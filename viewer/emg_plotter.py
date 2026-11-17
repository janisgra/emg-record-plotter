#!/usr/bin/env python3
"""
emg_plotter.py -- fast live plot + CSV recorder for ESP32 EMG UDP streams

Listens for the EMG wire format used by ../firmware and by the emg-hci
firmwares (emg_exfiltrator_node, emg_esp_ML_classifier-exfiltrator):

    [device_id:1][seq:4 BE][timestamp_us:4 BE][num_channels:1]
    [n_samples x num_channels x int32 BE]   (+ optional legacy XIAO trailer)

or the compact format of ../firmware (WIRE_FORMAT 2), marked by bit 7 of the
channel byte:  num_channels = 0x80 | n,  samples as uint16 BE.

* GPU-drawn plots (DearPyGui / ImPlot), one per channel, linked time axis.
* Min/max envelope for display: never hides a peak, however long the window.
* Lost packets are drawn as gaps (seq numbers), never stitched together.
* Rate is measured with the ESP32 clock (packet timestamps), so it shows what
  the device really samples, independent of WiFi timing.
* Per-channel levels and clipping (0 / 4095 = ADC rails) for the last second.
* Records to CSV in the emg_classifier_ui format:
      emg_raw_<YYYYmmdd_HHMMSS>.csv : timestamp_s,sequence,ch0..chN
  (timestamp_s = PC wall-clock receive time of the packet)

Usage
    python emg_plotter.py                    # listen on UDP 5005
    python emg_plotter.py --window 2 --y adc
    python emg_plotter.py --record           # start recording immediately
Keys: Space = pause display, R = start/stop recording
"""

from __future__ import annotations

import argparse
import queue
import signal
import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg

# ============================================================================
# Protocol / sizing
# ============================================================================
HEADER = struct.Struct(">BIIB")      # device_id, seq, timestamp_us, num_channels
LEGACY_SPP = 10                      # XIAO packets carry an IMU/temp trailer
ADC_MAX = 4095                       # 12-bit ESP32-S3 ADC
SEQ_MASK = 0xFFFFFFFF
RING_SECONDS = 60                    # history kept per device
MAX_RATE = 25000                     # ring sizing (>= 20.8 kHz ADC max per ch)
DEFAULT_RATE = 20000.0               # time base until the device clock is known
PLOT_BINS = 1600                     # envelope bins per line (2 points each)
STALE_S = 1.0                        # "NO DATA" after this long

# ============================================================================
# Look
# ============================================================================
C = {
    "bg": (22, 22, 27), "panel": (32, 32, 39), "frame": (45, 45, 54),
    "plot_bg": (14, 14, 18), "accent": (100, 150, 255), "ok": (100, 210, 120),
    "warn": (255, 200, 100), "err": (255, 100, 100), "text": (225, 225, 230),
    "dim": (140, 140, 150),
}
CH_COLORS = [(0, 255, 135), (0, 205, 255), (255, 110, 110), (255, 215, 60),
             (185, 140, 255), (255, 160, 70), (120, 225, 120), (210, 210, 210)]
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/TTF/DejaVuSans.ttf",
                   "C:/Windows/Fonts/segoeui.ttf"]


# ============================================================================
# Per-device stream state
# ============================================================================
class DeviceStream:
    """Mirror ring buffer: every sample is stored at i and i+cap, so any
    window of up to `cap` samples is one contiguous numpy view (no copies)."""

    def __init__(self, dev_id: int, nch: int, addr: str):
        self.id, self.nch, self.addr = dev_id, nch, addr
        self.cap = RING_SECONDS * MAX_RATE
        self.buf = np.full((2 * self.cap, nch), np.nan, dtype=np.float32)
        self.total = 0                  # samples written (incl. gap fill)
        self.last_seq: int | None = None
        self.spp = 0
        self.rx_pkts = self.lost_pkts = self.late_pkts = self.resyncs = 0
        self.last_rx = 0.0
        self.ref: tuple[int, int] | None = None   # (seq, ts_us) for rate
        self.gen_rate = 0.0             # samples/s/ch by the ESP32 clock
        self.rx_hist: deque = deque()   # (time, samples) of last ~2 s

    def _put(self, rows: np.ndarray) -> None:
        n = rows.shape[0]
        i = self.total % self.cap
        first = min(n, self.cap - i)
        self.buf[i:i + first] = rows[:first]
        self.buf[i + self.cap:i + self.cap + first] = rows[:first]
        if first < n:                   # wrapped
            rest = n - first
            self.buf[0:rest] = rows[first:]
            self.buf[self.cap:self.cap + rest] = rows[first:]
        self.total += n

    def write(self, rows: np.ndarray) -> None:
        self._put(rows)

    def write_gap(self, n: int) -> None:
        n = min(n, self.cap // 2)
        if n > 0:
            self._put(np.full((n, self.nch), np.nan, dtype=np.float32))

    def view(self, k: int, total: int) -> np.ndarray:
        """Last k samples ending at `total` as a contiguous view."""
        k = min(k, total, self.cap - 1)
        start = (total - k) % self.cap
        return self.buf[start:start + k]

    def rate(self) -> float:
        return self.gen_rate if self.gen_rate > 0 else DEFAULT_RATE

    def rx_rate(self, now: float) -> float:
        while self.rx_hist and now - self.rx_hist[0][0] > 2.0:
            self.rx_hist.popleft()
        if len(self.rx_hist) < 2:
            return 0.0
        span = now - self.rx_hist[0][0]
        return sum(n for _, n in self.rx_hist) / span if span > 0 else 0.0


# ============================================================================
# CSV recorder (emg_classifier_ui raw format)
# ============================================================================
class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.q: queue.Queue = queue.Queue(maxsize=20000)
        self.active = False
        self.dev_id = -1
        self.path: Path | None = None
        self.rows = 0
        self.dropped = 0
        self.t_start = 0.0
        self._thread: threading.Thread | None = None

    def start(self, dev_id: int, nch: int, rate: float) -> None:
        if self.active:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"emg_raw_{stamp}.csv"
        f = open(self.path, "w", buffering=1 << 20)
        f.write(f"# EMG raw recording  channels={nch}  date={stamp}  device={dev_id}"
                f"  sample_rate={rate:.0f}  source=emg-record-plotter\n")
        f.write("timestamp_s,sequence," + ",".join(f"ch{c}" for c in range(nch)) + "\n")
        self.rows = self.dropped = 0
        self.dev_id, self.t_start, self.active = dev_id, time.time(), True
        self._thread = threading.Thread(target=self._writer, args=(f,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.q.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, dev_id: int, t: float, seq: int, samples: np.ndarray) -> None:
        if self.active and dev_id == self.dev_id:
            try:
                self.q.put_nowait((t, seq, samples))
            except queue.Full:
                self.dropped += samples.shape[0]

    def _writer(self, f) -> None:
        last_flush = time.monotonic()
        while True:
            item = self.q.get()
            if item is None:
                break
            t, seq, s = item
            prefix = f"{t:.6f},{seq},"
            f.write("".join(prefix + ",".join(map(str, row)) + "\n" for row in s.tolist()))
            self.rows += s.shape[0]
            if time.monotonic() - last_flush > 0.5:
                f.flush()
                last_flush = time.monotonic()
        f.close()


# ============================================================================
# UDP receiver
# ============================================================================
class Receiver(threading.Thread):
    def __init__(self, bind_ip: str, port: int, device_filter: int | None, recorder: Recorder):
        super().__init__(daemon=True)
        self.bind_ip, self.port = bind_ip, port
        self.device_filter = device_filter
        self.recorder = recorder
        self.lock = threading.Lock()
        self.devices: dict[int, DeviceStream] = {}
        self.bad_pkts = 0
        self.running = True

    def run(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        s.bind((self.bind_ip, self.port))
        s.settimeout(0.2)
        while self.running:
            try:
                data, addr = s.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self._on_packet(data, addr[0])
        s.close()

    # NB: not "_handle" -- Python 3.13's threading.Thread uses that attribute.
    def _on_packet(self, data: bytes, ip: str) -> None:
        if len(data) < HEADER.size:
            self.bad_pkts += 1
            return
        dev_id, seq, ts, nch_byte = HEADER.unpack_from(data)
        compact = bool(nch_byte & 0x80)          # firmware WIRE_FORMAT 2: uint16 samples
        nch = nch_byte & 0x7F
        if not 0 < nch <= 16 or (self.device_filter is not None and dev_id != self.device_filter):
            return
        bps = 2 if compact else 4
        payload = len(data) - HEADER.size
        if payload % (bps * nch) == 0:
            spp = payload // (bps * nch)
        else:
            spp = -1 if compact else LEGACY_SPP  # legacy XIAO: 10 samples + trailer
        if spp <= 0 or payload < spp * bps * nch:
            self.bad_pkts += 1
            return
        raw = np.frombuffer(data, dtype=">u2" if compact else ">i4",
                            count=spp * nch, offset=HEADER.size).reshape(spp, nch)
        now = time.monotonic()

        with self.lock:
            dev = self.devices.get(dev_id)
            if dev is None or dev.nch != nch:
                dev = self.devices[dev_id] = DeviceStream(dev_id, nch, ip)
            dev.spp = spp
            if dev.last_seq is not None:
                ahead = (seq - dev.last_seq) & SEQ_MASK
                if ahead == 0 or ahead >= 0x80000000:          # older than last
                    if ((dev.last_seq - seq) & SEQ_MASK) < 1000:
                        dev.late_pkts += 1                     # late / duplicate
                        return
                    dev.resyncs += 1                           # device rebooted
                    dev.ref = None
                elif ahead > 1:
                    lost = ahead - 1
                    if lost < 20000:
                        dev.lost_pkts += lost
                        dev.write_gap(lost * spp)              # keep time honest
                    else:
                        dev.resyncs += 1
                        dev.ref = None
            dev.last_seq = seq
            dev.write(raw.astype(np.float32))
            dev.rx_pkts += 1
            dev.last_rx = now
            dev.rx_hist.append((now, spp))
            # generated rate from the ESP32 clock (independent of WiFi timing)
            if dev.ref is None:
                dev.ref = (seq, ts)
            else:
                dseq = (seq - dev.ref[0]) & SEQ_MASK
                dts = (ts - dev.ref[1]) & SEQ_MASK
                if dts >= 1_000_000 and dseq > 0:
                    r = dseq * spp / (dts / 1e6)
                    if 100 < r < 200_000:
                        dev.gen_rate = r if dev.gen_rate == 0 else 0.8 * dev.gen_rate + 0.2 * r
                    dev.ref = (seq, ts)

        self.recorder.submit(dev_id, time.time(), seq, raw.astype(np.int32))


# ============================================================================
# UI
# ============================================================================
def envelope(v: np.ndarray, bins: int):
    """(n, nch) -> sample offsets (2*bins,) and min/max interleaved (2*bins, nch).
    NaN-only bins (lost packets) stay NaN so the line shows a gap."""
    n = v.shape[0]
    if n <= 2 * bins:
        return np.arange(n, dtype=np.float64), v
    step = n // bins
    m = step * bins
    b = v[n - m:].reshape(bins, step, v.shape[1])
    lo = np.fmin.reduce(b, axis=1)
    hi = np.fmax.reduce(b, axis=1)
    out = np.empty((2 * bins, v.shape[1]), dtype=np.float32)
    out[0::2], out[1::2] = lo, hi
    centers = (n - m) + np.arange(bins) * step + step / 2.0
    return np.repeat(centers, 2), out


class PlotterUI:
    def __init__(self, rx: Receiver, rec: Recorder, args):
        self.rx, self.rec = rx, rec
        self.window_s = args.window
        self.y_mode = {"adc": "ADC 0-4095", "auto": "Auto", "free": "Free"}[args.y]
        self.paused = False
        self.sel: int | None = args.device
        self.built_for: tuple[int, int] | None = None   # (device, nch)
        self.auto_lim: dict[int, tuple[float, float]] = {}
        self.dev_items: list[str] = []

    # ---------------------------------------------------------------- setup
    def build(self) -> None:
        dpg.create_context()
        with dpg.font_registry():
            for path in FONT_CANDIDATES:
                if Path(path).exists():
                    dpg.bind_font(dpg.add_font(path, 17))
                    break

        with dpg.theme() as t_global:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, C["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C["panel"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, C["frame"])
                dpg.add_theme_color(dpg.mvThemeCol_Button, (55, 75, 120))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (75, 105, 165))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, C["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, C["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, C["accent"])
                dpg.add_theme_color(dpg.mvThemeCol_Text, C["text"])
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)
                dpg.add_theme_color(dpg.mvPlotCol_PlotBg, C["plot_bg"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_FrameBg, C["panel"], category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (60, 60, 72), category=dpg.mvThemeCat_Plots)
            with dpg.theme_component(dpg.mvInfLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 90, 90, 150), category=dpg.mvThemeCat_Plots)
        dpg.bind_theme(t_global)

        self.ch_themes = []
        for col in CH_COLORS:
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.0, category=dpg.mvThemeCat_Plots)
            self.ch_themes.append(t)
        with dpg.theme() as self.t_rec:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (205, 60, 60))

        with dpg.window(tag="main", no_scrollbar=True):
            with dpg.group(horizontal=True):
                dpg.add_text("EMG Record Plotter", color=C["accent"])
                dpg.add_text("", tag="live")
                dpg.add_spacer(width=12)
                dpg.add_text("Device")
                dpg.add_combo([], tag="dev_combo", width=230, callback=self._on_device)
                dpg.add_text("Window")
                dpg.add_slider_float(tag="win", default_value=self.window_s, min_value=0.2,
                                     max_value=30.0, width=170, format="%.1f s",
                                     callback=lambda s, a: setattr(self, "window_s", a))
                dpg.add_text("Y")
                dpg.add_radio_button(["ADC 0-4095", "Auto", "Free"], tag="ymode",
                                     default_value=self.y_mode, horizontal=True,
                                     callback=self._on_ymode)
                dpg.add_button(label="Pause", tag="btn_pause", width=90, callback=self._toggle_pause)
                dpg.add_button(label="Record", tag="btn_rec", width=150, callback=self._toggle_record)
            with dpg.group(horizontal=True):
                dpg.add_child_window(tag="plot_area", width=-300, height=-34, border=False)
                with dpg.child_window(tag="info", width=-1, height=-34):
                    dpg.add_text("waiting for data ...", tag="info_dev", color=C["dim"])
                    dpg.add_text("", tag="info_rates")
                    dpg.add_separator()
                    dpg.add_text("Levels (last 1 s)", color=C["accent"])
                    with dpg.group(tag="info_levels"):
                        pass
                    dpg.add_separator()
                    dpg.add_text("Recording", color=C["accent"])
                    dpg.add_text("off", tag="info_rec", color=C["dim"], wrap=270)
            dpg.add_text("", tag="status", color=C["dim"])

        with dpg.handler_registry():
            for key, fn in (("mvKey_Spacebar", self._toggle_pause), ("mvKey_R", self._toggle_record)):
                k = getattr(dpg, key, None)
                if k is not None:
                    dpg.add_key_press_handler(k, callback=lambda s, a, u=fn: u())

        dpg.create_viewport(title="EMG Record Plotter", width=1600, height=960,
                            clear_color=C["bg"], vsync=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

    def _build_plots(self, dev_id: int, nch: int) -> None:
        if dpg.does_item_exist("subplots"):
            dpg.delete_item("subplots")
        if dpg.does_item_exist("info_levels"):
            dpg.delete_item("info_levels", children_only=True)
        with dpg.subplots(nch, 1, tag="subplots", parent="plot_area", width=-1, height=-1,
                          link_all_x=True, no_title=True, no_menus=False):
            for c in range(nch):
                with dpg.plot(tag=f"plot{c}", no_mouse_pos=False):
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f"x{c}", no_tick_labels=(c < nch - 1),
                                      label="time (s)" if c == nch - 1 else "")
                    with dpg.plot_axis(dpg.mvYAxis, tag=f"y{c}", label=f"CH{c}"):
                        dpg.add_line_series([], [], tag=f"s{c}")
                        dpg.add_inf_line_series([0.0, float(ADC_MAX)], horizontal=True, tag=f"rail{c}")
                    dpg.bind_item_theme(f"s{c}", self.ch_themes[c % len(self.ch_themes)])
        for c in range(nch):
            with dpg.group(horizontal=True, parent="info_levels"):
                dpg.add_text(f"CH{c}", color=CH_COLORS[c % len(CH_COLORS)])
                dpg.add_text("", tag=f"clip{c}")
            dpg.add_text("", tag=f"lv{c}", parent="info_levels", indent=12, color=C["dim"])
        self.built_for = (dev_id, nch)
        self.auto_lim.clear()
        self._apply_ymode()

    # ------------------------------------------------------------ callbacks
    def _on_device(self, sender, value) -> None:
        try:
            self.sel = int(value.split()[1])
        except (IndexError, ValueError):
            pass

    def _on_ymode(self, sender, value) -> None:
        self.y_mode = value
        self._apply_ymode()

    def _apply_ymode(self) -> None:
        if not self.built_for:
            return
        for c in range(self.built_for[1]):
            dpg.configure_item(f"rail{c}", show=(self.y_mode != "Auto"))
            if self.y_mode == "Free":
                dpg.set_axis_limits_auto(f"y{c}")
            elif self.y_mode == "ADC 0-4095":
                dpg.set_axis_limits(f"y{c}", -0.03 * ADC_MAX, 1.03 * ADC_MAX)

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        dpg.set_item_label("btn_pause", "Resume" if self.paused else "Pause")

    def _toggle_record(self) -> None:
        if self.rec.active:
            self.rec.stop()
            dpg.set_item_label("btn_rec", "Record")
            dpg.bind_item_theme("btn_rec", 0)
        elif self.built_for:
            dev_id, nch = self.built_for
            with self.rx.lock:
                rate = self.rx.devices[dev_id].rate() if dev_id in self.rx.devices else 0.0
            self.rec.start(dev_id, nch, rate)
            dpg.bind_item_theme("btn_rec", self.t_rec)

    # --------------------------------------------------------------- frame
    def update(self) -> None:
        now = time.monotonic()
        with self.rx.lock:
            devs = {i: (d.nch, d.addr) for i, d in self.rx.devices.items()}
        items = [f"Device {i}  ({n} ch, {a})" for i, (n, a) in sorted(devs.items())]
        if items != self.dev_items:
            self.dev_items = items
            dpg.configure_item("dev_combo", items=items)
        if not devs:
            dpg.set_value("status", f"listening on UDP {self.rx.port} - no packets yet")
            return
        if self.sel not in devs:
            self.sel = sorted(devs)[0]
        dpg.set_value("dev_combo", next(s for s in items if s.split()[1] == str(self.sel)))
        nch = devs[self.sel][0]
        if self.built_for != (self.sel, nch):
            self._build_plots(self.sel, nch)

        with self.rx.lock:
            d = self.rx.devices[self.sel]
            total, rate, spp = d.total, d.rate(), d.spp
            rx_rate, age = d.rx_rate(now), now - d.last_rx
            rx_pkts, lost, late, resync = d.rx_pkts, d.lost_pkts, d.late_pkts, d.resyncs
        live = age < STALE_S

        # ---- plots (mirror ring -> contiguous views, no copy)
        if not self.paused:
            k = int(self.window_s * rate) + 1
            v = d.view(k, total)
            if v.shape[0] > 1:
                off, env = envelope(v, PLOT_BINS)
                x = ((off - v.shape[0]) / rate).tolist()
                for c in range(nch):
                    dpg.set_value(f"s{c}", [x, env[:, c].tolist()])
                    dpg.set_axis_limits(f"x{c}", -self.window_s, 0.0)
                    if self.y_mode == "Auto":
                        self._auto_y(c, env[:, c])
            # levels over the last second
            last = d.view(int(rate), total)
            if last.shape[0] > 0:
                with np.errstate(all="ignore"):
                    mean = np.nanmean(last, axis=0)
                    lo, hi = np.nanmin(last, axis=0), np.nanmax(last, axis=0)
                    rms = np.nanstd(last, axis=0)
                    valid = np.sum(~np.isnan(last), axis=0)
                    clip = 100.0 * np.sum((last <= 0) | (last >= ADC_MAX), axis=0) / np.maximum(valid, 1)
                for c in range(nch):
                    if valid[c] == 0:
                        continue
                    dpg.set_value(f"lv{c}", f"mean {mean[c]:4.0f}  p-p {hi[c] - lo[c]:4.0f}  rms {rms[c]:4.0f}")
                    dpg.set_value(f"clip{c}", f"clip {clip[c]:4.1f}%")
                    dpg.configure_item(f"clip{c}", color=C["err"] if clip[c] > 0.5 else C["ok"])

        # ---- header / info / status
        dpg.set_value("live", "LIVE" if live else ("PAUSED" if self.paused else "NO DATA"))
        dpg.configure_item("live", color=C["ok"] if live else C["err"])
        loss = 100.0 * lost / max(rx_pkts + lost, 1)
        dpg.set_value("info_dev", f"Device {self.sel}  -  {devs[self.sel][1]}")
        dpg.configure_item("info_dev", color=C["text"])
        dpg.set_value("info_rates",
                      f"Rate   {rate:8.0f} S/s/ch  (ESP32 clock)\n"
                      f"RX     {rx_rate:8.0f} S/s/ch\n"
                      f"Loss   {loss:8.1f} %   ({lost} pkts)\n"
                      f"Pkts   {rx_pkts:8d}   x {spp} samples\n"
                      + (f"Late {late}  resync {resync}\n" if late or resync else ""))
        if self.rec.active:
            dur = time.time() - self.rec.t_start
            size = self.rec.path.stat().st_size / 1e6 if self.rec.path and self.rec.path.exists() else 0
            dpg.set_item_label("btn_rec", f"Stop  {int(dur // 60):02d}:{int(dur % 60):02d}")
            dpg.set_value("info_rec", f"{self.rec.path.name}\n{self.rec.rows:,} rows  {size:.1f} MB"
                          + (f"\n{self.rec.dropped} samples dropped" if self.rec.dropped else ""))
            dpg.configure_item("info_rec", color=C["err"])
        elif self.rec.path:
            dpg.set_value("info_rec", f"saved {self.rec.path}\n{self.rec.rows:,} rows")
            dpg.configure_item("info_rec", color=C["dim"])
        dpg.set_value("status", f"UDP {self.rx.port}   |   last packet {age * 1000:6.0f} ms ago"
                      f"   |   Space = pause   R = record   |   {dpg.get_frame_rate():.0f} fps")

    def _auto_y(self, c: int, y: np.ndarray) -> None:
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        pad = max((hi - lo) * 0.08, 20.0)
        lo, hi = lo - pad, hi + pad
        plo, phi = self.auto_lim.get(c, (lo, hi))
        # expand instantly, shrink smoothly
        lo = lo if lo < plo else plo + 0.1 * (lo - plo)
        hi = hi if hi > phi else phi + 0.1 * (hi - phi)
        self.auto_lim[c] = (lo, hi)
        dpg.set_axis_limits(f"y{c}", lo, hi)


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Fast live plot + CSV recorder for ESP32 EMG UDP streams")
    ap.add_argument("-p", "--port", type=int, default=5005, help="UDP port (default 5005)")
    ap.add_argument("--bind", default="0.0.0.0", help="bind address (default all)")
    ap.add_argument("-d", "--device", type=int, default=None, help="only accept this device id")
    ap.add_argument("-w", "--window", type=float, default=5.0, help="time window in s (default 5)")
    ap.add_argument("--y", choices=["adc", "auto", "free"], default="adc",
                    help="Y axis: fixed ADC range (default), auto, or free zoom")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent.parent / "recordings"),
                    help="recording directory (default ../recordings)")
    ap.add_argument("--record", action="store_true", help="start recording as soon as data arrives")
    args = ap.parse_args()

    rec = Recorder(Path(args.out))
    rx = Receiver(args.bind, args.port, args.device, rec)
    rx.start()
    ui = PlotterUI(rx, rec, args)
    ui.build()
    # Ctrl+C / kill: leave the render loop so the recording is closed cleanly
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: dpg.stop_dearpygui())
    try:
        while dpg.is_dearpygui_running():
            ui.update()
            if args.record and not rec.active and ui.built_for:
                ui._toggle_record()
                args.record = False
            dpg.render_dearpygui_frame()
    finally:
        rec.stop()
        rx.running = False
        dpg.destroy_context()


if __name__ == "__main__":
    main()
