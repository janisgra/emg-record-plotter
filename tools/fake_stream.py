#!/usr/bin/env python3
"""
fake_stream.py -- synthetic EMG UDP stream for testing the viewer without hardware

    python tools/fake_stream.py                        # compact (firmware default): 20 kHz, 160 samples/pkt
    python tools/fake_stream.py --loss 0.05            # drop 5 % of packets
    python tools/fake_stream.py --format plain         # legacy int32 packets, 25 samples/pkt
    python tools/fake_stream.py --format xiao          # legacy XIAO packets (10 + trailer)
    python tools/fake_stream.py --format plain --rate 2475 --spp 20   # ML-firmware decimated stream

Signal per channel: ~2000 baseline, EMG-like bursts, small 50 Hz hum; CH2
bursts are large enough to hit the 0 / 4095 rails (to see clipping).
"""
import argparse
import socket
import struct
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=5005)
ap.add_argument("--rate", type=float, default=20000.0, help="samples/s per channel")
ap.add_argument("--spp", type=int, default=None, help="samples per packet (default per format)")
ap.add_argument("--channels", type=int, default=4)
ap.add_argument("--device", type=int, default=6)
ap.add_argument("--format", choices=["compact", "plain", "xiao"], default="compact")
ap.add_argument("--loss", type=float, default=0.0, help="fraction of packets to drop")
ap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = forever)")
a = ap.parse_args()
if a.format == "xiao":
    a.spp = 10
elif a.spp is None:
    a.spp = 160 if a.format == "compact" else 25

rng = np.random.default_rng(1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t0 = time.monotonic()
n = seq = sent = 0
print(f"{a.format} stream -> {a.host}:{a.port}  {a.rate:.0f} Hz x {a.channels} ch, {a.spp} samples/pkt, loss {a.loss:.0%}")
try:
    while not a.duration or time.monotonic() - t0 < a.duration:
        due = int((time.monotonic() - t0) * a.rate)
        while n + a.spp <= due:
            t = (n + np.arange(a.spp)) / a.rate
            cols = []
            for c in range(a.channels):
                burst = np.clip(np.sin(2 * np.pi * 0.35 * t + c * 1.6), 0, None)
                amp = 60 + burst * (1500 if c == 2 else 700)
                v = 2000 + 40 * np.sin(2 * np.pi * 50 * t) + rng.normal(0, 1, a.spp) * amp / 2.5
                cols.append(np.clip(v, 0, 4095))
            data = np.stack(cols, axis=1)
            if a.format == "compact":
                body, nch_byte = data.astype(">u2").tobytes(), 0x80 | a.channels
            else:
                body, nch_byte = data.astype(">i4").tobytes(), a.channels
            pkt = struct.pack(">BIIB", a.device, seq, int(t[0] * 1e6) & 0xFFFFFFFF, nch_byte) + body
            if a.format == "xiao":
                pkt += b"\x00\x00"                      # IMU / temp flags
            if rng.random() >= a.loss:
                sock.sendto(pkt, (a.host, a.port))
                sent += 1
            seq += 1
            n += a.spp
        time.sleep(0.002)
except KeyboardInterrupt:
    pass
print(f"generated {seq} packets, sent {sent}")
