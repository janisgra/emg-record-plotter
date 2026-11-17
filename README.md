# emg-record-plotter

Raw 4-channel EMG straight from an ESP32-S3 to a fast live plot and a CSV
recorder, without the ML firmware or the emg_classifier_ui stack (no CUDA,
no ONNX, no on-device inference).

```
firmware/   ESP-IDF exfiltration firmware (ESP32-S3-DevKitC N16R8, XIAO ESP32-S3)
viewer/     emg_plotter.py: DearPyGui live plot + CSV recorder
tools/      fake_stream.py: synthetic stream for testing without hardware
recordings/ default output of the recorder (created on first recording)
```

## Quick start

```bash
# 1. flash (board on /dev/ttyACM1, see firmware/platformio.ini)
cd firmware && pio run -e esp32s3-n16r8 -t upload && pio device monitor

# 2. plot (any Python with dearpygui + numpy, e.g. the anaconda base env)
python3 viewer/emg_plotter.py                 # UDP 5005, 5 s window, fixed ADC axis
python3 viewer/emg_plotter.py -w 1 --y auto   # 1 s window, autoscaled
python3 viewer/emg_plotter.py --record        # record from the first packet on
```

No hardware at hand: `python3 tools/fake_stream.py --loss 0.05` and start the viewer.

## Firmware

Port of emg-hci `firmware/emg_exfiltrator_node/src/esp32_mini_4ch.cpp`:

| | |
|---|---|
| ADC | continuous/DMA, 80 kHz total = **20 kHz per channel**, 12-bit, 12 dB attenuation (~0-3.1 V). Hardware max is 83.333 kHz total: `-DADC_SAMPLE_FREQ_HZ=83333` in platformio.ini. 80 kHz matches all earlier recordings and the trained models. |
| Processing | none, raw ADC codes |
| Packets | **compact** (default): uint16 samples, 160 samples/packet, ~126 pkt/s, 1290 B (~1.3 Mbit/s). `-DWIRE_FORMAT=1` = legacy int32, 25 samples/packet, ~800 pkt/s (emg_classifier_ui compatible) |
| Tasks | ADC task (core 1) only fills a queue; sender (core 0) does `sendto` and, while WiFi buffers are full (`ENOMEM`), keeps the packet and retries |
| Buffers | 51 ms ADC DMA pool; **2 s stall buffer in PSRAM** so WiFi pauses delay data instead of dropping it |
| WiFi | power save off; TX power 8 dBm on the N16R8 env (board next to the router; remove `-DWIFI_TX_POWER_DBM=8` if it is far away) |
| Target | `192.168.7.240:5005`, WiFi `GL-NET`; change the CONFIG block in `src/main.cpp` |

### Why the compact format (measured 2026-11-17, N16R8, RSSI -22 dBm, channel 11)

| firmware | frames/s needed | got out of the ESP32 | PC received |
|---|---|---|---|
| legacy int32, 20 dBm | 807 | ~550, rest `ENOMEM` | ~68 % |
| legacy int32, 8 dBm | 807 | ~700 | ~85-95 % |
| compact uint16, 8 dBm | 126 | 126, `fail 0` | ~95 %, ~1 s stalls lost data |
| compact + 2 s PSRAM stall buffer | 126 | 126, `fail 0`, `qdrop 0` | **~100 %** (stalls caught up) |

The ESP32 -> router hop could not carry ~800 small frames/s here; it also stalls
for ~1 s every 20-40 s. Packets that left the ESP32 practically always arrived.

| env | board | EMG pins (CH0..CH3) | device id |
|---|---|---|---|
| `esp32s3-n16r8` | ESP32-S3-DevKitC-1 N16R8 | GPIO1, 2, 3, 4 (same as the ML firmware) | 6 |
| `xiao-esp32s3` | Seeed XIAO ESP32-S3 | GPIO5 (A4), 2 (A1), 3 (A2), 4 (A3) | 5 |

Channel order from the thesis electrode table: CH0 Flexor Carpi Radialis,
CH1 Flexor Carpi Ulnaris, CH2 Extensor Digitorum, CH3 Extensor Carpi Ulnaris.
Front-end outputs must stay within 0-3.3 V (ESP32-S3 pin limit 3.6 V);
a mid-rail biased signal idles near code ~2000.

Serial status, once per second:

```
EMG: 20168 S/s/ch | tx 126 pkt/s, retry 0, fail 0, qdrop 0, buf 0/250, dma-ovf 0 | RSSI -22
EMG: min/mean/max clip: CH0 1520/2010/2480 0% | CH1 ...
```

`retry` = WiFi buffers were full and the sender waited (harmless),
`buf` = packets waiting in the stall buffer, `qdrop` = stall longer than the
buffer (data lost), `fail` = other send errors, `dma-ovf` = ADC data lost on
the device, `clip` = samples at 0 or 4095.

## PC side: WiFi power saving

If the PC is on WiFi (not cable), its power saving holds incoming packets and
releases them in bursts (measured: ping to the router 1-85 ms instead of
1-3 ms), which makes the plot jerky and can lose catch-up bursts at the router.
Best: connect the PC by Ethernet to the router. Otherwise disable power save
for that connection:

```bash
nmcli connection modify "GL-NET 1" 802-11-wireless.powersave 2   # 2 = disable
nmcli connection up "GL-NET 1"
```

## Viewer

* GPU-drawn (DearPyGui/ImPlot) plots, one per channel, linked time axis
* min/max envelope for display: peaks are never hidden, even in a 30 s window
* lost packets are drawn as **gaps** (from sequence numbers), never stitched
* rate measured with the **ESP32 clock** (packet timestamps) + received rate + loss
* per-channel mean / peak-to-peak / RMS / **clip %** for the last second
* Y axis: fixed ADC range with rail lines (default), auto, or free zoom
* Space = pause display, R = start/stop recording

Accepts the compact format of this firmware and every sender using the
legacy format, including the emg-hci firmwares (`emg_exfiltrator_node`,
`emg_esp_ML_classifier-exfiltrator`) and XIAO packets with the IMU/temperature trailer:

```
legacy : [device_id:1][seq:4 BE][timestamp_us:4 BE][num_channels:1]       [n x num_channels x int32 BE]
compact: [device_id:1][seq:4 BE][timestamp_us:4 BE][0x80|num_channels:1]  [n x num_channels x uint16 BE]
```

### Recording format

Same as emg_classifier_ui's raw recordings, so the existing analysis code reads it:

```
# EMG raw recording  channels=4  date=20261117_224507  device=6  sample_rate=20160  source=emg-record-plotter
timestamp_s,sequence,ch0,ch1,ch2,ch3
1794951907.273083,21,2113,1988,2028,2013
```

`timestamp_s` = PC wall-clock receive time of the packet (all rows of one
packet share it), `sequence` = packet number, so lost packets show up as
gaps in `sequence`.

## Compatibility notes

* emg_classifier_ui only understands the legacy format (`-DWIRE_FORMAT=1`),
  only accepts device id 6 (`config::receiver::EXPECTED_DEVICE_ID`) and
  packets up to 512 B (legacy default of 25 samples/packet fits).
* `pip install -r viewer/requirements.txt` if dearpygui is missing.
