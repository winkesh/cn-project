# Phantom Frames
**Generative loss recovery for real-time video streaming**

Netflix and YouTube never show you a glitch from packet loss, because they hide it behind TCP
retransmissions and 10-30 seconds of buffer. That is exactly why they stall (the spinning wheel) and
why "live" streams run seconds behind reality. Low-latency streaming of **any content** (live sports,
live events, game streaming, cloud gaming, video calls) cannot afford that buffer, and is left with three
bad options: retransmit (too slow once RTT approaches the deadline), send redundancy (FEC, costs bandwidth
all the time), or freeze until a keyframe arrives.

Phantom Frames takes a fourth path: **never retransmit, and let the receiver synthesise what the network
lost**, within a 100 ms playout deadline.

```
 sender                         lossy link                       receiver
 frame -> 40x40 JPEG tiles  ->  UDP / RTP-compatible  ->  reassemble at playout deadline (100 ms)
        -> interleave tiles     (loss, bursts, delay)      -> lost tiles = mask
           across packets                                   -> RepairNet fills only the lost tiles
        <- loss-aware refresh  <------ feedback (100 ms) ---- received pixels are kept bit-exact
```

* **Transport**: RTP-compatible header (RFC 3550) plus a header extension (frame id, packet index,
  flags, send time, 144-bit skip map). Wireshark decodes it as RTP.
* **Interleaving**: neighbouring tiles go into different packets, so a burst of losses becomes
  scattered holes that are easy to fill instead of one large gap.
* **Conditional replenishment + loss-aware refresh** (live mode): unchanged tiles are not re-sent;
  tiles the receiver reports lost are force-refreshed in the next frame.
* **RepairNet**: a small mask-conditioned U-Net (~1.9 M params) that takes the previous two
  output frames plus the partially received frame and fills only the lost tiles. Trained on loss
  masks produced by the real packetizer layout under random and Gilbert-Elliott bursty loss.
  It needs no future frames, so it adds no latency.
* **Baselines**: freeze, copy-tile, OpenCV inpainting, optical-flow extrapolation, NACK
  retransmission within the deadline, XOR FEC (RFC 5109 style), and TCP.

## Setup (Python 3.12)
From the project folder (works the same in cmd, PowerShell and VS Code's terminal):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
On Linux/macOS activate with `source .venv/bin/activate`. Without an NVIDIA GPU install plain
`pip install torch torchvision` and use `--model flow` on the receiver.

## Data
Standard open test sequences, covering four content types (all streamed and stored as 640x360 at ~30 fps):

| Type | Train | Test (held out) |
|---|---|---|
| people | Johnny, KristenAndSara, vidyo1/3/4 (Xiph) | FourPeople (Xiph) |
| motion / scenery | parkrun, mobcal, ducks_take_off, in_to_tree, old_town_cross (Xiph) | shields (Xiph) |
| sports-like | | park_joy (Xiph) |
| film | Sintel trailer (Blender, CC-BY) | Tears of Steel (Blender, CC-BY) |

```bash
python -m tools.fetch_clips          # downloads + converts everything into data/train and data/test
```
Any other mp4/avi/mov dropped into `data/train/` or `data/test/` is picked up too.

## Run
```bash
python -m pytest -q tests                                  # correctness tests
python -m train.train_repairnet --minutes 40               # train RepairNet -> weights/repairnet.pt
python -m eval.run_matrix --frames 300                     # all experiments -> results/results.csv
python -m eval.plots                                       # figures -> results/figs/
python -m phantom.app                                      # the demo (web page), see below
```

### Demo app (upload a video or paste a stream URL)
Runs on any machine with the Python environment above; a CUDA GPU makes it real time, and no other
service is needed. Then open the address it prints in a browser.
```bash
python -m phantom.app                                  # 360p, http://127.0.0.1:8000 (any laptop GPU)
python -m phantom.app --res 720                        # 720p or 1080p: needs a stronger CPU, see below
python -m phantom.app --host 0.0.0.0 --port 8000       # also reachable from other devices on the network
```
By default the app picks the fastest repair backend on the machine (`--backend auto`: a TensorRT engine if
`python -m tools.build_accel` built one, else PyTorch, else DirectML) and repairs at 360p when the stream is
larger. How far the resolution goes was measured on the running demo (game clip, 10 % loss, browser attached):

| Machine | 360p | 720p | 1080p |
|---|---|---|---|
| Laptop: i5-12450HX + RTX 3050, TensorRT FP16 | 30 fps | 12-14 fps | below that |
| Desktop: 16 cores + RTX 5060 Ti, TensorRT INT8+FP16 | 30 fps | 30 fps | 23 fps |

Above 360p the limit is the CPU, not the GPU: every 720p frame is 576 JPEG tiles and packets (1080p: 1296), and
sending, receiving and the browser's H.264 stream share one Python process. `python -m tools.res_bench` times
each stage on a machine.
`--host 0.0.0.0` lets a phone or another laptop on the same network open `http://<this-machine's-IP>:8000`;
any network path between the browser and the server works (LAN, VPN, port forward).
Upload any video or paste a stream URL (`.mp4`, HLS `.m3u8`, `rtsp://`). Each frame goes through the real
packetizer, a lossy channel and the real receiver; the left side shows what the loss does (no repair, a normal
decoder that drops damaged frames, or copy-last-tile), the right side shows Phantom regenerating the lost tiles
(RepairNet, or framegen with one frame of look-ahead). Both sides lose exactly the same packets. Loss rate,
random/bursty loss, side-by-side or wipe view are live controls. Video reaches the browser as H.264
(fragmented MP4 over HTTP into Media Source Extensions), with an MJPEG fallback.

### Streaming from another machine
The demo app is also the receiver: it listens on UDP port 5004 and shows a stream sent from another machine
on the same Wi-Fi or hotspot, in the same page and with the same controls.

1. **Receiving machine** (ideally the one with an NVIDIA GPU): start the demo and open it in the browser.
   ```bash
   python -m phantom.app
   ```
2. **Sending machine** (the repo installed too): stream its webcam. It finds the receiver by itself.
   ```bash
   python -m phantom.send
   ```

The page switches to *Live from &lt;sender's IP&gt;* as soon as packets arrive: left without Phantom, right with
Phantom, from the same real UDP packets. The loss slider and the random/bursty switch drop packets as they
arrive, on top of anything the real network loses, and the sender prints the loss the receiver reports back.
A live stream has no original to compare against, so the PSNR cards show a dash. Pick a sample to leave the
live stream; *Watch* under the samples goes back to it.

`python -m phantom.send --src video.mp4` streams a file instead of the webcam. If the sender says
*No receiver found* (some networks block broadcasts), give the address the demo printed when it started:
`python -m phantom.send --to 192.168.1.42`. The first time, Windows asks whether Python may use the network:
allow it (a phone hotspot is a *public* network). Both machines must use the same `PHANTOM_RES` (default 360p).

Optional: `python -m phantom.live.relay` sits between the two to add delay and jitter
(`python -m phantom.live.relay --help`); point the sender at it with `--to`.

## Running RepairNet fast on consumer hardware
The demo picks the fastest working backend on its own (`python -m phantom.app --backend auto|torch|trt|dml`):
a TensorRT engine built for this machine, else PyTorch on CUDA, else ONNX Runtime + DirectML on any DirectX 12
GPU (NVIDIA, AMD, Intel), else the CPU. Streams above 360p are repaired at 360p when full-resolution repair would
not fit the frame budget (`--repair-res auto|full|360`): the network fills the lost tiles from a 360p copy and
the upscaled result is pasted into the lost tiles only, so received pixels keep full resolution and the cost no
longer grows with the stream resolution.

```bash
python -m tools.build_accel        # ONNX (FP32, FP16, INT8) + TensorRT engines for this GPU, then verify
python -m tools.dml_check          # RepairNet through DirectML on every GPU in this PC
```
TensorRT 11 builds strongly typed networks, so precision lives in the ONNX file: an FP16 export for the FP16
engine, and explicit INT8 quantize/dequantize nodes calibrated on real training frames for INT8
(`phantom/accel.py`). Quantizing alone leaves every operation between the INT8 convolutions in FP32, which costs
back what INT8 saves; converting those to FP16 (`int8fp16`) is what makes INT8 pay off. Engines only load on the
operating system they were built on, so build them on the machine that runs the demo (on a laptop GPU,
`--opt-level 3` built in under a minute without trouble). The build tool measures every engine and the app uses
the fastest one that stays within 0.3 dB of PyTorch.

GPU compute per frame at 360p (inputs already on the GPU):

| GPU | PyTorch FP16 | TensorRT FP16 | TensorRT INT8 (+FP16) | Used |
|---|---|---|---|---|
| RTX 3050 laptop (Ampere) | 9.25 ms | **5.03 ms (1.84x)** | 5.71 ms (1.62x) | FP16 |
| RTX 5060 Ti (Blackwell) | 4.49 ms | 2.58 ms (1.74x) | **1.94 ms (2.31x)** | INT8+FP16 |

Repair quality and end-to-end time (frames uploaded from RAM each call), PSNR of the lost regions:

| Backend | GPU | Quality | Time / frame |
|---|---|---|---|
| Copy last tile (baseline) | – | 26.8 dB | – |
| PyTorch FP16 | RTX 5060 Ti | 31.4 dB | 6.0 ms |
| TensorRT FP16 | RTX 5060 Ti | 31.4 dB (identical) | 3.2 ms |
| TensorRT INT8+FP16 | RTX 5060 Ti | 31.2 dB | 2.8 ms |
| PyTorch FP16 | RTX 3050 laptop | 30.7 dB | 10.6 ms |
| TensorRT FP16 | RTX 3050 laptop | 30.7 dB (identical) | 6.6 ms |
| DirectML FP16 | RTX 3050 laptop | identical to CPU | 17.9 ms |
| DirectML FP32 | Intel UHD (integrated) | identical to CPU | 385 ms (105 ms at 180p) |

RTX 5060 Ti rows: 64 held-out samples; laptop rows: 16 (DirectML: 6), so compare quality within a machine, not
across. Which INT8 wins depends on the GPU: on Blackwell INT8 convolutions are ~25 % faster than FP16, on the
Ampere laptop GPU the INT8 conversions cost more than they save. Low-res repair at 720p keeps about 75 % of the
full-resolution gain (28.7 vs 29.8 dB in the lost regions, copy-tile 25.4 dB) for a quarter of the compute; on a
near-static video call it gains little over copy-tile.

## Experiments
| ID | Varies | Fixed | Question |
|---|---|---|---|
| E1 | random loss 0-30 % | RTT 50 ms, jitter buffer 100 ms | How does quality degrade with loss? |
| E2 | burst length 1-8 packets | 5 % mean loss | Does interleaving + synthesis survive bursts? |
| E3 | RTT 20-200 ms | 5 % loss | At what RTT does NACK stop beating synthesis? |
| E4 | interleaving on/off | 5 % bursty loss | Ablation of the packetisation co-design |

## Layout
```
phantom/        core library
  packet.py     RTP-compatible packet format
  tiles.py      tile grid, JPEG tiles, masks
  sender.py     packetizer, interleaving, XOR FEC, retransmission buffer
  receiver.py   reassembly, FEC recovery, missing-tile detection
  channel.py    loss models (Bernoulli, Gilbert-Elliott), link with delay/jitter/bottleneck
  conceal.py    freeze / copy / inpaint / flow / RepairNet / framegen
  sim.py        discrete-event simulator (none / nack / fec / tcp)
  models/       RepairNet, framegen network
  live/         real UDP sender and receiver (discovery, playout, feedback), lossy relay
  app/          the demo: web server, streaming engine, page (also the receiver for other machines)
  send.py       python -m phantom.send: stream a webcam or file to the demo on another machine
train/          RepairNet training
eval/           experiment runner and plots
tools/          clip download/preparation, demo video renderer
tests/          unit tests
```
