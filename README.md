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
python -m demo.live_demo --src data/test/clip.mp4          # live side-by-side demo over real UDP
```

### Demo app (upload a video or paste a stream URL)
Runs on any machine with the Python environment above; a CUDA GPU makes it real time, and no other
service is needed. Then open the address it prints in a browser.
```bash
python -m phantom.app                                  # 360p, http://127.0.0.1:8000 (any laptop GPU)
PHANTOM_RES=720 python -m phantom.app                  # 720p, needs a faster GPU for 30 fps
python -m phantom.app --host 0.0.0.0 --port 8000       # also reachable from other devices on the network
```
`--host 0.0.0.0` lets a phone or another laptop on the same network open `http://<this-machine's-IP>:8000`;
any network path between the browser and the server works (LAN, VPN, port forward).
Upload any video or paste a stream URL (`.mp4`, HLS `.m3u8`, `rtsp://`). Each frame goes through the real
packetizer, a lossy channel and the real receiver; the left side shows what the loss does (no repair, a normal
decoder that drops damaged frames, or copy-last-tile), the right side shows Phantom regenerating the lost tiles
(RepairNet, or framegen with one frame of look-ahead). Both sides lose exactly the same packets. Loss rate,
random/bursty loss, side-by-side or wipe view are live controls. Video reaches the browser as H.264
(fragmented MP4 over HTTP into Media Source Extensions), with an MJPEG fallback.

### Two-machine demo over a real network
One machine sends, the other receives and repairs. Both on the same Wi-Fi or hotspot, both with the repo
installed (see Setup; `weights/repairnet.pt` is in the repo), commands run from the project folder.

**Receiver machine** (ideally the one with an NVIDIA GPU), start it first:
```bash
python -m phantom.receive
```
**Sender machine** (streams its webcam; finds the receiver on the network by itself):
```bash
python -m phantom.send
```

That is all. The receiver window shows the same packets twice: left without Phantom, right with Phantom.
Packet loss (10 % to start) is simulated as packets arrive at the receiver, on top of anything the real
network loses; the sender prints the loss the receiver reports back.

| Receiver key | Does |
|---|---|
| `+` / `-` | packet loss up / down 5 % |
| `b` | random or bursty loss |
| `c` | left side: no repair, normal decoder (freeze), copy last tile |
| `a` | outline the lost tiles |
| `f` / `q` | full screen / quit |

Options: `python -m phantom.send --src video.mp4` streams a file instead of the webcam;
`python -m phantom.receive --model flow` for a receiver without a GPU; `--loss 0.2` sets the starting loss.

If the sender says *No receiver found*: some networks block broadcasts, so pass the address the receiver
printed when it started: `python -m phantom.send --to 192.168.1.42`. The first time, Windows asks whether
Python may use the network; allow it (on a phone hotspot that is a *public* network). Both machines must use
the same `PHANTOM_RES` (default 360p).

Advanced: `phantom.live.relay` + `phantom.live.sender` + `demo.receiver_view` put the loss in a separate
relay process instead, with extra delay and jitter options (`python -m phantom.live.relay --help`).

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
  conceal.py    freeze / copy / inpaint / flow / RepairNet
  sim.py        discrete-event simulator (none / nack / fec / tcp)
  models/       RepairNet
  live/         real UDP sender, receiver, lossy relay
train/          RepairNet training
eval/           experiment runner and plots
demo/           live demo
tests/          unit tests
```
