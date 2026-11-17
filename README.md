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

## Setup (Windows, Python 3.12)
```bash
python -m venv %USERPROFILE%\.venvs\phantom
%USERPROFILE%\.venvs\phantom\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

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
Machine A sends video, machine B receives and repairs it. Packets cross your real network (Wi-Fi or
Ethernet) as UDP, and a relay on machine A adds controlled packet loss on top.

```
 Machine A                                             Machine B
 sender --UDP--> relay :5000 (drops X %) ==network==> receiver :5004 (repairs, shows window)
        <--------------- loss feedback -----------------------
```

1. **Both machines:** clone the repo and install (see Setup). `weights/repairnet.pt` is in the repo.
   Machine B should have an NVIDIA GPU; without one, start the receiver with `--model flow`.
2. **Both machines on the same network.** Find machine B's IP address (`ipconfig` on Windows,
   `ip addr` on Linux), for example `192.168.1.42`.
3. **Machine B (receiver):** start it first. Windows asks to allow Python through the firewall the first
   time: allow it on private networks (it needs inbound UDP port 5004).
   ```bash
   python -m demo.receiver_view --port 5004
   ```
4. **Machine A (relay):** forwards to B and drops 10 % of the video packets.
   ```bash
   python -m phantom.live.relay --listen 5000 --to 192.168.1.42:5004 --loss 0.10
   ```
   `--model ge --burst 4` makes the loss bursty, `--delay 30` adds 30 ms of one-way delay. Restart it to
   change the loss.
5. **Machine A (sender):** a video file, or `--src 0` for the webcam.
   ```bash
   python -m phantom.live.sender --to 127.0.0.1:5000 --src path/to/video.mp4
   ```
   No video at hand? `python -m tools.fetch_clips` downloads the test clips into `data/`.

Machine B's window shows the same received packets twice: left without Phantom, right with Phantom.
Keys: `c` cycles the left side (no repair, normal decoder, copy last tile), `a` outlines lost tiles,
`f` full screen, `q` quits. The sender prints the loss the receiver reports back.

Notes: both machines must use the same `PHANTOM_RES` (default 360p; the receiver warns otherwise).
The 360p stream needs about 20-30 Mbit/s, fine on Ethernet or decent Wi-Fi. To try everything on one
machine, use `--to 127.0.0.1:5004` for the relay.

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
