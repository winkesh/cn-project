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

Two-machine demo over a real network:
```bash
# machine B (receiver):  python -m demo.receiver_view --port 5004
# machine A (relay):     python -m phantom.live.relay --listen 5000 --to <B-ip>:5004 --loss 0.1
# machine A (sender):    python -m phantom.live.sender --to 127.0.0.1:5000 --src 0
```

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
