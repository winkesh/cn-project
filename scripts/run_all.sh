#!/usr/bin/env bash
# Full pipeline on a GPU machine: tests -> data -> train -> evaluate -> plots.
#   tmux new -s phantom 'bash scripts/run_all.sh 2>&1 | tee logs/run_all.log'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/.venvs/phantom/bin/python}
FG=${FG:-$HOME/framegen}                         # our framegen project (game data + checkpoints)
FG_CKPT=${FG_CKPT:-$FG/ckpt_cut_w0.75_leaky_d4_same/best.pt}
TRAIN_MIN=${TRAIN_MIN:-60}
mkdir -p logs results weights

echo "== $(date) tests";            $PY -m pytest -q tests
echo "== $(date) fetch clips";      $PY -m tools.fetch_clips --max-frames 300 | tee logs/fetch.log

echo "== $(date) game clips (held out from framegen training)"
cp --update=none "$FG_CKPT" weights/framegen.pt
for g in forza mw3; do
  [ -f data/test/game_$g.avi ] || $PY -m tools.prep_clip --src "$FG/clips_laptop/$g.mp4" \
      --out data/test/game_$g.avi --stride 2 --start 600 --frames 300
done

echo "== $(date) train ${TRAIN_MIN} min"
$PY -m train.train_repairnet --minutes "$TRAIN_MIN" --batch 32 --crop-tiles 6 --max-frames 300 \
    --triplets "$FG/data/hr/game_train" --triplets-max 1500 --out weights/repairnet.pt | tee logs/train.log
echo "== $(date) evaluate";         $PY -m eval.run_matrix --frames 300 --seeds 2 | tee logs/eval.log
echo "== $(date) plots";            $PY -m eval.plots | tee logs/plots.log
echo "== $(date) ALL_DONE"
