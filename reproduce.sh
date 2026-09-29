#!/bin/bash
# Reproduces the final submission (public leaderboard 0.983078) end to end, in the order it was built.
# Usage: nohup bash reproduce.sh > /workspace/er/reproduce.log 2>&1 &
# Needs: dataset under /workspace/er/data (or DATA_DIR), one GPU, ~90 GB RAM, ~60 GB free disk.
set -euo pipefail
cd "$(dirname "$0")"
export WORK_DIR=${WORK_DIR:-/workspace/er/work}
FINAL_OUT=${FINAL_OUT:-/workspace/er/output}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
export DATA_DIR
stage() { echo; echo "######## $1  ($(date +%H:%M:%S))"; }

stage "A. prepare, embeddings, blocking, features, first models (version 2)"
OUT_DIR=/workspace/er/output_v2 bash run_all.sh

stage "B. small cross-encoder on version-2 uncertain pairs (used by version 3)"
for u in trainA trainB test; do cp "$WORK_DIR/$u/p1.npy" "$WORK_DIR/$u/p1_sel.npy"; done
python -u src/ce.py --work-dir "$WORK_DIR"

stage "C. version 3: rarity features, French pseudo-labels, stage 1, stage 2"
bash run_v3.sh

stage "D. final: stronger pseudo-labels, large cross-encoder, stage 2"
bash run_final.sh

stage "E. France cutoff 0.99 (chosen on the public leaderboard) -> final files"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "$FINAL_OUT" --override France=threshold:0.99
python "$DATA_DIR/../utils/validate_submission.py" --matching "$FINAL_OUT/matching_results.tsv" \
       --candidate "$FINAL_OUT/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)): $FINAL_OUT/matching_results.tsv and $FINAL_OUT/candidate_pairs.tsv"
