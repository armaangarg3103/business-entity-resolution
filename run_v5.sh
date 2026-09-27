#!/bin/bash
# Round 2 on top of version 3 (needs its work dir: features with specificity columns, test/p2.parquet):
#   French pseudo-labels from the stronger stage-2 model -> stage 1 -> cross-encoder on a wider
#   uncertainty band -> stage 2 -> France-strict file -> validate
# Usage: nohup bash run_v5.sh > /workspace/er/run_v5.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_v5}
FRANCE=${FRANCE:-0.9}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }

stage "1/5 French pseudo-labels from stage 2"; python -u src/pseudo.py --work-dir "$WORK_DIR" --probs p2
for u in trainA trainB test; do rm -f "$WORK_DIR/$u/ce.npy"; done   # stage 2 must not use stale scores
stage "2/5 stage 1";                           python -u src/train.py  --work-dir "$WORK_DIR" --pseudo
for u in trainA trainB test; do cp "$WORK_DIR/$u/p1.npy" "$WORK_DIR/$u/p1_sel.npy"; done
stage "3/5 cross-encoder (wider band)";        python -u src/ce.py     --work-dir "$WORK_DIR" --lo 0.003 --hi 0.997 --max-train 2500000
stage "4/5 stage 2";                           python -u src/refine.py --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"
stage "5/5 France cutoff $FRANCE"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "${OUT_DIR}_fr" --override "France=threshold:$FRANCE" --matching-only
python "$DATA_DIR/../utils/validate_submission.py" --matching "${OUT_DIR}_fr/matching_results.tsv" \
       --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)); submit ${OUT_DIR}_fr/matching_results.tsv"
