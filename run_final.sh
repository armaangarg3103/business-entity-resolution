#!/bin/bash
# Final run on top of version 3 (its work dir: features with specificity columns, test/p2.parquet):
#   1 French pseudo-labels from the version-3 stage-2 model (strong teacher)
#   2 stage 1 retrained with them
#   3 cross-encoder (BACKBONE, default multilingual-e5-large, MIT) on all uncertain pairs + French pairs
#   4 stage 2 -> 5 France-strict file -> validate
# Usage: BACKBONE=intfloat/multilingual-e5-large nohup bash run_final.sh > /workspace/er/run_final.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_final}
FRANCE=${FRANCE:-0.9}
BACKBONE=${BACKBONE:-intfloat/multilingual-e5-large}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-2}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }

stage "1/5 French pseudo-labels from version-3 stage 2"
python -u src/pseudo.py --work-dir "$WORK_DIR" --probs p2
for u in trainA trainB test; do rm -f "$WORK_DIR/$u/ce.npy"; done   # never mix old scores with the new stage 1
stage "2/5 stage 1"
python -u src/train.py --work-dir "$WORK_DIR" --pseudo
for u in trainA trainB test; do cp "$WORK_DIR/$u/p1.npy" "$WORK_DIR/$u/p1_sel.npy"; done
stage "3/5 cross-encoder $BACKBONE, $EPOCHS epochs"
python -u src/ce.py --work-dir "$WORK_DIR" --backbone "$BACKBONE" --lo 0.003 --hi 0.997 --max-train 4000000 \
       --epochs "$EPOCHS" --batch "$BATCH" --lr 3e-5 --pseudo-file pseudo.parquet
stage "4/5 stage 2"
python -u src/refine.py --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"
stage "5/5 France cutoff $FRANCE"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "${OUT_DIR}_fr" --override "France=threshold:$FRANCE" --matching-only
python "$DATA_DIR/../utils/validate_submission.py" --matching "${OUT_DIR}_fr/matching_results.tsv" \
       --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)); submit ${OUT_DIR}_fr/matching_results.tsv"
