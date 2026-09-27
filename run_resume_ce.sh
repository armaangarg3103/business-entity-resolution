#!/bin/bash
# Switch cross-encoder mid-run WITHOUT redoing stage 1: use after killing run_final.sh during step 3.
# Usage: BACKBONE=intfloat/multilingual-e5-base nohup bash run_resume_ce.sh > /workspace/er/run_resume.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_final}
FRANCE=${FRANCE:-0.9}
BACKBONE=${BACKBONE:-intfloat/multilingual-e5-base}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-2}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }
for u in trainA trainB test; do rm -f "$WORK_DIR/$u/ce.npy"; done
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
