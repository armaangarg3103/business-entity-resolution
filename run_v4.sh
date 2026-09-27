#!/bin/bash
# Version 4 = version 3 + cross-encoder. Run AFTER run_v3.sh has finished and ce.py has finished:
#   stage 2 again (now with cross-encoder scores) -> France-strict variant -> validate
# Usage: nohup bash run_v4.sh > /workspace/er/run_v4.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_v4}
FRANCE=${FRANCE:-0.9}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
for u in trainA trainB test; do
  [ -f "$WORK_DIR/$u/ce.npy" ] || { echo "missing $WORK_DIR/$u/ce.npy - run src/ce.py first"; exit 1; }
done
echo "===== stage 2 with cross-encoder ($(date +%H:%M:%S))"
python -u src/refine.py --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"
echo "===== France cutoff $FRANCE ($(date +%H:%M:%S))"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "${OUT_DIR}_fr" --override "France=threshold:$FRANCE" --matching-only
python "$DATA_DIR/../utils/validate_submission.py" --matching "${OUT_DIR}_fr/matching_results.tsv" \
       --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)); submit ${OUT_DIR}_fr/matching_results.tsv"
