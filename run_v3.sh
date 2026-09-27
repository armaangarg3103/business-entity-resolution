#!/bin/bash
# Version 3 on top of an existing full run in WORK_DIR (records, embeddings, features already built):
#   specificity features -> French pseudo-labels -> stage 1 -> stage 2 -> France-strict variant -> validate
# Usage: nohup bash run_v3.sh > /workspace/er/run_v3.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_v3}
FRANCE=${FRANCE:-0.9}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }

stage "1/5 specificity features"; python -u src/extra.py   --work-dir "$WORK_DIR"
stage "2/5 French pseudo-labels"; python -u src/pseudo.py  --work-dir "$WORK_DIR"
stage "3/5 stage 1";              python -u src/train.py   --work-dir "$WORK_DIR" --pseudo
stage "4/5 stage 2";              python -u src/refine.py  --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"
stage "5/5 France cutoff $FRANCE"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "${OUT_DIR}_fr" --override "France=threshold:$FRANCE" --matching-only
python "$DATA_DIR/../utils/validate_submission.py" --matching "${OUT_DIR}_fr/matching_results.tsv" \
       --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)); submit ${OUT_DIR}_fr/matching_results.tsv"
