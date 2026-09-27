#!/bin/bash
# Round 3 on top of round 2 (run AFTER run_v5.sh has finished; reuses its stage 1 and p1_sel):
#   French pseudo-pairs from the round-2 stage-2 model -> larger cross-encoder (multilingual-e5-base,
#   MIT) trained on train pairs + French pseudo pairs -> stage 2 -> France-strict file -> validate
# Usage: nohup bash run_v6.sh > /workspace/er/run_v6.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output_v6}
FRANCE=${FRANCE:-0.9}
BACKBONE=${BACKBONE:-intfloat/multilingual-e5-base}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }

stage "1/4 French pseudo-pairs from round-2 stage 2"
python -u src/pseudo.py --work-dir "$WORK_DIR" --probs p2 --out pseudo_ce.parquet
stage "2/4 cross-encoder $BACKBONE"
python -u src/ce.py --work-dir "$WORK_DIR" --backbone "$BACKBONE" --lo 0.003 --hi 0.997 \
       --max-train ${MAX_TRAIN:-1500000} --batch 128 --lr 3e-5 --pseudo-file pseudo_ce.parquet
stage "3/4 stage 2"; python -u src/refine.py --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"
stage "4/4 France cutoff $FRANCE"
python -u src/variant.py --work-dir "$WORK_DIR" --out-dir "${OUT_DIR}_fr" --override "France=threshold:$FRANCE" --matching-only
python "$DATA_DIR/../utils/validate_submission.py" --matching "${OUT_DIR}_fr/matching_results.tsv" \
       --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
echo "done ($(date +%H:%M:%S)); submit ${OUT_DIR}_fr/matching_results.tsv"
