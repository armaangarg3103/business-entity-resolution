#!/bin/bash
# End-to-end pipeline: data -> blocking -> features -> model -> submission files.
# Usage:  bash run_all.sh                 (full run)
#         SAMPLE=0.02 bash run_all.sh     (quick smoke test on 2% of the data; not a valid submission)
# Settings (environment variables, all optional):
#   DATA_DIR     folder containing train/ and test/   (default: auto-detected under /workspace/er/data)
#   WORK_DIR     intermediate files                   (default: /workspace/er/work)
#   OUT_DIR      submission files                     (default: /workspace/er/output)
#   NUM_THREADS  CPU threads to use                   (default: all visible cores)
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${DATA_DIR:-}" ]; then
  f=$(find /workspace/er/data -name train_source1.tsv 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "Could not find train_source1.tsv under /workspace/er/data. Set DATA_DIR."; exit 1; }
  DATA_DIR=$(dirname "$(dirname "$f")")
fi
WORK_DIR=${WORK_DIR:-/workspace/er/work}
OUT_DIR=${OUT_DIR:-/workspace/er/output}
SAMPLE=${SAMPLE:-1.0}
echo "data=$DATA_DIR work=$WORK_DIR out=$OUT_DIR sample=$SAMPLE threads=${NUM_THREADS:-all}"

stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }
stage "1/5 prepare";  python -u src/prepare.py  --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" --sample "$SAMPLE"
stage "2/5 block";    python -u src/block.py    --work-dir "$WORK_DIR"
stage "3/5 features"; python -u src/features.py --work-dir "$WORK_DIR"
stage "4/5 train";    python -u src/train.py    --work-dir "$WORK_DIR"
stage "5/5 predict";  python -u src/predict.py  --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"

VALIDATOR="$DATA_DIR/../utils/validate_submission.py"
if [ -f "$VALIDATOR" ] && [ "$SAMPLE" = "1.0" ]; then
  stage "validate"
  python "$VALIDATOR" --matching "$OUT_DIR/matching_results.tsv" \
                      --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
fi
echo "done ($(date +%H:%M:%S))"
