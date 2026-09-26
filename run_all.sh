#!/bin/bash
# End-to-end pipeline: data -> blocking -> features -> model -> submission files.
# Usage:  bash run_all.sh                 (full run)
#         SAMPLE=0.02 bash run_all.sh     (quick smoke test on 2% of the data; not a valid submission)
# Settings (environment variables, all optional):
#   DATA_DIR     folder containing train/ and test/   (default: auto-detected under /workspace/er/data)
#   WORK_DIR     intermediate files                   (default: /workspace/er/work)
#   OUT_DIR      submission files                     (default: /workspace/er/output)
#   NUM_THREADS  CPU threads to use                   (default: the pod's CPU limit)
#   USE_EMB      1 = add GPU multilingual embeddings  (default: 1; 0 = TF-IDF only baseline)
#   SKIP_PREP    1 = reuse prepared records from an earlier run in WORK_DIR
# To rerun only the modelling after a code change:
#   python -u src/train.py --work-dir W && python -u src/refine.py --work-dir W --out-dir O
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
USE_EMB=${USE_EMB:-1}
THREADS=$(cd src && python -c "from common import N_THREADS; print(N_THREADS)")
echo "data=$DATA_DIR work=$WORK_DIR out=$OUT_DIR sample=$SAMPLE threads=$THREADS"

stage() { echo; echo "===== $1  ($(date +%H:%M:%S))"; }
if [ "${SKIP_PREP:-0}" = "1" ]; then
  echo "reusing prepared records in $WORK_DIR"
else
  stage "1/6 prepare"; python -u src/prepare.py --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" --sample "$SAMPLE"
fi
if [ "$USE_EMB" = "1" ]; then
  stage "2/6 embed (GPU)"; python -u src/embed.py --work-dir "$WORK_DIR"
  EMB_K=5
else
  echo "skipping embeddings (USE_EMB=0)"; EMB_K=0
fi
stage "3/6 block";    python -u src/block.py    --work-dir "$WORK_DIR" --emb-k "$EMB_K"
stage "4/6 features"; python -u src/features.py --work-dir "$WORK_DIR"
stage "5/6 train (stage 1, cross-fitted)"; python -u src/train.py --work-dir "$WORK_DIR"
stage "6/6 refine (stage 2) + write submission"; python -u src/refine.py --work-dir "$WORK_DIR" --out-dir "$OUT_DIR"

VALIDATOR="$DATA_DIR/../utils/validate_submission.py"
if [ -f "$VALIDATOR" ] && [ "$SAMPLE" = "1.0" ]; then
  stage "validate"
  python "$VALIDATOR" --matching "$OUT_DIR/matching_results.tsv" \
                      --candidate "$OUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
fi
echo "done ($(date +%H:%M:%S))"
