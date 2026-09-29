#!/bin/bash
# Builds <team>_submission.zip in the layout the challenge asks for:
#   output/matching_results.tsv, output/candidate_pairs.tsv,
#   code/business_entity_resolution/ (this repo), Documentation_template.md
# Usage: TEAM=my_team bash make_zip.sh
set -euo pipefail
cd "$(dirname "$0")"
TEAM=${TEAM:?set TEAM=<your team name>}
MATCH=${MATCH:-/workspace/er/output_final_fr99/matching_results.tsv}
CAND=${CAND:-/workspace/er/output_final/candidate_pairs.tsv}
DEST=${DEST:-/workspace/er/${TEAM}_submission.zip}
DATA_DIR=${DATA_DIR:-$(dirname "$(dirname "$(find /workspace/er/data -name train_source1.tsv | head -1)")")}
python "$DATA_DIR/../utils/validate_submission.py" --matching "$MATCH" --candidate "$CAND" --test-dir "$DATA_DIR/test"
python - "$DEST" "$MATCH" "$CAND" <<'PY'
import subprocess, sys, zipfile
dest, match, cand = sys.argv[1:4]
files = subprocess.check_output(["git", "ls-files"], text=True).split()
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(match, "output/matching_results.tsv")
    z.write(cand, "output/candidate_pairs.tsv")
    for f in files:
        z.write(f, f"code/business_entity_resolution/{f}")
    z.write("docs/Documentation_template.md", "Documentation_template.md")
print(f"written {dest} with {len(files)} code files")
PY
ls -la "$DEST"
