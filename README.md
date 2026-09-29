# Business Entity Resolution (Amazon ML Challenge 2026)

For every Source 1 business, the pipeline finds all matching Source 2 and Source 3 records. The final submission scored **0.983078** on the public leaderboard, with a validation macro F0.5 of **0.9915**. The full method is in `docs/Documentation_template.md`.

## Reproduce the final submission

```bash
# 1. dataset: unzip the challenge data anywhere under /workspace/er/data (or set DATA_DIR)
# 2. dependencies (NVIDIA NGC PyTorch container, Python 3.10)
bash setup.sh
# 3. everything, in the order used for the final submission (several hours on one H100 slice)
nohup bash reproduce.sh > /workspace/er/reproduce.log 2>&1 &
```

The last step writes `/workspace/er/output/matching_results.tsv` and `/workspace/er/output/candidate_pairs.tsv`, then runs the official validator on them. GPU training is not bit-for-bit deterministic, so a rerun gives very close but not identical files.

`reproduce.sh` chains these scripts:

| Step | Script | What it produces |
|---|---|---|
| A | `run_all.sh` | Prepared records, embeddings, candidates, features, and the version-2 models |
| B | `src/ce.py` with its default settings | Small cross-encoder scores, used by version 3 |
| C | `run_v3.sh` | Rarity features, French pseudo-labels, stage 1, stage 2 (version 3) |
| D | `run_final.sh` | Stronger pseudo-labels, large cross-encoder, stage 2 (final model) |
| E | `src/variant.py` with France at 0.99 | The submitted files |

To package the submission zip after a run, use `TEAM=<team_name> bash make_zip.sh`.

## Pipeline

| Script | Role |
|---|---|
| `src/prepare.py` | Loads the TSVs, splits train into halves A and B by Source 1 entity, each about the size of test. Normalizes text: script folding, abbreviations, numbers, aliases. Discovers generic name words per country. |
| `src/embed.py` | multilingual-e5-small embeddings of raw names and addresses, on the GPU |
| `src/block.py` | Per country: TF-IDF top 8 plus embedding top 5 Source 1 candidates for each Source 2/3 record |
| `src/features.py` | Fuzzy name and address scores, number overlap, embedding cosines, blocking score and rank, margin over competing candidates |
| `src/extra.py` | Country-relative rarity features: IDF-weighted cosines fitted per country, and same-name ambiguity counts |
| `src/pseudo.py` | French pseudo-labels from confident test predictions, since France has no training labels |
| `src/train.py` | Stage 1: XGBoost, cross-fitted over A and B, with negative sampling and optional pseudo-labels |
| `src/ce.py` | Cross-encoder: multilingual-e5 fine-tuned on text pairs, cross-fitted, scoring only uncertain pairs |
| `src/refine.py` | Stage 2 with group features. Picks the decision rule on B: expected-F0.5 per entity, or a global threshold. Writes both TSVs. |
| `src/variant.py` | Rebuilds the submission with a different decision rule for chosen countries |
| `src/diagnose.py` | Error analysis on a train half |
| `src/gbm.py`, `src/common.py` | XGBoost on GPU with LightGBM fallback; normalization, metric and output helpers |

Country is used only to group records for blocking and for the France cutoff. It is never a model input. No external data, APIs or lookups are used. All models are MIT or Apache-2.0 licensed and at most 560M parameters.

## Settings

| Environment variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | found under `/workspace/er/data` | folder containing `train/` and `test/` |
| `WORK_DIR` | `/workspace/er/work` | intermediate files |
| `NUM_THREADS` | the pod's CPU limit | CPU threads |
| `GBM_BACKEND` | `auto` (XGBoost if a GPU is visible) | `xgb` or `lgb` |
| `BACKBONE` | `intfloat/multilingual-e5-large` in `run_final.sh` | cross-encoder model |
