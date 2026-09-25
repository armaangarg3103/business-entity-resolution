# Business Entity Resolution (Amazon ML Challenge 2026)

Finds, for every Source 1 business, all matching Source 2 / Source 3 records.
The pipeline runs in five stages: normalize, block, featurize, LightGBM, then decide.
It is tuned for the challenge metric, which is macro F0.5 per Source 1 entity.

## Quick start (GPU server)

```bash
# 1. get the code
cd /workspace/er/code
git clone https://github.com/armaangarg3103/business-entity-resolution.git business_entity_resolution
cd business_entity_resolution

# 2. install dependencies (re-run after a pod restart)
bash setup.sh

# 3. smoke test on 2% of the data (a few minutes)
SAMPLE=0.02 WORK_DIR=/workspace/er/work_small OUT_DIR=/workspace/er/output_small bash run_all.sh

# 4. full run in the background, so closing the browser does not kill it
nohup bash run_all.sh > /workspace/er/run.log 2>&1 &
tail -f /workspace/er/run.log
```

The dataset folder, the one holding `train/` and `test/`, is found automatically under `/workspace/er/data`.
Set `DATA_DIR` to point elsewhere.
The thread count is detected from the pod's CPU limit. Set `NUM_THREADS` to override it.

To get new code later, run `git pull` inside the repo folder.

## Outputs

| File | Meaning |
|---|---|
| `output/matching_results.tsv` | final matches; this is what you upload to the leaderboard |
| `output/candidate_pairs.tsv` | every pair the model scored, a superset of the matches |
| `work/decision.json` | tuned threshold and validation F0.5, overall and per country |
| `work/model.txt` | trained LightGBM model |

After a full run, `run_all.sh` runs the official validator automatically.

## How it works

| Stage | Script | What it does |
|---|---|---|
| 1 | `src/prepare.py` | Splits train by S1 entity into two halves, A and B, each about the size of the test set. Normalizes text: ascii-folds any script, lowercases, canonicalizes abbreviations, extracts address numbers, and splits DBA / "formerly" aliases. Generic name words such as inc, pvt or sarl are **discovered per country from word frequency**, not hand-listed, so France works without French training data. |
| 2 | `src/block.py` | Each S2/S3 record keeps its top-K most similar S1 records of the same country. The search uses sparse TF-IDF over name words, name character trigrams, address words and numbers. Reports blocking recall on the train halves. |
| 3 | `src/features.py` | About 40 country-independent features: fuzzy name and address scores, number overlap, the blocking score and rank, and the **margin over the best competing candidate** of the same record. |
| 4 | `src/train.py` | Trains LightGBM on half A with early stopping on half B. Picks the probability threshold that maximizes macro F0.5 on B. |
| 5 | `src/predict.py` | Scores the test pairs. Each S2/S3 record is linked only to its single best S1, and only above the threshold, because ground truth never links one record to two entities. Writes both TSVs. |

Country is used only to group records for blocking. It is never a model feature, so unseen countries work.
No external data, APIs or lookups are used.

## Tuning knobs

| Setting | Where | Effect |
|---|---|---|
| `--top-k` (default 8) | `block.py` | more candidates means higher recall ceiling, but more pairs to score |
| `--max-df` (default 0.01) | `block.py` | drops very common tokens; lower is faster, higher gives more recall |
| `--threshold` | `predict.py` | overrides the tuned threshold; higher means more precision |

## Roadmap

1. Embedding blocking on the GPU with multilingual-e5-small, merged with the TF-IDF candidates.
2. Fine-tune the encoder contrastively on the training pairs.
3. Cross-source features, for example an S2 record that strongly matches an S3 record already linked to the same S1.
4. Per-entity decision that maximizes expected F0.5, instead of one global threshold.
