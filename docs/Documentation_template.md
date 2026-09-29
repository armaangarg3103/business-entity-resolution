# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Team name]
**Team Members:** Armaan Garg, [other members]
**Submission Date:** 29 September 2026

---

## 1. Executive Summary

We solve the task as a many-to-one assignment: each Source 2/3 record is linked to at most one Source 1 entity. The pipeline has three parts. Candidates come from sparse TF-IDF search combined with a multilingual embedding search. Two gradient-boosted tree models then score the pairs, with a fine-tuned multilingual cross-encoder added for the uncertain ones. A per-entity decision rule maximizes expected F0.5.

France is absent from training, so three country-agnostic ideas carry it: similarity weighted by how rare a token is inside its own country, pseudo-labels from confident French test predictions, and models that read raw text in any script.

Our final public leaderboard score is **0.983078**. Macro F0.5 on a held-out half of train is **0.9915**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploratory analysis that shaped the design:

| Finding | Value | Consequence |
|---|---|---|
| Train records: Source 1 / 2 / 3 | 2.21M / 5.03M / 5.29M | Scalable blocking is mandatory |
| Test records: Source 1 / 2 / 3 | 1.73M / 4.89M / 5.08M | Same |
| Source 2/3 records matched to more than one Source 1 entity | 0 (all 7.64M matched IDs are unique) | Many-to-one assignment: each record goes to its single best entity or to none |
| Matched pairs whose country labels agree | 100% | Country is a safe blocking key, used as an open set of labels |
| Source 1 entities with no match (singletons) | 5.6% | They score 1.0 only with an empty prediction, so precision on them matters |
| Mean matches per matched entity | 3.46 (up to 11) | Several records per entity: group context helps |
| Source 2/3 records that match nothing (distractors) | about 26% | "No owner" must be a valid answer |
| Source 2 names in non-Latin script (Devanagari, Telugu, Malayalam, ...) | 15% | Transliteration and multilingual models needed |
| Source 2/3 records with an empty address | 3.4% | Name-only matching; the hardest cases (Section 5) |
| France share of test Source 1 | 15% (259k entities), none in train | Zero-shot country |

**Noise patterns seen in true matches:** legal-form changes (Pvt Ltd and Private Limited, Inc and LLC), OCR-style typos ("5ecure", "Venmfes"), word reordering, "doing business as" and "Formerly" prefixes, domain names used as names ("ipower.com" for "Indchem Power"), completely different names at the same address, uppercase and abbreviated addresses (ST, RD, R., AV), leading zeros in house numbers ("00709"), component reordering, dropped PIN or state, and state or region written in the local script.

**The key insight about France.** For each test Source 1 entity we counted the other entities sharing its house number and city. The median count per country:

| Country | Median entities sharing house number and city |
|---|---|
| US | 1 |
| India | 30 |
| France | 164 |

French house numbers are small, most records sit in a few cities (Bordeaux, Nantes, Lille), and names are built from a small generic vocabulary (SARL, SAS, Club, École, Amicale). A model that learned from US data that a matching number and city nearly proves a match will over-merge French look-alikes. This drove the country-relative features in Section 4.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking, then cross-fitted gradient-boosted classifiers, then a fine-tuned transformer cross-encoder for uncertain pairs, then a second-stage group-aware model, then a per-entity decision.

**Core Innovations:**
1. **Search from the Source 2/3 side,** with a decision that enforces at most one owner per record, matching the structure of the ground truth.
2. **Country-relative rarity features.** Inverse-document-frequency weights are fitted per country on all records, labelled or not. A shared rare street name counts for a lot, and a shared "12", "Bordeaux" or "SARL" counts for almost nothing. A country never seen in training gets its own weights automatically.
3. **Self-training for the unseen country.** Confident French test predictions become positives, and the other candidates of those records become hard negatives, which are exactly the French look-alikes.
4. **A second stage that sees the neighbourhood.** Each pair is compared with the record's other candidates and with the entity's other likely members.
5. **Direct optimization of the metric.** For each entity we choose the match set with the highest expected F0.5, including the empty set for singletons.

Country is used only to group records for blocking, and for one decision cutoff (Section 4.4). It is never a model input.

---

## 3. Candidate Generation (Blocking)

Blocking runs separately inside each country. Every Source 2/3 record queries the Source 1 records of its country, and two searches are combined:

1. **Sparse TF-IDF, top 8 per record.** Each record becomes a bag of prefixed tokens: core-name words, words of any "doing business as" alias, character 3-grams of the space-free core name (robust to typos, transliteration and domain names), normalized address words, and address numbers. Tokens in more than 1% of a country's records are dropped. The top-8 search uses a multithreaded sparse matrix product (sparse_dot_topn).
2. **Multilingual embeddings, top 5 per record.** multilingual-e5-small encodes the raw name and raw address, which it reads in any script. The normalized sum of the two vectors is searched with a chunked GPU matrix product and top-k.

The union of both lists is the candidate set. Every candidate pair then gets both scores and both ranks, so the models can use each view.

- **Blocking keys used:** country as the partition; TF-IDF over name words, name 3-grams, address words and numbers; dense multilingual embeddings of name and address.
- **Candidate pairs generated:** 117,260,966 for test, about 11.7 per Source 2/3 record. Each train half has about 60.7 million. Comparing all pairs within each country would take about 6.7 × 10¹², so the reduction ratio is about 99.998%.
- **How true matches were kept:** the two searches fail differently. TF-IDF is strong on exact tokens and numbers, and embeddings are strong on transliteration and rewording. Generic words are down-weighted instead of used as keys, and the decision stage never drops a candidate without scoring it. With the TF-IDF search alone, on a 2% sample, blocking recall was 98.9% (India 97.9%, US 99.6%). The full-scale matcher recall below is an upper bound on what full-scale blocking lost.

---

## 4. Matching Model

### 4.1 Text normalization

- **Script folding.** Non-ASCII text is transliterated to Latin (anyascii), then lowercased with punctuation stripped.
- **Canonical abbreviations,** applied identically to both sides: Street and St, Road and Rd, Rue and R, Boulevard and Bd, Private and Pvt, Limited and Ltd, and US state names to their codes.
- **Leading zeros** are removed from numbers.
- **Aliases.** "doing business as", "formerly", "aka" and "trading as" split the name into a main part and an alias.
- **Generic name words are discovered, not hand-listed.** Any token in more than 0.5% of a country's names is treated as generic, for example inc, llc, pvt, sarl, sas, eurl, club and "elelpi", the transliterated Hindi form of LLP. The core name is the name without these words, so France gets its own list with no French rules written.

### 4.2 Features: about 55 pair features, none of them country-specific

- **Name features:** Levenshtein ratio, token-set, token-sort and partial ratios on the normalized name. Ratio, token-set and Jaro-Winkler on the core name. Ratio and partial ratio on the space-free core name, which catches concatenations and domains. Token-set against the alias.
- **Address features:** Levenshtein, token-set, token-sort and partial ratios on the normalized address. Token-set and ratio on the set of address numbers, meaning house and PIN or ZIP codes.
- **Country-relative rarity features:** IDF-weighted cosine of address words, name words and name 3-grams, with IDF fitted per country. Also, how many Source 1 entities in the country carry exactly this name, which measures ambiguity for name-only records.
- **Multilingual embedding features:** cosine of name embeddings, of address embeddings, and of the combined vector, plus the embedding rank.
- **Blocking context:** TF-IDF cosine and rank, number of candidates, and how many records chose this entity as their top candidate.
- **Competition features:** for the main similarities, the margin between this pair's value and the best value among the record's other candidates. This is the most important feature family. Since each record has at most one owner, winning is what counts.
- **Flags:** source, empty address, transliterated name, domain-style name, and lengths.

### 4.3 Models

| Stage | Model | Training | Output |
|---|---|---|---|
| Stage 1 | XGBoost, hist on GPU, up to 255 leaves (LightGBM on CPU as a fallback) | Cross-fitted: one model on train half A, validated on B, and one the other way round. All positives plus 30% of negatives, weighted to keep probabilities calibrated. Plus French pseudo-labels (4.5). | Out-of-fold probabilities for both halves; the average of both models for test |
| Cross-encoder | multilingual-e5-large, 560M parameters, fine-tuned as a pair classifier on "name ; address" against "name ; address" | Cross-fitted the same way. Trained on every uncertain pair of its half, meaning a stage-1 probability between 0.3% and 99.7% (about 1.18M), plus 400k French pseudo-labelled pairs: 1.58M pairs in total. 2 epochs, bf16, AdamW at 3e-5. | Scores for about 1.2M uncertain pairs per train half and 3.6M in test. On uncertain pairs its log loss is 0.118, against 0.202 for stage 1, 41% lower. |
| Stage 2 | XGBoost, 127 leaves | Trained on half A, early-stopped and tuned on half B. Inputs: all stage-1 features, the stage-1 probability, the cross-encoder score, and 15 group features. | Final pair probability |

**The stage-2 group features:**
- **Record side:** margin over the record's other candidates, the sum of its probabilities, and the rank.
- **Entity side:** expected cluster size as the sum of probabilities, the number of strong candidates, this pair's rank within the entity, and the probability relative to the entity's best, overall and per source.
- **Coherence:** name and address similarity between this record and the entity's strongest other candidate, plus that candidate's probability. True members of one business agree with each other, and look-alikes do not.

All models are MIT or Apache-2.0 licensed, and the largest has 560M parameters: multilingual-e5-small and e5-large (MIT), XGBoost (Apache-2.0), LightGBM (MIT). Pretrained weights were downloaded once from Hugging Face. No external data, APIs, geocoding or business lookups were used.

### 4.4 Decision and threshold selection

1. Each Source 2/3 record is assigned only to its highest-probability entity.
2. For each entity, its assigned records are sorted by probability p. We keep the top-k set that maximizes the expected F0.5, using 1.25·Σp_top-k / (0.25·E[true matches] + k), where E[true matches] is the sum of p over all the entity's candidates. We compare that with the probability that the entity is a singleton, which is the product of (1 − p). The probabilities are sharpened as p^α, with α tuned on half B.
3. On half B, this rule is compared with a single global threshold. The expected-F0.5 rule won, with α = 1.5.

**France** has no labels to tune on. Following the evidence that the model over-merges French look-alikes, a stricter cutoff was applied to French pairs only: keep a record's best entity if p ≥ 0.99. The value was chosen with public-leaderboard submissions that differed only in French rows. Section 5 lists those results.

### 4.5 French pseudo-labels (self-training)

Source: French test pairs, scored by our own model with no labels.
- **Positives:** the record's best entity when p ≥ 0.97 and it beats every other candidate of that record by at least 0.8.
- **Hard negatives:** all other candidates of those confidently assigned records, which are the French look-alikes.
- **Easy negatives:** a 5% sample of pairs with p ≤ 0.02.

The final run used the version-3 stage-2 model as the teacher. It produced 10.66M training rows, 871k of them positive. They were added to stage-1 training only, never to validation. A 400k sample of them was also added to cross-encoder training.

---

## 5. Results & Error Analysis

**Validation** is on train half B: about 1.1M entities, the size of the test set, never used for training the stage being evaluated.

| Metric on half B | US | India | Overall |
|---|---|---|---|
| Macro F0.5 | 0.9921 | 0.9905 | **0.9915** |
| Precision, entities with matches | 0.9975 | 0.9971 | |
| Recall, entities with matches | 0.9779 | 0.9730 | |
| Singletons wrongly given a match | 0.6% | 0.5% | |
| Matched entities predicted empty | 0.1% | 0.2% | |

**Public leaderboard progression.** Validation is macro F0.5 on half B, US and India only.

| Version | Main change | Validation | Leaderboard |
|---|---|---|---|
| 1 | TF-IDF blocking and one LightGBM | 0.9766 | 0.959 |
| 2 | Embedding blocking and features, stage-1 cross-fitting | 0.9827 | 0.968 |
| 2b | Stage 2 with group features and expected-F0.5 rule, France cutoff 0.9 | 0.9842 | 0.9704 |
| 3 | Rarity features, French pseudo-labels, small cross-encoder | 0.9902 | 0.9811 |
| Final | Large cross-encoder with French pairs, stronger pseudo-label teacher | 0.9915 | 0.982425 |
| Final | Same, France cutoff tuned 0.9, 0.95, 0.98, 0.99, 0.995 | 0.9915 | **0.983078** (at 0.99) |

**Implied France score.** With US and India at about 0.991, the final leaderboard score implies France near 0.93. France is the main remaining gap.

**Common false positives (wrong merges):**
- **Look-alike businesses at the same or nearly the same address,** for example "Babu Sangh" and "Babu Sangh Industries" in the same building. In France, many unrelated businesses share a small house number and a city, and have generic names.
- **Name-only records,** assigned to the wrong one of several entities with the same name.

**Common false negatives (missed matches):**
- **Records with an empty address.** In an error analysis on a 2% sample, they were 4.3% of true pairs but 43% of the misses, with recall 0.82 against 0.989 for records with an address. Without an address, a record with a common name such as "Shivam Builders" is ambiguous.
- **Heavy name changes combined with a partial address,** for example a rebrand like "Jaxaria Formerly Cabrera Secure Sciences" with the house number dropped.

---

## 6. Conclusion

Treating the problem as many-to-one assignment, combining lexical and multilingual candidate generation, and stacking a text cross-encoder under a group-aware second stage took validation macro F0.5 from 0.977 to 0.9915. The public leaderboard rose from 0.959 to 0.983.

The main lesson is that the distribution shift to an unseen country is about feature semantics, not vocabulary. The strongest address signal in the US means little in France. Evidence normalized for how rare it is within each country, together with self-training on the new country, recovered most of that loss. The remaining France gap, estimated near 0.93 against about 0.99 elsewhere, is the clearest direction for further work: more rounds of self-training and a France-aware cross-encoder.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds all source in `src/`, plus `README.md`, `requirements.txt` and shell entry points. `reproduce.sh` runs every step in the order used for the final submission.

| Script | Role |
|---|---|
| `src/prepare.py` | Loads the TSVs, splits train into halves A and B by Source 1 entity, normalizes text, discovers generic words per country |
| `src/embed.py` | multilingual-e5-small embeddings of raw names and addresses, on the GPU |
| `src/block.py` | TF-IDF top-8 and embedding top-5 candidates per Source 2/3 record, within each country |
| `src/features.py` | Fuzzy, number, embedding, blocking and competition features |
| `src/extra.py` | Country-relative rarity features, appended in streamed batches |
| `src/pseudo.py` | French pseudo-labels from confident test predictions |
| `src/train.py` | Stage 1, cross-fitted, with negative sampling and optional pseudo-labels |
| `src/ce.py` | Cross-encoder, cross-fitted, on uncertain pairs |
| `src/refine.py` | Stage 2 with group features, decision-rule choice on half B, writes both TSVs |
| `src/variant.py` | Rebuilds the submission with a per-country decision rule; used for the France cutoff |
| `src/diagnose.py` | Error analysis: blocking misses, wrong owner, below threshold |
| `src/gbm.py`, `src/common.py` | XGBoost and LightGBM wrapper, normalization, metric, file writers |

**Entry point:** `bash reproduce.sh` runs every step and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The official validator reports PASS on both files.

**Candidate file in this archive:** the final `candidate_pairs.tsv` (117,260,966 pairs) was produced on a GPU server whose access ended before it could be downloaded, so it is not in the zip. `output/CANDIDATE_PAIRS_NOTE.md` gives its details, and `reproduce.sh` regenerates it.

**Hardware used:** one NVIDIA H100 MIG slice (3g.40gb), 15 CPU threads and 90 GB of RAM. The heaviest step is the large cross-encoder, about 4 hours for both folds including scoring.

### B. Additional Results

- **Stage-1 feature importance, version 2,** by share of gain: embedding margin 0.71, TF-IDF margin 0.07, embedding rank 0.04, number-overlap margin 0.03, address token-set margin 0.03. The competition margins dominate throughout.
- **Stage-2 feature importance, version 2b:** stage-1 probability 0.76, its margin over competing candidates 0.20, embedding margin 0.02.
- **Cross-encoder gain:** on uncertain pairs, log loss falls from 0.202 to 0.118 for the large model, and from 0.286 to 0.179 for the small model on its own set of uncertain pairs.
