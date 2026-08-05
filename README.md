# smart-drive-failure-prediction

Predicting seven-day hard-drive failures from Backblaze SMART telemetry.

Given a drive's previous 30 days of daily SMART readings, predict whether it fails within the next 7 days.

## Status

| Component | File | State |
|---|---|---|
| Ingestion + labelling | [src/data_pipeline.py](src/data_pipeline.py) | Implemented, tested |
| Trailing-window features | [src/features.py](src/features.py) | Implemented, tested |
| Cross-sectional experiment | [src/experiment.py](src/experiment.py) | Implemented, run — underpowered, kept as a negative result |
| Drive-day panel experiment | [src/panel_experiment.py](src/panel_experiment.py) | Implemented, run |
| Training / evaluation modules | [src/train.py](src/train.py), [src/evaluate.py](src/evaluate.py) | Scaffolds — `NotImplementedError` |
| Inference API | [api/main.py](api/main.py) | Boots; scoring routes return 501 |
| SHAP explanations | — | Not started |
| Tests | [tests/](tests/) | 108 passing |

## Layout

```
data/raw/            Backblaze daily CSVs or quarterly .zip (git-ignored)
data/processed/      labelled and feature tables
artifacts/           models, metrics, metadata (git-ignored)
src/                 pipeline, features, experiments
api/                 FastAPI service
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Download the daily files from [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Quarterly ZIPs are read in place — no need to unzip.

## Usage

```bash
# Label a directory of daily CSVs
python -m src.data_pipeline --input-dir data/raw --output data/processed/labelled.parquet

# Trailing-window features
python -m src.features --labelled data/processed/labelled.parquet

# Drive-day panel experiment, straight from the archive
python -m src.panel_experiment --archive ~/Downloads/data_Q1_2026.zip

pytest
```

## Problem framing

**Label.** A row at date `t` is positive when the drive's failure date falls in `[t+1, t+7]`.

**Features.** For each SMART attribute: latest, mean, min, max, standard deviation, change and observation count over the 30 days before `t`. The panel experiment uses `[t-30, t-1]`, excluding `t` itself. Windows are defined in time rather than rows, because drives skip days.

**Exclusions.** Rows are dropped when either window runs past the edge of the data, when the drive has already failed, or when survival across the future window cannot be verified. A drive that disappears is not a drive that survived.

**Split.** Chronological for the panel design — a drive contributes many autocorrelated rows, so a random split leaks. The cross-sectional design aggregates to one row per drive and can use a stratified split safely.

**Imbalance.** `class_weight="balanced"` for logistic regression, `balanced_subsample` for the forest, `scale_pos_weight` for XGBoost, all from the training fold. Test folds keep their natural failure rate.

**Metrics.** Accuracy is not reported; the positive class is ~0.02%, so predicting "healthy" everywhere scores 99.98%. False alerts per 1,000 healthy uses truly-healthy drive-days as the denominator, so it is comparable across thresholds. Event-level recall counts a failure as detected if any alert fires in its 7-day pre-failure window.

## Results

Backblaze Q1 2026, read from `data_Q1_2026.zip`. 30,597,484 drive-day rows, 351,095 drives, 1,030 failure events. 53 eligible prediction dates (2026-01-31 to 2026-03-24) split chronologically; test is 2026-03-13 to 2026-03-24.

Test fold: 4,107,703 drive-days, 850 positive labels, **197 failure events**, base rate 0.000207.

| model | PR-AUC | lift | event recall | 95% CI | alerts/day | false alerts per 1,000 healthy |
|---|---|---|---|---|---|---|
| logistic regression | 0.0185 | 90× | 0.223 | 0.171–0.286 | 307 | 0.86 |
| random forest | 0.0469 | 227× | 0.254 | 0.198–0.319 | 223 | 0.60 |
| xgboost | 0.0558 | 270× | 0.213 | 0.162–0.276 | 125 | 0.32 |
| rule baseline | — | — | 0.761 | 0.697–0.816 | 10,231 | 29.74 |

Rule baseline: `smart_197_raw_max > 0 OR smart_5_raw_change > 0 OR smart_198_raw_max > 0`.

**Model selection is inconclusive.** The test fold has enough events (197) to attempt selection, but random forest's event-recall interval contains logistic regression's point estimate. PR-AUC ranks XGBoost first while event recall ranks the forest first. All three models are saved; none is designated best.

**The rule baseline's higher recall is not a like-for-like win.** It fires 10,231 alerts per day against XGBoost's 125 — 82× the operational load for 3.6× the failures caught. Per alert, XGBoost catches 28.1 events per 1,000 alerts against the rule's 1.2. Comparing recall at matched alert budgets is the outstanding piece of analysis.

**Every model degrades from validation to test**: −25% (logistic regression), −37% (forest), −48% (XGBoost). A random split would have hidden this. Single-fold estimates here overstate next month's performance.

## Known issues

- Logistic regression's selected threshold is `0.99999994`, the last float32 value below 1.0. It fires, but its operating point sits at the edge of float precision; the classifier needs calibrating.
- Training uses 4% negative downsampling (420,450 of 10,454,939 rows). Validation and test metrics come from complete folds, but the models' probability scale reflects the sampled prior.
- `train.py` and `evaluate.py` remain scaffolds while the experiments live in separate modules.

## Testing

```bash
pytest
```

108 tests against synthetic fixtures, not Backblaze data, so expected counts are hand-computable. Coverage includes the leakage properties: that features never see the prediction date, that truncating the dataset does not change earlier features, and that already-failed and unverifiable rows are excluded.
