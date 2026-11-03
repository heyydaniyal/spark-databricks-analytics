# To-Do List — Bank Churn Project

*Last updated: 3 Nov 2026, after Phase 4 modelling (GX_04). Source: agent session with Mariana.*

## GX_04 — Modelling (to finish later)

- [ ] **Re-add the final test evaluation (Step 9)** — removed on purpose for now. When it comes back:
  - freeze the decision on validation only first (winner: **Stacking GBT + RF**, validation AUC-PR 0.658);
  - score it **once** on the 1,501 test customers;
  - fill the Result cell with the real numbers (never guessed).
  - Note: two forward references to the test still exist meanwhile (title cell rule "The test set is opened once" + the Step 2 threshold paragraph) — they become consistent again once Step 9 returns.
- [ ] **Re-add the closing Summary section** — it was removed together with the test part.
- [ ] **Final QA** — read GX_04 top to bottom after Step 9 is back (numbers vs. charts vs. markdown flow).

## Data / environment

- [ ] **Restore the missing Unity Catalog tables.** The workspace UC was reset; only `workspace.churn.gold_churn_features` was rebuilt (verified identical to GX_02's recorded split: train 7,000 / val 1,499 / test 1,501, 0 overlap). Still missing: bronze (churn + transactions), silver (customers + transactions), gold transaction aggregates, quality report. Re-run GX_01 and GX_02 when the other phases need them.
- `workspace.churn.ml_tmp` — scratch volume for Spark ML cross-validation caching on serverless (`SPARKML_TEMP_DFS_PATH`). GX_04's setup cell re-creates it automatically; nothing to do here.

## Team / delivery

- [ ] **Tell the team before merging:** `src/ml.py` is a shared file and it gained `make_pipeline()` and `churn_metrics()` (team decision was to invest in this file, not a new module).
- [ ] **Commit and push:** GX_04_modelling, src/ml.py, this to-do list.

## Later phases

- [ ] **GX_05 (next phase):** reuse `make_pipeline` + `churn_metrics` from `src/ml.py` — that is why they were moved there. New customers get scored with exactly the same preparation and metrics as Phase 4.
- [ ] **Threshold decision (default 0.5):** a business decision based on the bank's retention costs — flagged in the modelling conclusions; not a modelling choice.

## Already done (for reference)

- Phase 4 modelling end-to-end: baselines, grid search tuning (LR / RF / GBT), model selection (GBT + RF), out-of-fold stacking (winner on validation), all executed with real numbers.
- `make_pipeline` + `churn_metrics` moved to `src/ml.py`; GX_04 refactored to import them; full re-run reproduced every number identically.