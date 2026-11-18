# Four-person GitHub commit plan

Use this as a work split for four teammates. Replace Person 1-4 with the real contributors, and have each person commit only work they actually implemented or verified. These are proposed commit messages, not commits that have already been made.

## Person 1 - Data, preprocessing, and exploration

**Owns:** `pathsnap/data.py`, `explore.py`, `artifacts/exploration/`, `DATASET_NOTES.md`

1. `feat(data): add trip readers and partition inventory`
   - Read `.track`, `.nodes`, `.arcs`, and `.route` safely.
   - Record missing/invalid observations, timestamp gaps, graph connectivity, and file sizes.
2. `feat(features): generate GPS-to-road candidate features`
   - Implement candidate generation and document the route-label and geographic assumptions.
3. `feat(eda): add Folium trip and road-network maps`
   - Generate the required exploratory maps and partition summaries from supplied data.
4. `docs(data): record dataset findings and limitations`
   - Add measured findings with the partition used for each analysis; label the partition for every reported result and keep evaluator results post-selection only.

**Done when:** another teammate can run the exploration commands and reproduce the committed inventory, summaries, and maps from the provided files.

## Person 2 - Model experiments and optimisation

**Owns:** `pathsnap/models.py`, `train.py`, `artifacts/model_comparison.csv`, `artifacts/bayes_trials.csv`, `artifacts/model_config.json`, model plots.

1. `feat(models): add baseline candidate classifiers`
   - Keep the model families and configurations in the training source.
2. `feat(training): add trip-grouped model comparison`
   - Train at least five distinct models; record seeds, fit/inference time, metrics, and confusion-matrix counts.
3. `feat(optimisation): add Bayesian search and retained trials`
   - Save the actual search space, trial results, selected parameters, and runtime.
4. `results(models): record measured student-test comparison`
   - Commit only results actually produced by the training script on permitted partitions. Include classifier-only and HMM comparisons where available.
5. `docs(models): explain selection and model limitations`
   - Base conclusions on the committed experiment outputs; do not add unrun numbers.

**Done when:** the comparison and search outputs can be regenerated from source, and the report's claims match those files.

## Person 3 - Route decoder and API

**Owns:** `pathsnap/decoder.py`, `pathsnap/service.py`, API configuration and server-side evaluator-data handling.

1. `feat(decoder): implement candidate-sequence Viterbi reconstruction`
   - Document states, emissions, transitions, GPS-gap behavior, and fallback behavior.
2. `feat(api): add model-loading and prediction endpoints`
   - Provide `/health`, `/trips`, and `/predict/{trip_id}`; load the committed model artifact/configuration.
3. `feat(api): keep evaluator labels server-side`
   - Read evaluator routes only on Laptop 1; return predictions and permitted result summaries without sending reference labels or route geometry to the browser.
4. `docs(api): document Laptop 1 configuration and launch`
   - Document environment variables, host/port, logs, and expected evaluator-data directory.

**Done when:** Laptop 1 can list the ten provided evaluator IDs and return route results through the API without exposing labels to Laptop 2. Confirm the ten IDs load from the server-side evaluator directory and predictions do not return reference labels.

## Person 4 - UI, deployment, and final integration

**Owns:** `ui/`, `run_ui.py`, `requirements.txt`, `.gitignore`, `README.md`, `REPORT_TEMPLATE.md`, screenshots, final report assembly.

1. `feat(ui): add Laptop 2 evaluator trip selector and result map`
   - Load trip options from Laptop 1, submit the selection, and show the returned route and metrics.
2. `feat(ui): add local trip preview for development`
   - Keep the local preview clearly separate from model inference and label supplied reference routes as reference data.
3. `build(deploy): add run instructions and dependency configuration`
   - Document Laptop 1 and Laptop 2 setup, network address configuration, and data/model paths.
4. `docs(report): assemble report from verified outputs`
   - Add screenshots, actual result tables, and Folium maps. Keep unsupported sections marked pending; do not manufacture demo screenshots or scores.
5. `chore(repo): prepare public submission`
   - Review the tree for secrets, unwanted raw data, generated caches, and large files before publication.

**Done when:** the UI runs on Laptop 2, the API runs on Laptop 1, and the README/report match the committed code and observed demonstration.

## Integration order

1. Person 1 commits data loading, candidate generation, and reproducible exploration.
2. Person 2 builds on Person 1's feature interface and commits real comparison and tuning outputs.
3. Person 3 uses the selected model artifact and decoder; keep all evaluator labels on Laptop 1.
4. Person 4 integrates the API and UI, writes run instructions, and assembles the report from committed evidence.
5. All four review the complete repository together before the team publishes it.

## GitHub checklist

- Commit the source, dependency/configuration files, run instructions, and experiment code.
- Commit model artifacts only if permitted by size limits; include their generation code and configuration.
- Include actual CSVs, logs, plots, confusion matrices, and search records used in the report.
- Do not commit passwords, API keys, or private credentials.
- Do not commit evaluator labels into the student UI or prediction pipeline.
- Do not invent authorship: use each teammate's own Git identity and commits.
- The public repository contains the trained artifact but no evaluator trip files or evaluator-derived scores. Keep evaluator files on Laptop 1 and capture the live two-laptop UI during the team demonstration.
