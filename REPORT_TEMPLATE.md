# PathSnap technical report: evidence template

> Complete this report only after running the submitted scripts. Do not replace the marked fields with estimates. The supplied workspace includes the training and student-test partitions; it does not include evaluator-test trips.

## 1. Solution summary

Describe the implemented system in your own words, including the API/UI laptop split and the team’s contributions. Disclose AI assistance under the event policy.

## 2. Dataset understanding and exploration

Run `python explore.py --data "../Student folder"`. Use `artifacts/exploration/partition_summary.csv`, `raw_inventory.csv`, `sample_trip_summary.json`, `dataset_exploration.png`, and `sample_trajectory.html` plus its static map export. State which partition each analysis uses. Review missing values, file sizes, weak component counts, isolated nodes, and timestamp gaps.

## 3. Candidate generation and features

Explain the nearest-edge candidate generator and six features in `pathsnap/models.py`. Labels are derived from the supplied route by selecting its geometrically nearest segment. Discuss midpoint-based candidate retrieval, route-label ambiguity, sampling gaps, and per-trip graph identifiers.

## 4. Experiments

Run `python train.py --data "../Student folder"`. Use the actual rows from `artifacts/model_comparison.csv` and `artifacts/bayes_trials.csv`. The script compares five classifiers with five-fold GroupKFold by trip, then tunes the leading CV model with Bayesian optimisation. Report the random seed, runtime, package versions, the number of completed trials, and any interrupted runs.

| Model | CV macro F1 mean ± SD | Candidate precision | Candidate recall | Candidate macro F1 | Candidate log loss | Candidate miss rate | Top-1 accuracy | HMM Top-1 | Route continuity | Fit time | Inference time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Populate from model_comparison.csv | | | | | | | | | | | |

Explain each attempted model’s fit to this task, strengths, limitations, objective, initial configuration, and student-test results. Justify selection using the measured outputs, not training scores alone.

## 5. Bayesian optimisation

Include the actual search space, F1 objective, GroupKFold grouping, best parameters, trial table, elapsed time, and tuned-versus-untuned student-test comparison. Discuss search limitations.

## 6. HMM and Viterbi route reconstruction

Explain the candidate edge state, classifier probability emission, directed endpoint transition score, gap relaxation above 30 seconds, recovery behavior, and Viterbi recurrence in `pathsnap/decoder.py`. Use the measured classifier-only versus HMM comparison and continuity analysis from `artifacts/`.

## 7. API, UI, and deployment

Include screenshots of the running UI and API logs from two laptops. Record actual successful requests, environment configuration, network address, and setup. Explain how ground-truth labels remain inside the API process.

## 8. Limitations and conclusion

Use observed failures, data-quality findings, and experimental evidence. Do not claim results that were not run.

The generated development report is `artifacts/PathSnap_Technical_Report.pdf`. The team must add the actual evaluator demo evidence after the reserved partition is provided and make the required AI-assistance disclosure.
