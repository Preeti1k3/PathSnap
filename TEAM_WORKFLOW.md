# Four-person GitHub workflow

Use one shared repository and separate branches. Replace Person 1-4 with the teammates' real names. These are proposed messages, not commits that already exist. Each person should commit work they personally implemented or ran.

## Repository folders and owners

```text
pathsnap/                         # GitHub repository root
├── README.md                     # P4: install, run, two-laptop guide
├── requirements.txt              # P4: dependencies
├── .gitignore                    # P4: local files and secrets
├── DATASET_NOTES.md              # P1: observed dataset counts/findings
├── REPORT_TEMPLATE.md            # P4: complete from evidence
├── TEAM_COMMIT_PLAN.md            # task summary
├── TEAM_WORKFLOW.md               # this guide
├── explore.py                     # P1: exploration and Folium maps
├── train.py                       # P2: model comparison and Bayesian search
├── run_ui.py                      # P4: Laptop 2 UI server
├── pathsnap/
│   ├── data.py                    # P1: readers, cleaning, features
│   ├── models.py                  # P2: model definitions
│   ├── decoder.py                 # P3: HMM/Viterbi
│   └── service.py                 # P3: Laptop 1 API/model loading
├── ui/index.html                  # P4: evaluator UI/local preview
└── artifacts/
    ├── exploration/               # P1: inventory, summaries, Folium maps
    ├── model_comparison.csv       # P2: actual run output
    ├── bayes_trials.csv           # P2: actual search record
    ├── model_config.json          # P2: selected configuration
    ├── tuned_vs_untuned_student_test.csv # P2: measured comparison
    └── model.joblib               # P2 creates; P3 consumes
```

Measured training and student-test summaries are present under `artifacts/`. Keep raw supplied data outside the checkout. Never commit evaluator files or evaluator-derived scores.

## Person 1 - data, preprocessing, features, exploration

Owns `pathsnap/data.py`, `explore.py`, `DATASET_NOTES.md`, `artifacts/exploration/`.

```text
feat(data): add trip readers and partition inventory
feat(features): generate GPS-to-road candidate features
feat(eda): add Folium trip and road-network maps
docs(data): record measured findings and limitations
```

Commit reproducible outputs from supplied partitions; name the partition and state assumptions. Do not use evaluator trips to tune or select models; keep evaluator results on Laptop 1 and out of the public repository.

## Person 2 - models, experiments, optimisation

Owns `pathsnap/models.py`, `train.py`, and generated model/results under `artifacts/`.

```text
feat(models): add baseline candidate classifiers
feat(training): compare models with trip-grouped cross-validation
feat(optimisation): add Bayesian search and retain trial results
results(models): add measured student-test comparison
docs(models): explain selection using recorded results
```

Run at least five distinct models. Retain actual configurations, metrics, timings, confusion-matrix values, and search trials. Make report claims match generated files. Record incomplete runs honestly.

## Person 3 - decoder and Laptop 1 API

Owns `pathsnap/decoder.py`, `pathsnap/service.py`.

```text
feat(decoder): add Viterbi route reconstruction
feat(api): load model artifact and expose prediction endpoints
feat(api): score routes server-side and protect reference labels
docs(api): document Laptop 1 launch and configuration
```

Keep evaluator labels and scoring on Laptop 1. The browser receives trip IDs, predictions, and approved metrics, never evaluator ground truth. The ten-trip route is implemented; capture a real two-laptop UI screenshot during the team demonstration.

## Person 4 - UI, deployment, report

Owns `ui/`, `run_ui.py`, `requirements.txt`, `.gitignore`, `README.md`, `REPORT_TEMPLATE.md`, `TEAM_WORKFLOW.md`.

```text
feat(ui): add evaluator trip selector and route results view
feat(ui): add clearly labeled local trip preview
build(deploy): document two-laptop setup and dependencies
docs(report): assemble report from verified experiment outputs
chore(repo): prepare public submission and remove accidental data
```

Use screenshots from the running API/UI. Do not present the local reference-route preview as model inference.

## Create the GitHub repository once

One teammate creates an empty public GitHub repository and adds the other three as collaborators. Avoid creating a README or `.gitignore` on GitHub if pushing this existing project as the initial commit.

From the parent directory that contains this project, the repository owner runs:

```bash
cd pathsnap
git init -b main
git add .
git status --short
git commit -m "chore(repo): add PathSnap project scaffold"
git remote add origin https://github.com/TEAM-ORG/REPOSITORY.git
git push -u origin main
```

Replace `TEAM-ORG/REPOSITORY` with the real repository path. Check `.gitignore` before staging; never add raw evaluator data, local credentials, virtual environments, or caches.

## Each teammate's branch and commits

Each person clones the same repository and configures their own Git identity. Person 1 example:

```bash
git clone https://github.com/TEAM-ORG/REPOSITORY.git
cd REPOSITORY
git switch -c feat/data-eda
git config user.name "YOUR REAL NAME"
git config user.email "YOUR GITHUB EMAIL"
```

Branch names:

```text
feat/data-eda
feat/model-experiments
feat/decoder-api
feat/ui-deployment
```

Stage only files you own, inspect the staged diff, commit, push, and open a pull request to `main`. Example:

```bash
git add pathsnap/data.py explore.py DATASET_NOTES.md artifacts/exploration/
git diff --cached
git commit -m "feat(data): add trip readers and partition inventory"
git push -u origin feat/data-eda
```

Have another teammate review each pull request. Coordinate before editing shared files. After merges, update your branch:

```bash
git fetch origin
git switch YOUR-BRANCH
git rebase origin/main
```

If needed, use `git push --force-with-lease` only on your own feature branch, never on `main`.

## Merge order and two-laptop integration

1. Merge Person 1's readers, candidate features, and exploration.
2. Person 2 updates from `main`, runs real experiments, and merges source plus actual results.
3. Person 3 consumes the selected artifact and merges decoder/API code.
4. Person 4 integrates the UI, deployment instructions, screenshots, and evidence-based report.
5. All four confirm report numbers match committed result files and no secrets/evaluator labels were committed.

Laptop 1: install dependencies, put `artifacts/model.joblib` and the separate `evaluator_test/` directory on the server, set `PATHSNAP_DATA`, `PATHSNAP_EVALUATOR_DIR`, and `PATHSNAP_MODEL`, then run `uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000`.

Laptop 2: run `python3 run_ui.py`, open `http://localhost:8080`, enter Laptop 1's LAN address, and load trips from the API.

The Git checkout is `PathSnap`. The model artifact is included; raw evaluator data and evaluator-derived scores are deliberately excluded. Run and screenshot the UI/API two-laptop demonstration before final submission.
