
# PathSnap – Distributed Map-Matching & Route Reconstruction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-yellow" alt="Frontend">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" alt="Project Status"> 
  <img src="https://img.shields.io/badge/Course-UE24CS352A-purple" alt="Course">
</p>

<p align="center">
  <b>An intelligent GPS trajectory map-matching and route reconstruction system powered by Machine Learning.</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [System Workflow](#-system-workflow)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset Description](#-dataset-description)
- [Feature Engineering](#-feature-engineering)
- [Model Training and Optimisation](#-model-training-and-optimisation)
- [Route Reconstruction](#-route-reconstruction)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Execution](#-execution)
- [Two-Laptop Deployment](#-two-laptop-deployment)
- [Evaluation Metrics](#-evaluation-metrics)
- [Generated Artifacts](#-generated-artifacts)
- [Limitations](#-limitations)
- [Team Contributions](#-team-contributions)
- [Future Improvements](#-future-improvements)

---

## 🚀 Overview

**PathSnap** is a Machine Learning-based map-matching system that reconstructs vehicle routes from noisy GPS trajectories using road-network information.

GPS observations collected from mobile devices and navigation systems often contain positional inaccuracies, measurement noise, and irregularities. Consequently, directly plotting GPS coordinates may not accurately represent the actual roads travelled.

PathSnap addresses this challenge by identifying candidate road segments for each GPS observation, extracting spatial and movement-related features, and applying Machine Learning classifiers to determine the most likely road segments.

A sequence-based Viterbi decoder further improves route consistency by considering relationships between consecutive predictions.

The system also incorporates a FastAPI backend and a browser-based frontend, allowing distributed execution across two laptops.

### Key Features

- GPS trajectory and road-network exploration.
- GPS-to-road candidate generation.
- Spatial and movement-based feature extraction.
- Six Machine Learning classification models.
- Trip-grouped cross-validation.
- Bayesian hyperparameter optimisation.
- Viterbi-based route reconstruction.
- FastAPI-based prediction service.
- Interactive browser-based route interface.
- Two-laptop distributed deployment.

---

## 🎯 Problem Statement

GPS trajectories contain noisy and inaccurate location observations that do not always correspond directly to road-network geometries.

The objective of this project is to develop a Machine Learning-based map-matching system that associates GPS observations with the most likely road segments and reconstructs a continuous route using the available road-network information.

The system aims to improve the accuracy and consistency of route reconstruction by combining candidate generation, feature-based classification, and sequential decoding.

---

## 🎯 Objectives

1. Analyse GPS trajectory and road-network datasets.
2. Preprocess GPS observations and road-network structures.
3. Generate candidate road segments for individual GPS observations.
4. Extract meaningful spatial and movement-related features.
5. Implement and compare multiple Machine Learning classifiers.
6. Evaluate models using trip-grouped cross-validation.
7. Perform Bayesian hyperparameter optimisation.
8. Implement Viterbi-based route reconstruction.
9. Develop a backend service for route prediction.
10. Build a frontend for route selection and visualisation.
11. Integrate the backend and frontend through a distributed two-laptop architecture.

---

## ⚙️ System Workflow

The PathSnap pipeline consists of the following stages:

### 1. Data Exploration

- Inspect GPS tracks, timestamps, road nodes, road arcs, and reference routes.
- Analyse trajectory lengths and road-network connectivity.
- Validate coordinates and route identifiers.
- Generate trajectory and road-network visualisations.

### 2. Data Preprocessing

- Load individual trip files.
- Validate GPS observations and road-network structures.
- Prepare road segments for spatial processing.
- Handle invalid or missing observations.

### 3. Candidate Generation

For each GPS observation, identify a set of possible road segments from the supplied road network.

Candidate road segments are evaluated using their spatial relationship with the GPS observation.

### 4. Feature Engineering

Extract features that describe the relationship between GPS observations and candidate road segments.

Examples include:

- GPS-to-road distance.
- Heading difference.
- Road-segment length.
- GPS speed.
- Time difference between observations.
- Along-segment projected position.
- Candidate rank.

### 5. Machine Learning Classification

Train multiple classifiers to score candidate road segments and identify the most likely candidate for each observation.

### 6. Model Evaluation and Optimisation

- Perform five-fold GroupKFold cross-validation.
- Compare classifier performance.
- Select the leading model.
- Perform Bayesian hyperparameter optimisation.
- Evaluate the tuned model on student-test trips.

### 7. Route Reconstruction

Apply a Viterbi-based decoder to improve the sequential consistency of predicted road segments.

### 8. Backend and Frontend Integration

Deploy the trained model through a FastAPI backend and connect it to a browser-based user interface.

---

## 🤖 Machine Learning Models

PathSnap implements six candidate-scoring Machine Learning models.

| Model | Algorithm | Category |
|---|---|---|
| Logistic Regression | `LogisticRegression` | Linear Classification |
| Gaussian Naive Bayes | `GaussianNB` | Probabilistic Classification |
| Decision Tree | `DecisionTreeClassifier` | Tree-Based Learning |
| Extra Trees | `ExtraTreesClassifier` | Ensemble Learning |
| Histogram Gradient Boosting | `HistGradientBoostingClassifier` | Gradient Boosting |
| K-Nearest Neighbors | `KNeighborsClassifier` | Instance-Based Learning |

All six models are defined in:

```text
pathsnap/models.py
```

The model comparison and training pipeline is implemented in:

```text
train.py
```

### Model Comparison

All models are evaluated using the same candidate-feature dataset and trip-grouped cross-validation procedure.

The leading model is selected using cross-validation Macro F1-score before Bayesian hyperparameter optimisation.

---

## 📂 Dataset Description

The dataset contains GPS trajectories and corresponding road-network information.

Each trip contains four files:

| File | Description |
|---|---|
| `.track` | GPS longitude, latitude, and timestamp |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Connections between road-network nodes |
| `.route` | Reference sequence of road arc IDs |

### Dataset Partitions

| Partition | Number of Trips |
|---|---:|
| Training | 80 |
| Student Testing | 10 |
| Evaluator Testing | 10 expected |

The evaluator partition is provided separately by the organisers and is not included in the current workspace.

### Dataset Directory Structure

```text
Student folder/
│
├── train/
│   ├── trip_1/
│   │   ├── *.track
│   │   ├── *.nodes
│   │   ├── *.arcs
│   │   └── *.route
│   │
│   └── ...
│
├── student_test/
│   └── ...
│
└── evaluator_test/
    └── ...
```

**Important:** The actual supplied trip directories use their dataset-specific identifiers. The example above illustrates the expected partition organisation.

Raw datasets and evaluator ground-truth labels must not be committed to the public repository.

---

## 🧮 Feature Engineering

Feature extraction is performed to represent the spatial relationship between each GPS observation and its candidate road segments.

The features include:

| Feature | Description |
|---|---|
| Distance | Distance between GPS observation and candidate road segment |
| Heading Difference | Difference between GPS movement direction and road direction |
| Segment Length | Length of the candidate road segment |
| GPS Speed | Estimated movement speed from consecutive observations |
| Time Gap | Time difference between GPS observations |
| Along-Segment Position | Projected position of the observation along the road segment |
| Candidate Rank | Relative ranking of the generated candidate |

These features are used by the classification models to score candidate road segments.

---

## 📈 Model Training and Optimisation

The training pipeline is implemented in `train.py`.

### Cross-Validation

The project uses:

```python
GroupKFold(n_splits=5)
```

Grouping is performed at the trip level to reduce data leakage between training and validation folds.

### Bayesian Hyperparameter Optimisation

Bayesian optimisation is performed on the leading model selected through cross-validation.

The current training configuration performs:

- 12 Bayesian optimisation trials.
- Model-specific hyperparameter search.
- Evaluation of the tuned model on student-test data.

The optimisation results are saved for analysis and reproducibility.

---

## 🗺️ Route Reconstruction

After candidate scoring, PathSnap applies a Viterbi-based decoder to improve route continuity.

Independent classification can select road segments that are individually plausible but inconsistent with the surrounding trajectory.

The decoder considers the sequence of candidate predictions and applies a continuity penalty to obtain a more consistent route.

Implementation:

```text
pathsnap/decoder.py
```

The project compares classifier predictions with decoded route predictions.

---

## 🏗️ Project Architecture

PathSnap follows a modular architecture consisting of four major components.

| Component | Responsibility |
|---|---|
| Data Processing | Loading, preprocessing, candidate generation, and feature extraction |
| Machine Learning | Model training, comparison, and optimisation |
| Route Decoder and API | Sequential route reconstruction and prediction service |
| Frontend | Trip selection, route visualisation, and results display |

---

## 📁 Project Structure

```text
pathsnap/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── DATASET_NOTES.md
├── REPORT_TEMPLATE.md
├── TEAM_COMMIT_PLAN.md
├── TEAM_WORKFLOW.md
│
├── explore.py
├── train.py
├── run_ui.py
│
├── pathsnap/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── decoder.py
│   └── service.py
│
├── ui/
│   └── index.html
│
└── artifacts/
    ├── exploration/
    ├── dataset_inventory.csv
    ├── model_comparison.csv
    ├── bayes_trials.csv
    ├── model_config.json
    ├── tuned_student_test.json
    └── model.joblib
```

### Important Files

| File | Function |
|---|---|
| `explore.py` | Dataset exploration and visualisation |
| `pathsnap/data.py` | Data loading, preprocessing, and feature generation |
| `pathsnap/models.py` | Machine Learning model definitions |
| `train.py` | Model training and optimisation |
| `pathsnap/decoder.py` | Viterbi route reconstruction |
| `pathsnap/service.py` | FastAPI backend |
| `run_ui.py` | Frontend server launcher |
| `ui/index.html` | Browser-based user interface |
| `artifacts/` | Model and experiment outputs |

---

## 🛠️ Technologies Used

### Programming and Data Processing

- Python
- NumPy
- Pandas

### Machine Learning

- Scikit-learn
- Scikit-Optimize

### Geospatial Processing and Visualisation

- Shapely
- Folium

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Model Persistence and Version Control

- Joblib
- Git
- GitHub

---

## 💻 Installation

### Prerequisites

- Python 3.10–3.12
- Git
- pip
- Modern web browser

### 1. Clone the Repository

```bash
git clone https://github.com/TEAM-ORG/REPOSITORY.git

cd REPOSITORY
```

Replace `TEAM-ORG/REPOSITORY` with the actual GitHub repository path.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 5. Configure the Dataset

Keep the supplied dataset outside the repository checkout.

Example:

```text
project-parent/
│
├── Student folder/
│   ├── train/
│   ├── student_test/
│   └── evaluator_test/
│
└── pathsnap/
```

---

## ▶️ Execution

### 1. Dataset Exploration

Run:

```bash
python explore.py --data "../Student folder"
```

To inspect a specific trip:

```bash
python explore.py \
    --data "../Student folder" \
    --trip "../Student folder/train/00000034"
```

Exploration outputs are stored in:

```text
artifacts/exploration/
```

### 2. Train and Compare Models

Run:

```bash
python train.py \
    --data "../Student folder" \
    --output artifacts \
    --seed 42
```

The training pipeline performs:

1. Dataset loading.
2. Candidate generation.
3. Feature extraction.
4. Training of six classifiers.
5. Five-fold trip-grouped cross-validation.
6. Model comparison.
7. Bayesian hyperparameter optimisation.
8. Student-test evaluation.
9. Model and result persistence.

### 3. Start the Backend

After the model artifact has been generated:

```bash
uvicorn pathsnap.service:app \
    --host 0.0.0.0 \
    --port 8000
```

### 4. Start the Frontend

In another terminal:

```bash
python3 run_ui.py
```

Open:

```text
http://localhost:8080
```

---

## 🌐 Two-Laptop Deployment

PathSnap supports distributed execution across two laptops connected to the same local network.

### Laptop 1 – Backend

Laptop 1 is responsible for:

- Hosting the FastAPI application.
- Loading the trained model.
- Processing prediction requests.
- Performing route reconstruction.
- Handling evaluator-side scoring.

Configure the environment variables:

```bash
export PATHSNAP_DATA="/path/to/Student folder"

export PATHSNAP_EVALUATOR_DIR="/path/to/Student folder/evaluator_test"

export PATHSNAP_MODEL="/path/to/artifacts/model.joblib"
```

Start the API:

```bash
uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000
```

Check API readiness:

```text
http://localhost:8000/health
```

API documentation:

```text
http://localhost:8000/docs
```

### Laptop 2 – Frontend

Start the frontend server:

```bash
python3 run_ui.py
```

Open the browser:

```text
http://localhost:8080
```

In the Evaluator UI:

1. Enter Laptop 1's local IP address.
2. Select **Load trips**.
3. Choose an available trip.
4. View the returned route predictions and evaluation results.

To find Laptop 1's local IP address on macOS:

```bash
ipconfig getifaddr en0
```

Example backend address:

```text
http://192.168.1.20:8000
```

Both laptops must be connected to the same network, and Laptop 1 must allow incoming connections on port `8000`.

### Local Trip Preview

The application also includes a Local Trip Preview feature.

It allows supplied trip files to be visualised directly in the browser.

This is a data exploration feature and should not be interpreted as Machine Learning inference.

---

## 📊 Evaluation Metrics

The project evaluates both candidate classification and route reconstruction performance.

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of positive predictions that are correct |
| Recall | Measures the proportion of actual positive samples identified |
| F1-score | Harmonic mean of precision and recall |
| Macro F1-score | Average F1-score across classes |
| Confusion Matrix | Shows classification outcomes |
| Candidate Log Loss | Evaluates predicted candidate probabilities |
| Top-1 Accuracy | Measures whether the highest-scoring candidate is correct |
| Candidate Miss Rate | Measures how often the reference candidate is absent from the candidate set |
| Route Continuity | Evaluates consistency between consecutive predicted road segments |
| HMM Top-1 Accuracy | Evaluates decoded route predictions |
| Inference Time | Measures prediction execution time |

Actual experiment results should be reported using the generated files rather than manually estimated values.

---

## 📦 Generated Artifacts

The training pipeline generates the following outputs:

| Artifact | Description |
|---|---|
| `dataset_inventory.csv` | Dataset inventory |
| `model_comparison.csv` | Model comparison results |
| `bayes_trials.csv` | Bayesian optimisation results |
| `model_config.json` | Selected model configuration |
| `tuned_student_test.json` | Tuned model evaluation |
| `model.joblib` | Persisted trained model |
| `exploration/` | Dataset exploration outputs |

These files are created after executing the corresponding scripts.

---

## ⚠️ Limitations

- The evaluator partition is not included in the current workspace.
- Supplied reference routes are treated as labels and are not independently verified as physical ground truth.
- Candidate generation currently uses nearest network-edge midpoints, which can introduce ambiguity for nearby or parallel roads.
- Candidate recall must be measured to determine whether the correct road segment is included in the generated candidate set.
- The current decoder uses a continuity penalty rather than a probabilistic road-network travel-time model.
- Final evaluator results and screenshots must be generated using the actual evaluator partition.

---

## 👥 Team Contributions

The project follows a four-member development workflow.

| Member | Responsibility | Files |
|---|---|---|
| Person 1 | Data preprocessing, feature engineering, and EDA | `pathsnap/data.py`, `explore.py` |
| Person 2 | Machine Learning models, experiments, and optimisation | `pathsnap/models.py`, `train.py` |
| Person 3 | Viterbi decoder and backend API | `pathsnap/decoder.py`, `pathsnap/service.py` |
| Person 4 | Frontend, deployment, and documentation | `ui/`, `run_ui.py`, `README.md` |

### Development Workflow

The team follows a shared GitHub repository workflow:

- Each member works on a separate feature branch.
- Changes are committed independently.
- Pull requests are reviewed before merging.
- Components are integrated in dependency order.
- Experiment results are committed only after actual execution.

---

## 🔮 Future Improvements

- Improve candidate generation using point-to-segment distance.
- Incorporate road-network connectivity and transition probabilities.
- Improve GPS noise handling.
- Experiment with additional sequence-modelling approaches.
- Optimise inference latency.
- Improve route visualisation and frontend interaction.
- Evaluate generalisation using the organiser-provided evaluator partition.

---

## 📚 Academic Information

**Course:** UE24CS352A – Machine Learning

**Project:** PathSnap – Distributed Map-Matching & Route Reconstruction

**Institution:** PES University

---

<p align="center">
  <b>PathSnap</b><br>
  Reconstructing routes. Connecting observations. Making GPS data meaningful.
</p>
