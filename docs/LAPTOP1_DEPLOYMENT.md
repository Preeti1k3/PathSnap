# Laptop 1 Backend Service & Deployment Guide

This document describes the launch configuration, environment setup, networking, and API contract for **Laptop 1** in the PathSnap distributed two-laptop architecture.

---

## 1. Laptop 1 Responsibilities

In accordance with the hackathon guidelines and project architecture:
- **Backend Host**: Runs the FastAPI application (`pathsnap.service:app`).
- **Model Inference**: Loads the trained model artifact (`artifacts/model.joblib`) produced by Person 2.
- **Sequence Reconstruction**: Executes the HMM / Viterbi decoder (`pathsnap/decoder.py`).
- **Server-Side Scoring & Data Protection**: Computes performance metrics (`/evaluate`) on the private reference route (`.route`), strictly guaranteeing that ground-truth labels are never exposed to the frontend / Laptop 2.
- **Live Terminal Logging**: Displays live request logs (timestamp, client IP, route length, inference duration, continuity scores) for evaluation judges to inspect.

---

## 2. Environment Variables

The service is configured via standard environment variables:

| Variable | Default Value | Description |
|---|---|---|
| `PATHSNAP_DATA` | `"Student folder"` | Root directory containing trip datasets. |
| `PATHSNAP_EVALUATOR_DIR` | `"$PATHSNAP_DATA/evaluator_test"` | Directory containing the ten reserved evaluation trips. Automatically falls back to `student_test` if `evaluator_test` is not yet available. |
| `PATHSNAP_MODEL` | `"artifacts/model.joblib"` | Filepath to the trained scikit-learn model artifact. If absent, the API falls back to a spatial distance-and-heading heuristic. |

---

## 3. Launch Instructions

### Prerequisites
Ensure all dependencies are installed:
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
```

### Windows (PowerShell)
```powershell
# Optional: Set custom paths if datasets/artifacts are stored in a different location
$env:PATHSNAP_DATA = "Student folder"
$env:PATHSNAP_EVALUATOR_DIR = "Student folder/student_test"
$env:PATHSNAP_MODEL = "artifacts/model.joblib"

# Start Uvicorn bound to all interfaces
uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000 --log-level info
```

### Windows (CMD)
```cmd
set PATHSNAP_DATA=Student folder
set PATHSNAP_EVALUATOR_DIR=Student folder/student_test
set PATHSNAP_MODEL=artifacts/model.joblib

uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000 --log-level info
```

### Linux / macOS (Bash)
```bash
export PATHSNAP_DATA="Student folder"
export PATHSNAP_EVALUATOR_DIR="Student folder/student_test"
export PATHSNAP_MODEL="artifacts/model.joblib"

uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000 --log-level info
```

---

## 4. Local Area Network (LAN) & Firewall Setup

Both Laptop 1 (Backend) and Laptop 2 (UI) must be connected to the same Wi-Fi network or mobile hotspot.

### Step 4.1: Find Laptop 1's LAN IP Address
- **Windows**: Run `ipconfig` in PowerShell/CMD and look for `IPv4 Address` under your active Wireless or Ethernet adapter (e.g., `192.168.1.45`).
- **macOS**: Run `ipconfig getifaddr en0`.
- **Linux**: Run `hostname -I`.

### Step 4.2: Open Inbound Port 8000 on Laptop 1
If Laptop 2 cannot connect, allow port 8000 through the Windows Defender Firewall:
```powershell
# Run PowerShell as Administrator (if needed)
New-NetFirewallRule -DisplayName "PathSnap Backend Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 4.3: Connecting from Laptop 2
On Laptop 2, open `http://localhost:8080` in the browser and enter Laptop 1's LAN address:
```text
http://<LAPTOP_1_IP>:8000
```
Example:
```text
http://192.168.1.45:8000
```

---

## 5. API Endpoints Specification

### 1. `GET /health`
Verifies that the service is running, reports model load status, and lists available trip count.
- **Response `200 OK`**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_type": "HistGradientBoostingClassifier",
  "evaluator_dir": "Student folder/student_test",
  "trips_available": 10
}
```

### 2. `GET /trips`
Lists available trip IDs in the evaluator partition for the UI selector.
- **Response `200 OK`**:
```json
{
  "trips": [
    "00000000",
    "00000007",
    "00000009",
    "00000040"
  ]
}
```

### 3. `POST /predict`
Performs candidate generation, feature extraction, model scoring, and Viterbi route reconstruction.
- **Request Body**:
```json
{
  "trip_id": "00000000",
  "use_viterbi": true,
  "k_candidates": 8
}
```
- **Response `200 OK`**:
```json
{
  "trip_id": "00000000",
  "predicted_route": [17633, 211, 17638, 17647, ...],
  "observation_count": 2503,
  "route_length": 305,
  "route_continuity": 0.9812,
  "inference_time_ms": 142.5,
  "gps_points": [[8.5816, 49.8578], ...],
  "route_geometry": [[[8.5816, 49.8578], [8.5846, 49.8582]], ...]
}
```
*Note: Ground-truth reference routes are never returned by this endpoint.*

### 4. `POST /evaluate`
Evaluates the reconstructed route against private reference route data stored on Laptop 1.
- **Request Body**:
```json
{
  "trip_id": "00000000",
  "k_candidates": 8
}
```
- **Response `200 OK`**:
```json
{
  "trip_id": "00000000",
  "status": "evaluated_server_side",
  "reference_labels_protected": true,
  "observation_count": 2503,
  "ground_truth_route_arcs": 179,
  "metrics": {
    "continuity": {
      "classifier_only": 0.8981,
      "viterbi_decoded": 0.9812,
      "improvement": 0.0831
    },
    "route_f1": {
      "classifier_only": 0.5767,
      "viterbi_decoded": 0.7378,
      "improvement": 0.1611
    },
    "lcs_ratio": {
      "classifier_only": 0.9665,
      "viterbi_decoded": 0.9274,
      "improvement": -0.0391
    },
    "route_length": {
      "ground_truth": 179,
      "classifier_only": 725,
      "viterbi_decoded": 305
    }
  },
  "inference_time_ms": 185.3
}
```

### 5. `POST /reload_model`
Hot-reloads the model artifact from disk without taking down the server.
- **Response `200 OK`**:
```json
{
  "status": "reloaded",
  "model_loaded": true,
  "model_type": "HistGradientBoostingClassifier",
  "model_path": "artifacts/model.joblib"
}
```

---

## 6. Live Evaluator Demo Checklist

During the live assessment:
1. Keep the Laptop 1 terminal in full view displaying Uvicorn request logs.
2. Have Person 4 on Laptop 2 select trips from the frontend UI.
3. Observe incoming `POST /predict` and `POST /evaluate` log lines printing in real-time.
4. Verify to the evaluators that no `.route` data or hardcoded outputs are returned over the network.
