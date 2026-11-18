"""
pathsnap/service.py - Laptop 1 FastAPI prediction backend for PathSnap.

Responsible for:
- Serving prediction and route reconstruction requests from Laptop 2.
- Loading the trained Machine Learning model artifact from `artifacts/model.joblib`.
- Exposing REST API endpoints: /health, /trips, /predict.
- Providing logging on Laptop 1 for evaluation inspection.
"""

from __future__ import annotations

import os
import glob
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

from pathsnap.decoder import ViterbiDecoder

# Configure logging for live terminal inspection during hackathon evaluation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap.service")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Route Reconstruction Backend (Laptop 1)",
    version="1.0.0",
)

# Enable CORS for Laptop 2 frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
DATA_DIR = Path(os.getenv("PATHSNAP_DATA", "Student folder"))
EVAL_DIR = Path(os.getenv("PATHSNAP_EVALUATOR_DIR", str(DATA_DIR / "evaluator_test")))
if not EVAL_DIR.exists():
    EVAL_DIR = DATA_DIR / "student_test"
if not EVAL_DIR.exists():
    EVAL_DIR = DATA_DIR / "train"

MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL", "artifacts/model.joblib"))

# Global model state
_model = None
_model_meta: Dict[str, Any] = {"loaded": False, "type": "none"}
_decoder = ViterbiDecoder()


def get_model():
    """Lazy loader for trained model artifact with heuristic fallback."""
    global _model, _model_meta
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        try:
            _model = joblib.load(str(MODEL_PATH))
            _model_meta = {
                "loaded": True,
                "type": type(_model).__name__,
                "path": str(MODEL_PATH),
            }
            logger.info(f"Loaded model artifact from {MODEL_PATH} ({_model_meta['type']})")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
            _model = None
    else:
        logger.warning(f"Model artifact not found at {MODEL_PATH}. Operating in heuristic fallback mode.")
        _model_meta = {"loaded": False, "type": "heuristic_distance_heading_fallback"}
    return _model


@app.on_event("startup")
def startup_event():
    get_model()
    logger.info(f"PathSnap service started. Evaluator directory: {EVAL_DIR}")
    logger.info(f"Available trips: {len(list_available_trips())}")


def list_available_trips() -> List[str]:
    """Scans evaluation directory and returns sorted trip identifiers."""
    if not EVAL_DIR.exists():
        return []
    trips = []
    for item in sorted(EVAL_DIR.iterdir()):
        if item.is_dir() and (item / f"{item.name}.track").exists():
            trips.append(item.name)
    return trips


def load_raw_trip_data(trip_id: str) -> Dict[str, Any]:
    """Loads track, nodes, and arcs files for a specified trip ID."""
    folder = EVAL_DIR / trip_id
    if not folder.is_dir():
        # Fallback check in student_test or train
        alt = DATA_DIR / "student_test" / trip_id
        if alt.is_dir():
            folder = alt
        else:
            alt = DATA_DIR / "train" / trip_id
            if alt.is_dir():
                folder = alt

    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip directory '{trip_id}' not found.")

    base = folder / trip_id
    track_path = folder / f"{trip_id}.track"
    nodes_path = folder / f"{trip_id}.nodes"
    arcs_path = folder / f"{trip_id}.arcs"

    if not track_path.exists() or not nodes_path.exists() or not arcs_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing trajectory or road-network files for '{trip_id}'.")

    track = pd.read_csv(track_path, sep=r"\s+", names=["lon", "lat", "time"])
    track = track.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    nodes = pd.read_csv(nodes_path, sep=r"\s+", names=["lon", "lat"]).to_numpy(float)
    arcs = pd.read_csv(arcs_path, sep=r"\s+", names=["from", "to"]).to_numpy(int)

    valid_arcs = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    arcs = arcs[valid_arcs]

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "folder": folder,
    }


def generate_candidate_features(trip_data: Dict[str, Any], k: int = 8) -> pd.DataFrame:
    """
    Generates spatial and movement features for candidate road arcs.
    Uses pathsnap.data if available, otherwise runs internal spatial BallTree.
    """
    try:
        from pathsnap.data import Trip, candidate_table
        trip_obj = Trip(
            trip_id=trip_data["trip_id"],
            track=trip_data["track"],
            nodes=trip_data["nodes"],
            arcs=trip_data["arcs"],
            route=None,  # Do not pass route for inference
        )
        return candidate_table(trip_obj, k=k)
    except Exception:
        # Internal candidate extraction fallback
        from sklearn.neighbors import BallTree

        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        track = trip_data["track"]

        xy = nodes[arcs]
        mid = xy.mean(axis=1)
        tree = BallTree(np.radians(mid[:, ::-1]), metric="haversine")

        gps = track[["lon", "lat"]].to_numpy(float)
        n = len(gps)
        k_val = min(k, len(arcs))
        _, ids = tree.query(np.radians(gps[:, ::-1]), k=k_val)

        a, b = xy[:, 0, :], xy[:, 1, :]
        d_lon = np.radians(b[:, 0] - a[:, 0])
        lat1 = np.radians(a[:, 1])
        lat2 = np.radians(b[:, 1])
        mid_bearing = np.arctan2(
            np.sin(d_lon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        )
        edge_len = np.linalg.norm(
            (b - a) * np.array([111_320 * np.cos(np.radians(gps[:, 1].mean())), 111_320]),
            axis=1,
        )

        pt_bearing = np.zeros(n)
        if n > 1:
            d_l = np.radians(gps[1:, 0] - gps[:-1, 0])
            l1 = np.radians(gps[:-1, 1])
            l2 = np.radians(gps[1:, 1])
            pt_bearing[:-1] = np.arctan2(
                np.sin(d_l) * np.cos(l2),
                np.cos(l1) * np.sin(l2) - np.sin(l1) * np.cos(l2) * np.cos(d_l)
            )
            pt_bearing[-1] = pt_bearing[-2]

        dt = np.r_[1.0, np.diff(track["time"].to_numpy(float))]
        speed = np.r_[
            0.0,
            np.linalg.norm(np.diff(gps, axis=0) * np.array([111_320 * np.cos(np.radians(gps[:, 1].mean())), 111_320]), axis=1) / np.maximum(dt[1:], 1e-3)
        ]

        rows = []
        earth_r = 6_371_008.8
        for i in range(n):
            ids_i = ids[i]
            dlat = np.radians(gps[i, 1] - mid[ids_i, 1]) * earth_r
            dlon = np.radians(gps[i, 0] - mid[ids_i, 0]) * earth_r * np.cos(np.radians(gps[i, 1]))
            dist = np.hypot(dlat, dlon)
            head_err = np.abs(np.arctan2(np.sin(pt_bearing[i] - mid_bearing[ids_i]), np.cos(pt_bearing[i] - mid_bearing[ids_i])))
            for rank, arc_id in enumerate(ids_i):
                rows.append((
                    i, arc_id, float(dist[rank]), float(head_err[rank]),
                    float(edge_len[arc_id]), float(speed[i]), float(dt[i]), float(rank)
                ))

        return pd.DataFrame(
            rows,
            columns=["obs", "arc_id", "distance_m", "heading_error", "edge_length_m", "speed_mps", "time_gap_s", "candidate_rank"]
        )


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Unique trip directory identifier")
    use_viterbi: bool = Field(True, description="Whether to apply Viterbi sequence decoding")
    k_candidates: int = Field(8, description="Number of road candidates per GPS fix")


@app.get("/health")
def health(request: Request):
    """Health check endpoint confirming API readiness and model load status."""
    logger.info(f"GET /health from {request.client.host if request.client else 'unknown'}")
    return {
        "status": "ready",
        "model_loaded": _model_meta["loaded"],
        "model_type": _model_meta["type"],
        "evaluator_dir": str(EVAL_DIR),
        "trips_available": len(list_available_trips()),
    }


@app.get("/trips")
def get_trips(request: Request):
    """Returns available trip IDs for the evaluator UI dropdown."""
    trips = list_available_trips()
    logger.info(f"GET /trips from {request.client.host if request.client else 'unknown'} (found {len(trips)} trips)")
    return {"trips": trips}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    """
    Reconstructs the vehicle route from noisy GPS observations.
    
    Returns predicted route arcs, GPS trajectory, and road coordinates for UI rendering.
    Strictly protects server-side reference route labels.
    """
    t0 = time.time()
    trip_data = load_raw_trip_data(req.trip_id)
    track = trip_data["track"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]

    # 1. Candidate Generation & Feature Extraction
    df_cands = generate_candidate_features(trip_data, k=req.k_candidates)

    # 2. Candidate Probability Scoring
    feature_cols = ["distance_m", "heading_error", "edge_length_m", "speed_mps", "time_gap_s", "candidate_rank"]
    model = get_model()

    if model is not None and hasattr(model, "predict_proba"):
        X = df_cands[feature_cols].to_numpy()
        probs = model.predict_proba(X)[:, 1]
    else:
        # Heuristic emission model: distance + heading error penalty
        d = df_cands["distance_m"].to_numpy(float)
        h = df_cands["heading_error"].to_numpy(float)
        scores = np.exp(-0.5 * (d / 15.0) ** 2) * np.cos(np.minimum(h, np.pi / 2))
        probs = np.maximum(scores, 1e-6)

    df_cands["score"] = probs

    # 3. Group by observation
    candidate_arcs_per_step: List[List[int]] = []
    candidate_probs_per_step: List[List[float]] = []
    pointwise_arcs: List[int] = []

    for _, group in df_cands.groupby("obs"):
        c_list = group["arc_id"].tolist()
        p_list = group["score"].tolist()
        candidate_arcs_per_step.append(c_list)
        candidate_probs_per_step.append(p_list)
        pointwise_arcs.append(c_list[int(np.argmax(p_list))])

    # 4. Sequence Decoding (Viterbi vs Pointwise)
    timestamps = track["time"].tolist()
    if req.use_viterbi:
        decoded_arcs = _decoder.decode(candidate_arcs_per_step, candidate_probs_per_step, arcs, timestamps)
    else:
        decoded_arcs = pointwise_arcs

    predicted_route = _decoder.compress_route(decoded_arcs)
    continuity = _decoder.calculate_continuity(decoded_arcs, arcs)
    inference_ms = round((time.time() - t0) * 1000, 2)

    # Extract road segment geometries for UI map rendering
    route_geometry = []
    for arc_id in predicted_route:
        if arc_id < len(arcs):
            u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
            coord1 = [float(nodes[u, 0]), float(nodes[u, 1])]
            coord2 = [float(nodes[v, 0]), float(nodes[v, 1])]
            route_geometry.append([coord1, coord2])

    gps_points = track[["lon", "lat"]].to_numpy(float).tolist()

    logger.info(
        f"POST /predict | Trip: {req.trip_id} | Fixes: {len(track)} | Route: {len(predicted_route)} arcs | "
        f"Continuity: {continuity:.4f} | {inference_ms}ms"
    )

    return {
        "trip_id": req.trip_id,
        "predicted_route": predicted_route,
        "observation_count": len(track),
        "route_length": len(predicted_route),
        "route_continuity": round(continuity, 4),
        "inference_time_ms": inference_ms,
        "gps_points": gps_points,
        "route_geometry": route_geometry,
    }


class EvaluateRequest(BaseModel):
    trip_id: str = Field(..., description="Unique trip directory identifier to evaluate")
    k_candidates: int = Field(8, description="Number of road candidates per GPS fix")


@app.post("/evaluate")
def evaluate(req: EvaluateRequest, request: Request):
    """
    Evaluates model predictions and Viterbi route reconstruction against
    server-side reference route data.

    SECURITY REQUIREMENT:
    Reference route labels (.route) are read strictly on Laptop 1 and are
    NEVER returned in the API response or transmitted to Laptop 2 / client UI.
    Only calculated performance metrics are returned.
    """
    t0 = time.time()
    trip_data = load_raw_trip_data(req.trip_id)
    folder = trip_data["folder"]
    route_path = folder / f"{req.trip_id}.route"

    if not route_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference route '{req.trip_id}.route' not found on server for evaluation."
        )

    # 1. Read ground-truth reference route strictly on the server
    ref_df = pd.read_csv(route_path, sep=r"\s+", header=None)
    reference_route = ref_df.iloc[:, 0].to_numpy(int).tolist()

    track = trip_data["track"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]

    # 2. Candidate Generation & Feature Extraction
    df_cands = generate_candidate_features(trip_data, k=req.k_candidates)

    # 3. Model Scoring
    feature_cols = ["distance_m", "heading_error", "edge_length_m", "speed_mps", "time_gap_s", "candidate_rank"]
    model = get_model()

    if model is not None and hasattr(model, "predict_proba"):
        X = df_cands[feature_cols].to_numpy()
        probs = model.predict_proba(X)[:, 1]
    else:
        d = df_cands["distance_m"].to_numpy(float)
        h = df_cands["heading_error"].to_numpy(float)
        scores = np.exp(-0.5 * (d / 15.0) ** 2) * np.cos(np.minimum(h, np.pi / 2))
        probs = np.maximum(scores, 1e-6)

    df_cands["score"] = probs

    # 4. Extract candidates, probabilities, and pointwise predictions
    candidate_arcs_per_step: List[List[int]] = []
    candidate_probs_per_step: List[List[float]] = []
    raw_top1_arcs: List[int] = []
    ref_pointwise: Optional[List[int]] = None

    if "reference_arc_id" in df_cands.columns:
        ref_pointwise = []

    for _, group in df_cands.groupby("obs"):
        c_list = group["arc_id"].tolist()
        p_list = group["score"].tolist()
        candidate_arcs_per_step.append(c_list)
        candidate_probs_per_step.append(p_list)
        raw_top1_arcs.append(c_list[int(np.argmax(p_list))])
        if ref_pointwise is not None:
            ref_pointwise.append(int(group["reference_arc_id"].iloc[0]))

    # 5. Sequence Decoding
    timestamps = track["time"].tolist()
    decoded_arcs = _decoder.decode(candidate_arcs_per_step, candidate_probs_per_step, arcs, timestamps)

    # 6. Comprehensive Server-Side Comparison
    comparison = _decoder.compare(
        raw_classifier_arcs=raw_top1_arcs,
        decoded_arcs=decoded_arcs,
        reference_route=reference_route,
        arcs=arcs,
        reference_pointwise_arcs=ref_pointwise,
    )

    eval_ms = round((time.time() - t0) * 1000, 2)

    logger.info(
        f"POST /evaluate | Trip: {req.trip_id} | Ref Arcs: {len(reference_route)} | "
        f"Viterbi F1: {comparison['route_f1']['viterbi_decoded']:.4f} "
        f"(+{comparison['route_f1']['improvement']:.4f}) | "
        f"Continuity: {comparison['continuity']['viterbi_decoded']:.4f} | {eval_ms}ms"
    )

    # Return approved evaluation metrics only - never reference route contents
    return {
        "trip_id": req.trip_id,
        "status": "evaluated_server_side",
        "reference_labels_protected": True,
        "observation_count": len(track),
        "ground_truth_route_arcs": len(reference_route),
        "metrics": comparison,
        "inference_time_ms": eval_ms,
    }


@app.post("/reload_model")
def reload_model():
    """Allows hot-reloading the model artifact after Person 2 completes training."""
    global _model, _model_meta
    _model = None
    m = get_model()
    return {
        "status": "reloaded",
        "model_loaded": _model_meta["loaded"],
        "model_type": _model_meta["type"],
        "model_path": _model_meta.get("path"),
    }

