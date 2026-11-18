from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import folium
from pathsnap.data import load_trip, candidate_table, _point_segment_distance
from pathsnap.decoder import viterbi
from pathsnap.models import FEATURES

DATA_ROOT=Path(os.getenv("PATHSNAP_DATA","../Student folder")).resolve()
EVALUATOR_ROOT=Path(os.getenv("PATHSNAP_EVALUATOR_DIR",str(DATA_ROOT/"evaluator_test"))).resolve()
MODEL_PATH=Path(os.getenv("PATHSNAP_MODEL","artifacts/model.joblib"))
CONFIG_PATH=Path(os.getenv("PATHSNAP_CONFIG","artifacts/model_config.json"))
app=FastAPI(title="PathSnap API",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET","POST"],allow_headers=["*"])
app.mount("/ui",StaticFiles(directory=str(Path(__file__).parent.parent/"ui"),html=True),name="ui")

@lru_cache(maxsize=1)
def decoder_weight():
    if not CONFIG_PATH.exists():
        return 1.8
    import json
    return float(json.loads(CONFIG_PATH.read_text()).get("decoder_transition_weight", 1.8))

@lru_cache(maxsize=1)
def model():
    if not MODEL_PATH.exists():
        raise HTTPException(503,"Trained model artifact is missing; run train.py first.")
    return joblib.load(MODEL_PATH)

def evaluator_trip_folders():
    if not EVALUATOR_ROOT.is_dir():
        return []
    return [folder for folder in sorted(EVALUATOR_ROOT.iterdir())
            if folder.is_dir() and (folder / f"{folder.name}.track").is_file()]

@app.get("/health")
def health():
    folders = evaluator_trip_folders()
    return {"status":"ok", "model_ready":MODEL_PATH.exists(),
            "evaluator_data_ready":len(folders) > 0,
            "evaluator_trip_count":len(folders)}

@app.get("/trips")
def trips():
    folders = evaluator_trip_folders()
    if not folders:
        return {"trips":[], "message":(
            f"Laptop 1 API cannot find evaluator trips at {EVALUATOR_ROOT}. "
            "Copy the evaluator_test folder to Laptop 1, set PATHSNAP_EVALUATOR_DIR "
            "to that folder's full path, and restart the API.")}
    # Only identifiers and observation counts are sent to the UI.
    result=[]
    for folder in folders:
        track_path=folder/(folder.name+".track")
        with track_path.open() as stream:
            observations=sum(1 for _ in stream)
        result.append({"trip_id":folder.name,"observations":observations})
    return {"trips":result, "message":f"{len(result)} evaluator trips ready on Laptop 1."}

@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    folder=EVALUATOR_ROOT/trip_id
    if Path(trip_id).name != trip_id or not folder.is_dir():
        raise HTTPException(404,"Unknown evaluator trip.")
    # Prediction inputs are loaded without the reference route. Route-derived labels
    # are read only after inference, for the server-side evaluation response.
    trip=load_trip(folder,include_route=False)
    table=candidate_table(trip,include_targets=False)
    clf=model(); raw=clf.predict_proba(table[FEATURES])[:,1]
    groups=list(table.groupby("obs",sort=True))
    cand=np.array([g.arc_id.to_numpy(int) for _,g in groups])
    probs=np.array([raw[g.index.to_numpy()] for _,g in groups])
    top=cand[np.arange(len(cand)),np.argmax(probs,axis=1)]
    gaps=np.r_[0.,np.diff(trip.track.time.to_numpy(float))]
    decoded=viterbi(cand,probs,trip.arcs,gaps,decoder_weight())
    # Ground truth is kept on Laptop 1 and is not included in model features/response.
    scored_trip=load_trip(folder,include_route=True)
    labels=np.full(len(trip.track),-1,int)
    if scored_trip.route is not None and len(scored_trip.route):
        route=np.unique(scored_trip.route)
        gps=trip.track[["lon","lat"]].to_numpy(float)
        route_edges=trip.nodes[trip.arcs[route]]
        distances=_point_segment_distance(gps,route_edges[:,0],route_edges[:,1])
        labels=route[np.argmin(distances,axis=1)]
    valid=labels>=0
    continuity=float(np.mean([a==b or trip.arcs[a,1]==trip.arcs[b,0] for a,b in zip(decoded[:-1],decoded[1:])])) if len(decoded)>1 else 1.0
    gps=trip.track[["lon","lat"]].to_numpy(float)
    center=[float(gps[:,1].mean()),float(gps[:,0].mean())]
    fmap=folium.Map(location=center,zoom_start=14,control_scale=True)
    gps_group=folium.FeatureGroup(name="Raw GPS observations",show=True)
    folium.PolyLine(gps[:,::-1].tolist(),color="#2563eb",weight=2,opacity=.45).add_to(gps_group)
    marker_stride=max(1,int(np.ceil(len(gps)/2500)))
    for lon,lat in gps[::marker_stride]:
        folium.CircleMarker([float(lat),float(lon)],radius=2,color="#1d4ed8",fill=True,
                            fill_opacity=.8,weight=1).add_to(gps_group)
    gps_group.add_to(fmap)
    candidate_stride=max(1,int(np.ceil(len(cand)/1200)))
    candidate_arcs=np.unique(cand[::candidate_stride].ravel())
    network_features=[]
    for arc_id in candidate_arcs:
        edge=trip.nodes[trip.arcs[int(arc_id)]]
        network_features.append({"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":edge.tolist()}})
    folium.GeoJson({"type":"FeatureCollection","features":network_features},name="Nearby road candidates",
                   style_function=lambda _: {"color":"#94a3b8","weight":2,"opacity":.45}).add_to(fmap)
    def route_parts_for(arc_ids):
        parts=[]
        for arc_id in arc_ids:
            edge=trip.nodes[trip.arcs[int(arc_id)]][:,::-1].tolist()
            if parts and parts[-1][-1] == edge[0]:
                parts[-1].append(edge[1])
            else:
                parts.append(edge)
        return parts
    folium.PolyLine(route_parts_for(top),color="#7c3aed",weight=3,opacity=.75,
                    dash_array="6, 5",tooltip="Classifier-only route").add_to(fmap)
    folium.PolyLine(route_parts_for(decoded),color="#16a34a",weight=5,opacity=.95,
                    tooltip="Classifier + Viterbi route").add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return {"trip_id":trip_id,"points":trip.track[["lon","lat","time"]].to_dict("records"),
            "predicted_arc_ids":decoded.tolist(),"classifier_arc_ids":top.tolist(),
            "map_html":fmap.get_root().render(),
            "metrics":{"classifier_top1":float(np.mean(top[valid]==labels[valid])) if valid.any() else None,
                       "hmm_top1":float(np.mean(decoded[valid]==labels[valid])) if valid.any() else None,
                       "classifier_route_continuity":float(np.mean([trip.arcs[a,1]==trip.arcs[b,0] or a==b for a,b in zip(top[:-1],top[1:])])) if len(top)>1 else 1.0,
                       "route_continuity":continuity}}

