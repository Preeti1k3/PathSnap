from __future__ import annotations
import os
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import folium
from pathsnap.data import load_trip, candidate_table
from pathsnap.decoder import viterbi
from pathsnap.models import FEATURES

DATA_ROOT=Path(os.getenv("PATHSNAP_DATA","../Student folder")).resolve()
EVALUATOR_ROOT=Path(os.getenv("PATHSNAP_EVALUATOR_DIR",str(DATA_ROOT/"evaluator_test"))).resolve()
MODEL_PATH=Path(os.getenv("PATHSNAP_MODEL","artifacts/model.joblib"))
app=FastAPI(title="PathSnap API",version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET","POST"],allow_headers=["*"])
app.mount("/ui",StaticFiles(directory=str(Path(__file__).parent.parent/"ui"),html=True),name="ui")

def model():
    if not MODEL_PATH.exists():
        raise HTTPException(503,"Trained model artifact is missing; run train.py first.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health(): return {"status":"ok","model_ready":MODEL_PATH.exists()}

@app.get("/trips")
def trips():
    if not EVALUATOR_ROOT.exists(): return {"trips":[],"message":"Evaluator trip data is not installed on this server."}
    # Only identifiers and observation counts are sent to the UI.
    return {"trips":[{"trip_id":p.name,"observations":sum(1 for _ in open(p/(p.name+".track")))}
                      for p in sorted(EVALUATOR_ROOT.iterdir()) if p.is_dir()]}

@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    folder=EVALUATOR_ROOT/trip_id
    if not folder.is_dir(): raise HTTPException(404,"Unknown evaluator trip.")
    trip=load_trip(folder,include_route=True)
    table=candidate_table(trip)
    clf=model(); raw=clf.predict_proba(table[FEATURES])[:,1]
    groups=list(table.groupby("obs",sort=True))
    cand=np.array([g.arc_id.to_numpy(int) for _,g in groups])
    probs=np.array([raw[g.index.to_numpy()] for _,g in groups])
    top=cand[np.arange(len(cand)),np.argmax(probs,axis=1)]
    gaps=np.r_[0.,np.diff(trip.track.time.to_numpy(float))]
    decoded=viterbi(cand,probs,trip.arcs,gaps)
    # Route labels are used only inside this server-side scoring function.
    labels=np.full(len(trip.track),-1,int)
    if trip.route is not None and len(trip.route):
        xy=trip.nodes[trip.arcs]; mids=xy.mean(axis=1); route=np.unique(trip.route)
        from sklearn.neighbors import BallTree
        route=route[route<len(trip.arcs)]
        bt=BallTree(np.radians(mids[route,::-1]),metric="haversine")
        gps=trip.track[["lon","lat"]].to_numpy(float)
        labels=route[bt.query(np.radians(gps[:,::-1]),k=1,return_distance=False)[:,0]]
    valid=labels>=0
    continuity=float(np.mean([trip.arcs[a,1]==trip.arcs[b,0] for a,b in zip(decoded[:-1],decoded[1:])])) if len(decoded)>1 else 1.0
    gps=trip.track[["lon","lat"]].to_numpy(float)
    center=[float(gps[:,1].mean()),float(gps[:,0].mean())]
    fmap=folium.Map(location=center,zoom_start=14,control_scale=True)
    folium.PolyLine(gps[:,::-1].tolist(),color="#2684ff",weight=3,opacity=.65,tooltip="GPS observations").add_to(fmap)
    for arc_id in decoded:
        edge=trip.nodes[trip.arcs[int(arc_id)]]
        folium.PolyLine(edge[:,::-1].tolist(),color="#e4572e",weight=5,opacity=.9,tooltip="Predicted road segment").add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return {"trip_id":trip_id,"points":trip.track[["lon","lat","time"]].to_dict("records"),
            "predicted_arc_ids":decoded.tolist(),"classifier_arc_ids":top.tolist(),
            "map_html":fmap.get_root().render(),
            "metrics":{"classifier_top1":float(np.mean(top[valid]==labels[valid])) if valid.any() else None,
                       "hmm_top1":float(np.mean(decoded[valid]==labels[valid])) if valid.any() else None,
                       "route_continuity":continuity}}

