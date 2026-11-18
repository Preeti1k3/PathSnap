from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_M = 6_371_008.8


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None


def load_trip(folder: str | Path, include_route: bool = True) -> Trip:
    folder = Path(folder)
    trip_id = folder.name
    base = folder / trip_id
    track = pd.read_csv(str(base) + ".track", sep=r"\s+", names=["lon", "lat", "time"])
    nodes = pd.read_csv(str(base) + ".nodes", sep=r"\s+", names=["lon", "lat"]).to_numpy(float)
    arcs = pd.read_csv(str(base) + ".arcs", sep=r"\s+", names=["from", "to"]).to_numpy(int)
    route_path = Path(str(base) + ".route")
    route = pd.read_csv(route_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(int) if include_route and route_path.exists() else None
    track = track.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    valid = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    arcs = arcs[valid]
    if route is not None:
        route = route[(route >= 0) & (route < len(arcs))]
    return Trip(trip_id, track, nodes, arcs, route)


def inventory(root: str | Path) -> pd.DataFrame:
    rows = []
    for part in ("train", "student_test", "evaluator_test"):
        parent = Path(root) / part
        if not parent.exists():
            continue
        for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
            trip = load_trip(folder)
            base=folder/folder.name
            raw_track=pd.read_csv(str(base)+".track",sep=r"\s+",names=["lon","lat","time"])
            intervals=np.diff(raw_track.time.dropna().to_numpy(float)) if raw_track.time.notna().all() else np.diff(raw_track.time.dropna().to_numpy(float))
            degree=np.zeros(len(trip.nodes),dtype=int)
            parent_id=np.arange(len(trip.nodes))
            def find(x):
                while parent_id[x]!=x:
                    parent_id[x]=parent_id[parent_id[x]];x=parent_id[x]
                return x
            for a,b in trip.arcs:
                degree[a]+=1;degree[b]+=1
                ra,rb=find(int(a)),find(int(b))
                if ra!=rb: parent_id[ra]=rb
            active=np.flatnonzero(degree)
            rows.append({"partition": part, "trip_id": trip.trip_id, "observations": len(trip.track),
                         "nodes": len(trip.nodes), "arcs": len(trip.arcs),
                         "route_arcs": len(trip.route) if trip.route is not None else 0,
                         "invalid_gps_rows": int(raw_track.isna().any(axis=1).sum()),
                         "timestamp_reversals": int((intervals<0).sum()),
                         "gaps_over_5s": int((intervals>5).sum()),
                         "max_gap_s": float(intervals.max()) if len(intervals) else 0.0,
                         "isolated_nodes": int((degree==0).sum()),
                         "active_components": len({find(int(i)) for i in active}),
                         "track_bytes": (Path(str(base)+".track").stat().st_size),
                         "nodes_bytes": (Path(str(base)+".nodes").stat().st_size),
                         "arcs_bytes": (Path(str(base)+".arcs").stat().st_size),
                         "route_bytes": (Path(str(base)+".route").stat().st_size) if Path(str(base)+".route").exists() else 0})
    return pd.DataFrame(rows)


def _bearing(a, b):
    lon1, lat1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lon2, lat2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    dl = lon2 - lon1
    return np.arctan2(np.sin(dl) * np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dl))


def candidate_table(trip: Trip, k: int = 8):
    """Return per-observation road candidates and labels from supplied route geometry.

    Labels are assigned by nearest route segment in local metric coordinates. Candidate
    generation is independent of labels and uses all network edges.
    """
    xy = trip.nodes[trip.arcs]
    mid = xy.mean(axis=1)
    # Haversine BallTree uses [latitude, longitude] radians.
    tree = BallTree(np.radians(mid[:, ::-1]), metric="haversine")
    gps = trip.track[["lon", "lat"]].to_numpy(float)
    n = len(gps)
    k = min(k, len(trip.arcs))
    _, ids = tree.query(np.radians(gps[:, ::-1]), k=k)
    a, b = xy[:, 0, :], xy[:, 1, :]
    mid_bearing = _bearing(a, b)
    length = np.linalg.norm((b-a) * np.array([111_320*np.cos(np.radians(gps[:,1].mean())), 111_320]), axis=1)
    point_bearing = np.zeros(n)
    if n > 1:
        point_bearing[:-1] = _bearing(gps[:-1], gps[1:])
        point_bearing[-1] = point_bearing[-2]
    dt = np.r_[1., np.diff(trip.track.time.to_numpy(float))]
    speed = np.r_[0., np.linalg.norm(np.diff(gps, axis=0) * np.array([111_320*np.cos(np.radians(gps[:,1].mean())),111_320]),axis=1) / np.maximum(dt[1:],1e-3)]
    labels = np.full(n, -1, int)
    if trip.route is not None and len(trip.route):
        route_ids = np.unique(trip.route)
        route_ids = route_ids[route_ids < len(xy)]
        # For each fix, use the nearest segment in the provided route.
        rt = BallTree(np.radians(mid[route_ids, ::-1]), metric="haversine")
        labels = route_ids[rt.query(np.radians(gps[:, ::-1]), k=1, return_distance=False)[:, 0]]
    rows=[]
    for i in range(n):
        ids_i=ids[i]
        dlat=np.radians(gps[i,1]-mid[ids_i,1])*EARTH_M
        dlon=np.radians(gps[i,0]-mid[ids_i,0])*EARTH_M*np.cos(np.radians(gps[i,1]))
        delta=np.arctan2(np.sin(point_bearing[i]-mid_bearing[ids_i]),np.cos(point_bearing[i]-mid_bearing[ids_i]))
        for rank, arc in enumerate(ids_i):
            rows.append((i, arc, float(np.hypot(dlat[rank],dlon[rank])), float(abs(delta[rank])),
                         float(length[arc]), float(speed[i]), float(dt[i]), float(rank), int(arc==labels[i]), int(labels[i])))
    return pd.DataFrame(rows, columns=["obs", "arc_id", "distance_m", "heading_error", "edge_length_m", "speed_mps", "time_gap_s", "candidate_rank", "label", "reference_arc_id"])

