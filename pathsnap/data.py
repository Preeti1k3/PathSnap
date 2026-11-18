from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_M = 6_371_008.8
TRACK_COLUMNS = ["lon", "lat", "time"]


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None


def _read_whitespace(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one headerless whitespace-delimited dataset file."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="c")


def load_trip(folder: str | Path, include_route: bool = True) -> Trip:
    """Read and validate one trip directory named with its trip ID."""
    folder = Path(folder)
    trip_id = folder.name
    base = folder / trip_id

    track = _read_whitespace(Path(str(base) + ".track"), TRACK_COLUMNS)
    for col in TRACK_COLUMNS:
        track[col] = pd.to_numeric(track[col], errors="coerce")
    track = track.replace([np.inf, -np.inf], np.nan).dropna()
    track = (track.drop_duplicates("time", keep="first")
                  .sort_values("time", kind="stable")
                  .reset_index(drop=True))
    if track.empty:
        raise ValueError(f"Trip {trip_id} has no valid GPS observations")

    nodes = _read_whitespace(Path(str(base) + ".nodes"), ["lon", "lat"]).to_numpy(float)
    raw_arcs = _read_whitespace(Path(str(base) + ".arcs"), ["from", "to"]).to_numpy(int)
    valid_arc = ((raw_arcs[:, 0] >= 0) & (raw_arcs[:, 0] < len(nodes)) &
                 (raw_arcs[:, 1] >= 0) & (raw_arcs[:, 1] < len(nodes)))
    # Preserve route arc references when malformed arc rows are removed.
    arc_remap = np.full(len(raw_arcs), -1, dtype=int)
    arc_remap[valid_arc] = np.arange(int(valid_arc.sum()))
    arcs = raw_arcs[valid_arc]

    route_path = Path(str(base) + ".route")
    route = None
    if include_route and route_path.exists() and route_path.stat().st_size:
        route_raw = _read_whitespace(route_path, ["arc_id"])["arc_id"].to_numpy(int)
        in_range = (route_raw >= 0) & (route_raw < len(arc_remap))
        route_raw = route_raw[in_range]
        route = arc_remap[route_raw]
        route = route[route >= 0]
    if not len(arcs):
        raise ValueError(f"Trip {trip_id} has no valid road segments")
    if not np.isfinite(nodes).all():
        raise ValueError(f"Trip {trip_id} contains invalid road-node coordinates")
    return Trip(trip_id, track, nodes, arcs, route)


def _component_count(n_nodes: int, arcs: np.ndarray) -> tuple[int, int]:
    parent = np.arange(n_nodes)
    degree = np.zeros(n_nodes, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for a, b in arcs:
        a, b = int(a), int(b)
        degree[a] += 1
        degree[b] += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    active = np.flatnonzero(degree)
    components = len({find(int(x)) for x in active})
    return int((degree == 0).sum()), components


def inventory(root: str | Path) -> pd.DataFrame:
    """Build trip-level data-quality and network inventory for present partitions."""
    rows = []
    root = Path(root)
    direct_trip_dirs = [p for p in root.iterdir() if p.is_dir() and (p / f"{p.name}.track").is_file()] if root.is_dir() else []
    if direct_trip_dirs:
        partitions = [("evaluator_test", root)]
    else:
        partitions = [(part, root / part) for part in ("train", "student_test", "evaluator_test")]
    for part, parent in partitions:
        if not parent.is_dir():
            continue
        for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
            trip = load_trip(folder)
            base = folder / folder.name
            raw = _read_whitespace(Path(str(base) + ".track"), TRACK_COLUMNS)
            numeric = raw.apply(pd.to_numeric, errors="coerce").to_numpy(float)
            invalid_rows = int((~np.isfinite(numeric)).any(axis=1).sum())
            times = numeric[:, 2]
            times = times[np.isfinite(times)]
            gaps = np.diff(times)
            isolated, components = _component_count(len(trip.nodes), trip.arcs)
            sizes = {ext: Path(str(base) + ext).stat().st_size if Path(str(base) + ext).exists() else 0
                     for ext in (".track", ".nodes", ".arcs", ".route")}
            gps = trip.track[["lon", "lat"]].to_numpy(float)
            rows.append({
                "partition": part, "trip_id": trip.trip_id,
                "track_rows_raw": len(raw), "observations_clean": len(trip.track),
                "invalid_gps_rows": invalid_rows,
                "timestamp_reversals": int((gaps < 0).sum()),
                "gaps_over_5s": int((gaps > 5).sum()),
                "gaps_over_30s": int((gaps > 30).sum()),
                "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
                "median_gap_s": float(np.median(gaps)) if len(gaps) else 0.0,
                "longitude_min": float(gps[:, 0].min()), "longitude_max": float(gps[:, 0].max()),
                "latitude_min": float(gps[:, 1].min()), "latitude_max": float(gps[:, 1].max()),
                "nodes": len(trip.nodes), "arcs": len(trip.arcs),
                "route_arcs": len(trip.route) if trip.route is not None else 0,
                "isolated_nodes": isolated, "weak_components_active_nodes": components,
                "track_bytes": sizes[".track"], "nodes_bytes": sizes[".nodes"],
                "arcs_bytes": sizes[".arcs"], "route_bytes": sizes[".route"],
            })
    return pd.DataFrame(rows)


def _bearing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lon1, lat1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lon2, lat2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    dlon = lon2 - lon1
    return np.arctan2(np.sin(dlon) * np.cos(lat2),
                      np.cos(lat1) * np.sin(lat2) -
                      np.sin(lat1) * np.cos(lat2) * np.cos(dlon))


def _point_segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Local equirectangular point-to-segment distance in metres, shape (N, M)."""
    lat0 = np.radians(float(np.mean(points[:, 1])))
    scale = np.array([111_320.0 * np.cos(lat0), 111_320.0])
    p = points * scale
    a, b = starts * scale, ends * scale
    v = b - a
    denom = np.sum(v * v, axis=1)
    w = p[:, None, :] - a[None, :, :]
    frac = np.sum(w * v[None, :, :], axis=2) / np.maximum(denom[None, :], 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    nearest = a[None, :, :] + frac[:, :, None] * v[None, :, :]
    return np.linalg.norm(p[:, None, :] - nearest, axis=2)


def candidate_table(trip: Trip, k: int = 8, include_targets: bool = True) -> pd.DataFrame:
    """Create nearest-road candidates and route-derived binary labels per GPS fix.

    Each row is one (observation, candidate arc). The target arc is the closest
    supplied reference-route segment by local metric point-to-segment distance.
    Candidate search uses a haversine BallTree over segment midpoints; final feature
    distance is point-to-segment distance in a local equirectangular projection.
    """
    edges = trip.nodes[trip.arcs]
    starts, ends = edges[:, 0], edges[:, 1]
    midpoints = (starts + ends) / 2
    gps = trip.track[["lon", "lat"]].to_numpy(float)
    n, n_arcs = len(gps), len(trip.arcs)
    k = min(max(int(k), 1), n_arcs)

    tree = BallTree(np.radians(midpoints[:, ::-1]), metric="haversine")
    candidate_ids = tree.query(np.radians(gps[:, ::-1]), k=k, return_distance=False)
    labels = np.full(n, -1, dtype=int)
    if include_targets and trip.route is not None and len(trip.route):
        route_ids = np.unique(trip.route)
        route_edges = edges[route_ids]
        route_dist = _point_segment_distance(gps, route_edges[:, 0], route_edges[:, 1])
        labels = route_ids[np.argmin(route_dist, axis=1)]

    # Broadcast projection features over the K candidates.
    lat0 = np.radians(float(np.mean(gps[:, 1])))
    scale = np.array([111_320.0 * np.cos(lat0), 111_320.0])
    points = gps * scale
    ca, cb = starts[candidate_ids] * scale, ends[candidate_ids] * scale
    vec = cb - ca
    denom = np.maximum(np.sum(vec * vec, axis=2), 1e-12)
    rel = points[:, None, :] - ca
    frac = np.clip(np.sum(rel * vec, axis=2) / denom, 0.0, 1.0)
    nearest = ca + frac[:, :, None] * vec
    distance = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    lengths = np.linalg.norm(vec, axis=2)

    arc_bearing = _bearing(starts.reshape(-1, 2), ends.reshape(-1, 2))[candidate_ids]
    obs_bearing = np.zeros(n)
    if n > 1:
        obs_bearing[:-1] = _bearing(gps[:-1], gps[1:])
        obs_bearing[-1] = obs_bearing[-2]
    heading_error = np.abs(np.arctan2(np.sin(obs_bearing[:, None] - arc_bearing),
                                      np.cos(obs_bearing[:, None] - arc_bearing)))
    gaps = np.r_[1.0, np.diff(trip.track.time.to_numpy(float))]
    step_m = np.zeros(n)
    if n > 1:
        step_m[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = step_m / np.maximum(gaps, 1e-3)

    obs_idx = np.repeat(np.arange(n), k)
    arc_flat = candidate_ids.ravel()
    return pd.DataFrame({
        "obs": obs_idx,
        "arc_id": arc_flat,
        "distance_m": distance.ravel(),
        "heading_error": heading_error.ravel(),
        "edge_length_m": lengths.ravel(),
        "speed_mps": np.repeat(speed, k),
        "time_gap_s": np.repeat(gaps, k),
        "candidate_rank": np.tile(np.arange(k), n),
        "label": (candidate_ids == labels[:, None]).astype(np.int8).ravel(),
        "reference_arc_id": np.repeat(labels, k),
    })
