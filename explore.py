from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from pathsnap.data import inventory, load_trip


def make_map(trip, output: Path) -> None:
    gps = trip.track[["lon", "lat"]].to_numpy(float)
    center = [float(gps[:, 1].mean()), float(gps[:, 0].mean())]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True)

    # Plot nearby network edges as a GeoJSON feature collection to keep the HTML
    # compact. The per-trip network itself is retained in the original data.
    edge_xy = trip.nodes[trip.arcs]
    midpoints = edge_xy.mean(axis=1)
    lat0 = np.radians(center[0])
    delta = midpoints - np.array([center[1], center[0]])
    approx_m = np.linalg.norm(delta * np.array([111_320*np.cos(lat0), 111_320]), axis=1)
    local_edges = np.flatnonzero(approx_m < 1800)
    if len(local_edges) > 8000:
        local_edges = local_edges[::int(np.ceil(len(local_edges)/8000))]
    road_features = [{"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":edge_xy[i].tolist()}}
                     for i in local_edges]
    folium.GeoJson({"type":"FeatureCollection","features":road_features},
                   name="Nearby road network",
                   style_function=lambda _: {"color":"#94a3b8","weight":2,"opacity":.55}).add_to(fmap)

    gps_group = folium.FeatureGroup(name="Raw GPS observations", show=True)
    stride = max(1, int(np.ceil(len(gps)/2500)))
    for lon, lat in gps[::stride]:
        folium.CircleMarker([float(lat), float(lon)], radius=2, color="#2563eb",
                            fill=True, fill_opacity=.85, weight=1).add_to(gps_group)
    folium.PolyLine(gps[:, ::-1].tolist(), color="#2563eb", weight=2, opacity=.35).add_to(gps_group)
    gps_group.add_to(fmap)

    if trip.route is not None and len(trip.route):
        route_parts = []
        for arc_id in trip.route:
            edge = trip.nodes[trip.arcs[int(arc_id)]][:, ::-1].tolist()
            if route_parts and route_parts[-1][-1] == edge[0]:
                route_parts[-1].append(edge[1])
            else:
                route_parts.append(edge)
        folium.PolyLine(route_parts, color="#16a34a", weight=5, opacity=.9,
                        tooltip="Supplied reference route").add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(output)


def make_static_map(trip, output: Path) -> None:
    """Export the same GPS/network/reference layers used by the Folium map."""
    gps = trip.track[["lon", "lat"]].to_numpy(float)
    center_lat = float(gps[:, 1].mean())
    center_lon = float(gps[:, 0].mean())
    scale = np.array([111_320*np.cos(np.radians(center_lat)), 111_320])
    gps_xy = (gps - np.array([center_lon, center_lat])) * scale
    edges = trip.nodes[trip.arcs]
    mids = edges.mean(axis=1)
    delta = mids - np.array([center_lon, center_lat])
    local = np.linalg.norm(delta * scale, axis=1) < 1800
    road_xy = (edges[local] - np.array([center_lon, center_lat])) * scale
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.add_collection(LineCollection(road_xy, colors="#94a3b8", linewidths=.45, alpha=.5, label="Nearby road network"))
    stride = max(1, int(np.ceil(len(gps_xy)/2500)))
    ax.scatter(gps_xy[::stride, 0], gps_xy[::stride, 1], s=8, c="#2563eb", alpha=.8, label="Raw GPS observations", zorder=3)
    if trip.route is not None and len(trip.route):
        route_xy = trip.nodes[trip.arcs[trip.route]]
        route_xy = (route_xy - np.array([center_lon, center_lat])) * scale
        ax.add_collection(LineCollection(route_xy, colors="#16a34a", linewidths=2.2, alpha=.9, label="Supplied reference route", zorder=4))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local east-west distance (m)")
    ax.set_ylabel("Local north-south distance (m)")
    ax.set_title(f"Trip {trip.trip_id}: GPS, road network, and reference route")
    ax.grid(alpha=.2)
    ax.legend(loc="best")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect PathSnap partitions and make a Folium map")
    parser.add_argument("--data", default="../Student folder")
    parser.add_argument("--output", default="artifacts/exploration")
    parser.add_argument("--evaluator", default=None, help="Separate server-side evaluator partition directory")
    parser.add_argument("--trip", default=None, help="Trip directory or trip ID; defaults to the first training trip")
    args = parser.parse_args()
    root, out = Path(args.data).resolve(), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    inv = inventory(root)
    if args.evaluator:
        eval_inv = inventory(Path(args.evaluator).resolve())
        if len(eval_inv):
            eval_inv["partition"] = "evaluator_test"
            inv = pd.concat([inv, eval_inv], ignore_index=True)
    inv.to_csv(out / "raw_inventory.csv", index=False)
    partition_summary = (inv.groupby("partition", dropna=False)
                         .agg(trips=("trip_id", "count"),
                              gps_rows=("track_rows_raw", "sum"),
                              clean_observations=("observations_clean", "sum"),
                              mean_observations=("observations_clean", "mean"),
                              invalid_gps_rows=("invalid_gps_rows", "sum"),
                              gaps_over_5s=("gaps_over_5s", "sum"),
                              gaps_over_30s=("gaps_over_30s", "sum"),
                              nodes=("nodes", "sum"), arcs=("arcs", "sum"),
                              route_arcs=("route_arcs", "sum"),
                              isolated_nodes=("isolated_nodes", "sum"),
                              active_components_sum=("weak_components_active_nodes", "sum"),
                              track_bytes=("track_bytes", "sum"), nodes_bytes=("nodes_bytes", "sum"),
                              arcs_bytes=("arcs_bytes", "sum"), route_bytes=("route_bytes", "sum"))
                         .reset_index())
    partition_summary.to_csv(out / "partition_summary.csv", index=False)

    train = inv[inv.partition.eq("train")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].hist(train.observations_clean, bins=15, color="#0f766e")
    axes[0].set(title="Training trip length", xlabel="GPS observations", ylabel="Trips")
    axes[1].hist(train.max_gap_s, bins=15, color="#3b82f6")
    axes[1].set(title="Largest time gap per trip", xlabel="Seconds", ylabel="Trips")
    axes[2].scatter(train.nodes, train.arcs, alpha=.75, color="#f97316")
    axes[2].set(title="Road network size", xlabel="Nodes", ylabel="Arcs")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Training partition exploration (80 trips)", fontsize=14, fontweight="bold")
    fig.savefig(out / "dataset_exploration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.trip:
        requested = Path(args.trip)
        folder = requested if requested.is_dir() else root / "train" / str(args.trip)
    else:
        folder = sorted(p for p in (root / "train").iterdir() if p.is_dir())[0]
    trip = load_trip(folder)
    make_map(trip, out / "sample_trajectory.html")
    make_static_map(trip, out / "sample_trajectory.png")
    trip_row = inv[inv.trip_id.eq(trip.trip_id)].iloc[0].to_dict()
    (out / "sample_trip_summary.json").write_text(json.dumps(trip_row, indent=2, default=float))
    print(partition_summary.to_string(index=False))
    print(f"Folium map: {out / 'sample_trajectory.html'}")


if __name__ == "__main__":
    main()
