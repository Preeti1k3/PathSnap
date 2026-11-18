
from pathlib import Path
import pandas as pd


def main():
    inventory_path = Path("artifacts/exploration/raw_inventory.csv")

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {inventory_path}"
        )

    inv = pd.read_csv(inventory_path)

    required = [
        "partition",
        "trip_id",
        "track_rows",
        "nodes_rows",
        "arcs_rows",
    ]

    missing = [col for col in required if col not in inv.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    summary = (
        inv.groupby("partition")
        .agg(
            trips=("trip_id", "count"),
            gps_observations=("track_rows", "sum"),
            mean_observations=("track_rows", "mean"),
            nodes_mean=("nodes_rows", "mean"),
            arcs_mean=("arcs_rows", "mean"),
        )
        .round(2)
    )

    output_path = Path(
        "artifacts/exploration/partition_summary.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)

    print("\nDATASET PARTITION SUMMARY")
    print(summary)

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
