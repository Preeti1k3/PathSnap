from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import folium
from pathsnap.data import inventory, load_trip

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='../Student folder');p.add_argument('--output',default='artifacts/exploration');p.add_argument('--trip',default=None);a=p.parse_args()
    root=Path(a.data).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    inv=inventory(root);inv.to_csv(out/'dataset_inventory.csv',index=False)
    summary=inv.groupby('partition').agg(trips=('trip_id','count'),gps_observations=('observations','sum'),mean_observations=('observations','mean'),nodes_mean=('nodes','mean'),arcs_mean=('arcs','mean')).round(2)
    summary.to_csv(out/'partition_summary.csv')
    folder=Path(a.trip) if a.trip else sorted((root/'train').iterdir())[0]
    trip=load_trip(folder);gps=trip.track[['lon','lat']].to_numpy()
    m=folium.Map([float(gps[:,1].mean()),float(gps[:,0].mean())],zoom_start=14,control_scale=True)
    folium.PolyLine(gps[:,::-1].tolist(),color='blue',weight=3,tooltip='Raw GPS').add_to(m)
    if trip.route is not None:
        coords=[]
        for arc in trip.route:
            coords.extend(trip.nodes[trip.arcs[arc]][:,::-1].tolist())
        folium.PolyLine(coords,color='green',weight=4,tooltip='Supplied reference route').add_to(m)
    m.save(out/'sample_trajectory.html')
    intervals=trip.track.time.diff().dropna()
    pd.DataFrame([{'trip_id':trip.trip_id,'gps_points':len(trip.track),'route_edges':len(trip.route) if trip.route is not None else 0,'invalid_gps_rows_dropped':0,'gap_gt_5s':int((intervals>5).sum()),'gap_gt_30s':int((intervals>30).sum()),'median_gap_s':float(intervals.median()) if len(intervals) else None,'longitude_min':float(gps[:,0].min()),'longitude_max':float(gps[:,0].max()),'latitude_min':float(gps[:,1].min()),'latitude_max':float(gps[:,1].max())}]).to_csv(out/'sample_trip_summary.csv',index=False)
    print(summary)
if __name__=='__main__':main()

