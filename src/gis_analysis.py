"""Sesi 3 - GIS: analisis spasial (grid hotspot, ringkasan wilayah, GeoJSON)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (DRIVER_COLOR, DRIVER_LABEL, DRIVERS, FIG_DIR, GEOJSON, KABUPATEN, REPORT_DIR)
from .features import load_alerts


def grid_hotspots(df, cell=0.1):
    g = df.assign(cell_lat=(np.floor(df["lat"] / cell) * cell).round(2),
                  cell_lon=(np.floor(df["lon"] / cell) * cell).round(2))
    t = (g.groupby(["cell_lat", "cell_lon"])
         .agg(alerts=("alert_id", "count"), area_ha=("area_ha", "sum"),
              top_driver=("driver", lambda s: s.value_counts().index[0]))
         .reset_index().sort_values("alerts", ascending=False))
    t["area_ha"] = t["area_ha"].round(1)
    return t


def kabupaten_summary(df):
    s = (df.groupby("kabupaten")
         .agg(alerts=("alert_id", "count"), area_ha=("area_ha", "sum"),
              pct_protected=("in_protected", "mean"),
              dominant_driver=("driver", lambda x: x.value_counts().index[0]))
         .sort_values("area_ha", ascending=False))
    s["area_ha"] = s["area_ha"].round(1)
    s["pct_protected"] = (s["pct_protected"] * 100).round(1)
    return s


def export_geojson(df, path=GEOJSON):
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
              "properties": {"id": r.alert_id, "date": r.date.strftime("%Y-%m-%d"), "driver": r.driver,
                             "area_ha": r.area_ha, "kabupaten": r.kabupaten, "in_protected": int(r.in_protected)}}
             for r in df.itertuples()]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))


def plot_maps(df, hot):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    ax = axes[0]
    for d in DRIVERS:
        s = df[df["driver"] == d]
        ax.scatter(s["lon"], s["lat"], s=5, alpha=.55, color=DRIVER_COLOR[d], label=DRIVER_LABEL[d], linewidths=0)
    for k, (la, lo) in KABUPATEN.items():
        ax.annotate(k, (lo, la), fontsize=7, ha="center", color="#333")
    ax.set_title("Sebaran alert menurut penyebab"); ax.legend(markerscale=3, fontsize=8, frameon=False)
    ax2 = axes[1]
    hb = ax2.hexbin(df["lon"], df["lat"], gridsize=35, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax2, shrink=.7, label="Jumlah alert per sel")
    ax2.set_title("Kepadatan alert (hotspot)")
    for a in axes:
        a.set_xlabel("Bujur"); a.set_aspect("equal")
    axes[0].set_ylabel("Lintang")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_spatial_maps.png", dpi=150); plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_alerts()
    hot = grid_hotspots(df)
    hot.head(15).to_csv(REPORT_DIR / "top_hotspot_cells.csv", index=False)
    kab = kabupaten_summary(df)
    kab.to_csv(REPORT_DIR / "kabupaten_summary.csv")
    export_geojson(df)
    plot_maps(df, hot)
    print(kab.to_string()); print("\nTop 5 hotspot cells:"); print(hot.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
