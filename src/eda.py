"""Sesi 1 - Data analysis: eksplorasi data alert deforestasi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DRIVER_COLOR, DRIVER_LABEL, DRIVERS, FEATURES, FIG_DIR, REPORT_DIR
from .features import add_features, load_alerts

PAL = [DRIVER_COLOR[d] for d in DRIVERS]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = add_features(load_alerts())
    df["driver_id"] = df["driver"].map(DRIVER_LABEL)
    order = [DRIVER_LABEL[d] for d in DRIVERS]
    pal = dict(zip(order, PAL))

    # 1. Distribusi driver
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["driver_id"].value_counts().reindex(order)
    ax.barh(order, counts.values, color=PAL, edgecolor="black", linewidth=.5)
    for i, v in enumerate(counts.values):
        ax.text(v + 20, i, f"{v} ({v / len(df):.0%})", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Jumlah alert"); ax.set_title("Distribusi penyebab deforestasi")
    ax.set_xlim(0, counts.max() * 1.25); fig.tight_layout(); fig.savefig(FIG_DIR / "01_driver_distribution.png", dpi=150); plt.close(fig)

    # 2. Tren bulanan
    monthly = df.groupby([df["date"].dt.to_period("M").dt.to_timestamp(), "driver_id"]).size().unstack(fill_value=0)[order]
    fig, ax = plt.subplots(figsize=(9, 4))
    monthly.plot.area(ax=ax, color=PAL, linewidth=0, alpha=.9)
    ax.set_xlabel(""); ax.set_ylabel("Alert / bulan"); ax.set_title("Tren bulanan alert (kebakaran memuncak di musim kering)")
    ax.legend(fontsize=8, ncol=2, frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR / "02_monthly_trend.png", dpi=150); plt.close(fig)

    # 3. Perbandingan fitur kunci
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (col, title) in zip(axes.ravel(), [
        ("ndvi_drop", "Penurunan NDVI"), ("nbr_drop", "Penurunan NBR (indikator terbakar)"),
        ("dist_road_km", "Jarak ke jalan (km)"), ("dist_settlement_km", "Jarak ke permukiman (km)"),
        ("slope_deg", "Kemiringan lereng (derajat)"), ("compactness", "Kekompakan bentuk patch")]):
        sns.boxplot(data=df, y="driver_id", x=col, order=order, palette=pal, hue="driver_id", legend=False,
                    ax=ax, fliersize=1, linewidth=.8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(""); ax.set_xlabel("")
        if col not in ("ndvi_drop", "dist_road_km"):
            ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_feature_by_driver.png", dpi=150); plt.close(fig)

    # 4. Korelasi
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[FEATURES].corr(), cmap="vlag", center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Korelasi antar fitur"); fig.tight_layout(); fig.savefig(FIG_DIR / "04_correlation.png", dpi=150); plt.close(fig)

    summary = df.groupby("driver")[["area_ha", "ndvi_drop", "nbr_drop", "dist_road_km", "dist_settlement_km", "slope_deg", "elevation_m", "compactness", "in_protected"]].mean().round(2)
    summary.insert(0, "n_alerts", df["driver"].value_counts())
    summary.to_csv(REPORT_DIR / "eda_summary_by_driver.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
