"""Konfigurasi global proyek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models"

SEED = 42
N_ALERTS = 6000

DRIVERS = [
    "smallholder_agriculture",
    "oil_palm",
    "mining",
    "infrastructure",
    "logging",
    "fire_natural",
]
DRIVER_LABEL = {
    "smallholder_agriculture": "Pertanian rakyat",
    "oil_palm": "Kebun sawit",
    "mining": "Pertambangan",
    "infrastructure": "Infrastruktur",
    "logging": "Pembalakan",
    "fire_natural": "Kebakaran/alami",
}
# Palet Okabe-Ito (ramah buta warna)
DRIVER_COLOR = {
    "smallholder_agriculture": "#E69F00",
    "oil_palm": "#009E73",
    "mining": "#D55E00",
    "infrastructure": "#0072B2",
    "logging": "#CC79A7",
    "fire_natural": "#F0E442",
}

# Titik pusat APROKSIMASI kabupaten/kota di Provinsi Bengkulu (lat, lon)
KABUPATEN = {
    "Mukomuko": (-2.58, 101.10),
    "Bengkulu Utara": (-3.30, 102.00),
    "Lebong": (-3.15, 102.25),
    "Rejang Lebong": (-3.45, 102.55),
    "Kepahiang": (-3.65, 102.58),
    "Bengkulu Tengah": (-3.75, 102.30),
    "Kota Bengkulu": (-3.80, 102.27),
    "Seluma": (-4.00, 102.45),
    "Bengkulu Selatan": (-4.35, 102.75),
    "Kaur": (-4.80, 103.40),
}

FEATURES = [
    "log_area", "ndvi_before", "ndvi_after", "ndvi_drop", "nbr_drop",
    "dist_road_km", "dist_river_km", "dist_settlement_km",
    "slope_deg", "elevation_m", "compactness", "in_protected", "dry_season",
]

DATA_CSV = DATA_DIR / "deforestation_alerts_synthetic.csv"
PRED_CSV = DATA_DIR / "alerts_with_predictions.csv"
GEOJSON = DATA_DIR / "alerts.geojson"
