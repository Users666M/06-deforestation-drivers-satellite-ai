"""Membuat dataset SINTETIS alert deforestasi Bengkulu.

Struktur kolom meniru data nyata (mis. GLAD / Global Forest Watch + fitur citra
satelit + layer GIS), tetapi nilainya dibangkitkan dari distribusi per-driver.
Ganti dengan data asli bila tersedia.
"""
import numpy as np
import pandas as pd

from .config import DATA_CSV, DATA_DIR, DRIVERS, KABUPATEN, N_ALERTS, SEED

BASE_PRIOR = {"smallholder_agriculture": .32, "oil_palm": .22, "mining": .08,
              "infrastructure": .10, "logging": .14, "fire_natural": .14}

REGION_BIAS = {
    "Mukomuko": {"oil_palm": 3.0, "logging": 1.5},
    "Bengkulu Utara": {"oil_palm": 2.5, "mining": 2.0},
    "Bengkulu Tengah": {"mining": 3.0, "infrastructure": 1.5},
    "Kota Bengkulu": {"infrastructure": 4.0},
    "Lebong": {"smallholder_agriculture": 2.0, "logging": 2.0},
    "Rejang Lebong": {"smallholder_agriculture": 3.0},
    "Kepahiang": {"smallholder_agriculture": 3.0},
    "Seluma": {"oil_palm": 2.0, "smallholder_agriculture": 1.5},
    "Bengkulu Selatan": {"smallholder_agriculture": 1.5, "fire_natural": 1.5},
    "Kaur": {"fire_natural": 2.0, "oil_palm": 1.5},
}

# log_area=(mu,sd) lognormal | ndvi_after,nbr,slope,elev,compact=(mean,sd)
# road/river/settle = skala eksponensial (km) | prot = peluang di kawasan lindung
PARAMS = {
    "smallholder_agriculture": dict(log_area=(.6, .6), ndvi_after=(.45, .10), nbr=(.10, .08), road=2.0,
                                    river=3.0, settle=1.5, slope=(15, 8), elev=(600, 300), compact=(.55, .12), prot=.15),
    "oil_palm": dict(log_area=(1.8, .6), ndvi_after=(.30, .08), nbr=(.18, .10), road=3.0,
                     river=4.0, settle=6.0, slope=(6, 4), elev=(80, 60), compact=(.70, .10), prot=.05),
    "mining": dict(log_area=(1.4, .7), ndvi_after=(.15, .08), nbr=(.10, .08), road=4.0,
                   river=1.5, settle=8.0, slope=(10, 6), elev=(120, 100), compact=(.60, .12), prot=.08),
    "infrastructure": dict(log_area=(1.0, .5), ndvi_after=(.25, .08), nbr=(.08, .06), road=.3,
                           river=3.0, settle=1.0, slope=(7, 5), elev=(300, 250), compact=(.25, .10), prot=.04),
    "logging": dict(log_area=(1.2, .6), ndvi_after=(.50, .10), nbr=(.12, .08), road=5.0,
                    river=3.0, settle=9.0, slope=(20, 8), elev=(700, 300), compact=(.45, .12), prot=.30),
    "fire_natural": dict(log_area=(2.0, .8), ndvi_after=(.35, .12), nbr=(.45, .15), road=4.0,
                         river=3.0, settle=5.0, slope=(12, 8), elev=(200, 150), compact=(.40, .12), prot=.15),
}


def generate(n: int = N_ALERTS, seed: int = SEED, label_noise: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kabs = list(KABUPATEN)
    kab_idx = rng.integers(0, len(kabs), n)

    drv = np.empty(n, dtype=object)
    for i, k in enumerate(kab_idx):
        w = np.array([BASE_PRIOR[d] * REGION_BIAS.get(kabs[k], {}).get(d, 1.0) for d in DRIVERS])
        drv[i] = rng.choice(DRIVERS, p=w / w.sum())

    cols = {c: np.zeros(n) for c in [
        "area_ha", "ndvi_before", "ndvi_after", "nbr_drop", "dist_road_km", "dist_river_km",
        "dist_settlement_km", "slope_deg", "elevation_m", "compactness", "in_protected"]}
    days = np.arange(1096)  # 2023-01-01 .. 2025-12-31
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(days, unit="D")
    dry = np.isin(dates.month, [7, 8, 9, 10])
    date_offset = np.zeros(n, dtype=int)

    for d in DRIVERS:
        m = drv == d
        c = int(m.sum())
        p = PARAMS[d]
        cols["area_ha"][m] = np.clip(rng.lognormal(*p["log_area"], c), .1, 200)
        before = np.clip(rng.normal(.78, .05, c), .5, .95)
        cols["ndvi_before"][m] = before
        cols["ndvi_after"][m] = np.clip(np.minimum(before - .05, rng.normal(*p["ndvi_after"], c)), 0, .9)
        cols["nbr_drop"][m] = np.clip(rng.normal(*p["nbr"], c), 0, 1)
        cols["dist_road_km"][m] = rng.exponential(p["road"], c)
        cols["dist_river_km"][m] = rng.exponential(p["river"], c)
        cols["dist_settlement_km"][m] = rng.exponential(p["settle"], c)
        cols["slope_deg"][m] = np.clip(np.abs(rng.normal(*p["slope"], c)), 0, 45)
        cols["elevation_m"][m] = np.clip(rng.normal(*p["elev"], c), 0, 2000)
        cols["compactness"][m] = np.clip(rng.normal(*p["compact"], c), .05, 1)
        cols["in_protected"][m] = rng.random(c) < p["prot"]
        w = np.where(dry, 1.0, .25) if d == "fire_natural" else np.ones(len(days))
        date_offset[m] = rng.choice(days, size=c, p=w / w.sum())

    lat = np.array([KABUPATEN[kabs[k]][0] for k in kab_idx]) + rng.normal(0, .16, n)
    lon = np.array([KABUPATEN[kabs[k]][1] for k in kab_idx]) + rng.normal(0, .16, n)

    # noise label: anotator manusia tidak selalu benar
    flip = rng.random(n) < label_noise
    drv_obs = drv.copy()
    drv_obs[flip] = rng.choice(DRIVERS, int(flip.sum()))

    df = pd.DataFrame({
        "alert_id": [f"BKL-{i:05d}" for i in range(n)],
        "date": (pd.to_datetime("2023-01-01") + pd.to_timedelta(date_offset, unit="D")).strftime("%Y-%m-%d"),
        "lat": lat.round(5), "lon": lon.round(5),
        "kabupaten": [kabs[k] for k in kab_idx],
        **{k: v.round(4) for k, v in cols.items()},
        "driver": drv_obs,
    })
    df["in_protected"] = df["in_protected"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_CSV, index=False)
    print(f"Saved {len(df)} rows -> {DATA_CSV}")
    print(df["driver"].value_counts().to_string())


if __name__ == "__main__":
    main()
