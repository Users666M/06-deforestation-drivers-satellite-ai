"""Sesi 2 - HPC: benchmark pemrosesan paralel.

Mensimulasikan ekstraksi fitur dari banyak tile citra satelit (NDVI + smoothing +
deteksi penurunan vegetasi). Tiap tile independen -> "embarrassingly parallel".
Hasil speedup bergantung jumlah core mesin. Jalankan ulang di laptop/server/cluster Anda.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, REPORT_DIR

N_TILES, TILE, PASSES = 64, 1024, 6


def process_tile(seed: int) -> float:
    """NDVI sebelum/sesudah -> smoothing 3x3 -> persentase piksel yang kehilangan vegetasi."""
    rng = np.random.default_rng(seed)
    red0, nir0, red1, nir1 = (rng.random((TILE, TILE), dtype=np.float32) for _ in range(4))
    ndvi0 = (nir0 - red0) / (nir0 + red0 + 1e-6)
    ndvi1 = (nir1 - red1) / (nir1 + red1 + 1e-6)
    d = ndvi0 - ndvi1
    for _ in range(PASSES):  # smoothing 3x3 berulang (box filter)
        p = np.pad(d, 1, mode="edge")
        d = sum(p[i:i + TILE, j:j + TILE] for i in range(3) for j in range(3)) / 9.0
    return float((d > 0.05).mean())


def run(workers: int) -> float:
    t0 = time.perf_counter()
    if workers == 1:
        res = [process_tile(s) for s in range(N_TILES)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(process_tile, range(N_TILES), chunksize=max(1, N_TILES // (workers * 4))))
    assert len(res) == N_TILES
    return time.perf_counter() - t0


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cores = os.cpu_count() or 1
    worker_list = sorted({1, 2, 4, cores})
    print(f"CPU cores terdeteksi: {cores} | workers diuji: {worker_list}")
    rows = []
    for w in worker_list:
        t = min(run(w) for _ in range(2))  # ambil yang tercepat dari 2 kali jalan
        rows.append({"workers": w, "seconds": round(t, 3)})
        print(f"workers={w:2d}  {t:6.2f}s")
    df = pd.DataFrame(rows)
    df["speedup"] = (df["seconds"].iloc[0] / df["seconds"]).round(2)
    df["efficiency"] = (df["speedup"] / df["workers"]).round(2)
    df["cpu_cores_machine"] = cores
    df.to_csv(REPORT_DIR / "hpc_benchmark.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["workers"], df["speedup"], "o-", color="#0072B2", label="Speedup terukur")
    ax.plot(df["workers"], df["workers"], "--", color="gray", label="Ideal (linear)")
    ax.set_xlabel("Jumlah worker"); ax.set_ylabel("Speedup (x)"); ax.set_title(f"Benchmark paralel ({cores} core)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR / "08_hpc_speedup.png", dpi=150); plt.close(fig)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
