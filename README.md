# Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI
Studi kasus terintegrasi (sesi 6), menggabungkan analisis data, GIS, ML, HPC, dan dashboard.

Studi kasus berbasis materi *Informatics Practitioner Lecture Series* (Informatika Unib): analisis data, HPC, GIS, machine learning, HCI & visualisasi data, serta atribusi deforestasi dengan citra satelit dan AI.

> Data = **sintetis** (untuk pembelajaran). Laporan lengkap: [`docs/laporan_studi_kasus.md`](docs/laporan_studi_kasus.md)

![dashboard](reports/figures/09_dashboard_preview.png)

## Hasil singkat
- Random Forest: akurasi **86%**, macro-F1 **0,835** (hold-out), spatial CV macro-F1 0,830
- 6 kelas penyebab: pertanian rakyat, sawit, tambang, infrastruktur, pembalakan, kebakaran/alami
- Dashboard interaktif tanpa instalasi: buka `reports/dashboard.html`

## Struktur
```
src/
  generate_data.py       # data sintetis
  eda.py                 # sesi 1: analisis data
  parallel_benchmark.py  # sesi 2: HPC / paralel
  gis_analysis.py        # sesi 3: GIS (hotspot, GeoJSON)
  train_model.py         # sesi 4: machine learning
  build_dashboard.py     # sesi 5: HCI & dataviz
tests/                   # tes otomatis
data/                    # CSV + GeoJSON
reports/                 # grafik, metrik, dashboard.html
docs/laporan_studi_kasus.md
```

## Menjalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_all.sh            # atau jalankan modul satu per satu: python -m src.generate_data, dst.
```
Lalu buka `reports/dashboard.html`. Jalankan `python -m pytest -q` untuk tes.

## Memakai data asli
Pertahankan nama kolom di `data/deforestation_alerts_synthetic.csv` (lihat `src/config.py` -> `FEATURES`), ganti isinya dengan data GLAD/GFW + layer GIS, lalu jalankan ulang pipeline mulai dari `src.eda`.

## Catatan
Benchmark HPC di `reports/` dijalankan pada 1 core sehingga tanpa speedup. Jalankan ulang di mesin multi-core untuk hasil yang bermakna.

## Repo lain dalam rangkaian studi kasus ini
- `01-data-analysis-and-ai`
- `02-hpc-parallel-processing`
- `03-gis-spatial-analysis`
- `04-machine-learning-intro`
- `05-hci-data-visualization`
