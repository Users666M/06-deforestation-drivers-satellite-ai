#!/usr/bin/env bash
# Menjalankan seluruh pipeline dari awal sampai akhir.
set -euo pipefail
python -m src.generate_data        # 1. data (sintetis)
python -m src.eda                  # 2. analisis data
python -m src.gis_analysis         # 3. analisis spasial + GeoJSON
python -m src.train_model          # 4. machine learning
python -m src.parallel_benchmark   # 5. benchmark paralel (HPC)
python -m src.build_dashboard      # 6. dashboard HTML
python -m pytest -q
echo "Selesai. Buka reports/dashboard.html di browser."
