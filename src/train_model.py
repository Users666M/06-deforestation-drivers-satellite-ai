"""Sesi 4 - Machine Learning: klasifikasi penyebab deforestasi.

Evaluasi: (1) hold-out stratified, (2) CV biasa, (3) spatial CV (leave-kabupaten-out)
untuk menguji generalisasi ke wilayah yang belum pernah dilihat model.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (DRIVER_LABEL, DRIVERS, FEATURES, FIG_DIR, MODEL_DIR, PRED_CSV, REPORT_DIR, SEED)
from .features import add_features, get_xy, load_alerts


def candidate_models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    df = load_alerts()
    X, y = get_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for name, m in candidate_models().items():
        sc = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro")
        cv[name] = {"cv_f1_macro_mean": round(float(sc.mean()), 4), "cv_f1_macro_std": round(float(sc.std()), 4)}
        print(f"{name:22s} CV macro-F1 = {sc.mean():.3f} +/- {sc.std():.3f}")
    best_name = max(cv, key=lambda k: cv[k]["cv_f1_macro_mean"])
    best = candidate_models()[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)

    metrics = {
        "best_model": best_name, "cv": cv,
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_f1_macro": round(float(f1_score(yte, pred, average="macro")), 4),
    }
    rep = classification_report(yte, pred, labels=DRIVERS, digits=3)
    (REPORT_DIR / "classification_report.txt").write_text(rep)
    print(f"\nBest: {best_name}\n{rep}")

    # Spatial CV
    groups = df["kabupaten"]
    spatial = cross_val_score(candidate_models()[best_name], X, y, cv=GroupKFold(5), groups=groups, scoring="f1_macro")
    metrics["spatial_cv_f1_macro_mean"] = round(float(spatial.mean()), 4)
    metrics["spatial_cv_f1_macro_std"] = round(float(spatial.std()), 4)
    print(f"Spatial CV macro-F1 = {spatial.mean():.3f} +/- {spatial.std():.3f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yte, pred, labels=DRIVERS, display_labels=[DRIVER_LABEL[d] for d in DRIVERS],
                                            normalize="true", cmap="Blues", ax=ax, xticks_rotation=40, values_format=".2f")
    ax.set_title(f"Confusion matrix (test, ternormalisasi) - {best_name}"); fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrix.png", dpi=150); plt.close(fig)

    # Permutation importance
    pi = permutation_importance(best, Xte, yte, n_repeats=5, random_state=SEED, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#0072B2"); ax.set_xlabel("Penurunan macro-F1 saat fitur diacak"); ax.set_title("Pentingnya fitur (permutation importance)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_feature_importance.png", dpi=150); plt.close(fig)
    metrics["top_features"] = imp.sort_values(ascending=False).head(5).round(4).to_dict()

    # Prediksi out-of-fold untuk seluruh data (dipakai dashboard)
    oof = cross_val_predict(candidate_models()[best_name], X, y, cv=skf)
    out = add_features(df)[["alert_id", "date", "lat", "lon", "kabupaten", "area_ha", "in_protected", "driver"]].copy()
    out["predicted_driver"] = oof
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(PRED_CSV, index=False)
    metrics["oof_accuracy"] = round(float(accuracy_score(y, oof)), 4)

    joblib.dump(best.fit(X, y), MODEL_DIR / "driver_classifier.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
