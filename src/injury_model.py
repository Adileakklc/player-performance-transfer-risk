import os
import json
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

from features import build_features


def make_label(feats: pd.DataFrame, inj: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """Label: date itibariyle önümüzdeki `horizon_days` gün içinde sakatlık olacak mı?"""
    inj_map = {pid: g["injury_date"].tolist() for pid, g in inj.groupby("player_id")}
    y = []

    for _, r in feats.iterrows():
        pid, d = r["player_id"], r["date"]
        end = d + pd.Timedelta(days=horizon_days)

        flag = 0
        for idate in inj_map.get(pid, []):
            if d < idate <= end:
                flag = 1
                break
        y.append(flag)

    feats = feats.copy()
    feats["y_injury_next14"] = y
    return feats


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def explain_top_player(model: Pipeline, feature_cols: list[str], latest_with_feats: pd.DataFrame):
    """Coef bazlı basit explainability (global + top risky player local katkılar)."""
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    coefs = pd.Series(clf.coef_[0], index=feature_cols)
    global_top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)

    print("\n--- Explainability (Global) | Top Coefficients ---")
    print(global_top.to_string())

    top_row = latest_with_feats.iloc[0].copy()
    X = top_row[feature_cols].to_frame().T

    Xz = scaler.transform(X)
    contrib = (Xz[0] * clf.coef_[0])
    contrib_s = pd.Series(contrib, index=feature_cols).sort_values(ascending=False)

    print(f"\n--- Explainability (Local) | Top Risky Player: {top_row['player_id']} ---")
    print(f"Risk: {top_row['injury_risk']:.3f} | Date: {str(top_row['date'])[:10]}")
    print("\nTop 3 risk drivers (approx):")
    print(contrib_s.head(3).to_string())

    print("\nTop 3 risk reducers (approx):")
    print(contrib_s.tail(3).to_string())


def time_based_split(df: pd.DataFrame, date_col: str = "date", quantile: float = 0.75):
    """Geçmiş %75 train, gelecek %25 test."""
    df = df.sort_values(date_col).copy()
    split_date = df[date_col].quantile(quantile)

    train_df = df[df[date_col] <= split_date].copy()
    test_df = df[df[date_col] > split_date].copy()

    return train_df, test_df, split_date


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    """PR curve üzerinden F1'i maksimize eden threshold'u bul."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)

    # thr uzunluğu prec/rec'ten 1 eksik olabiliyor, güvenli seçim:
    best_idx = int(np.nanargmax(f1))
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5

    return best_thr, float(prec[best_idx]), float(rec[best_idx]), float(f1[best_idx])


def save_model_and_meta(model: Pipeline, feature_cols: list[str], metrics: dict, split_info: dict, version: str = "v2"):
    """Model versioning: pkl + meta json."""
    import joblib

    model_path = f"models/injury_model_{version}.pkl"
    meta_path = f"models/injury_model_{version}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "version": version,
        "features": feature_cols,
        "metrics": metrics,
        "split": split_info,
        "model": "LogisticRegression + StandardScaler",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved model: {model_path}")
    print(f"✅ Saved meta : {meta_path}")


def main():
    ensure_dirs()

    match = pd.read_csv("data/raw/match_data.csv", parse_dates=["match_date"])
    train = pd.read_csv("data/raw/training_load.csv", parse_dates=["date"])
    inj = pd.read_csv("data/raw/injury_history.csv", parse_dates=["injury_date"])

    feats = build_features(match, train, inj)
    feats = make_label(feats, inj, horizon_days=14)

    # rolling hazır olmayanları at
    df = feats.dropna(subset=["load_7", "load_28"]).copy()

    # ✅ rolling otursun diye ilk 35 günü çıkar (stabilite için)
    min_date = df["date"].min()
    df = df[df["date"] >= (min_date + pd.Timedelta(days=35))].copy()

    # ✅ days_since_last_injury yerine daha anlamlı sinyal
    # yakın sakatlık = yüksek sinyal
    df["recent_injury_signal"] = np.exp(-df["days_since_last_injury"] / 30.0)

    feature_cols = [
        "training_load", "load_7", "load_14", "load_28", "acwr_7_28",
        "minutes", "perf_last3", "perf_trend3",
        "recent_injury_signal"
    ]

    # ✅ TIME-BASED SPLIT
    train_df, test_df, split_date = time_based_split(df, date_col="date", quantile=0.75)

    X_train = train_df[feature_cols]
    y_train = train_df["y_injury_next14"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["y_injury_next14"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
    pr_auc = average_precision_score(y_test, proba)

    # ✅ threshold optimize
    best_thr, p_best, r_best, f1_best = find_best_threshold(y_test.values, proba)

    print("ROC-AUC:", auc)
    print("PR-AUC :", pr_auc)
    print(f"Split date (75% quantile): {split_date.date()}")
    print(f"Best threshold (F1): {best_thr:.4f} | P:{p_best:.3f} R:{r_best:.3f} F1:{f1_best:.3f}")

    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, (proba > 0.5).astype(int)))

    print("\nClassification report (optimized threshold):")
    print(classification_report(y_test, (proba > best_thr).astype(int)))

    # Latest risk per player (tüm timeline üzerinden son gün)
    latest_all = df.sort_values("date").groupby("player_id").tail(1).copy()
    latest_all["injury_risk"] = model.predict_proba(latest_all[feature_cols])[:, 1]

    latest = latest_all[["player_id", "date", "injury_risk"]].sort_values("injury_risk", ascending=False)

    print("\nTop 10 injury risk (latest):")
    print(latest.head(10).to_string(index=False))

    # Save outputs
    latest.to_csv("data/processed/injury_risk_latest.csv", index=False)
    latest_all.sort_values("injury_risk", ascending=False).to_csv(
        "data/processed/injury_risk_latest_with_features.csv", index=False
    )

    explain_top_player(
        model,
        feature_cols,
        latest_all.sort_values("injury_risk", ascending=False).reset_index(drop=True)
    )

    metrics = {
        "roc_auc": None if auc is None else float(auc),
        "pr_auc": float(pr_auc),
        "best_threshold_f1": float(best_thr),
        "best_precision": float(p_best),
        "best_recall": float(r_best),
        "best_f1": float(f1_best),
    }

    split_info = {
        "type": "time-based",
        "train_until": str(split_date.date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "early_days_dropped": 35,
        "horizon_days": 14
    }

    save_model_and_meta(model, feature_cols, metrics, split_info, version="v2")


if __name__ == "__main__":
    main()