import os
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_transfer_label(df: pd.DataFrame, seed: int = 42):
    """
    Sentetik label: oyuncu bu sezon içinde takımdan ayrılma/transfer olma riski.
    Rule-based probability -> Bernoulli sample
    """
    rng = np.random.default_rng(seed)
    d = df.copy()

    contract_term = np.clip((12 - d["contract_months_left"]) / 12.0, -1, 2)
    interest_term = (d["interest_level"] - 3) / 2.0
    agent_term = d["agent_risk"].astype(float)

    salary_term = np.clip((d["salary_k"] - d["salary_k"].median()) / (d["salary_k"].std() + 1e-9), -2, 2)
    value_term = np.clip((d["market_value_m"] - d["market_value_m"].median()) / (d["market_value_m"].std() + 1e-9), -2, 2)

    age_term = np.where((d["age"] >= 23) & (d["age"] <= 28), 0.25, 0.0) + np.where(d["age"] >= 30, -0.15, 0.0)

    # biraz noise ekleyelim ki rule'la birebir olmasın
    noise = rng.normal(0, 0.25, size=len(d))

    logit = (
        -1.4
        + 1.2 * contract_term
        + 0.9 * interest_term
        + 0.7 * agent_term
        + 0.5 * salary_term
        - 0.15 * value_term
        + age_term
        + noise
    )

    p = sigmoid(logit)
    y = (rng.random(len(d)) < p).astype(int)

    d["y_transfer_risk"] = y
    d["p_transfer_risk_rule"] = p
    return d


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1))
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    return best_thr, float(prec[best_idx]), float(rec[best_idx]), float(f1[best_idx])


def save_model_and_meta(model, feature_cols, metrics, version="v2"):
    import joblib

    model_path = f"models/transfer_model_{version}.pkl"
    meta_path = f"models/transfer_model_{version}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "version": version,
        "features": feature_cols,
        "metrics": metrics,
        "model": "LogisticRegression + StandardScaler",
        "target": "transfer_risk (leaving/transfer likelihood)"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved model: {model_path}")
    print(f"✅ Saved meta : {meta_path}")


def explain_top(model: Pipeline, feature_cols: list[str], df_scored: pd.DataFrame):
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    coefs = pd.Series(clf.coef_[0], index=feature_cols)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    print("\n--- Explainability (Global) | Coefficients ---")
    print(top.to_string())

    top_row = df_scored.sort_values("transfer_risk", ascending=False).iloc[0]
    X = top_row[feature_cols].to_frame().T
    Xz = scaler.transform(X)
    contrib = Xz[0] * clf.coef_[0]
    contrib_s = pd.Series(contrib, index=feature_cols).sort_values(ascending=False)

    print(f"\n--- Explainability (Local) | Top Risk Player: {top_row['player_id']} ---")
    print(f"Transfer Risk: {top_row['transfer_risk']:.3f}")
    print("\nTop 3 risk drivers (approx):")
    print(contrib_s.head(3).to_string())


def cross_val_metrics(X, y, model, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, pras = [], []
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        if len(np.unique(y.iloc[te])) > 1:
            aucs.append(roc_auc_score(y.iloc[te], proba))
        pras.append(average_precision_score(y.iloc[te], proba))
    return float(np.mean(aucs)) if aucs else None, float(np.mean(pras))


def main():
    ensure_dirs()

    df = pd.read_csv("data/raw/transfer_candidates.csv")
    df = make_transfer_label(df, seed=42)

    feature_cols = [
        "age",
        "market_value_m",
        "contract_months_left",
        "salary_k",
        "interest_level",
        "agent_risk",
    ]

    X = df[feature_cols].copy()
    y = df["y_transfer_risk"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    # ✅ CV metrikleri (asıl güvenilecek kısım)
    cv_roc_auc, cv_pr_auc = cross_val_metrics(X, y, model, n_splits=5)
    print("CV ROC-AUC:", cv_roc_auc)
    print("CV PR-AUC :", cv_pr_auc)

    # final fit (tüm veriye)
    model.fit(X, y)
    proba_all = model.predict_proba(X)[:, 1]

    best_thr, p_best, r_best, f1_best = find_best_threshold(y.values, proba_all)
    print(f"\nBest threshold (F1) on ALL: {best_thr:.4f} | P:{p_best:.3f} R:{r_best:.3f} F1:{f1_best:.3f}")

    # Score everyone
    df_scored = df.copy()
    df_scored["transfer_risk"] = proba_all

    out = df_scored[["player_id", "age", "market_value_m", "contract_months_left", "salary_k", "interest_level", "agent_risk", "transfer_risk"]] \
        .sort_values("transfer_risk", ascending=False)

    print("\nTop 10 transfer risk:")
    print(out.head(10).to_string(index=False))

    out.to_csv("data/processed/transfer_risk_latest.csv", index=False)

    metrics = {
        "cv_roc_auc": cv_roc_auc,
        "cv_pr_auc": cv_pr_auc,
        "best_threshold_f1_all": float(best_thr),
        "best_precision_all": float(p_best),
        "best_recall_all": float(r_best),
        "best_f1_all": float(f1_best),
        "positives": int(y.sum()),
        "rows": int(len(df)),
    }
    save_model_and_meta(model, feature_cols, metrics, version="v2")

    explain_top(model, feature_cols, out)


if __name__ == "__main__":
    main()