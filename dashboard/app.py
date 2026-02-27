import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scout Panel", layout="wide")

DATA_DECISION = "data/processed/player_decision_board.csv"
DATA_INJ_FEATS = "data/processed/injury_risk_latest_with_features.csv"
DATA_TRANSFER = "data/processed/transfer_risk_latest.csv"
DATA_PLAYERS = "data/raw/players.csv"

MODELS_INJ = "models/injury_model_v2.pkl"
MODELS_TR = "models/transfer_model_v2.pkl"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_DECISION):
        return None, None, None, None

    df = pd.read_csv(DATA_DECISION, parse_dates=["date"])

    players = pd.read_csv(DATA_PLAYERS) if os.path.exists(DATA_PLAYERS) else None
    inj_feats = pd.read_csv(DATA_INJ_FEATS) if os.path.exists(DATA_INJ_FEATS) else None
    tr = pd.read_csv(DATA_TRANSFER) if os.path.exists(DATA_TRANSFER) else None

    if players is not None:
        df = df.merge(players[["player_id", "player_name", "position"]], on="player_id", how="left")

    if inj_feats is not None and "player_id" in inj_feats.columns:
        keep_cols = [c for c in inj_feats.columns if c not in ["injury_risk"]]
        df = df.merge(inj_feats[keep_cols], on=["player_id"], how="left", suffixes=("", "_inj"))

    if tr is not None:
        df = df.merge(tr[["player_id", "transfer_risk"]], on="player_id", how="left", suffixes=("", "_tr"))
        if "transfer_risk_tr" in df.columns:
            df["transfer_risk"] = df["transfer_risk"].fillna(df["transfer_risk_tr"])
            df.drop(columns=["transfer_risk_tr"], inplace=True, errors="ignore")

    return df, players, inj_feats, tr


def style_action(df: pd.DataFrame):
    def color_row(row):
        a = row.get("action", "")
        if a == "AL":
            return ["background-color: #E8F5E9"] * len(row)
        if a == "İZLE":
            return ["background-color: #FFFDE7"] * len(row)
        if a == "ALMA":
            return ["background-color: #FFEBEE"] * len(row)
        return [""] * len(row)

    return df.style.apply(color_row, axis=1)


def radar_plot(player_row: pd.Series):
    metrics = {
        "Injury Risk": float(player_row.get("injury_risk", np.nan)),
        "Transfer Risk": float(player_row.get("transfer_risk", np.nan)),
        "Value (norm)": float(player_row.get("market_value_m", np.nan)),
        "Contract (norm)": float(player_row.get("contract_months_left", np.nan)),
        "Interest (norm)": float(player_row.get("interest_level", np.nan)),
    }

    def norm(x, lo, hi):
        if np.isnan(x):
            return 0.0
        return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))

    values = [
        metrics["Injury Risk"] if not np.isnan(metrics["Injury Risk"]) else 0.0,
        metrics["Transfer Risk"] if not np.isnan(metrics["Transfer Risk"]) else 0.0,
        norm(metrics["Value (norm)"], 0.0, 15.0),
        1.0 - norm(metrics["Contract (norm)"], 0.0, 48.0),
        norm(metrics["Interest (norm)"], 1.0, 5.0),
    ]
    labels = ["Injury", "Transfer", "Value", "Short Contract", "Interest"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Player Radar | {player_row.get('player_id')}", pad=15)
    return fig


def coeff_plot(model_path: str, title: str, top_n: int = 10):
    try:
        model = joblib.load(model_path)
        clf = model.named_steps["clf"]
        coefs = clf.coef_[0]

        meta_path = model_path.replace(".pkl", "_meta.json")
        feats = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feats = meta.get("features", None)

        if feats is None or len(feats) != len(coefs):
            feats = [f"f{i}" for i in range(len(coefs))]

        s = pd.Series(coefs, index=feats).sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
        s = s.sort_values()

        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.barh(s.index, s.values)
        ax.set_title(title)
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        return fig, None
    except Exception as e:
        return None, str(e)


def get_feature_cols_from_meta(model_path: str, df_feats: pd.DataFrame):
    meta_path = model_path.replace(".pkl", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feats = meta.get("features", None)
        if feats and all(c in df_feats.columns for c in feats):
            return feats
    # fallback: df_feats'te işe yarar kolonları al (injury_risk/y gibi hariç)
    bad = {"player_id", "date", "injury_risk", "y_injury_next14"}
    return [c for c in df_feats.columns if c not in bad and pd.api.types.is_numeric_dtype(df_feats[c])]


st.title("⚽ Scout Panel — Injury + Transfer + Decision Board")

df, players, inj_feats, tr = load_data()

if df is None:
    st.error(
        "Gerekli dosyalar yok. Önce şu scriptleri çalıştır:\n\n"
        "py src/data/generate_synthetic.py\n"
        "py src/injury_model.py\n"
        "py src/transfer_model.py\n"
        "py src/decision_board.py"
    )
    st.stop()

# ===== KPI CARDS =====
high_injury_thr = st.sidebar.slider("High injury threshold", 0.0, 1.0, 0.65, 0.01)

k1, k2, k3 = st.columns(3)
high_injury_count = int((df["injury_risk"] >= high_injury_thr).sum())
watchlist_count = int((df["action"] == "İZLE").sum())
buy_count = int((df["action"] == "AL").sum())

k1.metric("🔴 High Injury Risk", high_injury_count)
k2.metric("🟡 Watchlist", watchlist_count)
k3.metric("🟢 Buy Candidates", buy_count)

# ===== Position Distribution =====
st.subheader("Risk Distribution by Position")
pos_metric = st.selectbox("Metric", ["injury_risk", "transfer_risk", "final_score"], index=0)

if "position" in df.columns:
    pos_agg = (
        df.groupby("position")[pos_metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    st.bar_chart(pos_agg.set_index("position"))
else:
    st.info("Position bilgisi yok (players.csv merge edilmemiş olabilir).")

# ===== Sidebar filters =====
st.sidebar.header("Filters")

actions = sorted(df["action"].dropna().unique().tolist()) if "action" in df.columns else []
selected_actions = st.sidebar.multiselect("Action", actions, default=actions)

pos_list = sorted(df["position"].dropna().unique().tolist()) if "position" in df.columns else []
selected_pos = st.sidebar.multiselect("Position", pos_list, default=pos_list) if len(pos_list) else []

if "age" in df.columns:
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
else:
    age_min, age_max = 18, 35
age_range = st.sidebar.slider("Age", min_value=age_min, max_value=age_max, value=(age_min, age_max))

if "contract_months_left" in df.columns:
    contract_min = int(df["contract_months_left"].min())
    contract_max = int(df["contract_months_left"].max())
else:
    contract_min, contract_max = 1, 48
contract_range = st.sidebar.slider(
    "Contract months left",
    min_value=contract_min,
    max_value=contract_max,
    value=(contract_min, contract_max),
)

top_n = st.sidebar.slider("Top N", min_value=5, max_value=20, value=10)

# Apply filters
f = df.copy()
if selected_actions:
    f = f[f["action"].isin(selected_actions)]
if selected_pos and "position" in f.columns:
    f = f[f["position"].isin(selected_pos)]
if "age" in f.columns:
    f = f[(f["age"] >= age_range[0]) & (f["age"] <= age_range[1])]
if "contract_months_left" in f.columns:
    f = f[(f["contract_months_left"] >= contract_range[0]) & (f["contract_months_left"] <= contract_range[1])]

# ===== Layout =====
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    st.subheader("Decision Board")
    show_cols = [
        c for c in [
            "player_id", "player_name", "position",
            "injury_risk", "transfer_risk", "final_score", "action",
            "market_value_m", "contract_months_left", "salary_k", "interest_level", "agent_risk"
        ] if c in f.columns
    ]
    st.dataframe(
        style_action(f[show_cols].sort_values("final_score", ascending=False)),
        use_container_width=True,
        height=520,
    )

with c2:
    st.subheader("Top Injury Risk")
    tmp = f.sort_values("injury_risk", ascending=False).head(top_n)
    st.bar_chart(tmp.set_index("player_id")["injury_risk"])

    st.subheader("Top Transfer Risk")
    tmp2 = f.sort_values("transfer_risk", ascending=False).head(top_n)
    st.bar_chart(tmp2.set_index("player_id")["transfer_risk"])

with c3:
    st.subheader("Player Radar")
    pid_list = f["player_id"].tolist()
    pick = st.selectbox("Select player", pid_list, index=0 if pid_list else 0)
    if pick and len(f):
        row = f[f["player_id"] == pick].iloc[0]
        fig = radar_plot(row)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Explainability")

    # ===== SHAP Global =====
    st.markdown("### SHAP Global Feature Importance (Injury Model)")
    try:
        if os.path.exists(MODELS_INJ) and os.path.exists(DATA_INJ_FEATS):
            model = joblib.load(MODELS_INJ)
            df_feats = pd.read_csv(DATA_INJ_FEATS)

            feature_cols = get_feature_cols_from_meta(MODELS_INJ, df_feats)
            X = df_feats[feature_cols].fillna(0.0)

            X_scaled = model.named_steps["scaler"].transform(X)
            clf = model.named_steps["clf"]

            explainer = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer(X_scaled)

            shap.plots.bar(shap_values, show=False)
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("SHAP için model veya feature dosyası yok.")
    except Exception as e:
        st.info(f"SHAP yüklenemedi: {e}")

    # ===== Coeff plots =====
    st.caption("Model coefficients (top 10).")
    fig1, err1 = coeff_plot(MODELS_INJ, "Injury Model Coefficients", top_n=10)
    if fig1 is not None:
        st.pyplot(fig1, clear_figure=True)
    else:
        st.info(f"Injury model yüklenemedi: {err1}")

    fig2, err2 = coeff_plot(MODELS_TR, "Transfer Model Coefficients", top_n=10)
    if fig2 is not None:
        st.pyplot(fig2, clear_figure=True)
    else:
        st.info(f"Transfer model yüklenemedi: {err2}")

st.caption("Data sources: decision_board + injury/transfer outputs (synthetic).")