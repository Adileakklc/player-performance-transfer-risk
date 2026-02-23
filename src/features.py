import pandas as pd
import numpy as np

def safe_div(a, b):
    return a / (b + 1e-9)

def build_features(match_df: pd.DataFrame, train_df: pd.DataFrame, inj_df: pd.DataFrame) -> pd.DataFrame:
    # --- training rolling + ACWR ---
    tr = train_df.sort_values(["player_id", "date"]).copy()
    tr = tr.groupby(["player_id", "date"], as_index=False)["training_load"].sum()
    tr["load_7"] = tr.groupby("player_id")["training_load"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    tr["load_14"] = tr.groupby("player_id")["training_load"].transform(lambda s: s.rolling(14, min_periods=5).mean())
    tr["load_28"] = tr.groupby("player_id")["training_load"].transform(lambda s: s.rolling(28, min_periods=10).mean())
    tr["acwr_7_28"] = safe_div(tr["load_7"], tr["load_28"])

    # --- last 3 match performance + trend ---
    m = match_df.sort_values(["player_id", "match_date"]).copy()
    m["perf_idx"] = (0.4*m["xg"].fillna(0) + 0.3*m["xa"].fillna(0) + 0.2*m["pass_accuracy"].fillna(0) + 0.1*m["distance_km"].fillna(0))
    m["perf_last3"] = m.groupby("player_id")["perf_idx"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    m["perf_prev3"] = m.groupby("player_id")["perf_idx"].transform(lambda s: s.shift(3).rolling(3, min_periods=1).mean())
    m["perf_trend3"] = m["perf_last3"] - m["perf_prev3"].fillna(m["perf_last3"])
    m = m.rename(columns={"match_date": "date"})[["player_id", "date", "minutes", "perf_last3", "perf_trend3"]]

    # --- timeline union ---
    timeline = pd.concat([tr[["player_id","date"]], m[["player_id","date"]]], ignore_index=True).drop_duplicates()
    feats = timeline.merge(tr, on=["player_id","date"], how="left").merge(m, on=["player_id","date"], how="left")
    feats = feats.sort_values(["player_id","date"])
    feats["training_load"] = feats["training_load"].fillna(0)
    for c in ["load_7","load_14","load_28","acwr_7_28"]:
        feats[c] = feats.groupby("player_id")[c].ffill()
    for c in ["minutes","perf_last3","perf_trend3"]:
        feats[c] = feats.groupby("player_id")[c].ffill().fillna(0)

    # --- days since last injury ---
    inj = inj_df.sort_values(["player_id","injury_date"])[["player_id","injury_date"]]
    out = []
    for pid, g in feats.groupby("player_id"):
        g = g.sort_values("date")
        inj_p = inj[inj["player_id"] == pid][["injury_date"]].sort_values("injury_date")
        if inj_p.empty:
            g["days_since_last_injury"] = 9999
        else:
            g2 = pd.merge_asof(g, inj_p.rename(columns={"injury_date":"last_injury_date"}), left_on="date", right_on="last_injury_date", direction="backward")
            g2["days_since_last_injury"] = (g2["date"] - g2["last_injury_date"]).dt.days.fillna(9999).astype(int)
            g = g2.drop(columns=["last_injury_date"])
        out.append(g)

    feats = pd.concat(out, ignore_index=True)
    return feats