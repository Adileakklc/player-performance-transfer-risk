# src/data/generate_synthetic.py
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

def make_players(n=20, team="TR_Club_A", season="2025-2026"):
    positions = ["GK","CB","FB","DM","CM","AM","W","ST"]
    pos_probs = [0.10, 0.18, 0.14, 0.12, 0.16, 0.10, 0.10, 0.10]
    players = []
    for i in range(1, n+1):
        pos = RNG.choice(positions, p=pos_probs)
        players.append({
            "season": season,
            "team": team,
            "player_id": f"P{i:02d}",
            "player_name": f"Player_{i:02d}",
            "position": pos,
            "age": int(RNG.integers(18, 35)),
        })
    return pd.DataFrame(players)

def make_matches(n_matches=28, season="2025-2026"):
    start = pd.Timestamp("2025-08-10")
    dates = [start + pd.Timedelta(days=int(7*k + RNG.integers(-1,2))) for k in range(n_matches)]
    return pd.DataFrame({
        "season": season,
        "match_id": [f"M{k+1:02d}" for k in range(n_matches)],
        "match_date": dates
    }).sort_values("match_date").reset_index(drop=True)

def gen_match_data(players, matches):
    rows = []
    for _, m in matches.iterrows():
        for _, p in players.iterrows():
            play_prob = 0.85 if p["position"] != "GK" else 0.70
            played = RNG.random() < play_prob
            minutes = int(RNG.integers(10, 91)) if played else 0

            pos = p["position"]
            base_dist = 9.5 if pos in ["CM","DM"] else (8.0 if pos in ["FB","W","AM"] else 6.5)
            distance_km = float(max(0, RNG.normal(base_dist, 1.2) * (minutes/90))) if minutes>0 else 0.0

            sprints = int(max(0, RNG.normal(18 if pos in ["W","FB"] else 10, 5) * (minutes/90))) if minutes>0 else 0
            hir = int(max(0, RNG.normal(35 if pos in ["CM","DM","FB"] else 25, 8) * (minutes/90))) if minutes>0 else 0

            xg = float(max(0, RNG.normal(0.25 if pos=="ST" else 0.10, 0.12) * (minutes/90))) if minutes>0 else 0.0
            xa = float(max(0, RNG.normal(0.15 if pos in ["AM","W"] else 0.07, 0.10) * (minutes/90))) if minutes>0 else 0.0

            passes_attempted = int(max(0, RNG.normal(55 if pos in ["CB","DM","CM"] else 35, 15) * (minutes/90))) if minutes>0 else 0
            pass_accuracy = float(np.clip(RNG.normal(0.86 if pos in ["CB","DM","CM"] else 0.80, 0.05), 0.60, 0.95)) if minutes>0 else 0.0

            duels = int(max(0, RNG.normal(10 if pos in ["CB","DM","ST"] else 7, 3) * (minutes/90))) if minutes>0 else 0
            duels_won = int(np.clip(RNG.normal(duels*0.52, 2), 0, duels)) if minutes>0 else 0

            yellow = int((RNG.random() < 0.08) and minutes>0)
            red = int((RNG.random() < 0.01) and minutes>0)

            rows.append({
                "season": p["season"],
                "team": p["team"],
                "match_id": m["match_id"],
                "match_date": m["match_date"],
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "position": pos,
                "minutes": minutes,
                "distance_km": round(distance_km, 2),
                "sprints": sprints,
                "high_intensity_runs": hir,
                "xg": round(xg, 3),
                "xa": round(xa, 3),
                "passes_attempted": passes_attempted,
                "pass_accuracy": round(pass_accuracy, 3),
                "duels": duels,
                "duels_won": duels_won,
                "yellow": yellow,
                "red": red
            })
    return pd.DataFrame(rows)

def gen_training_load(players, matches):
    start = matches["match_date"].min() - pd.Timedelta(days=7)
    end = matches["match_date"].max() + pd.Timedelta(days=7)
    days = pd.date_range(start, end, freq="D")
    rows = []
    for d in days:
        is_training_day = d.weekday() in [0,1,3,4]  # Pzt-Salı-Perş-Cuma
        for _, p in players.iterrows():
            if not is_training_day:
                continue
            duration = int(np.clip(RNG.normal(75, 20), 30, 120))
            rpe = float(np.clip(RNG.normal(6.0, 1.5), 1, 10))
            rows.append({
                "date": d,
                "player_id": p["player_id"],
                "session_duration_min": duration,
                "rpe": round(rpe, 1),
                "training_load": round(duration * rpe, 1),
            })
    return pd.DataFrame(rows)

# ---------- NEW: workload-driven injury generation ----------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _prepare_daily_load(training: pd.DataFrame) -> pd.DataFrame:
    # daily load per player (ensure daily granularity)
    df = training.copy().sort_values(["player_id", "date"])
    df = df.groupby(["player_id", "date"], as_index=False)["training_load"].sum()

    # rolling means
    df["load_7"] = df.groupby("player_id")["training_load"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["load_28"] = df.groupby("player_id")["training_load"].transform(lambda s: s.rolling(28, min_periods=10).mean())
    df["acwr_7_28"] = df["load_7"] / (df["load_28"] + 1e-9)

    # weekly trend proxy: last 7 avg - prev 7 avg
    df["load_prev7"] = df.groupby("player_id")["training_load"].transform(lambda s: s.shift(7).rolling(7, min_periods=3).mean())
    df["delta_7"] = (df["load_7"] - df["load_prev7"]).fillna(0)

    return df

def _match_congestion(matches: pd.DataFrame) -> pd.DataFrame:
    # Count matches in last 7 days at each match date
    m = matches[["match_date"]].copy().sort_values("match_date")
    out = []
    for d in m["match_date"]:
        cnt7 = ((m["match_date"] > d - pd.Timedelta(days=7)) & (m["match_date"] <= d)).sum()
        cnt3 = ((m["match_date"] > d - pd.Timedelta(days=3)) & (m["match_date"] <= d)).sum()
        out.append({"match_date": d, "matches_last7": int(cnt7), "matches_last3": int(cnt3)})
    return pd.DataFrame(out)

def gen_injuries(players, matches, training):
    """
    Injury generation driven by:
    - ACWR high -> risk ↑
    - delta_7 positive -> risk ↑
    - match congestion (3 days / 7 days) -> risk ↑
    - previous injury -> reinjury risk ↑
    """
    injury_types = ["hamstring","ankle","groin","knee","calf"]

    daily = _prepare_daily_load(training)
    cong = _match_congestion(matches)

    start = matches["match_date"].min()
    end = matches["match_date"].max()

    # Use match days as "risk evaluation" points (simpler & realistic: injuries around matches)
    match_days = matches["match_date"].sort_values().unique()

    rows = []
    for _, p in players.iterrows():
        pid = p["player_id"]
        pos = p["position"]

        # position baseline injury tendency (very rough)
        pos_bias = {
            "GK": -0.3, "CB": -0.1, "FB": 0.05, "DM": 0.03,
            "CM": 0.04, "AM": 0.05, "W": 0.08, "ST": 0.07
        }.get(pos, 0.0)

        had_injury_before = False

        # evaluate risk on each match day
        for d in match_days:
            # get latest daily load row up to date d
            dload = daily[(daily["player_id"] == pid) & (daily["date"] <= d)].sort_values("date").tail(1)
            if dload.empty:
                continue

            acwr = float(dload["acwr_7_28"].iloc[0]) if not pd.isna(dload["acwr_7_28"].iloc[0]) else 1.0
            delta7 = float(dload["delta_7"].iloc[0])

            # match congestion factors
            c = cong[cong["match_date"] == d].iloc[0]
            cong7 = c["matches_last7"]
            cong3 = c["matches_last3"]

            # normalize terms to reasonable scale
            acwr_term = np.clip(acwr - 1.0, -0.5, 1.5)          # >1 means acute>chronic
            delta_term = np.clip(delta7 / 300.0, -1.0, 2.0)     # scale load change
            cong_term = 0.25 * max(0, cong7 - 1) + 0.35 * max(0, cong3 - 1)

            reinj_term = 0.35 if had_injury_before else 0.0

            # logit risk
            logit = -2.4 + 1.6 * acwr_term + 1.0 * delta_term + 0.8 * cong_term + 1.2 * reinj_term + pos_bias
            p_inj = _sigmoid(logit)

            # stochastic injury occurrence
            if RNG.random() < p_inj:
                itype = RNG.choice(injury_types)
                base_days = RNG.normal(14, 9)
                # more severe when acwr high
                days_out = int(np.clip(base_days + 10 * max(0, acwr_term), 3, 60))

                rows.append({
                    "player_id": pid,
                    "injury_date": pd.Timestamp(d),
                    "injury_type": itype,
                    "days_out": days_out,
                    "is_reinjury": int(had_injury_before)
                })

                had_injury_before = True

                # simple “cooldown”: after an injury, skip next few match days
                # (simulate absence / rehabilitation)
                # We'll break out if too many injuries
                if len([r for r in rows if r["player_id"] == pid]) >= 5:
                    break

    return pd.DataFrame(rows).sort_values(["player_id", "injury_date"]).reset_index(drop=True)

def gen_transfer(players):
    rows = []
    for _, p in players.iterrows():
        market_value = float(np.clip(RNG.normal(2.5, 2.0), 0.2, 15.0))
        contract = int(np.clip(RNG.normal(18, 10), 1, 48))
        salary = int(np.clip(RNG.normal(180, 120), 30, 800))
        interest = int(RNG.integers(1, 6))
        agent_risk = int(RNG.random() < 0.18)
        rows.append({
            "player_id": p["player_id"],
            "age": int(p["age"]),
            "market_value_m": round(market_value, 2),
            "contract_months_left": contract,
            "salary_k": salary,
            "interest_level": interest,
            "agent_risk": agent_risk
        })
    return pd.DataFrame(rows)

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    players = make_players(n=20)
    matches = make_matches(n_matches=28)
    match_data = gen_match_data(players, matches)
    training = gen_training_load(players, matches)

    # ✅ UPDATED: injuries now depend on workload + congestion
    injuries = gen_injuries(players, matches, training)

    transfer = gen_transfer(players)

    players.to_csv(out_dir/"players.csv", index=False)
    matches.to_csv(out_dir/"matches.csv", index=False)
    match_data.to_csv(out_dir/"match_data.csv", index=False)
    training.to_csv(out_dir/"training_load.csv", index=False)
    injuries.to_csv(out_dir/"injury_history.csv", index=False)
    transfer.to_csv(out_dir/"transfer_candidates.csv", index=False)

    print("✅ Generated:", out_dir)

if __name__ == "__main__":
    main()
