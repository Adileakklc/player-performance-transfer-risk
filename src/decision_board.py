import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)


def action_rule(injury_risk: float, transfer_risk: float) -> str:
    """
    Basit ama savunulabilir karar kuralı (TR kulübü):
    - ALMA: injury çok yüksek
    - AL: injury düşük + transfer uygun (pazarda alınabilir)
    - İZLE: arada kalanlar
    """
    # Thresholds: senin output dağılımına göre ayarlı (0-1 arası)
    if injury_risk >= 0.70:
        return "ALMA"
    if injury_risk <= 0.45 and transfer_risk <= 0.70:
        return "AL"
    # düşük riskli, düşük transfer riski: izleme (hemen düşmeyebilir)
    if injury_risk <= 0.35 and transfer_risk < 0.70:
        return "İZLE"
    
    return "İZLE"


def main():
    ensure_dirs()

    inj = pd.read_csv("data/processed/injury_risk_latest.csv", parse_dates=["date"])
    tr  = pd.read_csv("data/processed/transfer_risk_latest.csv")

    # merge
    df = inj.merge(tr[["player_id", "transfer_risk", "age", "market_value_m", "contract_months_left", "salary_k", "interest_level", "agent_risk"]],
                   on="player_id", how="left")

    # missing transfer risk? (shouldn't happen)
    df["transfer_risk"] = df["transfer_risk"].fillna(df["transfer_risk"].median())

    # final score (weighted): injury daha ağır
    df["final_score"] = 0.65 * df["injury_risk"] + 0.35 * df["transfer_risk"]

    # decision
    df["action"] = df.apply(lambda r: action_rule(r["injury_risk"], r["transfer_risk"]), axis=1)

    # shortlist: AL + İZLE ilk sıralar
    out = df.sort_values(["action", "final_score"], ascending=[True, False])

    # Kaydet
    out.to_csv("data/processed/player_decision_board.csv", index=False)

    print("\n✅ Saved: data/processed/player_decision_board.csv")
    print("\nTop 12 Decision Board:")
    cols = ["player_id", "date", "injury_risk", "transfer_risk", "final_score", "action",
            "age", "market_value_m", "contract_months_left", "salary_k", "interest_level", "agent_risk"]
    print(out[cols].head(12).to_markdown(index=False))


if __name__ == "__main__":
    main()