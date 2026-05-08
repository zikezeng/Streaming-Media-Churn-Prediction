import os
from typing import List

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "processed", "clustered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed", "association_results_no_satisfaction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_cut(series: pd.Series, bins: List[float], labels: List[str]) -> pd.Series:
    """Discretize a numeric series into categorical labels."""
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create categorical groups used for association analysis.

    Satisfaction-related columns are intentionally excluded.
    """
    out = df.copy()

    # Behavioral / structural features only (no satisfaction)
    out["Spend_group"] = safe_cut(
        out["Monthly_Spend"],
        bins=[0, 20, 40, 60, 80, 1000],
        labels=["<=20", "20-40", "40-60", "60-80", "80+"],
    )

    out["Tickets_group"] = safe_cut(
        out["Support_Tickets_Raised"],
        bins=[-1, 0, 2, 5, 1000],
        labels=["0", "1-2", "3-5", "6+"],
    )

    out["Activity_group"] = safe_cut(
        out["Last_Activity"],
        bins=[0, 30, 90, 180, 365, 10000],
        labels=["<=30d", "31-90d", "91-180d", "181-365d", "365d+"],
    )

    out["Subscription_group"] = safe_cut(
        out["Subscription_Length"],
        bins=[0, 12, 24, 36, 120],
        labels=["<=12m", "13-24m", "25-36m", "37m+"],
    )

    out["Discount_group"] = safe_cut(
        out["Discount_Offered"],
        bins=[-1, 0, 10, 20, 1000],
        labels=["0", "1-10", "11-20", "20+"],
    )

    # Ensure categorical columns are string-like for transaction building
    for col in ["Gender", "Region", "Payment_Method", "Cluster", "Churned"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out


def build_transactions(df: pd.DataFrame, include_cluster: bool = False) -> List[List[str]]:
    """Convert a dataframe to transaction format for Apriori."""
    feature_cols = [
        "Gender",
        "Region",
        "Payment_Method",
        "Subscription_group",
        "Spend_group",
        "Tickets_group",
        "Activity_group",
        "Discount_group",
    ]
    if include_cluster and "Cluster" in df.columns:
        feature_cols = ["Cluster"] + feature_cols

    transactions: List[List[str]] = []
    for _, row in df.iterrows():
        items = []
        for col in feature_cols:
            if col in df.columns:
                items.append(f"{col}={row[col]}")
        items.append(f"Churned={row['Churned']}")
        transactions.append(items)

    return transactions


def run_apriori(
    df: pd.DataFrame,
    name: str,
    min_support: float = 0.08,
    lift_threshold: float = 1.0,
) -> pd.DataFrame:
    """Run Apriori and save churn-related rules."""
    print(f"\n{'=' * 88}")
    print(f"Running association analysis (no Satisfaction): {name}")
    print(f"Rows: {len(df)} | Min support: {min_support} | Lift threshold: {lift_threshold}")
    print(f"{'=' * 88}")

    transactions = build_transactions(df, include_cluster=False)

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    tx_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(tx_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    itemsets_path = os.path.join(OUTPUT_DIR, f"{name}_frequent_itemsets.csv")
    frequent_itemsets.sort_values(["support", "length"], ascending=[False, False]).to_csv(
        itemsets_path, index=False
    )

    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)
    except Exception as exc:
        print(f"Could not generate rules: {exc}")
        return pd.DataFrame()

    if rules.empty:
        print("No rules found. Try lowering min_support or lift_threshold.")
        return pd.DataFrame()

    rules_churn = rules[rules["consequents"].apply(lambda x: "Churned=1" in x)].copy()
    if rules_churn.empty:
        print("No churn-related rules found.")
        return pd.DataFrame()

    rules_churn = rules_churn.sort_values(["lift", "confidence", "support"], ascending=[False, False, False])

    rules_path = os.path.join(OUTPUT_DIR, f"{name}_rules_to_churn.csv")
    rules_churn.to_csv(rules_path, index=False)

    print("Top churn-related rules:")
    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    print(rules_churn[display_cols].head(15).to_string(index=False))
    print(f"\nSaved itemsets to: {itemsets_path}")
    print(f"Saved churn rules to: {rules_path}")

    return rules_churn

# =========================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    # No-Satisfaction version: drop satisfaction-related fields if present
    for col in ["Satisfaction_Score", "Satisfaction_group"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # without satisfaction
    df_prepared = prepare_dataframe(df)
    run_apriori(df_prepared, name="all_users_no_satisfaction", min_support=0.05)

    # High-risk cluster only, without satisfaction
    if "Cluster" in df_prepared.columns:
        cluster0 = df_prepared[df_prepared["Cluster"] == "0"].copy()
        if len(cluster0) > 0:
            run_apriori(cluster0, name="cluster0_no_satisfaction", min_support=0.08)
        else:
            print("\nCluster 0 is empty; skipping cluster-only analysis.")
    else:
        print("\nNo Cluster column found; skipping cluster-only analysis.")
