import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "processed", "clustered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "processed", "association_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================
df = pd.read_csv(INPUT_PATH)

# =========================
def bin_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()


    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 120],
        labels=["<=30", "31-45", "46-60", "60+"],
        include_lowest=True
    )

    df["Subscription_group"] = pd.cut(
        df["Subscription_Length"],
        bins=[0, 12, 24, 36, 120],
        labels=["<=12m", "13-24m", "25-36m", "37m+"],
        include_lowest=True
    )

    df["Tickets_group"] = pd.cut(
        df["Support_Tickets_Raised"],
        bins=[-1, 0, 2, 5, 1000],
        labels=["0", "1-2", "3-5", "6+"],
        include_lowest=True
    )

    df["Satisfaction_group"] = pd.cut(
        df["Satisfaction_Score"],
        bins=[0, 2, 4, 6, 8, 10],
        labels=["1-2 low", "3-4", "5-6", "7-8", "9-10 high"],
        include_lowest=True
    )

    df["Discount_group"] = pd.cut(
        df["Discount_Offered"],
        bins=[-1, 0, 10, 20, 1000],
        labels=["0", "1-10", "11-20", "20+"],
        include_lowest=True
    )

    df["Activity_group"] = pd.cut(
        df["Last_Activity"],
        bins=[0, 30, 90, 180, 365, 10000],
        labels=["<=30d", "31-90d", "91-180d", "181-365d", "365d+"],
        include_lowest=True
    )

    df["Spend_group"] = pd.cut(
        df["Monthly_Spend"],
        bins=[0, 20, 40, 60, 80, 1000],
        labels=["<=20", "20-40", "40-60", "60-80", "80+"],
        include_lowest=True
    )

    # 分类变量转字符串
    for col in ["Gender", "Region", "Payment_Method"]:
        df[col] = df[col].astype(str)

    return df

# =========================
def build_transactions(dataframe: pd.DataFrame) -> list[list[str]]:
    feature_cols = [
        "Gender", "Region", "Payment_Method",
        "Age_group", "Subscription_group", "Tickets_group",
        "Satisfaction_group", "Discount_group", "Activity_group", "Spend_group",
        "Churned"
    ]

    transactions = []
    for _, row in dataframe.iterrows():
        items = []
        for col in feature_cols:
            items.append(f"{col}={row[col]}")
        transactions.append(items)

    return transactions

# =========================
def run_apriori(transactions, name, min_support=0.05, lift_threshold=1.0):
    print(f"\n{'='*80}")
    print(f"Running association analysis for: {name}")
    print(f"Transactions: {len(transactions)}")
    print(f"Min support: {min_support}")
    print(f"{'='*80}")

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    tx_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(tx_df, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return

   
    itemsets_path = os.path.join(OUTPUT_DIR, f"{name}_frequent_itemsets.csv")
    frequent_itemsets.sort_values(
        ["support", "length"], ascending=[False, False]
    ).to_csv(itemsets_path, index=False)

    print("\nTop frequent itemsets:")
    print(
        frequent_itemsets
        .sort_values(["support", "length"], ascending=[False, False])
        .head(15)
        .to_string(index=False)
    )


    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_threshold)
    except Exception as e:
        print(f"Could not generate rules: {e}")
        return

    if rules.empty:
        print("No rules found. Try lowering min_support or lift_threshold.")
        return

    rules_churn = rules[
        rules["consequents"].apply(lambda x: "Churned=1" in x)
    ].copy()

    if rules_churn.empty:
        print("No churn rules found. Try lowering min_support.")
        return

    rules_churn = rules_churn.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False]
    )

    rules_path = os.path.join(OUTPUT_DIR, f"{name}_rules_to_churn.csv")
    rules_churn.to_csv(rules_path, index=False)

    print("\nTop rules leading to churn:")
    print(
        rules_churn[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].head(15).to_string(index=False)
    )

    print(f"\nSaved itemsets to: {itemsets_path}")
    print(f"Saved churn rules to: {rules_path}")

if __name__ == "__main__":
    df_all = bin_features(df)
    transactions_all = build_transactions(df_all)
    run_apriori(
        transactions_all,
        name="all_users",
        min_support=0.05,
        lift_threshold=1.0
    )

    if "Cluster" in df.columns:
        df_cluster0 = df[df["Cluster"] == 0].copy()

        if len(df_cluster0) > 0:
            df_cluster0 = bin_features(df_cluster0)
            transactions_cluster0 = build_transactions(df_cluster0)
            run_apriori(
                transactions_cluster0,
                name="cluster0",
                min_support=0.08,
                lift_threshold=1.0
            )
        else:
            print("\nCluster 0 has no rows, skipping Cluster 0 analysis.")
    else:
        print("\nNo Cluster column found, skipping Cluster 0 analysis.")