"""
Smart Shopping Basket - Market Basket Analysis
==============================================
Data mining project: Apriori vs FP-Growth + Association Rules.

Dataset (Kaggle): "Groceries Dataset" (Groceries_dataset.csv)
  columns: Member_number, Date, itemDescription   (38,765 rows)
  Download: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
  Place Groceries_dataset.csv next to this script (or pass --data path).

Usage:
  pip install pandas numpy matplotlib seaborn mlxtend networkx
  python smart_shopping_basket.py --data Groceries_dataset.csv
  python smart_shopping_basket.py --demo        # synthetic data, for testing
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

OUT = "output"


# ---------------------------------------------------------------- 1. LOAD
def load_groceries(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Member_number", "Date", "itemDescription"])
    df["itemDescription"] = df["itemDescription"].str.strip().str.lower()
    return df


def demo_data(n_baskets=6000, seed=42):
    """Synthetic groceries with planted associations (testing only)."""
    rng = np.random.default_rng(seed)
    items = ["whole milk", "bread", "butter", "eggs", "coffee", "sugar", "pasta",
             "tomato sauce", "cheese", "chips", "soda", "beer", "diapers",
             "yogurt", "apples", "bananas", "rice", "chicken", "onions", "salad"]
    rows = []
    for b in range(n_baskets):
        basket = set(rng.choice(items, size=rng.integers(1, 5), replace=False))
        if rng.random() < 0.25: basket |= {"bread", "butter"}
        if rng.random() < 0.15: basket |= {"coffee", "sugar"}
        if rng.random() < 0.12: basket |= {"pasta", "tomato sauce", "cheese"}
        if rng.random() < 0.08: basket |= {"beer", "chips"}
        for it in basket:
            rows.append((b % 800, pd.Timestamp("2015-01-01") + pd.Timedelta(days=b % 700), it))
    return pd.DataFrame(rows, columns=["Member_number", "Date", "itemDescription"])


# ---------------------------------------------------------- 2. PREPROCESS
def build_transactions(df):
    """One transaction = one member's purchases on one date."""
    baskets = (df.groupby(["Member_number", "Date"])["itemDescription"]
                 .apply(lambda s: sorted(set(s))).reset_index())
    return baskets["itemDescription"].tolist()


def encode(transactions):
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=[str(c) for c in te.columns_])


# ------------------------------------------------------------------- 3. EDA
def eda_plots(df, transactions):
    os.makedirs(OUT, exist_ok=True)
    top = df["itemDescription"].value_counts().head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top.values, y=top.index, color="steelblue")
    plt.title("Top 15 most purchased items")
    plt.xlabel("Purchases")
    plt.tight_layout(); plt.savefig(f"{OUT}/top_items.png", dpi=130); plt.close()

    sizes = pd.Series([len(t) for t in transactions])
    plt.figure(figsize=(7, 4))
    sns.histplot(sizes, bins=range(1, sizes.max() + 2), color="darkorange")
    plt.title("Basket size distribution")
    plt.xlabel("Items per basket")
    plt.tight_layout(); plt.savefig(f"{OUT}/basket_sizes.png", dpi=130); plt.close()


# ------------------------------------------------- 4. MINE (Apriori, FP-G)
def run_miners(basket_df, min_support):
    t0 = time.perf_counter()
    ap = apriori(basket_df, min_support=min_support, use_colnames=True)
    t_ap = time.perf_counter() - t0

    t0 = time.perf_counter()
    fp = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    t_fp = time.perf_counter() - t0
    return ap, fp, t_ap, t_fp


def compare_runtime(basket_df, supports):
    rows = []
    for s in supports:
        ap, fp, t_ap, t_fp = run_miners(basket_df, s)
        rows.append({"min_support": s, "apriori_s": t_ap, "fpgrowth_s": t_fp,
                     "apriori_itemsets": len(ap), "fpgrowth_itemsets": len(fp)})
    res = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    plt.plot(res["min_support"], res["apriori_s"], "o-", label="Apriori")
    plt.plot(res["min_support"], res["fpgrowth_s"], "s-", label="FP-Growth")
    plt.xlabel("Minimum support"); plt.ylabel("Runtime (s)")
    plt.title("Apriori vs FP-Growth runtime"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/runtime_comparison.png", dpi=130); plt.close()
    return res


# ------------------------------------------------------------ 5. RULES
def make_rules(freq, min_conf=0.1, min_lift=1.0):
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def rule_plots(rules):
    if rules.empty:
        return
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                     cmap="viridis", s=40, alpha=.8)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support"); plt.ylabel("Confidence")
    plt.title("Association rules: support vs confidence")
    plt.tight_layout(); plt.savefig(f"{OUT}/rules_scatter.png", dpi=130); plt.close()

    top = rules.head(20)
    G = nx.DiGraph()
    for _, r in top.iterrows():
        G.add_edge(r["antecedents"], r["consequents"], w=r["lift"])
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, k=1.1, seed=1)
    nx.draw_networkx(G, pos, node_color="lightsteelblue", node_size=1800,
                     font_size=8, arrowsize=15,
                     width=[G[u][v]["w"] for u, v in G.edges()])
    plt.title("Top rules by lift (edge width = lift)"); plt.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/rules_network.png", dpi=130); plt.close()


# ---------------------------------------------- 6. SMART BASKET RECOMMENDER
def recommend(rules, basket, top_n=5):
    """Suggest items to add to a customer's current basket."""
    basket = {b.lower() for b in basket}
    scores = {}
    for _, r in rules.iterrows():
        ante = set(r["antecedents"].split(", "))
        cons = set(r["consequents"].split(", ")) - basket
        if ante <= basket and cons:
            for c in cons:
                scores[c] = max(scores.get(c, 0), r["confidence"] * r["lift"])
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]


# ------------------------------------------------------------------ MAIN
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", default="Groceries_dataset.csv")
    ap_.add_argument("--demo", action="store_true")
    ap_.add_argument("--min_support", type=float, default=0.002)
    ap_.add_argument("--min_conf", type=float, default=0.05)
    a = ap_.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = demo_data() if a.demo else load_groceries(a.data)
    if a.demo:
        a.min_support, a.min_conf = 0.02, 0.3
    print(f"Rows: {len(df):,} | Unique items: {df['itemDescription'].nunique()}")

    transactions = build_transactions(df)
    print(f"Transactions (member+date baskets): {len(transactions):,}")
    basket_df = encode(transactions)
    eda_plots(df, transactions)

    apr, fpg, t_ap, t_fp = run_miners(basket_df, a.min_support)
    print(f"\nApriori   : {len(apr)} itemsets in {t_ap:.3f}s")
    print(f"FP-Growth : {len(fpg)} itemsets in {t_fp:.3f}s")
    same = {frozenset(x) for x in apr["itemsets"]} == {frozenset(x) for x in fpg["itemsets"]}
    print(f"Same frequent itemsets from both algorithms: {same}")

    sups = [a.min_support * m for m in (4, 2, 1.5, 1)]
    cmp_df = compare_runtime(basket_df, sups)
    cmp_df.to_csv(f"{OUT}/runtime_comparison.csv", index=False)
    print("\nRuntime comparison:\n", cmp_df.round(4).to_string(index=False))

    rules = make_rules(fpg, a.min_conf)
    rules.to_csv(f"{OUT}/association_rules.csv", index=False)
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    print(f"\n{len(rules)} rules. Top 10 by lift:\n", rules[cols].head(10).round(3).to_string(index=False))
    rule_plots(rules)

    if not rules.empty:
        sample = rules.iloc[0]["antecedents"].split(", ")
        print(f"\nBasket {sample} -> suggestions:", recommend(rules, sample))
    print(f"\nAll outputs saved in ./{OUT}/")


if __name__ == "__main__":
    main()