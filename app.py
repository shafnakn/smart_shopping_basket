"""
Smart Shopping Basket - Streamlit frontend
==========================================
Run:  streamlit run app.py
Needs smart_shopping_basket.py in the same folder.
"""
import os
import time

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, fpgrowth

from smart_shopping_basket import (
    build_transactions,
    demo_data,
    encode,
    load_groceries,
    make_rules,
    recommend,
)

st.set_page_config(page_title="Smart Shopping Basket", page_icon="🛒", layout="wide")


# ------------------------------------------------------------ data helpers
@st.cache_data(show_spinner="Loading data...")
def get_data(source, uploaded_bytes, path):
    if source == "Demo (synthetic)":
        return demo_data()
    if source == "Upload CSV" and uploaded_bytes is not None:
        import io
        return load_groceries(io.BytesIO(uploaded_bytes))
    return load_groceries(path)


@st.cache_data(show_spinner="Preparing baskets...")
def get_baskets(df):
    transactions = build_transactions(df)
    return transactions, encode(transactions)


@st.cache_data(show_spinner="Mining frequent itemsets...")
def mine(basket_df, algo, min_support):
    t0 = time.perf_counter()
    fn = apriori if algo == "Apriori" else fpgrowth
    freq = fn(basket_df, min_support=min_support, use_colnames=True)
    return freq, time.perf_counter() - t0


@st.cache_data(show_spinner="Comparing algorithms...")
def compare(basket_df, supports):
    rows = []
    for s in supports:
        t0 = time.perf_counter()
        a = apriori(basket_df, min_support=s, use_colnames=True)
        t_a = time.perf_counter() - t0
        t0 = time.perf_counter()
        f = fpgrowth(basket_df, min_support=s, use_colnames=True)
        t_f = time.perf_counter() - t0
        rows.append({"min_support": s, "Apriori (s)": t_a, "FP-Growth (s)": t_f,
                     "itemsets": len(f)})
    return pd.DataFrame(rows)


def itemset_text(s):
    return ", ".join(sorted(s))


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🛒 Smart Basket")
source = st.sidebar.radio("Data source", ["Kaggle CSV file", "Upload CSV", "Demo (synthetic)"])
uploaded, path = None, "Groceries_dataset.csv"
if source == "Upload CSV":
    f = st.sidebar.file_uploader("Groceries_dataset.csv", type="csv")
    uploaded = f.getvalue() if f else None
elif source == "Kaggle CSV file":
    path = st.sidebar.text_input("CSV path", "Groceries_dataset.csv")

demo = source == "Demo (synthetic)"
st.sidebar.markdown("---")
algo = st.sidebar.selectbox("Algorithm", ["FP-Growth", "Apriori"])
min_support = st.sidebar.slider("Minimum support", 0.001, 0.10, 0.02 if demo else 0.003,
                                step=0.001, format="%.3f")
min_conf = st.sidebar.slider("Minimum confidence", 0.05, 1.0, 0.3 if demo else 0.05, step=0.05)
min_lift = st.sidebar.slider("Minimum lift", 1.0, 10.0, 1.0, step=0.1)

# -------------------------------------------------------------- load / run
st.title("Smart Shopping Basket")
st.caption("Market basket analysis with Apriori, FP-Growth and association rules")

if source == "Upload CSV" and uploaded is None:
    st.info("Upload the Kaggle Groceries CSV in the sidebar to begin.")
    st.stop()
if source == "Kaggle CSV file" and not os.path.exists(path):
    st.warning(f"File not found: `{path}`. Download the Groceries Dataset from Kaggle and place it "
               "next to app.py, upload it, or switch to the demo data.")
    st.stop()

try:
    df = get_data(source, uploaded, path)
    transactions, basket_df = get_baskets(df)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the data: {e}")
    st.stop()

freq, elapsed = mine(basket_df, algo, min_support)
if freq.empty:
    st.warning("No frequent itemsets at this support. Lower the minimum support.")
    st.stop()

rules = pd.DataFrame()
if (freq["itemsets"].apply(len) >= 2).any():
    rules = make_rules(freq, min_conf, min_lift)

tab_over, tab_sets, tab_rules, tab_cmp, tab_rec = st.tabs(
    ["Overview", "Frequent itemsets", "Association rules", "Apriori vs FP-Growth", "Smart basket"])

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Purchase rows", f"{len(df):,}")
    c2.metric("Baskets", f"{len(transactions):,}")
    c3.metric("Unique items", f"{df['itemDescription'].nunique():,}")
    c4.metric("Avg basket size", f"{sum(map(len, transactions)) / len(transactions):.2f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Top 15 items")
        st.bar_chart(df["itemDescription"].value_counts().head(15), horizontal=True)
    with right:
        st.subheader("Basket size distribution")
        st.bar_chart(pd.Series([len(t) for t in transactions]).value_counts().sort_index())

# ------------------------------------------------------------- itemsets
with tab_sets:
    st.subheader(f"{len(freq):,} frequent itemsets ({algo}, {elapsed:.3f}s)")
    show = freq.copy()
    show["size"] = show["itemsets"].apply(len)
    show["itemsets"] = show["itemsets"].apply(itemset_text)
    size_filter = st.multiselect("Itemset size", sorted(show["size"].unique()),
                                 default=sorted(show["size"].unique()))
    show = show[show["size"].isin(size_filter)].sort_values("support", ascending=False)
    st.dataframe(show.round(4), width="stretch", hide_index=True)

# ---------------------------------------------------------------- rules
with tab_rules:
    if rules.empty:
        st.info("No rules at these thresholds. Lower support, confidence or lift.")
    else:
        st.subheader(f"{len(rules):,} association rules")
        search = st.text_input("Filter by item name")
        view = rules
        if search:
            m = (view["antecedents"].str.contains(search, case=False)
                 | view["consequents"].str.contains(search, case=False))
            view = view[m]
        cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
        st.dataframe(view[cols].round(4), width="stretch", hide_index=True)
        st.download_button("Download rules (CSV)", view[cols].to_csv(index=False),
                           "association_rules.csv", "text/csv")
        st.subheader("Support vs confidence (colour = lift)")
        st.scatter_chart(view, x="support", y="confidence", color="lift")

# ----------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("Runtime comparison")
    st.caption("On small catalogues Apriori can be as fast or faster. FP-Growth tends to win on "
               "large, dense data and low support.")
    sups = tuple(round(min_support * m, 4) for m in (4, 2, 1.5, 1))
    res = compare(basket_df, sups)
    st.line_chart(res.set_index("min_support")[["Apriori (s)", "FP-Growth (s)"]])
    st.dataframe(res.round(4), width="stretch", hide_index=True)

# -------------------------------------------------------- smart basket
with tab_rec:
    st.subheader("Build a basket, get suggestions")
    items = sorted(basket_df.columns)
    chosen = st.multiselect("Items in the customer's basket", items)
    if not chosen:
        st.info("Pick one or more items.")
    elif rules.empty:
        st.info("No rules available. Lower the thresholds in the sidebar.")
    else:
        recs = recommend(rules, chosen, top_n=8)
        if recs:
            out = pd.DataFrame(recs, columns=["Suggested item", "Score (confidence × lift)"])
            st.dataframe(out.round(3), width="stretch", hide_index=True)
        else:
            st.info("No rule matches this basket. Try other items or lower the thresholds.")