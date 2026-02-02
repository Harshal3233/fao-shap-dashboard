import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="FAOSTAT XAI Case Study Dashboard", layout="wide")

# =========================================================
# Helpers
# =========================================================
def safe_to_dense(x):
    """Convert sparse matrix to dense only when needed."""
    try:
        return x.toarray()
    except Exception:
        return np.asarray(x)

def clamp_int(val, lo, hi):
    return int(max(lo, min(hi, val)))

# =========================================================
# Data
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    def load_country(path: str, country: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["Country"] = country
        return df

    df = pd.concat(
        [
            load_country("data/italy.csv", "Italy"),
            load_country("data/france.csv", "France"),
            load_country("data/germany.csv", "Germany"),
            load_country("data/spain.csv", "Spain"),
        ],
        ignore_index=True,
    )

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df[["Country", "Item", "Value"]].copy()

    df["Country"] = df["Country"].astype(str)
    df["Item"] = df["Item"].astype(str)
    return df

FEATURES = ["Country", "Item"]
TARGET = "Value"

df = load_data()

# =========================================================
# Model training
# =========================================================
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = RandomForestRegressor(
        n_estimators=140,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return pipeline, X_train, X_test, y_train, y_test, mae, r2

pipeline, X_train, X_test, y_train, y_test, mae, r2 = train_model(df)

median_value = float(df["Value"].median()) if len(df) else np.nan
relative_mae = (
    (mae / median_value) * 100
    if median_value and not np.isnan(median_value)
    else np.nan
)

# =========================================================
# SHAP (LAZY INIT + BUTTON TRIGGER)
# =========================================================
def get_explainer():
    """
    Create the SHAP explainer only when needed.
    This avoids the app sitting 'in the oven' on startup.
    """
    if "shap_explainer" not in st.session_state:
        rf_model = pipeline.named_steps["model"]
        st.session_state["shap_explainer"] = shap.TreeExplainer(rf_model)
    return st.session_state["shap_explainer"]

def transform_with_names(pipeline: Pipeline, X: pd.DataFrame):
    pre = pipeline.named_steps["preprocess"]
    X_trans = pre.transform(X)  # may be sparse
    names = pre.named_transformers_["cat"].get_feature_names_out(FEATURES)
    return X_trans, names

@st.cache_data(show_spinner=False)
def aggregated_shap_summary_cached(X_sample: pd.DataFrame, seed: int = 42):
    """
    Cached aggregated SHAP so Insights tab doesn't recompute endlessly.
    Note: we compute SHAP on transformed sample and store summary only.
    """
    explainer = get_explainer()

    X_trans, names = transform_with_names(pipeline, X_sample)
    X_dense = safe_to_dense(X_trans)

    shap_matrix = explainer.shap_values(X_dense, check_additivity=False)
    shap_df = pd.DataFrame(shap_matrix, columns=names)
    shap_df["Country"] = X_sample["Country"].values

    country_cols = [c for c in names if c.startswith("Country_")]
    item_cols = [c for c in names if c.startswith("Item_")]

    summary = (
        shap_df.groupby("Country")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_abs_country_effect": float(g[country_cols].abs().mean().mean()) if country_cols else 0.0,
                    "avg_abs_item_effect": float(g[item_cols].abs().mean().mean()) if item_cols else 0.0,
                }
            )
        )
        .reset_index()
    )

    summary["item_to_country_ratio"] = summary["avg_abs_item_effect"] / (summary["avg_abs_country_effect"] + 1e-9)
    summary = summary.sort_values("item_to_country_ratio", ascending=False)

    return summary, names

# =========================================================
# Header
# =========================================================
st.title("FAOSTAT Explainable ML Case Study Dashboard (Italy • France • Germany • Spain)")

# =========================================================
# Metrics row
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df)}")
c2.metric("MAE (tonnes)", f"{mae:,.0f}")
c3.metric("Relative MAE", f"{relative_mae:.1f}%" if not np.isnan(relative_mae) else "NA")
c4.metric("R²", f"{r2:.3f}")

with st.expander("How to read these metrics", expanded=False):
    st.markdown(f"""
- **MAE (tonnes)**: average absolute error on the held-out test split.
- **Relative MAE (%)**: MAE scaled by the **median** production value (robust for skew).
- **R²**: variance explained. With only (Country, Item), R² may be modest because real drivers
  like yield/area/weather aren’t included.
""")

# =========================================================
# Tabs (4 tabs)
# =========================================================
tab0, tab1, tab2, tab3 = st.tabs(["Case study", "Explore", "Predict + Explain", "Insights"])

# ----------------------------
# TAB 0: Case study
# ----------------------------
with tab0:
    st.subheader("What this dashboard is doing, and what you can do next")
    st.markdown("""
### What you're seeing now (current phase)
This is a compact explainable-ML case study using **Country + Item** to learn patterns in FAOSTAT-style production values.
The model is used as an **interpretation engine**, not a crystal ball.

### What I would do in a further phase (next steps)
1. **Add real drivers**: harvested area, yield, irrigation, rainfall/temperature proxies, fertilizer use.
2. **Time-awareness**: add year and trend features; check structural breaks.
3. **Validation by group**: evaluate by leaving one country out (generalization).
4. **Better explanations**: add partial dependence / ICE plots and SHAP interaction checks.
5. **Story outputs**: auto-generate “country profile cards” with key crops + explanation highlights.

If you want, I can also restructure the project into `src/` modules so it’s easier to maintain.
""")

# ----------------------------
# TAB 1: Explore
# ----------------------------
with tab1:
    st.subheader("Explore country production patterns")
    st.caption("Goal: understand production composition by item, then compare structures across countries.")

    country = st.selectbox("Country", sorted(df["Country"].unique()), key="expl_country")
    df_c = df[df["Country"] == country].copy()

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df_c.sort_values("Value", ascending=False).head(30), use_container_width=True)
    with right:
        st.markdown("#### Quick stats")
        st.write(f"Items: **{df_c['Item'].nunique()}**")
        st.write(f"Total production: **{df_c['Value'].sum():,.0f} t**")
        st.write(f"Median item production: **{df_c['Value'].median():,.0f} t**")

    topn = st.slider("Top-N items", 5, 30, 10, key="expl_topn")

    view_mode = st.radio(
        "Bar chart view",
        ["Total production (tonnes)", "Share of country total (%)"],
        horizontal=True,
        key="expl_view",
    )
    use_log = st.checkbox("Use log scale (for tonnes)", value=False, key="expl_log")

    top_items = (
        df_c.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(topn)
    )

    if view_mode == "Share of country total (%)":
        denom = df_c["Value"].sum()
        top_items["Value"] = (top_items["Value"] / denom) * 100 if denom else 0
        y_label = "Share (%)"
    else:
        y_label = "Tonnes"

    fig = px.bar(
        top_items,
        x="Item",
        y="Value",
        title=f"{country}: Top {topn} items",
        labels={"Value": y_label},
    )
    if use_log and view_mode == "Total production (tonnes)":
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download this country subset (CSV)",
        df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_data.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Compare two countries (structure)")
    st.caption("This comparison answers: do two countries differ mainly by scale, or by composition?")

    colA, colB = st.columns(2)
    countries_sorted = sorted(df["Country"].unique())
    with colA:
        cA = st.selectbox("Country A", countries_sorted, index=0, key="cmp_A")
    with colB:
        cB = st.selectbox("Country B", countries_sorted, index=1 if len(countries_sorted) > 1 else 0, key="cmp_B")

    dfA = df[df["Country"] == cA].groupby("Item", as_index=False)["Value"].sum()
    dfB = df[df["Country"] == cB].groupby("Item", as_index=False)["Value"].sum()

    top_global = (
        pd.concat([dfA, dfB], ignore_index=True)
        .groupby("Item", as_index=False)["Value"].sum()
        .sort_values("Value", ascending=False)
        .head(15)["Item"]
    )

    dfA2 = dfA[dfA["Item"].isin(top_global)].assign(Country=cA)
    dfB2 = dfB[dfB["Item"].isin(top_global)].assign(Country=cB)
    cmp = pd.concat([dfA2, dfB2], ignore_index=True)

    cmp_mode = st.radio("Compare as", ["Tonnes", "Share (%)"], horizontal=True, key="cmp_mode")
    if cmp_mode == "Share (%)":
        cmp["Value"] = cmp.groupby("Country")["Value"].transform(lambda s: (s / s.sum() * 100) if s.sum() else 0)

    fig_cmp = px.bar(
        cmp,
        x="Item",
        y="Value",
        color="Country",
        barmode="group",
        title=f"{cA} vs {cB}: Top items comparison",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ----------------------------
# TAB 2: Predict + Explain
# ----------------------------
with tab2:
    st.subheader("Predict + Explain (SHAP)")
    st.caption("Goal: explain one selected (Country, Item) estimate using SHAP contributions.")

    cA, cB = st.columns(2)
    with cA:
        sel_country = st.selectbox("Country", sorted(df["Country"].unique()), key="p_country")
    with cB:
        items_for_country = sorted(df[df["Country"] == sel_country]["Item"].unique())
        sel_item = st.selectbox("Item", items_for_country, key="p_item")

    row = pd.DataFrame([{"Country": sel_country, "Item": sel_item}])
    pred = float(pipeline.predict(row)[0])
    st.metric("Model estimate (tonnes)", f"{pred:,.0f}")

    with st.expander("How to read the SHAP waterfall", expanded=False):
        st.markdown("""
- The model starts from a **baseline** (average prediction).
- It adds/subtracts feature contributions (Country_* and Item_*).
- The final sum becomes the prediction.
""")

    run_shap = st.button("Compute SHAP explanation", type="primary")

    if run_shap:
        with st.spinner("Computing SHAP (this can take a moment)..."):
            explainer = get_explainer()

            X_trans, names = transform_with_names(pipeline, row)
            X_dense = safe_to_dense(X_trans)

            shap_vals = explainer.shap_values(X_dense, check_additivity=False)[0]

            exp = shap.Explanation(
                values=shap_vals,
                base_values=explainer.expected_value,
                feature_names=names,
            )

        st.markdown("#### SHAP explanation (waterfall)")
        plt.figure()
        shap.waterfall_plot(exp, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        st.markdown("#### Top SHAP contributors (with direction)")
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:15]
        contrib = pd.DataFrame(
            {"feature": np.array(names)[top_idx], "shap_value": shap_vals[top_idx]}
        )
        contrib["direction"] = np.where(contrib["shap_value"] >= 0, "↑ increases", "↓ decreases")
        contrib = contrib.sort_values(by="shap_value", ascending=False)

        st.dataframe(contrib, use_container_width=True)

        st.download_button(
            "Download SHAP contributors (CSV)",
            contrib.to_csv(index=False).encode("utf-8"),
            file_name=f"shap_contrib_{sel_country}_{sel_item}.csv".replace(" ", "_"),
            mime="text/csv",
        )
    else:
        st.info("Click **Compute SHAP explanation** to generate the waterfall + contributors (keeps the app fast).")

# ----------------------------
# TAB 3: Insights
# ----------------------------
with tab3:
    st.subheader("Insights (aggregated explainability)")
    st.caption("Goal: compare the strength of crop-driven vs country-driven effects across countries.")

    with st.expander("What do these aggregated SHAP numbers mean?", expanded=True):
        st.markdown("""
These metrics summarize SHAP values over a sample:

- **avg_abs_item_effect**: typical strength of Item_* effects.
- **avg_abs_country_effect**: typical strength of Country_* effects.
- **item_to_country_ratio**: >1 means item effects dominate; <1 means country effects dominate.

Because we use **absolute values**, this measures *strength*, not direction.
""")

    st.markdown("### A) Crop vs Country: aggregated SHAP comparison")

    # Safe bounds so slider never breaks
    max_allowed = max(20, len(X_test))
    slider_max = max(20, min(160, max_allowed))
    default_val = max(20, min(80, slider_max))

    sample_n = st.slider(
        "Sample size used for aggregated SHAP (speed vs stability)",
        20,
        slider_max,
        default_val,
        key="ins_sample_n",
    )

    # Use a deterministic sample for stable caching
    X_sample = X_test.sample(sample_n, random_state=42)

    # Button-triggered compute so it doesn't constantly rerun while you tweak other UI
    compute_insights = st.button("Compute aggregated SHAP", type="primary", key="ins_btn")

    if compute_insights:
        with st.spinner("Aggregating SHAP..."):
            summary, names = aggregated_shap_summary_cached(X_sample, seed=42)

        st.dataframe(summary, use_container_width=True)

        metric = st.selectbox(
            "Choose what to plot",
            ["avg_abs_item_effect", "avg_abs_country_effect", "item_to_country_ratio"],
            key="ins_metric",
        )

        fig_metric = px.bar(
            summary.sort_values(metric, ascending=False),
            x="Country",
            y=metric,
            title=f"Country comparison: {metric}",
        )
        st.plotly_chart(fig_metric, use_container_width=True)

        st.caption("Tip: **item_to_country_ratio > 1** means item effects dominate; **< 1** means country effects dominate.")
    else:
        st.info("Click **Compute aggregated SHAP** to generate the table + plot (keeps the tab snappy).")

    st.divider()
    st.markdown("### B) Cross-country composition: top items heatmap")
    st.caption("Compare which items dominate total production across the four countries.")

    top_k = st.slider("How many top items to include", 10, 40, 20, key="heat_k")
    top_items_global = (
        df.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(top_k)["Item"]
        .tolist()
    )

    heat = (
        df[df["Item"].isin(top_items_global)]
        .groupby(["Item", "Country"], as_index=False)["Value"]
        .sum()
    )
    pivot = heat.pivot(index="Item", columns="Country", values="Value").fillna(0)

    view = st.radio("Heatmap values", ["Tonnes", "Share within country (%)"], horizontal=True, key="heat_view")
    pivot_view = pivot.copy()
    if view == "Share within country (%)":
        pivot_view = pivot_view.div(pivot_view.sum(axis=0).replace(0, np.nan), axis=1) * 100
        pivot_view = pivot_view.fillna(0)

    st.dataframe(pivot_view, use_container_width=True)

    try:
        fig_h = px.imshow(
            pivot_view,
            aspect="auto",
            title=f"Top {top_k} items: {view}",
        )
        st.plotly_chart(fig_h, use_container_width=True)
    except Exception:
        st.info("Heatmap rendering skipped (table above is available).")

    with st.expander("How to interpret this heatmap", expanded=False):
        st.markdown("""
- **Tonnes** shows absolute magnitude.
- **Share (%)** shows composition (structure) within each country.
""")
