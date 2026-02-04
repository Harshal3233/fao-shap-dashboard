import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="FAOSTAT XAI Case Study", layout="wide")


# =========================================================
# Data
# =========================================================
@st.cache_data
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

    # Minimal columns for this study
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df[["Country", "Item", "Value"]].copy()

    df["Country"] = df["Country"].astype(str)
    df["Item"] = df["Item"].astype(str)

    return df


df = load_data()
FEATURES = ["Country", "Item"]
TARGET = "Value"


# =========================================================
# Model training
# =========================================================
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
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
relative_mae = (mae / median_value) * 100 if median_value and not np.isnan(median_value) else np.nan


# =========================================================
# SHAP explainer
# IMPORTANT: do NOT cache a function that takes 'pipeline' as param
# =========================================================
rf_model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)


def transform_with_names(pipeline: Pipeline, X: pd.DataFrame):
    pre = pipeline.named_steps["preprocess"]
    X_trans = pre.transform(X).toarray()
    names = pre.named_transformers_["cat"].get_feature_names_out(FEATURES)
    return X_trans, names


# =========================================================
# Header
# =========================================================
st.title("🌾 FAOSTAT Explainable ML Dashboard (Case Study)")
st.caption("Italy • France • Germany • Spain | Explainable ML + Interactive Comparison")


# =========================================================
# Narrative block: Motivation -> Problem -> Solution -> Theory -> Next steps
# (More detailed, structured, and intentionally not '100% complete')
# =========================================================
with st.expander("📘 Case Study Narrative (Motivation → Problem → Solution → Theory → Next steps)", expanded=True):
    st.markdown("""
## 1) Motivation (why this matters)
In international agricultural analysis, decision-makers often need to compare countries transparently:
- What is different across countries?
- Which crops dominate production?
- Are observed differences primarily about crop composition or broader structural context?

In many practical settings, the objective is not purely forecasting, but **interpretable comparison** and **clear communication**.

## 2) The issue (the analytical gap)
Simple totals and rankings tell *what* is produced, but not *why* differences occur.  
A common gap is separating two intertwined drivers:
- **Item effects**: “This country looks high because it produces high-volume crops.”
- **Country effects**: “This country looks high because country context amplifies output (structure, geography, systems).”

Without a structured approach, it’s easy to confuse:
- scale vs composition,
- dominance of a few crops vs broad structural differences.

## 3) The solution (what this project provides)
This project builds a reproducible, explainable workflow using FAOSTAT production data to:
- compare production composition across countries (Explore)
- generate a structured estimate for any (Country, Item) pair (Predict)
- explain estimates using SHAP contributions (Explain)
- aggregate SHAP to quantify “Crop vs Country” influence (Insights)

The emphasis is **interpretability and comparison**, not claiming real-world forecasting capability.

## 4) Theory (how to interpret the explainability)
### 4.1 Why a model at all?
The model acts as a structured way to learn patterns in the dataset, so that we can ask:
- “What does the model attribute differences to, on average?”
- “How strong is the role of crop vs country in the learned structure?”

### 4.2 What SHAP represents (simple rule-set)
SHAP decomposes each prediction into additive contributions:
- Positive SHAP value → pushes the estimate upward
- Negative SHAP value → pushes the estimate downward
- Contributions sum back to the prediction (baseline + effects)

### 4.3 Aggregated SHAP (the key comparison)
We compute:
- **avg_abs_item_effect**: how strongly crop choice typically shifts estimates up/down
- **avg_abs_country_effect**: how strongly country context typically shifts estimates up/down

Because we take absolute values, this measures **strength**, not direction.

## 5) If I continue this project (planned next steps)
To deepen analytical value, the next iteration would include:
1. **Multi-year time series** to study trends and shocks (not only a snapshot).
2. **Additional drivers** (if available): area harvested, yield, rainfall proxies, prices.
3. **Better validation strategy**: country-aware splits and sensitivity checks.
4. **Policy-style outputs**: short briefs, confidence ranges, and scenario comparisons.
5. **More granular aggregation**: clustering countries by crop profiles, not only totals.

These steps would transform the dashboard from a structural prototype into a richer analytical product.
    """)


# =========================================================
# Metrics row + interpretability
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df)}")
c2.metric("MAE (tonnes)", f"{mae:,.0f}")
c3.metric("Relative MAE", f"{relative_mae:.1f}%" if not np.isnan(relative_mae) else "NA")
c4.metric("R²", f"{r2:.3f}")

with st.expander("How to interpret MAE / Relative MAE / R²", expanded=False):
    st.markdown(f"""
- **MAE (tonnes)** is the average absolute error on held-out test data.
- **Relative MAE (%)** scales MAE by the median production value so the error is easier to interpret.
  Here, Relative MAE ≈ **{relative_mae:.1f}%**.
- **R²** can be modest because the model intentionally uses only (Country, Item) and does not include
  real drivers like yield, harvested area, weather, management inputs, etc. This is expected for a minimal structural prototype.
    """)


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["Explore", "Predict + Explain", "Insights"])


# ----------------------------
# TAB 1: Explore
# ----------------------------
with tab1:
    st.subheader("Explore")
    st.caption("Compare production composition within a country and across countries (structure vs scale).")

    country = st.selectbox("Country", sorted(df["Country"].unique()))
    df_c = df[df["Country"] == country].copy()

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df_c.sort_values("Value", ascending=False).head(30), use_container_width=True)
    with right:
        st.markdown("#### Quick stats")
        st.write(f"Items: **{df_c['Item'].nunique()}**")
        st.write(f"Total production: **{df_c['Value'].sum():,.0f} t**")
        st.write(f"Median item production: **{df_c['Value'].median():,.0f} t**")

    topn = st.slider("Top-N items", 5, 30, 10)

    view_mode = st.radio(
        "Bar chart view",
        ["Total production (tonnes)", "Share of country total (%)"],
        horizontal=True
    )
    use_log = st.checkbox("Use log scale (only for tonnes)", value=False)

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
        labels={"Value": y_label}
    )
    if use_log and view_mode == "Total production (tonnes)":
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this chart", expanded=False):
        st.markdown("""
- **Tonnes** shows absolute magnitude (bigger countries/crops dominate).
- **Share (%)** shows structure: which crops dominate within the country regardless of scale.
Use Share (%) when comparing countries fairly.
        """)

    st.download_button(
        " Download this country subset (CSV)",
        df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_data.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Compare two countries")
    st.caption("This answers: Are differences mainly scale-driven or composition-driven?")

    colA, colB = st.columns(2)
    countries_sorted = sorted(df["Country"].unique())
    with colA:
        cA = st.selectbox("Country A", countries_sorted, index=0, key="cA")
    with colB:
        cB = st.selectbox("Country B", countries_sorted, index=1 if len(countries_sorted) > 1 else 0, key="cB")

    dfA = df[df["Country"] == cA].groupby("Item", as_index=False)["Value"].sum()
    dfB = df[df["Country"] == cB].groupby("Item", as_index=False)["Value"].sum()

    top_global = (
        pd.concat([dfA, dfB])
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
        title=f"{cA} vs {cB}: Top items comparison"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    with st.expander("How to interpret this comparison", expanded=False):
        st.markdown("""
- If the **Tonnes** chart differs a lot but the **Share (%)** chart looks similar, differences are mostly **scale**.
- If the **Share (%)** chart differs strongly, differences are mainly **composition/structure**.
        """)


# ----------------------------
# TAB 2: Predict + Explain
# ----------------------------
with tab2:
    st.subheader("Predict + Explain (SHAP)")
    st.caption("Pick a (Country, Item) pair to view an estimate and an explanation of what drove it.")

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
- It then adds/subtracts contributions from the active features (Country_* and Item_*).
- The final value is the estimate for your selected (Country, Item).

Positive bars push the estimate up; negative bars push it down.
        """)

    X_trans, names = transform_with_names(pipeline, row)
    shap_vals = explainer.shap_values(X_trans, check_additivity=False)[0]

    exp = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        feature_names=names,
    )

    st.markdown("#### SHAP explanation (waterfall)")
    plt.figure()
    shap.waterfall_plot(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.markdown("#### Top contributors (with direction)")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:15]
    contrib = pd.DataFrame(
        {"feature": np.array(names)[top_idx], "shap_value": shap_vals[top_idx]}
    )
    contrib["direction"] = np.where(contrib["shap_value"] >= 0, "↑ increases", "↓ decreases")
    contrib = contrib.sort_values(by="shap_value", ascending=False)

    st.dataframe(contrib, use_container_width=True)

    st.download_button(
        " Download SHAP contributors (CSV)",
        contrib.to_csv(index=False).encode("utf-8"),
        file_name=f"shap_contrib_{sel_country}_{sel_item}.csv".replace(" ", "_"),
        mime="text/csv",
    )


# ----------------------------
# TAB 3: Insights
# ----------------------------
with tab3:
    st.subheader("Insights (Crop vs Country)")
    st.caption("Aggregated SHAP compares how strong crop choice vs country context is, on average.")

    with st.expander("What do 'avg_abs_item_effect' and 'avg_abs_country_effect' mean?", expanded=True):
        st.markdown("""
These metrics summarize SHAP values over a sample of rows.

- **avg_abs_item_effect**: average absolute SHAP magnitude for Item_* features.
  It answers: *how strongly crop choice typically moves estimates up/down?*

- **avg_abs_country_effect**: average absolute SHAP magnitude for Country_* features.
  It answers: *how strongly country context typically shifts estimates up/down?*

Because we take **absolute values**, this measures **strength of influence**, not direction.

✅ Interpretation:
- If **avg_abs_item_effect > avg_abs_country_effect**, variation is mainly **crop-driven**.
- If **avg_abs_country_effect is high**, **country context** plays a stronger structural role.
        """)

    st.markdown("### A) Aggregated SHAP strength comparison")
    sample_n = st.slider(
        "Sample size for aggregated SHAP (speed vs stability)",
        20,
        min(200, len(X_test)),
        min(80, len(X_test)),
    )
    X_sample = X_test.sample(sample_n, random_state=42)

    X_trans, names = transform_with_names(pipeline, X_sample)
    shap_matrix = explainer.shap_values(X_trans, check_additivity=False)

    shap_df = pd.DataFrame(shap_matrix, columns=names)
    shap_df["Country"] = X_sample["Country"].values

    country_cols = [c for c in names if c.startswith("Country_")]
    item_cols = [c for c in names if c.startswith("Item_")]

    summary = (
        shap_df.groupby("Country")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_abs_country_effect": g[country_cols].abs().mean().mean() if country_cols else 0.0,
                    "avg_abs_item_effect": g[item_cols].abs().mean().mean() if item_cols else 0.0,
                }
            )
        )
        .reset_index()
    )
    summary["item_to_country_ratio"] = summary["avg_abs_item_effect"] / (summary["avg_abs_country_effect"] + 1e-9)
    summary = summary.sort_values("item_to_country_ratio", ascending=False)

    st.dataframe(summary, use_container_width=True)

    metric = st.selectbox(
        "Choose what to plot",
        ["avg_abs_item_effect", "avg_abs_country_effect", "item_to_country_ratio"],
    )

    fig_metric = px.bar(
        summary.sort_values(metric, ascending=False),
        x="Country",
        y=metric,
        title=f"Country comparison: {metric}",
    )
    st.plotly_chart(fig_metric, use_container_width=True)

    st.caption("Tip: ratio > 1 ⇒ crop effects dominate; ratio < 1 ⇒ country context dominates.")

    st.divider()
    st.markdown("### B) Cross-country composition heatmap (Top items)")
    st.caption("Compare which items dominate across countries. Switch between Tonnes and within-country share.")

    top_k = st.slider("How many top items to include", 10, 40, 20)

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

    view = st.radio("Heatmap values", ["Tonnes", "Share within country (%)"], horizontal=True)
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

    with st.expander("How to read this heatmap", expanded=False):
        st.markdown("""
- **Tonnes** highlights absolute scale (large producers dominate visually).
- **Share (%)** highlights structure: which crops matter *within* each country.
Use Share (%) to avoid confusing scale with composition.
        """)

    st.divider()
    st.subheader("Roadmap: Phase 2")
    st.markdown("""
If this project is extended, the next steps would focus on turning this prototype into a richer analytical system:

- **Multi-year panel** (add trends, shocks, and seasonal narratives)
- **Add drivers** (area harvested, yields, trade, rainfall proxies)
- **Robust validation** (country-aware splits + sensitivity analysis)
- **Clustering** (group countries by crop profiles, not only totals)
- **Policy deliverables** (short briefs, confidence bands, scenario comparisons)


    """)
