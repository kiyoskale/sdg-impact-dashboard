import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from src.analytics import build_correlation_matrix, build_wide_frame, calculate_normalized_growth, compute_summary_stats, shared_years
from src.db_manager import get_available_regions, get_countries_indicator, get_country_indicators, get_year_bounds

st.set_page_config(page_title="SDG Impact Dashboard", layout="wide")
st.title("SDG Impact Dashboard")
st.caption("Tracking progress towards the Sustainable Development Goals")

INDICATORS = {
    "Poverty (SDG 1)": "SI.POV.DDAY",
    "Maternal Mortality (SDG 3)": "SH.STA.MMRT",
    "Education Expenditure (SDG 4)": "SE.XPD.TOTL.GD.ZS",
    "Access to Electricity (SDG 7)": "EG.ELC.ACCS.ZS",
}
NAMES = {code: label for label, code in INDICATORS.items()}

countries = get_available_regions()
default = countries.index("Morocco") if "Morocco" in countries else 0
country = st.selectbox("Region / Country", countries, index=default)
label = st.selectbox("SDG indicator", list(INDICATORS))
code = INDICATORS[label]
min_year, max_year = get_year_bounds()
start, end = st.slider("Year range", min_year, max_year, (min_year, max_year))
data = get_countries_indicator([country], code, start, end).dropna(subset=["value"])

if data.empty:
    st.info("No data for this country, indicator and period.")
    st.stop()
st.line_chart(data, x="year", y="value")
st.dataframe(data)

# --- Statistical insights ---
st.header("Statistical Insights")
stats = compute_summary_stats(data)
col1, col2, col3 = st.columns(3)
col1.metric("Standard deviation", "—" if pd.isna(stats["std"]) else f"{stats['std']:.2f}")
col2.metric("Peak growth year", stats["peak_growth_year"] or "—")
col3.metric("Average value", f"{stats['mean']:.2f}")
st.dataframe(pd.Series(stats, name=label), width="content")

st.subheader("Correlation between indicators")
method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
multi = get_country_indicators(country, list(INDICATORS.values()), start, end)
frames = [group for _, group in multi.groupby("indicator_code")]
corr = build_correlation_matrix(frames, method=method).dropna(how="all").dropna(axis=1, how="all")
if corr.empty:
    st.info("Not enough overlapping years (minimum 3) to correlate indicators.")
else:
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr.rename(index=NAMES, columns=NAMES), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    left.pyplot(fig)
    right.pyplot(sns.pairplot(build_wide_frame(frames)[corr.columns].rename(columns=NAMES)).figure)
    st.caption("Indicators that both trend over time can correlate strongly without a causal link.")

st.subheader("Country benchmark")
peers = st.multiselect("Compare countries", countries, default=[c for c in dict.fromkeys([country, "Algeria", "Tunisia"]) if c in countries])
bench = get_countries_indicator(peers, code, start, end).dropna(subset=["value"])
years = shared_years(bench)
if len(peers) < 2 or not years:
    st.info("Choose at least two countries that report this indicator in a common year.")
else:
    base = st.selectbox("Baseline year", years)
    growth = calculate_normalized_growth(bench, base)
    left, right = st.columns(2)
    left.line_chart(growth, x="year", y="growth_pct", color="country_name")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=bench, x="country_name", y="value", ax=ax)
    right.pyplot(fig)

plt.close("all")
