
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Diversity ROI – Outreach Optimization", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_diversity_roi_data.csv")
    # derive helpful fields
    df["date_first_touch"] = pd.to_datetime(df["date_first_touch"])
    df["enroll_dt"] = pd.to_datetime(df["enroll_dt"], errors="coerce")
    df["is_urp"] = df["race_ethnicity"].isin(["Black","Hispanic"]).astype(int)
    # time-to-enroll (days)
    df["time_to_enroll_days"] = (df["enroll_dt"] - df["date_first_touch"]).dt.days
    return df

df = load_data()

st.title("Community Engagement & Diversity Analytics")
st.caption("Synthetic demo – optimize outreach for diverse recruitment and retention")

# --- Sidebar filters
with st.sidebar:
    st.header("Filters")
    trials = st.multiselect("Trials", sorted(df["trial_id"].unique().tolist()), default=sorted(df["trial_id"].unique().tolist()))
    channels = st.multiselect("Channels", sorted(df["outreach_channel"].unique().tolist()), default=sorted(df["outreach_channel"].unique().tolist()))
    regions = st.multiselect("Regions", sorted(df["region"].unique().tolist()), default=sorted(df["region"].unique().tolist()))
    start_date = st.date_input("Start date", df["date_first_touch"].min())
    end_date = st.date_input("End date", df["date_first_touch"].max())
    urp_only = st.checkbox("URP only (Black & Hispanic)", value=False)

mask = (
    df["trial_id"].isin(trials) &
    df["outreach_channel"].isin(channels) &
    df["region"].isin(regions) &
    (df["date_first_touch"].dt.date >= pd.to_datetime(start_date)) &
    (df["date_first_touch"].dt.date <= pd.to_datetime(end_date))
)
if urp_only:
    mask &= df["is_urp"] == 1

f = df.loc[mask].copy()

# --- KPI tiles
def safe_div(n, d):
    return (n / d) if d else np.nan

qualified = int(f["eligible_flag"].sum())
enrolled = int(f["enrolled"].sum())
retained3 = int(f["retained_3m"].sum())
diversity_pct = 100.0 * safe_div(f["is_urp"].sum(), len(f)) if len(f) else np.nan
cpl = safe_div(f["outreach_cost"].sum(), qualified)
cpe = safe_div(f["outreach_cost"].sum(), enrolled)
cpr = safe_div(f["outreach_cost"].sum(), retained3)
tte = float(np.nanmedian(f.loc[f["time_to_enroll_days"].notna(), "time_to_enroll_days"])) if enrolled else np.nan

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Qualified Leads", f"{qualified:,}")
kpi2.metric("Cost per Enrolled", "—" if np.isnan(cpe) else f"${cpe:,.0f}")
kpi3.metric("Cost per Retained (3m)", "—" if np.isnan(cpr) else f"${cpr:,.0f}")
kpi4.metric("URP Share", "—" if np.isnan(diversity_pct) else f"{diversity_pct:,.1f}%")
kpi5.metric("Median Time-to-Enroll (days)", "—" if np.isnan(tte) else f"{tte:,.0f}")

st.divider()

# --- Map
st.subheader("Geographic view")
map_df = f.groupby(["zip","lat","lon","region"], as_index=False).agg(
    enrolled=("enrolled","sum"),
    retained_3m=("retained_3m","sum")
)
st.map(map_df.rename(columns={"lat":"latitude","lon":"longitude"}), size="enrolled")

# --- Channel performance
st.subheader("Channel performance")
perf = f.groupby("outreach_channel", as_index=False).agg(
    outreach_cost=("outreach_cost","sum"),
    qualified=("eligible_flag","sum"),
    enrolled=("enrolled","sum"),
    retained_3m=("retained_3m","sum")
)
perf["CPE"] = perf["outreach_cost"] / perf["enrolled"].replace(0, np.nan)
perf["CPR_3m"] = perf["outreach_cost"] / perf["retained_3m"].replace(0, np.nan)

c1, c2 = st.columns(2)
bar1 = alt.Chart(perf).mark_bar().encode(
    x=alt.X("outreach_channel:N", title="Channel"),
    y=alt.Y("CPE:Q", title="Cost per Enrolled (USD)")
)
c1.altair_chart(bar1, use_container_width=True)

bar2 = alt.Chart(perf).mark_bar().encode(
    x=alt.X("outreach_channel:N", title="Channel"),
    y=alt.Y("CPR_3m:Q", title="Cost per Retained 3m (USD)")
)
c2.altair_chart(bar2, use_container_width=True)

# --- Funnel
st.subheader("Funnel")
funnel = pd.DataFrame({
    "stage": ["Touched","Qualified","Enrolled","Retained 3m"],
    "count": [len(f), f["eligible_flag"].sum(), f["enrolled"].sum(), f["retained_3m"].sum()]
})
funnel_chart = alt.Chart(funnel).mark_bar().encode(
    x=alt.X("stage:N", sort=["Touched","Qualified","Enrolled","Retained 3m"]),
    y=alt.Y("count:Q")
)
st.altair_chart(funnel_chart, use_container_width=True)

# --- Recommendations (very simple heuristic for demo)
st.subheader("Recommendations")
if len(perf):
    perf_sorted = perf.sort_values("CPR_3m", ascending=True).head(2)["outreach_channel"].tolist()
    worst = perf.sort_values("CPR_3m", ascending=False).head(1)["outreach_channel"].tolist()
    st.write(f"- Consider shifting budget **toward**: {', '.join(perf_sorted)}")
    st.write(f"- Consider reducing spend **from**: {', '.join(worst)}")
    st.caption("Heuristic demo only — replace with optimization model in production.")
else:
    st.info("Not enough data for recommendations with current filters.")
