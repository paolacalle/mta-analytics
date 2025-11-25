from typing import Tuple, List
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from math import log

total_cols_by_type = {
    "subway" : "subway_total_estimated_ridership",
    "bus" : "bus_total_estimated_ridership",
    "lirr" : "lirr_total_estimated_ridership",
    "metro_north" : "metro_north_total_estimated_ridership",
    "access_a_ride" : "access_a_ride_total_scheduled_trips",
    "sirr" : "sirr_total_estimated_ridership"
}

pct_cols_by_type = {
    "subway" : "subway_pct_of_prepandemic_day",
    "bus" : "bus_pct_of_prepandemic_day",
    "lirr" : "lirr_pct_of_prepandemic_day",
    "metro_north" : "metro_north_pct_of_prepandemic_day",
    "access_a_ride" : "access_a_ride_pct_of_prepandemic_day",
    "sirr" : "sirr_pct_of_prepandemic_day"
}

def render_mode_selection() -> Tuple[List[str], List[str]]:
    """
    Renders a multi-select widget for selecting ridership modes.
    Returns selected modes and corresponding column names.
    """
    ridership_types = list(total_cols_by_type.keys()) + ["all"]
    selected_modes = st.multiselect(
        "Select Ridership Modes to Display",
        options=ridership_types,
        default=["subway"]
    )
    
    if "all" in selected_modes:
        selected_modes = list(total_cols_by_type.keys())
        
    col_names = [total_cols_by_type[mode] for mode in selected_modes]
    return selected_modes, col_names

def render_eda_table(df_ridership: pd.DataFrame) -> None:
    """
    Renders an exploratory data analysis table of ridership data.
    Includes summary stats and quick insights.
    """
    st.markdown("### Ridership Data Overview")
    
    # Basic stats
    total_days = df_ridership.shape[0]
    start_date = df_ridership["date"].min()
    end_date = df_ridership["date"].max()
    start_year = start_date.year
    end_year = end_date.year
    
    days_per_year = (
        df_ridership.groupby(df_ridership["date"].dt.year)
        .nunique()["date"]
        .reset_index(name="days_recorded")
        .rename(columns={"date": "year"})
    )
    
    # Summary boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days Recorded", f"{total_days}")
    with col2:
        st.metric("Date Range", f"{start_date.date()} to {end_date.date()}")
    with col3:
        st.metric("Years Covered", f"{start_year}–{end_year}")
    
    st.markdown("### Days Covered Per Year")
    st.write(days_per_year)
    
    # drop down -- note about year 2024
    with st.expander("Note on 2024 Data"):
        st.markdown(
            """
            2024 was a leap year, so there are 366 days in total.
            A leap day (February 29) was included in the data.
            """
        )
        
    # drop down -- not about year 2025
    with st.expander("Note on 2025 Data"):
        st.markdown(
            """
            As of now, 2025 data is partial and does not cover the full year.
            Please interpret 2025 ridership trends with caution.
            """
        )

# Ridership over time 
def render_ridership_over_daily_time(
    df_ridership: pd.DataFrame,
    selected_modes: List[str],
    col_names: List[str]
    ) -> None:
    """
    Renders a line chart of ridership over time.
    """
    st.markdown("### Subway Ridership Over Time")
    
    # select many modes to display
    if df_ridership.empty:
        st.info("No ridership data available.")
        return

    # line graph for each ridership type except overall
    fig = go.Figure()
    
    df_ridership_c = df_ridership.copy()
        
    # group by date and sum ridership values (in case of duplicates)
    df_ridership_grouped = df_ridership_c.groupby("date").sum().reset_index()

    for ridership_type, col_name in zip(selected_modes, col_names):
        fig.add_trace(
            go.Scatter(
                x=df_ridership_grouped["date"],
                y=df_ridership_grouped[col_name],
                mode="lines",
                name=ridership_type.replace("_", " ").title()
            )
        )
        
    fig.update_layout(
    title="Daily Ridership by Mode",
    xaxis_title="Date",
    yaxis_title="Estimated Ridership",
    hovermode="x unified",
    template="plotly_white"
    )
    
    # highlight covid pandemic period
    fig.add_vrect(
        x0="2020-03-01", x1="2021-06-01",
        fillcolor="LightSalmon", opacity=0.5,
        layer="below", line_width=0,
        annotation_text="COVID-19 Pandemic", annotation_position="top left"
    )

    st.plotly_chart(fig, use_container_width=True)
    
def render_ridership_yearly_time(
    df_ridership: pd.DataFrame,
    selected_modes: List[str],
    col_names: List[str]
) -> None:
    """
    Renders a grouped bar chart of yearly ridership totals
    for the selected modes and adds quick analysis.
    """
    st.markdown("### Yearly Ridership Totals by Mode")

    if df_ridership.empty:
        st.info("No ridership data available.")
        return

    df_temp = df_ridership.copy()
    df_temp["year"] = df_temp["date"].dt.year

    # group by year and sum totals for the selected columns
    yearly = (
        df_temp[["year"] + col_names]
        .groupby("year", as_index=False)
        .sum()
    )

    # melt to long format for nice plotting
    yearly_long = yearly.melt(
        id_vars="year",
        value_vars=col_names,
        var_name="mode_col",
        value_name="total_ridership",
    )

    # map column names back to friendly mode labels
    # assume order of col_names aligns with selected_modes
    col_to_mode = dict(zip(col_names, selected_modes))
    yearly_long["mode"] = yearly_long["mode_col"].map(col_to_mode)
    yearly_long["mode_label"] = yearly_long["mode"].str.replace("_", " ").str.title()

    fig = px.bar(
        yearly_long,
        x="year",
        y="total_ridership",
        color="mode_label",
        barmode="group",
        labels={
            "year": "Year",
            "total_ridership": "Total Estimated Ridership",
            "mode_label": "Mode",
        },
        title="Yearly Total Estimated Ridership by Mode",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- Quick analysis text ----------
    st.markdown("#### Quick Yearly Insights")

    # overall total across selected modes per year
    yearly["total_all_selected_modes"] = yearly[col_names].sum(axis=1)

    # best and worst years overall
    best_year = int(yearly.loc[yearly["total_all_selected_modes"].idxmax(), "year"])
    worst_year = int(yearly.loc[yearly["total_all_selected_modes"].idxmin(), "year"])

    # example: subway-specific if it’s selected
    insights = []

    insights.append(
        f"- Across the selected modes, **{best_year}** has the highest total ridership, "
        f"while **{worst_year}** has the lowest (pandemic years usually dominate the low end)."
    )

    # for each selected mode, find best and worst years (excluding 2025)
    yearly_filtered = yearly[yearly["year"] != 2025]
    for ridership_type, col_name in zip(selected_modes, col_names):
        best_mode_year = int(yearly_filtered.loc[yearly_filtered[col_name].idxmax(), "year"])
        worst_mode_year = int(yearly_filtered.loc[yearly_filtered[col_name].idxmin(), "year"])

        insights.append(
            f"- For **{ridership_type.replace('_', ' ').title()}**, "
            f"**{best_mode_year}** had the highest ridership, "
            f"while **{worst_mode_year}** had the lowest."
        )

    st.markdown("\n".join(insights))
    
def render_proportional_difference_yearly(df_ridership, selected_modes, col_names):
    st.markdown("### Proportional Ridership Share by Mode (Yearly)")
    
    st.latex(r"""
    \text{share}_{\text{mode},\,\text{year}}
    =
    \frac{\text{ridership}_{\text{mode},\,\text{year}}}
    {\sum_{\text{modes}} \text{ridership}_{\text{mode},\,\text{year}}}
    """)
    
    
    df_temp = df_ridership.copy()
    df_temp["year"] = df_temp["date"].dt.year
    
    # exclude incomplete 2025 data
    df_temp = df_temp[df_temp["year"] != 2025]
    
    # 1. Aggregate annual totals per mode
    yearly = df_temp.groupby("year")[col_names].sum().reset_index()
    
    # 2. Compute proportions
    yearly["total_all"] = yearly[col_names].sum(axis=1)
    for col, mode in zip(col_names, selected_modes):
        yearly[f"{mode}_prop"] = yearly[col] / yearly["total_all"]
    
    # 3. Prep long-form data
    prop_cols = [f"{mode}_prop" for mode in selected_modes]
    yearly_long = yearly.melt(
        id_vars="year",
        value_vars=prop_cols,
        var_name="mode_prop",
        value_name="proportion"
    )
    
    # Clean names for display
    yearly_long["mode"] = yearly_long["mode_prop"].str.replace("_prop", "")
    yearly_long["mode_label"] = yearly_long["mode"].str.replace("_", " ").str.title()
    
    # 4. Plot proportions as line chart
    fig = px.line(
        yearly_long,
        x="year",
        y="proportion",
        color="mode_label",
        markers=True,
        title="Proportional Ridership Share Across Modes Over Time",
        labels={"proportion": "Share of Total Annual Ridership"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Quick insights
    st.markdown("#### Insights")
    insights = []
    
    for mode in selected_modes:
        prop_col = f"{mode}_prop"
        
        first_year = yearly["year"].iloc[0]
        last_year = yearly["year"].iloc[-1]
        
        first_prop = yearly.loc[yearly["year"] == first_year, prop_col].iloc[0]
        last_prop = yearly.loc[yearly["year"] == last_year, prop_col].iloc[0]
        
        diff_pct = (last_prop - first_prop) * 100
        
        direction = "increased" if diff_pct > 0 else "decreased"
        
        insights.append(
            f"- **{mode.replace('_',' ').title()} ridership share {direction} by {abs(diff_pct):.2f}%** "
            f"from {first_year} to {last_year}."
        )
    
    st.markdown("\n".join(insights))
    
def render_proportional_difference_daily(
    df_ridership: pd.DataFrame,
    selected_modes: List[str],
    col_names: List[str]
) -> None:
    st.markdown("### Proportional Ridership Share by Mode (Daily)")
    
    st.latex(r"""
    \text{share}_{\text{mode},\,\text{day}}
    =
    \frac{\text{ridership}_{\text{mode},\,\text{day}}}
    {\sum_{\text{modes}} \text{ridership}_{\text{mode},\,\text{day}}}
    """)
    
    df_temp = df_ridership.copy()
    
    # 1. Compute daily total across selected modes
    df_temp["total_all_selected"] = df_temp[col_names].sum(axis=1)
    
    # 2. Compute proportions
    for col, mode in zip(col_names, selected_modes):
        df_temp[f"{mode}_prop"] = df_temp[col] / df_temp["total_all_selected"]
    
    # 3. Prep long-form data
    prop_cols = [f"{mode}_prop" for mode in selected_modes]
    daily_long = df_temp.melt(
        id_vars="date",
        value_vars=prop_cols,
        var_name="mode_prop",
        value_name="proportion"
    )
    
    # Clean names for display
    daily_long["mode"] = daily_long["mode_prop"].str.replace("_prop", "")
    daily_long["mode_label"] = daily_long["mode"].str.replace("_", " ").str.title()
    
    # 4. Plot proportions as line chart
    fig = px.scatter(
        daily_long,
        x="date",
        y="proportion",
        color="mode_label",
        title="Proportional Ridership Share Across Modes Over Time",
        labels={"proportion": "Share of Total Daily Ridership"}
    )
    
    st.plotly_chart(fig, use_container_width=True)


from typing import List
import pandas as pd
import streamlit as st
import plotly.express as px

# --- 1. WEEKDAY vs WEEKEND SUMMARY (for a chosen mode) ---

def render_weekday_weekend_summary(df_ridership: pd.DataFrame) -> None:
    """
    Show average weekday vs weekend ridership for a selected mode,
    plus the weekend/weekday ratio.
    """
    st.markdown("### Weekday vs Weekend Ridership (Single Mode)")
    
    if df_ridership.empty:
        st.info("No ridership data available.")
        return

    # choose mode
    mode = st.selectbox(
        "Select mode for weekday/weekend comparison",
        options=list(total_cols_by_type.keys()),
        index=0,
        format_func=lambda m: m.replace("_", " ").title()
    )
    col_name = total_cols_by_type[mode]

    df = df_ridership.copy()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    weekday_avg = df[~df["is_weekend"]][col_name].mean()
    weekend_avg = df[df["is_weekend"]][col_name].mean()
    ratio = weekend_avg / weekday_avg if weekday_avg > 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg Weekday Ridership", f"{weekday_avg:,.0f}")
    with c2:
        st.metric("Avg Weekend Ridership", f"{weekend_avg:,.0f}")
    with c3:
        st.metric("Weekend as % of Weekday", f"{ratio*100:,.1f}%")

    # bar chart
    df_bar = pd.DataFrame({
        "day_type": ["Weekday", "Weekend"],
        "ridership": [weekday_avg, weekend_avg],
    })

    fig = px.bar(
        df_bar,
        x="day_type",
        y="ridership",
        title=f"Average {mode.replace('_',' ').title()} Ridership: Weekday vs Weekend",
        labels={"ridership": "Average Daily Ridership", "day_type": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"- On average, **weekend {mode.replace('_',' ').title()} ridership is "
        f"{ratio*100:,.1f}% of weekday ridership.**"
    )


# --- 2. WEEKEND/WEEKDAY RATIO OVER TIME (YEARLY) ---

def render_weekend_weekday_ratio_over_time(df_ridership: pd.DataFrame) -> None:
    """
    Plot how the weekend/weekday ridership ratio changes by year for a selected mode.
    """
    st.markdown("### 2. Weekend/Weekday Ridership Ratio Over Time")
    
    if df_ridership.empty:
        st.info("No ridership data available.")
        return

    mode = st.selectbox(
        "Select mode for ratio over time",
        options=list(total_cols_by_type.keys()),
        index=0,
        format_func=lambda m: m.replace("_", " ").title(),
        key="ratio_mode_select",
    )
    col_name = total_cols_by_type[mode]

    df = df_ridership.copy()
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    grouped = (
        df.groupby(["year", "is_weekend"])[col_name]
        .mean()
        .reset_index()
    )

    weekday = grouped[~grouped["is_weekend"]].rename(columns={col_name: "weekday_avg"})[
        ["year", "weekday_avg"]
    ]
    weekend = grouped[grouped["is_weekend"]].rename(columns={col_name: "weekend_avg"})[
        ["year", "weekend_avg"]
    ]

    merged = pd.merge(weekday, weekend, on="year", how="inner")
    merged["weekend_weekday_ratio"] = merged["weekend_avg"] / merged["weekday_avg"]

    fig = px.line(
        merged,
        x="year",
        y="weekend_weekday_ratio",
        markers=True,
        labels={"weekend_weekday_ratio": "Weekend / Weekday Ridership"},
        title=f"Weekend-to-Weekday Ridership Ratio by Year ({mode.replace('_',' ').title()})",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not merged.empty:
        first_year = int(merged["year"].iloc[0])
        last_year = int(merged["year"].iloc[-1])
        first_ratio = merged["weekend_weekday_ratio"].iloc[0]
        last_ratio = merged["weekend_weekday_ratio"].iloc[-1]
        change = (last_ratio - first_ratio) * 100

        direction = "higher" if change > 0 else "lower"
        st.markdown(
            f"- The weekend/weekday ratio for **{mode.replace('_',' ').title()}** is "
            f"**{abs(change):.1f} percentage points {direction}** in {last_year} "
            f"compared to {first_year}."
        )


# --- 3. MODE-SPECIFIC WEEKDAY vs WEEKEND COMPARISON ---

def render_mode_weekday_weekend_comparison(df_ridership: pd.DataFrame) -> None:
    """
    Compare weekday vs weekend ridership across all modes in one grouped bar chart.
    """
    st.markdown("### 3. Weekday vs Weekend Comparison Across Modes")
    
    if df_ridership.empty:
        st.info("No ridership data available.")
        return

    df = df_ridership.copy()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    records = []
    for mode, col_name in total_cols_by_type.items():
        weekday_avg = df[~df["is_weekend"]][col_name].mean()
        weekend_avg = df[df["is_weekend"]][col_name].mean()
        records.append(
            {"mode": mode, "day_type": "Weekday", "avg_ridership": weekday_avg}
        )
        records.append(
            {"mode": mode, "day_type": "Weekend", "avg_ridership": weekend_avg}
        )

    df_modes = pd.DataFrame(records)
    df_modes["mode_label"] = df_modes["mode"].str.replace("_", " ").str.title()

    fig = px.bar(
        df_modes,
        x="mode_label",
        y="avg_ridership",
        color="day_type",
        barmode="group",
        labels={
            "mode_label": "Mode",
            "avg_ridership": "Average Daily Ridership",
            "day_type": "",
        },
        title="Average Weekday vs Weekend Ridership by Mode",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- Modes with **small differences** between weekday and weekend (e.g., buses / Access-A-Ride) "
        "tend to serve essential or non-commute trips.\n"
        "- Modes with **large drops** on weekends (e.g., commuter rail) are more commute-oriented."
    )


def render_system_recovery(df: pd.DataFrame) -> None:
    st.markdown("## System-wide Recovery by Mode")


    df = df.copy()
    df["year"] = df["date"].dt.year
    
    # exlude incomplete 2025 data
    df = df[df["year"] != 2025]

    # yearly avg % of pre-pandemic by mode
    pct_cols = list(pct_cols_by_type.values())
    yearly_pct = (
        df[["year"] + pct_cols]
        .groupby("year", as_index=False)
        .mean()
    )

    yearly_long = yearly_pct.melt(
        id_vars="year",
        value_vars=pct_cols,
        var_name="mode_col",
        value_name="avg_pct_prepandemic",
    )
    # clean labels
    col2mode = {v: k for k, v in pct_cols_by_type.items()}
    yearly_long["mode"] = yearly_long["mode_col"].map(col2mode)
    yearly_long["mode_label"] = yearly_long["mode"].str.replace("_", " ").str.title()

    fig = px.line(
        yearly_long,
        x="year",
        y="avg_pct_prepandemic",
        color="mode_label",
        markers=True,
        labels={
            "year": "Year",
            "avg_pct_prepandemic": "Avg % of Pre-Pandemic Day",
            "mode_label": "Mode",
        },
        title="Average % of Pre-Pandemic Ridership by Mode and Year",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Quick Takeaways")
    lines = []
    for mode, col in pct_cols_by_type.items():
        series = yearly_pct.set_index("year")[col].dropna()
        if series.empty:
            continue
        start_y, end_y = series.index[0], series.index[-1]
        change = series.iloc[-1] - series.iloc[0]
        direction = "higher" if change > 0 else "lower"
        lines.append(
            f"- **{mode.replace('_',' ').title()}** is about **{abs(change):.1f} percentage points {direction}** in {end_y} compared to {start_y}."
        )
    st.markdown("\n".join(lines))
    
def render_weekday_weekend_recovery(df: pd.DataFrame) -> None:
    st.markdown("## 2. Weekday vs Weekend Recovery by Mode")

    df = df.copy()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["day_type"] = np.where(df["is_weekend"], "Weekend", "Weekday")

    records = []
    for mode, col in pct_cols_by_type.items():
        for day_type in ["Weekday", "Weekend"]:
            m = (
                df[df["day_type"] == day_type][col]
                .mean()
            )
            records.append(
                {
                    "mode": mode,
                    "mode_label": mode.replace("_", " ").title(),
                    "day_type": day_type,
                    "avg_pct_prepandemic": m,
                }
            )

    summary = pd.DataFrame(records)

    fig = px.bar(
        summary,
        x="mode_label",
        y="avg_pct_prepandemic",
        color="day_type",
        barmode="group",
        labels={
            "mode_label": "Mode",
            "avg_pct_prepandemic": "Avg % of Pre-Pandemic Day",
            "day_type": "",
        },
        title="Average Recovery Level: Weekday vs Weekend by Mode",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- Modes with **higher weekend recovery** (relative to weekdays) indicate stronger leisure/tourism patterns.\n"
        "- Modes with **similar weekday/weekend levels** likely serve essential or non-commute trips."
    )
    
def render_mode_correlations(df: pd.DataFrame) -> None:
    st.markdown("## 3. Cross-Mode Correlation (Elasticity Proxy)")

    df = df.copy()
    corr_df = df[list(total_cols_by_type.values())].corr()

    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
        title="Correlation Between Daily Ridership by Mode",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- Strong **positive correlations** suggest modes respond to the same shocks (weather, holidays, system-wide disruptions).\n"
        "- Weaker correlations indicate modes serving different rider populations or trip purposes."
    )
    
