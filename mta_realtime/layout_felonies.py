from typing import Dict, Tuple, List
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from math import log
from statsmodels.stats.proportion import proportions_ztest

from .layout_ridership import total_cols_by_type

def render_crime_eda(df_crime: pd.DataFrame) -> None:
    """
    Exploratory Data Analysis of Crime Data.
    """
    st.markdown("## Exploratory Data Analysis of Crime Data")

    if df_crime.empty:
        st.info("No crime data available.")
        return

    st.markdown("### Crime Type Distribution")
    crime_type_counts = df_crime.groupby('felony_type')['felony_count'].sum()\
        .reset_index(name='count').sort_values(by='count', ascending=False)

    fig = px.bar(
        crime_type_counts,
        x='felony_type',
        y='count',
        title='Distribution of Crime Types',
        labels={'felony_type': 'Felony Type', 'count': 'Number of Incidents'},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Crime Trends")
    monthly_trends = (
        df_crime.groupby('month')['felony_count']
        .sum()
        .reset_index(name='incident_count')
    )

    fig2 = px.line(
        monthly_trends,
        x='month',
        y='incident_count',
        title='Monthly Crime Incident Trends',
        labels={'month': 'Month', 'incident_count': 'Number of Incidents'},
    )
    st.plotly_chart(fig2, use_container_width=True)

def _render_crimes_per_million_explainer():
    st.markdown("### What does *Crimes per Million Riders* mean?")

    st.latex(r"""
    \text{Crimes per Million Riders} =
    \frac{\text{Felony Count}}{\text{Total Ridership}} \times 1{,}000{,}000
    """)

    st.markdown(
        """
        - This metric shows **how many crimes occur for every 1,000,000 riders**.
        - It **normalizes** crime by ridership, so we can fairly compare:
          - different **years** (e.g., low-ridership COVID years vs 2024), and  
          - different **agencies** (NYCT vs LIRR vs MNR vs SIR).
        - If ridership drops but crimes stay similar, **crimes per million will rise**
          even if the system isn’t actually more dangerous in total.
        - If ridership grows and crimes per million fall, it means the system is
          **safer per rider** even if the raw number of crimes increased.
        """
    )

def render_crime_rate_trend(df_crime: pd.DataFrame) -> None:
    """
    Crimes per Million Riders over time (overall + by agency).
    """
    st.markdown("## 1. Crimes per Million Riders Over Time")
    _render_crimes_per_million_explainer()

    if df_crime.empty:
        st.info("No crime data available.")
        return

    df = df_crime.copy().sort_values("month")

    # overall trend
    overall = (
        df.groupby("month")["crimes_per_100k_ridership"]
        .mean()
        .reset_index(name="crimes_per_million")
    )

    fig = px.line(
        overall,
        x="month",
        y="crimes_per_million",
        title="System-wide Crimes per Million Riders Over Time",
        labels={"month": "Month", "crimes_per_million": "Crimes per Million Riders"},
    )
    
    # highlight coronavirus period
    fig.add_vrect(
        x0="2020-03-01",
        x1="2021-06-01",
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
        annotation_text="COVID-19 Pandemic",
        annotation_position="top left",
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # by agency
    st.markdown("#### By Agency")
    st.expander("Explanation of Agencies").markdown(
        """
        Agencies refer to the part of the transit system responsible for the subway lines.
        Different agencies may have different safety records, so analyzing crime rates by agency
        can provide insights into which parts of the system are safer or more dangerous.
        
        Agencies included:
        - MTA New York City Transit (NYCT)
        - MTA Long Island Rail Road (LIRR)
        - MTA Metro-North Railroad (MNR)
        - Staten Island Railway (SIR)
        """
    )

    agencies = sorted(df["agency"].dropna().unique().tolist())
    selected_agencies = st.multiselect(
        "Select agencies to display",
        options=agencies,
        default=agencies,
    )

    if selected_agencies:
        df_agency = df[df["agency"].isin(selected_agencies)]
        agency_series = (
            df_agency.groupby(["month", "agency"])["crimes_per_100k_ridership"]
            .mean()
            .reset_index()
        )
        fig2 = px.line(
            agency_series,
            x="month",
            y="crimes_per_100k_ridership",
            color="agency",
            title="Crimes per Million Riders by Agency",
        )
        
        # highlight coronavirus period
        fig2.add_vrect(
            x0="2020-03-01",
            x1="2021-06-01",
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
            annotation_text="COVID-19 Pandemic",
            annotation_position="top left",
        )
    
        st.plotly_chart(fig2, use_container_width=True)
        
    

    st.markdown(
        "- This normalizes crime by ridership, showing how **risk per rider** evolves over time, "
        "rather than just raw incident counts."
    )


def render_crime_type_shift(
    df_crime: pd.DataFrame
    ) -> None:
    """
    2) Crime-type mix: early pandemic vs recovery.
    Splits at 2022-01 by default.
    """
    st.markdown("## 2. Crime Type Mix: Early Pandemic vs Recovery")

    if df_crime.empty:
        st.info("No crime data available.")
        return

    df = df_crime.copy()

    split_year = st.number_input(
        "Split year between early pandemic and recovery",
        min_value=2020,
        max_value=2030,
        value=2022,
        step=1,
    )

    df["period"] = df["month"].dt.year.apply(
        lambda y: "Early Pandemic" if y < split_year else "Recovery"
    )

    # total felony counts per type & period
    grouped = (
        df.groupby(["period", "felony_type"])["felony_count"]
        .sum()
        .reset_index()
    )

    # convert to shares within each period
    totals = grouped.groupby("period")["felony_count"].transform("sum")
    grouped["share"] = grouped["felony_count"] / totals

    fig = px.bar(
        grouped,
        x="felony_type",
        y="share",
        color="period",
        barmode="group",
        title=f"Felony Type Share: Early Pandemic vs {split_year}+ Recovery",
        labels={"share": "Share of Total Felonies", "felony_type": "Felony Type"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- This shows whether the **mix of crime types** shifted after the initial COVID shock.\n"
        "- For example, grand larceny vs assault proportions may change between periods."
    )

    # --- Statistical significance ---
    st.markdown("### Statistical Significance of Crime Type Shifts in Proportions")
    
    significance_results = []
    crime_types = grouped["felony_type"].unique()
    
    early_total = grouped[grouped["period"] == "Early Pandemic"]["felony_count"].sum()
    recovery_total = grouped[grouped["period"] == "Recovery"]["felony_count"].sum()
    
    for ctype in crime_types:
        early_count = grouped[
            (grouped["period"] == "Early Pandemic") & (grouped["felony_type"] == ctype)
        ]["felony_count"].values
        
        recovery_count = grouped[
            (grouped["period"] == "Recovery") & (grouped["felony_type"] == ctype)
        ]["felony_count"].values
        
        if len(early_count) == 0 or len(recovery_count) == 0:
            continue
        
        counts = np.array([early_count[0], recovery_count[0]])
        nobs = np.array([early_total, recovery_total])
        
        stat, pval = proportions_ztest(counts, nobs)
        significance_results.append((ctype, pval, early_count[0], recovery_count[0]))
    
    sig_df = pd.DataFrame(
        significance_results,
        columns=["Felony Type", "p-value", "Early Pandemic Count", "Recovery Count"]
    )
    sig_df["Significant Change"] = sig_df["p-value"] < 0.005
    st.dataframe(sig_df.style.format({"p-value": "{:.4f}"}))

    # --- Dropdown for detailed analysis of one felony type ---
    if not sig_df.empty:
        st.markdown("### Detailed View by Felony Type")
        selected_type = st.selectbox(
            "Choose a felony type to analyze",
            options=sig_df["Felony Type"].tolist(),
        )

        row = sig_df[sig_df["Felony Type"] == selected_type].iloc[0]

        # Get shares too
        early_row = grouped[
            (grouped["period"] == "Early Pandemic") &
            (grouped["felony_type"] == selected_type)
        ].iloc[0]
        recovery_row = grouped[
            (grouped["period"] == "Recovery") &
            (grouped["felony_type"] == selected_type)
        ].iloc[0]

        early_share = early_row["share"] * 100
        recovery_share = recovery_row["share"] * 100

        st.markdown(
            f"""
            **{selected_type}**

            - Early Pandemic: **{row['Early Pandemic Count']} incidents**
                ({early_share:.2f}% of all felonies)  
            - Recovery: **{row['Recovery Count']} incidents**
                ({recovery_share:.2f}% of all felonies)  
            - p-value: **{row['p-value']:.4f}**  
            - Statistically significant change (α = 0.005)? **{row['Significant Change']}**
            """
        )

    
def render_agency_safety_comparison(df_crime: pd.DataFrame) -> None:
    """
    3) Compare agencies by average crimes per million riders.
    """
    st.markdown("## 3. Agency Safety Comparison (Crimes per Million Riders)")

    if df_crime.empty:
        st.info("No crime data available.")
        return

    df = df_crime.copy()

    # average crime rate per agency over full period
    agency_rates = (
        df.groupby("agency")["crimes_per_100k_ridership"]
        .mean()
        .reset_index(name="avg_crimes_per_million")
        .sort_values("avg_crimes_per_million", ascending=False)
    )

    fig = px.bar(
        agency_rates,
        x="agency",
        y="avg_crimes_per_million",
        title="Average Crimes per Million Riders by Agency",
        labels={
            "agency": "Agency",
            "avg_crimes_per_million": "Avg Crimes per Million Riders",
        },
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- This comparison controls for ridership volume, answering **“Which agencies have higher or lower crime per rider?”**\n"
        "- Agencies with high raw crime counts can still appear relatively safe **once normalized by ridership**."
    )

def render_seasonal_crime_pattern(df_crime: pd.DataFrame) -> None:
    """
    4) Seasonal pattern of crime: average per-month crime rate.
    """
    st.markdown("## 4. Seasonal Crime Patterns")

    if df_crime.empty:
        st.info("No crime data available.")
        return

    df = df_crime.copy()
    df["month_num"] = df["month"].dt.month
    df["month_name"] = df["month"].dt.month_name()

    seasonal = (
        df.groupby("month_num")[["crimes_per_100k_ridership", "felony_count"]]
        .mean()
        .reset_index()
    )
    seasonal["month_name"] = seasonal["month_num"].apply(
        lambda m: pd.to_datetime(str(m), format="%m").strftime("%B")
    )

    fig1 = px.bar(
        seasonal,
        x="month_name",
        y="crimes_per_100k_ridership",
        title="Average Crimes per Million Riders by Month",
        labels={
            "month_name": "Month",
            "crimes_per_100k_ridership": "Crimes per Million Riders",
        },
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        seasonal,
        x="month_name",
        y="felony_count",
        title="Average Monthly Felony Count (Unnormalized)",
        labels={"month_name": "Month", "felony_count": "Average Felony Count"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "- The **normalized chart** shows risk per rider; the **raw felony count** chart shows absolute volume.\n"
        "- Peaks in summer or holiday months can reflect more riders, more tourism, or different policing patterns."
    )
    
    
    # clustering crime seasonality
    st.markdown("""
        ### Clustering Seasonal Crime Patterns
        We can use clustering to identify groups of agencies with similar seasonal crime trends.
        We use K-Means clustering on the per-agency monthly crime rates.
    """)
    
    monthly_matrix = (
        df.groupby(["month_num", "agency"])["crimes_per_100k_ridership"]
        .mean()
        .reset_index()
        .pivot(index="agency", columns="month_num", values="crimes_per_100k_ridership")
        .fillna(0)
    )
    
    scaler = StandardScaler()
    monthly_scaled = scaler.fit_transform(monthly_matrix)
    
    # select number of clusters
    n_clusters = st.slider(
        "Select number of clusters for seasonal patterns",
        min_value=2,
        max_value=4,
        value=2,
        step=1,
    )
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    monthly_matrix["cluster"] = kmeans.fit_predict(monthly_scaled)
    
    st.markdown("#### Clustered Seasonal Patterns by Agency")
    for cluster_id in sorted(monthly_matrix["cluster"].unique()):
        cluster_data = monthly_matrix[monthly_matrix["cluster"] == cluster_id].drop(columns=["cluster"])
        cluster_avg = cluster_data.mean().reset_index()
        cluster_avg.columns = ["month_num", "avg_crimes_per_million"]
        cluster_avg["month_name"] = cluster_avg["month_num"].apply(
            lambda m: pd.to_datetime(str(m), format="%m").strftime("%B")
        )
        
        fig = px.line(
            cluster_avg,
            x="month_name",
            y="avg_crimes_per_million",
            title=f"Cluster {cluster_id} Average Seasonal Crime Pattern",
            labels={
                "month_name": "Month",
                "avg_crimes_per_million": "Avg Crimes per Million Riders",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(
            f"- **Cluster {cluster_id}** includes agencies: "
            f"{', '.join(cluster_data.index.tolist())}.\n"
            "- This clustering reveals groups of agencies with similar seasonal crime trends."
        )
    
    
    


agency_to_mode_key = {
    "NYCT": "subway",          # NYCT = New York City Transit --> Subway (and sometimes Bus)
    "LIRR": "lirr",            # Long Island Rail Road
    "MNR": "metro_north",      # Metro-North Railroad
    "SIR": "sirr",             # Staten Island Railway
}

def render_crime_vs_ridership(
    df_crime: pd.DataFrame,
    df_ridership: pd.DataFrame
) -> None:
    """
    5) Crime vs ridership: do more riders mean more crime per rider, or less?
    Joins monthly crime with monthly ridership for matching agencies.
    """
    st.markdown("## 5. Crime vs Ridership Relationship")

    if df_crime.empty or df_ridership.empty:
        st.info("Need both crime and ridership data.")
        return

    # --- build monthly ridership per mode ---
    rid = df_ridership.copy()
    rid["month"] = rid["date"].dt.to_period("M").dt.to_timestamp()
    # sum daily ridership to monthly by mode
    rid_monthly = (
        rid.groupby("month")[list(total_cols_by_type.values())]
        .sum()
        .reset_index()
    )

    # --- prepare crime monthly per agency ---
    crime = df_crime.copy()
    crime["month"] = crime["month"].dt.to_period("M").dt.to_timestamp()

    agencies = sorted(crime["agency"].dropna().unique())
    selected_agency = st.selectbox(
        "Select agency to analyze",
        options=agencies,
    )

    if selected_agency not in agency_to_mode_key:
        st.warning(
            f"No mode mapping configured for agency '{selected_agency}'. "
            "Edit `agency_to_mode_key` to map this agency to a ridership mode."
        )
        return

    mode_key = agency_to_mode_key[selected_agency]
    ridership_col = total_cols_by_type[mode_key]

    crime_agency = (
        crime[crime["agency"] == selected_agency]
        .groupby("month")[["felony_count", "crimes_per_100k_ridership"]]
        .sum()
        .reset_index()
    )

    merged = pd.merge(crime_agency, rid_monthly[["month", ridership_col]],
                    on="month", how="inner")
    merged.rename(columns={ridership_col: "Monthly Ridership"}, inplace=True)

    if merged.empty:
        st.info("No overlapping months between crime and ridership for this agency.")
        return

    # scatter: crime rate vs ridership
    fig = px.scatter(
        merged,
        x="Monthly Ridership",
        y="crimes_per_100k_ridership",
        trendline="ols",
        title=f"Crimes per Million Riders vs Monthly Ridership – {selected_agency}",
        labels={
            "Monthly Ridership": "Monthly Ridership",
            "crimes_per_100k_ridership": "Crimes per Million Riders",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # correlation numbers
    corr_raw = merged["felony_count"].corr(merged["Monthly Ridership"])
    corr_rate = merged["crimes_per_100k_ridership"].corr(merged["Monthly Ridership"])

    st.markdown(
        f"""
        **Correlation summary for {selected_agency}:**
        - Raw felony count vs ridership: **{corr_raw:.2f}**  
        - Crimes per million riders vs ridership: **{corr_rate:.2f}**
        
        Typically you'll see:
        - A **strong positive** correlation for raw felonies (more riders --> more incidents overall).  
        - A **weak or negative** correlation for crimes per million riders (more riders --> *lower* risk per rider),
          meaning the system can actually become **safer per person** as it gets busier.
        """
    )

