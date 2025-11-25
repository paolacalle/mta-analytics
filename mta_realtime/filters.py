# mta_realtime/filters.py

from typing import Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

from .config import FEED_GROUPS
from .data import (
    load_ridership, 
    load_realtime, 
    normalize_arrival_times, 
    normalize_departure_times,
    load_felonies 
)
 

@st.cache_data(ttl=30)
def _load_and_prepare_realtime(feed_key: str) -> pd.DataFrame:
    df = load_realtime(feed_key)
    if df.empty:
        return df
    df = normalize_arrival_times(df)
    df = normalize_departure_times(df)
    df = calculate_predicted_delays(df)
    return df

# @st.cache_data(ttl=30)
# def _load_and_prepare_trip_shapes() -> pd.DataFrame:
#     df = load_ridership()
#     return df

@st.cache_data(ttl=30)
def _load_and_prepare_ridership() -> pd.DataFrame:
    df = load_ridership()
    return df

# @st.cache_data(ttl=30)
# def _load_and_prepare_trip_shapes(feed_key: str) -> pd.DataFrame:
#     df = load_trip_shapes(feed_key)
#     return df

@st.cache_data(ttl=30)
def _load_and_prepare_felonies() -> pd.DataFrame:
    df = load_felonies()
    return df

def p1_sidebar_controls() -> Tuple[str, pd.DataFrame, int, int]:
    """
    Renders the sidebar:
      - feed group selector
      - route multi-select
      - time window slider (minutes from first arrival)

    Returns:
      feed_key, df_filtered, lower_min, upper_min
    """
    st.header("Feed & Filters")

    # Feed group selection
    feed_label = st.selectbox("Feed group", list(FEED_GROUPS.keys()))
    feed_key = FEED_GROUPS[feed_label]

    st.caption(
        "Each feed group bundles several lines together, "
        "matching how MTA serves GTFS-Realtime data."
    )

    with st.spinner("Fetching realtime data from MTA..."):
        df = _load_and_prepare_realtime(feed_key)

    if df.empty:
        st.warning("No realtime data returned for this feed at the moment.")
        return feed_key, df, 0, 0

    # Route filter
    routes = sorted(df["route_id"].unique())
    selected_routes = st.multiselect(
        "Filter by route",
        options=routes,
        default=routes,
    )

    if selected_routes:
        df = df[df["route_id"].isin(selected_routes)]

    # Time window slider
    st.markdown("---")
    st.subheader("Time window")

    min_min = float(df["diff_from_first_minutes"].min())
    max_min = float(df["diff_from_first_minutes"].max())

    st.caption(
        f"Data spans from **{min_min:.1f}** to **{max_min:.1f}** minutes "
        f"from the first arrival in this feed."
    )

    lower_min, upper_min = st.slider(
        "Select window (minutes from first arrival)",
        min_value=int(np.floor(min_min)),
        max_value=int(np.ceil(max_min)),
        value=(
            int(np.floor(min_min)),
            min(int(np.floor(min_min)) + 30, int(np.ceil(max_min))),
        ),
    )

    st.caption(
        f"Filtering between **{lower_min}** and **{upper_min}** minutes "
        f"from the first arrival."
    )

    # Apply time-window mask
    mask = (df["diff_from_first_minutes"] >= lower_min) & (
        df["diff_from_first_minutes"] <= upper_min
    )
    df_filtered = df[mask].copy()

    st.success(
        f"Window is {upper_min - lower_min} minutes: "
        f"showing {len(df_filtered)} arrivals in that range."
    )

    return feed_key, df_filtered, lower_min, upper_min

def filter_train_id_route(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters the dataframe for a specific train_id and route_id.
    """
    
    selected_route_id = st.selectbox("Select Route ID", options=df["route_id"].unique())
    filtered_df_by_route = df[df["route_id"] == selected_route_id]
    selected_train_id = st.selectbox("Select Train ID", options=filtered_df_by_route["train_id"].unique())
    
    df_filtered = df[
        (df["train_id"] == selected_train_id) &
        (df["route_id"] == selected_route_id)
    ].copy()
    
    return df_filtered, selected_train_id, selected_route_id
    
    
def calculate_predicted_delays(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates predicted delays for each stop based on arrival and departure times.
    Adds a 'predicted_delay_minutes' column to the dataframe.
    """
    df = df.copy()
    df["predicted_delay_minutes"] = (
        (df["arrival_time"] - df["departure_time"]).dt.total_seconds() / 60.0
    )
    return df
    
def p2_sidebar_controls() -> Tuple[str, pd.DataFrame, int, int]:
    """
    Renders the sidebar for Page 2:
      - feed group selector
      - time window slider (minutes from first arrival)
      - route multi-select
    Returns:
        feed_key, df_filtered, lower_min, upper_min
    """
    
    
    # filter by train_id and route_id
    st.header("Feed & Filters")
    feed_label = st.selectbox("Feed group", list(FEED_GROUPS.keys()))
    feed_key = FEED_GROUPS[feed_label]
    st.caption(
        "Each feed group bundles several lines together, "
        "matching how MTA serves GTFS-Realtime data."
    )
    # with st.spinner("Fetching realtime data from MTA..."):
    #     df = _load_and_prepare_trip_shapes(feed_key)
        
    # if df.empty:
    #     st.warning("No realtime data returned for this feed at the moment.")
    #     return feed_key, df, 0, 0
    
def p3_sidebar_controls() -> pd.DataFrame:
    """
    Renders the sidebar for Page 3:
      - ridership data loader
    Returns:
        df_ridership 
    """
    
    st.header("Ridership Data")
    with st.spinner("Loading ridership data..."):
        df_ridership = _load_and_prepare_ridership()
    if df_ridership.empty:
        st.warning("No ridership data available at the moment.")
    return df_ridership
    
    
def p4_sidebar_controls() -> pd.DataFrame:
    """
    Renders the sidebar for Page 4:
      - felonies data loader
    Returns:
        df_felonies 
    """ 
    
    st.header("Major Felonies Data")
    with st.spinner("Loading major felonies data..."):
        df_felonies = _load_and_prepare_felonies()
    if df_felonies.empty:
        st.warning("No major felonies data available at the moment.")
        
    df_ridership = _load_and_prepare_ridership()
    if df_ridership.empty:
        st.warning("No ridership data available at the moment.")
    return df_felonies, df_ridership

    
    
