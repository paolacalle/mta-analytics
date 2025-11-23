# mta_realtime/layout.py

from typing import Tuple, List
import pandas as pd
import plotly.express as px
import streamlit as st
from mta_realtime.filters import filter_train_id_route

def render_top_section(df_filtered: pd.DataFrame, feed_key: str) -> None:
    """
    Renders:
      - upcoming arrivals table
      - bar chart of active trains by route & direction
    """
    st.markdown("### Upcoming Arrivals and Active Trains")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"Upcoming Arrivals for Feed Group {feed_key}")
        st.dataframe(
            df_filtered[
                [
                    "route_id",
                    "direction",
                    "train_id",
                    "stop_name",
                    "arrival_time",
                    "diff_from_first_minutes",
                    "status",
                ]
            ]
            .sort_values("arrival_time")
            .reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    with col2:
        st.subheader("Trains by Route and Direction")

        if df_filtered.empty:
            st.info("No arrivals in the selected time window.")
            return

        tmp = df_filtered.copy()
        tmp["route_id"] = tmp["route_id"].astype("category")
        counts = (
            tmp.groupby(["route_id", "direction"])["train_id"]
            .nunique()
            .reset_index(name="active_trains")
        )

        if counts.empty:
            st.info("No trains found for selected filters.")
            return

        fig = px.bar(
            counts,
            x="route_id",
            y="active_trains",
            color="direction",
            barmode="group",
            labels={
                "route_id": "Route",
                "active_trains": "Number of Active Trains",
                "direction": "Direction",
            },
            title="Active Trains per Route and Direction (within selected window)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_stop_direction_filters(df_filtered: pd.DataFrame) -> None:
    """
    Renders:
      - stop multi-select
      - direction multi-select
      - status distribution pie chart
    """
    st.markdown("---")
    st.markdown("### Stop & Direction Filters")

    df_view = df_filtered.copy()

    # Stop filter
    stops = sorted(df_view["stop_name"].unique())
    selected_stops = st.multiselect(
        "Filter by stop",
        options=stops,
        default=[],
    )

    if selected_stops:
        df_view = df_view[df_view["stop_name"].isin(selected_stops)]
        st.success(f"Filtered to {len(df_view)} arrivals at selected stops.")
    else:
        st.info("No stops selected: showing all stops in the time window.")

    # Direction filter
    directions = sorted(df_view["direction"].unique())
    selected_directions = st.multiselect(
        "Filter by direction",
        options=directions,
        default=[],
    )

    if selected_directions:
        df_view = df_view[df_view["direction"].isin(selected_directions)]
        st.success(f"Filtered to {len(df_view)} arrivals in selected directions.")
    else:
        st.info("No directions selected: showing all directions in the time window.")

    # Status distribution
    st.subheader("Train Status Distribution")

    if df_view.empty:
        st.info("No trains to display for the current filters.")
        return

    status_counts = df_view["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    fig2 = px.pie(
        status_counts,
        names="status",
        values="count",
        title="Distribution of Train Statuses (filtered view)",
    )
    st.plotly_chart(fig2, use_container_width=True)
    
def render_follow_train_id_route(
    df_filtered: pd.DataFrame
) -> None:
    """
    Renders:
      - line chart of arrival times for a specific train_id and route_id
    """
    st.markdown("---")
    st.markdown("### Follow Specific Train ID and Route ID")
    
    with st.expander("What does this Train ID mean?"):
        st.markdown(
        """
        **Train ID** is an internal NYCT identifier that links the GTFS-RT trip to the
        rail operations system. It’s built from several pieces:

        - The **first character** is the *trip type*  
        - `0` = normal scheduled revenue trip  
        - other symbols indicate things like reroutes, skip-stop service, or short-turn trains.
        - The **second character** is the **line** (for example `6` for the 6 train).
        - The **next characters** encode the **origin time** of the trip.  
        The last character in this block may indicate whether the train leaves on the
        exact minute or about 30 seconds after.
        - After that comes a three-letter **origin location** code and a three-letter
        **destination location** code.

        So a Train ID like `06 0123+ PEL/BBR` can be read as:  
        trip type `0`, line `6`, origin time around 01:23, from PEL to BBR.
        
        For a daily commuter, the Train ID can help identify your specific train
        even if the route has multiple trains departing around the same time.
        
        For more than day-to-day use of Train IDs, see the 
        [NYCT-GTFS documentation](https://github.com/TransitApp/nyct-gtfs-realtime-validator/wiki/Train-IDs).
        """
        )
    
    df_train, train_id, route_id = filter_train_id_route(df_filtered)
    st.info(f"Showing data for Train ID: {train_id} on Route ID: {route_id}")
    
    if df_train.empty:
        st.info("No data available for the selected Train ID and Route ID.")
        return
    
    fig = px.line(
        df_train,
        x="arrival_time",
        y="stop_name",
        title=f"Arrival Times for Train ID {train_id} on Route {route_id}",
        labels={
            "arrival_time": "Arrival Time",
            "stop_name": "Stop Name",
        }
    )
    
    # add departure times as markers
    fig.add_scatter(
        x=df_train["departure_time"],
        y=df_train["stop_name"],
        mode="markers",
        name="Departure Time",
        marker=dict(symbol="circle-open", size=10, color="red"),
    )
    
    # add arrival times as markers
    fig.add_scatter(
        x=df_train["arrival_time"],
        y=df_train["stop_name"],
        mode="markers",
        name="Arrival Time",
        marker=dict(symbol="circle", size=10, color="blue"),
    )
    
    st.plotly_chart(fig, use_container_width=True)

