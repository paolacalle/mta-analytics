# pages/1_Live_Arrivals.py
import streamlit as st

from mta_realtime.filters import p1_sidebar_controls
from mta_realtime.layout import (
    render_top_section, 
    render_stop_direction_filters, 
    render_follow_train_id_route
)

st.set_page_config(page_title="Live Arrivals", page_icon="", layout="wide")

st.title("Live Subway Arrivals (MTA GTFS-Realtime)")

st.markdown(
    """
This page pulls **live train predictions** from the MTA GTFS-Realtime API using the
[`nyct-gtfs`](https://pypi.org/project/nyct-gtfs/) library.

Use the sidebar to select a **feed group**, **routes**, and **time window**
to inspect upcoming trains and their stops.
"""
)

# Sidebar: feed, routes, time window → returns filtered dataframe
with st.sidebar:
    feed_key, df_filtered, lower_min, upper_min = p1_sidebar_controls()

if df_filtered.empty:
    st.warning("No arrivals to display for the selected filters.")
else:
    render_top_section(df_filtered, feed_key)
    render_stop_direction_filters(df_filtered)
    render_follow_train_id_route(df_filtered)
