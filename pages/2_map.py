# pages/2_map.py
import streamlit as st

from mta_realtime.filters import p2_sidebar_controls
from mta_realtime.layout import (
    render_top_section, 
    render_stop_direction_filters, 
    render_follow_train_id_route
)


st.set_page_config(page_title="Live Arrivals", page_icon="", layout="wide")
st.title("Live Subway Arrivals (MTA GTFS-Realtime)")

st.markdown(
    """
    Displays Live Subway Arrivals on a Map.
    """
)

with st.sidebar:
    p2_sidebar_controls()