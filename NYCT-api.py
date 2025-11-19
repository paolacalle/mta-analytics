import streamlit as st
import pandas as pd
from nyct_gtfs import NYCTFeed

@st.cache_data(ttl=30)  # re-fetch every 30 seconds
def load_realtime(feed_group: str) -> pd.DataFrame:
    feed = NYCTFeed(feed_group)
    rows = []
    for train in feed.trips:
        for stu in train.stop_time_updates:
            rows.append({
                "route_id": train.route_id,
                "direction": train.direction,
                "train_id": train.trip_id,
                "stop_id": stu.stop_id,
                "stop_name": stu.stop_name,
                "arrival_time": stu.arrival,
            })
    return pd.DataFrame(rows)

st.title("MTA Realtime Subway Dashboard")

feed_group = st.selectbox("Feed group", ["A", "B", "N", "1"])
df = load_realtime(feed_group)

route_filter = st.multiselect("Route", sorted(df["route_id"].unique()), default=None)
if route_filter:
    df = df[df["route_id"].isin(route_filter)]

st.dataframe(df.head(50))
