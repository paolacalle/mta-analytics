import streamlit as st
import pandas as pds
import tabs.tab1 as t1


st.title("MTA Dashboard")

tab1, tab2, tab3 = st.tabs(["Live Arrivals", "Line Performance", "Station Info"])

with tab1:
    t1.tab1()

with tab2:
    st.subheader("On-Time Performance")
    st.write("Plot your KPIs here...")

with tab3:
    st.subheader("Station Map / Details")
    st.write("Maps, stop listings, etc...")
