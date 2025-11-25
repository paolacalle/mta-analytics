import streamlit as st

from mta_realtime.filters import p4_sidebar_controls
from mta_realtime.layout_felonies import (
    render_crime_eda,
    render_crime_rate_trend,
    render_crime_type_shift,
    render_agency_safety_comparison,
    render_seasonal_crime_pattern,
    render_crime_vs_ridership
)


st.title("Major Felony Incidents Near Subway Stations")

st.markdown(
    """
    This page presents an analysis of major felony incidents occurring near NYC subway stations.
    The data is sourced from the NYPD's Major Felony Incident reports and is visualized to show
    trends over time and across different subway lines.
    """
)

with st.sidebar:
    df_felonies, df_ridership = p4_sidebar_controls()
    
if df_felonies.empty:
    st.warning("No felony incident data to display.")
else:
    render_crime_eda(df_felonies)
    render_crime_rate_trend(df_felonies)
    render_crime_type_shift(df_felonies)
    render_agency_safety_comparison(df_felonies)
    render_seasonal_crime_pattern(df_felonies)
    render_crime_vs_ridership(df_felonies, df_ridership)