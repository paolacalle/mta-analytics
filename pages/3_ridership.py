# pages/1_Live_Arrivals.py
import streamlit as st
# from PIL import Image 
from mta_realtime.filters import p3_sidebar_controls
from mta_realtime.layout_ridership import (
    render_eda_table,
    render_mode_selection,
    render_ridership_over_daily_time,
    render_ridership_yearly_time,
    render_proportional_difference_yearly,
    render_proportional_difference_daily,
    render_weekday_weekend_summary,
    render_weekend_weekday_ratio_over_time,
    render_mode_weekday_weekend_comparison,
    render_system_recovery,
    render_weekday_weekend_recovery,
    render_mode_correlations
)

# MTA_LOGO = Image.open("../assets/mta_logo.png")
st.set_page_config(page_title="Ridership", page_icon="", layout="wide")

st.title("Subway Ridership Over Time")

st.markdown(
    """
"""
)

with st.sidebar:
    df_ridership = p3_sidebar_controls()
    
if df_ridership.empty:
    st.warning("No ridership data to display.")
else:
    render_eda_table(df_ridership)
    selected_modes, col_names = render_mode_selection()
    render_ridership_over_daily_time(df_ridership, selected_modes, col_names)
    render_ridership_yearly_time(df_ridership, selected_modes, col_names)
    
    render_proportional_difference_yearly(df_ridership, selected_modes, col_names)
    render_proportional_difference_daily(df_ridership, selected_modes, col_names)
    
    # render_weekday_weekend_summary(df_ridership)
    # render_weekend_weekday_ratio_over_time(df_ridership)
    # render_mode_weekday_weekend_comparison(df_ridership)
    
    render_system_recovery(df_ridership)
    render_weekday_weekend_recovery(df_ridership)
    render_mode_correlations(df_ridership)
    
    # render_anomaly_detection(df_ridership)
    
    
    