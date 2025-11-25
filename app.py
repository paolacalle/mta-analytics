# app.py
import streamlit as st
from PIL import Image 

MTA_LOGO = Image.open("assets/mta_logo.png")
st.set_page_config(
    page_title="MTA Ridership & Safety Analytics",
    page_icon=MTA_LOGO,
    layout="wide",
)

st.title("MTA Ridership & Safety Analytics Dashboard")

st.markdown("""
This app explores **MTA ridership and safety patterns** using open data from the MTA
(open data portal) starting in 2020.

Use the pages in the sidebar to:

- **Ridership Analysis** – explore daily and yearly ridership trends by mode  
  (Subway, Bus, LIRR, Metro-North, Access-A-Ride, SIRR), recovery versus pre-pandemic
  levels, weekday vs weekend behavior, proportional mode shares, and ML-based anomaly
  detection and clustering.

- **Crime & Safety Analysis** – analyze felony counts and **crimes per 100k riders**
  by agency (NYCT, LIRR, MNR, SIR), track changes in crime type mix between early
  pandemic and recovery, study seasonal crime patterns, and link crime risk to
  ridership levels.
  
- **Real-time Train Delays** – visualize real-time train arrival data (MTA GTFS-realtime
  feeds), predicted vs actual arrival times.

---

### Data sources
- **MTA Daily Ridership: Beginning 2020** – estimates of ridership and `%` of comparable
  pre-pandemic day by mode.
- **MTA Transit Crime data** – monthly felony counts and crimes per 100k riders
  by agency and felony type.

### Tech stack
- **Python**, **Streamlit**
- **Pandas**, **NumPy** for data processing
- **Plotly** for interactive visualizations
- Hosted on **Streamlit Community Cloud**
""")
