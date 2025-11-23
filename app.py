# app.py
import streamlit as st
from PIL import Image 

MTA_LOGO = Image.open("assets/mta_logo.png")
st.set_page_config(
    page_title="MTA Realtime Performance Dashboard",
    page_icon=MTA_LOGO,
    layout="wide",
)

st.title("MTA Realtime Performance Dashboard")
st.markdown("""
            
This app explores **NYC Subway realtime data** using the MTA GTFS-Realtime API.

Use the pages in the sidebar to:
- **Live Arrivals** – view current train predictions by feed group and route.
- **On-Time Performance** – analyze delays and on-time metrics (from processed data).
- **oStation Summary** – look at aggregated stats by station.

### Tech stack
- **Python**, **Streamlit**
- **nyct-gtfs** for GTFS-Realtime API access
- **Pandas**, **Plotly** for data processing and visualization
- Hosted on **Streamlit Cloud**
""")
