# MTA Analytics

Interactive Streamlit dashboard for exploring **MTA ridership and safety patterns** using open data beginning in 2020.

The app focuses on:

- How different MTA modes (Subway, Bus, LIRR, Metro-North, Access-A-Ride, SIRR) have recovered since COVID
- How **felony crime** in the system has evolved over time, normalized by ridership
- Simple ML-driven insights like anomaly detection and clustering of daily patterns

> Built with Python + Streamlit as a demo of real-world transit analytics.

Deployed at: [here](https://mta-dashboard.streamlit.app/)

---

## What this app does

### 1. Ridership Analysis

Uses the **MTA Daily Ridership: Beginning 2020** dataset to analyze:

- **Daily & yearly ridership trends** by mode  
- **% of pre-pandemic levels** over time  
- **Proportional mode share** (e.g. subway vs bus vs rail)  
- **Weekday vs weekend behavior** and hybrid work patterns  
- **Seasonality** (which months are busiest, and how that affects risk)  
- Simple **ML-based anomaly detection** (e.g. COVID waves)  
- Clustering of **daily ridership profiles** into “regimes” (pandemic lows, normal weekdays, weekends/holidays, etc.)

### 2. Crime & Safety Analysis

Uses monthly crime data with:

- `Month`, `Agency` (NYCT, LIRR, MNR, SIR), `Police Force`, `Felony Type`, `Felony Count`, `Crimes per 100k riders`

Key analyses include:

- **Crimes per 100k riders over time** → risk per rider, not just raw incidents  
- Comparison of **crime levels by agency**, normalized by ridership  
- Changes in **felony type mix** (e.g. assault vs grand larceny)  
  - Early pandemic vs recovery period, with **proportion z-tests** to check if changes are statistically significant  
- **Seasonal crime patterns** (by month) in both raw counts and normalized form  
- Linking **crime vs ridership** (e.g. more riders --> more incidents, but often *lower* risk per rider)

### 3. (Optional) Real-time GTFS

The repo also contains a `mta_realtime/` module for working with the **MTA GTFS-Realtime API** to:

- Pull live trip updates
- Inspect active trains by route/line
- Experiment with on-time performance metrics

The main focus of the current app is the open-data, historical analytics side, but the GTFS code is kept around for future expansion.

---

## Project structure

High-level layout:

```bash
mta-analytics/
├── app.py                # Main Streamlit entry point
├── pages/                # Streamlit multi-page views (ridership, crime, etc.)
├── mta_realtime/         # GTFS-Realtime utilities (legacy / optional)
├── data/                 # CSV data files (ridership + crime)
├── assets/               # Logos, images, etc.
├── requirements.txt      # Python dependencies
├── ENVIRONMENT.md        # Environment / setup notes
└── resoures.md           # Extra links / references (typo kept intentionally)
```

---

## Getting started

1. Clone the repo
```bash
git clone https://github.com/paolacalle/mta-analytics.git
cd mta-analytics
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run app
```bash 
streamlit run app.py
```