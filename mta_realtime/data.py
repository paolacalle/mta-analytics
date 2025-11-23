# mta_realtime/data.py

from typing import Dict
import pandas as pd
from nyct_gtfs import NYCTFeed


def load_realtime(feed_key: str) -> pd.DataFrame:
    """
    Pull realtime trip updates for a given feed group code (e.g., 'A', '1', 'N').
    Returns a DataFrame with one row per (train, stop) prediction.
    """
    feed = NYCTFeed(feed_key)
    rows = []

    for train in feed.trips:
        for stu in train.stop_time_updates:
            rows.append(
                {
                    "route_id": train.route_id,
                    "direction": train.direction,
                    "train_id": train.trip_id,
                    "stop_id": stu.stop_id,
                    "stop_name": stu.stop_name,
                    "arrival_time": stu.arrival,  # datetime
                    "status": train.location_status,
                    "departure_time": stu.departure,  # datetime
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "route_id",
                "direction",
                "train_id",
                "stop_id",
                "stop_name",
                "arrival_time",
                "status",
            ]
        )

    df = pd.DataFrame(rows).sort_values(["route_id", "arrival_time"])
    return df


def normalize_arrival_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure arrival_time is timezone-aware and converted to America/New_York.
    Adds a 'diff_from_first_minutes' column (float minutes from earliest arrival).
    """
    df = df.copy()

    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    df["arrival_time"] = df["arrival_time"].dt.tz_convert("America/New_York")

    first_arrival = df["arrival_time"].min()
    df["diff_from_first_minutes"] = (
        df["arrival_time"] - first_arrival
    ).dt.total_seconds() / 60.0

    return df

def normalize_departure_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure departure_time is timezone-aware and converted to America/New_York.
    """
    df = df.copy()

    df["departure_time"] = pd.to_datetime(df["departure_time"], utc=True)
    df["departure_time"] = df["departure_time"].dt.tz_convert("America/New_York")

    return df