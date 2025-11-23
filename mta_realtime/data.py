# mta_realtime/data.py

from typing import Dict
import pandas as pd
from nyct_gtfs import NYCTFeed
from nyct_gtfs.gtfs_static_types import TripShapes

def load_trip_shapes(feed_key: str) -> Dict[str, pd.DataFrame]:
    """
    Load static trip shapes for NYC Subway from nyct-gtfs.
    Returns a dictionary mapping route_id to DataFrame of shape points.
    """
    feed = NYCTFeed(feed_key) 
    trips = feed.trips

    route_shapes: Dict[str, pd.DataFrame] = {}

    for trip in trips:
        route_id = trip.route_id
        shape = trip._tripe_shapes
        
        df_shape = pd.DataFrame(
            {
                "route_id": route_id,
                "shape_id": shape.shape_id,
                "shape_pt_lat": [pt.lat for pt in shape.shape_points],
                "shape_pt_lon": [pt.lon for pt in shape.shape_points],
                "shape_pt_sequence": [pt.sequence for pt in shape.shape_points],
            }
        ).sort_values("shape_pt_sequence")
        route_shapes[route_id] = df_shape

    return route_shapes


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
                    "has_delay" : train.has_delay_alert
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
                "departure_time",
                "has_delay"
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