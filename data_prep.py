"""
Shared cleaning/feature engineering pipeline used by both the Part A notebook
and the Part B Streamlit app so the two stay consistent.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

#BEGIN
DATA_PATH = "business-licences.geojson"

STATUS_KEEP = "Issued"
TOP_N_BUSINESS_TYPES = 20
MIN_AREA_BUSINESS_COUNT = 300
EXCLUDED_AREAS = ["Out of Town"]


def load_raw(path=DATA_PATH):
    """Read the GeoJSON export."""
    return gpd.read_file(path)


def clean_businesses(gdf, top_n=TOP_N_BUSINESS_TYPES):
    # Cleaning 1: keep only Issued licences (drop pending/cancelled/inactive)
    df = gdf[gdf["status"] == STATUS_KEEP].copy()

    # Cleaning 3: flag rows with usable coordinates instead of dropping them
    df["lon"] = df.geometry.x
    df["lat"] = df.geometry.y
    df["has_geo"] = df.geometry.notna()

    # Cleaning 6: group missing feepaid with the median fee for that business type
    # feepaid: among Issued rows. Group with the median fee for that business type,
    # falling back to the overall median for any type with no observed fee at all.
    df["feepaid"] = df.groupby("businesstype")["feepaid"].transform(
        lambda s: s.fillna(s.median())
    )
    df["feepaid"] = df["feepaid"].fillna(df["feepaid"].median())

    # Cleaning 7: consolidate the businesstype categories into the top N
    # plus "Other" for the long tail.
    top_types = df["businesstype"].value_counts().nlargest(top_n).index
    df["businesstype_grouped"] = np.where(
        df["businesstype"].isin(top_types), df["businesstype"], "Other"
    )

    df["issueddate"] = pd.to_datetime(df["issueddate"], utc=True).dt.tz_localize(None)
    df["expireddate"] = pd.to_datetime(df["expireddate"])

    return df


def geo_subset(df, n_sample=20000, random_state=42):
    """Rows with usable coordinates, downsampled for fast/plottable clustering."""
    geo_df = df[df["has_geo"]].copy()
    if n_sample is not None and len(geo_df) > n_sample:
        geo_df = geo_df.sample(n_sample, random_state=random_state)
    return geo_df.reset_index(drop=True)


def area_composition_matrix(
    df,
    # Cleaning 5: use localarea
    area_col="localarea",
    min_count=MIN_AREA_BUSINESS_COUNT,
    excluded_areas=EXCLUDED_AREAS,
):
    
    # Return a DataFrame with the composition of business types by area.
    d = df[df[area_col].notna()].copy()
    if excluded_areas:
        d = d[~d[area_col].isin(excluded_areas)]

    counts = d[area_col].value_counts()
    keep_areas = counts[counts >= min_count].index
    d = d[d[area_col].isin(keep_areas)]

    comp = pd.crosstab(d[area_col], d["businesstype_grouped"], normalize="index") * 100

    centroids = (
        d[d["has_geo"]]
        .groupby(area_col)[["lat", "lon"]]
        .mean()
        .reindex(comp.index)
    )
    comp["business_count"] = counts.reindex(comp.index)
    comp["centroid_lat"] = centroids["lat"]
    comp["centroid_lon"] = centroids["lon"]

    return comp
#END
