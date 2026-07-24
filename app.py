# Part B - Neighborhood/Zipcode Similarity Explorer (Streamlit app)

# Imports
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from data_prep import (
    load_raw,
    clean_businesses,
    area_composition_matrix,
    MIN_AREA_BUSINESS_COUNT,
    EXCLUDED_AREAS,
)

st.set_page_config(
    page_title="Vancouver Business Licences Neighbourhood Similarity Explorer",
    layout="wide",
)

META_COLS = ["business_count", "centroid_lat", "centroid_lon"]


# B1 — Choose your granularity and engineer an area-level feature vector
  # The key shift here: you're moving from row = business (Part A) to row = area. That means you need to engineer a new feature vector for each area, built up from the businesses inside it.

  # Step 1 — pick your unit of analysis. Choose one:
    # Postal FSA (first 3 characters of postalcode) — more zipcode-like, ~54% coverage
    # localarea (25 named neighborhoods) — cleaner, near-complete coverage

  # Step 2 — engineer the features. For each area, turn "what business types are present here" into a numeric feature vector: one feature (column) per business type, where the value is the percentage of that area's businesses belonging to that type. This is essentially a frequency-encoding of businesstype, aggregated up to the area level and row-normalized so an area with 50 businesses and one with 5,000 are still comparable on the same 0–100% scale.

  # Step 3 — clean up thin areas. Areas with very few businesses produce noisy percentages (one business can swing a whole feature by a large margin). Consider a minimum business-count threshold before including an area, and justify your cutoff.

  # Hints:
    # pd.crosstab(df["area_column"], df["businesstype"], normalize="index") * 100 does steps 1–2 in a single line — it directly gives you a row-normalized area × business-type matrix. pd.pivot_table with aggfunc="count" followed by dividing each row by its sum works too if you want more control.
    # The result of that crosstab is your feature matrix for clustering — each row is one "sample" (an area), each column is one "feature" (a business type's share of that area).
    # To apply the count threshold from Step 3, compute df["area_column"].value_counts() first, keep only areas above your cutoff, then build the crosstab from the filtered data (or build the crosstab first and drop rows where the row's original business count is too low — either order works).
    # You do not need to scale this feature matrix the way you scaled Size/Industry/Lifecycle in A3 — the values are already on a comparable 0–100% scale, but it's fine to standardize anyway if you prefer being consistent with your A3 pipeline.

@st.cache_data(show_spinner="Loading and cleaning business licence data...")
def get_composition_matrix():
    gdf = load_raw()
    df = clean_businesses(gdf)
    return area_composition_matrix(df)


comp = get_composition_matrix()
feature_cols = [c for c in comp.columns if c not in META_COLS]

st.title("Vancouver Business Licences Neighbourhood Similarity Explorer")

# B2 — Interactive K-means in Streamlit
  # In your app, add a slider/control for K, and re-run K-means on the composition matrix live as the user changes it. Show a PCA scatter of areas colored by cluster.

#BEGIN
st.sidebar.header("Clustering controls")
max_k = min(10, len(comp) - 1)
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=4)

X = comp[feature_cols].values
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

result = comp.copy()
result["cluster"] = labels.astype(str)

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X)
result["pca1"], result["pca2"] = pca_coords[:, 0], pca_coords[:, 1]
result = result.reset_index()
area_col = result.columns[0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("PCA projection (by business-type mix)")
    fig_pca = px.scatter(
        result,
        x="pca1",
        y="pca2",
        color="cluster",
        size="business_count",
        text=area_col,
        hover_name=area_col,
        labels={"pca1": "PC1", "pca2": "PC2"},
    )
    fig_pca.update_traces(textposition="top center")
    st.plotly_chart(fig_pca, width="stretch")
#END

with col2:
# B3 — Geographic visualization
  # Plot each area as a point at its centroid lat/lon, colored by cluster, sized by business count. Should update live with the K selection.

    st.subheader("Geographic view (area centroids)")
    fig_map = px.scatter_map(
        result,
        lat="centroid_lat",
        lon="centroid_lon",
        color="cluster",
        size="business_count",
        hover_name=area_col,
        zoom=10.3,
        height=480,
        map_style="carto-positron",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, width="stretch")

# B4 — Cluster membership and interpretation
  # Show which specific neighborhoods/zipcodes fall into each cluster (a list or on-map highlight), and interpret:
    # Markdown/writeup required: Does the grouping seem meaningful given what you know (or can look up) about these areas? Any surprising groupings? Explain using the actual cluster membership, not just the general shape of the plot.
    # Optional/suggested: cluster profiling by dominant business type per area-cluster, as in A3, deepens this but is not required.

#BEGIN
st.subheader("Cluster membership")
membership_cols = st.columns(k)
for c, col in zip(sorted(result["cluster"].unique(), key=int), membership_cols):
    areas = result[result["cluster"] == c].sort_values("business_count", ascending=False)
    with col:
        st.markdown(f"**Cluster {c}** ({len(areas)} areas)")
        st.markdown("\n".join(f"- {a} ({n:,})" for a, n in zip(areas[area_col], areas["business_count"])))

st.subheader("Dominant business types per cluster")
profile = result.groupby("cluster")[feature_cols].mean()
for c in profile.index:
    top = profile.loc[c].sort_values(ascending=False).head(3)
    st.markdown(f"**Cluster {c}:** " + ", ".join(f"{t} ({v:.1f}%)" for t, v in top.items()))
#END

st.subheader("Interpretation")
st.markdown(
    """
**At K=4, this grouping is genuinely meaningful, and it does cut across
geography.** One cluster of 14 areas spans opposite ends of the city:
**Kerrisdale, Dunbar-Southlands, West Point Grey, Arbutus-Ridge** on the
west side sit in the *same* cluster as **Renfrew-Collingwood,
Killarney, Victoria-Fraserview, Hastings-Sunrise** on the east side, plus
Kensington-Cedar Cottage, Riley Park, Oakridge, Shaughnessy, South Cambie,
and Kitsilano. Every one in those areas is dominated by long term rental
licences, because they are all primarily residential neighbourhoods rather
than commercial districts. K-means is correctly picking up residential
character neighbourhood as a real pattern that has nothing to do with
which side of the city you are on.

**Downtown, Marpole, Mount Pleasant, Sunset, Grandview-Woodland,
West End** form a mixed commercial/dense cluster, while **Fairview**
and **Strathcona** are distinctive enough to stand alone as
singleton clusters at this K value.

This lines up with the same conclusion from Part A: the location based
K-means/DBSCAN clusters and the Size/Industry/Lifecycle feature
clusters on individual businesses had essentially zero agreement. The
neighbourhood level result geography and what kind of business mix
an area has are close to independent signals in this dataset.
"""
)

# B5 — Suggested reflection questions (optional, not required)
  # Do business-similar areas also tend to be geographically close, or does similarity cut across geography?
  # How does this compare to your Part A "Industry" clustering — consistent story, or different?
  # Pick one area you didn't expect to see grouped with another — what do they have in common?

with st.expander("B5: Reflection Questions"):
    st.markdown(
        """
- **Do areas of similar businesses also tend to be geographically close?** No,
    because at K=4 the largest cluster contains west side areas and east
    side areas together, driven by a shared long term rental dominated
    business mix rather than proximity.
- **How does this compare to Part A's Industry clustering?** The same
    near zero Adjusted Rand Index between geographic and feature based
    clusters shows up at both the individual business level and here at
    the neighbourhood level.
- **One unexpected pairing:** Dunbar-Southlands and Renfrew-Collingwood
    land in the same cluster at K=4 despite being on opposite sides of
    the city and having very different reputations. What they share is a
    business landscape that is overwhelmingly long term rental plus
    general contractor. For example, both are primarily residential
    areas with light home services activity rather than commercial hubs.
"""
    )