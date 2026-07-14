# Part B — Neighborhood/Zipcode Similarity Explorer (Streamlit app)

# Imports
import streamlit as st
import plotly.express as px


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


# B2 — Interactive K-means in Streamlit
  # In your app, add a slider/control for K, and re-run K-means on the composition matrix live as the user changes it. Show a PCA scatter of areas colored by cluster.


# B3 — Geographic visualization
  # Plot each area as a point at its centroid lat/lon, colored by cluster, sized by business count. Should update live with the K selection.


# B4 — Cluster membership and interpretation
  # Show which specific neighborhoods/zipcodes fall into each cluster (a list or on-map highlight), and interpret:
    # Markdown/writeup required: Does the grouping seem meaningful given what you know (or can look up) about these areas? Any surprising groupings? Explain using the actual cluster membership, not just the general shape of the plot.
    # Optional/suggested: cluster profiling by dominant business type per area-cluster, as in A3, deepens this but is not required.


# B5 — Suggested reflection questions (optional, not required)
  # Do business-similar areas also tend to be geographically close, or does similarity cut across geography?
  # How does this compare to your Part A "Industry" clustering — consistent story, or different?
  # Pick one area you didn't expect to see grouped with another — what do they have in common?