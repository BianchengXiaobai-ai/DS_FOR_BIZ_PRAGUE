# Data folder

Git does not track large CSV/GZ files. Put these files here locally:

| File | Source |
|------|--------|
| `airbnb_listings_features.csv.gz` | Part 1 output (required for modeling) |
| `listings.csv.gz` | Inside Airbnb Prague (optional fallback) |

If files are missing, `data_prepro_func.get_release_df()` tries the GitHub Release tag `data`.
