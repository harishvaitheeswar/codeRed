import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# === Sample CSV creation (run once) ===
crop_data = """latitude,longitude,pesticide_amount,date
12.9716,77.5946,5.0,2024-07-01
12.9750,77.5900,3.5,2024-07-02
12.9800,77.6000,4.2,2024-07-03
12.9850,77.6100,6.1,2024-07-04
"""

health_data = """latitude,longitude,health_issue_score
12.9740,77.5950,7
12.9790,77.6050,5
12.9830,77.6150,8
"""

with open('crop_spraying_data.csv', 'w') as f:
    f.write(crop_data)

with open('health_survey_data.csv', 'w') as f:
    f.write(health_data)

print("Sample CSV files created!")

# === Load datasets ===
crop_df = pd.read_csv('crop_spraying_data.csv')
health_df = pd.read_csv('health_survey_data.csv')

# Convert to GeoDataFrames
crop_gdf = gpd.GeoDataFrame(
    crop_df, geometry=gpd.points_from_xy(crop_df.longitude, crop_df.latitude))
crop_gdf.set_crs(epsg=4326, inplace=True)  # WGS84 lat/lon

health_gdf = gpd.GeoDataFrame(
    health_df, geometry=gpd.points_from_xy(health_df.longitude, health_df.latitude))
health_gdf.set_crs(epsg=4326, inplace=True)

# Project to metric CRS for distance calculations (use suitable UTM zone)
crop_gdf = crop_gdf.to_crs(epsg=32644)  # Replace with your zone if different
health_gdf = health_gdf.to_crs(epsg=32644)

# Buffer health points by 5km radius
health_gdf['buffer'] = health_gdf.geometry.buffer(5000)
health_buffers = health_gdf.set_geometry('buffer')

# Spatial join: crop spraying points within health buffers
joined = gpd.sjoin(crop_gdf, health_buffers, how='inner', predicate='within')

# Aggregate pesticide exposure for each health survey location
exposure_summary = joined.groupby('index_right').pesticide_amount.sum()

# Merge exposure back into health_gdf
health_gdf['pesticide_exposure'] = exposure_summary
health_gdf['pesticide_exposure'] = health_gdf['pesticide_exposure'].fillna(0)

# Correlation between pesticide exposure and health issues
corr = health_gdf[['pesticide_exposure', 'health_issue_score']].corr().iloc[0,1]
print(f"Correlation between pesticide exposure and health issues: {corr:.3f}")

# Plot pesticide exposure near health survey points
fig, ax = plt.subplots(figsize=(10,6))
health_gdf.plot(column='pesticide_exposure', cmap='OrRd', legend=True, ax=ax)
ax.set_title('Pesticide Exposure near Health Survey Points')
plt.show()
