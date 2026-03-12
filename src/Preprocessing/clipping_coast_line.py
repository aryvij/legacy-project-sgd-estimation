import geopandas as gpd

# Load your files
boundary_fp = r"C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\input\shapefiles\coast_line\coastline.shp"
coastline_fp = r"C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\input\shapefiles\coastline_check\ne_10m_coastline.shp"

boundary = gpd.read_file(boundary_fp)
coastline = gpd.read_file(coastline_fp)

# Ensure both are in the same CRS (projected CRS recommended for buffering in meters)
if boundary.crs != coastline.crs:
    coastline = coastline.to_crs(boundary.crs)

# Buffer the coastline (e.g., 500 meters on each side)
buffer_distance = 3500  # meters
# If CRS is geographic (degrees), you might want to reproject to a UTM CRS first
if boundary.crs.is_geographic:
    utm_crs = boundary.estimate_utm_crs()
    boundary = boundary.to_crs(utm_crs)
    coastline = coastline.to_crs(utm_crs)

coastline_buffer = coastline.buffer(buffer_distance)

# Create a GeoDataFrame for the buffer
coastline_buffer_gdf = gpd.GeoDataFrame(geometry=coastline_buffer, crs=coastline.crs)

# Clip the boundary to the buffered coastline area to get coastal boundary
coastal_boundary = gpd.overlay(boundary, coastline_buffer_gdf, how='intersection')

# Save result
output_fp = r"C:\Users\aryapv\OneDrive - KTH\Modelling_SGD_Arya\SGD_model\data\output\coastal_boundary.shp"
coastal_boundary.to_file(output_fp)

print(f"Coastal boundary extracted and saved to: {output_fp}")
