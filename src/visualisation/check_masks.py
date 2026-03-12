import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from pathlib import Path

# ---- EDIT these to match your setup ----
catchment_id = 204
base_ws = Path("data/output/model_runs") / f"mf6_{catchment_id}"
dem_tif = base_ws / "soil_thickness.tif"   # only used for its transform & CRS
ghb_tif = base_ws / f"ghb_mask_{catchment_id}.tif"
chd_tif = base_ws / "chd_mask_inland_band.tif"
rivers_shp = Path("data/input/shapefiles/surface_water/scandinavian_waters_lines_shp/scandinavian_waters_lines.shp")
# ----------------------------------------

# 1) grab transform & CRS
with rasterio.open(dem_tif) as ds:
    tr = ds.transform
    crs = ds.crs
    shape = (ds.height, ds.width)

# 2) rasterize the river lines
rivers = gpd.read_file(rivers_shp).to_crs(crs)
riv_mask = rasterize(
    [(geom, 1) for geom in rivers.geometry],
    out_shape=shape, transform=tr, fill=0, dtype="uint8"
)

# 3) read your saved masks
with rasterio.open(ghb_tif) as ds: ghb_mask = ds.read(1)
with rasterio.open(chd_tif) as ds: chd_mask = ds.read(1)

# 4) plot them
for data, title in [(ghb_mask, "Coastal GHB Mask"),
                    (chd_mask, "Inland CHD Band Mask"),
                    (riv_mask, "River (RIV) Mask")]:
    plt.figure()
    plt.imshow(data, origin="upper")
    plt.title(title)
    plt.axis("off")

plt.show()
