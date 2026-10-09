import os
import sys
import subprocess
import math
import h3
import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path

H3_RES = 9
BOUNDARY_FILE = "data/raw/chennai_tamil_nadu_india_boundary.geojson"
EXPECTED_CELL_COUNT = 4100
EXPECTED_CELL_TOLERANCE = 0.05
AREA_TOLERANCE = 0.03

def get_cells_for_polygon(poly, res):
    """Get H3 cells for a single shapely Polygon."""
    if hasattr(h3, 'polygon_to_cells'):
        # h3 >= 4.0
        ext = [(lat, lon) for lon, lat in poly.exterior.coords]
        holes = [[(lat, lon) for lon, lat in hole.coords] for hole in poly.interiors]
        
        # Use LatLngPoly for h3 >= 4.0
        if hasattr(h3, 'LatLngPoly'):
            return h3.polygon_to_cells(h3.LatLngPoly(ext, *holes), res)
        else:
            return h3.polygon_to_cells(h3.Polygon(ext, *holes), res)
    else:
        # h3 < 4.0
        geo = {
            "type": "Polygon",
            "coordinates": [list(poly.exterior.coords)] + [list(hole.coords) for hole in poly.interiors]
        }
        return h3.polyfill(geo, res, geo_json_conformant=True)

def main():
    if not Path(BOUNDARY_FILE).exists():
        print(f"FAIL: {BOUNDARY_FILE} does not exist.")
        sys.exit(1)
        
    gdf_bound = gpd.read_file(BOUNDARY_FILE)
    if gdf_bound.empty:
        print("FAIL: empty boundary.")
        sys.exit(1)
        
    geom = gdf_bound.geometry.iloc[0]
    
    cells = set()
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = geom.geoms
    else:
        print(f"FAIL: unsupported geometry {geom.geom_type}")
        sys.exit(1)
        
    for poly in polys:
        cells.update(get_cells_for_polygon(poly, H3_RES))
        
    cells = list(cells)
    total_cells = len(cells)
    
        # Validation 1: cell count vs boundary area / actual mean cell area
    bnd_km2 = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:32644").geometry.area.sum() / 1e6
    mean_cell_km2 = sum(h3.cell_area(c, unit="km^2") for c in cells) / total_cells
    expected = bnd_km2 / mean_cell_km2
    print(f"Cells: {total_cells} | expected about {expected:.0f} (boundary {bnd_km2:.1f} km2 / mean cell {mean_cell_km2:.4f} km2)")
    if abs(total_cells - expected) / expected > EXPECTED_CELL_TOLERANCE:
        print(f"FAIL: cell count is not within {EXPECTED_CELL_TOLERANCE*100}% of the expected count")
        sys.exit(1)
        
    # Validation 2: no duplicates
    if len(cells) != len(set(cells)):
        print(f"FAIL: duplicate cells found. Unique: {len(set(cells))}, Total: {total_cells}")
        sys.exit(1)
        
    # Create GeoDataFrame
    records = []
    is_v4 = hasattr(h3, 'cell_to_latlng')
    get_latlng = h3.cell_to_latlng if is_v4 else h3.h3_to_geo
    get_boundary = h3.cell_to_boundary if is_v4 else h3.h3_to_geo_boundary
    get_parent = h3.cell_to_parent if is_v4 else h3.h3_to_parent
    
    for c in cells:
        lat, lon = get_latlng(c)
        parent = get_parent(c, 8)
        bnd = get_boundary(c)
        poly = Polygon([(b_lon, b_lat) for b_lat, b_lon in bnd])
        records.append({
            "h3_id": c,
            "lat": lat,
            "lon": lon,
            "parent_res8": parent,
            "geometry": poly
        })
        
    df = pd.DataFrame(records)
    gdf_cells = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    
    # Calculate area in EPSG:32644 (UTM Zone 44N - suitable for Chennai)
    gdf_cells_proj = gdf_cells.to_crs("EPSG:32644")
    gdf_cells["area_km2"] = gdf_cells_proj.geometry.area / 1e6
    
    # Validation 3: sum of cell areas vs boundary area
    bnd_area = gdf_bound.to_crs("EPSG:32644").geometry.area.sum() / 1e6
    cells_area = gdf_cells["area_km2"].sum()
    if abs(cells_area - bnd_area) / bnd_area > AREA_TOLERANCE:
        print(f"FAIL: cell area sum {cells_area:.2f} vs boundary area {bnd_area:.2f} differs by more than {AREA_TOLERANCE*100}%")
        sys.exit(1)
        
    # Validation 4: cells whose centre is within 50m of the polygon edge
    gdf_bound_proj = gdf_bound.to_crs("EPSG:32644")
    edge_proj = gdf_bound_proj.geometry.boundary
    # buffer by 50 meters
    buf_proj = edge_proj.buffer(50)
    
    gdf_centers = gdf_cells.copy()
    gdf_centers.geometry = gpd.points_from_xy(gdf_cells.lon, gdf_cells.lat)
    gdf_centers_proj = gdf_centers.to_crs("EPSG:32644")
    
    poly_buf = buf_proj.iloc[0]
    edge_cells = gdf_centers_proj[gdf_centers_proj.geometry.within(poly_buf)]
    edge_cells_count = len(edge_cells)
    
    print("All validations passed.")
    print(f"Total cells: {total_cells}")
    print(f"Sum of cell areas: {cells_area:.2f} km2")
    print(f"Boundary area: {bnd_area:.2f} km2")
    print(f"Cells whose centre is within 50m of the polygon edge: {edge_cells_count}")
    
    # Save files
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_geojson = "data/processed/chennai_h3_res9.geojson"
    out_csv = "data/processed/chennai_h3_res9.csv"
    
    gdf_cells.to_file(out_geojson, driver="GeoJSON")
    gdf_cells.drop(columns=["geometry"]).to_csv(out_csv, index=False)
    print(f"Saved {out_geojson} and {out_csv}")
    
    # Folium map
    Path("reports").mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = gdf_bound.total_bounds
    map_center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=map_center, zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Esri Streets").add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    
    # add hexes
    folium.GeoJson(
        gdf_cells.__geo_interface__,
        name="H3 Grid",
        style_function=lambda x: {'fillColor': '#ff7800', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.1}
    ).add_to(m)
    
    # add boundary
    folium.GeoJson(
        gdf_bound.__geo_interface__,
        name="Boundary",
        style_function=lambda x: {'fillColor': 'none', 'color': '#0000ff', 'weight': 3}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    map_path = "reports/chennai_grid.html"
    m.save(map_path)
    print(f"Saved map to {map_path}")
    
    # Upload to AWS S3
    for f in [out_geojson, out_csv]:
        fname = os.path.basename(f)
        cmd = ["aws", "s3", "cp", f, f"s3://urban-heat-risk-sai-001/processed/{fname}", "--profile", "climateguard"]
        print(f"Uploading {f}...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAIL: S3 upload failed for {f}")
            print(res.stderr)
            sys.exit(1)
        else:
            print(f"Uploaded {fname}")

if __name__ == '__main__':
    main()
