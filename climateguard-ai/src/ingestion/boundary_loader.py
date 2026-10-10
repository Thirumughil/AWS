"""Chennai municipal boundary loader and validation gate."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import explain_validity

from src.ingestion.config import (
    CHENNAI_BOUNDARY_PATH,
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    CHENNAI_BBOX,
    EXPECTED_AREA_KM2_MIN,
    EXPECTED_AREA_KM2_MAX,
)


class ChennaiBoundaryLoader:
    """Loads and validates the official Chennai administrative boundary GeoJSON.
    
    Strictly reads the existing boundary file without re-downloading or modifying it.
    """

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = Path(file_path or CHENNAI_BOUNDARY_PATH)

    def file_exists(self) -> bool:
        """Check if boundary file exists on disk."""
        return self.file_path.is_file()

    def load_raw_data(self) -> Dict[str, Any]:
        """Load raw GeoJSON dictionary from disk."""
        if not self.file_exists():
            raise FileNotFoundError(
                f"Chennai boundary GeoJSON not found at: {self.file_path}. "
                "Per safety rules, automatic re-download is disabled."
            )
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_and_validate(self) -> Dict[str, Any]:
        """Perform comprehensive validation of geometry, CRS, bounds, and area.
        
        Returns a dictionary containing:
          - 'status': 'PASS' or 'FAIL'
          - 'gdf': GeoPandas GeoDataFrame (if valid)
          - 'area_km2': float
          - 'bounds': dict of bounding coordinates
          - 'validation_report': detailed checklist
        """
        raw_data = self.load_raw_data()
        features = raw_data.get("features", [])

        if not features:
            raise ValueError(f"Boundary file {self.file_path} contains no features.")

        primary_feature = features[0]
        geom_dict = primary_feature.get("geometry")
        if not geom_dict:
            raise ValueError("Primary boundary feature has no geometry object.")

        shapely_geom = shape(geom_dict)
        geom_valid = shapely_geom.is_valid
        geom_reason = explain_validity(shapely_geom) if not geom_valid else "Valid"

        if shapely_geom.is_empty:
            raise ValueError("Boundary geometry is empty.")

        # Load into GeoPandas
        gdf = gpd.read_file(self.file_path)
        if gdf.crs is None:
            gdf.set_crs(GEOGRAPHIC_CRS, inplace=True)

        # Reproject to UTM Zone 44N (EPSG:32644) for metric area computation
        gdf_projected = gdf.to_crs(PROJECTED_CRS)
        calculated_area_m2 = gdf_projected.geometry.area.sum()
        calculated_area_km2 = round(calculated_area_m2 / 1_000_000.0, 2)

        # Check bounds
        bounds = shapely_geom.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        bounds_check = (
            abs(minx - CHENNAI_BBOX["west"]) < 0.05 and
            abs(maxx - CHENNAI_BBOX["east"]) < 0.05 and
            abs(miny - CHENNAI_BBOX["south"]) < 0.05 and
            abs(maxy - CHENNAI_BBOX["north"]) < 0.05
        )

        area_check = EXPECTED_AREA_KM2_MIN <= calculated_area_km2 <= EXPECTED_AREA_KM2_MAX

        status = "PASS" if (geom_valid and bounds_check and area_check) else "FAIL"

        report = {
            "status": status,
            "file_path": str(self.file_path),
            "file_size_bytes": self.file_path.stat().st_size,
            "feature_count": len(features),
            "geometry_type": shapely_geom.geom_type,
            "geometry_valid": bool(geom_valid),
            "geometry_detail": geom_reason,
            "area_km2": calculated_area_km2,
            "area_check": bool(area_check),
            "expected_area_range_km2": [EXPECTED_AREA_KM2_MIN, EXPECTED_AREA_KM2_MAX],
            "bounds_wgs84": {
                "min_lon": minx,
                "min_lat": miny,
                "max_lon": maxx,
                "max_lat": maxy,
            },
            "bounds_check": bool(bounds_check),
            "properties": primary_feature.get("properties", {}),
        }

        return {
            "status": status,
            "gdf": gdf,
            "gdf_projected": gdf_projected,
            "area_km2": calculated_area_km2,
            "geometry": shapely_geom,
            "report": report,
        }
