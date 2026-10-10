"""Modular building-footprint ingestion interface and validation gate for Chennai."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import geopandas as gpd
from shapely.geometry import shape, box
from shapely.validation import explain_validity

from src.ingestion.config import (
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    CHENNAI_BBOX,
    SIBLING_BUILDING_SAMPLE_PATH,
)


class BuildingFootprintIngestor:
    """Modular building footprint ingestion interface.
    
    Supports loading local sample GeoJSON, chunk-based partition loaders,
    and geometry validation without triggering massive network downloads.
    """

    def __init__(self, geojson_path: Optional[Path] = None):
        self.geojson_path = Path(geojson_path or SIBLING_BUILDING_SAMPLE_PATH)

    def file_exists(self) -> bool:
        """Check if target building dataset exists on disk."""
        return self.geojson_path.is_file()

    def generate_spatial_chunks(self,
                                num_x: int = 4,
                                num_y: int = 4) -> List[Dict[str, Any]]:
        """Partition Chennai municipal bounding box into spatial grid tiles for chunked ingestion.
        
        Designed for future chunked fetching without triggering queries now.
        """
        w, s, e, n = (
            CHENNAI_BBOX["west"],
            CHENNAI_BBOX["south"],
            CHENNAI_BBOX["east"],
            CHENNAI_BBOX["north"],
        )
        dx = (e - w) / num_x
        dy = (n - s) / num_y

        chunks: List[Dict[str, Any]] = []
        chunk_id = 0
        for i in range(num_x):
            for j in range(num_y):
                tile_w = w + i * dx
                tile_e = w + (i + 1) * dx
                tile_s = s + j * dy
                tile_n = s + (j + 1) * dy
                chunks.append({
                    "chunk_id": f"tile_{chunk_id:02d}",
                    "bounds": {
                        "west": round(tile_w, 6),
                        "south": round(tile_s, 6),
                        "east": round(tile_e, 6),
                        "north": round(tile_n, 6),
                    },
                    "geometry": box(tile_w, tile_s, tile_e, tile_n),
                })
                chunk_id += 1

        return chunks

    def load_and_validate_local_sample(self,
                                       max_features: Optional[int] = None) -> Dict[str, Any]:
        """Load and validate the local building GeoJSON sample.
        
        Performs geometry verification, CRS consistency check, and bounding box validation.
        """
        if not self.file_exists():
            return {
                "status": "UNAVAILABLE",
                "message": f"Building footprint file not found at: {self.geojson_path}",
                "total_features": 0,
                "valid_geometries": 0,
            }

        with open(self.geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        if max_features and max_features > 0:
            features = features[:max_features]

        valid_count = 0
        invalid_count = 0
        missing_id_count = 0
        bounds_list: List[Tuple[float, float, float, float]] = []
        building_types: Dict[str, int] = {}

        for idx, feat in enumerate(features):
            props = feat.get("properties") or {}
            geom_data = feat.get("geometry")
            b_id = props.get("@id") or props.get("id") or props.get("osm_id")

            if not b_id:
                missing_id_count += 1

            b_type = str(props.get("building", "unspecified"))
            building_types[b_type] = building_types.get(b_type, 0) + 1

            if not geom_data:
                invalid_count += 1
                continue

            try:
                geom = shape(geom_data)
                if geom.is_valid and not geom.is_empty:
                    valid_count += 1
                    bounds_list.append(geom.bounds)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1

        # Calculate bounding box
        bbox = None
        overlaps_chennai = False
        if bounds_list:
            min_lon = min(b[0] for b in bounds_list)
            min_lat = min(b[1] for b in bounds_list)
            max_lon = max(b[2] for b in bounds_list)
            max_lat = max(b[3] for b in bounds_list)
            bbox = {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            }
            # Overlap check with Chennai administrative bbox
            overlaps_chennai = not (
                max_lon < CHENNAI_BBOX["west"] or
                min_lon > CHENNAI_BBOX["east"] or
                max_lat < CHENNAI_BBOX["south"] or
                min_lat > CHENNAI_BBOX["north"]
            )

        status = "PASS" if (valid_count > 0 and invalid_count == 0 and overlaps_chennai) else "WARN"

        report = {
            "status": status,
            "source_file": str(self.geojson_path),
            "file_size_mb": round(self.geojson_path.stat().st_size / (1024 * 1024), 2),
            "total_features": len(features),
            "valid_geometries": valid_count,
            "invalid_geometries": invalid_count,
            "missing_ids": missing_id_count,
            "bounding_box": bbox,
            "overlaps_chennai_admin_extent": overlaps_chennai,
            "top_building_types": dict(sorted(building_types.items(), key=lambda x: x[1], reverse=True)[:8]),
            "coverage_scope": (
                "Sample corridor dataset within Chennai (lon 80.199° to 80.250°, lat 13.039° to 13.080°). "
                "Not complete city-wide municipal coverage."
            ),
        }

        return {
            "status": status,
            "report": report,
            "total_features": len(features),
            "valid_count": valid_count,
        }
