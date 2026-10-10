"""Critical infrastructure and cooling facilities ingestor for Chennai."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from shapely.geometry import Point

from src.ingestion.config import (
    CACHE_DIR,
    FACILITY_CATEGORIES,
    OVERPASS_ENDPOINTS,
    USER_AGENT,
    CHENNAI_BBOX,
)


class FacilitiesIngestor:
    """Ingests, caches, and validates critical infrastructure and heat relief assets.
    
    Includes hospitals, schools, parks, community halls, and drinking water sources.
    Idempotent and uses local file caching.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "chennai_facilities_cache.json"

    def has_cached_data(self) -> bool:
        """Check if cached facilities file exists."""
        return self.cache_file.is_file()

    def load_cached_facilities(self) -> Optional[Dict[str, Any]]:
        """Load facilities from local cache if present."""
        if not self.has_cached_data():
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def query_overpass_counts(self,
                              poly_coords_str: Optional[str] = None,
                              timeout_sec: int = 35) -> Dict[str, int]:
        """Query Overpass API for counts in each facility category with mirror fallbacks."""
        if not poly_coords_str:
            # Fall back to bounding box query if no polygon string provided
            w, s, e, n = (
                CHENNAI_BBOX["west"],
                CHENNAI_BBOX["south"],
                CHENNAI_BBOX["east"],
                CHENNAI_BBOX["north"],
            )
            area_filter = f"({s},{w},{n},{e})"
        else:
            area_filter = f'(poly:"{poly_coords_str}")'

        queries = {
            "hospitals": f"""[out:json][timeout:{timeout_sec}];(nwr["amenity"="hospital"]{area_filter};);out count;""",
            "schools": f"""[out:json][timeout:{timeout_sec}];(nwr["amenity"="school"]{area_filter};);out count;""",
            "community_centres": f"""[out:json][timeout:{timeout_sec}];(nwr["amenity"="community_centre"]{area_filter};);out count;""",
            "libraries": f"""[out:json][timeout:{timeout_sec}];(nwr["amenity"="library"]{area_filter};);out count;""",
            "parks": f"""[out:json][timeout:{timeout_sec}];(nwr["leisure"="park"]{area_filter};);out count;""",
            "drinking_water": f"""[out:json][timeout:{timeout_sec}];(nwr["amenity"="drinking_water"]{area_filter};);out count;""",
        }

        results: Dict[str, int] = {}
        headers = {"User-Agent": USER_AGENT}

        for category, q in queries.items():
            count = 0
            for endpoint in OVERPASS_ENDPOINTS:
                try:
                    resp = requests.post(endpoint, data={"data": q}, headers=headers, timeout=timeout_sec + 10)
                    if resp.status_code == 200:
                        elems = resp.json().get("elements", [])
                        if elems and "tags" in elems[0]:
                            count = int(elems[0]["tags"].get("total", 0))
                        break
                except Exception:
                    continue
            results[category] = count
            time.sleep(1.0)  # Rate limiting pace

        return results

    def ingest_and_validate(self,
                            boundary_polygon: Optional[Any] = None,
                            force_refresh: bool = False) -> Dict[str, Any]:
        """Ingest facilities, prioritizing local cache, and return validation report."""
        cached_data = self.load_cached_facilities() if not force_refresh else None
        from_cache = cached_data is not None

        if from_cache and cached_data:
            counts = cached_data.get("counts", {})
        else:
            # Format boundary coordinates for Overpass if available
            poly_str = None
            if boundary_polygon and hasattr(boundary_polygon, "simplify"):
                simplified = boundary_polygon.simplify(0.005)
                if simplified.geom_type == "MultiPolygon":
                    simplified = max(simplified.geoms, key=lambda a: a.area)
                coords = list(simplified.exterior.coords)
                poly_str = " ".join(f"{lat:.5f} {lon:.5f}" for lon, lat in coords)

            counts = self.query_overpass_counts(poly_coords_str=poly_str)

            # Persist to local cache
            cache_payload = {
                "city": "Chennai, Tamil Nadu, India",
                "counts": counts,
                "cached_timestamp": time.time(),
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)

        # Validation checks
        total_assets = sum(counts.values())
        has_essential_assets = (counts.get("hospitals", 0) >= 5 and counts.get("schools", 0) >= 20)
        status = "PASS" if (total_assets > 0 and has_essential_assets) else "WARN"

        report = {
            "status": status,
            "from_cache": from_cache,
            "cache_file": str(self.cache_file),
            "total_assets_found": total_assets,
            "category_counts": counts,
            "hospitals_pass": counts.get("hospitals", 0) >= 5,
            "schools_pass": counts.get("schools", 0) >= 20,
            "cooling_relief_assets": {
                "parks": counts.get("parks", 0),
                "drinking_water": counts.get("drinking_water", 0),
                "community_centres": counts.get("community_centres", 0),
            },
        }

        return {
            "status": status,
            "counts": counts,
            "report": report,
        }
