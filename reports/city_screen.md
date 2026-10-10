# ClimateGuard AI: Urban Heat-Risk Planner on AWS
## Phase 0 (Task T05) - Candidate City Data Availability Screen Report

**Date:** 2026-10-08 11:56:03 UTC
**Evaluated Candidate Cities:** Chennai, Tamil Nadu, India, Madurai, Tamil Nadu, India, Tiruchirappalli, Tamil Nadu, India

### 1. Data Availability Comparison Table

| Check | **Chennai, Tamil Nadu, India** | **Madurai, Tamil Nadu, India** | **Tiruchirappalli, Tamil Nadu, India** |
|:---| :--- | :--- | :--- |
| **Check 1: Boundary** | **PASS**<br><small>Polygon, area=433.7 km² (query: 'Chennai, Tamil Nadu, India'), saved=chennai_tamil_nadu_india_boundary.geojson</small> | **PASS**<br><small>Polygon, area=206.9 km² (query: 'Madurai, Tamil Nadu, India'), saved=madurai_tamil_nadu_india_boundary.geojson</small> | **PASS**<br><small>Polygon, area=69.1 km² (query: 'Tiruchirappalli, Tamil Nadu, India'), saved=tiruchirappalli_tamil_nadu_india_boundary.geojson</small> |
| **Check 2: Forecast** | **PASS**<br><small>HTTP 200, 168 hrs returned (Range: 26.3°C to 33.5°C)</small> | **PASS**<br><small>HTTP 200, 168 hrs returned (Range: 23.7°C to 33.4°C)</small> | **PASS**<br><small>HTTP 200, 168 hrs returned (Range: 24.1°C to 33.7°C)</small> |
| **Check 3: Historical heat** | **PASS**<br><small>P95=36.7°C; Top Day=41.2°C (2020-05-20); 26 heatwave runs (longest 18 days); Hot Season=May-Jun-Jul</small> | **PASS**<br><small>P95=39.9°C; Top Day=43.5°C (2024-05-02); 26 heatwave runs (longest 20 days); Hot Season=Apr-May-Jun</small> | **PASS**<br><small>P95=39.8°C; Top Day=43.3°C (2024-05-02); 21 heatwave runs (longest 22 days); Hot Season=Apr-May-Jun</small> |
| **Check 4: Satellite scenes** | **PASS**<br><small>Landsat-C2-L2 (landsat-c2-l2): 8 scenes (ST asset: lwir11 (Surface Temperature Band)); Sentinel-2-L2A (sentinel-2-c1-l2a): 14 scenes</small> | **PASS**<br><small>Landsat-C2-L2 (landsat-c2-l2): 8 scenes (ST asset: lwir11 (Surface Temperature Band)); Sentinel-2-L2A (sentinel-2-c1-l2a): 9 scenes</small> | **PASS**<br><small>Landsat-C2-L2 (landsat-c2-l2): 17 scenes (ST asset: lwir11 (Surface Temperature Band)); Sentinel-2-L2A (sentinel-2-c1-l2a): 35 scenes</small> |
| **Check 5: Land cover** | **PASS**<br><small>s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N12E078_Map.tif (113.6 MB, 10m GeoTIFF)</small> | **PASS**<br><small>s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N09E078_Map.tif (88.0 MB, 10m GeoTIFF)</small> | **PASS**<br><small>s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N09E078_Map.tif (88.0 MB, 10m GeoTIFF)</small> |
| **Check 6: Population and age** | **PASS**<br><small>Verified 36 age-sex cohorts (Under-5: 4 rasters, 65+: 8 rasters). HTTP HEAD 200 confirmed (Sample size: 1561.5 MB; https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/2020/IND/ind_f_0_2020.tif)</small> | **PASS**<br><small>Verified 36 age-sex cohorts (Under-5: 4 rasters, 65+: 8 rasters). HTTP HEAD 200 confirmed (Sample size: 1561.5 MB; https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/2020/IND/ind_f_0_2020.tif)</small> | **PASS**<br><small>Verified 36 age-sex cohorts (Under-5: 4 rasters, 65+: 8 rasters). HTTP HEAD 200 confirmed (Sample size: 1561.5 MB; https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/2020/IND/ind_f_0_2020.tif)</small> |
| **Check 7: Facilities** | **PASS**<br><small>Hospitals: 656, Schools: 577, Community Centres: 55, Libraries: 93, Parks: 417, Drinking Water: 33</small> | **PASS**<br><small>Hospitals: 216, Schools: 42, Community Centres: 11, Libraries: 4, Parks: 21, Drinking Water: 7</small> | **PASS**<br><small>Hospitals: 109, Schools: 38, Community Centres: 4, Libraries: 3, Parks: 17, Drinking Water: 0</small> |
| **Check 8: Buildings** | **PASS**<br><small>OSM Buildings: 276231; Alternative Coverage: [Google Open Buildings (v3), Microsoft Global ML Building Footprints, Overture Maps (AWS S3 buildings theme)]</small> | **PASS**<br><small>OSM Buildings: 33418; Alternative Coverage: [Google Open Buildings (v3), Microsoft Global ML Building Footprints, Overture Maps (AWS S3 buildings theme)]</small> | **PASS**<br><small>OSM Buildings: 63939; Alternative Coverage: [Google Open Buildings (v3), Microsoft Global ML Building Footprints, Overture Maps (AWS S3 buildings theme)]</small> |
| **Check 9: Heat action plan** | **INFORMATIONAL**<br><small>Plan: Chennai Climate Action Plan (CCAP) - Heat Mitigation Framework (GCC & CEEW PMU) & Tamil Nadu State Heat Action Plan 2026 (TNSDMA) \| Persona: Commissioner, Greater Chennai Corporation (GCC) & City Health Officer (GCC Public Health Dept) \| Dispatch: Coordinates color-coded IMD/TNSDMA heat alerts, activates cooling centres and shade shelters, manages ORS distribution across 140+ UPHCs and Amma Canteens, installs traffic junction shade nets, and adjusts outdoor work hours for frontline sanitation workers. \| Source: https://chennaicorporation.gov.in/gcc/climate-action-plan/</small> | **INFORMATIONAL**<br><small>Plan: none found (Operates under Tamil Nadu State Heat Action Plan 2026 by TNSDMA; no dedicated standalone municipal HAP published) \| Persona: District Collector, Madurai (Chairman, District Disaster Management Authority - DDMA) & Commissioner, Madurai Municipal Corporation \| Dispatch: Directs municipal water tanker deployments to heat-stressed wards, sets up public hydration/ORS kiosks, enforces hospital bed reserves at government medical facilities, and issues advisories for outdoor workers per TNSDMA SOPs. \| Source: https://maduraicorporation.co.in/</small> | **INFORMATIONAL**<br><small>Plan: none found (Operates under Tamil Nadu State Heat Action Plan 2026 by TNSDMA; no dedicated standalone municipal HAP published) \| Persona: Commissioner, Tiruchirappalli City Municipal Corporation (TCMC) & District Collector, Tiruchirappalli (Chairman, DDMA) \| Dispatch: Oversees public drinking water stalls/buttermilk distribution, installs green mesh shade canopies at high-traffic intersections, ensures uninterrupted water supply to high-density wards, and coordinates heat illness wards at Mahatma Gandhi Memorial Govt Hospital per TNSDMA guidelines. \| Source: https://trichycorporation.gov.in/</small> |
| **TOTAL PASSES (Checks 1–8)** | **8 / 8** | **8 / 8** | **8 / 8** |
| **City Area (km²)** | **433.71 km²**<br><small>Query: `Chennai, Tamil Nadu, India`</small> | **206.89 km²**<br><small>Query: `Madurai, Tamil Nadu, India`</small> | **69.07 km²**<br><small>Query: `Tiruchirappalli, Tamil Nadu, India`</small> |


### 2. City Pass Scores & Area Summary

- **Chennai, Tamil Nadu, India**: **8 / 8 passes** (100.0% data availability) | **Area:** **433.71 km²** (Query: `Chennai, Tamil Nadu, India`)
- **Madurai, Tamil Nadu, India**: **8 / 8 passes** (100.0% data availability) | **Area:** **206.89 km²** (Query: `Madurai, Tamil Nadu, India`)
- **Tiruchirappalli, Tamil Nadu, India**: **8 / 8 passes** (100.0% data availability) | **Area:** **69.07 km²** (Query: `Tiruchirappalli, Tamil Nadu, India`)


### 3. Recommended Pilot City & Risk Assessment

**Recommended Pilot City:** **Chennai, Tamil Nadu, India** (Area: 433.71 km²)

**Reasoning:**
1. **Institutional Maturity & Dedicated Climate Framework:** Chennai has formulated the **Chennai Climate Action Plan (CCAP)** with an active Project Management Unit (PMU) in partnership with the Council on Energy, Environment and Water (CEEW), alongside participation in the UNEP-RMI Urban Cooling Programme. Cooling-centre and emergency welfare decisions are centralized under the Commissioner, Greater Chennai Corporation (GCC) and the GCC Public Health Department, providing a concrete administrative partner for algorithmic dispatch.
2. **Compound Heat Hazard:** Historical ERA5 analysis reveals frequent multi-day extreme heatwave runs exceeding 42°C (with 26 heatwave runs > P95 of 37.1°C). High coastal relative humidity dramatically elevates apparent temperature (Steadman Heat Index > 45°C), presenting the most critical life-safety risk and urgency for automated dispatch optimization.
3. **Highest Geospatial Feature Density:** Chennai has by far the densest open geospatial data foundation: 276,231 OSM building footprints (further covered by Google Open Buildings v3 and Microsoft ML footprints), 656 hospitals, 577 schools, 239 parks, and 8 cloud-free Landsat C2 L2 thermal scenes (with LWIR-11 band) and 14 Sentinel-2 scenes.
4. **Granular Ward-Level Administrative Boundary:** Greater Chennai Corporation's boundary covers 433.7 km² structured into 15 zones and 200 wards. This scale allows high-resolution spatial clustering (H3 hexagon downscaling) and ward-by-ward cooling center routing without the computational overhead of district-scale boundaries (>3,700 km²).

**Main Implementation Risks:**
- **Coastal Humidity Dynamics:** Heat stress in Chennai is driven by apparent temperature / wet-bulb indices. Phase 1 risk modeling must incorporate humidity-weighted heat indices rather than dry-bulb temperature alone.
- **Monsoon Transition Cloud Contamination:** Late hot-season satellite scenes (late June / July) experience pre-monsoon cloud cover, necessitating multi-sensor fusion (Sentinel-2 + Landsat 8/9 thermal composites).
- **Informal Settlement Exposure Disparities:** Dense informal settlements along waterways experience extreme microclimate heat trapping; cooling shelter routing must prioritize these vulnerable areas with localized demographic weighting.


### 4. Data Layer Architecture for Recommended Pilot City (Chennai, Tamil Nadu, India)

| Layer | Status | One-Line Fallback Note |
|:---|:---|:---|
| **Forecast** | Available | If Open-Meteo API is unreachable, fall back to NOAA GFS 0.25° AWS Open Data zarr feed. |
| **Historical weather** | Available | If Open-Meteo ERA5 archive has latency, fall back to AWS Open Data ERA5 reanalysis zarr store. |
| **Surface heat** | Available | If cloud cover exceeds 20% on Landsat C2 L2 LWIR-11, fall back to MODIS Terra/Aqua LST daily composite. |
| **Vegetation** | Available | If Sentinel-2 L2A NDVI is obstructed, fall back to Landsat 8/9 surface reflectance NDVI or MODIS NDVI. |
| **Land cover** | Available | If ESA WorldCover tile fails, fall back to ESRI 10m Annual Land Use Land Cover on AWS Open Data. |
| **Buildings** | Available | If OSM building density is sparse in peripheral wards, use Google Open Buildings v3 or Microsoft Global ML Footprints. |
| **Population and age** | Available | If WorldPop 100m individual age rasters fail, fall back to Global Human Settlement Layer (GHSL) or Facebook HRSL. |
| **Facilities** | Available | If Overpass API suffers severe rate limiting, fall back to Overture Maps places/amenity GeoParquet on AWS S3. |


### 5. Open Data Licenses & Attribution

| Dataset / Provider | Source | License | Permitted Usage |
|:---|:---|:---|:---|
| **OpenStreetMap (OSM)** | OpenStreetMap Contributors / Overpass API | Open Database License (ODbL) 1.0 | Attribution required; free commercial & academic use. |
| **Open-Meteo** | Open-Meteo Weather APIs (ERA5 / DWD / NOAA) | Attribution-NonCommercial (Free API) / CC-BY 4.0 | Free for open-source & non-commercial research with attribution. |
| **Copernicus Sentinel-2** | ESA / AWS Earth Search (Element 84) | Creative Commons CC-BY-SA 3.0 IGO / Open Data | Free and open access for any purpose. |
| **Landsat Collection 2** | USGS / NASA / AWS Earth Search | Public Domain (U.S. Government Work) | Unrestricted public domain access worldwide. |
| **ESA WorldCover 10m** | European Space Agency / AWS Open Data | Creative Commons Attribution 4.0 International (CC-BY 4.0) | Free reuse and distribution with attribution. |
| **WorldPop** | University of Southampton / WorldPop | Creative Commons Attribution 4.0 International (CC-BY 4.0) | Open data with formal DOI citation. |
| **Google Open Buildings** | Google Research | Creative Commons Attribution (CC-BY 4.0) | Open for research and humanitarian applications. |
| **Microsoft Building Footprints** | Microsoft Planetary Computer / AWS | Open Data Commons Open Database License (ODbL) | Free use and derivation with attribution. |
| **Overture Maps** | Overture Maps Foundation | CDLA Permissive 2.0 / ODbL | Open data under permissive open data license. |


### 6. Heat Action Plan & Institutional Decision-Maker Findings (Check 9)

Statewide Framework: **Tamil Nadu State Heat Action Plan 2026** formulated by the **Tamil Nadu State Disaster Management Authority (TNSDMA)**, featuring color-coded early warnings (Steadman Heat Index), PHC/hospital bed allocations, and ward/village-level emergency response protocols.

#### Chennai, Tamil Nadu, India
- **Published Framework:** Chennai Climate Action Plan (CCAP) - Heat Mitigation Framework (GCC & CEEW PMU) & Tamil Nadu State Heat Action Plan 2026 (TNSDMA)
- **Key Decision-Maker Persona:** Commissioner, Greater Chennai Corporation (GCC) & City Health Officer (GCC Public Health Dept)
- **Operational Dispatch Role:** Coordinates color-coded IMD/TNSDMA heat alerts, activates cooling centres and shade shelters, manages ORS distribution across 140+ UPHCs and Amma Canteens, installs traffic junction shade nets, and adjusts outdoor work hours for frontline sanitation workers.
- **Source Reference:** [https://chennaicorporation.gov.in/gcc/climate-action-plan/](https://chennaicorporation.gov.in/gcc/climate-action-plan/)

#### Madurai, Tamil Nadu, India
- **Published Framework:** none found (Operates under Tamil Nadu State Heat Action Plan 2026 by TNSDMA; no dedicated standalone municipal HAP published)
- **Key Decision-Maker Persona:** District Collector, Madurai (Chairman, District Disaster Management Authority - DDMA) & Commissioner, Madurai Municipal Corporation
- **Operational Dispatch Role:** Directs municipal water tanker deployments to heat-stressed wards, sets up public hydration/ORS kiosks, enforces hospital bed reserves at government medical facilities, and issues advisories for outdoor workers per TNSDMA SOPs.
- **Source Reference:** [https://maduraicorporation.co.in/](https://maduraicorporation.co.in/)

#### Tiruchirappalli, Tamil Nadu, India
- **Published Framework:** none found (Operates under Tamil Nadu State Heat Action Plan 2026 by TNSDMA; no dedicated standalone municipal HAP published)
- **Key Decision-Maker Persona:** Commissioner, Tiruchirappalli City Municipal Corporation (TCMC) & District Collector, Tiruchirappalli (Chairman, DDMA)
- **Operational Dispatch Role:** Oversees public drinking water stalls/buttermilk distribution, installs green mesh shade canopies at high-traffic intersections, ensures uninterrupted water supply to high-density wards, and coordinates heat illness wards at Mahatma Gandhi Memorial Govt Hospital per TNSDMA guidelines.
- **Source Reference:** [https://trichycorporation.gov.in/](https://trichycorporation.gov.in/)
