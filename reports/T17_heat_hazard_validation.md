# T17 ? Heat Hazard Validation

## Structural checks
- Rows: 559,555
- Unique H3 cells: 3,859
- Unique forecast timestamps: 145
- Forecast range: 2026-10-10 23:00:00 to 2026-10-16 23:00:00
- Duplicate time/H3 pairs: 0
- Invalid timestamps: 0
- Missing local heat-index / wet-bulb values: 0
- Hazard scores within [0,1]: passed

## Recorded sensitivity
The local anomaly adjustment factor is 0.3 in the prototype.
Sensitivity results from the generation run:
- Factor 0.0: mean 34.08 C, P95 37.27 C, max 37.69 C
- Factor 0.3: mean 34.11 C, P95 37.89 C, max 42.12 C
- Factor 0.5: mean 34.18 C, P95 38.57 C, max 45.43 C

## Limitations
- Hazard score is a within-hour relative percentile, not probability of illness.
- Local temperature adjustment is an assumption derived from a surface anomaly.
- Heat-index and wet-bulb estimates require scientific calibration before operational use.
- Refreshed output was uploaded to S3; structural validity does not imply health-risk validation.
