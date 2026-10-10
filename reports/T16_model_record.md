# T16 ? LST/UHI Model Record

## Selected estimator
- Saved model: `gboost`
- Estimator class: `HistGradientBoostingRegressor`
- Features: 20
- H3 cells in current feature table: 3,859
- City-level Landsat date records: 36

## Recorded validation results
- Gradient-boosting spatial CV R?: 0.8267
- Gradient-boosting spatial CV RMSE (?C): 0.9649
- Ridge spatial CV R?: 0.672
- Ridge spatial CV RMSE (?C): 1.3273
- City-mean leave-one-date-out RMSE (?C): 2.463

## Implementation decision
The original plan specified LightGBM. The current selected estimator is
scikit-learn HistGradientBoostingRegressor, retained as a documented
substitute for the time-constrained prototype. Do not describe it as LightGBM.

## Known limitations
- Building height and nearest-water distance are not included in the
  current 20 model features.
- The city-mean LST model has recorded leave-one-date-out RMSE of about
  2.46 ?C. This is not an afternoon air-temperature or heat-index forecast.
- The surface-temperature pattern represents Landsat morning observations,
  not daily maximum surface temperature.
- `src/explain_hotspot.py` currently explains the ridge JSON artifact.
  Its contributions must not be presented as explanations of predictions
  from the selected gradient-boosting model.

## Operational caution
Use the reported validation metrics with their documented split methodology.
Treat the model as a prototype until feature limitations and the
explanation-model mismatch are resolved.
