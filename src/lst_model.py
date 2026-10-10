#!/usr/bin/env python
"""
T16: land-surface-temperature (LST) model for the Chennai hex grid.

    python src/lst_model.py                     # train, evaluate, save the model and tables
    python src/lst_model.py --day 2026-05-21    # predict a hex LST map for one day (after training)
    python src/lst_model.py --upload            # also copy the outputs to S3

Two parts, kept deliberately simple so every number can be explained:

 1. WHERE is it hotter?   For each clear Landsat date, a hex's anomaly = its LST minus the city mean that day.
    A model learns the average anomaly of a hex from static features (built-up share, tree share, building cover,
    NDVI, neighbourhood averages, coordinates). It is scored on city blocks it has never seen (spatial cross-validation).
 2. HOW HOT is the city today?   The city-mean LST of a Landsat date is regressed on the ERA5 / forecast air
    temperature around the satellite pass (about 10:30 local time). It is scored by leave-one-date-out.

    hex LST for a day = city-mean LST(day) + hex anomaly

Limits to state in the README: Landsat sees the surface around 10:30 in the morning, not the afternoon peak; the model
explains the typical heat pattern, not weather-specific effects; it was trained on Mar-Aug dates of 2022-2026.

Outputs
    data/processed/chennai_hex_features.csv      static features per hex
    data/processed/chennai_hex_lst_pattern.csv   observed and predicted anomaly per hex
    data/processed/chennai_lst_city_dates.csv    city-mean LST per training date with the ERA5 predictors
    models/lst_anomaly.json (ridge) or lst_anomaly.joblib (gradient boosting), models/lst_city_model.json
    reports/lst_model_metrics.csv
Needs: numpy pandas h3   (scikit-learn is optional and adds a gradient-boosting model)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scikit-learn is optional: some Windows setups block its compiled files (Application Control), and the
# script then runs with its own ridge regression alone. Only the gradient-boosting model needs scikit-learn.
HGB = None
if not os.environ.get("LST_NO_SKLEARN"):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    except Exception as err:                       # ImportError, or "DLL load failed ... Application Control"
        print(f"Note: scikit-learn is not usable here ({str(err)[:80]}); using the built-in ridge model only.")

import h3

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

MIN_SCENE_VALID = 0.80          # a date counts only if at least this share of its hex-pixels are clear
MIN_HEX_VALID = 0.50            # a hex counts on a date only if at least this share of its pixels are clear
MIN_DATES_PER_HEX = 5           # a hex needs this many clear dates to be a training row
LST_RANGE_C = (15, 65)          # values outside this are treated as errors
PASS_HOURS = (10, 11)           # local hours averaged to represent the ~10:30 satellite pass
CV_FOLDS = 5
SEED = 7

PROFILE = os.environ.get("AWS_PROFILE", "climateguard")
S3_PREFIX = "s3://urban-heat-risk-sai-001/processed/"
V4 = hasattr(h3, "grid_disk")

LAND = ["built_frac", "tree_frac", "grass_shrub_frac", "crop_frac", "bare_frac", "water_frac"]
BLDG = ["bldg_cover_frac", "small_bldg_frac"]
NEIGHBOUR_OF = ["built_frac", "tree_frac", "bldg_cover_frac", "ndvi_mean"]
NEIGHBOUR_K = (1, 3)


def disk(c, k):
    return h3.grid_disk(c, k) if V4 else h3.k_ring(c, k)


def need(p):
    if not p.exists():
        sys.exit(f"Missing file: {p}\nRestore it from S3 (aws s3 sync) or re-run the earlier step.")
    return p


# ----------------------------------------------------------------- features
def build_features(grid, lst, s2, lc, ex, use_coords):
    f = grid[["h3_id", "lat", "lon", "parent_res8"]].copy()
    f = f.merge(lc[["h3_id"] + LAND], on="h3_id", how="left")
    f = f.merge(ex[["h3_id"] + BLDG + ["nearest_park_m"]], on="h3_id", how="left")

    # NDVI: Sentinel-2 where available (sharper), otherwise Landsat's
    s2ok = s2[s2["s2_valid_frac"] >= MIN_HEX_VALID].groupby("h3_id")["ndvi_s2"].mean().rename("ndvi_s2_mean")
    l8ok = lst[lst["lst_valid_frac"] >= MIN_HEX_VALID].groupby("h3_id")["ndvi_l8"].mean().rename("ndvi_l8_mean")
    f = f.merge(s2ok, on="h3_id", how="left").merge(l8ok, on="h3_id", how="left")
    f["ndvi_mean"] = f["ndvi_s2_mean"].fillna(f["ndvi_l8_mean"])
    f = f.drop(columns=["ndvi_s2_mean", "ndvi_l8_mean"])

    # neighbourhood averages (a hot block heats its surroundings)
    idx = {h: i for i, h in enumerate(f["h3_id"])}
    base = f[NEIGHBOUR_OF].to_numpy(dtype=float)
    for k in NEIGHBOUR_K:
        out = np.full_like(base, np.nan)
        for h, i in idx.items():
            nb = [idx[x] for x in disk(h, k) if x in idx]
            with np.errstate(all="ignore"):
                out[i] = np.nanmean(base[nb], axis=0)
        for j, col in enumerate(NEIGHBOUR_OF):
            f[f"{col}_nb{k}"] = out[:, j]

    cols = LAND + BLDG + ["nearest_park_m", "ndvi_mean"] + \
           [f"{c}_nb{k}" for k in NEIGHBOUR_K for c in NEIGHBOUR_OF]
    if use_coords:
        cols += ["lat", "lon"]
    return f, cols


# ----------------------------------------------------------------- part 1: where
def load_anomalies(lst):
    ds = lst.groupby("date")["lst_valid_frac"].mean()
    good = ds[ds >= MIN_SCENE_VALID].index
    x = lst[lst["date"].isin(good) & (lst["lst_valid_frac"] >= MIN_HEX_VALID)].copy()
    x = x[x["lst_c"].between(*LST_RANGE_C)]
    city = x.groupby("date")["lst_c"].agg(city_lst_c="mean", n_hexes="size").reset_index()
    x = x.merge(city[["date", "city_lst_c"]], on="date")
    x["anom"] = x["lst_c"] - x["city_lst_c"]
    return x, city, len(ds), len(good)


def r2(y, p):
    ss = ((y - p) ** 2).sum()
    st = ((y - y.mean()) ** 2).sum()
    return 1 - ss / st if st > 0 else np.nan


class Ridge:
    """Weighted ridge regression with median imputation and standardisation (numpy only)."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def _prep(self, X):
        return np.where(np.isnan(X), self.med, X)

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(med), 0.0, med)
        Xi = self._prep(X)
        self.mu = np.average(Xi, axis=0, weights=w)
        sd = np.sqrt(np.average((Xi - self.mu) ** 2, axis=0, weights=w))
        self.sd = np.where(sd == 0, 1.0, sd)
        Z = (Xi - self.mu) / self.sd
        self.intercept = float(np.average(y, weights=w))
        A = Z.T @ (Z * w[:, None]) + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (w * (y - self.intercept)))
        return self

    def predict(self, X):
        return ((self._prep(X) - self.mu) / self.sd) @ self.coef + self.intercept

    def to_dict(self):
        return {"kind": "ridge", "alpha": self.alpha, "median": self.med.tolist(), "mean": self.mu.tolist(),
                "std": self.sd.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept}


def make_models():
    models = {"ridge": Ridge(alpha=10.0)}
    if HGB is not None:
        models["gboost"] = HGB(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=30,
                               l2_regularization=1.0, random_state=SEED)
    return models


def group_folds(groups, k, seed=SEED):
    """Assign whole neighbourhood blocks to folds, so a block is never in both training and testing."""
    uniq = np.unique(groups)
    np.random.default_rng(seed).shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def spatial_cv(X, y, w, groups):
    fold = group_folds(groups, CV_FOLDS)
    res = {}
    for name in make_models():
        oof = np.full(len(y), np.nan)
        for f in range(CV_FOLDS):
            te = fold == f
            m = make_models()[name]
            m.fit(X[~te], y[~te], sample_weight=w[~te])
            oof[te] = m.predict(X[te])
        res[name] = oof
    return res


def permutation_rise(model, X, y, repeats=5, seed=SEED):
    """How much the error (C) rises when one feature's values are shuffled."""
    rng = np.random.default_rng(seed)
    base = np.sqrt(np.mean((y - model.predict(X)) ** 2))
    rise = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            rise[j] += np.sqrt(np.mean((y - model.predict(Xp)) ** 2)) - base
    return rise / repeats


# ----------------------------------------------------------------- part 2: how hot
def hourly_predictors(hourly, day):
    """Predictors for one day from an hourly table with time, temperature_2m, relative_humidity_2m, apparent_temperature."""
    h = hourly[hourly["time"].dt.normalize() == pd.Timestamp(day)]
    if h.empty:
        return None
    p = h[h["time"].dt.hour.isin(PASS_HOURS)]
    if p.empty:
        return None
    return {"t_pass": p["temperature_2m"].mean(), "app_pass": p["apparent_temperature"].mean(),
            "rh_pass": p["relative_humidity_2m"].mean(), "t_max": h["temperature_2m"].max()}


def fit_ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def loo_rmse(X, y):
    err = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        c = fit_ols(X[keep], y[keep])
        err.append(y[i] - (c[0] + X[i] @ c[1:]))
    return float(np.sqrt(np.mean(np.square(err))))


def city_model(city, hourly):
    rows = []
    for d in city["date"]:
        p = hourly_predictors(hourly, d)
        rows.append(p if p else {"t_pass": np.nan, "app_pass": np.nan, "rh_pass": np.nan, "t_max": np.nan})
    c = pd.concat([city.reset_index(drop=True), pd.DataFrame(rows)], axis=1).dropna()
    y = c["city_lst_c"].to_numpy()
    options = {"t_pass": ["t_pass"], "app_pass": ["app_pass"], "t_max": ["t_max"], "t_pass+rh_pass": ["t_pass", "rh_pass"]}
    table = []
    for name, cols in options.items():
        X = c[cols].to_numpy()
        coef = fit_ols(X, y)
        pred = coef[0] + X @ coef[1:]
        table.append({"predictor": name, "cols": cols, "coef": coef, "r2": r2(y, pred),
                      "loo_rmse_c": loo_rmse(X, y)})
    best = min(table, key=lambda t: t["loo_rmse_c"])
    simple = [t for t in table if len(t["cols"]) == 1 and t["loo_rmse_c"] <= best["loo_rmse_c"] + 0.05]
    if simple and len(best["cols"]) > 1:
        best = min(simple, key=lambda t: t["loo_rmse_c"])           # prefer one variable when it is nearly as good
    return c, table, best


# ----------------------------------------------------------------- train
def train(args):
    grid = pd.read_csv(need(PROCESSED / "chennai_h3_res9.csv"))
    lst = pd.read_csv(need(PROCESSED / "chennai_hex_lst.csv"))
    s2 = pd.read_csv(need(PROCESSED / "chennai_hex_ndvi_s2.csv"))
    lc = pd.read_csv(need(PROCESSED / "chennai_hex_landcover.csv"))
    ex = pd.read_csv(need(PROCESSED / "chennai_hex_exposure.csv"))
    hourly = pd.read_csv(need(RAW / "chennai_era5_hourly.csv"), parse_dates=["time"])

    print(f"{len(grid)} hexes | {lst['date'].nunique()} Landsat dates in the table")
    feat, cols = build_features(grid, lst, s2, lc, ex, not args.no_coords)
    x, city, n_all, n_good = load_anomalies(lst)
    print(f"Dates with at least {MIN_SCENE_VALID:.0%} clear pixels: {n_good} of {n_all} | "
          f"hex-date rows used: {len(x):,} | features: {len(cols)}")

    hexy = x.groupby("h3_id")["anom"].agg(anom_obs="mean", n_dates="size").reset_index()
    tr = feat.merge(hexy, on="h3_id", how="inner")
    tr = tr[tr["n_dates"] >= MIN_DATES_PER_HEX].reset_index(drop=True)
    print(f"Training hexes: {len(tr)} of {len(feat)} (need {MIN_DATES_PER_HEX}+ clear dates) | "
          f"observed anomaly: mean {tr['anom_obs'].mean():+.2f}, std {tr['anom_obs'].std():.2f}, "
          f"range {tr['anom_obs'].min():+.1f} to {tr['anom_obs'].max():+.1f} C")

    X = tr[cols].to_numpy(dtype=float)
    y = tr["anom_obs"].to_numpy()
    w = tr["n_dates"].to_numpy(dtype=float)
    groups = tr["parent_res8"].astype(str).to_numpy()
    print(f"Spatial cross-validation: {CV_FOLDS} folds over {len(set(groups))} neighbourhood blocks (res-8 cells)")
    oofs = spatial_cv(X, y, w, groups)

    metrics, scores = [], {}
    base_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    print(f"\nPart 1: hex anomaly (C), scored on blocks the model has not seen")
    print(f"  {'model':<10}{'R2':>7}{'RMSE':>8}{'MAE':>8}")
    print(f"  {'mean only':<10}{0:>7.2f}{base_rmse:>8.2f}{np.mean(np.abs(y - y.mean())):>8.2f}")
    for name, oof in oofs.items():
        sc = (r2(y, oof), float(np.sqrt(np.mean((y - oof) ** 2))), float(np.mean(np.abs(y - oof))))
        scores[name] = sc
        print(f"  {name:<10}{sc[0]:>7.2f}{sc[1]:>8.2f}{sc[2]:>8.2f}")
        metrics += [(f"anomaly_cv_r2_{name}", sc[0]), (f"anomaly_cv_rmse_c_{name}", sc[1])]
    best_name = min(scores, key=lambda k: scores[k][1])
    metrics.append(("anomaly_baseline_rmse_c", base_rmse))
    print(f"  -> using {best_name}")

    # per-date check of the out-of-fold pattern
    tr["oof"] = oofs[best_name]
    xx = x.merge(tr[["h3_id", "oof", "anom_obs"]], on="h3_id")
    per = []
    for d, g in xx.groupby("date"):
        per.append({"date": d, "r2": r2(g["anom"].to_numpy(), g["oof"].to_numpy()),
                    "rmse": float(np.sqrt(np.mean((g["anom"] - g["oof"]) ** 2))),
                    "ceiling_r2": r2(g["anom"].to_numpy(), g["anom_obs"].to_numpy())})
    per = pd.DataFrame(per)
    print(f"\n  On each of the {len(per)} dates (held-out blocks): R2 median {per['r2'].median():.2f} "
          f"(worst {per['r2'].min():.2f}, best {per['r2'].max():.2f}), RMSE median {per['rmse'].median():.2f} C")
    print(f"  Ceiling for ANY model using fixed hex features: R2 {per['ceiling_r2'].median():.2f} "
          "(each hex's own average pattern); the gap below that is day-to-day variation no static model can see")
    metrics += [("anomaly_perdate_r2_median", per["r2"].median()), ("anomaly_perdate_rmse_c_median", per["rmse"].median()),
                ("anomaly_ceiling_r2_median", per["ceiling_r2"].median())]

    # final fit on every training hex
    final = make_models()[best_name]
    final.fit(X, y, sample_weight=w)
    imp = pd.Series(permutation_rise(final, X, y), index=cols).sort_values(ascending=False)
    print("\n  Most useful features (rise in error when shuffled, C):")
    for k, v in imp.head(8).items():
        print(f"    {k:<24}{v:>6.3f}")

    Xall = feat[cols].to_numpy(dtype=float)
    feat["anomaly_model"] = final.predict(Xall)
    centre = float(feat.loc[feat["h3_id"].isin(tr["h3_id"]), "anomaly_model"].mean())
    feat["anomaly_model"] = (feat["anomaly_model"] - centre).round(3)

    # part 2
    c, table, best = city_model(city, hourly)
    print(f"\nPart 2: city-mean LST from air temperature around the ~10:30 pass ({len(c)} dates)")
    print(f"  {'predictor':<16}{'R2':>7}{'leave-one-out RMSE':>20}")
    for t in table:
        print(f"  {t['predictor']:<16}{t['r2']:>7.2f}{t['loo_rmse_c']:>20.2f}{'   <- used' if t is best else ''}")
    b = best["coef"]
    eq = f"city LST = {b[0]:.2f} " + " ".join(f"{v:+.2f} x {n}" for v, n in zip(b[1:], best["cols"]))
    print(f"  {eq}")
    print(f"  city LST range in training: {c['city_lst_c'].min():.1f} to {c['city_lst_c'].max():.1f} C "
          f"(air temperature at the pass: {c['t_pass'].min():.1f} to {c['t_pass'].max():.1f} C)")
    metrics += [("city_loo_rmse_c", best["loo_rmse_c"]), ("city_r2", best["r2"])]

    # save
    for p in (PROCESSED, REPORTS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
    feat.drop(columns=["parent_res8"]).to_csv(PROCESSED / "chennai_hex_features.csv", index=False)
    pat = feat[["h3_id", "anomaly_model"]].merge(hexy, on="h3_id", how="left").merge(
        tr[["h3_id", "oof"]].rename(columns={"oof": "anomaly_cv"}), on="h3_id", how="left")
    pat.round(3).to_csv(PROCESSED / "chennai_hex_lst_pattern.csv", index=False)
    c.round(3).to_csv(PROCESSED / "chennai_lst_city_dates.csv", index=False)
    if best_name == "ridge":
        (MODELS / "lst_anomaly.json").write_text(json.dumps(
            {**final.to_dict(), "features": cols, "centre": centre}, indent=1))
    else:
        import joblib
        joblib.dump({"model": final, "features": cols, "centre": centre, "name": best_name},
                    MODELS / "lst_anomaly.joblib")
    (MODELS / "lst_city_model.json").write_text(json.dumps({
        "predictors": best["cols"], "intercept": float(b[0]), "coef": [float(v) for v in b[1:]],
        "loo_rmse_c": best["loo_rmse_c"], "r2": best["r2"], "n_dates": int(len(c)), "pass_hours": list(PASS_HOURS),
        "anomaly_cv_rmse_c": scores[best_name][1]}, indent=2))
    pd.DataFrame(metrics, columns=["metric", "value"]).round(4).to_csv(REPORTS / "lst_model_metrics.csv", index=False)
    print(f"\nSaved tables to {PROCESSED}, model to {MODELS}, metrics to {REPORTS / 'lst_model_metrics.csv'}")
    return [PROCESSED / "chennai_hex_features.csv", PROCESSED / "chennai_hex_lst_pattern.csv",
            PROCESSED / "chennai_lst_city_dates.csv"]


# ----------------------------------------------------------------- predict one day
def predict_day(day):
    mj = need(MODELS / "lst_city_model.json")
    cm = json.loads(mj.read_text())
    pat = pd.read_csv(need(PROCESSED / "chennai_hex_lst_pattern.csv"))
    sources = []
    fc = RAW / "chennai_forecast.csv"
    if fc.exists():
        sources.append(("forecast file", pd.read_csv(fc, parse_dates=["time"])))
    sources.append(("ERA5 history", pd.read_csv(need(RAW / "chennai_era5_hourly.csv"), parse_dates=["time"])))
    p, src = None, None
    for name, h in sources:
        p = hourly_predictors(h, day)
        if p:
            src = name
            break
    if not p:
        sys.exit(f"No hourly weather for {day} in the forecast file or the ERA5 file.")
    city_lst = cm["intercept"] + sum(c * p[k] for c, k in zip(cm["coef"], cm["predictors"]))
    out = pat[["h3_id", "anomaly_model"]].copy()
    out["lst_pred_c"] = (city_lst + out["anomaly_model"]).round(2)
    dest = PROCESSED / f"chennai_hex_lst_pred_{day}.csv"
    out.to_csv(dest, index=False)

    print(f"{day}: weather from {src} | air temperature around the pass {p['t_pass']:.1f} C, "
          f"apparent {p['app_pass']:.1f} C, humidity {p['rh_pass']:.0f}%")
    print(f"  predicted city-mean surface temperature {city_lst:.1f} C (typical error +/- {cm['loo_rmse_c']:.1f} C)")
    print(f"  hex range: 5th to 95th percentile {out['lst_pred_c'].quantile(0.05):.1f} to "
          f"{out['lst_pred_c'].quantile(0.95):.1f} C (hex-level error about +/- {cm['anomaly_cv_rmse_c']:.1f} C)")
    cd = PROCESSED / "chennai_lst_city_dates.csv"
    if cd.exists():
        c = pd.read_csv(cd)
        r = c[c["date"].astype(str).str[:10] == day]
        if len(r):
            print(f"  Landsat saw {r.iloc[0]['city_lst_c']:.1f} C that day (a training date, so this is not an independent test)")
    print(f"Saved {dest}")
    return [dest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="predict the hex LST map for this date, YYYY-MM-DD")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--no-coords", action="store_true", help="leave latitude/longitude out of the features")
    args = ap.parse_args()
    files = predict_day(args.day) if args.day else train(args)
    if args.upload:
        for f in files:
            cmd = ["aws", "s3", "cp", str(f), S3_PREFIX + f.name, "--profile", PROFILE]
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as err:
                print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()