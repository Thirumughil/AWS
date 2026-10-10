#!/usr/bin/env python
"""
T37: backtest of the LST model against Landsat scenes it was NOT trained on.

    python src/backtest_lst.py                       # the two chosen days + every clear Landsat date
    python src/backtest_lst.py --days 2026-05-21     # only the days you name (still scores all dates in the summary)
    python src/backtest_lst.py --upload

For each test date the model is rebuilt WITHOUT that date, then asked to predict that day's hex temperature map:
  * the hex pattern comes from blocks of the city it has never seen (spatial hold-out) AND from dates it has never seen,
  * the city-mean temperature comes from ERA5 air temperature around the ~10:30 pass, using a city model fitted without that date,
  * predicted hex LST = predicted city-mean + predicted hex anomaly, compared with what Landsat measured.

Reports
  city-mean error (C), hex-level RMSE (C) against a "no pattern" baseline (every hex = city mean),
  pattern skill (R2 and rank correlation of the anomaly), and how many of Landsat's hottest 10% of hexes the model finds.

Reads : data/processed/chennai_h3_res9, _hex_lst, _hex_ndvi_s2, _hex_landcover, _hex_exposure CSVs; data/raw/chennai_era5_hourly.csv
Writes: reports/lst_backtest.csv (one row per date), reports/lst_backtest_summary.csv,
        data/processed/chennai_backtest_<day>.csv for the named days (observed vs predicted per hex)
Note  : the feature table is built once from all dates. Only the NDVI fallback (Landsat NDVI, used where Sentinel-2 is missing)
        sees the test date; NDVI is not temperature, so the effect is negligible.
Needs : the same packages as lst_model.py (numpy pandas h3). Run from the project folder, with lst_model.py in src/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lst_model as L

DAYS = ["2026-05-21", "2024-05-31"]
TOP_FRAC = 0.10


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def backtest_date(day, lst, feat, cols, hourly, model_name):
    held = lst[lst["day"] == day]
    if held.empty:
        return None, None, "no Landsat rows for this date in chennai_hex_lst.csv"
    rest = lst[lst["day"] != day]

    # --- hex pattern without this date, scored on blocks the model has not seen
    x, city, _, _ = L.load_anomalies(rest)
    hexy = x.groupby("h3_id")["anom"].agg(anom_obs="mean", n_dates="size").reset_index()
    tr = feat.merge(hexy, on="h3_id", how="inner")
    tr = tr[tr["n_dates"] >= L.MIN_DATES_PER_HEX].reset_index(drop=True)
    X = tr[cols].to_numpy(dtype=float)
    y = tr["anom_obs"].to_numpy()
    w = tr["n_dates"].to_numpy(dtype=float)
    fold = L.group_folds(tr["parent_res8"].astype(str).to_numpy(), L.CV_FOLDS)
    oof = np.full(len(y), np.nan)
    for f in range(L.CV_FOLDS):
        te = fold == f
        m = L.make_models()[model_name]
        m.fit(X[~te], y[~te], sample_weight=w[~te])
        oof[te] = m.predict(X[te])
    final = L.make_models()[model_name]
    final.fit(X, y, sample_weight=w)
    pred = pd.Series(final.predict(feat[cols].to_numpy(dtype=float)), index=feat["h3_id"].to_numpy())
    pred.loc[tr["h3_id"].to_numpy()] = oof                      # training hexes: use the unseen-block prediction
    pred_anom = pred - float(pred.loc[tr["h3_id"].to_numpy()].mean())

    # --- city-mean temperature for the day, from weather only, with a city model fitted without this date
    _, _, best = L.city_model(city, hourly)
    p = L.hourly_predictors(hourly, day)
    if p is None:
        return None, None, "no ERA5 hourly weather for this date"
    city_pred = best["coef"][0] + sum(c * p[k] for c, k in zip(best["coef"][1:], best["cols"]))

    # --- what Landsat saw
    scene_valid = float(held["lst_valid_frac"].mean())
    hv = held[(held["lst_valid_frac"] >= L.MIN_HEX_VALID) & held["lst_c"].between(*L.LST_RANGE_C)]
    hv = hv.groupby("h3_id", as_index=False)["lst_c"].mean()
    if len(hv) < 100:
        return None, None, f"only {len(hv)} clear hexes on this date"
    city_obs = float(hv["lst_c"].mean())
    hv["lst_obs"] = hv["lst_c"]
    hv["anom_obs"] = hv["lst_obs"] - city_obs
    hv["anom_pred"] = pred_anom.reindex(hv["h3_id"].to_numpy()).to_numpy()
    hv = hv.dropna(subset=["anom_pred"]).reset_index(drop=True)
    hv["lst_pred"] = city_pred + hv["anom_pred"]

    err = hv["lst_obs"] - hv["lst_pred"]
    k = max(1, int(round(TOP_FRAC * len(hv))))
    top_pred = set(hv["lst_pred"].nlargest(k).index)
    top_obs = set(hv["lst_obs"].nlargest(k).index)
    row = {
        "date": day, "clear_scene": scene_valid >= L.MIN_SCENE_VALID, "scene_valid": round(scene_valid, 3),
        "n_hexes": len(hv), "train_dates": int(len(city)), "predictor": "+".join(best["cols"]),
        "air_temp_pass_c": round(p["t_pass"], 1),
        "city_obs_c": round(city_obs, 2), "city_pred_c": round(float(city_pred), 2),
        "city_err_c": round(float(city_pred - city_obs), 2),
        "hex_rmse_c": float(np.sqrt(np.mean(err ** 2))),
        "hex_rmse_nopattern_c": float(np.sqrt(np.mean((hv["lst_obs"] - city_pred) ** 2))),
        "hex_mae_c": float(np.mean(np.abs(err))),
        "anom_rmse_c": float(np.sqrt(np.mean((hv["anom_obs"] - hv["anom_pred"]) ** 2))),
        "anom_r2": float(L.r2(hv["anom_obs"].to_numpy(), hv["anom_pred"].to_numpy())),
        "anom_spearman": spearman(hv["anom_obs"].to_numpy(), hv["anom_pred"].to_numpy()),
        "top10_found": len(top_pred & top_obs) / k,
    }
    return row, hv[["h3_id", "lst_obs", "lst_pred", "anom_obs", "anom_pred"]].round(3), None


def show(r):
    note = "" if r["clear_scene"] else "   (scene below the clear-pixel threshold used for training; used only as a test)"
    print(f"\n{r['date']}   clear pixels {r['scene_valid']:.0%}, {r['n_hexes']:,} hexes compared{note}")
    print(f"  City mean:   Landsat {r['city_obs_c']:.1f} C | model {r['city_pred_c']:.1f} C | error {r['city_err_c']:+.1f} C "
          f"(air temperature around the pass {r['air_temp_pass_c']:.1f} C)")
    print(f"  Hex pattern: R2 {r['anom_r2']:.2f}, rank correlation {r['anom_spearman']:.2f}, RMSE {r['anom_rmse_c']:.2f} C")
    print(f"  Hex LST:     RMSE {r['hex_rmse_c']:.2f} C, MAE {r['hex_mae_c']:.2f} C "
          f"(a map with no pattern, every hex at the city mean, would be {r['hex_rmse_nopattern_c']:.2f} C)")
    print(f"  Hottest {TOP_FRAC:.0%} of hexes: the model finds {r['top10_found']:.0%} of Landsat's (chance: {TOP_FRAC:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS, help="dates to report in detail, YYYY-MM-DD")
    ap.add_argument("--model", default="ridge", help="ridge (default) or gboost, if scikit-learn works here")
    ap.add_argument("--no-coords", action="store_true", help="leave latitude/longitude out of the features")
    ap.add_argument("--only-named", action="store_true", help="skip the all-dates summary (faster)")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if args.model not in L.make_models():
        sys.exit(f"Model '{args.model}' is not available here. Choices: {list(L.make_models())}")

    P = L.PROCESSED
    grid = pd.read_csv(L.need(P / "chennai_h3_res9.csv"))
    lst = pd.read_csv(L.need(P / "chennai_hex_lst.csv"))
    s2 = pd.read_csv(L.need(P / "chennai_hex_ndvi_s2.csv"))
    lc = pd.read_csv(L.need(P / "chennai_hex_landcover.csv"))
    ex = pd.read_csv(L.need(P / "chennai_hex_exposure.csv"))
    hourly = pd.read_csv(L.need(L.RAW / "chennai_era5_hourly.csv"), parse_dates=["time"])
    lst["day"] = lst["date"].astype(str).str[:10]
    feat, cols = L.build_features(grid, lst, s2, lc, ex, not args.no_coords)

    scene = lst.groupby("day")["lst_valid_frac"].mean()
    clear = list(scene[scene >= L.MIN_SCENE_VALID].index)
    todo = list(dict.fromkeys(list(args.days) + ([] if args.only_named else clear)))
    print(f"{len(lst['day'].unique())} Landsat dates in the table, {len(clear)} clear | model: {args.model} | "
          f"testing {len(todo)} dates, each with that date left out")

    rows, saved = [], []
    for day in todo:
        row, hexes, why = backtest_date(day, lst, feat, cols, hourly, args.model)
        if row is None:
            print(f"\n{day}: skipped, {why}")
            if day in args.days:
                idx = pd.to_datetime(scene.index)
                near = scene[np.abs((idx - pd.Timestamp(day)).days) <= 20]
                if len(near):
                    print("   Landsat dates within 20 days of it: " +
                          ", ".join(f"{d} ({v:.0%} clear)" for d, v in near.items()))
            continue
        row["named"] = day in args.days
        rows.append(row)
        if row["named"]:
            show(row)
            dest = P / f"chennai_backtest_{day}.csv"
            hexes.to_csv(dest, index=False)
            saved.append(dest)
    if not rows:
        sys.exit("\nNothing could be tested.")

    res = pd.DataFrame(rows)
    allc = res[res["clear_scene"]]
    summary = []
    if len(allc) >= 3:
        print(f"\nAll {len(allc)} clear dates, each tested with that date left out:")
        s = {
            "dates_tested": len(allc),
            "city_err_rmse_c": float(np.sqrt((allc["city_err_c"] ** 2).mean())),
            "city_err_mean_c": float(allc["city_err_c"].mean()),
            "city_err_abs_max_c": float(allc["city_err_c"].abs().max()),
            "hex_rmse_median_c": float(allc["hex_rmse_c"].median()),
            "hex_rmse_nopattern_median_c": float(allc["hex_rmse_nopattern_c"].median()),
            "anom_r2_median": float(allc["anom_r2"].median()),
            "anom_spearman_median": float(allc["anom_spearman"].median()),
            "top10_found_median": float(allc["top10_found"].median()),
            "top10_found_worst": float(allc["top10_found"].min()),
        }
        print(f"  City-mean error: RMSE {s['city_err_rmse_c']:.2f} C, mean {s['city_err_mean_c']:+.2f} C, "
              f"worst {s['city_err_abs_max_c']:.1f} C")
        print(f"  Hex LST RMSE: median {s['hex_rmse_median_c']:.2f} C (no-pattern map: {s['hex_rmse_nopattern_median_c']:.2f} C)")
        print(f"  Hex pattern: R2 median {s['anom_r2_median']:.2f}, rank correlation median {s['anom_spearman_median']:.2f}")
        print(f"  Hottest 10% found: median {s['top10_found_median']:.0%}, worst {s['top10_found_worst']:.0%} (chance 10%)")
        summary = list(s.items())

    L.REPORTS.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(L.REPORTS / "lst_backtest.csv", index=False)
    if summary:
        pd.DataFrame(summary, columns=["metric", "value"]).round(4).to_csv(L.REPORTS / "lst_backtest_summary.csv", index=False)
    print(f"\nSaved {L.REPORTS / 'lst_backtest.csv'}" + (f", {L.REPORTS / 'lst_backtest_summary.csv'}" if summary else "")
          + (f" and {len(saved)} per-hex file(s) in {P}" if saved else ""))

    if args.upload:
        for f in saved:
            cmd = ["aws", "s3", "cp", str(f), L.S3_PREFIX + f.name, "--profile", L.PROFILE]
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as err:
                print("Upload problem:", err, "\n  run: " + " ".join(cmd))
    print("\nDone.")


if __name__ == "__main__":
    main()