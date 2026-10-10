#!/usr/bin/env python3
"""Explain why an H3 hex is predicted hotter or cooler than average Chennai.

The LST anomaly model is a linear ridge model. Feature contributions are exact
relative to the model's mean feature vector and sum to the prediction deviation.
Missing feature values are filled with the training median, as in training.

Usage from the project root:
  python src/explain_hotspot.py 89618c4d2abffff
  python src/explain_hotspot.py --top-risk 5
  python src/explain_hotspot.py 89618c4d2abffff --json

The explanation describes model associations, not proven causal effects.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "lst_anomaly.json"
FEATURES = ROOT / "data" / "processed" / "chennai_hex_features.csv"
RISK = ROOT / "data" / "processed" / "chennai_hex_risk.csv"

LABEL = {
    "built_frac": "built-up share",
    "tree_frac": "tree cover",
    "grass_shrub_frac": "grass and shrub cover",
    "crop_frac": "cropland share",
    "bare_frac": "bare-ground share",
    "water_frac": "water share",
    "bldg_cover_frac": "building footprint cover",
    "small_bldg_frac": "share of very small buildings (under 40 m2)",
    "nearest_park_m": "distance to the nearest park",
    "ndvi_mean": "greenness (NDVI)",
    "lat": "north-south position",
    "lon": "east-west position",
}
NEIGHBOUR = {"nb1": "immediate surroundings", "nb3": "wider neighbourhood"}

# Related features are strongly correlated; grouped totals are easier to interpret.
GROUP_OF = {
    "built_frac": "built-up surface and building density",
    "bldg_cover_frac": "built-up surface and building density",
    "small_bldg_frac": "built-up surface and building density",
    "bare_frac": "built-up surface and building density",
    "tree_frac": "vegetation and greenness",
    "grass_shrub_frac": "vegetation and greenness",
    "crop_frac": "vegetation and greenness",
    "ndvi_mean": "vegetation and greenness",
    "water_frac": "water and parks nearby",
    "nearest_park_m": "water and parks nearby",
    "lat": "position in the city (north-south / east-west trend)",
    "lon": "position in the city (north-south / east-west trend)",
}
PHRASE = {
    "built-up surface and building density": (
        "more built-up surface and building density than average",
        "less built-up surface and building density than average",
    ),
    "vegetation and greenness": (
        "less vegetation and greenness than average",
        "more vegetation and greenness than average",
    ),
    "water and parks nearby": (
        "little water or parkland nearby",
        "water or parkland nearby",
    ),
}


def label(name):
    base, _, suffix = name.rpartition("_")
    if suffix in NEIGHBOUR and base in LABEL:
        return f"{LABEL[base]} of the {NEIGHBOUR[suffix]}"
    return LABEL.get(name, name)


def phrase(group_name, up):
    options = PHRASE.get(group_name)
    return (options[0] if up else options[1]) if options else group_name


def group(name):
    base, _, suffix = name.rpartition("_")
    return GROUP_OF.get(base if suffix in NEIGHBOUR else name, "other")


def numeric(value):
    try:
        result = float(value)
        return None if result != result else result
    except (TypeError, ValueError):
        return None


class Explainer:
    def __init__(self, model_path=MODEL, features_path=FEATURES, risk_path=RISK):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Train src/lst_model.py or provide models/lst_anomaly.json."
            )
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if model.get("kind") != "ridge":
            raise ValueError(
                "This explainer requires the ridge model (models/lst_anomaly.json); "
                "a gradient-boosting model needs a different explainer."
            )
        self.model = model
        self.names = model["features"]

        with open(features_path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Feature file is empty: {features_path}")
        missing = [name for name in self.names if name not in rows[0]]
        if missing:
            raise ValueError(f"Feature table is missing model features: {missing}")
        self.rows = {row["h3_id"]: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("Feature table contains duplicate H3 IDs")

        self.z_mean = []
        for index, name in enumerate(self.names):
            zs = [self._z(row, index) for row in rows]
            self.z_mean.append(sum(zs) / len(zs))
        self.value_median = {
            name: statistics.median(
                [v for v in (numeric(row[name]) for row in rows) if v is not None] or [0.0]
            )
            for name in self.names
        }

        self.risk = {}
        if Path(risk_path).exists():
            with open(risk_path, newline="", encoding="utf-8-sig") as handle:
                risk_rows = list(csv.DictReader(handle))
            self.risk = {row["h3_id"]: row for row in risk_rows}

    def _value(self, row, index):
        value = numeric(row[self.names[index]])
        return self.model["median"][index] if value is None else value

    def _z(self, row, index):
        std = self.model["std"][index]
        if not std:
            raise ValueError(f"Model standard deviation is zero for {self.names[index]}")
        return (self._value(row, index) - self.model["mean"][index]) / std

    def predicted_anomaly(self, row):
        return (
            self.model["intercept"]
            + sum(
                self.model["coef"][j] * self._z(row, j)
                for j in range(len(self.names))
            )
            - self.model["centre"]
        )

    def risk_breakdown(self, h3_id):
        row = self.risk.get(h3_id)
        if row is None:
            return None
        parts = {
            key: numeric(row.get(key + "_score"))
            for key in ("hazard", "exposure", "vulnerability")
        }
        score = numeric(row.get("risk_score"))
        if any(value is None for value in parts.values()) or score is None:
            return None
        strongest = max(parts, key=parts.get)
        vulnerable = numeric(row.get("pop_vulnerable"))
        output = {
            "risk_score": round(score, 2),
            "risk_category": row.get("risk_category"),
            "hazard_score": round(parts["hazard"], 2),
            "exposure_score": round(parts["exposure"], 2),
            "vulnerability_score": round(parts["vulnerability"], 2),
            "pop_vulnerable": None if vulnerable is None else round(vulnerable),
            "strongest_component": strongest,
        }
        summary = (
            f"Risk score {output['risk_score']} ({output['risk_category']}) = "
            f"100 x hazard {output['hazard_score']} x exposure {output['exposure_score']} x "
            f"vulnerability {output['vulnerability_score']}; the strongest component is {strongest}"
        )
        if output["pop_vulnerable"] is not None and output["pop_vulnerable"] != 0:
            summary += f", with about {output['pop_vulnerable']:,} vulnerable residents."
        else:
            summary += "."
        output["summary"] = summary
        return output

    def explain(self, h3_id, top=4):
        row = self.rows.get(h3_id)
        if row is None:
            raise KeyError(f"{h3_id} is not in the hex feature table")

        contributions = []
        for j, name in enumerate(self.names):
            contribution = self.model["coef"][j] * (self._z(row, j) - self.z_mean[j])
            contributions.append({
                "feature": name,
                "label": label(name),
                "value": round(self._value(row, j), 3),
                "city_median": round(self.value_median[name], 3),
                "contribution_c": round(contribution, 3),
            })
        contributions.sort(key=lambda item: -abs(item["contribution_c"]))

        predicted = self.predicted_anomaly(row)
        predicted_average = sum(self.predicted_anomaly(r) for r in self.rows.values()) / len(self.rows)
        difference = predicted - predicted_average

        groups = {}
        for item in contributions:
            item["group"] = group(item["feature"])
            groups.setdefault(item["group"], []).append(item)
        grouped = [
            {
                "group": group_name,
                "contribution_c": round(sum(x["contribution_c"] for x in items), 3),
                "features": items[:top],
            }
            for group_name, items in groups.items()
        ]
        grouped.sort(key=lambda item: -abs(item["contribution_c"]))

        pushed_up = [item for item in grouped if item["contribution_c"] > 0.1]
        pushed_down = [item for item in grouped if item["contribution_c"] < -0.1]
        direction = "hotter" if difference > 0 else "cooler"
        details = []
        if pushed_up:
            details.append(
                "pushed up by " + ", ".join(
                    f"{phrase(item['group'], True)} ({item['contribution_c']:+.1f} C)"
                    for item in pushed_up[:3]
                )
            )
        if pushed_down:
            details.append(
                "held down by " + ", ".join(
                    f"{phrase(item['group'], False)} ({item['contribution_c']:+.1f} C)"
                    for item in pushed_down[:3]
                )
            )
        summary = (
            f"Predicted to run {abs(difference):.1f} C {direction} than the average Chennai hex "
            "on a typical clear morning"
        )
        if details:
            summary += ": " + "; ".join(details)
        summary += "."

        risk = self.risk_breakdown(h3_id)
        if risk:
            summary += " " + risk["summary"]

        return {
            "h3_id": h3_id,
            "risk": risk,
            "predicted_anomaly_c": round(predicted, 2),
            "vs_average_hex_c": round(difference, 2),
            "sum_of_contributions_c": round(sum(item["contribution_c"] for item in contributions), 2),
            "groups": grouped,
            "summary": summary,
            "note": (
                "Morning (~10:30) land-surface pattern from the ridge model; typical hex-level error "
                "about 1.3 C. Related terms (a hex and its neighbours) can offset each other, so read "
                "the group totals. The explanation shows what the model associates with heat, not proven causes."
            ),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hexes", nargs="*", help="H3 IDs")
    parser.add_argument("--top-risk", type=int, default=0, help="Explain the N highest-risk hexes")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    ids = list(args.hexes)
    if args.top_risk:
        if not RISK.exists():
            raise SystemExit(f"Missing {RISK}")
        with open(RISK, newline="", encoding="utf-8-sig") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: -(numeric(r.get("risk_score")) or 0))
        ids.extend(row["h3_id"] for row in rows[:args.top_risk])
    if not ids:
        raise SystemExit("Give one or more H3 IDs, or --top-risk N")

    try:
        explainer = Explainer()
        results = [explainer.explain(h3_id) for h3_id in ids]
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"Could not explain hotspot: {exc}") from exc

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        return

    for result in results:
        print(f"\n{result['h3_id']}: {result['summary']}")
        print(
            f"  contributions add up to {result['sum_of_contributions_c']:+.2f} C = "
            f"model anomaly {result['predicted_anomaly_c']:+.2f} C "
            f"({result['vs_average_hex_c']:+.2f} C versus average hex)"
        )
        for group_item in result["groups"]:
            print(f"    {group_item['contribution_c']:+.2f} C  {group_item['group']}")
            for feature in group_item["features"][:3]:
                print(
                    f"         {feature['contribution_c']:+.2f} C  {feature['label']}: "
                    f"{feature['value']} (city median {feature['city_median']})"
                )
        print(f"  Note: {result['note']}")
    print("\nNote: correlated terms for a hex and its surroundings can offset each other; read group totals.")


if __name__ == "__main__":
    main()
