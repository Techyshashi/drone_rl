"""NETRA Command Center - CSV / JSON export. NULL measurements are exported as empty CSV cells / JSON null."""
import csv
import json
import os

from experiment_database import NETRA_VERSION, utc_now
from metrics_engine import compute_metrics, experiment_metrics

CSV_COLUMNS = [
    "experiment_id", "episode_number", "episode_id", "reflex_mode", "goal_id", "outcome", "status", "success",
    "collision", "duration_s", "path_length_m", "final_goal_dist_m", "remaining_survivors",
    "reflex_activation_pct", "min_clearance_m", "telemetry_samples", "termination_reason", "error_details",
    "started_at", "ended_at", "recorded_at", "policy_actions_json", "fused_actions_json",
]


def export_episodes_csv(path, episodes):
    """One row per episode. Returns the number of rows written."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for e in episodes:
            w.writerow({k: ("" if e.get(k) is None else e.get(k)) for k in CSV_COLUMNS})
    return len(episodes)


def _decode(rec):
    out = dict(rec)
    for k in ("policy_actions_json", "fused_actions_json", "config_json"):
        if k in out and isinstance(out[k], str):
            try:
                out[k.replace("_json", "")] = json.loads(out[k])
            except ValueError:
                out[k.replace("_json", "")] = None
            del out[k]
    return out


def build_report(db, experiment_ids):
    exps = db.list_experiments(experiment_ids=list(experiment_ids))
    eps = db.get_episodes(experiment_ids=[x["experiment_id"] for x in exps])
    report = {
        "generator": f"NETRA Command Center {NETRA_VERSION}",
        "exported_at": utc_now(),
        "metric_definitions": "success/collision/timeout rates = count / completed episodes; interrupted episodes "
                              "are excluded; means skip episodes where the measurement is null.",
        "overall": compute_metrics(eps),
        "experiments": [],
    }
    for x in sorted(exps, key=lambda r: r["started_at"]):
        mine = [e for e in eps if e["experiment_id"] == x["experiment_id"]]
        report["experiments"].append({
            "experiment": _decode(x),
            "metrics": experiment_metrics(x, eps),
            "episodes": [_decode(e) for e in sorted(mine, key=lambda r: r["episode_number"])],
        })
    return report


def export_report_json(path, db, experiment_ids):
    report = build_report(db, experiment_ids)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return len(report["experiments"])


def safe_default_name(prefix, ext):
    return f"{prefix}_{utc_now().replace(':', '').replace('-', '').replace('.', '_')}.{ext}"
