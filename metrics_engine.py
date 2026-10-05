"""
NETRA Command Center - metrics computed from recorded episode rows (never from UI counters).

Definitions
  eligible episode : status == 'COMPLETED'  (interrupted / still-running episodes are excluded)
  success          : outcome == 'SUCCESS'
  collision        : outcome == 'COLLISION'
  timeout          : outcome == 'TIMEOUT'
  other failure    : outcome == 'OTHER_FAILURE' (simulator crash mid-episode or unrecognised result)
  failed           : eligible - success      (so success + collision + timeout + other == eligible)
  rate (%)         : count / eligible * 100  (None when eligible == 0)
  means            : averaged over the episodes that HAVE the measurement (NULLs are skipped, never read as 0);
                     the number of contributing episodes is reported next to each mean.
"""
import math

MIN_SAMPLE = 30          # below this many eligible episodes per mode, comparisons are flagged as inconclusive
MEAN_FIELDS = {
    "mean_duration_s": "duration_s",
    "mean_path_m": "path_length_m",
    "mean_final_dist_m": "final_goal_dist_m",
    "mean_clearance_m": "min_clearance_m",
    "mean_reflex_pct": "reflex_activation_pct",
}


def eligible(episodes):
    return [e for e in episodes if e.get("status") == "COMPLETED"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, as percentages. None if n == 0."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def compute_metrics(episodes):
    """Metrics for any list of episode dicts (any mix of experiments)."""
    el = eligible(episodes)
    n = len(el)
    cnt = {o: sum(1 for e in el if e["outcome"] == o) for o in ("SUCCESS", "COLLISION", "TIMEOUT", "OTHER_FAILURE")}
    m = {
        "episodes_recorded": len(episodes),
        "episodes_completed": n,
        "episodes_interrupted": sum(1 for e in episodes if e.get("status") == "INTERRUPTED"),
        "successes": cnt["SUCCESS"],
        "collisions": cnt["COLLISION"],
        "timeouts": cnt["TIMEOUT"],
        "other_failures": cnt["OTHER_FAILURE"],
        "failures": n - cnt["SUCCESS"],
    }
    rate = lambda c: (100.0 * c / n) if n else None
    m["success_rate"], m["collision_rate"], m["timeout_rate"] = rate(cnt["SUCCESS"]), rate(cnt["COLLISION"]), rate(cnt["TIMEOUT"])
    m["other_failure_rate"] = rate(cnt["OTHER_FAILURE"])
    m["success_ci"] = wilson(cnt["SUCCESS"], n)
    m["collision_ci"] = wilson(cnt["COLLISION"], n)
    m["timeout_ci"] = wilson(cnt["TIMEOUT"], n)
    for key, col in MEAN_FIELDS.items():
        m[key], m[key + "_n"] = _mean([e.get(col) for e in el])
    m["small_sample"] = n < MIN_SAMPLE
    return m


def mode_summary(experiments, episodes, mode):
    """Summary for one reflex mode: experiment count + metrics of that mode's episodes."""
    exps = [x for x in experiments if x["reflex_mode"] == mode]
    ids = {x["experiment_id"] for x in exps}
    eps = [e for e in episodes if e["reflex_mode"] == mode and e["experiment_id"] in ids]
    m = compute_metrics(eps)
    m["mode"] = mode
    m["experiments"] = len(exps)
    m["survivors_covered"] = sorted({e["goal_id"] for e in eps if e.get("goal_id") is not None})
    return m


def experiment_metrics(experiment, episodes):
    eps = [e for e in episodes if e["experiment_id"] == experiment["experiment_id"]]
    m = compute_metrics(eps)
    m["experiment_id"] = experiment["experiment_id"]
    return m


def verify_consistency(episodes):
    """True if every eligible episode is counted in exactly one outcome bucket (used by tests / self-check)."""
    m = compute_metrics(episodes)
    return (m["successes"] + m["collisions"] + m["timeouts"] + m["other_failures"] == m["episodes_completed"]
            and m["failures"] == m["collisions"] + m["timeouts"] + m["other_failures"])


def compare(on, off):
    """Plain-language findings. Only states a difference when the 95% intervals do not overlap."""
    lines = []
    for name, m in (("Reflex ON", on), ("Reflex OFF", off)):
        n = m["episodes_completed"]
        if n == 0:
            lines.append(f"{name}: no completed episodes recorded yet.")
        elif n < MIN_SAMPLE:
            lines.append(f"{name}: only {n} completed episodes - below {MIN_SAMPLE}, too few for firm conclusions.")
    if on["episodes_completed"] == 0 or off["episodes_completed"] == 0:
        lines.append("Run both modes to enable a comparison.")
        return lines
    a, b = on["success_ci"], off["success_ci"]
    d = on["success_rate"] - off["success_rate"]
    if a[0] > b[1]:
        lines.append(f"Success rate is higher with reflex ON ({d:+.1f} points); 95% intervals do not overlap.")
    elif b[0] > a[1]:
        lines.append(f"Success rate is higher with reflex OFF ({-d:+.1f} points for OFF); 95% intervals do not overlap.")
    else:
        lines.append(f"Success rates differ by {d:+.1f} points (ON minus OFF), but the 95% intervals overlap: "
                     "the recorded data does not support a performance difference.")
    c, e = on["collision_ci"], off["collision_ci"]
    dc = on["collision_rate"] - off["collision_rate"]
    if c[1] < e[0]:
        lines.append(f"Collision rate is lower with reflex ON ({dc:+.1f} points); intervals do not overlap.")
    elif e[1] < c[0]:
        lines.append(f"Collision rate is lower with reflex OFF ({-dc:+.1f} points for OFF); intervals do not overlap.")
    else:
        lines.append(f"Collision rates differ by {dc:+.1f} points (ON minus OFF); intervals overlap, no supported difference.")
    if len(on.get("survivors_covered", [])) > 1 or len(off.get("survivors_covered", [])) > 1 \
            or on.get("survivors_covered") != off.get("survivors_covered"):
        lines.append("Note: modes were tested on different / mixed survivors, which can bias the comparison.")
    return lines


def fmt(v, digits=1, suffix=""):
    return "n/a" if v is None else f"{v:.{digits}f}{suffix}"
