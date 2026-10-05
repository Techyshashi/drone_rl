"""
NETRA Command Center - experiment lifecycle (no Qt).

prepare_launch()   validates paths / policy files and snapshots the configuration (with SHA-256 hashes)
ExperimentSession  owns one experiment: creates its row, turns EP / TEL data into episode rows, records
                   interrupted / crashed episodes, and closes the experiment with the right status.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone

from experiment_database import DatabaseError, NETRA_VERSION, utc_now
from telemetry_manager import EpisodeAccumulator

POLICY_FILE, VECNORM_FILE = "ppo_nidar.zip", "vecnorm.pkl"
TERMINATION = {"SUCCESS": "goal_reached", "COLLISION": "collision", "TIMEOUT": "time_limit"}


# ---------------------------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(directory):
    try:
        flags = 0x08000000 if sys.platform.startswith("win") else 0         # CREATE_NO_WINDOW
        out = subprocess.run(["git", "-C", directory, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=3, creationflags=flags)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def new_experiment_id(mode):
    return f"EXP-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{mode}-{uuid.uuid4().hex[:4]}"


def inspect_policy_zip(path):
    """Returns (errors, info). Checks it is an intact Stable-Baselines3 zip (policy weights + data)."""
    errs, info = [], {}
    if not zipfile.is_zipfile(path):
        return [f"{os.path.basename(path)} is not a valid zip archive"], info
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                errs.append(f"{os.path.basename(path)} is corrupt (bad member {bad})")
            names = set(z.namelist())
            for need in ("data", "policy.pth"):
                if need not in names:
                    errs.append(f"{os.path.basename(path)} is not a Stable-Baselines3 model (no '{need}' entry)")
            if "data" in names and not errs:
                try:
                    data = json.loads(z.read("data").decode("utf-8", "replace"))
                    info["policy_class"] = str(data.get("policy_class", {}).get(":type:", ""))
                    info["sb3_version"] = data.get("_stable_baselines3_version")
                except ValueError:
                    pass
    except (OSError, zipfile.BadZipFile) as e:
        errs.append(f"cannot read {os.path.basename(path)}: {e}")
    return errs, info


def inspect_vecnorm(path):
    errs = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(2)
        if size < 64:
            errs.append(f"{os.path.basename(path)} is empty / too small ({size} bytes)")
        elif head[:1] != b"\x80":
            errs.append(f"{os.path.basename(path)} does not look like a pickle file")
    except OSError as e:
        errs.append(f"cannot read {os.path.basename(path)}: {e}")
    return errs


class LaunchPlan:
    def __init__(self):
        self.errors, self.warnings = [], []
        self.policy_path = self.vecnorm_path = self.run_dir = None
        self.policy_sha = self.vecnorm_sha = None
        self.config = {}

    @property
    def ok(self):
        return not self.errors


def prepare_launch(cfg, run_dir, reflex_on):
    """Validate everything needed to start a run and build the configuration snapshot to be stored."""
    plan = LaunchPlan()
    plan.run_dir = run_dir
    script = os.path.join(cfg["files_ppo"], "run_policy_gui.py")
    plan.policy_path = os.path.join(run_dir, POLICY_FILE)
    plan.vecnorm_path = os.path.join(run_dir, VECNORM_FILE)

    if not os.path.isdir(run_dir):
        plan.errors.append(f"Run folder does not exist: {run_dir}")
    for label, p in (("run_policy_gui.py", script), ("x2.xml", cfg["x2"]), ("arena xml", cfg["arena"]),
                     ("arena layout json", cfg["layout"])):
        if not os.path.isfile(p):
            plan.errors.append(f"{label} not found: {p}")
    py = cfg["python"]
    if not (os.path.isfile(py) or shutil.which(py)):
        plan.errors.append(f"Python interpreter not found: {py}")
    if int(cfg["episodes"]) < 1:
        plan.errors.append("Episode count must be at least 1")

    pol_ok = os.path.isfile(plan.policy_path)
    vec_ok = os.path.isfile(plan.vecnorm_path)
    if not pol_ok:
        plan.errors.append(f"Policy file missing: {plan.policy_path}")
    else:
        e, info = inspect_policy_zip(plan.policy_path)
        plan.errors += e
        plan.config["policy_info"] = info
    if not vec_ok:
        plan.errors.append(f"Normalization file missing: {plan.vecnorm_path} (the policy cannot be run without "
                           "the statistics it was trained with)")
    else:
        plan.errors += inspect_vecnorm(plan.vecnorm_path)

    low = run_dir.lower().replace("-", "_")
    if reflex_on and "no_reflex" in low:
        plan.warnings.append("Reflex ON panel points at a folder whose name contains 'no_reflex'.")
    if not reflex_on and "no_reflex" not in low and "noreflex" not in low:
        plan.warnings.append("Reflex OFF panel folder name does not contain 'no_reflex' - check it holds the "
                             "policy trained WITHOUT the reflex.")

    if plan.ok:
        plan.policy_sha, plan.vecnorm_sha = sha256_file(plan.policy_path), sha256_file(plan.vecnorm_path)
        files = {}
        for key, p in (("x2_xml", cfg["x2"]), ("arena_xml", cfg["arena"]), ("arena_layout", cfg["layout"]),
                       ("run_policy_gui", script)):
            files[key] = {"path": p, "sha256": sha256_file(p)}
        plan.config.update({
            "run_policy_script": script,
            "python": py,
            "reflex_flag": "--no-reflex" if not reflex_on else None,
            "environment_files": files,
            "policy_size_bytes": os.path.getsize(plan.policy_path),
            "policy_mtime": datetime.fromtimestamp(os.path.getmtime(plan.policy_path), timezone.utc).isoformat(),
            "vecnorm_mtime": datetime.fromtimestamp(os.path.getmtime(plan.vecnorm_path), timezone.utc).isoformat(),
            "episode_time_limit_s": 240.0,                     # ep_seconds in run_policy_gui.py
            "spawn": "pad",
        })
    return plan


# ---------------------------------------------------------------------------------------------- session
class ExperimentSession:
    """One experiment = one reflex mode, one policy, one configuration. Survives process restarts
    (Restart Episode / Reset Simulation) so the same experiment can be continued."""

    def __init__(self, db, reflex_mode, plan, launch_cfg, previous_id=None, error_cb=None):
        self.db, self.mode, self.plan, self.cfg = db, reflex_mode, plan, dict(launch_cfg)
        self.previous_id = previous_id
        self.error_cb = error_cb or (lambda m: None)
        self.experiment_id = new_experiment_id(reflex_mode)
        self.configured = int(launch_cfg["episodes"])
        self.acc = EpisodeAccumulator()
        self.offset = 0
        self._last_number = 0
        self._mem_completed = 0          # completed episodes seen this session, even if a DB write failed
        self._last_touch = 0.0
        self.closed = False
        self.last_outcome = None

    # ---- lifecycle
    def start(self):
        cfg = self.cfg
        goal = int(cfg["goal"])
        snapshot = dict(self.plan.config)
        snapshot.update({"goal_setting": goal, "episodes": self.configured, "speed": cfg["speed"],
                         "headless": bool(cfg["headless"]), "software_version": NETRA_VERSION,
                         "reflex_mode": self.mode})
        self.db.create_experiment({
            "experiment_id": self.experiment_id, "reflex_mode": self.mode, "status": "RUNNING",
            "policy_path": self.plan.policy_path, "policy_file": os.path.basename(self.plan.policy_path),
            "policy_sha256": self.plan.policy_sha, "vecnorm_path": self.plan.vecnorm_path,
            "vecnorm_file": os.path.basename(self.plan.vecnorm_path), "vecnorm_sha256": self.plan.vecnorm_sha,
            "goal_setting": goal, "goal_label": "random each episode" if goal == 0 else f"survivor {goal}",
            "configured_episodes": self.configured, "sim_speed": float(cfg["speed"]),
            "headless": 1 if cfg["headless"] else 0, "software_version": NETRA_VERSION,
            "git_commit": git_commit(cfg["files_ppo"]), "config_json": json.dumps(snapshot),
            "previous_experiment_id": self.previous_id,
        })
        return self.experiment_id

    def remaining(self):
        try:
            done = max(self.db.count_completed(self.experiment_id), self._mem_completed)
        except DatabaseError:
            done = self._mem_completed
        return max(0, self.configured - done)

    def begin_segment(self):
        """Call before every (re)start of the simulator process. Local episode 1 maps to offset + 1."""
        try:
            self.offset = max(self.db.next_episode_number(self.experiment_id) - 1, self._last_number)
        except DatabaseError:
            self.offset = self._last_number
        self.acc.reset()
        return self.remaining()

    def close(self, status, reason=None):
        if self.closed:
            return
        self.closed = True
        try:
            self.db.set_experiment_status(self.experiment_id, status, reason)
        except DatabaseError as e:
            self.error_cb(f"could not set {self.experiment_id} to {status}: {e}")

    # ---- data in
    def on_telemetry(self, tel):
        self.acc.add(tel)
        now = datetime.now().timestamp()
        if now - self._last_touch > 5:
            self._last_touch = now
            self.db.touch_experiment(self.experiment_id)

    def _number(self, local_ep):
        n = self.offset + int(local_ep)
        self._last_number = max(self._last_number, n)
        return n

    def _base(self, number):
        return {
            "episode_id": f"{self.experiment_id}-E{number:04d}", "experiment_id": self.experiment_id,
            "episode_number": number, "reflex_mode": self.mode, "recorded_at": utc_now(),
            "remaining_survivors": None,
        }

    def _save(self, rec):
        try:
            self.db.save_episode(rec)
            return True
        except DatabaseError as e:
            self.error_cb(f"DATABASE WRITE ERROR: {e}")
            return False

    def on_episode(self, ep):
        """Handle an EP line. Returns (record, saved_flag)."""
        number = self._number(ep["ep"])
        res = str(ep.get("result", ""))
        outcome = res if res in TERMINATION else "OTHER_FAILURE"
        mine = self.acc.local_ep == ep["ep"] and self.acc.samples > 0
        rec = self._base(number)
        rec.update({
            "goal_id": ep.get("goal"), "outcome": outcome, "status": "COMPLETED",
            "success": 1 if outcome == "SUCCESS" else 0, "collision": 1 if outcome == "COLLISION" else 0,
            "duration_s": ep.get("time"), "path_length_m": ep.get("path"), "final_goal_dist_m": ep.get("left"),
            "reflex_activation_pct": (ep["reflex"] * 100.0) if (self.mode == "ON" and ep.get("reflex") is not None) else None,
            "min_clearance_m": self.acc.min_clearance if mine else None,
            "telemetry_samples": self.acc.samples if mine else None,
            "policy_actions_json": self.acc.policy_actions_json() if mine else None,
            "fused_actions_json": self.acc.fused_actions_json() if mine else None,
            "termination_reason": TERMINATION.get(outcome, f"unrecognised result {res!r}"),
            "started_at": self.acc.started_at if mine else None, "ended_at": utc_now(),
        })
        self.acc.reset()
        self.last_outcome = outcome
        self._mem_completed += 1
        return rec, self._save(rec)

    def has_inflight(self):
        return self.acc.active

    def interrupt(self, reason, crashed=False, error=None):
        """Record the episode that was running when the simulator stopped. Returns the record or None.
        crashed=False -> outcome INTERRUPTED (excluded from rates).  crashed=True -> OTHER_FAILURE with the
        error text; its partial duration / path / distance are left NULL so they cannot skew the means."""
        if not self.acc.active:
            return None
        a = self.acc
        rec = self._base(self._number(a.local_ep))
        partial = {"last_time_s": a.last_t, "last_path_m": a.last_path, "last_goal_dist_m": a.last_goal_dist}
        rec.update({
            "goal_id": a.goal, "min_clearance_m": a.min_clearance, "telemetry_samples": a.samples,
            "policy_actions_json": a.policy_actions_json(), "fused_actions_json": a.fused_actions_json(),
            "termination_reason": reason, "started_at": a.started_at, "ended_at": utc_now(),
        })
        if crashed:
            self._mem_completed += 1
            rec.update({"outcome": "OTHER_FAILURE", "status": "COMPLETED", "success": 0, "collision": 0,
                        "error_details": ((error or "simulator process crashed") + " | last known: "
                                          + json.dumps(partial))})
        else:
            rec.update({"outcome": "INTERRUPTED", "status": "INTERRUPTED", "success": None, "collision": None,
                        "duration_s": a.last_t, "path_length_m": a.last_path, "final_goal_dist_m": a.last_goal_dist})
        a.reset()
        self.last_outcome = rec["outcome"]
        self._save(rec)
        return rec
