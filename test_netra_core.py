"""
Tests for the NETRA core (database, metrics, sessions, run control, exports) - no Qt, no MuJoCo needed.

A fake run_policy_gui.py prints the same TEL / EP protocol as the real one, so the real RunController,
ExperimentSession, parser, database and exporters are exercised against a genuine child process.

    python test_netra_core.py
"""
import csv
import json
import os
import queue
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_database import DatabaseError, ExperimentDatabase                      # noqa: E402
from experiment_manager import prepare_launch                                          # noqa: E402
from metrics_engine import compare, compute_metrics, mode_summary, verify_consistency  # noqa: E402
from report_exporter import build_report, export_episodes_csv, export_report_json     # noqa: E402
from simulation_manager import RunController                                           # noqa: E402

FAKE = r'''
import json, os, sys, time, argparse
ap = argparse.ArgumentParser()
for f in ("--run","--x2","--arena","--layout"): ap.add_argument(f)
ap.add_argument("--goal", type=int, default=0); ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--speed"); ap.add_argument("--tel-every"); ap.add_argument("--no-reflex", action="store_true")
ap.add_argument("--headless", action="store_true")
a = ap.parse_args()
plan = json.loads(os.environ.get("FAKE_PLAN", '["SUCCESS","COLLISION","TIMEOUT","SUCCESS","SUCCESS"]'))
steps, delay = int(os.environ.get("FAKE_STEPS", "5")), float(os.environ.get("FAKE_DELAY", "0.01"))
crash_at, exit_after = int(os.environ.get("FAKE_CRASH_AT", "0")), int(os.environ.get("FAKE_EXIT_AFTER", "0"))
print("loaded fake run", flush=True)
for ep in range(1, a.episodes + 1):
    goal = a.goal or (ep % 6) + 1
    for k in range(1, steps + 1):
        tel = dict(ep=ep, goal=goal, t=k*0.1, path=k*0.05, x=k*0.1, y=0.0, z=1.0, speed=0.5, vz=0.0, heading=0.0,
                   goal_dist=5.0-k*0.1, min_range=round(2.0 - 0.1*k - 0.05*ep, 3), ranges=[2.0]*12,
                   alpha=0.2 if not a.no_reflex else 0.0, a=[0.1*k,0.0,0.0], ap=[0.2*k,0.0,0.0])
        print("TEL " + json.dumps(tel), flush=True)
        time.sleep(delay)
        if crash_at == ep and k == steps:
            print("Traceback: simulated MuJoCo crash", flush=True); os._exit(3)
    res = plan[(ep - 1) % len(plan)]
    print("EP " + json.dumps(dict(ep=ep, goal=goal, result=res, time=12.5+ep, path=3.0+ep, left=0.0 if res=="SUCCESS" else 2.5,
                                   reflex=0.25)), flush=True)
    if exit_after and ep == exit_after:
        sys.exit(0)
print("DONE {}", flush=True)
'''


def write(path, text, mode="w"):
    with open(path, mode, encoding=None if "b" in mode else "utf-8") as f:
        f.write(text)


class SubprocessRunner:
    """Behaves like the QProcess adapter: output/exit are queued and pumped on the 'GUI' (test) thread."""

    def __init__(self):
        self.q, self.proc, self.controller, self._gen = queue.Queue(), None, None, 0

    def start(self, python, args, cwd, env):
        import subprocess
        e = dict(os.environ)
        e["PYTHONPATH"] = env["PYTHONPATH_PREFIX"] + os.pathsep + e.get("PYTHONPATH", "")
        self._gen += 1
        self.proc = subprocess.Popen([python] + args, cwd=cwd, env=e, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return True

    def _read(self, p):
        for line in p.stdout:
            self.q.put(("out", line))
        code = p.wait()
        p.stdout.close()
        self.q.put(("exit", code))

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.is_running():
            gen = self._gen
            self.proc.terminate()
            threading.Timer(1.5, lambda: self.proc.kill() if gen == self._gen and self.is_running() else None).start()

    def pump(self, until=lambda: False, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            try:
                kind, val = self.q.get(timeout=0.05)
            except queue.Empty:
                if until():
                    return True
                continue
            self.controller.on_output(val) if kind == "out" else self.controller.on_exit(val)
            if until():
                return True
        return until()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="netra_test_")
        self.db = ExperimentDatabase(os.path.join(self.tmp, "data", "test.db"))
        sim = os.path.join(self.tmp, "files_ppo")
        os.makedirs(sim)
        write(os.path.join(sim, "run_policy_gui.py"), FAKE)
        ns = os.path.join(self.tmp, "nidar_sim")
        os.makedirs(ns)
        for n, t in (("nidar_arena.xml", "<mujoco/>"), ("nidar_arena_layout.json", "{}")):
            write(os.path.join(ns, n), t)
        x2 = os.path.join(self.tmp, "x2.xml")
        write(x2, "<mujoco/>")
        self.cfg = dict(python=sys.executable, files_ppo=sim, nidar_sim=ns, x2=x2,
                        arena=os.path.join(ns, "nidar_arena.xml"), layout=os.path.join(ns, "nidar_arena_layout.json"),
                        goal=3, episodes=5, speed=1.0, headless=True)
        self.run_on = self.make_run("runs/nidar")
        self.run_off = self.make_run("runs/no_reflex")
        for k in ("FAKE_PLAN", "FAKE_STEPS", "FAKE_DELAY", "FAKE_CRASH_AT", "FAKE_EXIT_AFTER"):
            os.environ.pop(k, None)
        self.events = []

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, rel):
        d = os.path.join(self.tmp, rel)
        os.makedirs(d)
        with zipfile.ZipFile(os.path.join(d, "ppo_nidar.zip"), "w") as z:
            z.writestr("data", json.dumps({"policy_class": {":type:": "<class 'MlpPolicy'>"},
                                           "_stable_baselines3_version": "2.x"}))
            z.writestr("policy.pth", b"weights" * 20)
        with open(os.path.join(d, "vecnorm.pkl"), "wb") as f:
            f.write(b"\x80\x04" + b"x" * 200)
        return d

    def controller(self, mode):
        runner = SubprocessRunner()
        c = RunController(self.db, mode, runner,
                          log=lambda m: self.events.append(("log", m)), error=lambda m: self.events.append(("err", m)),
                          episode=lambda r, s: self.events.append(("ep", r["episode_number"], r["outcome"], s)),
                          state=lambda s, m: self.events.append(("state", s)))
        runner.controller = c
        return c, runner

    def run_to_end(self, mode="ON", run=None, **over):
        cfg = dict(self.cfg, **over)
        c, r = self.controller(mode)
        self.assertTrue(c.launch(cfg, run or (self.run_on if mode == "ON" else self.run_off)))
        self.assertTrue(r.pump(lambda: not c.is_running() and r.q.empty()))
        return c, r


class TestCore(Base):
    def test_1_reflex_on_every_episode_stored(self):
        c, _ = self.run_to_end("ON")
        eps = self.db.get_episodes(experiment_ids=[c.session.experiment_id])
        self.assertEqual([e["episode_number"] for e in eps], [1, 2, 3, 4, 5])
        self.assertEqual([e["outcome"] for e in eps], ["SUCCESS", "COLLISION", "TIMEOUT", "SUCCESS", "SUCCESS"])
        self.assertEqual(self.db.get_experiment(c.session.experiment_id)["status"], "COMPLETED")
        e1 = eps[0]
        self.assertEqual((e1["success"], e1["collision"], e1["status"], e1["goal_id"]), (1, 0, "COMPLETED", 3))
        self.assertAlmostEqual(e1["reflex_activation_pct"], 25.0)
        self.assertAlmostEqual(e1["min_clearance_m"], 1.45, places=2)           # min over TEL samples of ep 1
        self.assertEqual(e1["telemetry_samples"], 5)
        self.assertEqual(json.loads(e1["fused_actions_json"])["last"], [0.5, 0.0, 0.0])
        self.assertEqual(json.loads(e1["policy_actions_json"])["last"], [1.0, 0.0, 0.0])
        self.assertEqual(e1["termination_reason"], "goal_reached")
        self.assertEqual(eps[1]["termination_reason"], "collision")
        self.assertIsNone(e1["remaining_survivors"])

    def test_2_reflex_off_separate_and_null_reflex_pct(self):
        a, _ = self.run_to_end("ON")
        b, _ = self.run_to_end("OFF")
        self.assertNotEqual(a.session.experiment_id, b.session.experiment_id)
        off = self.db.get_episodes(experiment_ids=[b.session.experiment_id])
        self.assertTrue(all(e["reflex_mode"] == "OFF" and e["reflex_activation_pct"] is None for e in off))
        self.assertEqual(len(self.db.get_episodes(reflex="ON")), 5)
        self.assertEqual(len(self.db.get_episodes(reflex="OFF")), 5)
        x = self.db.get_experiment(b.session.experiment_id)
        self.assertEqual(x["policy_path"], os.path.join(self.run_off, "ppo_nidar.zip"))
        self.assertEqual(len(x["policy_sha256"]), 64)
        self.assertEqual(json.loads(x["config_json"])["reflex_flag"], "--no-reflex")

    def test_3_persistence_across_restart(self):
        c, _ = self.run_to_end("ON")
        eid = c.session.experiment_id
        self.db.close()
        db2 = ExperimentDatabase(self.db.path)
        self.addCleanup(db2.close)
        self.assertEqual(len(db2.get_episodes(experiment_ids=[eid])), 5)
        self.assertEqual(db2.get_experiment(eid)["status"], "COMPLETED")
        db2.set_settings({"files_ppo": "C:/x", "episodes": 7})
        self.assertEqual(db2.all_settings()["episodes"], "7")
        self.db = db2

    def test_4_metrics_match_outcomes(self):
        c, _ = self.run_to_end("ON", episodes=5)
        m = compute_metrics(self.db.get_episodes())
        self.assertEqual((m["episodes_completed"], m["successes"], m["collisions"], m["timeouts"], m["failures"]),
                         (5, 3, 1, 1, 2))
        self.assertAlmostEqual(m["success_rate"], 60.0)
        self.assertAlmostEqual(m["collision_rate"], 20.0)
        self.assertAlmostEqual(m["timeout_rate"], 20.0)
        self.assertAlmostEqual(m["mean_duration_s"], sum(12.5 + i for i in range(1, 6)) / 5)
        self.assertEqual(m["mean_clearance_m_n"], 5)
        self.assertTrue(m["small_sample"])
        self.assertTrue(verify_consistency(self.db.get_episodes()))
        on = mode_summary(self.db.list_experiments(), self.db.get_episodes(), "ON")
        off = mode_summary(self.db.list_experiments(), self.db.get_episodes(), "OFF")
        self.assertEqual((on["experiments"], off["experiments"], off["episodes_completed"]), (1, 0, 0))
        text = " ".join(compare(on, off))
        self.assertIn("Run both modes", text)
        self.assertNotIn("higher", text)                       # never claims an effect without data

    def test_5_restart_episode_keeps_history(self):
        os.environ["FAKE_DELAY"], os.environ["FAKE_STEPS"] = "0.05", "20"
        c, r = self.controller("ON")
        self.assertTrue(c.launch(dict(self.cfg, episodes=3), self.run_on))
        eid = c.session.experiment_id
        self.assertTrue(r.pump(lambda: len(self.db.get_episodes(experiment_ids=[eid])) >= 1))
        before = {e["episode_id"]: e for e in self.db.get_episodes(experiment_ids=[eid]) if e["status"] == "COMPLETED"}
        r.pump(lambda: c.session.has_inflight(), 10)
        c.request("restart_episode")
        self.assertTrue(r.pump(lambda: c.state == "RUNNING" and r.is_running() and
                               any(e[0] == "log" and "restarted" in e[1] for e in self.events), 15))
        os.environ["FAKE_DELAY"] = "0.005"                      # speed up the rest
        self.assertTrue(r.pump(lambda: not c.is_running() and r.q.empty(), 60))
        eps = self.db.get_episodes(experiment_ids=[eid])
        done = [e for e in eps if e["status"] == "COMPLETED"]
        self.assertEqual(len(done), 3)
        self.assertEqual(len([e for e in eps if e["outcome"] == "INTERRUPTED"]), 1)
        for k, old in before.items():                           # completed rows untouched
            self.assertEqual(next(e for e in eps if e["episode_id"] == k), old)
        self.assertEqual(len({e["episode_number"] for e in eps}), len(eps))      # no duplicate numbers
        self.assertEqual(self.db.get_experiment(eid)["status"], "COMPLETED")
        m = compute_metrics(eps)
        self.assertEqual((m["episodes_completed"], m["episodes_interrupted"]), (3, 1))   # interrupted excluded

    def test_6_restart_experiment_new_id(self):
        os.environ["FAKE_DELAY"], os.environ["FAKE_STEPS"] = "0.05", "20"
        c, r = self.controller("ON")
        c.launch(dict(self.cfg, episodes=4), self.run_on)
        old = c.session.experiment_id
        r.pump(lambda: c.session.has_inflight(), 10)
        c.request("restart_experiment")
        self.assertTrue(r.pump(lambda: c.is_running() and c.session.experiment_id != old and c.session.has_inflight(), 20))
        new = c.session.experiment_id
        self.assertNotEqual(old, new)
        self.assertEqual(self.db.get_experiment(old)["status"], "CANCELLED")
        self.assertEqual(self.db.get_experiment(new)["previous_experiment_id"], old)
        c.request("stop")
        r.pump(lambda: not c.is_running() and r.q.empty(), 20)
        self.assertEqual(self.db.get_experiment(new)["status"], "CANCELLED")

    def test_7_crash_keeps_previous_episodes_and_recovery(self):
        os.environ["FAKE_CRASH_AT"] = "3"
        c, _ = self.run_to_end("ON")
        eid = c.session.experiment_id
        eps = self.db.get_episodes(experiment_ids=[eid])
        self.assertEqual([e["outcome"] for e in eps], ["SUCCESS", "COLLISION", "OTHER_FAILURE"])
        self.assertIsNone(eps[2]["duration_s"])                 # partial values are not recorded as measurements
        self.assertIn("code 3", eps[2]["error_details"])
        self.assertIn("exit code 3", eps[2]["termination_reason"])
        self.assertEqual(self.db.get_experiment(eid)["status"], "FAILED")
        self.assertEqual(c.state, "FAILED")
        # unexpected app death: an experiment left RUNNING is marked INTERRUPTED on next start
        self.db.create_experiment({"experiment_id": "EXP-ORPHAN", "reflex_mode": "OFF", "configured_episodes": 5})
        self.db.save_episode({"episode_id": "EXP-ORPHAN-E0001", "experiment_id": "EXP-ORPHAN", "episode_number": 1,
                              "reflex_mode": "OFF", "outcome": "SUCCESS", "status": "COMPLETED", "success": 1,
                              "collision": 0})
        self.assertEqual(self.db.recover_orphans(), ["EXP-ORPHAN"])
        self.assertEqual(self.db.get_experiment("EXP-ORPHAN")["status"], "INTERRUPTED")
        self.assertEqual(len(self.db.get_episodes(experiment_ids=["EXP-ORPHAN"])), 1)

    def test_8_exports(self):
        a, _ = self.run_to_end("ON")
        b, _ = self.run_to_end("OFF")
        p = os.path.join(self.tmp, "out.csv")
        n = export_episodes_csv(p, self.db.get_episodes())
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual((n, len(rows)), (10, 10))
        self.assertEqual(rows[0]["reflex_mode"], "ON")
        self.assertEqual(next(r for r in rows if r["reflex_mode"] == "OFF")["reflex_activation_pct"], "")
        pj = os.path.join(self.tmp, "out.json")
        export_report_json(pj, self.db, [a.session.experiment_id, b.session.experiment_id])
        with open(pj, encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(len(rep["experiments"]), 2)
        self.assertEqual(rep["overall"]["episodes_completed"], 10)
        ex = rep["experiments"][0]
        self.assertEqual(len(ex["episodes"]), 5)
        self.assertEqual(ex["metrics"]["successes"], 3)
        self.assertIsInstance(ex["episodes"][0]["fused_actions"], dict)
        self.assertIn("policy_sha256", ex["experiment"])

    def test_9_missing_files_and_bad_paths(self):
        c, _ = self.controller("ON")
        bad = os.path.join(self.tmp, "runs", "empty")
        os.makedirs(bad)
        self.assertFalse(c.launch(self.cfg, bad))
        self.assertTrue(any("Policy file missing" in e[1] for e in self.events if e[0] == "err"))
        self.assertTrue(any("Normalization file missing" in e[1] for e in self.events if e[0] == "err"))
        self.assertFalse(c.launch(self.cfg, os.path.join(self.tmp, "nope")))
        os.remove(os.path.join(self.run_on, "vecnorm.pkl"))
        self.assertFalse(c.launch(self.cfg, self.run_on))
        # corrupt files
        write(os.path.join(self.run_on, "ppo_nidar.zip"), "not a zip")
        write(os.path.join(self.run_on, "vecnorm.pkl"), "tiny")
        plan = prepare_launch(self.cfg, self.run_on, True)
        self.assertFalse(plan.ok)
        self.assertTrue(any("not a valid zip" in e for e in plan.errors))
        self.assertTrue(any("empty / too small" in e for e in plan.errors))
        self.assertFalse(prepare_launch(dict(self.cfg, python="/no/such/python"), self.run_off, False).ok)
        self.assertEqual(self.db.counts(), (0, 0))               # refused launches create nothing
        # name heuristic
        self.assertTrue(prepare_launch(self.cfg, self.run_off, True).warnings)

    def test_10_stop_all_and_window_closed(self):
        os.environ["FAKE_DELAY"], os.environ["FAKE_STEPS"] = "0.05", "30"
        on, ron = self.controller("ON")
        off, roff = self.controller("OFF")
        on.launch(self.cfg, self.run_on)
        off.launch(self.cfg, self.run_off)
        self.assertTrue(ron.pump(lambda: on.session.has_inflight(), 10) and roff.pump(lambda: off.session.has_inflight(), 10))
        for c in (on, off):
            c.request("stop")
        self.assertTrue(ron.pump(lambda: not on.is_running() and ron.q.empty(), 20))
        self.assertTrue(roff.pump(lambda: not off.is_running() and roff.q.empty(), 20))
        for c in (on, off):
            self.assertEqual(self.db.get_experiment(c.session.experiment_id)["status"], "CANCELLED")
            self.assertEqual([e["outcome"] for e in self.db.get_episodes(experiment_ids=[c.session.experiment_id])],
                             ["INTERRUPTED"])
        # viewer window closed by the user: clean exit before all episodes were done
        os.environ.update(FAKE_DELAY="0.005", FAKE_STEPS="5", FAKE_EXIT_AFTER="2")
        c, _ = self.run_to_end("ON")
        self.assertEqual(self.db.get_experiment(c.session.experiment_id)["status"], "CANCELLED")
        self.assertEqual(len([e for e in self.db.get_episodes(experiment_ids=[c.session.experiment_id])
                              if e["status"] == "COMPLETED"]), 2)

    def test_11_reset_simulation_then_continue(self):
        os.environ["FAKE_DELAY"], os.environ["FAKE_STEPS"] = "0.03", "20"
        c, r = self.controller("ON")
        c.launch(dict(self.cfg, episodes=3), self.run_on)
        eid = c.session.experiment_id
        r.pump(lambda: c.session.has_inflight(), 10)
        c.request("reset_sim")
        r.pump(lambda: not c.is_running() and r.q.empty(), 20)
        self.assertEqual(c.state, "IDLE")
        self.assertTrue(c.can_continue())                        # experiment kept open, nothing auto-resumed
        self.assertFalse(c.is_running())
        self.assertEqual(self.db.get_experiment(eid)["status"], "RUNNING")
        n_before = len(self.db.get_episodes())
        os.environ["FAKE_DELAY"] = "0.005"
        self.assertTrue(c.continue_experiment())
        r.pump(lambda: not c.is_running() and r.q.empty(), 40)
        self.assertGreater(len(self.db.get_episodes()), n_before)
        self.assertEqual(self.db.get_experiment(eid)["status"], "COMPLETED")

    def test_12_duplicates_spool_and_clear_history(self):
        rec = {"episode_id": "E-1-E0001", "experiment_id": "E-1", "episode_number": 1, "reflex_mode": "ON",
               "outcome": "SUCCESS", "status": "COMPLETED", "success": 1, "collision": 0}
        self.db.create_experiment({"experiment_id": "E-1", "reflex_mode": "ON", "configured_episodes": 2})
        self.assertTrue(self.db.save_episode(rec))
        self.assertFalse(self.db.save_episode(dict(rec)))                   # retry does not duplicate
        self.assertEqual(len(self.db.get_episodes()), 1)
        # simulated write failure -> spooled, error raised, then recovered
        real = self.db._tx
        calls = {"n": 0}

        def flaky(fn):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise DatabaseError("disk I/O error (simulated)")
            return real(fn)
        self.db._tx = flaky
        rec2 = dict(rec, episode_id="E-1-E0002", episode_number=2)
        with self.assertRaises(DatabaseError):
            self.db.save_episode(rec2)
        self.db._tx = real
        self.assertEqual(self.db.spool_size(), 1)
        self.assertEqual(len(self.db.get_episodes()), 1)
        self.assertEqual(self.db.flush_spool(), 1)
        self.assertEqual((len(self.db.get_episodes()), self.db.spool_size()), (2, 0))
        # clear history: backup first, settings survive
        self.db.set_settings({"keep": "me"})
        bak = self.db.clear_history()
        self.assertTrue(os.path.isfile(bak))
        self.assertEqual(self.db.counts(), (0, 0))
        self.assertEqual(self.db.get_setting("keep"), "me")
        con = sqlite3.connect(bak)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 2)
        con.close()

    def test_13_filters_and_comparison_logic(self):
        self.run_to_end("ON", goal=2)
        self.run_to_end("OFF", goal=4)
        self.assertEqual(len(self.db.list_experiments(reflex="ON")), 1)
        self.assertEqual(len(self.db.list_experiments(goal=4)), 1)
        self.assertEqual(len(self.db.list_experiments(outcome="COLLISION")), 2)
        self.assertEqual(len(self.db.list_experiments(search="-OFF-")), 1)
        self.assertEqual(len(self.db.list_experiments(date_from="2999-01-01T00:00:00.000Z")), 0)
        # synthetic large samples: overlapping vs non-overlapping intervals
        mk = lambda s, n: {"episodes_completed": n, "success_rate": 100 * s / n,
                           "success_ci": compute_metrics([{"status": "COMPLETED", "outcome": "SUCCESS"}] * s +
                                                         [{"status": "COMPLETED", "outcome": "TIMEOUT"}] * (n - s))["success_ci"],
                           "collision_rate": 0.0, "collision_ci": (0.0, 3.0), "survivors_covered": [3]}
        self.assertIn("do not overlap", compare(mk(95, 100), mk(40, 100))[0])
        txt = " ".join(compare(mk(6, 10), mk(5, 10)))
        self.assertIn("does not support a performance difference", txt)
        self.assertIn("too few for firm conclusions", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
