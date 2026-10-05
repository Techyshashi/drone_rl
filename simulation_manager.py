"""
NETRA Command Center - simulator process control.

RunController   (no Qt)  one per reflex mode. Owns the current ExperimentSession and decides what every
                         process exit means: finished, stopped by the user, restarted, reset, window closed, crashed.
QtProcessRunner (Qt)     thin adapter that runs run_policy_gui.py in a QProcess and forwards output / exit.

The MuJoCo simulation itself lives in the child process (run_policy_gui.py, unchanged), so the GUI never blocks.
Because that process owns the simulator state, "restart episode" and "reset simulation" are implemented by
stopping the process (the unfinished episode is recorded as INTERRUPTED) and starting a fresh one.
"""
import os

from experiment_database import DatabaseError
from experiment_manager import ExperimentSession, prepare_launch
from telemetry_manager import StreamParser

# UI state -> label
STATE_LABEL = {"IDLE": "Idle", "RUNNING": "Running", "COMPLETED": "Completed", "CANCELLED": "Interrupted (stopped)",
               "INTERRUPTED": "Interrupted", "FAILED": "Failed"}
ACTION_REASON = {"stop": "stopped by user", "restart_episode": "episode restarted by user",
                 "reset_sim": "simulation reset by user", "restart_experiment": "experiment restarted by user",
                 "exit": "application closed"}


def build_env(cfg):
    base = cfg["nidar_sim"]
    return {"PYTHONPATH_PREFIX": base, "PYTHONUNBUFFERED": "1"}


def build_args(cfg, run_dir, mode, episodes):
    script = os.path.join(cfg["files_ppo"], "run_policy_gui.py")
    args = ["-u", script, "--run", run_dir, "--goal", str(int(cfg["goal"])), "--episodes", str(int(episodes)),
            "--x2", cfg["x2"], "--arena", cfg["arena"], "--layout", cfg["layout"], "--speed", str(cfg["speed"]),
            "--tel-every", "1"]
    if mode == "OFF":
        args.append("--no-reflex")
    if cfg["headless"]:
        args.append("--headless")
    return args


class RunController:
    """runner must provide: start(python, args, cwd, env) -> bool, stop(), is_running() -> bool."""

    def __init__(self, db, mode, runner, **cb):
        self.db, self.mode, self.runner = db, mode, runner
        noop = lambda *a, **k: None
        self.cb = {k: cb.get(k, noop) for k in ("log", "telemetry", "episode", "state", "error", "reset_live")}
        self.parser = StreamParser()
        self.session = None
        self.last_launch = None                 # (cfg, run_dir) of the latest launch -> used by restart experiment
        self.state = "IDLE"
        self._pending = None
        self._tail = []

    # ---- queries
    def is_running(self):
        return self.runner.is_running()

    def has_open_session(self):
        return self.session is not None and not self.session.closed

    def can_continue(self):
        return self.has_open_session() and not self.is_running() and self.session.remaining() > 0

    def _set_state(self, state, msg=""):
        self.state = state
        self.cb["state"](state, msg)

    # ---- launching
    def launch(self, cfg, run_dir, previous_id=None):
        """Start a NEW experiment (any still-open one is closed as CANCELLED)."""
        if self.is_running():
            return False
        if self.has_open_session():
            self.session.close("CANCELLED", "superseded by a new launch")
        plan = prepare_launch(cfg, run_dir, self.mode == "ON")
        for w in plan.warnings:
            self.cb["log"](f"WARNING: {w}")
        if not plan.ok:
            for e in plan.errors:
                self.cb["error"](e)
            self._set_state("FAILED", "launch refused: " + plan.errors[0])
            return False
        sess = ExperimentSession(self.db, self.mode, plan, cfg, previous_id, error_cb=self.cb["error"])
        try:
            sess.start()
        except DatabaseError as e:
            self.cb["error"](f"DATABASE ERROR - experiment not started: {e}")
            self._set_state("FAILED", "database error")
            return False
        self.session, self.last_launch = sess, (dict(cfg), run_dir)
        self.cb["log"](f"experiment {sess.experiment_id} created "
                       f"(policy sha256 {plan.policy_sha[:12]}, vecnorm sha256 {plan.vecnorm_sha[:12]})")
        return self._spawn()

    def continue_experiment(self):
        """Run the remaining episodes of the open experiment with its ORIGINAL configuration."""
        if not self.can_continue():
            return False
        return self._spawn()

    def _spawn(self):
        s = self.session
        remaining = s.begin_segment()
        if remaining <= 0:
            s.close("COMPLETED")
            self._set_state("COMPLETED", "all configured episodes already recorded")
            return False
        cfg = s.cfg
        args = build_args(cfg, s.plan.run_dir, self.mode, remaining)
        self._tail.clear()
        self.parser = StreamParser()
        self.cb["reset_live"]()
        self.cb["log"](f"> {cfg['python']} {' '.join(args)}")
        if not self.runner.start(cfg["python"], args, cfg["files_ppo"], build_env(cfg)):
            self.cb["error"]("simulator process failed to start (check the Python interpreter path)")
            s.close("FAILED", "process failed to start")
            self._set_state("FAILED", "process failed to start")
            return False
        self._set_state("RUNNING", f"experiment {s.experiment_id}")
        return True

    # ---- process events
    def on_output(self, text):
        for kind, payload in self.parser.feed(text):
            self._handle(kind, payload)

    def _handle(self, kind, payload):
        s = self.session
        if s is None:
            return
        if kind == "TEL":
            s.on_telemetry(payload)
            self.cb["telemetry"](payload)
        elif kind == "EP":
            rec, saved = s.on_episode(payload)
            self.cb["episode"](rec, saved)
        elif kind == "LOG":
            self._tail = (self._tail + [payload])[-15:]
            self.cb["log"](payload)

    def on_exit(self, code):
        for kind, payload in self.parser.flush():
            self._handle(kind, payload)
        s, action, self._pending = self.session, self._pending, None
        if s is None or s.closed:
            return
        if action:
            self._apply_action(action)
            return
        if s.remaining() <= 0:
            s.close("COMPLETED")
            self._set_state("COMPLETED", f"{s.configured} episodes recorded")
        elif code == 0:
            s.interrupt("simulator window closed before the run finished")
            s.close("CANCELLED", "simulator window closed before all episodes finished")
            self._set_state("CANCELLED", "window closed - unfinished episode recorded as INTERRUPTED")
        else:
            err = f"simulator exited with code {code}: " + (" | ".join(self._tail[-4:]) or "no output")
            s.interrupt(f"simulator crashed (exit code {code})", crashed=True, error=err)
            s.close("FAILED", err[:500])
            self.cb["error"](err)
            self._set_state("FAILED", err[:200])

    # ---- user actions
    def request(self, action):
        """action: stop | restart_episode | reset_sim | restart_experiment | exit"""
        if self.is_running():
            if self._pending is None or action == "exit":
                self._pending = action
                self.runner.stop()
            return
        if action == "restart_episode":
            return                                   # nothing running to restart
        self._apply_action(action)

    def _apply_action(self, action):
        s = self.session
        reason = ACTION_REASON[action]
        if s is not None and not s.closed:
            s.interrupt(reason)
        if action == "restart_episode":
            self.cb["log"](f"episode restarted - partial episode (if any) kept as INTERRUPTED; "
                           f"{s.remaining()} episode(s) still to run")
            self._spawn()
        elif action == "reset_sim":
            self.cb["reset_live"]()
            left = s.remaining() if s and not s.closed else 0
            self._set_state("IDLE", "simulation reset" + (f" - experiment still open, {left} episode(s) left" if left else ""))
        elif action == "stop":
            if s is not None and not s.closed:
                s.close("CANCELLED", reason)
            self._set_state("CANCELLED", reason)
        elif action == "restart_experiment":
            old = s.experiment_id if s else None
            if s is not None and not s.closed:
                s.close("CANCELLED", reason)
            if self.last_launch:
                cfg, run_dir = self.last_launch
                self.launch(cfg, run_dir, previous_id=old)
        elif action == "exit":
            if s is not None and not s.closed:
                s.close("INTERRUPTED", reason)
            self._set_state("INTERRUPTED", reason)


# ------------------------------------------------------------------------------------------------ Qt adapter
try:
    from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QTimer
except ImportError:                                           # allows importing the module without PyQt5
    QObject = None

if QObject is not None:
    class QtProcessRunner(QObject):
        """Runs the simulator in a QProcess; forwards stdout/stderr text and the exit code to a RunController."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.controller = None
            self._gen = 0                       # incremented per start; stale kill-timers must not hit a newer run
            self.proc = QProcess(self)
            self.proc.setProcessChannelMode(QProcess.MergedChannels)
            self.proc.readyReadStandardOutput.connect(self._read)
            self.proc.finished.connect(self._finished)
            self.proc.errorOccurred.connect(self._error)
            self._failed_to_start = False

        def start(self, python, args, cwd, env):
            e = QProcessEnvironment.systemEnvironment()
            old = e.value("PYTHONPATH", "")
            e.insert("PYTHONPATH", env["PYTHONPATH_PREFIX"] + (os.pathsep + old if old else ""))
            e.insert("PYTHONUNBUFFERED", env["PYTHONUNBUFFERED"])
            self.proc.setProcessEnvironment(e)
            self.proc.setWorkingDirectory(cwd)
            self._failed_to_start = False
            self._gen += 1
            self.proc.start(python, args)
            return self.proc.waitForStarted(5000)

        def is_running(self):
            return self.proc.state() != QProcess.NotRunning

        def stop(self, kill_after_ms=1500):
            """Polite terminate (closes the MuJoCo window), then kill if still alive."""
            if self.is_running():
                gen = self._gen
                self.proc.terminate()
                QTimer.singleShot(kill_after_ms, lambda: self.proc.kill()
                                  if (gen == self._gen and self.is_running()) else None)

        def kill_now(self, wait_ms=2000):
            if self.is_running():
                self.proc.kill()
                self.proc.waitForFinished(wait_ms)

        def _read(self):
            if self.controller:
                self.controller.on_output(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace"))

        def _finished(self, code, _status):
            if self.controller:
                self.controller.on_exit(code)

        def _error(self, err):
            if err == QProcess.FailedToStart and self.controller:
                self.controller.cb["error"]("process error: FailedToStart")
