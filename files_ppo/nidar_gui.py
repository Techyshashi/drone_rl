"""
NIDAR launcher (PyQt5): start the trained policy in MuJoCo with the reflex ON or OFF, with live readings,
or fly the drone yourself. Every MuJoCo window runs as its own process, so the GUI never freezes.

Put nidar_gui.py and run_policy_gui.py in  ...\\new_try\\files_ppo  (next to nidar_env.py), then:
    pip install PyQt5
    python nidar_gui.py
"""
import json
import os
import sys
import time

from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QSettings, QTimer, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # ...\new_try
MONO = QFont("Consolas", 10)
GREEN, AMBER, RED = "#1D9E75", "#BA7517", "#E24B4A"


def kill_later(proc, ms=1500):
    """terminate politely (closes the window), then kill if it is still alive."""
    if proc.state() != QProcess.NotRunning:
        proc.terminate()
        QTimer.singleShot(ms, lambda: proc.kill() if proc.state() != QProcess.NotRunning else None)


def make_env(cfg):
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONPATH", cfg["nidar_sim"] + os.pathsep + env.value("PYTHONPATH", ""))
    env.insert("PYTHONUNBUFFERED", "1")
    return env


# ---------------------------------------------------------------------------------------------- radar
class Radar(QWidget):
    ANG = [-150 + 30 * i for i in range(12)]                   # same 12 rays as the policy; 0 = nose, + = left

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.r = [5.0] * 12

    def set_ranges(self, r):
        self.r = r
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        R = min(self.width(), self.height()) / 2 - 10
        ring = QColor(self.palette().mid().color())
        p.setPen(QPen(ring, 1))
        for f in (0.25, 0.5, 0.75, 1.0):
            p.drawEllipse(c, R * f, R * f)
        p.drawLine(c, QPointF(c.x(), c.y() - R))
        p.setPen(self.palette().text().color())
        p.drawText(QPointF(c.x() + 4, c.y() - R + 12), "5 m")
        import math
        pts = []
        for th, d in zip(self.ANG, self.r):
            t = math.radians(th)
            rr = min(d, 5.0) / 5.0 * R
            pts.append(QPointF(c.x() - math.sin(t) * rr, c.y() - math.cos(t) * rr))
        p.setPen(QPen(QColor(55, 138, 221), 1.5))
        p.setBrush(QColor(55, 138, 221, 60))
        p.drawPolygon(QPolygonF(pts))
        for pt, d in zip(pts, self.r):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(RED if d < 0.5 else AMBER if d < 1.0 else GREEN))
            p.drawEllipse(pt, 3.5, 3.5)
        p.setPen(Qt.NoPen)
        p.setBrush(self.palette().text().color())
        p.drawPolygon(QPolygonF([QPointF(c.x(), c.y() - 9), QPointF(c.x() - 6, c.y() + 7), QPointF(c.x() + 6, c.y() + 7)]))


# ---------------------------------------------------------------------------------------------- policy panel
class RunPanel(QGroupBox):
    log = pyqtSignal(str)

    def __init__(self, title, tag, reflex, default_run, get_cfg):
        super().__init__(title)
        self.tag, self.reflex, self.get_cfg = tag, reflex, get_cfg
        self.results, self._buf, self._bar_state = [], "", None
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(lambda e: self.log.emit(f"[{tag}] process error: {e}"))

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Run folder"))
        self.run_edit = QLineEdit(default_run)
        self.run_edit.textChanged.connect(self._file_info)
        row.addWidget(self.run_edit, 1)
        b = QPushButton("Browse")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setFont(QFont("Consolas", 8))
        lay.addWidget(self.info)

        row = QHBoxLayout()
        self.btn_go = QPushButton(f"Launch  (reflex {'ON' if reflex else 'OFF'})")
        self.btn_go.clicked.connect(self.launch)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        self.status = QLabel("idle")
        row.addWidget(self.btn_go)
        row.addWidget(self.btn_stop)
        row.addWidget(self.status, 1)
        lay.addLayout(row)

        body = QHBoxLayout()
        grid = QGridLayout()
        self.v = {}
        for r, (key, name) in enumerate([("pos", "position"), ("spd", "speed / heading"), ("goal", "goal"),
                                         ("time", "time / path"), ("pol", "policy action"),
                                         ("fus", "fused action")]):
            grid.addWidget(QLabel(name), r, 0)
            lb = QLabel("-")
            lb.setFont(MONO)
            grid.addWidget(lb, r, 1)
            self.v[key] = lb
        self.bar_min, self.bar_ref = QProgressBar(), QProgressBar()
        for bar in (self.bar_min, self.bar_ref):
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
        self.bar_min.setFormat("nearest wall -")
        self.bar_ref.setFormat("reflex -")
        grid.addWidget(QLabel("clearance"), 6, 0)
        grid.addWidget(self.bar_min, 6, 1)
        grid.addWidget(QLabel("reflex now"), 7, 0)
        grid.addWidget(self.bar_ref, 7, 1)
        grid.setRowStretch(8, 1)
        body.addLayout(grid, 1)
        self.radar = Radar()
        body.addWidget(self.radar)
        lay.addLayout(body)

        self.summary = QLabel("no episodes yet")
        lay.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "survivor", "result", "time s", "path m", "left m", "reflex %"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(110)
        lay.addWidget(self.table)
        self._file_info()

    # ---- helpers
    def run_dir(self):
        d = self.run_edit.text().strip()
        return d if os.path.isabs(d) else os.path.join(self.get_cfg()["files_ppo"], d)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with ppo_nidar.zip + vecnorm.pkl", self.run_dir())
        if d:
            self.run_edit.setText(d)

    def _file_info(self):
        parts = []
        for f in ("ppo_nidar.zip", "vecnorm.pkl"):
            p = os.path.join(self.run_dir(), f)
            if os.path.isfile(p):
                parts.append(f"{f} {os.path.getsize(p) / 1024:.0f} KB, "
                             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}")
            else:
                parts.append(f"{f} MISSING")
        self.info.setText("   |   ".join(parts))

    def running(self):
        return self.proc.state() != QProcess.NotRunning

    def _set_running(self, on):
        self.btn_go.setEnabled(not on)
        self.btn_stop.setEnabled(on)

    # ---- start / stop
    def launch(self):
        if self.running():
            return
        cfg = self.get_cfg()
        self._file_info()
        run = self.run_dir()
        script = os.path.join(cfg["files_ppo"], "run_policy_gui.py")
        missing = [p for p in (os.path.join(run, "ppo_nidar.zip"), os.path.join(run, "vecnorm.pkl"), script,
                               cfg["x2"], cfg["arena"], cfg["layout"]) if not os.path.isfile(p)]
        if missing:
            QMessageBox.warning(self, "Missing files", "Not found:\n" + "\n".join(missing))
            return
        args = ["-u", script, "--run", run, "--goal", str(cfg["goal"]), "--episodes", str(cfg["episodes"]),
                "--x2", cfg["x2"], "--arena", cfg["arena"], "--layout", cfg["layout"], "--speed", str(cfg["speed"])]
        if not self.reflex:
            args.append("--no-reflex")
        if cfg["headless"]:
            args.append("--headless")
        self.results, self._buf = [], ""
        self.table.setRowCount(0)
        self.summary.setText("running...")
        for lb in self.v.values():
            lb.setText("-")
        self.proc.setProcessEnvironment(make_env(cfg))
        self.proc.setWorkingDirectory(cfg["files_ppo"])
        self.log.emit(f"[{self.tag}] > {cfg['python']} {' '.join(args)}")
        self.proc.start(cfg["python"], args)
        self.status.setText("starting...")
        self._set_running(True)

    def stop(self):
        kill_later(self.proc)

    def _finished(self, code, _):
        self._flush()
        self._set_running(False)
        self.status.setText(f"finished (exit code {code})" if code == 0 else f"stopped / crashed (exit code {code})")
        if code != 0:
            self.log.emit(f"[{self.tag}] exit code {code} - see the messages above")

    # ---- output parsing
    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self._line(ln.rstrip("\r"))

    def _flush(self):
        self._read()
        if self._buf.strip():
            self._line(self._buf.strip())
        self._buf = ""

    def _line(self, ln):
        if not ln:
            return
        try:
            if ln.startswith("TEL "):
                return self._tel(json.loads(ln[4:]))
            if ln.startswith("EP "):
                return self._ep(json.loads(ln[3:]))
            if ln.startswith("DONE"):
                return
        except (ValueError, KeyError):
            pass
        self.log.emit(f"[{self.tag}] {ln}")

    @staticmethod
    def _act(a):
        return f"F{a[0]:+.2f}  L{a[1]:+.2f}  Y{a[2]:+.2f}"

    def _tel(self, d):
        n = self.get_cfg()["episodes"]
        self.status.setText(f"running - episode {d['ep']}/{n}")
        self.v["pos"].setText(f"x {d['x']:6.2f}  y {d['y']:6.2f}  z {d['z']:4.2f} m")
        self.v["spd"].setText(f"{d['speed']:.2f} m/s  vz {d['vz']:+.2f}   {d['heading']:+.0f} deg")
        self.v["goal"].setText(f"survivor {d['goal']}  -  {d['goal_dist']:.1f} m left")
        self.v["time"].setText(f"{d['t']:.1f} s   {d['path']:.1f} m")
        self.v["pol"].setText(self._act(d["ap"]))
        self.v["fus"].setText(self._act(d["a"]))
        m = d["min_range"]
        self.bar_min.setValue(int(min(m, 5.0) / 5.0 * 100))
        self.bar_min.setFormat(f"nearest wall {m:.2f} m")
        state = "red" if m < 0.5 else "amber" if m < 1.0 else "green"
        if state != self._bar_state:
            self._bar_state = state
            col = {"red": RED, "amber": AMBER, "green": GREEN}[state]
            self.bar_min.setStyleSheet(f"QProgressBar::chunk{{background:{col}}}")
        self.bar_ref.setValue(int(d["alpha"] * 100))
        self.bar_ref.setFormat(f"reflex {d['alpha']:.0%}")
        self.radar.set_ranges(d["ranges"])

    def _ep(self, d):
        self.results.append(d)
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = [d["ep"], f"survivor {d['goal']}", d["result"], d["time"], d["path"], d["left"],
                f"{d['reflex'] * 100:.0f}"]
        for c, val in enumerate(vals):
            it = QTableWidgetItem(str(val))
            it.setTextAlignment(Qt.AlignCenter)
            if c == 2:
                it.setForeground(QColor(GREEN if val == "SUCCESS" else RED if val == "COLLISION" else AMBER))
            self.table.setItem(r, c, it)
        self.table.scrollToBottom()
        n = len(self.results)
        ok = [x for x in self.results if x["result"] == "SUCCESS"]
        col = sum(x["result"] == "COLLISION" for x in self.results)
        tmo = n - len(ok) - col
        s = f"n={n} | success {len(ok)} ({len(ok) / n:.0%}) | collisions {col} | timeouts {tmo}"
        if ok:
            s += f" | mean time {sum(x['time'] for x in ok) / len(ok):.0f} s | mean path {sum(x['path'] for x in ok) / len(ok):.1f} m"
        self.summary.setText(s)


# ---------------------------------------------------------------------------------------------- tool launcher
class Tool(QWidget):
    log = pyqtSignal(str)

    def __init__(self, label, tag, build, tip, get_cfg):
        super().__init__()
        self.label, self.tag, self.build, self.get_cfg = label, tag, build, get_cfg
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(lambda e: self.log.emit(f"[{tag}] process error: {e}"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton(label)
        self.btn.setToolTip(tip)
        self.btn.clicked.connect(self.toggle)
        lay.addWidget(self.btn)

    def running(self):
        return self.proc.state() != QProcess.NotRunning

    def toggle(self):
        if self.running():
            kill_later(self.proc)
            return
        cfg = self.get_cfg()
        args, cwd, need = self.build(cfg)
        missing = [p for p in need if not os.path.exists(p)]
        if missing:
            QMessageBox.warning(self, "Missing files", "Not found:\n" + "\n".join(missing))
            return
        self.proc.setProcessEnvironment(make_env(cfg))
        self.proc.setWorkingDirectory(cwd)
        self.log.emit(f"[{self.tag}] > {cfg['python']} {' '.join(args)}")
        self.proc.start(cfg["python"], args)
        self.btn.setText("Stop: " + self.label)

    def _read(self):
        for ln in bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            if ln.strip():
                self.log.emit(f"[{self.tag}] {ln}")

    def _finished(self, code, _):
        self.btn.setText(self.label)
        self.log.emit(f"[{self.tag}] closed (exit code {code})")


# ---------------------------------------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIDAR drone launcher - MuJoCo reflex ON / OFF / manual flight")
        self.resize(1250, 900)
        self.st = QSettings("nidar", "launcher")
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # paths
        gp = QGroupBox("Project paths")
        g = QGridLayout(gp)
        self.paths = {}
        defaults = [("files_ppo", "files_ppo (nidar_env.py, runs\\)", HERE, True),
                    ("nidar_sim", "nidar_sim (fly_x2_keyboard.py, depth.py)", os.path.join(ROOT, "nidar_sim"), True),
                    ("x2", "skydio_x2\\x2.xml", os.path.join(ROOT, "skydio_x2", "x2.xml"), False),
                    ("scene", "skydio_x2\\scene.xml", os.path.join(ROOT, "skydio_x2", "scene.xml"), False),
                    ("python", "python.exe", sys.executable, False)]
        for r, (key, label, default, is_dir) in enumerate(defaults):
            g.addWidget(QLabel(label), r, 0)
            e = QLineEdit(self.st.value(key, default))
            g.addWidget(e, r, 1)
            b = QPushButton("Browse")
            b.clicked.connect(lambda _, k=key, d=is_dir: self._browse(k, d))
            g.addWidget(b, r, 2)
            self.paths[key] = e
        root.addWidget(gp)

        # settings row
        row = QHBoxLayout()
        row.addWidget(QLabel("Survivor"))
        self.goal = QComboBox()
        self.goal.addItems(["random each episode"] + [f"survivor {i}" for i in range(1, 7)])
        self.goal.setCurrentIndex(int(self.st.value("goal", 3)))
        row.addWidget(self.goal)
        row.addWidget(QLabel("Episodes"))
        self.eps = QSpinBox()
        self.eps.setRange(1, 500)
        self.eps.setValue(int(self.st.value("eps", 5)))
        row.addWidget(self.eps)
        row.addWidget(QLabel("Speed x"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 5.0)
        self.speed.setSingleStep(0.25)
        self.speed.setValue(float(self.st.value("speed", 1.0)))
        row.addWidget(self.speed)
        self.headless = QCheckBox("Headless (no window, fast)")
        row.addWidget(self.headless)
        row.addStretch(1)
        b_both = QPushButton("Launch both (compare)")
        b_both.clicked.connect(lambda: [p.launch() for p in self.panels])
        b_stop = QPushButton("Stop all")
        b_stop.clicked.connect(self.stop_all)
        row.addWidget(b_both)
        row.addWidget(b_stop)
        root.addLayout(row)

        # panels
        self.panels = [
            RunPanel("Reflex ON  (policy trained with the reflex)", "ON", True, os.path.join("runs", "nidar"), self.cfg),
            RunPanel("Reflex OFF  (policy trained without the reflex)", "OFF", False, os.path.join("runs", "no_reflex"), self.cfg)]
        sp = QSplitter(Qt.Horizontal)
        for p in self.panels:
            sp.addWidget(p)
        root.addWidget(sp, 1)

        # manual tools
        gt = QGroupBox("Normal flying (you fly with the keyboard)")
        h = QHBoxLayout(gt)
        self.tools = [
            Tool("Manual flight + ego / depth / IMU (depth.py)", "manual",
                 lambda c: (["-u", os.path.join(c["nidar_sim"], "depth.py"), "--x2", c["x2"], "--arena", c["arena"],
                             "--layout", c["layout"]], c["nidar_sim"], [os.path.join(c["nidar_sim"], "depth.py"), c["x2"]]),
                 "W/S A/D forward and sideways, R/F up/down, Q/E yaw, M mode, X brake, P reset. "
                 "Keep the MuJoCo window unfocused.", self.cfg),
            Tool("Hold-key flight (hold.py)", "hold",
                 lambda c: (["-u", os.path.join(c["nidar_sim"], "hold.py"), "--x2", c["x2"]], c["nidar_sim"],
                            [os.path.join(c["nidar_sim"], "hold.py"), c["x2"]]),
                 "Same keys as manual flight, without the camera panel.", self.cfg),
            Tool("Scene viewer (skydio_x2 scene.xml)", "scene",
                 lambda c: (["-m", "mujoco.viewer", "--mjcf", c["scene"]], c["nidar_sim"], [c["scene"]]),
                 "Plain MuJoCo viewer of the drone model.", self.cfg)]
        for t in self.tools:
            h.addWidget(t)
        h.addWidget(QLabel("keys: W/S A/D move   R/F up/down   Q/E yaw   M mode   X brake   P reset"))
        root.addWidget(gt)

        # log
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(4000)
        self.console.setFont(QFont("Consolas", 9))
        self.console.setMaximumHeight(150)
        root.addWidget(self.console)
        for w in self.panels + self.tools:
            w.log.connect(self.write)
        self.write("ready. Pick the settings and press Launch. Readings appear in each panel.")

    def _browse(self, key, is_dir):
        cur = self.paths[key].text()
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "Choose folder", cur)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", cur)
        if p:
            self.paths[key].setText(p)

    def cfg(self):
        t = lambda k: self.paths[k].text().strip()
        ns = t("nidar_sim")
        return dict(python=t("python"), files_ppo=t("files_ppo"), nidar_sim=ns, x2=t("x2"), scene=t("scene"),
                    arena=os.path.join(ns, "nidar_arena.xml"), layout=os.path.join(ns, "nidar_arena_layout.json"),
                    goal=self.goal.currentIndex(), episodes=self.eps.value(), speed=self.speed.value(),
                    headless=self.headless.isChecked())

    def write(self, msg):
        self.console.appendPlainText(f"{time.strftime('%H:%M:%S')}  {msg}")

    def stop_all(self):
        for w in self.panels + self.tools:
            if w.running():
                kill_later(w.proc)

    def closeEvent(self, e):
        for k, ed in self.paths.items():
            self.st.setValue(k, ed.text())
        self.st.setValue("goal", self.goal.currentIndex())
        self.st.setValue("eps", self.eps.value())
        self.st.setValue("speed", self.speed.value())
        for w in self.panels + self.tools:
            if w.running():
                w.proc.kill()
                w.proc.waitForFinished(1000)
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Main()
    win.show()
    sys.exit(app.exec_())
