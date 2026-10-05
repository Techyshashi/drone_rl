"""
NETRA Command Center - Autonomous Flight Research & Simulation (PyQt5)

Upgrade of the NIDAR launcher. Same MuJoCo runner (run_policy_gui.py), same trained policies, same reflex
ON / OFF side-by-side comparison, same manual-flight tools - plus a persistent SQLite experiment database,
automatic metrics, history / export, and restart / reset controls.

    pip install PyQt5
    python netra_launcher.py

Place this file and its modules (experiment_database.py, experiment_manager.py, metrics_engine.py,
report_exporter.py, simulation_manager.py, telemetry_manager.py) in one folder; project folders are
auto-detected and can be changed in the Project Configuration box.
"""
import math
import os
import sys
import time

from PyQt5.QtCore import (Qt, QDate, QItemSelectionModel, QPointF, QProcess, QProcessEnvironment, QRectF, QSettings,
                          QTimer, pyqtSignal)
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit, QDialog,
                             QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                             QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from experiment_database import (DatabaseError, ExperimentDatabase, NETRA_VERSION, local_date_to_utc,
                                 utc_to_local_str)
from experiment_manager import prepare_launch
from metrics_engine import MIN_SAMPLE, compare, compute_metrics, experiment_metrics, mode_summary
from report_exporter import export_episodes_csv, export_report_json, safe_default_name
from simulation_manager import QtProcessRunner, RunController, STATE_LABEL

APP_NAME = "NETRA Command Center"
APP_SUB = "Autonomous Flight Research & Simulation"

# ---- Oracle Redwood-inspired palette
BG, BRAND, TEXT, TEXT2 = "#FCFBFA", "#C74634", "#312D2A", "#70736E"
SURFACE, BORDER, ACCENT = "#F5F4F2", "#DFDCD8", "#2C5967"
GREEN, AMBER, RED = "#2E7D5B", "#B7791F", "#C74634"
OUTCOME_COLOR = {"SUCCESS": GREEN, "COLLISION": RED, "TIMEOUT": AMBER, "OTHER_FAILURE": "#8A4B9B", "INTERRUPTED": TEXT2}
STATE_STYLE = {"IDLE": (SURFACE, TEXT2), "RUNNING": ("#DCEBEF", ACCENT), "COMPLETED": ("#DDF0E7", GREEN),
               "CANCELLED": ("#FBEFD9", AMBER), "INTERRUPTED": ("#FBEFD9", AMBER), "FAILED": ("#FBE3DF", RED)}
MONO = QFont("Consolas", 10)

QSS = f"""
* {{ color:{TEXT}; font-family:"Oracle Sans","Helvetica Neue",Arial,sans-serif; font-size:10pt; }}
QMainWindow, QDialog {{ background:{BG}; }}
QLabel, QCheckBox {{ background:transparent; }}
QScrollArea {{ background:transparent; border:none; }}
QScrollArea > QWidget > QWidget {{ background:{BG}; }}
QGroupBox {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:8px; margin-top:16px; padding:14px 10px 10px 10px; font-weight:600; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; color:{TEXT}; }}
QGroupBox QGroupBox {{ background:#FFFFFF; }}
QPushButton {{ background:#FFFFFF; border:1px solid {BORDER}; border-radius:6px; padding:6px 12px; }}
QPushButton:hover {{ border-color:{BRAND}; }}
QPushButton:disabled {{ color:#B5B2AD; background:{SURFACE}; border-color:{BORDER}; }}
QPushButton#primary {{ background:{BRAND}; color:#FFFFFF; border:1px solid {BRAND}; font-weight:600; }}
QPushButton#primary:hover {{ background:#B03C2D; }}
QPushButton#primary:disabled {{ background:#E3B9B2; border-color:#E3B9B2; color:#FFFFFF; }}
QPushButton#danger {{ color:{BRAND}; border-color:{BRAND}; }}
QPushButton#danger:disabled {{ color:#E3B9B2; border-color:{BORDER}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{ background:#FFFFFF; border:1px solid {BORDER}; border-radius:6px; padding:4px 6px; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {{ border-color:{ACCENT}; }}
QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:8px; background:{BG}; top:-1px; }}
QTabBar::tab {{ padding:9px 20px; background:{SURFACE}; border:1px solid {BORDER}; border-bottom:none; border-top-left-radius:8px; border-top-right-radius:8px; color:{TEXT2}; margin-right:2px; }}
QTabBar::tab:selected {{ background:{BG}; color:{BRAND}; font-weight:600; }}
QTableWidget {{ background:#FFFFFF; alternate-background-color:#FAF9F8; gridline-color:#EDEBE8; border:1px solid {BORDER}; border-radius:6px; selection-background-color:#F4DCD8; selection-color:{TEXT}; }}
QHeaderView::section {{ background:{SURFACE}; color:{TEXT2}; border:none; border-bottom:1px solid {BORDER}; padding:5px; font-weight:600; }}
QProgressBar {{ border:1px solid {BORDER}; border-radius:6px; background:#FFFFFF; text-align:center; min-height:16px; }}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:5px; }}
QPlainTextEdit {{ background:#FFFFFF; border:1px solid {BORDER}; border-radius:6px; font-family:Consolas,"Courier New",monospace; font-size:9pt; }}
QSplitter::handle {{ background:transparent; }}
QStatusBar {{ background:{SURFACE}; color:{TEXT2}; }}
QToolTip {{ background:#FFFFFF; border:1px solid {BORDER}; color:{TEXT}; }}
"""

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------ path detection
def _find_up(start, markers, levels=6):
    cur = os.path.abspath(start)
    for _ in range(levels):
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def autodetect():
    """Locate the folder holding run_policy_gui.py and the project root (nidar_sim / skydio_x2)."""
    sim = None
    for c in (HERE, os.path.join(HERE, "nidar_rl"), os.path.join(HERE, "files_ppo", "nidar_rl"),
              os.path.join(HERE, "files_ppo")):
        if os.path.isfile(os.path.join(c, "run_policy_gui.py")):
            sim = c
            break
    if sim is None:
        for dp, dn, fn in os.walk(HERE):
            if dp[len(HERE):].count(os.sep) > 3:
                dn[:] = []
                continue
            if "run_policy_gui.py" in fn:
                sim = dp
                break
    root = _find_up(sim or HERE, ["nidar_sim", "skydio_x2"]) or os.path.dirname(HERE)
    return (sim or HERE), root


def guess_run_dir(sim, off):
    parent = os.path.dirname(sim)
    cands = ([os.path.join(sim, "runs", "no_reflex"), os.path.join(parent, "drive_dl_nr", "no_reflex"),
              os.path.join(parent, "drive_dl", "nidar_runs", "no_reflex")] if off else
             [os.path.join(sim, "runs", "nidar"), os.path.join(parent, "drive_dl", "nidar_runs", "run1"),
              os.path.join(parent, "drive_dl", "nidar_runs", "nidar")])
    for c in cands:
        if os.path.isfile(os.path.join(c, "ppo_nidar.zip")):
            return c
    return cands[0]


def default_settings():
    sim, root = autodetect()
    return {"files_ppo": sim, "nidar_sim": os.path.join(root, "nidar_sim"),
            "x2": os.path.join(root, "skydio_x2", "x2.xml"), "scene": os.path.join(root, "skydio_x2", "scene.xml"),
            "python": sys.executable, "run_on": guess_run_dir(sim, False), "run_off": guess_run_dir(sim, True),
            "goal": "3", "episodes": "5", "speed": "1.0", "headless": "0"}


def load_settings(db):
    s = default_settings()
    stored = db.all_settings()
    if "files_ppo" not in stored:                                   # first run: import the old NIDAR launcher values
        old = QSettings("nidar", "launcher")
        for k in ("files_ppo", "nidar_sim", "x2", "scene", "python", "goal", "eps", "speed"):
            v = old.value(k)
            if v not in (None, ""):
                s["episodes" if k == "eps" else k] = str(v)
    s.update({k: v for k, v in stored.items() if v is not None})
    return s


# ------------------------------------------------------------------------------------------------ small helpers
def fnum(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def cell(text, color=None, italic=False, align=Qt.AlignCenter, bold=False):
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(align)
    if color:
        it.setForeground(QColor(color))
    if italic or bold:
        f = it.font()
        f.setItalic(italic)
        f.setBold(bold)
        it.setFont(f)
    return it


EP_HEADERS = ["#", "Survivor", "Outcome", "Time s", "Path m", "Left m", "Clearance m", "Reflex %"]


def make_episode_table(show_exp=False, min_h=110):
    hdr = (["Mode", "Experiment"] if show_exp else []) + EP_HEADERS
    t = QTableWidget(0, len(hdr))
    t.setHorizontalHeaderLabels(hdr)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    if show_exp:
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.setMinimumHeight(min_h)
    return t


def fill_episode_table(t, eps, show_exp=False):
    t.setRowCount(len(eps))
    off = 2 if show_exp else 0
    for r, e in enumerate(eps):
        interrupted = e["status"] == "INTERRUPTED"
        vals = [e["episode_number"], f"survivor {e['goal_id']}" if e["goal_id"] else "—", e["outcome"],
                fnum(e["duration_s"]), fnum(e["path_length_m"]), fnum(e["final_goal_dist_m"]),
                fnum(e["min_clearance_m"], 2), fnum(e["reflex_activation_pct"], 0)]
        if show_exp:
            t.setItem(r, 0, cell(e["reflex_mode"]))
            t.setItem(r, 1, cell(e["experiment_id"], align=Qt.AlignLeft | Qt.AlignVCenter))
        for c, v in enumerate(vals):
            t.setItem(r, off + c, cell(v, color=OUTCOME_COLOR.get(e["outcome"]) if c == 2 else (TEXT2 if interrupted else None),
                                       italic=interrupted, bold=(c == 2)))
        t.item(r, off + 2).setToolTip(e.get("termination_reason") or "")
    if not show_exp:
        t.scrollToBottom()


def metric_rows(m, with_reflex=True):
    n = m["episodes_completed"]

    def rate(key, cnt, ci):
        v = m[key]
        if v is None:
            return "n/a"
        return f"{v:.1f} %   ({m[cnt]}/{n},  95% CI {m[ci][0]:.0f}-{m[ci][1]:.0f})"

    def mean(key, d=1):
        return "n/a" if m[key] is None else f"{m[key]:.{d}f}   (n={m[key + '_n']})"
    rows = [
        ("Experiments", str(m.get("experiments", 1))),
        ("Episodes completed (eligible)", str(n)),
        ("Episodes interrupted (excluded)", str(m["episodes_interrupted"])),
        ("Successful episodes", str(m["successes"])),
        ("Failed episodes", str(m["failures"])),
        ("Success rate", rate("success_rate", "successes", "success_ci")),
        ("Collision rate", rate("collision_rate", "collisions", "collision_ci")),
        ("Timeout rate", rate("timeout_rate", "timeouts", "timeout_ci")),
        ("Other-failure rate (crashes)", "n/a" if m["other_failure_rate"] is None else
         f"{m['other_failure_rate']:.1f} %   ({m['other_failures']}/{n})"),
        ("Mean episode duration (s)", mean("mean_duration_s")),
        ("Mean path length (m)", mean("mean_path_m")),
        ("Mean final distance to goal (m)", mean("mean_final_dist_m")),
        ("Mean min. obstacle clearance (m)", mean("mean_clearance_m", 2)),
        ("Mean reflex activation (%)", mean("mean_reflex_pct") if with_reflex else "n/a (reflex off)"),
        ("Sample size", "no data" if n == 0 else (f"too small ({n} < {MIN_SAMPLE})" if m["small_sample"] else f"adequate ({n})")),
    ]
    return rows


# ------------------------------------------------------------------------------------------------ widgets
class Radar(QWidget):
    ANG = [-150 + 30 * i for i in range(12)]                    # same 12 rays as the policy; 0 = nose, + = left

    def __init__(self):
        super().__init__()
        self.setMinimumSize(180, 180)
        self.r = [5.0] * 12

    def set_ranges(self, r):
        self.r = r
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        R = min(self.width(), self.height()) / 2 - 10
        p.setPen(QPen(QColor(BORDER), 1))
        for f in (0.25, 0.5, 0.75, 1.0):
            p.drawEllipse(c, R * f, R * f)
        p.drawLine(c, QPointF(c.x(), c.y() - R))
        p.setPen(QColor(TEXT2))
        p.drawText(QPointF(c.x() + 4, c.y() - R + 12), "5 m")
        pts = []
        for th, d in zip(self.ANG, self.r):
            t = math.radians(th)
            rr = min(d, 5.0) / 5.0 * R
            pts.append(QPointF(c.x() - math.sin(t) * rr, c.y() - math.cos(t) * rr))
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.setBrush(QColor(44, 89, 103, 50))
        p.drawPolygon(QPolygonF(pts))
        for pt, d in zip(pts, self.r):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(RED if d < 0.5 else AMBER if d < 1.0 else GREEN))
            p.drawEllipse(pt, 3.5, 3.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(TEXT))
        p.drawPolygon(QPolygonF([QPointF(c.x(), c.y() - 9), QPointF(c.x() - 6, c.y() + 7), QPointF(c.x() + 6, c.y() + 7)]))


class BarChart(QWidget):
    """Compact grouped bar chart. series = [(name, color, [value or None per category])]."""

    def __init__(self, title, vmax=None, decimals=1):
        super().__init__()
        self.title, self.vmax, self.dec = title, vmax, decimals
        self.cats, self.series = [], []
        self.setMinimumHeight(190)
        self.setMinimumWidth(200)

    def set_data(self, cats, series):
        self.cats, self.series = cats, series
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 8, 8)
        bold = QFont(self.font())
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QColor(TEXT))
        p.drawText(QPointF(12, 22), self.title)
        p.setFont(self.font())
        x = w - 12
        for name, color, _ in reversed(self.series):                  # legend, right aligned
            tw = p.fontMetrics().horizontalAdvance(name)
            x -= tw
            p.setPen(QColor(TEXT2))
            p.drawText(QPointF(x, 22), name)
            x -= 14
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(x, 13, 9, 9), 2, 2)
            x -= 10
        if not self.cats or not self.series:
            return
        left, right, top, bottom = 14, 14, 36, 28
        pw, ph = w - left - right, h - top - bottom
        vals = [v for _, _, vs in self.series for v in vs if v is not None]
        vmax = self.vmax or ((max(vals) * 1.2) if vals and max(vals) > 0 else 1.0)
        base = top + ph
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(QPointF(left, base), QPointF(left + pw, base))
        gw = pw / len(self.cats)
        ns = len(self.series)
        bw = min(38.0, gw * 0.7 / ns)
        gap = 5.0
        for gi, cat in enumerate(self.cats):
            total = bw * ns + gap * (ns - 1)
            start = left + gi * gw + (gw - total) / 2
            for si, (_, color, vs) in enumerate(self.series):
                v = vs[gi]
                bx = start + si * (bw + gap)
                p.setPen(QColor(TEXT2))
                if v is None:
                    p.drawText(QRectF(bx - 6, base - 20, bw + 12, 16), Qt.AlignCenter, "n/a")
                    continue
                bh = max(1.0, min(v / vmax, 1.0) * ph)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color))
                p.drawRoundedRect(QRectF(bx, base - bh, bw, bh), 3, 3)
                p.setPen(QColor(TEXT))
                p.drawText(QRectF(bx - 10, base - bh - 17, bw + 20, 15), Qt.AlignCenter, f"{v:.{self.dec}f}")
            p.setPen(QColor(TEXT2))
            p.drawText(QRectF(left + gi * gw, base + 4, gw, 18), Qt.AlignCenter, cat)


# ------------------------------------------------------------------------------------------------ run panel
class RunPanel(QGroupBox):
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    changed = pyqtSignal()

    def __init__(self, title, mode, db, get_cfg, default_run):
        super().__init__(title)
        self.mode, self.db, self.get_cfg, self.peer = mode, db, get_cfg, None
        self.view_exp_id = None
        self._done = 0
        self.runner = QtProcessRunner(self)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Run folder"))
        self.run_edit = QLineEdit(default_run)
        self.run_edit.textChanged.connect(self._file_info)
        self.run_edit.editingFinished.connect(self.changed.emit)
        row.addWidget(self.run_edit, 1)
        b = QPushButton("Browse")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setFont(QFont("Consolas", 8))
        self.info.setStyleSheet(f"color:{TEXT2}")
        lay.addWidget(self.info)

        row = QHBoxLayout()
        self.chip = QLabel("Idle")
        row.addWidget(self.chip)
        self.lbl_exp = QLabel("no experiment")
        self.lbl_exp.setFont(QFont("Consolas", 9))
        row.addWidget(self.lbl_exp, 1)
        self.lbl_prog = QLabel("")
        row.addWidget(self.lbl_prog)
        lay.addLayout(row)
        self.lbl_msg = QLabel("")
        self.lbl_msg.setWordWrap(True)
        self.lbl_msg.setStyleSheet(f"color:{TEXT2}")
        lay.addWidget(self.lbl_msg)

        row = QHBoxLayout()
        self.btn_go = QPushButton()
        self.btn_go.setObjectName("primary")
        self.btn_go.clicked.connect(self.launch)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(lambda: self.ctrl.request("stop"))
        row.addWidget(self.btn_go, 1)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.btn_rep = QPushButton("Restart episode")
        self.btn_rep.setToolTip("Stop the current episode (kept as INTERRUPTED) and start a fresh one in the same experiment.")
        self.btn_rep.clicked.connect(lambda: self.ctrl.request("restart_episode"))
        self.btn_rex = QPushButton("Restart experiment")
        self.btn_rex.setToolTip("Cancel this experiment (records kept) and start a NEW experiment with the same settings.")
        self.btn_rex.clicked.connect(self._restart_experiment)
        self.btn_rsim = QPushButton("Reset simulation")
        self.btn_rsim.setToolTip("Stop the simulator and clear live telemetry. The experiment stays open; nothing restarts "
                                 "until you press Continue.")
        self.btn_rsim.clicked.connect(lambda: self.ctrl.request("reset_sim"))
        for bt in (self.btn_rep, self.btn_rex, self.btn_rsim):
            row.addWidget(bt)
        lay.addLayout(row)

        # ---- live telemetry (not persisted)
        live = QGroupBox("Live telemetry   (streaming - not yet persisted)")
        body = QHBoxLayout(live)
        grid = QGridLayout()
        self.v = {}
        for r, (key, name) in enumerate([("pos", "position"), ("spd", "speed / heading"), ("goal", "goal"),
                                         ("time", "time / path"), ("pol", "PPO action"), ("fus", "fused action")]):
            grid.addWidget(QLabel(name), r, 0)
            lb = QLabel("-")
            lb.setFont(MONO)
            grid.addWidget(lb, r, 1)
            self.v[key] = lb
        self.bar_min, self.bar_ref = QProgressBar(), QProgressBar()
        for bar in (self.bar_min, self.bar_ref):
            bar.setRange(0, 100)
        self._bar_state = None
        grid.addWidget(QLabel("clearance"), 6, 0)
        grid.addWidget(self.bar_min, 6, 1)
        grid.addWidget(QLabel("reflex now"), 7, 0)
        grid.addWidget(self.bar_ref, 7, 1)
        grid.setRowStretch(8, 1)
        body.addLayout(grid, 1)
        self.radar = Radar()
        body.addWidget(self.radar)
        lay.addWidget(live)

        # ---- persisted results
        saved = QGroupBox("Episode results   (read from the database)")
        sl = QVBoxLayout(saved)
        self.lbl_source = QLabel("")
        self.lbl_source.setStyleSheet(f"color:{TEXT2}")
        sl.addWidget(self.lbl_source)
        self.lbl_latest = QLabel("latest outcome: -")
        sl.addWidget(self.lbl_latest)
        self.summary = QLabel("no episodes recorded")
        self.summary.setWordWrap(True)
        sl.addWidget(self.summary)
        self.table = make_episode_table()
        sl.addWidget(self.table)
        lay.addWidget(saved)

        self.reset_live()
        self._file_info()
        self.ctrl = RunController(db, mode, self.runner, log=self._cb_log, telemetry=self._on_tel,
                                  episode=self._on_episode, state=self._on_state, error=self._cb_error,
                                  reset_live=self.reset_live)
        self.runner.controller = self.ctrl
        self._apply_state("IDLE", "")
        latest = db.list_experiments(reflex=mode)
        if latest:
            self.view_exp_id = latest[0]["experiment_id"]
            self.reload_persisted()

    # ---- paths
    def run_dir(self):
        d = self.run_edit.text().strip()
        return d if os.path.isabs(d) else os.path.join(self.get_cfg()["files_ppo"], d)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with ppo_nidar.zip + vecnorm.pkl", self.run_dir())
        if d:
            self.run_edit.setText(d)
            self.changed.emit()

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

    # ---- actions
    def launch(self):
        if self.peer is not None and not self.ctrl.can_continue():
            if os.path.normcase(os.path.abspath(self.run_dir())) == os.path.normcase(os.path.abspath(self.peer.run_dir())):
                if QMessageBox.question(self, APP_NAME, "Reflex ON and Reflex OFF point at the SAME run folder.\n"
                                        "The comparison would not be meaningful. Launch anyway?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return
        self.changed.emit()                                           # persist settings before launching
        if self.ctrl.can_continue():
            self.ctrl.continue_experiment()
        else:
            self.ctrl.launch(self.get_cfg(), self.run_dir())
        self.view_exp_id = None
        self.reload_persisted()

    def _restart_experiment(self):
        if QMessageBox.question(self, APP_NAME, "Cancel the current experiment (all recorded episodes are kept) and "
                                "start a NEW experiment with the same settings?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.ctrl.request("restart_experiment")

    def shutdown(self):
        c = self.ctrl
        if c.is_running() or c.has_open_session():
            c.request("exit")
            if c.is_running():
                self.runner.kill_now()
                if c.has_open_session():
                    c.on_exit(-1)

    # ---- controller callbacks
    def _cb_log(self, msg):
        self.log.emit(f"[{self.mode}] {msg}")

    def _cb_error(self, msg):
        self.error.emit(f"[{self.mode}] {msg}")

    def _on_state(self, state, msg):
        self._apply_state(state, msg)
        self.log.emit(f"[{self.mode}] {STATE_LABEL[state]}" + (f": {msg}" if msg else ""))
        self.reload_persisted()
        self.changed.emit()

    def _apply_state(self, state, msg):
        bg, fg = STATE_STYLE[state]
        self.chip.setText(STATE_LABEL[state])
        self.chip.setStyleSheet(f"background:{bg};color:{fg};border-radius:10px;padding:3px 12px;font-weight:600;")
        self.lbl_msg.setText(msg)
        s = self.ctrl.session if hasattr(self, "ctrl") else None
        self.lbl_exp.setText(s.experiment_id if s else "no experiment")
        self._update_progress()
        self._refresh_buttons()

    def _update_progress(self):
        s = self.ctrl.session if hasattr(self, "ctrl") else None
        if s is None:
            self.lbl_prog.setText("")
            return
        self._done = s.configured - s.remaining()
        self.lbl_prog.setText(f"completed {self._done}/{s.configured}")

    def _refresh_buttons(self):
        c = self.ctrl
        running = c.is_running()
        if c.can_continue():
            self.btn_go.setText(f"Continue experiment  ({c.session.remaining()} left)")
        else:
            self.btn_go.setText(f"Launch  (reflex {self.mode})")
        self.btn_go.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_rep.setEnabled(running)
        self.btn_rex.setEnabled(c.last_launch is not None)
        self.btn_rsim.setEnabled(running or c.has_open_session())

    def reset_live(self):
        for lb in self.v.values():
            lb.setText("-")
        self.radar.set_ranges([5.0] * 12)
        self.bar_min.setValue(0)
        self.bar_min.setFormat("nearest wall -")
        self.bar_ref.setValue(0)
        self.bar_ref.setFormat("reflex -")

    @staticmethod
    def _act(a):
        return f"F{a[0]:+.2f}  L{a[1]:+.2f}  Y{a[2]:+.2f}"

    def _on_tel(self, d):
        s = self.ctrl.session
        if s is not None:
            self.lbl_prog.setText(f"episode {s.offset + d['ep']}  |  completed {self._done}/{s.configured}")
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

    def _on_episode(self, rec, saved):
        self.lbl_latest.setText(f"latest outcome: episode {rec['episode_number']} - {rec['outcome']}"
                                + ("" if saved else "   (NOT SAVED - see console)"))
        self.lbl_latest.setStyleSheet(f"color:{OUTCOME_COLOR.get(rec['outcome'], TEXT)};font-weight:600;")
        self._update_progress()
        self.reload_persisted()
        self.changed.emit()

    # ---- persisted view
    def reload_persisted(self):
        s = self.ctrl.session if hasattr(self, "ctrl") else None
        eid = s.experiment_id if s else self.view_exp_id
        if not eid:
            self.lbl_source.setText("no experiment recorded yet for this mode")
            self.table.setRowCount(0)
            self.summary.setText("no episodes recorded")
            return
        try:
            eps = self.db.get_episodes(experiment_ids=[eid])
        except DatabaseError as e:
            self.error.emit(f"[{self.mode}] DATABASE READ ERROR: {e}")
            return
        self.lbl_source.setText(("current experiment " if s else "last recorded experiment ") + eid)
        fill_episode_table(self.table, eps)
        m = compute_metrics(eps)
        if m["episodes_completed"]:
            txt = (f"n={m['episodes_completed']} completed | success {m['successes']} ({m['success_rate']:.0f}%) | "
                   f"collisions {m['collisions']} | timeouts {m['timeouts']}")
            if m["other_failures"]:
                txt += f" | crashes {m['other_failures']}"
            if m["mean_duration_s"] is not None:
                txt += f" | mean time {m['mean_duration_s']:.0f} s"
            if m["mean_path_m"] is not None:
                txt += f" | mean path {m['mean_path_m']:.1f} m"
        else:
            txt = "no completed episodes yet"
        if m["episodes_interrupted"]:
            txt += f"  (+{m['episodes_interrupted']} interrupted, excluded)"
        self.summary.setText(txt)


# ------------------------------------------------------------------------------------------------ manual tools
def make_env(cfg):
    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONPATH", cfg["nidar_sim"] + os.pathsep + env.value("PYTHONPATH", ""))
    env.insert("PYTHONUNBUFFERED", "1")
    return env


def kill_later(proc, ms=1500):
    if proc.state() != QProcess.NotRunning:
        proc.terminate()
        QTimer.singleShot(ms, lambda: proc.kill() if proc.state() != QProcess.NotRunning else None)


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
            QMessageBox.warning(self, APP_NAME, "Not found:\n" + "\n".join(missing))
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


# ------------------------------------------------------------------------------------------------ dashboard
class Dashboard(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Scope"))
        self.scope = QComboBox()
        self.scope.addItems(["All recorded history", "Latest experiment of each mode", "Experiments selected in History"])
        self.scope.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.scope)
        row.addWidget(QLabel("Survivor"))
        self.surv = QComboBox()
        self.surv.addItems(["All survivors"] + [f"survivor {i}" for i in range(1, 7)])
        self.surv.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.surv)
        row.addStretch(1)
        b = QPushButton("Refresh from database")
        b.clicked.connect(self.refresh)
        row.addWidget(b)
        lay.addLayout(row)
        note = QLabel("Every figure is computed from the stored episode records. Interrupted episodes are excluded from "
                      "all rates and means.")
        note.setStyleSheet(f"color:{TEXT2}")
        lay.addWidget(note)

        top = QHBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Reflex ON", "Reflex OFF"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(430)
        top.addWidget(self.table, 3)
        charts = QVBoxLayout()
        self.ch_rates = BarChart("Outcome rates (%)", vmax=100.0)
        self.ch_path = BarChart("Mean path length (m)")
        self.ch_time = BarChart("Mean completion time (s)")
        for c in (self.ch_rates, self.ch_path, self.ch_time):
            charts.addWidget(c)
        top.addLayout(charts, 2)
        lay.addLayout(top)

        self.findings = QLabel()
        self.findings.setWordWrap(True)
        self.findings.setStyleSheet(f"background:#FFFFFF;border:1px solid {BORDER};border-radius:8px;padding:10px;")
        lay.addWidget(self.findings)
        lay.addWidget(QLabel("Episode-by-episode results"))
        self.eps = make_episode_table(show_exp=True, min_h=220)
        lay.addWidget(self.eps)

    def reset(self):
        self.scope.setCurrentIndex(0)
        self.surv.setCurrentIndex(0)

    def refresh(self, *_):
        db = self.main.db
        goal = self.surv.currentIndex() or None
        try:
            exps = db.list_experiments(goal=goal)
            idx = self.scope.currentIndex()
            if idx == 1:
                exps = [next((x for x in exps if x["reflex_mode"] == m), None) for m in ("ON", "OFF")]
                exps = [x for x in exps if x]
            elif idx == 2:
                sel = set(self.main.history.selected_ids())
                exps = [x for x in exps if x["experiment_id"] in sel]
            eps = db.get_episodes(experiment_ids=[x["experiment_id"] for x in exps], goal=goal)
        except DatabaseError as e:
            self.main.report_error(f"DATABASE READ ERROR: {e}")
            return
        on, off = mode_summary(exps, eps, "ON"), mode_summary(exps, eps, "OFF")
        ron, roff = metric_rows(on), metric_rows(off, with_reflex=False)
        self.table.setRowCount(len(ron))
        for r, ((label, a), (_, b)) in enumerate(zip(ron, roff)):
            self.table.setItem(r, 0, cell(label, align=Qt.AlignLeft | Qt.AlignVCenter, bold=label in ("Success rate", "Collision rate")))
            self.table.setItem(r, 1, cell(a))
            self.table.setItem(r, 2, cell(b))
        S = [("Reflex ON", BRAND, on), ("Reflex OFF", ACCENT, off)]
        self.ch_rates.set_data(["Success", "Collision", "Timeout"],
                               [(n, c, [m["success_rate"], m["collision_rate"], m["timeout_rate"]]) for n, c, m in S])
        self.ch_path.set_data(["mean path"], [(n, c, [m["mean_path_m"]]) for n, c, m in S])
        self.ch_time.set_data(["mean time"], [(n, c, [m["mean_duration_s"]]) for n, c, m in S])
        self.findings.setText("<b>Findings</b><br>" + "<br>".join(compare(on, off)) +
                              f"<br><span style='color:{TEXT2}'>Samples: ON n={on['episodes_completed']}, "
                              f"OFF n={off['episodes_completed']} completed episodes.</span>")
        shown = eps[-2000:]
        fill_episode_table(self.eps, shown, show_exp=True)


# ------------------------------------------------------------------------------------------------ history
HIST_COLS = ["Experiment ID", "Started", "Mode", "Survivor", "Status", "Done / configured", "Success %",
             "Collision %", "Mean time s", "Mean path m", "Policy"]
SORTS = [("Date (newest first)", "started_at"), ("Success rate", "success_rate"), ("Collision rate", "collision_rate"),
         ("Mean duration", "mean_duration_s"), ("Mean path length", "mean_path_m")]


class CompareDialog(QDialog):
    def __init__(self, parent, rows):
        super().__init__(parent)
        self.setWindowTitle("Compare experiments")
        self.resize(max(700, 220 + 230 * len(rows)), 560)
        lay = QVBoxLayout(self)
        base = metric_rows(rows[0][1])
        t = QTableWidget(len(base), 1 + len(rows))
        t.setHorizontalHeaderLabels(["Metric"] + [f"{x['reflex_mode']}  {x['experiment_id'][4:19]}" for x, _ in rows])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setAlternatingRowColors(True)
        for r, (label, _) in enumerate(base):
            t.setItem(r, 0, cell(label, align=Qt.AlignLeft | Qt.AlignVCenter))
        for c, (x, m) in enumerate(rows, start=1):
            for r, (_, v) in enumerate(metric_rows(m, with_reflex=x["reflex_mode"] == "ON")):
                t.setItem(r, c, cell(v))
            t.horizontalHeaderItem(c).setToolTip(x["experiment_id"])
        lay.addWidget(t)
        b = QPushButton("Close")
        b.clicked.connect(self.accept)
        lay.addWidget(b, 0, Qt.AlignRight)


class History(QWidget):
    use_in_dashboard = pyqtSignal()

    def __init__(self, main):
        super().__init__()
        self.main, self.rows = main, []
        lay = QVBoxLayout(self)
        f = QGridLayout()
        self.chk_from, self.chk_to = QCheckBox("From"), QCheckBox("To")
        self.d_from, self.d_to = QDateEdit(QDate.currentDate().addDays(-30)), QDateEdit(QDate.currentDate())
        for d in (self.d_from, self.d_to):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
        f.addWidget(self.chk_from, 0, 0)
        f.addWidget(self.d_from, 0, 1)
        f.addWidget(self.chk_to, 0, 2)
        f.addWidget(self.d_to, 0, 3)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Any reflex mode", "Reflex ON", "Reflex OFF"])
        self.cb_out = QComboBox()
        self.cb_out.addItems(["Any outcome", "SUCCESS", "COLLISION", "TIMEOUT", "OTHER_FAILURE", "INTERRUPTED"])
        self.cb_surv = QComboBox()
        self.cb_surv.addItems(["Any survivor"] + [f"survivor {i}" for i in range(1, 7)])
        f.addWidget(self.cb_mode, 0, 4)
        f.addWidget(self.cb_out, 0, 5)
        f.addWidget(self.cb_surv, 0, 6)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Search experiment ID...")
        f.addWidget(self.ed_search, 1, 0, 1, 4)
        self.cb_sort = QComboBox()
        self.cb_sort.addItems([s[0] for s in SORTS])
        self.chk_desc = QCheckBox("Descending")
        self.chk_desc.setChecked(True)
        f.addWidget(QLabel("Sort by"), 1, 4, Qt.AlignRight)
        f.addWidget(self.cb_sort, 1, 5)
        f.addWidget(self.chk_desc, 1, 6)
        lay.addLayout(f)
        for w in (self.chk_from, self.chk_to, self.chk_desc):
            w.stateChanged.connect(self.refresh)
        for w in (self.d_from, self.d_to):
            w.dateChanged.connect(self.refresh)
        for w in (self.cb_mode, self.cb_out, self.cb_surv, self.cb_sort):
            w.currentIndexChanged.connect(self.refresh)
        self.ed_search.textChanged.connect(self.refresh)

        row = QHBoxLayout()
        for text, fn in (("Refresh", self.refresh), ("Compare selected", self.compare_selected),
                         ("Show selected in dashboard", lambda: self.use_in_dashboard.emit()),
                         ("Export CSV", self.export_csv), ("Export JSON report", self.export_json)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        row.addStretch(1)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"color:{TEXT2}")
        row.addWidget(self.lbl_count)
        lay.addLayout(row)

        self.table = QTableWidget(0, len(HIST_COLS))
        self.table.setHorizontalHeaderLabels(HIST_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._show_selected)
        lay.addWidget(self.table, 3)
        self.lbl_eps = QLabel("Select an experiment to inspect its episodes.")
        lay.addWidget(self.lbl_eps)
        self.eps = make_episode_table(show_exp=True, min_h=160)
        lay.addWidget(self.eps, 2)

    # ---- filters
    def filters(self):
        return dict(
            date_from=local_date_to_utc(self.d_from.date().toPyDate()) if self.chk_from.isChecked() else None,
            date_to=local_date_to_utc(self.d_to.date().toPyDate(), True) if self.chk_to.isChecked() else None,
            reflex={0: None, 1: "ON", 2: "OFF"}[self.cb_mode.currentIndex()],
            outcome=self.cb_out.currentText() if self.cb_out.currentIndex() else None,
            goal=self.cb_surv.currentIndex() or None,
            search=self.ed_search.text().strip() or None)

    def reset_filters(self):
        widgets = (self.chk_from, self.chk_to, self.chk_desc, self.cb_mode, self.cb_out, self.cb_surv, self.cb_sort,
                   self.ed_search)
        for w in widgets:
            w.blockSignals(True)
        self.chk_from.setChecked(False)
        self.chk_to.setChecked(False)
        self.chk_desc.setChecked(True)
        for w in (self.cb_mode, self.cb_out, self.cb_surv, self.cb_sort):
            w.setCurrentIndex(0)
        self.ed_search.clear()
        for w in widgets:
            w.blockSignals(False)
        self.table.clearSelection()
        self.refresh()

    def selected_ids(self):
        return [self.table.item(i.row(), 0).text() for i in self.table.selectionModel().selectedRows()]

    def refresh(self, *_):
        keep = set(self.selected_ids())
        try:
            exps = self.main.db.list_experiments(**self.filters())
            eps = self.main.db.get_episodes(experiment_ids=[x["experiment_id"] for x in exps])
        except DatabaseError as e:
            self.main.report_error(f"DATABASE READ ERROR: {e}")
            return
        rows = [(x, experiment_metrics(x, eps)) for x in exps]
        key = SORTS[self.cb_sort.currentIndex()][1]
        val = (lambda r: r[0]["started_at"]) if key == "started_at" else (lambda r: r[1][key])
        have = [r for r in rows if val(r) is not None]
        have.sort(key=val, reverse=self.chk_desc.isChecked())
        self.rows = have + [r for r in rows if val(r) is None]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for r, (x, m) in enumerate(self.rows):
            vals = [x["experiment_id"], utc_to_local_str(x["started_at"], "%Y-%m-%d %H:%M"), x["reflex_mode"],
                    x["goal_label"] or "—", x["status"], f"{m['episodes_completed']} / {x['configured_episodes']}",
                    fnum(m["success_rate"]), fnum(m["collision_rate"]), fnum(m["mean_duration_s"]),
                    fnum(m["mean_path_m"]), x["policy_file"] or "—"]
            for c, v in enumerate(vals):
                col = {"COMPLETED": GREEN, "FAILED": RED, "INTERRUPTED": AMBER, "CANCELLED": AMBER}.get(v) if c == 4 else None
                it = cell(v, color=col, align=Qt.AlignLeft | Qt.AlignVCenter if c == 0 else Qt.AlignCenter)
                if c == 10:
                    it.setToolTip(f"{x['policy_path']}\nsha256 {x['policy_sha256']}\nvecnorm sha256 {x['vecnorm_sha256']}")
                self.table.setItem(r, c, it)
            if x["experiment_id"] in keep:
                self.table.selectionModel().select(self.table.model().index(r, 0),
                                                   QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.table.blockSignals(False)
        self.lbl_count.setText(f"{len(self.rows)} experiment(s), {len(eps)} episode record(s) match")
        self._show_selected()

    def _show_selected(self):
        ids = self.selected_ids()
        if not ids:
            self.eps.setRowCount(0)
            self.lbl_eps.setText("Select an experiment to inspect its episodes.")
            return
        try:
            eps = self.main.db.get_episodes(experiment_ids=ids)
        except DatabaseError as e:
            self.main.report_error(f"DATABASE READ ERROR: {e}")
            return
        self.lbl_eps.setText(f"Episodes of {len(ids)} selected experiment(s): {len(eps)} record(s)")
        fill_episode_table(self.eps, eps, show_exp=True)

    # ---- actions
    def compare_selected(self):
        sel = set(self.selected_ids())
        rows = [r for r in self.rows if r[0]["experiment_id"] in sel]
        if len(rows) < 2:
            QMessageBox.information(self, APP_NAME, "Select two or more experiments (Ctrl/Shift-click) to compare.")
            return
        CompareDialog(self, rows[:6]).exec_()

    def _scope_ids(self):
        ids = self.selected_ids()
        return ids if ids else [x["experiment_id"] for x, _ in self.rows]

    def export_csv(self):
        ids = self._scope_ids()
        if not ids:
            QMessageBox.information(self, APP_NAME, "Nothing to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export episodes (CSV)", os.path.join(
            os.path.expanduser("~"), safe_default_name("netra_episodes", "csv")), "CSV (*.csv)")
        if path:
            try:
                n = export_episodes_csv(path, self.main.db.get_episodes(experiment_ids=ids))
                self.main.write(f"exported {n} episode record(s) from {len(ids)} experiment(s) to {path}")
            except (OSError, DatabaseError) as e:
                self.main.report_error(f"EXPORT ERROR: {e}")

    def export_json(self):
        ids = self._scope_ids()
        if not ids:
            QMessageBox.information(self, APP_NAME, "Nothing to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export experiment report (JSON)", os.path.join(
            os.path.expanduser("~"), safe_default_name("netra_report", "json")), "JSON (*.json)")
        if path:
            try:
                n = export_report_json(path, self.main.db, ids)
                self.main.write(f"exported report with {n} experiment(s) to {path}")
            except (OSError, DatabaseError) as e:
                self.main.report_error(f"EXPORT ERROR: {e}")


# ------------------------------------------------------------------------------------------------ main window
class Main(QMainWindow):
    def __init__(self, db):
        super().__init__()
        self.db, self._restart_requested = db, False
        self.setWindowTitle(f"{APP_NAME} - {APP_SUB}")
        self.resize(1360, 960)
        self.s = load_settings(db)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_header())
        self._build_banner(root)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Live Control")
        self.dashboard = Dashboard(self)
        self.tabs.addTab(self.dashboard, "Comparison Dashboard")
        self.history = History(self)
        self.history.use_in_dashboard.connect(self._selected_to_dashboard)
        self.tabs.addTab(self.history, "Experiment History")

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        sp = QSplitter(Qt.Vertical)
        sp.addWidget(self.tabs)
        box = QGroupBox("Console and System Logs")
        bl = QVBoxLayout(box)
        bl.addWidget(self.console)
        sp.addWidget(box)
        sp.setStretchFactor(0, 5)
        sp.setStretchFactor(1, 1)
        sp.setSizes([760, 150])
        root.addWidget(sp, 1)

        for w in self.panels + self.tools:
            w.log.connect(self.write)
        for p in self.panels:
            p.error.connect(self.report_error)
            p.changed.connect(self._on_panel_changed)
        self.sb_db = QLabel()
        self.statusBar().addPermanentWidget(self.sb_db)
        self.statusBar().showMessage(f"{APP_NAME} {NETRA_VERSION}")
        self.refresh_views()
        self.write(f"{APP_NAME} {NETRA_VERSION} ready. Database: {db.path}")

    # ---- construction
    def _build_header(self):
        h = QHBoxLayout()
        col = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setStyleSheet(f"color:{BRAND};font-size:22pt;font-weight:700;")
        sub = QLabel(APP_SUB)
        sub.setStyleSheet(f"color:{TEXT2};font-size:10pt;")
        col.addWidget(title)
        col.addWidget(sub)
        h.addLayout(col)
        h.addStretch(1)
        for text, name, fn in (("Stop all", "danger", self.stop_all), ("Reset UI", None, self.reset_ui),
                               ("Restart application", None, self.restart_app),
                               ("Clear history...", "danger", self.clear_history)):
            b = QPushButton(text)
            if name:
                b.setObjectName(name)
            b.clicked.connect(fn)
            h.addWidget(b)
        return h

    def _build_banner(self, root):
        self.banner = QFrame()
        bl = QHBoxLayout(self.banner)
        self.banner_lbl = QLabel()
        self.banner_lbl.setWordWrap(True)
        bl.addWidget(self.banner_lbl, 1)
        x = QPushButton("Dismiss")
        x.clicked.connect(self.banner.hide)
        bl.addWidget(x)
        self.banner.hide()
        root.addWidget(self.banner)

    def _build_live_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        # Project configuration
        gp = QGroupBox("Project Configuration")
        g = QGridLayout(gp)
        self.paths = {}
        spec = [("files_ppo", "Simulator folder (run_policy_gui.py, nidar_env.py)", True),
                ("nidar_sim", "nidar_sim (fly_x2_keyboard.py, depth.py)", True),
                ("x2", "skydio_x2/x2.xml", False), ("scene", "skydio_x2/scene.xml", False),
                ("python", "Python interpreter / venv", False)]
        for r, (key, label, is_dir) in enumerate(spec):
            g.addWidget(QLabel(label), r, 0)
            e = QLineEdit(self.s[key])
            e.editingFinished.connect(self.save_settings)
            g.addWidget(e, r, 1)
            b = QPushButton("Browse")
            b.clicked.connect(lambda _, k=key, d=is_dir: self._browse(k, d))
            g.addWidget(b, r, 2)
            self.paths[key] = e
        bv = QPushButton("Validate paths and policies")
        bv.clicked.connect(self.validate_all)
        g.addWidget(bv, len(spec), 1, 1, 2, Qt.AlignRight)
        lay.addWidget(gp)

        # Experiment configuration
        ge = QGroupBox("Experiment Configuration")
        row = QHBoxLayout(ge)
        row.addWidget(QLabel("Survivor"))
        self.goal = QComboBox()
        self.goal.addItems(["random each episode"] + [f"survivor {i}" for i in range(1, 7)])
        self.goal.setCurrentIndex(max(0, min(6, int(self.s["goal"]))))
        row.addWidget(self.goal)
        row.addWidget(QLabel("Episodes"))
        self.eps = QSpinBox()
        self.eps.setRange(1, 1000)
        self.eps.setValue(int(self.s["episodes"]))
        row.addWidget(self.eps)
        row.addWidget(QLabel("Speed x"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 5.0)
        self.speed.setSingleStep(0.25)
        self.speed.setValue(float(self.s["speed"]))
        row.addWidget(self.speed)
        self.headless = QCheckBox("Headless (no window, fast)")
        self.headless.setChecked(self.s["headless"] == "1")
        row.addWidget(self.headless)
        row.addStretch(1)
        b_both = QPushButton("Launch both (compare)")
        b_both.setObjectName("primary")
        b_both.clicked.connect(self.launch_both)
        row.addWidget(b_both)
        for w in (self.goal,):
            w.currentIndexChanged.connect(self.save_settings)
        for w in (self.eps, self.speed):
            w.valueChanged.connect(self.save_settings)
        self.headless.stateChanged.connect(self.save_settings)
        lay.addWidget(ge)

        # Reflex ON / OFF panels
        self.panels = [
            RunPanel("Reflex ON  -  policy trained with reflex", "ON", self.db, self.cfg, self.s["run_on"]),
            RunPanel("Reflex OFF  -  policy trained without reflex", "OFF", self.db, self.cfg, self.s["run_off"])]
        self.panels[0].peer, self.panels[1].peer = self.panels[1], self.panels[0]
        sp = QSplitter(Qt.Horizontal)
        for p in self.panels:
            sp.addWidget(p)
        lay.addWidget(sp, 1)

        # Manual flight and simulation tools
        gt = QGroupBox("Manual Flight and Simulation Tools")
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
        k = QLabel("keys: W/S A/D move   R/F up/down   Q/E yaw   M mode   X brake   P reset")
        k.setStyleSheet(f"color:{TEXT2}")
        h.addWidget(k)
        lay.addWidget(gt)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    # ---- configuration
    def _browse(self, key, is_dir):
        cur = self.paths[key].text()
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "Choose folder", cur)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", cur)
        if p:
            self.paths[key].setText(p)
            self.save_settings()

    def cfg(self):
        t = lambda k: self.paths[k].text().strip()
        ns = t("nidar_sim")
        return dict(python=t("python"), files_ppo=t("files_ppo"), nidar_sim=ns, x2=t("x2"), scene=t("scene"),
                    arena=os.path.join(ns, "nidar_arena.xml"), layout=os.path.join(ns, "nidar_arena_layout.json"),
                    goal=self.goal.currentIndex(), episodes=self.eps.value(), speed=self.speed.value(),
                    headless=self.headless.isChecked())

    def save_settings(self, *_):
        if not hasattr(self, "panels"):
            return
        items = {k: e.text().strip() for k, e in self.paths.items()}
        items.update(run_on=self.panels[0].run_edit.text().strip(), run_off=self.panels[1].run_edit.text().strip(),
                     goal=self.goal.currentIndex(), episodes=self.eps.value(), speed=self.speed.value(),
                     headless="1" if self.headless.isChecked() else "0")
        try:
            self.db.set_settings(items)
        except DatabaseError as e:
            self.report_error(f"could not save settings: {e}")

    def validate_all(self):
        cfg, lines, ok = self.cfg(), [], True
        for p in self.panels:
            plan = prepare_launch(cfg, p.run_dir(), p.mode == "ON")
            lines.append(f"Reflex {p.mode}: " + ("OK" if plan.ok else "PROBLEMS FOUND"))
            lines += [f"   error: {e}" for e in plan.errors] + [f"   warning: {w}" for w in plan.warnings]
            if plan.ok:
                info = plan.config.get("policy_info", {})
                lines.append(f"   policy sha256 {plan.policy_sha[:16]}...  vecnorm sha256 {plan.vecnorm_sha[:16]}..."
                             + (f"  sb3 {info.get('sb3_version')}" if info.get("sb3_version") else ""))
            ok = ok and plan.ok
        for ln in lines:
            self.write(ln)
        (QMessageBox.information if ok else QMessageBox.warning)(self, APP_NAME, "\n".join(lines))

    # ---- actions
    def launch_both(self):
        self.save_settings()
        for p in self.panels:
            if not p.ctrl.is_running():
                p.launch()

    def stop_all(self):
        for p in self.panels:
            if p.ctrl.is_running():
                p.ctrl.request("stop")
        for t in self.tools:
            if t.running():
                kill_later(t.proc)

    def reset_ui(self):
        for p in self.panels:
            p.reset_live()
            p.lbl_latest.setText("latest outcome: -")
            p.lbl_latest.setStyleSheet("")
        self.history.reset_filters()
        self.dashboard.reset()
        self.banner.hide()
        self.write("UI reset (live displays and filters only - experiment history untouched)")
        self.refresh_views()

    def clear_history(self):
        e, n = self.db.counts()
        if QMessageBox.warning(
                self, "Clear history",
                f"This permanently deletes ALL {e} experiment(s) and {n} episode record(s) from the database.\n\n"
                "Settings are kept. A full backup copy of the database is saved first, so the data can be restored "
                "by copying that file back.\n\nDelete the history?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        for p in self.panels:
            if p.ctrl.is_running():
                QMessageBox.information(self, APP_NAME, "Stop the running simulations first.")
                return
        for p in self.panels:
            if p.ctrl.has_open_session():
                p.ctrl.request("stop")
        try:
            dest = self.db.clear_history()
        except DatabaseError as ex:
            self.report_error(f"CLEAR HISTORY FAILED (nothing was deleted if backup failed): {ex}")
            return
        for p in self.panels:
            p.view_exp_id = None
            p.ctrl.session = None
            p.reload_persisted()
            p._apply_state("IDLE", "")
        self.write(f"history cleared; backup saved to {dest}")
        self.refresh_views()

    def restart_app(self):
        running = any(p.ctrl.is_running() for p in self.panels)
        if QMessageBox.question(self, "Restart application",
                                ("Simulations are running and will be stopped (their episodes are recorded as "
                                 "interrupted). " if running else "") +
                                "Restart NETRA Command Center? Nothing is resumed automatically after the restart.",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._restart_requested = True
        self.close()

    def _selected_to_dashboard(self):
        self.dashboard.scope.setCurrentIndex(2)
        self.tabs.setCurrentWidget(self.dashboard)
        self.dashboard.refresh()

    def _on_panel_changed(self):
        self.save_settings()
        self.refresh_views()

    def refresh_views(self):
        self.dashboard.refresh()
        self.history.refresh()
        try:
            e, n = self.db.counts()
            sp = self.db.spool_size()
            self.sb_db.setText(f"Database: {self.db.path}   |   {e} experiments, {n} episodes"
                               + (f"   |   {sp} PENDING in spool!" if sp else ""))
        except Exception as ex:                                         # status bar must never crash the UI
            self.sb_db.setText(f"Database: {ex}")

    # ---- messaging
    def write(self, msg):
        self.console.appendPlainText(f"{time.strftime('%H:%M:%S')}  {msg}")

    def show_banner(self, msg, error=True):
        col, bg = (RED, "#FBE9E6") if error else (ACCENT, "#E6EFF1")
        self.banner.setStyleSheet(f"QFrame{{background:{bg};border:1px solid {col};border-radius:8px;}}")
        self.banner_lbl.setText(msg)
        self.banner.show()

    def report_error(self, msg):
        self.write("ERROR  " + msg)
        if "DATABASE" in msg.upper() or "SPOOL" in msg.upper():
            self.show_banner(msg)

    def startup_notices(self, orphans, imported):
        notes = []
        if orphans:
            notes.append(f"The previous session ended unexpectedly: {len(orphans)} experiment(s) were marked INTERRUPTED "
                         "(completed episodes are intact). Nothing was resumed automatically.")
        if imported:
            notes.append(f"{imported} episode record(s) saved to the spool file after an earlier database error were "
                         "imported now.")
        for n in notes:
            self.write(n)
        if notes:
            self.show_banner(" ".join(notes), error=False)

    # ---- shutdown
    def closeEvent(self, e):
        if not self._restart_requested and any(p.ctrl.is_running() for p in self.panels):
            if QMessageBox.question(self, "Quit", "Simulations are running. Stop them and quit?\n"
                                    "Unfinished episodes will be recorded as interrupted.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                e.ignore()
                return
        self.save_settings()
        for p in self.panels:
            p.shutdown()
        for t in self.tools:
            if t.running():
                t.proc.kill()
                t.proc.waitForFinished(1000)
        self.db.close()
        if self._restart_requested:
            QProcess.startDetached(sys.executable, [os.path.abspath(sys.argv[0])] + sys.argv[1:])
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    app.setApplicationName(APP_NAME)
    try:
        db = ExperimentDatabase()
        orphans = db.recover_orphans()
        imported = db.flush_spool()
    except DatabaseError as e:
        QMessageBox.critical(None, APP_NAME, f"The experiment database could not be opened:\n{e}\n\n"
                             "Experiment results cannot be recorded, so the application will not start.")
        return 1
    win = Main(db)
    win.startup_notices(orphans, imported)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
