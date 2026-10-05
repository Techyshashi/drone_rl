"""
Skydio X2 in the NIDAR arena: onboard (ego) camera view + live IMU panel + 6-DOF hold-to-fly keys.

Needs only fly_x2_keyboard.py (same folder), nidar_arena.xml, nidar_arena_layout.json, mujoco>=3.2, numpy.
The ego view and IMU readings open in a separate Tk window (standard library, no extra install).

Keys (hold; works whichever window has focus, but keep the MuJoCo viewer window UNfocused so its own
shortcuts don't fire):
  VEL mode (default)  W/S x velocity   A/D y velocity   R/F z   Q/E yaw rate
  ATT mode (M)        W/S pitch angle  A/D roll angle   R/F z   Q/E yaw rate
  M toggle mode   X brake   P reset to launch pad

Camera: --nose picks the body axis the camera looks along (default +x, verified with ego.png).
"""
import argparse
import ctypes
import json
import math
import os
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer

from fly_x2_keyboard import (Drone, quat_to_rpy, wrap, DEFAULT_X2, DEFAULT_ARENA, DEFAULT_LAYOUT,
                             Z_MIN, Z_MAX, KV, A_MAX, TILT_MAX, KZ, KVZ, AZ_MAX,
                             KA_P, KA_D, KY_P, KY_D)

# ------------------------------------------------------------------ camera mounting
CAM_DIST = 0.15                 # metres from the body origin along the nose axis (unchanged)
CAM_FOVY = 90.0                 # vertical field of view, degrees (unchanged)
NOSES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
         "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}
_UP = np.array([0.0, 0.0, 1.0])


def camera_frame(nose):
    """Return (pos, 3x3 row-major matrix) for a camera looking along `nose` (body frame), upright.
    Columns are the camera axes in the body frame: x_cam (image right) = nose x up,
    y_cam (image up) = up, z_cam = -nose (MuJoCo cameras look along -z)."""
    f = np.array(nose, dtype=float)
    pos = (CAM_DIST * f).tolist()
    mat = np.column_stack([np.cross(f, _UP), _UP, -f]).ravel()
    return pos, mat


_u32 = ctypes.windll.user32
VK = {"W": 0x57, "S": 0x53, "A": 0x41, "D": 0x44, "Q": 0x51, "E": 0x45,
      "R": 0x52, "F": 0x46, "P": 0x50, "M": 0x4D, "X": 0x58}
HOLD_SPEED, HOLD_YAW, HOLD_ACC, Z_RATE = 0.6, 0.8, 1.5, 0.4
ATT_MAX, ATT_RATE = 0.25, 1.0


# ------------------------------------------------------------------ model with an onboard camera
def build_model_ego(x2_xml, arena_xml, nose=NOSES["+x"]):
    if not hasattr(mujoco, "MjSpec"):
        raise SystemExit(f"mujoco {mujoco.__version__} has no MjSpec; run: pip install -U mujoco")
    if not os.path.isfile(x2_xml):
        raise SystemExit(f"x2.xml not found: {x2_xml}")
    spec = mujoco.MjSpec.from_file(x2_xml)
    wb = spec.worldbody
    types = {"box": mujoco.mjtGeom.mjGEOM_BOX, "plane": mujoco.mjtGeom.mjGEOM_PLANE}
    n = 0
    for g in ET.parse(arena_xml).getroot().find("worldbody").findall("geom"):
        kw = dict(type=types[g.get("type")],
                  pos=[float(v) for v in g.get("pos").split()],
                  size=[float(v) for v in g.get("size").split()],
                  rgba=[float(v) for v in g.get("rgba").split()])
        if g.get("name"):
            kw["name"] = g.get("name")
        if g.get("contype") is not None:
            kw["contype"] = int(g.get("contype"))
        if g.get("conaffinity") is not None:
            kw["conaffinity"] = int(g.get("conaffinity"))
        wb.add_geom(**kw)
        n += 1
    try:   # extra overhead light for better shading in the camera image (optional)
        wb.add_light(pos=[7.4, 7.4, 10.0], dir=[0, 0, -1], castshadow=False)
    except Exception as e:
        print("[build] note: could not add light:", e)

    body = spec.body("x2")
    if body is None:
        raise SystemExit("body 'x2' not found in x2.xml")
    cam_pos, cam_mat = camera_frame(nose)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, cam_mat)
    body.add_camera(name="ego", pos=cam_pos, quat=quat.tolist(), fovy=CAM_FOVY)

    model = spec.compile()

    # sanity checks: the vision camera must exist and be rigidly attached to the drone body
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    if cid < 0:
        raise SystemExit("camera 'ego' missing from compiled model")
    if model.cam_bodyid[cid] != mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2"):
        raise SystemExit("camera 'ego' is not attached to body 'x2'")

    model.vis.map.znear = 0.001            # znear is a fraction of stat.extent; default would clip near walls
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.6
    print(f"[build] x2 + {n} arena geoms + camera 'ego' (nose {list(map(float, nose))}, pos {cam_pos}) | "
          f"mujoco {mujoco.__version__} | extent {model.stat.extent:.1f} m -> "
          f"znear {model.vis.map.znear * model.stat.extent:.3f} m")
    return model


# ------------------------------------------------------------------ 6-DOF controller (same as the working hold script)
class Drone6(Drone):
    def __init__(self, model, layout):
        super().__init__(model, layout)
        self.mode = "VEL"
        self.roll_cmd = self.pitch_cmd = 0.0
        self._p_was = self._m_was = False

    def reset(self, data):
        super().reset(data)
        self.mode = "VEL"
        self.roll_cmd = self.pitch_cmd = 0.0

    def brake(self):
        self.mode = "VEL"
        self.roll_cmd = self.pitch_cmd = 0.0
        self.cmd["vx"] = self.cmd["vy"] = self.cmd["yaw_rate"] = 0.0

    def toggle_mode(self):
        if self.mode == "VEL":
            self.mode = "ATT"
            self.cmd["vx"] = self.cmd["vy"] = 0.0
            self.roll_cmd = self.pitch_cmd = 0.0
        else:
            self.brake()

    def control(self, data):
        dt = self.m.opt.timestep
        p, q = self.pose(data)
        v = data.qvel[self.dadr:self.dadr + 3]
        w = data.qvel[self.dadr + 3:self.dadr + 6]
        roll, pitch, yaw = quat_to_rpy(q)
        c = self.cmd
        if self.mode == "VEL":
            cy, sy = math.cos(yaw), math.sin(yaw)
            vx_b = cy * v[0] + sy * v[1]
            vy_b = -sy * v[0] + cy * v[1]
            ax = max(-A_MAX, min(A_MAX, KV * (c["vx"] - vx_b)))
            ay = max(-A_MAX, min(A_MAX, KV * (c["vy"] - vy_b)))
            pitch_des = max(-TILT_MAX, min(TILT_MAX, ax / self.g))
            roll_des = max(-TILT_MAX, min(TILT_MAX, -ay / self.g))
        else:
            pitch_des, roll_des = self.pitch_cmd, self.roll_cmd
        az = max(-AZ_MAX, min(AZ_MAX, KZ * (c["z"] - p[2]) - KVZ * v[2]))
        thrust = self.mass * (self.g + az) / max(math.cos(roll) * math.cos(pitch), 0.5)
        self.yaw_target += c["yaw_rate"] * dt
        err = max(-0.8, min(0.8, wrap(self.yaw_target - yaw)))
        self.yaw_target = yaw + err
        tau = self.inertia * np.array([KA_P * (roll_des - roll) - KA_D * w[0],
                                       KA_P * (pitch_des - pitch) - KA_D * w[1],
                                       KY_P * err - KY_D * w[2]])
        ctrl = self.Binv @ np.array([thrust, *tau])
        for i in range(self.m.nu):
            if self.limited[i]:
                ctrl[i] = min(max(ctrl[i], self.crange[i, 0]), self.crange[i, 1])
        data.ctrl[:] = ctrl


def held(name):
    return bool(_u32.GetAsyncKeyState(VK[name]) & 0x8000)


def approach(cur, target, rate, dt):
    return cur + max(-rate * dt, min(rate * dt, target - cur))


def poll_keys(drone, dt):
    c = drone.cmd
    m_now = held("M")
    if m_now and not drone._m_was:
        drone.toggle_mode()
    drone._m_was = m_now
    p_now = held("P")
    if p_now and not drone._p_was:
        drone.pending_reset = True
    drone._p_was = p_now
    if held("X"):
        drone.brake()
    c["yaw_rate"] = approach(c["yaw_rate"], HOLD_YAW * (held("Q") - held("E")), 3.0, dt)
    if held("R"):
        c["z"] = min(Z_MAX, c["z"] + Z_RATE * dt)
    if held("F"):
        c["z"] = max(Z_MIN, c["z"] - Z_RATE * dt)
    if drone.mode == "VEL":
        c["vx"] = approach(c["vx"], HOLD_SPEED * (held("W") - held("S")), HOLD_ACC, dt)
        c["vy"] = approach(c["vy"], HOLD_SPEED * (held("A") - held("D")), HOLD_ACC, dt)
    else:
        drone.pitch_cmd = approach(drone.pitch_cmd, ATT_MAX * (held("W") - held("S")), ATT_RATE, dt)
        drone.roll_cmd = approach(drone.roll_cmd, -ATT_MAX * (held("A") - held("D")), ATT_RATE, dt)


# ------------------------------------------------------------------ Tk panel: ego image + IMU numbers + rolling plots
class Panel:
    def __init__(self, zoom, hist=200):
        self.root = tk.Tk()
        self.root.title("X2 ego view + live IMU")
        self.alive = True
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.zoom = zoom
        self._photo = None
        self.img = tk.Label(self.root, text="(no ego image)", bg="black", fg="white")
        self.img.grid(row=0, column=0, rowspan=3, padx=4, pady=4)
        self.txt = tk.Label(self.root, font=("Consolas", 10), justify="left", anchor="nw", width=46)
        self.txt.grid(row=0, column=1, sticky="nw", padx=4)
        self.cg = tk.Canvas(self.root, width=380, height=110, bg="black", highlightthickness=0)
        self.cg.grid(row=1, column=1, padx=4, pady=2)
        self.ca = tk.Canvas(self.root, width=380, height=110, bg="black", highlightthickness=0)
        self.ca.grid(row=2, column=1, padx=4, pady=2)
        self.hg = deque(maxlen=hist)
        self.ha = deque(maxlen=hist)

    def _close(self):
        self.alive = False

    @staticmethod
    def _plot(canvas, hist, rng, title):
        canvas.delete("all")
        w, h = int(canvas["width"]), int(canvas["height"])
        canvas.create_line(0, h / 2, w, h / 2, fill="#444")
        canvas.create_text(4, 2, anchor="nw", fill="#aaa", font=("Consolas", 8),
                           text=f"{title}  (+/-{rng})   red=x green=y blue=z")
        if len(hist) < 2:
            return
        arr = np.array(hist)
        xs = np.linspace(0, w, len(arr))
        for k, col in enumerate(("#ff5555", "#55ff55", "#5599ff")):
            ys = h / 2 - np.clip(arr[:, k] / rng, -1, 1) * (h / 2 - 2)
            canvas.create_line(*np.column_stack([xs, ys]).ravel().tolist(), fill=col)

    def update(self, rgb, text, gyro, acc):
        if not self.alive:
            return
        if rgb is not None:
            h, w, _ = rgb.shape
            ppm = b"P6\n%d %d\n255\n" % (w, h) + np.ascontiguousarray(rgb).tobytes()
            photo = tk.PhotoImage(width=w, height=h, data=ppm, format="PPM")
            if self.zoom > 1:
                photo = photo.zoom(self.zoom)
            self._photo = photo
            self.img.configure(image=photo, text="")
        self.txt.configure(text=text)
        self.hg.append(gyro)
        self.ha.append(acc)
        self._plot(self.cg, self.hg, 3.0, "gyro rad/s")
        self._plot(self.ca, self.ha, 20.0, "accel m/s^2")
        try:
            self.root.update()
        except tk.TclError:
            self.alive = False


def sensor(model, data, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if i < 0:
        return np.zeros(3)
    a, n = model.sensor_adr[i], model.sensor_dim[i]
    return data.sensordata[a:a + n].copy()


def imu_text(model, drone, d, t):
    p, q = drone.pose(d)
    v = d.qvel[drone.dadr:drone.dadr + 3]
    r, pi, ya = np.degrees(quat_to_rpy(q))
    gyro = sensor(model, d, "body_gyro")
    acc = sensor(model, d, "body_linacc")
    quat = sensor(model, d, "body_quat")
    mot = d.ctrl
    s = (f"mode {drone.mode:<3}   t={t:6.1f} s   box {drone.grid_box(p):>7}"
         f"{'   COLLISION' if drone.touching(d) else ''}\n"
         f"pos  x={p[0]:6.2f} y={p[1]:6.2f} z={p[2]:5.2f} m\n"
         f"vel  x={v[0]:+6.2f} y={v[1]:+6.2f} z={v[2]:+6.2f} m/s\n"
         f"att  roll={r:+6.1f} pitch={pi:+6.1f} yaw={ya:+7.1f} deg\n"
         f"gyro  x={gyro[0]:+7.3f} y={gyro[1]:+7.3f} z={gyro[2]:+7.3f} rad/s\n"
         f"acc   x={acc[0]:+7.3f} y={acc[1]:+7.3f} z={acc[2]:+7.3f} |a|={np.linalg.norm(acc):6.2f}\n"
         f"quat  w={quat[0]:+.3f} x={quat[1]:+.3f} y={quat[2]:+.3f} z={quat[3]:+.3f}\n"
         f"motors N  {mot[0]:5.2f} {mot[1]:5.2f} {mot[2]:5.2f} {mot[3]:5.2f}")
    return s, gyro, acc


# ------------------------------------------------------------------ main loop
def run(model, drone, args):
    d = mujoco.MjData(model)
    drone.reset(d)
    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    except Exception as e:
        print(f"[ego] could not create offscreen renderer ({e}); continuing with IMU only")
    panel = Panel(args.zoom)
    print(__doc__)
    ego_every = max(1, int(round(1.0 / args.fps / model.opt.timestep)))
    try:
        with mujoco.viewer.launch_passive(model, d, show_left_ui=False, show_right_ui=False) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = drone.bid
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 90, -70
            dt = model.opt.timestep
            nxt, k = time.time(), 0
            while v.is_running() and panel.alive:
                if drone.pending_reset:
                    drone.reset(d)
                    drone.pending_reset = False
                poll_keys(drone, dt)
                drone.control(d)
                mujoco.mj_step(model, d)
                k += 1
                if k % 2 == 0:
                    v.sync()
                if k % ego_every == 0:
                    rgb = None
                    if renderer is not None:
                        renderer.update_scene(d, camera="ego")
                        rgb = renderer.render()
                    text, gyro, acc = imu_text(model, drone, d, d.time)
                    panel.update(rgb, text, gyro, acc)
                nxt += dt
                now = time.time()
                if nxt < now - 0.1:
                    nxt = now                      # running slower than real time: don't try to catch up
                time.sleep(max(0.0, nxt - now))
    finally:
        if renderer is not None:
            renderer.close()
        try:
            panel.root.destroy()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--zoom", type=int, default=2, help="integer enlargement of the ego image")
    ap.add_argument("--fps", type=float, default=20.0, help="ego view / IMU panel refresh rate")
    ap.add_argument("--nose", choices=list(NOSES), default="+x",
                    help="body axis the camera looks along (default +x)")
    a = ap.parse_args()
    layout = json.load(open(a.layout, encoding="utf-8"))
    model = build_model_ego(a.x2, a.arena, NOSES[a.nose])
    run(model, Drone6(model, layout), a)


if __name__ == "__main__":
    main()