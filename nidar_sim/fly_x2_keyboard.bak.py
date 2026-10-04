"""
Fly the MuJoCo Menagerie Skydio X2 through the NIDAR arena with the keyboard.

Needs: mujoco >= 3.2 (MjSpec), numpy.
Files:  nidar_arena.xml + nidar_arena_layout.json (from nidar_arena_gen.py), and x2.xml from mujoco_menagerie.

  python fly_x2_keyboard.py --x2 <path>\\x2.xml --inspect     # print what the model actually contains
  python fly_x2_keyboard.py --x2 <path>\\x2.xml --selftest    # headless hover / forward-flight check
  python fly_x2_keyboard.py --x2 <path>\\x2.xml               # keyboard flight in the viewer

Design notes
  * The X2 has 4 thrust motors. Instead of assuming the rotor layout / signs, the mixer is MEASURED:
    each actuator is driven with ctrl = 1 at identity attitude and the resulting generalized force on the
    free joint is read from qfrc_actuator. u = [thrust, tau_x, tau_y, tau_z] is then solved for ctrl.
  * The viewer's key callback only reports key PRESSES (no release), so keys change latched setpoints.
"""
import argparse
import json
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

# ------------------------------------------------------------------ user-tunable settings
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_X2 = os.environ.get(
    "X2_XML", r"C:\Users\HP\Downloads\mujoco_menagerie\skydio_x2\x2.xml")
DEFAULT_ARENA = os.path.join(HERE, "nidar_arena.xml")
DEFAULT_LAYOUT = os.path.join(HERE, "nidar_arena_layout.json")

START_YAW_DEG = 90.0      # assumes the X2 nose is +x (hypothesis); 90 deg -> nose points +y, into the gate
START_Z = 0.25            # spawn height above the launch pad (m)
TAKEOFF_Z = 1.0           # initial altitude setpoint (m)
Z_MIN, Z_MAX = 0.25, 2.0  # altitude setpoint limits (ceiling net is at 2.44 m)

V_STEP, V_MAX = 0.10, 0.8       # m/s per key press, max horizontal speed
YAW_STEP, YAW_MAX = 0.3, 1.2    # rad/s per key press, max yaw rate
Z_STEP = 0.10                   # m per key press

# gains (conservative first guess; torques are scaled by the model's inertia)
KV, A_MAX, TILT_MAX = 2.0, 2.5, 0.30      # velocity P gain (1/s), accel limit (m/s^2), tilt limit (rad)
KZ, KVZ, AZ_MAX = 6.0, 4.0, 3.0           # altitude PD
KA_P, KA_D = 40.0, 10.0                   # roll/pitch (rad/s^2 per rad, per rad/s)
KY_P, KY_D = 10.0, 5.0                    # yaw

# GLFW key codes
KEYMAP = {
    87: "W", 83: "S", 65: "A", 68: "D",    # forward / back / left / right
    81: "Q", 69: "E",                      # yaw left / right
    82: "R", 70: "F",                      # up / down
    88: "X",                               # stop (zero velocity + yaw rate, hold position)
    80: "P",                               # reset to launch pad
}
HELP = """
 W/S  forward/back     A/D  strafe left/right     Q/E  yaw left/right
 R/F  altitude up/down X    stop (hover in place) P    reset to launch pad
 Each press changes a latched setpoint; press X to brake. Close the window to quit.
"""


# ------------------------------------------------------------------ helpers
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def build_model(x2_xml, arena_xml):
    """Load x2.xml (so its assets resolve) and add the arena geoms from nidar_arena.xml."""
    if not hasattr(mujoco, "MjSpec"):
        raise SystemExit(f"mujoco {mujoco.__version__} has no MjSpec; run: pip install -U mujoco")
    if not os.path.isfile(x2_xml):
        raise SystemExit(f"x2.xml not found: {x2_xml}\nPass --x2 <full path to ...\\skydio_x2\\x2.xml>")
    spec = mujoco.MjSpec.from_file(x2_xml)
    wb = spec.worldbody
    types = {"box": mujoco.mjtGeom.mjGEOM_BOX, "plane": mujoco.mjtGeom.mjGEOM_PLANE}
    root = ET.parse(arena_xml).getroot()
    n = 0
    for g in root.find("worldbody").findall("geom"):
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
    model = spec.compile()
    print(f"[build] x2 model + {n} arena geoms | mujoco {mujoco.__version__} | timestep {model.opt.timestep}")
    return model


def find_drone_body(model):
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free:
        raise SystemExit("no free joint found - is this the x2 model?")
    return int(model.jnt_bodyid[free[0]]), int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])


def measure_mixer(model, dofadr, qposadr):
    """Columns = wrench [Fx Fy Fz tx ty tz] produced by ctrl_i = 1 at identity attitude."""
    d = mujoco.MjData(model)
    cols = []
    for i in range(model.nu):
        mujoco.mj_resetData(model, d)
        d.qpos[qposadr + 3:qposadr + 7] = [1, 0, 0, 0]
        d.ctrl[:] = 0
        d.ctrl[i] = 1.0
        mujoco.mj_forward(model, d)
        cols.append(d.qfrc_actuator[dofadr:dofadr + 6].copy())
    B = np.array(cols).T                  # 6 x nu
    return B


class Drone:
    def __init__(self, model, layout):
        self.m = model
        self.bid, self.qadr, self.dadr = find_drone_body(model)
        if model.nu != 4:
            raise SystemExit(f"expected 4 motors, model has nu={model.nu}")
        self.B6 = measure_mixer(model, self.dadr, self.qadr)
        self.B4 = self.B6[[2, 3, 4, 5], :]          # thrust(z), tau_x, tau_y, tau_z
        if np.linalg.matrix_rank(self.B4, tol=1e-6) < 4:
            raise SystemExit(f"mixer is rank-deficient:\n{np.round(self.B6, 4)}")
        self.Binv = np.linalg.inv(self.B4)
        self.mass = float(model.body_subtreemass[self.bid])
        self.inertia = np.array(model.body_inertia[self.bid], dtype=float)
        self.g = -float(model.opt.gravity[2])
        self.limited = model.actuator_ctrllimited.astype(bool)
        self.crange = model.actuator_ctrlrange.copy()
        self.layout = layout
        pad = layout["launch_pad"]["xy"]
        self.start_xy = (pad[0], pad[1])
        self.cmd = {}
        self.yaw_target = 0.0
        self.pending_reset = False

    # ---- state
    def reset(self, data):
        mujoco.mj_resetData(self.m, data)
        yaw = math.radians(START_YAW_DEG)
        data.qpos[self.qadr:self.qadr + 3] = [self.start_xy[0], self.start_xy[1], START_Z]
        data.qpos[self.qadr + 3:self.qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.ctrl[:] = 0
        self.cmd = dict(vx=0.0, vy=0.0, yaw_rate=0.0, z=TAKEOFF_Z)
        self.yaw_target = yaw
        mujoco.mj_forward(self.m, data)

    def pose(self, data):
        p = data.qpos[self.qadr:self.qadr + 3].copy()
        q = data.qpos[self.qadr + 3:self.qadr + 7].copy()
        return p, q

    # ---- keyboard
    def on_key(self, keycode):
        k = KEYMAP.get(keycode)
        if k is None:
            return
        c = self.cmd
        clip = lambda v, lim: max(-lim, min(lim, v))
        if k == "W": c["vx"] = clip(c["vx"] + V_STEP, V_MAX)
        elif k == "S": c["vx"] = clip(c["vx"] - V_STEP, V_MAX)
        elif k == "A": c["vy"] = clip(c["vy"] + V_STEP, V_MAX)
        elif k == "D": c["vy"] = clip(c["vy"] - V_STEP, V_MAX)
        elif k == "Q": c["yaw_rate"] = clip(c["yaw_rate"] + YAW_STEP, YAW_MAX)
        elif k == "E": c["yaw_rate"] = clip(c["yaw_rate"] - YAW_STEP, YAW_MAX)
        elif k == "R": c["z"] = min(Z_MAX, c["z"] + Z_STEP)
        elif k == "F": c["z"] = max(Z_MIN, c["z"] - Z_STEP)
        elif k == "X": c["vx"] = c["vy"] = c["yaw_rate"] = 0.0
        elif k == "P": self.pending_reset = True

    # ---- control law
    def control(self, data):
        dt = self.m.opt.timestep
        p, q = self.pose(data)
        v = data.qvel[self.dadr:self.dadr + 3]
        w = data.qvel[self.dadr + 3:self.dadr + 6]       # body-frame angular velocity
        roll, pitch, yaw = quat_to_rpy(q)
        c = self.cmd

        cy, sy = math.cos(yaw), math.sin(yaw)
        vx_b = cy * v[0] + sy * v[1]
        vy_b = -sy * v[0] + cy * v[1]
        ax = max(-A_MAX, min(A_MAX, KV * (c["vx"] - vx_b)))
        ay = max(-A_MAX, min(A_MAX, KV * (c["vy"] - vy_b)))
        pitch_des = max(-TILT_MAX, min(TILT_MAX, ax / self.g))     # +pitch tilts thrust toward +x_body
        roll_des = max(-TILT_MAX, min(TILT_MAX, -ay / self.g))     # +roll tilts thrust toward -y_body

        az = max(-AZ_MAX, min(AZ_MAX, KZ * (c["z"] - p[2]) - KVZ * v[2]))
        thrust = self.mass * (self.g + az) / max(math.cos(roll) * math.cos(pitch), 0.5)

        self.yaw_target += c["yaw_rate"] * dt
        err = wrap(self.yaw_target - yaw)
        err = max(-0.8, min(0.8, err))
        self.yaw_target = yaw + err                                # anti-windup when blocked

        tau = self.inertia * np.array([KA_P * (roll_des - roll) - KA_D * w[0],
                                       KA_P * (pitch_des - pitch) - KA_D * w[1],
                                       KY_P * err - KY_D * w[2]])
        ctrl = self.Binv @ np.array([thrust, *tau])
        for i in range(self.m.nu):
            if self.limited[i]:
                ctrl[i] = min(max(ctrl[i], self.crange[i, 0]), self.crange[i, 1])
        data.ctrl[:] = ctrl

    # ---- reporting
    def grid_box(self, p):
        L = self.layout
        P, T, N = L["cell_pitch_m"], L["wall_thickness_m"], L["grid_n"]
        c, r = int((p[0] - T) // P), int((p[1] - T) // P)
        return f"{chr(ord('A') + c)}{r + 1}" if 0 <= c < N and 0 <= r < N else "outside"

    def touching(self, data):
        for i in range(data.ncon):
            con = data.contact[i]
            r1 = self.m.body_rootid[self.m.geom_bodyid[con.geom1]]
            r2 = self.m.body_rootid[self.m.geom_bodyid[con.geom2]]
            if (r1 == self.bid) != (r2 == self.bid):
                return True
        return False


# ------------------------------------------------------------------ modes
def do_inspect(model, drone):
    print("\n=== INSPECT (observations from the compiled model) ===")
    print("timestep:", model.opt.timestep, "| gravity:", model.opt.gravity)
    print("bodies:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)][:12])
    print("drone body:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, drone.bid),
          "| mass:", round(drone.mass, 4), "| principal inertia:", np.round(drone.inertia, 5))
    print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
    print("ctrl limited:", drone.limited, "| ctrlrange:\n", drone.crange)
    print("measured wrench per unit ctrl (rows Fx Fy Fz tx ty tz, cols = motors):\n", np.round(drone.B6, 4))
    hover = drone.mass * drone.g / max(drone.B4[0].sum(), 1e-9)
    print("approx. per-motor ctrl for hover:", round(hover, 3))
    print("camera/sensor names:",
          [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)])
    print("keyframes:", model.nkey)


def do_selftest(model, drone):
    d = mujoco.MjData(model)
    drone.reset(d)
    steps = lambda sec: int(sec / model.opt.timestep)
    for _ in range(steps(6.0)):
        drone.control(d)
        mujoco.mj_step(model, d)
    p, q = drone.pose(d)
    print(f"[selftest] after 6 s hover cmd: z={p[2]:.3f} (target {drone.cmd['z']}), "
          f"xy drift=({p[0]-drone.start_xy[0]:+.3f},{p[1]-drone.start_xy[1]:+.3f}), "
          f"rpy(deg)={np.round(np.degrees(quat_to_rpy(q)), 1)}, touching={drone.touching(d)}")
    drone.cmd["vx"] = 0.5
    p0, _ = drone.pose(d)
    for _ in range(steps(3.0)):
        drone.control(d)
        mujoco.mj_step(model, d)
    p1, q1 = drone.pose(d)
    yaw = quat_to_rpy(q1)[2]
    fwd = (p1[0] - p0[0]) * math.cos(yaw) + (p1[1] - p0[1]) * math.sin(yaw)
    print(f"[selftest] 3 s with vx=0.5: moved {fwd:+.2f} m along body-x (expect about +1.2..1.5), "
          f"z={p1[2]:.3f}, speed={np.linalg.norm(d.qvel[drone.dadr:drone.dadr+3]):.2f} m/s")
    print("[selftest] if z diverges or the drone tumbles, paste this output and the --inspect output.")


def run_viewer(model, drone):
    import mujoco.viewer
    d = mujoco.MjData(model)
    drone.reset(d)
    print(HELP)
    with mujoco.viewer.launch_passive(model, d, key_callback=drone.on_key) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = drone.bid
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 90, -70
        dt = model.opt.timestep
        nxt, last_hud, k = time.time(), 0.0, 0
        while v.is_running():
            if drone.pending_reset:
                drone.reset(d)
                drone.pending_reset = False
            drone.control(d)
            mujoco.mj_step(model, d)
            k += 1
            if k % 2 == 0:
                v.sync()
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - last_hud > 0.5:
                last_hud = time.time()
                p, q = drone.pose(d)
                c = drone.cmd
                print(f"\rpos=({p[0]:5.2f},{p[1]:5.2f},{p[2]:4.2f}) box={drone.grid_box(p):>7} "
                      f"cmd vx={c['vx']:+.1f} vy={c['vy']:+.1f} yaw={c['yaw_rate']:+.1f} z={c['z']:.1f} "
                      f"{'COLLISION' if drone.touching(d) else '         '}", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    layout = json.load(open(a.layout, encoding="utf-8"))
    model = build_model(a.x2, a.arena)
    drone = Drone(model, layout)
    if a.inspect:
        do_inspect(model, drone)
    elif a.selftest:
        do_selftest(model, drone)
    else:
        run_viewer(model, drone)


if __name__ == "__main__":
    main()