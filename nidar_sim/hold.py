"""
Hold-to-fly keyboard control for the Skydio X2 in the NIDAR arena (Windows).
Keep the TERMINAL focused (not the MuJoCo window) so the viewer's own shortcuts don't fire.

Two modes (toggle with M):
  VEL (default)  W/S = x velocity (forward/back)   A/D = y velocity (left/right)
                 R/F = z (altitude)                Q/E = yaw rate.   Roll and pitch are set automatically.
  ATT            W/S = PITCH angle (nose down/up)  A/D = ROLL angle  (left/right)
                 R/F = z (altitude)                Q/E = yaw rate.   x/y motion is the RESULT of the tilt.
Other keys: X = brake (back to VEL mode, zero velocity)   P = reset to launch pad   M = toggle mode
A quadrotor has only 4 independent inputs (thrust + 3 torques), so roll/pitch and x/y velocity cannot be
commanded independently: in VEL mode you command x,y,z,yaw; in ATT mode you command roll,pitch,z,yaw.
"""
import argparse
import ctypes
import json
import math
import time

import numpy as np
import mujoco
import mujoco.viewer

from fly_x2_keyboard import (build_model, Drone, quat_to_rpy, wrap, DEFAULT_X2, DEFAULT_ARENA,
                             DEFAULT_LAYOUT, Z_MIN, Z_MAX, KV, A_MAX, TILT_MAX, KZ, KVZ, AZ_MAX,
                             KA_P, KA_D, KY_P, KY_D)

_u32 = ctypes.windll.user32
VK = {"W": 0x57, "S": 0x53, "A": 0x41, "D": 0x44, "Q": 0x51, "E": 0x45,
      "R": 0x52, "F": 0x46, "P": 0x50, "M": 0x4D, "X": 0x58}
HOLD_SPEED, HOLD_YAW, HOLD_ACC, Z_RATE = 0.6, 0.8, 1.5, 0.4   # m/s, rad/s, m/s^2, m/s
ATT_MAX, ATT_RATE = 0.25, 1.0                                  # max tilt (rad ~ 14 deg), tilt change rate rad/s


class Drone6(Drone):
    def __init__(self, model, layout):
        super().__init__(model, layout)
        self.mode = "VEL"
        self.roll_cmd = 0.0
        self.pitch_cmd = 0.0
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
        # +pitch tilts thrust toward +x (forward); +roll tilts thrust toward -y, so "left" is negative roll
        drone.pitch_cmd = approach(drone.pitch_cmd, ATT_MAX * (held("W") - held("S")), ATT_RATE, dt)
        drone.roll_cmd = approach(drone.roll_cmd, -ATT_MAX * (held("A") - held("D")), ATT_RATE, dt)


def run(model, drone):
    d = mujoco.MjData(model)
    drone.reset(d)
    print(__doc__)
    with mujoco.viewer.launch_passive(model, d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = drone.bid
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 90, -70
        dt = model.opt.timestep
        nxt, last_hud, k = time.time(), 0.0, 0
        while v.is_running():
            if drone.pending_reset:
                drone.reset(d)
                drone.pending_reset = False
            poll_keys(drone, dt)
            drone.control(d)
            mujoco.mj_step(model, d)
            k += 1
            if k % 2 == 0:
                v.sync()
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - last_hud > 0.4:
                last_hud = time.time()
                p, q = drone.pose(d)
                r, pi, ya = np.degrees(quat_to_rpy(q))
                print(f"\r{drone.mode} x={p[0]:5.2f} y={p[1]:5.2f} z={p[2]:4.2f} | "
                      f"roll={r:+5.1f} pitch={pi:+5.1f} yaw={ya:+6.1f} | box={drone.grid_box(p):>7} "
                      f"{'COLLISION' if drone.touching(d) else '         '}", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    a = ap.parse_args()
    layout = json.load(open(a.layout, encoding="utf-8"))
    model = build_model(a.x2, a.arena)
    run(model, Drone6(model, layout))


if __name__ == "__main__":
    main()