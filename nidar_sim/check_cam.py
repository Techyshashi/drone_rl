import sys, json, zlib, struct
import numpy as np, mujoco
from new_try.nidar_sim.old_code.fly_x2_ego import build_model_ego, Drone6
from fly_x2_keyboard import DEFAULT_ARENA, DEFAULT_LAYOUT

def save_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n"
        + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))

def add_markers(scn, d, bid):
    R, c = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    for axis, rgba in ((0, [1, 0, 0, 1]), (1, [0, 1, 0, 1])):   # red = body +x, green = body +y
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.03, 0, 0]), c + R[:, axis] * 0.45,
                            np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

m = build_model_ego(sys.argv[1], DEFAULT_ARENA)
layout = json.load(open(DEFAULT_LAYOUT, encoding="utf-8"))
dr = Drone6(m, layout)
d = mujoco.MjData(m)
dr.reset(d)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = d.xpos[dr.bid]
cam.distance, cam.azimuth, cam.elevation = 1.5, 90, -89
r.update_scene(d, cam); add_markers(r.scene, d, dr.bid)
save_png("topdown.png", r.render())

r.update_scene(d, camera="ego"); add_markers(r.scene, d, dr.bid)
save_png("ego.png", r.render())
print("saved topdown.png and ego.png")