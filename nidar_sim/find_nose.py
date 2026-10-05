import sys, numpy as np, mujoco
from new_try.nidar_sim.old_code.fly_x2_ego import build_model_ego
from fly_x2_keyboard import DEFAULT_ARENA

m = build_model_ego(sys.argv[1], DEFAULT_ARENA)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "x2")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
R, c = d.xmat[bid].reshape(3, 3), d.xpos[bid]

def under(b):
    while b > 0:
        if b == bid: return True
        b = m.body_parentid[b]
    return False

for g in range(m.ngeom):
    if not under(m.geom_bodyid[g]): continue
    mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else ""
    p = R.T @ (d.geom_xpos[g] - c)
    print(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '-':<20} {mesh:<24} body-frame xyz = {np.round(p, 3)}")