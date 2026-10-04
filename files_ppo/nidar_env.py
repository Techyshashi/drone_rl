"""
NIDAR maze environment for the Skydio X2 (Gymnasium). Headless: no rendering, so it runs on Colab as is.

Architecture mapping
  1. MuJoCo environment ........ model from fly_x2_keyboard.build_model (X2 + arena)
  2. Observation (22-D) ........ 4 drone state + 3 goal/route hint + 3 previous action + 12 directional ranges
  3. PPO policy ................ outputs a = [forward, left, yaw]  in [-1, 1]
  4. Reflex + fusion ........... reflex.ReflexLayer (switch off with use_reflex=False)
  5. Drone control ............. Drone.control (velocity controller + motor mixer), unchanged
  6. Episode evaluation ........ info dict: success / collision / path_len / sim_time / goal_dist / reflex_rate

Conventions: the drone's front (gimbal / camera) is body -x, so "forward" = -x_body and "left" = -y_body.
Ranges come from mujoco.mj_ray in the horizontal plane (they can later be replaced by columns of the depth image).
The goal hint is the direction to the next cell centre on the shortest path through the maze (BFS over the wall layout).
"""
import json
import math
import os
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from fly_x2_keyboard import Drone, build_model, quat_to_rpy, TAKEOFF_Z
from reflex import ReflexLayer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_X2 = os.path.join(HERE, "skydio_x2", "x2.xml")
DEFAULT_ARENA = os.path.join(HERE, "nidar_arena.xml")
DEFAULT_LAYOUT = os.path.join(HERE, "nidar_arena_layout.json")

RAY_ANGLES = np.deg2rad(np.arange(-150, 181, 30))       # 12 rays; 0 = nose, + = left
N_RAYS = len(RAY_ANGLES)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridMap:
    """14x14 cell graph built from the wall rectangles; BFS gives the shortest route to a goal cell."""

    def __init__(self, layout):
        self.P, self.T, self.N = layout["cell_pitch_m"], layout["wall_thickness_m"], layout["grid_n"]
        self.walls = np.array(layout["walls_xyxy"], dtype=float)
        N = self.N
        self.open = {}
        for c in range(N):
            for r in range(N):
                for dc, dr in DIRS:
                    nc, nr = c + dc, r + dr
                    if not (0 <= nc < N and 0 <= nr < N):
                        self.open[(c, r, dc, dr)] = False
                        continue
                    if dc != 0:
                        x, y = self.P * (c + 1 if dc == 1 else c) + self.T / 2, self.center(c, r)[1]
                    else:
                        x, y = self.center(c, r)[0], self.P * (r + 1 if dr == 1 else r) + self.T / 2
                    self.open[(c, r, dc, dr)] = not self._wall_at(x, y)

    def _wall_at(self, x, y):
        w = self.walls
        return bool(np.any((w[:, 0] <= x) & (x <= w[:, 2]) & (w[:, 1] <= y) & (y <= w[:, 3])))

    def center(self, c, r):
        h = (self.P - self.T) / 2
        return np.array([self.T + self.P * c + h, self.T + self.P * r + h])

    def cell_of(self, p):
        c = int(np.clip(math.floor((p[0] - self.T) / self.P), 0, self.N - 1))
        r = int(np.clip(math.floor((p[1] - self.T) / self.P), 0, self.N - 1))
        return c, r

    def bfs(self, goal):
        N = self.N
        dist = np.full((N, N), np.inf)
        dist[goal] = 0
        q = deque([goal])
        while q:
            c, r = q.popleft()
            for dc, dr in DIRS:
                if self.open[(c, r, dc, dr)] and not np.isfinite(dist[c + dc, r + dr]):
                    dist[c + dc, r + dr] = dist[c, r] + 1
                    q.append((c + dc, r + dr))
        nxt = np.full((N, N, 2), -1, dtype=int)
        for c in range(N):
            for r in range(N):
                if np.isfinite(dist[c, r]) and dist[c, r] > 0:
                    for dc, dr in DIRS:
                        if self.open[(c, r, dc, dr)] and dist[c + dc, r + dr] == dist[c, r] - 1:
                            nxt[c, r] = (c + dc, r + dr)
                            break
        return dist, nxt


class NidarEnv(gym.Env):
    metadata = {"render_modes": []}
    V_MAX, V_LAT, YAW_MAX = 0.8, 0.5, 1.0          # m/s forward, m/s sideways, rad/s yaw at |action| = 1
    R_MAX = 5.0                                    # ray length (m)
    POLICY_DT = 0.1                                # 10 Hz policy

    def __init__(self, x2_xml=DEFAULT_X2, arena_xml=DEFAULT_ARENA, layout_json=DEFAULT_LAYOUT,
                 use_reflex=True, spawn="mixed", max_start_dist=None, heading_noise=math.pi,
                 ep_seconds=180.0, goal_radius=0.5, range_noise=0.0, fixed_goal=None):
        super().__init__()
        self.layout = json.load(open(layout_json, encoding="utf-8"))
        self.model = build_model(x2_xml, arena_xml)
        self.data = mujoco.MjData(self.model)
        self.drone = Drone(self.model, self.layout)
        self.grid = GridMap(self.layout)
        self.substeps = max(1, int(round(self.POLICY_DT / self.model.opt.timestep)))
        self.reflex = ReflexLayer(RAY_ANGLES, dt=self.POLICY_DT, r_max=self.R_MAX)
        self.use_reflex = use_reflex
        self.spawn, self.max_start_dist, self.heading_noise = spawn, max_start_dist, heading_noise
        self.ep_seconds, self.goal_radius, self.range_noise = ep_seconds, goal_radius, range_noise
        self.fixed_goal = fixed_goal
        self.pad = np.array(self.layout["launch_pad"]["xy"], dtype=float)
        gate_xy = np.array(self.layout["gates"]["entry_exit"]["xy"], dtype=float)
        self.gate_cell = self.grid.cell_of(gate_xy)
        self.gate_xy = self.grid.center(*self.gate_cell)
        self.goals = [np.array(s["xy"], dtype=float) for s in self.layout["survivors"]]
        self.goal_cells = [self.grid.cell_of(g) for g in self.goals]
        self.maps = [self.grid.bfs(gc) for gc in self.goal_cells]
        self.viewer_sync = None                     # optional callable, invoked after every physics step
        self._geomid = np.zeros(1, dtype=np.int32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(4 + 3 + 3 + N_RAYS,), dtype=np.float32)

    # ---------------------------------------------------------------- curriculum
    def set_curriculum(self, max_start_dist, heading_noise, spawn):
        self.max_start_dist, self.heading_noise, self.spawn = max_start_dist, heading_noise, spawn

    # ---------------------------------------------------------------- geometry helpers
    @staticmethod
    def _frame(yaw):
        """world-xy unit vectors of the drone's nose and left (nose = body -x)."""
        return np.array([-math.cos(yaw), -math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])

    def _geo(self, p):
        """(remaining route length in m, world xy of the next waypoint)"""
        c, r = self.grid.cell_of(p)
        if p[1] < self.grid.T and (c, r) == self.gate_cell and (c, r) != self.gcell:
            # still outside the arena (launch pad): aim at the gate first, so the drone enters straight
            return float(np.linalg.norm(self.gate_xy - p[:2]) + self.dist[c, r] * self.grid.P), self.gate_xy
        if (c, r) == self.gcell:
            return float(np.linalg.norm(self.goal_xy - p[:2])), self.goal_xy
        n = self.nxt[c, r]
        if n[0] < 0:                                   # not on any route (should not happen)
            return float(np.linalg.norm(self.goal_xy - p[:2])), self.goal_xy
        wp = self.grid.center(n[0], n[1])
        return float(np.linalg.norm(wp - p[:2]) + self.dist[n[0], n[1]] * self.grid.P), wp

    def _cast(self):
        p, q = self.drone.pose(self.data)
        yaw = quat_to_rpy(q)[2]
        nose, left = self._frame(yaw)
        out = np.full(N_RAYS, self.R_MAX)
        for i, th in enumerate(RAY_ANGLES):
            dxy = math.cos(th) * nose + math.sin(th) * left
            vec = np.array([dxy[0], dxy[1], 0.0])
            dist = mujoco.mj_ray(self.model, self.data, p, vec, None, 1, self.drone.bid, self._geomid)
            if dist >= 0.0:
                out[i] = min(dist, self.R_MAX)
        if self.range_noise > 0:
            out = np.clip(out + self.np_random.normal(0, self.range_noise, N_RAYS), 0.0, self.R_MAX)
        return out

    def _obs(self):
        p, q = self.drone.pose(self.data)
        yaw = quat_to_rpy(q)[2]
        nose, left = self._frame(yaw)
        v = self.data.qvel[self.drone.dadr:self.drone.dadr + 2]
        wz = self.data.qvel[self.drone.dadr + 5]
        rem, wp = self._geo(p)
        dv = wp - p[:2]
        n = float(np.linalg.norm(dv)) + 1e-6
        o = np.concatenate([
            [v @ nose / self.V_MAX, v @ left / self.V_MAX, wz / self.YAW_MAX,
             np.clip((p[2] - TAKEOFF_Z) / 0.5, -1, 1)],
            [dv @ nose / n, dv @ left / n, min(rem, 15.0) / 15.0],
            self.prev_a,
            self._ranges / self.R_MAX])
        return o.astype(np.float32)

    def _sim(self, n):
        for _ in range(n):
            self.drone.control(self.data)
            mujoco.mj_step(self.model, self.data)
            if self.viewer_sync is not None:
                self.viewer_sync()
            if self.drone.touching(self.data):
                return True
        return False

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        gi = self.fixed_goal if self.fixed_goal is not None else int(rng.integers(len(self.goals)))
        self.goal_i, self.goal_xy, self.gcell = gi, self.goals[gi], self.goal_cells[gi]
        self.dist, self.nxt = self.maps[gi]
        mode = self.spawn
        if mode == "mixed":
            mode = "pad" if rng.random() < 0.25 else "random"
        dr, d = self.drone, self.data
        dr.reset(d)
        if mode == "pad":
            x, y, z = self.pad[0] + rng.uniform(-.08, .08), self.pad[1] + rng.uniform(-.08, .08), 0.25
            nose_h, settle = math.pi / 2 + rng.uniform(-.15, .15), 250
        else:
            mx = np.inf if self.max_start_dist is None else self.max_start_dist
            cells = np.argwhere(np.isfinite(self.dist) & (self.dist >= 1) & (self.dist <= mx))
            c, r = cells[int(rng.integers(len(cells)))]
            ctr = self.grid.center(c, r)
            x, y = ctr[0] + rng.uniform(-.12, .12), ctr[1] + rng.uniform(-.12, .12)
            _, wp = self._geo(np.array([x, y]))
            nose_h = math.atan2(wp[1] - y, wp[0] - x) + rng.uniform(-self.heading_noise, self.heading_noise)
            z, settle = TAKEOFF_Z, 40
        yaw = nose_h + math.pi                              # nose = body -x
        d.qpos[dr.qadr:dr.qadr + 3] = [x, y, z]
        d.qpos[dr.qadr + 3:dr.qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        dr.yaw_target = yaw
        dr.cmd = dict(vx=0.0, vy=0.0, yaw_rate=0.0, z=TAKEOFF_Z)
        mujoco.mj_forward(self.model, d)
        self._sim(settle)
        self.reflex.reset()
        self.prev_a = np.zeros(3)
        self.prev_pol = np.zeros(3)
        self.steps = 0
        self.path_len = 0.0
        self.sum_alpha = 0.0
        self.sum_delta = 0.0
        self._ranges = self._cast()
        self._rem, _ = self._geo(dr.pose(d)[0])
        return self._obs(), {"goal": gi, "spawn": mode}

    def step(self, action):
        a_pol = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.use_reflex:
            a, ri = self.reflex(self._ranges, a_pol, self.data.qvel[self.drone.dadr + 5])
        else:
            a, ri = a_pol, dict(alpha=0.0, delta=0.0)
        c = self.drone.cmd
        c["vx"] = -a[0] * self.V_MAX                        # controller's +x is the REAR of the drone
        c["vy"] = -a[1] * self.V_LAT                      # controller +y = image-right
        c["yaw_rate"] = a[2] * self.YAW_MAX
        p0 = self.drone.pose(self.data)[0]
        collided = self._sim(self.substeps)
        p = self.drone.pose(self.data)[0]
        self.path_len += float(np.linalg.norm(p[:2] - p0[:2]))
        self._ranges = self._cast()
        rem, _ = self._geo(p)
        self.steps += 1
        self.sum_alpha += ri["alpha"]
        self.sum_delta += ri["delta"]

        success = (not collided) and float(np.linalg.norm(p[:2] - self.goal_xy)) < self.goal_radius
        rmin = float(self._ranges.min())
        reward = (self._rem - rem) - 0.02 - 0.02 * ri["alpha"] \
            - 0.02 * float(np.sum((a_pol - self.prev_pol) ** 2)) \
            - 0.2 * float(np.clip((0.45 - rmin) / 0.15, 0.0, 1.0))
        if collided:
            reward -= 20.0
        elif success:
            reward += 50.0
        self._rem = rem
        self.prev_a, self.prev_pol = a.copy(), a_pol.copy()
        t = self.steps * self.POLICY_DT
        terminated = bool(collided or success)
        truncated = bool((not terminated) and t >= self.ep_seconds)
        info = dict(success=bool(success), collision=bool(collided), timeout=truncated,
                    path_len=self.path_len, sim_time=t, goal_dist=rem, min_range=rmin,
                    reflex_rate=self.sum_alpha / self.steps, reflex_delta=self.sum_delta / self.steps,
                    goal=self.goal_i)
        return self._obs(), float(reward), terminated, truncated, info
