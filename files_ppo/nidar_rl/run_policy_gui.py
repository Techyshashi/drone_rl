"""
Telemetry runner used by nidar_gui.py. Same as run_policy.py (MuJoCo viewer + trained policy), plus
machine-readable lines the GUI parses:
    TEL {...}   every policy step (10 Hz): pose, speed, ranges, reflex activity, actions
    EP  {...}   at the end of every episode
You can also run it by hand (put it next to nidar_env.py):
    python run_policy_gui.py --run runs\\nidar --goal 3                      # reflex ON  (policy trained with reflex)
    python run_policy_gui.py --run runs\\no_reflex --no-reflex --goal 3      # reflex OFF (policy trained without it)
    add --headless for no window (fast), --speed 2 for 2x real time.
"""
import argparse
import json
import math
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nidar_env import NidarEnv, DEFAULT_X2, DEFAULT_ARENA, DEFAULT_LAYOUT
from fly_x2_keyboard import quat_to_rpy


def emit(tag, d):
    print(tag + " " + json.dumps(d, separators=(",", ":")), flush=True)


class Pacer:
    """Draw every physics step and keep the simulation at (speed x) real time."""

    def __init__(self, viewer, dt, speed):
        self.v, self.dt, self.speed, self.nxt = viewer, dt, max(0.05, speed), time.time()

    def sync(self):
        self.v.sync()
        self.nxt += self.dt / self.speed
        time.sleep(max(0.0, self.nxt - time.time()))

    def reset(self):
        self.nxt = time.time()


def play(env, venv, model, a, alive, on_episode=None):
    obs, ep, last_sum, k = venv.reset(), 0, 0.0, 0
    while alive() and ep < a.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        i = infos[0]
        if done[0]:
            res = "SUCCESS" if i["success"] else ("COLLISION" if i["collision"] else "TIMEOUT")
            emit("EP", dict(ep=ep + 1, goal=i["goal"] + 1, result=res, time=round(i["sim_time"], 1),
                            path=round(i["path_len"], 1), left=round(i["goal_dist"], 1),
                            reflex=round(i["reflex_rate"], 3)))
            print(f"episode {ep + 1}: survivor_{i['goal'] + 1} {res} | {i['sim_time']:.0f} s | "
                  f"path {i['path_len']:.1f} m | left {i['goal_dist']:.1f} m | reflex active {i['reflex_rate']:.0%}",
                  flush=True)
            ep, last_sum = ep + 1, 0.0
            if on_episode:
                on_episode()
            continue
        k += 1
        alpha = float(env.sum_alpha - last_sum)            # reflex urgency in THIS step (0..1)
        last_sum = float(env.sum_alpha)
        if k % a.tel_every == 0:
            p, q = env.drone.pose(env.data)
            nose, _ = env._frame(quat_to_rpy(q)[2])
            v = env.data.qvel[env.drone.dadr:env.drone.dadr + 3]
            emit("TEL", dict(
                ep=ep + 1, goal=i["goal"] + 1, t=round(i["sim_time"], 2), path=round(i["path_len"], 2),
                x=round(float(p[0]), 3), y=round(float(p[1]), 3), z=round(float(p[2]), 3),
                speed=round(float(math.hypot(v[0], v[1])), 3), vz=round(float(v[2]), 3),
                heading=round(math.degrees(math.atan2(nose[1], nose[0])), 1),
                goal_dist=round(i["goal_dist"], 2), min_range=round(i["min_range"], 3),
                ranges=[round(float(x), 2) for x in env._ranges],
                alpha=round(alpha, 3), a=[round(float(x), 3) for x in env.prev_a],
                ap=[round(float(x), 3) for x in env.prev_pol]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with ppo_nidar.zip and vecnorm.pkl")
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--goal", type=int, default=0, help="survivor 1-6 (0 = random each episode)")
    ap.add_argument("--no-reflex", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--headless", action="store_true", help="no viewer, run as fast as possible")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time (viewer only)")
    ap.add_argument("--tel-every", type=int, default=1, help="emit telemetry every N policy steps")
    ap.add_argument("--cam-dist", type=float, default=4.0)
    ap.add_argument("--cam-elev", type=float, default=-60.0)
    a = ap.parse_args()

    env = NidarEnv(x2_xml=a.x2, arena_xml=a.arena, layout_json=a.layout, use_reflex=not a.no_reflex,
                   spawn="pad", fixed_goal=(a.goal - 1) if a.goal else None, ep_seconds=240.0)
    venv = VecNormalize.load(os.path.join(a.run, "vecnorm.pkl"), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(os.path.join(a.run, "ppo_nidar.zip"), device="cpu")
    print(f"loaded {a.run} | reflex {'OFF' if a.no_reflex else 'ON'} | goal "
          f"{a.goal or 'random'} | {'headless' if a.headless else 'viewer'}", flush=True)

    if a.headless:
        play(env, venv, model, a, lambda: True)
    else:
        import mujoco
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = env.drone.bid
            v.cam.distance, v.cam.azimuth, v.cam.elevation = a.cam_dist, 90, a.cam_elev
            pacer = Pacer(v, env.model.opt.timestep, a.speed)
            env.viewer_sync = pacer.sync
            play(env, venv, model, a, v.is_running, pacer.reset)
    emit("DONE", {})


if __name__ == "__main__":
    main()
