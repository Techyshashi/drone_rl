"""
Watch a trained policy (+ reflex layer) fly the NIDAR maze in the MuJoCo viewer (Windows or Linux with a display).

  python run_policy.py --run runs\\nidar --goal 3                 # survivor 3, with reflex
  python run_policy.py --run runs\\nidar --goal 3 --no-reflex     # policy alone
  python run_policy.py --run runs\\nidar --x2 "C:\\...\\skydio_x2\\x2.xml"

--run is the folder that contains ppo_nidar.zip and vecnorm.pkl (both copied back from Colab).
The drone starts on the launch pad each episode. Close the viewer window to stop.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nidar_env import NidarEnv, DEFAULT_X2, DEFAULT_ARENA, DEFAULT_LAYOUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--goal", type=int, default=0, help="survivor 1-6 (0 = random each episode)")
    ap.add_argument("--no-reflex", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args()

    env = NidarEnv(x2_xml=a.x2, arena_xml=a.arena, layout_json=a.layout, use_reflex=not a.no_reflex,
                   spawn="pad", fixed_goal=(a.goal - 1) if a.goal else None, ep_seconds=240.0)
    venv = VecNormalize.load(os.path.join(a.run, "vecnorm.pkl"), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(os.path.join(a.run, "ppo_nidar.zip"), device="cpu")

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = env.drone.bid
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.0, 90, -60
        dt, nxt = env.model.opt.timestep, [time.time()]

        def sync():                                   # called after every physics step: draw and run in real time
            v.sync()
            nxt[0] += dt
            time.sleep(max(0.0, nxt[0] - time.time()))

        env.viewer_sync = sync
        obs, ep = venv.reset(), 0
        while v.is_running() and ep < a.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(action)
            if done[0]:
                i = infos[0]
                res = "SUCCESS" if i["success"] else ("COLLISION" if i["collision"] else "timeout")
                print(f"episode {ep + 1}: survivor_{i['goal'] + 1} {res} | {i['sim_time']:.0f} s | path {i['path_len']:.1f} m "
                      f"| left {i['goal_dist']:.1f} m | reflex active {i['reflex_rate']:.0%}")
                ep += 1
                nxt[0] = time.time()


if __name__ == "__main__":
    main()
