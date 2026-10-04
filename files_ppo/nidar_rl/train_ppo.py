"""
Train the goal-directed PPO policy (architecture box 3) with the reflex layer (box 4) in the loop, then evaluate (box 6).
Colab / T4 edition: GPU for the network, resumable after a disconnect, checkpoints saved to Google Drive.

Colab cells
  from google.colab import drive; drive.mount('/content/drive')       # checkpoints survive a disconnect
  !pip install -q mujoco gymnasium stable-baselines3
  %cd /content/nidar_rl                                                # fly_x2_keyboard.py, nidar_env.py, reflex.py,
                                                                       # train_ppo.py, nidar_arena.xml,
                                                                       # nidar_arena_layout.json, skydio_x2/
  !python -u train_ppo.py --envs 8 --device cuda                      # same hyper-parameters as the CPU version
  !python -u train_ppo.py --envs 8 --device cuda --gpu-preset         # bigger batch + 256x256 net (uses the T4 better)

Session died?  Run the SAME command again with --resume added; it continues from the last checkpoint.

Notes
  * MuJoCo stepping and ray casts always run on the CPU cores. The GPU only runs the policy network, so --envs
    (= CPU cores) is what sets the speed. Check `fps` in the log.
  * Outputs (in --out): ppo_nidar.zip (policy) and vecnorm.pkl (observation statistics). BOTH are needed by run_policy.py.
  * --out defaults to /content/drive/MyDrive/nidar_runs/run1 when Drive is mounted, otherwise <here>/runs/nidar.

Evaluate a saved policy, with and without the reflex (ablation):
  python train_ppo.py --eval-only <RUN_DIR> --episodes 30
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from nidar_env import NidarEnv, DEFAULT_X2, DEFAULT_ARENA, DEFAULT_LAYOUT

# curriculum: start close to the goal and facing roughly the right way, then widen
STAGES = [
    dict(max_start_dist=3, heading_noise=0.4, spawn="random", steps=200_000),
    dict(max_start_dist=8, heading_noise=1.0, spawn="random", steps=400_000),
    dict(max_start_dist=20, heading_noise=math.pi, spawn="mixed", steps=600_000),
    dict(max_start_dist=None, heading_noise=math.pi, spawn="mixed", steps=1_000_000),
]

DRIVE_OUT = "/content/drive/MyDrive/nidar_runs/run1"


def env_kwargs(a, use_reflex):
    return dict(x2_xml=a.x2, arena_xml=a.arena, layout_json=a.layout, use_reflex=use_reflex)


def make_eval_env(a, use_reflex, vecnorm_stats):
    """Pad-spawn evaluation env that uses the training observation statistics."""
    kw = env_kwargs(a, use_reflex)
    kw.update(spawn="pad", ep_seconds=240.0)
    venv = VecNormalize(DummyVecEnv([lambda: NidarEnv(**kw)]), training=False, norm_obs=True, norm_reward=False)
    venv.obs_rms = vecnorm_stats
    return venv


def evaluate(model, venv, n_episodes, label):
    """Episode evaluation (box 6): success, collisions, path length, time, reflex activity, policy latency."""
    obs = venv.reset()
    infos_done, lat = [], []
    while len(infos_done) < n_episodes:
        t0 = time.perf_counter()
        act, _ = model.predict(obs, deterministic=True)
        lat.append(time.perf_counter() - t0)
        obs, _, done, infos = venv.step(act)
        if done[0]:
            infos_done.append(infos[0])
    ok = [i for i in infos_done if i["success"]]
    res = dict(
        success=len(ok) / n_episodes,
        collision=float(np.mean([i["collision"] for i in infos_done])),
        timeout=float(np.mean([i["timeout"] for i in infos_done])),
        path_m=float(np.mean([i["path_len"] for i in ok])) if ok else float("nan"),
        time_s=float(np.mean([i["sim_time"] for i in ok])) if ok else float("nan"),
        goal_dist_m=float(np.mean([i["goal_dist"] for i in infos_done])),
        reflex_active=float(np.mean([i["reflex_rate"] for i in infos_done])),
        policy_ms=1000 * float(np.mean(lat)))
    print(f"[eval {label:14}] n={n_episodes} success {res['success']:.0%} | collision {res['collision']:.0%} | "
          f"timeout {res['timeout']:.0%} | path {res['path_m']:.1f} m | time {res['time_s']:.0f} s | "
          f"final goal dist {res['goal_dist_m']:.1f} m | reflex active {res['reflex_active']:.0%} | "
          f"policy {res['policy_ms']:.2f} ms/step")
    return res


def save_run(model, venv, out):
    """Write policy + normalisation stats (via temp files, so a disconnect can't leave a half-written checkpoint)."""
    tmp_m, tmp_v = os.path.join(out, "_tmp_ppo.zip"), os.path.join(out, "_tmp_vecnorm.pkl")
    model.save(tmp_m)
    venv.save(tmp_v)
    os.replace(tmp_m, os.path.join(out, "ppo_nidar.zip"))
    os.replace(tmp_v, os.path.join(out, "vecnorm.pkl"))


class CheckpointCB(BaseCallback):
    """Saves the latest policy + vecnorm every `every` timesteps, so --resume loses little work."""

    def __init__(self, venv, out, every):
        super().__init__()
        self.venv, self.out, self.every, self.last = venv, out, every, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            save_run(self.model, self.venv, self.out)
        return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = DRIVE_OUT if os.path.isdir("/content/drive/MyDrive") else os.path.join(here, "runs", "nidar")
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2", default=DEFAULT_X2)
    ap.add_argument("--arena", default=DEFAULT_ARENA)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--envs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of steps of every stage")
    ap.add_argument("--no-reflex", action="store_true", help="train without the reflex layer (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=24, help="episodes per evaluation")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="device for the policy network: auto = cuda if available, else cpu")
    ap.add_argument("--gpu-preset", action="store_true",
                    help="n_steps 2048, batch 2048, 256x256 net (~144k params). Changes learning behaviour vs. the baseline.")
    ap.add_argument("--n-steps", type=int, default=None, help="rollout length per env (default 1024; preset 2048)")
    ap.add_argument("--batch-size", type=int, default=None, help="PPO minibatch (default 512; preset 2048)")
    ap.add_argument("--net", type=int, default=None, help="hidden units per layer (default 128; preset 256)")
    ap.add_argument("--save-every", type=int, default=50_000, help="checkpoint every N timesteps")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ppo_nidar.zip + vecnorm.pkl")
    ap.add_argument("--eval-only", metavar="RUN_DIR", help="skip training; evaluate RUN_DIR/ppo_nidar.zip")
    a = ap.parse_args()

    n_steps = a.n_steps or (2048 if a.gpu_preset else 1024)
    batch = a.batch_size or (2048 if a.gpu_preset else 512)
    width = a.net or (256 if a.gpu_preset else 128)

    if a.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but no GPU is visible "
              "(Runtime > Change runtime type > T4 GPU). Using cpu.")
        a.device = "cpu"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"torch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cuda":
        torch.set_num_threads(1)                    # CPU cores belong to the MuJoCo workers

    if a.eval_only:
        stats = VecNormalize.load(os.path.join(a.eval_only, "vecnorm.pkl"),
                                  DummyVecEnv([lambda: NidarEnv(**env_kwargs(a, True))])).obs_rms
        model = PPO.load(os.path.join(a.eval_only, "ppo_nidar.zip"), device=device)
        for use in (True, False):
            evaluate(model, make_eval_env(a, use, stats), a.episodes, "reflex ON" if use else "reflex OFF")
        return

    os.makedirs(a.out, exist_ok=True)
    print(f"output folder: {a.out}")
    cls = SubprocVecEnv if a.envs > 1 else DummyVecEnv
    venv = make_vec_env(NidarEnv, n_envs=a.envs, seed=a.seed, vec_env_cls=cls,
                        env_kwargs=env_kwargs(a, not a.no_reflex),
                        monitor_kwargs=dict(info_keywords=("success", "collision")))

    ck_model, ck_vec = os.path.join(a.out, "ppo_nidar.zip"), os.path.join(a.out, "vecnorm.pkl")
    if a.resume and os.path.exists(ck_model) and os.path.exists(ck_vec):
        venv = VecNormalize.load(ck_vec, venv)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(ck_model, env=venv, device=device)
        print(f"RESUMED from {a.out} at {model.num_timesteps:,} timesteps")
    else:
        if a.resume:
            print("--resume given but no checkpoint found, starting from scratch")
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
        model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=0.005, clip_range=0.2, seed=a.seed, device=device, verbose=1,
                    policy_kwargs=dict(net_arch=dict(pi=[width, width], vf=[width, width])))
    n_par = sum(p.numel() for p in model.policy.parameters())
    print(f"policy parameters: {n_par:,}  (paper: < 170,000) | envs {a.envs} | n_steps {model.n_steps} | "
          f"batch {model.batch_size}")

    bounds = np.cumsum([int(st["steps"] * a.scale) for st in STAGES])      # cumulative end of every stage
    cb = CheckpointCB(venv, a.out, a.save_every)
    for i, st in enumerate(STAGES):
        if model.num_timesteps >= bounds[i]:
            print(f"stage {i + 1}/{len(STAGES)} already done, skipping")
            continue
        venv.env_method("set_curriculum", st["max_start_dist"], st["heading_noise"], st["spawn"])
        todo = int(bounds[i] - model.num_timesteps)
        print(f"\n=== stage {i + 1}/{len(STAGES)}: start <= {st['max_start_dist']} cells from goal, "
              f"heading noise {st['heading_noise']:.2f} rad, spawn {st['spawn']}, {todo:,} steps to go ===")
        cb.last = model.num_timesteps
        model.learn(total_timesteps=todo, reset_num_timesteps=False, callback=cb)
        save_run(model, venv, a.out)
        evaluate(model, make_eval_env(a, not a.no_reflex, venv.obs_rms), max(6, a.episodes // 2), f"stage {i + 1}")

    print("\n=== final evaluation from the launch pad ===")
    for use in (True, False):
        evaluate(model, make_eval_env(a, use, venv.obs_rms), a.episodes, "reflex ON" if use else "reflex OFF")
    print(f"saved: {ck_model}  and  {ck_vec}")


if __name__ == "__main__":
    main()