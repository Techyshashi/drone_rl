"""
Reflex layer (architecture box 4): fruit-fly-inspired heuristic + action fusion with the PPO policy.

Actions are normalised to [-1, 1]:  a = [forward, left, yaw_left]   (forward = the drone's NOSE / camera side).
Inputs are the directional ranges (metres) from the observation builder. Ray i points at angle angles[i]
from the nose (0 = straight ahead, + = left).

Three fly-inspired behaviours:
  1. LOOMING AVOIDANCE : an obstacle that is close OR expanding fast (closing speed / range ~ 1/time-to-contact)
                         triggers repulsion away from it and a turn toward the more open side.
  2. CORRIDOR CENTERING: always-on lateral nudge that balances left/right range (like flies/bees balancing
                         lateral optic flow).
  3. FORWARD BRAKING   : forward speed is capped by the clearance straight ahead.

Fusion:  the policy action has its motion INTO obstacles cancelled (in proportion to how urgent each ray is),
the repulsion/turn/centering corrections are added, and the forward cap is applied.  With nothing nearby the
output equals the policy action (plus the small centering nudge).

Later extension (planned in the diagram): replace `ReflexLayer.__call__` with a decoder reading flybrain spiking
activity. Keep the same signature: (ranges, a_policy) -> (a_fused, info).
"""
import numpy as np


class ReflexLayer:
    def __init__(self, angles, dt=0.1, r_max=5.0,
                 front_react=1.0, front_crit=0.50,      # start reacting / fully urgent, straight ahead (m)
                 side_react=0.48, side_crit=0.34,       # same, at +/-90 deg (X2 half-width is ~0.31 m)
                 slow_range=1.4, stop_range=0.55,       # forward cap: full speed beyond slow_range, zero at stop_range
                 k_rep=0.4, k_yaw=1.0, k_center=0.25, loom_ref=1.0, loom_yaw_max=0.3):
        self.th = np.asarray(angles, dtype=float)
        self.dt, self.r_max = dt, r_max
        s = np.abs(np.sin(self.th))
        self.r_react = front_react + (side_react - front_react) * s
        self.r_crit = front_crit + (side_crit - front_crit) * s
        self.w = np.where(np.abs(self.th) <= np.deg2rad(100), 1.0, 0.0)      # rear rays are ignored
        self.e = np.stack([np.cos(self.th), np.sin(self.th)], axis=1)         # (forward, left) unit vectors
        self.front = np.abs(self.th) <= np.deg2rad(30) + 1e-6
        self.left = (self.th > 0) & (self.th <= np.deg2rad(100))
        self.right = (self.th < 0) & (self.th >= -np.deg2rad(100))
        self.iL = int(np.argmin(np.abs(self.th - np.pi / 2)))
        self.iR = int(np.argmin(np.abs(self.th + np.pi / 2)))
        self.slow, self.stop = slow_range, stop_range
        self.k_rep, self.k_yaw, self.k_center, self.loom_ref = k_rep, k_yaw, k_center, loom_ref
        self.loom_yaw_max = loom_yaw_max
        self.reset()

    def reset(self):
        self.prev = None
        self._turn = 1.0

    def __call__(self, ranges, a_pol, yaw_rate=0.0):
        r = np.minimum(np.asarray(ranges, dtype=float), self.r_max)
        a_pol = np.asarray(a_pol, dtype=float)

        # --- looming: closing speed / range ~ 1 / time-to-contact
        closing = np.zeros_like(r) if self.prev is None else np.maximum(0.0, (self.prev - r) / self.dt)
        self.prev = r.copy()
        prox = np.clip((self.r_react - r) / (self.r_react - self.r_crit), 0.0, 1.0)
        loom = np.clip(closing / np.maximum(r, 0.1) / self.loom_ref, 0.0, 1.0) * (r < 2.0)
        if abs(yaw_rate) > self.loom_yaw_max:
            loom[:] = 0.0
        u = np.maximum(prox, 0.6 * loom) * self.w                  # urgency per ray, 0..1

        a_xy = a_pol[:2].copy()
        # 1a) cancel the part of the commanded motion that points into an obstacle
        for i in np.nonzero(u > 0.0)[0]:
            into = a_xy @ self.e[i]
            if into > 0.0:
                a_xy -= u[i] * into * self.e[i]
        # 1b) repulsion
        su = float(u.sum())
        if su > 0.0:
            a_xy += self.k_rep * (-(u[:, None] * self.e).sum(axis=0) / max(1.0, su))

        # 1c) turn toward the more open side when something is ahead
        yaw = a_pol[2]
        g_f = float(u[self.front].max())
        if g_f > 0.0:
            mL = float(np.mean(np.minimum(r[self.left], 3.0)))
            mR = float(np.mean(np.minimum(r[self.right], 3.0)))
            asym = (mL - mR) / (mL + mR + 1e-6)
            if abs(asym) > 0.05:
                self._turn = float(np.sign(asym))                  # remember, so a symmetric dead end doesn't dither
                direction = float(np.clip(2.0 * asym, -1.0, 1.0))
            else:
                direction = self._turn
            yaw += self.k_yaw * g_f ** 2 * direction

        # 2) corridor centering (always on, gentle)
        rL, rR = r[self.iL], r[self.iR]
        if rL < 1.2 and rR < 1.2:
            a_xy[1] += self.k_center * (rL - rR) / (rL + rR)

        # 3) forward braking
        cap = float(np.clip((r[self.front].min() - self.stop) / (self.slow - self.stop), 0.0, 1.0))
        a_xy[0] = max(-0.5, min(a_xy[0], cap))

        a = np.clip(np.array([a_xy[0], a_xy[1], yaw]), -1.0, 1.0)
        return a, dict(alpha=float(u.max()), delta=float(np.linalg.norm(a - a_pol)))
