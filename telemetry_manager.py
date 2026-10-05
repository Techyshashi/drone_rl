"""
NETRA Command Center - live telemetry handling (no Qt, so it can be unit-tested).

run_policy_gui.py prints:
    TEL {...}   every policy step: pose, ranges, min_range, reflex alpha, fused action 'a', policy action 'ap'
    EP  {...}   once per finished episode: ep, goal, result, time, path, left, reflex
    DONE {}     when the run ends
Everything else is ordinary log text. This module splits the byte stream into those pieces and turns the
TEL samples of the episode in progress into the extra measurements the EP line does not carry
(minimum clearance, action statistics).
"""
import json

from experiment_database import utc_now


class StreamParser:
    """Feed arbitrary text chunks; get complete lines back as (kind, payload)."""

    def __init__(self):
        self._buf = ""

    def feed(self, text):
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        return [self._classify(ln.rstrip("\r")) for ln in lines if ln.strip()]

    def flush(self):
        out = [self._classify(self._buf.strip())] if self._buf.strip() else []
        self._buf = ""
        return out

    @staticmethod
    def _classify(ln):
        try:
            if ln.startswith("TEL "):
                return ("TEL", json.loads(ln[4:]))
            if ln.startswith("EP "):
                return ("EP", json.loads(ln[3:]))
            if ln.startswith("DONE"):
                return ("DONE", {})
        except ValueError:
            pass
        return ("LOG", ln)


class EpisodeAccumulator:
    """Statistics over the TEL samples of ONE episode. Values stay None until a sample provides them."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.local_ep = None
        self.goal = None
        self.started_at = None
        self.samples = 0
        self.min_clearance = None
        self.last_pol = self.last_fused = None
        self._sum_pol = self._sum_fused = None
        self.last_t = self.last_path = self.last_goal_dist = None

    @property
    def active(self):
        return self.samples > 0

    def add(self, tel):
        if self.local_ep is not None and tel.get("ep") != self.local_ep:
            self.reset()                                   # a new episode started without an EP line for the old one
        if self.local_ep is None:
            self.local_ep, self.started_at = tel.get("ep"), utc_now()
        self.goal = tel.get("goal", self.goal)
        self.samples += 1
        mr = tel.get("min_range")
        if mr is not None:
            self.min_clearance = mr if self.min_clearance is None else min(self.min_clearance, mr)
        self.last_t, self.last_path, self.last_goal_dist = tel.get("t"), tel.get("path"), tel.get("goal_dist")
        pol, fused = tel.get("ap"), tel.get("a")
        if pol is not None:
            self.last_pol = list(pol)
            self._sum_pol = list(pol) if self._sum_pol is None else [s + v for s, v in zip(self._sum_pol, pol)]
        if fused is not None:
            self.last_fused = list(fused)
            self._sum_fused = list(fused) if self._sum_fused is None else [s + v for s, v in zip(self._sum_fused, fused)]

    def _action_json(self, last, total):
        if last is None:
            return None
        n = self.samples
        return json.dumps({"order": ["forward", "lateral", "yaw"], "last": last,
                           "mean": [round(s / n, 4) for s in total], "samples": n})

    def policy_actions_json(self):
        return self._action_json(self.last_pol, self._sum_pol)

    def fused_actions_json(self):
        return self._action_json(self.last_fused, self._sum_fused)
