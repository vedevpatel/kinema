"""Check 12 - deployment readiness.

The "low loss, failed robot" check. Training can succeed while the
dataset is undeployable: episodes shorter than the policy's action
chunk, stats that cannot produce the normalization the policy expects,
or a state/action layout the policy does not assume. Grounded in the
openpi #906 failure mode (normalization mismatch between training and
inference is the community's top diagnosis).
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY


def _load_policies() -> dict:
    """Known-policy declarations from src/kinema/policies.json.

    One entry per policy: chunk_size (action horizon), quantile_norm (the
    normalizer consumes q01/q99), needs_images / needs_language (the policy
    is vision- / language-conditioned), typical_resolution ([w, h] the
    policy is usually trained at). Add a policy by editing the JSON - no
    code change."""
    with resources.files("kinema").joinpath("policies.json").open() as f:
        return json.load(f)


POLICY_DEFAULTS = _load_policies()


def _camera_features(ctx) -> dict[str, dict]:
    """Declared image/video features (embedded or MP4 chunks)."""
    return {k: f for k, f in (ctx.info.get("features") or {}).items()
            if isinstance(f, dict) and f.get("dtype") in ("image", "video")}


def _spatial_size(feat: dict) -> tuple[int, int] | None:
    """(width, height) from a [C, H, W]-style declared shape."""
    shape = feat.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        h, w = int(shape[-2]), int(shape[-1])
        if h > 0 and w > 0:
            return w, h
    return None


def _has_task_strings(ctx) -> bool:
    return any(meta.tasks for meta in ctx.episodes if meta is not None and meta.tasks)


class DeploymentReadinessCheck(Check):
    id = "deployment-readiness"
    title = "Deployment readiness (would this dataset serve a policy?)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        policy = POLICY_DEFAULTS.get(cfg.policy.lower()) if cfg.policy else None
        if cfg.policy and policy is None:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    f"unknown policy {cfg.policy!r}; running policy-agnostic checks "
                    f"(known: {', '.join(sorted(POLICY_DEFAULTS))})",
                )
            )

        # 1. stats coverage: training consumes observation.state + action
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.INFO,
                        f"no {feat} column found (checked the usual aliases) - "
                        "state/action checks could not run",
                    )
                )
                continue
            if feat not in ctx.stats:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{feat}: no normalization stats (meta/stats.json or per-episode "
                        "stats) - the policy cannot normalize inputs/outputs at inference",
                    )
                )

        # 2. quantile normalization support (openpi normalizes with q01/q99)
        if policy and policy["quantile_norm"]:
            for feat in (STATE_KEY, ACTION_KEY):
                entry = ctx.stats.get(feat) or {}
                if entry and not ("q01" in entry and "q99" in entry):
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"{cfg.policy} normalizes with q01/q99 quantiles but stats.json "
                            f"has none for {feat} - recompute stats with quantiles or the "
                            "openpi normalizer will estimate them itself",
                        )
                    )

        # 3. episode length vs action chunk
        if policy:
            chunk = policy["chunk_size"]
            short = [e.index for e in ctx.episodes if 0 < e.length < chunk]
            if short:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{len(short)}/{ctx.total_episodes} episodes are shorter than "
                        f"{cfg.policy}'s action chunk ({chunk} frames): {short[:10]}"
                        f"{'...' if len(short) > 10 else ''} - they train on padded or "
                        "truncated chunks",
                        details={"episodes": short, "chunk_size": chunk},
                    )
                )

        # 4. dims: info.json feature shapes vs the actual arrays
        feats = ctx.info.get("features") or {}
        for key, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            declared = (feats.get(ctx.feature_keys.get(key, key)) or {}).get("shape")
            # compare element counts: [8], [1, 8] and [8, 1] all describe 8 dims
            if declared and int(np.prod(declared)) != arr.shape[1]:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{key}: info.json declares shape {declared} but the data has "
                        f"{arr.shape[1]} dims",
                    )
                )

        # 5. what the policy conditions on (declared in policies.json)
        if policy:
            cams = _camera_features(ctx)
            if policy.get("needs_images") and not cams:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{cfg.policy} is trained with camera input but this dataset declares "
                        "no image or video features - a state-only dataset cannot serve it as-is",
                    )
                )
            typical = policy.get("typical_resolution")
            if typical and cams:
                tw, th = int(typical[0]), int(typical[1])
                small = []
                for cam_key, cam_feat in cams.items():
                    size = _spatial_size(cam_feat)
                    if size and (size[0] < tw or size[1] < th):
                        small.append(f"{cam_key} {size[0]}x{size[1]}")
                if small:
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.INFO,
                            f"{cfg.policy} is usually trained at {tw}x{th}; "
                            f"{', '.join(sorted(small))} is smaller - usable (the loader "
                            "resizes), but the policy sees less detail than it was built for",
                        )
                    )
            if policy.get("needs_language") and not _has_task_strings(ctx):
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{cfg.policy} conditions on language but no episode declares a task "
                        "string - training here teaches a task-agnostic policy",
                    )
                )

        if not res.findings:
            res.summary = (
                f"ready for {cfg.policy}" if cfg.policy else "no deployment blockers found"
            )
        else:
            n_fail = sum(1 for f in res.findings if f.severity == Severity.FAIL)
            n_warn = sum(1 for f in res.findings if f.severity == Severity.WARN)
            res.summary = f"{n_fail} blocker(s), {n_warn} warning(s)"
            if cfg.policy:
                res.summary += f" for {cfg.policy}"
        return res
