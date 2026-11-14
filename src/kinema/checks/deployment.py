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
    code change. A top-level "_aliases" object maps alternate spellings
    (pi0.5) to the canonical entry (pi05)."""
    with resources.files("kinema").joinpath("policies.json").open() as f:
        return json.load(f)


_POLICIES = _load_policies()
POLICY_DEFAULTS = {k: v for k, v in _POLICIES.items() if not k.startswith("_")}
POLICY_ALIASES = {k.lower(): v for k, v in _POLICIES.get("_aliases", {}).items()}

# fields a per-call policy_overrides dict may set - the policies.json schema
OVERRIDEABLE_POLICY_FIELDS = (
    "chunk_size",
    "quantile_norm",
    "needs_images",
    "needs_language",
    "typical_resolution",
)


def _override_problem(key: str, val) -> str | None:
    """Why an override value has the wrong type, or None when it is fine.
    Types are checked, not coerced: a string "false" is truthy, so an
    unchecked quantile_norm="false" would silently keep quantile norm on."""
    if key == "chunk_size":
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            return "a positive integer"
    elif key in ("quantile_norm", "needs_images", "needs_language"):
        if not isinstance(val, bool):
            return "a boolean (true/false, not a string)"
    elif key == "typical_resolution":
        if not (
            isinstance(val, (list, tuple))
            and len(val) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in val)
        ):
            return "a [width, height] pair of positive integers"
    return None


def canonical_policy_name(name: str) -> str:
    """Resolve an alias (pi0.5) to the canonical policies.json key (pi05)."""
    low = name.lower()
    return POLICY_ALIASES.get(low, low)


def resolve_policy(name: str | None, overrides: dict | None = None):
    """The declared defaults for a policy, merged with per-call overrides.

    Returns (canonical_name, entry); (canonical_name_or_None, None) when the
    policy is unknown. Overrides are validated against the policies.json
    schema so a typo fails loudly instead of tuning nothing."""
    if not name:
        return None, None
    canon = canonical_policy_name(name)
    entry = POLICY_DEFAULTS.get(canon)
    if entry is None:
        return canon, None
    merged = dict(entry)
    for key, val in (overrides or {}).items():
        if key not in OVERRIDEABLE_POLICY_FIELDS:
            raise ValueError(
                f"unknown policy override {key!r} "
                f"(valid: {', '.join(OVERRIDEABLE_POLICY_FIELDS)})"
            )
        problem = _override_problem(key, val)
        if problem:
            raise ValueError(f"policy override {key!r} must be {problem}, got {val!r}")
        merged[key] = val
    return canon, merged


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

    def _quantile_entry(self, ctx, feat: str) -> dict:
        """The stats entry to test for q01/q99.

        The raw meta/stats.json wins when it carries quantiles: the
        effective stats (per-episode aggregates on v2.1) keep only
        mean/std/min/max/count and would report quantiles missing from a
        dataset whose stats.json ships them. Which file a given training
        pipeline actually reads varies by pipeline and is not something
        this check can see - pipelines that compute their own norm stats
        read neither."""
        raw = ctx.stats_json.get(feat) if getattr(ctx, "stats_json", None) else None
        if isinstance(raw, dict) and "q01" in raw and "q99" in raw:
            return raw
        return ctx.stats.get(feat) or {}

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        canon, policy = resolve_policy(cfg.policy, cfg.policy_overrides or None)
        if cfg.policy and policy is None:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    f"unknown policy {cfg.policy!r}; running policy-agnostic checks "
                    f"(known: {', '.join(sorted(POLICY_DEFAULTS))})",
                )
            )
        # the name findings cite: the canonical policies.json key
        pname = canon or cfg.policy

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
                        details={"feature": feat},
                    )
                )

        # 2. quantile normalization support (pi0-family normalizers consume q01/q99)
        if policy and policy["quantile_norm"]:
            for feat in (STATE_KEY, ACTION_KEY):
                entry = self._quantile_entry(ctx, feat)
                if entry and not ("q01" in entry and "q99" in entry):
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"{pname} normalizes with q01/q99 quantiles but "
                            f"{ctx.stats_source} has none for {feat} - recompute stats "
                            "with quantiles, unless your training pipeline computes its "
                            "own norm stats",
                            details={"feature": feat, "policy": pname, "stats": sorted(entry)},
                        )
                    )

        # 3. episode length vs action chunk
        if policy:
            chunk = int(policy["chunk_size"])
            short = [e.index for e in ctx.episodes if 0 < e.length < chunk]
            if short:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{len(short)}/{ctx.total_episodes} episodes are shorter than "
                        f"{pname}'s action chunk ({chunk} frames): {short[:10]}"
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
                        details={"feature": key, "declared": declared, "actual": arr.shape[1]},
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
                        f"{pname} is trained with camera input but this dataset declares "
                        "no image or video features - a state-only dataset cannot serve it as-is",
                        details={"policy": pname},
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
                            f"{pname} is usually trained at {tw}x{th}; "
                            f"{', '.join(sorted(small))} is smaller - usable (the loader "
                            "resizes), but the policy sees less detail than it was built for",
                            details={"policy": pname, "expected": [tw, th], "smaller": sorted(small)},
                        )
                    )
            if policy.get("needs_language") and not _has_task_strings(ctx):
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{pname} conditions on language but no episode declares a task "
                        "string - training here teaches a task-agnostic policy",
                        details={"policy": pname},
                    )
                )

        if not res.findings:
            res.summary = (
                f"ready for {pname}" if cfg.policy else "no deployment blockers found"
            )
        else:
            n_fail = sum(1 for f in res.findings if f.severity == Severity.FAIL)
            n_warn = sum(1 for f in res.findings if f.severity == Severity.WARN)
            res.summary = f"{n_fail} blocker(s), {n_warn} warning(s)"
            if cfg.policy:
                res.summary += f" for {pname}"
        return res
