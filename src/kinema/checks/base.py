"""Shared check framework: severities, findings, config, registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from kinema.dataset import DatasetContext


class Severity(IntEnum):
    OK = 0
    INFO = 1
    WARN = 2
    FAIL = 3
    SKIP = 4  # check did not run; not a verdict on the data


@dataclass
class Finding:
    check_id: str
    severity: Severity
    message: str
    episode_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    check_id: str
    title: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    skipped: bool = False
    skip_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> Severity:
        if self.skipped:
            return Severity.SKIP
        if not self.findings:
            return Severity.OK
        return max((f.severity for f in self.findings), default=Severity.OK)


@dataclass
class CheckConfig:
    video: bool = False
    policy: str | None = None
    # per-run policies.json overrides (the MCP diagnosis layer passes these;
    # not a threshold, so --config JSON files cannot set it)
    policy_overrides: dict = field(default_factory=dict)
    # stats-drift (mean in std devs, std as a ratio, min/max as range fraction)
    stats_mean_warn_sigma: float = 0.10
    stats_mean_fail_sigma: float = 0.50
    stats_std_warn: float = 0.05
    stats_std_fail: float = 0.25
    stats_std_oversize_fail: float = 3.0  # stored action std >= 3x real: FAIL
    stats_range_warn: float = 0.01
    stats_range_fail: float = 0.10
    # sampled loads without per-episode stats: gross-corruption backstop
    stats_sample_std_ratio: float = 3.0
    stats_sample_mean_sigma: float = 2.0
    # ...and the sharper test above it: bootstrap the sampling distribution
    # of the pooled stats over loaded episodes (the episode is the unit -
    # frames within one are autocorrelated) and flag stored stats outside
    # the (1 - alpha) interval
    stats_sample_min_episodes: int = 8
    stats_sample_bootstrap: int = 400
    stats_sample_alpha: float = 0.001
    # timestamp-grid
    grid_warn_frac: float = 0.10  # fraction of the frame period
    # calibration-drift
    calib_warn_range_frac: float = 0.05
    calib_fail_range_frac: float = 0.15
    calib_min_corr: float = 0.90  # action must track state to be comparable
    calib_scale_warn: float = 0.05  # offset drifts > 5% of range across the range
    calib_scale_fail: float = 0.15
    calib_scale_min_r2: float = 0.60  # ...and a line explains the offset
    calib_scale_line_vs_step: float = 1.5  # ...and beats the best two-level step (backlash)
    calib_gripper_warn: float = 0.15  # gripper offset at its free end
    # delta commands: measured step vs commanded step (regression slope)
    calib_delta_scale_warn: float = 0.5  # |slope - 1| at/above this gates
    calib_delta_scale_info: float = 0.25  # milder disagreement: report only
    calib_delta_min_r2: float = 0.5  # steps must track before a scale verdict
    # idle / stuck
    idle_eps_frac: float = 0.01  # fraction of per-joint action std
    idle_min_seconds: float = 2.0
    idle_warn_fraction: float = 0.30
    idle_state_frac: float = 0.02  # state moves < 2% of its range over an idle run
    stuck_range_frac: float = 0.005  # episode range < 0.5% of the dataset range
    stuck_warn_share: float = 0.25  # stuck in <= 25% of episodes: anomalous
    stuck_unused_share: float = 0.90  # stuck in >= 90%: unused DOF (INFO)
    follow_min_command: float = 0.10  # commanded >= 10% of the joint's range...
    follow_ratio: float = 0.10  # ...but the follower moved < 10% of that
    # action saturation (frames pinned exactly at the limit while moving)
    saturation_warn_fraction: float = 0.05
    # outliers
    outlier_iqr_k: float = 1.5


# runtime flags live on CheckConfig but are not thresholds: --video/--policy
# and the MCP layer's policy_overrides dict
_CONFIG_ONLY = ("video", "policy", "policy_overrides")


def load_config_overrides(path: str) -> dict[str, Any]:
    """Threshold overrides from a JSON file (`kinema check --config`).

    Keys are CheckConfig field names; unknown keys, non-numeric values and
    the runtime flags are rejected so a typo never silently tunes nothing.
    The return is typed Any so `CheckConfig(**overrides)` type-checks -
    the values are validated numeric above."""
    with open(path) as f:
        data = json.load(f)
    known = {f.name for f in fields(CheckConfig)} - set(_CONFIG_ONLY)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of threshold overrides")
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown config keys: {', '.join(unknown)} "
            f"(valid: {', '.join(sorted(known))})"
        )
    bad = sorted(
        k for k, v in data.items() if isinstance(v, bool) or not isinstance(v, (int, float))
    )
    if bad:
        raise ValueError(f"{path}: values must be numbers: {', '.join(bad)}")
    return dict(data)


class Check:
    """Base class. Subclasses set id/title and implement run()."""

    id: str = ""
    title: str = ""
    requires_video: bool = False

    def run(self, ctx: "DatasetContext", cfg: CheckConfig) -> CheckResult:
        raise NotImplementedError

    def skip(self, reason: str) -> CheckResult:
        return CheckResult(self.id, self.title, skipped=True, skip_reason=reason)


def _registry() -> list[Check]:
    from kinema.checks.actions import ActionSaturationCheck, NanAndNormalizationCheck
    from kinema.checks.calibration import CalibrationDriftCheck
    from kinema.checks.deployment import DeploymentReadinessCheck
    from kinema.checks.duplicates import DuplicateCheck
    from kinema.checks.idle import IdleSegmentsCheck, StuckActuatorCheck
    from kinema.checks.metadata import (
        CameraCoverageCheck,
        EmbeddedImagesCheck,
        RewardDoneCheck,
        TaskStringsCheck,
    )
    from kinema.checks.outliers import OutlierEpisodesCheck
    from kinema.checks.stats_drift import StatsDriftCheck
    from kinema.checks.timestamps import DroppedFramesCheck, TimestampGridCheck
    from kinema.checks.visual import VisualQualityCheck

    return [
        StatsDriftCheck(),
        TimestampGridCheck(),
        DroppedFramesCheck(),
        CalibrationDriftCheck(),
        IdleSegmentsCheck(),
        StuckActuatorCheck(),
        ActionSaturationCheck(),
        NanAndNormalizationCheck(),
        DuplicateCheck(),
        OutlierEpisodesCheck(),
        EmbeddedImagesCheck(),
        CameraCoverageCheck(),
        TaskStringsCheck(),
        RewardDoneCheck(),
        VisualQualityCheck(),
        DeploymentReadinessCheck(),
    ]


ALL_CHECKS: list[Check] = _registry()
