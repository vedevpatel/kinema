"""Shared check framework: severities, findings, config, registry."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # stats-drift
    stats_rel_tol: float = 0.01
    stats_fail_tol: float = 0.10
    # timestamp-grid
    grid_warn_frac: float = 0.10  # fraction of the frame period
    # calibration-drift
    calib_warn_range_frac: float = 0.05
    calib_fail_range_frac: float = 0.15
    # idle / stuck
    idle_eps_frac: float = 0.01  # fraction of per-joint action std
    idle_min_seconds: float = 2.0
    idle_warn_fraction: float = 0.30
    stuck_warn_fraction: float = 0.80
    # action saturation
    saturation_warn_fraction: float = 0.05
    # outliers
    outlier_iqr_k: float = 1.5
    # duplicates
    near_dup_max_pairs: int = 400


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
        VisualQualityCheck(),
        DeploymentReadinessCheck(),
    ]


ALL_CHECKS: list[Check] = _registry()
