"""Check registry. Importing this package registers all built-in checks."""

from kinema.checks.base import (
    ALL_CHECKS,
    Check,
    CheckConfig,
    CheckResult,
    Finding,
    Severity,
    load_config_overrides,
)

__all__ = [
    "ALL_CHECKS",
    "Check",
    "CheckConfig",
    "CheckResult",
    "Finding",
    "Severity",
    "load_config_overrides",
]
