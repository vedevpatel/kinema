"""Check registry. Importing this package registers all built-in checks."""

from kinema.checks.base import ALL_CHECKS, Check, CheckConfig, CheckResult, Finding, Severity

__all__ = ["ALL_CHECKS", "Check", "CheckConfig", "CheckResult", "Finding", "Severity"]
