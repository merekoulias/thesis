"""Fairness audit package (baseline audit of credit scorers)."""

from typing import Any

__all__ = ["run_audit"]


def __getattr__(name: str) -> Any:
    if name == "run_audit":
        from src.fairness.audit import run_audit as _run_audit

        return _run_audit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
