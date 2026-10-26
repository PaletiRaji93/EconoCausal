"""Project-specific exceptions. Raised instead of returning fake results."""
import importlib


class EconoCausalError(Exception):
    """Base class for all EconoCausal errors."""


class DependencyMissingError(EconoCausalError):
    """An optional/heavy dependency (EconML, DoWhy, ...) is not installed."""


class DataValidationError(EconoCausalError):
    """The input data failed validation (message explains how to fix it)."""


class LeakageError(EconoCausalError):
    """A feature would leak post-treatment or target information."""


class InsufficientSupportError(EconoCausalError):
    """The data cannot support the requested causal estimate (overlap/dose support)."""


def require(module: str, pip_name: str | None = None):
    """Import `module` or raise DependencyMissingError with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        pkg = pip_name or module.split(".")[0]
        raise DependencyMissingError(
            f"Required package '{pkg}' is not installed. Run: pip install {pkg}"
        ) from exc
