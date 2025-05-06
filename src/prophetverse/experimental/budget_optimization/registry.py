import threading
from typing import Type, Callable, Dict, List, Union
import importlib
import pkg_resources  # for entry-point discovery

from prophetverse.experimental.budget_optimization.constraints.base import (
    Constraint,
    UtilityFunction,
)
from prophetverse.experimental.budget_optimization.translation_result import (
    TranslationResult,
)

# Separate registries for clarity
_CONSTRAINT_TRANSLATORS: Dict[str, Dict[Type[Constraint], Callable]] = {}
_UTILITY_TRANSLATORS: Dict[str, Dict[Type[UtilityFunction], Callable]] = {}
_registry_lock = threading.RLock()


def register_constraint_translator(
    backend_name: str, constraint_type: Type[Constraint]
):
    """Decorator to register a translator for a Constraint on a given backend."""

    def decorator(fn: Callable[[Constraint, Any], TranslationResult]):
        with _registry_lock:
            _CONSTRAINT_TRANSLATORS.setdefault(backend_name, {})[constraint_type] = fn
        return fn

    return decorator


def register_utility_translator(backend_name: str, util_type: Type[UtilityFunction]):
    """Decorator to register a translator for a UtilityFunction on a given backend."""

    def decorator(fn: Callable[[UtilityFunction, Any], TranslationResult]):
        with _registry_lock:
            _UTILITY_TRANSLATORS.setdefault(backend_name, {})[util_type] = fn
        return fn

    return decorator


def get_constraint_translator(backend_name: str, constraint: Constraint):
    return _CONSTRAINT_TRANSLATORS.get(backend_name, {}).get(type(constraint))


def get_utility_translator(backend_name: str, util: UtilityFunction):
    return _UTILITY_TRANSLATORS.get(backend_name, {}).get(type(util))


def list_translators(backend_name: str) -> Dict[str, List[str]]:
    """Return discovered translators for introspection.
    Returns a dict with keys 'constraints' and 'utilities' listing type names."""
    cons = [cls.__name__ for cls in _CONSTRAINT_TRANSLATORS.get(backend_name, {})]
    utils = [cls.__name__ for cls in _UTILITY_TRANSLATORS.get(backend_name, {})]
    return {"constraints": cons, "utilities": utils}


def discover_entrypoint_plugins(group: str = "budget_optimizer.plugins"):
    """
    Discover and import all entry-point plugins under the given setuptools group.
    Users can define entry-points in setup.py to auto-register translators.
    """
    for ep in pkg_resources.iter_entry_points(group=group):
        try:
            importlib.import_module(ep.module_name)
        except ImportError:
            # optionally log or warn
            pass
