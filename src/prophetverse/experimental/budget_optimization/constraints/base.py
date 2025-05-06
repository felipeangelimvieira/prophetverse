from dataclasses import dataclass


@dataclass(frozen=True)
class Constraint:
    """Base class for all high-level budget constraints."""

    pass


@dataclass(frozen=True)
class UtilityFunction:
    """Base class for any scalar objective computed on model output."""

    pass
