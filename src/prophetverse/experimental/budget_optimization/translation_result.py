from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationResult:
    """
    Well-typed container for translator outputs.
    Fields:
      - kwargs: dict of backend-specific args (e.g. bounds, A_eq, b_eq)
      - objective_fn: for utilities: Callable[[array], float]
      - gradient_fn: optional Callable[[array], array]
    """

    kwargs: Dict[str, Any]
    objective_fn: Optional[Callable] = None
    gradient_fn: Optional[Callable] = None
