"""Compatibility shim: re-export `Sum` from `multiply.py`.

The real implementations live in `prophetverse.effects.multiply` to keep a
single source of truth for chained-effects logic. This module keeps the old
import path working for callers that import `Sum` from
`prophetverse.effects.sum`.
"""

from prophetverse.effects.multiply import Sum

__all__ = ["Sum"]
