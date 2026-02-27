"""Panel (multi-series) effects with hierarchical hyperpriors.

This sub-package provides geo-level (panel) versions of standard effects.
Each effect inherits from :class:`BaseGeoEffect`, which supplies a shared
mechanism for sampling parameters either **shared** across all series or
**per-series** via non-centred hierarchical priors.
"""

from prophetverse.effects.panel.base import BaseGeoEffect
from prophetverse.effects.panel.geo_hill import GeoHillEffect
from prophetverse.effects.panel.geo_michaelis_menten import GeoMichaelisMentenEffect
from prophetverse.effects.panel.geo_geometric_adstock import GeoGeometricAdstockEffect
from prophetverse.effects.panel.geo_weibull_adstock import GeoWeibullAdstockEffect

__all__ = [
    "BaseGeoEffect",
    "GeoHillEffect",
    "GeoMichaelisMentenEffect",
    "GeoGeometricAdstockEffect",
    "GeoWeibullAdstockEffect",
]
