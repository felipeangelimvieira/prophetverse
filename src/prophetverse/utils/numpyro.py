import numpyro
from typing import Any, List, Union


class CacheMessenger(numpyro.primitives.Messenger):
    """
    A Messenger that remembers the first value drawn at each sample site,
    and on subsequent visits just returns that cached value.
    """

    def __init__(self, cache_types: Union[List[str], str] = "sample"):
        super().__init__(fn=None)
        self._cache: dict[str, Any] = {}
        self.cache_types = cache_types

        self._cache_types = (
            [cache_types] if isinstance(cache_types, str) else cache_types
        )

    def process_message(self, msg):
        # only intercept actual sample sites
        if msg["type"] in self._cache_types and msg["name"] in self._cache:
            # short‐circuit: return the cached value

            for k, v in self._cache[msg["name"]].items():
                msg[k] = v
            # Avoid errors in tracers above due to duplicated names
            msg["name"] = msg["name"] + "_cached_" + str(id(msg)) + ":ignore"

    def postprocess_message(self, msg):
        # after a real sample has been taken, cache it
        if msg["type"] in self._cache_types:
            if msg["name"] not in self._cache:
                self._cache[msg["name"]] = msg
