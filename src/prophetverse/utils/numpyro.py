import numpyro


class CacheMessenger(numpyro.primitives.Messenger):
    """
    A Messenger that remembers the first value drawn at each sample site,
    and on subsequent visits just returns that cached value.
    """

    def __init__(self):
        super().__init__(fn=None)
        self._cache: dict[str, Any] = {}

    def process_message(self, msg):
        # only intercept actual sample sites
        if not numpyro.primitives._PYRO_STACK:
            return
        if msg["type"] == "sample" and msg["name"] in self._cache:
            # short‐circuit: return the cached value
            msg["value"] = self._cache[msg["name"]]
            msg["stop"] = True  # skip all further handlers

    def postprocess_message(self, msg):
        if not numpyro.primitives._PYRO_STACK:
            return
        # after a real sample has been taken, cache it
        if msg["type"] == "sample":
            if msg["name"] not in self._cache:
                self._cache[msg["name"]] = msg["value"]
            else:
                # if the sample site is already in cache, just return the cached value
                msg["value"] = self._cache[msg["name"]]
