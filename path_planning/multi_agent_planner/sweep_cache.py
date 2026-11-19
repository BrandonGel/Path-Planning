"""Planner-side memo for the map's collision-sweep queries (the map itself does not cache)."""
import weakref


class SweepCache(dict):
    """A bounded memo ``key -> result``, emptied when it reaches ``max_entries`` so long runs with ever-new
    query points stay bounded. Results are shared between callers: do not modify them."""

    def __init__(self, max_entries: int = 250_000):
        super().__init__()
        self.max_entries = max_entries

    def get_or_compute(self, key, compute):
        hit = self.get(key)
        if hit is None:
            hit = compute()
            if len(self) >= self.max_entries:
                self.clear()
            self[key] = hit
        return hit


# graph_map -> (its sweep_version, {name: SweepCache}); weak, so a cache dies with its map.
_SHARED = weakref.WeakKeyDictionary()


def shared_sweep_cache(graph_map, name: str = "collision_sweep") -> SweepCache:
    """The ``SweepCache`` called ``name`` that every planner on ``graph_map`` shares. Planners built anew
    per call (Neural-ATTF's low-level SIPP) reuse each other's sweeps through it. It is replaced by an empty
    one whenever the map's collision sweep is rebound (``GraphSampler.sweep_version`` changes)."""
    version = getattr(graph_map, "sweep_version", 0)
    entry = _SHARED.get(graph_map)
    if entry is None or entry[0] != version:
        entry = _SHARED[graph_map] = (version, {})
    cache = entry[1].get(name)
    if cache is None:
        cache = entry[1][name] = SweepCache()
    return cache
