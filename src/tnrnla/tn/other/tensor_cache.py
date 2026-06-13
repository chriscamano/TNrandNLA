from collections import OrderedDict


class tensorCache:
    """
    Lightweight LRU cache for tensor-derived views (reshape/transpose/conj/etc.).
    Values are stored as views when builders return views, so memory overhead stays low.
    """

    def __init__(self, enabled=True, max_entries=4096):
        self._enabled = bool(enabled)
        self._max_entries = int(max_entries)
        self._store = OrderedDict()

    @property
    def enabled(self):
        return self._enabled

    def set_enabled(self, enabled=True):
        self._enabled = bool(enabled)
        if not self._enabled:
            self._store.clear()

    def clear(self):
        self._store.clear()

    def get(self, key, builder):
        if not self._enabled:
            return builder()

        store = self._store
        if key in store:
            value = store.pop(key)
            store[key] = value
            return value

        value = builder()
        store[key] = value
        if len(store) > self._max_entries:
            store.popitem(last=False)
        return value


# Convenience alias for conventional class-style imports.
TensorCache = tensorCache
