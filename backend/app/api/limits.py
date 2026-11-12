import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window limits: N requests per client per window, plus a global cap over its own window.
    Only allowed requests are recorded, so a denied request does not extend a client's wait."""

    def __init__(self, per_client: int, client_window_s: float, global_cap: int, global_window_s: float,
                 clock=time.monotonic):
        self.per_client, self.client_window_s = per_client, client_window_s
        self.global_cap, self.global_window_s = global_cap, global_window_s
        self.clock = clock
        self._clients: dict[str, deque] = {}
        self._global: deque = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _trim(q: deque, now: float, window: float) -> None:
        while q and now - q[0] >= window:
            q.popleft()

    def check(self, client: str) -> tuple[bool, str, int]:
        """(allowed, reason, retry_after_seconds)."""
        with self._lock:
            now = self.clock()
            self._trim(self._global, now, self.global_window_s)
            q = self._clients.setdefault(client, deque())
            self._trim(q, now, self.client_window_s)
            if len(self._clients) > 1000:  # forget clients with no recent requests
                for key in list(self._clients):
                    self._trim(self._clients[key], now, self.client_window_s)
                    if not self._clients[key] and key != client:
                        del self._clients[key]
            if len(self._global) >= self.global_cap:
                wait = int(self.global_window_s - (now - self._global[0]))
                return False, "daily demo limit reached", max(1, wait)
            if len(q) >= self.per_client:
                wait = int(self.client_window_s - (now - q[0]))
                return False, "too many questions from this client", max(1, wait)
            q.append(now)
            self._global.append(now)
            return True, "", 0