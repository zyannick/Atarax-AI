import functools
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from contextlib import contextmanager
from typing import Awaitable, Callable, Dict, Any, Optional
import asyncio
from typing import Callable, TypeVar, Any, cast

F = TypeVar("F")

API_REQUEST_LATENCY_SECONDS = Histogram(
    "api_request_latency_seconds", "Latency of API requests", ["method", "endpoint"]
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total number of HTTP requests", ["method", "endpoint"]
)

VAULT_UNLOCK_ATTEMPTS_TOTAL = Counter(
    "vault_unlock_attempts_total", "Total number of vault unlock attempts", ["status"]
)


class Katalepsis:

    _instance: Optional["Katalepsis"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "Katalepsis":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return

        self._buffer_lock = threading.Lock()
        self.metrics_buffer: list = []
        self.cache_stats = {"hits": 0, "misses": 0}
        self._initialized = True

    @contextmanager
    def measure_time(
        self,
        metric: Histogram,
        labels: Optional[Dict[str, str]] = None,
        record_count: bool = True,
    ):
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            final_labels = (labels or {}).copy()
            final_labels["status"] = status

            try:
                if final_labels:
                    metric.labels(**final_labels).observe(duration)
                    if record_count:
                        HTTP_REQUESTS_TOTAL.labels(**final_labels).inc()
                else:
                    metric.observe(duration)
            except Exception:
                pass

    def instrument_async_api(
        self, method: str = "POST", endpoint_prefix: str = "/v1/"
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:

        def decorator(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs): # type: ignore
                endpoint_path = f"{endpoint_prefix}{func.__name__}"
                labels = {"method": method, "endpoint": endpoint_path}

                with self.measure_time(API_REQUEST_LATENCY_SECONDS, labels):
                    return await func(*args, **kwargs)

            return wrapper # type: ignore

        return decorator

    def instrument_sync_api(
        self, method: str = "POST", endpoint_prefix: str = "/v1/"
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs): # type: ignore
                endpoint_path = f"{endpoint_prefix}{func.__name__}"
                labels = {"method": method, "endpoint": endpoint_path}

                with self.measure_time(API_REQUEST_LATENCY_SECONDS, labels):
                    return func(*args, **kwargs)

            return wrapper # type: ignore

        return decorator

    def instrument_api(
        self, func: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        return self.instrument_async_api()(func)
