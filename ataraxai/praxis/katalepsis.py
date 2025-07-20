import functools
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from contextlib import contextmanager
from typing import Callable, Dict, Any
import asyncio
from typing import Callable, TypeVar, Any, cast

F = TypeVar("F", bound=Callable[..., Any])

API_REQUEST_LATENCY_SECONDS = Histogram(
    "api_request_latency_seconds", "Latency of API requests", ["method", "endpoint"]
)


class Katalepsis:
    def __init__(self):
        self.metrics_buffer = []
        self.cache_stats = {"hits": 0, "misses": 0}

    @contextmanager
    def measure_time(self, metric: Histogram, labels: Dict[str, Any] = {}):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if labels:
                metric.labels(**labels).observe(duration)
            else:
                metric.observe(duration)

    def instrument_api(self, func: Callable) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            endpoint_path = f"/v1/{func.__name__}"
            with self.measure_time(
                API_REQUEST_LATENCY_SECONDS,
                labels={"method": "POST", "endpoint": endpoint_path},
            ):
                return await func(*args, **kwargs)  # type: ignore

        return cast(F, wrapper)
