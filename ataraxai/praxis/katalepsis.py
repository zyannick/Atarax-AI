import functools
import inspect
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

from fastapi import Request
from prometheus_client import Counter, Histogram

F = TypeVar("F", bound=Callable[..., Any])

API_REQUEST_LATENCY_SECONDS = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

VAULT_UNLOCK_ATTEMPTS_TOTAL = Counter(
    "vault_unlock_attempts_total",
    "Total number of vault unlock attempts",
    ["method", "endpoint", "status"],
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
        self._initialized = True

    @contextmanager
    def measure_time(self, metric: Histogram, labels: Optional[Dict[str, str]] = None):
        """A context manager to time a block of code and record the duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if labels:
                metric.labels(**labels).observe(duration)
            else:
                metric.observe(duration)

    def instrument_api(self, method: str = "POST") -> Callable[[F], F]:
        """
        A versatile decorator that instruments both sync and async functions.
        It records latency and counts for API endpoints.

        Usage:
            @katalepsis_monitor.instrument_api(method="GET")
            async def my_endpoint():
                ...
        """

        def decorator(func: F) -> F:
            def record_metrics(start_time: float, status_code: str, *args, **kwargs):  # type: ignore
                duration = time.time() - start_time

                request = next((arg for arg in args if isinstance(arg, Request)), None)  # type: ignore
                if not request:
                    request = kwargs.get("request")  # type: ignore

                if request and hasattr(request, "scope"):
                    endpoint_path = request.scope.get("route", {}).get("path", f"/{func.__name__}")  # type: ignore
                else:
                    endpoint_path = f"/{func.__name__}"  # type: ignore

                labels = {"method": method.upper(), "endpoint": endpoint_path, "status": status_code}  # type: ignore
                API_REQUEST_LATENCY_SECONDS.labels(**labels).observe(duration)
                HTTP_REQUESTS_TOTAL.labels(**labels).inc()

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    try:
                        response = await func(*args, **kwargs)
                        record_metrics(start_time, "success", *args, **kwargs)
                        return response
                    except Exception:
                        record_metrics(start_time, "error", *args, **kwargs)
                        raise

                return async_wrapper  # type: ignore

            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    try:
                        response = func(*args, **kwargs)
                        record_metrics(start_time, "success", *args, **kwargs)
                        return response
                    except Exception:
                        record_metrics(start_time, "error", *args, **kwargs)
                        raise

                return sync_wrapper  # type: ignore

        return decorator


katalepsis_monitor = Katalepsis()
