import functools
import threading
from prometheus_client import Counter, Histogram
import time
from contextlib import contextmanager
from typing import Callable, Dict, Any, Optional
from typing import TypeVar
import inspect

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
    """
    A Singleton class for application-wide observability and metrics collection.
    Provides decorators and context managers to instrument code.
    """
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
            endpoint_path = f"/v1/{func.__name__}"
            
            # Handle async functions
            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    status_code = "success"
                    start_time = time.time()
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        status_code = "error"
                        raise # Re-raise the exception after catching it
                    finally:
                        duration = time.time() - start_time
                        labels = {"method": method, "endpoint": endpoint_path, "status": status_code}
                        API_REQUEST_LATENCY_SECONDS.labels(**labels).observe(duration)
                        HTTP_REQUESTS_TOTAL.labels(**labels).inc()
                return async_wrapper # type: ignore
            
            # Handle sync functions
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    status_code = "success"
                    start_time = time.time()
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        status_code = "error"
                        raise
                    finally:
                        duration = time.time() - start_time
                        labels = {"method": method, "endpoint": endpoint_path, "status": status_code}
                        API_REQUEST_LATENCY_SECONDS.labels(**labels).observe(duration)
                        HTTP_REQUESTS_TOTAL.labels(**labels).inc()
                return sync_wrapper # type: ignore
        return decorator


katalepsis_monitor = Katalepsis()