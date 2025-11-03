import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

from pybreaker import CircuitBreaker, CircuitBreakerError

from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger


class RequestPriority(IntEnum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RequestTimeoutError(Exception):
    def __init__(self, timeout_duration: float, message: str = ""):
        self.timeout_duration = timeout_duration
        default_message = f"Request timed out after {timeout_duration} seconds"
        super().__init__(message or default_message)


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    submission_time: float = field(compare=False)
    future: asyncio.Future = field(compare=False)
    func: Callable = field(compare=False)
    args: Any = field(compare=False, default=())
    kwargs: Any = field(compare=False, default_factory=dict)
    timeout: Optional[float] = field(compare=False, default=None)

    def is_expired(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.submission_time > self.timeout

    def remaining_time(self) -> Optional[float]:
        if self.timeout is None:
            return None
        elapsed = time.monotonic() - self.submission_time
        remaining = self.timeout - elapsed
        return max(0, remaining)


class RequestManager:

    def __init__(
        self,
        rate_limit: int = 10,
        bucket_capacity: int = 20,
        breaker_fail_max: int = 5,
        breaker_reset_timeout: int = 60,
        max_queue_size: int = 1000,
        concurrent_workers: int = 5,
        default_timeout: Optional[float] = None,
        cleanup_interval: float = 30.0,
        logger: logging.Logger = AtaraxAILogger("RequestManager").get_logger(),
    ):
        """
        Initializes the RequestManager with rate limiting, circuit breaking, and request queue management.

        Args:
            rate_limit (int, optional): Maximum number of requests allowed per second. Defaults to 10.
            bucket_capacity (int, optional): Maximum burst size for the token bucket rate limiter. Defaults to 20.
            breaker_fail_max (int, optional): Maximum consecutive failures before opening the circuit breaker. Defaults to 5.
            breaker_reset_timeout (int, optional): Time in seconds before attempting to reset the circuit breaker. Defaults to 60.
            max_queue_size (int, optional): Maximum number of requests that can be queued. Defaults to 1000.
            concurrent_workers (int, optional): Number of concurrent worker tasks processing requests. Defaults to 5.
            default_timeout (float, optional): Default timeout for requests in seconds. None means no timeout.
            cleanup_interval (float, optional): Interval for cleaning up expired requests. Defaults to 30.0.
        """
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._processor_tasks: List[asyncio.Task] = []
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
        self._concurrent_workers = concurrent_workers
        self._default_timeout = default_timeout
        self._cleanup_interval = cleanup_interval
        self.logger = logger
        self._rate_limit = rate_limit
        self._bucket_capacity = float(bucket_capacity)
        self._tokens = self._bucket_capacity
        self._last_refill_time = time.monotonic()
        self._rate_lock = asyncio.Lock()

        self._breaker = CircuitBreaker(
            fail_max=breaker_fail_max, reset_timeout=breaker_reset_timeout
        )

        self._running_requests: set[asyncio.Task] = set()

        self._metrics = {
            "requests_submitted": 0,
            "requests_processed": 0,
            "requests_failed": 0,
            "requests_rejected": 0,
            "requests_timed_out": 0,
            "requests_expired_in_queue": 0,
            "circuit_open_count": 0,
        }

        self.logger.info(
            f"RequestManager initialized with rate limit: {rate_limit}/s, burst capacity: {bucket_capacity}"
        )
        self.logger.info(
            f"Circuit Breaker initialized with fail_max={breaker_fail_max}, reset_timeout={breaker_reset_timeout}s"
        )
        self.logger.info(
            f"Queue max size: {max_queue_size}, concurrent workers: {concurrent_workers}"
        )
        self.logger.info(
            f"Default timeout: {default_timeout}s, cleanup interval: {cleanup_interval}s"
        )

    async def _refill_token_bucket(self):
        async with self._rate_lock:
            now = time.monotonic()
            time_delta = now - self._last_refill_time
            new_tokens = time_delta * self._rate_limit
            self._tokens = min(self._bucket_capacity, self._tokens + new_tokens)
            self._last_refill_time = now

    async def _wait_for_token(self) -> bool:
        while not self._shutdown_event.is_set():
            await self._refill_token_bucket()

            async with self._rate_lock:
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                return False
            except asyncio.TimeoutError:
                continue

        return False

    async def submit_request(
        self,
        request_name: str,
        func: Callable,
        *args: Any,
        priority: RequestPriority = RequestPriority.MEDIUM,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a request for processing.

        Args:
            func: The function to execute
            *args: Arguments for the function
            priority: Request priority
            timeout: Timeout in seconds for this specific request.
                    If None, uses default_timeout from constructor.
                    If both are None, no timeout is applied.
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution

        Raises:
            CircuitBreakerError: If circuit breaker is open
            asyncio.QueueFull: If the request queue is full
            RequestTimeoutError: If the request times out
        """
        self.logger.debug(
            f"Submitting request '{request_name}' with priority {priority.name}"
        )
        if self._breaker.current_state == "open":
            self._metrics["requests_rejected"] += 1
            raise CircuitBreakerError("Circuit breaker is open")

        effective_timeout = timeout if timeout is not None else self._default_timeout

        future = asyncio.Future() # type: ignore

        request = PrioritizedRequest(
            priority=priority.value,
            submission_time=time.monotonic(),
            timeout=effective_timeout,
            future=future,
            func=func,
            args=args,
            kwargs=kwargs,
        )

        try:
            self._queue.put_nowait(request)
            self._metrics["requests_submitted"] += 1
            timeout_info = (
                f" (timeout: {effective_timeout}s)" if effective_timeout else ""
            )
            self.logger.info(
                f"Submitted request with priority {priority.name}{timeout_info}. Queue size: {self._queue.qsize()}"
            )
        except asyncio.QueueFull:
            self._metrics["requests_rejected"] += 1
            self.logger.warning(
                f"Queue is full! Rejecting request with priority {priority.name}"
            )
            raise

        return future

    async def _execute_request(self, request: PrioritizedRequest):
        try:
            if request.is_expired():
                self._metrics["requests_expired_in_queue"] += 1
                error_msg = f"Request expired in queue after {request.timeout}s"
                request.future.set_exception(
                    RequestTimeoutError(request.timeout if request.timeout else 0.0, error_msg)
                )
                return

            remaining_timeout = request.remaining_time()
            if remaining_timeout is not None and remaining_timeout <= 0:
                self._metrics["requests_timed_out"] += 1
                error_msg = (
                    f"Request timed out before execution (timeout: {request.timeout}s)"
                )
                request.future.set_exception(
                    RequestTimeoutError(request.timeout if request.timeout else 0.0, error_msg)
                )
                return

            result = await (
                asyncio.wait_for(
                    self._breaker.call_async(
                        request.func, *request.args, **request.kwargs
                    ),
                    timeout=remaining_timeout,
                )
                if remaining_timeout is not None
                else self._breaker.call_async(
                    request.func, *request.args, **request.kwargs
                )
            )

            if not request.future.done():
                request.future.set_result(result)
            self._metrics["requests_processed"] += 1

        except asyncio.CancelledError:
            if not request.future.done():
                request.future.cancel()
            raise

        except asyncio.TimeoutError:
            self._metrics["requests_timed_out"] += 1
            timeout_duration = request.timeout or 0
            error_msg = f"Request execution timed out after {timeout_duration}s"
            if not request.future.done():
                request.future.set_exception(
                    RequestTimeoutError(timeout_duration, error_msg)
                )

        except CircuitBreakerError as e:
            if not request.future.done():
                request.future.set_exception(e)
            self._metrics["requests_rejected"] += 1
            self._metrics["circuit_open_count"] += 1

        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)
            self._metrics["requests_failed"] += 1

    async def _cleanup_expired_requests(self):
        self.logger.info("Started expired request cleanup task")

        while not self._shutdown_event.is_set():
            try:
                temp_requests = []
                expired_count = 0

                while not self._queue.empty():
                    try:
                        request = self._queue.get_nowait()

                        if request.is_expired():
                            expired_count += 1
                            self._metrics["requests_expired_in_queue"] += 1
                            error_msg = (
                                f"Request expired in queue after {request.timeout}s"
                            )
                            request.future.set_exception(
                                RequestTimeoutError(request.timeout, error_msg)
                            )
                            self._queue.task_done()
                        else:
                            temp_requests.append(request)
                    except asyncio.QueueEmpty:
                        break

                for request in temp_requests:
                    try:
                        self._queue.put_nowait(request)
                    except asyncio.QueueFull:
                        self.logger.warning(
                            "Queue full during cleanup, dropping request"
                        )
                        request.future.set_exception(
                            Exception("Queue full during cleanup")
                        )

                if expired_count > 0:
                    self.logger.info(
                        f"Cleaned up {expired_count} expired requests from queue"
                    )

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self._cleanup_interval
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            except asyncio.CancelledError:
                self.logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}", exc_info=True)
                await asyncio.sleep(1.0)

        self.logger.info("Expired request cleanup task stopped")

    async def _process_requests(self, worker_id: int):
        self.logger.info(f"Request processor worker {worker_id} started.")

        while not self._shutdown_event.is_set():
            try:
                try:
                    request: PrioritizedRequest = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                self.logger.debug(
                    f"Worker {worker_id} processing request with priority {request.priority}. Queue size: {self._queue.qsize()}"
                )

                if not await self._wait_for_token():
                    try:
                        self._queue.put_nowait(request)
                    except asyncio.QueueFull:
                        request.future.set_exception(Exception("Service shutting down"))
                    break

                task = asyncio.create_task(self._execute_request(request))
                self._running_requests.add(task)
                task.add_done_callback(lambda t: self._running_requests.discard(t))

                self._queue.task_done()

            except asyncio.CancelledError:
                self.logger.info(f"Request processor worker {worker_id} shutting down.")
                break
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in worker {worker_id}: {e}", exc_info=True
                )

        self.logger.info(f"Request processor worker {worker_id} stopped.")

    async def start(self):
        if self._processor_tasks:
            self.logger.warning("RequestManager is already running.")
            return

        self._shutdown_event.clear()
        self._processor_tasks = []

        for i in range(self._concurrent_workers):
            task = asyncio.create_task(self._process_requests(i))
            self._processor_tasks.append(task)

        if self._default_timeout is not None or self._cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())

        self.logger.info(
            f"RequestManager started with {self._concurrent_workers} processor workers."
        )
        if self._cleanup_task:
            self.logger.info("Started expired request cleanup task.")

    async def stop(self, timeout: float = 5.0):
        if not self._processor_tasks:
            self.logger.info("RequestManager is not running.")
            return

        self.logger.info("Stopping RequestManager...")
        self._shutdown_event.set()

        all_tasks = self._processor_tasks[:]
        if self._cleanup_task:
            all_tasks.append(self._cleanup_task)

        try:
            await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Graceful shutdown timed out, cancelling tasks.")
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=1.0
                )
            except asyncio.TimeoutError:
                self.logger.error("Failed to cancel all tasks.")

        for task in list(self._running_requests):
            if not task.done():
                task.cancel()

        self._processor_tasks = []
        self._cleanup_task = None
        self.logger.info("RequestManager stopped.")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "queue_size": self._queue.qsize(),
            "available_tokens": self._tokens,
            "circuit_state": self._breaker.current_state,
            "circuit_failure_count": self._breaker.fail_counter,
            "active_workers": len([t for t in self._processor_tasks if not t.done()]),
            "cleanup_task_running": self._cleanup_task
            and not self._cleanup_task.done(),
        }

    async def health_check(self) -> Dict[str, Any]:
        metrics = self.get_metrics()

        is_healthy = (
            self._breaker.current_state != "open"
            and metrics["queue_size"] < self._queue.maxsize * 0.9
            and metrics["active_workers"] > 0
        )

        return {
            "healthy": is_healthy,
            "circuit_breaker_state": self._breaker.current_state,
            "queue_utilization": (
                metrics["queue_size"] / self._queue.maxsize
                if self._queue.maxsize
                else 0
            ),
            "active_workers": metrics["active_workers"],
            "total_requests_processed": metrics["requests_processed"],
            "total_requests_failed": metrics["requests_failed"],
            "total_requests_timed_out": metrics["requests_timed_out"],
            "requests_expired_in_queue": metrics["requests_expired_in_queue"],
        }
