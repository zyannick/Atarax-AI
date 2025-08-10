import asyncio
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Coroutine, Optional, Dict
from pybreaker import CircuitBreaker, CircuitBreakerError

from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
logger = AtaraxAILogger("RequestManager").get_logger()


class RequestPriority(IntEnum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    submission_time: float = field(compare=False)
    future: asyncio.Future = field(compare=False)
    item: Coroutine = field(compare=False)


class RequestManager:
    
    def __init__(
        self,
        rate_limit: int = 10,      
        bucket_capacity: int = 20, 
        breaker_fail_max: int = 5,  
        breaker_reset_timeout: int = 60, 
        max_queue_size: int = 1000, 
        concurrent_workers: int = 5  
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

        Attributes:
            _queue (asyncio.PriorityQueue): Queue for managing incoming requests.
            _processor_tasks (list): List of worker task references.
            _shutdown_event (asyncio.Event): Event to signal shutdown.
            _concurrent_workers (int): Number of concurrent worker tasks.
            _rate_limit (int): Rate limit for requests per second.
            _bucket_capacity (float): Burst capacity for the token bucket.
            _tokens (float): Current number of tokens in the bucket.
            _last_refill_time (float): Last time the token bucket was refilled.
            _rate_lock (asyncio.Lock): Lock for synchronizing rate limiter access.
            _breaker (CircuitBreaker): Circuit breaker instance for handling failures.
            _metrics (dict): Dictionary for tracking request and circuit breaker metrics.
        """
        self._queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._processor_tasks = []
        self._shutdown_event = asyncio.Event()
        self._concurrent_workers = concurrent_workers

        self._rate_limit = rate_limit
        self._bucket_capacity = float(bucket_capacity)
        self._tokens = self._bucket_capacity
        self._last_refill_time = time.monotonic()
        self._rate_lock = asyncio.Lock()

        self._breaker = CircuitBreaker(
            fail_max=breaker_fail_max, 
            reset_timeout=breaker_reset_timeout
        )

        self._metrics = {
            'requests_submitted': 0,
            'requests_processed': 0,
            'requests_failed': 0,
            'requests_rejected': 0,
            'circuit_open_count': 0
        }

        logger.info(f"RequestManager initialized with rate limit: {rate_limit}/s, burst capacity: {bucket_capacity}")
        logger.info(f"Circuit Breaker initialized with fail_max={breaker_fail_max}, reset_timeout={breaker_reset_timeout}s")
        logger.info(f"Queue max size: {max_queue_size}, concurrent workers: {concurrent_workers}")

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
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.01)
                return False  
            except asyncio.TimeoutError:
                continue  
        
        return False

    async def submit_request(
        self,
        coro: Coroutine,
        priority: RequestPriority = RequestPriority.MEDIUM
    ) -> Any:
        if self._breaker.current_state == 'open':
            self._metrics['requests_rejected'] += 1
            raise CircuitBreakerError("Circuit breaker is open")

        future = asyncio.Future()
        
        request = PrioritizedRequest(
            priority=priority.value,
            submission_time=time.monotonic(),
            future=future,
            item=coro
        )
        
        try:
            self._queue.put_nowait(request)
            self._metrics['requests_submitted'] += 1
            logger.info(f"Submitted request with priority {priority.name}. Queue size: {self._queue.qsize()}")
        except asyncio.QueueFull:
            self._metrics['requests_rejected'] += 1
            logger.warning(f"Queue is full! Rejecting request with priority {priority.name}")
            raise
        
        return await future

    async def _execute_request(self, request: PrioritizedRequest):
        try:
            result = await self._breaker.call_async(request.item) # type: ignore
            request.future.set_result(result)
            self._metrics['requests_processed'] += 1
            logger.debug(f"Request with priority {request.priority} completed successfully")
            
        except CircuitBreakerError as e:
            logger.error(f"Circuit is open! Rejecting request. Error: {e}")
            request.future.set_exception(e)
            self._metrics['requests_rejected'] += 1
            self._metrics['circuit_open_count'] += 1
            
        except Exception as e:
            logger.error(f"Request failed. Error: {e}")
            request.future.set_exception(e)
            self._metrics['requests_failed'] += 1

    async def _process_requests(self, worker_id: int):
        logger.info(f"Request processor worker {worker_id} started.")
        
        while not self._shutdown_event.is_set():
            try:
                try:
                    request: PrioritizedRequest = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue  
                
                logger.debug(f"Worker {worker_id} processing request with priority {request.priority}. Queue size: {self._queue.qsize()}")

                if not await self._wait_for_token():
                    try:
                        self._queue.put_nowait(request)
                    except asyncio.QueueFull:
                        request.future.set_exception(Exception("Service shutting down"))
                    break
                
                await self._execute_request(request)
                self._queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Request processor worker {worker_id} shutting down.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_id}: {e}", exc_info=True)

        logger.info(f"Request processor worker {worker_id} stopped.")

    async def start(self):
        if self._processor_tasks:
            logger.warning("RequestManager is already running.")
            return
        
        self._shutdown_event.clear()
        self._processor_tasks = []
        
        for i in range(self._concurrent_workers):
            task = asyncio.create_task(self._process_requests(i))
            self._processor_tasks.append(task)
        
        logger.info(f"RequestManager started with {self._concurrent_workers} processor workers.")

    async def stop(self, timeout: float = 5.0):
        if not self._processor_tasks:
            logger.info("RequestManager is not running.")
            return

        logger.info("Stopping RequestManager...")
        
        self._shutdown_event.set()
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._processor_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Graceful shutdown timed out, cancelling tasks.")
            for task in self._processor_tasks:
                if not task.done():
                    task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._processor_tasks, return_exceptions=True),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.error("Failed to cancel all processor tasks.")

        self._processor_tasks = []
        logger.info("RequestManager stopped.")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            'queue_size': self._queue.qsize(),
            'available_tokens': self._tokens,
            'circuit_state': self._breaker.current_state,
            'circuit_failure_count': self._breaker.fail_counter,
            'active_workers': len([t for t in self._processor_tasks if not t.done()])
        }

    async def health_check(self) -> Dict[str, Any]:
        metrics = self.get_metrics()
        
        is_healthy = (
            self._breaker.current_state != 'open' and
            metrics['queue_size'] < self._queue.maxsize * 0.9 and 
            metrics['active_workers'] > 0  
        )
        
        return {
            'healthy': is_healthy,
            'circuit_breaker_state': self._breaker.current_state,
            'queue_utilization': metrics['queue_size'] / self._queue.maxsize if self._queue.maxsize else 0,
            'active_workers': metrics['active_workers'],
            'total_requests_processed': metrics['requests_processed'],
            'total_requests_failed': metrics['requests_failed']
        }
