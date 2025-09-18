import asyncio
import time
from unittest import mock

import pytest
from pybreaker import CircuitBreakerError

from ataraxai.gateway.request_manager import (
    PrioritizedRequest,
    RequestManager,
    RequestPriority,
    RequestTimeoutError,
)


@pytest.fixture
def mock_logger():
    logger = mock.Mock()
    logger.get_logger.return_value = logger
    return logger


@pytest.fixture
def request_manager(mock_logger: mock.Mock):
    return RequestManager(
        rate_limit=5,
        bucket_capacity=5,
        breaker_fail_max=2,
        breaker_reset_timeout=1,
        max_queue_size=10,
        concurrent_workers=2,
        default_timeout=1.0,
        cleanup_interval=0.5,
        logger=mock_logger,
    )


@pytest.mark.asyncio
async def test_submit_and_process_request(request_manager: RequestManager):
    await request_manager.start()

    async def dummy_func(x):
        await asyncio.sleep(0.1)
        return x * 2

    future = await request_manager.submit_request(
        "test1", dummy_func, 5, priority=RequestPriority.HIGH
    )
    result = await asyncio.wait_for(future, timeout=2)
    assert result == 10

    await request_manager.stop()


@pytest.mark.asyncio
async def test_request_timeout(request_manager: RequestManager):
    await request_manager.start()

    async def slow_func():
        await asyncio.sleep(2)
        return "done"

    future = await request_manager.submit_request(
        "timeout_test", slow_func, priority=RequestPriority.MEDIUM, timeout=0.5
    )
    with pytest.raises(RequestTimeoutError):
        await asyncio.wait_for(future, timeout=2)

    await request_manager.stop()


@pytest.mark.asyncio
async def test_queue_full(request_manager: RequestManager):
    await request_manager.start()

    async def dummy_func():
        return "ok"

    for _ in range(request_manager._queue.maxsize):
        await request_manager.submit_request(
            "fill", dummy_func, priority=RequestPriority.LOW
        )

    with pytest.raises(asyncio.QueueFull):
        await request_manager.submit_request(
            "overflow", dummy_func, priority=RequestPriority.LOW
        )

    await request_manager.stop()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_rejects_after_failures():
    manager = RequestManager(breaker_fail_max=2, concurrent_workers=1)

    async def failing_task():
        raise ValueError("This task is designed to fail")

    await manager.start()

    with pytest.raises(ValueError):
        future = await manager.submit_request("failing_task_1", failing_task)
        await future

    with pytest.raises(CircuitBreakerError):
        future = await manager.submit_request("failing_task_2", failing_task)
        await future

    with pytest.raises(CircuitBreakerError):
        future = await manager.submit_request("failing_task_3", failing_task)
        await future

    await manager.stop()


@pytest.mark.asyncio
async def test_metrics_and_health_check(request_manager: RequestManager):
    await request_manager.start()

    async def dummy_func(x):
        return x

    future = await request_manager.submit_request(
        "metrics", dummy_func, 42, priority=RequestPriority.MEDIUM
    )
    await asyncio.wait_for(future, timeout=2)

    metrics = request_manager.get_metrics()
    assert metrics["requests_submitted"] >= 1
    assert metrics["requests_processed"] >= 1
    assert metrics["queue_size"] >= 0

    health = await request_manager.health_check()
    assert isinstance(health, dict)
    assert "healthy" in health

    await request_manager.stop()


@pytest.mark.asyncio
async def test_expired_request_cleanup(request_manager: RequestManager):
    await request_manager.start()

    async def dummy_func():
        await asyncio.sleep(2)
        return "ok"

    future = await request_manager.submit_request(
        "expire", dummy_func, priority=RequestPriority.LOW, timeout=0.1
    )
    await asyncio.sleep(1)

    with pytest.raises(RequestTimeoutError):
        await future

    await request_manager.stop()


@pytest.mark.asyncio
async def test_priority_ordering(request_manager: RequestManager):
    await request_manager.start()

    results = []

    async def dummy_func(x):
        await asyncio.sleep(0.05)
        results.append(x)
        return x

    futures = []
    futures.append(
        await request_manager.submit_request(
            "low", dummy_func, "low", priority=RequestPriority.LOW
        )
    )
    futures.append(
        await request_manager.submit_request(
            "high", dummy_func, "high", priority=RequestPriority.HIGH
        )
    )
    futures.append(
        await request_manager.submit_request(
            "medium", dummy_func, "medium", priority=RequestPriority.MEDIUM
        )
    )

    await asyncio.gather(*futures)

    assert results[0] == "high"
    assert results[1] == "medium"
    assert results[2] == "low"

    await request_manager.stop()


@pytest.mark.asyncio
async def test_stop_cancels_tasks(request_manager: RequestManager):
    await request_manager.start()

    async def dummy_func():
        await asyncio.sleep(2)
        return "ok"

    future = await request_manager.submit_request(
        "cancel", dummy_func, priority=RequestPriority.LOW
    )

    await asyncio.sleep(0.05)

    await request_manager.stop(timeout=0.1)

    await asyncio.sleep(0.1)

    assert future.done()
    if future.exception():
        assert isinstance(future.exception(), (asyncio.CancelledError, Exception))


def test_get_metrics_structure(request_manager: RequestManager):
    metrics = request_manager.get_metrics()
    assert isinstance(metrics, dict)
    assert "queue_size" in metrics
    assert "available_tokens" in metrics
    assert "circuit_state" in metrics
    assert "active_workers" in metrics
    assert "cleanup_task_running" in metrics


def test_prioritized_request_expiry_and_remaining_time():
    pr = PrioritizedRequest(
        priority=1,
        submission_time=time.monotonic(),
        future=mock.Mock(),
        func=lambda: None,
        timeout=0.1,
    )
    assert pr.remaining_time() is not None
    time.sleep(0.2)
    assert pr.is_expired()


@pytest.mark.asyncio
async def test_submit_request_sets_future_and_priority(request_manager: RequestManager):
    async def dummy_func(x):
        return x

    future = await request_manager.submit_request(
        "priority_test", dummy_func, 123, priority=RequestPriority.HIGH
    )
    assert isinstance(future, asyncio.Future)
    assert not future.done()
    await request_manager.stop()


def test_prioritized_request_no_timeout():
    pr = PrioritizedRequest(
        priority=2, submission_time=time.monotonic(), future=mock.Mock(), func=lambda: 1
    )
    assert pr.is_expired() is False
    assert pr.remaining_time() is None


def test_request_timeout_error_message():
    err = RequestTimeoutError(2.5)
    assert "timed out after 2.5 seconds" in str(err)
    err2 = RequestTimeoutError(1.0, "Custom message")
    assert str(err2) == "Custom message"


@pytest.mark.asyncio
async def test_stop_when_not_running(request_manager: RequestManager):
    await request_manager.stop()


@pytest.mark.asyncio
async def test_start_when_already_running(request_manager: RequestManager):
    await request_manager.start()
    await request_manager.start()
    await request_manager.stop()


def test_metrics_queue_and_tokens(request_manager: RequestManager):
    metrics = request_manager.get_metrics()
    assert isinstance(metrics["queue_size"], int)
    assert isinstance(metrics["available_tokens"], float)
    assert metrics["circuit_state"] in ("closed", "open", "half-open")
