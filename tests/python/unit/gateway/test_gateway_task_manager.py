import asyncio
import pytest
from unittest import mock
from ataraxai.gateway.gateway_task_manager import GatewayTaskManager, TaskStatus

@pytest.mark.asyncio
async def test_create_task_and_success_result():
    manager = GatewayTaskManager()
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    task_id = manager.create_task(future)

    future.set_result("done")
    await asyncio.sleep(0)

    status = manager.get_task_status(task_id)
    assert status["task_id"] == task_id
    assert status["status"] == TaskStatus.SUCCESS
    assert status["result"] == "done"
    assert status["error"] is None

@pytest.mark.asyncio
async def test_create_task_and_failed_result():
    manager = GatewayTaskManager()
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    task_id = manager.create_task(future)

    # Simulate task failure
    future.set_exception(ValueError("fail"))
    await asyncio.sleep(0)  # Let callbacks run

    status = manager.get_task_status(task_id)
    assert status["status"] == TaskStatus.FAILED
    assert status["result"] is None
    assert "fail" in status["error"]

def test_cancel_task_success():
    manager = GatewayTaskManager()
    future = mock.Mock(spec=asyncio.Future)
    future.cancelled = mock.Mock(return_value=False)
    future.cancel = mock.Mock()
    task_id = manager.create_task(future)

    cancelled = manager.cancel_task(task_id)
    assert cancelled
    future.cancel.assert_called_once()

def test_cancel_task_not_pending():
    manager = GatewayTaskManager()
    future = mock.Mock(spec=asyncio.Future)
    task_id = manager.create_task(future)
    manager._tasks[task_id].status = TaskStatus.SUCCESS

    cancelled = manager.cancel_task(task_id)
    assert not cancelled

def test_get_task_status_none_for_unknown_id():
    manager = GatewayTaskManager()
    assert manager.get_task_status("unknown_id") is None