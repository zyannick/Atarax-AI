from typing import Any, Any, Dict, List
from unittest.mock import Mock, AsyncMock

import pytest
from ataraxai.praxis.modules.prompt_engine.chain_runner import ChainRunner

def test_chain_runner_init_sets_attributes():
    mock_task_manager = Mock()
    mock_context_manager = Mock()
    mock_prompt_manager = Mock()
    mock_core_ai_service = Mock()
    mock_chat_context = Mock()
    mock_rag_manager = Mock()

    runner = ChainRunner(
        task_manager=mock_task_manager,
        context_manager=mock_context_manager,
        prompt_manager=mock_prompt_manager,
        core_ai_service_manager=mock_core_ai_service,
        chat_context=mock_chat_context,
        rag_manager=mock_rag_manager
    )

    assert runner.task_manager is mock_task_manager
    assert runner.context_manager is mock_context_manager
    assert runner.prompt_manager is mock_prompt_manager
    assert runner.core_ai_service_manager is mock_core_ai_service
    assert runner.chat_context is mock_chat_context
    assert runner.rag_manager is mock_rag_manager

def test_chain_runner_init_sets_dependencies_dict():
    mock_task_manager = Mock()
    mock_context_manager = Mock()
    mock_prompt_manager = Mock()
    mock_core_ai_service = Mock()
    mock_chat_context = Mock()
    mock_rag_manager = Mock()

    runner = ChainRunner(
        task_manager=mock_task_manager,
        context_manager=mock_context_manager,
        prompt_manager=mock_prompt_manager,
        core_ai_service_manager=mock_core_ai_service,
        chat_context=mock_chat_context,
        rag_manager=mock_rag_manager
    )

    expected_dependencies = {
        "context_manager": mock_context_manager,
        "prompt_manager": mock_prompt_manager,
        "core_ai_service_manager": mock_core_ai_service,
        "chat_context": mock_chat_context,
        "rag_manager": mock_rag_manager
    }
    assert runner.dependencies == expected_dependencies


@pytest.mark.asyncio
async def test_run_chain_single_step_success():
    mock_task_manager = Mock()
    mock_context_manager = Mock()
    mock_prompt_manager = Mock()
    mock_core_ai_service = Mock()
    mock_chat_context = Mock()
    mock_rag_manager = Mock()

    mock_task = Mock()
    mock_task.run = AsyncMock(return_value="step_result")
    mock_task.handle_error = AsyncMock()

    mock_task_manager.get_task.return_value = mock_task

    runner = ChainRunner(
        task_manager=mock_task_manager,
        context_manager=mock_context_manager,
        prompt_manager=mock_prompt_manager,
        core_ai_service_manager=mock_core_ai_service,
        chat_context=mock_chat_context,
        rag_manager=mock_rag_manager
    )

    chain_definition: List[Dict[str, Any]] = [
        {
            "task_id": "task1",
            "inputs": {"input1": "value1"}
        }
    ]

    result = await runner.run_chain(chain_definition, "user query")

    mock_task.run.assert_awaited_once() 

    assert result == "step_result", f"Expected 'step_result', got {result}"

@pytest.mark.asyncio
async def test_run_chain_multiple_steps_with_reference():
    mock_task_manager = Mock()
    mock_context_manager = Mock()
    mock_prompt_manager = Mock()
    mock_core_ai_service = Mock()
    mock_chat_context = Mock()
    mock_rag_manager = Mock()

    mock_task1 = AsyncMock()
    mock_task1.run = AsyncMock(return_value="result1")
    mock_task1.handle_error = AsyncMock()
    mock_task2 = AsyncMock()
    mock_task2.run = AsyncMock(return_value="result2")
    mock_task2.handle_error = AsyncMock()

    def get_task_side_effect(task_id):
        if task_id == "task1":
            return mock_task1
        elif task_id == "task2":
            return mock_task2
        else:
            raise ValueError("Unknown task_id")

    mock_task_manager.get_task.side_effect = get_task_side_effect

    runner = ChainRunner(
        task_manager=mock_task_manager,
        context_manager=mock_context_manager,
        prompt_manager=mock_prompt_manager,
        core_ai_service_manager=mock_core_ai_service,
        chat_context=mock_chat_context,
        rag_manager=mock_rag_manager
    )

    chain_definition = [
        {
            "task_id": "task1",
            "inputs": {"input1": "value1"}
        },
        {
            "task_id": "task2",
            "inputs": {"input2": "{{step_0.output}}"}
        }
    ]
    result = await runner.run_chain(chain_definition, "user query")
    assert result == "result2"
    assert mock_task1.run.call_count == 1
    assert mock_task2.run.call_count == 1
    mock_task1.handle_error.assert_not_called()
    mock_task2.handle_error.assert_not_called()
    args, kwargs = mock_task2.run.call_args
    assert args[0]["input2"] == "result1"

@pytest.mark.asyncio
async def test_run_chain_handles_exception_and_calls_handle_error():
    mock_task_manager = Mock()
    mock_context_manager = Mock()
    mock_prompt_manager = Mock()
    mock_core_ai_service = Mock()
    mock_chat_context = Mock()
    mock_rag_manager = Mock()

    mock_task = AsyncMock()
    mock_task.run.side_effect = Exception("fail!")
    mock_task.handle_error = AsyncMock(return_value="handled_error")

    mock_task_manager.get_task.return_value = mock_task

    runner = ChainRunner(
        task_manager=mock_task_manager,
        context_manager=mock_context_manager,
        prompt_manager=mock_prompt_manager,
        core_ai_service_manager=mock_core_ai_service,
        chat_context=mock_chat_context,
        rag_manager=mock_rag_manager
    )

    chain_definition = [
        {
            "task_id": "task1",
            "inputs": {"input1": "value1"}
        }
    ]
    result = await runner.run_chain(chain_definition, "user query")
    assert result == "handled_error"
    mock_task.run.assert_called_once()
    mock_task.handle_error.assert_called_once()
