import logging
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest

from ataraxai.praxis.modules.prompt_engine.chain_runner import ChainRunner


@pytest.fixture
def chain_runner():
    chain_task_manager = Mock()
    context_manager = Mock()
    prompt_manager = Mock()
    core_ai_service_manager = Mock()
    chat_context = Mock()
    rag_manager = Mock()
    logger = Mock(spec=logging.Logger)
    return ChainRunner(
        chain_task_manager=chain_task_manager,
        context_manager=context_manager,
        prompt_manager=prompt_manager,
        core_ai_service_manager=core_ai_service_manager,
        chat_context=chat_context,
        rag_manager=rag_manager,
        logger=logger,
    )


@pytest.mark.asyncio
async def test_run_chain_success(chain_runner : ChainRunner):
    mock_task_1 = Mock()
    mock_task_1.run = AsyncMock(return_value="result_1")
    mock_task_2 = Mock()
    mock_task_2.run = AsyncMock(return_value="result_2")

    chain_runner.chain_task_manager.get_task.side_effect = [mock_task_1, mock_task_2]

    chain_definition = [
        {"task_id": "task_1", "inputs": {"input_a": "foo"}},
        {"task_id": "task_2", "inputs": {"input_b": "{{step_0.output}}"}},
    ]

    result = await chain_runner.run_chain(chain_definition, "user_query")
    assert result == "result_2"
    assert chain_runner.chain_task_manager.get_task.call_count == 2
    mock_task_1.run.assert_awaited_once()
    mock_task_2.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_chain_error_handling(chain_runner : ChainRunner):
    mock_task_1 = Mock()
    mock_task_1.run = AsyncMock(side_effect=Exception("fail"))
    mock_task_1.handle_error = AsyncMock(return_value="error_handled")

    chain_runner.chain_task_manager.get_task.return_value = mock_task_1

    chain_definition = [
        {"task_id": "task_1", "inputs": {"input_a": "foo"}},
    ]

    result = await chain_runner.run_chain(chain_definition, "user_query")
    assert result == "error_handled"
    mock_task_1.run.assert_awaited_once()
    mock_task_1.handle_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_chain_input_reference(chain_runner : ChainRunner):
    mock_task_1 = Mock()
    mock_task_1.run = AsyncMock(return_value="output_1")
    mock_task_2 = Mock()
    mock_task_2.run = AsyncMock(return_value="output_2")

    chain_runner.chain_task_manager.get_task.side_effect = [mock_task_1, mock_task_2]

    chain_definition = [
        {"task_id": "task_1", "inputs": {"input_a": "foo"}},
        {"task_id": "task_2", "inputs": {"input_b": "{{step_0.output}}"}},
    ]

    result = await chain_runner.run_chain(chain_definition, "user_query")
    assert result == "output_2"
    mock_task_2.run.assert_awaited_with(
        {"input_b": "output_1"}, chain_runner.dependencies
    )


@pytest.mark.asyncio
async def test_run_chain_multiple_steps(chain_runner : ChainRunner):
    mock_task_1 = Mock()
    mock_task_1.run = AsyncMock(return_value="out_1")
    mock_task_2 = Mock()
    mock_task_2.run = AsyncMock(return_value="out_2")
    mock_task_3 = Mock()
    mock_task_3.run = AsyncMock(return_value="out_3")

    chain_runner.chain_task_manager.get_task.side_effect = [
        mock_task_1,
        mock_task_2,
        mock_task_3,
    ]

    chain_definition = [
        {"task_id": "task_1", "inputs": {"input_a": "foo"}},
        {"task_id": "task_2", "inputs": {"input_b": "{{step_0.output}}"}},
        {"task_id": "task_3", "inputs": {"input_c": "{{step_1.output}}"}},
    ]

    result = await chain_runner.run_chain(chain_definition, "user_query")
    assert result == "out_3"
    mock_task_1.run.assert_awaited_once()
    mock_task_2.run.assert_awaited_once()
    mock_task_3.run.assert_awaited_once()
    mock_task_2.run.assert_awaited_with({"input_b": "out_1"}, chain_runner.dependencies)
    mock_task_3.run.assert_awaited_with({"input_c": "out_2"}, chain_runner.dependencies)


@pytest.mark.asyncio
async def test_run_chain_no_inputs(chain_runner : ChainRunner):
    mock_task = Mock()
    mock_task.run = AsyncMock(return_value="no_input_result")
    chain_runner.chain_task_manager.get_task.return_value = mock_task

    chain_definition = [
        {"task_id": "task_no_input"},
    ]

    result = await chain_runner.run_chain(chain_definition, "user_query")
    assert result == "no_input_result"
    mock_task.run.assert_awaited_once_with({}, chain_runner.dependencies)


@pytest.mark.asyncio
async def test_run_chain_input_reference_missing_key(chain_runner : ChainRunner):
    mock_task_1 = Mock()
    mock_task_1.run = AsyncMock(return_value="output_1")
    mock_task_2 = Mock()
    mock_task_2.run = AsyncMock(return_value="output_2")

    chain_runner.chain_task_manager.get_task.side_effect = [mock_task_1, mock_task_2]

    chain_definition = [
        {"task_id": "task_1", "inputs": {"input_a": "foo"}},
        {"task_id": "task_2", "inputs": {"input_b": "{{step_0.nonexistent}}"}},
    ]

    with pytest.raises(KeyError):
        await chain_runner.run_chain(chain_definition, "user_query")
