import pytest
from unittest import mock
from ataraxai.praxis.modules.rag.watch_directories_manager import WatchedDirectoriesManager

@pytest.fixture
def mock_rag_config_manager():
    mgr = mock.Mock()
    mgr.config.rag_watched_directories = ["dir1", "dir2"]
    return mgr

@pytest.fixture
def mock_manifest():
    manifest = mock.Mock()
    manifest.is_file_in_manifest.return_value = False
    return manifest

@pytest.fixture
def mock_processing_queue():
    queue = mock.AsyncMock()
    return queue

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.mark.asyncio
async def test_add_directories_updates_config_and_scans_files(
    mock_rag_config_manager : mock.Mock, mock_manifest: mock.Mock, mock_processing_queue: mock.AsyncMock, mock_logger: mock.Mock
):
    manager = WatchedDirectoriesManager(
        mock_rag_config_manager, mock_manifest, mock_processing_queue, mock_logger
    )
    directories_to_add = {"dir3"}
    with mock.patch("os.walk") as mock_walk, \
         mock.patch.object(manager, "_scan_and_queue_files", new=mock.AsyncMock()) as scan_mock:
        mock_walk.return_value = [("dir3", [], ["file1.txt"])]
        result = await manager.add_directories(directories_to_add)
        assert result is True
        call_args, _ = mock_rag_config_manager.set.call_args
        actual_list = call_args[1]

        expected_content = {"dir1", "dir2", "dir3"}

        assert set(actual_list) == expected_content

        assert call_args[0] == "rag_watched_directories"


@pytest.mark.asyncio
async def test_remove_directories_updates_config_and_scans_files(
    mock_rag_config_manager: mock.Mock, 
    mock_manifest: mock.Mock, 
    mock_processing_queue: mock.AsyncMock, 
    mock_logger: mock.Mock
):
    manager = WatchedDirectoriesManager(
        mock_rag_config_manager, mock_manifest, mock_processing_queue, mock_logger
    )
    directories_to_remove = {"dir1"}
    

    with mock.patch.object(manager, "_scan_and_queue_files", new=mock.MagicMock()) as scan_mock:
        
        result = await manager.remove_directories(directories_to_remove)
        
        assert result is True
        mock_rag_config_manager.set.assert_called_once()
        
        scan_mock.assert_called_with("dir1", "remove")


@pytest.mark.asyncio
async def test_scan_and_queue_files_adds_and_removes_files(mock_manifest: mock.Mock, mock_processing_queue: mock.AsyncMock, mock_logger: mock.Mock):
    rag_config_manager = mock.Mock()
    manager = WatchedDirectoriesManager(
        rag_config_manager, mock_manifest, mock_processing_queue, mock_logger
    )
    with mock.patch("os.walk") as mock_walk, \
         mock.patch("asyncio.get_running_loop") as mock_loop, \
         mock.patch("asyncio.run_coroutine_threadsafe") as mock_run_coroutine:
        mock_walk.return_value = [("/watched", [], ["a.txt", "b.txt"])]
        mock_manifest.is_file_in_manifest.side_effect = [False, True]
        await manager._scan_and_queue_files("/watched", "add")
        assert mock_manifest.add_file.call_count == 1
        assert mock_run_coroutine.call_count == 1
        mock_manifest.is_file_in_manifest.side_effect = [True, False]
        await manager._scan_and_queue_files("/watched", "remove")
        assert mock_manifest.remove_file.call_count == 1
        assert mock_run_coroutine.call_count == 2