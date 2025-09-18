import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.modules.rag.ataraxai_rag_manager import AtaraxAIRAGManager





@pytest.fixture
def mock_rag_config_manager():
    config = mock.Mock()
    config.rag_embedder_model = "test-embedder"
    config.rag_use_reranking = False
    config.rag_n_result = 3
    config.rag_n_result_final = 2
    config.rag_use_hyde = False
    config.rag_watched_directories = []
    config.rag_cross_encoder_model = "test-cross-encoder"
    config.model_dump.return_value = {"dummy": "config"}
    rag_config_manager = mock.Mock()
    rag_config_manager.config = config
    return rag_config_manager

@pytest.fixture
def mock_core_ai_service():
    service = mock.Mock()
    service.generate_completion.return_value = "Hypothetical answer"
    return service

@pytest.fixture
def manager(tmp_path : Path, mock_rag_config_manager : mock.Mock, mock_core_ai_service : mock.Mock):
    with mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.AtaraxAIEmbedder"), \
         mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.RAGStore"), \
         mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.RAGManifest"), \
         mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.WatchedDirectoriesManager"), \
         mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.ResilientIndexer"):
        return AtaraxAIRAGManager(
            rag_config_manager=mock_rag_config_manager,
            app_data_root_path=tmp_path,
            core_ai_service=mock_core_ai_service,
        )

@pytest.mark.asyncio
async def test_query_knowledge_simple(manager : AtaraxAIRAGManager):
    manager.rag_store.query = mock.Mock(return_value={"documents": [["doc1", "doc2", "doc3"]]})
    result = await manager.query_knowledge("test query")
    assert result == ["doc1", "doc2", "doc3"]

@pytest.mark.asyncio
async def test_query_knowledge_empty_query(manager : AtaraxAIRAGManager):
    with pytest.raises(ValueError):
        await manager.query_knowledge("")

@pytest.mark.asyncio
async def test_query_knowledge_error_logs_and_returns_empty(manager : AtaraxAIRAGManager):
    with mock.patch.object(manager, "_simple_query", side_effect=Exception("fail")):
        result = await manager.query_knowledge("test query")
        assert result == []

@pytest.mark.asyncio
async def test_simple_query_returns_documents(manager : AtaraxAIRAGManager):
    manager.rag_store.query = mock.Mock(return_value={"documents": [["docA", "docB"]]})
    docs = await manager._simple_query("query", None)
    assert docs == ["docA", "docB"]

@pytest.mark.asyncio
async def test_simple_query_returns_empty_if_no_documents(manager : AtaraxAIRAGManager):
    manager.rag_store.query = mock.Mock(return_value={"documents": [[]]})
    docs = await manager._simple_query("query", None)
    assert docs == []

@pytest.mark.asyncio
async def test_advanced_query_no_docs_returns_empty(manager : AtaraxAIRAGManager):
    manager.rag_config_manager.config.rag_use_reranking = True
    manager.rag_store.query = mock.Mock(return_value={"documents": [[]]})
    docs = await manager._advanced_query("query", None)
    assert docs == []

@pytest.mark.asyncio
async def test_advanced_query_with_reranking(manager : AtaraxAIRAGManager):
    manager.rag_config_manager.config.rag_use_reranking = True
    manager.rag_config_manager.config.rag_n_result_final = 2
    manager.rag_store.query = mock.Mock(return_value={"documents": [["d1", "d2", "d3"]]})
    with mock.patch.object(manager, "_rerank_documents", return_value=["d3", "d2", "d1"]):
        docs = await manager._advanced_query("query", None)
        assert docs == ["d3", "d2"]

@pytest.mark.asyncio
async def test_generate_hypothetical_document_success(manager : AtaraxAIRAGManager):
    result = await manager._generate_hypothetical_document("What is AI?")
    assert result == "Hypothetical answer"

@pytest.mark.asyncio
async def test_generate_hypothetical_document_error_returns_query(manager : AtaraxAIRAGManager):
    manager.core_ai_service.generate_completion.side_effect = Exception("fail")
    result = await manager._generate_hypothetical_document("What is AI?")
    assert result == "What is AI?"

@pytest.mark.asyncio
async def test_get_cross_encoder_initializes_once(manager : AtaraxAIRAGManager  ):
    manager.rag_config_manager.config.rag_use_reranking = True
    with mock.patch("ataraxai.praxis.modules.rag.ataraxai_rag_manager.CrossEncoder") as mock_ce:
        mock_ce.return_value = mock.Mock()
        ce = await manager.get_cross_encoder()
        assert ce is not None
        ce2 = await manager.get_cross_encoder()
        assert ce2 is ce

@pytest.mark.asyncio
async def test_rerank_documents_returns_sorted(manager : AtaraxAIRAGManager):
    ce_mock = mock.Mock()
    ce_mock.predict.return_value = [0.2, 0.9, 0.5]
    with mock.patch.object(manager, "get_cross_encoder", return_value=ce_mock):
        docs = await manager._rerank_documents("q", ["a", "b", "c"])
        assert docs == ["b", "c", "a"]

@pytest.mark.asyncio
async def test_add_and_remove_watch_directories(manager : AtaraxAIRAGManager):
    manager.directory_manager.add_directories = mock.AsyncMock(return_value=True)
    manager.directory_manager.remove_directories = mock.AsyncMock(return_value=True)
    assert await manager.add_watch_directories(["/tmp"]) is True
    assert await manager.remove_watch_directories(["/tmp"]) is True

@pytest.mark.asyncio
async def test_start_and_stop(manager : AtaraxAIRAGManager):
    manager.file_watcher_manager.start = mock.AsyncMock()
    manager.file_watcher_manager.stop = mock.AsyncMock()
    manager.directory_manager.add_directories = mock.AsyncMock()
    with mock.patch("asyncio.create_task") as create_task:
        create_task.return_value = mock.Mock(done=mock.Mock(return_value=True))
        await manager.start()
        await manager.stop()
