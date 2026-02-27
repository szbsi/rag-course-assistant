"""
Tests for RAGSystem.query() in backend/rag_system.py.

All heavy components (AIGenerator, VectorStore, SessionManager, DocumentProcessor)
are replaced with MagicMock so only the orchestration logic is tested.

Exposes:
  - Bug B3: generate_response exception propagates uncaught → 500 "query failed"
"""
import pytest
from unittest.mock import MagicMock, patch

# conftest.py already patched sys.modules and added BACKEND_DIR to sys.path
from rag_system import RAGSystem


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "./test_chroma"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.DEEPSEEK_API_KEY = "test-key"
    cfg.DEEPSEEK_MODEL = "deepseek-chat"
    return cfg


@pytest.fixture
def rag(mock_config):
    """RAGSystem with all heavy components replaced by MagicMock."""
    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator") as mock_ai_cls, \
         patch("rag_system.SessionManager") as mock_sm_cls:
        system = RAGSystem(mock_config)
        # Replace auto-created instances with fresh mocks for easy configuration
        system.ai_generator = mock_ai_cls.return_value
        system.session_manager = mock_sm_cls.return_value
        system.tool_manager = MagicMock()
        return system


# ─── Normal query flow ───────────────────────────────────────────────────────

def test_query_returns_response_string_and_sources_list(rag):
    """A successful query returns (str, list) with the AI's answer and sources."""
    rag.session_manager.get_conversation_history.return_value = None
    rag.ai_generator.generate_response.return_value = "Here is the answer."
    rag.tool_manager.get_last_sources.return_value = [{"text": "MCP - Lesson 1", "url": None}]

    response, sources = rag.query("What is MCP?", session_id="sess1")

    assert response == "Here is the answer."
    assert len(sources) == 1


def test_query_skips_session_when_session_id_is_none(rag):
    """With session_id=None, session history is neither read nor written."""
    rag.ai_generator.generate_response.return_value = "answer"
    rag.tool_manager.get_last_sources.return_value = []

    rag.query("test query", session_id=None)

    rag.session_manager.get_conversation_history.assert_not_called()
    rag.session_manager.add_exchange.assert_not_called()


def test_query_passes_conversation_history_to_generator(rag):
    """When a session exists, its history is forwarded to generate_response."""
    rag.session_manager.get_conversation_history.return_value = (
        "user: hi\nassistant: hello"
    )
    rag.ai_generator.generate_response.return_value = "answer"
    rag.tool_manager.get_last_sources.return_value = []

    rag.query("follow up question", session_id="sess1")

    call_kwargs = rag.ai_generator.generate_response.call_args[1]
    assert "hi" in call_kwargs.get("conversation_history", "")


def test_query_resets_sources_after_retrieval(rag):
    """tool_manager.reset_sources() is called after every query."""
    rag.ai_generator.generate_response.return_value = "answer"
    rag.tool_manager.get_last_sources.return_value = []

    rag.query("test", session_id="sess1")

    rag.tool_manager.reset_sources.assert_called_once()


def test_query_updates_session_history_with_exchange(rag):
    """After a query the Q&A pair is saved to the session."""
    rag.session_manager.get_conversation_history.return_value = None
    rag.ai_generator.generate_response.return_value = "The answer is 42"
    rag.tool_manager.get_last_sources.return_value = []

    rag.query("What is the answer?", session_id="sess1")

    rag.session_manager.add_exchange.assert_called_once_with(
        "sess1", "What is the answer?", "The answer is 42"
    )


# ─── Bug B3: uncaught exception from generate_response ───────────────────────

def test_query_returns_error_message_when_generator_raises(rag):
    """
    BUG B3 (fixed): generate_response exception is now caught inside query().
    query() returns a graceful error tuple instead of propagating the exception.
    """
    rag.session_manager.get_conversation_history.return_value = None
    rag.ai_generator.generate_response.side_effect = Exception("API timeout")

    # After fix: no exception raised; an error message tuple is returned
    response, sources = rag.query("test question", session_id="sess1")
    assert "error" in response.lower()
    assert sources == []
