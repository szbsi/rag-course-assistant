"""
Tests for CourseSearchTool.execute() in backend/search_tools.py

All VectorStore interactions are replaced with MagicMock so no ChromaDB or
embedding model is needed to run these tests.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock

# conftest.py already patched sys.modules and added BACKEND_DIR to sys.path
from vector_store import SearchResults
from search_tools import CourseSearchTool


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# ─── Happy path ──────────────────────────────────────────────────────────────

def test_execute_returns_formatted_results(tool, mock_store):
    """Successful search returns formatted text with header and content."""
    mock_store.search.return_value = SearchResults(
        documents=["Lesson content here"],
        metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
        distances=[0.2],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

    result = tool.execute("what is MCP?")

    assert "[MCP Course - Lesson 1]" in result
    assert "Lesson content here" in result


def test_execute_stores_sources_after_search(tool, mock_store):
    """After a successful search, last_sources is populated."""
    mock_store.search.return_value = SearchResults(
        documents=["content"],
        metadata=[{"course_title": "RAG Course", "lesson_number": 2}],
        distances=[0.3],
    )
    mock_store.get_lesson_link.return_value = None

    tool.execute("RAG pipeline")

    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["text"] == "RAG Course - Lesson 2"


def test_execute_deduplicates_sources(tool, mock_store):
    """The same (course, lesson) combination appears only once in sources."""
    mock_store.search.return_value = SearchResults(
        documents=["chunk A", "chunk B"],
        metadata=[
            {"course_title": "Course X", "lesson_number": 3},
            {"course_title": "Course X", "lesson_number": 3},
        ],
        distances=[0.1, 0.2],
    )
    mock_store.get_lesson_link.return_value = None

    tool.execute("something")

    assert len(tool.last_sources) == 1


def test_execute_attaches_lesson_link_to_sources(tool, mock_store):
    """When a lesson link is available, it is stored in source url."""
    mock_store.search.return_value = SearchResults(
        documents=["content"],
        metadata=[{"course_title": "Course", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/l1"

    tool.execute("query")

    assert tool.last_sources[0]["url"] == "https://example.com/l1"


def test_execute_source_url_is_none_when_no_lesson_link(tool, mock_store):
    """When get_lesson_link returns None, source url is None."""
    mock_store.search.return_value = SearchResults(
        documents=["content"],
        metadata=[{"course_title": "Course", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store.get_lesson_link.return_value = None

    tool.execute("query")

    assert tool.last_sources[0]["url"] is None


# ─── Error / empty paths ──────────────────────────────────────────────────────

def test_execute_returns_error_string_on_search_error(tool, mock_store):
    """When vector_store returns an error, execute returns the error string directly."""
    mock_store.search.return_value = SearchResults.empty("Search error: connection refused")

    result = tool.execute("anything")

    # execute() at line 77: return results.error (no added prefix)
    assert "Search error" in result


def test_execute_returns_no_results_message_when_empty(tool, mock_store):
    """When search returns no documents, a 'No relevant content found' message is returned."""
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )

    result = tool.execute("unknown topic")

    assert "No relevant content found" in result


def test_execute_includes_course_name_in_no_results_message(tool, mock_store):
    """The course filter name is mentioned in the empty-results message."""
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    result = tool.execute("topic", course_name="MCP Course")

    assert "MCP Course" in result


# ─── Parameter forwarding ─────────────────────────────────────────────────────

def test_execute_passes_filters_to_vector_store(tool, mock_store):
    """course_name and lesson_number are forwarded correctly to vector_store.search()."""
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    tool.execute("query", course_name="RAG", lesson_number=5)

    mock_store.search.assert_called_once_with(
        query="query", course_name="RAG", lesson_number=5
    )
