"""
Tests for AIGenerator in backend/ai_generator.py.

Verifies that the generator correctly:
  - Returns direct text when no tool call is needed
  - Calls the search tool when finish_reason == "tool_calls"
  - Sends tool results back to the API for synthesis
  - Supports up to MAX_TOOL_ROUNDS sequential tool calls
  - Stops early when the model returns stop mid-loop
  - Falls back gracefully when tool execution fails
  - Handles the DeepSeek DSML fallback format
  - Exposes Bug B2: uncaught exception from execute_tool
  - Exposes Bug B5: None content returned from API
"""
import json
import pytest
from unittest.mock import MagicMock, patch

# conftest.py already patched sys.modules and added BACKEND_DIR to sys.path
from ai_generator import AIGenerator


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_response(content=None, finish_reason="stop", tool_calls=None):
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_tool_call(name, arguments_dict):
    """Build a mock ToolCall object."""
    tc = MagicMock()
    tc.id = "call_test_001"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


# ─── Fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def generator():
    """AIGenerator with its OpenAI client replaced by a MagicMock."""
    with patch("ai_generator.OpenAI"):
        gen = AIGenerator(api_key="test-key", model="deepseek-chat")
        gen.client = MagicMock()
        return gen


# ─── Direct response (no tool) ───────────────────────────────────────────────

def test_generate_response_returns_direct_text_when_no_tool_call(generator):
    """finish_reason=stop → content is returned as-is."""
    generator.client.chat.completions.create.return_value = make_response(
        content="Here is the answer.", finish_reason="stop"
    )

    result = generator.generate_response("What is RAG?")

    assert result == "Here is the answer."


# ─── Tool calling (standard OpenAI format) ───────────────────────────────────

def test_generate_response_calls_tool_when_finish_reason_is_tool_calls(generator):
    """finish_reason=tool_calls → tool_manager.execute_tool is called once."""
    tc = make_tool_call("search_course_content", {"query": "RAG pipeline"})
    first_resp = make_response(finish_reason="tool_calls", tool_calls=[tc])
    # Loop: after tool execution, model returns stop (no second tool needed)
    intermediate_resp = make_response(finish_reason="stop")
    synthesis_resp = make_response(content="Final answer", finish_reason="stop")
    generator.client.chat.completions.create.side_effect = [
        first_resp, intermediate_resp, synthesis_resp
    ]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "Relevant RAG content"

    result = generator.generate_response("What is RAG?", tool_manager=mock_tm)

    mock_tm.execute_tool.assert_called_once_with(
        "search_course_content", query="RAG pipeline"
    )
    assert result == "Final answer"


def test_handle_tool_execution_sends_tool_result_back_to_api(generator):
    """Tool result is included as a 'tool' role message in the intermediate API call."""
    tc = make_tool_call("search_course_content", {"query": "MCP"})
    first_resp = make_response(finish_reason="tool_calls", tool_calls=[tc])
    # Intermediate call (index 1) sees tool results; model returns stop
    intermediate_resp = make_response(finish_reason="stop")
    synthesis_resp = make_response(content="Answer using MCP content", finish_reason="stop")
    generator.client.chat.completions.create.side_effect = [
        first_resp, intermediate_resp, synthesis_resp
    ]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "[MCP Course] MCP content..."

    generator.generate_response("Explain MCP", tool_manager=mock_tm)

    # call index 1 is the intermediate loop call; messages include the tool result
    intermediate_call_msgs = generator.client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_messages = [m for m in intermediate_call_msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "[MCP Course]" in tool_messages[0]["content"]


# ─── Bug B2: uncaught exception from execute_tool ────────────────────────────

def test_generate_response_tool_manager_exception_is_handled(generator):
    """
    BUG B2 (fixed): execute_tool exception is now caught and returned as a tool
    result string so the AI can synthesise a graceful response instead of crashing.
    """
    tc = make_tool_call("search_course_content", {"query": "test"})
    first_resp = make_response(finish_reason="tool_calls", tool_calls=[tc])
    second_resp = make_response(content="I could not retrieve results.", finish_reason="stop")
    generator.client.chat.completions.create.side_effect = [first_resp, second_resp]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = RuntimeError("ChromaDB connection failed")

    # After fix: no exception raised; a response string is returned
    result = generator.generate_response("test query", tool_manager=mock_tm)
    assert result is not None
    assert isinstance(result, str)


# ─── Bug B5: None content from API ───────────────────────────────────────────

def test_generate_response_returns_non_none_when_api_content_is_none(generator):
    """
    BUG B5: API returns None content → generate_response returns None →
    Pydantic's `answer: str` field rejects it → 500 error.

    This test FAILS (confirming the bug) when generate_response returns None.
    After the fix (`return content or ""`), it should PASS.
    """
    generator.client.chat.completions.create.return_value = make_response(
        content=None, finish_reason="stop"
    )

    result = generator.generate_response("test")

    # Expected after fix: not None. Currently FAILS because result is None.
    assert result is not None


# ─── DSML fallback format ────────────────────────────────────────────────────

def test_generate_response_dsml_fallback_triggers_tool_call(generator):
    """When finish_reason=stop but content contains DSML markup, the tool is still called."""
    dsml_content = (
        "<\uff5cDSML\uff5cfunction_calls>"
        "<\uff5cDSML\uff5cinvoke name=\"search_course_content\">"
        "<\uff5cDSML\uff5cparameter name=\"query\">what is RAG</\uff5cDSML\uff5cparameter>"
        "</\uff5cDSML\uff5cinvoke>"
        "</\uff5cDSML\uff5cfunction_calls>"
    )
    first_resp = make_response(content=dsml_content, finish_reason="stop")
    second_resp = make_response(content="RAG answer", finish_reason="stop")
    generator.client.chat.completions.create.side_effect = [first_resp, second_resp]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "RAG search results"

    result = generator.generate_response("what is RAG?", tool_manager=mock_tm)

    mock_tm.execute_tool.assert_called_once_with(
        "search_course_content", query="what is RAG"
    )
    assert result == "RAG answer"


def test_parse_dsml_tool_call_extracts_tool_name_and_params(generator):
    """_parse_dsml_tool_call correctly extracts the tool name and typed parameters."""
    dsml = (
        "<\uff5cDSML\uff5cinvoke name=\"search_course_content\">"
        "<\uff5cDSML\uff5cparameter name=\"query\">lesson 5</\uff5cDSML\uff5cparameter>"
        "<\uff5cDSML\uff5cparameter name=\"lesson_number\">5</\uff5cDSML\uff5cparameter>"
        "</\uff5cDSML\uff5cinvoke>"
    )

    result = generator._parse_dsml_tool_call(dsml)

    assert result is not None
    assert result["tool_name"] == "search_course_content"
    assert result["params"]["query"] == "lesson 5"
    assert result["params"]["lesson_number"] == 5  # numeric string → int


# ─── Sequential tool calling (multi-round) ───────────────────────────────────

def test_two_round_tool_calling_executes_both_tools(generator):
    """Two sequential tool calls: both tools execute and results are synthesized."""
    tc1 = make_tool_call("search_course_content", {"query": "RAG pipeline"})
    tc2 = make_tool_call("search_course_content", {"query": "vector stores"})

    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc1])  # initial
    resp2 = make_response(finish_reason="tool_calls", tool_calls=[tc2])  # loop intermediate: wants more
    resp3 = make_response(finish_reason="stop")                          # loop intermediate: done
    resp4 = make_response(content="Combined answer", finish_reason="stop")  # synthesis

    generator.client.chat.completions.create.side_effect = [resp1, resp2, resp3, resp4]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = ["RAG results", "Vector store results"]

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    result = generator.generate_response(
        "Explain RAG and vector stores", tools=tools, tool_manager=mock_tm
    )

    assert mock_tm.execute_tool.call_count == 2
    assert generator.client.chat.completions.create.call_count == 4
    assert result == "Combined answer"


def test_second_round_messages_include_both_tool_results(generator):
    """The final synthesis call's messages contain tool results from both rounds."""
    tc1 = make_tool_call("search_course_content", {"query": "first topic"})
    tc2 = make_tool_call("search_course_content", {"query": "second topic"})

    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc1])
    resp2 = make_response(finish_reason="tool_calls", tool_calls=[tc2])
    resp3 = make_response(finish_reason="stop")
    resp4 = make_response(content="Final", finish_reason="stop")

    generator.client.chat.completions.create.side_effect = [resp1, resp2, resp3, resp4]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = ["[Course A] result", "[Course B] result"]

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    generator.generate_response("query", tools=tools, tool_manager=mock_tm)

    # Synthesis call is at index 3; its messages must contain both tool results
    synthesis_msgs = generator.client.chat.completions.create.call_args_list[3][1]["messages"]
    tool_messages = [m for m in synthesis_msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "[Course A]" in tool_messages[0]["content"]
    assert "[Course B]" in tool_messages[1]["content"]


def test_tool_loop_stops_when_model_returns_stop_after_round1(generator):
    """If model returns stop after the first tool result, no second tool call is made."""
    tc = make_tool_call("search_course_content", {"query": "RAG"})
    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc])
    resp2 = make_response(finish_reason="stop")                              # model done
    resp3 = make_response(content="One-search answer", finish_reason="stop") # synthesis

    generator.client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "Search results"

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    result = generator.generate_response("What is RAG?", tools=tools, tool_manager=mock_tm)

    assert mock_tm.execute_tool.call_count == 1
    assert generator.client.chat.completions.create.call_count == 3
    assert result == "One-search answer"


def test_tool_loop_respects_max_rounds_limit(generator):
    """Even if the model keeps requesting tools, execution stops at MAX_TOOL_ROUNDS."""
    tc = make_tool_call("search_course_content", {"query": "test"})

    # Model always wants more tool calls; the loop must enforce the limit
    tool_resp = make_response(finish_reason="tool_calls", tool_calls=[tc])
    synthesis_resp = make_response(content="Synthesized", finish_reason="stop")

    # initial + 2 intermediate (both tool_calls) + 1 synthesis = 4 calls
    generator.client.chat.completions.create.side_effect = [
        tool_resp, tool_resp, tool_resp, synthesis_resp
    ]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "result"

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    generator.generate_response("test", tools=tools, tool_manager=mock_tm)

    assert mock_tm.execute_tool.call_count == AIGenerator.MAX_TOOL_ROUNDS


def test_first_round_failure_falls_back_to_synthesis(generator):
    """If the tool call in round 1 fails, the loop exits and synthesis runs directly."""
    tc = make_tool_call("search_course_content", {"query": "test"})
    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc])
    resp_synthesis = make_response(content="Sorry, search failed.", finish_reason="stop")

    generator.client.chat.completions.create.side_effect = [resp1, resp_synthesis]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = RuntimeError("ChromaDB down")

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    result = generator.generate_response("test query", tools=tools, tool_manager=mock_tm)

    # No intermediate call (failure → break immediately); initial + synthesis = 2 calls
    assert generator.client.chat.completions.create.call_count == 2
    assert isinstance(result, str)


def test_second_round_failure_uses_first_round_result(generator):
    """Round 1 succeeds; round 2 fails. Synthesis proceeds using round 1's result."""
    tc1 = make_tool_call("search_course_content", {"query": "first"})
    tc2 = make_tool_call("search_course_content", {"query": "second"})

    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc1])
    resp2 = make_response(finish_reason="tool_calls", tool_calls=[tc2])  # intermediate
    resp_synthesis = make_response(content="Partial answer", finish_reason="stop")

    generator.client.chat.completions.create.side_effect = [resp1, resp2, resp_synthesis]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = [
        "Good results from round 1",
        RuntimeError("DB unavailable"),
    ]

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    result = generator.generate_response("query", tools=tools, tool_manager=mock_tm)

    assert mock_tm.execute_tool.call_count == 2
    assert generator.client.chat.completions.create.call_count == 3
    assert isinstance(result, str)


def test_execute_tool_calls_returns_true_when_tool_succeeds(generator):
    """_execute_tool_calls returns True when at least one tool call succeeds."""
    tc = make_tool_call("search_course_content", {"query": "test"})
    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [tc]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "Success result"

    messages = []
    result = generator._execute_tool_calls(assistant_msg, messages, mock_tm)

    assert result is True
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"


def test_execute_tool_calls_returns_false_when_all_fail(generator):
    """_execute_tool_calls returns False when every tool call raises an exception."""
    tc = make_tool_call("search_course_content", {"query": "test"})
    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [tc]

    mock_tm = MagicMock()
    mock_tm.execute_tool.side_effect = RuntimeError("Failure")

    messages = []
    result = generator._execute_tool_calls(assistant_msg, messages, mock_tm)

    assert result is False
    assert len(messages) == 1
    assert "Tool execution failed" in messages[0]["content"]


def test_intermediate_api_call_includes_tools_but_synthesis_does_not(generator):
    """Intermediate loop calls include tools; the final synthesis call does not."""
    tc = make_tool_call("search_course_content", {"query": "RAG"})
    resp1 = make_response(finish_reason="tool_calls", tool_calls=[tc])
    resp2 = make_response(finish_reason="stop")
    resp3 = make_response(content="Final", finish_reason="stop")

    generator.client.chat.completions.create.side_effect = [resp1, resp2, resp3]

    mock_tm = MagicMock()
    mock_tm.execute_tool.return_value = "results"

    tools = [{"type": "function", "function": {"name": "search_course_content"}}]
    generator.generate_response("query", tools=tools, tool_manager=mock_tm)

    all_calls = generator.client.chat.completions.create.call_args_list
    assert "tools" in all_calls[1][1]      # intermediate call has tools
    assert "tools" not in all_calls[2][1]  # synthesis call does not
