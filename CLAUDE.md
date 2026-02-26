# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Always use `uv` to run the server and manage all dependencies — never use `pip` or `python` directly.

On Intel Mac (x86_64), Python 3.12 must be used due to dependency compatibility (torch and onnxruntime dropped Intel Mac support in newer versions).

```bash
# Start the server (from repo root)
cd backend && uv run --python 3.12 uvicorn app:app --reload --port 8000

# Or use the shell script
./run.sh
```

Web UI: `http://localhost:8000` | API docs: `http://localhost:8000/docs`

## Environment

Create a `.env` file in the repo root:
```
DEEPSEEK_API_KEY=your_key_here
```

The `.env` is loaded by `backend/config.py` via `python-dotenv` and consumed as `config.DEEPSEEK_API_KEY` and `config.DEEPSEEK_MODEL`.

## pyproject.toml Constraints

The following overrides exist due to Intel Mac (x86_64) + macOS 26 compatibility:
- `sentence-transformers==2.7.0` (not 5.x — newer versions require torch 2.7+ which dropped Intel Mac)
- `onnxruntime<1.20` (1.20+ dropped Intel Mac x86_64 wheels)
- `torch<2.3` (2.3+ dropped Intel Mac x86_64 wheels)
- `requires-python = ">=3.12"` (changed from 3.13)

## Architecture

The system is a full-stack RAG app: a FastAPI backend serves both the REST API and the static frontend.

```
frontend/          ← Static HTML/CSS/JS, served at /
backend/
  app.py           ← FastAPI entry point; mounts frontend, defines /api/query and /api/courses
  config.py        ← All settings (API key, model, chunk size, DB path, etc.)
  rag_system.py    ← Main orchestrator; composes all components below
  document_processor.py  ← Parses .txt course files, chunks text
  vector_store.py  ← ChromaDB wrapper; two collections: course_catalog + course_content
  ai_generator.py  ← DeepSeek API client (OpenAI-compatible SDK)
  search_tools.py  ← Tool definitions + ToolManager for AI function calling
  session_manager.py     ← Per-session conversation history (in-memory)
  models.py        ← Pydantic models: Course, Lesson, CourseChunk
docs/              ← Course .txt files loaded at startup
```

### Query Flow

1. Frontend POSTs `{query, session_id}` to `/api/query`
2. `app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` fetches session history, calls `AIGenerator.generate_response()` with tool definitions
4. **First DeepSeek call**: model decides to invoke `search_course_content` tool
5. `ToolManager` → `CourseSearchTool` → `VectorStore.search()` → ChromaDB semantic search (top 5 chunks)
6. **Second DeepSeek call**: model synthesizes tool results into a final answer
7. Session history updated; `{answer, sources, session_id}` returned to frontend

### Tool Calling Format

Tools use **OpenAI-compatible format** (not Anthropic format):
```python
{"type": "function", "function": {"name": ..., "parameters": {...}}}
```
Tool results are appended as `{"role": "tool", "tool_call_id": ..., "content": ...}`.

### Document Format

Course files in `docs/` must follow this structure:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content...>

Lesson 1: <title>
...
```

`DocumentProcessor` chunks each lesson's content into ~800-character segments with 100-character overlap. Chunks are stored in ChromaDB with `course_title` and `lesson_number` metadata for filtered retrieval.

### ChromaDB Collections

- **`course_catalog`**: One document per course (title, instructor, serialized lessons list) — used for fuzzy course name resolution
- **`course_content`**: One document per chunk — used for semantic search; filtered by `course_title` / `lesson_number`

Duplicate courses are skipped on startup by checking `get_existing_course_titles()` against already-stored titles.
