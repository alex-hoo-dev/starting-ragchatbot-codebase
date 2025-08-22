# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- `./run.sh` - Quick start script that starts the backend server
- `cd backend && uv run uvicorn app:app --reload --port 8000` - Manual server start with hot reload
- Backend serves at `http://localhost:8000` with web interface
- API docs available at `http://localhost:8000/docs`

### Dependencies
- `uv sync` - Install Python dependencies using uv package manager
- Python 3.13+ required
- Environment variables: Set `ANTHROPIC_API_KEY` in `.env` file

### Code Quality Tools
- `./scripts/format.sh` - Format code with black and sort imports with isort
- `./scripts/lint.sh` - Run linting checks (flake8, black --check, isort --check)
- `./scripts/typecheck.sh` - Run type checking with mypy
- `./scripts/quality.sh` - Run full quality pipeline (format, lint, typecheck, tests)
- `uv run black .` - Format code with black
- `uv run isort .` - Sort imports with isort
- `uv run flake8 backend/ main.py` - Run flake8 linting
- `uv run mypy backend/ main.py` - Run type checking

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for querying course materials. The architecture follows a full-stack pattern with clear separation of concerns:

### Backend Components (`backend/`)

**Core RAG Pipeline:**
- `rag_system.py` - Main orchestrator that coordinates all components
- `vector_store.py` - ChromaDB integration for semantic search with dual collections:
  - `course_catalog` - Course metadata (titles, instructors)  
  - `course_content` - Actual course content chunks
- `ai_generator.py` - Anthropic Claude integration with tool-based search
- `search_tools.py` - Tool-based search system that AI uses to query knowledge base

**Data Processing:**
- `document_processor.py` - Converts course documents into structured chunks
- `models.py` - Pydantic models (Course, Lesson, CourseChunk)
- `session_manager.py` - Conversation history management

**Web Layer:**
- `app.py` - FastAPI application with `/api/query` and `/api/courses` endpoints
- `config.py` - Configuration with environment variable loading

### Frontend (`frontend/`)
- Simple HTML/CSS/JS interface for chat interaction
- Queries backend API endpoints

### Key Design Patterns

**Tool-Based Architecture**: The AI doesn't directly access the vector store. Instead, it uses a `CourseSearchTool` that provides structured search capabilities, allowing for more controlled and reliable information retrieval.

**Dual Vector Collections**: Separates course metadata from content for more efficient search patterns - catalog search for course resolution, content search for actual material.

**Session Management**: Maintains conversation context with configurable history limits for better multi-turn interactions.

**Chunking Strategy**: Documents are processed into 800-character chunks with 100-character overlap for optimal semantic search performance.

## Data Flow

1. Documents in `docs/` are processed on startup
2. Course metadata → `course_catalog` collection  
3. Course content → `course_content` collection (chunked)
4. User queries trigger tool-based search through AI
5. AI synthesizes search results into conversational responses

## Key Configuration

- ChromaDB storage: `./chroma_db`
- Embedding model: `all-MiniLM-L6-v2`
- AI model: `claude-sonnet-4-20250514`
- Default chunk size: 800 chars, 100 overlap
- Max search results: 5
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files