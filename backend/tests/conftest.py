"""
Pytest configuration and fixtures for the RAG chatbot tests.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_vector_store():
    """Fixture providing mock vector store"""
    mock_store = Mock()
    mock_store.search.return_value = Mock(
        documents=[], metadata=[], distances=[], error=None, is_empty=lambda: True
    )
    mock_store.get_lesson_link.return_value = None
    mock_store.get_existing_course_titles.return_value = []
    mock_store.get_course_count.return_value = 0
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing mock Anthropic client"""
    client = Mock()
    # Default response
    mock_response = Mock()
    mock_response.content = [Mock(type="text", text="Mock response")]
    mock_response.stop_reason = "stop"
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def sample_course_data():
    """Fixture providing sample course data for testing"""
    from models import Course, CourseChunk, Lesson

    course = Course(
        title="Python Programming Basics",
        course_link="https://example.com/python-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction to Python",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Data Types",
                lesson_link="https://example.com/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Control Structures",
                lesson_link="https://example.com/lesson3",
            ),
        ],
    )

    chunks = [
        CourseChunk(
            content="Python is a high-level programming language.",
            course_title="Python Programming Basics",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python can store different types of data.",
            course_title="Python Programming Basics",
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Control structures like if statements control program flow.",
            course_title="Python Programming Basics",
            lesson_number=3,
            chunk_index=2,
        ),
    ]

    return {"course": course, "chunks": chunks}


@pytest.fixture
def mock_rag_system():
    """Fixture providing mock RAG system for API tests"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a mock answer",
        [{"text": "Mock source text", "link": "https://example.com/lesson1"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Programming Basics", "Advanced Python"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag


@pytest.fixture
def test_app():
    """Fixture providing a test FastAPI app without static file mounting"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create a clean test app
    app = FastAPI(title="Test RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define request/response models inline
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test answer",
        [{"text": "Test source", "link": "https://example.com/test"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    
    # Define endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        from fastapi import HTTPException
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            
            source_items = [
                SourceItem(text=source.get("text", ""), link=source.get("link"))
                for source in sources
            ]
            
            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        from fastapi import HTTPException
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    return app


@pytest.fixture
async def test_client(test_app):
    """Fixture providing async test client"""
    from fastapi.testclient import TestClient
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Auto-suppress warnings during tests"""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*resource_tracker.*")
