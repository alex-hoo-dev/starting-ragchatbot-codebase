"""
Pytest configuration and fixtures for the RAG chatbot tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

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
        documents=[],
        metadata=[],
        distances=[],
        error=None,
        is_empty=lambda: True
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
    from models import Course, Lesson, CourseChunk
    
    course = Course(
        title="Python Programming Basics",
        course_link="https://example.com/python-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="Control Structures", lesson_link="https://example.com/lesson3")
        ]
    )
    
    chunks = [
        CourseChunk(
            content="Python is a high-level programming language.",
            course_title="Python Programming Basics",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python can store different types of data.",
            course_title="Python Programming Basics", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Control structures like if statements control program flow.",
            course_title="Python Programming Basics",
            lesson_number=3,
            chunk_index=2
        )
    ]
    
    return {"course": course, "chunks": chunks}


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Auto-suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*resource_tracker.*")