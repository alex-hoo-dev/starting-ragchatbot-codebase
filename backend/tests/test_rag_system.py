import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystem:
    """Test suite for RAG system's content query handling"""

    def setup_method(self):
        """Set up test fixtures before each test"""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_chroma"
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 2

        # Mock all the components
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_processor,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_generator,
            patch("rag_system.SessionManager") as mock_session_manager,
            patch("rag_system.ToolManager") as mock_tool_manager,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
        ):

            # Initialize RAG system with mocks
            self.rag_system = RAGSystem(self.mock_config)

            # Store references to mocks for testing
            self.mock_doc_processor = self.rag_system.document_processor
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager
            self.mock_tool_manager = self.rag_system.tool_manager
            self.mock_search_tool = self.rag_system.search_tool

    def test_query_without_session(self):
        """Test processing a query without session ID"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = (
            "This is about Python programming."
        )

        # Mock tool manager sources
        self.mock_tool_manager.get_last_sources.return_value = [
            {"text": "Python Basics - Lesson 1", "link": "https://example.com/lesson1"}
        ]

        # Execute query
        response, sources = self.rag_system.query("What is Python?")

        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args

        assert (
            "Answer this question about course materials: What is Python?"
            in call_args[1]["query"]
        )
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] == self.mock_tool_manager

        # Verify response and sources
        assert response == "This is about Python programming."
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Basics - Lesson 1"

        # Verify sources were reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()

        # Verify session was not used
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()

    def test_query_with_session(self):
        """Test processing a query with session ID"""
        session_id = "test_session_123"

        # Mock session manager
        self.mock_session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )

        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = (
            "Follow-up response about Python."
        )

        # Mock tool manager sources
        self.mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        response, sources = self.rag_system.query(
            "Tell me more about functions", session_id
        )

        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator was called with history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"

        # Verify session was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id,
            "Tell me more about functions",
            "Follow-up response about Python.",
        )

        assert response == "Follow-up response about Python."
        assert sources == []

    def test_query_with_sources(self):
        """Test query that returns sources from search tool"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = (
            "Here's information about data science."
        )

        # Mock tool manager sources
        mock_sources = [
            {"text": "Data Science 101 - Lesson 1", "link": "https://example.com/ds1"},
            {"text": "Data Science 101 - Lesson 2"},  # No link
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources

        # Execute query
        response, sources = self.rag_system.query("What is data science?")

        assert response == "Here's information about data science."
        assert len(sources) == 2
        assert sources[0]["text"] == "Data Science 101 - Lesson 1"
        assert sources[0]["link"] == "https://example.com/ds1"
        assert sources[1]["text"] == "Data Science 101 - Lesson 2"
        assert "link" not in sources[1]

    def test_query_tool_manager_integration(self):
        """Test that RAG system properly integrates with tool manager"""
        # Mock tool manager methods
        mock_tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        self.mock_tool_manager.get_last_sources.return_value = []

        # Mock AI generator
        self.mock_ai_generator.generate_response.return_value = "Response using tools."

        # Execute query
        response, sources = self.rag_system.query("Test query")

        # Verify tool definitions were retrieved and passed to AI
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["tools"] == mock_tool_definitions
        assert call_args[1]["tool_manager"] == self.mock_tool_manager

    def test_query_prompt_formatting(self):
        """Test that query prompts are properly formatted"""
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.mock_tool_manager.get_last_sources.return_value = []

        test_query = "How do I learn machine learning?"
        self.rag_system.query(test_query)

        # Check that prompt is properly formatted
        call_args = self.mock_ai_generator.generate_response.call_args
        expected_prompt = f"Answer this question about course materials: {test_query}"
        assert call_args[1]["query"] == expected_prompt

    def test_query_error_handling(self):
        """Test query error handling when AI generator fails"""
        # Mock AI generator to raise exception
        self.mock_ai_generator.generate_response.side_effect = Exception(
            "API call failed"
        )
        self.mock_tool_manager.get_last_sources.return_value = []

        # Execute query should raise exception
        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("Test query")

        assert "API call failed" in str(exc_info.value)

    def test_query_session_error_handling(self):
        """Test query handling when session operations fail"""
        session_id = "test_session"

        # Mock session manager to fail on history retrieval
        self.mock_session_manager.get_conversation_history.side_effect = Exception(
            "Session error"
        )

        # Mock AI generator to succeed
        self.mock_ai_generator.generate_response.return_value = (
            "Response despite session error"
        )
        self.mock_tool_manager.get_last_sources.return_value = []

        # Query should still work even if session fails
        with pytest.raises(Exception):
            self.rag_system.query("Test query", session_id)

    def test_query_empty_response(self):
        """Test handling of empty AI response"""
        self.mock_ai_generator.generate_response.return_value = ""
        self.mock_tool_manager.get_last_sources.return_value = []

        response, sources = self.rag_system.query("Test query")

        assert response == ""
        assert sources == []

    def test_query_none_response(self):
        """Test handling of None AI response"""
        self.mock_ai_generator.generate_response.return_value = None
        self.mock_tool_manager.get_last_sources.return_value = []

        response, sources = self.rag_system.query("Test query")

        assert response is None
        assert sources == []

    def test_query_complex_conversation_flow(self):
        """Test complex conversation flow with multiple exchanges"""
        session_id = "complex_session"

        # Mock conversation history
        self.mock_session_manager.get_conversation_history.return_value = (
            "User asked about Python. AI explained basics."
        )

        # Mock AI response with sources
        self.mock_ai_generator.generate_response.return_value = (
            "Advanced Python concepts include classes and modules."
        )
        mock_sources = [{"text": "Advanced Python - Lesson 5"}]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources

        # Execute query
        response, sources = self.rag_system.query(
            "What are advanced Python concepts?", session_id
        )

        # Verify full flow
        self.mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        call_args = self.mock_ai_generator.generate_response.call_args
        assert "Advanced Python concepts" in call_args[1]["query"]
        assert (
            call_args[1]["conversation_history"]
            == "User asked about Python. AI explained basics."
        )

        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id,
            "What are advanced Python concepts?",
            "Advanced Python concepts include classes and modules.",
        )

        assert response == "Advanced Python concepts include classes and modules."
        assert len(sources) == 1
        assert sources[0]["text"] == "Advanced Python - Lesson 5"

    def test_get_course_analytics(self):
        """Test course analytics functionality"""
        # Mock vector store analytics
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Python Basics",
            "Data Science 101",
            "Machine Learning",
            "Web Development",
            "AI Ethics",
        ]

        analytics = self.rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Python Basics" in analytics["course_titles"]
        assert "Data Science 101" in analytics["course_titles"]

        # Verify vector store methods were called
        self.mock_vector_store.get_course_count.assert_called_once()
        self.mock_vector_store.get_existing_course_titles.assert_called_once()

    def test_initialization_components(self):
        """Test that all components are properly initialized"""
        # Verify all components exist
        assert self.rag_system.document_processor is not None
        assert self.rag_system.vector_store is not None
        assert self.rag_system.ai_generator is not None
        assert self.rag_system.session_manager is not None
        assert self.rag_system.tool_manager is not None
        assert self.rag_system.search_tool is not None

        # Verify search tool was registered
        # Note: We can't easily test the registration without examining the mock calls
        # but we can verify the tool exists
        assert hasattr(self.rag_system, "search_tool")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
