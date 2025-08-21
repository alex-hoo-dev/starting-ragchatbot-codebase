import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults


class TestRAGSystemIntegration:
    """Integration tests for the full RAG system pipeline"""
    
    def setup_method(self):
        """Set up test fixtures for integration testing"""
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
        
        # We'll use real ToolManager and CourseSearchTool but mock VectorStore
        self.mock_vector_store = Mock()
        
        # Mock other components
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore', return_value=self.mock_vector_store), \
             patch('rag_system.SessionManager'), \
             patch('ai_generator.anthropic.Anthropic'):
            
            self.rag_system = RAGSystem(self.mock_config)
    
    def test_successful_content_query_flow(self):
        """Test the complete flow for a successful content query"""
        # Mock vector store search to return content
        mock_search_results = SearchResults(
            documents=[
                "Python is a high-level programming language known for its simplicity.",
                "Python supports multiple programming paradigms including object-oriented programming."
            ],
            metadata=[
                {"course_title": "Python Fundamentals", "lesson_number": 1},
                {"course_title": "Python Fundamentals", "lesson_number": 2}
            ],
            distances=[0.2, 0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/python-lesson"
        
        # Mock AI generator to simulate tool use
        mock_tool_use_content = [
            Mock(type="text", text="I'll search for information about Python."),
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "Python programming language"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(
            content=[Mock(type="text", text="Python is a versatile programming language perfect for beginners and experts alike. It's known for its readable syntax and extensive libraries.")]
        )
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response, 
            mock_final_response
        ]
        
        # Execute the query
        response, sources = self.rag_system.query("What is Python programming?")
        
        # Verify the complete flow
        # 1. AI generator should be called twice (initial + follow-up)
        assert self.rag_system.ai_generator.client.messages.create.call_count == 2
        
        # 2. Vector store search should be called via the tool
        self.mock_vector_store.search.assert_called_once_with(
            query="Python programming language",
            course_name=None,
            lesson_number=None
        )
        
        # 3. Response should be from the final AI call
        assert "Python is a versatile programming language" in response
        
        # 4. Sources should be populated from the search tool
        assert len(sources) == 2
        assert sources[0]["text"] == "Python Fundamentals - Lesson 1"
        assert sources[1]["text"] == "Python Fundamentals - Lesson 2"
        assert "link" in sources[0]
        assert "link" in sources[1]
    
    def test_query_failed_scenario_no_results(self):
        """Test scenario where query fails due to no search results"""
        # Mock vector store to return empty results
        mock_empty_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_empty_results
        
        # Mock AI generator tool use
        mock_tool_use_content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "nonexistent topic"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(
            content=[Mock(type="text", text="I couldn't find any information about that topic in the course materials.")]
        )
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Execute the query
        response, sources = self.rag_system.query("Tell me about quantum computing")
        
        # Verify vector store was searched but returned empty
        self.mock_vector_store.search.assert_called_once()
        
        # Tool should return "No relevant content found"
        # AI should get this as tool result and respond appropriately
        assert "couldn't find any information" in response
        assert len(sources) == 0
    
    def test_query_failed_scenario_search_error(self):
        """Test scenario where query fails due to search error"""
        # Mock vector store to return search error
        mock_error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_error_results
        
        # Mock AI generator tool use
        mock_tool_use_content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "test query"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(
            content=[Mock(type="text", text="I'm experiencing technical difficulties accessing the course materials right now.")]
        )
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Execute the query
        response, sources = self.rag_system.query("Test query")
        
        # Verify the error handling
        self.mock_vector_store.search.assert_called_once()
        assert "technical difficulties" in response
        assert len(sources) == 0
    
    def test_query_failed_scenario_ai_error(self):
        """Test scenario where query fails due to AI generator error"""
        # Mock AI generator to raise exception
        self.rag_system.ai_generator.client.messages.create.side_effect = Exception("API rate limit exceeded")
        
        # Execute query should raise exception
        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("Test query")
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    def test_query_with_course_filter_success(self):
        """Test successful query with course filtering"""
        # Mock vector store with course-specific results
        mock_search_results = SearchResults(
            documents=["Advanced Python concepts including decorators and metaclasses."],
            metadata=[{"course_title": "Advanced Python Programming", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/advanced-python"
        
        # Mock AI generator to use course filter
        mock_tool_use_content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "decorators", "course_name": "Advanced Python"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(
            content=[Mock(type="text", text="Decorators in Python are a powerful feature that allows you to modify functions.")]
        )
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Execute query
        response, sources = self.rag_system.query("Explain Python decorators in the Advanced Python course")
        
        # Verify course filter was used
        self.mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="Advanced Python", 
            lesson_number=None
        )
        
        assert "Decorators in Python" in response
        assert len(sources) == 1
        assert "Advanced Python Programming" in sources[0]["text"]
    
    def test_conversation_flow_with_context(self):
        """Test conversation flow with context preservation"""
        session_id = "test_session"
        
        # Mock session manager
        self.rag_system.session_manager.get_conversation_history.return_value = \
            "User: What is Python?\nAI: Python is a programming language.\n"
        
        # Mock vector store results
        mock_search_results = SearchResults(
            documents=["Python functions are defined using the def keyword."],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.15],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_search_results
        
        # Mock AI generator follow-up
        mock_tool_use_content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "Python functions"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(
            content=[Mock(type="text", text="Building on what we discussed about Python, functions are reusable blocks of code.")]
        )
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Execute follow-up query
        response, sources = self.rag_system.query("How do I define functions?", session_id)
        
        # Verify context was used
        self.rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify conversation was updated
        self.rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id,
            "How do I define functions?",
            "Building on what we discussed about Python, functions are reusable blocks of code."
        )
        
        assert "Building on what we discussed" in response
    
    def test_tool_registration_and_execution(self):
        """Test that tools are properly registered and executed"""
        # Verify search tool was registered
        assert hasattr(self.rag_system, 'search_tool')
        assert hasattr(self.rag_system, 'tool_manager')
        
        # Test tool definition retrieval
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) >= 1
        
        # Find the search tool definition
        search_tool_def = None
        for tool_def in tool_definitions:
            if tool_def.get('name') == 'search_course_content':
                search_tool_def = tool_def
                break
        
        assert search_tool_def is not None
        assert 'description' in search_tool_def
        assert 'input_schema' in search_tool_def
    
    def test_sources_handling_and_reset(self):
        """Test that sources are properly handled and reset"""
        # Mock search results with sources
        mock_search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_search_results
        self.mock_vector_store.get_lesson_link.return_value = "https://test.com/lesson1"
        
        # Mock AI tool use
        mock_tool_use_content = [
            Mock(type="tool_use", name="search_course_content", 
                 input={"query": "test"}, id="tool_123")
        ]
        mock_initial_response = Mock(content=mock_tool_use_content, stop_reason="tool_use")
        mock_final_response = Mock(content=[Mock(type="text", text="Test response")])
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # First query
        response1, sources1 = self.rag_system.query("First query")
        assert len(sources1) == 1
        assert "Test Course" in sources1[0]["text"]
        
        # Sources should be reset for next query
        # Mock empty results for second query
        mock_empty_results = SearchResults([], [], [], None)
        self.mock_vector_store.search.return_value = mock_empty_results
        
        self.rag_system.ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        response2, sources2 = self.rag_system.query("Second query")
        assert len(sources2) == 0  # Should be empty since no results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])