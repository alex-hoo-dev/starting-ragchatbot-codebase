import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        # Create mock vector store
        self.mock_vector_store = Mock()
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Initialize CourseSearchTool with mock
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_basic_query_success(self):
        """Test successful execution of basic query without filters"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is course content about Python programming."],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute(query="Python programming")
        
        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python programming",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert isinstance(result, str)
        assert "Python Basics" in result
        assert "Lesson 1" in result
        assert "This is course content about Python programming." in result
        
        # Verify sources were stored
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert "link" in self.search_tool.last_sources[0]
    
    def test_execute_with_course_filter(self):
        """Test execution with course name filter"""
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 2}],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="functions", 
            course_name="Advanced Python"
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        assert "Advanced Python" in result
        assert "functions" in result or "Advanced Python concepts" in result
    
    def test_execute_with_lesson_filter(self):
        """Test execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content about loops"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="loops", 
            lesson_number=3
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="loops",
            course_name=None,
            lesson_number=3
        )
        
        assert "Lesson 3" in result
        assert "loops" in result or "Lesson 3 content about loops" in result
    
    def test_execute_with_both_filters(self):
        """Test execution with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 5}],
            distances=[0.15],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="machine learning",
            course_name="Data Science",
            lesson_number=5
        )
        
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name="Data Science",
            lesson_number=5
        )
        
        assert "Data Science" in result
        assert "Lesson 5" in result
    
    def test_execute_empty_results(self):
        """Test execution when no results are found"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(query="nonexistent topic")
        
        assert "No relevant content found" in result
        assert self.search_tool.last_sources == []
    
    def test_execute_empty_results_with_filters(self):
        """Test execution when no results are found with filters applied"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(
            query="nonexistent", 
            course_name="Python Basics",
            lesson_number=1
        )
        
        assert "No relevant content found" in result
        assert "in course 'Python Basics'" in result
        assert "in lesson 1" in result
    
    def test_execute_with_search_error(self):
        """Test execution when search returns an error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(query="test query")
        
        assert result == "Database connection failed"
        assert self.search_tool.last_sources == []
    
    def test_execute_multiple_results(self):
        """Test execution with multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First result about Python", 
                "Second result about Python"
            ],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Advanced", "lesson_number": 2}
            ],
            distances=[0.2, 0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(query="Python")
        
        # Should contain both results
        assert "Python Basics" in result
        assert "Python Advanced" in result
        assert "First result about Python" in result
        assert "Second result about Python" in result
        
        # Should have two sources
        assert len(self.search_tool.last_sources) == 2
    
    def test_execute_missing_metadata(self):
        """Test execution with missing metadata fields"""
        mock_results = SearchResults(
            documents=["Content without complete metadata"],
            metadata=[{"course_title": "Test Course"}],  # Missing lesson_number
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(query="test")
        
        assert "Test Course" in result
        # Should handle missing lesson_number gracefully
        assert "Content without complete metadata" in result
        assert len(self.search_tool.last_sources) == 1
    
    def test_execute_no_lesson_link(self):
        """Test execution when no lesson link is available"""
        # Mock get_lesson_link to return None
        self.mock_vector_store.get_lesson_link.return_value = None
        
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute(query="test")
        
        # Should still work without lesson link
        assert "Test Course" in result
        assert len(self.search_tool.last_sources) == 1
        # Source should not have a link
        assert "link" not in self.search_tool.last_sources[0]
    
    def test_get_tool_definition(self):
        """Test the tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        assert isinstance(definition, dict)
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]
        
        # Verify all expected properties exist
        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
    
    def test_sources_reset_behavior(self):
        """Test that sources are properly managed across multiple calls"""
        mock_results = SearchResults(
            documents=["First search result"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # First search
        self.search_tool.execute(query="first query")
        assert len(self.search_tool.last_sources) == 1
        
        # Second search should replace sources
        mock_results.documents = ["Second search result"]
        mock_results.metadata = [{"course_title": "Course 2", "lesson_number": 2}]
        
        self.search_tool.execute(query="second query")
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0]["text"] == "Course 2 - Lesson 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])