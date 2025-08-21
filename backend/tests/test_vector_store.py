import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk, Lesson


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        # Mock ChromaDB components
        self.mock_client = Mock()
        self.mock_course_catalog = Mock()
        self.mock_course_content = Mock()
        self.mock_embedding_function = Mock()
        
        # Set up mock collections
        self.mock_client.get_or_create_collection.side_effect = [
            self.mock_course_catalog,
            self.mock_course_content
        ]
        
        with patch('vector_store.chromadb.PersistentClient', return_value=self.mock_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction', 
                   return_value=self.mock_embedding_function):
            
            self.vector_store = VectorStore(
                chroma_path="./test_chroma",
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )
    
    def test_search_basic_query(self):
        """Test basic search without filters"""
        # Mock successful ChromaDB query
        mock_chroma_results = {
            'documents': [['Python is a programming language.', 'Variables store data.']],
            'metadatas': [[
                {'course_title': 'Python Basics', 'lesson_number': 1, 'chunk_index': 0},
                {'course_title': 'Python Basics', 'lesson_number': 2, 'chunk_index': 1}
            ]],
            'distances': [[0.2, 0.3]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        # Execute search
        results = self.vector_store.search("Python programming")
        
        # Verify ChromaDB was called correctly
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["Python programming"],
            n_results=5,
            where=None
        )
        
        # Verify results
        assert isinstance(results, SearchResults)
        assert len(results.documents) == 2
        assert results.documents[0] == 'Python is a programming language.'
        assert results.documents[1] == 'Variables store data.'
        assert len(results.metadata) == 2
        assert results.metadata[0]['course_title'] == 'Python Basics'
        assert results.error is None
        assert not results.is_empty()
    
    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        # Mock course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [['Advanced Python']],
            'metadatas': [[{'title': 'Advanced Python Programming'}]]
        }
        
        # Mock content search
        mock_chroma_results = {
            'documents': [['Classes and objects in Python.']],
            'metadatas': [[{'course_title': 'Advanced Python Programming', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        # Execute search
        results = self.vector_store.search("classes", course_name="Advanced Python")
        
        # Verify course resolution
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["Advanced Python"],
            n_results=1
        )
        
        # Verify content search with filter
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["classes"],
            n_results=5,
            where={"course_title": "Advanced Python Programming"}
        )
        
        assert len(results.documents) == 1
        assert "Classes and objects" in results.documents[0]
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_chroma_results = {
            'documents': [['Lesson 2 content about loops.']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 2}]],
            'distances': [[0.15]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("loops", lesson_number=2)
        
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["loops"],
            n_results=5,
            where={"lesson_number": 2}
        )
        
        assert results.metadata[0]['lesson_number'] == 2
    
    def test_search_with_both_filters(self):
        """Test search with both course and lesson filters"""
        # Mock course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [['Data Science']],
            'metadatas': [[{'title': 'Data Science 101'}]]
        }
        
        mock_chroma_results = {
            'documents': [['Machine learning algorithms.']],
            'metadatas': [[{'course_title': 'Data Science 101', 'lesson_number': 5}]],
            'distances': [[0.05]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search(
            "machine learning", 
            course_name="Data Science", 
            lesson_number=5
        )
        
        # Should use AND filter
        expected_filter = {
            "$and": [
                {"course_title": "Data Science 101"},
                {"lesson_number": 5}
            ]
        }
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["machine learning"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_course_not_found(self):
        """Test search when course name cannot be resolved"""
        # Mock course resolution failure
        self.mock_course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        results = self.vector_store.search("test", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
        
        # Content search should not be called
        self.mock_course_content.query.assert_not_called()
    
    def test_search_empty_results(self):
        """Test search that returns no results"""
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("nonexistent topic")
        
        assert results.is_empty()
        assert results.error is None
        assert len(results.documents) == 0
    
    def test_search_with_limit(self):
        """Test search with custom result limit"""
        mock_chroma_results = {
            'documents': [['Result 1', 'Result 2', 'Result 3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("test query", limit=3)
        
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where=None
        )
        
        assert len(results.documents) == 3
    
    def test_search_exception_handling(self):
        """Test search error handling when ChromaDB fails"""
        self.mock_course_content.query.side_effect = Exception("ChromaDB connection failed")
        
        results = self.vector_store.search("test query")
        
        assert "Search error: ChromaDB connection failed" in results.error
        assert results.is_empty()
    
    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        self.mock_course_catalog.query.return_value = {
            'documents': [['Python Programming']],
            'metadatas': [[{'title': 'Python Programming Fundamentals'}]]
        }
        
        # Use private method for testing
        resolved_title = self.vector_store._resolve_course_name("Python")
        
        assert resolved_title == "Python Programming Fundamentals"
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
    
    def test_resolve_course_name_failure(self):
        """Test course name resolution failure"""
        self.mock_course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        resolved_title = self.vector_store._resolve_course_name("Nonexistent")
        
        assert resolved_title is None
    
    def test_resolve_course_name_exception(self):
        """Test course name resolution with exception"""
        self.mock_course_catalog.query.side_effect = Exception("Query failed")
        
        resolved_title = self.vector_store._resolve_course_name("Test")
        
        assert resolved_title is None
    
    def test_build_filter_no_filters(self):
        """Test filter building with no parameters"""
        filter_dict = self.vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self):
        """Test filter building with course only"""
        filter_dict = self.vector_store._build_filter("Python Basics", None)
        assert filter_dict == {"course_title": "Python Basics"}
    
    def test_build_filter_lesson_only(self):
        """Test filter building with lesson only"""
        filter_dict = self.vector_store._build_filter(None, 3)
        assert filter_dict == {"lesson_number": 3}
    
    def test_build_filter_both_parameters(self):
        """Test filter building with both course and lesson"""
        filter_dict = self.vector_store._build_filter("Advanced Python", 2)
        expected = {
            "$and": [
                {"course_title": "Advanced Python"},
                {"lesson_number": 2}
            ]
        }
        assert filter_dict == expected
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        # Create test chunks
        chunks = [
            CourseChunk(
                content="Python basics content",
                course_title="Python Programming",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables and data types",
                course_title="Python Programming", 
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        self.vector_store.add_course_content(chunks)
        
        # Verify collection.add was called correctly
        self.mock_course_content.add.assert_called_once()
        call_args = self.mock_course_content.add.call_args[1]
        
        assert len(call_args["documents"]) == 2
        assert call_args["documents"][0] == "Python basics content"
        assert call_args["documents"][1] == "Variables and data types"
        
        assert len(call_args["metadatas"]) == 2
        assert call_args["metadatas"][0]["course_title"] == "Python Programming"
        assert call_args["metadatas"][0]["lesson_number"] == 1
        assert call_args["metadatas"][0]["chunk_index"] == 0
        
        assert len(call_args["ids"]) == 2
        assert call_args["ids"][0] == "Python_Programming_0"
        assert call_args["ids"][1] == "Python_Programming_1"
    
    def test_add_course_content_empty(self):
        """Test adding empty course content"""
        self.vector_store.add_course_content([])
        
        # Should not call add method
        self.mock_course_content.add.assert_not_called()
    
    def test_get_existing_course_titles(self):
        """Test retrieving existing course titles"""
        self.mock_course_catalog.get.return_value = {
            'ids': ['Python Basics', 'Data Science 101', 'Web Development']
        }
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert len(titles) == 3
        assert 'Python Basics' in titles
        assert 'Data Science 101' in titles
        assert 'Web Development' in titles
        
        self.mock_course_catalog.get.assert_called_once()
    
    def test_get_existing_course_titles_error(self):
        """Test retrieving course titles with error"""
        self.mock_course_catalog.get.side_effect = Exception("Database error")
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_course_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3', 'Course 4']
        }
        
        count = self.vector_store.get_course_count()
        
        assert count == 4
    
    def test_get_course_count_error(self):
        """Test getting course count with error"""
        self.mock_course_catalog.get.side_effect = Exception("Connection error")
        
        count = self.vector_store.get_course_count()
        
        assert count == 0


class TestSearchResults:
    """Test suite for SearchResults class"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results with data"""
        chroma_results = {
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'meta1': 'value1'}, {'meta2': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['Doc 1', 'Doc 2']
        assert results.metadata == [{'meta1': 'value1'}, {'meta2': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
        assert not results.is_empty()
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()
    
    def test_from_chroma_no_data(self):
        """Test creating SearchResults from ChromaDB results with no data"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "No results found"
        
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
        assert results.is_empty()
    
    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [], None)
        assert empty_results.is_empty()
        
        # Non-empty results
        non_empty_results = SearchResults(['doc'], [{}], [0.1], None)
        assert not non_empty_results.is_empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])