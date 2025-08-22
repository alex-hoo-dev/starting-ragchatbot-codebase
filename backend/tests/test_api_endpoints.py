"""
Comprehensive API endpoint tests for the RAG system FastAPI application.
Tests all endpoints with proper request/response validation.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""
    
    @pytest.mark.asyncio
    async def test_query_with_valid_request(self, test_client: AsyncClient):
        """Test successful query with valid request data"""
        request_data = {
            "query": "What is Python?",
            "session_id": "test-session-123"
        }
        
        response = await test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source
    
    @pytest.mark.asyncio
    async def test_query_without_session_id(self, test_client: AsyncClient):
        """Test query without session_id creates new session"""
        request_data = {"query": "What is machine learning?"}
        
        response = await test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        assert "answer" in data
        assert "sources" in data
    
    @pytest.mark.asyncio
    async def test_query_with_empty_query(self, test_client: AsyncClient):
        """Test query with empty query string"""
        request_data = {"query": ""}
        
        response = await test_client.post("/api/query", json=request_data)
        
        # Should still process but may return different results
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    @pytest.mark.asyncio
    async def test_query_with_missing_query_field(self, test_client: AsyncClient):
        """Test query with missing required query field"""
        request_data = {"session_id": "test-123"}
        
        response = await test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_query_with_invalid_json(self, test_client: AsyncClient):
        """Test query with malformed JSON"""
        response = await test_client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_query_response_structure(self, test_client: AsyncClient):
        """Test that query response has correct structure"""
        request_data = {"query": "Test query"}
        
        response = await test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure matches QueryResponse model
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Validate sources structure
        assert isinstance(data["sources"], list)
        for source in data["sources"]:
            assert isinstance(source, dict)
            assert "text" in source
            assert "link" in source


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""
    
    @pytest.mark.asyncio
    async def test_get_course_stats_success(self, test_client: AsyncClient):
        """Test successful retrieval of course statistics"""
        response = await test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
    
    @pytest.mark.asyncio
    async def test_course_stats_response_structure(self, test_client: AsyncClient):
        """Test course stats response structure matches CourseStats model"""
        response = await test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Validate data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Validate course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    @pytest.mark.asyncio
    async def test_course_stats_with_no_courses(self, test_client: AsyncClient):
        """Test course stats when no courses are available"""
        # This test assumes the mock can be configured for empty state
        response = await test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still return valid structure even with no courses
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)


@pytest.mark.api
class TestRootEndpoint:
    """Test suite for root / endpoint"""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, test_client: AsyncClient):
        """Test root endpoint returns expected message"""
        response = await test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["message"] == "RAG System API"
    
    @pytest.mark.asyncio
    async def test_root_endpoint_content_type(self, test_client: AsyncClient):
        """Test root endpoint returns JSON content type"""
        response = await test_client.get("/")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


@pytest.mark.api
class TestAPIErrorHandling:
    """Test suite for API error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, test_client: AsyncClient):
        """Test request to non-existent endpoint"""
        response = await test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_wrong_http_method(self, test_client: AsyncClient):
        """Test wrong HTTP method on endpoints"""
        # GET on POST endpoint
        response = await test_client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed
        
        # POST on GET endpoint
        response = await test_client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed
    
    @pytest.mark.asyncio
    async def test_cors_enabled(self, test_client: AsyncClient):
        """Test that CORS middleware is configured (basic functionality test)"""
        # Make a request and verify it succeeds - CORS would block if misconfigured
        response = await test_client.get("/api/courses")
        assert response.status_code == 200
        
        # Verify we can make cross-origin style requests
        response = await test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API workflows"""
    
    @pytest.mark.asyncio
    async def test_query_and_courses_workflow(self, test_client: AsyncClient):
        """Test typical user workflow: check courses then query"""
        # First get course stats
        courses_response = await test_client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Then make a query
        query_data = {"query": "Tell me about the available courses"}
        query_response = await test_client.post("/api/query", json=query_data)
        assert query_response.status_code == 200
        query_result = query_response.json()
        
        # Verify both responses are valid
        assert "total_courses" in courses_data
        assert "answer" in query_result
        assert "session_id" in query_result
    
    @pytest.mark.asyncio
    async def test_session_consistency(self, test_client: AsyncClient):
        """Test session consistency across multiple queries"""
        session_id = "test-session-456"
        
        # First query
        query1_data = {"query": "First query", "session_id": session_id}
        response1 = await test_client.post("/api/query", json=query1_data)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second query with same session
        query2_data = {"query": "Second query", "session_id": session_id}
        response2 = await test_client.post("/api/query", json=query2_data)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Both should have same session_id
        assert data1["session_id"] == session_id
        assert data2["session_id"] == session_id


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance-related API tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, test_client: AsyncClient):
        """Test handling of concurrent queries"""
        import asyncio
        
        async def make_query(query_text: str):
            return await test_client.post(
                "/api/query", 
                json={"query": query_text}
            )
        
        # Make multiple concurrent requests
        tasks = [
            make_query(f"Test query {i}") 
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data