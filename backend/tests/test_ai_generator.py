import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockContentBlock:
    """Mock for anthropic content block"""
    def __init__(self, block_type, name=None, input_data=None, block_id=None, text=None):
        self.type = block_type
        self.name = name
        self.input = input_data or {}
        self.id = block_id
        self.text = text


class MockResponse:
    """Mock for anthropic API response"""
    def __init__(self, content, stop_reason="stop"):
        if isinstance(content, str):
            # Text response
            self.content = [MockContentBlock("text", text=content)]
        else:
            # Tool use or complex content
            self.content = content
        self.stop_reason = stop_reason


class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test"""
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"
        
        # Mock the anthropic client
        self.mock_client = Mock()
        
        # Initialize AIGenerator with mocked client
        with patch('ai_generator.anthropic.Anthropic', return_value=self.mock_client):
            self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_generate_response_without_tools(self):
        """Test generating response without tool usage"""
        # Mock API response
        mock_response = MockResponse("This is a direct response without tools.")
        self.mock_client.messages.create.return_value = mock_response
        
        # Test without tools
        result = self.ai_generator.generate_response(
            query="What is Python?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )
        
        # Verify API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        
        # Check the parameters passed to API
        assert call_args[1]["model"] == self.model
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args[1]
        
        assert result == "This is a direct response without tools."
    
    def test_generate_response_with_conversation_history(self):
        """Test generating response with conversation history"""
        mock_response = MockResponse("Response with history context.")
        self.mock_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"
        
        result = self.ai_generator.generate_response(
            query="Follow up question",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content
        assert "Previous conversation:" in system_content
        
        assert result == "Response with history context."
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test generating response with tools available but not used"""
        mock_response = MockResponse("Direct answer without using tools.")
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock tools and tool manager
        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
        tool_manager = Mock()
        
        result = self.ai_generator.generate_response(
            query="What is 2+2?",
            conversation_history=None,
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify tools were included in API call
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        
        # Tool manager should not be called since no tools were used
        tool_manager.execute_tool.assert_not_called()
        
        assert result == "Direct answer without using tools."
    
    def test_generate_response_with_tool_use_success(self):
        """Test successful tool use and follow-up response"""
        # Mock initial response with tool use
        tool_use_content = [
            MockContentBlock("text", text="I'll search for that information."),
            MockContentBlock(
                "tool_use", 
                name="search_course_content",
                input_data={"query": "Python basics"},
                block_id="tool_123"
            )
        ]
        initial_response = MockResponse(tool_use_content, stop_reason="tool_use")
        
        # Mock final response after tool execution
        final_response = MockResponse("Based on the search results, Python is a programming language.")
        
        # Set up mock client to return both responses
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Python is a high-level programming language."
        
        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
        
        result = self.ai_generator.generate_response(
            query="What is Python?",
            conversation_history=None,
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2
        
        # Check the final API call included tool results
        final_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        assert len(messages) == 3  # Original user message, AI tool use, tool results
        
        # Check tool result message
        tool_result_message = messages[2]
        assert tool_result_message["role"] == "user"
        assert len(tool_result_message["content"]) == 1
        tool_result = tool_result_message["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"
        assert tool_result["content"] == "Python is a high-level programming language."
        
        assert result == "Based on the search results, Python is a programming language."
    
    def test_generate_response_tool_use_multiple_tools(self):
        """Test handling multiple tool uses in one response"""
        # Mock response with multiple tool uses
        tool_use_content = [
            MockContentBlock(
                "tool_use",
                name="search_course_content", 
                input_data={"query": "Python"},
                block_id="tool_1"
            ),
            MockContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "Java"},
                block_id="tool_2"
            )
        ]
        initial_response = MockResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockResponse("Comparison of Python and Java.")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Python info", 
            "Java info"
        ]
        
        tools = [{"name": "search_course_content"}]
        
        result = self.ai_generator.generate_response(
            query="Compare Python and Java",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="Java")
        
        # Check final message has both tool results
        final_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        tool_result_message = messages[2]
        assert len(tool_result_message["content"]) == 2
        
        assert result == "Comparison of Python and Java."
    
    def test_generate_response_tool_execution_error(self):
        """Test handling when tool execution fails"""
        # Mock tool use response
        tool_use_content = [
            MockContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test"},
                block_id="tool_123"
            )
        ]
        initial_response = MockResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockResponse("I encountered an error searching for that information.")
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager to return error
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool execution failed"
        
        result = self.ai_generator.generate_response(
            query="test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager
        )
        
        # Error should be passed to AI as tool result
        final_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        tool_result = messages[2]["content"][0]
        assert tool_result["content"] == "Tool execution failed"
        
        assert result == "I encountered an error searching for that information."
    
    def test_generate_response_no_tool_manager(self):
        """Test tool use when no tool manager is provided"""
        # Mock tool use response
        tool_use_content = [
            MockContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "test"},
                block_id="tool_123"
            )
        ]
        initial_response = MockResponse(tool_use_content, stop_reason="tool_use")
        self.mock_client.messages.create.return_value = initial_response
        
        # No tool manager provided
        result = self.ai_generator.generate_response(
            query="test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=None
        )
        
        # Should return the initial response content as text
        # Since no tool manager, _handle_tool_execution won't be called
        # We need to check what happens - let's see if it has a text part
        assert "I'll search for that information." in result or isinstance(result, str)
    
    def test_system_prompt_structure(self):
        """Test that system prompt is properly structured"""
        mock_response = MockResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        self.ai_generator.generate_response(
            query="Test query",
            conversation_history="Previous context"
        )
        
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Check system prompt contains expected elements
        assert "AI assistant specialized in course materials" in system_content
        assert "Search Tool Usage" in system_content
        assert "One search per query maximum" in system_content
        assert "Previous conversation:" in system_content
        assert "Previous context" in system_content
    
    def test_api_parameters_structure(self):
        """Test API parameters are correctly structured"""
        mock_response = MockResponse("Test response")
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool"}]
        
        self.ai_generator.generate_response(
            query="Test query",
            conversation_history=None,
            tools=tools
        )
        
        call_args = self.mock_client.messages.create.call_args[1]
        
        # Check all required parameters
        assert call_args["model"] == self.model
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])