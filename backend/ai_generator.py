import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Maximum 2 search operations per query** - use them strategically
- You may search once, analyze results, then search again with refined parameters if needed
- For complex queries requiring multiple searches: first search broadly, then search specifically based on initial findings
- Synthesize all search results into accurate, fact-based responses
- If searches yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex queries**: May require multiple searches to gather complete information (e.g., comparing courses, multi-part questions)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or describe your search strategy


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use":
            if tool_manager:
                return self._handle_tool_execution(response, api_params, tool_manager)
            else:
                # No tool manager available, return fallback response
                return "I need access to search tools to answer this question, but they are currently unavailable."
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int = 2):
        """
        Handle execution of tool calls with support for multiple rounds.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool execution rounds (default: 2)
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0
        
        # Execute up to max_rounds of tool calling
        while round_count < max_rounds and current_response.stop_reason == "tool_use":
            round_count += 1
            
            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            tool_execution_failed = False
            
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, 
                            **content_block.input
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Handle tool execution failure gracefully
                        tool_execution_failed = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}"
                        })
            
            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # If tool execution failed, terminate with error handling
            if tool_execution_failed:
                break
            
            # If this is the last allowed round, make final call without tools
            if round_count >= max_rounds:
                break
            
            # Prepare parameters for next round (keep tools available)
            next_round_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools"),
                "tool_choice": base_params.get("tool_choice")
            }
            
            try:
                # Get response for next potential round
                current_response = self.client.messages.create(**next_round_params)
            except Exception as e:
                # Handle API errors gracefully
                print(f"API error in round {round_count + 1}: {e}")
                break
        
        # Make final API call without tools to get concluding response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        try:
            # If we ended due to non-tool response, use that response
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text
            
            # Otherwise, make final call without tools
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            # Final fallback for API errors
            print(f"API error in final response: {e}")
            return "I encountered a technical issue while processing your request. Please try again."