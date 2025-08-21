#!/usr/bin/env python3
"""
Diagnostic script to test the real RAG system and identify where "query failed" occurs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from models import Course, Lesson, CourseChunk


def test_vector_store_directly():
    """Test the vector store directly to see if it has data and can search."""
    print("=== TESTING VECTOR STORE DIRECTLY ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        # Check if we have any courses
        existing_titles = vector_store.get_existing_course_titles()
        course_count = vector_store.get_course_count()
        
        print(f"Existing course titles: {existing_titles}")
        print(f"Course count: {course_count}")
        
        if course_count == 0:
            print("❌ NO COURSES FOUND - This is likely the main issue!")
            return False
        
        # Test basic search
        print("\nTesting basic search...")
        results = vector_store.search("Python programming")
        print(f"Search results: {len(results.documents)} documents")
        print(f"Error: {results.error}")
        print(f"Is empty: {results.is_empty()}")
        
        if results.documents:
            print(f"First result: {results.documents[0][:100]}...")
            print(f"First metadata: {results.metadata[0]}")
        
        return len(results.documents) > 0
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False


def test_course_search_tool():
    """Test the CourseSearchTool directly."""
    print("\n=== TESTING COURSE SEARCH TOOL ===")
    
    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)
        
        # Test the execute method
        result = search_tool.execute("Python programming")
        print(f"CourseSearchTool result: {result}")
        print(f"Sources: {search_tool.last_sources}")
        
        # Check if result indicates failure
        if "No relevant content found" in result:
            print("❌ CourseSearchTool found no content")
            return False
        elif "error" in result.lower():
            print(f"❌ CourseSearchTool returned error: {result}")
            return False
        else:
            print("✅ CourseSearchTool returned content")
            return True
            
    except Exception as e:
        print(f"❌ CourseSearchTool error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ai_generator_mock():
    """Test AI generator with mock to see tool flow."""
    print("\n=== TESTING AI GENERATOR TOOL FLOW ===")
    
    try:
        from ai_generator import AIGenerator
        from unittest.mock import Mock
        
        # Create mock client that simulates tool use
        mock_client = Mock()
        
        # Mock initial response with tool use
        class MockContent:
            def __init__(self, content_type, **kwargs):
                self.type = content_type
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        tool_use_response = Mock()
        tool_use_response.content = [
            MockContent("tool_use", name="search_course_content", 
                       input={"query": "Python"}, id="tool_123")
        ]
        tool_use_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [MockContent("text", text="Python is a programming language.")]
        
        mock_client.messages.create.side_effect = [tool_use_response, final_response]
        
        # Create AI generator with mock
        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client
        
        # Create tool manager
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test with tools
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        print(f"AI Generator result: {result}")
        print(f"Tool was called: {mock_client.messages.create.call_count == 2}")
        
        return "Python is a programming language" in result
        
    except Exception as e:
        print(f"❌ AI Generator test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_system_components():
    """Test RAG system component by component."""
    print("\n=== TESTING RAG SYSTEM COMPONENTS ===")
    
    try:
        rag_system = RAGSystem(config)
        
        # Test component initialization
        print(f"Document processor: {rag_system.document_processor is not None}")
        print(f"Vector store: {rag_system.vector_store is not None}")
        print(f"AI generator: {rag_system.ai_generator is not None}")
        print(f"Session manager: {rag_system.session_manager is not None}")
        print(f"Tool manager: {rag_system.tool_manager is not None}")
        print(f"Search tool: {rag_system.search_tool is not None}")
        
        # Test tool definitions
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        print(f"Tool definitions: {len(tool_defs)}")
        
        if tool_defs:
            print(f"First tool: {tool_defs[0]['name']}")
        
        # Test analytics (shows if we have data)
        analytics = rag_system.get_course_analytics()
        print(f"Analytics: {analytics}")
        
        return analytics["total_courses"] > 0
        
    except Exception as e:
        print(f"❌ RAG system components error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_directory():
    """Check if the docs directory exists and has content."""
    print("\n=== CHECKING DATA DIRECTORY ===")
    
    docs_path = "../docs"
    if os.path.exists(docs_path):
        files = [f for f in os.listdir(docs_path) 
                if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        print(f"Found {len(files)} document files in {docs_path}")
        for f in files[:5]:  # Show first 5 files
            print(f"  - {f}")
        return len(files) > 0
    else:
        print(f"❌ Docs directory {docs_path} does not exist!")
        return False


def main():
    """Run all diagnostic tests."""
    print("🔍 RAG SYSTEM DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Check if we have training data
    has_data = check_data_directory()
    
    # Test each component
    vector_store_works = test_vector_store_directly()
    search_tool_works = test_course_search_tool()
    ai_tool_flow_works = test_ai_generator_mock()
    rag_components_work = test_rag_system_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSTIC SUMMARY")
    print(f"📁 Has training data: {'✅' if has_data else '❌'}")
    print(f"🗃️  Vector store works: {'✅' if vector_store_works else '❌'}")
    print(f"🔧 Search tool works: {'✅' if search_tool_works else '❌'}")
    print(f"🤖 AI tool flow works: {'✅' if ai_tool_flow_works else '❌'}")
    print(f"🏗️  RAG components work: {'✅' if rag_components_work else '❌'}")
    
    # Identify the likely issue
    if not has_data:
        print("\n🎯 LIKELY ISSUE: No training documents found in ../docs directory")
    elif not vector_store_works:
        print("\n🎯 LIKELY ISSUE: Vector store has no data or can't search")
    elif not search_tool_works:
        print("\n🎯 LIKELY ISSUE: CourseSearchTool is not working properly")
    elif not ai_tool_flow_works:
        print("\n🎯 LIKELY ISSUE: AI generator is not calling tools correctly")
    else:
        print("\n🎯 All components appear to work - the issue may be elsewhere")


if __name__ == "__main__":
    main()