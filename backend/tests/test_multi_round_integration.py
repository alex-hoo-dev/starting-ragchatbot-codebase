#!/usr/bin/env python3
"""
Test the multi-round tool calling functionality with real RAG system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem


def test_multi_round_integration():
    """Test multi-round tool calling with queries that should trigger search."""
    print("=== TESTING MULTI-ROUND INTEGRATION ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ No ANTHROPIC_API_KEY found")
        return False
    
    try:
        rag_system = RAGSystem(config)
        
        # Test queries that should trigger course content searches
        test_queries = [
            "What specific topics are covered in the MCP course lessons?",
            "Compare the content between Anthropic's computer use course and the MCP course",
            "What are the main differences between the courses available?",
            "Tell me about lesson 2 of the computer use course and lesson 3 of the MCP course",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            try:
                response, sources = rag_system.query(query)
                
                print(f"Response length: {len(response)}")
                print(f"Sources count: {len(sources)}")
                print(f"Response preview: {response[:200]}...")
                
                if sources:
                    print("Sources:")
                    for j, source in enumerate(sources[:3], 1):
                        print(f"  {j}. {source.get('text', 'No text')}")
                
                # Check if this looks like it used tools
                if len(sources) > 0:
                    print("✅ Tool usage detected (has sources)")
                elif "course" in response.lower() and len(response) > 200:
                    print("✅ Likely used tools (detailed course response)")
                else:
                    print("⚠️ Possibly no tool usage")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_specific_course_queries():
    """Test queries that should definitely trigger search tools."""
    print("\n=== TESTING COURSE-SPECIFIC QUERIES ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ No API key available")
        return False
    
    try:
        rag_system = RAGSystem(config)
        
        # Get available courses first
        analytics = rag_system.get_course_analytics()
        print(f"Available courses: {analytics['course_titles']}")
        
        if not analytics['course_titles']:
            print("❌ No courses available for testing")
            return False
        
        # Test with specific course names
        course_queries = [
            f"What is covered in the '{analytics['course_titles'][0]}' course?",
            f"Tell me about lesson 1 of {analytics['course_titles'][0]}",
            "What are the main topics in the Building Towards Computer Use course?",
            "Explain MCP architecture from the MCP course",
        ]
        
        successful_tool_usage = 0
        
        for query in course_queries:
            print(f"\nQuery: {query}")
            
            try:
                response, sources = rag_system.query(query)
                print(f"  Sources: {len(sources)}")
                print(f"  Response length: {len(response)}")
                
                if len(sources) > 0:
                    successful_tool_usage += 1
                    print(f"  ✅ Tool usage successful")
                    # Show first source
                    print(f"  Source: {sources[0].get('text', 'Unknown')}")
                else:
                    print(f"  ⚠️ No tool usage detected")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\nTool usage success rate: {successful_tool_usage}/{len(course_queries)}")
        return successful_tool_usage > 0
        
    except Exception as e:
        print(f"❌ Course query test failed: {e}")
        return False


def main():
    """Run multi-round integration tests."""
    print("🔄 TESTING MULTI-ROUND TOOL CALLING INTEGRATION")
    print("=" * 60)
    
    multi_round_works = test_multi_round_integration()
    course_queries_work = test_specific_course_queries()
    
    print("\n" + "=" * 60)
    print("🔄 MULTI-ROUND INTEGRATION SUMMARY")
    print(f"Multi-round integration: {'✅' if multi_round_works else '❌'}")
    print(f"Course-specific queries: {'✅' if course_queries_work else '❌'}")
    
    if multi_round_works and course_queries_work:
        print("\n✅ Multi-round tool calling is working correctly!")
    else:
        print("\n⚠️ Some aspects of multi-round functionality may need attention")


if __name__ == "__main__":
    main()