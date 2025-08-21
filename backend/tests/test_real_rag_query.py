#!/usr/bin/env python3
"""
Test the real RAG system with actual queries to identify the "query failed" issue.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem


def test_real_rag_query():
    """Test a real query through the RAG system."""
    print("=== TESTING REAL RAG SYSTEM QUERY ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ No ANTHROPIC_API_KEY found in environment")
        print("Set ANTHROPIC_API_KEY in .env file to test with real API")
        return False
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(config)
        
        # Test query
        print("Executing query: 'What is Python programming?'")
        response, sources = rag_system.query("What is Python programming?")
        
        print(f"\nResponse: {response}")
        print(f"Sources: {sources}")
        print(f"Response length: {len(response) if response else 0}")
        print(f"Number of sources: {len(sources) if sources else 0}")
        
        # Check for failure indicators
        if not response:
            print("❌ Empty response")
            return False
        elif "query failed" in response.lower():
            print("❌ Response contains 'query failed'")
            return False
        elif len(response.strip()) < 10:
            print("❌ Response too short, might be an error")
            return False
        else:
            print("✅ Query completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Exception during query: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_queries():
    """Test multiple different types of queries."""
    print("\n=== TESTING MULTIPLE QUERY TYPES ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ No API key available for testing")
        return False
        
    queries = [
        "What is Python?",
        "How do I use MCP?",
        "Tell me about computer use with Claude",
        "What is retrieval augmented generation?",
        "This query should not match any content xyzabc123"
    ]
    
    results = []
    
    try:
        rag_system = RAGSystem(config)
        
        for query in queries:
            print(f"\nTesting: '{query}'")
            try:
                response, sources = rag_system.query(query)
                success = response and "query failed" not in response.lower()
                results.append(success)
                print(f"  Result: {'✅ Success' if success else '❌ Failed'}")
                print(f"  Response length: {len(response) if response else 0}")
                print(f"  Sources: {len(sources) if sources else 0}")
                if not success:
                    print(f"  Response: {response}")
            except Exception as e:
                results.append(False)
                print(f"  ❌ Exception: {e}")
        
        success_rate = sum(results) / len(results)
        print(f"\nSuccess rate: {success_rate:.1%} ({sum(results)}/{len(results)})")
        
        return success_rate > 0.8
        
    except Exception as e:
        print(f"❌ Exception in multiple query test: {e}")
        return False


def main():
    """Run real query tests."""
    print("🧪 TESTING REAL RAG SYSTEM")
    print("=" * 50)
    
    single_query_works = test_real_rag_query()
    multiple_queries_work = test_multiple_queries()
    
    print("\n" + "=" * 50)
    print("🧪 REAL QUERY TEST SUMMARY")
    print(f"Single query works: {'✅' if single_query_works else '❌'}")
    print(f"Multiple queries work: {'✅' if multiple_queries_work else '❌'}")
    
    if not single_query_works or not multiple_queries_work:
        print("\n🎯 LIKELY ISSUE: The RAG system may be failing at the AI API level")
    else:
        print("\n✅ RAG system appears to work correctly - issue might be in frontend/API layer")


if __name__ == "__main__":
    main()