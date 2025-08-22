#!/usr/bin/env python3
"""
Test the FastAPI endpoint directly to identify where "query failed" comes from.
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from app import app
from fastapi.testclient import TestClient


def test_api_endpoint_directly():
    """Test the /api/query endpoint directly."""
    print("=== TESTING API ENDPOINT DIRECTLY ===")

    try:
        with TestClient(app) as client:
            # Test query
            response = client.post(
                "/api/query",
                json={"query": "What is Python programming?", "session_id": None},
            )

            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                print(f"Answer: {data.get('answer', 'No answer')}")
                print(f"Sources: {data.get('sources', 'No sources')}")
                print(f"Session ID: {data.get('session_id', 'No session')}")

                answer = data.get("answer", "")
                if "query failed" in answer.lower():
                    print("❌ API returned 'query failed'")
                    return False
                elif len(answer) > 100:
                    print("✅ API returned substantial response")
                    return True
                else:
                    print("⚠️ API returned short response")
                    return False
            else:
                print(f"❌ API returned error status: {response.status_code}")
                print(f"Error detail: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Exception testing API endpoint: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api_with_various_queries():
    """Test API with various query types."""
    print("\n=== TESTING API WITH VARIOUS QUERIES ===")

    queries = [
        "What is Python?",
        "How do I use MCP?",
        "",  # Empty query
        "   ",  # Whitespace only
        "a" * 1000,  # Very long query
        "What is quantum computing xyz123?",  # Query that shouldn't match
    ]

    results = []

    try:
        with TestClient(app) as client:
            for i, query in enumerate(queries, 1):
                print(f"\nTest {i}: '{query[:50]}{'...' if len(query) > 50 else ''}'")

                try:
                    response = client.post(
                        "/api/query",
                        json={"query": query, "session_id": f"test_session_{i}"},
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])

                        success = bool(answer and "query failed" not in answer.lower())
                        results.append(success)

                        print(f"  Status: {'✅ Success' if success else '❌ Failed'}")
                        print(f"  Answer length: {len(answer)}")
                        print(f"  Sources count: {len(sources)}")

                        if not success:
                            print(f"  Answer: {answer}")

                    else:
                        results.append(False)
                        print(f"  ❌ HTTP {response.status_code}: {response.text}")

                except Exception as e:
                    results.append(False)
                    print(f"  ❌ Exception: {e}")

            success_rate = sum(results) / len(results) if results else 0
            print(
                f"\nAPI Success Rate: {success_rate:.1%} ({sum(results)}/{len(results)})"
            )

            return success_rate > 0.7

    except Exception as e:
        print(f"❌ Exception in API testing: {e}")
        return False


def test_api_error_handling():
    """Test API error handling scenarios."""
    print("\n=== TESTING API ERROR HANDLING ===")

    try:
        with TestClient(app) as client:
            # Test malformed requests
            test_cases = [
                {},  # Empty body
                {"query": None},  # Null query
                {"not_query": "test"},  # Wrong field name
                {"query": ["not", "string"]},  # Wrong type
            ]

            for i, case in enumerate(test_cases, 1):
                print(f"\nError Test {i}: {case}")

                try:
                    response = client.post("/api/query", json=case)
                    print(f"  Status: {response.status_code}")

                    if response.status_code != 200:
                        print(f"  ✅ Correctly rejected bad request")
                    else:
                        data = response.json()
                        print(f"  ⚠️ Accepted bad request: {data}")

                except Exception as e:
                    print(f"  ❌ Exception on bad request: {e}")

            return True

    except Exception as e:
        print(f"❌ Exception in error handling test: {e}")
        return False


async def test_api_startup():
    """Test if the API startup process works correctly."""
    print("\n=== TESTING API STARTUP PROCESS ===")

    try:
        # The startup event should have been called when we imported app
        # Let's check if the system is properly initialized

        from app import rag_system

        analytics = rag_system.get_course_analytics()
        print(f"Courses loaded: {analytics['total_courses']}")
        print(f"Course titles: {analytics['course_titles']}")

        if analytics["total_courses"] > 0:
            print("✅ RAG system initialized correctly")
            return True
        else:
            print("❌ RAG system has no courses loaded")
            return False

    except Exception as e:
        print(f"❌ Exception checking startup: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all API tests."""
    print("🌐 TESTING API LAYER")
    print("=" * 50)

    startup_works = asyncio.run(test_api_startup())
    direct_api_works = test_api_endpoint_directly()
    various_queries_work = test_api_with_various_queries()
    error_handling_works = test_api_error_handling()

    print("\n" + "=" * 50)
    print("🌐 API TEST SUMMARY")
    print(f"Startup works: {'✅' if startup_works else '❌'}")
    print(f"Direct API works: {'✅' if direct_api_works else '❌'}")
    print(f"Various queries work: {'✅' if various_queries_work else '❌'}")
    print(f"Error handling works: {'✅' if error_handling_works else '❌'}")

    if not startup_works:
        print("\n🎯 ISSUE: API startup/initialization problem")
    elif not direct_api_works:
        print("\n🎯 ISSUE: API endpoint failing for normal queries")
    elif not various_queries_work:
        print("\n🎯 ISSUE: API failing for some query types")
    else:
        print("\n✅ API layer works correctly - issue must be in frontend")


if __name__ == "__main__":
    main()
