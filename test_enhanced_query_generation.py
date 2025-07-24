#!/usr/bin/env python3
"""
Integration test for enhanced generate_graph_query tool
Testing Phase 1.1 implementation
"""

import asyncio
import json
import sys
import io

# Configure UTF-8 encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Test cases for enhanced NLP entity extraction
test_cases = [
    {
        "query": "Find all facts about Einstein",
        "expected_entities": ["Einstein"],
        "expected_in_query": ["Einstein", "MATCH"],
        "language": "cypher"
    },
    {
        "query": "Show relationships between Einstein and Newton",
        "expected_entities": ["Einstein", "Newton"],
        "expected_in_query": ["Einstein", "Newton", "path"],
        "language": "cypher"
    },
    {
        "query": "What are Einstein's discoveries?",
        "expected_entities": ["Einstein"],
        "expected_in_query": ["Einstein"],
        "language": "cypher"
    },
    {
        "query": 'Find information about "Theory of Relativity"',
        "expected_entities": ["Theory of Relativity"],
        "expected_in_query": ["Theory of Relativity"],
        "language": "cypher"
    },
    {
        "query": "Show connections between Einstein and Tesla related to electricity",
        "expected_entities": ["Einstein", "Tesla", "electricity"],
        "expected_in_query": ["Einstein", "Tesla"],
        "language": "cypher"
    },
    {
        "query": "Find the shortest path between Einstein and Tesla",
        "expected_entities": ["Einstein", "Tesla"],
        "expected_in_query": ["shortestPath", "Einstein", "Tesla"],
        "language": "cypher"
    },
    {
        "query": "Who invented the telephone?",
        "expected_entities": ["telephone"],
        "expected_in_query": ["invented"],
        "language": "cypher"
    },
    {
        "query": "How many scientists are there?",
        "expected_entities": ["scientists"],
        "expected_in_query": ["COUNT"],
        "language": "cypher"
    },
    {
        "query": "Find all facts about Einstein",
        "expected_entities": ["Einstein"],
        "expected_in_query": ["SELECT", "WHERE", "Einstein"],
        "language": "sparql"
    },
    {
        "query": "Find all facts about Einstein",
        "expected_entities": ["Einstein"],
        "expected_in_query": ["g.V()", "Einstein"],
        "language": "gremlin"
    }
]

async def test_generate_graph_query():
    """Test the enhanced generate_graph_query tool via MCP"""
    try:
        import subprocess
        import requests
        import time
        
        print("🔍 Testing Enhanced generate_graph_query Tool")
        print("=" * 60)
        
        # First, let's test if MCP server is running
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                print("✅ MCP server is running")
            else:
                print("❌ MCP server returned status:", response.status_code)
                return
        except:
            print("❌ MCP server is not running. Please start it first.")
            print("Run: cargo run --bin llmkg_mcp_server")
            return
        
        # Test each case
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}: {test_case['query']}")
            print("-" * 40)
            
            # Prepare MCP request
            mcp_request = {
                "method": "generate_graph_query",
                "params": {
                    "natural_query": test_case["query"],
                    "query_language": test_case["language"],
                    "include_explanation": True
                }
            }
            
            try:
                # Send request to MCP server
                response = requests.post(
                    "http://localhost:8080/mcp/execute",
                    json=mcp_request,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check extracted entities
                    extracted_entities = result.get("result", {}).get("extracted_entities", [])
                    print(f"Extracted entities: {extracted_entities}")
                    
                    # Verify expected entities
                    entities_correct = all(
                        entity in extracted_entities 
                        for entity in test_case["expected_entities"]
                    )
                    
                    # Check generated query
                    generated_query = result.get("result", {}).get("generated_query", "")
                    print(f"Generated query: {generated_query[:100]}...")
                    
                    # Verify expected content in query
                    query_correct = all(
                        expected in generated_query 
                        for expected in test_case["expected_in_query"]
                    )
                    
                    if entities_correct and query_correct:
                        print("✅ Test passed!")
                        results.append(True)
                    else:
                        print("❌ Test failed!")
                        if not entities_correct:
                            print(f"   Missing entities: {[e for e in test_case['expected_entities'] if e not in extracted_entities]}")
                        if not query_correct:
                            print(f"   Missing in query: {[e for e in test_case['expected_in_query'] if e not in generated_query]}")
                        results.append(False)
                    
                    # Show explanation if available
                    explanation = result.get("result", {}).get("explanation", "")
                    if explanation:
                        print(f"Explanation: {explanation}")
                    
                else:
                    print(f"❌ Request failed with status: {response.status_code}")
                    print(f"Response: {response.text}")
                    results.append(False)
                    
            except Exception as e:
                print(f"❌ Error during test: {str(e)}")
                results.append(False)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print(f"Total tests: {len(results)}")
        print(f"Passed: {sum(results)}")
        print(f"Failed: {len(results) - sum(results)}")
        print(f"Success rate: {sum(results)/len(results)*100:.1f}%")
        
        if sum(results) == len(results):
            print("\n🎉 All tests passed! Phase 1.1 implementation is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Review the implementation.")
            
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_generate_graph_query())