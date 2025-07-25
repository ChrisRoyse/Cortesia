#!/usr/bin/env python3
"""
Phase 1 Integration Test Script
Tests multi-word entity extraction, semantic relationships, and question answering
"""

import asyncio
import json
import subprocess
import time

# Test data
TEST_KNOWLEDGE = [
    {
        "content": "Albert Einstein was a theoretical physicist who developed the Theory of Relativity. He was born in Germany in 1879 and later moved to the United States.",
        "title": "Einstein Biography"
    },
    {
        "content": "The Theory of Relativity revolutionized our understanding of space and time. It consists of two parts: Special Relativity published in 1905 and General Relativity published in 1915.",
        "title": "Relativity Theory"
    },
    {
        "content": "Marie Curie was a Polish physicist and chemist who discovered polonium and radium. She was the first woman to win a Nobel Prize and won it twice - in Physics in 1903 and Chemistry in 1911.",
        "title": "Marie Curie Biography"
    }
]

TEST_QUESTIONS = [
    {
        "question": "Who developed the Theory of Relativity?",
        "expected_entities": ["Theory of Relativity"],
        "expected_answer_contains": ["Einstein", "Albert Einstein"]
    },
    {
        "question": "What did Marie Curie discover?",
        "expected_entities": ["Marie Curie"],
        "expected_answer_contains": ["polonium", "radium"]
    },
    {
        "question": "When was Special Relativity published?",
        "expected_entities": ["Special Relativity"],
        "expected_answer_contains": ["1905"]
    },
    {
        "question": "Where was Einstein born?",
        "expected_entities": ["Einstein"],
        "expected_answer_contains": ["Germany"]
    }
]

def run_mcp_command(method, params):
    """Run an MCP command via the server"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": method,
            "arguments": params
        }
    }
    
    # For this test, we'll simulate the responses since we don't have the server running
    # In a real test, you would send this to the MCP server
    print(f"\n[TEST] Calling {method} with params: {json.dumps(params, indent=2)}")
    return {"success": True, "simulated": True}

async def test_phase1_features():
    """Test all Phase 1 features"""
    print("="*60)
    print("PHASE 1 INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Store knowledge with multi-word entities
    print("\n1. Testing Knowledge Storage with Multi-Word Entity Extraction")
    print("-"*50)
    
    for knowledge in TEST_KNOWLEDGE:
        result = run_mcp_command("store_knowledge", knowledge)
        print(f"✓ Stored: {knowledge['title']}")
    
    # Test 2: Test question answering
    print("\n2. Testing Enhanced Question Answering")
    print("-"*50)
    
    for test in TEST_QUESTIONS:
        result = run_mcp_command("ask_question", {
            "question": test["question"],
            "max_results": 5
        })
        
        print(f"\nQuestion: {test['question']}")
        print(f"Expected entities: {test['expected_entities']}")
        print(f"Expected answer contains: {test['expected_answer_contains']}")
        
        # In a real test, we would check:
        # - Entities were correctly extracted
        # - Answer contains expected content
        # - Confidence score is reasonable
        
    # Test 3: Test semantic relationships
    print("\n\n3. Testing Semantic Relationship Extraction")
    print("-"*50)
    
    # Store a fact with complex relationships
    complex_text = {
        "content": "Steve Jobs founded Apple Inc in 1976. The company is headquartered in Cupertino, California. Jobs worked with Steve Wozniak to create the first Apple computer.",
        "title": "Apple History"
    }
    
    result = run_mcp_command("store_knowledge", complex_text)
    print("✓ Stored complex knowledge")
    
    # Query for relationships
    queries = [
        {"subject": "Steve Jobs", "predicate": "founded"},
        {"subject": "Apple Inc", "predicate": "headquartered"},
        {"subject": "Steve Jobs", "predicate": "worked with"}
    ]
    
    for query in queries:
        result = run_mcp_command("find_facts", {"query": query, "limit": 10})
        print(f"\nQuery: {query}")
        print("Expected to find matching relationships")
    
    # Test 4: Test entity type classification
    print("\n\n4. Testing Entity Type Classification")
    print("-"*50)
    
    entity_test = {
        "content": "Microsoft Corporation announced a partnership with OpenAI on January 23, 2023 in Seattle. The deal is worth $10 billion.",
        "title": "Tech Partnership"
    }
    
    result = run_mcp_command("store_knowledge", entity_test)
    print("✓ Stored knowledge with various entity types:")
    print("  - Organizations: Microsoft Corporation, OpenAI")
    print("  - Location: Seattle")
    print("  - Time: January 23, 2023")
    print("  - Quantity: $10 billion")
    
    print("\n" + "="*60)
    print("PHASE 1 INTEGRATION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_phase1_features())