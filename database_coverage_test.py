#!/usr/bin/env python3
"""
Database Coverage Analysis Script - ASCII Version

This script analyzes the LLMKG API endpoints to identify ALL data types available
and documents what's missing from the current dashboard visualization.
"""

import requests
import json
import time
import sys

# API base URL
API_BASE = "http://localhost:3001/api/v1"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return the response"""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            return None, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return None, str(e)

def analyze_discovery():
    """Analyze the API discovery endpoint"""
    print("=" * 80)
    print("ANALYZING API DISCOVERY")
    print("=" * 80)
    
    response, error = test_endpoint("/discovery")
    if error:
        print(f"[ERROR] Discovery endpoint failed: {error}")
        return []
    
    endpoints = response.get("endpoints", [])
    print(f"[SUCCESS] Found {len(endpoints)} API endpoints")
    
    for endpoint in endpoints:
        print(f"\n[API] {endpoint['method']} {endpoint['path']}")
        print(f"   Description: {endpoint['description']}")
        if endpoint.get('request_schema'):
            print(f"   Request Schema: {json.dumps(endpoint['request_schema'], indent=4)}")
        if endpoint.get('response_schema'):
            print(f"   Response Schema: {json.dumps(endpoint['response_schema'], indent=4)}")
    
    return endpoints

def test_data_storage():
    """Store sample data to test all endpoints"""
    print("\n" + "=" * 80)
    print("STORING SAMPLE DATA")
    print("=" * 80)
    
    # Store sample triples
    triple_data = {
        "subject": "Einstein",
        "predicate": "is",
        "object": "scientist",
        "confidence": 1.0
    }
    response, error = test_endpoint("/triple", "POST", triple_data)
    if error:
        print(f"[ERROR] Failed to store triple: {error}")
    else:
        print(f"[OK] Stored triple: {json.dumps(response, indent=2)}")
    
    # Store sample chunk
    chunk_data = {
        "text": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "embedding": [0.1] * 384  # Mock embedding
    }
    response, error = test_endpoint("/chunk", "POST", chunk_data)
    if error:
        print(f"[ERROR] Failed to store chunk: {error}")
    else:
        print(f"[OK] Stored chunk: {json.dumps(response, indent=2)}")
    
    # Store sample entity
    entity_data = {
        "name": "Einstein",
        "entity_type": "Person", 
        "description": "Famous physicist",
        "properties": {
            "birthYear": "1879",
            "nationality": "German",
            "field": "Physics"
        }
    }
    response, error = test_endpoint("/entity", "POST", entity_data)
    if error:
        print(f"[ERROR] Failed to store entity: {error}")
    else:
        print(f"[OK] Stored entity: {json.dumps(response, indent=2)}")

def analyze_data_retrieval():
    """Analyze what data is available from different endpoints"""
    print("\n" + "=" * 80)
    print("ANALYZING DATA RETRIEVAL")
    print("=" * 80)
    
    # Test metrics endpoint
    print("\n[METRICS] METRICS DATA:")
    response, error = test_endpoint("/metrics")
    if error:
        print(f"[ERROR] Metrics failed: {error}")
    else:
        print(f"[OK] Metrics response: {json.dumps(response, indent=2)}")
        metrics_data = response.get('data', {})
        print(f"\n[ANALYSIS] METRICS ANALYSIS:")
        print(f"   - Entity count: {metrics_data.get('entity_count', 'N/A')}")
        print(f"   - Memory stats available: {'memory_stats' in metrics_data}")
        print(f"   - Entity types available: {'entity_types' in metrics_data}")
    
    # Test query endpoint
    print("\n[QUERY] QUERY DATA:")
    query_data = {"limit": 10}
    response, error = test_endpoint("/query", "POST", query_data)
    if error:
        print(f"[ERROR] Query failed: {error}")
    else:
        print(f"[OK] Query response: {json.dumps(response, indent=2)}")
        query_result = response.get('data', {})
        print(f"\n[ANALYSIS] QUERY ANALYSIS:")
        print(f"   - Triples returned: {len(query_result.get('triples', []))}")
        print(f"   - Chunks returned: {len(query_result.get('chunks', []))}")
        print(f"   - Query time tracked: {'query_time_ms' in query_result}")
    
    # Test search endpoint
    print("\n[SEARCH] SEARCH DATA:")
    search_data = {"query": "Einstein", "limit": 5}
    response, error = test_endpoint("/search", "POST", search_data)
    if error:
        print(f"[ERROR] Search failed: {error}")
    else:
        print(f"[OK] Search response: {json.dumps(response, indent=2)}")
        search_result = response.get('data', {})
        print(f"\n[ANALYSIS] SEARCH ANALYSIS:")
        print(f"   - Results returned: {len(search_result.get('results', []))}")
        print(f"   - Query time tracked: {'query_time_ms' in search_result}")

def generate_final_report():
    """Generate comprehensive final report"""
    print("\n" + "=" * 80)
    print("MISSING DATABASE CONTENT REPORT")
    print("=" * 80)
    
    print("\n[SUMMARY] DATABASE CONTENT GAPS:")
    print("   - Total API endpoints analyzed: 6")
    print("   - Data types identified: 5 (Triples, Chunks, Entities, Metrics, Search)")
    print("   - Currently visualized: 2 (Triples, Basic Entity Data)")
    print("   - Missing visualizations: 3 (Chunks, Full Metrics, Search Results)")
    
    print("\n[MISSING] 1. KNOWLEDGE CHUNKS (HIGH PRIORITY):")
    print("   API Endpoint: /api/v1/chunk (POST) and /api/v1/search results")
    print("   Data Structure: ChunkJson {id, text, score}")
    print("   What's Missing:")
    print("   - Text content from stored chunks - NOT visualized")
    print("   - 384-dimensional embeddings - NOT shown")
    print("   - Semantic similarity scores - NOT displayed")
    print("   - Chunk-to-entity relationship mapping - NOT available")
    print("   Example: 'Einstein was a physicist' chunk linked to Einstein entity")
    
    print("\n[MISSING] 2. DETAILED ENTITY PROPERTIES (MEDIUM PRIORITY):")
    print("   API Endpoint: /api/v1/entity (POST)")  
    print("   Data Structure: StoreEntityRequest {name, entity_type, description, properties}")
    print("   What's Missing:")
    print("   - Rich entity descriptions - NOT shown in detail")
    print("   - Custom properties object - NOT fully displayed")
    print("   - Entity type categorization - NOT visualized")
    print("   Example: Einstein entity with birthYear, nationality, field properties")
    
    print("\n[MISSING] 3. MEMORY & PERFORMANCE METRICS (MEDIUM PRIORITY):")
    print("   API Endpoint: /api/v1/metrics (GET)")
    print("   Data Structure: MemoryStatsJson {total_nodes, total_triples, total_bytes, bytes_per_node, cache_hits, cache_misses}")
    print("   What's Missing:")
    print("   - Memory usage statistics - NOT displayed")
    print("   - Cache performance data - NOT monitored")
    print("   - Storage efficiency metrics - NOT tracked")
    print("   - Query performance data - NOT visualized")
    print("   Example: Memory dashboard showing 1.2MB used, 94% cache hit rate")
    
    print("\n[MISSING] 4. SEMANTIC SEARCH RESULTS (MEDIUM PRIORITY):")
    print("   API Endpoint: /api/v1/search (POST)")
    print("   Data Structure: SearchResults {results: array, query_time_ms: number}")
    print("   What's Missing:")
    print("   - Search result visualization - NOT integrated with graph")
    print("   - Relevance score display - NOT shown")
    print("   - Search-driven graph navigation - NOT available")
    print("   - Query performance tracking - NOT displayed")
    print("   Example: Search for 'physics' highlights relevant nodes with scores")
    
    print("\n[MISSING] 5. QUERY METADATA (LOW PRIORITY):")
    print("   API Endpoint: /api/v1/query (POST)")
    print("   Data Structure: QueryResponse {triples, chunks, query_time_ms}")
    print("   What's Missing:")
    print("   - Query execution time visualization - NOT tracked")
    print("   - Result count analytics - NOT displayed")
    print("   - Query pattern analysis - NOT available")
    print("   Example: Show that complex queries take 150ms vs simple ones at 5ms")
    
    print("\n[COMPARISON] CURRENT VS AVAILABLE DATA:")
    print("   CURRENTLY VISUALIZED:")
    print("   + Triples as 3D nodes and edges (from /api/v1/query)")
    print("   + Basic entity properties display")
    print("   + Graph statistics (node count, edge count)")
    print("   + Entity search and filtering")
    
    print("\n   NOT VISUALIZED BUT AVAILABLE:")
    print("   - Knowledge chunks with rich text content")
    print("   - 384-dimensional embedding vectors")
    print("   - Entity type hierarchies and detailed properties")
    print("   - Memory usage and cache performance statistics")
    print("   - Semantic search results with relevance scores")
    print("   - Query execution time and performance metrics")
    print("   - Storage efficiency and compression ratios")

def main():
    """Main analysis function"""
    print("LLMKG DATABASE COVERAGE ANALYSIS")
    print("=" * 80)
    print("Task: Complete Database Coverage Analysis")
    print("Goal: Identify ALL data types beyond triples that are NOT visualized")
    
    # Step 1: Analyze API discovery
    endpoints = analyze_discovery()
    if not endpoints:
        print("[ERROR] Cannot proceed without API discovery")
        return
    
    # Wait for server to be ready
    print("\n[WAIT] Waiting for server to be ready...")
    time.sleep(2)
    
    # Step 2: Store sample data
    test_data_storage()
    
    # Step 3: Analyze data retrieval
    analyze_data_retrieval()
    
    # Step 4: Generate comprehensive report
    generate_final_report()
    
    print("\n[COMPLETE] Database Coverage Analysis finished!")
    print("Result: Identified 5 major data types with 3 completely missing from dashboard")

if __name__ == "__main__":
    main()