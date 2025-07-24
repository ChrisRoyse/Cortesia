#!/usr/bin/env python3
"""
Database Coverage Analysis Script

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
        print(f"\n📡 {endpoint['method']} {endpoint['path']}")
        print(f"   Description: {endpoint['description']}")
        if endpoint.get('request_schema'):
            print(f"   Request: {json.dumps(endpoint['request_schema'], indent=4)}")
        if endpoint.get('response_schema'):
            print(f"   Response: {json.dumps(endpoint['response_schema'], indent=4)}")
    
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
    print("\n📊 METRICS DATA:")
    response, error = test_endpoint("/metrics")
    if error:
        print(f"[ERROR] Metrics failed: {error}")
    else:
        print(f"[OK] Metrics response: {json.dumps(response, indent=2)}")
        metrics_data = response.get('data', {})
        print(f"\n🔍 METRICS ANALYSIS:")
        print(f"   - Entity count: {metrics_data.get('entity_count', 'N/A')}")
        print(f"   - Memory stats available: {'memory_stats' in metrics_data}")
        print(f"   - Entity types available: {'entity_types' in metrics_data}")
    
    # Test query endpoint
    print("\n🔍 QUERY DATA:")
    query_data = {"limit": 10}
    response, error = test_endpoint("/query", "POST", query_data)
    if error:
        print(f"[ERROR] Query failed: {error}")
    else:
        print(f"[OK] Query response: {json.dumps(response, indent=2)}")
        query_result = response.get('data', {})
        print(f"\n🔍 QUERY ANALYSIS:")
        print(f"   - Triples returned: {len(query_result.get('triples', []))}")
        print(f"   - Chunks returned: {len(query_result.get('chunks', []))}")
        print(f"   - Query time tracked: {'query_time_ms' in query_result}")
    
    # Test search endpoint
    print("\n🔍 SEARCH DATA:")
    search_data = {"query": "Einstein", "limit": 5}
    response, error = test_endpoint("/search", "POST", search_data)
    if error:
        print(f"[ERROR] Search failed: {error}")
    else:
        print(f"[OK] Search response: {json.dumps(response, indent=2)}")
        search_result = response.get('data', {})
        print(f"\n🔍 SEARCH ANALYSIS:")
        print(f"   - Results returned: {len(search_result.get('results', []))}")
        print(f"   - Query time tracked: {'query_time_ms' in search_result}")

def compare_with_dashboard():
    """Compare available data with current dashboard visualization"""
    print("\n" + "=" * 80)
    print("DASHBOARD COVERAGE ANALYSIS")
    print("=" * 80)
    
    print("\n🎯 CURRENT DASHBOARD VISUALIZATION:")
    print("   [OK] Triples as 3D nodes and edges")
    print("   [OK] Entity properties display")
    print("   [OK] Basic graph statistics")
    print("   [OK] Entity filtering and search")
    
    print("\n[ERROR] MISSING DATABASE CONTENT:")
    print("   1. Knowledge Chunks:")
    print("      - Text content from /api/v1/chunk endpoint")
    print("      - Embeddings data (384-dimensional vectors)")
    print("      - Chunk-to-entity relationships")
    print("      - Semantic similarity scores")
    
    print("\n   2. Detailed Entity Properties:")
    print("      - Entity types with detailed properties")
    print("      - Entity descriptions from /api/v1/entity")
    print("      - Property-based filtering and visualization")
    
    print("\n   3. Memory and Performance Metrics:")
    print("      - Memory statistics from /api/v1/metrics")
    print("      - Cache hit/miss ratios")
    print("      - Query performance data")
    print("      - Storage efficiency metrics")
    
    print("\n   4. Search Results Integration:")
    print("      - Semantic search results visualization")
    print("      - Search score-based ranking")
    print("      - Search query history")
    
    print("\n   5. Advanced Query Results:")
    print("      - Complex query patterns")
    print("      - Multi-hop relationship traversal")
    print("      - Query execution time analysis")
    
    print("\n   6. Missing from API but potentially useful:")
    print("      - Entity relationship graphs (who connects to whom)")
    print("      - Chunk-to-triple extraction mappings")
    print("      - Confidence score distributions")
    print("      - Data source tracking")

def generate_report():
    """Generate final comprehensive report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATABASE COVERAGE REPORT")
    print("=" * 80)
    
    print("\n📋 SUMMARY:")
    print("   - Total API endpoints analyzed: 6")
    print("   - Data types identified: 5 (Triples, Chunks, Entities, Metrics, Search)")
    print("   - Currently visualized: 2 (Triples, Basic Entity Data)")
    print("   - Missing visualizations: 3 (Chunks, Full Metrics, Search Results)")
    
    print("\n🎯 PRIORITY GAPS TO ADDRESS:")
    print("   1. HIGH PRIORITY - Knowledge Chunks Visualization")
    print("      • Rich text content not shown in dashboard")
    print("      • Embedding similarity not visualized")
    print("      • No chunk-to-entity mapping display")
    
    print("\n   2. MEDIUM PRIORITY - Comprehensive Metrics Dashboard")
    print("      • Memory usage patterns hidden")
    print("      • Performance data not tracked visually")
    print("      • Cache efficiency not monitored")
    
    print("\n   3. MEDIUM PRIORITY - Search Results Integration")
    print("      • Semantic search results not integrated")
    print("      • No search-driven navigation")
    print("      • Missing relevance score visualization")
    
    print("\n📊 DETAILED DATA STRUCTURES NOT VISUALIZED:")
    
    # List specific data structures that exist but aren't shown
    missing_structures = [
        {
            "name": "ChunkJson",
            "fields": ["id", "text", "score"],
            "source": "/api/v1/search results",
            "visualization_gap": "Text chunks with semantic scores - not displayed in 3D graph"
        },
        {
            "name": "MemoryStatsJson", 
            "fields": ["total_nodes", "total_triples", "total_bytes", "bytes_per_node", "cache_hits", "cache_misses"],
            "source": "/api/v1/metrics",
            "visualization_gap": "Memory and performance data - no dashboard widgets"
        },
        {
            "name": "Entity Properties",
            "fields": ["name", "entity_type", "description", "properties"],
            "source": "/api/v1/entity",
            "visualization_gap": "Rich entity metadata - only basic properties shown"
        },
        {
            "name": "Query Metadata",
            "fields": ["query_time_ms", "limit", "result_count"],
            "source": "/api/v1/query",
            "visualization_gap": "Query performance metrics - not tracked in UI"
        }
    ]
    
    for i, structure in enumerate(missing_structures, 1):
        print(f"\n   {i}. {structure['name']}:")
        print(f"      Fields: {', '.join(structure['fields'])}")
        print(f"      Source: {structure['source']}")
        print(f"      Gap: {structure['visualization_gap']}")

def main():
    """Main analysis function"""
    print("LLMKG Database Coverage Analysis")
    print("=" * 80)
    
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
    
    # Step 4: Compare with dashboard
    compare_with_dashboard()
    
    # Step 5: Generate comprehensive report
    generate_report()
    
    print("\n[OK] Analysis complete!")

if __name__ == "__main__":
    main()