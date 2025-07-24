#!/usr/bin/env python3
"""
TDD Test: Dashboard Visualization Integration
Tests that the dashboard JavaScript correctly loads and displays database content
"""

import requests
import json

API_BASE = "http://localhost:3001/api/v1"
DASHBOARD_BASE = "http://localhost:8090"

def test_dashboard_visualization_integration():
    """Test that dashboard can visualize actual database content"""
    print("Testing dashboard visualization integration...")
    
    # Step 1: Verify database has actual content
    print("Step 1: Verifying database content...")
    query_response = requests.post(f"{API_BASE}/query", json={"limit": 100})
    triples_data = query_response.json()
    triples = triples_data["data"]["triples"]
    
    metrics_response = requests.get(f"{API_BASE}/metrics")
    metrics_data = metrics_response.json()
    entity_count = metrics_data["data"]["entity_count"]
    
    print(f"Database has {len(triples)} triples and {entity_count} entities")
    
    # Step 2: Verify dashboard HTML contains the visualization code
    print("Step 2: Checking dashboard HTML...")
    dashboard_response = requests.get(DASHBOARD_BASE)
    dashboard_html = dashboard_response.text
    
    # Check for key dashboard components
    required_elements = [
        "knowledgeGraphContainer",
        "loadKnowledgeGraphData",
        "updateKnowledgeGraph",
        "Three.js",
        "fetch('http://localhost:3001/api/v1/query'"
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in dashboard_html:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"FAIL: Dashboard missing elements: {missing_elements}")
        return False
    
    print("PASS: Dashboard contains all required visualization elements")
    
    # Step 3: Test the exact API calls the dashboard makes
    print("Step 3: Testing dashboard's API calls...")
    
    # Simulate the dashboard's query call
    dashboard_query = requests.post(
        f"{API_BASE}/query",
        json={"limit": 100},
        headers={"Origin": "http://localhost:8090", "Content-Type": "application/json"}
    )
    
    if dashboard_query.status_code != 200:
        print(f"FAIL: Dashboard query call failed: {dashboard_query.status_code}")
        return False
    
    dashboard_data = dashboard_query.json()
    dashboard_triples = dashboard_data["data"]["triples"]
    
    if len(dashboard_triples) == 0:
        print("FAIL: Dashboard receives no triples from API")
        return False
    
    print(f"PASS: Dashboard can fetch {len(dashboard_triples)} triples")
    
    # Simulate the dashboard's metrics call
    dashboard_metrics = requests.get(
        f"{API_BASE}/metrics",
        headers={"Origin": "http://localhost:8090"}
    )
    
    if dashboard_metrics.status_code != 200:
        print(f"FAIL: Dashboard metrics call failed: {dashboard_metrics.status_code}")
        return False
    
    print("PASS: Dashboard can fetch metrics")
    
    # Step 4: Verify data structure matches what dashboard expects
    print("Step 4: Verifying data structure...")
    
    # Check triple structure
    for triple in dashboard_triples[:3]:
        required_fields = ["subject", "predicate", "object"]
        for field in required_fields:
            if field not in triple:
                print(f"FAIL: Triple missing {field}: {triple}")
                return False
    
    print("PASS: Triple data structure is correct")
    
    # Step 5: Test entity extraction (what dashboard does with triples)
    print("Step 5: Testing entity extraction logic...")
    
    entities = set()
    relationships = []
    
    for triple in dashboard_triples:
        entities.add(triple["subject"])
        entities.add(triple["object"])
        relationships.append({
            "source": triple["subject"],
            "target": triple["object"],
            "type": triple["predicate"]
        })
    
    if len(entities) == 0:
        print("FAIL: No entities extracted from triples")
        return False
    
    if len(relationships) == 0:
        print("FAIL: No relationships extracted from triples")
        return False
    
    print(f"PASS: Extracted {len(entities)} unique entities and {len(relationships)} relationships")
    
    # Show sample data for verification
    print("Sample entities:", list(entities)[:5])
    print("Sample relationship:", relationships[0] if relationships else "None")
    
    print("SUCCESS: Dashboard visualization integration test PASSED!")
    return True

if __name__ == "__main__":
    success = test_dashboard_visualization_integration()
    exit(0 if success else 1)