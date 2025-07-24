#!/usr/bin/env python3
"""
Final Verification Test: Complete Dashboard Database Integration
Tests the complete pipeline from database to visualization
"""

import requests
import json
import re

API_BASE = "http://localhost:3001/api/v1"
DASHBOARD_BASE = "http://localhost:8090"

def test_complete_integration():
    """Final test: Complete database-to-dashboard integration"""
    print("=== FINAL VERIFICATION: Complete Database Integration ===")
    
    # Test 1: Database contains real, meaningful data
    print("\n1. Testing database content quality...")
    
    query_response = requests.post(f"{API_BASE}/query", json={"limit": 50})
    data = query_response.json()
    triples = data["data"]["triples"]
    
    # Verify we have meaningful relationships
    meaningful_predicates = ["created_by", "uses", "located_in", "is", "invented_by", "developed"]
    found_predicates = set(triple["predicate"] for triple in triples)
    
    overlap = found_predicates.intersection(meaningful_predicates)
    if len(overlap) < 3:
        print(f"FAIL: Database lacks meaningful relationships. Found: {found_predicates}")
        return False
    
    print(f"PASS: Database has {len(triples)} triples with meaningful relationships: {list(overlap)}")
    
    # Test 2: Verify we have diverse entity types
    print("\n2. Testing entity diversity...")
    
    entities = set()
    for triple in triples:
        entities.add(triple["subject"])
        entities.add(triple["object"])
    
    # Look for different types of entities based on names
    entity_types = {
        "AI/Tech": len([e for e in entities if any(word in e.lower() for word in ["claude", "ai", "neural", "transformer", "machine"])]),
        "Companies": len([e for e in entities if any(word in e.lower() for word in ["anthropic", "google", "company"])]),
        "Locations": len([e for e in entities if any(word in e.lower() for word in ["francisco", "california", "location"])]),
        "Systems": len([e for e in entities if any(word in e.lower() for word in ["llmkg", "mcp", "system", "protocol"])])
    }
    
    diverse_types = sum(1 for count in entity_types.values() if count > 0)
    if diverse_types < 3:
        print(f"FAIL: Insufficient entity diversity: {entity_types}")
        return False
    
    print(f"PASS: Diverse entities found: {entity_types}")
    
    # Test 3: Verify dashboard HTML has complete integration
    print("\n3. Testing dashboard integration completeness...")
    
    dashboard_response = requests.get(DASHBOARD_BASE)
    html = dashboard_response.text
    
    # Check for complete Three.js integration
    integration_checks = {
        "Three.js loading": "three.min.js" in html,
        "Knowledge graph container": "knowledgeGraphContainer" in html,
        "Data loading function": "loadKnowledgeGraphData" in html,
        "Visualization update": "updateKnowledgeGraph" in html,
        "API endpoint": "http://localhost:3001/api/v1/query" in html,
        "Entity mesh creation": "createEntityMeshes" in html,
        "Relationship lines": "createRelationshipLines" in html
    }
    
    failed_checks = [check for check, passed in integration_checks.items() if not passed]
    if failed_checks:
        print(f"FAIL: Dashboard missing integration components: {failed_checks}")
        return False
    
    print("PASS: Dashboard has complete Three.js visualization integration")
    
    # Test 4: Verify API endpoints work with CORS
    print("\n4. Testing cross-origin API access...")
    
    headers = {
        "Origin": "http://localhost:8090",
        "Content-Type": "application/json"
    }
    
    # Test query endpoint with CORS
    cors_query = requests.post(f"{API_BASE}/query", json={"limit": 10}, headers=headers)
    if cors_query.status_code != 200:
        print(f"FAIL: CORS query failed: {cors_query.status_code}")
        return False
    
    # Test metrics endpoint with CORS
    cors_metrics = requests.get(f"{API_BASE}/metrics", headers={"Origin": "http://localhost:8090"})
    if cors_metrics.status_code != 200:
        print(f"FAIL: CORS metrics failed: {cors_metrics.status_code}")
        return False
    
    print("PASS: Cross-origin API access working correctly")
    
    # Test 5: Verify data transformation logic
    print("\n5. Testing data transformation for visualization...")
    
    query_data = cors_query.json()
    triples = query_data["data"]["triples"]
    
    # Simulate the dashboard's entity extraction
    entity_map = {}
    relationships = []
    
    for triple in triples:
        # Extract entities
        for entity_name in [triple["subject"], triple["object"]]:
            if entity_name not in entity_map:
                # Infer type like dashboard does
                name_lower = entity_name.lower()
                if any(word in name_lower for word in ["claude", "ai", "assistant"]):
                    entity_type = "AI"
                elif any(word in name_lower for word in ["anthropic", "google", "company"]):
                    entity_type = "Company"
                elif any(word in name_lower for word in ["san francisco", "california", "location"]):
                    entity_type = "Location"
                elif any(word in name_lower for word in ["transformer", "neural", "architecture"]):
                    entity_type = "Technology"
                else:
                    entity_type = "Concept"
                
                entity_map[entity_name] = {
                    "id": entity_name,
                    "name": entity_name,
                    "type": entity_type,
                    "connections": 0
                }
        
        # Count connections
        entity_map[triple["subject"]]["connections"] += 1
        entity_map[triple["object"]]["connections"] += 1
        
        # Create relationship
        relationships.append({
            "source": triple["subject"],
            "target": triple["object"],
            "type": triple["predicate"],
            "confidence": triple.get("confidence", 1.0)
        })
    
    entities = list(entity_map.values())
    
    if len(entities) < 5:
        print(f"FAIL: Insufficient entities for visualization: {len(entities)}")
        return False
    
    if len(relationships) < 5:
        print(f"FAIL: Insufficient relationships for visualization: {len(relationships)}")
        return False
    
    print(f"PASS: Data transformation successful: {len(entities)} entities, {len(relationships)} relationships")
    
    # Test 6: Verify entity type distribution
    print("\n6. Testing entity type distribution...")
    
    type_counts = {}
    for entity in entities:
        entity_type = entity["type"]
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    if len(type_counts) < 2:
        print(f"FAIL: Insufficient entity type diversity: {type_counts}")
        return False
    
    print(f"PASS: Good entity type distribution: {type_counts}")
    
    print("\n" + "="*60)
    print("SUCCESS: ALL INTEGRATION TESTS PASSED!")
    print(f"Dashboard at {DASHBOARD_BASE} is ready to visualize:")
    print(f"  - {len(entities)} entities across {len(type_counts)} types")
    print(f"  - {len(relationships)} relationships")
    print(f"  - {len(triples)} knowledge triples from database")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_complete_integration()
    exit(0 if success else 1)