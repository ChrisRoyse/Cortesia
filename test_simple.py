#!/usr/bin/env python3
"""
TDD Test: Simple Dashboard Database Integration Test
"""

import requests
import json
import time

API_BASE = "http://localhost:3001/api/v1"

def test_database_integration():
    """Test database has data and dashboard can access it"""
    print("TDD RED PHASE: Testing database integration...")
    
    try:
        # Test 1: API server running
        print("Test 1: Checking API server...")
        response = requests.get(f"{API_BASE}/discovery", timeout=5)
        if response.status_code != 200:
            print(f"FAIL: API server not responding: {response.status_code}")
            return False
        print("PASS: API server is running")
        
        # Test 2: Database has data
        print("Test 2: Checking database content...")
        response = requests.post(f"{API_BASE}/query", json={"limit": 10}, timeout=5)
        if response.status_code != 200:
            print(f"FAIL: Query failed: {response.status_code}")
            return False
            
        data = response.json()
        if data["status"] != "success":
            print(f"FAIL: Query unsuccessful: {data}")
            return False
            
        triples = data["data"]["triples"]
        if len(triples) == 0:
            print("FAIL: No triples found in database")
            return False
            
        print(f"PASS: Database contains {len(triples)} triples")
        print(f"Sample: {triples[0]['subject']} {triples[0]['predicate']} {triples[0]['object']}")
        
        # Test 3: Metrics endpoint
        print("Test 3: Checking metrics...")
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        if response.status_code != 200:
            print(f"FAIL: Metrics failed: {response.status_code}")
            return False
            
        metrics = response.json()["data"]
        print(f"PASS: Metrics shows {metrics['entity_count']} entities")
        
        print("All database integration tests PASSED!")
        return True
        
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        return False

if __name__ == "__main__":
    time.sleep(2)  # Wait for server
    success = test_database_integration()
    exit(0 if success else 1)