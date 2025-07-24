#!/usr/bin/env python3
"""
TDD Test: Dashboard Database Integration
Tests that the dashboard can fetch and display real database content.
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:3001/api/v1"
DASHBOARD_BASE = "http://localhost:8090"

def test_api_server_is_running():
    """Test 1: FAILING - API server should be accessible"""
    try:
        response = requests.get(f"{API_BASE}/discovery", timeout=5)
        assert response.status_code == 200, f"API server not responding: {response.status_code}"
        print("PASS: API server is running")
        return True
    except Exception as e:
        print(f"FAIL: API server not accessible: {e}")
        return False

def test_dashboard_server_is_running():
    """Test 2: FAILING - Dashboard server should be accessible"""
    try:
        response = requests.get(DASHBOARD_BASE, timeout=5)
        assert response.status_code == 200, f"Dashboard not responding: {response.status_code}"
        print("✅ PASS: Dashboard server is running")
        return True
    except Exception as e:
        print(f"❌ FAIL: Dashboard not accessible: {e}")
        return False

def test_database_has_real_data():
    """Test 3: FAILING - Database should contain actual data"""
    try:
        # Test query endpoint
        response = requests.post(f"{API_BASE}/query", 
                               json={"limit": 10}, 
                               timeout=5)
        assert response.status_code == 200, f"Query failed: {response.status_code}"
        
        data = response.json()
        assert data["status"] == "success", f"Query unsuccessful: {data}"
        assert "data" in data, "No data field in response"
        assert "triples" in data["data"], "No triples in response"
        
        triples = data["data"]["triples"]
        assert len(triples) > 0, f"No triples found in database: {triples}"
        
        # Verify triple structure
        for triple in triples[:3]:  # Check first 3
            assert "subject" in triple, f"Triple missing subject: {triple}"
            assert "predicate" in triple, f"Triple missing predicate: {triple}"
            assert "object" in triple, f"Triple missing object: {triple}"
        
        print(f"✅ PASS: Database contains {len(triples)} triples")
        print(f"   Sample: {triples[0]['subject']} {triples[0]['predicate']} {triples[0]['object']}")
        return triples
    except Exception as e:
        print(f"❌ FAIL: Database data test failed: {e}")
        return []

def test_metrics_endpoint_working():
    """Test 4: FAILING - Metrics endpoint should return database statistics"""
    try:
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        assert response.status_code == 200, f"Metrics failed: {response.status_code}"
        
        data = response.json()
        assert data["status"] == "success", f"Metrics unsuccessful: {data}"
        assert "data" in data, "No data in metrics response"
        
        metrics = data["data"]
        assert "entity_count" in metrics, "No entity count in metrics"
        assert metrics["entity_count"] > 0, f"No entities in database: {metrics['entity_count']}"
        
        print(f"✅ PASS: Metrics shows {metrics['entity_count']} entities")
        return metrics
    except Exception as e:
        print(f"❌ FAIL: Metrics test failed: {e}")
        return {}

def test_dashboard_can_fetch_data():
    """Test 5: FAILING - Dashboard should successfully fetch data from API"""
    try:
        # Simulate what the dashboard JavaScript does
        response = requests.post(f"{API_BASE}/query",
                               json={"limit": 100},
                               headers={"Origin": "http://localhost:8090"},
                               timeout=5)
        
        assert response.status_code == 200, f"Dashboard data fetch failed: {response.status_code}"
        
        data = response.json()
        assert data["status"] == "success", "Dashboard fetch unsuccessful"
        assert len(data["data"]["triples"]) > 0, "Dashboard received no data"
        
        print(f"✅ PASS: Dashboard can fetch {len(data['data']['triples'])} triples")
        return True
    except Exception as e:
        print(f"❌ FAIL: Dashboard data fetch test failed: {e}")
        return False

def run_all_tests():
    """Run all tests - this should FAIL initially (TDD Red phase)"""
    print("🔴 TDD RED PHASE: Running failing tests...")
    print("=" * 50)
    
    results = {
        "api_running": test_api_server_is_running(),
        "dashboard_running": test_dashboard_server_is_running(),
        "database_has_data": test_database_has_real_data(),
        "metrics_working": test_metrics_endpoint_working(),
        "dashboard_fetch": test_dashboard_can_fetch_data()
    }
    
    print("\n" + "=" * 50)
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    if passed == total:
        print(f"🟢 ALL TESTS PASSED ({passed}/{total}) - Ready for GREEN phase!")
        return True
    else:
        print(f"🔴 TESTS FAILING ({passed}/{total}) - Need to implement fixes")
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
        return False

if __name__ == "__main__":
    # Wait a moment for server to fully start
    print("Waiting for server to start...")
    time.sleep(3)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)