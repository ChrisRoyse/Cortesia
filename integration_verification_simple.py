#!/usr/bin/env python3
"""
Data Structure Integration Verification Test
Verifies that all data structures between modules are compatible and integrate correctly.
"""

import json
import sys
import subprocess
import time
from pathlib import Path

def run_rust_test(test_name):
    """Run specific Rust test and capture output."""
    try:
        result = subprocess.run(
            ["cargo", "test", test_name, "--", "--nocapture"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out"
    except Exception as e:
        return False, "", str(e)

def check_compilation_status():
    """Check if the project compiles successfully."""
    print("Checking compilation status...")
    
    try:
        result = subprocess.run(
            ["cargo", "check"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("  PASS: Project compiles successfully")
            return True
        else:
            print(f"  FAIL: Compilation errors found:")
            print(f"     {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("  FAIL: Compilation check timed out")
        return False
    except Exception as e:
        print(f"  FAIL: Compilation check failed: {e}")
        return False

def verify_triple_compatibility():
    """Verify Triple structure is compatible across modules."""
    print("Verifying Triple structure compatibility...")
    
    # Test that Triple can be created and used consistently
    success, stdout, stderr = run_rust_test("test_triple_new_valid_creation")
    
    if success:
        print("  PASS: Triple creation and basic operations work")
        triple_basic = True
    else:
        print(f"  FAIL: Triple compatibility issue: {stderr}")
        triple_basic = False
    
    # Test Triple hashing consistency (needed for temporal tracking)
    success, stdout, stderr = run_rust_test("test_triple_hash_consistency")
    
    if success:
        print("  PASS: Triple hashing is consistent for temporal tracking")
        triple_hash = True
    else:
        print(f"  FAIL: Triple hashing issue: {stderr}")
        triple_hash = False
        
    return triple_basic and triple_hash

def verify_knowledge_node_compatibility():
    """Verify KnowledgeNode structures work across modules."""
    print("Verifying KnowledgeNode compatibility...")
    
    # Test chunk node creation (used in storage and temporal tracking)
    success, stdout, stderr = run_rust_test("test_knowledge_node_new_chunk_valid")
    
    if success:
        print("  PASS: KnowledgeNode chunk creation works")
        chunk_ok = True
    else:
        print(f"  FAIL: KnowledgeNode chunk issue: {stderr}")
        chunk_ok = False
    
    # Test triple extraction (needed for temporal tracking)
    success, stdout, stderr = run_rust_test("test_knowledge_node_get_triples")
    
    if success:
        print("  PASS: Triple extraction from nodes works")
        extract_ok = True
    else:
        print(f"  FAIL: Triple extraction issue: {stderr}")
        extract_ok = False
        
    return chunk_ok and extract_ok

def verify_temporal_integration():
    """Verify temporal tracking integrates with storage."""
    print("Verifying temporal tracking integration...")
    
    # Test temporal recording works
    success, stdout, stderr = run_rust_test("test_temporal_recording")
    
    if success:
        print("  PASS: Temporal recording of operations works")
        temporal_record = True
    else:
        print(f"  FAIL: Temporal recording issue: {stderr}")
        temporal_record = False
    
    # Test point-in-time queries
    success, stdout, stderr = run_rust_test("test_point_in_time_query")
    
    if success:
        print("  PASS: Point-in-time queries work")
        temporal_query = True
    else:
        print(f"  FAIL: Point-in-time query issue: {stderr}")
        temporal_query = False
        
    return temporal_record and temporal_query

def verify_engine_query_integration():
    """Verify KnowledgeEngine queries integrate with other systems."""
    print("Verifying KnowledgeEngine query integration...")
    
    # Test basic storage and retrieval
    success, stdout, stderr = run_rust_test("test_store_triple_basic")
    
    if success:
        print("  PASS: Basic storage and retrieval works")
        storage_ok = True
    else:
        print(f"  FAIL: Storage issue: {stderr}")
        storage_ok = False
    
    # Test query filtering (needed for reasoning chains)
    success, stdout, stderr = run_rust_test("test_query_triples_subject_filter")
    
    if success:
        print("  PASS: Query filtering works for reasoning chains")
        query_ok = True
    else:
        print(f"  FAIL: Query filtering issue: {stderr}")
        query_ok = False
        
    return storage_ok and query_ok

def generate_integration_report(results):
    """Generate a detailed integration report."""
    print("Generating integration analysis report...")
    
    report = {
        "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_results": results,
        "data_flow_analysis": {
            "storage_to_temporal": {
                "description": "Triple objects flow from KnowledgeEngine to TemporalIndex",
                "compatibility": "Triple structure is consistent and hashable",
                "status": "Compatible" if results.get("triple_compatibility", False) else "Issues Found"
            },
            "temporal_to_query": {
                "description": "TemporalTriple data feeds back into query systems",
                "compatibility": "TemporalTriple wraps Triple maintaining structure",
                "status": "Compatible" if results.get("temporal_integration", False) else "Issues Found"
            },
            "engine_to_reasoning": {
                "description": "KnowledgeResult feeds into reasoning chain premises",
                "compatibility": "Query results provide structured Triple data",
                "status": "Compatible" if results.get("engine_integration", False) else "Issues Found"
            },
            "node_to_systems": {
                "description": "KnowledgeNode works across storage and retrieval",
                "compatibility": "Node content extraction maintains Triple integrity",
                "status": "Compatible" if results.get("node_compatibility", False) else "Issues Found"
            }
        },
        "integration_assessment": {
            "overall_status": "PASS" if all(results.values()) else "ISSUES_FOUND",
            "critical_issues": [],
            "recommendations": []
        }
    }
    
    # Add specific issues and recommendations
    if not results.get("compilation", False):
        report["integration_assessment"]["critical_issues"].append("Project compilation fails - blocking all integration")
        report["integration_assessment"]["recommendations"].append("Fix compilation errors before testing integration")
    
    if not results.get("triple_compatibility", False):
        report["integration_assessment"]["critical_issues"].append("Triple data structure incompatibilities")
        report["integration_assessment"]["recommendations"].append("Ensure Triple hashing and serialization work across modules")
    
    if not results.get("temporal_integration", False):
        report["integration_assessment"]["critical_issues"].append("Temporal tracking system not properly integrated")
        report["integration_assessment"]["recommendations"].append("Verify TemporalTriple wrapping and indexing functionality")
    
    if all(results.values()):
        report["integration_assessment"]["recommendations"].extend([
            "All data structures integrate correctly - ready for production",
            "Consider performance testing under load",
            "Add integration monitoring in production"
        ])
    
    # Save report
    with open("integration_verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("  Report saved to integration_verification_report.json")
    return report

def main():
    """Main verification function."""
    print("Starting Data Structure Integration Verification")
    print("=" * 60)
    
    results = {}
    
    # Step 1: Check compilation
    print("\nStep 1: Compilation Check")
    results["compilation"] = check_compilation_status()
    
    if not results["compilation"]:
        print("\nCompilation failed - cannot proceed with integration tests")
        generate_integration_report(results)
        return False
    
    # Step 2: Verify data structure compatibility
    print("\nStep 2: Data Structure Compatibility")
    results["triple_compatibility"] = verify_triple_compatibility()
    results["node_compatibility"] = verify_knowledge_node_compatibility()
    
    # Step 3: Verify integration points
    print("\nStep 3: Integration Point Verification")
    results["temporal_integration"] = verify_temporal_integration()
    results["engine_integration"] = verify_engine_query_integration()
    
    # Generate comprehensive report
    print("\nStep 4: Report Generation")
    report = generate_integration_report(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("SUCCESS: All integration tests passed!")
        print("- Data structures are compatible across all modules")
        print("- Integration points work correctly")
        print("- System is ready for end-to-end testing")
        return True
    else:
        print("ISSUES FOUND: Some integration tests failed")
        print("- Data structure integration issues detected")
        print("- Check the detailed report for specific problems")
        print("- Review compilation errors and fix before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)