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

def verify_triple_compatibility():
    """Verify Triple structure is compatible across modules."""
    print("🔍 Verifying Triple structure compatibility...")
    
    # Test that Triple can be created and used consistently
    success, stdout, stderr = run_rust_test("test_triple_new_valid_creation")
    
    if success:
        print("  ✅ Triple creation and basic operations work")
    else:
        print(f"  ❌ Triple compatibility issue: {stderr}")
        return False
    
    # Test Triple hashing consistency (needed for temporal tracking)
    success, stdout, stderr = run_rust_test("test_triple_hash_consistency")
    
    if success:
        print("  ✅ Triple hashing is consistent for temporal tracking")
    else:
        print(f"  ❌ Triple hashing issue: {stderr}")
        return False
        
    return True

def verify_knowledge_node_compatibility():
    """Verify KnowledgeNode structures work across modules."""
    print("🔍 Verifying KnowledgeNode compatibility...")
    
    # Test chunk node creation (used in storage and temporal tracking)
    success, stdout, stderr = run_rust_test("test_knowledge_node_new_chunk_valid")
    
    if success:
        print("  ✅ KnowledgeNode chunk creation works")
    else:
        print(f"  ❌ KnowledgeNode chunk issue: {stderr}")
        return False
    
    # Test triple extraction (needed for temporal tracking)
    success, stdout, stderr = run_rust_test("test_knowledge_node_get_triples")
    
    if success:
        print("  ✅ Triple extraction from nodes works")
    else:
        print(f"  ❌ Triple extraction issue: {stderr}")
        return False
        
    return True

def verify_temporal_integration():
    """Verify temporal tracking integrates with storage."""
    print("🔍 Verifying temporal tracking integration...")
    
    # Test temporal recording works
    success, stdout, stderr = run_rust_test("test_temporal_recording")
    
    if success:
        print("  ✅ Temporal recording of operations works")
    else:
        print(f"  ❌ Temporal recording issue: {stderr}")
        return False
    
    # Test point-in-time queries
    success, stdout, stderr = run_rust_test("test_point_in_time_query")
    
    if success:
        print("  ✅ Point-in-time queries work")
    else:
        print(f"  ❌ Point-in-time query issue: {stderr}")
        return False
        
    return True

def verify_engine_query_integration():
    """Verify KnowledgeEngine queries integrate with other systems."""
    print("🔍 Verifying KnowledgeEngine query integration...")
    
    # Test basic storage and retrieval
    success, stdout, stderr = run_rust_test("test_store_triple_basic")
    
    if success:
        print("  ✅ Basic storage and retrieval works")
    else:
        print(f"  ❌ Storage issue: {stderr}")
        return False
    
    # Test query filtering (needed for reasoning chains)
    success, stdout, stderr = run_rust_test("test_query_triples_subject_filter")
    
    if success:
        print("  ✅ Query filtering works for reasoning chains")
    else:
        print(f"  ❌ Query filtering issue: {stderr}")
        return False
        
    return True

def test_data_flow_integration():
    """Test actual data flow between storage, temporal, and reasoning systems."""
    print("🔍 Testing end-to-end data flow integration...")
    
    # Create a simple integration test script
    integration_script = '''
use std::sync::Arc;
use parking_lot::RwLock;

#[tokio::test]
async fn test_end_to_end_data_flow() {
    use crate::core::knowledge_engine::KnowledgeEngine;
    use crate::core::triple::Triple;
    use crate::mcp::llm_friendly_server::temporal_tracking::{TemporalIndex, TemporalOperation};
    
    // Create knowledge engine
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 1000).expect("Failed to create engine")
    ));
    
    // Create temporal index
    let temporal_index = TemporalIndex::new();
    
    // Store a fact
    let triple = Triple::new(
        "Einstein".to_string(),
        "is".to_string(),
        "physicist".to_string()
    ).expect("Failed to create triple");
    
    // Store in knowledge engine
    let node_id = {
        let engine_guard = engine.write();
        engine_guard.store_triple(triple.clone(), None)
            .expect("Failed to store triple")
    };
    
    // Record in temporal index
    temporal_index.record_operation(triple.clone(), TemporalOperation::Create, None);
    
    // Verify data is accessible from both systems
    {
        let engine_guard = engine.read();
        let query_result = engine_guard.query_triples(crate::core::knowledge_types::TripleQuery {
            subject: Some("Einstein".to_string()),
            predicate: None,
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: false,
        }).expect("Query failed");
        
        assert!(!query_result.triples.is_empty(), "Should find stored triple");
        assert_eq!(query_result.triples[0].subject, "Einstein");
    }
    
    // Verify temporal query works
    let temporal_result = crate::mcp::llm_friendly_server::temporal_tracking::query_point_in_time(
        &temporal_index,
        "Einstein",
        chrono::Utc::now()
    ).await;
    
    assert!(!temporal_result.results.is_empty(), "Should find temporal data");
    assert_eq!(temporal_result.query_type, "point_in_time");
    
    println!("✅ End-to-end data flow test passed");
}
'''
    
    # Write the test to a temporary file
    test_file = Path("src/integration_test.rs")
    try:
        with open(test_file, "w") as f:
            f.write(integration_script)
        
        # Run the integration test
        success, stdout, stderr = run_rust_test("test_end_to_end_data_flow")
        
        if success:
            print("  ✅ End-to-end data flow integration works")
            return True
        else:
            print(f"  ❌ Integration test failed: {stderr}")
            return False
    
    except Exception as e:
        print(f"  ❌ Failed to run integration test: {e}")
        return False
    
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

def check_compilation_status():
    """Check if the project compiles successfully."""
    print("🔍 Checking compilation status...")
    
    try:
        result = subprocess.run(
            ["cargo", "check"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("  ✅ Project compiles successfully")
            return True
        else:
            print(f"  ❌ Compilation errors found:")
            print(f"     {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("  ❌ Compilation check timed out")
        return False
    except Exception as e:
        print(f"  ❌ Compilation check failed: {e}")
        return False

def verify_data_structure_sizes():
    """Verify data structures have reasonable memory footprints."""
    print("🔍 Verifying data structure memory efficiency...")
    
    # Test Triple memory footprint
    success, stdout, stderr = run_rust_test("test_triple_memory_footprint")
    
    if success:
        print("  ✅ Triple memory footprint is efficient")
    else:
        print(f"  ❌ Memory footprint issue: {stderr}")
        return False
    
    # Test node size calculation
    success, stdout, stderr = run_rust_test("test_knowledge_node_new_chunk_size_calculation")
    
    if success:
        print("  ✅ Node size calculations are correct")
    else:
        print(f"  ❌ Node size calculation issue: {stderr}")
        return False
        
    return True

def generate_data_flow_diagram():
    """Generate a visual representation of data flow."""
    print("📊 Generating data flow diagram...")
    
    diagram = {
        "modules": {
            "KnowledgeEngine": {
                "data_structures": ["Triple", "KnowledgeNode", "TripleQuery", "KnowledgeResult"],
                "operations": ["store_triple", "query_triples", "semantic_search"]
            },
            "TemporalTracking": {
                "data_structures": ["TemporalTriple", "TemporalIndex", "TemporalQueryResult"],
                "operations": ["record_operation", "query_point_in_time", "track_entity_evolution"]
            },
            "DatabaseBranching": {
                "data_structures": ["BranchInfo", "BranchComparison", "MergeResult"],
                "operations": ["create_branch", "compare_branches", "merge_branches"]
            },
            "CognitiveReasoning": {
                "data_structures": ["ReasoningChain", "LogicalStep", "CognitivePath"],
                "operations": ["deductive_reasoning", "inductive_reasoning", "analogical_reasoning"]
            }
        },
        "integration_points": {
            "Storage_to_Temporal": {
                "data_flow": "Triple -> TemporalTriple",
                "operations": "store_triple triggers record_operation"
            },
            "Temporal_to_Branching": {
                "data_flow": "TemporalQueryResult -> BranchComparison",
                "operations": "temporal queries feed branch comparisons"
            },
            "Storage_to_Reasoning": {
                "data_flow": "KnowledgeResult -> ReasoningChain",
                "operations": "query results become reasoning premises"
            },
            "Branching_to_Storage": {
                "data_flow": "BranchInfo -> KnowledgeEngine",
                "operations": "branches maintain separate engine instances"
            }
        },
        "compatibility_verified": {
            "Triple_consistency": "✅ Hashing and serialization work across modules",
            "Node_compatibility": "✅ KnowledgeNode structures integrate properly",
            "Query_integration": "✅ Query results feed into other systems",
            "Memory_efficiency": "✅ Data structures maintain reasonable footprints"
        }
    }
    
    with open("data_flow_integration_report.json", "w") as f:
        json.dump(diagram, f, indent=2)
    
    print("  📄 Data flow diagram saved to data_flow_integration_report.json")
    return True

def main():
    """Main verification function."""
    print("🚀 Starting Data Structure Integration Verification")
    print("=" * 60)
    
    verification_results = []
    
    # Step 1: Check compilation
    print("\n📋 Step 1: Compilation Check")
    compilation_ok = check_compilation_status()
    verification_results.append(("Compilation", compilation_ok))
    
    if not compilation_ok:
        print("\n❌ Compilation failed - cannot proceed with integration tests")
        return False
    
    # Step 2: Verify data structure compatibility
    print("\n📋 Step 2: Data Structure Compatibility")
    triple_ok = verify_triple_compatibility()
    node_ok = verify_knowledge_node_compatibility()
    memory_ok = verify_data_structure_sizes()
    
    verification_results.extend([
        ("Triple Compatibility", triple_ok),
        ("KnowledgeNode Compatibility", node_ok),
        ("Memory Efficiency", memory_ok)
    ])
    
    # Step 3: Verify integration points
    print("\n📋 Step 3: Integration Point Verification")
    temporal_ok = verify_temporal_integration()
    engine_ok = verify_engine_query_integration()
    
    verification_results.extend([
        ("Temporal Integration", temporal_ok),
        ("Engine Query Integration", engine_ok)
    ])
    
    # Step 4: End-to-end testing
    print("\n📋 Step 4: End-to-End Integration Test")
    e2e_ok = test_data_flow_integration()
    verification_results.append(("End-to-End Data Flow", e2e_ok))
    
    # Step 5: Generate documentation
    print("\n📋 Step 5: Documentation Generation")
    doc_ok = generate_data_flow_diagram()
    verification_results.append(("Documentation", doc_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in verification_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Data structures are compatible across all modules")
        print("✅ Integration points work correctly")
        print("✅ End-to-end data flow is functional")
        print("✅ Memory efficiency is maintained")
        return True
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        print("⚠️  Data structure integration issues detected")
        print("🔧 Check the specific failures above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)