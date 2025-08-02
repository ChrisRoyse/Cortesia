# Task 31: Phase 2 Integration Tests

**Estimated Time**: 15-20 minutes  
**Dependencies**: 30_api_endpoints.md  
**Stage**: Integration & Testing  

## Objective
Implement comprehensive integration tests to validate that Phase 3 knowledge graph components correctly integrate with Phase 2 neuromorphic allocation engine, ensuring seamless data flow, proper TTFS encoding usage, and cortical column coordination.

## Specific Requirements

### 1. Phase 2 Component Integration Validation
- Test proper communication with cortical column managers
- Validate TTFS encoding integration and neural pathway storage
- Verify allocation engine coordination with knowledge graph operations
- Ensure memory pool synchronization between phases

### 2. Data Flow Testing
- Test end-to-end data flow from allocation request to graph storage
- Validate neural pathway metadata persistence in graph database
- Verify inheritance system integration with allocation priorities
- Test performance monitoring data exchange between phases

### 3. Error Handling and Resilience
- Test graceful degradation when Phase 2 components are unavailable
- Validate error propagation and recovery mechanisms
- Test timeout handling and retry logic for Phase 2 communications
- Verify data consistency during partial failures

## Implementation Steps

### 1. Create Phase 2 Integration Test Suite
```rust
// tests/integration/phase2_integration_tests.rs
use std::sync::Arc;
use tokio::test;

use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::mcp::brain_inspired_server::BrainInspiredServer;
use llmkg::core::activation_processors::{CorticalColumnManager, TTFSEncoder};
use llmkg::core::memory::MemoryPool;

#[tokio::test]
async fn test_cortical_column_integration() {
    // Initialize Phase 2 components
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_test());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_test());
    let memory_pool = Arc::new(MemoryPool::new_for_test());
    
    // Initialize Phase 3 components with Phase 2 integration
    let brain_graph = BrainEnhancedGraphCore::new_with_phase2_integration(
        cortical_manager.clone(),
        ttfs_encoder.clone(),
        memory_pool.clone(),
    ).await.expect("Failed to create brain graph");
    
    // Test concept allocation with cortical column coordination
    let concept_request = MemoryAllocationRequest {
        concept_id: "test_integration_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Integration test content for Phase 2 coordination".to_string(),
        semantic_embedding: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        priority: AllocationPriority::High,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec!["cortical_area_1".to_string()],
        user_id: "test_user".to_string(),
        request_id: "req_001".to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(concept_request)
        .await
        .expect("Failed to allocate memory with cortical coordination");
    
    // Verify cortical column was properly selected
    assert!(allocation_result.cortical_column_id.is_some());
    assert!(allocation_result.neural_pathway_id.is_some());
    
    // Verify TTFS encoding was applied
    assert!(allocation_result.ttfs_encoding.is_some());
    assert_eq!(allocation_result.ttfs_encoding.as_ref().unwrap().len(), 5);
    
    // Verify memory pool integration
    let memory_slot = memory_pool
        .get_memory_slot(&allocation_result.memory_slot.slot_id)
        .await
        .expect("Failed to retrieve memory slot");
    
    assert_eq!(memory_slot.concept_id, Some(allocation_result.memory_slot.concept_id.unwrap()));
    assert_eq!(memory_slot.allocation_status, AllocationStatus::Active);
    
    println!("✓ Cortical column integration test passed");
}

#[tokio::test]
async fn test_ttfs_encoding_integration() {
    let brain_graph = setup_integrated_brain_graph().await;
    
    // Test semantic search with TTFS encoding
    let search_request = SearchRequest {
        query_text: "neural pathway test".to_string(),
        search_type: SearchType::TTFS,
        similarity_threshold: Some(0.7),
        limit: Some(10),
        user_context: UserContext::default(),
        use_ttfs_encoding: true,
        cortical_area_filter: Some(vec!["visual_cortex".to_string()]),
    };
    
    let search_result = brain_graph
        .search_memory_with_ttfs(search_request)
        .await
        .expect("Failed to search with TTFS encoding");
    
    // Verify TTFS-specific search results
    assert!(search_result.results.len() > 0);
    assert!(search_result.ttfs_search_metadata.is_some());
    
    let ttfs_metadata = search_result.ttfs_search_metadata.unwrap();
    assert!(ttfs_metadata.encoding_time_ms > 0);
    assert!(ttfs_metadata.neural_pathway_matches > 0);
    assert!(ttfs_metadata.cortical_column_hits.len() > 0);
    
    // Verify search results include TTFS encoding information
    for result in &search_result.results {
        assert!(result.ttfs_encoding.is_some());
        assert!(result.neural_pathway_id.is_some());
        assert!(result.cortical_column_id.is_some());
    }
    
    println!("✓ TTFS encoding integration test passed");
}

#[tokio::test]
async fn test_neural_pathway_storage() {
    let brain_graph = setup_integrated_brain_graph().await;
    
    // Create a concept with complex neural pathway
    let allocation_request = MemoryAllocationRequest {
        concept_id: "neural_pathway_test".to_string(),
        concept_type: ConceptType::Semantic,
        content: "Complex concept with multiple neural pathways".to_string(),
        semantic_embedding: Some(generate_test_embedding(512)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements {
            memory_size_mb: 10,
            computational_units: 5,
            bandwidth_mbps: 100,
        },
        locality_hints: vec!["associative_memory".to_string()],
        user_id: "test_user".to_string(),
        request_id: "req_002".to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate memory");
    
    // Verify neural pathway was stored in graph database
    let neural_pathway = brain_graph
        .get_neural_pathway(&allocation_result.neural_pathway_id.unwrap())
        .await
        .expect("Failed to retrieve neural pathway");
    
    assert_eq!(neural_pathway.pathway_id, allocation_result.neural_pathway_id.unwrap());
    assert_eq!(neural_pathway.source_concept_id, allocation_result.memory_slot.concept_id.unwrap());
    assert!(neural_pathway.encoding_strength > 0.0);
    assert!(neural_pathway.synaptic_weights.len() > 0);
    
    // Test pathway-based retrieval
    let pathway_concepts = brain_graph
        .get_concepts_by_neural_pathway(&neural_pathway.pathway_id)
        .await
        .expect("Failed to retrieve concepts by pathway");
    
    assert!(pathway_concepts.len() >= 1);
    assert!(pathway_concepts.iter().any(|c| c.concept_id == allocation_result.memory_slot.concept_id.unwrap()));
    
    println!("✓ Neural pathway storage test passed");
}

#[tokio::test]
async fn test_allocation_priority_inheritance() {
    let brain_graph = setup_integrated_brain_graph().await;
    
    // Create parent concept with high priority
    let parent_allocation = MemoryAllocationRequest {
        concept_id: "high_priority_parent".to_string(),
        concept_type: ConceptType::Abstract,
        content: "High priority parent concept".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Critical,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "test_user".to_string(),
        request_id: "req_003".to_string(),
        version_info: None,
    };
    
    let parent_result = brain_graph
        .allocate_memory_with_cortical_coordination(parent_allocation)
        .await
        .expect("Failed to allocate parent concept");
    
    // Create child concept that should inherit priority characteristics
    let child_allocation = MemoryAllocationRequest {
        concept_id: "child_concept".to_string(),
        concept_type: ConceptType::Specific,
        content: "Child concept inheriting priority".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal, // Should be elevated due to inheritance
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "test_user".to_string(),
        request_id: "req_004".to_string(),
        version_info: None,
    };
    
    // Create inheritance relationship
    brain_graph
        .create_inheritance_relationship(
            &parent_result.memory_slot.concept_id.unwrap(),
            &child_allocation.concept_id,
            InheritanceType::DirectSubclass,
        )
        .await
        .expect("Failed to create inheritance relationship");
    
    let child_result = brain_graph
        .allocate_memory_with_cortical_coordination(child_allocation)
        .await
        .expect("Failed to allocate child concept");
    
    // Verify priority inheritance affected allocation
    assert!(child_result.priority_boost_applied);
    assert!(child_result.allocation_time_ms < 50); // Should be faster due to priority
    
    // Verify cortical column proximity (should be near parent)
    let parent_column = parent_result.cortical_column_id.unwrap();
    let child_column = child_result.cortical_column_id.unwrap();
    
    let column_distance = brain_graph
        .calculate_cortical_column_distance(&parent_column, &child_column)
        .await
        .expect("Failed to calculate column distance");
    
    assert!(column_distance < 3.0); // Should be allocated nearby
    
    println!("✓ Allocation priority inheritance test passed");
}

#[tokio::test]
async fn test_phase2_error_handling() {
    // Test graceful degradation when Phase 2 components fail
    let cortical_manager = Arc::new(CorticalColumnManager::new_with_failure_simulation());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_test());
    let memory_pool = Arc::new(MemoryPool::new_for_test());
    
    let brain_graph = BrainEnhancedGraphCore::new_with_phase2_integration(
        cortical_manager,
        ttfs_encoder,
        memory_pool,
    ).await.expect("Failed to create brain graph");
    
    // Attempt allocation when cortical manager is failing
    let allocation_request = MemoryAllocationRequest {
        concept_id: "error_test_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Testing error handling".to_string(),
        semantic_embedding: Some(vec![0.1, 0.2, 0.3]),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "test_user".to_string(),
        request_id: "req_005".to_string(),
        version_info: None,
    };
    
    let result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await;
    
    // Should succeed with fallback allocation strategy
    assert!(result.is_ok());
    
    let allocation_result = result.unwrap();
    assert!(allocation_result.fallback_allocation_used);
    assert!(allocation_result.cortical_column_id.is_none()); // No cortical coordination
    assert!(allocation_result.neural_pathway_id.is_some()); // Still has pathway (from TTFS)
    
    println!("✓ Phase 2 error handling test passed");
}

async fn setup_integrated_brain_graph() -> BrainEnhancedGraphCore {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_test());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_test());
    let memory_pool = Arc::new(MemoryPool::new_for_test());
    
    BrainEnhancedGraphCore::new_with_phase2_integration(
        cortical_manager,
        ttfs_encoder,
        memory_pool,
    ).await.expect("Failed to create integrated brain graph")
}

fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}
```

### 2. Create Performance Integration Tests
```rust
// tests/integration/phase2_performance_tests.rs
use std::time::Instant;
use tokio::test;

#[tokio::test]
async fn test_allocation_performance_with_phase2() {
    let brain_graph = setup_integrated_brain_graph().await;
    
    // Test allocation performance with Phase 2 integration
    let start_time = Instant::now();
    
    let mut allocation_times = Vec::new();
    
    for i in 0..100 {
        let request_start = Instant::now();
        
        let allocation_request = MemoryAllocationRequest {
            concept_id: format!("perf_test_concept_{}", i),
            concept_type: ConceptType::Episodic,
            content: format!("Performance test content {}", i),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "test_user".to_string(),
            request_id: format!("req_perf_{}", i),
            version_info: None,
        };
        
        let _result = brain_graph
            .allocate_memory_with_cortical_coordination(allocation_request)
            .await
            .expect("Failed to allocate memory");
        
        allocation_times.push(request_start.elapsed().as_millis());
    }
    
    let total_time = start_time.elapsed();
    let avg_allocation_time = allocation_times.iter().sum::<u128>() / allocation_times.len() as u128;
    let max_allocation_time = *allocation_times.iter().max().unwrap();
    
    // Performance assertions
    assert!(avg_allocation_time < 10, "Average allocation time {} ms exceeds 10ms limit", avg_allocation_time);
    assert!(max_allocation_time < 50, "Max allocation time {} ms exceeds 50ms limit", max_allocation_time);
    assert!(total_time.as_secs() < 5, "Total test time {} seconds exceeds 5s limit", total_time.as_secs());
    
    println!("✓ Phase 2 performance integration test passed");
    println!("  Average allocation time: {} ms", avg_allocation_time);
    println!("  Max allocation time: {} ms", max_allocation_time);
    println!("  Total test time: {} ms", total_time.as_millis());
}
```

### 3. Create Data Consistency Tests
```rust
// tests/integration/phase2_consistency_tests.rs
#[tokio::test]
async fn test_data_consistency_across_phases() {
    let brain_graph = setup_integrated_brain_graph().await;
    
    // Allocate multiple related concepts
    let concept_ids = vec!["concept_a", "concept_b", "concept_c"];
    let mut allocation_results = Vec::new();
    
    for concept_id in &concept_ids {
        let allocation_request = MemoryAllocationRequest {
            concept_id: concept_id.to_string(),
            concept_type: ConceptType::Semantic,
            content: format!("Content for {}", concept_id),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "test_user".to_string(),
            request_id: format!("req_{}", concept_id),
            version_info: None,
        };
        
        let result = brain_graph
            .allocate_memory_with_cortical_coordination(allocation_request)
            .await
            .expect("Failed to allocate memory");
        
        allocation_results.push(result);
    }
    
    // Verify data consistency between Phase 2 memory pool and Phase 3 graph
    for result in &allocation_results {
        let concept_id = result.memory_slot.concept_id.as_ref().unwrap();
        
        // Check Phase 3 graph storage
        let graph_concept = brain_graph
            .get_concept(concept_id)
            .await
            .expect("Failed to retrieve concept from graph");
        
        assert_eq!(graph_concept.concept_id, *concept_id);
        assert!(graph_concept.neural_pathway_id.is_some());
        assert!(graph_concept.cortical_column_id.is_some());
        
        // Check Phase 2 memory pool consistency
        let memory_slot = brain_graph
            .get_memory_pool()
            .get_memory_slot(&result.memory_slot.slot_id)
            .await
            .expect("Failed to retrieve memory slot");
        
        assert_eq!(memory_slot.concept_id, Some(concept_id.clone()));
        assert_eq!(memory_slot.allocation_status, AllocationStatus::Active);
        
        // Check TTFS encoding consistency
        assert_eq!(
            result.ttfs_encoding.as_ref().unwrap(),
            &memory_slot.ttfs_encoding.unwrap()
        );
    }
    
    println!("✓ Data consistency test passed");
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All Phase 2 components integrate correctly with Phase 3 knowledge graph
- [ ] TTFS encoding is properly applied and stored in graph database
- [ ] Cortical column coordination works for memory allocation
- [ ] Neural pathway metadata is correctly persisted
- [ ] Inheritance system integrates with allocation priorities

### Performance Requirements
- [ ] Integrated allocation operations complete within 10ms average
- [ ] Phase 2 communication overhead < 2ms per operation
- [ ] Error handling and fallback mechanisms work within 5ms
- [ ] Memory consistency checks complete within 1ms

### Testing Requirements
- [ ] All integration test scenarios pass
- [ ] Performance benchmarks meet requirements
- [ ] Error simulation tests demonstrate resilience
- [ ] Data consistency validation succeeds

## Validation Steps

1. **Run Phase 2 integration tests**:
   ```bash
   cargo test --test phase2_integration_tests
   ```

2. **Verify performance integration**:
   ```bash
   cargo test --test phase2_performance_tests
   ```

3. **Check data consistency**:
   ```bash
   cargo test --test phase2_consistency_tests
   ```

4. **Run complete integration suite**:
   ```bash
   cargo test integration::phase2
   ```

## Files to Create/Modify
- `tests/integration/phase2_integration_tests.rs` - Main integration test suite
- `tests/integration/phase2_performance_tests.rs` - Performance integration tests
- `tests/integration/phase2_consistency_tests.rs` - Data consistency tests
- `tests/integration/mod.rs` - Test module definitions

## Success Metrics
- Integration test success rate: 100%
- Average allocation time with Phase 2: < 10ms
- Data consistency validation: 100% pass rate
- Error handling resilience: 99.9% uptime under failure simulation

## Next Task
Upon completion, proceed to **32_performance_benchmarks.md** to run comprehensive performance benchmarks across all Phase 3 components.