# Task 33: Data Integrity Tests

**Estimated Time**: 15-20 minutes  
**Dependencies**: 32_performance_benchmarks.md  
**Stage**: Integration & Testing  

## Objective
Implement comprehensive data integrity and consistency validation tests to ensure that all Phase 3 knowledge graph operations maintain data accuracy, referential integrity, inheritance consistency, and proper synchronization between Phase 2 and Phase 3 components.

## Specific Requirements

### 1. Referential Integrity Validation
- Verify all concept references point to valid, existing concepts
- Test that inheritance relationships maintain parent-child consistency
- Validate neural pathway references correspond to actual Phase 2 pathways
- Ensure cortical column mappings are accurate and up-to-date

### 2. Data Consistency Checks
- Test property inheritance chain consistency across updates
- Validate temporal versioning maintains accurate historical states
- Verify cache synchronization with underlying database state
- Test concurrent operation data consistency under load

### 3. Cross-Phase Data Synchronization
- Validate Phase 2 memory pool state matches Phase 3 graph state
- Test TTFS encoding consistency between phases
- Verify allocation metadata synchronization accuracy
- Test error state handling and data recovery mechanisms

## Implementation Steps

### 1. Create Core Data Integrity Test Suite
```rust
// tests/integrity/data_integrity_tests.rs
use std::sync::Arc;
use tokio::test;
use uuid::Uuid;

use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::types::*;
use llmkg::versioning::TemporalVersionManager;

#[tokio::test]
async fn test_referential_integrity() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create parent concept
    let parent_request = MemoryAllocationRequest {
        concept_id: "parent_concept".to_string(),
        concept_type: ConceptType::Abstract,
        content: "Parent concept for integrity testing".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "integrity_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let parent_result = brain_graph
        .allocate_memory_with_cortical_coordination(parent_request)
        .await
        .expect("Failed to allocate parent concept");
    
    // Create child concept with inheritance relationship
    let child_request = MemoryAllocationRequest {
        concept_id: "child_concept".to_string(),
        concept_type: ConceptType::Specific,
        content: "Child concept for integrity testing".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "integrity_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let child_result = brain_graph
        .allocate_memory_with_cortical_coordination(child_request)
        .await
        .expect("Failed to allocate child concept");
    
    // Create inheritance relationship
    brain_graph
        .create_inheritance_relationship(
            &parent_result.memory_slot.concept_id.unwrap(),
            &child_result.memory_slot.concept_id.unwrap(),
            InheritanceType::DirectSubclass,
        )
        .await
        .expect("Failed to create inheritance relationship");
    
    // Validate referential integrity
    let integrity_report = brain_graph
        .validate_referential_integrity()
        .await
        .expect("Failed to validate referential integrity");
    
    assert_eq!(integrity_report.broken_references.len(), 0, 
              "Found broken references: {:?}", integrity_report.broken_references);
    assert_eq!(integrity_report.orphaned_concepts.len(), 0,
              "Found orphaned concepts: {:?}", integrity_report.orphaned_concepts);
    assert_eq!(integrity_report.invalid_inheritance_chains.len(), 0,
              "Found invalid inheritance chains: {:?}", integrity_report.invalid_inheritance_chains);
    
    // Test specific relationship integrity
    let parent_concept = brain_graph
        .get_concept(&parent_result.memory_slot.concept_id.unwrap())
        .await
        .expect("Failed to retrieve parent concept");
    
    let child_concept = brain_graph
        .get_concept(&child_result.memory_slot.concept_id.unwrap())
        .await
        .expect("Failed to retrieve child concept");
    
    // Verify bidirectional relationship consistency
    assert!(parent_concept.child_relationships.iter()
           .any(|r| r.target_concept_id == child_result.memory_slot.concept_id.unwrap()));
    assert!(child_concept.parent_relationships.iter()
           .any(|r| r.source_concept_id == parent_result.memory_slot.concept_id.unwrap()));
    
    // Verify neural pathway references are valid
    if let Some(neural_pathway_id) = &parent_concept.neural_pathway_id {
        let neural_pathway = brain_graph
            .get_neural_pathway(neural_pathway_id)
            .await
            .expect("Failed to retrieve neural pathway");
        
        assert_eq!(neural_pathway.source_concept_id, parent_concept.concept_id);
    }
    
    // Verify cortical column references are valid
    if let Some(cortical_column_id) = &parent_concept.cortical_column_id {
        let cortical_column = brain_graph
            .get_cortical_column_manager()
            .get_column_info(cortical_column_id)
            .await
            .expect("Failed to retrieve cortical column");
        
        assert!(cortical_column.allocated_concepts.contains(&parent_concept.concept_id));
    }
    
    println!("✓ Referential integrity test passed");
}

#[tokio::test]
async fn test_inheritance_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create inheritance hierarchy: grandparent -> parent -> child
    let concepts = vec![
        ("grandparent", ConceptType::Abstract),
        ("parent", ConceptType::Semantic),
        ("child", ConceptType::Specific),
    ];
    
    let mut concept_ids = Vec::new();
    
    // Allocate all concepts
    for (concept_name, concept_type) in concepts {
        let request = MemoryAllocationRequest {
            concept_id: concept_name.to_string(),
            concept_type,
            content: format!("Content for {}", concept_name),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "integrity_test_user".to_string(),
            request_id: Uuid::new_v4().to_string(),
            version_info: None,
        };
        
        let result = brain_graph
            .allocate_memory_with_cortical_coordination(request)
            .await
            .expect("Failed to allocate concept");
        
        concept_ids.push(result.memory_slot.concept_id.unwrap());
    }
    
    // Create inheritance relationships
    brain_graph
        .create_inheritance_relationship(
            &concept_ids[0], // grandparent
            &concept_ids[1], // parent
            InheritanceType::DirectSubclass,
        )
        .await
        .expect("Failed to create grandparent->parent relationship");
    
    brain_graph
        .create_inheritance_relationship(
            &concept_ids[1], // parent
            &concept_ids[2], // child
            InheritanceType::DirectSubclass,
        )
        .await
        .expect("Failed to create parent->child relationship");
    
    // Add properties to grandparent
    brain_graph
        .add_concept_property(
            &concept_ids[0],
            ConceptProperty {
                key: "fundamental_property".to_string(),
                value: PropertyValue::String("fundamental_value".to_string()),
                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                metadata: PropertyMetadata::default(),
            },
        )
        .await
        .expect("Failed to add property to grandparent");
    
    // Add properties to parent (overriding and new)
    brain_graph
        .add_concept_property(
            &concept_ids[1],
            ConceptProperty {
                key: "fundamental_property".to_string(),
                value: PropertyValue::String("overridden_value".to_string()),
                inheritance_behavior: PropertyInheritanceBehavior::Override,
                metadata: PropertyMetadata::default(),
            },
        )
        .await
        .expect("Failed to add overriding property to parent");
    
    brain_graph
        .add_concept_property(
            &concept_ids[1],
            ConceptProperty {
                key: "parent_specific".to_string(),
                value: PropertyValue::String("parent_value".to_string()),
                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                metadata: PropertyMetadata::default(),
            },
        )
        .await
        .expect("Failed to add parent-specific property");
    
    // Validate inheritance chain consistency
    let child_resolved_properties = brain_graph
        .resolve_inherited_properties(&concept_ids[2], true)
        .await
        .expect("Failed to resolve child properties");
    
    // Child should inherit from both grandparent and parent
    assert!(child_resolved_properties.resolved_properties.contains_key("fundamental_property"));
    assert!(child_resolved_properties.resolved_properties.contains_key("parent_specific"));
    
    // Should have the overridden value, not the grandparent's original
    let fundamental_prop = &child_resolved_properties.resolved_properties["fundamental_property"];
    if let PropertyValue::String(value) = &fundamental_prop.value {
        assert_eq!(value, "overridden_value", "Property inheritance override failed");
    }
    
    // Verify inheritance chain structure
    assert_eq!(child_resolved_properties.inheritance_chain.len(), 3);
    assert_eq!(child_resolved_properties.inheritance_chain[0], concept_ids[2]); // child
    assert_eq!(child_resolved_properties.inheritance_chain[1], concept_ids[1]); // parent
    assert_eq!(child_resolved_properties.inheritance_chain[2], concept_ids[0]); // grandparent
    
    // Test inheritance consistency validation
    let consistency_report = brain_graph
        .validate_inheritance_consistency()
        .await
        .expect("Failed to validate inheritance consistency");
    
    assert_eq!(consistency_report.circular_dependencies.len(), 0,
              "Found circular inheritance dependencies: {:?}", consistency_report.circular_dependencies);
    assert_eq!(consistency_report.property_conflicts.len(), 0,
              "Found property inheritance conflicts: {:?}", consistency_report.property_conflicts);
    assert_eq!(consistency_report.broken_chains.len(), 0,
              "Found broken inheritance chains: {:?}", consistency_report.broken_chains);
    
    println!("✓ Inheritance consistency test passed");
}

#[tokio::test]
async fn test_temporal_versioning_integrity() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concept with initial content
    let initial_request = MemoryAllocationRequest {
        concept_id: "versioned_concept".to_string(),
        concept_type: ConceptType::Semantic,
        content: "Initial content for versioning test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "integrity_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let initial_result = brain_graph
        .allocate_memory_with_cortical_coordination(initial_request)
        .await
        .expect("Failed to allocate initial concept");
    
    let concept_id = initial_result.memory_slot.concept_id.unwrap();
    
    // Perform multiple updates to create version history
    let updates = vec![
        "Updated content version 1",
        "Updated content version 2", 
        "Updated content version 3",
    ];
    
    let mut version_ids = vec![initial_result.memory_slot.slot_id.clone()];
    
    for (i, update_content) in updates.iter().enumerate() {
        let update_request = MemoryUpdateRequest {
            concept_id: concept_id.clone(),
            update_type: UpdateType::ContentModification,
            new_content: Some(update_content.to_string()),
            property_updates: None,
            relationship_updates: None,
            metadata: UpdateMetadata {
                user_id: "integrity_test_user".to_string(),
                update_reason: format!("Test update {}", i + 1),
                ..Default::default()
            },
        };
        
        let update_result = brain_graph
            .update_memory(update_request)
            .await
            .expect("Failed to update concept");
        
        version_ids.push(update_result.new_version_id);
    }
    
    // Validate temporal versioning integrity
    let version_history = brain_graph
        .get_concept_version_history(&concept_id)
        .await
        .expect("Failed to retrieve version history");
    
    assert_eq!(version_history.versions.len(), 4, "Expected 4 versions (initial + 3 updates)");
    
    // Verify version sequence and timestamps
    for i in 1..version_history.versions.len() {
        let current_version = &version_history.versions[i];
        let previous_version = &version_history.versions[i - 1];
        
        assert!(current_version.created_at > previous_version.created_at,
               "Version timestamps not in correct order");
        assert_eq!(current_version.version_number, previous_version.version_number + 1,
                  "Version numbers not sequential");
    }
    
    // Test version content integrity
    let version_2 = brain_graph
        .get_concept_at_version(&concept_id, &version_ids[2])
        .await
        .expect("Failed to retrieve concept at version 2");
    
    assert_eq!(version_2.content, "Updated content version 1");
    
    // Test temporal query integrity
    let temporal_query_result = brain_graph
        .query_concept_at_timestamp(&concept_id, version_history.versions[2].created_at)
        .await
        .expect("Failed to query concept at specific timestamp");
    
    assert_eq!(temporal_query_result.content, "Updated content version 1");
    
    // Validate no data corruption in version chain
    for (i, version_id) in version_ids.iter().enumerate() {
        let version_concept = brain_graph
            .get_concept_at_version(&concept_id, version_id)
            .await
            .expect("Failed to retrieve version");
        
        // Verify version-specific content
        let expected_content = if i == 0 {
            "Initial content for versioning test"
        } else {
            &updates[i - 1]
        };
        
        assert_eq!(version_concept.content, expected_content,
                  "Version {} content corrupted", i);
    }
    
    println!("✓ Temporal versioning integrity test passed");
}

#[tokio::test]
async fn test_cross_phase_data_synchronization() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Test allocation synchronization between phases
    let allocation_request = MemoryAllocationRequest {
        concept_id: "sync_test_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Cross-phase synchronization test content".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::High,
        resource_requirements: ResourceRequirements {
            memory_size_mb: 50,
            computational_units: 10,
            bandwidth_mbps: 200,
        },
        locality_hints: vec!["sync_test_area".to_string()],
        user_id: "integrity_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate memory for sync test");
    
    let concept_id = allocation_result.memory_slot.concept_id.unwrap();
    
    // Verify Phase 3 graph state
    let graph_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve concept from graph");
    
    // Verify Phase 2 memory pool state
    let memory_slot = brain_graph
        .get_memory_pool()
        .get_memory_slot(&allocation_result.memory_slot.slot_id)
        .await
        .expect("Failed to retrieve memory slot");
    
    // Validate synchronization between phases
    assert_eq!(memory_slot.concept_id, Some(concept_id.clone()));
    assert_eq!(memory_slot.allocation_status, AllocationStatus::Active);
    assert_eq!(memory_slot.resource_requirements, allocation_result.memory_slot.resource_requirements);
    
    // Verify TTFS encoding synchronization
    assert_eq!(
        graph_concept.ttfs_encoding,
        memory_slot.ttfs_encoding,
        "TTFS encoding not synchronized between phases"
    );
    
    // Verify cortical column assignment synchronization
    assert_eq!(
        graph_concept.cortical_column_id,
        memory_slot.cortical_column_id,
        "Cortical column assignment not synchronized"
    );
    
    // Verify neural pathway synchronization
    assert_eq!(
        graph_concept.neural_pathway_id,
        memory_slot.neural_pathway_id,
        "Neural pathway assignment not synchronized"
    );
    
    // Test update synchronization
    let update_request = MemoryUpdateRequest {
        concept_id: concept_id.clone(),
        update_type: UpdateType::ContentModification,
        new_content: Some("Updated content for sync test".to_string()),
        property_updates: None,
        relationship_updates: None,
        metadata: UpdateMetadata::default(),
    };
    
    let update_result = brain_graph
        .update_memory(update_request)
        .await
        .expect("Failed to update memory for sync test");
    
    // Verify update synchronization
    let updated_graph_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve updated concept from graph");
    
    let updated_memory_slot = brain_graph
        .get_memory_pool()
        .get_memory_slot(&allocation_result.memory_slot.slot_id)
        .await
        .expect("Failed to retrieve updated memory slot");
    
    assert_eq!(updated_graph_concept.content, "Updated content for sync test");
    assert_eq!(updated_memory_slot.content, "Updated content for sync test");
    assert_eq!(updated_graph_concept.updated_at, updated_memory_slot.updated_at);
    
    // Test deallocation synchronization
    brain_graph
        .deallocate_memory(&concept_id)
        .await
        .expect("Failed to deallocate memory");
    
    // Verify deallocation is synchronized
    let deallocated_memory_slot = brain_graph
        .get_memory_pool()
        .get_memory_slot(&allocation_result.memory_slot.slot_id)
        .await
        .expect("Failed to retrieve deallocated memory slot");
    
    assert_eq!(deallocated_memory_slot.allocation_status, AllocationStatus::Deallocated);
    
    // Graph concept should be marked as deallocated but still accessible for queries
    let deallocated_graph_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve deallocated concept from graph");
    
    assert_eq!(deallocated_graph_concept.allocation_status, Some(AllocationStatus::Deallocated));
    
    println!("✓ Cross-phase data synchronization test passed");
}

#[tokio::test]
async fn test_concurrent_data_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create base concept for concurrent operations
    let base_request = MemoryAllocationRequest {
        concept_id: "concurrent_test_concept".to_string(),
        concept_type: ConceptType::Semantic,
        content: "Base content for concurrent test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "integrity_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let base_result = brain_graph
        .allocate_memory_with_cortical_coordination(base_request)
        .await
        .expect("Failed to allocate base concept");
    
    let concept_id = base_result.memory_slot.concept_id.unwrap();
    
    // Perform concurrent updates
    let concurrent_updates = 50;
    let update_tasks: Vec<_> = (0..concurrent_updates)
        .map(|i| {
            let graph = brain_graph.clone();
            let id = concept_id.clone();
            tokio::spawn(async move {
                let update_request = MemoryUpdateRequest {
                    concept_id: id,
                    update_type: UpdateType::PropertyAddition,
                    new_content: None,
                    property_updates: Some(vec![PropertyUpdate {
                        operation: PropertyOperation::Add,
                        property: ConceptProperty {
                            key: format!("concurrent_prop_{}", i),
                            value: PropertyValue::String(format!("value_{}", i)),
                            inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                            metadata: PropertyMetadata::default(),
                        },
                    }]),
                    relationship_updates: None,
                    metadata: UpdateMetadata::default(),
                };
                
                graph.update_memory(update_request).await
            })
        })
        .collect();
    
    // Wait for all updates to complete
    let update_results = futures::future::join_all(update_tasks).await;
    
    // Verify all updates succeeded
    let successful_updates = update_results
        .iter()
        .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
        .count();
    
    assert_eq!(successful_updates, concurrent_updates, 
              "Not all concurrent updates succeeded");
    
    // Verify final state consistency
    let final_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve final concept state");
    
    // Should have all concurrent properties
    assert_eq!(final_concept.properties.len(), concurrent_updates,
              "Not all concurrent properties were persisted");
    
    // Verify each property exists and has correct value
    for i in 0..concurrent_updates {
        let prop_key = format!("concurrent_prop_{}", i);
        let prop = final_concept.properties.iter()
            .find(|p| p.key == prop_key)
            .expect(&format!("Property {} not found", prop_key));
        
        if let PropertyValue::String(value) = &prop.value {
            assert_eq!(value, &format!("value_{}", i), 
                      "Property {} has incorrect value", prop_key);
        } else {
            panic!("Property {} has incorrect type", prop_key);
        }
    }
    
    // Verify version consistency
    let version_history = brain_graph
        .get_concept_version_history(&concept_id)
        .await
        .expect("Failed to retrieve version history");
    
    // Should have initial version + all updates
    assert!(version_history.versions.len() >= concurrent_updates + 1,
           "Version history missing updates");
    
    println!("✓ Concurrent data consistency test passed");
}

async fn setup_integrity_test_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_test());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_test());
    let memory_pool = Arc::new(MemoryPool::new_for_test());
    
    Arc::new(
        BrainEnhancedGraphCore::new_with_phase2_integration(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create integrity test graph")
    )
}

fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}

#[derive(Debug)]
pub struct IntegrityReport {
    pub broken_references: Vec<BrokenReference>,
    pub orphaned_concepts: Vec<String>,
    pub invalid_inheritance_chains: Vec<InvalidInheritanceChain>,
}

#[derive(Debug)]
pub struct ConsistencyReport {
    pub circular_dependencies: Vec<CircularDependency>,
    pub property_conflicts: Vec<PropertyConflict>,
    pub broken_chains: Vec<BrokenChain>,
}

#[derive(Debug)]
pub struct BrokenReference {
    pub source_concept_id: String,
    pub reference_type: String,
    pub target_id: String,
}

#[derive(Debug)]
pub struct InvalidInheritanceChain {
    pub concept_id: String,
    pub issue_description: String,
}

#[derive(Debug)]
pub struct CircularDependency {
    pub concept_ids: Vec<String>,
}

#[derive(Debug)]
pub struct PropertyConflict {
    pub concept_id: String,
    pub property_key: String,
    pub conflict_description: String,
}

#[derive(Debug)]
pub struct BrokenChain {
    pub start_concept_id: String,
    pub broken_at_concept_id: String,
}
```

### 2. Create Cache Consistency Tests
```rust
// tests/integrity/cache_consistency_tests.rs
#[tokio::test]
async fn test_cache_database_synchronization() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Allocate concept that will be cached
    let allocation_request = MemoryAllocationRequest {
        concept_id: "cache_test_concept".to_string(),
        concept_type: ConceptType::Semantic,
        content: "Content for cache consistency test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "cache_test_user".to_string(),
        request_id: Uuid::new_v4().to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate concept for cache test");
    
    let concept_id = allocation_result.memory_slot.concept_id.unwrap();
    
    // Retrieve concept to populate cache
    let cached_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve concept for caching");
    
    // Verify cache hit
    let cache_stats_before = brain_graph.get_cache_statistics().await;
    
    let retrieved_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve cached concept");
    
    let cache_stats_after = brain_graph.get_cache_statistics().await;
    
    assert!(cache_stats_after.hits > cache_stats_before.hits, "Cache hit not registered");
    assert_eq!(cached_concept.content, retrieved_concept.content, "Cached content doesn't match");
    
    // Directly update database (simulating external change)
    brain_graph
        .direct_database_update(&concept_id, "Updated content bypassing cache")
        .await
        .expect("Failed to update database directly");
    
    // Cache should be invalidated or updated
    let post_update_concept = brain_graph
        .get_concept(&concept_id)
        .await
        .expect("Failed to retrieve concept after direct update");
    
    assert_eq!(post_update_concept.content, "Updated content bypassing cache",
              "Cache not properly invalidated after direct database update");
    
    println!("✓ Cache-database synchronization test passed");
}

#[tokio::test]
async fn test_inheritance_cache_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create inheritance hierarchy
    let parent_id = "cache_parent".to_string();
    let child_id = "cache_child".to_string();
    
    // Allocate parent and child
    let parent_request = create_test_allocation_request(&parent_id, ConceptType::Abstract);
    let child_request = create_test_allocation_request(&child_id, ConceptType::Specific);
    
    brain_graph.allocate_memory_with_cortical_coordination(parent_request).await.unwrap();
    brain_graph.allocate_memory_with_cortical_coordination(child_request).await.unwrap();
    
    // Create inheritance relationship
    brain_graph
        .create_inheritance_relationship(&parent_id, &child_id, InheritanceType::DirectSubclass)
        .await
        .expect("Failed to create inheritance relationship");
    
    // Add property to parent
    brain_graph
        .add_concept_property(
            &parent_id,
            ConceptProperty {
                key: "inherited_prop".to_string(),
                value: PropertyValue::String("original_value".to_string()),
                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                metadata: PropertyMetadata::default(),
            },
        )
        .await
        .expect("Failed to add property to parent");
    
    // Resolve child properties to populate inheritance cache
    let initial_resolved = brain_graph
        .resolve_inherited_properties(&child_id, true)
        .await
        .expect("Failed to resolve initial properties");
    
    assert!(initial_resolved.resolved_properties.contains_key("inherited_prop"));
    
    // Update parent property
    brain_graph
        .update_concept_property(
            &parent_id,
            "inherited_prop",
            PropertyValue::String("updated_value".to_string()),
        )
        .await
        .expect("Failed to update parent property");
    
    // Child's resolved properties should reflect the update
    let updated_resolved = brain_graph
        .resolve_inherited_properties(&child_id, true)
        .await
        .expect("Failed to resolve updated properties");
    
    let updated_prop = &updated_resolved.resolved_properties["inherited_prop"];
    if let PropertyValue::String(value) = &updated_prop.value {
        assert_eq!(value, "updated_value", "Inheritance cache not properly invalidated");
    }
    
    println!("✓ Inheritance cache consistency test passed");
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All referential integrity checks pass without errors
- [ ] Inheritance consistency validation confirms proper hierarchy maintenance
- [ ] Temporal versioning preserves accurate historical states
- [ ] Cross-phase data synchronization maintains consistency between Phase 2 and 3
- [ ] Concurrent operations maintain data consistency under load

### Performance Requirements
- [ ] Integrity validation completes within 5 seconds for 10K nodes
- [ ] Cache consistency checks run in < 100ms
- [ ] Cross-phase synchronization validation < 50ms per concept
- [ ] Concurrent consistency tests handle 100+ simultaneous operations

### Testing Requirements
- [ ] All data integrity test scenarios pass
- [ ] Cache consistency validation succeeds
- [ ] Concurrent access maintains data accuracy
- [ ] Error scenarios properly preserve data integrity

## Validation Steps

1. **Run core data integrity tests**:
   ```bash
   cargo test --test data_integrity_tests
   ```

2. **Execute cache consistency tests**:
   ```bash
   cargo test --test cache_consistency_tests
   ```

3. **Run concurrent integrity validation**:
   ```bash
   cargo test test_concurrent_data_consistency --release
   ```

4. **Validate cross-phase synchronization**:
   ```bash
   cargo test test_cross_phase_data_synchronization
   ```

## Files to Create/Modify
- `tests/integrity/data_integrity_tests.rs` - Main integrity test suite
- `tests/integrity/cache_consistency_tests.rs` - Cache consistency tests
- `tests/integrity/concurrent_integrity_tests.rs` - Concurrent operation tests
- `tests/integrity/mod.rs` - Test module definitions
- `src/core/brain_enhanced_graph/integrity_validator.rs` - Integrity validation implementation

## Success Metrics
- Referential integrity validation: 100% pass rate
- Inheritance consistency: 0 conflicts detected
- Cache synchronization: < 1ms invalidation time
- Cross-phase consistency: 100% synchronization accuracy
- Concurrent operation safety: 0 data corruption incidents

## Next Task
Upon completion, proceed to **34_concurrent_access_tests.md** to test concurrent access patterns and thread safety.