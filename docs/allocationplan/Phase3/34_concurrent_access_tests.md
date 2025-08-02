# Task 34: Concurrent Access Tests

**Estimated Time**: 15-20 minutes  
**Dependencies**: 33_data_integrity_tests.md  
**Stage**: Integration & Testing  

## Objective
Implement comprehensive concurrent access testing to validate thread safety, deadlock prevention, resource contention handling, and performance under high concurrency loads across all Phase 3 knowledge graph components and their integration with Phase 2 systems.

## Specific Requirements

### 1. Thread Safety Validation
- Test concurrent read/write operations on shared graph structures
- Validate lock-free data structures and atomic operations
- Test concurrent access to inheritance resolution cache
- Verify thread-safe neural pathway and cortical column access

### 2. Deadlock Prevention Testing
- Test complex multi-resource locking scenarios
- Validate lock ordering consistency across components
- Test timeout mechanisms for long-running operations
- Verify graceful handling of resource contention

### 3. High Concurrency Load Testing
- Test system behavior under 1000+ concurrent operations
- Validate connection pooling effectiveness under load
- Test cache performance and hit rates under concurrent access
- Measure throughput degradation patterns with increasing concurrency

## Implementation Steps

### 1. Create Thread Safety Test Suite
```rust
// tests/concurrency/thread_safety_tests.rs
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;
use tokio::{task::JoinHandle, time::timeout};
use rayon::prelude::*;

use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::types::*;

#[tokio::test]
async fn test_concurrent_memory_allocation() {
    let brain_graph = setup_concurrency_test_graph().await;
    let concurrent_operations = 500;
    let success_counter = Arc::new(AtomicUsize::new(0));
    let error_counter = Arc::new(AtomicUsize::new(0));
    
    println!("🔄 Testing {} concurrent memory allocations...", concurrent_operations);
    
    let allocation_tasks: Vec<JoinHandle<()>> = (0..concurrent_operations)
        .map(|i| {
            let graph = brain_graph.clone();
            let success_counter = success_counter.clone();
            let error_counter = error_counter.clone();
            
            tokio::spawn(async move {
                let request = MemoryAllocationRequest {
                    concept_id: format!("concurrent_concept_{}", i),
                    concept_type: ConceptType::Episodic,
                    content: format!("Concurrent test content {}", i),
                    semantic_embedding: Some(generate_test_embedding(256)),
                    priority: AllocationPriority::Normal,
                    resource_requirements: ResourceRequirements::default(),
                    locality_hints: vec![],
                    user_id: format!("concurrent_user_{}", i % 10), // Simulate 10 users
                    request_id: format!("concurrent_req_{}", i),
                    version_info: None,
                };
                
                match timeout(
                    Duration::from_millis(5000),
                    graph.allocate_memory_with_cortical_coordination(request)
                ).await {
                    Ok(Ok(_)) => {
                        success_counter.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(Err(e)) => {
                        println!("Allocation error for concept {}: {:?}", i, e);
                        error_counter.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(_) => {
                        println!("Allocation timeout for concept {}", i);
                        error_counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    
    // Wait for all tasks to complete
    futures::future::join_all(allocation_tasks).await;
    
    let successes = success_counter.load(Ordering::SeqCst);
    let errors = error_counter.load(Ordering::SeqCst);
    
    println!("✓ Concurrent allocation results: {} successes, {} errors", successes, errors);
    
    // Should have very high success rate (>95%)
    let success_rate = successes as f64 / concurrent_operations as f64;
    assert!(success_rate > 0.95, 
           "Success rate {:.2}% below required 95%", success_rate * 100.0);
    
    // Verify all successful allocations are properly stored
    let stored_concepts = brain_graph.get_all_concept_ids().await.unwrap();
    assert!(stored_concepts.len() >= successes,
           "Not all successful allocations were properly stored");
    
    println!("✓ Concurrent memory allocation test passed");
}

#[tokio::test]
async fn test_concurrent_search_operations() {
    let brain_graph = setup_populated_graph_for_search().await;
    let concurrent_searches = 200;
    let search_types = vec![
        SearchType::Semantic,
        SearchType::TTFS,
        SearchType::Hierarchical,
        SearchType::SpreadingActivation,
    ];
    
    println!("🔍 Testing {} concurrent search operations...", concurrent_searches);
    
    let search_queries = vec![
        "artificial intelligence neural networks",
        "machine learning algorithms",
        "cognitive science neuroscience", 
        "computer vision processing",
        "natural language understanding",
    ];
    
    let results_counter = Arc::new(AtomicUsize::new(0));
    let timeout_counter = Arc::new(AtomicUsize::new(0));
    
    let search_tasks: Vec<JoinHandle<()>> = (0..concurrent_searches)
        .map(|i| {
            let graph = brain_graph.clone();
            let query = search_queries[i % search_queries.len()].to_string();
            let search_type = search_types[i % search_types.len()].clone();
            let results_counter = results_counter.clone();
            let timeout_counter = timeout_counter.clone();
            
            tokio::spawn(async move {
                let search_request = SearchRequest {
                    query_text: query,
                    search_type,
                    similarity_threshold: Some(0.7),
                    limit: Some(10),
                    user_context: UserContext::default(),
                    use_ttfs_encoding: true,
                    cortical_area_filter: None,
                };
                
                match timeout(
                    Duration::from_millis(2000),
                    match search_request.search_type {
                        SearchType::Semantic => graph.search_memory_with_semantic_similarity(search_request),
                        SearchType::TTFS => graph.search_memory_with_ttfs(search_request),
                        SearchType::Hierarchical => {
                            let hierarchical_request = HierarchicalSearchRequest {
                                concept_id: "root_concept".to_string(),
                                search_direction: SearchDirection::Descendants,
                                max_depth: 3,
                                include_properties: true,
                                filter_criteria: Some(search_request.query_text),
                            };
                            graph.perform_hierarchical_search(hierarchical_request)
                        }
                        SearchType::SpreadingActivation => {
                            let spreading_request = SpreadingActivationRequest {
                                seed_concept_ids: vec!["concept_1".to_string()],
                                activation_threshold: 0.5,
                                max_hops: 3,
                                decay_factor: 0.8,
                                max_results: 10,
                            };
                            graph.perform_spreading_activation_search(spreading_request)
                        }
                    }
                ).await {
                    Ok(Ok(result)) => {
                        results_counter.fetch_add(result.results.len(), Ordering::SeqCst);
                    }
                    Ok(Err(e)) => {
                        println!("Search error for operation {}: {:?}", i, e);
                    }
                    Err(_) => {
                        println!("Search timeout for operation {}", i);
                        timeout_counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    
    futures::future::join_all(search_tasks).await;
    
    let total_results = results_counter.load(Ordering::SeqCst);
    let timeouts = timeout_counter.load(Ordering::SeqCst);
    
    println!("✓ Concurrent search results: {} total results found, {} timeouts", 
             total_results, timeouts);
    
    // Should have very low timeout rate (<5%)
    let timeout_rate = timeouts as f64 / concurrent_searches as f64;
    assert!(timeout_rate < 0.05,
           "Timeout rate {:.2}% exceeds 5% threshold", timeout_rate * 100.0);
    
    // Should find reasonable number of results
    assert!(total_results > 0, "No search results found in concurrent tests");
    
    println!("✓ Concurrent search operations test passed");
}

#[tokio::test] 
async fn test_concurrent_inheritance_resolution() {
    let brain_graph = setup_inheritance_hierarchy_for_concurrency().await;
    let concurrent_resolutions = 300;
    
    println!("🔗 Testing {} concurrent inheritance resolutions...", concurrent_resolutions);
    
    // Test concepts at different depths in hierarchy
    let test_concepts = vec![
        "depth_1_concept",
        "depth_3_concept", 
        "depth_5_concept",
        "depth_7_concept",
        "depth_10_concept",
    ];
    
    let resolution_counter = Arc::new(AtomicUsize::new(0));
    let cache_hit_counter = Arc::new(AtomicUsize::new(0));
    
    let resolution_tasks: Vec<JoinHandle<()>> = (0..concurrent_resolutions)
        .map(|i| {
            let graph = brain_graph.clone();
            let concept_id = test_concepts[i % test_concepts.len()].to_string();
            let resolution_counter = resolution_counter.clone();
            let cache_hit_counter = cache_hit_counter.clone();
            
            tokio::spawn(async move {
                match timeout(
                    Duration::from_millis(1000),
                    graph.resolve_inherited_properties(&concept_id, true)
                ).await {
                    Ok(Ok(resolved_properties)) => {
                        resolution_counter.fetch_add(1, Ordering::SeqCst);
                        
                        if resolved_properties.cache_hit {
                            cache_hit_counter.fetch_add(1, Ordering::SeqCst);
                        }
                        
                        // Verify inheritance chain is valid
                        assert!(!resolved_properties.inheritance_chain.is_empty(),
                               "Empty inheritance chain for {}", concept_id);
                        assert!(resolved_properties.resolved_properties.len() > 0,
                               "No resolved properties for {}", concept_id);
                    }
                    Ok(Err(e)) => {
                        println!("Inheritance resolution error for {}: {:?}", concept_id, e);
                    }
                    Err(_) => {
                        println!("Inheritance resolution timeout for {}", concept_id);
                    }
                }
            })
        })
        .collect();
    
    futures::future::join_all(resolution_tasks).await;
    
    let total_resolutions = resolution_counter.load(Ordering::SeqCst);
    let cache_hits = cache_hit_counter.load(Ordering::SeqCst);
    
    println!("✓ Inheritance resolution results: {} completed, {} cache hits", 
             total_resolutions, cache_hits);
    
    // Should have high success rate
    let success_rate = total_resolutions as f64 / concurrent_resolutions as f64;
    assert!(success_rate > 0.98,
           "Resolution success rate {:.2}% below 98%", success_rate * 100.0);
    
    // Should have good cache hit rate after initial resolutions
    let cache_hit_rate = cache_hits as f64 / total_resolutions as f64;
    assert!(cache_hit_rate > 0.3,
           "Cache hit rate {:.2}% too low", cache_hit_rate * 100.0);
    
    println!("✓ Concurrent inheritance resolution test passed");
}

#[tokio::test]
async fn test_concurrent_read_write_operations() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    // First, populate with some initial data
    let initial_concepts = 50;
    for i in 0..initial_concepts {
        let request = MemoryAllocationRequest {
            concept_id: format!("base_concept_{}", i),
            concept_type: ConceptType::Semantic,
            content: format!("Base content {}", i),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "setup_user".to_string(),
            request_id: format!("setup_req_{}", i),
            version_info: None,
        };
        
        brain_graph.allocate_memory_with_cortical_coordination(request).await.unwrap();
    }
    
    println!("🔄 Testing concurrent read/write operations...");
    
    let read_operations = 200;
    let write_operations = 100;
    let update_operations = 50;
    
    let read_success_counter = Arc::new(AtomicUsize::new(0));
    let write_success_counter = Arc::new(AtomicUsize::new(0));
    let update_success_counter = Arc::new(AtomicUsize::new(0));
    
    // Spawn read tasks
    let read_tasks: Vec<JoinHandle<()>> = (0..read_operations)
        .map(|i| {
            let graph = brain_graph.clone();
            let counter = read_success_counter.clone();
            
            tokio::spawn(async move {
                let concept_id = format!("base_concept_{}", i % initial_concepts);
                
                match timeout(
                    Duration::from_millis(500),
                    graph.get_concept(&concept_id)
                ).await {
                    Ok(Ok(_)) => {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            })
        })
        .collect();
    
    // Spawn write tasks
    let write_tasks: Vec<JoinHandle<()>> = (0..write_operations)
        .map(|i| {
            let graph = brain_graph.clone();
            let counter = write_success_counter.clone();
            
            tokio::spawn(async move {
                let request = MemoryAllocationRequest {
                    concept_id: format!("concurrent_write_concept_{}", i),
                    concept_type: ConceptType::Episodic,
                    content: format!("Concurrent write content {}", i),
                    semantic_embedding: Some(generate_test_embedding(256)),
                    priority: AllocationPriority::Normal,
                    resource_requirements: ResourceRequirements::default(),
                    locality_hints: vec![],
                    user_id: "write_user".to_string(),
                    request_id: format!("write_req_{}", i),
                    version_info: None,
                };
                
                match timeout(
                    Duration::from_millis(2000),
                    graph.allocate_memory_with_cortical_coordination(request)
                ).await {
                    Ok(Ok(_)) => {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            })
        })
        .collect();
    
    // Spawn update tasks
    let update_tasks: Vec<JoinHandle<()>> = (0..update_operations)
        .map(|i| {
            let graph = brain_graph.clone();
            let counter = update_success_counter.clone();
            
            tokio::spawn(async move {
                let concept_id = format!("base_concept_{}", i % initial_concepts);
                let update_request = MemoryUpdateRequest {
                    concept_id,
                    update_type: UpdateType::ContentModification,
                    new_content: Some(format!("Updated content {}", i)),
                    property_updates: None,
                    relationship_updates: None,
                    metadata: UpdateMetadata::default(),
                };
                
                match timeout(
                    Duration::from_millis(1500),
                    graph.update_memory(update_request)
                ).await {
                    Ok(Ok(_)) => {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            })
        })
        .collect();
    
    // Execute all operations concurrently
    let (read_results, write_results, update_results) = tokio::join!(
        futures::future::join_all(read_tasks),
        futures::future::join_all(write_tasks),
        futures::future::join_all(update_tasks)
    );
    
    let read_successes = read_success_counter.load(Ordering::SeqCst);
    let write_successes = write_success_counter.load(Ordering::SeqCst);
    let update_successes = update_success_counter.load(Ordering::SeqCst);
    
    println!("✓ Concurrent R/W results: {} reads, {} writes, {} updates", 
             read_successes, write_successes, update_successes);
    
    // Validate success rates
    assert!(read_successes as f64 / read_operations as f64 > 0.95,
           "Read success rate too low");
    assert!(write_successes as f64 / write_operations as f64 > 0.90,
           "Write success rate too low");
    assert!(update_successes as f64 / update_operations as f64 > 0.85,
           "Update success rate too low");
    
    // Verify data consistency after concurrent operations
    let final_concept_count = brain_graph.get_concept_count().await.unwrap();
    assert!(final_concept_count >= initial_concepts + write_successes,
           "Concept count inconsistent after concurrent operations");
    
    println!("✓ Concurrent read/write operations test passed");
}

#[tokio::test]
async fn test_deadlock_prevention() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    println!("🔒 Testing deadlock prevention mechanisms...");
    
    // Create concepts that will be involved in potential deadlock scenarios
    let concept_pairs = vec![
        ("deadlock_test_a", "deadlock_test_b"),
        ("deadlock_test_c", "deadlock_test_d"),
        ("deadlock_test_e", "deadlock_test_f"),
    ];
    
    // Allocate initial concepts
    for (concept_a, concept_b) in &concept_pairs {
        let request_a = create_test_allocation_request(concept_a, ConceptType::Abstract);
        let request_b = create_test_allocation_request(concept_b, ConceptType::Specific);
        
        brain_graph.allocate_memory_with_cortical_coordination(request_a).await.unwrap();
        brain_graph.allocate_memory_with_cortical_coordination(request_b).await.unwrap();
    }
    
    let deadlock_test_duration = Duration::from_millis(5000);
    let operation_counter = Arc::new(AtomicUsize::new(0));
    
    // Spawn tasks that could potentially deadlock
    let deadlock_tasks: Vec<JoinHandle<()>> = concept_pairs.into_iter()
        .enumerate()
        .flat_map(|(i, (concept_a, concept_b))| {
            let graph_1 = brain_graph.clone();
            let graph_2 = brain_graph.clone();
            let counter_1 = operation_counter.clone();
            let counter_2 = operation_counter.clone();
            let concept_a = concept_a.to_string();
            let concept_b = concept_b.to_string();
            
            vec![
                // Task 1: A -> B relationship then property update
                tokio::spawn(async move {
                    let start_time = std::time::Instant::now();
                    while start_time.elapsed() < deadlock_test_duration {
                        // Create relationship A -> B
                        if let Ok(_) = timeout(
                            Duration::from_millis(100),
                            graph_1.create_relationship(
                                &concept_a,
                                &concept_b,
                                RelationshipType::Association,
                                RelationshipMetadata::default(),
                            )
                        ).await {
                            // Then update property on A
                            let property_update = ConceptProperty {
                                key: format!("deadlock_prop_{}", i),
                                value: PropertyValue::String("test_value".to_string()),
                                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                                metadata: PropertyMetadata::default(),
                            };
                            
                            let _ = timeout(
                                Duration::from_millis(100),
                                graph_1.add_concept_property(&concept_a, property_update)
                            ).await;
                            
                            counter_1.fetch_add(1, Ordering::SeqCst);
                        }
                        
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }),
                
                // Task 2: B -> A relationship then property update
                tokio::spawn(async move {
                    let start_time = std::time::Instant::now();
                    while start_time.elapsed() < deadlock_test_duration {
                        // Create relationship B -> A
                        if let Ok(_) = timeout(
                            Duration::from_millis(100),
                            graph_2.create_relationship(
                                &concept_b,
                                &concept_a,
                                RelationshipType::Dependency,
                                RelationshipMetadata::default(),
                            )
                        ).await {
                            // Then update property on B
                            let property_update = ConceptProperty {
                                key: format!("deadlock_prop_{}", i),
                                value: PropertyValue::String("test_value".to_string()),
                                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                                metadata: PropertyMetadata::default(),
                            };
                            
                            let _ = timeout(
                                Duration::from_millis(100),
                                graph_2.add_concept_property(&concept_b, property_update)
                            ).await;
                            
                            counter_2.fetch_add(1, Ordering::SeqCst);
                        }
                        
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }),
            ]
        })
        .collect();
    
    // Wait for all tasks to complete or timeout
    let task_results = timeout(
        deadlock_test_duration + Duration::from_millis(1000),
        futures::future::join_all(deadlock_tasks)
    ).await;
    
    let total_operations = operation_counter.load(Ordering::SeqCst);
    
    // Verify tasks completed (no deadlock occurred)
    assert!(task_results.is_ok(), "Deadlock detected - tasks did not complete within timeout");
    
    println!("✓ Deadlock prevention test passed: {} operations completed", total_operations);
    
    // Verify system is still responsive after stress test
    let post_test_concept = brain_graph
        .get_concept("deadlock_test_a")
        .await
        .expect("System unresponsive after deadlock test");
    
    assert!(!post_test_concept.concept_id.is_empty(), "System state corrupted");
    
    println!("✓ System remains responsive after deadlock prevention test");
}

// Helper functions
async fn setup_concurrency_test_graph() -> Arc<BrainEnhancedGraphCore> {
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
        .expect("Failed to create concurrency test graph")
    )
}

async fn setup_populated_graph_for_search() -> Arc<BrainEnhancedGraphCore> {
    let graph = setup_concurrency_test_graph().await;
    
    // Populate with test data for search operations
    let test_concepts = vec![
        ("concept_1", "artificial intelligence machine learning algorithms"),
        ("concept_2", "neural networks deep learning models"),
        ("concept_3", "computer vision image processing"),
        ("concept_4", "natural language processing understanding"),
        ("concept_5", "cognitive science neuroscience research"),
        ("root_concept", "knowledge representation reasoning"),
    ];
    
    for (id, content) in test_concepts {
        let request = MemoryAllocationRequest {
            concept_id: id.to_string(),
            concept_type: ConceptType::Semantic,
            content: content.to_string(),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "search_setup_user".to_string(),
            request_id: format!("search_setup_{}", id),
            version_info: None,
        };
        
        graph.allocate_memory_with_cortical_coordination(request).await.unwrap();
    }
    
    graph
}

async fn setup_inheritance_hierarchy_for_concurrency() -> Arc<BrainEnhancedGraphCore> {
    let graph = setup_concurrency_test_graph().await;
    
    // Create inheritance hierarchy for concurrent testing
    for depth in 1..=10 {
        let concept_id = format!("depth_{}_concept", depth);
        let parent_id = if depth == 1 {
            None
        } else {
            Some(format!("depth_{}_concept", depth - 1))
        };
        
        let request = MemoryAllocationRequest {
            concept_id: concept_id.clone(),
            concept_type: ConceptType::Specific,
            content: format!("Concept at depth {}", depth),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "hierarchy_setup_user".to_string(),
            request_id: format!("hierarchy_setup_{}", depth),
            version_info: None,
        };
        
        graph.allocate_memory_with_cortical_coordination(request).await.unwrap();
        
        if let Some(parent_id) = parent_id {
            graph.create_inheritance_relationship(
                &parent_id,
                &concept_id,
                InheritanceType::DirectSubclass,
            ).await.unwrap();
        }
        
        // Add some properties for inheritance testing
        graph.add_concept_property(
            &concept_id,
            ConceptProperty {
                key: format!("depth_{}_property", depth),
                value: PropertyValue::String(format!("value_at_depth_{}", depth)),
                inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                metadata: PropertyMetadata::default(),
            },
        ).await.unwrap();
    }
    
    graph
}

fn create_test_allocation_request(concept_id: &str, concept_type: ConceptType) -> MemoryAllocationRequest {
    MemoryAllocationRequest {
        concept_id: concept_id.to_string(),
        concept_type,
        content: format!("Test content for {}", concept_id),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "test_user".to_string(),
        request_id: format!("test_req_{}", concept_id),
        version_info: None,
    }
}

fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}
```

### 2. Create High Load Stress Testing
```rust
// tests/concurrency/stress_tests.rs
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

#[tokio::test]
async fn test_extreme_concurrency_load() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    println!("⚡ Starting extreme concurrency load test...");
    
    let extreme_load_operations = 2000;
    let max_concurrent_tasks = 1000;
    
    let start_time = Instant::now();
    let completed_operations = Arc::new(AtomicUsize::new(0));
    let failed_operations = Arc::new(AtomicUsize::new(0));
    
    // Create semaphore to limit concurrent tasks
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_tasks));
    
    let mut task_handles = Vec::new();
    
    for i in 0..extreme_load_operations {
        let graph = brain_graph.clone();
        let semaphore = semaphore.clone();
        let completed = completed_operations.clone();
        let failed = failed_operations.clone();
        
        let handle = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            
            let operation_type = i % 4;
            let result = match operation_type {
                0 => {
                    // Allocation operation
                    let request = MemoryAllocationRequest {
                        concept_id: format!("stress_concept_{}", i),
                        concept_type: ConceptType::Episodic,
                        content: format!("Stress test content {}", i),
                        semantic_embedding: Some(generate_test_embedding(128)),
                        priority: AllocationPriority::Normal,
                        resource_requirements: ResourceRequirements::default(),
                        locality_hints: vec![],
                        user_id: format!("stress_user_{}", i % 50),
                        request_id: format!("stress_req_{}", i),
                        version_info: None,
                    };
                    
                    timeout(
                        Duration::from_millis(10000),
                        graph.allocate_memory_with_cortical_coordination(request)
                    ).await.map(|r| r.map(|_| ()))
                }
                1 => {
                    // Search operation
                    let search_request = SearchRequest {
                        query_text: format!("stress test query {}", i % 10),
                        search_type: SearchType::Semantic,
                        similarity_threshold: Some(0.7),
                        limit: Some(5),
                        user_context: UserContext::default(),
                        use_ttfs_encoding: false,
                        cortical_area_filter: None,
                    };
                    
                    timeout(
                        Duration::from_millis(5000),
                        graph.search_memory_with_semantic_similarity(search_request)
                    ).await.map(|r| r.map(|_| ()))
                }
                2 => {
                    // Read operation
                    let concept_id = format!("stress_concept_{}", (i / 4) * 4); // Read previously created concepts
                    timeout(
                        Duration::from_millis(1000),
                        graph.get_concept(&concept_id)
                    ).await.map(|r| r.map(|_| ()))
                }
                _ => {
                    // Property update operation
                    let concept_id = format!("stress_concept_{}", (i / 4) * 4);
                    let property = ConceptProperty {
                        key: format!("stress_prop_{}", i),
                        value: PropertyValue::String(format!("stress_value_{}", i)),
                        inheritance_behavior: PropertyInheritanceBehavior::Inherit,
                        metadata: PropertyMetadata::default(),
                    };
                    
                    timeout(
                        Duration::from_millis(2000),
                        graph.add_concept_property(&concept_id, property)
                    ).await.map(|r| r.map(|_| ()))
                }
            };
            
            match result {
                Ok(Ok(())) => {
                    completed.fetch_add(1, Ordering::SeqCst);
                }
                _ => {
                    failed.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        
        task_handles.push(handle);
    }
    
    // Wait for all operations to complete
    futures::future::join_all(task_handles).await;
    
    let total_time = start_time.elapsed();
    let completed = completed_operations.load(Ordering::SeqCst);
    let failed = failed_operations.load(Ordering::SeqCst);
    
    let success_rate = completed as f64 / extreme_load_operations as f64;
    let throughput = completed as f64 / total_time.as_secs_f64();
    
    println!("✓ Extreme load test results:");
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Completed operations: {}", completed);
    println!("  Failed operations: {}", failed);
    println!("  Success rate: {:.2}%", success_rate * 100.0);
    println!("  Throughput: {:.0} ops/sec", throughput);
    
    // Success criteria for extreme load
    assert!(success_rate > 0.80, "Success rate {:.2}% below 80% threshold", success_rate * 100.0);
    assert!(throughput > 100.0, "Throughput {:.0} ops/sec below 100 threshold", throughput);
    
    // Verify system stability after extreme load
    let post_load_concept = brain_graph
        .get_concept("stress_concept_0")
        .await
        .expect("System unresponsive after extreme load test");
    
    assert!(!post_load_concept.concept_id.is_empty(), "System state corrupted after extreme load");
    
    println!("✓ Extreme concurrency load test passed");
}

#[tokio::test]
async fn test_sustained_load_with_monitoring() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    println!("📊 Running sustained load test with performance monitoring...");
    
    let test_duration = Duration::from_secs(30);
    let operations_per_second = 100;
    let monitoring_interval = Duration::from_millis(500);
    
    let start_time = Instant::now();
    let operation_counter = Arc::new(AtomicUsize::new(0));
    let error_counter = Arc::new(AtomicUsize::new(0));
    let should_stop = Arc::new(AtomicBool::new(false));
    
    // Performance monitoring task
    let monitor_handle = {
        let graph = brain_graph.clone();
        let should_stop = should_stop.clone();
        let operation_counter = operation_counter.clone();
        
        tokio::spawn(async move {
            let mut monitoring_data = Vec::new();
            
            while !should_stop.load(Ordering::SeqCst) {
                let current_ops = operation_counter.load(Ordering::SeqCst);
                let elapsed = start_time.elapsed();
                
                let current_throughput = current_ops as f64 / elapsed.as_secs_f64();
                let cache_stats = graph.get_cache_statistics().await;
                
                monitoring_data.push(MonitoringPoint {
                    timestamp: elapsed,
                    operations_completed: current_ops,
                    current_throughput,
                    cache_hit_rate: cache_stats.hit_rate,
                    memory_usage_mb: get_memory_usage_mb(),
                });
                
                if monitoring_data.len() % 10 == 0 {
                    let latest = monitoring_data.last().unwrap();
                    println!("  Monitoring: {:.1}s, {} ops, {:.0} ops/sec, {:.1}% cache hits", 
                            latest.timestamp.as_secs_f64(),
                            latest.operations_completed,
                            latest.current_throughput,
                            latest.cache_hit_rate * 100.0);
                }
                
                tokio::time::sleep(monitoring_interval).await;
            }
        })
    };
    
    // Load generation task
    let load_handle = {
        let graph = brain_graph.clone();
        let operation_counter = operation_counter.clone();
        let error_counter = error_counter.clone();
        
        tokio::spawn(async move {
            let mut operation_id = 0;
            let interval = Duration::from_millis(1000 / operations_per_second as u64);
            
            while start_time.elapsed() < test_duration {
                let graph = graph.clone();
                let op_counter = operation_counter.clone();
                let err_counter = error_counter.clone();
                let current_op_id = operation_id;
                operation_id += 1;
                
                tokio::spawn(async move {
                    let request = MemoryAllocationRequest {
                        concept_id: format!("sustained_concept_{}", current_op_id),
                        concept_type: ConceptType::Episodic,
                        content: format!("Sustained test content {}", current_op_id),
                        semantic_embedding: Some(generate_test_embedding(128)),
                        priority: AllocationPriority::Normal,
                        resource_requirements: ResourceRequirements::default(),
                        locality_hints: vec![],
                        user_id: format!("sustained_user_{}", current_op_id % 20),
                        request_id: format!("sustained_req_{}", current_op_id),
                        version_info: None,
                    };
                    
                    match graph.allocate_memory_with_cortical_coordination(request).await {
                        Ok(_) => {
                            op_counter.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(_) => {
                            err_counter.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
                
                tokio::time::sleep(interval).await;
            }
        })
    };
    
    // Wait for load generation to complete
    load_handle.await.unwrap();
    
    // Stop monitoring
    should_stop.store(true, Ordering::SeqCst);
    monitor_handle.await.unwrap();
    
    let total_operations = operation_counter.load(Ordering::SeqCst);
    let total_errors = error_counter.load(Ordering::SeqCst);
    let final_throughput = total_operations as f64 / test_duration.as_secs_f64();
    
    println!("✓ Sustained load test completed:");
    println!("  Duration: {:.1}s", test_duration.as_secs_f64());
    println!("  Total operations: {}", total_operations);
    println!("  Total errors: {}", total_errors);
    println!("  Average throughput: {:.0} ops/sec", final_throughput);
    
    // Validate sustained performance
    assert!(final_throughput >= operations_per_second as f64 * 0.8,
           "Sustained throughput {:.0} below 80% of target {}", 
           final_throughput, operations_per_second);
    
    let error_rate = total_errors as f64 / (total_operations + total_errors) as f64;
    assert!(error_rate < 0.05, "Error rate {:.2}% exceeds 5% threshold", error_rate * 100.0);
    
    println!("✓ Sustained load test passed");
}

#[derive(Debug, Clone)]
struct MonitoringPoint {
    timestamp: Duration,
    operations_completed: usize,
    current_throughput: f64,
    cache_hit_rate: f64,
    memory_usage_mb: u64,
}

fn get_memory_usage_mb() -> u64 {
    // Simplified memory usage calculation
    // In a real implementation, this would use system monitoring
    42 // Placeholder value
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Thread safety validated for all shared data structures
- [ ] Deadlock prevention mechanisms work under stress
- [ ] Concurrent read/write operations maintain data consistency
- [ ] Resource contention handled gracefully
- [ ] System remains responsive under high concurrency

### Performance Requirements
- [ ] 1000+ concurrent operations supported
- [ ] Deadlock detection/prevention < 100ms overhead
- [ ] Thread contention minimal (< 5% performance degradation)
- [ ] Memory usage stable under concurrent load
- [ ] Cache effectiveness maintained under concurrency

### Testing Requirements
- [ ] All thread safety test scenarios pass
- [ ] Stress tests demonstrate system stability
- [ ] Performance monitoring shows acceptable metrics
- [ ] No data corruption under extreme load

## Validation Steps

1. **Run thread safety tests**:
   ```bash
   cargo test --test thread_safety_tests --release
   ```

2. **Execute stress tests**:
   ```bash
   cargo test --test stress_tests --release -- --test-threads=1
   ```

3. **Run deadlock prevention tests**:
   ```bash
   cargo test test_deadlock_prevention --release
   ```

4. **Monitor sustained load performance**:
   ```bash
   cargo test test_sustained_load_with_monitoring --release
   ```

## Files to Create/Modify
- `tests/concurrency/thread_safety_tests.rs` - Thread safety test suite
- `tests/concurrency/stress_tests.rs` - High load stress tests
- `tests/concurrency/deadlock_tests.rs` - Deadlock prevention tests
- `tests/concurrency/mod.rs` - Concurrency test module definitions

## Success Metrics
- Thread safety: 100% pass rate under 1000+ concurrent operations
- Deadlock prevention: 0 deadlocks detected in 5-minute stress test
- System stability: < 5% performance degradation under high load
- Resource contention: < 100ms maximum lock wait time
- Data consistency: 100% accuracy under concurrent access

## Next Task
Upon completion, proceed to **35_production_readiness.md** for final production readiness validation.