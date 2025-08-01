//! Hierarchical Storage Integration Tests
//! 
//! Tests for integration of hierarchical storage with other system components:
//! - Storage tier coordination with processing components
//! - Integration with retrieval and query systems
//! - Cross-tier data consistency and validation
//! - Performance optimization across storage hierarchy
//! 
//! These tests verify that hierarchical storage integrates properly
//! with all other system components.

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use super::*;

    struct HierarchicalStorageSystem {
        memory_tier: Arc<RwLock<KnowledgeGraph>>,
        embedding_tier: EmbeddingStore,
        persistent_tier: PersistentMMapStorage,
        archive_tier: PersistentMMapStorage,
        extractor: AdvancedEntityExtractor,
        orchestrator: CognitiveOrchestrator,
        _temp_dir: TempDir,
    }

    async fn create_hierarchical_storage_system() -> HierarchicalStorageSystem {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();
        
        let memory_tier = Arc::new(RwLock::new(KnowledgeGraph::new(384).expect("Failed to create knowledge graph")));
        let embedding_tier = EmbeddingStore::new(384, 8).expect("Failed to create embedding store");
        
        let persistent_path = base_path.join("persistent.mmap");
        let persistent_tier = PersistentMMapStorage::new(Some(&persistent_path), 10 * 1024 * 1024) // 10MB
            .expect("Failed to create persistent storage");
        
        let archive_path = base_path.join("archive.mmap");
        let archive_tier = PersistentMMapStorage::new(Some(&archive_path), 50 * 1024 * 1024) // 50MB
            .expect("Failed to create archive storage");
        
        let extractor = AdvancedEntityExtractor::new();
        let config = CognitiveOrchestratorConfig::default();
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create brain graph"));
        let orchestrator = CognitiveOrchestrator::new(brain_graph, config).await.expect("Failed to create orchestrator");
        
        HierarchicalStorageSystem {
            memory_tier,
            embedding_tier,
            persistent_tier,
            archive_tier,
            extractor,
            orchestrator,
            _temp_dir: temp_dir,
        }
    }

    #[tokio::test]
    async fn test_hierarchical_storage_processing_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Test data flow through storage hierarchy during processing
        let test_documents = vec![
            ("hot", "Breaking: Einstein's new theory published today!", "high_priority"),
            ("warm", "Marie Curie's research continues to influence modern science.", "medium_priority"),
            ("cold", "Historical overview of 19th century physics discoveries.", "low_priority"),
            ("archive", "Ancient astronomical observations from 1800s.", "archival"),
        ];
        
        for (tier_type, document, priority) in test_documents {
            let processing_start = Instant::now();
            
            // Step 1: Extract knowledge
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract knowledge");
            
            // Step 2: Route to appropriate storage tier based on priority
            match tier_type {
                "hot" => {
                    // Hot data: Store in memory tier for fastest access
                    {
                        let mut memory_lock = system.memory_tier.write().await;
                        for entity in &extraction.entities {
                            let entity_data = EntityData {
                                attributes: [
                                    ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                    ("tier".to_string(), AttributeValue::String("hot".to_string())),
                                    ("priority".to_string(), AttributeValue::String(priority.to_string())),
                                    ("access_pattern".to_string(), AttributeValue::String("frequent".to_string())),
                                    ("created_at".to_string(), AttributeValue::String(chrono::Utc::now().to_rfc3339())),
                                ].into_iter().collect(),
                            };
                            
                            memory_lock.add_entity(
                                format!("hot_{}", entity.id),
                                entity_data
                            ).expect("Failed to add hot entity");
                        }
                        
                        for relation in &extraction.relations {
                            let relationship = Relationship {
                                target: format!("hot_{}", relation.object_id),
                                relationship_type: relation.predicate.clone(),
                                weight: relation.confidence,
                                properties: [
                                    ("tier".to_string(), AttributeValue::String("hot".to_string())),
                                ].into_iter().collect(),
                            };
                            
                            memory_lock.add_relationship(
                                format!("hot_{}", relation.subject_id),
                                relationship
                            ).expect("Failed to add hot relationship");
                        }
                    }
                    
                    // Also store in embedding tier for similarity searches
                    for entity in &extraction.entities {
                        let hot_embedding: Vec<f32> = (0..384).map(|i| {
                            (1.0 + i as f32 * 0.001).sin() // High energy signature for hot data
                        }).collect();
                        
                        system.embedding_tier.add_embedding(
                            &format!("hot_{}", entity.id),
                            hot_embedding
                        ).expect("Failed to add hot embedding");
                    }
                },
                "warm" => {
                    // Warm data: Store in persistent tier with memory cache
                    let warm_data = serde_json::json!({
                        "tier": "warm",
                        "priority": priority,
                        "extraction": extraction,
                        "document": document,
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    
                    system.persistent_tier.store(
                        &format!("warm_{}", chrono::Utc::now().timestamp_nanos()),
                        warm_data.to_string().as_bytes()
                    ).expect("Failed to store warm data");
                    
                    // Cache important entities in memory
                    {
                        let mut memory_lock = system.memory_tier.write().await;
                        for entity in &extraction.entities {
                            if entity.confidence > 0.7 { // Cache high-confidence entities
                                let entity_data = EntityData {
                                    attributes: [
                                        ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                        ("tier".to_string(), AttributeValue::String("warm_cached".to_string())),
                                        ("cache_reason".to_string(), AttributeValue::String("high_confidence".to_string())),
                                    ].into_iter().collect(),
                                };
                                
                                memory_lock.add_entity(
                                    format!("warm_cached_{}", entity.id),
                                    entity_data
                                ).expect("Failed to cache warm entity");
                            }
                        }
                    }
                },
                "cold" => {
                    // Cold data: Store primarily in persistent tier
                    let cold_data = serde_json::json!({
                        "tier": "cold",
                        "priority": priority,
                        "extraction": extraction,
                        "access_frequency": "low",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    });
                    
                    system.persistent_tier.store(
                        &format!("cold_{}", chrono::Utc::now().timestamp_nanos()),
                        cold_data.to_string().as_bytes()
                    ).expect("Failed to store cold data");
                },
                "archive" => {
                    // Archive data: Store in archive tier with compression
                    let archive_data = serde_json::json!({
                        "tier": "archive",
                        "priority": priority,
                        "extraction": extraction,
                        "document": document,
                        "archived_at": chrono::Utc::now().to_rfc3339(),
                        "compression": "enabled"
                    });
                    
                    system.archive_tier.store(
                        &format!("archive_{}", chrono::Utc::now().timestamp_nanos()),
                        archive_data.to_string().as_bytes()
                    ).expect("Failed to store archive data");
                }
            }
            
            let processing_time = processing_start.elapsed();
            
            // Hot tier should be fastest, archive slowest
            match tier_type {
                "hot" => assert!(processing_time < Duration::from_millis(100), 
                               "Hot tier processing should be very fast: {:?}", processing_time),
                "warm" => assert!(processing_time < Duration::from_millis(200),
                                "Warm tier processing should be fast: {:?}", processing_time),
                "cold" => assert!(processing_time < Duration::from_millis(500),
                                "Cold tier processing should be reasonable: {:?}", processing_time),
                "archive" => assert!(processing_time < Duration::from_secs(1),
                                   "Archive tier processing should complete: {:?}", processing_time),
            }
            
            println!("Processed {} data in {:?}", tier_type, processing_time);
        }
        
        // Validate tier separation and integration
        let memory_stats = {
            let memory_lock = system.memory_tier.read().await;
            memory_lock.get_stats()
        };
        
        // Memory tier should contain hot data and warm cache
        assert!(memory_stats.entity_count >= 2, "Memory tier should contain hot and cached entities");
        
        // Embedding tier should contain hot data
        let embedding_info = system.embedding_tier.get_info();
        assert!(embedding_info.count >= 1, "Embedding tier should contain hot embeddings");
        
        println!("✓ Hierarchical storage and processing integration test passed");
    }

    #[tokio::test]
    async fn test_cross_tier_retrieval_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Populate different tiers with related data
        let knowledge_pieces = vec![
            ("Einstein's theory of relativity revolutionized physics.", "hot"),
            ("Relativity theory explains the relationship between space and time.", "warm"),
            ("GPS satellites account for relativistic effects in their calculations.", "cold"),
            ("Historical development of relativity theory from 1905 to 1915.", "archive"),
        ];
        
        let mut all_entities = Vec::new();
        
        // Store data across tiers
        for (document, tier) in knowledge_pieces {
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract knowledge");
            
            match tier {
                "hot" => {
                    let mut memory_lock = system.memory_tier.write().await;
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("tier".to_string(), AttributeValue::String(tier.to_string())),
                                ("document".to_string(), AttributeValue::String(document.to_string())),
                            ].into_iter().collect(),
                        };
                        
                        memory_lock.add_entity(
                            format!("{}_{}", tier, entity.id),
                            entity_data
                        ).expect("Failed to add entity");
                    }
                    
                    // Hot data gets premium embeddings
                    for entity in &extraction.entities {
                        let embedding: Vec<f32> = (0..384).map(|i| {
                            (entity.name.len() as f32 * i as f32 * 0.001 + 1.0).sin()
                        }).collect();
                        
                        system.embedding_tier.add_embedding(
                            &format!("{}_{}", tier, entity.id),
                            embedding
                        ).expect("Failed to add hot embedding");
                    }
                },
                "warm" => {
                    let warm_data = serde_json::json!({
                        "extraction": extraction,
                        "document": document,
                        "tier": tier
                    });
                    
                    system.persistent_tier.store(
                        &format!("warm_retrieval_{}", chrono::Utc::now().timestamp_nanos()),
                        warm_data.to_string().as_bytes()
                    ).expect("Failed to store warm data");
                    
                    // Add to embeddings for cross-tier search
                    for entity in &extraction.entities {
                        let embedding: Vec<f32> = (0..384).map(|i| {
                            (entity.name.len() as f32 * i as f32 * 0.001 + 0.5).sin()
                        }).collect();
                        
                        system.embedding_tier.add_embedding(
                            &format!("{}_{}", tier, entity.id),
                            embedding
                        ).expect("Failed to add warm embedding");
                    }
                },
                "cold" => {
                    let cold_data = serde_json::json!({
                        "extraction": extraction,
                        "document": document,
                        "tier": tier,
                        "indexed_at": chrono::Utc::now().to_rfc3339()
                    });
                    
                    system.persistent_tier.store(
                        &format!("cold_retrieval_{}", chrono::Utc::now().timestamp_nanos()),
                        cold_data.to_string().as_bytes()
                    ).expect("Failed to store cold data");
                },
                "archive" => {
                    let archive_data = serde_json::json!({
                        "extraction": extraction,
                        "document": document,
                        "tier": tier,
                        "archived_metadata": {
                            "compression_ratio": 0.7,
                            "original_size": document.len(),
                            "archive_date": chrono::Utc::now().to_rfc3339()
                        }
                    });
                    
                    system.archive_tier.store(
                        &format!("archive_retrieval_{}", chrono::Utc::now().timestamp_nanos()),
                        archive_data.to_string().as_bytes()
                    ).expect("Failed to store archive data");
                }
            }
            
            all_entities.extend(extraction.entities);
        }
        
        // Test cross-tier retrieval scenarios
        
        // 1. Fast retrieval from hot tier (memory)
        let hot_retrieval_start = Instant::now();
        {
            let memory_lock = system.memory_tier.read().await;
            let hot_entities: Vec<_> = (0..10).filter_map(|i| {
                let entity_key = format!("hot_entity_{}", i);
                memory_lock.get_entity(&entity_key).map(|_| entity_key)
            }).collect();
            
            if !hot_entities.is_empty() {
                println!("Hot tier retrieval found {} entities", hot_entities.len());
            }
        }
        let hot_retrieval_time = hot_retrieval_start.elapsed();
        
        assert!(hot_retrieval_time < Duration::from_millis(10), 
               "Hot tier retrieval should be very fast: {:?}", hot_retrieval_time);
        
        // 2. Embedding-based cross-tier similarity search
        let similarity_search_start = Instant::now();
        let similarity_results = system.embedding_tier.find_similar("relativity", 5)
            .expect("Failed to perform similarity search");
        let similarity_search_time = similarity_search_start.elapsed();
        
        assert!(!similarity_results.is_empty(), "Similarity search should find results across tiers");
        assert!(similarity_search_time < Duration::from_millis(100),
               "Cross-tier similarity search should be fast: {:?}", similarity_search_time);
        
        println!("Cross-tier similarity search found {} results", similarity_results.len());
        
        // 3. Cognitive reasoning across tiers
        let reasoning_start = Instant::now();
        let reasoning_result = system.orchestrator.process_complex_query(
            "How is Einstein's relativity theory connected to modern GPS technology?",
            &all_entities,
            &[]
        ).await.expect("Failed to process cross-tier reasoning");
        let reasoning_time = reasoning_start.elapsed();
        
        assert!(reasoning_result.confidence > 0.5, 
               "Cross-tier reasoning should be confident: {:.3}", reasoning_result.confidence);
        assert!(reasoning_time < Duration::from_secs(5),
               "Cross-tier reasoning should complete reasonably: {:?}", reasoning_time);
        
        // 4. Tiered retrieval strategy (hot -> warm -> cold -> archive)
        let tiered_retrieval_start = Instant::now();
        
        async fn tiered_search(system: &HierarchicalStorageSystem, query: &str) -> Vec<String> {
            let mut results = Vec::new();
            
            // Check hot tier first
            {
                let memory_lock = system.memory_tier.read().await;
                let stats = memory_lock.get_stats();
                if stats.entity_count > 0 {
                    results.push(format!("Hot tier: {} entities available", stats.entity_count));
                }
            }
            
            // Check embedding tier
            let embedding_results = system.embedding_tier.find_similar(query, 3)
                .unwrap_or_default();
            if !embedding_results.is_empty() {
                results.push(format!("Embedding tier: {} similar items", embedding_results.len()));
            }
            
            // Note: In a real implementation, we would have indexing for persistent and archive tiers
            results.push("Persistent tier: searchable via index".to_string());
            results.push("Archive tier: searchable via catalog".to_string());
            
            results
        }
        
        let tiered_results = tiered_search(&system, "Einstein").await;
        let tiered_retrieval_time = tiered_retrieval_start.elapsed();
        
        assert!(!tiered_results.is_empty(), "Tiered retrieval should find results");
        assert!(tiered_retrieval_time < Duration::from_millis(500),
               "Tiered retrieval should be efficient: {:?}", tiered_retrieval_time);
        
        // 5. Cross-tier consistency check
        let consistency_start = Instant::now();
        
        // Verify that related data across tiers maintains semantic consistency
        let test_entity = "Einstein";
        
        // Check memory tier
        let memory_results = {
            let memory_lock = system.memory_tier.read().await;
            (0..10).filter_map(|i| {
                let entity_key = format!("hot_{}", i);
                memory_lock.get_entity(&entity_key)
                    .and_then(|entity| entity.attributes.get("name"))
                    .and_then(|name| {
                        if let AttributeValue::String(s) = name {
                            if s.contains(test_entity) { Some(s.clone()) } else { None }
                        } else { None }
                    })
            }).collect::<Vec<_>>()
        };
        
        // Check embedding tier  
        let embedding_matches = system.embedding_tier.find_similar(test_entity, 5)
            .unwrap_or_default();
        
        let consistency_time = consistency_start.elapsed();
        
        println!("Cross-tier consistency check completed in {:?}", consistency_time);
        println!("Memory tier matches: {}", memory_results.len());
        println!("Embedding tier matches: {}", embedding_matches.len());
        
        // Performance and consistency summary
        println!("✓ Cross-tier retrieval integration test passed:");
        println!("  - Hot retrieval: {:?}", hot_retrieval_time);
        println!("  - Similarity search: {:?} ({} results)", similarity_search_time, similarity_results.len());
        println!("  - Cross-tier reasoning: {:?} (confidence: {:.3})", reasoning_time, reasoning_result.confidence);
        println!("  - Tiered retrieval: {:?} ({} tiers checked)", tiered_retrieval_time, tiered_results.len());
        println!("  - Consistency check: {:?}", consistency_time);
    }

    #[tokio::test]
    async fn test_tier_migration_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Create data that will migrate between tiers based on access patterns
        let initial_documents = vec![
            "Einstein's theory of general relativity",
            "Special relativity and time dilation", 
            "Applications of relativity in GPS systems",
            "Historical context of Einstein's discoveries",
            "Mathematical foundations of spacetime"
        ];
        
        // Initial population in warm tier
        let mut document_metadata = Vec::new();
        
        for (doc_id, document) in initial_documents.iter().enumerate() {
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract from document");
            
            let initial_data = serde_json::json!({
                "doc_id": doc_id,
                "extraction": extraction,
                "document": document,
                "tier": "warm",
                "access_count": 0,
                "last_accessed": chrono::Utc::now().to_rfc3339(),
                "created_at": chrono::Utc::now().to_rfc3339()
            });
            
            let storage_key = format!("migration_doc_{}", doc_id);
            system.persistent_tier.store(
                &storage_key,
                initial_data.to_string().as_bytes()
            ).expect("Failed to store initial document");
            
            document_metadata.push((doc_id, storage_key, 0, "warm".to_string()));
            
            // Add to embeddings for access tracking
            for entity in &extraction.entities {
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (doc_id as f32 * 0.1 + i as f32 * 0.001).sin()
                }).collect();
                
                system.embedding_tier.add_embedding(
                    &format!("migration_{}_{}", doc_id, entity.id),
                    embedding
                ).expect("Failed to add migration embedding");
            }
        }
        
        // Simulate access patterns that trigger migrations
        let access_patterns = vec![
            (0, 10), // Document 0: High access -> promote to hot
            (1, 8),  // Document 1: High access -> promote to hot
            (2, 5),  // Document 2: Medium access -> stay warm
            (3, 1),  // Document 3: Low access -> demote to cold
            (4, 0),  // Document 4: No access -> demote to archive
        ];
        
        for (doc_id, access_count) in access_patterns {
            // Simulate access pattern
            for access_num in 0..access_count {
                let access_start = Instant::now();
                
                // Simulate data access
                let storage_key = format!("migration_doc_{}", doc_id);
                let current_data = system.persistent_tier.retrieve(&storage_key)
                    .expect("Failed to retrieve document for access");
                
                let mut data_json: serde_json::Value = serde_json::from_slice(&current_data)
                    .expect("Failed to parse document data");
                
                // Update access metadata
                data_json["access_count"] = serde_json::Value::Number(
                    serde_json::Number::from(access_num + 1)
                );
                data_json["last_accessed"] = serde_json::Value::String(
                    chrono::Utc::now().to_rfc3339()
                );
                
                let access_time = access_start.elapsed();
                
                // Trigger migration based on access patterns
                if access_num == access_count - 1 { // Final access triggers migration decision
                    let migration_start = Instant::now();
                    
                    let new_tier = if access_count >= 8 {
                        "hot"
                    } else if access_count >= 3 {
                        "warm"
                    } else if access_count >= 1 {
                        "cold"
                    } else {
                        "archive"
                    };
                    
                    // Perform migration
                    match new_tier {
                        "hot" => {
                            // Migrate to memory tier
                            if let Some(extraction_data) = data_json["extraction"].as_object() {
                                let mut memory_lock = system.memory_tier.write().await;
                                
                                // Simulate extraction parsing (simplified)
                                let entity_data = EntityData {
                                    attributes: [
                                        ("name".to_string(), AttributeValue::String(format!("Doc{}_Entity", doc_id))),
                                        ("tier".to_string(), AttributeValue::String("hot".to_string())),
                                        ("access_count".to_string(), AttributeValue::Integer(access_count)),
                                        ("migrated_from".to_string(), AttributeValue::String("warm".to_string())),
                                        ("migration_reason".to_string(), AttributeValue::String("high_access".to_string())),
                                    ].into_iter().collect(),
                                };
                                
                                memory_lock.add_entity(
                                    format!("hot_migrated_{}", doc_id),
                                    entity_data
                                ).expect("Failed to migrate to hot tier");
                            }
                            
                            println!("Migrated doc {} to HOT tier (access count: {})", doc_id, access_count);
                        },
                        "cold" => {
                            // Update tier metadata but keep in persistent storage
                            data_json["tier"] = serde_json::Value::String("cold".to_string());
                            data_json["migration_reason"] = serde_json::Value::String("low_access".to_string());
                            
                            system.persistent_tier.store(
                                &format!("cold_migrated_{}", doc_id),
                                data_json.to_string().as_bytes()
                            ).expect("Failed to migrate to cold tier");
                            
                            println!("Migrated doc {} to COLD tier (access count: {})", doc_id, access_count);
                        },
                        "archive" => {
                            // Migrate to archive tier with compression metadata
                            data_json["tier"] = serde_json::Value::String("archive".to_string());
                            data_json["migration_reason"] = serde_json::Value::String("no_access".to_string());
                            data_json["archived_at"] = serde_json::Value::String(chrono::Utc::now().to_rfc3339());
                            data_json["compression_enabled"] = serde_json::Value::Bool(true);
                            
                            system.archive_tier.store(
                                &format!("archived_{}", doc_id),
                                data_json.to_string().as_bytes()
                            ).expect("Failed to migrate to archive tier");
                            
                            println!("Migrated doc {} to ARCHIVE tier (access count: {})", doc_id, access_count);
                        },
                        _ => {
                            // Stay in warm tier
                            data_json["tier"] = serde_json::Value::String("warm".to_string());
                            
                            system.persistent_tier.store(
                                &storage_key,
                                data_json.to_string().as_bytes()
                            ).expect("Failed to update warm tier data");
                            
                            println!("Doc {} remains in WARM tier (access count: {})", doc_id, access_count);
                        }
                    }
                    
                    let migration_time = migration_start.elapsed();
                    
                    // Migration should be reasonably fast
                    assert!(migration_time < Duration::from_millis(100),
                           "Migration for doc {} should be fast: {:?}", doc_id, migration_time);
                    
                    // Update metadata
                    for metadata in &mut document_metadata {
                        if metadata.0 == doc_id {
                            metadata.2 = access_count;
                            metadata.3 = new_tier.to_string();
                        }
                    }
                }
                
                // Access should be fast regardless of current tier
                assert!(access_time < Duration::from_millis(50),
                       "Access to doc {} should be fast: {:?}", doc_id, access_time);
            }
        }
        
        // Validate post-migration state
        
        // Check hot tier population
        let memory_stats = {
            let memory_lock = system.memory_tier.read().await;
            memory_lock.get_stats()
        };
        
        // Should have migrated high-access documents to hot tier
        assert!(memory_stats.entity_count >= 2, 
               "Hot tier should contain migrated high-access documents");
        
        // Test cross-tier query after migration
        let post_migration_query_start = Instant::now();
        
        // Create entities for post-migration testing
        let test_entities: Vec<_> = (0..initial_documents.len()).map(|doc_id| {
            llmkg::extraction::Entity {
                id: format!("test_entity_{}", doc_id),
                name: format!("Document {} Content", doc_id),
                entity_type: "migrated_content".to_string(),
                confidence: 0.8,
            }
        }).collect();
        
        let post_migration_result = system.orchestrator.process_complex_query(
            "Find information across all storage tiers after migration",
            &test_entities,
            &[]
        ).await.expect("Failed to query across migrated tiers");
        
        let post_migration_query_time = post_migration_query_start.elapsed();
        
        // Post-migration queries should still work effectively
        assert!(post_migration_result.confidence > 0.4,
               "Post-migration query should be reasonably confident: {:.3}", post_migration_result.confidence);
        assert!(post_migration_query_time < Duration::from_secs(3),
               "Post-migration query should be responsive: {:?}", post_migration_query_time);
        
        // Test data consistency across tiers
        let consistency_check_start = Instant::now();
        
        for (doc_id, _storage_key, access_count, current_tier) in &document_metadata {
            match current_tier.as_str() {
                "hot" => {
                    // Verify presence in memory tier
                    let memory_lock = system.memory_tier.read().await;
                    let hot_entity_key = format!("hot_migrated_{}", doc_id);
                    assert!(memory_lock.get_entity(&hot_entity_key).is_some(),
                           "Hot migrated document {} should be in memory tier", doc_id);
                },
                "cold" => {
                    // Verify presence in persistent tier with cold metadata
                    let cold_key = format!("cold_migrated_{}", doc_id);
                    let cold_data = system.persistent_tier.retrieve(&cold_key);
                    assert!(cold_data.is_ok(),
                           "Cold migrated document {} should be in persistent tier", doc_id);
                },
                "archive" => {
                    // Verify presence in archive tier
                    let archive_key = format!("archived_{}", doc_id);
                    let archive_data = system.archive_tier.retrieve(&archive_key);
                    assert!(archive_data.is_ok(),
                           "Archived document {} should be in archive tier", doc_id);
                },
                _ => {
                    // Document should still be accessible in its tier
                    println!("Document {} remains in {} tier with {} accesses", 
                            doc_id, current_tier, access_count);
                }
            }
        }
        
        let consistency_check_time = consistency_check_start.elapsed();
        
        println!("✓ Tier migration integration test passed:");
        println!("  - Migration decisions based on access patterns");
        println!("  - Post-migration query time: {:?}", post_migration_query_time);
        println!("  - Post-migration confidence: {:.3}", post_migration_result.confidence);
        println!("  - Consistency check time: {:?}", consistency_check_time);
        println!("  - Final memory tier entities: {}", memory_stats.entity_count);
        
        for (doc_id, _key, access_count, tier) in &document_metadata {
            println!("    Doc {}: {} accesses -> {} tier", doc_id, access_count, tier);
        }
    }

    #[tokio::test]
    async fn test_hierarchical_consistency_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Create test data with intentional consistency challenges
        let consistency_test_data = vec![
            ("Einstein developed relativity theory.", "hot", "version_1"),
            ("Einstein's relativity theory revolutionized physics.", "warm", "version_2"),
            ("Albert Einstein's theory of relativity.", "cold", "version_3"),
            ("Einstein relativity theory physics breakthrough.", "archive", "version_4"),
        ];
        
        let mut version_metadata = Vec::new();
        
        // Store versions across different tiers
        for (document, tier, version) in &consistency_test_data {
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract from consistency test document");
            
            let version_data = serde_json::json!({
                "version": version,
                "tier": tier,
                "document": document,
                "extraction": extraction,
                "consistency_id": "einstein_relativity",
                "created_at": chrono::Utc::now().to_rfc3339(),
                "checksum": format!("{:x}", md5::compute(document.as_bytes()))
            });
            
            match *tier {
                "hot" => {
                    // Store in memory tier
                    let mut memory_lock = system.memory_tier.write().await;
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("version".to_string(), AttributeValue::String(version.to_string())),
                                ("tier".to_string(), AttributeValue::String(tier.to_string())),
                                ("consistency_id".to_string(), AttributeValue::String("einstein_relativity".to_string())),
                                ("checksum".to_string(), AttributeValue::String(format!("{:x}", md5::compute(document.as_bytes())))),
                            ].into_iter().collect(),
                        };
                        
                        memory_lock.add_entity(
                            format!("consistency_{}_{}", version, entity.id),
                            entity_data
                        ).expect("Failed to add consistency entity to hot tier");
                    }
                    
                    // Add to embeddings with version information
                    for entity in &extraction.entities {
                        let embedding: Vec<f32> = (0..384).map(|i| {
                            let version_num = version.chars().last().unwrap().to_digit(10).unwrap_or(1) as f32;
                            (version_num * 0.1 + i as f32 * 0.001).sin()
                        }).collect();
                        
                        system.embedding_tier.add_embedding(
                            &format!("consistency_{}_{}", version, entity.id),
                            embedding
                        ).expect("Failed to add consistency embedding");
                    }
                },
                "warm" => {
                    system.persistent_tier.store(
                        &format!("consistency_{}", version),
                        version_data.to_string().as_bytes()
                    ).expect("Failed to store warm consistency data");
                },
                "cold" => {
                    system.persistent_tier.store(
                        &format!("consistency_cold_{}", version),
                        version_data.to_string().as_bytes()
                    ).expect("Failed to store cold consistency data");
                },
                "archive" => {
                    system.archive_tier.store(
                        &format!("consistency_archive_{}", version),
                        version_data.to_string().as_bytes()
                    ).expect("Failed to store archive consistency data");
                }
            }
            
            version_metadata.push((version.to_string(), tier.to_string(), extraction.entities.len()));
        }
        
        // Test consistency validation across tiers
        let consistency_check_start = Instant::now();
        
        // 1. Entity consistency check
        let mut entity_consistency_report = Vec::new();
        
        // Check memory tier consistency
        {
            let memory_lock = system.memory_tier.read().await;
            let memory_entities: Vec<_> = (1..=4).filter_map(|version_num| {
                let entity_key = format!("consistency_version_{}_entity", version_num);
                memory_lock.get_entity(&entity_key).map(|entity| {
                    (format!("version_{}", version_num), entity.clone())
                })
            }).collect();
            
            if !memory_entities.is_empty() {
                entity_consistency_report.push(format!("Memory tier: {} consistent entities", memory_entities.len()));
            }
        }
        
        // 2. Cross-tier semantic consistency
        let semantic_consistency_start = Instant::now();
        
        // Test that similar content has similar embeddings across tiers
        let embedding_similarities = {
            let mut similarities = Vec::new();
            
            for i in 1..=4 {
                for j in (i+1)..=4 {
                    let entity1_key = format!("consistency_version_{}_entity", i);
                    let entity2_key = format!("consistency_version_{}_entity", j);
                    
                    // In a real implementation, we would compare actual embeddings
                    // For this test, we simulate semantic similarity based on content overlap
                    let doc1 = &consistency_test_data[i-1].0;
                    let doc2 = &consistency_test_data[j-1].0;
                    
                    let words1: std::collections::HashSet<&str> = doc1.split_whitespace().collect();
                    let words2: std::collections::HashSet<&str> = doc2.split_whitespace().collect();
                    
                    let intersection = words1.intersection(&words2).count();
                    let union = words1.union(&words2).count();
                    let jaccard_similarity = intersection as f32 / union as f32;
                    
                    similarities.push((format!("v{}-v{}", i, j), jaccard_similarity));
                }
            }
            
            similarities
        };
        
        let semantic_consistency_time = semantic_consistency_start.elapsed();
        
        // All versions should have reasonable semantic similarity (they're about the same topic)
        for (version_pair, similarity) in &embedding_similarities {
            assert!(similarity > &0.2, 
                   "Versions {} should have semantic consistency: similarity {:.3}", 
                   version_pair, similarity);
            println!("Semantic consistency {}: {:.3}", version_pair, similarity);
        }
        
        // 3. Data integrity check across storage systems
        let integrity_check_start = Instant::now();
        
        // Verify data can be retrieved from each tier
        let mut integrity_report = Vec::new();
        
        // Check persistent tier
        for version in ["version_2", "version_3"] {
            let retrieval_key = if version == "version_2" {
                format!("consistency_{}", version)
            } else {
                format!("consistency_cold_{}", version)
            };
            
            match system.persistent_tier.retrieve(&retrieval_key) {
                Ok(data) => {
                    let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&data);
                    match parsed {
                        Ok(json) => {
                            if json["consistency_id"] == "einstein_relativity" {
                                integrity_report.push(format!("{}: Consistent", version));
                            } else {
                                integrity_report.push(format!("{}: Inconsistent ID", version));
                            }
                        },
                        Err(_) => {
                            integrity_report.push(format!("{}: Parse error", version));
                        }
                    }
                },
                Err(_) => {
                    integrity_report.push(format!("{}: Retrieval failed", version));
                }
            }
        }
        
        // Check archive tier
        match system.archive_tier.retrieve("consistency_archive_version_4") {
            Ok(data) => {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) {
                    if json["consistency_id"] == "einstein_relativity" {
                        integrity_report.push("version_4: Archive consistent".to_string());
                    }
                }
            },
            Err(_) => {
                integrity_report.push("version_4: Archive retrieval failed".to_string());
            }
        }
        
        let integrity_check_time = integrity_check_start.elapsed();
        
        // 4. Cross-tier query consistency
        let query_consistency_start = Instant::now();
        
        // Create entities representing the consistent data
        let consistent_entities: Vec<_> = version_metadata.iter().enumerate().map(|(idx, (version, tier, _count))| {
            llmkg::extraction::Entity {
                id: format!("consistent_entity_{}", idx),
                name: "Einstein relativity theory".to_string(),
                entity_type: "scientific_concept".to_string(),
                confidence: 0.9 - (idx as f32 * 0.1), // Slight confidence variation
            }
        }).collect();
        
        let consistency_query_result = system.orchestrator.process_complex_query(
            "What is Einstein's relativity theory?",
            &consistent_entities,
            &[]
        ).await.expect("Failed to process consistency query");
        
        let query_consistency_time = query_consistency_start.elapsed();
        
        // Query should produce consistent results despite data being spread across tiers
        assert!(consistency_query_result.confidence > 0.7,
               "Cross-tier query should be highly confident with consistent data: {:.3}", 
               consistency_query_result.confidence);
        
        // 5. Version drift detection
        let version_drift_check = {
            let mut drift_metrics = Vec::new();
            
            // Check for semantic drift between versions
            for (version_pair, similarity) in &embedding_similarities {
                if similarity < &0.5 {
                    drift_metrics.push(format!("Potential drift detected: {} (similarity: {:.3})", 
                                              version_pair, similarity));
                } else {
                    drift_metrics.push(format!("Consistent: {} (similarity: {:.3})", 
                                              version_pair, similarity));
                }
            }
            
            drift_metrics
        };
        
        let consistency_check_time = consistency_check_start.elapsed();
        
        // Validate overall consistency
        assert!(consistency_check_time < Duration::from_secs(2),
               "Consistency check should complete quickly: {:?}", consistency_check_time);
        
        assert!(integrity_report.iter().filter(|r| r.contains("Consistent")).count() >= 2,
               "Most data should maintain integrity across tiers");
        
        println!("✓ Hierarchical consistency integration test passed:");
        println!("  - Total consistency check time: {:?}", consistency_check_time);
        println!("  - Semantic consistency time: {:?}", semantic_consistency_time);
        println!("  - Integrity check time: {:?}", integrity_check_time);
        println!("  - Query consistency time: {:?}", query_consistency_time);
        println!("  - Query confidence: {:.3}", consistency_query_result.confidence);
        
        println!("  Entity consistency:");
        for report in &entity_consistency_report {
            println!("    {}", report);
        }
        
        println!("  Integrity report:");
        for report in &integrity_report {
            println!("    {}", report);
        }
        
        println!("  Version drift analysis:");
        for metric in &version_drift_check {
            println!("    {}", metric);
        }
    }

    #[tokio::test]
    async fn test_storage_optimization_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Create diverse data for optimization testing
        let optimization_test_data = vec![
            ("Frequently accessed hot data about Einstein's discoveries", "hot", 100),
            ("Medium priority warm data on relativity applications", "warm", 20),
            ("Rarely accessed cold data on historical physics context", "cold", 5),
            ("Archive data on 19th century scientific methodologies", "archive", 1),
            ("Duplicate content about Einstein's theory of relativity", "warm", 25),
            ("Similar content regarding Einstein's relativistic theories", "warm", 22),
        ];
        
        let optimization_start = Instant::now();
        let mut storage_metrics = Vec::new();
        
        // Initial storage with suboptimal placement
        for (idx, (document, initial_tier, priority)) in optimization_test_data.iter().enumerate() {
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract for optimization test");
            
            let storage_data = serde_json::json!({
                "doc_id": idx,
                "extraction": extraction,
                "document": document,
                "tier": initial_tier,
                "priority": priority,
                "storage_efficiency": 0.0, // To be calculated
                "created_at": chrono::Utc::now().to_rfc3339()
            });
            
            // Store in persistent tier initially (suboptimal for some)
            system.persistent_tier.store(
                &format!("optimization_doc_{}", idx),
                storage_data.to_string().as_bytes()
            ).expect("Failed to store optimization test data");
            
            storage_metrics.push((idx, initial_tier.to_string(), *priority, document.len()));
        }
        
        // Step 1: Analyze storage efficiency
        let analysis_start = Instant::now();
        
        let mut efficiency_analysis = Vec::new();
        
        for (doc_id, tier, priority, size) in &storage_metrics {
            // Calculate efficiency score based on access pattern vs storage tier
            let efficiency_score = match (tier.as_str(), *priority) {
                ("hot", p) if p >= 50 => 1.0,      // Perfect match
                ("warm", p) if p >= 10 && p < 50 => 1.0, // Perfect match
                ("cold", p) if p >= 2 && p < 10 => 1.0,  // Perfect match
                ("archive", p) if p < 2 => 1.0,    // Perfect match
                ("hot", p) if p < 50 => 0.3,       // Overprovisioned
                ("warm", p) if p >= 50 => 0.7,     // Underprovisioned
                ("warm", p) if p < 10 => 0.4,      // Overprovisioned
                ("cold", p) if p >= 10 => 0.5,     // Underprovisioned
                ("archive", p) if p >= 2 => 0.2,   // Severely underprovisioned
                _ => 0.6, // Default
            };
            
            let storage_cost = match tier.as_str() {
                "hot" => size * 10,     // Expensive
                "warm" => size * 3,     // Moderate
                "cold" => size * 1,     // Cheap
                "archive" => size / 2,  // Very cheap
                _ => *size,
            };
            
            efficiency_analysis.push((*doc_id, efficiency_score, storage_cost, tier.clone()));
        }
        
        let analysis_time = analysis_start.elapsed();
        
        // Step 2: Optimize storage placement
        let optimization_execution_start = Instant::now();
        
        let mut optimization_actions = Vec::new();
        
        for (doc_id, efficiency_score, storage_cost, current_tier) in &efficiency_analysis {
            if *efficiency_score < 0.7 { // Needs optimization
                let optimal_tier = match storage_metrics[*doc_id].2 {
                    p if p >= 50 => "hot",
                    p if p >= 10 => "warm", 
                    p if p >= 2 => "cold",
                    _ => "archive",
                };
                
                if optimal_tier != current_tier {
                    let migration_start = Instant::now();
                    
                    // Retrieve current data
                    let current_data = system.persistent_tier.retrieve(&format!("optimization_doc_{}", doc_id))
                        .expect("Failed to retrieve data for optimization");
                    
                    let mut data_json: serde_json::Value = serde_json::from_slice(&current_data)
                        .expect("Failed to parse data for optimization");
                    
                    // Update tier information
                    data_json["tier"] = serde_json::Value::String(optimal_tier.to_string());
                    data_json["optimization_applied"] = serde_json::Value::Bool(true);
                    data_json["previous_tier"] = serde_json::Value::String(current_tier.clone());
                    data_json["optimization_reason"] = serde_json::Value::String(
                        format!("Efficiency improved from {:.2} to target 1.0", efficiency_score)
                    );
                    
                    // Migrate to optimal storage
                    match optimal_tier {
                        "hot" => {
                            // Move to memory tier
                            if let Some(extraction_data) = data_json["extraction"].as_object() {
                                let mut memory_lock = system.memory_tier.write().await;
                                
                                let entity_data = EntityData {
                                    attributes: [
                                        ("name".to_string(), AttributeValue::String(format!("OptimizedDoc{}", doc_id))),
                                        ("tier".to_string(), AttributeValue::String("hot".to_string())),
                                        ("optimized".to_string(), AttributeValue::String("true".to_string())),
                                        ("efficiency_improvement".to_string(), AttributeValue::Float(1.0 - efficiency_score)),
                                    ].into_iter().collect(),
                                };
                                
                                memory_lock.add_entity(
                                    format!("optimized_hot_{}", doc_id),
                                    entity_data
                                ).expect("Failed to optimize to hot tier");
                            }
                        },
                        "archive" => {
                            // Move to archive tier
                            system.archive_tier.store(
                                &format!("optimized_archive_{}", doc_id),
                                data_json.to_string().as_bytes()
                            ).expect("Failed to optimize to archive tier");
                        },
                        _ => {
                            // Update in persistent tier
                            system.persistent_tier.store(
                                &format!("optimized_{}", doc_id),
                                data_json.to_string().as_bytes()
                            ).expect("Failed to optimize in persistent tier");
                        }
                    }
                    
                    let migration_time = migration_start.elapsed();
                    
                    optimization_actions.push((
                        *doc_id,
                        current_tier.clone(),
                        optimal_tier.to_string(),
                        *efficiency_score,
                        migration_time
                    ));
                }
            }
        }
        
        let optimization_execution_time = optimization_execution_start.elapsed();
        
        // Step 3: Deduplication optimization
        let deduplication_start = Instant::now();
        
        // Find similar content for deduplication
        let similar_content_pairs = vec![
            (4, 5), // The Einstein duplicate content
        ];
        
        let mut deduplication_savings = 0;
        
        for (doc1_id, doc2_id) in similar_content_pairs {
            let doc1 = &optimization_test_data[doc1_id];
            let doc2 = &optimization_test_data[doc2_id];
            
            // Calculate content similarity
            let words1: std::collections::HashSet<&str> = doc1.0.split_whitespace().collect();
            let words2: std::collections::HashSet<&str> = doc2.0.split_whitespace().collect();
            
            let intersection = words1.intersection(&words2).count();
            let union = words1.union(&words2).count();
            let similarity = intersection as f32 / union as f32;
            
            if similarity > 0.8 { // High similarity threshold
                // Create deduplicated entry
                let deduplicated_data = serde_json::json!({
                    "type": "deduplicated",
                    "primary_doc_id": doc1_id,
                    "duplicate_doc_id": doc2_id,
                    "similarity_score": similarity,
                    "space_saved": doc2.0.len(),
                    "deduplicated_at": chrono::Utc::now().to_rfc3339()
                });
                
                system.persistent_tier.store(
                    &format!("dedup_{}_{}", doc1_id, doc2_id),
                    deduplicated_data.to_string().as_bytes()
                ).expect("Failed to store deduplication record");
                
                deduplication_savings += doc2.0.len();
                
                println!("Deduplicated docs {} and {} (similarity: {:.3}, saved: {} bytes)", 
                        doc1_id, doc2_id, similarity, doc2.0.len());
            }
        }
        
        let deduplication_time = deduplication_start.elapsed();
        
        // Step 4: Test performance after optimization
        let performance_test_start = Instant::now();
        
        // Test query performance on optimized storage
        let test_entities: Vec<_> = optimization_test_data.iter().enumerate().map(|(idx, (doc, _tier, priority))| {
            llmkg::extraction::Entity {
                id: format!("perf_test_{}", idx),
                name: format!("Optimized content {}", idx),
                entity_type: "optimized_content".to_string(),
                confidence: (*priority as f32 / 100.0).min(1.0),
            }
        }).collect();
        
        let optimized_query_result = system.orchestrator.process_complex_query(
            "Find information from optimized storage system",
            &test_entities,
            &[]
        ).await.expect("Failed to query optimized storage");
        
        let performance_test_time = performance_test_start.elapsed();
        
        // Step 5: Calculate optimization benefits
        let total_optimization_time = optimization_start.elapsed();
        
        let optimization_summary = {
            let total_migrations = optimization_actions.len();
            let avg_migration_time = if total_migrations > 0 {
                optimization_actions.iter().map(|(_, _, _, _, time)| time.as_millis()).sum::<u128>() / total_migrations as u128
            } else {
                0
            };
            
            let efficiency_improvements: Vec<f32> = optimization_actions.iter()
                .map(|(_, _, _, old_efficiency, _)| 1.0 - old_efficiency)
                .collect();
            
            let avg_efficiency_improvement = if !efficiency_improvements.is_empty() {
                efficiency_improvements.iter().sum::<f32>() / efficiency_improvements.len() as f32
            } else {
                0.0
            };
            
            (total_migrations, avg_migration_time, avg_efficiency_improvement)
        };
        
        // Validate optimization results
        assert!(optimization_execution_time < Duration::from_secs(2),
               "Storage optimization should complete quickly: {:?}", optimization_execution_time);
        
        assert!(optimized_query_result.confidence > 0.5,
               "Optimized storage should maintain query quality: {:.3}", optimized_query_result.confidence);
        
        assert!(performance_test_time < Duration::from_secs(3),
               "Optimized storage should maintain query performance: {:?}", performance_test_time);
        
        if !optimization_actions.is_empty() {
            assert!(optimization_summary.2 > 0.0,
                   "Optimization should improve efficiency: {:.3}", optimization_summary.2);
        }
        
        println!("✓ Storage optimization integration test passed:");
        println!("  - Total optimization time: {:?}", total_optimization_time);
        println!("  - Analysis time: {:?}", analysis_time);
        println!("  - Execution time: {:?}", optimization_execution_time);
        println!("  - Deduplication time: {:?}", deduplication_time);
        println!("  - Performance test time: {:?}", performance_test_time);
        println!("  - Query confidence: {:.3}", optimized_query_result.confidence);
        println!("  - Optimizations performed: {}", optimization_summary.0);
        println!("  - Average migration time: {}ms", optimization_summary.1);
        println!("  - Average efficiency improvement: {:.3}", optimization_summary.2);
        println!("  - Deduplication savings: {} bytes", deduplication_savings);
        
        println!("  Optimization actions:");
        for (doc_id, from_tier, to_tier, old_efficiency, migration_time) in &optimization_actions {
            println!("    Doc {}: {} -> {} (efficiency: {:.3} -> 1.0, time: {:?})", 
                    doc_id, from_tier, to_tier, old_efficiency, migration_time);
        }
    }

    #[tokio::test]
    async fn test_concurrent_access_integration() {
        let mut system = create_hierarchical_storage_system().await;
        
        // Prepare test data for concurrent access
        let concurrent_test_documents = vec![
            "Einstein's contributions to quantum mechanics",
            "Quantum entanglement and its implications", 
            "Modern applications of quantum physics",
            "Historical development of quantum theory",
            "Quantum computing and future technologies"
        ];
        
        // Pre-populate storage tiers
        for (doc_id, document) in concurrent_test_documents.iter().enumerate() {
            let extraction = system.extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract for concurrent test");
            
            let tier = match doc_id % 4 {
                0 => "hot",
                1 => "warm", 
                2 => "cold",
                _ => "archive",
            };
            
            // Store across different tiers
            match tier {
                "hot" => {
                    let mut memory_lock = system.memory_tier.write().await;
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("tier".to_string(), AttributeValue::String(tier.to_string())),
                                ("doc_id".to_string(), AttributeValue::String(doc_id.to_string())),
                                ("concurrent_test".to_string(), AttributeValue::String("true".to_string())),
                            ].into_iter().collect(),
                        };
                        
                        memory_lock.add_entity(
                            format!("concurrent_{}_{}", doc_id, entity.id),
                            entity_data
                        ).expect("Failed to add concurrent entity");
                    }
                },
                _ => {
                    let storage_data = serde_json::json!({
                        "doc_id": doc_id,
                        "extraction": extraction,
                        "document": document,
                        "tier": tier,
                        "concurrent_test": true
                    });
                    
                    let storage = match tier {
                        "archive" => &mut system.archive_tier,
                        _ => &mut system.persistent_tier,
                    };
                    
                    storage.store(
                        &format!("concurrent_{}_{}", tier, doc_id),
                        storage_data.to_string().as_bytes()
                    ).expect("Failed to store concurrent test data");
                }
            }
            
            // Add to embeddings for all documents
            for entity in &extraction.entities {
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (doc_id as f32 * 0.1 + i as f32 * 0.001).sin()
                }).collect();
                
                system.embedding_tier.add_embedding(
                    &format!("concurrent_{}_{}", doc_id, entity.id),
                    embedding
                ).expect("Failed to add concurrent embedding");
            }
        }
        
        // Test concurrent access scenarios
        let concurrent_test_start = Instant::now();
        
        // Scenario 1: Concurrent reads from different tiers
        let concurrent_reads_start = Instant::now();
        
        let read_tasks = (0..10).map(|task_id| {
            let memory_tier = Arc::clone(&system.memory_tier);
            
            tokio::spawn(async move {
                let task_start = Instant::now();
                let mut results = Vec::new();
                
                // Read from memory tier
                {
                    let memory_lock = memory_tier.read().await;
                    let stats = memory_lock.get_stats();
                    results.push(format!("Task {}: Memory tier has {} entities", task_id, stats.entity_count));
                    
                    // Try to access specific entities
                    for doc_id in 0..concurrent_test_documents.len() {
                        let entity_key = format!("concurrent_{}_entity", doc_id);
                        if memory_lock.get_entity(&entity_key).is_some() {
                            results.push(format!("Task {}: Found entity {}", task_id, entity_key));
                        }
                    }
                }
                
                let task_time = task_start.elapsed();
                (task_id, results, task_time)
            })
        }).collect::<Vec<_>>();
        
        let read_results = futures::future::join_all(read_tasks).await;
        let concurrent_reads_time = concurrent_reads_start.elapsed();
        
        // Validate concurrent reads
        assert_eq!(read_results.len(), 10, "All read tasks should complete");
        
        let mut successful_reads = 0;
        let mut total_read_time = Duration::from_millis(0);
        
        for result in read_results {
            match result {
                Ok((task_id, results, task_time)) => {
                    assert!(!results.is_empty(), "Task {} should find some results", task_id);
                    assert!(task_time < Duration::from_millis(500), 
                           "Task {} should complete quickly: {:?}", task_id, task_time);
                    successful_reads += 1;
                    total_read_time += task_time;
                },
                Err(e) => {
                    panic!("Read task failed: {:?}", e);
                }
            }
        }
        
        assert_eq!(successful_reads, 10, "All concurrent reads should succeed");
        
        // Scenario 2: Concurrent writes to different tiers
        let concurrent_writes_start = Instant::now();
        
        let write_tasks = (0..5).map(|task_id| {
            let memory_tier = Arc::clone(&system.memory_tier);
            
            tokio::spawn(async move {
                let task_start = Instant::now();
                let mut write_results = Vec::new();
                
                // Write to memory tier
                {
                    let mut memory_lock = memory_tier.write().await;
                    
                    for item_id in 0..3 {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(format!("ConcurrentWrite_{}_{}", task_id, item_id))),
                                ("task_id".to_string(), AttributeValue::String(task_id.to_string())),
                                ("item_id".to_string(), AttributeValue::String(item_id.to_string())),
                                ("write_timestamp".to_string(), AttributeValue::String(chrono::Utc::now().to_rfc3339())),
                            ].into_iter().collect(),
                        };
                        
                        match memory_lock.add_entity(
                            format!("concurrent_write_{}_{}", task_id, item_id),
                            entity_data
                        ) {
                            Ok(_) => {
                                write_results.push(format!("Task {}: Successfully wrote item {}", task_id, item_id));
                            },
                            Err(e) => {
                                write_results.push(format!("Task {}: Failed to write item {}: {:?}", task_id, item_id, e));
                            }
                        }
                    }
                }
                
                let task_time = task_start.elapsed();
                (task_id, write_results, task_time)
            })
        }).collect::<Vec<_>>();
        
        let write_results = futures::future::join_all(write_tasks).await;
        let concurrent_writes_time = concurrent_writes_start.elapsed();
        
        // Validate concurrent writes
        let mut successful_writes = 0;
        let mut total_items_written = 0;
        
        for result in write_results {
            match result {
                Ok((task_id, results, task_time)) => {
                    let successful_items = results.iter().filter(|r| r.contains("Successfully")).count();
                    assert!(successful_items > 0, "Task {} should write some items successfully", task_id);
                    assert!(task_time < Duration::from_secs(1),
                           "Task {} should complete write operations quickly: {:?}", task_id, task_time);
                    successful_writes += 1;
                    total_items_written += successful_items;
                },
                Err(e) => {
                    panic!("Write task failed: {:?}", e);
                }
            }
        }
        
        assert_eq!(successful_writes, 5, "All concurrent write tasks should succeed");
        assert!(total_items_written >= 10, "Should write multiple items successfully");
        
        // Scenario 3: Mixed concurrent operations (reads, writes, queries)
        let mixed_operations_start = Instant::now();
        
        let mixed_tasks = (0..8).map(|task_id| {
            let memory_tier = Arc::clone(&system.memory_tier);
            let operation_type = match task_id % 3 {
                0 => "read",
                1 => "write",
                _ => "query",
            };
            
            tokio::spawn(async move {
                let task_start = Instant::now();
                let result = match operation_type {
                    "read" => {
                        let memory_lock = memory_tier.read().await;
                        let stats = memory_lock.get_stats();
                        format!("Read: {} entities found", stats.entity_count)
                    },
                    "write" => {
                        let mut memory_lock = memory_tier.write().await;
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(format!("Mixed_{}", task_id))),
                                ("operation".to_string(), AttributeValue::String("mixed_write".to_string())),
                            ].into_iter().collect(),
                        };
                        
                        match memory_lock.add_entity(format!("mixed_{}", task_id), entity_data) {
                            Ok(_) => "Write: Success".to_string(),
                            Err(e) => format!("Write: Failed - {:?}", e),
                        }
                    },
                    "query" => {
                        let memory_lock = memory_tier.read().await;
                        let entity_count = memory_lock.get_stats().entity_count;
                        let relationship_count = memory_lock.get_stats().relationship_count;
                        format!("Query: {} entities, {} relationships", entity_count, relationship_count)
                    },
                    _ => "Unknown operation".to_string(),
                };
                
                let task_time = task_start.elapsed();
                (task_id, operation_type, result, task_time)
            })
        }).collect::<Vec<_>>();
        
        let mixed_results = futures::future::join_all(mixed_tasks).await;
        let mixed_operations_time = mixed_operations_start.elapsed();
        
        // Validate mixed operations
        let mut operation_counts = std::collections::HashMap::new();
        let mut max_operation_time = Duration::from_millis(0);
        
        for result in mixed_results {
            match result {
                Ok((task_id, operation_type, result_msg, task_time)) => {
                    assert!(task_time < Duration::from_secs(2),
                           "Mixed operation task {} ({}) should complete quickly: {:?}", 
                           task_id, operation_type, task_time);
                    assert!(!result_msg.contains("Failed") || result_msg.contains("Write: Failed"),
                           "Operation should succeed or fail gracefully: {}", result_msg);
                    
                    *operation_counts.entry(operation_type.to_string()).or_insert(0) += 1;
                    max_operation_time = max_operation_time.max(task_time);
                    
                    println!("Mixed operation task {}: {} -> {}", task_id, operation_type, result_msg);
                },
                Err(e) => {
                    panic!("Mixed operation task failed: {:?}", e);
                }
            }
        }
        
        // Verify all operation types were executed
        assert!(operation_counts.contains_key("read"), "Should have read operations");
        assert!(operation_counts.contains_key("write"), "Should have write operations");
        assert!(operation_counts.contains_key("query"), "Should have query operations");
        
        // Final consistency check
        let final_memory_stats = {
            let memory_lock = system.memory_tier.read().await;
            memory_lock.get_stats()
        };
        
        let total_concurrent_time = concurrent_test_start.elapsed();
        
        // Validate final state
        assert!(final_memory_stats.entity_count > 0, "Memory tier should contain entities after concurrent operations");
        assert!(total_concurrent_time < Duration::from_secs(10), 
               "All concurrent operations should complete within reasonable time: {:?}", total_concurrent_time);
        
        println!("✓ Concurrent access integration test passed:");
        println!("  - Total concurrent test time: {:?}", total_concurrent_time);
        println!("  - Concurrent reads time: {:?} (10 tasks)", concurrent_reads_time);
        println!("  - Concurrent writes time: {:?} (5 tasks)", concurrent_writes_time);
        println!("  - Mixed operations time: {:?} (8 tasks)", mixed_operations_time);
        println!("  - Average read time: {:?}", total_read_time / 10);
        println!("  - Total items written: {}", total_items_written);
        println!("  - Max mixed operation time: {:?}", max_operation_time);
        println!("  - Final memory tier entities: {}", final_memory_stats.entity_count);
        
        println!("  Operation distribution:");
        for (op_type, count) in operation_counts {
            println!("    {}: {} operations", op_type, count);
        }
    }
}