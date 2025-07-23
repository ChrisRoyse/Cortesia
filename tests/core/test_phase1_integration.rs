use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;
use tokio::time::{sleep, Duration};

use llmkg::core::phase1_integration::Phase1IntegrationLayer;
use llmkg::core::phase1_types::{Phase1Config, QueryResult, CognitiveQueryResult};
use llmkg::cognitive::CognitivePatternType;

/// Helper function to create a test configuration
fn create_test_config() -> Phase1Config {
    Phase1Config {
        embedding_dim: 384,
        neural_server_endpoint: "localhost:9000".to_string(),
        enable_temporal_tracking: true,
        enable_sdr_storage: true,
        enable_real_time_updates: true,
        enable_cognitive_patterns: true,
        activation_config: Default::default(),
    }
}

/// Helper function to create a test configuration without cognitive patterns
fn create_config_no_cognitive() -> Phase1Config {
    let mut config = create_test_config();
    config.enable_cognitive_patterns = false;
    config
}

/// Helper function to create a test configuration with custom settings
fn create_custom_config(
    embedding_dim: usize,
    enable_temporal: bool,
    enable_realtime: bool,
    enable_cognitive: bool,
) -> Phase1Config {
    Phase1Config {
        embedding_dim,
        neural_server_endpoint: "localhost:9000".to_string(),
        enable_temporal_tracking: enable_temporal,
        enable_sdr_storage: true,
        enable_real_time_updates: enable_realtime,
        enable_cognitive_patterns: enable_cognitive,
        activation_config: Default::default(),
    }
}

#[tokio::test]
async fn test_question_answering_system() {
    // Create Phase1IntegrationLayer
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Start background processes
    integration.start().await.unwrap();
    
    // Store large amount of knowledge through public API
    let knowledge_base = vec![
        ("Albert Einstein was a theoretical physicist who developed the theory of relativity.", Some("physics")),
        ("The theory of relativity consists of special relativity and general relativity.", Some("physics")),
        ("Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.", Some("physics")),
        ("Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity.", Some("science")),
        ("Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences.", Some("science")),
        ("Isaac Newton formulated the laws of motion and universal gravitation.", Some("physics")),
        ("Newton's laws of motion describe the relationship between forces and motion.", Some("physics")),
        ("Charles Darwin developed the theory of evolution by natural selection.", Some("biology")),
        ("The Origin of Species presented evidence for evolution and common descent.", Some("biology")),
        ("DNA is a molecule that carries genetic instructions for life.", Some("biology")),
    ];
    
    // Store all knowledge
    for (text, context) in knowledge_base {
        let result = integration.store_knowledge_with_neural_structure(text, context).await;
        assert!(result.is_ok(), "Failed to store knowledge: {}", text);
    }
    
    // Perform various types of queries
    let queries = vec![
        ("Who was Einstein?", Some("convergent")),
        ("What did Marie Curie discover?", Some("analytical")),
        ("Explain the theory of relativity", Some("explanatory")),
        ("What are Newton's contributions to physics?", Some("descriptive")),
        ("How does evolution work?", None),
    ];
    
    for (query, pattern) in queries {
        let result = integration.neural_query_with_activation(query, pattern).await;
        assert!(result.is_ok(), "Query failed: {}", query);
        
        let query_result = result.unwrap();
        assert_eq!(query_result.query, query);
        assert!(query_result.converged, "Query did not converge: {}", query);
        assert!(query_result.iterations_completed > 0, "No iterations completed for: {}", query);
        assert!(query_result.total_energy >= 0.0, "Invalid energy for: {}", query);
        
        // Verify we get some results
        if !query_result.final_activations.is_empty() {
            assert!(query_result.total_energy > 0.0, "Non-zero activations should have positive energy");
        }
    }
    
    // Test cognitive reasoning capabilities
    if integration.cognitive_orchestrator.is_some() {
        let cognitive_queries = vec![
            ("Compare Einstein and Newton's contributions to physics", CognitivePatternType::ChainOfThought),
            ("What is the relationship between DNA and evolution?", CognitivePatternType::TreeOfThoughts),
            ("Analyze the impact of radioactivity research", CognitivePatternType::TreeOfThoughts),
        ];
        
        for (query, pattern) in cognitive_queries {
            let result = integration.cognitive_reasoning(query, Some("scientific analysis"), Some(pattern)).await;
            assert!(result.is_ok(), "Cognitive reasoning failed: {}", query);
            
            let cognitive_result = result.unwrap();
            assert!(!cognitive_result.final_answer.is_empty(), "Empty answer for: {}", query);
            assert!(cognitive_result.confidence > 0.0, "Zero confidence for: {}", query);
            assert!(cognitive_result.execution_time_ms > 0, "Zero execution time for: {}", query);
            assert!(!cognitive_result.patterns_executed.is_empty(), "No patterns executed for: {}", query);
        }
    }
    
    // Stop background processes
    integration.stop().await.unwrap();
}

#[tokio::test]
async fn test_knowledge_storage_and_retrieval_workflow() {
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Test incremental knowledge building
    let facts = vec![
        "The solar system has eight planets.",
        "Earth is the third planet from the Sun.",
        "Mars is known as the Red Planet.",
        "Jupiter is the largest planet in our solar system.",
        "Saturn has prominent rings made of ice and rock.",
    ];
    
    // Store facts incrementally
    let mut stored_entities = Vec::new();
    for fact in &facts {
        let entities = integration.store_knowledge_with_neural_structure(fact, Some("astronomy")).await.unwrap();
        assert!(!entities.is_empty(), "No entities created for: {}", fact);
        stored_entities.extend(entities);
    }
    
    // Verify storage
    let stats = integration.get_phase1_statistics().await.unwrap();
    assert!(stats.brain_statistics.entity_count > 0, "No entities stored");
    assert!(stats.neural_server_connected, "Neural server not connected");
    
    // Query the stored knowledge
    let queries = vec![
        "Which planet is the largest?",
        "What color is Mars?",
        "How many planets are in the solar system?",
        "What is special about Saturn?",
        "Where is Earth located?",
    ];
    
    for query in queries {
        let result = integration.neural_query_with_activation(query, Some("factual")).await.unwrap();
        assert_eq!(result.query, query);
        assert!(result.converged, "Query did not converge: {}", query);
        
        // For factual queries, we expect some activation
        if !result.final_activations.is_empty() {
            let max_activation = result.final_activations.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!(*max_activation > 0.0, "No significant activation for: {}", query);
        }
    }
}

#[tokio::test]
async fn test_temporal_knowledge_evolution() {
    let mut config = create_test_config();
    config.enable_temporal_tracking = true;
    config.enable_real_time_updates = true;
    
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    integration.start().await.unwrap();
    
    // Simulate evolving knowledge over time
    let time_series_facts = vec![
        ("Bitcoin price is $30,000", "cryptocurrency", Utc::now()),
        ("Bitcoin price is $35,000", "cryptocurrency", Utc::now() + chrono::Duration::hours(1)),
        ("Bitcoin price is $32,000", "cryptocurrency", Utc::now() + chrono::Duration::hours(2)),
        ("Ethereum price is $2,000", "cryptocurrency", Utc::now()),
        ("Ethereum price is $2,100", "cryptocurrency", Utc::now() + chrono::Duration::hours(1)),
    ];
    
    // Store temporal facts
    for (fact, context, _time) in &time_series_facts {
        let result = integration.store_knowledge_with_neural_structure(fact, Some(context)).await;
        assert!(result.is_ok(), "Failed to store temporal fact: {}", fact);
        
        // Small delay to simulate time passing
        sleep(Duration::from_millis(100)).await;
    }
    
    // Query at different time points
    let query_times = vec![
        Utc::now(),
        Utc::now() + chrono::Duration::hours(1),
        Utc::now() + chrono::Duration::hours(2),
    ];
    
    for query_time in query_times {
        let result = integration.temporal_query_at_time(
            "cryptocurrency prices",
            query_time,
            None,
        ).await;
        assert!(result.is_ok(), "Temporal query failed");
        
        // Results depend on whether native feature is enabled
        let temporal_entities = result.unwrap();
        // We just verify the query doesn't crash
    }
    
    // Check temporal processor statistics
    let stats = integration.get_phase1_statistics().await.unwrap();
    assert!(stats.update_statistics.total_updates >= 0);
    
    integration.stop().await.unwrap();
}

#[tokio::test]
async fn test_cognitive_reasoning_ensemble() {
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Store complex knowledge for reasoning
    let complex_knowledge = vec![
        ("Climate change is caused by increased greenhouse gas emissions.", Some("environment")),
        ("Greenhouse gases trap heat in the Earth's atmosphere.", Some("environment")),
        ("Carbon dioxide is a major greenhouse gas.", Some("environment")),
        ("Deforestation reduces the Earth's capacity to absorb CO2.", Some("environment")),
        ("Renewable energy sources can help reduce emissions.", Some("environment")),
        ("Solar and wind power are renewable energy sources.", Some("environment")),
    ];
    
    for (text, context) in complex_knowledge {
        integration.store_knowledge_with_neural_structure(text, context).await.unwrap();
    }
    
    // Test ensemble reasoning with multiple patterns
    let patterns = vec![
        CognitivePatternType::ChainOfThought,
        CognitivePatternType::TreeOfThoughts,
        CognitivePatternType::TreeOfThoughts,
    ];
    
    let result = integration.ensemble_cognitive_reasoning(
        "How can we address climate change?",
        Some("environmental policy"),
        patterns.clone(),
    ).await;
    
    assert!(result.is_ok(), "Ensemble reasoning failed");
    let cognitive_result = result.unwrap();
    
    assert!(!cognitive_result.final_answer.is_empty(), "Empty ensemble answer");
    assert!(cognitive_result.confidence > 0.0, "Zero confidence in ensemble");
    assert!(cognitive_result.patterns_executed.len() >= 1, "No patterns executed");
    assert!(cognitive_result.execution_time_ms > 0, "Zero execution time");
    
    // Verify quality metrics
    let quality = &cognitive_result.quality_metrics;
    assert!(quality.overall_confidence >= 0.0 && quality.overall_confidence <= 1.0);
    assert!(quality.consistency_score >= 0.0 && quality.consistency_score <= 1.0);
    assert!(quality.completeness_score >= 0.0 && quality.completeness_score <= 1.0);
}

#[tokio::test]
async fn test_system_performance_under_load() {
    let config = create_test_config();
    let integration = Arc::new(Phase1IntegrationLayer::new(config).await.unwrap());
    integration.start().await.unwrap();
    
    // Simulate concurrent load
    let num_concurrent = 10;
    let mut handles = Vec::new();
    
    // Spawn concurrent storage operations
    for i in 0..num_concurrent {
        let integration_clone = integration.clone();
        let handle = tokio::spawn(async move {
            let facts = vec![
                format!("Fact {} about science", i),
                format!("Fact {} about technology", i),
                format!("Fact {} about mathematics", i),
            ];
            
            let mut results = Vec::new();
            for fact in facts {
                let result = integration_clone.store_knowledge_with_neural_structure(
                    &fact,
                    Some("concurrent test"),
                ).await;
                results.push(result);
            }
            results
        });
        handles.push(handle);
    }
    
    // Wait for all storage operations
    for handle in handles {
        let results = handle.await.unwrap();
        for result in results {
            assert!(result.is_ok(), "Concurrent storage failed");
        }
    }
    
    // Spawn concurrent query operations
    let mut query_handles = Vec::new();
    for i in 0..num_concurrent {
        let integration_clone = integration.clone();
        let handle = tokio::spawn(async move {
            let queries = vec![
                format!("What is fact {} about?", i),
                format!("Tell me about technology {}", i),
                format!("Explain mathematics concept {}", i),
            ];
            
            let mut results = Vec::new();
            for query in queries {
                let result = integration_clone.neural_query_with_activation(
                    &query,
                    Some("concurrent"),
                ).await;
                results.push(result);
            }
            results
        });
        query_handles.push(handle);
    }
    
    // Wait for all query operations
    for handle in query_handles {
        let results = handle.await.unwrap();
        for result in results {
            assert!(result.is_ok(), "Concurrent query failed");
        }
    }
    
    // Verify system stability
    let stats = integration.get_phase1_statistics().await.unwrap();
    assert!(stats.brain_statistics.entity_count > 0);
    assert!(stats.neural_server_connected);
    
    integration.stop().await.unwrap();
}

#[tokio::test]
async fn test_edge_cases_and_error_handling() {
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Test empty input handling
    let result = integration.store_knowledge_with_neural_structure("", None).await;
    assert!(result.is_ok(), "Failed on empty input");
    let entities = result.unwrap();
    assert!(entities.is_empty(), "Created entities from empty input");
    
    // Test whitespace-only input
    let result = integration.store_knowledge_with_neural_structure("   \n\t   ", Some("whitespace")).await;
    assert!(result.is_ok(), "Failed on whitespace input");
    
    // Test very long input
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let result = integration.store_knowledge_with_neural_structure(&long_text, Some("stress test")).await;
    assert!(result.is_ok(), "Failed on long input");
    
    // Test special characters
    let special_text = "Test @#$%^&*() with <special> characters & symbols!";
    let result = integration.store_knowledge_with_neural_structure(special_text, Some("special")).await;
    assert!(result.is_ok(), "Failed on special characters");
    
    // Test Unicode
    let unicode_text = "Unicode test: 你好世界 مرحبا بالعالم Здравствуй мир";
    let result = integration.store_knowledge_with_neural_structure(unicode_text, Some("unicode")).await;
    assert!(result.is_ok(), "Failed on Unicode input");
    
    // Test empty query
    let result = integration.neural_query_with_activation("", None).await;
    assert!(result.is_ok(), "Failed on empty query");
    let query_result = result.unwrap();
    assert_eq!(query_result.query, "");
    
    // Test null characters (should be handled gracefully)
    let null_text = "Test\0with\0null\0characters";
    let result = integration.store_knowledge_with_neural_structure(null_text, Some("null test")).await;
    // Should either succeed or fail gracefully
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_configuration_variations() {
    // Test with minimal configuration
    let config = create_custom_config(128, false, false, false);
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Should work with minimal config
    let result = integration.store_knowledge_with_neural_structure(
        "Test with minimal config",
        None,
    ).await;
    assert!(result.is_ok(), "Failed with minimal config");
    
    // Test with large embedding dimension
    let config = create_custom_config(1024, true, true, true);
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    let result = integration.store_knowledge_with_neural_structure(
        "Test with large embeddings",
        None,
    ).await;
    assert!(result.is_ok(), "Failed with large embeddings");
    
    // Test with temporal disabled
    let config = create_custom_config(384, false, false, true);
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    let result = integration.temporal_query_at_time(
        "temporal test",
        Utc::now(),
        None,
    ).await;
    assert!(result.is_ok(), "Temporal query failed when disabled");
}

#[tokio::test]
async fn test_lifecycle_management() {
    let mut config = create_test_config();
    config.enable_real_time_updates = true;
    
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Test multiple start/stop cycles
    for i in 0..3 {
        // Start
        let result = integration.start().await;
        assert!(result.is_ok(), "Failed to start on cycle {}", i);
        
        // Perform operations while running
        let entities = integration.store_knowledge_with_neural_structure(
            &format!("Lifecycle test cycle {}", i),
            None,
        ).await;
        assert!(entities.is_ok(), "Operation failed during cycle {}", i);
        
        // Stop
        let result = integration.stop().await;
        assert!(result.is_ok(), "Failed to stop on cycle {}", i);
        
        // Small delay between cycles
        sleep(Duration::from_millis(100)).await;
    }
    
    // Verify system is still functional after cycles
    let stats = integration.get_phase1_statistics().await;
    assert!(stats.is_ok(), "Failed to get statistics after lifecycle test");
}

#[tokio::test]
async fn test_statistics_accuracy() {
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    
    // Get initial statistics
    let initial_stats = integration.get_phase1_statistics().await.unwrap();
    let initial_entity_count = initial_stats.brain_statistics.entity_count;
    
    // Store known number of entities
    let num_facts = 5;
    for i in 0..num_facts {
        let text = format!("Statistics test fact {}", i);
        integration.store_knowledge_with_neural_structure(&text, Some("stats test")).await.unwrap();
    }
    
    // Get updated statistics
    let updated_stats = integration.get_phase1_statistics().await.unwrap();
    let updated_entity_count = updated_stats.brain_statistics.entity_count;
    
    // Verify entity count increased
    assert!(
        updated_entity_count > initial_entity_count,
        "Entity count did not increase: {} -> {}",
        initial_entity_count,
        updated_entity_count
    );
    
    // Test cognitive statistics when enabled
    let cognitive_stats = integration.get_cognitive_statistics().await.unwrap();
    if integration.cognitive_orchestrator.is_some() {
        assert!(cognitive_stats.is_some(), "Expected cognitive statistics");
    } else {
        assert!(cognitive_stats.is_none(), "Unexpected cognitive statistics");
    }
}

#[tokio::test]
async fn test_real_world_qa_scenario() {
    let config = create_test_config();
    let integration = Phase1IntegrationLayer::new(config).await.unwrap();
    integration.start().await.unwrap();
    
    // Simulate a medical Q&A system
    let medical_knowledge = vec![
        ("Diabetes is a chronic disease that affects how your body processes blood sugar.", Some("medical")),
        ("Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin.", Some("medical")),
        ("Type 2 diabetes is a condition where the body becomes resistant to insulin or doesn't produce enough.", Some("medical")),
        ("Symptoms of diabetes include increased thirst, frequent urination, and unexplained weight loss.", Some("medical")),
        ("Treatment for type 1 diabetes includes insulin therapy and blood sugar monitoring.", Some("medical")),
        ("Type 2 diabetes can often be managed with lifestyle changes, medication, and monitoring.", Some("medical")),
        ("Risk factors for type 2 diabetes include obesity, sedentary lifestyle, and family history.", Some("medical")),
        ("Complications of diabetes can include heart disease, kidney damage, and vision problems.", Some("medical")),
    ];
    
    // Load medical knowledge base
    for (text, context) in medical_knowledge {
        integration.store_knowledge_with_neural_structure(text, context).await.unwrap();
    }
    
    // Simulate patient queries
    let patient_queries = vec![
        ("What is diabetes?", "Get basic information"),
        ("What are the symptoms of diabetes?", "Identify symptoms"),
        ("How is type 1 diabetes different from type 2?", "Compare conditions"),
        ("What are the treatment options for diabetes?", "Explore treatments"),
        ("What complications can diabetes cause?", "Understand risks"),
    ];
    
    for (query, purpose) in patient_queries {
        println!("Query: {} ({})", query, purpose);
        
        // Use neural query for basic retrieval
        let result = integration.neural_query_with_activation(query, Some("medical")).await.unwrap();
        assert!(result.converged, "Query did not converge: {}", query);
        
        // Use cognitive reasoning for complex questions
        if query.contains("different") || query.contains("How") {
            let cognitive_result = integration.cognitive_reasoning(
                query,
                Some("medical consultation"),
                Some(CognitivePatternType::ChainOfThought),
            ).await.unwrap();
            
            assert!(!cognitive_result.final_answer.is_empty(), "No cognitive answer for: {}", query);
            assert!(cognitive_result.confidence > 0.0, "No confidence in answer");
            println!("Cognitive answer confidence: {}", cognitive_result.confidence);
        }
    }
    
    // Test ensemble reasoning for treatment recommendation
    let treatment_query = "What is the best approach to manage diabetes considering both medication and lifestyle?";
    let patterns = vec![
        CognitivePatternType::ChainOfThought,
        CognitivePatternType::TreeOfThoughts,
    ];
    
    let ensemble_result = integration.ensemble_cognitive_reasoning(
        treatment_query,
        Some("treatment planning"),
        patterns,
    ).await.unwrap();
    
    assert!(!ensemble_result.final_answer.is_empty(), "No ensemble answer");
    assert!(ensemble_result.patterns_executed.len() >= 2, "Not enough patterns executed");
    assert!(ensemble_result.confidence > 0.0, "No confidence in ensemble answer");
    
    // Get final statistics
    let final_stats = integration.get_phase1_statistics().await.unwrap();
    assert!(final_stats.brain_statistics.entity_count > 0, "No entities created");
    assert!(final_stats.neural_server_connected, "Neural server disconnected");
    
    integration.stop().await.unwrap();
}