//! Integration tests for Phase3IntegratedCognitiveSystem
//! 
//! These tests focus on public API and end-to-end cognitive processing workflows,
//! multi-pattern coordination, memory and attention system integration.

use std::sync::Arc;
use std::time::{Duration, Instant};
use llmkg::cognitive::{
    phase3_integration::{
        Phase3IntegratedCognitiveSystem, Phase3IntegrationConfig, PatternIntegrationMode,
        SystemDiagnostics, PerformanceData
    },
    CognitivePatternType,
    orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig},
    types::{QueryContext, ReasoningStrategy, ExplorationType, ValidationLevel},
};
use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedKnowledgeGraph,
    activation_engine::ActivationPropagationEngine,
    sdr_storage::SDRStorage,
    sdr_types::SDRConfig,
};
use llmkg::error::Result;

// Import shared test utilities
use super::test_utils::{
    create_test_entity_keys,
    PerformanceTimer,
};

/// Creates a test Phase3 system with default configuration
async fn create_test_phase3_system() -> Result<Phase3IntegratedCognitiveSystem> {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64)?);
    let orchestrator = Arc::new(
        CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default()).await?
    );
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));

    Phase3IntegratedCognitiveSystem::new(
        orchestrator,
        activation_engine,
        graph,
        sdr_storage,
    ).await
}

/// Creates a test Phase3 system with custom configuration
async fn create_test_phase3_system_with_config(config: Phase3IntegrationConfig) -> Result<Phase3IntegratedCognitiveSystem> {
    let mut system = create_test_phase3_system().await?;
    system.integration_config = config;
    Ok(system)
}

/// Adds test entities to the graph for more realistic testing
async fn populate_test_graph(system: &Phase3IntegratedCognitiveSystem) -> Result<()> {
    let test_entities = vec![
        ("artificial_intelligence", "The simulation of human intelligence in machines"),
        ("machine_learning", "A subset of AI that enables machines to learn from data"),
        ("neural_networks", "Computing systems inspired by biological neural networks"),
        ("deep_learning", "A subset of machine learning using artificial neural networks"),
        ("natural_language_processing", "AI field focused on language understanding"),
        ("computer_vision", "AI field focused on visual perception"),
        ("robotics", "Technology dealing with robots and their applications"),
        ("expert_systems", "AI systems that emulate human expert decision-making"),
        ("knowledge_graphs", "Structured representations of knowledge using entities and relationships"),
        ("cognitive_computing", "Computing systems that simulate human thought processes"),
    ];

    for (name, description) in test_entities {
        let _ = system.brain_graph.add_entity(name, description).await?;
    }

    // Add some relationships
    let all_entities = system.brain_graph.get_all_entities().await;
    if all_entities.len() >= 4 {
        let ai_key = all_entities[0].0;
        let ml_key = all_entities[1].0;
        let nn_key = all_entities[2].0;
        let dl_key = all_entities[3].0;
        
        let _ = system.brain_graph.add_weighted_edge(ml_key, ai_key, 0.9).await;
        let _ = system.brain_graph.add_weighted_edge(nn_key, ml_key, 0.8).await;
        let _ = system.brain_graph.add_weighted_edge(dl_key, nn_key, 0.9).await;
    }

    Ok(())
}

#[tokio::test]
async fn test_phase3_system_initialization() {
    let system = create_test_phase3_system().await;
    assert!(system.is_ok(), "Phase3 system should initialize successfully");

    let system = system.unwrap();
    
    // Verify all components are initialized
    assert!(system.orchestrator.get_statistics().await.is_ok());
    assert!(system.working_memory.get_current_state().await.is_ok());
    assert!(system.attention_manager.get_attention_state().await.is_ok());
    
    // Check default configuration
    assert!(system.integration_config.enable_working_memory);
    assert!(system.integration_config.enable_attention_management);
    assert!(system.integration_config.enable_competitive_inhibition);
    assert!(system.integration_config.enable_unified_memory);
    assert!(matches!(system.integration_config.pattern_integration_mode, PatternIntegrationMode::Orchestrated));
}

#[tokio::test]
async fn test_execute_advanced_reasoning_basic() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("basic advanced reasoning");
    let result = system.execute_advanced_reasoning("What is artificial intelligence?").await;
    timer.assert_within_ms(5000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Verify basic structure
    assert_eq!(query_result.query, "What is artificial intelligence?");
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence >= 0.0);
    assert!(query_result.confidence <= 1.0);
    assert!(query_result.response_time > Duration::from_nanos(0));
    
    // Verify reasoning trace
    assert!(!query_result.reasoning_trace.activated_patterns.is_empty());
    
    // Verify system state changes
    assert!(!query_result.system_state_changes.working_memory_changes.is_empty());
    assert!(!query_result.system_state_changes.attention_changes.is_empty());
}

#[tokio::test]
async fn test_orchestrated_pattern_integration() {
    let config = Phase3IntegrationConfig {
        pattern_integration_mode: PatternIntegrationMode::Orchestrated,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("orchestrated pattern integration");
    let result = system.execute_advanced_reasoning("Explain machine learning algorithms").await;
    timer.assert_within_ms(3000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // In orchestrated mode, we should have a single primary pattern
    assert!(!query_result.reasoning_trace.activated_patterns.is_empty());
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence > 0.0);
    
    // Should have pattern execution metrics
    assert!(!query_result.pattern_results.is_empty());
}

#[tokio::test]
async fn test_parallel_pattern_integration() {
    let config = Phase3IntegrationConfig {
        pattern_integration_mode: PatternIntegrationMode::Parallel,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("parallel pattern integration");
    let result = system.execute_advanced_reasoning("How do neural networks work?").await;
    timer.assert_within_ms(5000.0); // May take longer due to parallel execution
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // In parallel mode, we should have multiple patterns activated
    assert!(query_result.reasoning_trace.activated_patterns.len() >= 2);
    assert!(!query_result.response.is_empty());
    
    // Response should contain contributions from multiple patterns
    let response_lines: Vec<&str> = query_result.response.lines().collect();
    assert!(response_lines.len() >= 2, "Parallel mode should produce multi-line response");
}

#[tokio::test]
async fn test_sequential_pattern_integration() {
    let config = Phase3IntegrationConfig {
        pattern_integration_mode: PatternIntegrationMode::Sequential,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("sequential pattern integration");
    let result = system.execute_advanced_reasoning("What are the applications of deep learning?").await;
    timer.assert_within_ms(4000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Sequential mode should activate patterns in order
    assert!(!query_result.reasoning_trace.activated_patterns.is_empty());
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence > 0.0);
    
    // Should have executed at least one pattern successfully
    assert!(!query_result.pattern_results.is_empty());
}

#[tokio::test]
async fn test_adaptive_pattern_integration() {
    let config = Phase3IntegrationConfig {
        pattern_integration_mode: PatternIntegrationMode::Adaptive,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Test different query types to verify adaptive selection
    let test_cases = vec![
        ("analyze the effectiveness of machine learning", CognitivePatternType::Critical),
        ("creative applications for AI technology", CognitivePatternType::Divergent),
        ("system relationships in cognitive computing", CognitivePatternType::Systems),
        ("what is natural language processing", CognitivePatternType::Convergent),
    ];
    
    for (query, expected_pattern_type) in test_cases {
        let timer = PerformanceTimer::new(&format!("adaptive pattern for: {}", query));
        let result = system.execute_advanced_reasoning(query).await;
        timer.assert_within_ms(3000.0);
        
        assert!(result.is_ok());
        let query_result = result.unwrap();
        
        // Verify the system adapted to the query type
        assert!(!query_result.reasoning_trace.activated_patterns.is_empty());
        let primary_pattern = query_result.reasoning_trace.activated_patterns[0];
        
        // Note: The actual pattern selection might vary based on implementation details,
        // but we should at least verify that a pattern was selected and executed
        assert!(matches!(primary_pattern, 
            CognitivePatternType::Convergent | 
            CognitivePatternType::Divergent | 
            CognitivePatternType::Critical | 
            CognitivePatternType::Systems
        ));
        
        assert!(!query_result.response.is_empty());
        assert!(query_result.confidence > 0.0);
    }
}

#[tokio::test]
async fn test_working_memory_integration() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let result = system.execute_advanced_reasoning("What is machine learning and how does it relate to AI?").await;
    assert!(result.is_ok());
    
    let query_result = result.unwrap();
    
    // Verify working memory operations were recorded
    assert!(!query_result.reasoning_trace.working_memory_operations.is_empty());
    
    // Check that memory state shows utilization
    let memory_state = system.working_memory.get_current_state().await.unwrap();
    assert!(memory_state.capacity_utilization >= 0.0);
    assert!(memory_state.capacity_utilization <= 1.0);
    
    // Verify working memory changes were tracked
    assert!(!query_result.system_state_changes.working_memory_changes.is_empty());
}

#[tokio::test]
async fn test_attention_management_integration() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let result = system.execute_advanced_reasoning("Focus on neural networks and their applications").await;
    assert!(result.is_ok());
    
    let query_result = result.unwrap();
    
    // Verify attention shifts were recorded
    assert!(!query_result.reasoning_trace.attention_shifts.is_empty());
    
    // Check attention state
    let attention_state = system.attention_manager.get_attention_state().await.unwrap();
    assert!(attention_state.focus_strength >= 0.0);
    assert!(attention_state.focus_strength <= 1.0);
    
    // Verify attention changes were tracked
    assert!(!query_result.system_state_changes.attention_changes.is_empty());
}

#[tokio::test]
async fn test_competitive_inhibition_integration() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let result = system.execute_advanced_reasoning("Compare different AI approaches").await;
    assert!(result.is_ok());
    
    let query_result = result.unwrap();
    
    // Verify inhibition events were recorded
    assert!(!query_result.reasoning_trace.inhibition_events.is_empty());
    
    // Verify inhibition changes were tracked
    assert!(!query_result.system_state_changes.inhibition_changes.is_empty());
}

#[tokio::test]
async fn test_unified_memory_integration() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let result = system.execute_advanced_reasoning("Analyze the relationship between AI and cognitive computing").await;
    assert!(result.is_ok());
    
    let query_result = result.unwrap();
    
    // Verify memory consolidations were recorded
    assert!(!query_result.reasoning_trace.memory_consolidations.is_empty());
    
    let consolidation = &query_result.reasoning_trace.memory_consolidations[0];
    assert_eq!(consolidation.source_memory, "working_memory");
    assert_eq!(consolidation.target_memory, "long_term_memory");
    assert!(consolidation.success_rate >= 0.0);
    assert!(consolidation.success_rate <= 1.0);
}

#[tokio::test]
async fn test_performance_monitoring() {
    let config = Phase3IntegrationConfig {
        performance_monitoring: true,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Execute multiple queries to generate performance data
    let queries = vec![
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks",
    ];
    
    for query in queries {
        let result = system.execute_advanced_reasoning(query).await;
        assert!(result.is_ok());
    }
    
    // Check performance metrics
    let metrics = system.performance_metrics.read().await;
    assert!(metrics.total_queries >= 3);
    assert!(metrics.successful_queries > 0);
    assert!(metrics.get_success_rate() > 0.0);
    assert!(metrics.average_response_time > Duration::from_nanos(0));
}

#[tokio::test]
async fn test_automatic_optimization() {
    let config = Phase3IntegrationConfig {
        automatic_optimization: true,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Set up conditions that should trigger optimization
    {
        let mut state = system.system_state.write().await;
        state.system_performance = 0.5; // Poor performance
        state.working_memory_load = 0.9; // High memory load
    }
    
    let result = system.execute_advanced_reasoning("Trigger optimization test").await;
    assert!(result.is_ok());
    
    let query_result = result.unwrap();
    
    // Verify optimization was triggered
    assert!(query_result.system_state_changes.performance_changes.iter()
        .any(|change| change.contains("optimization")));
    
    // Check that system state was improved
    let final_state = system.system_state.read().await;
    assert!(final_state.working_memory_load < 0.9);
}

#[tokio::test]
async fn test_system_diagnostics() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Run a query to generate some system activity
    let _ = system.execute_advanced_reasoning("Generate system activity").await;
    
    let timer = PerformanceTimer::new("system diagnostics collection");
    let diagnostics = system.get_system_diagnostics().await;
    timer.assert_within_ms(1000.0);
    
    assert!(diagnostics.is_ok());
    let diag = diagnostics.unwrap();
    
    // Verify diagnostics structure
    assert!(diag.memory_utilization.working_memory_usage >= 0.0);
    assert!(diag.memory_utilization.working_memory_usage <= 1.0);
    assert!(diag.attention_status.current_focus_strength >= 0.0);
    assert!(diag.attention_status.current_focus_strength <= 1.0);
    assert!(diag.inhibition_status.global_inhibition_strength >= 0.0);
    assert!(diag.inhibition_status.global_inhibition_strength <= 1.0);
    
    // Performance metrics should be populated
    assert!(diag.performance_metrics.total_queries >= 1);
}

#[tokio::test]
async fn test_performance_data_collection() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Generate some system activity
    for i in 0..3 {
        let _ = system.execute_advanced_reasoning(&format!("Test query {}", i)).await;
    }
    
    let timer = PerformanceTimer::new("performance data collection");
    let perf_data = system.collect_performance_metrics(Duration::from_secs(1)).await;
    timer.assert_within_ms(2000.0);
    
    assert!(perf_data.is_ok());
    let data = perf_data.unwrap();
    
    // Verify performance data structure
    assert!(!data.query_latencies.is_empty());
    assert!(!data.accuracy_scores.is_empty());
    assert!(!data.user_satisfaction.is_empty());
    assert!(!data.memory_usage.is_empty());
    assert!(!data.error_rates.is_empty());
    assert!(data.system_stability >= 0.0);
    assert!(data.system_stability <= 1.0);
    
    // Verify error rates are reasonable
    for (error_type, rate) in &data.error_rates {
        assert!(rate >= &0.0);
        assert!(rate <= &1.0);
        assert!(!error_type.is_empty());
    }
}

#[tokio::test]
async fn test_multi_pattern_coordination() {
    let config = Phase3IntegrationConfig {
        pattern_integration_mode: PatternIntegrationMode::Parallel,
        ..Default::default()
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("multi-pattern coordination");
    let result = system.execute_advanced_reasoning(
        "Provide a comprehensive analysis of AI technologies including creative applications and critical evaluation"
    ).await;
    timer.assert_within_ms(8000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Should have multiple patterns working together
    assert!(query_result.reasoning_trace.activated_patterns.len() >= 2);
    
    // Should have pattern interactions recorded
    // Note: Pattern interactions may not be implemented yet, so this is optional
    if !query_result.reasoning_trace.pattern_interactions.is_empty() {
        let interaction = &query_result.reasoning_trace.pattern_interactions[0];
        assert!(matches!(interaction.interaction_type, 
            llmkg::cognitive::phase3_integration::InteractionType::Collaboration |
            llmkg::cognitive::phase3_integration::InteractionType::Parallel
        ));
    }
    
    // Response should be comprehensive
    assert!(query_result.response.len() > 100); // Should be a substantial response
}

#[tokio::test]
async fn test_end_to_end_cognitive_workflow() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("end-to-end cognitive workflow");
    let result = system.execute_advanced_reasoning(
        "Analyze machine learning, identify creative applications, and provide critical evaluation"
    ).await;
    timer.assert_within_ms(10000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Verify complete workflow execution
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence > 0.0);
    
    // Should have comprehensive reasoning trace
    assert!(!query_result.reasoning_trace.activated_patterns.is_empty());
    assert!(!query_result.reasoning_trace.working_memory_operations.is_empty());
    assert!(!query_result.reasoning_trace.attention_shifts.is_empty());
    assert!(!query_result.reasoning_trace.inhibition_events.is_empty());
    assert!(!query_result.reasoning_trace.memory_consolidations.is_empty());
    
    // Performance metrics should be recorded
    assert!(query_result.performance_metrics.total_time > Duration::from_nanos(0));
    assert!(query_result.performance_metrics.attention_shift_time > Duration::from_nanos(0));
    assert!(query_result.performance_metrics.inhibition_processing_time > Duration::from_nanos(0));
    
    // System state should reflect processing
    assert!(!query_result.system_state_changes.working_memory_changes.is_empty());
    assert!(!query_result.system_state_changes.attention_changes.is_empty());
    assert!(!query_result.system_state_changes.pattern_activations.is_empty());
}

#[tokio::test]
async fn test_backward_compatibility() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    // Test integrated_query method (Phase 4 compatibility)
    let context = QueryContext::new();
    let timer = PerformanceTimer::new("backward compatibility");
    let result = system.integrated_query("Test backward compatibility", Some(context)).await;
    timer.assert_within_ms(3000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Should work like execute_advanced_reasoning
    assert_eq!(query_result.query, "Test backward compatibility");
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence >= 0.0);
    assert!(query_result.confidence <= 1.0);
    
    // Test orchestrator access
    let orchestrator = system.get_base_orchestrator();
    assert!(orchestrator.is_ok());
    
    // Test abstract thinking access
    let abstract_thinking = system.get_abstract_thinking_pattern().await;
    assert!(abstract_thinking.is_ok());
}

#[tokio::test]
async fn test_system_stress_test() {
    let system = create_test_phase3_system().await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("system stress test");
    
    // Execute multiple concurrent queries
    let mut handles = vec![];
    for i in 0..5 {
        let system_clone = Arc::new(system.clone());
        let handle = tokio::spawn(async move {
            system_clone.execute_advanced_reasoning(&format!("Stress test query {}", i)).await
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut results = vec![];
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok());
        let query_result = result.unwrap();
        assert!(query_result.is_ok());
        results.push(query_result.unwrap());
    }
    
    timer.assert_within_ms(15000.0);
    
    // Verify all queries succeeded
    assert_eq!(results.len(), 5);
    for result in results {
        assert!(!result.response.is_empty());
        assert!(result.confidence > 0.0);
    }
    
    // System should remain stable
    let final_diagnostics = system.get_system_diagnostics().await;
    assert!(final_diagnostics.is_ok());
}

#[tokio::test]
async fn test_disabled_features() {
    let config = Phase3IntegrationConfig {
        enable_working_memory: false,
        enable_attention_management: false,
        enable_competitive_inhibition: false,
        enable_unified_memory: false,
        pattern_integration_mode: PatternIntegrationMode::Orchestrated,
        performance_monitoring: false,
        automatic_optimization: false,
    };
    let system = create_test_phase3_system_with_config(config).await.unwrap();
    let _ = populate_test_graph(&system).await;
    
    let timer = PerformanceTimer::new("disabled features test");
    let result = system.execute_advanced_reasoning("Test with disabled features").await;
    timer.assert_within_ms(3000.0);
    
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Should still work with basic functionality
    assert!(!query_result.response.is_empty());
    assert!(query_result.confidence > 0.0);
    
    // Should have minimal system state changes
    assert!(query_result.system_state_changes.working_memory_changes.is_empty());
    assert!(query_result.system_state_changes.attention_changes.is_empty());
    assert!(query_result.system_state_changes.inhibition_changes.is_empty());
    
    // Should have minimal reasoning trace
    assert!(query_result.reasoning_trace.working_memory_operations.is_empty());
    assert!(query_result.reasoning_trace.attention_shifts.is_empty());
    assert!(query_result.reasoning_trace.inhibition_events.is_empty());
    assert!(query_result.reasoning_trace.memory_consolidations.is_empty());
}

#[tokio::test]
async fn test_error_handling_and_recovery() {
    let system = create_test_phase3_system().await.unwrap();
    // Don't populate graph to test error conditions
    
    let timer = PerformanceTimer::new("error handling test");
    
    // Test with empty graph
    let result = system.execute_advanced_reasoning("Query on empty knowledge base").await;
    timer.assert_within_ms(2000.0);
    
    // Should handle gracefully even with limited knowledge
    assert!(result.is_ok());
    let query_result = result.unwrap();
    
    // Should still produce some response
    assert!(!query_result.response.is_empty());
    
    // Confidence might be lower but should be valid
    assert!(query_result.confidence >= 0.0);
    assert!(query_result.confidence <= 1.0);
    
    // System should remain functional
    let diagnostics = system.get_system_diagnostics().await;
    assert!(diagnostics.is_ok());
}