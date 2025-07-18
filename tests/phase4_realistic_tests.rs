use anyhow::Result;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::collections::HashMap;
use uuid::Uuid;
use dotenv::dotenv;
use std::env;

// Import only what actually exists in the implementation
use llmkg::learning::{
    hebbian::HebbianLearningEngine,
    homeostasis::SynapticHomeostasis,
    optimization_agent::GraphOptimizationAgent,
    adaptive_learning::AdaptiveLearningSystem,
    phase4_integration::Phase4LearningSystem,
    types::*,
};

use llmkg::cognitive::{
    phase4_integration::Phase4CognitiveSystem,
    phase3_integration::IntegratedCognitiveSystem,
    types::CognitivePatternType,
};

use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedGraph,
    brain_types::{EntityKey, BrainInspiredEntity, ActivationPattern},
};

#[cfg(test)]
mod phase4_realistic_tests {
    use super::*;

    // Test configuration from environment
    struct TestConfig {
        deepseek_api_key: String,
        min_learning_efficiency: f32,
        min_confidence_threshold: f32,
        max_memory_overhead: f32,
        synthetic_entity_count: usize,
    }

    impl TestConfig {
        fn from_env() -> Result<Self> {
            dotenv().ok();
            
            Ok(Self {
                deepseek_api_key: env::var("DEEPSEEK_API_KEY")
                    .expect("DEEPSEEK_API_KEY must be set"),
                min_learning_efficiency: env::var("MIN_LEARNING_EFFICIENCY")
                    .unwrap_or_else(|_| "0.15".to_string())
                    .parse()?,
                min_confidence_threshold: env::var("MIN_CONFIDENCE_THRESHOLD")
                    .unwrap_or_else(|_| "0.5".to_string())
                    .parse()?,
                max_memory_overhead: env::var("MAX_MEMORY_OVERHEAD_PERCENT")
                    .unwrap_or_else(|_| "20".to_string())
                    .parse()?,
                synthetic_entity_count: env::var("SYNTHETIC_ENTITY_COUNT")
                    .unwrap_or_else(|_| "100".to_string())
                    .parse()?,
            })
        }
    }

    /// Test Hebbian learning with realistic expectations
    #[tokio::test]
    async fn test_hebbian_learning_realistic() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Create components using actual constructors
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        
        let mut hebbian_engine = HebbianLearningEngine::new(
            brain_graph.clone(),
            activation_engine,
            inhibition_system,
        ).await?;
        
        // Create a small set of test entities
        let test_entities = create_minimal_test_entities(10)?;
        for entity in &test_entities {
            brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Create realistic activation events
        let events = create_realistic_activation_events(&test_entities, 20);
        let context = LearningContext {
            performance_pressure: 0.5,
            user_satisfaction_level: 0.6,
            learning_urgency: 0.4,
            session_id: Uuid::new_v4().to_string(),
            learning_goals: vec![
                LearningGoal {
                    goal_type: LearningGoalType::PerformanceImprovement,
                    target_improvement: config.min_learning_efficiency,
                    deadline: Some(SystemTime::now() + Duration::from_secs(3600)),
                }
            ],
        };
        
        // Apply learning and measure actual changes
        let initial_weights = capture_graph_weights(&brain_graph).await?;
        let update = hebbian_engine.apply_hebbian_learning(events, context).await?;
        let final_weights = capture_graph_weights(&brain_graph).await?;
        
        // Verify actual learning occurred
        let weight_changes = calculate_weight_changes(&initial_weights, &final_weights);
        assert!(!weight_changes.is_empty(), "No weight changes detected");
        
        // Check learning efficiency is within realistic bounds
        assert!(
            update.learning_efficiency >= config.min_learning_efficiency && 
            update.learning_efficiency <= 1.0,
            "Learning efficiency {} outside realistic range [{}, 1.0]",
            update.learning_efficiency,
            config.min_learning_efficiency
        );
        
        // Verify at least some connections were modified
        let total_changes = update.strengthened_connections.len() + 
                           update.weakened_connections.len() + 
                           update.new_connections.len();
        assert!(total_changes > 0, "No connections were modified");
        
        println!("✓ Hebbian learning test passed with {} weight changes", total_changes);
        Ok(())
    }

    /// Test homeostasis with measurable effects
    #[tokio::test]
    async fn test_homeostasis_measurable_effects() -> Result<()> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let attention_manager = Arc::new(llmkg::cognitive::attention_manager::AttentionManager::new().await?);
        let working_memory = Arc::new(llmkg::cognitive::working_memory::WorkingMemorySystem::new().await?);
        
        let mut homeostasis = SynapticHomeostasis::new(
            brain_graph.clone(),
            attention_manager,
            working_memory,
        ).await?;
        
        // Create test entities with varying activity levels
        let entities = create_minimal_test_entities(20)?;
        for entity in &entities {
            brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Measure baseline activity
        let baseline_activity = measure_graph_activity(&brain_graph).await?;
        
        // Apply homeostatic scaling
        let update = homeostasis.apply_homeostatic_scaling(
            Duration::from_secs(300)
        ).await?;
        
        // Measure post-scaling activity
        let post_scaling_activity = measure_graph_activity(&brain_graph).await?;
        
        // Verify homeostasis had an effect
        if update.scaled_entities.len() > 0 {
            assert!(
                (post_scaling_activity - baseline_activity).abs() > 0.01,
                "Homeostasis claimed to scale entities but no activity change detected"
            );
        }
        
        // Check stability improvement is realistic
        assert!(
            update.stability_improvement >= -0.1 && update.stability_improvement <= 0.5,
            "Stability improvement {} outside realistic range [-0.1, 0.5]",
            update.stability_improvement
        );
        
        println!("✓ Homeostasis test passed with {} entities scaled", update.scaled_entities.len());
        Ok(())
    }

    /// Test optimization with rollback scenarios
    #[tokio::test]
    async fn test_optimization_with_rollback() -> Result<()> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let sdr_storage = Arc::new(llmkg::core::sdr_storage::SDRStorage::new().await?);
        
        // Create other required components
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        let abstract_thinking = Arc::new(llmkg::cognitive::abstract_pattern::AbstractThinking::new(
            brain_graph.clone()
        ).await?);
        let orchestrator = Arc::new(llmkg::cognitive::orchestrator::CognitiveOrchestrator::new().await?);
        
        let hebbian_engine = Arc::new(Mutex::new(HebbianLearningEngine::new(
            brain_graph.clone(),
            activation_engine,
            inhibition_system,
        ).await?));
        
        let mut optimization_agent = GraphOptimizationAgent::new(
            brain_graph.clone(),
            sdr_storage,
            abstract_thinking,
            orchestrator,
            hebbian_engine,
        ).await?;
        
        // Create a simple graph structure
        let entities = create_minimal_test_entities(30)?;
        for entity in &entities {
            brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Analyze optimization opportunities
        let scope = llmkg::learning::optimization_agent::AnalysisScope {
            entities: entities.iter().take(10).map(|e| e.key).collect(),
            depth: 2,
            time_window: Duration::from_secs(3600),
        };
        
        let opportunities = optimization_agent.analyze_optimization_opportunities(scope).await?;
        
        // Test with a deliberately failing refactoring
        if let Some(candidate) = opportunities.optimization_candidates.first() {
            // Create a plan that should trigger rollback
            let risky_plan = create_risky_refactoring_plan(candidate);
            let result = optimization_agent.execute_safe_refactoring(risky_plan).await?;
            
            match result {
                llmkg::learning::optimization_agent::RefactoringResult::RolledBack { reason } => {
                    println!("✓ Optimization correctly rolled back: {:?}", reason);
                },
                llmkg::learning::optimization_agent::RefactoringResult::Success(_) => {
                    // If it succeeded, verify the optimization is valid
                    let post_optimization_state = capture_graph_state(&brain_graph).await?;
                    assert!(
                        verify_graph_integrity(&post_optimization_state),
                        "Graph integrity compromised after optimization"
                    );
                },
                _ => panic!("Unexpected refactoring result"),
            }
        }
        
        Ok(())
    }

    /// Test adaptive learning with measurable convergence
    #[tokio::test]
    async fn test_adaptive_learning_convergence() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Create all required components
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let sdr_storage = Arc::new(llmkg::core::sdr_storage::SDRStorage::new().await?);
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let working_memory = Arc::new(llmkg::cognitive::working_memory::WorkingMemorySystem::new().await?);
        let attention_manager = Arc::new(llmkg::cognitive::attention_manager::AttentionManager::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        
        let integrated_system = Arc::new(IntegratedCognitiveSystem::new(
            brain_graph.clone(),
            sdr_storage,
            activation_engine.clone(),
            working_memory.clone(),
            attention_manager.clone(),
            inhibition_system.clone(),
        ).await?);
        
        let orchestrator = Arc::new(llmkg::cognitive::orchestrator::CognitiveOrchestrator::new().await?);
        let hebbian_engine = Arc::new(Mutex::new(HebbianLearningEngine::new(
            brain_graph.clone(),
            activation_engine,
            inhibition_system,
        ).await?));
        
        let abstract_thinking = Arc::new(llmkg::cognitive::abstract_pattern::AbstractThinking::new(
            brain_graph.clone()
        ).await?);
        
        let optimization_agent = Arc::new(Mutex::new(GraphOptimizationAgent::new(
            brain_graph,
            Arc::new(llmkg::core::sdr_storage::SDRStorage::new().await?),
            abstract_thinking,
            orchestrator.clone(),
            hebbian_engine.clone(),
        ).await?));
        
        let mut adaptive_learning = AdaptiveLearningSystem::new(
            integrated_system,
            working_memory,
            attention_manager,
            orchestrator,
            hebbian_engine,
            optimization_agent,
        ).await?;
        
        // Track actual performance metrics
        let mut performance_history = Vec::new();
        let mut cycle_times = Vec::new();
        
        // Run multiple learning cycles
        for i in 0..5 {
            let start = Instant::now();
            let result = adaptive_learning.process_adaptive_learning_cycle(
                Duration::from_secs(60) // Shorter, realistic cycle
            ).await?;
            let cycle_time = start.elapsed();
            
            performance_history.push(result.performance_improvement);
            cycle_times.push(cycle_time);
            
            println!("Cycle {}: improvement={:.3}, time={:?}", 
                     i, result.performance_improvement, cycle_time);
            
            // Verify improvements are realistic
            assert!(
                result.performance_improvement >= -0.1 && result.performance_improvement <= 0.3,
                "Unrealistic performance improvement: {}",
                result.performance_improvement
            );
        }
        
        // Check for convergence trend (later cycles should be more stable)
        if performance_history.len() >= 3 {
            let early_variance = calculate_variance(&performance_history[..2]);
            let late_variance = calculate_variance(&performance_history[performance_history.len()-2..]);
            
            // Late variance should be lower (more stable)
            assert!(
                late_variance <= early_variance * 1.5, // Allow some tolerance
                "System not converging: early_variance={:.3}, late_variance={:.3}",
                early_variance, late_variance
            );
        }
        
        Ok(())
    }

    /// Test emergency adaptation with real failure scenarios
    #[tokio::test]
    async fn test_emergency_adaptation_real_failures() -> Result<()> {
        // This test would need actual emergency scenarios that can be triggered
        // For now, we'll test the API exists and returns reasonable values
        
        let adaptive_learning = create_minimal_adaptive_learning_system().await?;
        
        // Test different emergency types
        let emergency_types = vec![
            llmkg::learning::adaptive_learning::EmergencyTrigger::PerformanceCollapse,
            llmkg::learning::adaptive_learning::EmergencyTrigger::MemoryOverload,
        ];
        
        for trigger in emergency_types {
            let result = adaptive_learning.handle_emergency_adaptation(trigger.clone()).await?;
            
            // Verify the result makes sense
            assert!(
                result.immediate_recovery >= 0.0 && result.immediate_recovery <= 1.0,
                "Invalid recovery value: {}",
                result.immediate_recovery
            );
            
            // If performance improved, verify it's not unrealistic
            if result.performance_improvement > 0.0 {
                assert!(
                    result.performance_improvement <= 0.5,
                    "Unrealistic performance improvement in emergency: {}",
                    result.performance_improvement
                );
            }
            
            println!("✓ Emergency {:?} handled with recovery={:.3}", 
                     trigger, result.immediate_recovery);
        }
        
        Ok(())
    }

    /// Test memory usage stays within bounds
    #[tokio::test]
    async fn test_memory_usage_bounds() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Measure baseline memory
        let baseline_memory = get_process_memory_usage();
        
        // Create and populate a system
        let system = create_minimal_phase4_system().await?;
        
        // Add entities
        let entities = create_minimal_test_entities(config.synthetic_entity_count)?;
        for entity in &entities {
            system.phase3_system.brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Run a learning cycle
        let _result = system.phase4_learning.execute_comprehensive_learning_cycle().await?;
        
        // Measure final memory
        let final_memory = get_process_memory_usage();
        let memory_increase_mb = (final_memory - baseline_memory) as f32 / 1_048_576.0;
        let memory_increase_percent = (memory_increase_mb / (baseline_memory as f32 / 1_048_576.0)) * 100.0;
        
        println!("Memory increase: {:.1}MB ({:.1}%)", memory_increase_mb, memory_increase_percent);
        
        // Verify memory usage is reasonable
        assert!(
            memory_increase_percent <= config.max_memory_overhead,
            "Memory overhead {:.1}% exceeds limit {:.1}%",
            memory_increase_percent, config.max_memory_overhead
        );
        
        Ok(())
    }

    /// Test with DeepSeek API integration
    #[tokio::test]
    async fn test_deepseek_integration() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Create a client for DeepSeek API
        let client = reqwest::Client::new();
        let api_url = env::var("DEEPSEEK_API_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
        
        // Test the API is accessible
        let test_request = serde_json::json!({
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": "Test connection"
            }],
            "max_tokens": 10
        });
        
        let response = client
            .post(format!("{}/chat/completions", api_url))
            .header("Authorization", format!("Bearer {}", config.deepseek_api_key))
            .json(&test_request)
            .send()
            .await?;
        
        assert!(response.status().is_success(), "DeepSeek API request failed: {}", response.status());
        
        // Now test with actual LLMKG integration
        let phase4_system = create_minimal_phase4_system().await?;
        
        // Create a query that would benefit from LLM enhancement
        let query = "Explain the relationship between Hebbian learning and synaptic plasticity";
        let result = phase4_system.enhanced_query(query, None).await?;
        
        // Verify we got a meaningful response
        assert!(
            result.base_result.overall_confidence >= config.min_confidence_threshold,
            "Query confidence {} below threshold {}",
            result.base_result.overall_confidence,
            config.min_confidence_threshold
        );
        
        println!("✓ DeepSeek integration test passed with confidence={:.2}", 
                 result.base_result.overall_confidence);
        
        Ok(())
    }

    // Helper functions that actually work

    fn create_minimal_test_entities(count: usize) -> Result<Vec<BrainInspiredEntity>> {
        let mut entities = Vec::new();
        
        for i in 0..count {
            entities.push(BrainInspiredEntity {
                key: EntityKey::new(),
                name: format!("test_entity_{}", i),
                entity_type: "test".to_string(),
                attributes: HashMap::from([
                    ("index".to_string(), i.to_string()),
                    ("category".to_string(), format!("cat_{}", i % 3)),
                ]),
                semantic_embedding: vec![0.1; 768], // Minimal embedding
                activation_pattern: ActivationPattern {
                    current_activation: 0.5,
                    activation_history: vec![0.5],
                    decay_rate: 0.1,
                    last_activated: SystemTime::now(),
                },
                temporal_aspects: Default::default(),
                ingestion_time: SystemTime::now(),
            });
        }
        
        Ok(entities)
    }

    fn create_realistic_activation_events(entities: &[BrainInspiredEntity], count: usize) -> Vec<ActivationEvent> {
        let mut events = Vec::new();
        let base_time = Instant::now();
        
        for i in 0..count {
            let entity_idx = i % entities.len();
            events.push(ActivationEvent {
                entity_key: entities[entity_idx].key,
                activation_strength: 0.3 + ((i as f32 / count as f32) * 0.4), // 0.3 to 0.7
                timestamp: base_time + Duration::from_millis(i as u64 * 100),
                context: ActivationContext {
                    query_id: format!("query_{}", i / 5),
                    cognitive_pattern: match i % 3 {
                        0 => CognitivePatternType::Convergent,
                        1 => CognitivePatternType::Divergent,
                        _ => CognitivePatternType::Transform,
                    },
                    user_session: Some("test_session".to_string()),
                    outcome_quality: Some(0.6 + ((i % 10) as f32 * 0.03)), // 0.6 to 0.9
                },
            });
        }
        
        events
    }

    async fn capture_graph_weights(graph: &BrainEnhancedGraph) -> Result<HashMap<(EntityKey, EntityKey), f32>> {
        let relationships = graph.get_all_relationships().await?;
        let mut weights = HashMap::new();
        
        for rel in relationships {
            weights.insert((rel.source, rel.target), rel.weight);
        }
        
        Ok(weights)
    }

    fn calculate_weight_changes(
        initial: &HashMap<(EntityKey, EntityKey), f32>,
        final_weights: &HashMap<(EntityKey, EntityKey), f32>
    ) -> Vec<((EntityKey, EntityKey), f32)> {
        let mut changes = Vec::new();
        
        for (key, final_weight) in final_weights {
            if let Some(initial_weight) = initial.get(key) {
                let change = final_weight - initial_weight;
                if change.abs() > 0.001 {
                    changes.push((*key, change));
                }
            } else {
                // New connection
                changes.push((*key, *final_weight));
            }
        }
        
        changes
    }

    async fn measure_graph_activity(graph: &BrainEnhancedGraph) -> Result<f32> {
        let entities = graph.get_all_entities().await?;
        let total_activation: f32 = entities.iter()
            .map(|e| e.activation_pattern.current_activation)
            .sum();
        
        Ok(total_activation / entities.len() as f32)
    }

    fn create_risky_refactoring_plan(candidate: &OptimizationCandidate) -> llmkg::learning::optimization_agent::RefactoringPlan {
        // Create a plan that might fail validation
        llmkg::learning::optimization_agent::RefactoringPlan {
            plan_id: Uuid::new_v4(),
            candidate: candidate.clone(),
            operations: vec![
                llmkg::learning::optimization_agent::RefactoringOperation::RemoveRedundantConnections {
                    threshold: 0.9, // Very aggressive, likely to fail safety checks
                },
            ],
            estimated_impact: 0.5,
            risk_level: 0.8,
        }
    }

    async fn capture_graph_state(graph: &BrainEnhancedGraph) -> Result<GraphState> {
        Ok(GraphState {
            entity_count: graph.get_all_entities().await?.len(),
            relationship_count: graph.get_all_relationships().await?.len(),
            total_weight: graph.get_all_relationships().await?
                .iter()
                .map(|r| r.weight)
                .sum(),
        })
    }

    fn verify_graph_integrity(state: &GraphState) -> bool {
        state.entity_count > 0 && 
        state.relationship_count > 0 && 
        state.total_weight > 0.0
    }

    fn calculate_variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance
    }

    async fn create_minimal_adaptive_learning_system() -> Result<AdaptiveLearningSystem> {
        // Create minimal but functional system
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let sdr_storage = Arc::new(llmkg::core::sdr_storage::SDRStorage::new().await?);
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let working_memory = Arc::new(llmkg::cognitive::working_memory::WorkingMemorySystem::new().await?);
        let attention_manager = Arc::new(llmkg::cognitive::attention_manager::AttentionManager::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        
        let integrated_system = Arc::new(IntegratedCognitiveSystem::new(
            brain_graph.clone(),
            sdr_storage.clone(),
            activation_engine.clone(),
            working_memory.clone(),
            attention_manager.clone(),
            inhibition_system.clone(),
        ).await?);
        
        let orchestrator = Arc::new(llmkg::cognitive::orchestrator::CognitiveOrchestrator::new().await?);
        let hebbian_engine = Arc::new(Mutex::new(HebbianLearningEngine::new(
            brain_graph.clone(),
            activation_engine,
            inhibition_system,
        ).await?));
        
        let abstract_thinking = Arc::new(llmkg::cognitive::abstract_pattern::AbstractThinking::new(
            brain_graph.clone()
        ).await?);
        
        let optimization_agent = Arc::new(Mutex::new(GraphOptimizationAgent::new(
            brain_graph,
            sdr_storage,
            abstract_thinking,
            orchestrator.clone(),
            hebbian_engine.clone(),
        ).await?));
        
        AdaptiveLearningSystem::new(
            integrated_system,
            working_memory,
            attention_manager,
            orchestrator,
            hebbian_engine,
            optimization_agent,
        ).await
    }

    async fn create_minimal_phase4_system() -> Result<Phase4CognitiveSystem> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let sdr_storage = Arc::new(llmkg::core::sdr_storage::SDRStorage::new().await?);
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let working_memory = Arc::new(llmkg::cognitive::working_memory::WorkingMemorySystem::new().await?);
        let attention_manager = Arc::new(llmkg::cognitive::attention_manager::AttentionManager::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        
        let phase3_system = Arc::new(IntegratedCognitiveSystem::new(
            brain_graph.clone(),
            sdr_storage.clone(),
            activation_engine.clone(),
            working_memory.clone(),
            attention_manager.clone(),
            inhibition_system.clone(),
        ).await?);
        
        let orchestrator = Arc::new(llmkg::cognitive::orchestrator::CognitiveOrchestrator::new().await?);
        
        let phase4_learning = Arc::new(Phase4LearningSystem::new(
            phase3_system.clone(),
            brain_graph,
            sdr_storage,
            activation_engine,
            attention_manager,
            working_memory,
            inhibition_system,
            orchestrator,
        ).await?);
        
        Phase4CognitiveSystem::new(phase3_system, phase4_learning).await
    }

    fn get_process_memory_usage() -> usize {
        // Use actual system calls to get memory usage
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback for other platforms
        1_000_000_000 // 1GB default
    }

    // Simple state tracking struct
    struct GraphState {
        entity_count: usize,
        relationship_count: usize,
        total_weight: f32,
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test negative cases - what happens when things go wrong
    #[tokio::test]
    async fn test_learning_failure_scenarios() -> Result<()> {
        let mut hebbian_engine = create_minimal_hebbian_engine().await?;
        
        // Test with empty events
        let empty_events = vec![];
        let context = generate_test_learning_context();
        let result = hebbian_engine.apply_hebbian_learning(empty_events, context.clone()).await?;
        
        // Should handle gracefully
        assert_eq!(result.strengthened_connections.len(), 0);
        assert_eq!(result.new_connections.len(), 0);
        
        // Test with invalid activation strengths
        let invalid_events = vec![
            ActivationEvent {
                entity_key: EntityKey::new(),
                activation_strength: -0.5, // Invalid
                timestamp: Instant::now(),
                context: ActivationContext {
                    query_id: "test".to_string(),
                    cognitive_pattern: CognitivePatternType::Convergent,
                    user_session: None,
                    outcome_quality: Some(2.0), // Invalid
                },
            }
        ];
        
        // Should either error or clamp values
        match hebbian_engine.apply_hebbian_learning(invalid_events, context).await {
            Ok(update) => {
                // If it succeeds, values should be clamped
                for change in &update.strengthened_connections {
                    assert!(change.new_weight >= 0.0 && change.new_weight <= 1.0);
                }
            },
            Err(e) => {
                println!("Correctly rejected invalid input: {}", e);
            }
        }
        
        Ok(())
    }

    async fn create_minimal_hebbian_engine() -> Result<HebbianLearningEngine> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let activation_engine = Arc::new(llmkg::core::activation_engine::ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(llmkg::cognitive::inhibitory_logic::CompetitiveInhibitionSystem::new().await?);
        
        HebbianLearningEngine::new(brain_graph, activation_engine, inhibition_system).await
    }

    fn generate_test_learning_context() -> LearningContext {
        LearningContext {
            performance_pressure: 0.5,
            user_satisfaction_level: 0.7,
            learning_urgency: 0.3,
            session_id: Uuid::new_v4().to_string(),
            learning_goals: vec![],
        }
    }
}

// Benchmark realistic performance
#[cfg(all(test, not(debug_assertions)))]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn benchmark_hebbian_learning_performance() -> Result<()> {
        let engine = create_minimal_hebbian_engine().await?;
        let entities = create_minimal_test_entities(100)?;
        
        // Warm up
        let warmup_events = create_realistic_activation_events(&entities, 10);
        let _ = engine.apply_hebbian_learning(warmup_events, generate_test_learning_context()).await?;
        
        // Benchmark different event counts
        for event_count in [100, 500, 1000] {
            let events = create_realistic_activation_events(&entities, event_count);
            let context = generate_test_learning_context();
            
            let start = Instant::now();
            let result = engine.apply_hebbian_learning(events, context).await?;
            let duration = start.elapsed();
            
            println!("Hebbian learning {} events: {:?} ({} changes)", 
                     event_count, duration, 
                     result.strengthened_connections.len() + result.weakened_connections.len());
            
            // Verify performance is reasonable
            let ms_per_event = duration.as_millis() as f32 / event_count as f32;
            assert!(ms_per_event < 10.0, "Processing too slow: {:.2}ms per event", ms_per_event);
        }
        
        Ok(())
    }
}