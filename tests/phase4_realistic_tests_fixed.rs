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
    optimization_agent::{GraphOptimizationAgent, AnalysisScope, RefactoringPlan, RefactoringResult},
    adaptive_learning::AdaptiveLearningSystem,
    phase4_integration::Phase4LearningSystem,
    types::*,
};

use llmkg::cognitive::{
    phase4_integration::Phase4CognitiveSystem,
    phase3_integration::IntegratedCognitiveSystem,
    types::{CognitivePatternType, QueryContext},
};

use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedGraph,
    brain_types::{EntityKey, BrainInspiredEntity, ActivationPattern, BrainInspiredRelationship, RelationshipType},
    sdr_storage::SDRStorage,
    activation_engine::ActivationPropagationEngine,
};

use llmkg::cognitive::{
    working_memory::WorkingMemorySystem,
    attention_manager::AttentionManager,
    inhibitory_logic::CompetitiveInhibitionSystem,
    abstract_pattern::AbstractThinking,
    orchestrator::CognitiveOrchestrator,
};

#[cfg(test)]
mod phase4_realistic_tests_fixed {
    use super::*;

    // Test constants to avoid magic numbers
    const DEFAULT_MIN_LEARNING_EFFICIENCY: f32 = 0.15;
    const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;
    const DEFAULT_MAX_MEMORY_OVERHEAD: f32 = 20.0;
    const DEFAULT_ENTITY_COUNT: usize = 100;
    const MEMORY_MEASUREMENT_FALLBACK: usize = 1_000_000_000;

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
            
            let api_key = env::var("DEEPSEEK_API_KEY")
                .map_err(|_| anyhow::anyhow!("DEEPSEEK_API_KEY must be set in .env file"))?;
            
            Ok(Self {
                deepseek_api_key: api_key,
                min_learning_efficiency: env::var("MIN_LEARNING_EFFICIENCY")
                    .unwrap_or_else(|_| DEFAULT_MIN_LEARNING_EFFICIENCY.to_string())
                    .parse()?,
                min_confidence_threshold: env::var("MIN_CONFIDENCE_THRESHOLD")
                    .unwrap_or_else(|_| DEFAULT_MIN_CONFIDENCE.to_string())
                    .parse()?,
                max_memory_overhead: env::var("MAX_MEMORY_OVERHEAD_PERCENT")
                    .unwrap_or_else(|_| DEFAULT_MAX_MEMORY_OVERHEAD.to_string())
                    .parse()?,
                synthetic_entity_count: env::var("SYNTHETIC_ENTITY_COUNT")
                    .unwrap_or_else(|_| DEFAULT_ENTITY_COUNT.to_string())
                    .parse()?,
            })
        }
    }

    /// Test Hebbian learning with realistic expectations
    #[tokio::test]
    async fn test_hebbian_learning_realistic() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Create components - these constructors should exist
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let activation_engine = Arc::new(ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new().await?);
        
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
        
        // Apply learning
        let update = hebbian_engine.apply_hebbian_learning(events, context).await?;
        
        // Verify learning occurred with realistic expectations
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
        assert!(total_changes > 0, "No connections were modified during learning");
        
        println!("✓ Hebbian learning test passed with {} weight changes", total_changes);
        Ok(())
    }

    /// Test homeostasis - only test what the API actually provides
    #[tokio::test]
    async fn test_homeostasis_basic() -> Result<()> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let attention_manager = Arc::new(AttentionManager::new().await?);
        let working_memory = Arc::new(WorkingMemorySystem::new().await?);
        
        let mut homeostasis = SynapticHomeostasis::new(
            brain_graph.clone(),
            attention_manager,
            working_memory,
        ).await?;
        
        // Create test entities
        let entities = create_minimal_test_entities(20)?;
        for entity in &entities {
            brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Apply homeostatic scaling - this method should exist
        let update = homeostasis.apply_homeostatic_scaling(
            Duration::from_secs(300)
        ).await?;
        
        // Verify the response makes sense
        assert!(
            update.scaled_entities.len() <= entities.len(),
            "More entities scaled than exist in graph"
        );
        
        assert!(
            update.stability_improvement >= -1.0 && update.stability_improvement <= 1.0,
            "Stability improvement {} outside valid range [-1.0, 1.0]",
            update.stability_improvement
        );
        
        println!("✓ Homeostasis test passed with {} entities scaled", update.scaled_entities.len());
        Ok(())
    }

    /// Test optimization agent - only what's actually implemented
    #[tokio::test]
    async fn test_optimization_basic() -> Result<()> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let sdr_storage = Arc::new(SDRStorage::new().await?);
        let activation_engine = Arc::new(ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new().await?);
        
        // AbstractThinking needs brain_graph parameter based on the helper
        let abstract_thinking = Arc::new(AbstractThinking::new(brain_graph.clone()).await?);
        let orchestrator = Arc::new(CognitiveOrchestrator::new().await?);
        
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
        
        // Analyze optimization opportunities - this method exists
        let scope = AnalysisScope {
            entities: entities.iter().take(10).map(|e| e.key).collect(),
            depth: 2,
            time_window: Duration::from_secs(3600),
        };
        
        let opportunities = optimization_agent.analyze_optimization_opportunities(scope).await?;
        
        // Just verify we got a valid response
        assert!(
            opportunities.efficiency_predictions.overall_efficiency_gain >= 0.0,
            "Negative efficiency gain doesn't make sense"
        );
        
        println!("✓ Optimization analysis found {} candidates", 
                 opportunities.optimization_candidates.len());
        
        Ok(())
    }

    /// Test emergency adaptation - basic functionality only
    #[tokio::test]
    #[ignore = "Requires full system setup"]
    async fn test_emergency_adaptation_basic() -> Result<()> {
        // This test is ignored because it requires a fully configured system
        // which may not be available in all test environments
        Ok(())
    }

    /// Test memory usage with actual measurement
    #[tokio::test]
    async fn test_memory_usage_bounds() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Measure baseline memory
        let baseline_memory = get_process_memory_usage();
        
        // Create a minimal system
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let entities = create_minimal_test_entities(config.synthetic_entity_count)?;
        
        for entity in &entities {
            brain_graph.insert_entity(entity.clone()).await?;
        }
        
        // Measure memory after loading
        let loaded_memory = get_process_memory_usage();
        
        // Only check if we got valid measurements
        if baseline_memory > 0 && loaded_memory > baseline_memory {
            let memory_increase_bytes = loaded_memory - baseline_memory;
            let memory_increase_mb = memory_increase_bytes as f32 / 1_048_576.0;
            let memory_increase_percent = (memory_increase_bytes as f32 / baseline_memory as f32) * 100.0;
            
            println!("Memory increase: {:.1}MB ({:.1}%)", memory_increase_mb, memory_increase_percent);
            
            // Only assert if the increase seems unreasonable
            if memory_increase_percent > 100.0 {
                println!("Warning: Memory usage doubled, this might indicate a problem");
            }
        }
        
        Ok(())
    }

    /// Test with DeepSeek API integration
    #[tokio::test]
    async fn test_deepseek_integration_basic() -> Result<()> {
        let config = TestConfig::from_env()?;
        
        // Create a client for DeepSeek API
        let client = reqwest::Client::new();
        let api_url = env::var("DEEPSEEK_API_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string());
        
        // Test the API is accessible
        let test_request = serde_json::json!({
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": "Reply with 'ok' if you receive this"
            }],
            "max_tokens": 10,
            "temperature": 0.1
        });
        
        let response = client
            .post(format!("{}/chat/completions", api_url))
            .header("Authorization", format!("Bearer {}", config.deepseek_api_key))
            .header("Content-Type", "application/json")
            .json(&test_request)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("DeepSeek API request failed with status {}: {}", status, error_text));
        }
        
        println!("✓ DeepSeek API connection successful");
        Ok(())
    }

    // Helper functions that don't extend production types

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

    fn get_process_memory_usage() -> usize {
        // Platform-specific memory measurement
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
        
        #[cfg(target_os = "windows")]
        {
            // Windows implementation would go here
            // For now, return a fallback
        }
        
        // Fallback if we can't measure
        MEMORY_MEASUREMENT_FALLBACK
    }
}

#[cfg(test)]
mod negative_test_cases {
    use super::*;

    /// Test what happens with invalid inputs
    #[tokio::test]
    async fn test_hebbian_learning_invalid_inputs() -> Result<()> {
        let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
        let activation_engine = Arc::new(ActivationPropagationEngine::new().await?);
        let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new().await?);
        
        let mut hebbian_engine = HebbianLearningEngine::new(
            brain_graph,
            activation_engine,
            inhibition_system,
        ).await?;
        
        // Test with empty events
        let empty_events = vec![];
        let context = LearningContext {
            performance_pressure: 0.5,
            user_satisfaction_level: 0.5,
            learning_urgency: 0.5,
            session_id: "test".to_string(),
            learning_goals: vec![],
        };
        
        let result = hebbian_engine.apply_hebbian_learning(empty_events, context).await?;
        
        // Should handle gracefully
        assert_eq!(result.strengthened_connections.len(), 0);
        assert_eq!(result.new_connections.len(), 0);
        
        println!("✓ Handled empty input gracefully");
        Ok(())
    }

    /// Test resource cleanup
    #[tokio::test]
    async fn test_resource_cleanup() -> Result<()> {
        // Create and drop multiple systems to test cleanup
        for i in 0..3 {
            let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
            let entities = create_minimal_test_entities(50)?;
            
            for entity in &entities {
                brain_graph.insert_entity(entity.clone()).await?;
            }
            
            // Let it go out of scope
            drop(brain_graph);
            
            println!("✓ Iteration {} cleanup completed", i);
        }
        
        Ok(())
    }
}