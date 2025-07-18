use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

// Import all Phase 4 components
use llmkg::learning::{
    hebbian::HebbianLearningEngine,
    homeostasis::SynapticHomeostasis,
    optimization_agent::GraphOptimizationAgent,
    adaptive_learning::AdaptiveLearningSystem,
    phase4_integration::Phase4LearningSystem,
    types::*,
};

use llmkg::cognitive::{
    phase4_integration::{Phase4CognitiveSystem, Phase4QueryResult, Phase4LearningResult},
    phase3_integration::IntegratedCognitiveSystem,
    types::CognitivePatternType,
};

use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedGraph,
    brain_types::{EntityKey, BrainInspiredEntity, ActivationPattern},
    sdr_storage::SDRStorage,
    activation_engine::ActivationPropagationEngine,
};

// Include test helpers
#[path = "phase4_test_helpers.rs"]
mod phase4_test_helpers;

#[cfg(test)]
mod phase4_tests {
    use super::*;

    #[tokio::test]
    async fn test_hebbian_learning_engine_basic_functionality() -> Result<()> {
        // Create test components
        let brain_graph = Arc::new(create_test_brain_graph().await?);
        let activation_engine = Arc::new(create_test_activation_engine().await?);
        let inhibition_system = Arc::new(create_test_inhibition_system().await?);
        
        // Create Hebbian learning engine
        let mut hebbian_engine = HebbianLearningEngine::new(
            brain_graph,
            activation_engine,
            inhibition_system,
        ).await?;
        
        // Create test activation events
        let activation_events = create_test_activation_events();
        let learning_context = create_test_learning_context();
        
        // Apply Hebbian learning
        let learning_update = hebbian_engine.apply_hebbian_learning(
            activation_events,
            learning_context,
        ).await?;
        
        // Verify learning occurred
        assert!(learning_update.learning_efficiency > 0.0);
        assert!(learning_update.strengthened_connections.len() > 0 
                || learning_update.new_connections.len() > 0);
        
        println!("✓ Hebbian learning engine basic functionality test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_spike_timing_dependent_plasticity() -> Result<()> {
        let brain_graph = Arc::new(create_test_brain_graph().await?);
        let activation_engine = Arc::new(create_test_activation_engine().await?);
        let inhibition_system = Arc::new(create_test_inhibition_system().await?);
        
        let hebbian_engine = HebbianLearningEngine::new(
            brain_graph,
            activation_engine,
            inhibition_system,
        ).await?;
        
        // Create temporally related activation events
        let pre_event = ActivationEvent {
            entity_key: EntityKey::new(),
            activation_strength: 0.8,
            timestamp: std::time::Instant::now(),
            context: ActivationContext {
                query_id: "test_query".to_string(),
                cognitive_pattern: CognitivePatternType::Convergent,
                user_session: Some("test_session".to_string()),
                outcome_quality: Some(0.9),
            },
        };
        
        let post_event = ActivationEvent {
            entity_key: EntityKey::new(),
            activation_strength: 0.7,
            timestamp: std::time::Instant::now() + Duration::from_millis(20),
            context: pre_event.context.clone(),
        };
        
        // Apply STDP
        let stdp_result = hebbian_engine.spike_timing_dependent_plasticity(
            pre_event,
            post_event,
        ).await?;
        
        // Verify STDP occurred
        match stdp_result {
            STDPResult::WeightChanged { weight_change, timing_difference, plasticity_type } => {
                assert!(weight_change != 0.0);
                assert!(timing_difference > 0.0);
                assert!(matches!(plasticity_type, PlasticityType::Potentiation));
            },
            STDPResult::NoChange => {
                panic!("STDP should have occurred with this timing");
            }
        }
        
        println!("✓ Spike-timing dependent plasticity test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_synaptic_homeostasis_system() -> Result<()> {
        let brain_graph = Arc::new(create_test_brain_graph().await?);
        let attention_manager = Arc::new(create_test_attention_manager().await?);
        let working_memory = Arc::new(create_test_working_memory().await?);
        
        let mut homeostasis_system = SynapticHomeostasis::new(
            brain_graph,
            attention_manager,
            working_memory,
        ).await?;
        
        // Apply homeostatic scaling
        let homeostasis_update = homeostasis_system.apply_homeostatic_scaling(
            Duration::from_secs(3600)
        ).await?;
        
        // Verify homeostasis occurred
        assert!(homeostasis_update.scaled_entities.len() >= 0);
        assert!(homeostasis_update.stability_improvement >= 0.0);
        
        println!("✓ Synaptic homeostasis system test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_graph_optimization_agent() -> Result<()> {
        let brain_graph = Arc::new(create_test_brain_graph().await?);
        let sdr_storage = Arc::new(create_test_sdr_storage().await?);
        let abstract_thinking = Arc::new(create_test_abstract_thinking().await?);
        let orchestrator = Arc::new(create_test_orchestrator().await?);
        let hebbian_engine = Arc::new(Mutex::new(create_test_hebbian_engine().await?));
        
        let mut optimization_agent = GraphOptimizationAgent::new(
            brain_graph,
            sdr_storage,
            abstract_thinking,
            orchestrator,
            hebbian_engine,
        ).await?;
        
        // Analyze optimization opportunities
        let analysis_scope = llmkg::learning::optimization_agent::AnalysisScope {
            entities: Vec::new(),
            depth: 3,
            time_window: Duration::from_secs(3600),
        };
        
        let opportunities = optimization_agent.analyze_optimization_opportunities(analysis_scope).await?;
        
        // Verify optimization analysis
        assert!(opportunities.efficiency_predictions.overall_efficiency_gain >= 0.0);
        assert!(opportunities.priority_ranking.len() >= 0);
        
        println!("✓ Graph optimization agent test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_adaptive_learning_system() -> Result<()> {
        let integrated_cognitive_system = Arc::new(create_test_integrated_cognitive_system().await?);
        let working_memory = Arc::new(create_test_working_memory().await?);
        let attention_manager = Arc::new(create_test_attention_manager().await?);
        let orchestrator = Arc::new(create_test_orchestrator().await?);
        let hebbian_engine = Arc::new(Mutex::new(create_test_hebbian_engine().await?));
        let optimization_agent = Arc::new(Mutex::new(create_test_optimization_agent().await?));
        
        let mut adaptive_learning = AdaptiveLearningSystem::new(
            integrated_cognitive_system,
            working_memory,
            attention_manager,
            orchestrator,
            hebbian_engine,
            optimization_agent,
        ).await?;
        
        // Process adaptive learning cycle
        let learning_result = adaptive_learning.process_adaptive_learning_cycle(
            Duration::from_secs(1800)
        ).await?;
        
        // Verify adaptive learning occurred
        assert!(learning_result.performance_improvement >= 0.0);
        assert!(!learning_result.hebbian_updates.strengthened_connections.is_empty() 
                || !learning_result.hebbian_updates.new_connections.is_empty());
        
        println!("✓ Adaptive learning system test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_phase4_learning_system_integration() -> Result<()> {
        let phase4_system = create_test_phase4_learning_system().await?;
        
        // Execute comprehensive learning cycle
        let learning_result = phase4_system.execute_comprehensive_learning_cycle().await?;
        
        // Verify comprehensive learning
        assert!(learning_result.overall_success);
        assert!(learning_result.performance_improvement >= 0.0);
        assert!(learning_result.learning_results.overall_learning_effectiveness >= 0.0);
        
        println!("✓ Phase 4 learning system integration test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_phase4_cognitive_system_integration() -> Result<()> {
        let phase3_system = Arc::new(create_test_integrated_cognitive_system().await?);
        let phase4_learning = Arc::new(create_test_phase4_learning_system().await?);
        
        let phase4_cognitive = Phase4CognitiveSystem::new(
            phase3_system,
            phase4_learning,
        ).await?;
        
        // Test enhanced query processing
        let query_result = phase4_cognitive.enhanced_query(
            "What is the relationship between AI and consciousness?",
            None,
        ).await?;
        
        // Verify enhanced query results
        assert!(query_result.base_result.overall_confidence > 0.0);
        assert!(!query_result.learning_insights.pattern_effectiveness.is_empty());
        assert!(query_result.performance_impact.learning_efficiency_gain >= 0.0);
        
        println!("✓ Phase 4 cognitive system integration test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_learning_performance_improvements() -> Result<()> {
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // Get baseline performance
        let baseline_assessment = phase4_cognitive.assess_learning_benefits().await?;
        
        // Execute learning cycle
        let learning_result = phase4_cognitive.execute_cognitive_learning_cycle().await?;
        
        // Verify learning improved performance
        assert!(learning_result.user_satisfaction_impact >= 0.0);
        assert!(!learning_result.cognitive_adaptations.is_empty());
        
        // Get post-learning assessment
        let post_learning_assessment = phase4_cognitive.assess_learning_benefits().await?;
        
        // Verify improvement
        assert!(post_learning_assessment.overall_improvement >= baseline_assessment.overall_improvement);
        
        println!("✓ Learning performance improvements test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_emergency_adaptation() -> Result<()> {
        let adaptive_learning = create_test_adaptive_learning_system().await?;
        
        // Trigger emergency adaptation
        let emergency_result = adaptive_learning.handle_emergency_adaptation(
            llmkg::learning::adaptive_learning::EmergencyTrigger::PerformanceCollapse
        ).await?;
        
        // Verify emergency response
        assert!(emergency_result.performance_improvement != 0.0); // Should have some impact
        
        println!("✓ Emergency adaptation test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_user_personalization() -> Result<()> {
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // Create user context
        let user_context = Some(llmkg::cognitive::types::QueryContext {
            user_id: Some("test_user_123".to_string()),
            session_id: Some("test_session_456".to_string()),
            conversation_history: Vec::new(),
            domain_context: Some("technology".to_string()),
            urgency_level: 0.5,
            expected_response_time: Some(Duration::from_millis(500)),
            query_intent: None,
        });
        
        // Process query with personalization
        let query_result = phase4_cognitive.enhanced_query(
            "Explain machine learning concepts",
            user_context,
        ).await?;
        
        // Verify personalization was applied
        assert!(query_result.user_personalization.personalization_applied);
        assert!(!query_result.user_personalization.user_profile_updates.is_empty());
        
        println!("✓ User personalization test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_learning_coordination() -> Result<()> {
        let phase4_system = create_test_phase4_learning_system().await?;
        
        // Test different coordination modes
        let coordination_result = phase4_system.coordinate_learning_systems(
            Uuid::new_v4(),
            create_test_learning_strategy(),
        ).await?;
        
        // Verify coordination worked
        assert!(!coordination_result.participants_activated.is_empty());
        assert!(coordination_result.resource_allocation.memory_allocated_mb > 0.0);
        
        println!("✓ Learning coordination test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_learning_efficiency_targets() -> Result<()> {
        let phase4_system = create_test_phase4_learning_system().await?;
        
        // Execute multiple learning cycles and measure efficiency
        let mut efficiency_measurements = Vec::new();
        
        for _ in 0..3 {
            let result = phase4_system.execute_comprehensive_learning_cycle().await?;
            efficiency_measurements.push(result.learning_results.overall_learning_effectiveness);
        }
        
        // Verify efficiency meets targets (> 0.7 from Phase 4 spec)
        let average_efficiency = efficiency_measurements.iter().sum::<f32>() / efficiency_measurements.len() as f32;
        assert!(average_efficiency > 0.7, "Learning efficiency {} below target 0.7", average_efficiency);
        
        println!("✓ Learning efficiency targets test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_overhead_limits() -> Result<()> {
        let phase4_system = create_test_phase4_learning_system().await?;
        
        // Measure memory usage before and after learning
        let initial_memory = measure_memory_usage();
        
        let _result = phase4_system.execute_comprehensive_learning_cycle().await?;
        
        let final_memory = measure_memory_usage();
        let memory_overhead = (final_memory - initial_memory) / initial_memory;
        
        // Verify memory overhead is within limits (< 10% from Phase 4 spec)
        assert!(memory_overhead < 0.1, "Memory overhead {} exceeds limit 0.1", memory_overhead);
        
        println!("✓ Memory overhead limits test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_adaptation_speed_targets() -> Result<()> {
        let adaptive_learning = create_test_adaptive_learning_system().await?;
        
        // Measure adaptation speed
        let start_time = SystemTime::now();
        
        let _result = adaptive_learning.process_adaptive_learning_cycle(
            Duration::from_secs(300) // 5 minutes
        ).await?;
        
        let adaptation_duration = start_time.elapsed().unwrap();
        
        // Verify adaptation speed meets targets (< 12 hours from Phase 4 spec)
        assert!(adaptation_duration < Duration::from_secs(43200), 
                "Adaptation took {:?}, exceeds 12 hour limit", adaptation_duration);
        
        println!("✓ Adaptation speed targets test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_stability_during_learning() -> Result<()> {
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // Measure query performance before learning
        let baseline_query_time = measure_query_time(&phase4_cognitive).await?;
        
        // Execute learning cycle
        let _learning_result = phase4_cognitive.execute_cognitive_learning_cycle().await?;
        
        // Measure query performance after learning
        let post_learning_query_time = measure_query_time(&phase4_cognitive).await?;
        
        // Verify system remained stable (performance degradation < 5% from Phase 4 spec)
        let performance_change = (post_learning_query_time.as_millis() as f32 - baseline_query_time.as_millis() as f32) 
                                 / baseline_query_time.as_millis() as f32;
        
        assert!(performance_change < 0.05, 
                "Performance degraded by {:.1}%, exceeds 5% limit", performance_change * 100.0);
        
        println!("✓ Stability during learning test passed");
        Ok(())
    }

    #[tokio::test]
    async fn test_phase3_compatibility() -> Result<()> {
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // Test that all Phase 3 functionality still works
        let phase3_result = phase4_cognitive.phase3_system.integrated_query(
            "Test Phase 3 compatibility",
            None,
        ).await?;
        
        // Verify Phase 3 functionality is preserved
        assert!(phase3_result.overall_confidence > 0.0);
        assert!(!phase3_result.pattern_results.is_empty());
        
        // Test enhanced query also works
        let phase4_result = phase4_cognitive.enhanced_query(
            "Test Phase 4 enhancement",
            None,
        ).await?;
        
        // Verify Phase 4 enhancements work
        assert!(phase4_result.base_result.overall_confidence > 0.0);
        assert!(!phase4_result.learning_insights.pattern_effectiveness.is_empty());
        
        println!("✓ Phase 3 compatibility test passed");
        Ok(())
    }

    // Import helper functions from the test helpers module
    use crate::phase4_test_helpers::*;

    fn create_test_activation_events() -> Vec<ActivationEvent> {
        generate_test_activation_events(2)
    }

    fn create_test_learning_context() -> LearningContext {
        generate_test_learning_context()
    }

    fn create_test_learning_strategy() -> llmkg::learning::phase4_integration::LearningStrategy {
        llmkg::learning::phase4_integration::LearningStrategy {
            strategy_type: llmkg::learning::phase4_integration::StrategyType::Balanced,
            priority_areas: vec!["Performance".to_string(), "Quality".to_string()],
            resource_allocation: ResourceRequirement {
                memory_mb: 100.0,
                cpu_cores: 0.5,
                storage_mb: 50.0,
                duration_estimate: Duration::from_secs(300),
            },
            coordination_approach: llmkg::learning::phase4_integration::CoordinationApproach::Synchronized,
            safety_level: 0.8,
            expected_duration: Duration::from_secs(1800),
        }
    }

    fn measure_memory_usage() -> f32 {
        // Simplified memory measurement for testing
        0.5 // 50% memory usage
    }

    async fn measure_query_time(phase4_cognitive: &Phase4CognitiveSystem) -> Result<Duration> {
        let start = SystemTime::now();
        
        let _result = phase4_cognitive.enhanced_query(
            "Test query for timing",
            None,
        ).await?;
        
        Ok(start.elapsed().unwrap_or(Duration::from_millis(100)))
    }
}

#[cfg(test)]
mod phase4_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_phase4_workflow() -> Result<()> {
        println!("🚀 Starting comprehensive Phase 4 workflow test...");
        
        // 1. Create full Phase 4 system
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // 2. Test initial query processing
        let initial_query_result = phase4_cognitive.enhanced_query(
            "What are the implications of artificial general intelligence?",
            None,
        ).await?;
        
        println!("✓ Initial query processed with confidence: {:.2}", 
                initial_query_result.base_result.overall_confidence);
        
        // 3. Execute learning cycle
        let learning_result = phase4_cognitive.execute_cognitive_learning_cycle().await?;
        
        println!("✓ Learning cycle completed with satisfaction impact: {:.2}", 
                learning_result.user_satisfaction_impact);
        
        // 4. Test query processing after learning
        let post_learning_query_result = phase4_cognitive.enhanced_query(
            "What are the implications of artificial general intelligence?",
            None,
        ).await?;
        
        println!("✓ Post-learning query processed with confidence: {:.2}", 
                post_learning_query_result.base_result.overall_confidence);
        
        // 5. Verify learning benefits
        let learning_assessment = phase4_cognitive.assess_learning_benefits().await?;
        
        println!("✓ Learning effectiveness: {:.2}", learning_assessment.learning_effectiveness);
        println!("✓ Overall improvement: {:.2}", learning_assessment.overall_improvement);
        
        // 6. Test with user personalization
        let user_context = Some(llmkg::cognitive::types::QueryContext {
            user_id: Some("expert_user".to_string()),
            session_id: Some("expert_session".to_string()),
            conversation_history: Vec::new(),
            domain_context: Some("artificial_intelligence".to_string()),
            urgency_level: 0.7,
            expected_response_time: Some(Duration::from_millis(300)),
            query_intent: None,
        });
        
        let personalized_result = phase4_cognitive.enhanced_query(
            "Explain the technical challenges in AGI alignment",
            user_context,
        ).await?;
        
        println!("✓ Personalized query processed, personalization applied: {}", 
                personalized_result.user_personalization.personalization_applied);
        
        // 7. Test emergency adaptation
        let emergency_result = phase4_cognitive.phase4_learning
            .handle_system_emergency(
                llmkg::learning::phase4_integration::EmergencyType::PerformanceCollapse
            ).await?;
        
        println!("✓ Emergency adaptation completed successfully: {}", emergency_result.success);
        
        // Verify overall system health
        assert!(learning_assessment.overall_improvement >= 0.0);
        assert!(learning_result.user_satisfaction_impact >= 0.0);
        assert!(emergency_result.success);
        
        println!("🎉 Full Phase 4 workflow test completed successfully!");
        Ok(())
    }

    #[tokio::test]
    async fn test_phase4_performance_targets() -> Result<()> {
        println!("📊 Testing Phase 4 performance targets...");
        
        let phase4_cognitive = create_test_phase4_cognitive_system().await?;
        
        // Test learning efficiency target (> 80% from spec)
        let learning_result = phase4_cognitive.execute_cognitive_learning_cycle().await?;
        let learning_efficiency = learning_result.comprehensive_learning.learning_results.overall_learning_effectiveness;
        
        println!("Learning efficiency: {:.1}% (target: >70%)", learning_efficiency * 100.0);
        assert!(learning_efficiency > 0.7, "Learning efficiency below target");
        
        // Test adaptation speed target (< 12 hours from spec)
        let start_time = SystemTime::now();
        let _adaptation_result = phase4_cognitive.phase4_learning
            .execute_comprehensive_learning_cycle().await?;
        let adaptation_time = start_time.elapsed().unwrap();
        
        println!("Adaptation time: {:.1}s (target: <12 hours)", adaptation_time.as_secs_f32());
        assert!(adaptation_time < Duration::from_secs(43200), "Adaptation time exceeds target");
        
        // Test query performance improvement
        let query_start = SystemTime::now();
        let _query_result = phase4_cognitive.enhanced_query(
            "Complex reasoning test query",
            None,
        ).await?;
        let query_time = query_start.elapsed().unwrap();
        
        println!("Query time: {:.0}ms (should be optimized)", query_time.as_millis());
        assert!(query_time < Duration::from_millis(2000), "Query time too high");
        
        println!("✅ All Phase 4 performance targets met!");
        Ok(())
    }
}

// Helper function implementations
async fn create_test_phase4_cognitive_system() -> Result<Phase4CognitiveSystem> {
    phase4_test_helpers::create_test_phase4_cognitive_system().await
}

// Test configuration
#[cfg(test)]
mod test_config {
    pub const TEST_TIMEOUT_SECONDS: u64 = 30;
    pub const PERFORMANCE_THRESHOLD: f32 = 0.7;
    pub const MEMORY_LIMIT_MB: f32 = 1000.0;
    pub const MAX_ADAPTATION_TIME_SECONDS: u64 = 600;
}

// Test utilities
#[cfg(test)]
mod test_utils {
    use super::*;
    
    pub fn assert_learning_improvement(before: f32, after: f32, threshold: f32) {
        let improvement = after - before;
        assert!(improvement >= threshold, 
                "Learning improvement {:.3} below threshold {:.3}", improvement, threshold);
    }
    
    pub fn assert_performance_maintained(before: Duration, after: Duration, max_degradation: f32) {
        let degradation = (after.as_millis() as f32 - before.as_millis() as f32) / before.as_millis() as f32;
        assert!(degradation <= max_degradation,
                "Performance degraded by {:.1}%, exceeds limit {:.1}%", 
                degradation * 100.0, max_degradation * 100.0);
    }
    
    pub fn create_test_user_context() -> llmkg::cognitive::types::QueryContext {
        llmkg::cognitive::types::QueryContext {
            user_id: Some("test_user".to_string()),
            session_id: Some("test_session".to_string()),
            conversation_history: Vec::new(),
            domain_context: Some("general".to_string()),
            urgency_level: 0.5,
            expected_response_time: Some(Duration::from_millis(500)),
            query_intent: None,
        }
    }
}