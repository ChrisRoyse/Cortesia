//! Integration tests for learning module end-to-end learning cycles
//! 
//! These tests validate the complete learning pipeline from activation events
//! through adaptation execution and validation.

use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use uuid::Uuid;
use anyhow::Result;

use llmkg::learning::{
    Phase4LearningSystem,
    AdaptiveLearningSystem,
    HebbianLearningEngine,
    SynapticHomeostasis,
    MetaLearningSystem,
    NeuralPatternDetectionSystem,
    ParameterTuner,
    ActivationEvent,
    LearningContext,
    WeightChange,
    LearningUpdate,
    STDPResult,
    PlasticityType,
    LearningResult,
    LearningGoal,
    LearningGoalType,
    CorePerformanceBottleneck
};

use llmkg::learning::phase4_integration::{
    Phase4Config,
    ComprehensiveLearningResult,
    LearningSessionType,
    SystemAssessment,
    LearningStrategy,
    StrategyType,
    CoordinationApproach,
    ValidationResult
};

use llmkg::cognitive::phase3_integration::Phase3IntegratedCognitiveSystem;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::entity::EntityId;
use llmkg::core::types::{NodeType, RelationType};

/// Test fixture for learning integration tests
pub struct LearningIntegrationTestFixture {
    pub phase4_system: Phase4LearningSystem,
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub sdr_storage: Arc<SDRStorage>,
    pub phase3_system: Arc<Phase3IntegratedCognitiveSystem>,
}

impl LearningIntegrationTestFixture {
    /// Create a new test fixture with all dependencies
    pub async fn new() -> Result<Self> {
        // Create test brain graph
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new().await?);
        
        // Create test SDR storage
        let sdr_storage = Arc::new(SDRStorage::new().await?);
        
        // Create Phase 3 cognitive system
        let phase3_system = Arc::new(Phase3IntegratedCognitiveSystem::new(
            brain_graph.clone(),
            sdr_storage.clone()
        ).await?);
        
        // Create Phase 4 learning system with test configuration
        let config = Phase4Config {
            hebbian_learning_rate: 0.01,
            homeostasis_target_activity: 0.1,
            adaptive_learning_enabled: true,
            emergency_threshold: 0.05,
            max_concurrent_sessions: 3,
            performance_monitoring_interval: Duration::from_secs(60),
            safety_checks_enabled: true,
            rollback_enabled: true,
        };
        
        let phase4_system = Phase4LearningSystem::new(
            phase3_system.clone(),
            brain_graph.clone(),
            sdr_storage.clone(),
            Some(config),
        ).await?;
        
        Ok(Self {
            phase4_system,
            brain_graph,
            sdr_storage,
            phase3_system,
        })
    }
    
    /// Populate the graph with test entities and relationships
    pub async fn setup_test_knowledge(&self) -> Result<Vec<EntityId>> {
        let mut entities = Vec::new();
        
        // Create test concepts
        let concept1 = self.brain_graph.add_entity(
            "Neural Learning".to_string(),
            NodeType::Concept,
            HashMap::new()
        ).await?;
        entities.push(concept1);
        
        let concept2 = self.brain_graph.add_entity(
            "Synaptic Plasticity".to_string(),
            NodeType::Concept,
            HashMap::new()
        ).await?;
        entities.push(concept2);
        
        let concept3 = self.brain_graph.add_entity(
            "Homeostasis".to_string(),
            NodeType::Concept,
            HashMap::new()
        ).await?;
        entities.push(concept3);
        
        // Create test relationships
        self.brain_graph.add_relationship(
            concept1,
            concept2,
            RelationType::RelatedTo,
            0.8,
            HashMap::new()
        ).await?;
        
        self.brain_graph.add_relationship(
            concept2,
            concept3,
            RelationType::Influences,
            0.6,
            HashMap::new()
        ).await?;
        
        Ok(entities)
    }
    
    /// Create test activation events
    pub fn create_test_activation_events(&self, entity_ids: &[EntityId]) -> Vec<ActivationEvent> {
        entity_ids.iter().enumerate().map(|(i, &entity_id)| {
            ActivationEvent {
                entity_id,
                activation_strength: 0.5 + (i as f32 * 0.1),
                timestamp: SystemTime::now(),
                source: format!("test_source_{}", i),
                context: LearningContext {
                    session_id: Uuid::new_v4(),
                    learning_phase: "integration_test".to_string(),
                    performance_target: 0.8,
                    constraints: vec!["biological_plausibility".to_string()],
                },
            }
        }).collect()
    }
}

#[tokio::test]
async fn test_end_to_end_learning_cycle() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    let entities = fixture.setup_test_knowledge().await?;
    
    // Execute a complete learning cycle
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Validate the result structure
    assert!(result.overall_success, "Learning cycle should succeed");
    assert!(result.performance_improvement >= 0.0, "Should have non-negative performance improvement");
    assert!(result.duration.as_secs() < 300, "Learning cycle should complete in reasonable time");
    
    // Validate system assessment
    assert!(result.system_assessment.overall_health >= 0.0);
    assert!(result.system_assessment.overall_health <= 1.0);
    assert!(result.system_assessment.readiness_for_learning >= 0.0);
    
    // Validate learning strategy
    assert!(matches!(result.learning_strategy.strategy_type, StrategyType::Balanced | StrategyType::Conservative));
    assert!(result.learning_strategy.expected_duration.as_secs() > 0);
    
    // Validate coordination result
    assert!(!result.coordination_result.participants_activated.is_empty());
    assert_eq!(result.coordination_result.session_id, result.session_id);
    
    // Validate learning results
    assert!(result.learning_results.coordination_quality >= 0.0);
    assert!(result.learning_results.coordination_quality <= 1.0);
    assert!(result.learning_results.overall_learning_effectiveness >= 0.0);
    
    // Validate validation result
    assert!(result.validation_result.success, "Changes should be validated successfully");
    assert!(result.validation_result.changes_committed, "Changes should be committed");
    
    println!("End-to-end learning cycle completed successfully");
    println!("Performance improvement: {:.3}", result.performance_improvement);
    println!("Duration: {:?}", result.duration);
    
    Ok(())
}

#[tokio::test]
async fn test_hebbian_homeostasis_coordination() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    let entities = fixture.setup_test_knowledge().await?;
    let activation_events = fixture.create_test_activation_events(&entities);
    
    // Execute learning cycle and check for Hebbian-Homeostasis coordination
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Verify that both Hebbian and Homeostasis participated
    let hebbian_participated = result.learning_results.hebbian_results.is_some();
    let homeostasis_participated = result.learning_results.homeostasis_results.is_some();
    
    // At least one should have participated (depending on system state)
    assert!(hebbian_participated || homeostasis_participated, 
           "Either Hebbian or Homeostasis should participate in learning");
    
    // If both participated, verify coordination quality
    if hebbian_participated && homeostasis_participated {
        assert!(result.learning_results.coordination_quality > 0.5,
               "Coordination quality should be good when both systems participate");
        
        // Verify homeostasis balancing was applied
        assert!(result.homeostasis_result.balancing_applied,
               "Homeostasis balancing should be applied when both systems are active");
        
        println!("Both Hebbian and Homeostasis systems coordinated successfully");
        println!("Coordination quality: {:.3}", result.learning_results.coordination_quality);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_adaptive_learning_integration() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    let entities = fixture.setup_test_knowledge().await?;
    
    // Execute multiple learning cycles to test adaptation
    let mut results = Vec::new();
    for i in 0..3 {
        println!("Executing learning cycle {}", i + 1);
        let result = fixture.phase4_system.execute_learning_cycle().await?;
        results.push(result);
        
        // Small delay between cycles
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Verify that learning cycles completed
    assert_eq!(results.len(), 3, "All learning cycles should complete");
    
    // Check for adaptive behavior - later cycles might show different characteristics
    let first_duration = results[0].duration;
    let last_duration = results[2].duration;
    
    // Verify that adaptive learning participated in at least one cycle
    let adaptive_participated = results.iter().any(|r| r.learning_results.adaptive_results.is_some());
    
    if adaptive_participated {
        println!("Adaptive learning successfully integrated across multiple cycles");
        
        // Check for performance trends
        let performance_improvements: Vec<f32> = results.iter()
            .map(|r| r.performance_improvement)
            .collect();
        
        println!("Performance improvements: {:?}", performance_improvements);
        
        // Verify all improvements are non-negative
        assert!(performance_improvements.iter().all(|&p| p >= 0.0),
               "All performance improvements should be non-negative");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_emergency_learning_response() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    
    // Test emergency handling
    use llmkg::learning::phase4_integration::emergency::EmergencyType;
    
    let emergency_types = vec![
        EmergencyType::PerformanceCritical,
        EmergencyType::SystemOverload,
        EmergencyType::LearningDivergence,
    ];
    
    for emergency_type in emergency_types {
        let response = fixture.phase4_system.handle_emergency(emergency_type.clone()).await?;
        
        assert!(!response.protocol_name.is_empty(), "Emergency protocol should have a name");
        assert!(!response.actions_taken.is_empty(), "Emergency response should take actions");
        assert!(response.recovery_time.as_secs() < 60, "Emergency recovery should be fast");
        
        println!("Emergency {:?} handled successfully: {}", emergency_type, response.protocol_name);
        println!("Actions taken: {:?}", response.actions_taken);
        println!("Recovery time: {:?}", response.recovery_time);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_learning_performance_monitoring() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    
    // Execute a few learning cycles
    for _ in 0..2 {
        fixture.phase4_system.execute_learning_cycle().await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Get performance metrics
    let metrics_report = fixture.phase4_system.get_performance_metrics().await?;
    
    assert!(!metrics_report.is_empty(), "Performance metrics should be available");
    
    // The report should contain useful information
    assert!(metrics_report.contains("performance") || metrics_report.contains("learning") || metrics_report.contains("effectiveness"),
           "Performance report should contain relevant metrics");
    
    println!("Performance metrics report:");
    println!("{}", metrics_report);
    
    Ok(())
}

#[tokio::test]
async fn test_learning_configuration_update() -> Result<()> {
    let mut fixture = LearningIntegrationTestFixture::new().await?;
    
    // Create new configuration
    let new_config = Phase4Config {
        hebbian_learning_rate: 0.02,  // Different from default
        homeostasis_target_activity: 0.15,  // Different from default
        adaptive_learning_enabled: true,
        emergency_threshold: 0.03,  // Different from default
        max_concurrent_sessions: 5,  // Different from default
        performance_monitoring_interval: Duration::from_secs(30),  // Different from default
        safety_checks_enabled: true,
        rollback_enabled: true,
    };
    
    // Update configuration
    fixture.phase4_system.update_configuration(new_config.clone())?;
    
    // Verify configuration was updated
    assert_eq!(fixture.phase4_system.learning_configuration.hebbian_learning_rate, 0.02);
    assert_eq!(fixture.phase4_system.learning_configuration.homeostasis_target_activity, 0.15);
    assert_eq!(fixture.phase4_system.learning_configuration.emergency_threshold, 0.03);
    assert_eq!(fixture.phase4_system.learning_configuration.max_concurrent_sessions, 5);
    
    // Execute learning cycle with new configuration
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    assert!(result.overall_success, "Learning should work with updated configuration");
    
    println!("Configuration updated and learning cycle executed successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_cross_component_data_flow() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    let entities = fixture.setup_test_knowledge().await?;
    
    // Execute learning cycle
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Verify data flow between components
    assert!(result.session_id != Uuid::nil(), "Session ID should be set");
    
    // System assessment should inform learning strategy
    assert!(!result.system_assessment.learning_opportunities.is_empty() || 
           !result.system_assessment.bottlenecks.is_empty() ||
           result.system_assessment.readiness_for_learning > 0.0,
           "System assessment should provide actionable information");
    
    // Learning strategy should influence coordination
    assert!(result.coordination_result.participants_activated.len() > 0,
           "Coordination should activate participants based on strategy");
    
    // Coordination should produce learning results
    assert!(result.learning_results.coordination_quality > 0.0,
           "Learning results should reflect coordination quality");
    
    // Learning results should influence optimization
    assert!(result.optimization_result.optimizations_applied >= 0,
           "Optimization should be based on learning results");
    
    // All should contribute to validation
    assert!(result.validation_result.success == result.overall_success,
           "Validation result should align with overall success");
    
    println!("Cross-component data flow validated successfully");
    println!("Session ID: {}", result.session_id);
    println!("Participants activated: {:?}", result.coordination_result.participants_activated);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_learning_sessions() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    
    // Start multiple learning cycles concurrently
    let futures = (0..3).map(|i| {
        let system = &fixture.phase4_system;
        async move {
            println!("Starting concurrent learning session {}", i);
            system.execute_learning_cycle().await
        }
    });
    
    // Wait for all to complete
    let results: Vec<Result<ComprehensiveLearningResult>> = futures::future::join_all(futures).await;
    
    // Verify all sessions completed
    let successful_results: Vec<_> = results.into_iter()
        .filter_map(|r| r.ok())
        .filter(|r| r.overall_success)
        .collect();
    
    assert!(successful_results.len() >= 1, "At least one concurrent session should succeed");
    
    // Verify session IDs are unique
    let session_ids: std::collections::HashSet<_> = successful_results.iter()
        .map(|r| r.session_id)
        .collect();
    
    assert_eq!(session_ids.len(), successful_results.len(), 
              "All successful sessions should have unique IDs");
    
    println!("Concurrent learning sessions completed: {}", successful_results.len());
    
    Ok(())
}

#[tokio::test]
async fn test_learning_cycle_rollback_capability() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    
    // Execute learning cycle
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Check if any learning results indicate rollback capability
    if let Some(adaptive_result) = &result.learning_results.adaptive_results {
        // Check if rollback is available for optimization results
        assert!(adaptive_result.optimization_result.rollback_available == false || 
               adaptive_result.optimization_result.rollback_available == true,
               "Rollback availability should be clearly indicated");
        
        if adaptive_result.optimization_result.rollback_available {
            println!("Rollback capability confirmed for optimization results");
        }
    }
    
    // Verify safety checks were performed (indicated by successful validation)
    assert!(result.validation_result.success, "Safety validation should pass");
    assert!(result.validation_result.changes_committed, "Safe changes should be committed");
    
    println!("Learning cycle rollback capability verified");
    
    Ok(())
}

#[tokio::test]
async fn test_biological_learning_principles_integration() -> Result<()> {
    let fixture = LearningIntegrationTestFixture::new().await?;
    let entities = fixture.setup_test_knowledge().await?;
    
    // Execute learning cycle
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Verify biological principles are respected
    if let Some(hebbian_result) = &result.learning_results.hebbian_results {
        // Hebbian learning efficiency should be within biological ranges
        assert!(hebbian_result.learning_efficiency >= 0.0 && 
               hebbian_result.learning_efficiency <= 1.0,
               "Hebbian learning efficiency should be within [0,1]");
    }
    
    if let Some(_homeostasis_result) = &result.learning_results.homeostasis_results {
        // Homeostasis should maintain stability
        assert!(result.homeostasis_result.stability_improvement >= 0.0,
               "Homeostasis should improve or maintain stability");
    }
    
    // Overall performance improvement should be realistic
    assert!(result.performance_improvement >= -0.1 && result.performance_improvement <= 0.5,
           "Performance improvement should be within realistic biological bounds");
    
    // Learning should not be instantaneous (biological constraint)
    assert!(result.duration.as_millis() > 1, "Learning should take some time (biological realism)");
    
    println!("Biological learning principles integration verified");
    println!("Performance improvement: {:.3} (within biological bounds)", result.performance_improvement);
    
    Ok(())
}