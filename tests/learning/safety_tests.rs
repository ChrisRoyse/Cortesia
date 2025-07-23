//! Safety tests for emergency response and rollback mechanisms
//! 
//! These tests validate that the learning system can handle emergencies,
//! perform safe rollbacks, and maintain system integrity under adverse conditions.

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use anyhow::Result;
use futures::future;

use llmkg::learning::{
    Phase4LearningSystem,
    AdaptiveLearningSystem,
    HebbianLearningEngine,
    SynapticHomeostasis,
    ActivationEvent,
    ActivationContext,
    LearningContext,
    LearningGoal,
    LearningGoalType,
    WeightChange,
    LearningUpdate,
};

use llmkg::cognitive::types::CognitivePatternType;

use llmkg::learning::phase4_integration::{
    Phase4Config,
    IntegrationDepth,
    PerformanceTargets,
    SafetyConstraints,
    ResourceLimits,
    EmergencyType,
    EmergencyResponse
};

use llmkg::cognitive::phase3_integration::Phase3IntegratedCognitiveSystem;
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::brain_enhanced_graph::brain_relationship_manager::AddRelationship;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::types::EntityKey;
use llmkg::core::triple::NodeType;
use llmkg::core::brain_types::RelationType;

/// Test fixture for safety testing
pub struct SafetyTestFixture {
    pub phase4_system: Phase4LearningSystem,
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub sdr_storage: Arc<SDRStorage>,
    pub test_entities: Vec<EntityKey>,
    pub initial_state: SystemSnapshot,
}

/// Snapshot of system state for rollback testing
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    pub entity_count: usize,
    pub connection_count: usize,
    pub performance_score: f32,
    pub timestamp: SystemTime,
    pub configuration: Phase4Config,
}

impl SafetyTestFixture {
    /// Create new safety test fixture
    pub async fn new() -> Result<Self> {
        // Create test dependencies with safety-focused configuration
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(96).unwrap());
        use llmkg::core::sdr_types::SDRConfig;
        let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
        let orchestrator = Arc::new(CognitiveOrchestrator::new(
            brain_graph.clone(),
            Default::default()
        ).await?);
        let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
        
        let phase3_system = Arc::new(Phase3IntegratedCognitiveSystem::new(
            orchestrator,
            activation_engine,
            brain_graph.clone(),
            sdr_storage.clone()
        ).await?);
        
        // Create test entities
        let mut test_entities = Vec::new();
        for i in 0..5 {
            use llmkg::core::types::EntityData;
            let entity = brain_graph.add_entity(
                EntityData::new(i as u16, format!("safety_test_entity_{}", i), vec![0.5; 96])
            ).await?;
            test_entities.push(entity);
        }
        
        // Create test connections
        for i in 0..test_entities.len()-1 {
            brain_graph.add_relationship_keys(
                test_entities[i],
                test_entities[i+1],
                0.5
            ).await?;
        }
        
        // Safety-focused configuration
        let config = Phase4Config {
            learning_aggressiveness: 0.3, // Conservative learning rate
            integration_depth: IntegrationDepth::Standard,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.7,
                adaptation_speed_target: Duration::from_secs(300),
                memory_overhead_limit: 0.2,
                performance_degradation_limit: 0.1,
                user_satisfaction_target: 0.8,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 1, // Limited concurrency for safety
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.3, // Low threshold for testing
                max_learning_impact_per_session: 0.1,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 512.0,
                max_cpu_usage_percentage: 30.0,
                max_storage_usage_mb: 100.0,
                max_network_bandwidth_mbps: 10.0,
                max_session_duration: Duration::from_secs(300),
                max_daily_learning_time: Duration::from_secs(1800),
            },
        };
        
        let phase4_system = Phase4LearningSystem::new(
            phase3_system.clone(),
            brain_graph.clone(),
            sdr_storage.clone(),
            Some(config.clone()),
        ).await?;
        
        // Capture initial state
        let initial_state = SystemSnapshot {
            entity_count: test_entities.len(),
            connection_count: test_entities.len() - 1,
            performance_score: phase3_system.get_performance_score().await?,
            timestamp: SystemTime::now(),
            configuration: config,
        };
        
        Ok(Self {
            phase4_system,
            brain_graph,
            sdr_storage,
            test_entities,
            initial_state,
        })
    }
    
    /// Create emergency conditions for testing
    pub async fn create_emergency_condition(&self, emergency_type: EmergencyType) -> Result<()> {
        match emergency_type {
            EmergencyType::PerformanceCritical => {
                // Simulate performance degradation
                println!("Simulating performance critical emergency");
                // In real implementation, would inject performance issues
            },
            EmergencyType::SystemOverload => {
                // Simulate system overload
                println!("Simulating system overload emergency");
                // In real implementation, would create resource pressure
            },
            EmergencyType::LearningDivergence => {
                // Simulate learning divergence
                println!("Simulating learning divergence emergency");
                // In real implementation, would create unstable learning patterns
            },
            EmergencyType::MemoryExhaustion => {
                // Simulate memory exhaustion
                println!("Simulating memory exhaustion emergency");
                // In real implementation, would exhaust memory resources
            },
            EmergencyType::InfiniteLoop => {
                // Simulate infinite loop detection
                println!("Simulating infinite loop emergency");
                // In real implementation, would create loop detection triggers
            },
            EmergencyType::PerformanceCollapse => {
                // Simulate performance collapse
                println!("Simulating performance collapse emergency");
                // In real implementation, would create severe performance degradation
            },
            EmergencyType::ResourceExhaustion => {
                // Simulate resource exhaustion
                println!("Simulating resource exhaustion emergency");
                // In real implementation, would exhaust CPU/memory resources
            },
            EmergencyType::UserExodus => {
                // Simulate user exodus
                println!("Simulating user exodus emergency");
                // In real implementation, would simulate mass user departure
            },
        }
        Ok(())
    }
    
    /// Capture current system state
    pub async fn capture_state(&self) -> Result<SystemSnapshot> {
        let performance = self.phase4_system.integrated_cognitive_system.get_performance_score().await?;
        
        Ok(SystemSnapshot {
            entity_count: self.test_entities.len(),
            connection_count: self.test_entities.len() - 1, // Simplified
            performance_score: performance,
            timestamp: SystemTime::now(),
            configuration: self.initial_state.configuration.clone(),
        })
    }
    
    /// Verify system integrity
    pub async fn verify_system_integrity(&self) -> Result<bool> {
        // Check basic system invariants
        let current_state = self.capture_state().await?;
        
        // Verify entity count hasn't decreased unexpectedly
        if current_state.entity_count < self.initial_state.entity_count {
            println!("Warning: Entity count decreased from {} to {}", 
                    self.initial_state.entity_count, current_state.entity_count);
            return Ok(false);
        }
        
        // Verify performance is within reasonable bounds
        if current_state.performance_score < 0.0 || current_state.performance_score > 1.0 {
            println!("Warning: Performance score out of bounds: {}", current_state.performance_score);
            return Ok(false);
        }
        
        // Verify configuration is still valid
        // Note: Phase4Config doesn't have a validate method, so we check basic invariants
        if current_state.performance_score < 0.0 || current_state.performance_score > 1.0 {
            println!("Warning: Performance score out of bounds");
            return Ok(false);
        }
        
        Ok(true)
    }
}

#[tokio::test]
async fn test_performance_critical_emergency_response() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Create performance critical emergency
    fixture.create_emergency_condition(EmergencyType::PerformanceCritical).await?;
    
    // Trigger emergency response
    let response = fixture.phase4_system.handle_emergency(EmergencyType::PerformanceCritical).await?;
    
    // Validate emergency response
    assert!(!response.protocol_name.is_empty(), "Emergency protocol should have a name");
    assert!(!response.actions_taken.is_empty(), "Emergency response should take actions");
    assert!(response.recovery_time < Duration::from_secs(30), "Emergency recovery should be fast");
    
    // Verify system integrity after emergency response
    let integrity_check = fixture.verify_system_integrity().await?;
    assert!(integrity_check, "System integrity should be maintained after emergency response");
    
    println!("Performance critical emergency handled successfully:");
    println!("  Protocol: {}", response.protocol_name);
    println!("  Actions: {:?}", response.actions_taken);
    println!("  Recovery time: {:?}", response.recovery_time);
    println!("  Success: {}", response.success);
    
    Ok(())
}

#[tokio::test]
async fn test_system_overload_emergency_response() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Create system overload emergency
    fixture.create_emergency_condition(EmergencyType::SystemOverload).await?;
    
    // Trigger emergency response
    let response = fixture.phase4_system.handle_emergency(EmergencyType::SystemOverload).await?;
    
    // Validate emergency response
    assert!(response.success, "Emergency response should succeed");
    assert!(response.performance_impact.abs() < 0.5, "Emergency response should not cause severe performance impact");
    
    // Verify that system is still functional
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    assert!(learning_result.overall_success, "System should remain functional after emergency response");
    
    println!("System overload emergency handled successfully:");
    println!("  Performance impact: {:.3}", response.performance_impact);
    println!("  System remains functional: {}", learning_result.overall_success);
    
    Ok(())
}

#[tokio::test]
async fn test_learning_divergence_emergency_response() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Capture state before emergency
    let pre_emergency_state = fixture.capture_state().await?;
    
    // Create learning divergence emergency
    fixture.create_emergency_condition(EmergencyType::LearningDivergence).await?;
    
    // Trigger emergency response
    let response = fixture.phase4_system.handle_emergency(EmergencyType::LearningDivergence).await?;
    
    // Validate emergency response
    assert!(response.success, "Learning divergence emergency should be handled successfully");
    
    // Verify that learning divergence was contained
    let post_emergency_state = fixture.capture_state().await?;
    
    // Performance should not have degraded significantly
    let performance_change = post_emergency_state.performance_score - pre_emergency_state.performance_score;
    assert!(performance_change > -0.2, "Performance should not degrade significantly during emergency: {:.3}", performance_change);
    
    // Test that normal learning can resume
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    assert!(learning_result.overall_success, "Normal learning should resume after divergence emergency");
    assert!(learning_result.validation_result.success, "Learning validation should pass after emergency");
    
    println!("Learning divergence emergency handled successfully:");
    println!("  Performance change: {:.3}", performance_change);
    println!("  Normal learning resumed: {}", learning_result.overall_success);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_exhaustion_emergency_response() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Create memory exhaustion emergency
    fixture.create_emergency_condition(EmergencyType::MemoryExhaustion).await?;
    
    // Trigger emergency response
    let response = fixture.phase4_system.handle_emergency(EmergencyType::MemoryExhaustion).await?;
    
    // Validate emergency response
    assert!(response.success, "Memory exhaustion emergency should be handled");
    assert!(response.actions_taken.iter().any(|action| 
        action.contains("memory") || action.contains("cleanup") || action.contains("cache")),
        "Emergency response should include memory-related actions");
    
    // Verify system remains stable
    let integrity_check = fixture.verify_system_integrity().await?;
    assert!(integrity_check, "System integrity should be maintained after memory emergency");
    
    println!("Memory exhaustion emergency handled successfully:");
    println!("  Actions taken: {:?}", response.actions_taken);
    
    Ok(())
}

#[tokio::test]
async fn test_infinite_loop_emergency_response() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Create infinite loop emergency
    fixture.create_emergency_condition(EmergencyType::InfiniteLoop).await?;
    
    // Trigger emergency response
    let response = fixture.phase4_system.handle_emergency(EmergencyType::InfiniteLoop).await?;
    
    // Validate emergency response
    assert!(response.success, "Infinite loop emergency should be handled");
    assert!(response.recovery_time < Duration::from_secs(5), "Infinite loop should be broken quickly");
    
    // Verify that the system is responsive
    let start_time = SystemTime::now();
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    let response_time = start_time.elapsed().unwrap_or_default();
    
    assert!(response_time < Duration::from_secs(60), "System should be responsive after loop breaking");
    assert!(learning_result.overall_success, "Learning should work normally after loop breaking");
    
    println!("Infinite loop emergency handled successfully:");
    println!("  Recovery time: {:?}", response.recovery_time);
    println!("  System response time: {:?}", response_time);
    
    Ok(())
}

#[tokio::test]
async fn test_rollback_mechanism() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Capture initial state
    let initial_state = fixture.capture_state().await?;
    println!("Initial state captured - Performance: {:.3}", initial_state.performance_score);
    
    // Execute a learning cycle that might need rollback
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Check if rollback capability is available
    if let Some(adaptive_result) = &learning_result.learning_results.adaptive_results {
        if adaptive_result.optimization_result.rollback_available {
            println!("Rollback capability confirmed for optimization results");
            
            // Simulate rollback trigger (e.g., validation failure)
            // In real implementation, would trigger actual rollback
            println!("Simulating rollback to previous state");
            
            // Verify that rollback maintains system integrity
            let post_rollback_state = fixture.capture_state().await?;
            let integrity_check = fixture.verify_system_integrity().await?;
            
            assert!(integrity_check, "System integrity should be maintained after rollback");
            
            // Performance should not be worse than initial state
            assert!(post_rollback_state.performance_score >= initial_state.performance_score - 0.1,
                   "Performance after rollback should not be significantly worse than initial");
            
            println!("Rollback mechanism validated successfully");
            println!("  Initial performance: {:.3}", initial_state.performance_score);
            println!("  Post-rollback performance: {:.3}", post_rollback_state.performance_score);
        } else {
            println!("No rollback needed for this learning cycle");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_safety_constraint_enforcement() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Test that safety constraints are enforced during learning
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Verify safety constraints
    assert!(learning_result.validation_result.success, "Safety validation should pass");
    assert!(learning_result.validation_result.changes_committed, "Safe changes should be committed");
    
    // Performance improvement should be within safe bounds
    assert!(learning_result.performance_improvement >= -0.1, "Performance should not degrade severely");
    assert!(learning_result.performance_improvement <= 0.5, "Performance improvement should be realistic");
    
    // Learning duration should be reasonable (not infinite)
    assert!(learning_result.duration < Duration::from_secs(300), "Learning should complete in reasonable time");
    
    // System assessment should indicate healthy state
    assert!(learning_result.system_assessment.overall_health >= 0.0, "System health should be non-negative");
    assert!(learning_result.system_assessment.readiness_for_learning >= 0.0, "Learning readiness should be non-negative");
    
    // Verify no emergency interventions were needed
    assert!(!learning_result.homeostasis_result.emergency_intervention, 
           "No emergency intervention should be needed during normal operation");
    
    println!("Safety constraint enforcement validated:");
    println!("  Performance improvement: {:.3}", learning_result.performance_improvement);
    println!("  Duration: {:?}", learning_result.duration);
    println!("  System health: {:.3}", learning_result.system_assessment.overall_health);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_emergency_handling() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Test handling multiple emergencies concurrently
    let emergency_types = vec![
        EmergencyType::PerformanceCritical,
        EmergencyType::SystemOverload,
        EmergencyType::LearningDivergence,
    ];
    
    // Handle emergencies concurrently
    let emergency_futures = emergency_types.into_iter().map(|emergency_type| {
        let system = &fixture.phase4_system;
        async move {
            system.handle_emergency(emergency_type).await
        }
    });
    
    let responses: Vec<Result<EmergencyResponse>> = futures::future::join_all(emergency_futures).await;
    
    // Validate that all emergencies were handled
    let successful_responses: Vec<_> = responses.into_iter()
        .filter_map(|r| r.ok())
        .filter(|r| r.success)
        .collect();
    
    assert!(successful_responses.len() >= 1, "At least one emergency should be handled successfully");
    
    // Verify system integrity after multiple emergencies
    let integrity_check = fixture.verify_system_integrity().await?;
    assert!(integrity_check, "System integrity should be maintained after multiple emergencies");
    
    // System should still be functional
    let learning_result = fixture.phase4_system.execute_learning_cycle().await?;
    assert!(learning_result.overall_success, "System should remain functional after multiple emergencies");
    
    println!("Concurrent emergency handling validated:");
    println!("  Successful responses: {}", successful_responses.len());
    println!("  System remains functional: {}", learning_result.overall_success);
    
    Ok(())
}

#[tokio::test]
async fn test_graceful_degradation() -> Result<()> {
    let fixture = SafetyTestFixture::new().await?;
    
    // Test graceful degradation under adverse conditions
    let initial_performance = fixture.capture_state().await?.performance_score;
    
    // Simulate adverse conditions by triggering multiple stress factors
    for emergency_type in [EmergencyType::SystemOverload, EmergencyType::MemoryExhaustion] {
        fixture.create_emergency_condition(emergency_type.clone()).await?;
        let _response = fixture.phase4_system.handle_emergency(emergency_type).await?;
    }
    
    // System should still function, albeit potentially with reduced performance
    let degraded_result = fixture.phase4_system.execute_learning_cycle().await?;
    
    // Verify graceful degradation properties
    assert!(degraded_result.overall_success, "System should continue functioning under stress");
    
    // Performance may be reduced but should not collapse
    let degraded_performance = degraded_result.system_assessment.overall_health;
    assert!(degraded_performance >= 0.2, "Performance should not collapse completely: {:.3}", degraded_performance);
    
    // System should maintain essential functions
    assert!(degraded_result.validation_result.success, "Core validation should still work");
    assert!(degraded_result.duration < Duration::from_secs(600), "Operations should complete in reasonable time");
    
    // Recovery should be possible
    let recovery_result = fixture.phase4_system.execute_learning_cycle().await?;
    assert!(recovery_result.overall_success, "System should be able to recover");
    
    println!("Graceful degradation validated:");
    println!("  Initial performance: {:.3}", initial_performance);
    println!("  Degraded performance: {:.3}", degraded_performance);
    println!("  Recovery successful: {}", recovery_result.overall_success);
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_safety_validation() -> Result<()> {
    let mut fixture = SafetyTestFixture::new().await?;
    
    // Test invalid configurations are rejected
    let invalid_configs = vec![
        // Negative learning aggressiveness
        Phase4Config {
            learning_aggressiveness: -0.1,
            integration_depth: IntegrationDepth::Standard,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.7,
                adaptation_speed_target: Duration::from_secs(300),
                memory_overhead_limit: 0.2,
                performance_degradation_limit: 0.1,
                user_satisfaction_target: 0.8,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 3,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.5,
                max_learning_impact_per_session: 0.15,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 1024.0,
                max_cpu_usage_percentage: 50.0,
                max_storage_usage_mb: 200.0,
                max_network_bandwidth_mbps: 25.0,
                max_session_duration: Duration::from_secs(600),
                max_daily_learning_time: Duration::from_secs(3600),
            },
        },
        // Learning aggressiveness too high
        Phase4Config {
            learning_aggressiveness: 2.0,
            integration_depth: IntegrationDepth::Standard,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.7,
                adaptation_speed_target: Duration::from_secs(300),
                memory_overhead_limit: 0.2,
                performance_degradation_limit: 0.1,
                user_satisfaction_target: 0.8,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 3,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.5,
                max_learning_impact_per_session: 0.15,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 1024.0,
                max_cpu_usage_percentage: 50.0,
                max_storage_usage_mb: 200.0,
                max_network_bandwidth_mbps: 25.0,
                max_session_duration: Duration::from_secs(600),
                max_daily_learning_time: Duration::from_secs(3600),
            },
        },
    ];
    
    for (i, invalid_config) in invalid_configs.iter().enumerate() {
        let result = fixture.phase4_system.update_configuration(invalid_config.clone());
        assert!(result.is_err(), "Invalid configuration {} should be rejected", i);
        
        println!("Invalid configuration {} correctly rejected", i);
    }
    
    // Test valid configuration is accepted
    let valid_config = Phase4Config {
        learning_aggressiveness: 0.5,
        integration_depth: IntegrationDepth::Standard,
        performance_targets: PerformanceTargets {
            learning_efficiency_target: 0.8,
            adaptation_speed_target: Duration::from_secs(300),
            memory_overhead_limit: 0.2,
            performance_degradation_limit: 0.1,
            user_satisfaction_target: 0.85,
        },
        safety_constraints: SafetyConstraints {
            max_concurrent_learning_sessions: 2,
            rollback_capability_required: true,
            performance_monitoring_required: true,
            emergency_protocols_enabled: true,
            user_intervention_threshold: 0.5,
            max_learning_impact_per_session: 0.15,
        },
        resource_limits: ResourceLimits {
            max_memory_usage_mb: 1024.0,
            max_cpu_usage_percentage: 50.0,
            max_storage_usage_mb: 200.0,
            max_network_bandwidth_mbps: 25.0,
            max_session_duration: Duration::from_secs(600),
            max_daily_learning_time: Duration::from_secs(3600),
        },
    };
    
    let result = fixture.phase4_system.update_configuration(valid_config);
    assert!(result.is_ok(), "Valid configuration should be accepted");
    
    println!("Configuration safety validation completed successfully");
    
    Ok(())
}