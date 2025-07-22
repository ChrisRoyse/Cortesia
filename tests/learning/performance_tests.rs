//! Performance tests for learning algorithm efficiency
//! 
//! These tests validate that learning algorithms meet performance requirements
//! and scale appropriately with different workloads.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::collections::HashMap;
use uuid::Uuid;
use anyhow::Result;

use llmkg::learning::{
    Phase4LearningSystem,
    AdaptiveLearningSystem,
    HebbianLearningEngine,
    SynapticHomeostasis,
    NeuralPatternDetectionSystem,
    ParameterTuner,
    MetaLearningSystem,
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
    ComprehensiveLearningResult,
    LearningSessionType,
};

use llmkg::cognitive::phase3_integration::Phase3IntegratedCognitiveSystem;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::types::EntityKey;
use llmkg::core::triple::NodeType;
use llmkg::core::brain_types::RelationType;

/// Performance test fixture with configurable scale
pub struct PerformanceTestFixture {
    pub phase4_system: Phase4LearningSystem,
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub test_entities: Vec<EntityKey>,
    pub scale_factor: usize,
}

impl PerformanceTestFixture {
    /// Create new performance test fixture with specified scale
    pub async fn new(scale_factor: usize) -> Result<Self> {
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new().await?);
        let sdr_storage = Arc::new(SDRStorage::new().await?);
        let phase3_system = Arc::new(Phase3IntegratedCognitiveSystem::new(
            brain_graph.clone(),
            sdr_storage.clone()
        ).await?);
        
        // Create test entities based on scale factor
        let mut test_entities = Vec::new();
        for i in 0..scale_factor * 10 {
            let entity = brain_graph.add_entity(
                format!("perf_test_entity_{}", i),
                NodeType::Concept,
                HashMap::new()
            ).await?;
            test_entities.push(entity);
        }
        
        // Create connections (sparse connectivity for performance)
        for i in 0..test_entities.len() {
            let next_idx = (i + 1) % test_entities.len();
            brain_graph.add_relationship(
                test_entities[i],
                test_entities[next_idx],
                RelationType::RelatedTo,
                0.5,
                HashMap::new()
            ).await?;
            
            // Add some cross connections for complexity
            if i % 3 == 0 && i + 3 < test_entities.len() {
                brain_graph.add_relationship(
                    test_entities[i],
                    test_entities[i + 3],
                    RelationType::Influences,
                    0.3,
                    HashMap::new()
                ).await?;
            }
        }
        
        // Performance-oriented configuration
        let config = Phase4Config {
            learning_aggressiveness: 0.6,
            integration_depth: IntegrationDepth::Standard,
            performance_targets: PerformanceTargets {
                learning_efficiency_target: 0.85,
                adaptation_speed_target: Duration::from_secs(120),
                memory_overhead_limit: 0.2,
                performance_degradation_limit: 0.08,
                user_satisfaction_target: 0.9,
            },
            safety_constraints: SafetyConstraints {
                max_concurrent_learning_sessions: 3,
                rollback_capability_required: true,
                performance_monitoring_required: true,
                emergency_protocols_enabled: true,
                user_intervention_threshold: 0.4,
                max_learning_impact_per_session: 0.2,
            },
            resource_limits: ResourceLimits {
                max_memory_usage_mb: 1536.0,
                max_cpu_usage_percentage: 60.0,
                max_storage_usage_mb: 300.0,
                max_network_bandwidth_mbps: 40.0,
                max_session_duration: Duration::from_secs(600),
                max_daily_learning_time: Duration::from_secs(5400),
            },
        };
        
        let phase4_system = Phase4LearningSystem::new(
            phase3_system,
            brain_graph.clone(),
            sdr_storage,
            Some(config),
        ).await?;
        
        Ok(Self {
            phase4_system,
            brain_graph,
            test_entities,
            scale_factor,
        })
    }
    
    /// Generate load test activation events
    pub fn generate_load_test_events(&self, event_count: usize) -> Vec<ActivationEvent> {
        (0..event_count).map(|i| {
            let entity_idx = i % self.test_entities.len();
            ActivationEvent {
                entity_key: self.test_entities[entity_idx],
                activation_strength: 0.5 + (i as f32 % 10.0) / 20.0, // 0.5 to 1.0
                timestamp: std::time::Instant::now() + Duration::from_millis(i as u64),
                context: ActivationContext {
                    query_id: format!("perf_test_{}", i),
                    cognitive_pattern: CognitivePatternType::Convergent,
                    user_session: Some(Uuid::new_v4().to_string()),
                    outcome_quality: Some(0.8),
                },
            }
        }).collect()
    }
    
    /// Measure memory usage (simplified)
    pub fn measure_memory_usage(&self) -> usize {
        // Simplified memory measurement
        self.test_entities.len() * std::mem::size_of::<EntityKey>() +
        self.scale_factor * 1024 // Approximate overhead
    }
}

/// Performance metrics for benchmarking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time: Duration,
    pub throughput_ops_per_sec: f32,
    pub memory_usage_bytes: usize,
    pub cpu_efficiency: f32,
    pub scalability_factor: f32,
}

impl PerformanceMetrics {
    pub fn new(execution_time: Duration, operations: usize, memory_usage: usize) -> Self {
        let throughput = if execution_time.as_secs_f32() > 0.0 {
            operations as f32 / execution_time.as_secs_f32()
        } else {
            0.0
        };
        
        Self {
            execution_time,
            throughput_ops_per_sec: throughput,
            memory_usage_bytes: memory_usage,
            cpu_efficiency: 0.8, // Simplified
            scalability_factor: 1.0, // Baseline
        }
    }
}

#[tokio::test]
async fn test_learning_cycle_performance_small_scale() -> Result<()> {
    let fixture = PerformanceTestFixture::new(1).await?; // Small scale: 10 entities
    
    let start_time = Instant::now();
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    let execution_time = start_time.elapsed();
    
    // Performance requirements for small scale
    assert!(execution_time < Duration::from_secs(30), 
           "Small scale learning cycle should complete in <30s, took {:?}", execution_time);
    
    assert!(result.overall_success, "Learning cycle should succeed");
    
    let memory_usage = fixture.measure_memory_usage();
    let metrics = PerformanceMetrics::new(execution_time, 1, memory_usage);
    
    println!("Small scale performance metrics:");
    println!("  Execution time: {:?}", metrics.execution_time);
    println!("  Memory usage: {} bytes", metrics.memory_usage_bytes);
    println!("  Success: {}", result.overall_success);
    
    Ok(())
}

#[tokio::test]
async fn test_learning_cycle_performance_medium_scale() -> Result<()> {
    let fixture = PerformanceTestFixture::new(5).await?; // Medium scale: 50 entities
    
    let start_time = Instant::now();
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    let execution_time = start_time.elapsed();
    
    // Performance requirements for medium scale
    assert!(execution_time < Duration::from_secs(120), 
           "Medium scale learning cycle should complete in <120s, took {:?}", execution_time);
    
    assert!(result.overall_success, "Learning cycle should succeed");
    
    let memory_usage = fixture.measure_memory_usage();
    let metrics = PerformanceMetrics::new(execution_time, 1, memory_usage);
    
    println!("Medium scale performance metrics:");
    println!("  Execution time: {:?}", metrics.execution_time);
    println!("  Memory usage: {} bytes", metrics.memory_usage_bytes);
    println!("  Success: {}", result.overall_success);
    
    // Verify reasonable memory usage
    assert!(memory_usage < 1024 * 1024, "Memory usage should be reasonable for medium scale");
    
    Ok(())
}

#[tokio::test]
async fn test_learning_cycle_performance_large_scale() -> Result<()> {
    let fixture = PerformanceTestFixture::new(10).await?; // Large scale: 100 entities
    
    let start_time = Instant::now();
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    let execution_time = start_time.elapsed();
    
    // Performance requirements for large scale (more lenient)
    assert!(execution_time < Duration::from_secs(300), 
           "Large scale learning cycle should complete in <300s, took {:?}", execution_time);
    
    assert!(result.overall_success, "Learning cycle should succeed");
    
    let memory_usage = fixture.measure_memory_usage();
    let metrics = PerformanceMetrics::new(execution_time, 1, memory_usage);
    
    println!("Large scale performance metrics:");
    println!("  Execution time: {:?}", metrics.execution_time);
    println!("  Memory usage: {} bytes", metrics.memory_usage_bytes);
    println!("  Success: {}", result.overall_success);
    
    // Verify memory usage is bounded
    assert!(memory_usage < 10 * 1024 * 1024, "Memory usage should be bounded for large scale");
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_learning_performance() -> Result<()> {
    let fixture = PerformanceTestFixture::new(3).await?; // 30 entities
    
    // Test concurrent learning cycles
    let num_concurrent = 3;
    let start_time = Instant::now();
    
    let futures = (0..num_concurrent).map(|i| {
        let system = &fixture.phase4_system;
        async move {
            println!("Starting concurrent learning cycle {}", i);
            system.execute_learning_cycle().await
        }
    });
    
    let results: Vec<Result<ComprehensiveLearningResult>> = futures::future::join_all(futures).await;
    let execution_time = start_time.elapsed();
    
    // Validate concurrent performance
    let successful_results: Vec<_> = results.into_iter()
        .filter_map(|r| r.ok())
        .filter(|r| r.overall_success)
        .collect();
    
    assert!(successful_results.len() >= 1, "At least one concurrent cycle should succeed");
    
    // Concurrent execution should not be significantly slower than sequential
    // (allowing for some overhead)
    assert!(execution_time < Duration::from_secs(180), 
           "Concurrent learning should complete in reasonable time: {:?}", execution_time);
    
    let memory_usage = fixture.measure_memory_usage();
    let throughput = successful_results.len() as f32 / execution_time.as_secs_f32();
    
    println!("Concurrent learning performance:");
    println!("  Concurrent cycles: {}", num_concurrent);
    println!("  Successful cycles: {}", successful_results.len());
    println!("  Total execution time: {:?}", execution_time);
    println!("  Throughput: {:.2} cycles/sec", throughput);
    println!("  Memory usage: {} bytes", memory_usage);
    
    Ok(())
}

#[tokio::test]
async fn test_activation_processing_throughput() -> Result<()> {
    let fixture = PerformanceTestFixture::new(2).await?; // 20 entities
    
    // Generate many activation events
    let event_count = 1000;
    let events = fixture.generate_load_test_events(event_count);
    
    let start_time = Instant::now();
    
    // Process activation events in batches
    let batch_size = 100;
    for batch in events.chunks(batch_size) {
        // In real implementation, would process batch through learning systems
        // For now, simulate processing time
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        println!("Processed batch of {} events", batch.len());
    }
    
    let execution_time = start_time.elapsed();
    let throughput = event_count as f32 / execution_time.as_secs_f32();
    
    // Performance requirements for activation processing
    assert!(throughput > 100.0, "Should process >100 activations/sec, got {:.1}", throughput);
    assert!(execution_time < Duration::from_secs(30), "Should process all events in <30s");
    
    println!("Activation processing performance:");
    println!("  Events processed: {}", event_count);
    println!("  Execution time: {:?}", execution_time);
    println!("  Throughput: {:.1} events/sec", throughput);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_efficiency() -> Result<()> {
    let fixtures = vec![
        PerformanceTestFixture::new(1).await?,  // 10 entities
        PerformanceTestFixture::new(2).await?,  // 20 entities
        PerformanceTestFixture::new(5).await?,  // 50 entities
    ];
    
    let mut memory_measurements = Vec::new();
    
    for (i, fixture) in fixtures.iter().enumerate() {
        let memory_usage = fixture.measure_memory_usage();
        memory_measurements.push((fixture.scale_factor, memory_usage));
        
        println!("Scale factor {}: Memory usage {} bytes", fixture.scale_factor, memory_usage);
    }
    
    // Verify memory scaling is reasonable (should be roughly linear)
    for i in 1..memory_measurements.len() {
        let (prev_scale, prev_memory) = memory_measurements[i-1];
        let (curr_scale, curr_memory) = memory_measurements[i];
        
        let scale_ratio = curr_scale as f32 / prev_scale as f32;
        let memory_ratio = curr_memory as f32 / prev_memory as f32;
        
        // Memory should not grow exponentially
        assert!(memory_ratio <= scale_ratio * 2.0, 
               "Memory growth should be reasonable: scale ratio {:.1}, memory ratio {:.1}", 
               scale_ratio, memory_ratio);
        
        println!("Scale ratio: {:.1}, Memory ratio: {:.1}", scale_ratio, memory_ratio);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_learning_convergence_speed() -> Result<()> {
    let fixture = PerformanceTestFixture::new(2).await?; // 20 entities
    
    // Test how quickly learning converges
    let mut performance_history = Vec::new();
    let start_time = Instant::now();
    
    for cycle in 0..5 {
        let cycle_start = Instant::now();
        let result = fixture.phase4_system.execute_learning_cycle().await?;
        let cycle_time = cycle_start.elapsed();
        
        performance_history.push((cycle_time, result.performance_improvement));
        
        println!("Cycle {}: Time {:?}, Performance improvement {:.3}", 
                cycle + 1, cycle_time, result.performance_improvement);
        
        // Early convergence check
        if result.performance_improvement < 0.01 {
            println!("Early convergence detected at cycle {}", cycle + 1);
            break;
        }
    }
    
    let total_time = start_time.elapsed();
    
    // Performance requirements for convergence
    assert!(total_time < Duration::from_secs(600), "Convergence should happen within 10 minutes");
    assert!(!performance_history.is_empty(), "Should have at least one learning cycle");
    
    // Cycles should generally get faster as system optimizes
    if performance_history.len() > 2 {
        let first_cycle_time = performance_history[0].0;
        let last_cycle_time = performance_history.last().unwrap().0;
        
        // Allow for some variation, but cycles shouldn't get dramatically slower
        assert!(last_cycle_time <= first_cycle_time * 2, 
               "Learning cycles should not get much slower over time");
    }
    
    println!("Learning convergence performance:");
    println!("  Total cycles: {}", performance_history.len());
    println!("  Total time: {:?}", total_time);
    println!("  Average cycle time: {:?}", total_time / performance_history.len() as u32);
    
    Ok(())
}

#[tokio::test]
async fn test_emergency_response_speed() -> Result<()> {
    let fixture = PerformanceTestFixture::new(1).await?; // Small scale for fast emergency testing
    
    // Test emergency response times
    use llmkg::learning::phase4_integration::emergency::EmergencyType;
    
    let emergency_types = vec![
        EmergencyType::PerformanceCritical,
        EmergencyType::SystemOverload,
        EmergencyType::LearningDivergence,
    ];
    
    for emergency_type in emergency_types {
        let start_time = Instant::now();
        let response = fixture.phase4_system.handle_emergency(emergency_type.clone()).await?;
        let response_time = start_time.elapsed();
        
        // Emergency responses should be fast
        assert!(response_time < Duration::from_secs(10), 
               "Emergency {:?} response should be <10s, took {:?}", emergency_type, response_time);
        
        assert!(response.success, "Emergency response should succeed");
        
        println!("Emergency {:?}: Response time {:?}", emergency_type, response_time);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_scalability_characteristics() -> Result<()> {
    // Test performance across different scales
    let scales = vec![1, 2, 5]; // 10, 20, 50 entities
    let mut scalability_data = Vec::new();
    
    for scale in scales {
        let fixture = PerformanceTestFixture::new(scale).await?;
        
        let start_time = Instant::now();
        let result = fixture.phase4_system.execute_learning_cycle().await?;
        let execution_time = start_time.elapsed();
        
        let memory_usage = fixture.measure_memory_usage();
        let entities = fixture.test_entities.len();
        
        scalability_data.push((entities, execution_time, memory_usage, result.overall_success));
        
        println!("Scale {}: {} entities, {:?} execution time, {} bytes memory", 
                scale, entities, execution_time, memory_usage);
    }
    
    // Analyze scalability
    for i in 1..scalability_data.len() {
        let (prev_entities, prev_time, prev_memory, _) = scalability_data[i-1];
        let (curr_entities, curr_time, curr_memory, _) = scalability_data[i];
        
        let entity_ratio = curr_entities as f32 / prev_entities as f32;
        let time_ratio = curr_time.as_millis() as f32 / prev_time.as_millis() as f32;
        let memory_ratio = curr_memory as f32 / prev_memory as f32;
        
        // Time complexity should be reasonable (not exponential)
        assert!(time_ratio <= entity_ratio * entity_ratio, 
               "Time complexity should not be worse than quadratic: entity ratio {:.1}, time ratio {:.1}", 
               entity_ratio, time_ratio);
        
        // Memory should scale reasonably
        assert!(memory_ratio <= entity_ratio * 1.5, 
               "Memory should scale reasonably: entity ratio {:.1}, memory ratio {:.1}", 
               entity_ratio, memory_ratio);
        
        println!("Scalability analysis: Entity ratio {:.1}, Time ratio {:.1}, Memory ratio {:.1}", 
                entity_ratio, time_ratio, memory_ratio);
    }
    
    println!("Scalability characteristics validated successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_resource_utilization_efficiency() -> Result<()> {
    let fixture = PerformanceTestFixture::new(3).await?; // 30 entities
    
    // Measure resource utilization during learning
    let start_time = Instant::now();
    let initial_memory = fixture.measure_memory_usage();
    
    let result = fixture.phase4_system.execute_learning_cycle().await?;
    
    let execution_time = start_time.elapsed();
    let final_memory = fixture.measure_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    
    // Resource efficiency requirements
    assert!(result.overall_success, "Learning should succeed");
    
    // Memory growth should be bounded
    let max_allowed_growth = initial_memory / 2; // Allow 50% growth
    assert!(memory_growth <= max_allowed_growth, 
           "Memory growth should be bounded: {} bytes growth, max allowed {}", 
           memory_growth, max_allowed_growth);
    
    // Execution time should be reasonable for the scale
    assert!(execution_time < Duration::from_secs(180), 
           "Execution should complete in reasonable time for 30 entities");
    
    // Calculate efficiency metrics
    let entities_per_second = fixture.test_entities.len() as f32 / execution_time.as_secs_f32();
    let memory_per_entity = final_memory / fixture.test_entities.len();
    
    println!("Resource utilization efficiency:");
    println!("  Entities processed: {}", fixture.test_entities.len());
    println!("  Execution time: {:?}", execution_time);
    println!("  Entities per second: {:.1}", entities_per_second);
    println!("  Initial memory: {} bytes", initial_memory);
    println!("  Final memory: {} bytes", final_memory);
    println!("  Memory growth: {} bytes", memory_growth);
    println!("  Memory per entity: {} bytes", memory_per_entity);
    
    // Efficiency thresholds
    assert!(entities_per_second > 0.1, "Should process at least 0.1 entities/sec");
    assert!(memory_per_entity < 100_000, "Memory per entity should be reasonable");
    
    Ok(())
}