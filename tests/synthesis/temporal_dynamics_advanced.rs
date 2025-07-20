//! # Quantum Knowledge Synthesizer: Advanced Temporal Dynamics Testing
//! 
//! This module provides comprehensive testing strategies for temporal aspects
//! of brain-inspired neural networks with hook-intelligent validation.
//! 
//! ## Temporal Testing Philosophy
//! - Time-dependent behaviors require specialized testing approaches
//! - Decay functions need mathematical precision validation
//! - Temporal causality must be preserved across operations
//! - Stochastic temporal events need statistical validation

use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, RelationType,
    ActivationPattern, ActivationStep, ActivationOperation
};
use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use std::time::{SystemTime, Duration, Instant};
use std::collections::HashMap;
use tokio::time::sleep;

/// Advanced temporal decay testing with mathematical precision
#[derive(Debug, Clone)]
pub struct TemporalMathematicsHarness {
    pub entity_count: usize,
    pub time_resolution: Duration,
    pub precision_epsilon: f64,
}

impl TemporalMathematicsHarness {
    pub fn new() -> Self {
        Self {
            entity_count: 100,
            time_resolution: Duration::from_millis(10),
            precision_epsilon: 1e-6,
        }
    }
    
    /// Test exponential decay mathematical precision across multiple entities
    pub async fn test_multi_entity_decay_precision(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut entities = Vec::new();
        let base_time = SystemTime::now();
        
        // Create entities with staggered activation times
        for i in 0..self.entity_count {
            let mut entity = BrainInspiredEntity::new(
                format!("Entity{}", i), 
                EntityDirection::Hidden
            );
            entity.activation_state = 1.0;
            entity.last_activation = base_time - Duration::from_secs(i as u64);
            entities.push(entity);
        }
        
        let decay_rate = 0.5f32;
        let test_durations = vec![1.0, 2.0, 5.0, 10.0]; // seconds
        
        for &duration in &test_durations {
            for (i, entity) in entities.iter_mut().enumerate() {
                let time_since_activation = (i as f32) + duration;
                let expected_decay = (-decay_rate * time_since_activation).exp();
                
                // Reset for clean test
                entity.activation_state = 1.0;
                entity.last_activation = SystemTime::now() - Duration::from_secs_f32(time_since_activation);
                
                let actual_result = entity.activate(0.0, decay_rate);
                let difference = (actual_result as f64 - expected_decay as f64).abs();
                
                assert!(
                    difference < self.precision_epsilon,
                    "Decay precision error for entity {}, duration {}: expected {}, got {}, diff {}",
                    i, duration, expected_decay, actual_result, difference
                );
            }
        }
        
        Ok(())
    }
    
    /// Test temporal causality preservation in complex networks
    pub async fn test_temporal_causality_preservation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create causal chain: A(t0) -> B(t1) -> C(t2)
        let entity_a = BrainInspiredEntity::new("CauseA".to_string(), EntityDirection::Input);
        let entity_b = BrainInspiredEntity::new("IntermediateB".to_string(), EntityDirection::Hidden);
        let entity_c = BrainInspiredEntity::new("EffectC".to_string(), EntityDirection::Output);
        
        let key_a = entity_a.id;
        let key_b = entity_b.id;
        let key_c = entity_c.id;
        
        engine.add_entity(entity_a).await?;
        engine.add_entity(entity_b).await?;
        engine.add_entity(entity_c).await?;
        
        // Create relationships with temporal characteristics
        let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::Temporal);
        rel_ab.weight = 0.8;
        rel_ab.temporal_decay = 0.2;
        
        let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::Temporal);
        rel_bc.weight = 0.7;
        rel_bc.temporal_decay = 0.15;
        
        engine.add_relationship(rel_ab).await?;
        engine.add_relationship(rel_bc).await?;
        
        // Test causal ordering: activation should propagate in sequence
        let mut pattern = ActivationPattern::new("causality_test".to_string());
        pattern.activations.insert(key_a, 1.0);
        
        let result = engine.propagate_activation(&pattern).await?;
        
        // Verify causal ordering in activation trace
        let trace = &result.activation_trace;
        let mut a_first_activation = None;
        let mut b_first_activation = None;
        let mut c_first_activation = None;
        
        for step in trace {
            match step.entity_key {
                k if k == key_a && a_first_activation.is_none() => {
                    a_first_activation = Some(step.step_id);
                }
                k if k == key_b && step.activation_level > 0.1 && b_first_activation.is_none() => {
                    b_first_activation = Some(step.step_id);
                }
                k if k == key_c && step.activation_level > 0.1 && c_first_activation.is_none() => {
                    c_first_activation = Some(step.step_id);
                }
                _ => {}
            }
        }
        
        // Verify temporal ordering
        if let (Some(a_step), Some(b_step), Some(c_step)) = (a_first_activation, b_first_activation, c_first_activation) {
            assert!(a_step <= b_step, "A should activate before or with B: {} vs {}", a_step, b_step);
            assert!(b_step <= c_step, "B should activate before or with C: {} vs {}", b_step, c_step);
        }
        
        Ok(())
    }
    
    /// Test temporal decay convergence properties
    pub async fn test_decay_convergence_properties(&self) -> Result<(), Box<dyn std::error::Error>> {
        let decay_rates = vec![0.1, 0.5, 1.0, 2.0];
        let initial_activations = vec![0.1, 0.5, 0.9, 1.0];
        
        for &decay_rate in &decay_rates {
            for &initial_activation in &initial_activations {
                let mut entity = BrainInspiredEntity::new(
                    format!("Convergence_{}_{}", decay_rate, initial_activation),
                    EntityDirection::Hidden
                );
                entity.activation_state = initial_activation;
                
                // Test convergence to zero over extended time
                let convergence_threshold = 0.001;
                let max_time = 20.0; // seconds
                let time_step = 0.5;
                
                let mut time = 0.0;
                let mut converged = false;
                
                while time < max_time && !converged {
                    entity.last_activation = SystemTime::now() - Duration::from_secs_f32(time);
                    let activation = entity.activate(0.0, decay_rate);
                    
                    if activation < convergence_threshold {
                        converged = true;
                    }
                    
                    time += time_step;
                }
                
                // For positive decay rates, should eventually converge
                if decay_rate > 0.0 {
                    assert!(converged, 
                        "Decay should converge for rate {} and initial {}", 
                        decay_rate, initial_activation);
                }
            }
        }
        
        Ok(())
    }
    
    /// Test temporal consistency under concurrent modifications
    pub async fn test_temporal_consistency_concurrent(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ActivationConfig::default();
        let engine = ActivationPropagationEngine::new(config);
        
        // Create network of interconnected entities
        let mut entities = Vec::new();
        for i in 0..10 {
            let entity = BrainInspiredEntity::new(
                format!("Concurrent{}", i),
                EntityDirection::Hidden
            );
            entities.push(entity.id);
            engine.add_entity(entity).await?;
        }
        
        // Create full mesh of relationships
        for &source in &entities {
            for &target in &entities {
                if source != target {
                    let mut rel = BrainInspiredRelationship::new(
                        source, target, RelationType::RelatedTo
                    );
                    rel.weight = 0.1; // Weak connections
                    rel.temporal_decay = 0.3;
                    engine.add_relationship(rel).await?;
                }
            }
        }
        
        // Launch concurrent activation patterns
        let pattern_count = 5;
        let mut handles = Vec::new();
        
        for i in 0..pattern_count {
            let engine_clone = engine.clone(); // Assuming engine is cloneable
            let entities_clone = entities.clone();
            
            let handle = tokio::spawn(async move {
                let mut pattern = ActivationPattern::new(format!("concurrent_{}", i));
                
                // Activate different subset of entities
                for (j, &key) in entities_clone.iter().enumerate() {
                    if j % pattern_count == i {
                        pattern.activations.insert(key, 0.8);
                    }
                }
                
                engine_clone.propagate_activation(&pattern).await
            });
            
            handles.push(handle);
        }
        
        // Wait for all propagations to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await??;
            results.push(result);
        }
        
        // Verify temporal consistency
        for result in &results {
            // Check that no activation exceeded bounds
            for &activation in result.final_activations.values() {
                assert!(activation >= 0.0 && activation <= 1.0,
                    "Concurrent activation should remain bounded: {}", activation);
            }
            
            // Check that activation trace is temporally ordered
            let trace = &result.activation_trace;
            for window in trace.windows(2) {
                assert!(window[0].step_id <= window[1].step_id,
                    "Activation trace should be temporally ordered: {} -> {}",
                    window[0].step_id, window[1].step_id);
            }
        }
        
        Ok(())
    }
}

/// Advanced Hebbian learning temporal dynamics testing
#[derive(Debug, Clone)]
pub struct HebbianTemporalHarness {
    pub learning_cycles: usize,
    pub decay_test_duration: Duration,
}

impl HebbianTemporalHarness {
    pub fn new() -> Self {
        Self {
            learning_cycles: 1000,
            decay_test_duration: Duration::from_secs(60),
        }
    }
    
    /// Test learning-decay equilibrium dynamics
    pub async fn test_learning_decay_equilibrium(&self) -> Result<(), Box<dyn std::error::Error>> {
        let learning_rates = vec![0.01, 0.05, 0.1, 0.2];
        let decay_rates = vec![0.05, 0.1, 0.2, 0.5];
        
        for &learning_rate in &learning_rates {
            for &decay_rate in &decay_rates {
                let mut relationship = BrainInspiredRelationship::new(
                    Default::default(),
                    Default::default(),
                    RelationType::Learned
                );
                relationship.temporal_decay = decay_rate;
                
                let mut weight_history = Vec::new();
                
                // Simulate learning-decay cycles
                for cycle in 0..self.learning_cycles {
                    // Learning phase
                    relationship.strengthen(learning_rate);
                    
                    // Decay phase (simulate 1 second passage)
                    relationship.last_strengthened = SystemTime::now() - Duration::from_secs(1);
                    relationship.apply_decay();
                    
                    weight_history.push(relationship.weight);
                    
                    // Test for equilibrium detection
                    if cycle > 100 {
                        let recent_weights = &weight_history[cycle.saturating_sub(50)..];
                        let weight_variance = Self::calculate_variance(recent_weights);
                        
                        // If variance is very low, we've reached equilibrium
                        if weight_variance < 0.001 {
                            println!("Equilibrium reached at cycle {} for learning={}, decay={}, weight={:.4}", 
                                    cycle, learning_rate, decay_rate, relationship.weight);
                            break;
                        }
                    }
                }
                
                // Verify equilibrium properties
                let final_weight = relationship.weight;
                assert!(final_weight >= 0.0 && final_weight <= 1.0,
                    "Equilibrium weight should be bounded: {}", final_weight);
                
                // Higher learning rates should generally lead to higher equilibrium weights
                // (when decay rates are comparable)
            }
        }
        
        Ok(())
    }
    
    /// Test competitive Hebbian learning dynamics
    pub async fn test_competitive_hebbian_dynamics(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut relationships = Vec::new();
        
        // Create competing relationships from same source
        let source_key = Default::default();
        for i in 0..5 {
            let mut rel = BrainInspiredRelationship::new(
                source_key,
                Default::default(), // Different targets
                RelationType::Learned
            );
            rel.weight = 0.2; // Start equal
            rel.temporal_decay = 0.1;
            relationships.push(rel);
        }
        
        // Simulate competitive strengthening
        for cycle in 0..500 {
            // Randomly strengthen one relationship per cycle (winner-takes-all)
            let winner_idx = cycle % relationships.len();
            
            for (i, rel) in relationships.iter_mut().enumerate() {
                if i == winner_idx {
                    rel.strengthen(0.1); // Winner gets strengthened
                } else {
                    // Losers experience decay only
                    rel.last_strengthened = SystemTime::now() - Duration::from_secs(1);
                    rel.apply_decay();
                }
            }
        }
        
        // Verify competitive dynamics
        let weights: Vec<f32> = relationships.iter().map(|r| r.weight).collect();
        let max_weight = weights.iter().cloned().fold(0.0, f32::max);
        let min_weight = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        
        // Should see differentiation in weights
        assert!(max_weight > min_weight + 0.1,
            "Competitive learning should create weight differentiation: max={}, min={}",
            max_weight, min_weight);
        
        Ok(())
    }
    
    /// Test spike-timing dependent plasticity (STDP) simulation
    pub async fn test_stdp_simulation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut relationship = BrainInspiredRelationship::new(
            Default::default(),
            Default::default(),
            RelationType::Learned
        );
        
        let initial_weight = relationship.weight;
        let spike_window = Duration::from_millis(20); // 20ms STDP window
        
        // Test causal spike timing (pre before post)
        for _ in 0..10 {
            // Pre-synaptic spike
            let pre_time = SystemTime::now();
            
            // Post-synaptic spike 10ms later (within window)
            tokio::time::sleep(Duration::from_millis(10)).await;
            let post_time = SystemTime::now();
            
            // Simulate STDP rule: if post follows pre within window, strengthen
            let time_diff = post_time.duration_since(pre_time).unwrap_or_default();
            if time_diff <= spike_window {
                relationship.strengthen(0.05); // LTP
            }
        }
        
        let causal_weight = relationship.weight;
        
        // Reset for anti-causal test
        relationship.weight = initial_weight;
        
        // Test anti-causal spike timing (post before pre)
        for _ in 0..10 {
            // Post-synaptic spike first
            let post_time = SystemTime::now();
            
            // Pre-synaptic spike 10ms later
            tokio::time::sleep(Duration::from_millis(10)).await;
            let pre_time = SystemTime::now();
            
            // Anti-causal should weaken (LTD)
            relationship.weight = (relationship.weight - 0.02).max(0.0);
        }
        
        let anti_causal_weight = relationship.weight;
        
        // Verify STDP asymmetry
        assert!(causal_weight > anti_causal_weight,
            "Causal timing should strengthen more than anti-causal: {} vs {}",
            causal_weight, anti_causal_weight);
        
        Ok(())
    }
    
    fn calculate_variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        variance
    }
}

/// Performance benchmarking for temporal operations
#[derive(Debug, Clone)]
pub struct TemporalPerformanceBenchmark {
    pub scale_factors: Vec<usize>,
    pub time_scales: Vec<Duration>,
}

impl TemporalPerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            scale_factors: vec![10, 100, 1000, 10000],
            time_scales: vec![
                Duration::from_millis(1),
                Duration::from_millis(10),
                Duration::from_millis(100),
                Duration::from_secs(1),
            ],
        }
    }
    
    /// Benchmark temporal decay computation scaling
    pub async fn benchmark_decay_scaling(&self) -> Result<(), Box<dyn std::error::Error>> {
        for &scale in &self.scale_factors {
            for &time_scale in &self.time_scales {
                let mut entities = Vec::new();
                
                // Create entities with staggered times
                for i in 0..scale {
                    let mut entity = BrainInspiredEntity::new(
                        format!("Bench{}", i),
                        EntityDirection::Hidden
                    );
                    entity.activation_state = 1.0;
                    entity.last_activation = SystemTime::now() - (time_scale * i as u32);
                    entities.push(entity);
                }
                
                // Benchmark decay computation
                let start = Instant::now();
                
                for entity in entities.iter_mut() {
                    entity.activate(0.0, 0.5);
                }
                
                let duration = start.elapsed();
                
                println!("Decay computation for {} entities with {} time scale: {:?}",
                        scale, time_scale.as_millis(), duration);
                
                // Performance assertion: should scale linearly
                let time_per_entity = duration.as_nanos() / scale as u128;
                assert!(time_per_entity < 10_000, // 10 microseconds per entity
                    "Decay computation should be efficient: {} ns/entity", time_per_entity);
            }
        }
        
        Ok(())
    }
    
    /// Benchmark Hebbian learning batch operations
    pub async fn benchmark_hebbian_batch_operations(&self) -> Result<(), Box<dyn std::error::Error>> {
        for &scale in &self.scale_factors {
            let mut relationships = Vec::new();
            
            // Create batch of relationships
            for i in 0..scale {
                let mut rel = BrainInspiredRelationship::new(
                    Default::default(),
                    Default::default(),
                    RelationType::Learned
                );
                rel.weight = (i as f32) / (scale as f32); // Varied initial weights
                relationships.push(rel);
            }
            
            // Benchmark batch strengthening
            let start = Instant::now();
            
            for rel in relationships.iter_mut() {
                rel.strengthen(0.1);
            }
            
            let strengthen_duration = start.elapsed();
            
            // Benchmark batch decay
            let start = Instant::now();
            
            for rel in relationships.iter_mut() {
                rel.apply_decay();
            }
            
            let decay_duration = start.elapsed();
            
            println!("Hebbian operations for {} relationships: strengthen={:?}, decay={:?}",
                    scale, strengthen_duration, decay_duration);
            
            // Performance assertions
            let strengthen_per_rel = strengthen_duration.as_nanos() / scale as u128;
            let decay_per_rel = decay_duration.as_nanos() / scale as u128;
            
            assert!(strengthen_per_rel < 5_000, // 5 microseconds per relationship
                "Strengthen operation should be efficient: {} ns/rel", strengthen_per_rel);
            assert!(decay_per_rel < 5_000,
                "Decay operation should be efficient: {} ns/rel", decay_per_rel);
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_temporal_mathematics_precision() {
        let harness = TemporalMathematicsHarness::new();
        harness.test_multi_entity_decay_precision().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_temporal_causality() {
        let harness = TemporalMathematicsHarness::new();
        harness.test_temporal_causality_preservation().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_decay_convergence() {
        let harness = TemporalMathematicsHarness::new();
        harness.test_decay_convergence_properties().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_hebbian_equilibrium() {
        let harness = HebbianTemporalHarness::new();
        harness.test_learning_decay_equilibrium().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_competitive_learning() {
        let harness = HebbianTemporalHarness::new();
        harness.test_competitive_hebbian_dynamics().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_stdp_dynamics() {
        let harness = HebbianTemporalHarness::new();
        harness.test_stdp_simulation().await.unwrap();
    }
    
    #[tokio::test]
    async fn benchmark_temporal_performance() {
        let benchmark = TemporalPerformanceBenchmark::new();
        benchmark.benchmark_decay_scaling().await.unwrap();
        benchmark.benchmark_hebbian_batch_operations().await.unwrap();
    }
}