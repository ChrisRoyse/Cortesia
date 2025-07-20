/// Comprehensive temporal decay tests for BrainInspiredEntity
/// 
/// This module tests time-based activation patterns including:
/// - Exponential decay characteristics
/// - Activation accumulation over time
/// - Memory persistence and forgetting curves
/// - Real-time vs simulated temporal progression
/// - Edge cases with extreme time intervals

use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};
use llmkg::core::types::{EntityKey, AttributeValue};
use std::time::{SystemTime, Duration};
use std::thread;
use super::test_constants;
use super::test_helpers::*;

// ==================== Basic Temporal Decay Tests ====================

#[test]
fn test_activation_decay_over_time() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_INPUT.to_string(),
        EntityDirection::Input
    );
    
    // Initial activation
    let initial_activation = test_constants::ACTION_POTENTIAL;
    entity.activate(initial_activation, test_constants::STANDARD_DECAY_RATE);
    
    assert_float_eq(entity.activation_state, initial_activation, test_constants::ACTIVATION_EPSILON);
    
    // Wait for decay
    thread::sleep(Duration::from_millis(test_constants::DECAY_WAIT_MS));
    
    // Activate with zero to trigger decay calculation
    let decayed_activation = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    
    // Should be less than initial due to decay
    assert!(
        decayed_activation < initial_activation,
        "Expected decay: {} < {}",
        decayed_activation, initial_activation
    );
    
    // Should still be positive (exponential decay never reaches zero)
    assert!(decayed_activation > 0.0, "Decay should not reach zero");
}

#[test]
fn test_exponential_decay_characteristic() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_GATE.to_string(),
        EntityDirection::Gate
    );
    
    let initial_level = test_constants::SATURATION_LEVEL;
    let decay_rate = test_constants::FAST_DECAY_RATE;
    
    // Set initial state manually to test pure decay
    entity.activation_state = initial_level;
    entity.last_activation = SystemTime::now() - Duration::from_secs(1);
    
    // Calculate expected decay after 1 second
    let expected_decay = initial_level * (-decay_rate * 1.0).exp();
    
    let actual_decay = entity.activate(0.0, decay_rate);
    
    assert_float_eq(
        actual_decay,
        expected_decay,
        test_constants::LOOSE_TOLERANCE
    );
}

#[test]
fn test_different_decay_rates() {
    let decay_rates = vec![
        test_constants::NO_DECAY_RATE,
        test_constants::SLOW_DECAY_RATE,
        test_constants::STANDARD_DECAY_RATE,
        test_constants::FAST_DECAY_RATE,
    ];
    
    let initial_activation = test_constants::ACTION_POTENTIAL;
    let time_interval = Duration::from_millis(50);
    
    let mut results = Vec::new();
    
    for &decay_rate in &decay_rates {
        let mut entity = BrainInspiredEntity::new(
            format!("decay_test_{}", decay_rate),
            EntityDirection::Hidden
        );
        
        // Set initial state
        entity.activation_state = initial_activation;
        entity.last_activation = SystemTime::now() - time_interval;
        
        // Activate to trigger decay
        let final_activation = entity.activate(0.0, decay_rate);
        results.push(final_activation);
    }
    
    // Verify decay rate ordering: slower decay = higher final activation
    for i in 1..results.len() {
        assert!(
            results[i] <= results[i-1],
            "Faster decay should result in lower activation: rates[{}]={}, rates[{}]={}",
            i-1, decay_rates[i-1], i, decay_rates[i]
        );
    }
    
    // No decay should preserve activation exactly
    assert_float_eq(results[0], initial_activation, test_constants::ACTIVATION_EPSILON);
}

// ==================== Activation Accumulation Tests ====================

#[test]
fn test_activation_accumulation() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_OUTPUT.to_string(),
        EntityDirection::Output
    );
    
    let decay_rate = test_constants::SLOW_DECAY_RATE;
    let activation_increment = 0.2;
    let num_activations = 5;
    
    let mut expected_accumulation = 0.0;
    
    for i in 0..num_activations {
        let current_activation = entity.activate(activation_increment, decay_rate);
        
        // Should be accumulating (with some decay)
        if i > 0 {
            assert!(
                current_activation > activation_increment,
                "Activation should accumulate on iteration {}",
                i
            );
        }
        
        // Should not exceed saturation
        assert!(
            current_activation <= test_constants::SATURATION_LEVEL,
            "Activation should not exceed saturation"
        );
        
        // Small delay between activations
        thread::sleep(Duration::from_millis(1));
    }
    
    // Final activation should be significant accumulation
    assert!(
        entity.activation_state > activation_increment * 2.0,
        "Should show significant accumulation"
    );
}

#[test]
fn test_rapid_successive_activations() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_INPUT.to_string(),
        EntityDirection::Input
    );
    
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    let activation_level = 0.3;
    let rapid_count = 10;
    
    // Rapid successive activations (no delay)
    for _ in 0..rapid_count {
        entity.activate(activation_level, decay_rate);
    }
    
    // Should accumulate to near saturation due to rapid activation
    assert!(
        entity.activation_state > activation_level * 2.0,
        "Rapid activations should accumulate significantly"
    );
    
    // Should still respect saturation limit
    assert!(
        entity.activation_state <= test_constants::SATURATION_LEVEL,
        "Should not exceed saturation even with rapid activations"
    );
}

// ==================== Memory Persistence Tests ====================

#[test]
fn test_long_term_memory_persistence() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_HIDDEN.to_string(),
        EntityDirection::Hidden
    );
    
    // Strong initial activation with very slow decay
    let strong_activation = test_constants::SATURATION_LEVEL;
    let very_slow_decay = 0.001;
    
    entity.activate(strong_activation, very_slow_decay);
    
    // Simulate longer time passage (but still reasonable for tests)
    let long_interval = Duration::from_millis(200);
    thread::sleep(long_interval);
    
    // Check persistence
    let persistent_activation = entity.activate(0.0, very_slow_decay);
    
    // Should still retain significant activation
    assert!(
        persistent_activation > 0.5,
        "Long-term memory should persist: {}",
        persistent_activation
    );
    
    // Should be less than original due to some decay
    assert!(
        persistent_activation < strong_activation,
        "Should show some decay over time"
    );
}

#[test]
fn test_short_term_memory_forgetting() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_GATE.to_string(),
        EntityDirection::Gate
    );
    
    // Moderate activation with fast decay
    let moderate_activation = test_constants::MEDIUM_EXCITATORY;
    let fast_decay = test_constants::FAST_DECAY_RATE;
    
    entity.activate(moderate_activation, fast_decay);
    
    // Wait for significant decay
    thread::sleep(Duration::from_millis(100));
    
    // Check forgetting
    let forgotten_activation = entity.activate(0.0, fast_decay);
    
    // Should have decayed significantly
    assert!(
        forgotten_activation < moderate_activation * 0.5,
        "Short-term memory should decay quickly: {}",
        forgotten_activation
    );
}

// ==================== Simulated Time Tests ====================

#[test]
fn test_simulated_temporal_progression() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_OUTPUT.to_string(),
        EntityDirection::Output
    );
    
    let initial_activation = test_constants::ACTION_POTENTIAL;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    // Set initial state
    entity.activation_state = initial_activation;
    
    // Simulate different time intervals
    let time_intervals = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];
    let mut decay_values = Vec::new();
    
    for &interval in &time_intervals {
        // Manually set last_activation to simulate time passage
        entity.last_activation = SystemTime::now() - Duration::from_secs_f32(interval);
        entity.activation_state = initial_activation; // Reset for each test
        
        let decayed_value = entity.activate(0.0, decay_rate);
        decay_values.push(decayed_value);
    }
    
    // Verify decay progression: longer intervals = more decay
    for i in 1..decay_values.len() {
        assert!(
            decay_values[i] <= decay_values[i-1],
            "Longer time intervals should result in more decay: t[{}]={}, t[{}]={}",
            i-1, time_intervals[i-1], i, time_intervals[i]
        );
    }
    
    // First value (t=0) should equal initial activation
    assert_float_eq(decay_values[0], initial_activation, test_constants::ACTIVATION_EPSILON);
}

#[test]
fn test_zero_time_interval() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_INPUT.to_string(),
        EntityDirection::Input
    );
    
    let activation_level = test_constants::THRESHOLD_POTENTIAL;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    // Immediate successive activations (no time passage)
    let first_activation = entity.activate(activation_level, decay_rate);
    let immediate_second = entity.activate(activation_level, decay_rate);
    
    // Should accumulate without decay since no time passed
    assert!(
        immediate_second > first_activation,
        "Immediate activation should accumulate: {} -> {}",
        first_activation, immediate_second
    );
}

// ==================== Edge Case Temporal Tests ====================

#[test]
fn test_extreme_decay_rates() {
    let extreme_rates = vec![
        0.0,      // No decay
        0.001,    // Very slow
        10.0,     // Very fast
        100.0,    // Extremely fast
        f32::MAX, // Maximum possible
    ];
    
    for &rate in &extreme_rates {
        let mut entity = BrainInspiredEntity::new(
            format!("extreme_decay_{}", rate),
            EntityDirection::Hidden
        );
        
        // Set up for decay test
        entity.activation_state = test_constants::SATURATION_LEVEL;
        entity.last_activation = SystemTime::now() - Duration::from_millis(10);
        
        // Should not panic with extreme rates
        let result = entity.activate(0.0, rate);
        
        // Verify reasonable behavior
        assert_valid_activation(result);
        
        if rate == 0.0 {
            // No decay should preserve activation
            assert_float_eq(result, test_constants::SATURATION_LEVEL, test_constants::ACTIVATION_EPSILON);
        } else if rate > 10.0 {
            // Very fast decay should approach zero
            assert!(result < 0.1, "Very fast decay should approach zero");
        }
    }
}

#[test]
fn test_negative_decay_rate() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_GATE.to_string(),
        EntityDirection::Gate
    );
    
    let initial_activation = test_constants::THRESHOLD_POTENTIAL;
    let negative_decay = -0.1; // Negative decay rate (growth!)
    
    entity.activation_state = initial_activation;
    entity.last_activation = SystemTime::now() - Duration::from_millis(100);
    
    // Negative decay should cause growth
    let grown_activation = entity.activate(0.0, negative_decay);
    
    // Should be larger than initial (exponential growth)
    assert!(
        grown_activation > initial_activation,
        "Negative decay should cause growth: {} -> {}",
        initial_activation, grown_activation
    );
    
    // Should still be clamped to saturation
    assert!(
        grown_activation <= test_constants::SATURATION_LEVEL,
        "Growth should be clamped to saturation"
    );
}

#[test]
fn test_very_long_time_intervals() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_OUTPUT.to_string(),
        EntityDirection::Output
    );
    
    let initial_activation = test_constants::SATURATION_LEVEL;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    // Simulate very long time passage
    entity.activation_state = initial_activation;
    entity.last_activation = SystemTime::now() - Duration::from_secs(3600); // 1 hour ago
    
    let heavily_decayed = entity.activate(0.0, decay_rate);
    
    // Should decay significantly but not to exactly zero
    assert!(heavily_decayed < 0.01, "Should decay heavily over long periods");
    assert!(heavily_decayed > 0.0, "Should never reach exactly zero");
}

// ==================== Temporal Pattern Tests ====================

#[test]
fn test_periodic_activation_pattern() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_INPUT.to_string(),
        EntityDirection::Input
    );
    
    let activation_level = 0.4;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    let period_ms = 20;
    let num_periods = 5;
    
    let mut activation_history = Vec::new();
    
    for i in 0..num_periods {
        // Activate
        let current_activation = entity.activate(activation_level, decay_rate);
        activation_history.push(current_activation);
        
        // Wait for partial decay
        thread::sleep(Duration::from_millis(period_ms));
    }
    
    // Should see oscillating pattern approaching equilibrium
    for i in 1..activation_history.len() {
        // Each activation should be higher than just the increment
        assert!(
            activation_history[i] > activation_level,
            "Period {} should show accumulation",
            i
        );
    }
    
    // Later activations should approach equilibrium
    if activation_history.len() >= 3 {
        let stability_threshold = 0.1;
        let late_variation = (activation_history[num_periods - 1] - activation_history[num_periods - 2]).abs();
        
        assert!(
            late_variation < stability_threshold,
            "Should approach stability: variation = {}",
            late_variation
        );
    }
}

#[test]
fn test_burst_activation_pattern() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_HIDDEN.to_string(),
        EntityDirection::Hidden
    );
    
    let burst_size = 5;
    let burst_activation = 0.2;
    let decay_rate = test_constants::FAST_DECAY_RATE;
    
    // Rapid burst of activations
    for _ in 0..burst_size {
        entity.activate(burst_activation, decay_rate);
    }
    let burst_peak = entity.activation_state;
    
    // Wait for decay
    thread::sleep(Duration::from_millis(50));
    
    // Check post-burst level
    let post_burst = entity.activate(0.0, decay_rate);
    
    // Peak should be significant accumulation
    assert!(
        burst_peak > burst_activation * 2.0,
        "Burst should create significant peak: {}",
        burst_peak
    );
    
    // Post-burst should show decay
    assert!(
        post_burst < burst_peak,
        "Should decay after burst: {} -> {}",
        burst_peak, post_burst
    );
}

// ==================== Time Measurement Accuracy Tests ====================

#[test]
fn test_elapsed_time_measurement_accuracy() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_GATE.to_string(),
        EntityDirection::Gate
    );
    
    let activation_level = test_constants::ACTION_POTENTIAL;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    // Record start time
    let start_time = SystemTime::now();
    entity.activate(activation_level, decay_rate);
    
    // Wait precisely measured interval
    let wait_duration = Duration::from_millis(100);
    thread::sleep(wait_duration);
    
    // Activate again and check timing
    let measured_start = SystemTime::now();
    entity.activate(0.0, decay_rate);
    let measured_duration = measured_start.duration_since(start_time).unwrap();
    
    // Should be approximately the expected duration (within reasonable tolerance)
    let tolerance = Duration::from_millis(20); // 20ms tolerance for system timing
    let duration_diff = if measured_duration > wait_duration {
        measured_duration - wait_duration
    } else {
        wait_duration - measured_duration
    };
    
    assert!(
        duration_diff < tolerance,
        "Time measurement should be accurate: expected ~{:?}, got {:?}",
        wait_duration, measured_duration
    );
}

// ==================== Concurrent Temporal Access Tests ====================

#[test]
fn test_temporal_state_consistency() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_OUTPUT.to_string(),
        EntityDirection::Output
    );
    
    let activation_level = test_constants::MEDIUM_EXCITATORY;
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    // Multiple rapid activations to test timestamp consistency
    let num_activations = 10;
    let mut timestamps = Vec::new();
    
    for _ in 0..num_activations {
        entity.activate(activation_level, decay_rate);
        timestamps.push(entity.last_activation);
        
        // Tiny delay to ensure different timestamps
        thread::sleep(Duration::from_nanos(1000));
    }
    
    // Timestamps should be monotonically increasing
    for i in 1..timestamps.len() {
        assert!(
            timestamps[i] >= timestamps[i-1],
            "Timestamps should be monotonic: {:?} -> {:?}",
            timestamps[i-1], timestamps[i]
        );
    }
}

// ==================== Mathematical Property Tests ====================

#[test]
fn test_exponential_decay_mathematical_properties() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_INPUT.to_string(),
        EntityDirection::Input
    );
    
    let initial_value = 1.0;
    let decay_rate = 0.1;
    let time_constant = 1.0 / decay_rate; // Should decay to 1/e at this time
    
    // Set up entity state
    entity.activation_state = initial_value;
    entity.last_activation = SystemTime::now() - Duration::from_secs_f32(time_constant);
    
    let decayed_value = entity.activate(0.0, decay_rate);
    
    // Should be approximately 1/e ≈ 0.368 of original
    let expected_value = initial_value / std::f32::consts::E;
    
    assert_float_eq(
        decayed_value,
        expected_value,
        0.05 // Generous tolerance for floating point and timing
    );
}

#[test]
fn test_half_life_calculation() {
    let mut entity = BrainInspiredEntity::new(
        test_constants::TEST_CONCEPT_HIDDEN.to_string(),
        EntityDirection::Hidden
    );
    
    let initial_value = 1.0;
    let decay_rate = 0.693; // ln(2), should give half-life of 1 second
    
    entity.activation_state = initial_value;
    entity.last_activation = SystemTime::now() - Duration::from_secs(1);
    
    let half_life_value = entity.activate(0.0, decay_rate);
    
    // Should be approximately half the original value
    assert_float_eq(
        half_life_value,
        initial_value * 0.5,
        0.05
    );
}