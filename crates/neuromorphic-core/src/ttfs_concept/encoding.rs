//! TTFS encoding algorithms for converting features to spike patterns

use super::spike_pattern::{SpikePattern, SpikeEvent};
use std::time::Duration;

/// Configuration for TTFS encoding
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    /// Maximum spike time (milliseconds)
    pub max_spike_time_ms: u64,
    
    /// Number of neurons per feature
    pub neurons_per_feature: u32,
    
    /// Time constant tau (milliseconds)
    pub tau_ms: f32,
    
    /// Minimum feature strength to encode
    pub min_feature_threshold: f32,
    
    /// Base frequency for encoding (Hz)
    pub base_frequency: f32,
    
    /// Frequency modulation depth
    pub frequency_modulation: f32,
    
    /// Enable population coding
    pub use_population_coding: bool,
    
    /// Enable adaptive neuron allocation based on feature importance
    pub use_adaptive_allocation: bool,
    
    /// Temporal dependency strength for sequence encoding (0.0 to 1.0)
    pub temporal_dependency_strength: f32,
    
    /// Maximum temporal dependency window (milliseconds)
    pub max_dependency_window_ms: u64,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            max_spike_time_ms: 100,
            neurons_per_feature: 3,
            tau_ms: 20.0,
            min_feature_threshold: 0.1,
            base_frequency: 40.0,
            frequency_modulation: 0.5,
            use_population_coding: true,
            use_adaptive_allocation: false,
            temporal_dependency_strength: 0.3,
            max_dependency_window_ms: 50,
        }
    }
}

/// TTFS encoder for converting semantic features to spike patterns
pub struct TTFSEncoder {
    config: EncodingConfig,
}

impl Default for TTFSEncoder {
    fn default() -> Self {
        Self::new(EncodingConfig::default())
    }
}

impl TTFSEncoder {
    /// Create new encoder with custom config
    pub fn new(config: EncodingConfig) -> Self {
        Self {
            config,
        }
    }
    
    /// Encode feature vector into spike pattern
    pub fn encode(&self, features: &[f32]) -> SpikePattern {
        let mut events = Vec::new();
        
        for (feature_idx, &feature_value) in features.iter().enumerate() {
            // Skip weak features
            if feature_value.abs() < self.config.min_feature_threshold {
                continue;
            }
            
            // Encode using population coding or single neuron
            if self.config.use_population_coding {
                let feature_events = self.encode_feature_population(
                    feature_idx,
                    feature_value
                );
                events.extend(feature_events);
            } else {
                if let Some(event) = self.encode_feature_single(
                    feature_idx as u32,
                    feature_value
                ) {
                    events.push(event);
                }
            }
        }
        
        // Sort events by timestamp for proper ordering
        events.sort_by_key(|e| e.timestamp.as_nanos());
        
        SpikePattern::new(events)
    }
    
    /// Encode a single feature using population coding
    fn encode_feature_population(&self, 
                                feature_idx: usize, 
                                feature_value: f32) -> Vec<SpikeEvent> {
        let mut events = Vec::new();
        
        // Determine number of neurons based on adaptive allocation
        let num_neurons = if self.config.use_adaptive_allocation {
            self.calculate_adaptive_neuron_count(feature_value)
        } else {
            self.config.neurons_per_feature
        };
        
        // Distribute encoding across multiple neurons
        for i in 0..num_neurons {
            let neuron_id = (feature_idx as u32 * self.config.neurons_per_feature) + i;
            
            // Add variation to each neuron's encoding
            let jitter = if self.config.use_adaptive_allocation {
                // Adaptive jitter based on feature importance
                let importance_factor = feature_value * feature_value; // Quadratic scaling
                (i as f32 / num_neurons as f32) * 0.15 * importance_factor
            } else {
                // Standard jitter
                (i as f32 / self.config.neurons_per_feature as f32) * 0.2
            };
            
            let adjusted_value = (feature_value + jitter).clamp(0.0, 1.0);
            
            if let Some(event) = self.encode_feature_single(neuron_id, adjusted_value) {
                events.push(event);
            }
        }
        
        events
    }
    
    /// Calculate adaptive neuron count based on feature importance
    fn calculate_adaptive_neuron_count(&self, feature_value: f32) -> u32 {
        let importance = feature_value.abs();
        
        // More important features get more neurons (1 to 2x base allocation)
        let multiplier = 1.0 + importance;
        let adaptive_count = (self.config.neurons_per_feature as f32 * multiplier) as u32;
        
        // Clamp to reasonable bounds
        adaptive_count.clamp(1, self.config.neurons_per_feature * 2)
    }
    
    /// Encode a single feature value to a spike event
    fn encode_feature_single(&self, neuron_id: u32, feature_value: f32) -> Option<SpikeEvent> {
        // Ensure feature is in valid range
        let clamped_value = feature_value.clamp(0.0, 1.0);
        
        if clamped_value < self.config.min_feature_threshold {
            return None;
        }
        
        // TTFS encoding: stronger features spike earlier
        // t = -τ * ln(feature_strength)
        let spike_time_ms = if clamped_value >= 0.99 {
            0.0 // Maximum strength = immediate spike
        } else {
            -self.config.tau_ms * (clamped_value).ln()
        };
        
        // Clamp to maximum time
        let spike_time_ms = spike_time_ms.clamp(0.0, self.config.max_spike_time_ms as f32);
        
        // Calculate frequency based on feature strength
        let frequency = self.calculate_frequency(clamped_value);
        
        Some(SpikeEvent {
            neuron_id,
            timestamp: Duration::from_micros((spike_time_ms * 1000.0) as u64),
            amplitude: clamped_value,
            frequency,
        })
    }
    
    /// Calculate spike frequency based on feature strength
    fn calculate_frequency(&self, feature_value: f32) -> f32 {
        // Base frequency modulated by feature strength
        let modulation = self.config.frequency_modulation * feature_value;
        self.config.base_frequency * (1.0 + modulation)
    }
    
    /// Encode with temporal context (for sequences)
    pub fn encode_temporal(&self, 
                          features: &[f32], 
                          time_offset: Duration) -> SpikePattern {
        let mut pattern = self.encode(features);
        
        // Shift all spikes by time offset
        for event in &mut pattern.events {
            event.timestamp += time_offset;
        }
        
        pattern.duration += time_offset;
        pattern
    }
    
    /// Encode a sequence with temporal dependencies
    pub fn encode_sequence(&self, 
                          feature_sequence: &[Vec<f32>], 
                          step_duration: Duration) -> SpikePattern {
        let mut all_events = Vec::new();
        let mut previous_pattern: Option<SpikePattern> = None;
        
        for (step_idx, features) in feature_sequence.iter().enumerate() {
            let base_offset = step_duration * step_idx as u32;
            let mut step_pattern = self.encode(features);
            
            // Apply temporal dependencies if enabled
            if self.config.temporal_dependency_strength > 0.0 && previous_pattern.is_some() {
                step_pattern = self.apply_temporal_dependencies(
                    step_pattern, 
                    previous_pattern.as_ref().unwrap(),
                    base_offset
                );
            } else {
                // Simple time offset
                for event in &mut step_pattern.events {
                    event.timestamp += base_offset;
                }
            }
            
            all_events.extend(step_pattern.events.clone());
            previous_pattern = Some(step_pattern);
        }
        
        // Sort all events by timestamp
        all_events.sort_by_key(|e| e.timestamp.as_nanos());
        
        SpikePattern::new(all_events)
    }
    
    /// Apply temporal dependencies between sequence steps
    fn apply_temporal_dependencies(&self, 
                                  mut current_pattern: SpikePattern,
                                  previous_pattern: &SpikePattern,
                                  base_offset: Duration) -> SpikePattern {
        let dependency_strength = self.config.temporal_dependency_strength;
        let max_window = Duration::from_millis(self.config.max_dependency_window_ms);
        
        // Find recent spikes from previous pattern that could influence current pattern
        let cutoff_time = base_offset.saturating_sub(max_window);
        let recent_spikes: Vec<_> = previous_pattern.events.iter()
            .filter(|event| event.timestamp >= cutoff_time)
            .collect();
        
        // Adjust current pattern based on temporal dependencies
        for current_event in &mut current_pattern.events {
            let mut dependency_adjustment = Duration::ZERO;
            
            // Find the most relevant previous spike (same or nearby neuron)
            for &previous_event in &recent_spikes {
                let neuron_similarity = self.calculate_neuron_similarity(
                    current_event.neuron_id, 
                    previous_event.neuron_id
                );
                
                if neuron_similarity > 0.0 {
                    // Calculate temporal influence
                    let time_diff = base_offset.saturating_sub(previous_event.timestamp);
                    let time_factor = 1.0 - (time_diff.as_millis() as f32 / max_window.as_millis() as f32);
                    
                    // Dependency creates slight advancement (spikes come earlier)
                    let influence = dependency_strength * neuron_similarity * time_factor;
                    let adjustment_ms = influence * 5.0; // Max 5ms advancement
                    
                    dependency_adjustment = dependency_adjustment.saturating_add(
                        Duration::from_micros((adjustment_ms * 1000.0) as u64)
                    );
                }
            }
            
            // Apply base offset and dependency adjustment
            current_event.timestamp = base_offset + current_event.timestamp - dependency_adjustment;
        }
        
        current_pattern
    }
    
    /// Calculate similarity between neuron IDs for temporal dependencies
    fn calculate_neuron_similarity(&self, neuron_a: u32, neuron_b: u32) -> f32 {
        if neuron_a == neuron_b {
            return 1.0; // Same neuron
        }
        
        // Calculate feature indices
        let feature_a = neuron_a / self.config.neurons_per_feature;
        let feature_b = neuron_b / self.config.neurons_per_feature;
        
        if feature_a == feature_b {
            return 0.8; // Same feature, different population neuron
        }
        
        // Different features have weak similarity based on proximity
        let feature_distance = (feature_a as i32 - feature_b as i32).abs() as f32;
        (1.0 / (1.0 + feature_distance * 0.5)).max(0.1)
    }
    
    /// Validate that a spike pattern meets biological constraints
    pub fn validate_pattern(&self, pattern: &SpikePattern) -> Result<(), EncodingError> {
        // Check minimum inter-spike interval (refractory period)
        const MIN_ISI_MS: u64 = 2; // 2ms refractory period
        
        let mut events_by_neuron: std::collections::HashMap<u32, Vec<&SpikeEvent>> = 
            std::collections::HashMap::new();
        
        for event in &pattern.events {
            events_by_neuron.entry(event.neuron_id)
                .or_insert_with(Vec::new)
                .push(event);
        }
        
        for (neuron_id, neuron_events) in events_by_neuron {
            let mut sorted_events = neuron_events.clone();
            sorted_events.sort_by_key(|e| e.timestamp);
            
            for window in sorted_events.windows(2) {
                let isi = window[1].timestamp - window[0].timestamp;
                if isi < Duration::from_millis(MIN_ISI_MS) {
                    return Err(EncodingError::RefractoryViolation {
                        neuron_id,
                        isi,
                        minimum: Duration::from_millis(MIN_ISI_MS),
                    });
                }
            }
        }
        
        // Check maximum spike rate
        if pattern.density > 0.5 { // Max 0.5 spikes per ms
            return Err(EncodingError::ExcessiveSpikeRate(pattern.density));
        }
        
        Ok(())
    }
}

/// Clone implementation for TTFSEncoder to support batch building
impl Clone for TTFSEncoder {
    fn clone(&self) -> Self {
        Self::new(self.config.clone())
    }
}

/// Errors that can occur during encoding
#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    #[error("Refractory period violation for neuron {neuron_id}: ISI {isi:?} < minimum {minimum:?}")]
    RefractoryViolation {
        neuron_id: u32,
        isi: Duration,
        minimum: Duration,
    },
    
    #[error("Excessive spike rate: {0} spikes/ms exceeds biological limits")]
    ExcessiveSpikeRate(f32),
    
    #[error("Invalid feature vector: {0}")]
    InvalidFeatures(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ttfs_encoding_formula() {
        // Test with single neuron encoding to get exact formula results
        let config = EncodingConfig {
            use_population_coding: false,
            ..Default::default()
        };
        let encoder = TTFSEncoder::new(config);
        
        // Test edge cases
        let pattern1 = encoder.encode(&[1.0]); // Maximum strength
        assert_eq!(pattern1.first_spike_time(), Some(Duration::from_micros(0)));
        
        let pattern2 = encoder.encode(&[0.5]); // Medium strength
        let spike_time = pattern2.first_spike_time().unwrap();
        // Should be around -20 * ln(0.5) ≈ 13.86ms = 13860μs
        let spike_time_ms = spike_time.as_micros() as f32 / 1000.0;
        assert!(spike_time_ms >= 13.0 && spike_time_ms <= 15.0, 
                "Expected 13-15ms, got {}ms", spike_time_ms);
        
        let pattern3 = encoder.encode(&[0.05]); // Below threshold
        assert_eq!(pattern3.events.len(), 0);
        
        // Test population coding separately
        let pop_encoder = TTFSEncoder::default();
        let pop_pattern = pop_encoder.encode(&[0.5]);
        // With population coding + jitter, first spike should be earlier
        let pop_spike_time_ms = pop_pattern.first_spike_time().unwrap().as_micros() as f32 / 1000.0;
        assert!(pop_spike_time_ms < spike_time_ms, 
                "Population coding should produce earlier first spike");
    }
    
    #[test]
    fn test_population_coding() {
        let config = EncodingConfig {
            neurons_per_feature: 3,
            use_population_coding: true,
            ..Default::default()
        };
        
        let encoder = TTFSEncoder::new(config);
        let pattern = encoder.encode(&[0.8, 0.6]);
        
        // Should have 3 neurons per feature * 2 features = 6 spikes
        assert_eq!(pattern.events.len(), 6);
        
        // Check neuron IDs are correct
        let neuron_ids: std::collections::HashSet<_> = pattern.events.iter()
            .map(|e| e.neuron_id)
            .collect();
        assert_eq!(neuron_ids.len(), 6);
    }
    
    #[test]
    fn test_temporal_encoding() {
        let encoder = TTFSEncoder::default();
        let base_pattern = encoder.encode(&[0.7, 0.8]);
        
        let offset = Duration::from_millis(50);
        let temporal_pattern = encoder.encode_temporal(&[0.7, 0.8], offset);
        
        // All spikes should be shifted by offset
        for (base_event, temporal_event) in base_pattern.events.iter()
            .zip(temporal_pattern.events.iter()) {
            assert_eq!(temporal_event.timestamp, base_event.timestamp + offset);
        }
    }
    
    #[test]
    fn test_biological_constraints() {
        let encoder = TTFSEncoder::default();
        
        // Valid pattern - use single neuron per feature to avoid potential conflicts
        let config = EncodingConfig {
            use_population_coding: false,
            ..Default::default()
        };
        let single_encoder = TTFSEncoder::new(config);
        let valid_pattern = single_encoder.encode(&[0.7, 0.8, 0.9]);
        
        let validation_result = encoder.validate_pattern(&valid_pattern);
        if let Err(ref e) = validation_result {
            println!("Validation error: {:?}", e);
            println!("Pattern: {:?}", valid_pattern);
        }
        assert!(validation_result.is_ok());
        
        // Create pattern with refractory violation
        let invalid_pattern = SpikePattern::new(vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(0),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(1), // Too close!
                amplitude: 1.0,
                frequency: 50.0,
            },
        ]);
        
        assert!(encoder.validate_pattern(&invalid_pattern).is_err());
    }
    
    #[test]
    fn test_frequency_modulation() {
        let encoder = TTFSEncoder::default();
        
        let weak_pattern = encoder.encode(&[0.2]);
        let strong_pattern = encoder.encode(&[0.9]);
        
        // Stronger features should have higher frequency
        assert!(strong_pattern.events[0].frequency > weak_pattern.events[0].frequency);
    }
    
    #[test]
    fn test_adaptive_population_coding() {
        let config = EncodingConfig {
            neurons_per_feature: 3,
            use_population_coding: true,
            use_adaptive_allocation: true,
            ..Default::default()
        };
        
        let encoder = TTFSEncoder::new(config.clone());
        
        // Test with different feature strengths
        let weak_pattern = encoder.encode(&[0.3]); // Low importance
        let strong_pattern = encoder.encode(&[0.9]); // High importance
        
        // Strong features should get more neurons allocated
        // Due to adaptive allocation, strong features get up to 2x neurons
        assert!(strong_pattern.events.len() >= weak_pattern.events.len(),
                "Strong features should get at least as many neurons as weak features");
        
        // Test adaptive jitter calculation
        let pattern = encoder.encode(&[0.8, 0.2]);
        assert!(pattern.events.len() > 0, "Should have events for both features");
        
        // Verify neuron count calculation
        let weak_count = encoder.calculate_adaptive_neuron_count(0.3);
        let strong_count = encoder.calculate_adaptive_neuron_count(0.9);
        assert!(strong_count >= weak_count, "Strong features should get more neurons");
        assert!(strong_count <= config.neurons_per_feature * 2, "Should not exceed 2x base allocation");
    }
    
    #[test]
    fn test_sequence_encoding_without_dependencies() {
        let config = EncodingConfig {
            temporal_dependency_strength: 0.0, // Disable dependencies
            ..Default::default()
        };
        let encoder = TTFSEncoder::new(config);
        
        let sequence = vec![
            vec![0.8, 0.6],  // Step 1
            vec![0.5, 0.9],  // Step 2
            vec![0.7, 0.4],  // Step 3
        ];
        
        let step_duration = Duration::from_millis(20);
        let pattern = encoder.encode_sequence(&sequence, step_duration);
        
        // Should have events from all steps
        assert!(pattern.events.len() > 0, "Should have events from sequence");
        
        // Events should be spread across time
        let first_spike = pattern.first_spike_time().unwrap();
        let last_spike = pattern.last_spike_time().unwrap();
        assert!(last_spike > first_spike, "Should have temporal spread");
        
        // Check that events are properly spaced
        let expected_min_duration = step_duration * 2; // 3 steps = 2 intervals
        assert!(pattern.duration >= expected_min_duration, 
                "Pattern duration should span multiple steps");
    }
    
    #[test]
    fn test_sequence_encoding_with_dependencies() {
        let config = EncodingConfig {
            temporal_dependency_strength: 0.5, // Enable dependencies
            max_dependency_window_ms: 30,
            use_population_coding: false, // Simpler for testing
            ..Default::default()
        };
        let encoder = TTFSEncoder::new(config.clone());
        
        let sequence = vec![
            vec![0.8],  // Step 1: Strong feature
            vec![0.8],  // Step 2: Same strong feature (should show dependency)
        ];
        
        let step_duration = Duration::from_millis(25);
        let dependent_pattern = encoder.encode_sequence(&sequence, step_duration);
        
        // Compare with non-dependent encoding
        let no_dep_config = EncodingConfig {
            temporal_dependency_strength: 0.0,
            use_population_coding: false,
            ..config
        };
        let no_dep_encoder = TTFSEncoder::new(no_dep_config);
        let no_dep_pattern = no_dep_encoder.encode_sequence(&sequence, step_duration);
        
        // Dependent encoding should show temporal influence
        assert_eq!(dependent_pattern.events.len(), 2, "Should have events from both steps");
        assert_eq!(no_dep_pattern.events.len(), 2, "Should have events from both steps");
        
        // Find the second spike in each pattern
        let dep_second_spike = dependent_pattern.events.iter()
            .find(|e| e.timestamp >= step_duration)
            .expect("Should have second spike");
        let no_dep_second_spike = no_dep_pattern.events.iter()
            .find(|e| e.timestamp >= step_duration)
            .expect("Should have second spike");
        
        // Dependent spike should come earlier due to temporal priming
        assert!(dep_second_spike.timestamp <= no_dep_second_spike.timestamp,
                "Dependent encoding should show temporal advancement");
    }
    
    #[test]
    fn test_neuron_similarity_calculation() {
        let encoder = TTFSEncoder::default();
        
        // Same neuron
        assert_eq!(encoder.calculate_neuron_similarity(5, 5), 1.0);
        
        // Same feature, different population neuron (neurons 0,1,2 for feature 0)
        assert_eq!(encoder.calculate_neuron_similarity(0, 1), 0.8);
        assert_eq!(encoder.calculate_neuron_similarity(1, 2), 0.8);
        
        // Different features
        let sim_adjacent = encoder.calculate_neuron_similarity(0, 3); // Feature 0 vs Feature 1
        let sim_distant = encoder.calculate_neuron_similarity(0, 6); // Feature 0 vs Feature 2
        
        assert!(sim_adjacent > sim_distant, "Adjacent features should be more similar");
        assert!(sim_adjacent >= 0.1, "Should have minimum similarity");
        assert!(sim_distant >= 0.1, "Should have minimum similarity");
    }
    
    #[test]
    fn test_temporal_dependencies_boundary_conditions() {
        let config = EncodingConfig {
            temporal_dependency_strength: 1.0, // Maximum dependency
            max_dependency_window_ms: 10, // Very small window
            use_population_coding: false,
            ..Default::default()
        };
        let encoder = TTFSEncoder::new(config);
        
        // Test with steps outside dependency window
        let sequence = vec![
            vec![0.8],
            vec![0.8],
        ];
        
        let long_step = Duration::from_millis(50); // Longer than window
        let pattern = encoder.encode_sequence(&sequence, long_step);
        
        // Should still work but with minimal dependency effect
        assert_eq!(pattern.events.len(), 2, "Should handle long step durations");
        
        // Test empty sequence
        let empty_pattern = encoder.encode_sequence(&[], Duration::from_millis(10));
        assert_eq!(empty_pattern.events.len(), 0, "Should handle empty sequences");
        
        // Test single step sequence
        let single_pattern = encoder.encode_sequence(&[vec![0.5]], Duration::from_millis(10));
        assert!(single_pattern.events.len() > 0, "Should handle single step sequences");
    }
}