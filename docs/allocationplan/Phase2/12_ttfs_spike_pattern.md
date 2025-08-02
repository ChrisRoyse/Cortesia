# Task 12: TTFS Spike Pattern Structure

## Metadata
- **Micro-Phase**: 2.12
- **Duration**: 15-20 minutes
- **Dependencies**: Task 11 (SpikeEvent)
- **Output**: `src/ttfs_encoding/ttfs_spike_pattern.rs`

## Description
Create the TTFSSpikePattern structure that represents a complete Time-to-First-Spike encoded pattern with biological timing precision. This is the core data structure for neuromorphic concept encoding.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::SpikeEvent;
    use std::time::Duration;

    #[test]
    fn test_ttfs_spike_pattern_creation() {
        let concept_id = ConceptId::new("elephant");
        let first_spike_time = Duration::from_micros(500);
        let spike_sequence = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(2), Duration::from_micros(750), 0.8),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(1), 0.7),
        ];
        
        let pattern = TTFSSpikePattern::new(
            concept_id.clone(),
            first_spike_time,
            spike_sequence.clone(),
            Duration::from_millis(2)
        );
        
        assert_eq!(pattern.concept_id(), &concept_id);
        assert_eq!(pattern.first_spike_time(), first_spike_time);
        assert_eq!(pattern.spike_sequence().len(), 3);
        assert_eq!(pattern.total_duration(), Duration::from_millis(2));
    }
    
    #[test]
    fn test_ttfs_timing_validation() {
        let pattern = create_valid_ttfs_pattern();
        assert!(pattern.is_valid_ttfs());
        assert!(pattern.is_biologically_plausible());
        assert!(pattern.timing_precision_ns() < 100_000); // < 0.1ms precision
    }
    
    #[test]
    fn test_spike_sequence_ordering() {
        let mut spikes = vec![
            SpikeEvent::new(NeuronId(3), Duration::from_millis(2), 0.7),
            SpikeEvent::new(NeuronId(1), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(1), 0.8),
        ];
        
        let pattern = TTFSSpikePattern::new(
            ConceptId::new("test"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3)
        );
        
        // Should be sorted by timing
        let sorted_spikes = pattern.spike_sequence();
        assert!(sorted_spikes[0].timing < sorted_spikes[1].timing);
        assert!(sorted_spikes[1].timing < sorted_spikes[2].timing);
    }
    
    #[test]
    fn test_refractory_period_compliance() {
        let pattern = create_valid_ttfs_pattern();
        assert!(pattern.check_refractory_compliance());
        
        // Create pattern with violations
        let violation_spikes = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_micros(510), 0.8), // Too soon
        ];
        
        let violation_pattern = TTFSSpikePattern::new(
            ConceptId::new("violation"),
            Duration::from_micros(500),
            violation_spikes,
            Duration::from_millis(1)
        );
        
        assert!(!violation_pattern.check_refractory_compliance());
    }
    
    #[test]
    fn test_spike_pattern_encoding_features() {
        let pattern = create_valid_ttfs_pattern();
        
        let features = pattern.extract_neural_features();
        assert!(!features.is_empty());
        assert!(features.len() <= 128); // Standard feature vector size
        
        let encoding_confidence = pattern.encoding_confidence();
        assert!(encoding_confidence >= 0.0 && encoding_confidence <= 1.0);
    }
    
    #[test]
    fn test_temporal_compression() {
        let long_pattern = create_long_ttfs_pattern();
        let compressed = long_pattern.compress_temporal_window(Duration::from_millis(5));
        
        assert!(compressed.total_duration() <= Duration::from_millis(5));
        assert!(compressed.spike_sequence().len() <= long_pattern.spike_sequence().len());
        assert!(compressed.is_valid_ttfs());
    }
    
    // Helper functions
    fn create_valid_ttfs_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(1), 0.8),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(2), 0.7),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("valid_test"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3)
        )
    }
    
    fn create_long_ttfs_pattern() -> TTFSSpikePattern {
        let spikes: Vec<_> = (0..20)
            .map(|i| SpikeEvent::new(
                NeuronId(i), 
                Duration::from_millis(i as u64 * 500), 
                0.8 - (i as f32 * 0.02)
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new("long_test"),
            Duration::from_millis(0),
            spikes,
            Duration::from_millis(10000)
        )
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::SpikeEvent;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::collections::HashMap;

/// Unique identifier for concepts in the neuromorphic system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConceptId(String);

impl ConceptId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ConceptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// TTFS spike pattern representing a neuromorphically encoded concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTFSSpikePattern {
    /// Unique identifier of the encoded concept
    concept_id: ConceptId,
    
    /// Time-to-first-spike (the defining characteristic)
    first_spike_time: Duration,
    
    /// Sequence of spike events (sorted by timing)
    spike_sequence: Vec<SpikeEvent>,
    
    /// Total duration of the spike pattern
    total_duration: Duration,
    
    /// Encoding confidence score (0.0-1.0)
    encoding_confidence: f32,
    
    /// Neural features extracted for ruv-FANN processing
    neural_features: Vec<f32>,
    
    /// Refractory period compliance flag
    refractory_compliant: bool,
    
    /// Metadata for the spike pattern
    metadata: HashMap<String, String>,
}

impl TTFSSpikePattern {
    /// Create new TTFS spike pattern
    pub fn new(
        concept_id: ConceptId,
        first_spike_time: Duration,
        mut spike_sequence: Vec<SpikeEvent>,
        total_duration: Duration,
    ) -> Self {
        // Sort spike sequence by timing
        spike_sequence.sort_by(|a, b| a.timing.cmp(&b.timing));
        
        // Extract neural features
        let neural_features = Self::extract_features_from_spikes(&spike_sequence);
        
        // Check refractory compliance
        let refractory_compliant = Self::check_refractory_compliance_static(&spike_sequence);
        
        // Calculate encoding confidence
        let encoding_confidence = Self::calculate_encoding_confidence_static(
            &spike_sequence, 
            total_duration, 
            first_spike_time
        );
        
        Self {
            concept_id,
            first_spike_time,
            spike_sequence,
            total_duration,
            encoding_confidence,
            neural_features,
            refractory_compliant,
            metadata: HashMap::new(),
        }
    }
    
    /// Get concept identifier
    pub fn concept_id(&self) -> &ConceptId {
        &self.concept_id
    }
    
    /// Get time-to-first-spike
    pub fn first_spike_time(&self) -> Duration {
        self.first_spike_time
    }
    
    /// Get spike sequence
    pub fn spike_sequence(&self) -> &[SpikeEvent] {
        &self.spike_sequence
    }
    
    /// Get total duration
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }
    
    /// Get encoding confidence
    pub fn encoding_confidence(&self) -> f32 {
        self.encoding_confidence
    }
    
    /// Get neural features for network processing
    pub fn extract_neural_features(&self) -> &[f32] {
        &self.neural_features
    }
    
    /// Check if pattern is valid TTFS encoding
    pub fn is_valid_ttfs(&self) -> bool {
        // Must have at least one spike
        if self.spike_sequence.is_empty() {
            return false;
        }
        
        // First spike must match first_spike_time
        if self.spike_sequence[0].timing != self.first_spike_time {
            return false;
        }
        
        // All spikes must be within total duration
        if self.spike_sequence.iter().any(|spike| spike.timing > self.total_duration) {
            return false;
        }
        
        // Must be sorted by timing
        for window in self.spike_sequence.windows(2) {
            if window[0].timing > window[1].timing {
                return false;
            }
        }
        
        true
    }
    
    /// Check if pattern follows biological constraints
    pub fn is_biologically_plausible(&self) -> bool {
        // Check individual spike validity
        if !self.spike_sequence.iter().all(|spike| spike.is_biologically_valid()) {
            return false;
        }
        
        // Check total duration is reasonable (< 100ms for concepts)
        if self.total_duration > Duration::from_millis(100) {
            return false;
        }
        
        // Check first spike is within reasonable time (< 10ms)
        if self.first_spike_time > Duration::from_millis(10) {
            return false;
        }
        
        // Check refractory period compliance
        self.refractory_compliant
    }
    
    /// Get timing precision in nanoseconds
    pub fn timing_precision_ns(&self) -> u128 {
        if self.spike_sequence.len() < 2 {
            return 0;
        }
        
        // Find minimum time difference between consecutive spikes
        let min_diff = self.spike_sequence.windows(2)
            .map(|window| window[1].timing - window[0].timing)
            .min()
            .unwrap_or(Duration::new(0, 0));
        
        min_diff.as_nanos()
    }
    
    /// Check refractory period compliance
    pub fn check_refractory_compliance(&self) -> bool {
        self.refractory_compliant
    }
    
    /// Static method to check refractory compliance
    fn check_refractory_compliance_static(spikes: &[SpikeEvent]) -> bool {
        let refractory_period = Duration::from_millis(1); // 1ms refractory period
        let mut last_spike_per_neuron: HashMap<crate::ttfs_encoding::NeuronId, Duration> = HashMap::new();
        
        for spike in spikes {
            if let Some(&last_time) = last_spike_per_neuron.get(&spike.neuron_id) {
                if spike.timing - last_time < refractory_period {
                    return false; // Refractory violation
                }
            }
            last_spike_per_neuron.insert(spike.neuron_id, spike.timing);
        }
        
        true
    }
    
    /// Extract neural features from spike sequence
    fn extract_features_from_spikes(spikes: &[SpikeEvent]) -> Vec<f32> {
        let mut features = vec![0.0; 128]; // Standard feature vector size
        
        if spikes.is_empty() {
            return features;
        }
        
        // Feature 0-31: Spike timing distribution
        for (i, spike) in spikes.iter().take(32).enumerate() {
            features[i] = spike.timing.as_nanos() as f32 / 1_000_000.0; // Convert to milliseconds
        }
        
        // Feature 32-63: Spike amplitudes
        for (i, spike) in spikes.iter().take(32).enumerate() {
            features[32 + i] = spike.amplitude;
        }
        
        // Feature 64-95: Inter-spike intervals
        for (i, window) in spikes.windows(2).take(32).enumerate() {
            let interval = window[1].timing - window[0].timing;
            features[64 + i] = interval.as_nanos() as f32 / 1_000_000.0;
        }
        
        // Feature 96-127: Statistical features
        features[96] = spikes.len() as f32; // Spike count
        features[97] = spikes.iter().map(|s| s.amplitude).sum::<f32>() / spikes.len() as f32; // Mean amplitude
        features[98] = spikes.last().unwrap().timing.as_nanos() as f32 / 1_000_000.0; // Total duration
        
        // Feature 99: First spike time (TTFS)
        features[99] = spikes[0].timing.as_nanos() as f32 / 1_000_000.0;
        
        features
    }
    
    /// Calculate encoding confidence based on pattern quality
    fn calculate_encoding_confidence_static(
        spikes: &[SpikeEvent],
        total_duration: Duration,
        first_spike_time: Duration,
    ) -> f32 {
        if spikes.is_empty() {
            return 0.0;
        }
        
        let mut confidence = 1.0;
        
        // Penalty for very long patterns
        if total_duration > Duration::from_millis(50) {
            confidence *= 0.8;
        }
        
        // Penalty for late first spike
        if first_spike_time > Duration::from_millis(5) {
            confidence *= 0.9;
        }
        
        // Bonus for consistent amplitudes
        let amplitudes: Vec<f32> = spikes.iter().map(|s| s.amplitude).collect();
        let mean_amplitude = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
        let amplitude_variance = amplitudes.iter()
            .map(|a| (a - mean_amplitude).powi(2))
            .sum::<f32>() / amplitudes.len() as f32;
        
        if amplitude_variance < 0.1 {
            confidence *= 1.1;
        }
        
        confidence.clamp(0.0, 1.0)
    }
    
    /// Compress pattern to fit within a temporal window
    pub fn compress_temporal_window(&self, max_duration: Duration) -> TTFSSpikePattern {
        if self.total_duration <= max_duration {
            return self.clone();
        }
        
        let compression_ratio = max_duration.as_nanos() as f32 / self.total_duration.as_nanos() as f32;
        
        let compressed_spikes: Vec<SpikeEvent> = self.spike_sequence.iter()
            .map(|spike| {
                let new_timing = Duration::from_nanos(
                    (spike.timing.as_nanos() as f32 * compression_ratio) as u64
                );
                SpikeEvent {
                    neuron_id: spike.neuron_id,
                    timing: new_timing,
                    amplitude: spike.amplitude,
                    refractory_state: spike.refractory_state,
                }
            })
            .collect();
        
        let compressed_first_spike = Duration::from_nanos(
            (self.first_spike_time.as_nanos() as f32 * compression_ratio) as u64
        );
        
        TTFSSpikePattern::new(
            self.concept_id.clone(),
            compressed_first_spike,
            compressed_spikes,
            max_duration,
        )
    }
    
    /// Add metadata to the pattern
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// Get spike count
    pub fn spike_count(&self) -> usize {
        self.spike_sequence.len()
    }
    
    /// Get active neuron count
    pub fn active_neuron_count(&self) -> usize {
        use std::collections::HashSet;
        let neurons: HashSet<_> = self.spike_sequence.iter()
            .map(|spike| spike.neuron_id)
            .collect();
        neurons.len()
    }
    
    /// Calculate pattern density (spikes per millisecond)
    pub fn pattern_density(&self) -> f32 {
        if self.total_duration.as_millis() == 0 {
            return 0.0;
        }
        
        self.spike_sequence.len() as f32 / self.total_duration.as_millis() as f32
    }
    
    /// Create a test pattern for validation
    pub fn create_test_pattern(concept_name: &str) -> Self {
        let spikes = vec![
            SpikeEvent::new(
                crate::ttfs_encoding::NeuronId(1), 
                Duration::from_micros(500), 
                0.9
            ),
            SpikeEvent::new(
                crate::ttfs_encoding::NeuronId(2), 
                Duration::from_millis(1), 
                0.8
            ),
            SpikeEvent::new(
                crate::ttfs_encoding::NeuronId(3), 
                Duration::from_millis(2), 
                0.7
            ),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new(concept_name),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3),
        )
    }
}

impl PartialEq for TTFSSpikePattern {
    fn eq(&self, other: &Self) -> bool {
        self.concept_id == other.concept_id &&
        self.first_spike_time == other.first_spike_time &&
        self.spike_sequence == other.spike_sequence
    }
}

impl std::hash::Hash for TTFSSpikePattern {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.concept_id.hash(state);
        self.first_spike_time.as_nanos().hash(state);
        self.spike_sequence.len().hash(state);
    }
}
```

## Verification Steps
1. Create ConceptId and TTFSSpikePattern structures
2. Implement TTFS validation and biological plausibility checks
3. Add neural feature extraction for ruv-FANN compatibility
4. Implement temporal compression and refractory compliance
5. Ensure all tests pass

## Success Criteria
- [ ] TTFSSpikePattern struct compiles without errors
- [ ] TTFS validation correctly identifies valid patterns
- [ ] Neural feature extraction produces suitable vectors
- [ ] Refractory period compliance checking works
- [ ] All tests pass