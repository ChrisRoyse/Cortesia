# Task 19: Pattern Fixup Methods

## Metadata
- **Micro-Phase**: 2.19
- **Duration**: 25-30 minutes
- **Dependencies**: Task 18 (spike_pattern_validator)
- **Output**: `src/ttfs_encoding/pattern_fixup_methods.rs`

## Description
Implement automatic fixing methods for invalid TTFS spike patterns. These methods detect common violations and apply corrective transformations while preserving biological accuracy and concept encoding integrity.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{TTFSSpikePattern, SpikeEvent, NeuronId, ConceptId};
    use std::time::Duration;

    #[test]
    fn test_refractory_period_fixup() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        // Create pattern with refractory violation
        let violation_pattern = create_refractory_violation_pattern();
        let result = fixer.fix_pattern(violation_pattern).unwrap();
        
        assert!(result.is_valid_ttfs());
        assert!(result.check_refractory_compliance());
        assert_eq!(result.concept_id().as_str(), "refractory_test");
        
        // Verify spike ordering preserved
        let spikes = result.spike_sequence();
        for window in spikes.windows(2) {
            assert!(window[0].timing <= window[1].timing);
        }
    }
    
    #[test]
    fn test_timing_precision_fixup() {
        let mut config = FixupConfig::default();
        config.timing_precision_ns = 100; // 100ns precision
        let fixer = PatternFixer::new(config);
        
        let imprecise_pattern = create_imprecise_timing_pattern();
        let result = fixer.fix_pattern(imprecise_pattern).unwrap();
        
        assert!(result.timing_precision_ns() >= 100);
        assert!(result.is_valid_ttfs());
        
        // Verify all spike intervals meet precision requirements
        let spikes = result.spike_sequence();
        for window in spikes.windows(2) {
            let interval = window[1].timing - window[0].timing;
            assert!(interval.as_nanos() >= 100 || interval.is_zero());
        }
    }
    
    #[test]
    fn test_amplitude_normalization() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let invalid_amplitude_pattern = create_invalid_amplitude_pattern();
        let result = fixer.fix_pattern(invalid_amplitude_pattern).unwrap();
        
        // All amplitudes should be in valid range
        for spike in result.spike_sequence() {
            assert!(spike.amplitude >= 0.0);
            assert!(spike.amplitude <= 1.0);
        }
        
        assert!(result.is_valid_ttfs());
        assert!(result.is_biologically_plausible());
    }
    
    #[test]
    fn test_spike_ordering_fixup() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let unordered_pattern = create_unordered_spike_pattern();
        let result = fixer.fix_pattern(unordered_pattern).unwrap();
        
        // Verify spikes are now ordered by timing
        let spikes = result.spike_sequence();
        for window in spikes.windows(2) {
            assert!(window[0].timing <= window[1].timing);
        }
        
        // First spike should match first spike time
        assert_eq!(spikes[0].timing, result.first_spike_time());
        assert!(result.is_valid_ttfs());
    }
    
    #[test]
    fn test_pattern_duration_fixup() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let long_duration_pattern = create_long_duration_pattern();
        let result = fixer.fix_pattern(long_duration_pattern).unwrap();
        
        assert!(result.total_duration() <= Duration::from_millis(100));
        assert!(result.is_biologically_plausible());
        assert!(result.is_valid_ttfs());
        
        // Verify compression preserved relative timing
        let spikes = result.spike_sequence();
        if spikes.len() > 1 {
            let first_interval = spikes[1].timing - spikes[0].timing;
            assert!(first_interval > Duration::new(0, 0));
        }
    }
    
    #[test]
    fn test_neuron_allocation_fixup() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let excessive_neurons_pattern = create_excessive_neurons_pattern();
        let result = fixer.fix_pattern(excessive_neurons_pattern).unwrap();
        
        let unique_neurons: std::collections::HashSet<_> = result.spike_sequence()
            .iter()
            .map(|s| s.neuron_id)
            .collect();
        
        assert!(unique_neurons.len() <= 256); // Default max neurons
        assert!(result.spike_count() > 0);
        assert!(result.is_valid_ttfs());
    }
    
    #[test]
    fn test_biological_plausibility_fixup() {
        let fixer = PatternFixer::new(FixupConfig::biological());
        
        let implausible_pattern = create_biologically_implausible_pattern();
        let result = fixer.fix_pattern(implausible_pattern).unwrap();
        
        assert!(result.is_biologically_plausible());
        assert!(result.is_valid_ttfs());
        
        // Check spike frequency is within biological limits
        let duration_secs = result.total_duration().as_secs_f32();
        if duration_secs > 0.0 {
            let frequency = result.spike_count() as f32 / duration_secs;
            assert!(frequency <= 1000.0); // 1kHz biological limit
        }
    }
    
    #[test]
    fn test_batch_fixup() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let patterns = vec![
            create_refractory_violation_pattern(),
            create_unordered_spike_pattern(),
            create_invalid_amplitude_pattern(),
            create_long_duration_pattern(),
        ];
        
        let results = fixer.fix_batch(&patterns).unwrap();
        
        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(result.is_valid_ttfs());
            assert!(result.is_biologically_plausible());
        }
    }
    
    #[test]
    fn test_fixup_strategies() {
        let strategies = [
            FixupStrategy::Conservative,
            FixupStrategy::Aggressive,
            FixupStrategy::Biological,
            FixupStrategy::Performance,
        ];
        
        let problematic_pattern = create_multiple_violation_pattern();
        
        for strategy in strategies {
            let mut config = FixupConfig::default();
            config.strategy = strategy;
            let fixer = PatternFixer::new(config);
            
            let result = fixer.fix_pattern(problematic_pattern.clone()).unwrap();
            assert!(result.is_valid_ttfs());
            
            // Different strategies should produce different results
            match strategy {
                FixupStrategy::Conservative => {
                    // Should preserve original structure as much as possible
                    assert!(result.spike_count() >= problematic_pattern.spike_count() / 2);
                }
                FixupStrategy::Aggressive => {
                    // May significantly modify pattern for validity
                    assert!(result.is_biologically_plausible());
                }
                FixupStrategy::Biological => {
                    // Should prioritize biological accuracy
                    assert!(result.is_biologically_plausible());
                }
                FixupStrategy::Performance => {
                    // Should optimize for speed
                    assert!(result.spike_count() <= 32); // Simplified pattern
                }
            }
        }
    }
    
    #[test]
    fn test_fixup_preservation() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let original_pattern = create_slightly_invalid_pattern();
        let original_concept_id = original_pattern.concept_id().clone();
        let original_spike_count = original_pattern.spike_count();
        
        let result = fixer.fix_pattern(original_pattern).unwrap();
        
        // Concept ID should be preserved
        assert_eq!(result.concept_id(), &original_concept_id);
        
        // Spike count should be reasonably preserved
        assert!(result.spike_count() >= original_spike_count / 2);
        assert!(result.spike_count() <= original_spike_count * 2);
        
        // First spike should be preserved or adjusted minimally
        assert!(result.first_spike_time() <= Duration::from_millis(10));
    }
    
    #[test]
    fn test_fixup_statistics() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let patterns = vec![
            create_refractory_violation_pattern(),
            create_unordered_spike_pattern(),
            create_invalid_amplitude_pattern(),
        ];
        
        for pattern in &patterns {
            fixer.fix_pattern(pattern.clone()).unwrap();
        }
        
        let stats = fixer.statistics();
        assert_eq!(stats.total_fixups, 3);
        assert!(stats.success_rate >= 0.9);
        assert!(stats.average_fixup_time > Duration::new(0, 0));
    }
    
    #[test]
    fn test_unfixable_pattern_handling() {
        let fixer = PatternFixer::new(FixupConfig::default());
        
        let empty_pattern = create_empty_pattern();
        let result = fixer.fix_pattern(empty_pattern);
        
        // Should handle unfixable patterns gracefully
        assert!(result.is_err());
        
        if let Err(FixupError::UnfixablePattern(reason)) = result {
            assert!(reason.contains("empty") || reason.contains("no spikes"));
        } else {
            panic!("Expected UnfixablePattern error");
        }
    }
    
    // Helper functions
    fn create_refractory_violation_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(0), Duration::from_micros(550), 0.8), // 50μs later - violation
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("refractory_test"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_imprecise_timing_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_nanos(500_000), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_nanos(500_010), 0.8), // 10ns later
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("imprecise"),
            Duration::from_nanos(500_000),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_invalid_amplitude_pattern() -> TTFSSpikePattern {
        let mut spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
        ];
        // Simulate invalid amplitude (normally clamped)
        spikes.push(SpikeEvent {
            neuron_id: NeuronId(1),
            timing: Duration::from_millis(1),
            amplitude: 1.5, // Invalid
            refractory_state: crate::ttfs_encoding::RefractoryState::Ready,
        });
        
        TTFSSpikePattern::new(
            ConceptId::new("invalid_amplitude"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(2),
        )
    }
    
    fn create_unordered_spike_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_millis(2), 0.9), // Later spike first
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8), // Earlier spike second
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("unordered"),
            Duration::from_millis(1), // Inconsistent with actual first spike
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_long_duration_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_secs(1), 0.8), // 1 second later
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("long_duration"),
            Duration::from_micros(500),
            spikes,
            Duration::from_secs(2),
        )
    }
    
    fn create_excessive_neurons_pattern() -> TTFSSpikePattern {
        let spikes: Vec<_> = (0..1000) // Excessive neurons
            .map(|i| SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(500 + i as u64 * 10),
                0.7,
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new("excessive_neurons"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(10),
        )
    }
    
    fn create_biologically_implausible_pattern() -> TTFSSpikePattern {
        // Very high frequency pattern
        let spikes: Vec<_> = (0..100)
            .map(|i| SpikeEvent::new(
                NeuronId(i % 10),
                Duration::from_micros(i as u64), // 1MHz frequency
                0.8,
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new("implausible"),
            Duration::from_micros(0),
            spikes,
            Duration::from_micros(100),
        )
    }
    
    fn create_multiple_violation_pattern() -> TTFSSpikePattern {
        let mut spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_millis(2), 0.9), // Wrong order
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.8), // Refractory violation + wrong order
            SpikeEvent::new(NeuronId(1), Duration::from_micros(510), 1.5), // Invalid amplitude
        ];
        spikes[2].amplitude = 1.5; // Force invalid amplitude
        
        TTFSSpikePattern::new(
            ConceptId::new("multiple_violations"),
            Duration::from_micros(500),
            spikes,
            Duration::from_secs(1), // Too long
        )
    }
    
    fn create_slightly_invalid_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(0), Duration::from_micros(580), 0.8), // 80μs later - minor violation
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("slightly_invalid"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_empty_pattern() -> TTFSSpikePattern {
        TTFSSpikePattern::new(
            ConceptId::new("empty"),
            Duration::from_micros(500),
            vec![], // No spikes
            Duration::from_millis(1),
        )
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{
    TTFSSpikePattern, SpikeEvent, NeuronId, ConceptId,
    ValidationResult, SpikePatternValidator, ViolationType
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Fixup strategies for pattern correction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FixupStrategy {
    /// Conservative approach - minimal changes
    Conservative,
    /// Aggressive approach - significant modifications allowed
    Aggressive,
    /// Biological accuracy prioritized
    Biological,
    /// Performance optimized approach
    Performance,
}

/// Configuration for pattern fixup operations
#[derive(Debug, Clone)]
pub struct FixupConfig {
    /// Fixup strategy to use
    pub strategy: FixupStrategy,
    
    /// Minimum refractory period to enforce
    pub refractory_period: Duration,
    
    /// Required timing precision in nanoseconds
    pub timing_precision_ns: u64,
    
    /// Maximum pattern duration
    pub max_pattern_duration: Duration,
    
    /// Maximum first spike time
    pub max_first_spike_time: Duration,
    
    /// Valid amplitude range
    pub amplitude_range: (f32, f32),
    
    /// Maximum neurons per pattern
    pub max_neurons_per_pattern: usize,
    
    /// Maximum spike frequency (Hz)
    pub max_spike_frequency: f32,
    
    /// Whether to preserve original concept ID
    pub preserve_concept_id: bool,
    
    /// Whether to preserve relative timing relationships
    pub preserve_timing_ratios: bool,
    
    /// Maximum fixup iterations
    pub max_iterations: usize,
}

impl Default for FixupConfig {
    fn default() -> Self {
        Self {
            strategy: FixupStrategy::Conservative,
            refractory_period: Duration::from_micros(100),
            timing_precision_ns: 100,
            max_pattern_duration: Duration::from_millis(100),
            max_first_spike_time: Duration::from_millis(10),
            amplitude_range: (0.0, 1.0),
            max_neurons_per_pattern: 256,
            max_spike_frequency: 1000.0,
            preserve_concept_id: true,
            preserve_timing_ratios: true,
            max_iterations: 5,
        }
    }
}

impl FixupConfig {
    /// Create biological accuracy focused configuration
    pub fn biological() -> Self {
        Self {
            strategy: FixupStrategy::Biological,
            refractory_period: Duration::from_micros(100),
            max_spike_frequency: 500.0, // Stricter biological limit
            max_pattern_duration: Duration::from_millis(50),
            preserve_timing_ratios: true,
            ..Default::default()
        }
    }
    
    /// Create performance focused configuration
    pub fn performance() -> Self {
        Self {
            strategy: FixupStrategy::Performance,
            max_neurons_per_pattern: 32, // Reduced for performance
            timing_precision_ns: 1000, // Relaxed precision
            preserve_timing_ratios: false,
            max_iterations: 3,
            ..Default::default()
        }
    }
}

/// Fixup operation statistics
#[derive(Debug, Default)]
pub struct FixupStatistics {
    pub total_fixups: u64,
    pub successful_fixups: u64,
    pub failed_fixups: u64,
    pub average_fixup_time: Duration,
    pub fixup_type_counts: HashMap<String, u64>,
    pub success_rate: f32,
}

/// Errors that can occur during pattern fixup
#[derive(Debug, Clone)]
pub enum FixupError {
    /// Pattern cannot be fixed
    UnfixablePattern(String),
    /// Fixup operation failed
    FixupFailed(String),
    /// Maximum iterations exceeded
    MaxIterationsExceeded,
    /// Invalid configuration
    InvalidConfiguration(String),
}

impl std::fmt::Display for FixupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixupError::UnfixablePattern(msg) => write!(f, "Unfixable pattern: {}", msg),
            FixupError::FixupFailed(msg) => write!(f, "Fixup failed: {}", msg),
            FixupError::MaxIterationsExceeded => write!(f, "Maximum fixup iterations exceeded"),
            FixupError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for FixupError {}

/// Result type for fixup operations
pub type FixupResult<T> = Result<T, FixupError>;

/// Individual fixup operation record
#[derive(Debug, Clone)]
pub struct FixupOperation {
    pub operation_type: String,
    pub description: String,
    pub spikes_affected: usize,
    pub timing_adjustment: Duration,
}

/// Main pattern fixer
#[derive(Debug)]
pub struct PatternFixer {
    /// Configuration
    config: FixupConfig,
    
    /// Validator for checking fixes
    validator: SpikePatternValidator,
    
    /// Statistics
    statistics: std::sync::Mutex<FixupStatistics>,
}

impl PatternFixer {
    /// Create new pattern fixer
    pub fn new(config: FixupConfig) -> Self {
        let validation_config = crate::ttfs_encoding::ValidationConfig {
            refractory_period: config.refractory_period,
            timing_precision_ns: config.timing_precision_ns,
            max_pattern_duration: config.max_pattern_duration,
            max_first_spike_time: config.max_first_spike_time,
            amplitude_range: config.amplitude_range,
            max_neurons_per_pattern: config.max_neurons_per_pattern,
            biological_validation: true,
            performance_mode: matches!(config.strategy, FixupStrategy::Performance),
            ..Default::default()
        };
        
        Self {
            config,
            validator: SpikePatternValidator::new(validation_config),
            statistics: std::sync::Mutex::new(FixupStatistics::default()),
        }
    }
    
    /// Fix a single spike pattern
    pub fn fix_pattern(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let start_time = Instant::now();
        let mut current_pattern = pattern;
        let mut operations = Vec::new();
        
        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_fixups += 1;
        }
        
        // Check if pattern needs fixing
        let initial_validation = self.validator.validate_pattern(&current_pattern);
        if initial_validation.is_valid() {
            self.update_statistics(start_time, true, &operations);
            return Ok(current_pattern);
        }
        
        // Validate input pattern
        if current_pattern.spike_sequence().is_empty() {
            self.update_statistics(start_time, false, &operations);
            return Err(FixupError::UnfixablePattern("Pattern has no spikes".to_string()));
        }
        
        // Iterative fixing approach
        for iteration in 0..self.config.max_iterations {
            let validation_result = self.validator.validate_pattern(&current_pattern);
            
            if validation_result.is_valid() {
                self.update_statistics(start_time, true, &operations);
                return Ok(current_pattern);
            }
            
            // Apply fixes based on violations found
            let mut fixed = false;
            
            for violation in validation_result.violations() {
                match violation.violation_type {
                    ViolationType::RefractoryPeriod => {
                        current_pattern = self.fix_refractory_violations(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "refractory_fix".to_string(),
                            description: "Adjusted spike timings for refractory compliance".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: self.config.refractory_period,
                        });
                        fixed = true;
                    }
                    ViolationType::TimingPrecision => {
                        current_pattern = self.fix_timing_precision(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "timing_precision_fix".to_string(),
                            description: "Adjusted spike timings for precision requirements".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: Duration::from_nanos(self.config.timing_precision_ns),
                        });
                        fixed = true;
                    }
                    ViolationType::AmplitudeRange => {
                        current_pattern = self.fix_amplitude_violations(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "amplitude_fix".to_string(),
                            description: "Normalized spike amplitudes".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: Duration::new(0, 0),
                        });
                        fixed = true;
                    }
                    ViolationType::SpikeOrdering => {
                        current_pattern = self.fix_spike_ordering(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "ordering_fix".to_string(),
                            description: "Reordered spikes by timing".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: Duration::new(0, 0),
                        });
                        fixed = true;
                    }
                    ViolationType::PatternDuration => {
                        current_pattern = self.fix_pattern_duration(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "duration_fix".to_string(),
                            description: "Compressed pattern duration".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: self.config.max_pattern_duration,
                        });
                        fixed = true;
                    }
                    ViolationType::TTFSViolation => {
                        current_pattern = self.fix_ttfs_violations(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "ttfs_fix".to_string(),
                            description: "Fixed TTFS timing violations".to_string(),
                            spikes_affected: 1,
                            timing_adjustment: self.config.max_first_spike_time,
                        });
                        fixed = true;
                    }
                    ViolationType::BiologicalPlausibility => {
                        current_pattern = self.fix_biological_violations(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "biological_fix".to_string(),
                            description: "Adjusted pattern for biological plausibility".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: Duration::from_millis(1),
                        });
                        fixed = true;
                    }
                    ViolationType::NeuronAllocation => {
                        current_pattern = self.fix_neuron_allocation(current_pattern)?;
                        operations.push(FixupOperation {
                            operation_type: "neuron_fix".to_string(),
                            description: "Reduced neuron usage".to_string(),
                            spikes_affected: current_pattern.spike_count(),
                            timing_adjustment: Duration::new(0, 0),
                        });
                        fixed = true;
                    }
                    _ => continue,
                }
            }
            
            if !fixed {
                break; // No fixable violations found
            }
        }
        
        // Final validation
        let final_validation = self.validator.validate_pattern(&current_pattern);
        if final_validation.is_valid() {
            self.update_statistics(start_time, true, &operations);
            Ok(current_pattern)
        } else {
            self.update_statistics(start_time, false, &operations);
            Err(FixupError::MaxIterationsExceeded)
        }
    }
    
    /// Fix multiple patterns in batch
    pub fn fix_batch(&self, patterns: &[TTFSSpikePattern]) -> FixupResult<Vec<TTFSSpikePattern>> {
        let mut results = Vec::with_capacity(patterns.len());
        
        for pattern in patterns {
            results.push(self.fix_pattern(pattern.clone())?);
        }
        
        Ok(results)
    }
    
    /// Get fixup statistics
    pub fn statistics(&self) -> FixupStatistics {
        self.statistics.lock().unwrap().clone()
    }
    
    // Internal fixup methods
    
    fn fix_refractory_violations(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        let mut last_spike_per_neuron: HashMap<NeuronId, Duration> = HashMap::new();
        
        for spike in &mut spikes {
            if let Some(&last_time) = last_spike_per_neuron.get(&spike.neuron_id) {
                let interval = spike.timing.saturating_sub(last_time);
                if interval < self.config.refractory_period {
                    // Adjust spike timing to respect refractory period
                    spike.timing = last_time + self.config.refractory_period;
                }
            }
            last_spike_per_neuron.insert(spike.neuron_id, spike.timing);
        }
        
        // Sort spikes after timing adjustments
        spikes.sort_by(|a, b| a.timing.cmp(&b.timing));
        
        // Update first spike time if needed
        let first_spike_time = spikes.first().map(|s| s.timing).unwrap_or(pattern.first_spike_time());
        
        // Calculate new total duration
        let total_duration = spikes.last().map(|s| s.timing + Duration::from_millis(1))
            .unwrap_or(pattern.total_duration());
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            first_spike_time,
            spikes,
            total_duration,
        ))
    }
    
    fn fix_timing_precision(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        let precision_duration = Duration::from_nanos(self.config.timing_precision_ns);
        
        // Round timings to required precision
        for spike in &mut spikes {
            let nanos = spike.timing.as_nanos();
            let rounded_nanos = ((nanos as f64 / self.config.timing_precision_ns as f64).round() as u64) 
                * self.config.timing_precision_ns;
            spike.timing = Duration::from_nanos(rounded_nanos);
        }
        
        // Remove duplicate timings by slightly adjusting
        spikes.sort_by(|a, b| a.timing.cmp(&b.timing));
        for i in 1..spikes.len() {
            if spikes[i].timing == spikes[i-1].timing {
                spikes[i].timing += precision_duration;
            }
        }
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            spikes.first().map(|s| s.timing).unwrap_or(pattern.first_spike_time()),
            spikes,
            pattern.total_duration().max(
                spikes.last().map(|s| s.timing + Duration::from_millis(1))
                    .unwrap_or(pattern.total_duration())
            ),
        ))
    }
    
    fn fix_amplitude_violations(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        let (min_amp, max_amp) = self.config.amplitude_range;
        
        for spike in &mut spikes {
            spike.amplitude = spike.amplitude.clamp(min_amp, max_amp);
        }
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            pattern.first_spike_time(),
            spikes,
            pattern.total_duration(),
        ))
    }
    
    fn fix_spike_ordering(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        
        // Sort spikes by timing
        spikes.sort_by(|a, b| a.timing.cmp(&b.timing));
        
        // Update first spike time to match first spike
        let first_spike_time = spikes.first().map(|s| s.timing).unwrap_or(pattern.first_spike_time());
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            first_spike_time,
            spikes,
            pattern.total_duration(),
        ))
    }
    
    fn fix_pattern_duration(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        if pattern.total_duration() <= self.config.max_pattern_duration {
            return Ok(pattern);
        }
        
        // Compress pattern to fit within maximum duration
        let compression_ratio = self.config.max_pattern_duration.as_nanos() as f32 
            / pattern.total_duration().as_nanos() as f32;
        
        let compressed_spikes: Vec<SpikeEvent> = pattern.spike_sequence().iter()
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
            (pattern.first_spike_time().as_nanos() as f32 * compression_ratio) as u64
        );
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            compressed_first_spike,
            compressed_spikes,
            self.config.max_pattern_duration,
        ))
    }
    
    fn fix_ttfs_violations(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        
        // If first spike time is too late, adjust it
        let mut first_spike_time = pattern.first_spike_time();
        if first_spike_time > self.config.max_first_spike_time {
            first_spike_time = self.config.max_first_spike_time;
            
            // Adjust first spike timing to match
            if let Some(first_spike) = spikes.first_mut() {
                first_spike.timing = first_spike_time;
            }
        }
        
        // Ensure first spike time matches first spike
        if let Some(first_spike) = spikes.first() {
            if first_spike.timing != first_spike_time {
                first_spike_time = first_spike.timing.min(self.config.max_first_spike_time);
            }
        }
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            first_spike_time,
            spikes,
            pattern.total_duration(),
        ))
    }
    
    fn fix_biological_violations(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        
        // Check spike frequency and thin out if necessary
        let duration_secs = pattern.total_duration().as_secs_f32();
        if duration_secs > 0.0 {
            let current_frequency = spikes.len() as f32 / duration_secs;
            
            if current_frequency > self.config.max_spike_frequency {
                let keep_ratio = self.config.max_spike_frequency / current_frequency;
                let target_count = (spikes.len() as f32 * keep_ratio) as usize;
                
                // Keep evenly distributed spikes
                let mut filtered_spikes = Vec::new();
                let step = spikes.len() as f32 / target_count as f32;
                
                for i in 0..target_count {
                    let index = (i as f32 * step) as usize;
                    if index < spikes.len() {
                        filtered_spikes.push(spikes[index].clone());
                    }
                }
                
                spikes = filtered_spikes;
            }
        }
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            pattern.first_spike_time(),
            spikes,
            pattern.total_duration(),
        ))
    }
    
    fn fix_neuron_allocation(&self, pattern: TTFSSpikePattern) -> FixupResult<TTFSSpikePattern> {
        let unique_neurons: HashSet<_> = pattern.spike_sequence()
            .iter()
            .map(|spike| spike.neuron_id)
            .collect();
        
        if unique_neurons.len() <= self.config.max_neurons_per_pattern {
            return Ok(pattern);
        }
        
        // Reduce neuron usage by merging spikes
        let mut spikes: Vec<SpikeEvent> = pattern.spike_sequence().to_vec();
        
        match self.config.strategy {
            FixupStrategy::Performance => {
                // Keep only first N neurons worth of spikes
                let mut neuron_count = 0;
                let mut seen_neurons = HashSet::new();
                
                spikes.retain(|spike| {
                    if !seen_neurons.contains(&spike.neuron_id) {
                        seen_neurons.insert(spike.neuron_id);
                        neuron_count += 1;
                    }
                    neuron_count <= self.config.max_neurons_per_pattern
                });
            }
            _ => {
                // Merge spikes from excess neurons onto available neurons
                let mut neuron_mapping = HashMap::new();
                let available_neurons: Vec<_> = unique_neurons.into_iter().take(self.config.max_neurons_per_pattern).collect();
                
                for (i, spike) in spikes.iter_mut().enumerate() {
                    if !available_neurons.contains(&spike.neuron_id) {
                        let target_neuron = available_neurons[i % available_neurons.len()];
                        neuron_mapping.insert(spike.neuron_id, target_neuron);
                        spike.neuron_id = target_neuron;
                    }
                }
            }
        }
        
        Ok(TTFSSpikePattern::new(
            pattern.concept_id().clone(),
            pattern.first_spike_time(),
            spikes,
            pattern.total_duration(),
        ))
    }
    
    fn update_statistics(&self, start_time: Instant, success: bool, operations: &[FixupOperation]) {
        let mut stats = self.statistics.lock().unwrap();
        let fixup_time = start_time.elapsed();
        
        if success {
            stats.successful_fixups += 1;
        } else {
            stats.failed_fixups += 1;
        }
        
        // Update average time
        let total_fixups = stats.total_fixups;
        stats.average_fixup_time = Duration::from_nanos(
            ((stats.average_fixup_time.as_nanos() as u64 * (total_fixups - 1)) + fixup_time.as_nanos() as u64) / total_fixups
        );
        
        // Update operation counts
        for operation in operations {
            *stats.fixup_type_counts.entry(operation.operation_type.clone()).or_insert(0) += 1;
        }
        
        // Update success rate
        stats.success_rate = stats.successful_fixups as f32 / stats.total_fixups as f32;
    }
}
```

## Verification Steps
1. Implement comprehensive fixup strategies for all violation types
2. Add iterative fixing with configurable maximum iterations
3. Implement strategy-specific optimization approaches
4. Add preservation of timing relationships and concept identity
5. Implement batch processing capabilities
6. Add comprehensive statistics tracking

## Success Criteria
- [ ] All violation types can be automatically fixed
- [ ] Biological accuracy is preserved during fixes
- [ ] Concept identity and timing relationships are maintained
- [ ] Batch processing handles multiple patterns efficiently
- [ ] Statistics accurately track fixup operations
- [ ] All test cases pass with proper pattern correction