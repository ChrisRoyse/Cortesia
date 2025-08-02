# Task 18: Spike Pattern Validator

## Metadata
- **Micro-Phase**: 2.18
- **Duration**: 25-30 minutes
- **Dependencies**: Task 11 (spike_event_structure), Task 12 (ttfs_spike_pattern)
- **Output**: `src/ttfs_encoding/spike_pattern_validator.rs`

## Description
Implement comprehensive validation for TTFS spike patterns to ensure biological accuracy, timing constraints, and neuromorphic compatibility. This validator catches pattern violations before they reach the neural network.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{TTFSSpikePattern, SpikeEvent, NeuronId, ConceptId};
    use std::time::Duration;

    #[test]
    fn test_basic_pattern_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        let valid_pattern = create_valid_pattern();
        let result = validator.validate_pattern(&valid_pattern);
        
        assert!(result.is_valid());
        assert!(result.violations().is_empty());
        assert!(result.confidence_score() > 0.9);
    }
    
    #[test]
    fn test_refractory_period_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::strict());
        
        // Create pattern with refractory violation
        let violation_pattern = create_refractory_violation_pattern();
        let result = validator.validate_pattern(&violation_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::RefractoryPeriod));
        
        let violations = result.violations();
        assert!(violations.iter().any(|v| matches!(v.violation_type, ViolationType::RefractoryPeriod)));
        assert!(result.confidence_score() < 0.5);
    }
    
    #[test]
    fn test_timing_precision_validation() {
        let mut config = ValidationConfig::default();
        config.timing_precision_ns = 100; // 100ns precision requirement
        let validator = SpikePatternValidator::new(config);
        
        // Pattern with insufficient timing precision
        let imprecise_pattern = create_imprecise_timing_pattern();
        let result = validator.validate_pattern(&imprecise_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::TimingPrecision));
        
        // Pattern with adequate precision
        let precise_pattern = create_precise_timing_pattern();
        let result = validator.validate_pattern(&precise_pattern);
        
        assert!(result.is_valid());
        assert!(!result.has_violation(ViolationType::TimingPrecision));
    }
    
    #[test]
    fn test_amplitude_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        // Test amplitude range violations
        let invalid_amplitude_pattern = create_invalid_amplitude_pattern();
        let result = validator.validate_pattern(&invalid_amplitude_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::AmplitudeRange));
        
        // Test amplitude consistency
        let inconsistent_pattern = create_inconsistent_amplitude_pattern();
        let result = validator.validate_pattern(&inconsistent_pattern);
        
        assert!(result.has_warning(WarningType::AmplitudeInconsistency));
    }
    
    #[test]
    fn test_biological_plausibility() {
        let validator = SpikePatternValidator::new(ValidationConfig::biological());
        
        // Test spike frequency limits
        let high_frequency_pattern = create_high_frequency_pattern();
        let result = validator.validate_pattern(&high_frequency_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::BiologicalPlausibility));
        
        // Test pattern duration limits
        let long_duration_pattern = create_long_duration_pattern();
        let result = validator.validate_pattern(&long_duration_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::PatternDuration));
    }
    
    #[test]
    fn test_ttfs_specific_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        // Test first spike timing
        let late_first_spike = create_late_first_spike_pattern();
        let result = validator.validate_pattern(&late_first_spike);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::TTFSViolation));
        
        // Test spike ordering
        let unordered_pattern = create_unordered_spike_pattern();
        let result = validator.validate_pattern(&unordered_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::SpikeOrdering));
    }
    
    #[test]
    fn test_neuron_allocation_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        // Test neuron ID conflicts
        let conflict_pattern = create_neuron_conflict_pattern();
        let result = validator.validate_pattern(&conflict_pattern);
        
        assert!(!result.is_valid());
        assert!(result.has_violation(ViolationType::NeuronAllocation));
        
        // Test neuron capacity limits
        let excessive_neurons_pattern = create_excessive_neurons_pattern();
        let result = validator.validate_pattern(&excessive_neurons_pattern);
        
        assert!(result.has_warning(WarningType::ExcessiveNeuronUsage));
    }
    
    #[test]
    fn test_batch_validation() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        let patterns = vec![
            create_valid_pattern(),
            create_refractory_violation_pattern(),
            create_valid_pattern(),
            create_invalid_amplitude_pattern(),
        ];
        
        let results = validator.validate_batch(&patterns);
        
        assert_eq!(results.len(), 4);
        assert!(results[0].is_valid());
        assert!(!results[1].is_valid());
        assert!(results[2].is_valid());
        assert!(!results[3].is_valid());
        
        let summary = validator.batch_validation_summary(&results);
        assert_eq!(summary.total_patterns, 4);
        assert_eq!(summary.valid_patterns, 2);
        assert_eq!(summary.invalid_patterns, 2);
    }
    
    #[test]
    fn test_validation_performance() {
        let validator = SpikePatternValidator::new(ValidationConfig::performance());
        
        let patterns: Vec<_> = (0..100).map(|i| {
            if i % 2 == 0 {
                create_valid_pattern()
            } else {
                create_complex_pattern()
            }
        }).collect();
        
        let start = std::time::Instant::now();
        let results = validator.validate_batch(&patterns);
        let validation_time = start.elapsed();
        
        assert_eq!(results.len(), 100);
        assert!(validation_time < Duration::from_millis(10)); // <10ms for 100 patterns
        
        // Verify performance mode still catches critical violations
        let critical_violations = results.iter()
            .filter(|r| r.has_critical_violation())
            .count();
        assert!(critical_violations == 0); // No critical violations in test patterns
    }
    
    #[test]
    fn test_custom_validation_rules() {
        let mut config = ValidationConfig::default();
        config.custom_rules.push(CustomRule::new(
            "test_rule",
            |pattern| {
                if pattern.spike_count() < 3 {
                    Some(ValidationViolation::new(
                        ViolationType::Custom("minimum_spike_count".to_string()),
                        "Pattern must have at least 3 spikes".to_string(),
                        Severity::Warning,
                    ))
                } else {
                    None
                }
            }
        ));
        
        let validator = SpikePatternValidator::new(config);
        
        let small_pattern = create_small_spike_pattern();
        let result = validator.validate_pattern(&small_pattern);
        
        assert!(result.has_custom_violation("minimum_spike_count"));
    }
    
    #[test]
    fn test_validation_statistics() {
        let validator = SpikePatternValidator::new(ValidationConfig::default());
        
        // Validate various patterns
        let patterns = vec![
            create_valid_pattern(),
            create_refractory_violation_pattern(),
            create_invalid_amplitude_pattern(),
        ];
        
        for pattern in &patterns {
            validator.validate_pattern(pattern);
        }
        
        let stats = validator.statistics();
        assert_eq!(stats.total_validations, 3);
        assert!(stats.average_validation_time > Duration::new(0, 0));
        assert!(stats.violation_frequency.len() > 0);
    }
    
    // Helper functions
    fn create_valid_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.7),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("valid_test"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_refractory_violation_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(0), Duration::from_micros(550), 0.8), // 50μs later - violation
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("refractory_violation"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_imprecise_timing_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_millis(1), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8), // Same time
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("imprecise"),
            Duration::from_millis(1),
            spikes,
            Duration::from_millis(2),
        )
    }
    
    fn create_precise_timing_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_nanos(500_000), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_nanos(500_200), 0.8), // 200ns later
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("precise"),
            Duration::from_nanos(500_000),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_invalid_amplitude_pattern() -> TTFSSpikePattern {
        let mut spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 1.5), // Invalid amplitude
        ];
        // Manually set invalid amplitude (normally clamped in constructor)
        spikes[0].amplitude = 1.5;
        
        TTFSSpikePattern::new(
            ConceptId::new("invalid_amplitude"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(1),
        )
    }
    
    fn create_inconsistent_amplitude_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.1), // Very low amplitude
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.95), // High amplitude
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("inconsistent"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_high_frequency_pattern() -> TTFSSpikePattern {
        let spikes: Vec<_> = (0..50)
            .map(|i| SpikeEvent::new(
                NeuronId(i % 10),
                Duration::from_micros(100 * i as u64), // Very high frequency
                0.8,
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new("high_frequency"),
            Duration::from_micros(100),
            spikes,
            Duration::from_millis(5),
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
    
    fn create_late_first_spike_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_millis(50), 0.9), // Very late first spike
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("late_first_spike"),
            Duration::from_millis(50),
            spikes,
            Duration::from_millis(60),
        )
    }
    
    fn create_unordered_spike_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_millis(2), 0.9), // Later spike first
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8), // Earlier spike second
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("unordered"),
            Duration::from_millis(1), // First spike time doesn't match first spike
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_neuron_conflict_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(0), Duration::from_millis(2), 0.8), // Same neuron, adequate gap
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("neuron_conflict"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_excessive_neurons_pattern() -> TTFSSpikePattern {
        let spikes: Vec<_> = (0..1000) // Excessive number of neurons
            .map(|i| SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(500 + i as u64 * 100),
                0.7,
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new("excessive_neurons"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(100),
        )
    }
    
    fn create_complex_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.7),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(3), 0.6),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("complex"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(4),
        )
    }
    
    fn create_small_spike_pattern() -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new("small"),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(1),
        )
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{TTFSSpikePattern, SpikeEvent, NeuronId};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Validation configuration for spike pattern validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum refractory period between spikes from same neuron
    pub refractory_period: Duration,
    
    /// Required timing precision in nanoseconds
    pub timing_precision_ns: u64,
    
    /// Maximum pattern duration
    pub max_pattern_duration: Duration,
    
    /// Maximum first spike time for TTFS
    pub max_first_spike_time: Duration,
    
    /// Minimum and maximum amplitude ranges
    pub amplitude_range: (f32, f32),
    
    /// Maximum spike frequency (spikes per second)
    pub max_spike_frequency: f32,
    
    /// Maximum number of neurons per pattern
    pub max_neurons_per_pattern: usize,
    
    /// Enable biological plausibility checks
    pub biological_validation: bool,
    
    /// Enable performance mode (faster, less thorough)
    pub performance_mode: bool,
    
    /// Custom validation rules
    pub custom_rules: Vec<CustomRule>,
    
    /// Validation severity levels to report
    pub severity_filter: Vec<Severity>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            refractory_period: Duration::from_micros(100),
            timing_precision_ns: 100,
            max_pattern_duration: Duration::from_millis(100),
            max_first_spike_time: Duration::from_millis(10),
            amplitude_range: (0.0, 1.0),
            max_spike_frequency: 1000.0, // 1kHz
            max_neurons_per_pattern: 256,
            biological_validation: true,
            performance_mode: false,
            custom_rules: Vec::new(),
            severity_filter: vec![Severity::Error, Severity::Warning],
        }
    }
}

impl ValidationConfig {
    /// Create strict validation configuration
    pub fn strict() -> Self {
        Self {
            refractory_period: Duration::from_micros(100),
            timing_precision_ns: 10, // 10ns precision
            max_pattern_duration: Duration::from_millis(50),
            max_first_spike_time: Duration::from_millis(5),
            biological_validation: true,
            severity_filter: vec![Severity::Error, Severity::Warning, Severity::Info],
            ..Default::default()
        }
    }
    
    /// Create biological accuracy focused configuration
    pub fn biological() -> Self {
        Self {
            refractory_period: Duration::from_micros(100),
            max_spike_frequency: 500.0, // 500Hz biological limit
            max_pattern_duration: Duration::from_millis(50),
            biological_validation: true,
            ..Default::default()
        }
    }
    
    /// Create performance-focused configuration
    pub fn performance() -> Self {
        Self {
            performance_mode: true,
            biological_validation: false,
            severity_filter: vec![Severity::Error], // Only errors
            timing_precision_ns: 1000, // 1μs precision
            ..Default::default()
        }
    }
}

/// Types of validation violations
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Refractory period violation
    RefractoryPeriod,
    /// Timing precision insufficient
    TimingPrecision,
    /// Amplitude out of valid range
    AmplitudeRange,
    /// Pattern duration too long
    PatternDuration,
    /// TTFS-specific violation
    TTFSViolation,
    /// Spike ordering violation
    SpikeOrdering,
    /// Neuron allocation violation
    NeuronAllocation,
    /// Biological plausibility violation
    BiologicalPlausibility,
    /// Custom validation rule violation
    Custom(String),
}

/// Warning types for non-critical issues
#[derive(Debug, Clone, PartialEq)]
pub enum WarningType {
    /// Amplitude inconsistency
    AmplitudeInconsistency,
    /// Excessive neuron usage
    ExcessiveNeuronUsage,
    /// Suboptimal timing
    SuboptimalTiming,
    /// Pattern complexity warning
    PatternComplexity,
    /// Performance warning
    Performance,
}

/// Severity levels for validation issues
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    /// Critical error - pattern unusable
    Error,
    /// Warning - pattern usable but suboptimal
    Warning,
    /// Informational - minor issue
    Info,
}

/// Individual validation violation
#[derive(Debug, Clone)]
pub struct ValidationViolation {
    pub violation_type: ViolationType,
    pub message: String,
    pub severity: Severity,
    pub neuron_id: Option<NeuronId>,
    pub spike_index: Option<usize>,
    pub timestamp: Option<Duration>,
}

impl ValidationViolation {
    pub fn new(violation_type: ViolationType, message: String, severity: Severity) -> Self {
        Self {
            violation_type,
            message,
            severity,
            neuron_id: None,
            spike_index: None,
            timestamp: None,
        }
    }
    
    pub fn with_neuron(mut self, neuron_id: NeuronId) -> Self {
        self.neuron_id = Some(neuron_id);
        self
    }
    
    pub fn with_spike_index(mut self, spike_index: usize) -> Self {
        self.spike_index = Some(spike_index);
        self
    }
    
    pub fn with_timestamp(mut self, timestamp: Duration) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
}

/// Validation result for a spike pattern
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether the pattern is valid
    valid: bool,
    
    /// List of violations found
    violations: Vec<ValidationViolation>,
    
    /// List of warnings
    warnings: Vec<ValidationViolation>,
    
    /// Confidence score (0.0-1.0)
    confidence_score: f32,
    
    /// Validation time
    validation_time: Duration,
    
    /// Additional metadata
    metadata: HashMap<String, String>,
}

impl ValidationResult {
    pub fn new(valid: bool, confidence_score: f32, validation_time: Duration) -> Self {
        Self {
            valid,
            violations: Vec::new(),
            warnings: Vec::new(),
            confidence_score,
            validation_time,
            metadata: HashMap::new(),
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    
    pub fn violations(&self) -> &[ValidationViolation] {
        &self.violations
    }
    
    pub fn warnings(&self) -> &[ValidationViolation] {
        &self.warnings
    }
    
    pub fn confidence_score(&self) -> f32 {
        self.confidence_score
    }
    
    pub fn validation_time(&self) -> Duration {
        self.validation_time
    }
    
    pub fn has_violation(&self, violation_type: ViolationType) -> bool {
        self.violations.iter().any(|v| v.violation_type == violation_type)
    }
    
    pub fn has_warning(&self, warning_type: WarningType) -> bool {
        self.warnings.iter().any(|w| {
            if let ViolationType::Custom(ref name) = w.violation_type {
                match warning_type {
                    WarningType::AmplitudeInconsistency => name == "amplitude_inconsistency",
                    WarningType::ExcessiveNeuronUsage => name == "excessive_neuron_usage",
                    WarningType::SuboptimalTiming => name == "suboptimal_timing",
                    WarningType::PatternComplexity => name == "pattern_complexity",
                    WarningType::Performance => name == "performance",
                }
            } else {
                false
            }
        })
    }
    
    pub fn has_custom_violation(&self, custom_name: &str) -> bool {
        self.violations.iter().any(|v| {
            if let ViolationType::Custom(ref name) = v.violation_type {
                name == custom_name
            } else {
                false
            }
        })
    }
    
    pub fn has_critical_violation(&self) -> bool {
        self.violations.iter().any(|v| matches!(v.severity, Severity::Error))
    }
    
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        match violation.severity {
            Severity::Error => {
                self.valid = false;
                self.violations.push(violation);
            }
            Severity::Warning | Severity::Info => {
                self.warnings.push(violation);
            }
        }
    }
    
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Custom validation rule
#[derive(Clone)]
pub struct CustomRule {
    pub name: String,
    pub validator: fn(&TTFSSpikePattern) -> Option<ValidationViolation>,
}

impl CustomRule {
    pub fn new(name: &str, validator: fn(&TTFSSpikePattern) -> Option<ValidationViolation>) -> Self {
        Self {
            name: name.to_string(),
            validator,
        }
    }
}

impl std::fmt::Debug for CustomRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomRule")
            .field("name", &self.name)
            .finish()
    }
}

/// Batch validation summary
#[derive(Debug)]
pub struct BatchValidationSummary {
    pub total_patterns: usize,
    pub valid_patterns: usize,
    pub invalid_patterns: usize,
    pub total_violations: usize,
    pub most_common_violations: Vec<(ViolationType, usize)>,
    pub average_confidence: f32,
    pub total_validation_time: Duration,
}

/// Validation statistics
#[derive(Debug, Default)]
pub struct ValidationStatistics {
    pub total_validations: u64,
    pub valid_patterns: u64,
    pub invalid_patterns: u64,
    pub average_validation_time: Duration,
    pub peak_validation_time: Duration,
    pub violation_frequency: HashMap<String, u64>,
}

/// Main spike pattern validator
#[derive(Debug)]
pub struct SpikePatternValidator {
    /// Validation configuration
    config: ValidationConfig,
    
    /// Validation statistics
    statistics: std::sync::Mutex<ValidationStatistics>,
}

impl SpikePatternValidator {
    /// Create new spike pattern validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            statistics: std::sync::Mutex::new(ValidationStatistics::default()),
        }
    }
    
    /// Validate a single spike pattern
    pub fn validate_pattern(&self, pattern: &TTFSSpikePattern) -> ValidationResult {
        let start_time = Instant::now();
        let mut result = ValidationResult::new(true, 1.0, Duration::new(0, 0));
        
        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_validations += 1;
        }
        
        // Perform validation checks
        self.validate_basic_structure(pattern, &mut result);
        
        if !self.config.performance_mode {
            self.validate_refractory_periods(pattern, &mut result);
            self.validate_timing_precision(pattern, &mut result);
            self.validate_amplitudes(pattern, &mut result);
            self.validate_ttfs_specific(pattern, &mut result);
            self.validate_spike_ordering(pattern, &mut result);
            self.validate_neuron_allocation(pattern, &mut result);
            
            if self.config.biological_validation {
                self.validate_biological_plausibility(pattern, &mut result);
            }
            
            // Apply custom rules
            for rule in &self.config.custom_rules {
                if let Some(violation) = (rule.validator)(pattern) {
                    result.add_violation(violation);
                }
            }
        }
        
        // Calculate final confidence score
        result.confidence_score = self.calculate_confidence_score(&result, pattern);
        result.validation_time = start_time.elapsed();
        
        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            if result.is_valid() {
                stats.valid_patterns += 1;
            } else {
                stats.invalid_patterns += 1;
            }
            
            // Update timing statistics
            let total_validations = stats.total_validations;
            stats.average_validation_time = Duration::from_nanos(
                ((stats.average_validation_time.as_nanos() as u64 * (total_validations - 1)) 
                + result.validation_time.as_nanos() as u64) / total_validations
            );
            
            if result.validation_time > stats.peak_validation_time {
                stats.peak_validation_time = result.validation_time;
            }
            
            // Update violation frequency
            for violation in &result.violations {
                let violation_name = format!("{:?}", violation.violation_type);
                *stats.violation_frequency.entry(violation_name).or_insert(0) += 1;
            }
        }
        
        result
    }
    
    /// Validate multiple patterns in batch
    pub fn validate_batch(&self, patterns: &[TTFSSpikePattern]) -> Vec<ValidationResult> {
        if self.config.performance_mode && patterns.len() > 10 {
            // Parallel validation for performance mode
            use rayon::prelude::*;
            patterns.par_iter()
                .map(|pattern| self.validate_pattern(pattern))
                .collect()
        } else {
            patterns.iter()
                .map(|pattern| self.validate_pattern(pattern))
                .collect()
        }
    }
    
    /// Generate batch validation summary
    pub fn batch_validation_summary(&self, results: &[ValidationResult]) -> BatchValidationSummary {
        let total_patterns = results.len();
        let valid_patterns = results.iter().filter(|r| r.is_valid()).count();
        let invalid_patterns = total_patterns - valid_patterns;
        
        let total_violations = results.iter()
            .map(|r| r.violations().len())
            .sum();
        
        // Count violation types
        let mut violation_counts: HashMap<String, usize> = HashMap::new();
        for result in results {
            for violation in result.violations() {
                let violation_name = format!("{:?}", violation.violation_type);
                *violation_counts.entry(violation_name).or_insert(0) += 1;
            }
        }
        
        let mut most_common_violations: Vec<_> = violation_counts.into_iter()
            .map(|(name, count)| {
                // Convert string back to violation type (simplified)
                let violation_type = match name.as_str() {
                    "RefractoryPeriod" => ViolationType::RefractoryPeriod,
                    "TimingPrecision" => ViolationType::TimingPrecision,
                    "AmplitudeRange" => ViolationType::AmplitudeRange,
                    _ => ViolationType::Custom(name),
                };
                (violation_type, count)
            })
            .collect();
        most_common_violations.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_violations.truncate(5); // Top 5
        
        let average_confidence = results.iter()
            .map(|r| r.confidence_score())
            .sum::<f32>() / total_patterns as f32;
        
        let total_validation_time = results.iter()
            .map(|r| r.validation_time())
            .sum();
        
        BatchValidationSummary {
            total_patterns,
            valid_patterns,
            invalid_patterns,
            total_violations,
            most_common_violations,
            average_confidence,
            total_validation_time,
        }
    }
    
    /// Get validation statistics
    pub fn statistics(&self) -> ValidationStatistics {
        self.statistics.lock().unwrap().clone()
    }
    
    // Internal validation methods
    
    fn validate_basic_structure(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        // Check if pattern has spikes
        if pattern.spike_sequence().is_empty() {
            result.add_violation(ValidationViolation::new(
                ViolationType::TTFSViolation,
                "Pattern must contain at least one spike".to_string(),
                Severity::Error,
            ));
        }
        
        // Check pattern duration
        if pattern.total_duration() > self.config.max_pattern_duration {
            result.add_violation(ValidationViolation::new(
                ViolationType::PatternDuration,
                format!("Pattern duration {:?} exceeds maximum {:?}", 
                    pattern.total_duration(), self.config.max_pattern_duration),
                Severity::Error,
            ));
        }
    }
    
    fn validate_refractory_periods(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let mut last_spike_per_neuron: HashMap<NeuronId, Duration> = HashMap::new();
        
        for (i, spike) in pattern.spike_sequence().iter().enumerate() {
            if let Some(&last_time) = last_spike_per_neuron.get(&spike.neuron_id) {
                let interval = spike.timing.saturating_sub(last_time);
                if interval < self.config.refractory_period {
                    result.add_violation(
                        ValidationViolation::new(
                            ViolationType::RefractoryPeriod,
                            format!("Neuron {:?} spike interval {:?} violates refractory period {:?}", 
                                spike.neuron_id, interval, self.config.refractory_period),
                            Severity::Error,
                        )
                        .with_neuron(spike.neuron_id)
                        .with_spike_index(i)
                        .with_timestamp(spike.timing)
                    );
                }
            }
            last_spike_per_neuron.insert(spike.neuron_id, spike.timing);
        }
    }
    
    fn validate_timing_precision(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let spikes = pattern.spike_sequence();
        if spikes.len() < 2 {
            return;
        }
        
        let precision_ns = self.config.timing_precision_ns;
        
        for window in spikes.windows(2) {
            let interval = window[1].timing - window[0].timing;
            if interval.as_nanos() < precision_ns as u128 && interval.as_nanos() > 0 {
                result.add_violation(ValidationViolation::new(
                    ViolationType::TimingPrecision,
                    format!("Spike interval {:?} below required precision {}ns", 
                        interval, precision_ns),
                    Severity::Warning,
                ));
            }
        }
    }
    
    fn validate_amplitudes(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let (min_amp, max_amp) = self.config.amplitude_range;
        let mut amplitudes = Vec::new();
        
        for (i, spike) in pattern.spike_sequence().iter().enumerate() {
            amplitudes.push(spike.amplitude);
            
            if spike.amplitude < min_amp || spike.amplitude > max_amp {
                result.add_violation(
                    ValidationViolation::new(
                        ViolationType::AmplitudeRange,
                        format!("Spike amplitude {} outside valid range [{}, {}]", 
                            spike.amplitude, min_amp, max_amp),
                        Severity::Error,
                    )
                    .with_spike_index(i)
                    .with_neuron(spike.neuron_id)
                );
            }
        }
        
        // Check amplitude consistency
        if amplitudes.len() > 2 {
            let mean_amplitude = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
            let variance = amplitudes.iter()
                .map(|&a| (a - mean_amplitude).powi(2))
                .sum::<f32>() / amplitudes.len() as f32;
            
            if variance > 0.25 { // High variance threshold
                result.add_violation(ValidationViolation::new(
                    ViolationType::Custom("amplitude_inconsistency".to_string()),
                    format!("High amplitude variance: {:.3}", variance),
                    Severity::Warning,
                ));
            }
        }
    }
    
    fn validate_ttfs_specific(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let spikes = pattern.spike_sequence();
        if spikes.is_empty() {
            return;
        }
        
        // Check first spike time
        if pattern.first_spike_time() > self.config.max_first_spike_time {
            result.add_violation(ValidationViolation::new(
                ViolationType::TTFSViolation,
                format!("First spike time {:?} exceeds maximum {:?}", 
                    pattern.first_spike_time(), self.config.max_first_spike_time),
                Severity::Error,
            ));
        }
        
        // Verify first spike time matches first spike
        if spikes[0].timing != pattern.first_spike_time() {
            result.add_violation(ValidationViolation::new(
                ViolationType::TTFSViolation,
                "First spike time does not match first spike timing".to_string(),
                Severity::Error,
            ));
        }
    }
    
    fn validate_spike_ordering(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let spikes = pattern.spike_sequence();
        
        for (i, window) in spikes.windows(2).enumerate() {
            if window[0].timing > window[1].timing {
                result.add_violation(
                    ValidationViolation::new(
                        ViolationType::SpikeOrdering,
                        format!("Spikes not ordered by timing at index {}", i),
                        Severity::Error,
                    )
                    .with_spike_index(i)
                );
            }
        }
    }
    
    fn validate_neuron_allocation(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let unique_neurons: HashSet<_> = pattern.spike_sequence()
            .iter()
            .map(|spike| spike.neuron_id)
            .collect();
        
        if unique_neurons.len() > self.config.max_neurons_per_pattern {
            result.add_violation(ValidationViolation::new(
                ViolationType::Custom("excessive_neuron_usage".to_string()),
                format!("Pattern uses {} neurons, exceeds maximum {}", 
                    unique_neurons.len(), self.config.max_neurons_per_pattern),
                Severity::Warning,
            ));
        }
    }
    
    fn validate_biological_plausibility(&self, pattern: &TTFSSpikePattern, result: &mut ValidationResult) {
        let duration_secs = pattern.total_duration().as_secs_f32();
        if duration_secs > 0.0 {
            let spike_frequency = pattern.spike_count() as f32 / duration_secs;
            
            if spike_frequency > self.config.max_spike_frequency {
                result.add_violation(ValidationViolation::new(
                    ViolationType::BiologicalPlausibility,
                    format!("Spike frequency {:.1} Hz exceeds biological limit {:.1} Hz", 
                        spike_frequency, self.config.max_spike_frequency),
                    Severity::Error,
                ));
            }
        }
        
        // Additional biological checks can be added here
    }
    
    fn calculate_confidence_score(&self, result: &ValidationResult, pattern: &TTFSSpikePattern) -> f32 {
        if !result.is_valid() {
            return 0.0;
        }
        
        let mut confidence = 1.0;
        
        // Reduce confidence for warnings
        confidence -= result.warnings().len() as f32 * 0.1;
        
        // Bonus for good timing precision
        if pattern.timing_precision_ns() <= self.config.timing_precision_ns as u128 {
            confidence += 0.1;
        }
        
        // Bonus for biological plausibility
        if pattern.is_biologically_plausible() {
            confidence += 0.1;
        }
        
        confidence.clamp(0.0, 1.0)
    }
}
```

## Verification Steps
1. Implement comprehensive validation configuration system
2. Add validation for refractory periods, timing precision, and amplitudes
3. Implement TTFS-specific validation rules
4. Add biological plausibility checking
5. Implement batch validation with performance optimization
6. Add custom validation rules support

## Success Criteria
- [ ] Validator correctly identifies all violation types
- [ ] Biological constraints are properly enforced
- [ ] Batch validation processes 100+ patterns in <10ms
- [ ] Custom validation rules can be added dynamically
- [ ] Confidence scoring accurately reflects pattern quality
- [ ] All test cases pass with proper violation detection