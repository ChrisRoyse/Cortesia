# Task 14: Spike Validation Helpers

## Metadata
- **Micro-Phase**: 2.14
- **Duration**: 15-20 minutes
- **Dependencies**: Task 11 (SpikeEvent), Task 12 (TTFSSpikePattern)
- **Output**: `src/ttfs_encoding/spike_validation.rs`

## Description
Implement comprehensive spike validation helpers that ensure TTFS spike patterns meet biological constraints and neuromorphic encoding standards for reliable processing.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{SpikeEvent, TTFSSpikePattern, ConceptId, NeuronId};
    use std::time::Duration;

    #[test]
    fn test_biological_timing_validation() {
        let validator = SpikeValidator::new();
        
        // Valid biological timing
        let valid_spike = SpikeEvent::new(NeuronId(1), Duration::from_millis(5), 0.8);
        assert!(validator.validate_biological_timing(&valid_spike).is_valid);
        
        // Invalid - too long
        let invalid_spike = SpikeEvent::new(NeuronId(2), Duration::from_secs(1), 0.8);
        assert!(!validator.validate_biological_timing(&invalid_spike).is_valid);
    }
    
    #[test]
    fn test_refractory_period_validation() {
        let validator = SpikeValidator::new();
        let neuron_id = NeuronId(1);
        
        let spike1 = SpikeEvent::new(neuron_id, Duration::from_millis(1), 0.9);
        let spike2 = SpikeEvent::new(neuron_id, Duration::from_millis(3), 0.8); // 2ms later - valid
        let spike3 = SpikeEvent::new(neuron_id, Duration::from_millis(3100), 0.7); // 100μs later - invalid
        
        let spikes = vec![spike1, spike2, spike3];
        let result = validator.validate_refractory_periods(&spikes);
        
        assert!(!result.is_valid);
        assert!(result.violations.len() > 0);
    }
    
    #[test]
    fn test_pattern_coherence_validation() {
        let validator = SpikeValidator::new();
        
        // Coherent pattern
        let coherent_spikes = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.9),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.8),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(3), 0.7),
        ];
        
        let coherent_pattern = TTFSSpikePattern::new(
            ConceptId::new("coherent"),
            Duration::from_millis(1),
            coherent_spikes,
            Duration::from_millis(4)
        );
        
        let result = validator.validate_pattern_coherence(&coherent_pattern);
        assert!(result.is_valid);
        assert!(result.coherence_score > 0.7);
    }
    
    #[test]
    fn test_amplitude_consistency_validation() {
        let validator = SpikeValidator::new();
        
        let consistent_spikes = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.82),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(3), 0.78),
        ];
        
        let result = validator.validate_amplitude_consistency(&consistent_spikes);
        assert!(result.is_valid);
        assert!(result.variance < 0.1);
        
        let inconsistent_spikes = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.1),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.9),
            SpikeEvent::new(NeuronId(3), Duration::from_millis(3), 0.2),
        ];
        
        let result2 = validator.validate_amplitude_consistency(&inconsistent_spikes);
        assert!(!result2.is_valid);
        assert!(result2.variance > 0.2);
    }
    
    #[test]
    fn test_comprehensive_pattern_validation() {
        let validator = SpikeValidator::new();
        
        let valid_pattern = TTFSSpikePattern::create_test_pattern("test_concept");
        let result = validator.validate_complete_pattern(&valid_pattern);
        
        assert!(result.overall_valid);
        assert!(result.validation_score > 0.8);
        assert!(result.critical_violations.is_empty());
    }
    
    #[test]
    fn test_validation_with_corrections() {
        let validator = SpikeValidator::new();
        
        // Create pattern with correctable issues
        let spikes_with_issues = vec![
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 1.5), // Amplitude too high
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), -0.1), // Amplitude too low
            SpikeEvent::new(NeuronId(3), Duration::from_millis(3), 0.8),
        ];
        
        let corrected = validator.correct_spike_amplitudes(&spikes_with_issues);
        
        assert!(corrected.iter().all(|spike| spike.amplitude >= 0.0 && spike.amplitude <= 1.0));
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{SpikeEvent, TTFSSpikePattern, NeuronId};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::collections::HashMap;

/// Validation result for individual checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub score: f32,  // 0.0-1.0
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Comprehensive pattern validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternValidationResult {
    pub overall_valid: bool,
    pub validation_score: f32,
    pub critical_violations: Vec<String>,
    pub warnings: Vec<String>,
    pub component_results: HashMap<String, ValidationResult>,
    pub correction_suggestions: Vec<CorrectionSuggestion>,
}

/// Suggestion for correcting validation issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionSuggestion {
    pub issue_type: String,
    pub description: String,
    pub automatic_fix_available: bool,
    pub severity: SeverityLevel,
}

/// Severity levels for validation issues
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SeverityLevel {
    Critical,  // Must be fixed
    High,      // Should be fixed
    Medium,    // Recommended to fix
    Low,       // Optional improvement
}

/// Refractory period validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefractoryValidationResult {
    pub is_valid: bool,
    pub violations: Vec<RefractoryViolation>,
    pub compliance_percentage: f32,
}

/// Individual refractory period violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefractoryViolation {
    pub neuron_id: NeuronId,
    pub violation_time: Duration,
    pub expected_minimum: Duration,
    pub actual_interval: Duration,
}

/// Amplitude consistency validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplitudeValidationResult {
    pub is_valid: bool,
    pub mean_amplitude: f32,
    pub variance: f32,
    pub outliers: Vec<usize>, // Indices of outlier spikes
    pub consistency_score: f32,
}

/// Pattern coherence validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceValidationResult {
    pub is_valid: bool,
    pub coherence_score: f32,
    pub temporal_coherence: f32,
    pub amplitude_coherence: f32,
    pub neuron_distribution_score: f32,
}

/// Comprehensive spike pattern validator
pub struct SpikeValidator {
    /// Biological constraints
    max_spike_duration: Duration,
    min_refractory_period: Duration,
    max_refractory_period: Duration,
    
    /// Quality thresholds
    min_coherence_score: f32,
    max_amplitude_variance: f32,
    min_pattern_density: f32,
    max_pattern_density: f32,
}

impl SpikeValidator {
    /// Create new spike validator with default biological constraints
    pub fn new() -> Self {
        Self {
            max_spike_duration: Duration::from_millis(100),
            min_refractory_period: Duration::from_millis(1),
            max_refractory_period: Duration::from_millis(10),
            min_coherence_score: 0.6,
            max_amplitude_variance: 0.3,
            min_pattern_density: 0.1,
            max_pattern_density: 50.0,
        }
    }
    
    /// Create validator with custom constraints
    pub fn with_constraints(
        max_duration: Duration,
        min_refractory: Duration,
        max_refractory: Duration,
    ) -> Self {
        Self {
            max_spike_duration: max_duration,
            min_refractory_period: min_refractory,
            max_refractory_period: max_refractory,
            ..Self::new()
        }
    }
    
    /// Validate biological timing constraints for a single spike
    pub fn validate_biological_timing(&self, spike: &SpikeEvent) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        let mut score = 1.0;
        
        // Check spike timing is within biological limits
        if spike.timing > self.max_spike_duration {
            issues.push(format!(
                "Spike timing {:?} exceeds maximum biological duration {:?}",
                spike.timing, self.max_spike_duration
            ));
            score *= 0.0; // Critical failure
        }
        
        // Check amplitude is valid
        if spike.amplitude < 0.0 || spike.amplitude > 1.0 {
            issues.push(format!(
                "Spike amplitude {} is outside valid range [0.0, 1.0]",
                spike.amplitude
            ));
            score *= 0.0; // Critical failure
        }
        
        // Warning for very early spikes
        if spike.timing < Duration::from_micros(100) {
            warnings.push("Very early spike timing may not be biologically realistic".to_string());
            score *= 0.9;
        }
        
        // Warning for very low amplitudes
        if spike.amplitude < 0.1 {
            warnings.push("Very low spike amplitude may indicate weak signal".to_string());
            suggestions.push("Consider increasing amplitude or removing weak spike".to_string());
            score *= 0.95;
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            score,
            issues,
            warnings,
            suggestions,
        }
    }
    
    /// Validate refractory period compliance across spike sequence
    pub fn validate_refractory_periods(&self, spikes: &[SpikeEvent]) -> RefractoryValidationResult {
        let mut violations = Vec::new();
        let mut neuron_last_spike: HashMap<NeuronId, Duration> = HashMap::new();
        
        for spike in spikes {
            if let Some(&last_time) = neuron_last_spike.get(&spike.neuron_id) {
                let interval = spike.timing - last_time;
                
                if interval < self.min_refractory_period {
                    violations.push(RefractoryViolation {
                        neuron_id: spike.neuron_id,
                        violation_time: spike.timing,
                        expected_minimum: self.min_refractory_period,
                        actual_interval: interval,
                    });
                }
            }
            
            neuron_last_spike.insert(spike.neuron_id, spike.timing);
        }
        
        let total_possible_violations = spikes.len().saturating_sub(1);
        let compliance_percentage = if total_possible_violations == 0 {
            100.0
        } else {
            ((total_possible_violations - violations.len()) as f32 / total_possible_violations as f32) * 100.0
        };
        
        RefractoryValidationResult {
            is_valid: violations.is_empty(),
            violations,
            compliance_percentage,
        }
    }
    
    /// Validate amplitude consistency across spikes
    pub fn validate_amplitude_consistency(&self, spikes: &[SpikeEvent]) -> AmplitudeValidationResult {
        if spikes.is_empty() {
            return AmplitudeValidationResult {
                is_valid: false,
                mean_amplitude: 0.0,
                variance: 0.0,
                outliers: Vec::new(),
                consistency_score: 0.0,
            };
        }
        
        let amplitudes: Vec<f32> = spikes.iter().map(|s| s.amplitude).collect();
        let mean = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
        
        let variance = amplitudes.iter()
            .map(|a| (a - mean).powi(2))
            .sum::<f32>() / amplitudes.len() as f32;
        
        // Find outliers (more than 2 standard deviations from mean)
        let std_dev = variance.sqrt();
        let mut outliers = Vec::new();
        
        for (i, &amplitude) in amplitudes.iter().enumerate() {
            if (amplitude - mean).abs() > 2.0 * std_dev {
                outliers.push(i);
            }
        }
        
        let consistency_score = (1.0 - (variance / 0.25).min(1.0)).max(0.0);
        
        AmplitudeValidationResult {
            is_valid: variance <= self.max_amplitude_variance,
            mean_amplitude: mean,
            variance,
            outliers,
            consistency_score,
        }
    }
    
    /// Validate overall pattern coherence
    pub fn validate_pattern_coherence(&self, pattern: &TTFSSpikePattern) -> CoherenceValidationResult {
        let spikes = pattern.spike_sequence();
        
        if spikes.is_empty() {
            return CoherenceValidationResult {
                is_valid: false,
                coherence_score: 0.0,
                temporal_coherence: 0.0,
                amplitude_coherence: 0.0,
                neuron_distribution_score: 0.0,
            };
        }
        
        // Temporal coherence - smooth timing progression
        let temporal_coherence = self.calculate_temporal_coherence(spikes);
        
        // Amplitude coherence - consistent amplitudes
        let amplitude_result = self.validate_amplitude_consistency(spikes);
        let amplitude_coherence = amplitude_result.consistency_score;
        
        // Neuron distribution - balanced neuron usage
        let neuron_distribution_score = self.calculate_neuron_distribution_score(spikes);
        
        // Overall coherence score
        let coherence_score = (temporal_coherence * 0.4 + 
                              amplitude_coherence * 0.3 + 
                              neuron_distribution_score * 0.3);
        
        CoherenceValidationResult {
            is_valid: coherence_score >= self.min_coherence_score,
            coherence_score,
            temporal_coherence,
            amplitude_coherence,
            neuron_distribution_score,
        }
    }
    
    /// Comprehensive validation of complete spike pattern
    pub fn validate_complete_pattern(&self, pattern: &TTFSSpikePattern) -> PatternValidationResult {
        let mut component_results = HashMap::new();
        let mut critical_violations = Vec::new();
        let mut warnings = Vec::new();
        let mut correction_suggestions = Vec::new();
        
        let spikes = pattern.spike_sequence();
        
        // Individual spike validation
        let mut spike_scores = Vec::new();
        for (i, spike) in spikes.iter().enumerate() {
            let result = self.validate_biological_timing(spike);
            if !result.is_valid {
                critical_violations.extend(result.issues.iter().map(|issue| {
                    format!("Spike {}: {}", i, issue)
                }));
            }
            warnings.extend(result.warnings.iter().map(|warning| {
                format!("Spike {}: {}", i, warning)
            }));
            spike_scores.push(result.score);
        }
        
        let spike_validation_score = if spike_scores.is_empty() { 
            0.0 
        } else { 
            spike_scores.iter().sum::<f32>() / spike_scores.len() as f32 
        };
        
        // Refractory period validation
        let refractory_result = self.validate_refractory_periods(spikes);
        component_results.insert("refractory".to_string(), ValidationResult {
            is_valid: refractory_result.is_valid,
            score: refractory_result.compliance_percentage / 100.0,
            issues: refractory_result.violations.iter().map(|v| {
                format!("Neuron {:?}: interval {:?} < minimum {:?}", 
                       v.neuron_id, v.actual_interval, v.expected_minimum)
            }).collect(),
            warnings: Vec::new(),
            suggestions: if !refractory_result.violations.is_empty() {
                vec!["Increase inter-spike intervals for affected neurons".to_string()]
            } else {
                Vec::new()
            },
        });
        
        // Pattern coherence validation
        let coherence_result = self.validate_pattern_coherence(pattern);
        component_results.insert("coherence".to_string(), ValidationResult {
            is_valid: coherence_result.is_valid,
            score: coherence_result.coherence_score,
            issues: if !coherence_result.is_valid {
                vec!["Pattern lacks sufficient coherence".to_string()]
            } else {
                Vec::new()
            },
            warnings: Vec::new(),
            suggestions: if coherence_result.coherence_score < 0.8 {
                vec!["Consider smoothing spike timing or amplitude progression".to_string()]
            } else {
                Vec::new()
            },
        });
        
        // Calculate overall validation score
        let validation_score = (spike_validation_score * 0.4 +
                               (refractory_result.compliance_percentage / 100.0) * 0.3 +
                               coherence_result.coherence_score * 0.3);
        
        // Generate correction suggestions
        if !refractory_result.is_valid {
            correction_suggestions.push(CorrectionSuggestion {
                issue_type: "refractory_violation".to_string(),
                description: "Automatic refractory period correction available".to_string(),
                automatic_fix_available: true,
                severity: SeverityLevel::Critical,
            });
        }
        
        if coherence_result.coherence_score < 0.6 {
            correction_suggestions.push(CorrectionSuggestion {
                issue_type: "low_coherence".to_string(),
                description: "Pattern smoothing recommended".to_string(),
                automatic_fix_available: true,
                severity: SeverityLevel::High,
            });
        }
        
        PatternValidationResult {
            overall_valid: critical_violations.is_empty() && validation_score > 0.7,
            validation_score,
            critical_violations,
            warnings,
            component_results,
            correction_suggestions,
        }
    }
    
    /// Correct spike amplitudes to valid range
    pub fn correct_spike_amplitudes(&self, spikes: &[SpikeEvent]) -> Vec<SpikeEvent> {
        spikes.iter().map(|spike| {
            SpikeEvent {
                neuron_id: spike.neuron_id,
                timing: spike.timing,
                amplitude: spike.amplitude.clamp(0.0, 1.0),
                refractory_state: spike.refractory_state,
            }
        }).collect()
    }
    
    /// Calculate temporal coherence score
    fn calculate_temporal_coherence(&self, spikes: &[SpikeEvent]) -> f32 {
        if spikes.len() < 2 {
            return 1.0;
        }
        
        // Calculate intervals between consecutive spikes
        let intervals: Vec<Duration> = spikes.windows(2)
            .map(|window| window[1].timing - window[0].timing)
            .collect();
        
        if intervals.is_empty() {
            return 1.0;
        }
        
        // Calculate variance of intervals (lower variance = more coherent)
        let mean_interval_ns = intervals.iter()
            .map(|d| d.as_nanos() as f64)
            .sum::<f64>() / intervals.len() as f64;
        
        let variance = intervals.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_interval_ns;
                diff * diff
            })
            .sum::<f64>() / intervals.len() as f64;
        
        // Convert variance to coherence score (0-1)
        let normalized_variance = variance / (mean_interval_ns * mean_interval_ns);
        (1.0 - normalized_variance.min(1.0)).max(0.0) as f32
    }
    
    /// Calculate neuron distribution score
    fn calculate_neuron_distribution_score(&self, spikes: &[SpikeEvent]) -> f32 {
        use std::collections::HashSet;
        
        let unique_neurons: HashSet<NeuronId> = spikes.iter()
            .map(|spike| spike.neuron_id)
            .collect();
        
        let total_spikes = spikes.len() as f32;
        let unique_count = unique_neurons.len() as f32;
        
        // Score based on neuron diversity
        if total_spikes == 0.0 {
            return 0.0;
        }
        
        let diversity_ratio = unique_count / total_spikes;
        
        // Optimal range is 0.3-0.8 (not too redundant, not too scattered)
        if diversity_ratio >= 0.3 && diversity_ratio <= 0.8 {
            1.0
        } else if diversity_ratio < 0.3 {
            diversity_ratio / 0.3
        } else {
            (1.0 - diversity_ratio) / 0.2
        }
    }
}

impl Default for SpikeValidator {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create comprehensive validation structures and enums
2. Implement SpikeValidator with biological constraint checking
3. Add pattern coherence and amplitude consistency validation
4. Implement correction suggestions and automatic fixes
5. Ensure all tests pass

## Success Criteria
- [ ] SpikeValidator compiles without errors
- [ ] Biological timing validation catches invalid spikes
- [ ] Refractory period validation detects violations
- [ ] Pattern coherence validation provides meaningful scores
- [ ] All tests pass