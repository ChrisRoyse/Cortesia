# Task 27: Temporal and Exception Detection Columns

## Metadata
- **Micro-Phase**: 2.27
- **Duration**: 50-55 minutes
- **Dependencies**: Task 24 (architecture_selection_framework), Task 25 (semantic_column_setup), Task 26 (structural_column_setup)
- **Output**: `src/multi_column/temporal_column.rs`, `src/multi_column/exception_column.rs`

## Description
Implement temporal context processing column and exception detection column using optimally selected neural network architectures. The temporal column handles sequence processing, temporal patterns, and time-based reasoning using LSTM/TCN architectures. The exception column detects anomalies, inheritance violations, and contradictions using classification networks with specialized pattern recognition.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId};
    use crate::ruv_fann_integration::ArchitectureSelector;
    use std::time::{Duration, Instant};

    // Temporal Column Tests
    #[test]
    fn test_temporal_column_initialization() {
        let architecture_selector = ArchitectureSelector::new();
        
        // Should select architecture optimized for temporal/sequential processing
        let column = TemporalContextColumn::new_with_auto_selection(&architecture_selector).unwrap();
        
        // Verify temporal-optimized architecture selected
        assert!(column.selected_architecture.is_some());
        let arch = column.selected_architecture.as_ref().unwrap();
        assert!(arch.supported_tasks.contains(&TaskType::Temporal) || 
                arch.supported_tasks.contains(&TaskType::Semantic)); // LSTM for sequences
        assert!(arch.memory_profile.memory_footprint <= 50_000_000); // 50MB limit
        
        // Verify temporal components initialized
        assert!(column.is_ready());
        assert!(column.sequence_processor.is_initialized());
        assert!(column.temporal_memory.is_ready());
        assert!(column.pattern_detector.is_active());
    }
    
    #[test]
    fn test_temporal_sequence_processing() {
        let column = create_test_temporal_column();
        
        // Create temporal sequence pattern
        let sequence_pattern = create_temporal_sequence_pattern(vec![
            ("event1", Duration::from_millis(100)),
            ("event2", Duration::from_millis(200)),
            ("event3", Duration::from_millis(300)),
        ]);
        
        let start = Instant::now();
        let result = column.detect_sequences(&sequence_pattern).unwrap();
        let processing_time = start.elapsed();
        
        // Verify processing speed
        assert!(processing_time < Duration::from_millis(1)); // Sub-millisecond target
        
        // Verify result structure
        assert_eq!(result.column_id, ColumnId::Temporal);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.activation >= 0.0 && result.activation <= 1.0);
        
        // Verify temporal-specific outputs
        assert!(result.temporal_features.is_some());
        let temporal_features = result.temporal_features.unwrap();
        assert!(temporal_features.sequence_coherence >= 0.0);
        assert!(temporal_features.temporal_consistency >= 0.0);
        assert!(temporal_features.pattern_strength >= 0.0);
        assert_eq!(temporal_features.sequence_length, 3);
    }
    
    #[test]
    fn test_temporal_pattern_recognition() {
        let column = create_test_temporal_column();
        
        // Test increasing sequence
        let increasing_pattern = create_temporal_sequence_pattern(vec![
            ("value1", Duration::from_millis(100)),
            ("value2", Duration::from_millis(200)),
            ("value3", Duration::from_millis(300)),
            ("value4", Duration::from_millis(400)),
        ]);
        
        let increasing_result = column.detect_sequences(&increasing_pattern).unwrap();
        let increasing_features = increasing_result.temporal_features.unwrap();
        
        // Should detect ordered sequence
        assert!(increasing_features.sequence_coherence > 0.7);
        assert_eq!(increasing_features.pattern_type, TemporalPatternType::Sequential);
        
        // Test random sequence
        let random_pattern = create_temporal_sequence_pattern(vec![
            ("random1", Duration::from_millis(300)),
            ("random2", Duration::from_millis(100)),
            ("random3", Duration::from_millis(500)),
            ("random4", Duration::from_millis(50)),
        ]);
        
        let random_result = column.detect_sequences(&random_pattern).unwrap();
        let random_features = random_result.temporal_features.unwrap();
        
        // Should detect low coherence
        assert!(random_features.sequence_coherence < 0.4);
        assert_eq!(random_features.pattern_type, TemporalPatternType::Random);
        
        // Verify different pattern recognition
        assert!(increasing_features.sequence_coherence > random_features.sequence_coherence);
    }
    
    #[test]
    fn test_temporal_memory_integration() {
        let mut column = create_test_temporal_column();
        
        // Store temporal sequence
        let sequence1 = create_temporal_sequence_pattern(vec![
            ("morning", Duration::from_millis(100)),
            ("afternoon", Duration::from_millis(200)),
            ("evening", Duration::from_millis(300)),
        ]);
        
        let result1 = column.detect_sequences(&sequence1).unwrap();
        
        // Process similar sequence
        let sequence2 = create_temporal_sequence_pattern(vec![
            ("dawn", Duration::from_millis(110)),
            ("midday", Duration::from_millis(210)),
            ("dusk", Duration::from_millis(310)),
        ]);
        
        let result2 = column.detect_sequences(&sequence2).unwrap();
        
        // Should recognize temporal similarity
        assert!(result2.temporal_features.unwrap().memory_activation > 0.5);
        
        // Verify temporal memory learning
        let memory_stats = column.get_temporal_memory_statistics();
        assert!(memory_stats.stored_sequences > 0);
        assert!(memory_stats.pattern_matches > 0);
    }
    
    #[test]
    fn test_temporal_context_influence() {
        let column = create_test_temporal_column();
        
        // Test context-dependent temporal processing
        let base_sequence = create_temporal_sequence_pattern(vec![
            ("action", Duration::from_millis(100)),
            ("result", Duration::from_millis(200)),
        ]);
        
        // Process with past context
        let past_context = TemporalContext::new(ContextDirection::Past, Duration::from_secs(3600));
        let past_result = column.process_with_temporal_context(&base_sequence, &past_context).unwrap();
        
        // Process with future context
        let future_context = TemporalContext::new(ContextDirection::Future, Duration::from_secs(1800));
        let future_result = column.process_with_temporal_context(&base_sequence, &future_context).unwrap();
        
        // Context should influence processing
        assert_ne!(past_result.context_influence, future_result.context_influence);
        assert!(past_result.context_influence > 0.1);
        assert!(future_result.context_influence > 0.1);
        
        // Verify temporal direction affects interpretation
        assert!(past_result.temporal_direction_score != future_result.temporal_direction_score);
    }
    
    // Exception Column Tests
    #[test]
    fn test_exception_column_initialization() {
        let architecture_selector = ArchitectureSelector::new();
        
        // Should select architecture optimized for anomaly/exception detection
        let column = ExceptionDetectionColumn::new_with_auto_selection(&architecture_selector).unwrap();
        
        // Verify exception-optimized architecture selected
        assert!(column.selected_architecture.is_some());
        let arch = column.selected_architecture.as_ref().unwrap();
        assert!(arch.supported_tasks.contains(&TaskType::Exception) || 
                arch.supported_tasks.contains(&TaskType::Classification)); // MLP for anomaly detection
        assert!(arch.memory_profile.memory_footprint <= 50_000_000); // 50MB limit
        
        // Verify exception components initialized
        assert!(column.is_ready());
        assert!(column.anomaly_detector.is_active());
        assert!(column.inheritance_validator.is_ready());
        assert!(column.pattern_library.is_loaded());
    }
    
    #[test]
    fn test_inheritance_exception_detection() {
        let column = create_test_exception_column();
        
        // Test normal inheritance (should not trigger exception)
        let normal_inheritance = ConceptWithInheritance {
            concept_name: "dog".to_string(),
            inherited_properties: hashmap!{
                "is_alive" => "true",
                "has_fur" => "true",
                "can_move" => "true",
            },
            actual_properties: hashmap!{
                "is_alive" => "true",
                "has_fur" => "true", 
                "can_move" => "true",
                "barks" => "true", // Additional property
            },
            context: InheritanceContext::Biological,
        };
        
        let normal_result = column.find_inhibitions(&create_concept_spike_pattern(&normal_inheritance)).unwrap();
        
        // Should not detect exceptions
        assert!(normal_result.activation < 0.3); // Low activation for normal case
        let exception_features = normal_result.exception_features.unwrap();
        assert_eq!(exception_features.violation_type, ViolationType::None);
        assert!(exception_features.inheritance_violations.is_empty());
        
        // Test inheritance exception (penguin can't fly)
        let exception_inheritance = ConceptWithInheritance {
            concept_name: "penguin".to_string(),
            inherited_properties: hashmap!{
                "is_alive" => "true",
                "has_feathers" => "true",
                "can_fly" => "true", // Inherited from bird
            },
            actual_properties: hashmap!{
                "is_alive" => "true",
                "has_feathers" => "true",
                "can_fly" => "false", // Exception!
                "can_swim" => "true",
            },
            context: InheritanceContext::Biological,
        };
        
        let exception_result = column.find_inhibitions(&create_concept_spike_pattern(&exception_inheritance)).unwrap();
        
        // Should detect exception
        assert!(exception_result.activation > 0.7); // High activation for exception
        let exception_features = exception_result.exception_features.unwrap();
        assert_eq!(exception_features.violation_type, ViolationType::PropertyOverride);
        assert_eq!(exception_features.inheritance_violations.len(), 1);
        assert_eq!(exception_features.inheritance_violations[0].property, "can_fly");
    }
    
    #[test]
    fn test_anomaly_pattern_detection() {
        let column = create_test_exception_column();
        
        // Test normal pattern
        let normal_pattern = create_normal_pattern();
        let normal_result = column.detect_anomalies(&normal_pattern).unwrap();
        
        assert!(normal_result.anomaly_score < 0.3);
        assert_eq!(normal_result.anomaly_type, AnomalyType::None);
        
        // Test outlier pattern
        let outlier_pattern = create_outlier_pattern();
        let outlier_result = column.detect_anomalies(&outlier_pattern).unwrap();
        
        assert!(outlier_result.anomaly_score > 0.7);
        assert_eq!(outlier_result.anomaly_type, AnomalyType::Statistical);
        
        // Test contradiction pattern
        let contradiction_pattern = create_contradiction_pattern();
        let contradiction_result = column.detect_anomalies(&contradiction_pattern).unwrap();
        
        assert!(contradiction_result.anomaly_score > 0.8);
        assert_eq!(contradiction_result.anomaly_type, AnomalyType::Logical);
    }
    
    #[test]
    fn test_exception_pattern_learning() {
        let mut column = create_test_exception_column();
        
        // Introduce known exception pattern
        let platypus_exception = ConceptWithInheritance {
            concept_name: "platypus".to_string(),
            inherited_properties: hashmap!{
                "gives_birth" => "true", // From mammal
                "has_fur" => "true",
            },
            actual_properties: hashmap!{
                "lays_eggs" => "true", // Exception!
                "has_fur" => "true",
                "gives_birth" => "false",
            },
            context: InheritanceContext::Biological,
        };
        
        // Process exception multiple times to learn pattern
        for _ in 0..5 {
            let _ = column.find_inhibitions(&create_concept_spike_pattern(&platypus_exception)).unwrap();
        }
        
        // Test similar exception pattern (echidna)
        let echidna_exception = ConceptWithInheritance {
            concept_name: "echidna".to_string(),
            inherited_properties: hashmap!{
                "gives_birth" => "true", // From mammal
                "has_fur" => "true",
            },
            actual_properties: hashmap!{
                "lays_eggs" => "true", // Similar exception!
                "has_spines" => "true",
                "gives_birth" => "false",
            },
            context: InheritanceContext::Biological,
        };
        
        let echidna_result = column.find_inhibitions(&create_concept_spike_pattern(&echidna_exception)).unwrap();
        
        // Should recognize similar exception pattern
        let exception_features = echidna_result.exception_features.unwrap();
        assert!(exception_features.pattern_similarity > 0.6);
        assert!(exception_features.learned_pattern_match);
        
        // Verify pattern library updated
        let library_stats = column.get_pattern_library_statistics();
        assert!(library_stats.known_exception_patterns > 0);
    }
    
    #[test]
    fn test_context_dependent_exceptions() {
        let column = create_test_exception_column();
        
        // Test same concept in different contexts
        let bank_financial = ConceptWithInheritance {
            concept_name: "bank".to_string(),
            inherited_properties: hashmap!{
                "provides_service" => "true",
                "has_location" => "true",
            },
            actual_properties: hashmap!{
                "handles_money" => "true",
                "provides_loans" => "true",
                "has_location" => "true",
            },
            context: InheritanceContext::Financial,
        };
        
        let bank_geographic = ConceptWithInheritance {
            concept_name: "bank".to_string(),
            inherited_properties: hashmap!{
                "is_landform" => "true",
                "near_water" => "true",
            },
            actual_properties: hashmap!{
                "has_slope" => "true",
                "near_water" => "true",
                "handles_money" => "false", // Different context
            },
            context: InheritanceContext::Geographic,
        };
        
        let financial_result = column.find_inhibitions(&create_concept_spike_pattern(&bank_financial)).unwrap();
        let geographic_result = column.find_inhibitions(&create_concept_spike_pattern(&bank_geographic)).unwrap();
        
        // Context should influence exception detection
        assert_ne!(financial_result.activation, geographic_result.activation);
        
        // Verify context-specific exception handling
        let financial_features = financial_result.exception_features.unwrap();
        let geographic_features = geographic_result.exception_features.unwrap();
        assert_ne!(financial_features.context_relevance, geographic_features.context_relevance);
    }
    
    #[test]
    fn test_parallel_temporal_exception_processing() {
        let temporal_column = create_test_temporal_column();
        let exception_column = create_test_exception_column();
        
        // Create mixed patterns
        let patterns = vec![
            create_temporal_sequence_pattern(vec![("seq1", Duration::from_millis(100))]),
            create_temporal_sequence_pattern(vec![("seq2", Duration::from_millis(200))]),
        ];
        
        let exception_patterns = vec![
            create_concept_spike_pattern(&create_normal_inheritance_concept()),
            create_concept_spike_pattern(&create_exception_inheritance_concept()),
        ];
        
        // Process in parallel
        let start = Instant::now();
        let (temporal_results, exception_results) = rayon::join(
            || temporal_column.process_multiple_sequences_parallel(&patterns),
            || exception_column.process_multiple_exceptions_parallel(&exception_patterns)
        );
        let parallel_time = start.elapsed();
        
        // Verify results
        assert!(temporal_results.is_ok());
        assert!(exception_results.is_ok());
        assert_eq!(temporal_results.unwrap().len(), 2);
        assert_eq!(exception_results.unwrap().len(), 2);
        
        // Should be faster than sequential
        assert!(parallel_time < Duration::from_millis(5));
    }
    
    #[test]
    fn test_architecture_performance_validation() {
        let temporal_column = create_test_temporal_column();
        let exception_column = create_test_exception_column();
        
        // Verify temporal architecture
        let temporal_arch = temporal_column.get_selected_architecture();
        assert!(temporal_arch.supported_tasks.contains(&TaskType::Temporal) ||
                temporal_arch.supported_tasks.contains(&TaskType::Semantic));
        assert!(temporal_arch.memory_profile.memory_footprint <= 50_000_000);
        
        // Verify exception architecture  
        let exception_arch = exception_column.get_selected_architecture();
        assert!(exception_arch.supported_tasks.contains(&TaskType::Exception) ||
                exception_arch.supported_tasks.contains(&TaskType::Classification));
        assert!(exception_arch.memory_profile.memory_footprint <= 50_000_000);
        
        // Test actual performance
        let test_temporal = create_temporal_sequence_pattern(vec![("test", Duration::from_millis(100))]);
        let test_exception = create_concept_spike_pattern(&create_normal_inheritance_concept());
        
        let start = Instant::now();
        let temporal_result = temporal_column.detect_sequences(&test_temporal).unwrap();
        let temporal_time = start.elapsed();
        
        let start = Instant::now();
        let exception_result = exception_column.find_inhibitions(&test_exception).unwrap();
        let exception_time = start.elapsed();
        
        // Both should meet sub-millisecond targets
        assert!(temporal_time <= Duration::from_millis(1));
        assert!(exception_time <= Duration::from_millis(1));
        assert!(temporal_result.confidence >= 0.5);
        assert!(exception_result.confidence >= 0.5);
    }
    
    // Helper functions
    fn create_test_temporal_column() -> TemporalContextColumn {
        let architecture_selector = ArchitectureSelector::new();
        TemporalContextColumn::new_with_auto_selection(&architecture_selector).unwrap()
    }
    
    fn create_test_exception_column() -> ExceptionDetectionColumn {
        let architecture_selector = ArchitectureSelector::new();
        ExceptionDetectionColumn::new_with_auto_selection(&architecture_selector).unwrap()
    }
    
    fn create_temporal_sequence_pattern(events: Vec<(&str, Duration)>) -> TTFSSpikePattern {
        let concept_id = ConceptId::new(&format!("sequence_{}", events.len()));
        let first_spike_time = events.first().map(|(_, t)| *t).unwrap_or(Duration::from_millis(100));
        
        let spikes: Vec<SpikeEvent> = events.iter().enumerate().map(|(i, (name, timing))| {
            SpikeEvent::new(
                NeuronId(i + 50), // Different neuron range for temporal
                *timing,
                0.6 + (i as f32 * 0.1) % 0.4,
            )
        }).collect();
        
        let total_duration = events.last().map(|(_, t)| *t + Duration::from_millis(100))
            .unwrap_or(Duration::from_millis(500));
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_concept_spike_pattern(concept: &ConceptWithInheritance) -> TTFSSpikePattern {
        let concept_id = ConceptId::new(&concept.concept_name);
        let first_spike_time = Duration::from_micros(600);
        let spikes = create_exception_test_spikes(4);
        let total_duration = Duration::from_millis(3);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_exception_test_spikes(count: usize) -> Vec<SpikeEvent> {
        (0..count).map(|i| {
            SpikeEvent::new(
                NeuronId(i + 100), // Different neuron range for exceptions
                Duration::from_micros(200 + i as u64 * 300),
                0.3 + (i as f32 * 0.2) % 0.7,
            )
        }).collect()
    }
    
    fn create_normal_inheritance_concept() -> ConceptWithInheritance {
        ConceptWithInheritance {
            concept_name: "normal_dog".to_string(),
            inherited_properties: hashmap!{"is_alive" => "true", "has_fur" => "true"},
            actual_properties: hashmap!{"is_alive" => "true", "has_fur" => "true", "barks" => "true"},
            context: InheritanceContext::Biological,
        }
    }
    
    fn create_exception_inheritance_concept() -> ConceptWithInheritance {
        ConceptWithInheritance {
            concept_name: "exception_penguin".to_string(),
            inherited_properties: hashmap!{"can_fly" => "true", "has_feathers" => "true"},
            actual_properties: hashmap!{"can_fly" => "false", "has_feathers" => "true", "can_swim" => "true"},
            context: InheritanceContext::Biological,
        }
    }
    
    fn create_normal_pattern() -> TTFSSpikePattern {
        create_temporal_sequence_pattern(vec![
            ("normal1", Duration::from_millis(100)),
            ("normal2", Duration::from_millis(200)),
        ])
    }
    
    fn create_outlier_pattern() -> TTFSSpikePattern {
        create_temporal_sequence_pattern(vec![
            ("outlier", Duration::from_millis(5000)), // Very long delay
        ])
    }
    
    fn create_contradiction_pattern() -> TTFSSpikePattern {
        // Pattern that contradicts itself
        create_temporal_sequence_pattern(vec![
            ("true", Duration::from_millis(100)),
            ("false", Duration::from_millis(101)), // Immediate contradiction
        ])
    }
}
```

## Implementation
```rust
// src/multi_column/temporal_column.rs
use crate::ttfs_encoding::{TTFSSpikePattern, SpikeEvent, NeuronId, ConceptId};
use crate::ruv_fann_integration::{ArchitectureSelector, SelectedArchitecture, TaskType};
use crate::simd_spike_processor::SIMDSpikeProcessor;
use crate::multi_column::{ColumnVote, ColumnId};
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Temporal context processing cortical column
#[derive(Debug)]
pub struct TemporalContextColumn {
    /// Selected optimal neural network architecture
    selected_architecture: Option<SelectedArchitecture>,
    
    /// Neural network for temporal processing
    neural_network: Option<Box<dyn TemporalNeuralNetwork>>,
    
    /// Sequence processing engine
    sequence_processor: SequenceProcessor,
    
    /// Temporal memory for pattern storage
    temporal_memory: TemporalMemory,
    
    /// Pattern detection system
    pattern_detector: TemporalPatternDetector,
    
    /// Context processor for temporal interpretation
    context_processor: TemporalContextProcessor,
    
    /// Activation threshold
    activation_threshold: f32,
    
    /// Temporal analysis cache
    temporal_cache: DashMap<ConceptId, CachedTemporalResult>,
    
    /// SIMD processor
    simd_processor: SIMDSpikeProcessor,
    
    /// Performance monitoring
    performance_monitor: TemporalPerformanceMonitor,
}

/// Temporal context information
#[derive(Debug, Clone)]
pub struct TemporalContext {
    /// Context direction (past/future)
    pub direction: ContextDirection,
    
    /// Time horizon
    pub time_horizon: Duration,
    
    /// Context strength
    pub strength: f32,
    
    /// Context type
    pub context_type: TemporalContextType,
}

/// Context direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextDirection {
    Past,
    Present,
    Future,
    Bidirectional,
}

/// Temporal context types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalContextType {
    Causal,        // Cause-effect relationships
    Sequential,    // Ordered sequences
    Cyclical,      // Repeating patterns
    Conditional,   // If-then temporal logic
    Durational,    // Time-based durations
}

/// Temporal features extracted from processing
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    /// Sequence coherence score
    pub sequence_coherence: f32,
    
    /// Temporal consistency
    pub temporal_consistency: f32,
    
    /// Pattern strength
    pub pattern_strength: f32,
    
    /// Sequence length
    pub sequence_length: usize,
    
    /// Pattern type classification
    pub pattern_type: TemporalPatternType,
    
    /// Memory activation level
    pub memory_activation: f32,
    
    /// Temporal direction score
    pub temporal_direction_score: f32,
}

/// Temporal pattern types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalPatternType {
    Sequential,    // Ordered sequence
    Parallel,      // Simultaneous events
    Cyclical,      // Repeating pattern
    Random,        // No clear pattern
    Causal,        // Cause-effect chain
    Conditional,   // Conditional sequence
}

/// Column vote result for temporal processing
#[derive(Debug, Clone)]
pub struct TemporalColumnVote {
    /// Base column vote
    pub column_vote: ColumnVote,
    
    /// Temporal-specific features
    pub temporal_features: Option<TemporalFeatures>,
    
    /// Context influence
    pub context_influence: f32,
    
    /// Temporal direction score
    pub temporal_direction_score: f32,
    
    /// Memory activation details
    pub memory_activation_details: Option<TemporalMemoryActivation>,
}

/// Temporal memory activation
#[derive(Debug, Clone)]
pub struct TemporalMemoryActivation {
    /// Similar sequences found
    pub similar_sequences: Vec<SimilarSequence>,
    
    /// Prediction confidence
    pub prediction_confidence: f32,
    
    /// Memory influence strength
    pub memory_influence: f32,
}

/// Similar temporal sequence
#[derive(Debug, Clone)]
pub struct SimilarSequence {
    /// Sequence identifier
    pub sequence_id: String,
    
    /// Similarity score
    pub similarity_score: f32,
    
    /// Sequence pattern
    pub pattern: Vec<TemporalEvent>,
}

/// Temporal event in sequence
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event identifier
    pub event_id: String,
    
    /// Event timing
    pub timing: Duration,
    
    /// Event strength
    pub strength: f32,
}

/// Cached temporal result
#[derive(Debug, Clone)]
pub struct CachedTemporalResult {
    /// Column vote
    pub column_vote: ColumnVote,
    
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    
    /// Cache metadata
    pub cached_at: Instant,
    pub hit_count: u32,
}

/// Sequence processing engine
#[derive(Debug)]
pub struct SequenceProcessor {
    /// Sequence analysis algorithms
    analyzers: Vec<Box<dyn SequenceAnalyzer>>,
    
    /// Processing configuration
    config: SequenceProcessingConfig,
}

/// Sequence processing configuration
#[derive(Debug, Clone)]
pub struct SequenceProcessingConfig {
    /// Maximum sequence length to process
    pub max_sequence_length: usize,
    
    /// Minimum coherence threshold
    pub min_coherence_threshold: f32,
    
    /// Pattern detection sensitivity
    pub pattern_sensitivity: f32,
    
    /// Temporal resolution
    pub temporal_resolution: Duration,
}

/// Temporal memory for pattern storage
#[derive(Debug)]
pub struct TemporalMemory {
    /// Stored sequence patterns
    stored_sequences: DashMap<String, StoredSequence>,
    
    /// Pattern statistics
    pattern_stats: TemporalPatternStats,
    
    /// Memory capacity
    max_sequences: usize,
    
    /// Learning rate
    learning_rate: f32,
}

/// Stored temporal sequence
#[derive(Debug, Clone)]
pub struct StoredSequence {
    /// Sequence events
    pub events: Vec<TemporalEvent>,
    
    /// Pattern classification
    pub pattern_type: TemporalPatternType,
    
    /// Usage frequency
    pub usage_frequency: usize,
    
    /// Last access time
    pub last_accessed: Instant,
    
    /// Sequence strength
    pub strength: f32,
}

/// Temporal pattern statistics
#[derive(Debug, Default)]
pub struct TemporalPatternStats {
    pub stored_sequences: usize,
    pub pattern_matches: usize,
    pub total_predictions: usize,
    pub successful_predictions: usize,
    pub average_sequence_length: f32,
}

/// Temporal pattern detector
#[derive(Debug)]
pub struct TemporalPatternDetector {
    /// Pattern recognition models
    pattern_recognizers: HashMap<TemporalPatternType, Box<dyn PatternRecognizer>>,
    
    /// Detection thresholds
    detection_thresholds: HashMap<TemporalPatternType, f32>,
    
    /// Active state
    is_active: bool,
}

/// Temporal context processor
#[derive(Debug)]
pub struct TemporalContextProcessor {
    /// Context analyzers
    context_analyzers: HashMap<TemporalContextType, Box<dyn ContextAnalyzer>>,
    
    /// Context influence weights
    influence_weights: HashMap<TemporalContextType, f32>,
}

/// Performance monitoring for temporal column
#[derive(Debug, Default)]
pub struct TemporalPerformanceMonitor {
    /// Sequence processing times
    pub sequence_processing_times: Vec<Duration>,
    
    /// Pattern recognition accuracy
    pub pattern_recognition_accuracy: f32,
    
    /// Memory hit rate
    pub memory_hit_rate: f32,
    
    /// Cache performance
    pub cache_stats: CacheStatistics,
}

/// Cache statistics (reused from other columns)
#[derive(Debug, Default, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
    pub average_hit_time: Duration,
    pub average_miss_time: Duration,
}

/// Neural network abstraction for temporal processing
pub trait TemporalNeuralNetwork: Send + Sync {
    /// Process temporal sequence
    fn process_sequence(&self, sequence: &[f32]) -> Result<Vec<f32>, TemporalError>;
    
    /// Get architecture information
    fn architecture_info(&self) -> &SelectedArchitecture;
    
    /// Check readiness
    fn is_ready(&self) -> bool;
}

/// Sequence analysis trait
pub trait SequenceAnalyzer: Send + Sync {
    /// Analyze sequence coherence
    fn analyze_coherence(&self, sequence: &[TemporalEvent]) -> f32;
    
    /// Detect pattern type
    fn detect_pattern_type(&self, sequence: &[TemporalEvent]) -> TemporalPatternType;
}

/// Pattern recognition trait
pub trait PatternRecognizer: Send + Sync {
    /// Recognize pattern in sequence
    fn recognize(&self, sequence: &[TemporalEvent]) -> f32;
    
    /// Pattern type this recognizer handles
    fn pattern_type(&self) -> TemporalPatternType;
}

/// Context analysis trait
pub trait ContextAnalyzer: Send + Sync {
    /// Analyze context influence
    fn analyze_context(&self, context: &TemporalContext, sequence: &[TemporalEvent]) -> f32;
    
    /// Context type this analyzer handles
    fn context_type(&self) -> TemporalContextType;
}

/// Temporal processing errors
#[derive(Debug, thiserror::Error)]
pub enum TemporalError {
    #[error("Architecture selection failed: {0}")]
    ArchitectureSelectionFailed(String),
    
    #[error("Sequence processing failed: {0}")]
    SequenceProcessingFailed(String),
    
    #[error("Invalid temporal pattern: {0}")]
    InvalidTemporalPattern(String),
    
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,
    
    #[error("Network not initialized")]
    NetworkNotInitialized,
}

impl TemporalContextColumn {
    /// Create temporal column with automatic architecture selection
    pub fn new_with_auto_selection(selector: &ArchitectureSelector) -> Result<Self, TemporalError> {
        let start_time = Instant::now();
        
        // Select optimal architecture for temporal processing
        let temporal_candidates = selector.select_for_task_type(TaskType::Temporal);
        let semantic_candidates = selector.select_for_task_type(TaskType::Semantic); // LSTM fallback
        
        let mut all_candidates = temporal_candidates;
        all_candidates.extend(semantic_candidates);
        
        let selected_arch = all_candidates.into_iter()
            .filter(|arch| {
                // Prefer LSTM, TCN, or GRU for temporal processing
                matches!(arch.id, 4 | 5 | 20) || arch.supported_tasks.contains(&TaskType::Temporal)
            })
            .max_by(|a, b| {
                let score_a = Self::calculate_temporal_suitability_score(a);
                let score_b = Self::calculate_temporal_suitability_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .ok_or_else(|| TemporalError::ArchitectureSelectionFailed("No suitable temporal architecture found".to_string()))?;
        
        let neural_network = Self::load_temporal_neural_network(&selected_arch)?;
        
        let column = Self {
            selected_architecture: Some(selected_arch),
            neural_network: Some(neural_network),
            sequence_processor: SequenceProcessor::new(),
            temporal_memory: TemporalMemory::new(1000), // 1000 sequence capacity
            pattern_detector: TemporalPatternDetector::new(),
            context_processor: TemporalContextProcessor::new(),
            activation_threshold: 0.6, // Lower threshold for temporal nuances
            temporal_cache: DashMap::new(),
            simd_processor: SIMDSpikeProcessor::new(Default::default()),
            performance_monitor: TemporalPerformanceMonitor::default(),
        };
        
        let initialization_time = start_time.elapsed();
        println!("Temporal column initialized in {:?} with architecture: {}", 
                initialization_time, 
                column.selected_architecture.as_ref().unwrap().architecture.name);
        
        Ok(column)
    }
    
    /// Detect temporal sequences in spike pattern
    pub fn detect_sequences(&self, spike_pattern: &TTFSSpikePattern) -> Result<ColumnVote, TemporalError> {
        let start_time = Instant::now();
        
        // Check cache
        if let Some(cached) = self.check_cache(&spike_pattern.concept_id()) {
            return Ok(cached.column_vote);
        }
        
        // Extract temporal sequence from spike pattern
        let temporal_sequence = self.extract_temporal_sequence(spike_pattern)?;
        
        // Process through sequence analyzer
        let sequence_features = self.sequence_processor.analyze_sequence(&temporal_sequence)?;
        
        // Process through neural network
        let neural_input = self.prepare_temporal_neural_input(&temporal_sequence, &sequence_features);
        let neural_output = self.neural_network.as_ref()
            .ok_or(TemporalError::NetworkNotInitialized)?
            .process_sequence(&neural_input)?;
        
        // Extract temporal features
        let temporal_features = self.extract_temporal_features(&neural_output, &sequence_features);
        
        // Calculate confidence and activation
        let confidence = self.calculate_temporal_confidence(&temporal_features, &temporal_sequence);
        let activation = if confidence > self.activation_threshold { confidence } else { 0.0 };
        
        // Create column vote
        let column_vote = ColumnVote {
            column_id: ColumnId::Temporal,
            confidence,
            activation,
            neural_output: neural_output.clone(),
            processing_time: start_time.elapsed(),
        };
        
        // Cache result
        self.cache_result(spike_pattern.concept_id(), &column_vote, &temporal_features);
        
        // Update temporal memory
        self.update_temporal_memory(&temporal_sequence, &temporal_features);
        
        Ok(column_vote)
    }
    
    /// Process with temporal context
    pub fn process_with_temporal_context(&self, 
                                       spike_pattern: &TTFSSpikePattern, 
                                       context: &TemporalContext) -> Result<TemporalColumnVote, TemporalError> {
        let base_result = self.detect_sequences(spike_pattern)?;
        let temporal_sequence = self.extract_temporal_sequence(spike_pattern)?;
        
        // Apply context influence
        let context_influence = self.context_processor.calculate_context_influence(context, &temporal_sequence);
        
        // Calculate temporal direction score
        let temporal_direction_score = self.calculate_temporal_direction_score(context, &temporal_sequence);
        
        // Adjust confidence based on context
        let adjusted_confidence = (base_result.confidence + context_influence * 0.3).clamp(0.0, 1.0);
        let adjusted_activation = if adjusted_confidence > self.activation_threshold { 
            adjusted_confidence 
        } else { 
            0.0 
        };
        
        // Extract temporal features
        let sequence_features = self.sequence_processor.analyze_sequence(&temporal_sequence)?;
        let temporal_features = self.extract_temporal_features(&base_result.neural_output, &sequence_features);
        
        Ok(TemporalColumnVote {
            column_vote: ColumnVote {
                column_id: ColumnId::Temporal,
                confidence: adjusted_confidence,
                activation: adjusted_activation,
                neural_output: base_result.neural_output,
                processing_time: base_result.processing_time,
            },
            temporal_features: Some(temporal_features),
            context_influence,
            temporal_direction_score,
            memory_activation_details: self.get_memory_activation_details(&temporal_sequence),
        })
    }
    
    /// Process multiple sequences in parallel
    pub fn process_multiple_sequences_parallel(&self, spike_patterns: &[TTFSSpikePattern]) -> Result<Vec<ColumnVote>, TemporalError> {
        spike_patterns.par_iter()
            .map(|pattern| self.detect_sequences(pattern))
            .collect()
    }
    
    /// Get temporal memory statistics
    pub fn get_temporal_memory_statistics(&self) -> TemporalPatternStats {
        self.temporal_memory.pattern_stats.clone()
    }
    
    /// Check if column is ready
    pub fn is_ready(&self) -> bool {
        self.neural_network.is_some() && 
        self.sequence_processor.is_initialized() &&
        self.temporal_memory.is_ready() &&
        self.pattern_detector.is_active()
    }
    
    /// Get selected architecture
    pub fn get_selected_architecture(&self) -> &SelectedArchitecture {
        self.selected_architecture.as_ref().unwrap()
    }
    
    // Private helper methods
    
    fn calculate_temporal_suitability_score(architecture: &crate::ruv_fann_integration::ArchitectureCandidate) -> f32 {
        let mut score = architecture.performance_metrics.performance_score;
        
        // Strong preference for temporal architectures
        match architecture.id {
            4 => score += 0.3,  // LSTM - excellent for sequences
            20 => score += 0.25, // TCN - good for temporal
            5 => score += 0.2,   // GRU - good for sequences
            _ => {}
        }
        
        if architecture.supported_tasks.contains(&TaskType::Temporal) {
            score += 0.2;
        }
        
        score
    }
    
    fn load_temporal_neural_network(architecture: &SelectedArchitecture) -> Result<Box<dyn TemporalNeuralNetwork>, TemporalError> {
        match architecture.architecture.id {
            4 => Ok(Box::new(LSTMTemporalNetwork::new(architecture.clone())?)),
            20 => Ok(Box::new(TCNTemporalNetwork::new(architecture.clone())?)),
            5 => Ok(Box::new(GRUTemporalNetwork::new(architecture.clone())?)),
            _ => Ok(Box::new(LSTMTemporalNetwork::new(architecture.clone())?)), // Default to LSTM
        }
    }
    
    fn check_cache(&self, concept_id: &ConceptId) -> Option<CachedTemporalResult> {
        self.temporal_cache.get(concept_id).map(|entry| entry.value().clone())
    }
    
    fn cache_result(&self, concept_id: ConceptId, column_vote: &ColumnVote, temporal_features: &TemporalFeatures) {
        let cached_result = CachedTemporalResult {
            column_vote: column_vote.clone(),
            temporal_features: temporal_features.clone(),
            cached_at: Instant::now(),
            hit_count: 0,
        };
        
        self.temporal_cache.insert(concept_id, cached_result);
        
        // Cache management
        if self.temporal_cache.len() > 500 {
            let oldest_key = self.temporal_cache.iter()
                .min_by_key(|entry| entry.value().cached_at)
                .map(|entry| entry.key().clone());
            
            if let Some(key) = oldest_key {
                self.temporal_cache.remove(&key);
            }
        }
    }
    
    fn extract_temporal_sequence(&self, spike_pattern: &TTFSSpikePattern) -> Result<Vec<TemporalEvent>, TemporalError> {
        let spikes = spike_pattern.spike_sequence();
        
        let events = spikes.iter().enumerate().map(|(i, spike)| {
            TemporalEvent {
                event_id: format!("event_{}", i),
                timing: spike.timing,
                strength: spike.amplitude,
            }
        }).collect();
        
        Ok(events)
    }
    
    fn prepare_temporal_neural_input(&self, sequence: &[TemporalEvent], features: &SequenceFeatures) -> Vec<f32> {
        let mut input = vec![0.0; 128]; // Standard temporal input size
        
        // Encode sequence events
        for (i, event) in sequence.iter().take(32).enumerate() {
            let base_idx = i * 3;
            if base_idx + 2 < input.len() {
                input[base_idx] = event.timing.as_nanos() as f32 / 1_000_000.0; // Convert to ms
                input[base_idx + 1] = event.strength;
                input[base_idx + 2] = 1.0; // Event presence indicator
            }
        }
        
        // Add sequence-level features
        if input.len() > 96 {
            input[96] = features.coherence;
            input[97] = features.consistency;
            input[98] = sequence.len() as f32;
            input[99] = features.pattern_strength;
        }
        
        input
    }
    
    fn extract_temporal_features(&self, neural_output: &[f32], sequence_features: &SequenceFeatures) -> TemporalFeatures {
        TemporalFeatures {
            sequence_coherence: sequence_features.coherence,
            temporal_consistency: sequence_features.consistency,
            pattern_strength: sequence_features.pattern_strength,
            sequence_length: sequence_features.sequence_length,
            pattern_type: sequence_features.pattern_type,
            memory_activation: self.calculate_memory_activation(sequence_features),
            temporal_direction_score: self.calculate_direction_score(neural_output),
        }
    }
    
    fn calculate_temporal_confidence(&self, features: &TemporalFeatures, sequence: &[TemporalEvent]) -> f32 {
        let coherence_weight = 0.4;
        let consistency_weight = 0.3;
        let pattern_weight = 0.2;
        let length_weight = 0.1;
        
        let length_score = (sequence.len() as f32 / 10.0).min(1.0); // Normalize sequence length
        
        (features.sequence_coherence * coherence_weight +
         features.temporal_consistency * consistency_weight +
         features.pattern_strength * pattern_weight +
         length_score * length_weight).clamp(0.0, 1.0)
    }
    
    fn calculate_memory_activation(&self, _features: &SequenceFeatures) -> f32 {
        // Mock memory activation calculation
        0.6
    }
    
    fn calculate_direction_score(&self, _neural_output: &[f32]) -> f32 {
        // Mock temporal direction score
        0.7
    }
    
    fn calculate_temporal_direction_score(&self, context: &TemporalContext, _sequence: &[TemporalEvent]) -> f32 {
        match context.direction {
            ContextDirection::Past => 0.3,
            ContextDirection::Present => 1.0,
            ContextDirection::Future => 0.5,
            ContextDirection::Bidirectional => 0.8,
        }
    }
    
    fn get_memory_activation_details(&self, _sequence: &[TemporalEvent]) -> Option<TemporalMemoryActivation> {
        // Mock memory activation details
        Some(TemporalMemoryActivation {
            similar_sequences: Vec::new(),
            prediction_confidence: 0.7,
            memory_influence: 0.5,
        })
    }
    
    fn update_temporal_memory(&self, _sequence: &[TemporalEvent], _features: &TemporalFeatures) {
        // Mock memory update
    }
}

// Implementation details for supporting structures

#[derive(Debug, Clone)]
pub struct SequenceFeatures {
    pub coherence: f32,
    pub consistency: f32,
    pub pattern_strength: f32,
    pub sequence_length: usize,
    pub pattern_type: TemporalPatternType,
}

impl SequenceProcessor {
    pub fn new() -> Self {
        Self {
            analyzers: Vec::new(),
            config: SequenceProcessingConfig::default(),
        }
    }
    
    pub fn analyze_sequence(&self, sequence: &[TemporalEvent]) -> Result<SequenceFeatures, TemporalError> {
        let coherence = self.calculate_coherence(sequence);
        let consistency = self.calculate_consistency(sequence);
        let pattern_strength = self.calculate_pattern_strength(sequence);
        let pattern_type = self.detect_pattern_type(sequence);
        
        Ok(SequenceFeatures {
            coherence,
            consistency,
            pattern_strength,
            sequence_length: sequence.len(),
            pattern_type,
        })
    }
    
    pub fn is_initialized(&self) -> bool {
        true
    }
    
    fn calculate_coherence(&self, sequence: &[TemporalEvent]) -> f32 {
        if sequence.len() < 2 {
            return 1.0;
        }
        
        let mut coherence_sum = 0.0;
        let mut comparisons = 0;
        
        for window in sequence.windows(2) {
            let time_diff = window[1].timing.saturating_sub(window[0].timing).as_millis() as f32;
            let strength_diff = (window[1].strength - window[0].strength).abs();
            
            // Lower time differences and strength differences indicate higher coherence
            let time_coherence = 1.0 / (1.0 + time_diff / 1000.0); // Normalize by second
            let strength_coherence = 1.0 - strength_diff;
            
            coherence_sum += (time_coherence + strength_coherence) / 2.0;
            comparisons += 1;
        }
        
        if comparisons > 0 {
            coherence_sum / comparisons as f32
        } else {
            1.0
        }
    }
    
    fn calculate_consistency(&self, sequence: &[TemporalEvent]) -> f32 {
        if sequence.len() < 3 {
            return 1.0;
        }
        
        // Check for consistent timing intervals
        let intervals: Vec<_> = sequence.windows(2)
            .map(|w| w[1].timing.saturating_sub(w[0].timing).as_millis())
            .collect();
        
        let mean_interval = intervals.iter().sum::<u128>() as f32 / intervals.len() as f32;
        let variance = intervals.iter()
            .map(|&i| (i as f32 - mean_interval).powi(2))
            .sum::<f32>() / intervals.len() as f32;
        
        let coefficient_of_variation = if mean_interval > 0.0 {
            variance.sqrt() / mean_interval
        } else {
            1.0
        };
        
        (1.0 / (1.0 + coefficient_of_variation)).clamp(0.0, 1.0)
    }
    
    fn calculate_pattern_strength(&self, sequence: &[TemporalEvent]) -> f32 {
        if sequence.is_empty() {
            return 0.0;
        }
        
        let avg_strength = sequence.iter().map(|e| e.strength).sum::<f32>() / sequence.len() as f32;
        let length_factor = (sequence.len() as f32 / 10.0).min(1.0);
        
        (avg_strength * length_factor).clamp(0.0, 1.0)
    }
    
    fn detect_pattern_type(&self, sequence: &[TemporalEvent]) -> TemporalPatternType {
        if sequence.len() < 2 {
            return TemporalPatternType::Random;
        }
        
        let coherence = self.calculate_coherence(sequence);
        let consistency = self.calculate_consistency(sequence);
        
        match (coherence, consistency) {
            (c, s) if c > 0.8 && s > 0.8 => TemporalPatternType::Sequential,
            (c, s) if c > 0.6 && s < 0.4 => TemporalPatternType::Parallel,
            (c, s) if c > 0.7 && s > 0.6 => TemporalPatternType::Cyclical,
            (c, s) if c > 0.5 && s > 0.5 => TemporalPatternType::Causal,
            (c, s) if c > 0.4 && s > 0.3 => TemporalPatternType::Conditional,
            _ => TemporalPatternType::Random,
        }
    }
}

impl Default for SequenceProcessingConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 50,
            min_coherence_threshold: 0.3,
            pattern_sensitivity: 0.7,
            temporal_resolution: Duration::from_millis(10),
        }
    }
}

impl TemporalMemory {
    pub fn new(max_sequences: usize) -> Self {
        Self {
            stored_sequences: DashMap::new(),
            pattern_stats: TemporalPatternStats::default(),
            max_sequences,
            learning_rate: 0.1,
        }
    }
    
    pub fn is_ready(&self) -> bool {
        true
    }
}

impl TemporalPatternDetector {
    pub fn new() -> Self {
        Self {
            pattern_recognizers: HashMap::new(),
            detection_thresholds: HashMap::new(),
            is_active: true,
        }
    }
    
    pub fn is_active(&self) -> bool {
        self.is_active
    }
}

impl TemporalContextProcessor {
    pub fn new() -> Self {
        Self {
            context_analyzers: HashMap::new(),
            influence_weights: HashMap::new(),
        }
    }
    
    pub fn calculate_context_influence(&self, context: &TemporalContext, _sequence: &[TemporalEvent]) -> f32 {
        // Mock context influence calculation
        context.strength * 0.3
    }
}

impl TemporalContext {
    pub fn new(direction: ContextDirection, time_horizon: Duration) -> Self {
        Self {
            direction,
            time_horizon,
            strength: 1.0,
            context_type: TemporalContextType::Sequential,
        }
    }
}

// Neural network implementations for temporal processing

pub struct LSTMTemporalNetwork {
    architecture: SelectedArchitecture,
    lstm_weights: Vec<f32>,
    hidden_state: Vec<f32>,
    cell_state: Vec<f32>,
}

impl LSTMTemporalNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, TemporalError> {
        Ok(Self {
            architecture,
            lstm_weights: vec![0.4; 512], // LSTM weight matrix
            hidden_state: vec![0.0; 64],  // Hidden state
            cell_state: vec![0.0; 64],    // Cell state
        })
    }
}

impl TemporalNeuralNetwork for LSTMTemporalNetwork {
    fn process_sequence(&self, sequence: &[f32]) -> Result<Vec<f32>, TemporalError> {
        // Simplified LSTM forward pass
        let mut output = vec![0.0; 64];
        
        // Process sequence through LSTM cells
        for (i, &input) in sequence.iter().take(64).enumerate() {
            let weight_idx = i % self.lstm_weights.len();
            let weight = self.lstm_weights[weight_idx];
            
            // Simplified LSTM computation
            output[i] = (input * weight + self.hidden_state.get(i).unwrap_or(&0.0)).tanh();
        }
        
        Ok(output)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn is_ready(&self) -> bool {
        !self.lstm_weights.is_empty()
    }
}

pub struct TCNTemporalNetwork {
    architecture: SelectedArchitecture,
    conv_weights: Vec<f32>,
    dilation_levels: Vec<usize>,
}

impl TCNTemporalNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, TemporalError> {
        Ok(Self {
            architecture,
            conv_weights: vec![0.3; 256],
            dilation_levels: vec![1, 2, 4, 8],
        })
    }
}

impl TemporalNeuralNetwork for TCNTemporalNetwork {
    fn process_sequence(&self, sequence: &[f32]) -> Result<Vec<f32>, TemporalError> {
        // Simplified TCN forward pass
        let mut output = vec![0.0; 64];
        
        for (i, &input) in sequence.iter().take(64).enumerate() {
            let weight_idx = i % self.conv_weights.len();
            let weight = self.conv_weights[weight_idx];
            
            // Temporal convolution with dilation
            output[i] = (input * weight).max(0.0); // ReLU activation
        }
        
        Ok(output)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn is_ready(&self) -> bool {
        !self.conv_weights.is_empty()
    }
}

pub struct GRUTemporalNetwork {
    architecture: SelectedArchitecture,
    gru_weights: Vec<f32>,
    hidden_state: Vec<f32>,
}

impl GRUTemporalNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, TemporalError> {
        Ok(Self {
            architecture,
            gru_weights: vec![0.35; 384], // GRU weight matrix
            hidden_state: vec![0.0; 64],  // Hidden state
        })
    }
}

impl TemporalNeuralNetwork for GRUTemporalNetwork {
    fn process_sequence(&self, sequence: &[f32]) -> Result<Vec<f32>, TemporalError> {
        // Simplified GRU forward pass
        let mut output = vec![0.0; 64];
        
        for (i, &input) in sequence.iter().take(64).enumerate() {
            let weight_idx = i % self.gru_weights.len();
            let weight = self.gru_weights[weight_idx];
            
            // Simplified GRU computation
            output[i] = (input * weight + self.hidden_state.get(i).unwrap_or(&0.0)).tanh();
        }
        
        Ok(output)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn is_ready(&self) -> bool {
        !self.gru_weights.is_empty()
    }
}

// Exception Detection Column Implementation
// (This would normally be in a separate file: src/multi_column/exception_column.rs)

use std::collections::HashMap;

/// Exception detection cortical column
#[derive(Debug)]
pub struct ExceptionDetectionColumn {
    /// Selected optimal neural network architecture
    selected_architecture: Option<SelectedArchitecture>,
    
    /// Neural network for exception detection
    neural_network: Option<Box<dyn ExceptionNeuralNetwork>>,
    
    /// Anomaly detection engine
    anomaly_detector: AnomalyDetector,
    
    /// Inheritance validation system
    inheritance_validator: InheritanceValidator,
    
    /// Exception pattern library
    pattern_library: ExceptionPatternLibrary,
    
    /// Activation threshold
    activation_threshold: f32,
    
    /// Exception detection cache
    exception_cache: DashMap<ConceptId, CachedExceptionResult>,
    
    /// SIMD processor
    simd_processor: SIMDSpikeProcessor,
    
    /// Performance monitoring
    performance_monitor: ExceptionPerformanceMonitor,
}

/// Concept with inheritance information
#[derive(Debug, Clone)]
pub struct ConceptWithInheritance {
    /// Concept name
    pub concept_name: String,
    
    /// Inherited properties from parent
    pub inherited_properties: HashMap<String, String>,
    
    /// Actual properties of this concept
    pub actual_properties: HashMap<String, String>,
    
    /// Inheritance context
    pub context: InheritanceContext,
}

/// Inheritance context types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InheritanceContext {
    Biological,   // Biological taxonomy
    Mechanical,   // Mechanical/engineering
    Conceptual,   // Abstract concepts
    Geographic,   // Geographic entities
    Financial,    // Financial domain
    Legal,        // Legal domain
    General,      // General purpose
}

/// Exception features detected
#[derive(Debug, Clone)]
pub struct ExceptionFeatures {
    /// Type of violation detected
    pub violation_type: ViolationType,
    
    /// Inheritance violations found
    pub inheritance_violations: Vec<InheritanceViolation>,
    
    /// Anomaly score
    pub anomaly_score: f32,
    
    /// Context relevance
    pub context_relevance: f32,
    
    /// Pattern similarity to known exceptions
    pub pattern_similarity: f32,
    
    /// Whether this matches a learned pattern
    pub learned_pattern_match: bool,
}

/// Types of violations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViolationType {
    None,               // No violation
    PropertyOverride,   // Property value overridden
    PropertyMissing,    // Expected property missing
    LogicalConflict,    // Logical contradiction
    TypeMismatch,       // Type incompatibility
    ContextualError,    // Context-specific error
}

/// Inheritance violation details
#[derive(Debug, Clone)]
pub struct InheritanceViolation {
    /// Property name
    pub property: String,
    
    /// Expected value (from inheritance)
    pub expected_value: String,
    
    /// Actual value
    pub actual_value: String,
    
    /// Violation severity
    pub severity: ViolationSeverity,
    
    /// Explanation of violation
    pub explanation: String,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViolationSeverity {
    Minor,      // Small deviation
    Moderate,   // Significant deviation
    Major,      // Clear violation
    Critical,   // Fundamental contradiction
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub anomaly_score: f32,
    
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    
    /// Confidence in detection
    pub detection_confidence: f32,
    
    /// Anomaly description
    pub description: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    None,           // No anomaly
    Statistical,    // Statistical outlier
    Logical,        // Logical contradiction
    Temporal,       // Temporal inconsistency
    Contextual,     // Context-specific anomaly
    Structural,     // Structural inconsistency
}

/// Cached exception result
#[derive(Debug, Clone)]
pub struct CachedExceptionResult {
    /// Column vote
    pub column_vote: ColumnVote,
    
    /// Exception features
    pub exception_features: ExceptionFeatures,
    
    /// Cache metadata
    pub cached_at: Instant,
    pub hit_count: u32,
}

/// Column vote result for exception processing
#[derive(Debug, Clone)]
pub struct ExceptionColumnVote {
    /// Base column vote
    pub column_vote: ColumnVote,
    
    /// Exception-specific features
    pub exception_features: Option<ExceptionFeatures>,
    
    /// Anomaly detection result
    pub anomaly_result: Option<AnomalyResult>,
    
    /// Context influence on detection
    pub context_influence: f32,
}

/// Anomaly detection engine
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Detection algorithms
    detectors: Vec<Box<dyn AnomalyDetectionAlgorithm>>,
    
    /// Detection thresholds
    thresholds: HashMap<AnomalyType, f32>,
    
    /// Is active flag
    is_active: bool,
}

/// Inheritance validation system
#[derive(Debug)]
pub struct InheritanceValidator {
    /// Validation rules
    validation_rules: Vec<Box<dyn InheritanceRule>>,
    
    /// Context-specific validators
    context_validators: HashMap<InheritanceContext, Box<dyn ContextValidator>>,
    
    /// Ready state
    is_ready: bool,
}

/// Exception pattern library
#[derive(Debug)]
pub struct ExceptionPatternLibrary {
    /// Known exception patterns
    known_patterns: DashMap<String, ExceptionPattern>,
    
    /// Pattern statistics
    pattern_stats: ExceptionPatternStats,
    
    /// Loading state
    is_loaded: bool,
}

/// Exception pattern
#[derive(Debug, Clone)]
pub struct ExceptionPattern {
    /// Pattern identifier
    pub pattern_id: String,
    
    /// Pattern features
    pub features: Vec<f32>,
    
    /// Exception type
    pub exception_type: ViolationType,
    
    /// Context applicability
    pub applicable_contexts: Vec<InheritanceContext>,
    
    /// Usage frequency
    pub usage_frequency: usize,
}

/// Exception pattern statistics
#[derive(Debug, Default)]
pub struct ExceptionPatternStats {
    pub known_exception_patterns: usize,
    pub pattern_matches: usize,
    pub false_positives: usize,
    pub true_positives: usize,
}

/// Performance monitoring for exception column
#[derive(Debug, Default)]
pub struct ExceptionPerformanceMonitor {
    /// Detection times
    pub detection_times: Vec<Duration>,
    
    /// Detection accuracy
    pub detection_accuracy: f32,
    
    /// False positive rate
    pub false_positive_rate: f32,
    
    /// Cache performance
    pub cache_stats: CacheStatistics,
}

/// Neural network abstraction for exception detection
pub trait ExceptionNeuralNetwork: Send + Sync {
    /// Process exception detection input
    fn detect_exceptions(&self, input: &[f32]) -> Result<Vec<f32>, ExceptionError>;
    
    /// Get architecture information
    fn architecture_info(&self) -> &SelectedArchitecture;
    
    /// Check readiness
    fn is_ready(&self) -> bool;
}

/// Anomaly detection algorithm trait
pub trait AnomalyDetectionAlgorithm: Send + Sync {
    /// Detect anomalies in data
    fn detect(&self, data: &[f32]) -> AnomalyResult;
    
    /// Algorithm name
    fn name(&self) -> &str;
}

/// Inheritance rule trait
pub trait InheritanceRule: Send + Sync {
    /// Validate inheritance
    fn validate(&self, concept: &ConceptWithInheritance) -> Result<(), InheritanceViolation>;
    
    /// Rule name
    fn rule_name(&self) -> &str;
}

/// Context validator trait
pub trait ContextValidator: Send + Sync {
    /// Validate in specific context
    fn validate_context(&self, concept: &ConceptWithInheritance) -> f32;
    
    /// Context type
    fn context_type(&self) -> InheritanceContext;
}

/// Exception processing errors
#[derive(Debug, thiserror::Error)]
pub enum ExceptionError {
    #[error("Architecture selection failed: {0}")]
    ArchitectureSelectionFailed(String),
    
    #[error("Exception detection failed: {0}")]
    ExceptionDetectionFailed(String),
    
    #[error("Invalid inheritance structure: {0}")]
    InvalidInheritanceStructure(String),
    
    #[error("Network not initialized")]
    NetworkNotInitialized,
}

impl ExceptionDetectionColumn {
    /// Create exception column with automatic architecture selection
    pub fn new_with_auto_selection(selector: &ArchitectureSelector) -> Result<Self, ExceptionError> {
        let start_time = Instant::now();
        
        // Select optimal architecture for exception/anomaly detection
        let exception_candidates = selector.select_for_task_type(TaskType::Exception);
        let classification_candidates = selector.select_for_task_type(TaskType::Classification);
        
        let mut all_candidates = exception_candidates;
        all_candidates.extend(classification_candidates);
        
        let selected_arch = all_candidates.into_iter()
            .max_by(|a, b| {
                let score_a = Self::calculate_exception_suitability_score(a);
                let score_b = Self::calculate_exception_suitability_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .ok_or_else(|| ExceptionError::ArchitectureSelectionFailed("No suitable exception detection architecture found".to_string()))?;
        
        let neural_network = Self::load_exception_neural_network(&selected_arch)?;
        
        let column = Self {
            selected_architecture: Some(selected_arch),
            neural_network: Some(neural_network),
            anomaly_detector: AnomalyDetector::new(),
            inheritance_validator: InheritanceValidator::new(),
            pattern_library: ExceptionPatternLibrary::new(),
            activation_threshold: 0.7, // Higher threshold for exceptions
            exception_cache: DashMap::new(),
            simd_processor: SIMDSpikeProcessor::new(Default::default()),
            performance_monitor: ExceptionPerformanceMonitor::default(),
        };
        
        let initialization_time = start_time.elapsed();
        println!("Exception column initialized in {:?} with architecture: {}", 
                initialization_time, 
                column.selected_architecture.as_ref().unwrap().architecture.name);
        
        Ok(column)
    }
    
    /// Find inhibitory patterns (exceptions) in spike pattern
    pub fn find_inhibitions(&self, spike_pattern: &TTFSSpikePattern) -> Result<ColumnVote, ExceptionError> {
        let start_time = Instant::now();
        
        // Check cache
        if let Some(cached) = self.check_cache(&spike_pattern.concept_id()) {
            return Ok(cached.column_vote);
        }
        
        // Extract concept with inheritance from spike pattern
        let concept_with_inheritance = self.extract_concept_with_inheritance(spike_pattern)?;
        
        // Validate inheritance
        let inheritance_violations = self.inheritance_validator.validate(&concept_with_inheritance);
        
        // Detect anomalies
        let anomaly_result = self.anomaly_detector.detect_anomalies(&concept_with_inheritance);
        
        // Check against known exception patterns
        let pattern_match = self.pattern_library.find_matching_pattern(&concept_with_inheritance);
        
        // Prepare neural input
        let neural_input = self.prepare_exception_neural_input(&concept_with_inheritance, &inheritance_violations, &anomaly_result);
        
        // Process through neural network
        let neural_output = self.neural_network.as_ref()
            .ok_or(ExceptionError::NetworkNotInitialized)?
            .detect_exceptions(&neural_input)?;
        
        // Extract exception features
        let exception_features = self.extract_exception_features(&neural_output, inheritance_violations, &anomaly_result, pattern_match);
        
        // Calculate confidence and activation
        let confidence = self.calculate_exception_confidence(&exception_features, &concept_with_inheritance);
        let activation = if confidence > self.activation_threshold { confidence } else { 0.0 };
        
        // Create column vote
        let column_vote = ColumnVote {
            column_id: ColumnId::Exception,
            confidence,
            activation,
            neural_output: neural_output.clone(),
            processing_time: start_time.elapsed(),
        };
        
        // Cache result
        self.cache_result(spike_pattern.concept_id(), &column_vote, &exception_features);
        
        // Update pattern library if new exception learned
        if exception_features.violation_type != ViolationType::None {
            self.pattern_library.learn_pattern(&concept_with_inheritance, &exception_features);
        }
        
        Ok(column_vote)
    }
    
    /// Detect anomalies in spike pattern
    pub fn detect_anomalies(&self, spike_pattern: &TTFSSpikePattern) -> Result<AnomalyResult, ExceptionError> {
        let concept_with_inheritance = self.extract_concept_with_inheritance(spike_pattern)?;
        Ok(self.anomaly_detector.detect_anomalies(&concept_with_inheritance))
    }
    
    /// Process multiple exceptions in parallel
    pub fn process_multiple_exceptions_parallel(&self, spike_patterns: &[TTFSSpikePattern]) -> Result<Vec<ColumnVote>, ExceptionError> {
        spike_patterns.par_iter()
            .map(|pattern| self.find_inhibitions(pattern))
            .collect()
    }
    
    /// Get pattern library statistics
    pub fn get_pattern_library_statistics(&self) -> ExceptionPatternStats {
        self.pattern_library.pattern_stats.clone()
    }
    
    /// Check if column is ready
    pub fn is_ready(&self) -> bool {
        self.neural_network.is_some() && 
        self.anomaly_detector.is_active() &&
        self.inheritance_validator.is_ready() &&
        self.pattern_library.is_loaded()
    }
    
    /// Get selected architecture
    pub fn get_selected_architecture(&self) -> &SelectedArchitecture {
        self.selected_architecture.as_ref().unwrap()
    }
    
    // Private helper methods
    
    fn calculate_exception_suitability_score(architecture: &crate::ruv_fann_integration::ArchitectureCandidate) -> f32 {
        let mut score = architecture.performance_metrics.performance_score;
        
        // Prefer classification architectures for exception detection
        if architecture.supported_tasks.contains(&TaskType::Exception) {
            score += 0.3;
        }
        
        if architecture.supported_tasks.contains(&TaskType::Classification) {
            score += 0.2; // MLPs are good for anomaly detection
        }
        
        // Prefer smaller, faster networks for exception detection
        if architecture.memory_profile.memory_footprint < 30_000_000 {
            score += 0.1;
        }
        
        score
    }
    
    fn load_exception_neural_network(architecture: &SelectedArchitecture) -> Result<Box<dyn ExceptionNeuralNetwork>, ExceptionError> {
        // Default to MLP for exception detection
        Ok(Box::new(MLPExceptionNetwork::new(architecture.clone())?))
    }
    
    fn check_cache(&self, concept_id: &ConceptId) -> Option<CachedExceptionResult> {
        self.exception_cache.get(concept_id).map(|entry| entry.value().clone())
    }
    
    fn cache_result(&self, concept_id: ConceptId, column_vote: &ColumnVote, exception_features: &ExceptionFeatures) {
        let cached_result = CachedExceptionResult {
            column_vote: column_vote.clone(),
            exception_features: exception_features.clone(),
            cached_at: Instant::now(),
            hit_count: 0,
        };
        
        self.exception_cache.insert(concept_id, cached_result);
        
        // Cache management
        if self.exception_cache.len() > 300 {
            let oldest_key = self.exception_cache.iter()
                .min_by_key(|entry| entry.value().cached_at)
                .map(|entry| entry.key().clone());
            
            if let Some(key) = oldest_key {
                self.exception_cache.remove(&key);
            }
        }
    }
    
    fn extract_concept_with_inheritance(&self, spike_pattern: &TTFSSpikePattern) -> Result<ConceptWithInheritance, ExceptionError> {
        // Mock extraction - in real implementation would analyze spike pattern
        Ok(ConceptWithInheritance {
            concept_name: spike_pattern.concept_id().name().to_string(),
            inherited_properties: HashMap::new(),
            actual_properties: HashMap::new(),
            context: InheritanceContext::General,
        })
    }
    
    fn prepare_exception_neural_input(&self, 
                                    concept: &ConceptWithInheritance, 
                                    violations: &[InheritanceViolation], 
                                    anomaly: &AnomalyResult) -> Vec<f32> {
        let mut input = vec![0.0; 64]; // Smaller input for exception detection
        
        // Encode basic concept information
        input[0] = concept.inherited_properties.len() as f32 / 10.0; // Normalize
        input[1] = concept.actual_properties.len() as f32 / 10.0;
        input[2] = violations.len() as f32 / 5.0; // Normalize violation count
        input[3] = anomaly.anomaly_score;
        
        // Encode context
        input[4] = match concept.context {
            InheritanceContext::Biological => 1.0,
            InheritanceContext::Mechanical => 0.8,
            InheritanceContext::Conceptual => 0.6,
            InheritanceContext::Geographic => 0.4,
            InheritanceContext::Financial => 0.2,
            InheritanceContext::Legal => 0.1,
            InheritanceContext::General => 0.0,
        };
        
        // Encode violation details
        for (i, violation) in violations.iter().take(10).enumerate() {
            let base_idx = 5 + i * 2;
            if base_idx + 1 < input.len() {
                input[base_idx] = match violation.severity {
                    ViolationSeverity::Minor => 0.25,
                    ViolationSeverity::Moderate => 0.5,
                    ViolationSeverity::Major => 0.75,
                    ViolationSeverity::Critical => 1.0,
                };
                input[base_idx + 1] = 1.0; // Violation presence
            }
        }
        
        input
    }
    
    fn extract_exception_features(&self, 
                                neural_output: &[f32], 
                                violations: Vec<InheritanceViolation>, 
                                anomaly: &AnomalyResult,
                                pattern_match: Option<&ExceptionPattern>) -> ExceptionFeatures {
        let violation_type = if violations.is_empty() {
            if anomaly.anomaly_score > 0.7 {
                match anomaly.anomaly_type {
                    AnomalyType::Logical => ViolationType::LogicalConflict,
                    AnomalyType::Contextual => ViolationType::ContextualError,
                    _ => ViolationType::None,
                }
            } else {
                ViolationType::None
            }
        } else {
            ViolationType::PropertyOverride
        };
        
        let pattern_similarity = pattern_match.map(|_| 0.8).unwrap_or(0.0);
        let learned_pattern_match = pattern_match.is_some();
        
        ExceptionFeatures {
            violation_type,
            inheritance_violations: violations,
            anomaly_score: anomaly.anomaly_score,
            context_relevance: neural_output.get(0).copied().unwrap_or(0.5),
            pattern_similarity,
            learned_pattern_match,
        }
    }
    
    fn calculate_exception_confidence(&self, features: &ExceptionFeatures, _concept: &ConceptWithInheritance) -> f32 {
        let violation_weight = 0.4;
        let anomaly_weight = 0.3;
        let pattern_weight = 0.3;
        
        let violation_score = if features.violation_type == ViolationType::None { 0.0 } else { 1.0 };
        let pattern_score = if features.learned_pattern_match { features.pattern_similarity } else { 0.0 };
        
        (violation_score * violation_weight +
         features.anomaly_score * anomaly_weight +
         pattern_score * pattern_weight).clamp(0.0, 1.0)
    }
}

// Supporting structure implementations

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            thresholds: HashMap::new(),
            is_active: true,
        }
    }
    
    pub fn detect_anomalies(&self, concept: &ConceptWithInheritance) -> AnomalyResult {
        // Mock anomaly detection
        let anomaly_score = if concept.inherited_properties.len() != concept.actual_properties.len() {
            0.8 // High anomaly if property counts differ significantly
        } else {
            0.2 // Low anomaly for normal cases
        };
        
        let anomaly_type = if anomaly_score > 0.7 {
            AnomalyType::Logical
        } else {
            AnomalyType::None
        };
        
        AnomalyResult {
            anomaly_score,
            anomaly_type,
            detection_confidence: 0.85,
            description: format!("Detected {} anomaly with score {:.2}", 
                               format!("{:?}", anomaly_type).to_lowercase(), anomaly_score),
        }
    }
    
    pub fn is_active(&self) -> bool {
        self.is_active
    }
}

impl InheritanceValidator {
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
            context_validators: HashMap::new(),
            is_ready: true,
        }
    }
    
    pub fn validate(&self, concept: &ConceptWithInheritance) -> Vec<InheritanceViolation> {
        let mut violations = Vec::new();
        
        // Check for property overrides
        for (prop, inherited_value) in &concept.inherited_properties {
            if let Some(actual_value) = concept.actual_properties.get(prop) {
                if inherited_value != actual_value {
                    violations.push(InheritanceViolation {
                        property: prop.clone(),
                        expected_value: inherited_value.clone(),
                        actual_value: actual_value.clone(),
                        severity: ViolationSeverity::Moderate,
                        explanation: format!("Property '{}' overridden: expected '{}', got '{}'", 
                                           prop, inherited_value, actual_value),
                    });
                }
            }
        }
        
        violations
    }
    
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }
}

impl ExceptionPatternLibrary {
    pub fn new() -> Self {
        Self {
            known_patterns: DashMap::new(),
            pattern_stats: ExceptionPatternStats::default(),
            is_loaded: true,
        }
    }
    
    pub fn find_matching_pattern(&self, _concept: &ConceptWithInheritance) -> Option<&ExceptionPattern> {
        // Mock pattern matching
        None
    }
    
    pub fn learn_pattern(&self, _concept: &ConceptWithInheritance, _features: &ExceptionFeatures) {
        // Mock pattern learning
    }
    
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }
}

// Neural network implementation for exception detection

pub struct MLPExceptionNetwork {
    architecture: SelectedArchitecture,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
}

impl MLPExceptionNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, ExceptionError> {
        Ok(Self {
            architecture,
            weights: vec![vec![0.3; 64]; 3], // 3 layers for exception detection
            biases: vec![vec![0.0; 32]; 3],   // Smaller output for exception detection
        })
    }
}

impl ExceptionNeuralNetwork for MLPExceptionNetwork {
    fn detect_exceptions(&self, input: &[f32]) -> Result<Vec<f32>, ExceptionError> {
        let mut current = input.to_vec();
        
        for layer in 0..self.weights.len() {
            let mut next_layer = vec![0.0; self.biases[layer].len()];
            
            for (i, bias) in self.biases[layer].iter().enumerate() {
                let weighted_sum: f32 = current.iter()
                    .take(self.weights[layer].len())
                    .enumerate()
                    .map(|(j, &x)| x * self.weights[layer].get(j).unwrap_or(&0.3))
                    .sum();
                
                next_layer[i] = (weighted_sum + bias).max(0.0); // ReLU
            }
            
            current = next_layer;
        }
        
        Ok(current)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn is_ready(&self) -> bool {
        !self.weights.is_empty()
    }
}

// Helper macro for creating hash maps in tests
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
        let mut map = ::std::collections::HashMap::new();
        $( map.insert($key, $val); )*
        map
    }}
}

// Re-export for use in tests
pub(crate) use hashmap;
```

## Verification Steps
1. Implement temporal column with optimal architecture selection for sequence processing (LSTM/TCN/GRU)
2. Add temporal sequence analysis with coherence, consistency, and pattern recognition
3. Implement temporal memory system for storing and matching sequence patterns
4. Add temporal context processing with past/present/future directional analysis
5. Implement exception detection column with inheritance validation and anomaly detection
6. Add exception pattern library with learning capabilities for known exception types
7. Implement context-dependent exception detection for different domains (biological, financial, etc.)
8. Add parallel processing support for both temporal and exception columns

## Success Criteria
- [ ] Temporal column initializes with optimal sequence processing architecture in <200ms
- [ ] Temporal sequence processing completes in <1ms per pattern (sub-millisecond target)
- [ ] Temporal pattern recognition accurately classifies Sequential/Parallel/Cyclical/Random patterns
- [ ] Temporal memory integration shows learning and similarity matching capabilities
- [ ] Exception column initializes with optimal anomaly detection architecture in <200ms
- [ ] Exception detection processing completes in <1ms per pattern (sub-millisecond target)
- [ ] Inheritance violation detection accurately identifies property overrides and conflicts
- [ ] Exception pattern learning adapts to known exception types (platypus, penguin examples)
- [ ] Context-dependent processing shows different behavior across domains
- [ ] Parallel processing of both temporal and exception patterns provides >1.5x speedup
- [ ] Both columns meet memory constraints (<50MB each) and performance targets
- [ ] Integration with other cortical columns works seamlessly through column vote system