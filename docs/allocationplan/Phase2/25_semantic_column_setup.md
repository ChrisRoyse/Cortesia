# Task 25: Semantic Column Setup with Optimal Architecture

## Metadata
- **Micro-Phase**: 2.25
- **Duration**: 40-45 minutes
- **Dependencies**: Task 24 (architecture_selection_framework), Task 15 (ttfs_encoder_base), Task 20 (simd_spike_processor)
- **Output**: `src/multi_column/semantic_column.rs`

## Description
Implement semantic processing cortical column using intelligently selected optimal neural network architecture from ruv-FANN. This column handles semantic similarity, concept classification, and meaning extraction using Time-to-First-Spike encoding with sub-millisecond processing targets. The architecture is selected based on performance benchmarks rather than implementing all available options.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId};
    use crate::ruv_fann_integration::ArchitectureSelector;
    use std::time::{Duration, Instant};

    #[test]
    fn test_semantic_column_initialization() {
        let architecture_selector = ArchitectureSelector::new();
        
        // Should automatically select optimal architecture for semantic processing
        let column = SemanticProcessingColumn::new_with_auto_selection(&architecture_selector).unwrap();
        
        // Verify optimal architecture selected
        assert!(column.selected_architecture.is_some());
        let arch = column.selected_architecture.as_ref().unwrap();
        assert!(arch.supported_tasks.contains(&TaskType::Semantic));
        assert!(arch.performance_metrics.accuracy >= 0.85);
        assert!(arch.memory_profile.memory_footprint <= 50_000_000); // 50MB limit
        
        // Verify initialization completed successfully
        assert!(column.is_ready());
        assert_eq!(column.activation_threshold, 0.7);
        assert!(column.semantic_cache.is_empty());
    }
    
    #[test]
    fn test_semantic_spike_processing() {
        let column = create_test_semantic_column();
        
        // Create test spike pattern for semantic concept
        let spike_pattern = create_semantic_spike_pattern("African elephant", 0.9);
        
        let start = Instant::now();
        let result = column.process_spikes(&spike_pattern).unwrap();
        let processing_time = start.elapsed();
        
        // Verify processing speed
        assert!(processing_time < Duration::from_millis(1)); // Sub-millisecond target
        
        // Verify result structure
        assert_eq!(result.column_id, ColumnId::Semantic);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.activation >= 0.0 && result.activation <= 1.0);
        assert!(!result.neural_output.is_empty());
        
        // Verify semantic-specific processing
        assert!(result.semantic_features.is_some());
        let semantic_features = result.semantic_features.unwrap();
        assert_eq!(semantic_features.len(), 128); // Standard semantic feature size
        assert!(semantic_features.iter().all(|&f| f.is_finite()));
    }
    
    #[test]
    fn test_semantic_similarity_computation() {
        let column = create_test_semantic_column();
        
        // Test similar concepts
        let elephant_pattern = create_semantic_spike_pattern("elephant", 0.9);
        let mammoth_pattern = create_semantic_spike_pattern("mammoth", 0.8);
        
        let elephant_result = column.process_spikes(&elephant_pattern).unwrap();
        let mammoth_result = column.process_spikes(&mammoth_pattern).unwrap();
        
        let similarity = column.calculate_semantic_similarity(
            &elephant_result.semantic_features.unwrap(),
            &mammoth_result.semantic_features.unwrap()
        );
        
        // Similar concepts should have high similarity
        assert!(similarity > 0.7, "Similar concepts should have high similarity: {}", similarity);
        
        // Test dissimilar concepts
        let car_pattern = create_semantic_spike_pattern("car", 0.9);
        let car_result = column.process_spikes(&car_pattern).unwrap();
        
        let dissimilarity = column.calculate_semantic_similarity(
            &elephant_result.semantic_features.unwrap(),
            &car_result.semantic_features.unwrap()
        );
        
        // Dissimilar concepts should have low similarity
        assert!(dissimilarity < 0.3, "Dissimilar concepts should have low similarity: {}", dissimilarity);
    }
    
    #[test]
    fn test_ttfs_to_neural_input_conversion() {
        let column = create_test_semantic_column();
        
        // Test conversion with various spike patterns
        let early_spike_pattern = TTFSSpikePattern::new(
            ConceptId::new("test"),
            Duration::from_micros(100), // Very early spike
            create_test_spikes(5),
            Duration::from_millis(2),
        );
        
        let neural_input = column.prepare_neural_input(&early_spike_pattern).unwrap();
        
        // Verify input format
        assert_eq!(neural_input.len(), 256); // Standard input size for semantic processing
        assert!(neural_input.iter().all(|&x| x >= 0.0 && x <= 1.0)); // Normalized range
        
        // Verify early spikes produce higher activation values
        let early_activations: Vec<_> = neural_input.iter().take(5).cloned().collect();
        assert!(early_activations.iter().any(|&x| x > 0.8)); // Strong early activation
        
        // Test late spike pattern
        let late_spike_pattern = TTFSSpikePattern::new(
            ConceptId::new("test_late"),
            Duration::from_millis(5), // Late spike
            create_test_spikes(3),
            Duration::from_millis(10),
        );
        
        let late_neural_input = column.prepare_neural_input(&late_spike_pattern).unwrap();
        let late_activations: Vec<_> = late_neural_input.iter().take(3).cloned().collect();
        
        // Late spikes should produce lower activation values
        assert!(late_activations.iter().all(|&x| x < 0.5));
    }
    
    #[test]
    fn test_semantic_caching_performance() {
        let mut column = create_test_semantic_column();
        
        let test_pattern = create_semantic_spike_pattern("cached_concept", 0.8);
        
        // First processing - should be slow (cache miss)
        let start = Instant::now();
        let first_result = column.process_spikes(&test_pattern).unwrap();
        let first_time = start.elapsed();
        
        // Second processing - should be fast (cache hit)
        let start = Instant::now();
        let second_result = column.process_spikes(&test_pattern).unwrap();
        let second_time = start.elapsed();
        
        // Verify caching effectiveness
        assert!(second_time < first_time / 2, "Cached processing should be at least 2x faster");
        assert!(second_time < Duration::from_micros(100)); // Very fast cache retrieval
        
        // Results should be identical
        assert_eq!(first_result.confidence, second_result.confidence);
        assert_eq!(first_result.activation, second_result.activation);
        
        // Verify cache statistics
        let cache_stats = column.get_cache_statistics();
        assert_eq!(cache_stats.hits, 1);
        assert_eq!(cache_stats.misses, 1);
        assert_eq!(cache_stats.hit_rate, 0.5);
    }
    
    #[test]
    fn test_context_aware_processing() {
        let column = create_test_semantic_column();
        
        // Test context influence on semantic processing
        let base_pattern = create_semantic_spike_pattern("bank", 0.8);
        
        // Financial context
        let financial_context = SemanticContext::new(vec![
            "money".to_string(),
            "account".to_string(),
            "transaction".to_string(),
        ]);
        
        let financial_result = column.process_with_context(&base_pattern, &financial_context).unwrap();
        
        // River context
        let river_context = SemanticContext::new(vec![
            "water".to_string(),
            "river".to_string(),
            "shore".to_string(),
        ]);
        
        let river_result = column.process_with_context(&base_pattern, &river_context).unwrap();
        
        // Results should be different based on context
        assert_ne!(financial_result.confidence, river_result.confidence);
        
        // Context should influence semantic interpretation
        assert!(financial_result.context_influence > 0.1);
        assert!(river_result.context_influence > 0.1);
        assert_ne!(financial_result.context_influence, river_result.context_influence);
    }
    
    #[test]
    fn test_parallel_semantic_processing() {
        let column = create_test_semantic_column();
        
        // Create multiple semantic patterns
        let patterns = vec![
            create_semantic_spike_pattern("dog", 0.9),
            create_semantic_spike_pattern("cat", 0.8),
            create_semantic_spike_pattern("bird", 0.85),
            create_semantic_spike_pattern("fish", 0.7),
        ];
        
        let start = Instant::now();
        let results = column.process_multiple_spikes_parallel(&patterns).unwrap();
        let parallel_time = start.elapsed();
        
        // Verify results
        assert_eq!(results.len(), patterns.len());
        
        // Verify parallel processing speed improvement
        let start = Instant::now();
        let sequential_results: Vec<_> = patterns.iter()
            .map(|pattern| column.process_spikes(pattern).unwrap())
            .collect();
        let sequential_time = start.elapsed();
        
        // Parallel should be faster (with overhead consideration)
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        assert!(speedup > 1.5, "Parallel processing should provide speedup: {:.2}x", speedup);
        
        // Results should be equivalent
        for (parallel, sequential) in results.iter().zip(sequential_results.iter()) {
            assert!((parallel.confidence - sequential.confidence).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_semantic_feature_extraction() {
        let column = create_test_semantic_column();
        
        // Test feature extraction for hierarchical concepts
        let animal_pattern = create_semantic_spike_pattern("animal", 0.9);
        let dog_pattern = create_semantic_spike_pattern("dog", 0.9);
        let golden_retriever_pattern = create_semantic_spike_pattern("golden retriever", 0.9);
        
        let animal_features = column.extract_semantic_features(&animal_pattern).unwrap();
        let dog_features = column.extract_semantic_features(&dog_pattern).unwrap();
        let retriever_features = column.extract_semantic_features(&golden_retriever_pattern).unwrap();
        
        // Verify feature dimensionality
        assert_eq!(animal_features.len(), 128);
        assert_eq!(dog_features.len(), 128);
        assert_eq!(retriever_features.len(), 128);
        
        // Verify hierarchical relationships in feature space
        let animal_dog_similarity = column.calculate_semantic_similarity(&animal_features, &dog_features);
        let dog_retriever_similarity = column.calculate_semantic_similarity(&dog_features, &retriever_features);
        let animal_retriever_similarity = column.calculate_semantic_similarity(&animal_features, &retriever_features);
        
        // Closer concepts should be more similar
        assert!(dog_retriever_similarity > animal_dog_similarity);
        assert!(animal_dog_similarity > animal_retriever_similarity);
        assert!(dog_retriever_similarity > 0.7); // Strong similarity
    }
    
    #[test]
    fn test_activation_threshold_sensitivity() {
        let mut column = create_test_semantic_column();
        
        let test_pattern = create_semantic_spike_pattern("test_concept", 0.75);
        
        // Test with default threshold (0.7)
        let default_result = column.process_spikes(&test_pattern).unwrap();
        assert!(default_result.activation > 0.0); // Should activate
        
        // Test with high threshold (0.9)
        column.set_activation_threshold(0.9);
        let high_threshold_result = column.process_spikes(&test_pattern).unwrap();
        assert!(high_threshold_result.activation < default_result.activation); // Should activate less
        
        // Test with low threshold (0.3)
        column.set_activation_threshold(0.3);
        let low_threshold_result = column.process_spikes(&test_pattern).unwrap();
        assert!(low_threshold_result.activation > default_result.activation); // Should activate more
        
        // Verify threshold enforcement
        column.set_activation_threshold(0.8);
        let weak_pattern = create_semantic_spike_pattern("weak_concept", 0.4);
        let weak_result = column.process_spikes(&weak_pattern).unwrap();
        assert!(weak_result.activation < 0.1); // Should barely activate
    }
    
    #[test]
    fn test_semantic_memory_integration() {
        let column = create_test_semantic_column();
        
        // Test integration with existing semantic memory
        let known_concepts = vec![
            ("dog", vec![0.8, 0.9, 0.1, 0.2]),
            ("cat", vec![0.7, 0.8, 0.15, 0.25]),
            ("car", vec![0.1, 0.2, 0.9, 0.8]),
        ];
        
        column.preload_semantic_memory(&known_concepts).unwrap();
        
        // Test new concept processing
        let new_pattern = create_semantic_spike_pattern("puppy", 0.9);
        let result = column.process_spikes(&new_pattern).unwrap();
        
        // Should find similarity to existing "dog" concept
        assert!(result.memory_activation.is_some());
        let memory_activation = result.memory_activation.unwrap();
        assert!(memory_activation.similar_concepts.len() > 0);
        
        let most_similar = &memory_activation.similar_concepts[0];
        assert_eq!(most_similar.concept_name, "dog");
        assert!(most_similar.similarity_score > 0.6);
    }
    
    #[test]
    fn test_architecture_performance_validation() {
        let column = create_test_semantic_column();
        
        // Verify selected architecture meets performance requirements
        let architecture = column.get_selected_architecture();
        
        // Memory requirements
        assert!(architecture.memory_profile.memory_footprint <= 50_000_000); // 50MB max
        
        // Performance requirements
        assert!(architecture.performance_metrics.inference_time <= Duration::from_millis(1));
        assert!(architecture.performance_metrics.accuracy >= 0.85);
        
        // SIMD compatibility
        assert!(architecture.simd_compatibility.vectorization_efficiency >= 0.7);
        assert!(architecture.simd_compatibility.supports_4x_speedup);
        
        // Task suitability
        assert!(architecture.supported_tasks.contains(&TaskType::Semantic));
        
        // Verify actual performance matches predicted
        let test_pattern = create_semantic_spike_pattern("performance_test", 0.8);
        let start = Instant::now();
        let result = column.process_spikes(&test_pattern).unwrap();
        let actual_time = start.elapsed();
        
        assert!(actual_time <= architecture.performance_metrics.inference_time * 2); // Allow some overhead
        assert!(result.confidence >= architecture.performance_metrics.accuracy - 0.1); // Allow some variance
    }
    
    // Helper functions
    fn create_test_semantic_column() -> SemanticProcessingColumn {
        let architecture_selector = ArchitectureSelector::new();
        SemanticProcessingColumn::new_with_auto_selection(&architecture_selector).unwrap()
    }
    
    fn create_semantic_spike_pattern(concept_name: &str, relevance: f32) -> TTFSSpikePattern {
        let concept_id = ConceptId::new(concept_name);
        let first_spike_time = Duration::from_nanos((1000.0 / relevance) as u64);
        let spikes = create_test_spikes(8);
        let total_duration = Duration::from_millis(5);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_test_spikes(count: usize) -> Vec<SpikeEvent> {
        (0..count).map(|i| {
            SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(100 + i as u64 * 200),
                0.5 + (i as f32 * 0.1) % 0.5,
            )
        }).collect()
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{TTFSSpikePattern, SpikeEvent, NeuronId, ConceptId};
use crate::ruv_fann_integration::{ArchitectureSelector, SelectedArchitecture, TaskType};
use crate::simd_spike_processor::SIMDSpikeProcessor;
use crate::multi_column::{ColumnVote, ColumnId};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Semantic processing cortical column using intelligently selected neural architecture
#[derive(Debug)]
pub struct SemanticProcessingColumn {
    /// Selected optimal neural network architecture
    selected_architecture: Option<SelectedArchitecture>,
    
    /// Neural network instance (loaded from selected architecture)
    neural_network: Option<Box<dyn SemanticNeuralNetwork>>,
    
    /// Activation threshold for semantic relevance
    activation_threshold: f32,
    
    /// Semantic feature cache for performance
    semantic_cache: DashMap<ConceptId, CachedSemanticResult>,
    
    /// SIMD processor for parallel spike processing
    simd_processor: SIMDSpikeProcessor,
    
    /// Semantic memory for concept relationships
    semantic_memory: SemanticMemory,
    
    /// Context processor for contextual interpretation
    context_processor: ContextProcessor,
    
    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
}

/// Cached semantic processing result
#[derive(Debug, Clone)]
pub struct CachedSemanticResult {
    /// Column vote result
    pub column_vote: ColumnVote,
    
    /// Extracted semantic features
    pub semantic_features: Vec<f32>,
    
    /// Cache timestamp
    pub cached_at: Instant,
    
    /// Cache hit count
    pub hit_count: u32,
}

/// Column vote result for semantic processing
#[derive(Debug, Clone)]
pub struct SemanticColumnVote {
    /// Base column vote
    pub column_vote: ColumnVote,
    
    /// Semantic-specific features
    pub semantic_features: Option<Vec<f32>>,
    
    /// Memory activation results
    pub memory_activation: Option<MemoryActivation>,
    
    /// Context influence on processing
    pub context_influence: f32,
    
    /// Confidence breakdown
    pub confidence_breakdown: ConfidenceBreakdown,
}

/// Memory activation results
#[derive(Debug, Clone)]
pub struct MemoryActivation {
    /// Similar concepts found in memory
    pub similar_concepts: Vec<SimilarConcept>,
    
    /// Memory retrieval confidence
    pub retrieval_confidence: f32,
    
    /// Memory influence on current processing
    pub memory_influence: f32,
}

/// Similar concept from semantic memory
#[derive(Debug, Clone)]
pub struct SimilarConcept {
    /// Concept name
    pub concept_name: String,
    
    /// Similarity score
    pub similarity_score: f32,
    
    /// Feature vector
    pub features: Vec<f32>,
}

/// Confidence breakdown for semantic processing
#[derive(Debug, Clone)]
pub struct ConfidenceBreakdown {
    /// Neural network confidence
    pub neural_confidence: f32,
    
    /// Feature extraction confidence
    pub feature_confidence: f32,
    
    /// Memory consistency confidence
    pub memory_confidence: f32,
    
    /// Context relevance confidence
    pub context_confidence: f32,
}

/// Semantic context for contextual processing
#[derive(Debug, Clone)]
pub struct SemanticContext {
    /// Context words/concepts
    pub context_words: Vec<String>,
    
    /// Context strength
    pub context_strength: f32,
    
    /// Context type
    pub context_type: ContextType,
}

/// Context type for semantic interpretation
#[derive(Debug, Clone, Copy)]
pub enum ContextType {
    Domain,        // Domain-specific context (medical, legal, etc.)
    Temporal,      // Temporal context (past, present, future)
    Emotional,     // Emotional context (positive, negative, neutral)
    Cultural,      // Cultural context
    Technical,     // Technical/scientific context
    General,       // General conversational context
}

/// Semantic memory for concept storage and retrieval
#[derive(Debug)]
pub struct SemanticMemory {
    /// Stored concept embeddings
    concept_embeddings: DashMap<String, Vec<f32>>,
    
    /// Concept relationships
    concept_relationships: DashMap<String, Vec<ConceptRelation>>,
    
    /// Memory capacity
    max_concepts: usize,
    
    /// Memory usage statistics
    usage_stats: MemoryUsageStats,
}

/// Concept relationship in semantic memory
#[derive(Debug, Clone)]
pub struct ConceptRelation {
    /// Related concept name
    pub related_concept: String,
    
    /// Relationship strength
    pub strength: f32,
    
    /// Relationship type
    pub relation_type: RelationType,
}

/// Relationship type between concepts
#[derive(Debug, Clone, Copy)]
pub enum RelationType {
    Similarity,    // Semantic similarity
    Hierarchy,     // Is-a relationships
    PartOf,        // Part-of relationships
    Association,   // Associative relationships
    Opposition,    // Opposite/contrary relationships
}

/// Memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryUsageStats {
    pub total_concepts: usize,
    pub total_relationships: usize,
    pub memory_utilization: f32,
    pub average_retrieval_time: Duration,
}

/// Context processor for contextual semantic interpretation
#[derive(Debug)]
pub struct ContextProcessor {
    /// Context embeddings
    context_embeddings: HashMap<ContextType, Vec<f32>>,
    
    /// Context influence weights
    influence_weights: HashMap<ContextType, f32>,
}

/// Performance monitoring for semantic column
#[derive(Debug, Default)]
pub struct PerformanceMonitor {
    /// Processing times
    pub processing_times: Vec<Duration>,
    
    /// Cache statistics
    pub cache_stats: CacheStatistics,
    
    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
    
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Cache performance statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
    pub average_hit_time: Duration,
    pub average_miss_time: Duration,
}

/// Accuracy tracking metrics
#[derive(Debug, Default)]
pub struct AccuracyMetrics {
    pub total_predictions: u64,
    pub correct_predictions: u64,
    pub accuracy_rate: f32,
    pub confidence_accuracy_correlation: f32,
}

/// Resource usage tracking
#[derive(Debug, Default)]
pub struct ResourceUsage {
    pub peak_memory_usage: usize,
    pub average_cpu_usage: f32,
    pub total_neural_operations: u64,
    pub simd_utilization_rate: f32,
}

/// Neural network abstraction for semantic processing
pub trait SemanticNeuralNetwork: Send + Sync {
    /// Process neural input and return semantic features
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, SemanticError>;
    
    /// Get network architecture information
    fn architecture_info(&self) -> &SelectedArchitecture;
    
    /// Get network performance metrics
    fn performance_metrics(&self) -> &PerformanceMetrics;
    
    /// Check if network is ready for processing
    fn is_ready(&self) -> bool;
}

/// Performance metrics for neural network
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub inference_time: Duration,
    pub memory_usage: usize,
    pub accuracy: f32,
    pub throughput: f32,
}

/// Semantic processing errors
#[derive(Debug, thiserror::Error)]
pub enum SemanticError {
    #[error("Architecture selection failed: {0}")]
    ArchitectureSelectionFailed(String),
    
    #[error("Neural network not initialized")]
    NetworkNotInitialized,
    
    #[error("Invalid input format: {0}")]
    InvalidInput(String),
    
    #[error("Processing timeout")]
    ProcessingTimeout,
    
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,
    
    #[error("SIMD processing error: {0}")]
    SIMDError(String),
}

impl SemanticProcessingColumn {
    /// Create new semantic column with automatic architecture selection
    pub fn new_with_auto_selection(selector: &ArchitectureSelector) -> Result<Self, SemanticError> {
        let start_time = Instant::now();
        
        // Select optimal architecture for semantic processing
        let selected_arch = selector.select_for_task_type(TaskType::Semantic)
            .into_iter()
            .max_by(|a, b| a.performance_metrics.performance_score.partial_cmp(&b.performance_metrics.performance_score).unwrap())
            .ok_or_else(|| SemanticError::ArchitectureSelectionFailed("No suitable architecture found".to_string()))?;
        
        // Load the selected neural network
        let neural_network = Self::load_neural_network(&selected_arch)?;
        
        let column = Self {
            selected_architecture: Some(selected_arch),
            neural_network: Some(neural_network),
            activation_threshold: 0.7,
            semantic_cache: DashMap::new(),
            simd_processor: SIMDSpikeProcessor::new(Default::default()),
            semantic_memory: SemanticMemory::new(10000), // 10K concept capacity
            context_processor: ContextProcessor::new(),
            performance_monitor: PerformanceMonitor::default(),
        };
        
        let initialization_time = start_time.elapsed();
        println!("Semantic column initialized in {:?} with architecture: {}", 
                initialization_time, 
                column.selected_architecture.as_ref().unwrap().architecture.name);
        
        Ok(column)
    }
    
    /// Create semantic column with specific architecture
    pub fn new_with_specific_architecture(architecture: SelectedArchitecture) -> Result<Self, SemanticError> {
        let neural_network = Self::load_neural_network(&architecture)?;
        
        Ok(Self {
            selected_architecture: Some(architecture),
            neural_network: Some(neural_network),
            activation_threshold: 0.7,
            semantic_cache: DashMap::new(),
            simd_processor: SIMDSpikeProcessor::new(Default::default()),
            semantic_memory: SemanticMemory::new(10000),
            context_processor: ContextProcessor::new(),
            performance_monitor: PerformanceMonitor::default(),
        })
    }
    
    /// Process spike pattern and generate column vote
    pub fn process_spikes(&self, spike_pattern: &TTFSSpikePattern) -> Result<ColumnVote, SemanticError> {
        let start_time = Instant::now();
        
        // Check cache first
        if let Some(cached) = self.check_cache(&spike_pattern.concept_id()) {
            self.performance_monitor.cache_stats.hits += 1;
            return Ok(cached.column_vote);
        }
        
        self.performance_monitor.cache_stats.misses += 1;
        
        // Convert TTFS spikes to neural input
        let neural_input = self.prepare_neural_input(spike_pattern)?;
        
        // Process through neural network
        let neural_output = self.neural_network.as_ref()
            .ok_or(SemanticError::NetworkNotInitialized)?
            .forward(&neural_input)?;
        
        // Extract semantic features
        let semantic_features = self.extract_semantic_features_from_output(&neural_output);
        
        // Calculate confidence and activation
        let confidence = self.calculate_semantic_confidence(&neural_output, &semantic_features);
        let activation = if confidence > self.activation_threshold { confidence } else { 0.0 };
        
        // Create column vote
        let column_vote = ColumnVote {
            column_id: ColumnId::Semantic,
            confidence,
            activation,
            neural_output: neural_output.clone(),
            processing_time: start_time.elapsed(),
        };
        
        // Cache result
        self.cache_result(spike_pattern.concept_id(), &column_vote, &semantic_features);
        
        // Update performance metrics
        self.update_performance_metrics(start_time.elapsed(), confidence);
        
        Ok(column_vote)
    }
    
    /// Process spikes with semantic context
    pub fn process_with_context(&self, 
                               spike_pattern: &TTFSSpikePattern, 
                               context: &SemanticContext) -> Result<SemanticColumnVote, SemanticError> {
        let base_result = self.process_spikes(spike_pattern)?;
        
        // Apply context influence
        let context_influence = self.context_processor.calculate_context_influence(
            &base_result.neural_output, 
            context
        );
        
        // Adjust confidence based on context
        let adjusted_confidence = (base_result.confidence + context_influence * 0.2).clamp(0.0, 1.0);
        let adjusted_activation = if adjusted_confidence > self.activation_threshold { 
            adjusted_confidence 
        } else { 
            0.0 
        };
        
        // Create semantic-specific result
        let semantic_features = self.extract_semantic_features_from_output(&base_result.neural_output);
        
        Ok(SemanticColumnVote {
            column_vote: ColumnVote {
                column_id: ColumnId::Semantic,
                confidence: adjusted_confidence,
                activation: adjusted_activation,
                neural_output: base_result.neural_output,
                processing_time: base_result.processing_time,
            },
            semantic_features: Some(semantic_features),
            memory_activation: self.activate_semantic_memory(spike_pattern)?,
            context_influence,
            confidence_breakdown: self.calculate_confidence_breakdown(&base_result, context_influence),
        })
    }
    
    /// Process multiple spike patterns in parallel
    pub fn process_multiple_spikes_parallel(&self, 
                                          spike_patterns: &[TTFSSpikePattern]) -> Result<Vec<ColumnVote>, SemanticError> {
        spike_patterns.par_iter()
            .map(|pattern| self.process_spikes(pattern))
            .collect()
    }
    
    /// Extract semantic features from spike pattern
    pub fn extract_semantic_features(&self, spike_pattern: &TTFSSpikePattern) -> Result<Vec<f32>, SemanticError> {
        let neural_input = self.prepare_neural_input(spike_pattern)?;
        let neural_output = self.neural_network.as_ref()
            .ok_or(SemanticError::NetworkNotInitialized)?
            .forward(&neural_input)?;
        
        Ok(self.extract_semantic_features_from_output(&neural_output))
    }
    
    /// Calculate semantic similarity between feature vectors
    pub fn calculate_semantic_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        if features1.len() != features2.len() || features1.is_empty() {
            return 0.0;
        }
        
        // Use SIMD-accelerated cosine similarity
        self.simd_processor.calculate_correlation_simd(features1, features2)
    }
    
    /// Prepare neural input from TTFS spike pattern
    pub fn prepare_neural_input(&self, spike_pattern: &TTFSSpikePattern) -> Result<Vec<f32>, SemanticError> {
        let mut input = vec![0.0; 256]; // Standard semantic input size
        
        // Encode spike timing as activation strength
        for (i, spike) in spike_pattern.spike_sequence().iter().enumerate() {
            if i < input.len() {
                // Earlier spikes (shorter timing) produce higher activation
                let timing_ms = spike.timing.as_nanos() as f32 / 1_000_000.0;
                let activation = (10.0 - timing_ms.min(10.0)) / 10.0; // Normalize to 0-1
                input[i] = activation * spike.amplitude;
            }
        }
        
        // Add TTFS timing information
        let first_spike_ms = spike_pattern.first_spike_time().as_nanos() as f32 / 1_000_000.0;
        let ttfs_activation = (5.0 - first_spike_ms.min(5.0)) / 5.0; // Strong early spike bonus
        
        // Spread TTFS information across multiple input neurons
        for i in 128..144 { // Dedicated TTFS region
            input[i] = ttfs_activation;
        }
        
        // Add pattern duration information
        let duration_ms = spike_pattern.total_duration().as_nanos() as f32 / 1_000_000.0;
        let duration_feature = (20.0 - duration_ms.min(20.0)) / 20.0; // Shorter patterns preferred
        
        for i in 144..160 { // Duration region
            input[i] = duration_feature;
        }
        
        Ok(input)
    }
    
    /// Preload semantic memory with known concepts
    pub fn preload_semantic_memory(&self, concepts: &[(&str, Vec<f32>)]) -> Result<(), SemanticError> {
        for (concept_name, features) in concepts {
            self.semantic_memory.store_concept(concept_name.to_string(), features.clone());
        }
        Ok(())
    }
    
    /// Set activation threshold
    pub fn set_activation_threshold(&mut self, threshold: f32) {
        self.activation_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Check if column is ready for processing
    pub fn is_ready(&self) -> bool {
        self.neural_network.is_some() && 
        self.selected_architecture.is_some() &&
        self.neural_network.as_ref().unwrap().is_ready()
    }
    
    /// Get selected architecture information
    pub fn get_selected_architecture(&self) -> &SelectedArchitecture {
        self.selected_architecture.as_ref().unwrap()
    }
    
    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        self.performance_monitor.cache_stats.clone()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }
    
    // Private helper methods
    
    fn load_neural_network(architecture: &SelectedArchitecture) -> Result<Box<dyn SemanticNeuralNetwork>, SemanticError> {
        // Load the specific neural network based on selected architecture
        match architecture.architecture.id {
            1 => Ok(Box::new(MLPSemanticNetwork::new(architecture.clone())?)),
            4 => Ok(Box::new(LSTMSemanticNetwork::new(architecture.clone())?)),
            13 => Ok(Box::new(TransformerSemanticNetwork::new(architecture.clone())?)),
            _ => Err(SemanticError::ArchitectureSelectionFailed(
                format!("Unsupported architecture ID: {}", architecture.architecture.id)
            )),
        }
    }
    
    fn check_cache(&self, concept_id: &ConceptId) -> Option<CachedSemanticResult> {
        self.semantic_cache.get(concept_id).map(|entry| {
            let mut cached = entry.value().clone();
            cached.hit_count += 1;
            cached
        })
    }
    
    fn cache_result(&self, concept_id: ConceptId, column_vote: &ColumnVote, semantic_features: &[f32]) {
        let cached_result = CachedSemanticResult {
            column_vote: column_vote.clone(),
            semantic_features: semantic_features.to_vec(),
            cached_at: Instant::now(),
            hit_count: 0,
        };
        
        self.semantic_cache.insert(concept_id, cached_result);
        
        // Limit cache size
        if self.semantic_cache.len() > 1000 {
            // Remove oldest entries (simplified LRU)
            let oldest_key = self.semantic_cache.iter()
                .min_by_key(|entry| entry.value().cached_at)
                .map(|entry| entry.key().clone());
            
            if let Some(key) = oldest_key {
                self.semantic_cache.remove(&key);
            }
        }
    }
    
    fn extract_semantic_features_from_output(&self, neural_output: &[f32]) -> Vec<f32> {
        // Extract semantic features from neural network output
        // For most architectures, we use the final layer activations
        if neural_output.len() >= 128 {
            neural_output[neural_output.len() - 128..].to_vec()
        } else {
            // Pad with zeros if output is smaller
            let mut features = neural_output.to_vec();
            features.resize(128, 0.0);
            features
        }
    }
    
    fn calculate_semantic_confidence(&self, neural_output: &[f32], semantic_features: &[f32]) -> f32 {
        // Calculate confidence based on multiple factors
        let max_activation = neural_output.iter().cloned().fold(0.0f32, f32::max);
        let feature_diversity = self.calculate_feature_diversity(semantic_features);
        let network_confidence = self.neural_network.as_ref().unwrap()
            .performance_metrics().accuracy;
        
        // Weighted combination
        (max_activation * 0.4 + feature_diversity * 0.3 + network_confidence * 0.3).clamp(0.0, 1.0)
    }
    
    fn calculate_feature_diversity(&self, features: &[f32]) -> f32 {
        if features.is_empty() {
            return 0.0;
        }
        
        let mean = features.iter().sum::<f32>() / features.len() as f32;
        let variance = features.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / features.len() as f32;
        
        variance.sqrt().min(1.0)
    }
    
    fn activate_semantic_memory(&self, spike_pattern: &TTFSSpikePattern) -> Result<Option<MemoryActivation>, SemanticError> {
        let concept_name = spike_pattern.concept_id().name();
        let similar_concepts = self.semantic_memory.find_similar_concepts(&concept_name, 0.5, 3);
        
        if similar_concepts.is_empty() {
            return Ok(None);
        }
        
        let retrieval_confidence = similar_concepts.iter()
            .map(|c| c.similarity_score)
            .fold(0.0f32, f32::max);
        
        let memory_influence = retrieval_confidence * 0.3; // Moderate influence
        
        Ok(Some(MemoryActivation {
            similar_concepts,
            retrieval_confidence,
            memory_influence,
        }))
    }
    
    fn calculate_confidence_breakdown(&self, base_result: &ColumnVote, context_influence: f32) -> ConfidenceBreakdown {
        ConfidenceBreakdown {
            neural_confidence: base_result.confidence,
            feature_confidence: 0.85, // Mock for now
            memory_confidence: 0.80,  // Mock for now
            context_confidence: context_influence,
        }
    }
    
    fn update_performance_metrics(&self, processing_time: Duration, confidence: f32) {
        // Update performance tracking (simplified)
        // In real implementation, this would use atomic operations or locks
    }
}

// Concrete neural network implementations

/// MLP-based semantic network
pub struct MLPSemanticNetwork {
    architecture: SelectedArchitecture,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    performance_metrics: PerformanceMetrics,
}

impl MLPSemanticNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, SemanticError> {
        Ok(Self {
            architecture,
            weights: vec![vec![0.5; 256]; 3], // 3 layers, simplified
            biases: vec![vec![0.1; 128]; 3],
            performance_metrics: PerformanceMetrics {
                inference_time: Duration::from_micros(300),
                memory_usage: 15_000_000,
                accuracy: 0.87,
                throughput: 1000.0,
            },
        })
    }
}

impl SemanticNeuralNetwork for MLPSemanticNetwork {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, SemanticError> {
        // Simplified MLP forward pass
        let mut current = input.to_vec();
        
        for layer in 0..self.weights.len() {
            let mut next_layer = vec![0.0; self.biases[layer].len()];
            
            for (i, bias) in self.biases[layer].iter().enumerate() {
                let weighted_sum: f32 = current.iter()
                    .take(self.weights[layer].len())
                    .enumerate()
                    .map(|(j, &x)| x * self.weights[layer].get(j).unwrap_or(&0.5))
                    .sum();
                
                next_layer[i] = (weighted_sum + bias).max(0.0); // ReLU activation
            }
            
            current = next_layer;
        }
        
        Ok(current)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    fn is_ready(&self) -> bool {
        !self.weights.is_empty() && !self.biases.is_empty()
    }
}

/// LSTM-based semantic network
pub struct LSTMSemanticNetwork {
    architecture: SelectedArchitecture,
    lstm_weights: Vec<f32>,
    performance_metrics: PerformanceMetrics,
}

impl LSTMSemanticNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, SemanticError> {
        Ok(Self {
            architecture,
            lstm_weights: vec![0.3; 1024], // Simplified LSTM weights
            performance_metrics: PerformanceMetrics {
                inference_time: Duration::from_micros(800),
                memory_usage: 25_000_000,
                accuracy: 0.91,
                throughput: 800.0,
            },
        })
    }
}

impl SemanticNeuralNetwork for LSTMSemanticNetwork {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, SemanticError> {
        // Simplified LSTM forward pass
        let output_size = 128;
        let mut output = vec![0.0; output_size];
        
        for (i, &x) in input.iter().take(output_size).enumerate() {
            let weight = self.lstm_weights.get(i % self.lstm_weights.len()).unwrap_or(&0.3);
            output[i] = (x * weight).tanh(); // Tanh activation for LSTM
        }
        
        Ok(output)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    fn is_ready(&self) -> bool {
        !self.lstm_weights.is_empty()
    }
}

/// Transformer-based semantic network
pub struct TransformerSemanticNetwork {
    architecture: SelectedArchitecture,
    attention_weights: Vec<f32>,
    performance_metrics: PerformanceMetrics,
}

impl TransformerSemanticNetwork {
    pub fn new(architecture: SelectedArchitecture) -> Result<Self, SemanticError> {
        Ok(Self {
            architecture,
            attention_weights: vec![0.4; 512], // Simplified attention weights
            performance_metrics: PerformanceMetrics {
                inference_time: Duration::from_millis(1),
                memory_usage: 40_000_000,
                accuracy: 0.94,
                throughput: 500.0,
            },
        })
    }
}

impl SemanticNeuralNetwork for TransformerSemanticNetwork {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, SemanticError> {
        // Simplified transformer forward pass
        let output_size = 128;
        let mut output = vec![0.0; output_size];
        
        // Simplified attention mechanism
        for i in 0..output_size {
            let attention_sum: f32 = input.iter()
                .enumerate()
                .map(|(j, &x)| {
                    let weight_idx = (i * input.len() + j) % self.attention_weights.len();
                    x * self.attention_weights[weight_idx]
                })
                .sum();
            
            output[i] = attention_sum / input.len() as f32;
        }
        
        Ok(output)
    }
    
    fn architecture_info(&self) -> &SelectedArchitecture {
        &self.architecture
    }
    
    fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    fn is_ready(&self) -> bool {
        !self.attention_weights.is_empty()
    }
}

// Supporting implementations

impl SemanticContext {
    pub fn new(context_words: Vec<String>) -> Self {
        Self {
            context_words,
            context_strength: 1.0,
            context_type: ContextType::General,
        }
    }
}

impl SemanticMemory {
    pub fn new(max_concepts: usize) -> Self {
        Self {
            concept_embeddings: DashMap::new(),
            concept_relationships: DashMap::new(),
            max_concepts,
            usage_stats: MemoryUsageStats::default(),
        }
    }
    
    pub fn store_concept(&self, name: String, features: Vec<f32>) {
        self.concept_embeddings.insert(name, features);
    }
    
    pub fn find_similar_concepts(&self, query: &str, threshold: f32, limit: usize) -> Vec<SimilarConcept> {
        // Simplified similarity search
        vec![] // Mock implementation
    }
}

impl ContextProcessor {
    pub fn new() -> Self {
        Self {
            context_embeddings: HashMap::new(),
            influence_weights: HashMap::new(),
        }
    }
    
    pub fn calculate_context_influence(&self, _neural_output: &[f32], _context: &SemanticContext) -> f32 {
        0.1 // Mock context influence
    }
}
```

## Verification Steps
1. Implement automatic neural architecture selection for semantic processing with <10s selection time
2. Add TTFS spike pattern to neural input conversion with proper timing encoding
3. Implement semantic feature extraction and similarity computation using SIMD acceleration
4. Add semantic memory integration with concept storage and retrieval capabilities
5. Implement context-aware processing with multiple context types
6. Add comprehensive caching system with LRU eviction and performance tracking
7. Implement parallel processing support for multiple spike patterns
8. Add performance monitoring and metrics collection for optimization

## Success Criteria
- [ ] Semantic column initializes with optimal architecture in <200ms
- [ ] Spike processing completes in <1ms per pattern (sub-millisecond target)
- [ ] Architecture selection chooses optimal network based on performance benchmarks
- [ ] Memory usage stays within 50MB limit per column
- [ ] Semantic similarity computation achieves >0.8 accuracy on test concepts
- [ ] Cache hit rate reaches >90% after warmup period
- [ ] Parallel processing provides >1.5x speedup over sequential
- [ ] Context-aware processing shows measurable influence on semantic interpretation
- [ ] Selected architecture meets all performance, memory, and accuracy requirements
- [ ] Integration with ruv-FANN library successful with proper error handling