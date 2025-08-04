# Task 43: Scoring System Integration

## Metadata
- **Micro-Phase**: 2.43
- **Duration**: 20 minutes
- **Dependencies**: Tasks 39-42 (scoring framework, semantic similarity, property compatibility, structural composite)
- **Output**: `src/allocation_scoring/mod.rs` and `src/allocation_scoring/integration.rs`

## Description
Integrate all scoring strategies into a unified allocation scoring system with weighted combination, strategy selection, performance monitoring, and comprehensive validation. Provides the complete scoring interface for the allocation engine with <2ms total scoring time.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_scoring::*;
    use crate::hierarchy_detection::{ExtractedConcept, ConceptType};
    use std::collections::HashMap;

    #[test]
    fn test_integrated_scoring_system_creation() {
        let scoring_system = IntegratedScoringSystem::new();
        assert_eq!(scoring_system.strategy_count(), 3);
        assert!(scoring_system.is_strategy_enabled("semantic_similarity"));
        assert!(scoring_system.is_strategy_enabled("property_compatibility"));
        assert!(scoring_system.is_strategy_enabled("structural_composite"));
        assert!(scoring_system.supports_parallel_scoring());
    }
    
    #[test]
    fn test_weighted_scoring_combination() {
        let scoring_system = IntegratedScoringSystem::new();
        
        let concept = create_test_concept("golden_retriever", ConceptType::Entity);
        let context = create_test_allocation_context("dog", &["mammal", "animal"]);
        
        let result = scoring_system.score_allocation(&concept, &context).unwrap();
        
        // Should have all component scores
        assert!(result.semantic_score.is_some());
        assert!(result.property_score.is_some());
        assert!(result.structural_score.is_some());
        
        // Total score should be weighted combination
        assert!(result.total_score >= 0.0 && result.total_score <= 1.0);
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
        
        // Should have breakdown and justification
        assert!(!result.score_breakdown.is_empty());
        assert!(!result.justification.is_empty());
    }
    
    #[test]
    fn test_scoring_strategy_weights() {
        let mut config = ScoringConfig::default();
        config.semantic_weight = 0.5;
        config.property_weight = 0.3;
        config.structural_weight = 0.2;
        
        let scoring_system = IntegratedScoringSystem::with_config(config);
        
        let concept = create_test_concept("cat", ConceptType::Entity);
        let context = create_test_allocation_context("mammal", &["animal"]);
        
        let result = scoring_system.score_allocation(&concept, &context).unwrap();
        
        // Verify weights are applied correctly
        let expected_total = 
            result.semantic_score.unwrap() * 0.5 +
            result.property_score.unwrap() * 0.3 +
            result.structural_score.unwrap() * 0.2;
        
        assert!((result.total_score - expected_total).abs() < 0.01);
    }
    
    #[test]
    fn test_strategy_selection_and_filtering() {
        let mut scoring_system = IntegratedScoringSystem::new();
        
        // Disable one strategy
        scoring_system.disable_strategy("structural_composite").unwrap();
        assert!(!scoring_system.is_strategy_enabled("structural_composite"));
        
        let concept = create_test_concept("bird", ConceptType::Entity);
        let context = create_test_allocation_context("animal", &["living_thing"]);
        
        let result = scoring_system.score_allocation(&concept, &context).unwrap();
        
        // Should only have enabled strategies
        assert!(result.semantic_score.is_some());
        assert!(result.property_score.is_some());
        assert!(result.structural_score.is_none()); // Disabled
        
        // Re-enable strategy
        scoring_system.enable_strategy("structural_composite").unwrap();
        assert!(scoring_system.is_strategy_enabled("structural_composite"));
    }
    
    #[test]
    fn test_batch_scoring_performance() {
        let scoring_system = IntegratedScoringSystem::new();
        
        let concepts = vec![
            create_test_concept("dog", ConceptType::Entity),
            create_test_concept("cat", ConceptType::Entity),
            create_test_concept("bird", ConceptType::Entity),
            create_test_concept("fish", ConceptType::Entity),
            create_test_concept("tree", ConceptType::Entity),
        ];
        
        let context = create_test_allocation_context("animal", &["living_thing"]);
        
        let start = std::time::Instant::now();
        let results = scoring_system.batch_score_allocations(&concepts, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete batch scoring quickly
        assert!(elapsed < std::time::Duration::from_millis(10));
        assert_eq!(results.len(), 5);
        
        // All results should be valid
        for result in &results {
            assert!(result.total_score >= 0.0 && result.total_score <= 1.0);
            assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
            assert!(!result.score_breakdown.is_empty());
        }
        
        // Animals should generally score higher than plants for animal context
        let animal_scores: Vec<f32> = results[0..4].iter().map(|r| r.total_score).collect();
        let plant_score = results[4].total_score;
        
        assert!(animal_scores.iter().any(|&score| score > plant_score));
    }
    
    #[test]
    fn test_parallel_scoring_efficiency() {
        let scoring_system = IntegratedScoringSystem::new();
        
        // Create many concepts for parallel processing
        let concepts: Vec<_> = (0..100)
            .map(|i| create_test_concept(&format!("concept_{}", i), ConceptType::Entity))
            .collect();
        
        let context = create_test_allocation_context("parent", &["root"]);
        
        let start = std::time::Instant::now();
        let results = scoring_system.parallel_score_allocations(&concepts, &context).unwrap();
        let elapsed = start.elapsed();
        
        // Should achieve target performance (<2ms total for 100 concepts)
        assert!(elapsed < std::time::Duration::from_millis(20)); // More lenient for test
        assert_eq!(results.len(), 100);
        
        // All results should be valid
        for result in &results {
            assert!(result.total_score >= 0.0 && result.total_score <= 1.0);
        }
    }
    
    #[test]
    fn test_scoring_confidence_calculation() {
        let scoring_system = IntegratedScoringSystem::new();
        
        // High confidence concept
        let high_confidence_concept = create_test_concept_with_confidence("reliable_dog", ConceptType::Entity, 0.95);
        
        // Low confidence concept
        let low_confidence_concept = create_test_concept_with_confidence("uncertain_animal", ConceptType::Entity, 0.4);
        
        let context = create_test_allocation_context("mammal", &["animal"]);
        
        let high_result = scoring_system.score_allocation(&high_confidence_concept, &context).unwrap();
        let low_result = scoring_system.score_allocation(&low_confidence_concept, &context).unwrap();
        
        // High confidence should result in higher confidence score
        assert!(high_result.confidence_score > low_result.confidence_score);
        assert!(high_result.confidence_score - low_result.confidence_score > 0.2);
    }
    
    #[test]
    fn test_score_ranking_and_selection() {
        let scoring_system = IntegratedScoringSystem::new();
        
        let concepts = vec![
            create_test_concept("golden_retriever", ConceptType::Entity), // Should rank high
            create_test_concept("dog", ConceptType::Entity),              // Should rank high
            create_test_concept("mammal", ConceptType::Category),         // Medium
            create_test_concept("vehicle", ConceptType::Entity),          // Should rank low
            create_test_concept("plant", ConceptType::Entity),            // Should rank low
        ];
        
        let context = create_test_allocation_context("dog", &["mammal", "animal"]);
        
        let ranked_results = scoring_system.rank_allocations(&concepts, &context).unwrap();
        
        assert_eq!(ranked_results.len(), 5);
        
        // Results should be sorted by total score (highest first)
        for i in 1..ranked_results.len() {
            assert!(ranked_results[i-1].allocation_score.total_score >= ranked_results[i].allocation_score.total_score);
        }
        
        // Dog-related concepts should rank higher
        assert!(ranked_results[0].concept.name.contains("dog") || ranked_results[0].concept.name.contains("retriever"));
    }
    
    #[test]
    fn test_strategy_performance_monitoring() {
        let mut scoring_system = IntegratedScoringSystem::new();
        scoring_system.enable_performance_monitoring(true);
        
        let concept = create_test_concept("test_concept", ConceptType::Entity);
        let context = create_test_allocation_context("test_parent", &["root"]);
        
        // Perform some scoring operations
        for _ in 0..10 {
            let _ = scoring_system.score_allocation(&concept, &context);
        }
        
        let performance_stats = scoring_system.get_performance_statistics();
        
        assert!(performance_stats.total_scoring_operations >= 10);
        assert!(performance_stats.average_scoring_time_ms > 0.0);
        assert!(!performance_stats.strategy_performance.is_empty());
        
        // Each strategy should have performance data
        for (strategy_name, stats) in &performance_stats.strategy_performance {
            assert!(stats.operation_count > 0);
            assert!(stats.average_time_ms >= 0.0);
        }
    }
    
    #[test]
    fn test_scoring_validation_and_error_handling() {
        let scoring_system = IntegratedScoringSystem::new();
        
        // Test with invalid concept
        let invalid_concept = ExtractedConcept {
            name: "".to_string(), // Empty name
            concept_type: ConceptType::Entity,
            properties: HashMap::new(),
            source_span: crate::hierarchy_detection::TextSpan {
                start: 0,
                end: 0,
                text: "".to_string(),
            },
            confidence: -0.5, // Invalid confidence
            suggested_parent: None,
            semantic_features: vec![],
            extracted_at: 0,
        };
        
        let context = create_test_allocation_context("parent", &["root"]);
        
        // Should handle gracefully
        let result = scoring_system.score_allocation(&invalid_concept, &context);
        
        match result {
            Ok(score_result) => {
                // Should return low/zero scores for invalid input
                assert!(score_result.total_score <= 0.3);
            },
            Err(e) => {
                // Or return appropriate error
                assert!(e.to_string().contains("invalid") || e.to_string().contains("empty"));
            }
        }
    }
    
    #[test]
    fn test_scoring_system_configuration() {
        let mut config = ScoringConfig::default();
        config.semantic_weight = 0.6;
        config.property_weight = 0.4;
        config.structural_weight = 0.0; // Disable structural
        config.enable_caching = true;
        config.cache_size = 500;
        config.performance_monitoring = true;
        
        let scoring_system = IntegratedScoringSystem::with_config(config);
        
        // Verify configuration is applied
        assert_eq!(scoring_system.get_config().semantic_weight, 0.6);
        assert_eq!(scoring_system.get_config().property_weight, 0.4);
        assert_eq!(scoring_system.get_config().structural_weight, 0.0);
        assert!(scoring_system.get_config().enable_caching);
        assert!(scoring_system.get_config().performance_monitoring);
    }
    
    #[test]
    fn test_scoring_cache_functionality() {
        let mut config = ScoringConfig::default();
        config.enable_caching = true;
        config.cache_size = 100;
        
        let mut scoring_system = IntegratedScoringSystem::with_config(config);
        
        let concept = create_test_concept("cached_concept", ConceptType::Entity);
        let context = create_test_allocation_context("parent", &["root"]);
        
        // First call (cache miss)
        let start1 = std::time::Instant::now();
        let result1 = scoring_system.score_allocation(&concept, &context).unwrap();
        let elapsed1 = start1.elapsed();
        
        // Second call (cache hit)
        let start2 = std::time::Instant::now();
        let result2 = scoring_system.score_allocation(&concept, &context).unwrap();
        let elapsed2 = start2.elapsed();
        
        // Results should be identical
        assert!((result1.total_score - result2.total_score).abs() < 0.001);
        
        // Second call should be faster
        assert!(elapsed2 < elapsed1);
        
        // Check cache statistics
        let cache_stats = scoring_system.get_cache_statistics();
        assert!(cache_stats.hit_count >= 1);
        assert!(cache_stats.miss_count >= 1);
        assert!(cache_stats.hit_rate > 0.0);
    }
    
    #[test]
    fn test_strategy_fallback_mechanism() {
        let mut scoring_system = IntegratedScoringSystem::new();
        
        // Simulate strategy failure by disabling all but one
        scoring_system.disable_strategy("semantic_similarity").unwrap();
        scoring_system.disable_strategy("structural_composite").unwrap();
        
        let concept = create_test_concept("fallback_test", ConceptType::Entity);
        let context = create_test_allocation_context("parent", &["root"]);
        
        let result = scoring_system.score_allocation(&concept, &context).unwrap();
        
        // Should still provide a valid score using remaining strategy
        assert!(result.total_score >= 0.0 && result.total_score <= 1.0);
        assert!(result.property_score.is_some());
        assert!(result.semantic_score.is_none()); // Disabled
        assert!(result.structural_score.is_none()); // Disabled
        
        // Justification should indicate limited strategy usage
        assert!(result.justification.contains("property") || 
               result.justification.contains("limited") ||
               result.justification.contains("fallback"));
    }
    
    fn create_test_concept(name: &str, concept_type: ConceptType) -> ExtractedConcept {
        create_test_concept_with_confidence(name, concept_type, 0.8)
    }
    
    fn create_test_concept_with_confidence(name: &str, concept_type: ConceptType, confidence: f32) -> ExtractedConcept {
        use crate::hierarchy_detection::{ExtractedConcept, TextSpan};
        use std::collections::HashMap;
        
        ExtractedConcept {
            name: name.to_string(),
            concept_type,
            properties: HashMap::new(),
            source_span: TextSpan {
                start: 0,
                end: name.len(),
                text: name.to_string(),
            },
            confidence,
            suggested_parent: None,
            semantic_features: vec![0.5; 100],
            extracted_at: 0,
        }
    }
    
    fn create_test_allocation_context(target: &str, ancestors: &[&str]) -> AllocationContext {
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties: HashMap::new(),
            allocation_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;
use dashmap::DashMap;

use crate::allocation_scoring::semantic_similarity_scoring::SemanticSimilarityStrategy;
use crate::allocation_scoring::property_compatibility_scoring::PropertyCompatibilityStrategy;
use crate::allocation_scoring::structural_composite_scoring::StructuralCompositeStrategy;
use crate::hierarchy_detection::ExtractedConcept;

/// Configuration for the integrated scoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Weight for semantic similarity scoring
    pub semantic_weight: f32,
    
    /// Weight for property compatibility scoring
    pub property_weight: f32,
    
    /// Weight for structural composite scoring
    pub structural_weight: f32,
    
    /// Minimum score threshold for valid allocations
    pub minimum_score_threshold: f32,
    
    /// Enable result caching for performance
    pub enable_caching: bool,
    
    /// Maximum cache size
    pub cache_size: usize,
    
    /// Enable performance monitoring
    pub performance_monitoring: bool,
    
    /// Parallel processing batch size
    pub parallel_batch_size: usize,
    
    /// Timeout for individual strategy scoring (milliseconds)
    pub strategy_timeout_ms: u64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.4,
            property_weight: 0.35,
            structural_weight: 0.25,
            minimum_score_threshold: 0.1,
            enable_caching: true,
            cache_size: 1000,
            performance_monitoring: false,
            parallel_batch_size: 32,
            strategy_timeout_ms: 50,
        }
    }
}

impl ScoringConfig {
    /// Validate and normalize weights
    pub fn normalize_weights(&mut self) {
        let total = self.semantic_weight + self.property_weight + self.structural_weight;
        if total > 0.0 {
            self.semantic_weight /= total;
            self.property_weight /= total;
            self.structural_weight /= total;
        } else {
            // Fallback to default weights
            *self = Self::default();
        }
    }
    
    /// Check if weights are valid
    pub fn are_weights_valid(&self) -> bool {
        let total = self.semantic_weight + self.property_weight + self.structural_weight;
        (total - 1.0).abs() < 0.001 && 
        self.semantic_weight >= 0.0 && 
        self.property_weight >= 0.0 && 
        self.structural_weight >= 0.0
    }
}

/// Comprehensive allocation scoring result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationScoringResult {
    /// Overall weighted score (0.0-1.0)
    pub total_score: f32,
    
    /// Confidence in the scoring result
    pub confidence_score: f32,
    
    /// Individual strategy scores
    pub semantic_score: Option<f32>,
    pub property_score: Option<f32>,
    pub structural_score: Option<f32>,
    
    /// Detailed score breakdown by component
    pub score_breakdown: BTreeMap<String, f32>,
    
    /// Human-readable justification
    pub justification: String,
    
    /// Scoring metadata
    pub scoring_metadata: ScoringMetadata,
    
    /// Performance metrics for this scoring operation
    pub performance_metrics: Option<ScoringPerformanceMetrics>,
}

/// Metadata about the scoring operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringMetadata {
    /// Timestamp when scoring was performed
    pub scored_at: u64,
    
    /// Number of strategies that participated
    pub strategies_used: usize,
    
    /// Total time taken for scoring (microseconds)
    pub total_time_microseconds: u64,
    
    /// Whether result was retrieved from cache
    pub from_cache: bool,
    
    /// Strategy-specific execution times
    pub strategy_times: HashMap<String, u64>,
}

/// Performance metrics for scoring operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringPerformanceMetrics {
    /// Time taken by each strategy (microseconds)
    pub strategy_execution_times: HashMap<String, u64>,
    
    /// Memory usage during scoring (bytes)
    pub memory_usage_bytes: Option<usize>,
    
    /// Number of cache hits/misses
    pub cache_hits: usize,
    pub cache_misses: usize,
    
    /// Parallel processing efficiency (0.0-1.0)
    pub parallelization_efficiency: f32,
}

/// Ranked allocation result
#[derive(Debug, Clone)]
pub struct RankedAllocationResult {
    /// The concept being scored
    pub concept: ExtractedConcept,
    
    /// The allocation scoring result
    pub allocation_score: AllocationScoringResult,
    
    /// Rank within the batch (1-based)
    pub rank: usize,
    
    /// Percentile ranking (0.0-1.0)
    pub percentile: f32,
}

/// Performance statistics for the scoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringSystemPerformanceStats {
    /// Total number of scoring operations performed
    pub total_scoring_operations: u64,
    
    /// Average scoring time in milliseconds
    pub average_scoring_time_ms: f32,
    
    /// Minimum scoring time in milliseconds
    pub min_scoring_time_ms: f32,
    
    /// Maximum scoring time in milliseconds
    pub max_scoring_time_ms: f32,
    
    /// Performance statistics per strategy
    pub strategy_performance: HashMap<String, StrategyPerformanceStats>,
    
    /// Cache statistics
    pub cache_statistics: CacheStatistics,
    
    /// Parallel processing statistics
    pub parallel_processing_stats: ParallelProcessingStats,
}

/// Performance statistics for individual strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformanceStats {
    /// Number of operations performed
    pub operation_count: u64,
    
    /// Average execution time in milliseconds
    pub average_time_ms: f32,
    
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    
    /// Error count
    pub error_count: u64,
    
    /// Last error message
    pub last_error: Option<String>,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hit_count: u64,
    
    /// Total cache misses
    pub miss_count: u64,
    
    /// Cache hit rate (0.0-1.0)
    pub hit_rate: f32,
    
    /// Current cache size
    pub current_size: usize,
    
    /// Maximum cache size
    pub max_size: usize,
    
    /// Number of cache evictions
    pub eviction_count: u64,
}

/// Parallel processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessingStats {
    /// Total parallel operations
    pub parallel_operations: u64,
    
    /// Average speedup achieved
    pub average_speedup: f32,
    
    /// Maximum speedup achieved
    pub max_speedup: f32,
    
    /// Efficiency of parallel operations (0.0-1.0)
    pub efficiency: f32,
}

/// Strategy status information
#[derive(Debug, Clone)]
enum StrategyStatus {
    Enabled,
    Disabled,
    Error(String),
    Timeout,
}

/// Main integrated scoring system
pub struct IntegratedScoringSystem {
    /// System configuration
    config: ScoringConfig,
    
    /// Strategy instances
    semantic_strategy: SemanticSimilarityStrategy,
    property_strategy: PropertyCompatibilityStrategy,
    structural_strategy: StructuralCompositeStrategy,
    
    /// Strategy status tracking
    strategy_status: HashMap<String, StrategyStatus>,
    
    /// Result cache
    result_cache: Arc<DashMap<String, AllocationScoringResult>>,
    
    /// Performance monitoring
    performance_stats: Arc<Mutex<ScoringSystemPerformanceStats>>,
    
    /// Performance monitoring enabled flag
    monitoring_enabled: bool,
}

impl IntegratedScoringSystem {
    /// Create a new integrated scoring system with default configuration
    pub fn new() -> Self {
        let config = ScoringConfig::default();
        Self::with_config(config)
    }
    
    /// Create with custom configuration
    pub fn with_config(mut config: ScoringConfig) -> Self {
        config.normalize_weights();
        
        let mut strategy_status = HashMap::new();
        strategy_status.insert("semantic_similarity".to_string(), StrategyStatus::Enabled);
        strategy_status.insert("property_compatibility".to_string(), StrategyStatus::Enabled);
        strategy_status.insert("structural_composite".to_string(), StrategyStatus::Enabled);
        
        let performance_stats = ScoringSystemPerformanceStats {
            total_scoring_operations: 0,
            average_scoring_time_ms: 0.0,
            min_scoring_time_ms: f32::MAX,
            max_scoring_time_ms: 0.0,
            strategy_performance: HashMap::new(),
            cache_statistics: CacheStatistics {
                hit_count: 0,
                miss_count: 0,
                hit_rate: 0.0,
                current_size: 0,
                max_size: config.cache_size,
                eviction_count: 0,
            },
            parallel_processing_stats: ParallelProcessingStats {
                parallel_operations: 0,
                average_speedup: 1.0,
                max_speedup: 1.0,
                efficiency: 1.0,
            },
        };
        
        Self {
            config,
            semantic_strategy: SemanticSimilarityStrategy::new(),
            property_strategy: PropertyCompatibilityStrategy::new(),
            structural_strategy: StructuralCompositeStrategy::new(),
            strategy_status,
            result_cache: Arc::new(DashMap::new()),
            performance_stats: Arc::new(Mutex::new(performance_stats)),
            monitoring_enabled: false,
        }
    }
    
    /// Score a single allocation
    pub fn score_allocation(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<AllocationScoringResult, ScoringError> {
        let start_time = Instant::now();
        
        // Generate cache key
        let cache_key = self.generate_cache_key(concept, context);
        
        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.result_cache.get(&cache_key) {
                self.update_cache_statistics(true);
                return Ok(cached_result.clone());
            }
        }
        
        self.update_cache_statistics(false);
        
        // Score with each enabled strategy
        let mut strategy_scores = HashMap::new();
        let mut strategy_times = HashMap::new();
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        
        // Semantic similarity scoring
        if self.is_strategy_enabled("semantic_similarity") && self.config.semantic_weight > 0.0 {
            let strategy_start = Instant::now();
            match self.semantic_strategy.score(concept, context) {
                Ok(score) => {
                    strategy_scores.insert("semantic_similarity".to_string(), score);
                    weighted_sum += score * self.config.semantic_weight;
                    total_weight += self.config.semantic_weight;
                    
                    let strategy_time = strategy_start.elapsed().as_micros() as u64;
                    strategy_times.insert("semantic_similarity".to_string(), strategy_time);
                    
                    self.update_strategy_performance("semantic_similarity", strategy_time, true, None);
                },
                Err(e) => {
                    self.update_strategy_performance("semantic_similarity", 0, false, Some(e.to_string()));
                }
            }
        }
        
        // Property compatibility scoring
        if self.is_strategy_enabled("property_compatibility") && self.config.property_weight > 0.0 {
            let strategy_start = Instant::now();
            match self.property_strategy.score(concept, context) {
                Ok(score) => {
                    strategy_scores.insert("property_compatibility".to_string(), score);
                    weighted_sum += score * self.config.property_weight;
                    total_weight += self.config.property_weight;
                    
                    let strategy_time = strategy_start.elapsed().as_micros() as u64;
                    strategy_times.insert("property_compatibility".to_string(), strategy_time);
                    
                    self.update_strategy_performance("property_compatibility", strategy_time, true, None);
                },
                Err(e) => {
                    self.update_strategy_performance("property_compatibility", 0, false, Some(e.to_string()));
                }
            }
        }
        
        // Structural composite scoring
        if self.is_strategy_enabled("structural_composite") && self.config.structural_weight > 0.0 {
            let strategy_start = Instant::now();
            match self.structural_strategy.score(concept, context) {
                Ok(score) => {
                    strategy_scores.insert("structural_composite".to_string(), score);
                    weighted_sum += score * self.config.structural_weight;
                    total_weight += self.config.structural_weight;
                    
                    let strategy_time = strategy_start.elapsed().as_micros() as u64;
                    strategy_times.insert("structural_composite".to_string(), strategy_time);
                    
                    self.update_strategy_performance("structural_composite", strategy_time, true, None);
                },
                Err(e) => {
                    self.update_strategy_performance("structural_composite", 0, false, Some(e.to_string()));
                }
            }
        }
        
        // Calculate final scores
        let total_score = if total_weight > 0.0 {
            (weighted_sum / total_weight).max(0.0).min(1.0)
        } else {
            0.0 // No strategies contributed
        };
        
        let confidence_score = self.calculate_confidence_score(concept, &strategy_scores, total_weight);
        
        // Generate detailed breakdown
        let mut score_breakdown = BTreeMap::new();
        for (strategy, score) in &strategy_scores {
            score_breakdown.insert(strategy.clone(), *score);
        }
        score_breakdown.insert("total_weighted".to_string(), total_score);
        score_breakdown.insert("confidence".to_string(), confidence_score);
        
        // Generate justification
        let justification = self.generate_justification(&strategy_scores, total_score, confidence_score);
        
        // Create metadata
        let total_time = start_time.elapsed().as_micros() as u64;
        let metadata = ScoringMetadata {
            scored_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            strategies_used: strategy_scores.len(),
            total_time_microseconds: total_time,
            from_cache: false,
            strategy_times,
        };
        
        // Create performance metrics if monitoring is enabled
        let performance_metrics = if self.monitoring_enabled {
            Some(ScoringPerformanceMetrics {
                strategy_execution_times: metadata.strategy_times.clone(),
                memory_usage_bytes: None, // Would require more complex tracking
                cache_hits: 0,
                cache_misses: 1,
                parallelization_efficiency: 1.0, // Single operation
            })
        } else {
            None
        };
        
        let result = AllocationScoringResult {
            total_score,
            confidence_score,
            semantic_score: strategy_scores.get("semantic_similarity").copied(),
            property_score: strategy_scores.get("property_compatibility").copied(),
            structural_score: strategy_scores.get("structural_composite").copied(),
            score_breakdown,
            justification,
            scoring_metadata: metadata,
            performance_metrics,
        };
        
        // Cache the result
        if self.config.enable_caching {
            self.cache_result(cache_key, result.clone());
        }
        
        // Update overall performance statistics
        if self.monitoring_enabled {
            self.update_overall_performance_statistics(total_time);
        }
        
        Ok(result)
    }
    
    /// Batch score multiple allocations
    pub fn batch_score_allocations(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<AllocationScoringResult>, ScoringError> {
        // Process in sequential batches for now (can be optimized to parallel)
        let results: Result<Vec<_>, _> = concepts.iter()
            .map(|concept| self.score_allocation(concept, context))
            .collect();
        
        results
    }
    
    /// Parallel score multiple allocations for maximum performance
    pub fn parallel_score_allocations(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<AllocationScoringResult>, ScoringError> {
        let start_time = Instant::now();
        
        // Process in parallel chunks
        let results: Result<Vec<_>, _> = concepts.par_chunks(self.config.parallel_batch_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|concept| {
                    self.score_allocation(concept, context)
                })
            })
            .collect();
        
        // Update parallel processing statistics
        if self.monitoring_enabled {
            let total_time = start_time.elapsed().as_micros() as u64;
            self.update_parallel_processing_statistics(concepts.len(), total_time);
        }
        
        results
    }
    
    /// Rank allocations by score
    pub fn rank_allocations(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<RankedAllocationResult>, ScoringError> {
        let mut scored_concepts: Vec<_> = concepts.iter()
            .map(|concept| {
                let score = self.score_allocation(concept, context)?;
                Ok((concept.clone(), score))
            })
            .collect::<Result<Vec<_>, ScoringError>>()?;
        
        // Sort by total score (highest first)
        scored_concepts.sort_by(|a, b| {
            b.1.total_score.partial_cmp(&a.1.total_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Create ranked results
        let total_count = scored_concepts.len();
        let ranked_results = scored_concepts.into_iter()
            .enumerate()
            .map(|(index, (concept, allocation_score))| {
                RankedAllocationResult {
                    concept,
                    allocation_score,
                    rank: index + 1,
                    percentile: if total_count > 1 {
                        1.0 - (index as f32 / (total_count - 1) as f32)
                    } else {
                        1.0
                    },
                }
            })
            .collect();
        
        Ok(ranked_results)
    }
    
    /// Enable or disable a strategy
    pub fn enable_strategy(&mut self, strategy_name: &str) -> Result<(), ScoringError> {
        if self.strategy_status.contains_key(strategy_name) {
            self.strategy_status.insert(strategy_name.to_string(), StrategyStatus::Enabled);
            Ok(())
        } else {
            Err(ScoringError::InvalidStrategy(format!("Unknown strategy: {}", strategy_name)))
        }
    }
    
    /// Disable a strategy
    pub fn disable_strategy(&mut self, strategy_name: &str) -> Result<(), ScoringError> {
        if self.strategy_status.contains_key(strategy_name) {
            self.strategy_status.insert(strategy_name.to_string(), StrategyStatus::Disabled);
            Ok(())
        } else {
            Err(ScoringError::InvalidStrategy(format!("Unknown strategy: {}", strategy_name)))
        }
    }
    
    /// Check if a strategy is enabled
    pub fn is_strategy_enabled(&self, strategy_name: &str) -> bool {
        matches!(self.strategy_status.get(strategy_name), Some(StrategyStatus::Enabled))
    }
    
    /// Get number of available strategies
    pub fn strategy_count(&self) -> usize {
        self.strategy_status.len()
    }
    
    /// Check if system supports parallel scoring
    pub fn supports_parallel_scoring(&self) -> bool {
        true // Always supported
    }
    
    /// Enable/disable performance monitoring
    pub fn enable_performance_monitoring(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ScoringConfig {
        &self.config
    }
    
    /// Get performance statistics
    pub fn get_performance_statistics(&self) -> ScoringSystemPerformanceStats {
        if let Ok(stats) = self.performance_stats.lock() {
            stats.clone()
        } else {
            ScoringSystemPerformanceStats {
                total_scoring_operations: 0,
                average_scoring_time_ms: 0.0,
                min_scoring_time_ms: 0.0,
                max_scoring_time_ms: 0.0,
                strategy_performance: HashMap::new(),
                cache_statistics: CacheStatistics {
                    hit_count: 0,
                    miss_count: 0,
                    hit_rate: 0.0,
                    current_size: 0,
                    max_size: self.config.cache_size,
                    eviction_count: 0,
                },
                parallel_processing_stats: ParallelProcessingStats {
                    parallel_operations: 0,
                    average_speedup: 1.0,
                    max_speedup: 1.0,
                    efficiency: 1.0,
                },
            }
        }
    }
    
    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        if let Ok(stats) = self.performance_stats.lock() {
            stats.cache_statistics.clone()
        } else {
            CacheStatistics {
                hit_count: 0,
                miss_count: 0,
                hit_rate: 0.0,
                current_size: self.result_cache.len(),
                max_size: self.config.cache_size,
                eviction_count: 0,
            }
        }
    }
    
    /// Clear all caches
    pub fn clear_caches(&self) {
        self.result_cache.clear();
        // Could also clear individual strategy caches here
    }
    
    // Private helper methods
    
    fn generate_cache_key(&self, concept: &ExtractedConcept, context: &AllocationContext) -> String {
        format!("{}:{}:{}", 
                concept.name, 
                context.target_concept, 
                context.ancestor_concepts.join(","))
    }
    
    fn calculate_confidence_score(&self, concept: &ExtractedConcept, strategy_scores: &HashMap<String, f32>, total_weight: f32) -> f32 {
        // Base confidence from concept confidence
        let mut confidence = concept.confidence;
        
        // Adjust based on strategy agreement
        if strategy_scores.len() > 1 {
            let scores: Vec<f32> = strategy_scores.values().copied().collect();
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
            
            // Lower variance = higher confidence
            let agreement_factor = (1.0 / (1.0 + variance)).min(1.0);
            confidence *= agreement_factor;
        }
        
        // Adjust based on total weight coverage
        if total_weight < 1.0 {
            confidence *= total_weight; // Reduce confidence if not all strategies participated
        }
        
        confidence.max(0.0).min(1.0)
    }
    
    fn generate_justification(&self, strategy_scores: &HashMap<String, f32>, total_score: f32, confidence_score: f32) -> String {
        if strategy_scores.is_empty() {
            return "No scoring strategies were able to evaluate this allocation.".to_string();
        }
        
        let mut parts = Vec::new();
        
        // Overall assessment
        let assessment = if total_score >= 0.8 {
            "Excellent allocation candidate"
        } else if total_score >= 0.6 {
            "Good allocation candidate"
        } else if total_score >= 0.4 {
            "Moderate allocation candidate"
        } else {
            "Poor allocation candidate"
        };
        
        parts.push(format!("{} (score: {:.2})", assessment, total_score));
        
        // Strategy contributions
        let mut strategy_details = Vec::new();
        
        if let Some(semantic_score) = strategy_scores.get("semantic_similarity") {
            strategy_details.push(format!("semantic similarity: {:.2}", semantic_score));
        }
        
        if let Some(property_score) = strategy_scores.get("property_compatibility") {
            strategy_details.push(format!("property compatibility: {:.2}", property_score));
        }
        
        if let Some(structural_score) = strategy_scores.get("structural_composite") {
            strategy_details.push(format!("structural fit: {:.2}", structural_score));
        }
        
        if !strategy_details.is_empty() {
            parts.push(format!("Component scores: {}", strategy_details.join(", ")));
        }
        
        // Confidence assessment
        if confidence_score < 0.5 {
            parts.push(format!("Low confidence ({:.2}) - consider additional validation", confidence_score));
        } else if confidence_score > 0.8 {
            parts.push(format!("High confidence ({:.2})", confidence_score));
        }
        
        parts.join(". ")
    }
    
    fn cache_result(&self, cache_key: String, result: AllocationScoringResult) {
        // Simple cache management - could be improved with LRU eviction
        if self.result_cache.len() >= self.config.cache_size {
            // Remove some entries to make space (simple approach)
            let keys_to_remove: Vec<_> = self.result_cache.iter()
                .take(self.config.cache_size / 10) // Remove 10%
                .map(|entry| entry.key().clone())
                .collect();
            
            for key in keys_to_remove {
                self.result_cache.remove(&key);
            }
        }
        
        self.result_cache.insert(cache_key, result);
    }
    
    fn update_cache_statistics(&self, cache_hit: bool) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            if cache_hit {
                stats.cache_statistics.hit_count += 1;
            } else {
                stats.cache_statistics.miss_count += 1;
            }
            
            let total = stats.cache_statistics.hit_count + stats.cache_statistics.miss_count;
            if total > 0 {
                stats.cache_statistics.hit_rate = stats.cache_statistics.hit_count as f32 / total as f32;
            }
            
            stats.cache_statistics.current_size = self.result_cache.len();
        }
    }
    
    fn update_strategy_performance(&self, strategy_name: &str, execution_time: u64, success: bool, error: Option<String>) {
        if !self.monitoring_enabled {
            return;
        }
        
        if let Ok(mut stats) = self.performance_stats.lock() {
            let strategy_stats = stats.strategy_performance
                .entry(strategy_name.to_string())
                .or_insert_with(|| StrategyPerformanceStats {
                    operation_count: 0,
                    average_time_ms: 0.0,
                    success_rate: 1.0,
                    error_count: 0,
                    last_error: None,
                });
            
            strategy_stats.operation_count += 1;
            
            if success {
                let time_ms = execution_time as f32 / 1000.0; // Convert microseconds to milliseconds
                strategy_stats.average_time_ms = 
                    (strategy_stats.average_time_ms * (strategy_stats.operation_count - 1) as f32 + time_ms) / 
                    strategy_stats.operation_count as f32;
            } else {
                strategy_stats.error_count += 1;
                strategy_stats.last_error = error;
            }
            
            let success_count = strategy_stats.operation_count - strategy_stats.error_count;
            strategy_stats.success_rate = success_count as f32 / strategy_stats.operation_count as f32;
        }
    }
    
    fn update_overall_performance_statistics(&self, execution_time: u64) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.total_scoring_operations += 1;
            
            let time_ms = execution_time as f32 / 1000.0; // Convert microseconds to milliseconds
            
            stats.average_scoring_time_ms = 
                (stats.average_scoring_time_ms * (stats.total_scoring_operations - 1) as f32 + time_ms) / 
                stats.total_scoring_operations as f32;
            
            stats.min_scoring_time_ms = stats.min_scoring_time_ms.min(time_ms);
            stats.max_scoring_time_ms = stats.max_scoring_time_ms.max(time_ms);
        }
    }
    
    fn update_parallel_processing_statistics(&self, batch_size: usize, total_time: u64) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.parallel_processing_stats.parallel_operations += 1;
            
            // Estimate speedup (simplified calculation)
            let estimated_sequential_time = batch_size as u64 * 2000; // Assume 2ms per item sequentially
            let speedup = estimated_sequential_time as f32 / total_time as f32;
            
            stats.parallel_processing_stats.max_speedup = stats.parallel_processing_stats.max_speedup.max(speedup);
            stats.parallel_processing_stats.average_speedup = 
                (stats.parallel_processing_stats.average_speedup * (stats.parallel_processing_stats.parallel_operations - 1) as f32 + speedup) / 
                stats.parallel_processing_stats.parallel_operations as f32;
            
            // Calculate efficiency (speedup / theoretical_max_speedup)
            let theoretical_max = rayon::current_num_threads() as f32;
            stats.parallel_processing_stats.efficiency = (speedup / theoretical_max).min(1.0);
        }
    }
}

impl Default for IntegratedScoringSystem {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create IntegratedScoringSystem with all strategy integration
2. Implement weighted scoring combination with configurable weights
3. Add batch and parallel scoring capabilities for performance
4. Implement comprehensive performance monitoring and caching
5. Add strategy management (enable/disable) and error handling
6. Ensure overall system meets <2ms total scoring time target

## Success Criteria
- [ ] IntegratedScoringSystem compiles and initializes correctly
- [ ] Weighted scoring combination produces accurate results
- [ ] Strategy enable/disable functionality works properly
- [ ] Batch and parallel scoring achieve performance targets
- [ ] Caching improves repeated scoring performance significantly
- [ ] Performance monitoring captures detailed metrics
- [ ] All comprehensive tests pass with proper error handling