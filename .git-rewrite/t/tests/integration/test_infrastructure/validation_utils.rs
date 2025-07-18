// Validation Utilities
// Helper functions for validating test results

use std::collections::{HashMap, HashSet};
use crate::entity::EntityKey;
use crate::knowledge_graph::KnowledgeGraph;
use crate::embedding::EmbeddingStore;
use crate::query::{RagContext, GraphPath};

/// Result validation utilities
pub struct ResultValidator;

impl ResultValidator {
    /// Validate RAG context completeness
    pub fn validate_rag_context(context: &RagContext, expected_size: usize) -> Result<(), String> {
        if context.entities.len() != expected_size {
            return Err(format!(
                "Expected {} entities in context, got {}",
                expected_size,
                context.entities.len()
            ));
        }

        // Check for duplicates
        let unique_entities: HashSet<_> = context.entities.iter().collect();
        if unique_entities.len() != context.entities.len() {
            return Err("Duplicate entities found in context".to_string());
        }

        // Validate scores
        for score in &context.relevance_scores {
            if !score.is_finite() || *score < 0.0 || *score > 1.0 {
                return Err(format!("Invalid relevance score: {}", score));
            }
        }

        // Check score ordering (should be descending)
        for i in 1..context.relevance_scores.len() {
            if context.relevance_scores[i] > context.relevance_scores[i - 1] {
                return Err("Relevance scores not in descending order".to_string());
            }
        }

        Ok(())
    }

    /// Validate graph paths
    pub fn validate_graph_paths(
        paths: &[GraphPath],
        source: EntityKey,
        target: EntityKey,
        kg: &KnowledgeGraph,
    ) -> Result<(), String> {
        for (i, path) in paths.iter().enumerate() {
            // Check path starts and ends correctly
            if path.entities.first() != Some(&source) {
                return Err(format!("Path {} doesn't start with source entity", i));
            }
            if path.entities.last() != Some(&target) {
                return Err(format!("Path {} doesn't end with target entity", i));
            }

            // Check path continuity
            for j in 0..path.entities.len() - 1 {
                let from = path.entities[j];
                let to = path.entities[j + 1];
                
                let neighbors = kg.get_neighbors(from);
                let has_edge = neighbors.iter().any(|rel| rel.target() == to);
                
                if !has_edge {
                    return Err(format!(
                        "Path {} has invalid edge from {:?} to {:?}",
                        i, from, to
                    ));
                }
            }

            // Validate path length
            if path.entities.len() != path.relationships.len() + 1 {
                return Err(format!(
                    "Path {} has mismatched entities ({}) and relationships ({})",
                    i,
                    path.entities.len(),
                    path.relationships.len()
                ));
            }

            // Check for cycles (optional, depending on requirements)
            let unique_entities: HashSet<_> = path.entities.iter().collect();
            if unique_entities.len() != path.entities.len() {
                // Path contains cycles - this might be valid depending on use case
                // return Err(format!("Path {} contains cycles", i));
            }
        }

        Ok(())
    }

    /// Validate similarity search results
    pub fn validate_similarity_results(
        results: &[(EntityKey, f32)],
        query_embedding: &[f32],
        embedding_store: &EmbeddingStore,
        expected_count: usize,
    ) -> Result<(), String> {
        if results.len() != expected_count {
            return Err(format!(
                "Expected {} results, got {}",
                expected_count,
                results.len()
            ));
        }

        // Check ordering (distances should be ascending)
        for i in 1..results.len() {
            if results[i].1 < results[i - 1].1 {
                return Err("Similarity results not ordered by distance".to_string());
            }
        }

        // Verify distances are correct
        for (entity, reported_distance) in results {
            if let Ok(embedding) = embedding_store.get_embedding(*entity) {
                let actual_distance = calculate_euclidean_distance(query_embedding, &embedding);
                let diff = (reported_distance - actual_distance).abs();
                
                if diff > 1e-5 {
                    return Err(format!(
                        "Distance mismatch for {:?}: reported {}, actual {}",
                        entity, reported_distance, actual_distance
                    ));
                }
            } else {
                return Err(format!("Entity {:?} not found in embedding store", entity));
            }
        }

        Ok(())
    }

    /// Validate graph statistics
    pub fn validate_graph_statistics(
        kg: &KnowledgeGraph,
        expected_entities: usize,
        expected_relationships: usize,
    ) -> Result<(), String> {
        let actual_entities = kg.entity_count();
        let actual_relationships = kg.relationship_count();

        if actual_entities != expected_entities {
            return Err(format!(
                "Entity count mismatch: expected {}, got {}",
                expected_entities, actual_entities
            ));
        }

        if actual_relationships != expected_relationships {
            return Err(format!(
                "Relationship count mismatch: expected {}, got {}",
                expected_relationships, actual_relationships
            ));
        }

        Ok(())
    }

    /// Validate memory usage is within bounds
    pub fn validate_memory_usage(
        current_usage: u64,
        baseline: u64,
        max_allowed_overhead: u64,
    ) -> Result<(), String> {
        let overhead = current_usage.saturating_sub(baseline);
        
        if overhead > max_allowed_overhead {
            return Err(format!(
                "Memory usage exceeded: {} bytes overhead (max allowed: {})",
                overhead, max_allowed_overhead
            ));
        }

        Ok(())
    }
}

/// Performance validation utilities
pub struct PerformanceValidator;

impl PerformanceValidator {
    /// Validate query latency
    pub fn validate_latency(
        actual: std::time::Duration,
        max_allowed: std::time::Duration,
        operation: &str,
    ) -> Result<(), String> {
        if actual > max_allowed {
            return Err(format!(
                "{} latency too high: {:?} (max allowed: {:?})",
                operation, actual, max_allowed
            ));
        }
        Ok(())
    }

    /// Validate throughput
    pub fn validate_throughput(
        operations: usize,
        duration: std::time::Duration,
        min_ops_per_sec: f64,
    ) -> Result<(), String> {
        let actual_ops_per_sec = operations as f64 / duration.as_secs_f64();
        
        if actual_ops_per_sec < min_ops_per_sec {
            return Err(format!(
                "Throughput too low: {:.2} ops/sec (min required: {:.2})",
                actual_ops_per_sec, min_ops_per_sec
            ));
        }
        
        Ok(())
    }

    /// Validate scaling behavior
    pub fn validate_scaling(
        measurements: &[(usize, std::time::Duration)],
        max_complexity: f64,
    ) -> Result<(), String> {
        if measurements.len() < 2 {
            return Err("Not enough measurements for scaling analysis".to_string());
        }

        // Simple linear regression to estimate complexity
        let n = measurements.len() as f64;
        let sum_x: f64 = measurements.iter().map(|(size, _)| *size as f64).sum();
        let sum_y: f64 = measurements.iter().map(|(_, dur)| dur.as_secs_f64()).sum();
        let sum_xy: f64 = measurements.iter()
            .map(|(size, dur)| *size as f64 * dur.as_secs_f64())
            .sum();
        let sum_x2: f64 = measurements.iter()
            .map(|(size, _)| (*size as f64).powi(2))
            .sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        
        // Estimate complexity order
        let complexity = if slope < 0.001 {
            0.0 // O(1)
        } else {
            // Rough estimate of polynomial order
            let first = measurements.first().unwrap();
            let last = measurements.last().unwrap();
            
            let size_ratio = last.0 as f64 / first.0 as f64;
            let time_ratio = last.1.as_secs_f64() / first.1.as_secs_f64();
            
            time_ratio.log2() / size_ratio.log2()
        };

        if complexity > max_complexity {
            return Err(format!(
                "Complexity too high: O(n^{:.2}) (max allowed: O(n^{:.2}))",
                complexity, max_complexity
            ));
        }

        Ok(())
    }
}

/// Accuracy validation utilities
pub struct AccuracyValidator;

impl AccuracyValidator {
    /// Validate quantization accuracy
    pub fn validate_quantization_accuracy(
        original: &[f32],
        reconstructed: &[f32],
        max_error: f32,
    ) -> Result<(), String> {
        if original.len() != reconstructed.len() {
            return Err(format!(
                "Dimension mismatch: original {}, reconstructed {}",
                original.len(),
                reconstructed.len()
            ));
        }

        let error = calculate_euclidean_distance(original, reconstructed);
        
        if error > max_error {
            return Err(format!(
                "Quantization error too high: {} (max allowed: {})",
                error, max_error
            ));
        }

        Ok(())
    }

    /// Validate result overlap between two ranked lists
    pub fn validate_result_overlap<T: Eq + std::hash::Hash>(
        results1: &[T],
        results2: &[T],
        min_overlap: f64,
        top_k: usize,
    ) -> Result<(), String> {
        let k = top_k.min(results1.len()).min(results2.len());
        
        let set1: HashSet<_> = results1.iter().take(k).collect();
        let set2: HashSet<_> = results2.iter().take(k).collect();
        
        let intersection = set1.intersection(&set2).count();
        let overlap = intersection as f64 / k as f64;
        
        if overlap < min_overlap {
            return Err(format!(
                "Result overlap too low: {:.2} (min required: {:.2})",
                overlap, min_overlap
            ));
        }

        Ok(())
    }

    /// Validate false positive rate
    pub fn validate_false_positive_rate(
        false_positives: usize,
        total_negatives: usize,
        max_rate: f64,
    ) -> Result<(), String> {
        let actual_rate = false_positives as f64 / total_negatives as f64;
        
        if actual_rate > max_rate {
            return Err(format!(
                "False positive rate too high: {:.4} (max allowed: {:.4})",
                actual_rate, max_rate
            ));
        }

        Ok(())
    }
}

/// Concurrency validation utilities
pub struct ConcurrencyValidator;

impl ConcurrencyValidator {
    /// Validate thread safety by checking for data races
    pub fn validate_concurrent_results<T: Eq + std::hash::Hash>(
        sequential_results: Vec<T>,
        concurrent_results: Vec<Vec<T>>,
    ) -> Result<(), String> {
        // Flatten concurrent results
        let all_concurrent: HashSet<_> = concurrent_results
            .into_iter()
            .flatten()
            .collect();
        
        let sequential_set: HashSet<_> = sequential_results.into_iter().collect();
        
        // Check that concurrent execution produces same results
        if all_concurrent != sequential_set {
            return Err("Concurrent results differ from sequential".to_string());
        }

        Ok(())
    }

    /// Validate that operations complete within deadline under concurrency
    pub fn validate_concurrent_performance(
        thread_times: Vec<std::time::Duration>,
        max_allowed: std::time::Duration,
    ) -> Result<(), String> {
        for (i, &time) in thread_times.iter().enumerate() {
            if time > max_allowed {
                return Err(format!(
                    "Thread {} exceeded time limit: {:?} (max: {:?})",
                    i, time, max_allowed
                ));
            }
        }

        // Check for fairness - no thread should take significantly longer
        let avg_time = thread_times.iter().sum::<std::time::Duration>() / thread_times.len() as u32;
        let max_deviation = avg_time * 2; // Allow 2x average
        
        for (i, &time) in thread_times.iter().enumerate() {
            if time > avg_time + max_deviation {
                return Err(format!(
                    "Thread {} shows unfair scheduling: {:?} (avg: {:?})",
                    i, time, avg_time
                ));
            }
        }

        Ok(())
    }
}

/// Helper function to calculate euclidean distance
fn calculate_euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Integration test assertions
pub struct IntegrationAssertions;

impl IntegrationAssertions {
    /// Assert that a value is within a percentage of expected
    pub fn assert_within_percentage(actual: f64, expected: f64, percentage: f64) {
        let tolerance = expected * (percentage / 100.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "Value {} not within {}% of expected {} (tolerance: {})",
            actual, percentage, expected, tolerance
        );
    }

    /// Assert that a duration is within bounds
    pub fn assert_duration_in_range(
        actual: std::time::Duration,
        min: std::time::Duration,
        max: std::time::Duration,
    ) {
        assert!(
            actual >= min && actual <= max,
            "Duration {:?} not in range [{:?}, {:?}]",
            actual, min, max
        );
    }

    /// Assert collection properties
    pub fn assert_collection_properties<T>(
        collection: &[T],
        min_size: usize,
        max_size: usize,
        expected_unique: bool,
    ) where
        T: Eq + std::hash::Hash,
    {
        assert!(
            collection.len() >= min_size,
            "Collection size {} below minimum {}",
            collection.len(),
            min_size
        );
        assert!(
            collection.len() <= max_size,
            "Collection size {} above maximum {}",
            collection.len(),
            max_size
        );

        if expected_unique {
            let unique: HashSet<_> = collection.iter().collect();
            assert_eq!(
                unique.len(),
                collection.len(),
                "Collection contains duplicates"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::RagContext;

    #[test]
    fn test_rag_context_validation() {
        let context = RagContext {
            entities: vec![
                EntityKey::from_hash("a"),
                EntityKey::from_hash("b"),
                EntityKey::from_hash("c"),
            ],
            relevance_scores: vec![0.9, 0.7, 0.5],
            relationships: vec![],
            metadata: Default::default(),
        };

        assert!(ResultValidator::validate_rag_context(&context, 3).is_ok());
        assert!(ResultValidator::validate_rag_context(&context, 4).is_err());
    }

    #[test]
    fn test_performance_validation() {
        use std::time::Duration;

        // Test latency validation
        assert!(PerformanceValidator::validate_latency(
            Duration::from_millis(50),
            Duration::from_millis(100),
            "test_op"
        ).is_ok());

        assert!(PerformanceValidator::validate_latency(
            Duration::from_millis(150),
            Duration::from_millis(100),
            "test_op"
        ).is_err());

        // Test throughput validation
        assert!(PerformanceValidator::validate_throughput(
            1000,
            Duration::from_secs(1),
            500.0
        ).is_ok());

        assert!(PerformanceValidator::validate_throughput(
            100,
            Duration::from_secs(1),
            500.0
        ).is_err());
    }

    #[test]
    fn test_accuracy_validation() {
        let original = vec![1.0, 0.0, 0.0];
        let good_reconstruction = vec![0.95, 0.05, 0.0];
        let bad_reconstruction = vec![0.5, 0.5, 0.0];

        assert!(AccuracyValidator::validate_quantization_accuracy(
            &original,
            &good_reconstruction,
            0.1
        ).is_ok());

        assert!(AccuracyValidator::validate_quantization_accuracy(
            &original,
            &bad_reconstruction,
            0.1
        ).is_err());
    }
}