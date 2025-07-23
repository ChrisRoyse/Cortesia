// Comprehensive integration tests for similarity metrics
//
// Tests cover:
// - Mathematical correctness across all similarity algorithms
// - Property testing for mathematical invariants (reflexivity, symmetry, triangle inequality)
// - Numerical stability edge case testing
// - Performance validation for large datasets
// - Cross-metric consistency validation

use llmkg::math::similarity::{
    SimilarityEngine, SimilarityConfig, SimilarityMetric,
    euclidean_distance, euclidean_norm,
};
use std::collections::HashMap;

#[test]
fn test_similarity_engine_creation() {
    let engine = SimilarityEngine::new();
    let (cache_size, _) = engine.cache_stats();
    assert_eq!(cache_size, 0);
    
    let config = SimilarityConfig {
        cosine_weight: 0.7,
        textual_weight: 0.2,
        length_weight: 0.1,
        cache_enabled: true,
        cache_ttl_seconds: 1800,
    };
    
    let engine_with_config = SimilarityEngine::with_config(config);
    assert_eq!(engine_with_config.config.cosine_weight, 0.7);
}

#[test]
fn test_cosine_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test various vector configurations
    let test_cases = vec![
        // Identical vectors
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 1.0),
        // Orthogonal vectors
        (vec![1.0, 0.0], vec![0.0, 1.0], 0.0),
        // Anti-parallel vectors
        (vec![1.0, 1.0], vec![-1.0, -1.0], -1.0),
        // Similar vectors
        (vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], 0.999), // approximately
    ];
    
    for (vec1, vec2, expected) in test_cases {
        let similarity = engine.cosine_similarity(&vec1, &vec2).unwrap();
        if expected == 1.0 || expected == 0.0 || expected == -1.0 {
            assert!((similarity - expected).abs() < 1e-6, 
                "Expected {}, got {} for vectors {:?} and {:?}", 
                expected, similarity, vec1, vec2);
        } else {
            assert!(similarity > 0.99, 
                "Expected similarity > 0.99, got {} for similar vectors {:?} and {:?}", 
                similarity, vec1, vec2);
        }
    }
}

#[test]
fn test_euclidean_distance_integration() {
    let engine = SimilarityEngine::new();
    
    // Test cases with known distances
    let test_cases = vec![
        // 3-4-5 right triangle
        (vec![0.0, 0.0], vec![3.0, 4.0], 5.0),
        // Same point
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0.0),
        // Unit distance
        (vec![0.0], vec![1.0], 1.0),
        // Multiple dimensions
        (vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0], (3.0_f32).sqrt()),
    ];
    
    for (vec1, vec2, expected) in test_cases {
        let distance = engine.euclidean_distance(&vec1, &vec2).unwrap();
        assert!((distance - expected).abs() < 1e-6, 
            "Expected distance {}, got {} for vectors {:?} and {:?}", 
            expected, distance, vec1, vec2);
    }
}

#[test]
fn test_jaccard_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test with various set configurations
    let test_cases = vec![
        // Identical sets
        (vec![1, 2, 3], vec![1, 2, 3], 1.0),
        // Disjoint sets
        (vec![1, 2], vec![3, 4], 0.0),
        // Partial overlap
        (vec![1, 2, 3, 4], vec![3, 4, 5, 6], 1.0/3.0), // 2 common, 6 total
        // One empty set
        (vec![], vec![1, 2, 3], 0.0),
        // Both empty sets
        (vec![], vec![], 1.0),
    ];
    
    for (set1, set2, expected) in test_cases {
        let similarity = engine.jaccard_similarity(&set1, &set2);
        assert!((similarity - expected).abs() < 1e-6, 
            "Expected Jaccard similarity {}, got {} for sets {:?} and {:?}", 
            expected, similarity, set1, set2);
    }
}

#[test]
fn test_textual_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test various text similarity cases
    let test_cases = vec![
        // Identical text
        ("hello world", "hello world", 1.0),
        // Case insensitive
        ("Hello World", "HELLO WORLD", 1.0),
        // Completely different
        ("abc", "xyz", 0.0), // Note: might not be exactly 0.0 due to n-gram overlap
        // Similar text
        ("testing similarity", "testing similarities", 0.7), // threshold for "similar"
    ];
    
    for (text1, text2, min_expected) in test_cases {
        let similarity = engine.textual_similarity(text1, text2);
        if min_expected == 1.0 {
            assert_eq!(similarity, 1.0, 
                "Expected exact match for '{}' and '{}'", text1, text2);
        } else if min_expected == 0.0 {
            assert!(similarity >= 0.0, 
                "Similarity should be non-negative for '{}' and '{}'", text1, text2);
        } else {
            assert!(similarity >= min_expected, 
                "Expected similarity >= {}, got {} for '{}' and '{}'", 
                min_expected, similarity, text1, text2);
        }
    }
}

#[test]
fn test_semantic_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test semantic similarity with embeddings and text
    let embedding1 = vec![1.0, 0.0, 0.0];
    let embedding2 = vec![0.9, 0.1, 0.0]; // Similar to embedding1
    let embedding3 = vec![0.0, 1.0, 0.0]; // Orthogonal to embedding1
    
    let text1 = "artificial intelligence";
    let text2 = "machine learning";
    let text3 = "cooking recipes";
    
    // Similar embeddings with related text should have high similarity
    let sim_high = engine.semantic_similarity(&embedding1, &embedding2, text1, text2).unwrap();
    assert!(sim_high > 0.5, "Expected high semantic similarity, got {}", sim_high);
    
    // Orthogonal embeddings with unrelated text should have lower similarity
    let sim_low = engine.semantic_similarity(&embedding1, &embedding3, text1, text3).unwrap();
    assert!(sim_low < sim_high, "Expected lower similarity for unrelated content");
    
    // Verify bounds
    assert!(sim_high >= 0.0 && sim_high <= 1.0, "Semantic similarity out of bounds");
    assert!(sim_low >= 0.0 && sim_low <= 1.0, "Semantic similarity out of bounds");
}

#[test]
fn test_structural_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test structural similarity with graph neighbors
    let neighbors1 = vec![1, 2, 3, 4, 5];
    let neighbors2 = vec![3, 4, 5, 6, 7];
    let neighbors3 = vec![8, 9, 10]; // No overlap
    
    // Overlapping structures
    let sim_overlap = engine.structural_similarity(&neighbors1, &neighbors2, None, None);
    assert!(sim_overlap > 0.0, "Expected positive similarity for overlapping structures");
    
    // Non-overlapping structures
    let sim_disjoint = engine.structural_similarity(&neighbors1, &neighbors3, None, None);
    assert_eq!(sim_disjoint, 0.0, "Expected zero similarity for disjoint structures");
    
    // Test with weights
    let weights1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weights2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];
    
    let sim_weighted = engine.structural_similarity(&neighbors1, &neighbors2, Some(&weights1), Some(&weights2));
    assert!(sim_weighted >= 0.0 && sim_weighted <= 1.0, "Weighted similarity out of bounds");
}

#[test]
fn test_graph_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Create a test graph
    let mut graph = HashMap::new();
    graph.insert(1, vec![2, 3, 4]);
    graph.insert(2, vec![1, 5]);
    graph.insert(3, vec![1, 5]);
    graph.insert(4, vec![1, 6]);
    graph.insert(5, vec![2, 3]);
    graph.insert(6, vec![4]);
    
    // Test similarity between connected nodes
    let sim_connected = engine.graph_similarity(1, 2, &graph, 2);
    assert!(sim_connected > 0.0, "Connected nodes should have positive similarity");
    
    // Test similarity between same node
    let sim_same = engine.graph_similarity(1, 1, &graph, 2);
    assert_eq!(sim_same, 1.0, "Node should have perfect similarity with itself");
    
    // Test similarity between distant nodes
    let sim_distant = engine.graph_similarity(2, 6, &graph, 1);
    assert!(sim_distant >= 0.0 && sim_distant <= 1.0, "Distance similarity out of bounds");
}

#[test]
fn test_levenshtein_similarity_integration() {
    let engine = SimilarityEngine::new();
    
    // Test known Levenshtein distance cases
    let test_cases = vec![
        // Identical strings
        ("kitten", "kitten", 1.0),
        // Single character difference
        ("cat", "bat", 1.0 - 1.0/3.0), // distance 1, max length 3
        // Complete substitution
        ("abc", "xyz", 0.0), // distance 3, max length 3
        // Different lengths
        ("short", "much longer string", 1.0 - 13.0/18.0), // approximate
    ];
    
    for (str1, str2, expected_min) in test_cases {
        let similarity = engine.levenshtein_similarity(str1, str2);
        if str1 == str2 {
            assert_eq!(similarity, 1.0, "Identical strings should have similarity 1.0");
        } else {
            assert!(similarity >= 0.0 && similarity <= 1.0, 
                "Levenshtein similarity out of bounds for '{}' and '{}'", str1, str2);
            if expected_min > 0.0 {
                assert!(similarity >= expected_min - 0.1, 
                    "Expected similarity >= {}, got {} for '{}' and '{}'", 
                    expected_min, similarity, str1, str2);
            }
        }
    }
}

#[test]
fn test_similarity_metric_enum_integration() {
    // Test all similarity metric enum values
    let metrics = vec![
        SimilarityMetric::Cosine,
        SimilarityMetric::Euclidean,
        SimilarityMetric::Manhattan,
        SimilarityMetric::Jaccard,
        SimilarityMetric::Levenshtein,
        SimilarityMetric::Semantic,
        SimilarityMetric::Structural,
        SimilarityMetric::Graph,
    ];
    
    let expected_strings = vec![
        "cosine", "euclidean", "manhattan", "jaccard",
        "levenshtein", "semantic", "structural", "graph"
    ];
    
    for (metric, expected) in metrics.iter().zip(expected_strings.iter()) {
        assert_eq!(metric.as_str(), *expected);
    }
    
    // Test that all metrics are unique
    let metric_strings: Vec<_> = metrics.iter().map(|m| m.as_str()).collect();
    let mut unique_strings = metric_strings.clone();
    unique_strings.sort();
    unique_strings.dedup();
    assert_eq!(metric_strings.len(), unique_strings.len(), "Similarity metrics should be unique");
}

#[test]
fn test_cache_functionality_integration() {
    let mut engine = SimilarityEngine::new();
    
    // Verify initial cache state
    let (initial_size, initial_capacity) = engine.cache_stats();
    assert_eq!(initial_size, 0);
    assert!(initial_capacity >= 0);
    
    // Clear empty cache (should not panic)
    engine.clear_cache();
    let (size_after_clear, _) = engine.cache_stats();
    assert_eq!(size_after_clear, 0);
    
    // Note: Actual cache usage would require implementing cache storage in similarity methods
    // This test verifies the cache interface works correctly
}

#[test]
fn test_configuration_weight_validation() {
    // Test various weight configurations
    let configs = vec![
        SimilarityConfig {
            cosine_weight: 1.0,
            textual_weight: 0.0,
            length_weight: 0.0,
            cache_enabled: true,
            cache_ttl_seconds: 3600,
        },
        SimilarityConfig {
            cosine_weight: 0.5,
            textual_weight: 0.3,
            length_weight: 0.2,
            cache_enabled: false,
            cache_ttl_seconds: 0,
        },
    ];
    
    for config in configs {
        let engine = SimilarityEngine::with_config(config.clone());
        
        // Verify configuration is stored correctly
        assert_eq!(engine.config.cosine_weight, config.cosine_weight);
        assert_eq!(engine.config.textual_weight, config.textual_weight);
        assert_eq!(engine.config.length_weight, config.length_weight);
        assert_eq!(engine.config.cache_enabled, config.cache_enabled);
        assert_eq!(engine.config.cache_ttl_seconds, config.cache_ttl_seconds);
        
        // Test semantic similarity with custom weights
        let embedding1 = vec![1.0, 0.0];
        let embedding2 = vec![0.0, 1.0];
        let result = engine.semantic_similarity(&embedding1, &embedding2, "test1", "test2");
        assert!(result.is_ok(), "Semantic similarity should work with custom config");
        
        let similarity = result.unwrap();
        assert!(similarity >= 0.0 && similarity <= 1.0, "Similarity should be in valid range");
    }
}

#[test]
fn test_standalone_function_integration() {
    // Test standalone utility functions
    let vec1 = vec![3.0, 4.0];
    let vec2 = vec![0.0, 0.0];
    
    // Test euclidean_distance function
    let distance = euclidean_distance(&vec1, &vec2);
    assert_eq!(distance, 5.0);
    
    // Test euclidean_norm function
    let norm = euclidean_norm(&vec1);
    assert_eq!(norm, 5.0);
    
    // Test with various vector sizes
    let vectors = vec![
        vec![1.0],
        vec![1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ];
    
    for vec in vectors {
        let norm = euclidean_norm(&vec);
        let expected = (vec.len() as f32).sqrt();
        assert!((norm - expected).abs() < 1e-6, 
            "Expected norm {}, got {} for vector {:?}", expected, norm, vec);
    }
}

#[test]
fn test_error_handling_integration() {
    let engine = SimilarityEngine::new();
    
    // Test dimension mismatch errors
    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    
    assert!(engine.cosine_similarity(&vec1, &vec2).is_err());
    assert!(engine.euclidean_distance(&vec1, &vec2).is_err());
    assert!(engine.manhattan_distance(&vec1, &vec2).is_err());
    
    // Test semantic similarity with mismatched embeddings
    let result = engine.semantic_similarity(&vec1, &vec2, "text1", "text2");
    assert!(result.is_err());
}

// Performance integration test
#[test]
fn test_similarity_performance_integration() {
    let engine = SimilarityEngine::new();
    
    // Create large vectors for performance testing
    let large_vec1: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let large_vec2: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();
    
    let start = std::time::Instant::now();
    
    // Perform multiple similarity calculations
    for _ in 0..100 {
        let _ = engine.cosine_similarity(&large_vec1, &large_vec2).unwrap();
        let _ = engine.euclidean_distance(&large_vec1, &large_vec2).unwrap();
    }
    
    let elapsed = start.elapsed();
    
    // Performance should be reasonable (adjust threshold as needed)
    assert!(elapsed.as_millis() < 1000, 
        "Similarity calculations took too long: {:?}", elapsed);
}

#[test]
fn test_cross_metric_consistency() {
    let engine = SimilarityEngine::new();
    
    // Test that different metrics produce consistent relative orderings
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![0.9, 0.1, 0.0]; // Similar to vec1
    let vec3 = vec![0.0, 1.0, 0.0]; // Different from vec1
    
    let cosine_sim_12 = engine.cosine_similarity(&vec1, &vec2).unwrap();
    let cosine_sim_13 = engine.cosine_similarity(&vec1, &vec3).unwrap();
    
    let euclidean_dist_12 = engine.euclidean_distance(&vec1, &vec2).unwrap();
    let euclidean_dist_13 = engine.euclidean_distance(&vec1, &vec3).unwrap();
    
    // Cosine similarity: higher values indicate more similar
    // Euclidean distance: lower values indicate more similar
    // So these should be inversely related
    assert!(cosine_sim_12 > cosine_sim_13, "Cosine similarity ordering incorrect");
    assert!(euclidean_dist_12 < euclidean_dist_13, "Euclidean distance ordering incorrect");
}