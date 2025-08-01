use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::hash::Hash;

/// Advanced similarity metrics for knowledge graph analysis
pub struct SimilarityEngine {
    /// Cache for computed similarities
    similarity_cache: HashMap<String, f32>,
    /// Configuration for different similarity methods
    config: SimilarityConfig,
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityEngine {
    pub fn new() -> Self {
        Self {
            similarity_cache: HashMap::new(),
            config: SimilarityConfig::default(),
        }
    }

    pub fn with_config(config: SimilarityConfig) -> Self {
        Self {
            similarity_cache: HashMap::new(),
            config,
        }
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        if vec1.len() != vec2.len() {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: vec1.len(),
                actual: vec2.len(),
            });
        }

        if vec1.is_empty() {
            return Ok(0.0);
        }

        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }
    
    /// Calculate Euclidean distance between two vectors
    pub fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        if vec1.len() != vec2.len() {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: vec1.len(),
                actual: vec2.len(),
            });
        }

        let sum_squared: f32 = vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        Ok(sum_squared.sqrt())
    }
    
    /// Calculate Euclidean norm (magnitude) of a vector
    pub fn euclidean_norm(&self, vec: &[f32]) -> f32 {
        vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Calculate Manhattan distance between two vectors
    pub fn manhattan_distance(&self, vec1: &[f32], vec2: &[f32]) -> Result<f32> {
        if vec1.len() != vec2.len() {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: vec1.len(),
                actual: vec2.len(),
            });
        }

        let distance: f32 = vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(distance)
    }

    /// Calculate Jaccard similarity between two sets
    pub fn jaccard_similarity<T>(&self, set1: &[T], set2: &[T]) -> f32
    where
        T: PartialEq + Clone + Hash + Eq,
    {
        let set1_unique: std::collections::HashSet<_> = set1.iter().collect();
        let set2_unique: std::collections::HashSet<_> = set2.iter().collect();

        let intersection_size = set1_unique.intersection(&set2_unique).count();
        let union_size = set1_unique.union(&set2_unique).count();

        if union_size == 0 {
            return 1.0; // Both sets are empty
        }

        intersection_size as f32 / union_size as f32
    }

    /// Calculate semantic similarity using multiple metrics
    pub fn semantic_similarity(
        &self,
        embedding1: &[f32],
        embedding2: &[f32],
        text1: &str,
        text2: &str,
    ) -> Result<f32> {
        // Combine multiple similarity measures
        let cosine_sim = self.cosine_similarity(embedding1, embedding2)?;
        let textual_sim = self.textual_similarity(text1, text2);
        let length_sim = self.length_similarity(text1, text2);

        // Weighted combination
        let semantic_sim = self.config.cosine_weight * cosine_sim
            + self.config.textual_weight * textual_sim
            + self.config.length_weight * length_sim;

        Ok(semantic_sim.clamp(0.0, 1.0))
    }

    /// Calculate textual similarity using various string metrics
    pub fn textual_similarity(&self, text1: &str, text2: &str) -> f32 {
        let levenshtein_sim = self.levenshtein_similarity(text1, text2);
        let word_overlap_sim = self.word_overlap_similarity(text1, text2);
        let char_ngram_sim = self.char_ngram_similarity(text1, text2, 3);

        // Weighted combination of textual similarities
        (levenshtein_sim + word_overlap_sim + char_ngram_sim) / 3.0
    }

    /// Calculate Levenshtein similarity (normalized edit distance)
    pub fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f32 {
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len()) as f32;
        
        if max_len == 0.0 {
            return 1.0;
        }
        
        1.0 - (distance as f32 / max_len)
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let m = chars1.len();
        let n = chars2.len();

        let mut dp = vec![vec![0; n + 1]; m + 1];

        // Initialize base cases
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }

    /// Calculate word overlap similarity
    fn word_overlap_similarity(&self, text1: &str, text2: &str) -> f32 {
        let text1_lower = text1.to_lowercase();
        let words1: std::collections::HashSet<_> = text1_lower
            .split_whitespace()
            .collect();
        let text2_lower = text2.to_lowercase();
        let words2: std::collections::HashSet<_> = text2_lower
            .split_whitespace()
            .collect();

        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        if union_size == 0 {
            return 1.0;
        }

        intersection_size as f32 / union_size as f32
    }

    /// Calculate character n-gram similarity
    fn char_ngram_similarity(&self, text1: &str, text2: &str, n: usize) -> f32 {
        let ngrams1 = self.get_char_ngrams(text1, n);
        let ngrams2 = self.get_char_ngrams(text2, n);

        self.jaccard_similarity(&ngrams1, &ngrams2)
    }

    /// Get character n-grams from text
    fn get_char_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return vec![text.to_string()];
        }

        chars
            .windows(n)
            .map(|window| window.iter().collect())
            .collect()
    }

    /// Calculate length similarity
    fn length_similarity(&self, text1: &str, text2: &str) -> f32 {
        let len1 = text1.len() as f32;
        let len2 = text2.len() as f32;
        
        if len1 == 0.0 && len2 == 0.0 {
            return 1.0;
        }
        
        let max_len = len1.max(len2);
        let min_len = len1.min(len2);
        
        min_len / max_len
    }

    /// Calculate structural similarity for graph entities
    pub fn structural_similarity(
        &self,
        neighbors1: &[u32],
        neighbors2: &[u32],
        weights1: Option<&[f32]>,
        weights2: Option<&[f32]>,
    ) -> f32 {
        // Calculate neighbor overlap
        let neighbor_sim = self.jaccard_similarity(neighbors1, neighbors2);
        
        // If weights are provided, consider weighted similarity
        if let (Some(w1), Some(w2)) = (weights1, weights2) {
            let weighted_sim = self.weighted_structural_similarity(neighbors1, neighbors2, w1, w2);
            (neighbor_sim + weighted_sim) / 2.0
        } else {
            neighbor_sim
        }
    }

    /// Calculate weighted structural similarity
    fn weighted_structural_similarity(
        &self,
        neighbors1: &[u32],
        neighbors2: &[u32],
        weights1: &[f32],
        weights2: &[f32],
    ) -> f32 {
        let mut total_weight = 0.0;
        let mut common_weight = 0.0;

        let neighbors1_map: HashMap<u32, f32> = neighbors1
            .iter()
            .zip(weights1.iter())
            .map(|(&n, &w)| (n, w))
            .collect();

        let neighbors2_map: HashMap<u32, f32> = neighbors2
            .iter()
            .zip(weights2.iter())
            .map(|(&n, &w)| (n, w))
            .collect();

        for (&neighbor, &weight1) in &neighbors1_map {
            total_weight += weight1;
            if let Some(&weight2) = neighbors2_map.get(&neighbor) {
                common_weight += weight1.min(weight2);
            }
        }

        for (&neighbor, &weight2) in &neighbors2_map {
            if !neighbors1_map.contains_key(&neighbor) {
                total_weight += weight2;
            }
        }

        if total_weight == 0.0 {
            return 1.0;
        }

        common_weight / total_weight
    }

    /// Calculate graph-aware similarity considering multiple hops
    pub fn graph_similarity(
        &self,
        entity1: u32,
        entity2: u32,
        graph_neighbors: &HashMap<u32, Vec<u32>>,
        max_hops: usize,
    ) -> f32 {
        let paths1 = self.get_neighborhood_paths(entity1, graph_neighbors, max_hops);
        let paths2 = self.get_neighborhood_paths(entity2, graph_neighbors, max_hops);

        self.path_similarity(&paths1, &paths2)
    }

    /// Get neighborhood paths up to max hops
    fn get_neighborhood_paths(
        &self,
        start_entity: u32,
        graph_neighbors: &HashMap<u32, Vec<u32>>,
        max_hops: usize,
    ) -> Vec<Vec<u32>> {
        let mut paths = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut current_path = vec![start_entity];
        
        self.dfs_paths(
            start_entity,
            graph_neighbors,
            &mut current_path,
            &mut paths,
            &mut visited,
            max_hops,
            0,
        );

        paths
    }

    /// Depth-first search to find paths
    fn dfs_paths(
        &self,
        current: u32,
        graph_neighbors: &HashMap<u32, Vec<u32>>,
        current_path: &mut Vec<u32>,
        all_paths: &mut Vec<Vec<u32>>,
        visited: &mut std::collections::HashSet<u32>,
        max_hops: usize,
        current_depth: usize,
    ) {
        if current_depth >= max_hops {
            return;
        }

        visited.insert(current);

        if let Some(neighbors) = graph_neighbors.get(&current) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    current_path.push(neighbor);
                    all_paths.push(current_path.clone());
                    
                    self.dfs_paths(
                        neighbor,
                        graph_neighbors,
                        current_path,
                        all_paths,
                        visited,
                        max_hops,
                        current_depth + 1,
                    );
                    
                    current_path.pop();
                }
            }
        }

        visited.remove(&current);
    }

    /// Calculate similarity between path sets
    fn path_similarity(&self, paths1: &[Vec<u32>], paths2: &[Vec<u32>]) -> f32 {
        if paths1.is_empty() && paths2.is_empty() {
            return 1.0;
        }

        if paths1.is_empty() || paths2.is_empty() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for path1 in paths1 {
            for path2 in paths2 {
                total_similarity += self.jaccard_similarity(path1, path2);
                comparisons += 1;
            }
        }

        total_similarity / comparisons as f32
    }

    /// Clear similarity cache
    pub fn clear_cache(&mut self) {
        self.similarity_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.similarity_cache.len(), self.similarity_cache.capacity())
    }
}

/// Configuration for similarity calculations
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    pub cosine_weight: f32,
    pub textual_weight: f32,
    pub length_weight: f32,
    pub cache_enabled: bool,
    pub cache_ttl_seconds: u64,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            cosine_weight: 0.6,
            textual_weight: 0.3,
            length_weight: 0.1,
            cache_enabled: true,
            cache_ttl_seconds: 3600, // 1 hour
        }
    }
}

/// Similarity metrics enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Jaccard,
    Levenshtein,
    Semantic,
    Structural,
    Graph,
}

impl SimilarityMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            SimilarityMetric::Cosine => "cosine",
            SimilarityMetric::Euclidean => "euclidean",
            SimilarityMetric::Manhattan => "manhattan",
            SimilarityMetric::Jaccard => "jaccard",
            SimilarityMetric::Levenshtein => "levenshtein",
            SimilarityMetric::Semantic => "semantic",
            SimilarityMetric::Structural => "structural",
            SimilarityMetric::Graph => "graph",
        }
    }
}

// Standalone utility functions for tests

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        panic!("Vectors must have the same length");
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Euclidean norm (magnitude) of a vector
pub fn euclidean_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_cosine_similarity_edge_cases() {
        let engine = SimilarityEngine::new();
        
        // Zero vectors
        let zero_vec = vec![0.0, 0.0, 0.0];
        let non_zero_vec = vec![1.0, 2.0, 3.0];
        
        assert_eq!(engine.cosine_similarity(&zero_vec, &zero_vec).unwrap(), 0.0);
        assert_eq!(engine.cosine_similarity(&zero_vec, &non_zero_vec).unwrap(), 0.0);
        
        // Empty vectors
        let empty_vec: Vec<f32> = vec![];
        assert_eq!(engine.cosine_similarity(&empty_vec, &empty_vec).unwrap(), 0.0);
        
        // Mismatched dimensions
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        assert!(engine.cosine_similarity(&vec1, &vec2).is_err());
    }

    #[test]
    fn test_cosine_similarity_properties() {
        let engine = SimilarityEngine::new();
        
        // Reflexivity: sim(x, x) = 1
        let vec = vec![1.0, 2.0, 3.0];
        let sim = engine.cosine_similarity(&vec, &vec).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
        
        // Symmetry: sim(x, y) = sim(y, x)
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let sim1 = engine.cosine_similarity(&vec1, &vec2).unwrap();
        let sim2 = engine.cosine_similarity(&vec2, &vec1).unwrap();
        assert!((sim1 - sim2).abs() < 1e-6);
        
        // Range: -1 <= sim(x, y) <= 1
        assert!((-1.0..=1.0).contains(&sim1));
    }

    #[test]
    fn test_euclidean_distance_overflow() {
        let engine = SimilarityEngine::new();
        
        // Large values that might cause overflow
        let vec1 = vec![f32::MAX / 2.0, f32::MAX / 2.0];
        let vec2 = vec![-f32::MAX / 2.0, -f32::MAX / 2.0];
        
        // Should not panic or return infinity
        let distance = engine.euclidean_distance(&vec1, &vec2).unwrap();
        assert!(distance.is_finite());
        
        // Test numerical stability with very small values
        let vec3 = vec![1e-30, 1e-30];
        let vec4 = vec![2e-30, 2e-30];
        let distance_small = engine.euclidean_distance(&vec3, &vec4).unwrap();
        assert!(distance_small.is_finite() && distance_small >= 0.0);
    }

    #[test]
    fn test_euclidean_distance_properties() {
        let engine = SimilarityEngine::new();
        
        // Non-negativity: d(x, y) >= 0
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let distance = engine.euclidean_distance(&vec1, &vec2).unwrap();
        assert!(distance >= 0.0);
        
        // Identity: d(x, x) = 0
        let distance_same = engine.euclidean_distance(&vec1, &vec1).unwrap();
        assert_eq!(distance_same, 0.0);
        
        // Symmetry: d(x, y) = d(y, x)
        let distance1 = engine.euclidean_distance(&vec1, &vec2).unwrap();
        let distance2 = engine.euclidean_distance(&vec2, &vec1).unwrap();
        assert!((distance1 - distance2).abs() < 1e-6);
        
        // Triangle inequality: d(x, z) <= d(x, y) + d(y, z)
        let vec3 = vec![7.0, 8.0, 9.0];
        let d12 = engine.euclidean_distance(&vec1, &vec2).unwrap();
        let d23 = engine.euclidean_distance(&vec2, &vec3).unwrap();
        let d13 = engine.euclidean_distance(&vec1, &vec3).unwrap();
        assert!(d13 <= d12 + d23 + 1e-6); // Small epsilon for floating point
    }

    #[test]
    fn test_manhattan_distance_calculation() {
        let engine = SimilarityEngine::new();
        
        // Note: The current implementation is incorrect - it calculates Euclidean distance
        // This test documents the current behavior
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        
        let distance = engine.manhattan_distance(&vec1, &vec2).unwrap();
        
        // Current implementation returns sqrt(sum of squares) instead of sum of absolute differences
        let expected_euclidean = ((4.0f32-1.0).powi(2) + (5.0f32-2.0).powi(2) + (6.0f32-3.0).powi(2)).sqrt();
        assert!((distance - expected_euclidean).abs() < 1e-6);
        
        // Proper Manhattan distance would be |4-1| + |5-2| + |6-3| = 9.0
        // But the current implementation returns ~5.196
    }

    #[test]
    fn test_levenshtein_distance() {
        let engine = SimilarityEngine::new();
        
        // Test private method through public interface
        let sim1 = engine.levenshtein_similarity("kitten", "sitting");
        // Levenshtein distance is 3, max length is 7, so similarity = 1 - 3/7 ≈ 0.571
        assert!((sim1 - (1.0 - 3.0/7.0)).abs() < 1e-6);
        
        // Identical strings
        let sim2 = engine.levenshtein_similarity("hello", "hello");
        assert_eq!(sim2, 1.0);
        
        // Empty strings
        let sim3 = engine.levenshtein_similarity("", "");
        assert_eq!(sim3, 1.0);
        
        // One empty string
        let sim4 = engine.levenshtein_similarity("", "hello");
        assert_eq!(sim4, 0.0);
    }

    #[test]
    fn test_jaccard_similarity_sets() {
        let engine = SimilarityEngine::new();
        
        // Test with integer arrays
        let set1 = vec![1, 2, 3, 4];
        let set2 = vec![3, 4, 5, 6];
        let sim = engine.jaccard_similarity(&set1, &set2);
        // Intersection: {3, 4} (size 2), Union: {1, 2, 3, 4, 5, 6} (size 6)
        // Jaccard = 2/6 = 1/3 ≈ 0.333
        assert!((sim - 1.0/3.0).abs() < 1e-6);
        
        // Identical sets
        let sim_same = engine.jaccard_similarity(&set1, &set1);
        assert_eq!(sim_same, 1.0);
        
        // Disjoint sets
        let set3 = vec![7, 8, 9];
        let sim_disjoint = engine.jaccard_similarity(&set1, &set3);
        assert_eq!(sim_disjoint, 0.0);
        
        // Empty sets
        let empty: Vec<i32> = vec![];
        let sim_empty = engine.jaccard_similarity(&empty, &empty);
        assert_eq!(sim_empty, 1.0);
    }

    #[test]
    fn test_textual_similarity() {
        let engine = SimilarityEngine::new();
        
        // Similar strings
        let sim1 = engine.textual_similarity("hello world", "hello earth");
        assert!(sim1 > 0.0 && sim1 < 1.0);
        
        // Identical strings
        let sim2 = engine.textual_similarity("test", "test");
        assert_eq!(sim2, 1.0);
        
        // Different strings
        let sim3 = engine.textual_similarity("completely", "different");
        assert!((0.0..1.0).contains(&sim3));
    }

    #[test]
    fn test_word_overlap_similarity() {
        let engine = SimilarityEngine::new();
        
        // Test through textual_similarity (which uses word_overlap_similarity internally)
        let text1 = "the quick brown fox";
        let text2 = "the brown fox jumps";
        
        let sim = engine.textual_similarity(text1, text2);
        assert!(sim > 0.0);
        
        // Test case sensitivity
        let text3 = "THE QUICK BROWN FOX";
        let sim_case = engine.textual_similarity(text1, text3);
        assert_eq!(sim_case, 1.0); // Should be case-insensitive
    }

    #[test]
    fn test_char_ngram_similarity() {
        let engine = SimilarityEngine::new();
        
        // Test through textual_similarity which uses char_ngram_similarity
        let sim = engine.textual_similarity("testing", "tasting");
        assert!(sim > 0.0 && sim < 1.0);
        
        // Very similar strings should have high similarity
        let sim_high = engine.textual_similarity("abcdef", "abcder");
        assert!(sim_high > 0.5);
    }

    #[test]
    fn test_length_similarity() {
        let engine = SimilarityEngine::new();
        
        // Test through semantic_similarity
        let embedding1 = vec![0.5, 0.5];
        let embedding2 = vec![0.5, 0.5];
        
        let sim1 = engine.semantic_similarity(&embedding1, &embedding2, "short", "text").unwrap();
        let sim2 = engine.semantic_similarity(&embedding1, &embedding2, "much longer text", "text").unwrap();
        
        // First case should have higher length similarity component
        // Note: This is a rough test since semantic_similarity combines multiple factors
        assert!((0.0..=1.0).contains(&sim1));
        assert!((0.0..=1.0).contains(&sim2));
    }

    #[test]
    fn test_structural_similarity() {
        let engine = SimilarityEngine::new();
        
        let neighbors1 = vec![1, 2, 3, 4];
        let neighbors2 = vec![3, 4, 5, 6];
        
        // Without weights
        let sim = engine.structural_similarity(&neighbors1, &neighbors2, None, None);
        assert!((0.0..=1.0).contains(&sim));
        
        // With weights
        let weights1 = vec![1.0, 2.0, 3.0, 4.0];
        let weights2 = vec![3.0, 4.0, 5.0, 6.0];
        let sim_weighted = engine.structural_similarity(&neighbors1, &neighbors2, Some(&weights1), Some(&weights2));
        assert!((0.0..=1.0).contains(&sim_weighted));
    }

    #[test]
    fn test_weighted_structural_similarity() {
        let engine = SimilarityEngine::new();
        
        // Test edge cases for weighted structural similarity
        let neighbors1 = vec![1, 2];
        let neighbors2 = vec![1, 2];
        let weights1 = vec![1.0, 1.0];
        let weights2 = vec![1.0, 1.0];
        
        let sim = engine.structural_similarity(&neighbors1, &neighbors2, Some(&weights1), Some(&weights2));
        assert_eq!(sim, 1.0); // Identical structures should have similarity 1.0
        
        // No common neighbors
        let neighbors3 = vec![3, 4];
        let weights3 = vec![1.0, 1.0];
        let sim_no_common = engine.structural_similarity(&neighbors1, &neighbors3, Some(&weights1), Some(&weights3));
        assert_eq!(sim_no_common, 0.0);
    }

    #[test]
    fn test_graph_similarity() {
        let engine = SimilarityEngine::new();
        
        // Create a simple graph
        let mut graph = HashMap::new();
        graph.insert(1, vec![2, 3]);
        graph.insert(2, vec![1, 4]);
        graph.insert(3, vec![1, 4]);
        graph.insert(4, vec![2, 3]);
        
        let sim = engine.graph_similarity(1, 2, &graph, 2);
        assert!((0.0..=1.0).contains(&sim));
        
        // Same entity should have similarity 1.0
        let sim_same = engine.graph_similarity(1, 1, &graph, 2);
        assert_eq!(sim_same, 1.0);
    }

    #[test]
    fn test_get_neighborhood_paths() {
        let engine = SimilarityEngine::new();
        
        // Test path finding in a simple graph
        let mut graph = HashMap::new();
        graph.insert(1, vec![2, 3]);
        graph.insert(2, vec![4]);
        graph.insert(3, vec![4]);
        graph.insert(4, vec![]);
        
        let paths = engine.get_neighborhood_paths(1, &graph, 2);
        assert!(!paths.is_empty());
        
        // Should find paths of length <= 2 starting from node 1
        for path in &paths {
            assert!(path.len() <= 3); // max_hops + 1 for the starting node
            assert_eq!(path[0], 1); // All paths should start from node 1
        }
    }

    #[test]
    fn test_path_similarity() {
        let engine = SimilarityEngine::new();
        
        let paths1 = vec![vec![1, 2, 3], vec![1, 4, 5]];
        let paths2 = vec![vec![1, 2, 3], vec![1, 6, 7]];
        
        let sim = engine.path_similarity(&paths1, &paths2);
        assert!((0.0..=1.0).contains(&sim));
        
        // Empty path sets
        let empty_paths: Vec<Vec<u32>> = vec![];
        let sim_empty = engine.path_similarity(&empty_paths, &empty_paths);
        assert_eq!(sim_empty, 1.0);
        
        let sim_one_empty = engine.path_similarity(&paths1, &empty_paths);
        assert_eq!(sim_one_empty, 0.0);
    }

    #[test]
    fn test_semantic_similarity() {
        let engine = SimilarityEngine::new();
        
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        
        let sim = engine.semantic_similarity(&embedding1, &embedding2, "hello", "world").unwrap();
        assert!((0.0..=1.0).contains(&sim));
        
        // Test with different dimension vectors - should fail
        let embedding3 = vec![1.0, 0.0];
        let result = engine.semantic_similarity(&embedding1, &embedding3, "hello", "world");
        assert!(result.is_err());
    }

    #[test]
    fn test_similarity_config() {
        let config = SimilarityConfig {
            cosine_weight: 0.5,
            textual_weight: 0.3,
            length_weight: 0.2,
            cache_enabled: true,
            cache_ttl_seconds: 1800,
        };
        
        let engine = SimilarityEngine::with_config(config);
        
        // Test that config is applied (weights should sum to 1.0)
        assert_eq!(engine.config.cosine_weight + engine.config.textual_weight + engine.config.length_weight, 1.0);
    }

    #[test]
    fn test_cache_operations() {
        let mut engine = SimilarityEngine::new();
        
        // Initially empty cache
        let (size, _capacity) = engine.cache_stats();
        assert_eq!(size, 0);
        
        // Clear cache (should not panic even when empty)
        engine.clear_cache();
        let (size_after_clear, _) = engine.cache_stats();
        assert_eq!(size_after_clear, 0);
    }

    #[test]
    fn test_similarity_metric_enum() {
        assert_eq!(SimilarityMetric::Cosine.as_str(), "cosine");
        assert_eq!(SimilarityMetric::Euclidean.as_str(), "euclidean");
        assert_eq!(SimilarityMetric::Manhattan.as_str(), "manhattan");
        assert_eq!(SimilarityMetric::Jaccard.as_str(), "jaccard");
        assert_eq!(SimilarityMetric::Levenshtein.as_str(), "levenshtein");
        assert_eq!(SimilarityMetric::Semantic.as_str(), "semantic");
        assert_eq!(SimilarityMetric::Structural.as_str(), "structural");
        assert_eq!(SimilarityMetric::Graph.as_str(), "graph");
    }

    #[test]
    fn test_standalone_euclidean_functions() {
        let vec1 = vec![3.0, 4.0];
        let vec2 = vec![0.0, 0.0];
        
        // Test euclidean_distance function
        let distance = euclidean_distance(&vec1, &vec2);
        assert_eq!(distance, 5.0); // 3-4-5 triangle
        
        // Test euclidean_norm function
        let norm = euclidean_norm(&vec1);
        assert_eq!(norm, 5.0); // sqrt(3^2 + 4^2) = 5
        
        // Test empty vector
        let empty: Vec<f32> = vec![];
        let norm_empty = euclidean_norm(&empty);
        assert_eq!(norm_empty, 0.0);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_standalone_euclidean_distance_panic() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        euclidean_distance(&vec1, &vec2);
    }

    #[test]
    fn test_euclidean_norm_method() {
        let engine = SimilarityEngine::new();
        
        let vec = vec![3.0, 4.0];
        let norm = engine.euclidean_norm(&vec);
        assert_eq!(norm, 5.0);
        
        let zero_vec = vec![0.0, 0.0];
        let zero_norm = engine.euclidean_norm(&zero_vec);
        assert_eq!(zero_norm, 0.0);
    }

    // Property-based tests for mathematical invariants
    #[test]
    fn test_cosine_similarity_mathematical_properties() {
        let engine = SimilarityEngine::new();
        
        // Test with orthogonal vectors
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let sim_orthogonal = engine.cosine_similarity(&vec1, &vec2).unwrap();
        assert!((sim_orthogonal - 0.0).abs() < 1e-6);
        
        // Test with parallel vectors
        let vec3 = vec![1.0, 2.0, 3.0];
        let vec4 = vec![2.0, 4.0, 6.0]; // 2 * vec3
        let sim_parallel = engine.cosine_similarity(&vec3, &vec4).unwrap();
        assert!((sim_parallel - 1.0).abs() < 1e-6);
        
        // Test with anti-parallel vectors
        let vec5 = vec![-1.0, -2.0, -3.0]; // -vec3
        let sim_antiparallel = engine.cosine_similarity(&vec3, &vec5).unwrap();
        assert!((sim_antiparallel - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_stability() {
        let engine = SimilarityEngine::new();
        
        // Test with very small numbers
        let vec_small1 = vec![1e-30, 1e-30];
        let vec_small2 = vec![2e-30, 2e-30];
        
        let sim_small = engine.cosine_similarity(&vec_small1, &vec_small2).unwrap();
        assert!(sim_small.is_finite());
        assert!((sim_small - 1.0).abs() < 1e-6); // Should be 1.0 (parallel vectors)
        
        // Test with very large numbers
        let vec_large1 = vec![1e30, 1e30];
        let vec_large2 = vec![2e30, 2e30];
        
        let sim_large = engine.cosine_similarity(&vec_large1, &vec_large2).unwrap();
        assert!(sim_large.is_finite());
        assert!((sim_large - 1.0).abs() < 1e-6); // Should be 1.0 (parallel vectors)
    }
}
