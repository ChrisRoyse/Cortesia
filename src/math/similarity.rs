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
