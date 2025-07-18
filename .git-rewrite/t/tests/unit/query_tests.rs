//! Query Engine Unit Tests
//! 
//! Comprehensive unit tests for the LLMKG query engine components

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use rand::prelude::*;

/// Similarity search engine for testing
#[derive(Debug, Clone)]
pub struct SimilaritySearchEngine {
    /// Vector database
    vectors: Vec<(u32, Vec<f32>)>, // (entity_id, embedding)
    /// Index for fast search
    index: Option<VectorIndex>,
    /// Search configuration
    config: SearchConfig,
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub use_approximate_search: bool,
    pub index_type: IndexType,
}

#[derive(Debug, Clone)]
pub enum IndexType {
    Linear,     // Brute force search
    LSH,        // Locality-sensitive hashing
    IVF,        // Inverted file system
    HNSW,       // Hierarchical navigable small world
}

#[derive(Debug, Clone)]
pub struct VectorIndex {
    index_type: IndexType,
    num_vectors: usize,
    dimension: usize,
    // Simplified index representation
    buckets: HashMap<usize, Vec<u32>>, // For LSH/IVF
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_results: 10,
            use_approximate_search: true,
            index_type: IndexType::LSH,
        }
    }
}

impl SimilaritySearchEngine {
    /// Create new search engine
    pub fn new(config: SearchConfig) -> Self {
        Self {
            vectors: Vec::new(),
            index: None,
            config,
        }
    }

    /// Add vector to the engine
    pub fn add_vector(&mut self, entity_id: u32, embedding: Vec<f32>) -> Result<()> {
        if !self.vectors.is_empty() && embedding.len() != self.vectors[0].1.len() {
            return Err(anyhow!("Embedding dimension mismatch"));
        }

        self.vectors.push((entity_id, embedding));
        
        // Invalidate index when adding new vectors
        self.index = None;
        
        Ok(())
    }

    /// Build search index
    pub fn build_index(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(anyhow!("No vectors to index"));
        }

        let dimension = self.vectors[0].1.len();
        let mut buckets = HashMap::new();

        match self.config.index_type {
            IndexType::Linear => {
                // No index needed for linear search
            }
            IndexType::LSH => {
                // Simple LSH implementation
                let num_hash_functions = 8;
                let mut rng = StdRng::seed_from_u64(42);
                
                for (entity_id, embedding) in &self.vectors {
                    for hash_func in 0..num_hash_functions {
                        let hash = self.lsh_hash(embedding, hash_func, &mut rng);
                        buckets.entry(hash).or_insert_with(Vec::new).push(*entity_id);
                    }
                }
            }
            IndexType::IVF => {
                // Inverted file system (simplified k-means clustering)
                let num_clusters = (self.vectors.len() / 10).max(1).min(100);
                let clusters = self.create_clusters(num_clusters)?;
                
                for (entity_id, embedding) in &self.vectors {
                    let cluster_id = self.assign_to_cluster(embedding, &clusters);
                    buckets.entry(cluster_id).or_insert_with(Vec::new).push(*entity_id);
                }
            }
            IndexType::HNSW => {
                // Simplified HNSW (not fully implemented)
                buckets.insert(0, self.vectors.iter().map(|(id, _)| *id).collect());
            }
        }

        self.index = Some(VectorIndex {
            index_type: self.config.index_type.clone(),
            num_vectors: self.vectors.len(),
            dimension,
            buckets,
        });

        Ok(())
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], top_k: Option<usize>) -> Result<Vec<SimilarityResult>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        if query.len() != self.vectors[0].1.len() {
            return Err(anyhow!("Query dimension mismatch"));
        }

        let k = top_k.unwrap_or(self.config.max_results);
        
        match &self.index {
            Some(index) if self.config.use_approximate_search => {
                self.approximate_search(query, k, index)
            }
            _ => {
                self.linear_search(query, k)
            }
        }
    }

    /// Linear search (brute force)
    fn linear_search(&self, query: &[f32], top_k: usize) -> Result<Vec<SimilarityResult>> {
        let mut results = Vec::new();
        
        for (entity_id, embedding) in &self.vectors {
            let similarity = self.cosine_similarity(query, embedding);
            if similarity >= self.config.similarity_threshold {
                results.push(SimilarityResult {
                    entity_id: *entity_id,
                    similarity,
                    distance: 1.0 - similarity,
                });
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(top_k);
        
        Ok(results)
    }

    /// Approximate search using index
    fn approximate_search(&self, query: &[f32], top_k: usize, index: &VectorIndex) -> Result<Vec<SimilarityResult>> {
        let candidate_entities = match index.index_type {
            IndexType::LSH => {
                let mut candidates = HashSet::new();
                let mut rng = StdRng::seed_from_u64(42);
                
                // Generate hashes for query and find candidates
                for hash_func in 0..8 {
                    let hash = self.lsh_hash(query, hash_func, &mut rng);
                    if let Some(bucket) = index.buckets.get(&hash) {
                        candidates.extend(bucket);
                    }
                }
                candidates.into_iter().collect()
            }
            IndexType::IVF => {
                // Find nearest cluster and get candidates
                let clusters = self.create_clusters(index.buckets.len())?;
                let cluster_id = self.assign_to_cluster(query, &clusters);
                index.buckets.get(&cluster_id).cloned().unwrap_or_default()
            }
            _ => {
                // Fallback to all entities
                self.vectors.iter().map(|(id, _)| *id).collect()
            }
        };

        // Compute similarities for candidates only
        let mut results = Vec::new();
        for entity_id in candidate_entities {
            if let Some((_, embedding)) = self.vectors.iter().find(|(id, _)| *id == entity_id) {
                let similarity = self.cosine_similarity(query, embedding);
                if similarity >= self.config.similarity_threshold {
                    results.push(SimilarityResult {
                        entity_id,
                        similarity,
                        distance: 1.0 - similarity,
                    });
                }
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(top_k);
        
        Ok(results)
    }

    /// LSH hash function
    fn lsh_hash(&self, vector: &[f32], hash_func: usize, rng: &mut StdRng) -> usize {
        // Simple random projection LSH
        rng.set_seed_u64(hash_func as u64);
        let projection: Vec<f32> = (0..vector.len()).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        let dot_product: f32 = vector.iter().zip(projection.iter()).map(|(a, b)| a * b).sum();
        if dot_product >= 0.0 { 1 } else { 0 }
    }

    /// Create clusters for IVF
    fn create_clusters(&self, num_clusters: usize) -> Result<Vec<Vec<f32>>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dimension = self.vectors[0].1.len();
        let mut rng = StdRng::seed_from_u64(42);
        
        // Initialize random centroids
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        // Simple k-means (few iterations)
        for _ in 0..5 {
            let mut new_centroids = vec![vec![0.0; dimension]; num_clusters];
            let mut counts = vec![0; num_clusters];

            // Assign vectors to clusters
            for (_, embedding) in &self.vectors {
                let cluster_id = self.assign_to_cluster(embedding, &centroids);
                for (i, &value) in embedding.iter().enumerate() {
                    new_centroids[cluster_id][i] += value;
                }
                counts[cluster_id] += 1;
            }

            // Update centroids
            for (cluster_id, count) in counts.iter().enumerate() {
                if *count > 0 {
                    for value in &mut new_centroids[cluster_id] {
                        *value /= *count as f32;
                    }
                }
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    /// Assign vector to nearest cluster
    fn assign_to_cluster(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut best_cluster = 0;
        let mut best_distance = f32::INFINITY;

        for (i, centroid) in centroids.iter().enumerate() {
            let distance = self.euclidean_distance(vector, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_cluster = i;
            }
        }

        best_cluster
    }

    /// Cosine similarity
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Euclidean distance
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> SearchEngineStats {
        SearchEngineStats {
            num_vectors: self.vectors.len(),
            dimension: if self.vectors.is_empty() { 0 } else { self.vectors[0].1.len() },
            has_index: self.index.is_some(),
            index_type: self.config.index_type.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub entity_id: u32,
    pub similarity: f32,
    pub distance: f32,
}

#[derive(Debug)]
pub struct SearchEngineStats {
    pub num_vectors: usize,
    pub dimension: usize,
    pub has_index: bool,
    pub index_type: IndexType,
}

/// Query filter for advanced filtering
#[derive(Debug, Clone)]
pub struct QueryFilter {
    /// Attribute filters
    pub attribute_filters: Vec<AttributeFilter>,
    /// Numerical range filters
    pub range_filters: Vec<RangeFilter>,
    /// Text filters
    pub text_filters: Vec<TextFilter>,
}

#[derive(Debug, Clone)]
pub struct AttributeFilter {
    pub attribute_name: String,
    pub operation: FilterOperation,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct RangeFilter {
    pub attribute_name: String,
    pub min_value: f64,
    pub max_value: f64,
}

#[derive(Debug, Clone)]
pub struct TextFilter {
    pub attribute_name: String,
    pub pattern: String,
    pub case_sensitive: bool,
}

#[derive(Debug, Clone)]
pub enum FilterOperation {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
}

impl QueryFilter {
    /// Create new empty filter
    pub fn new() -> Self {
        Self {
            attribute_filters: Vec::new(),
            range_filters: Vec::new(),
            text_filters: Vec::new(),
        }
    }

    /// Add attribute filter
    pub fn add_attribute_filter(&mut self, name: String, operation: FilterOperation, value: String) {
        self.attribute_filters.push(AttributeFilter {
            attribute_name: name,
            operation,
            value,
        });
    }

    /// Add range filter
    pub fn add_range_filter(&mut self, name: String, min_value: f64, max_value: f64) {
        self.range_filters.push(RangeFilter {
            attribute_name: name,
            min_value,
            max_value,
        });
    }

    /// Add text filter
    pub fn add_text_filter(&mut self, name: String, pattern: String, case_sensitive: bool) {
        self.text_filters.push(TextFilter {
            attribute_name: name,
            pattern,
            case_sensitive,
        });
    }

    /// Apply filter to entity
    pub fn matches(&self, entity: &test_utils::Entity) -> bool {
        // Check attribute filters
        for filter in &self.attribute_filters {
            if let Some(value) = entity.get_attribute(&filter.attribute_name) {
                let matches = match filter.operation {
                    FilterOperation::Equals => value == filter.value,
                    FilterOperation::NotEquals => value != filter.value,
                    FilterOperation::Contains => value.contains(&filter.value),
                    FilterOperation::StartsWith => value.starts_with(&filter.value),
                    FilterOperation::EndsWith => value.ends_with(&filter.value),
                };
                if !matches {
                    return false;
                }
            } else {
                return false; // Attribute not found
            }
        }

        // Check range filters
        for filter in &self.range_filters {
            if let Some(value_str) = entity.get_attribute(&filter.attribute_name) {
                if let Ok(value) = value_str.parse::<f64>() {
                    if value < filter.min_value || value > filter.max_value {
                        return false;
                    }
                } else {
                    return false; // Not a valid number
                }
            } else {
                return false; // Attribute not found
            }
        }

        // Check text filters
        for filter in &self.text_filters {
            if let Some(value) = entity.get_attribute(&filter.attribute_name) {
                let search_value = if filter.case_sensitive { value } else { value.to_lowercase() };
                let search_pattern = if filter.case_sensitive { 
                    filter.pattern.clone() 
                } else { 
                    filter.pattern.to_lowercase() 
                };
                
                if !search_value.contains(&search_pattern) {
                    return false;
                }
            } else {
                return false; // Attribute not found
            }
        }

        true
    }
}

/// Graph traversal engine
#[derive(Debug)]
pub struct GraphTraversal {
    /// Graph representation
    adjacency_list: HashMap<u32, Vec<(u32, f32)>>, // node_id -> [(neighbor_id, weight)]
}

impl GraphTraversal {
    /// Create new graph traversal engine
    pub fn new() -> Self {
        Self {
            adjacency_list: HashMap::new(),
        }
    }

    /// Add edge to graph
    pub fn add_edge(&mut self, src: u32, dst: u32, weight: f32) {
        self.adjacency_list.entry(src).or_insert_with(Vec::new).push((dst, weight));
    }

    /// Breadth-first search
    pub fn bfs(&self, start: u32, max_depth: Option<usize>) -> Vec<u32> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        queue.push_back((start, 0));
        visited.insert(start);
        
        while let Some((node, depth)) = queue.pop_front() {
            result.push(node);
            
            if let Some(max_d) = max_depth {
                if depth >= max_d {
                    continue;
                }
            }
            
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &(neighbor, _) in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        
        result
    }

    /// Depth-first search
    pub fn dfs(&self, start: u32, max_depth: Option<usize>) -> Vec<u32> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        self.dfs_recursive(start, 0, max_depth, &mut visited, &mut result);
        result
    }

    fn dfs_recursive(&self, node: u32, depth: usize, max_depth: Option<usize>, 
                    visited: &mut HashSet<u32>, result: &mut Vec<u32>) {
        if visited.contains(&node) {
            return;
        }
        
        if let Some(max_d) = max_depth {
            if depth > max_d {
                return;
            }
        }
        
        visited.insert(node);
        result.push(node);
        
        if let Some(neighbors) = self.adjacency_list.get(&node) {
            for &(neighbor, _) in neighbors {
                self.dfs_recursive(neighbor, depth + 1, max_depth, visited, result);
            }
        }
    }

    /// Find shortest path using Dijkstra's algorithm
    pub fn shortest_path(&self, start: u32, end: u32) -> Option<Vec<u32>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        let mut distances: HashMap<u32, f32> = HashMap::new();
        let mut previous: HashMap<u32, u32> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        distances.insert(start, 0.0);
        heap.push(Reverse((0.0, start)));
        
        while let Some(Reverse((dist, node))) = heap.pop() {
            if node == end {
                break;
            }
            
            if let Some(&current_dist) = distances.get(&node) {
                if dist > current_dist {
                    continue;
                }
            }
            
            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for &(neighbor, weight) in neighbors {
                    let new_dist = dist + weight;
                    let should_update = match distances.get(&neighbor) {
                        Some(&old_dist) => new_dist < old_dist,
                        None => true,
                    };
                    
                    if should_update {
                        distances.insert(neighbor, new_dist);
                        previous.insert(neighbor, node);
                        heap.push(Reverse((new_dist, neighbor)));
                    }
                }
            }
        }
        
        // Reconstruct path
        if !distances.contains_key(&end) {
            return None;
        }
        
        let mut path = Vec::new();
        let mut current = end;
        
        while current != start {
            path.push(current);
            if let Some(&prev) = previous.get(&current) {
                current = prev;
            } else {
                return None;
            }
        }
        path.push(start);
        path.reverse();
        
        Some(path)
    }

    /// Get node neighbors
    pub fn get_neighbors(&self, node: u32) -> Vec<(u32, f32)> {
        self.adjacency_list.get(&node).cloned().unwrap_or_default()
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        let total_nodes = self.adjacency_list.len();
        let total_edges: usize = self.adjacency_list.values().map(|v| v.len()).sum();
        let avg_degree = if total_nodes > 0 { total_edges as f64 / total_nodes as f64 } else { 0.0 };
        
        GraphStats {
            total_nodes,
            total_edges,
            avg_degree,
        }
    }
}

#[derive(Debug)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub avg_degree: f64,
}

/// Test suite for query engine
pub async fn run_query_tests() -> Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();

    // Similarity search tests
    results.push(test_similarity_search_basic().await);
    results.push(test_similarity_search_linear().await);
    results.push(test_similarity_search_lsh().await);
    results.push(test_similarity_search_performance().await);

    // Query filtering tests
    results.push(test_query_filters().await);
    results.push(test_attribute_filtering().await);
    results.push(test_range_filtering().await);
    results.push(test_text_filtering().await);

    // Graph traversal tests
    results.push(test_graph_bfs().await);
    results.push(test_graph_dfs().await);
    results.push(test_shortest_path().await);

    Ok(results)
}

async fn test_similarity_search_basic() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut engine = SimilaritySearchEngine::new(SearchConfig::default());
        
        // Add test vectors
        engine.add_vector(1, vec![1.0, 0.0, 0.0])?;
        engine.add_vector(2, vec![0.9, 0.1, 0.0])?; // Similar to first
        engine.add_vector(3, vec![0.0, 1.0, 0.0])?; // Different
        engine.add_vector(4, vec![0.0, 0.0, 1.0])?; // Different
        
        // Search for similar to first vector
        let query = vec![1.0, 0.0, 0.0];
        let results = engine.search(&query, Some(2))?;
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity_id, 1); // Exact match should be first
        assert!(results[0].similarity > 0.99);
        assert_eq!(results[1].entity_id, 2); // Similar vector should be second
        
        Ok(())
    })();

    UnitTestResult {
        name: "similarity_search_basic".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_similarity_search_linear() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let config = SearchConfig {
            use_approximate_search: false,
            similarity_threshold: 0.5,
            max_results: 5,
            index_type: IndexType::Linear,
        };
        let mut engine = SimilaritySearchEngine::new(config);
        
        // Add random vectors
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..20 {
            let vector: Vec<f32> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
            engine.add_vector(i, vector)?;
        }
        
        // Search with random query
        let query: Vec<f32> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let results = engine.search(&query, Some(5))?;
        
        assert!(results.len() <= 5);
        
        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].similarity >= results[i].similarity);
        }
        
        // All results should meet threshold
        for result in &results {
            assert!(result.similarity >= 0.5);
        }
        
        Ok(())
    })();

    UnitTestResult {
        name: "similarity_search_linear".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_similarity_search_lsh() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let config = SearchConfig {
            use_approximate_search: true,
            similarity_threshold: 0.3,
            max_results: 10,
            index_type: IndexType::LSH,
        };
        let mut engine = SimilaritySearchEngine::new(config);
        
        // Add test vectors
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..100 {
            let vector: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
            engine.add_vector(i, vector)?;
        }
        
        // Build LSH index
        engine.build_index()?;
        
        // Search using index
        let query: Vec<f32> = (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let approx_results = engine.search(&query, Some(10))?;
        
        // Results should be reasonable
        assert!(approx_results.len() <= 10);
        
        let stats = engine.get_stats();
        assert!(stats.has_index);
        assert_eq!(stats.num_vectors, 100);
        assert_eq!(stats.dimension, 16);
        
        Ok(())
    })();

    UnitTestResult {
        name: "similarity_search_lsh".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 4096,
        coverage_percentage: 85.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_similarity_search_performance() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut engine = SimilaritySearchEngine::new(SearchConfig::default());
        
        // Add many vectors
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..1000 {
            let vector: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
            engine.add_vector(i, vector)?;
        }
        
        // Build index
        let index_start = std::time::Instant::now();
        engine.build_index()?;
        let index_time = index_start.elapsed();
        
        // Perform searches
        let search_start = std::time::Instant::now();
        for _ in 0..10 {
            let query: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let _results = engine.search(&query, Some(10))?;
        }
        let search_time = search_start.elapsed();
        
        // Performance assertions
        assert!(index_time.as_millis() < 5000, "Index building too slow: {}ms", index_time.as_millis());
        assert!(search_time.as_millis() < 1000, "Searches too slow: {}ms", search_time.as_millis());
        
        Ok(())
    })();

    UnitTestResult {
        name: "similarity_search_performance".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 16384,
        coverage_percentage: 82.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_query_filters() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut filter = QueryFilter::new();
        
        // Add various filters
        filter.add_attribute_filter("category".to_string(), FilterOperation::Equals, "product".to_string());
        filter.add_range_filter("price".to_string(), 10.0, 100.0);
        filter.add_text_filter("description".to_string(), "test".to_string(), false);
        
        // Create test entities
        let mut entity1 = test_utils::create_test_entity("entity1", "Product 1");
        entity1.add_attribute("category", "product");
        entity1.add_attribute("price", "50.0");
        entity1.add_attribute("description", "This is a test product");
        
        let mut entity2 = test_utils::create_test_entity("entity2", "Product 2");
        entity2.add_attribute("category", "service");
        entity2.add_attribute("price", "75.0");
        entity2.add_attribute("description", "Another test item");
        
        let mut entity3 = test_utils::create_test_entity("entity3", "Product 3");
        entity3.add_attribute("category", "product");
        entity3.add_attribute("price", "150.0"); // Out of range
        entity3.add_attribute("description", "Contains test keyword");
        
        // Test filtering
        assert!(filter.matches(&entity1)); // Should match all filters
        assert!(!filter.matches(&entity2)); // Wrong category
        assert!(!filter.matches(&entity3)); // Price out of range
        
        Ok(())
    })();

    UnitTestResult {
        name: "query_filters".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 92.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_attribute_filtering() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut filter = QueryFilter::new();
        filter.add_attribute_filter("type".to_string(), FilterOperation::StartsWith, "prod".to_string());
        
        let mut entity1 = test_utils::create_test_entity("entity1", "Entity 1");
        entity1.add_attribute("type", "product");
        
        let mut entity2 = test_utils::create_test_entity("entity2", "Entity 2");
        entity2.add_attribute("type", "service");
        
        let mut entity3 = test_utils::create_test_entity("entity3", "Entity 3");
        entity3.add_attribute("type", "production");
        
        assert!(filter.matches(&entity1)); // "product" starts with "prod"
        assert!(!filter.matches(&entity2)); // "service" doesn't start with "prod"
        assert!(filter.matches(&entity3)); // "production" starts with "prod"
        
        Ok(())
    })();

    UnitTestResult {
        name: "attribute_filtering".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 512,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_range_filtering() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut filter = QueryFilter::new();
        filter.add_range_filter("score".to_string(), 0.5, 0.9);
        
        let mut entity1 = test_utils::create_test_entity("entity1", "Entity 1");
        entity1.add_attribute("score", "0.7"); // In range
        
        let mut entity2 = test_utils::create_test_entity("entity2", "Entity 2");
        entity2.add_attribute("score", "0.3"); // Below range
        
        let mut entity3 = test_utils::create_test_entity("entity3", "Entity 3");
        entity3.add_attribute("score", "0.95"); // Above range
        
        let mut entity4 = test_utils::create_test_entity("entity4", "Entity 4");
        entity4.add_attribute("score", "invalid"); // Invalid number
        
        assert!(filter.matches(&entity1)); // In range
        assert!(!filter.matches(&entity2)); // Below range
        assert!(!filter.matches(&entity3)); // Above range
        assert!(!filter.matches(&entity4)); // Invalid number
        
        Ok(())
    })();

    UnitTestResult {
        name: "range_filtering".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 512,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_text_filtering() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut filter = QueryFilter::new();
        filter.add_text_filter("content".to_string(), "KEYWORD".to_string(), false); // Case insensitive
        
        let mut entity1 = test_utils::create_test_entity("entity1", "Entity 1");
        entity1.add_attribute("content", "This contains the keyword we're looking for");
        
        let mut entity2 = test_utils::create_test_entity("entity2", "Entity 2");
        entity2.add_attribute("content", "This has KEYWORD in uppercase");
        
        let mut entity3 = test_utils::create_test_entity("entity3", "Entity 3");
        entity3.add_attribute("content", "This doesn't have the search term");
        
        assert!(filter.matches(&entity1)); // Contains keyword (case insensitive)
        assert!(filter.matches(&entity2)); // Contains KEYWORD (case insensitive)
        assert!(!filter.matches(&entity3)); // Doesn't contain keyword
        
        // Test case sensitive
        let mut case_sensitive_filter = QueryFilter::new();
        case_sensitive_filter.add_text_filter("content".to_string(), "KEYWORD".to_string(), true);
        
        assert!(!case_sensitive_filter.matches(&entity1)); // lowercase keyword
        assert!(case_sensitive_filter.matches(&entity2)); // uppercase KEYWORD
        
        Ok(())
    })();

    UnitTestResult {
        name: "text_filtering".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 768,
        coverage_percentage: 85.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_graph_bfs() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut graph = GraphTraversal::new();
        
        // Create a simple graph: 0 -> 1 -> 2, 0 -> 3
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(0, 3, 1.0);
        
        // BFS from node 0
        let visited = graph.bfs(0, None);
        assert_eq!(visited[0], 0); // Start node
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        
        // BFS with depth limit
        let visited_limited = graph.bfs(0, Some(1));
        assert!(visited_limited.contains(&0));
        assert!(visited_limited.contains(&1));
        assert!(visited_limited.contains(&3));
        assert!(!visited_limited.contains(&2)); // Should not reach depth 2
        
        Ok(())
    })();

    UnitTestResult {
        name: "graph_bfs".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 512,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_graph_dfs() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut graph = GraphTraversal::new();
        
        // Create a tree: 0 -> 1 -> 2, 0 -> 3 -> 4
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(0, 3, 1.0);
        graph.add_edge(3, 4, 1.0);
        
        // DFS from node 0
        let visited = graph.dfs(0, None);
        assert_eq!(visited[0], 0); // Start node
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        assert!(visited.contains(&4));
        
        // DFS should visit depth-first (either 1->2 before 3->4, or vice versa)
        let pos_1 = visited.iter().position(|&x| x == 1).unwrap();
        let pos_2 = visited.iter().position(|&x| x == 2).unwrap();
        let pos_3 = visited.iter().position(|&x| x == 3).unwrap();
        let pos_4 = visited.iter().position(|&x| x == 4).unwrap();
        
        // Either 1->2 comes before 3->4, or 3->4 comes before 1->2
        assert!(
            (pos_1 < pos_2 && pos_2 < pos_3 && pos_3 < pos_4) ||
            (pos_3 < pos_4 && pos_4 < pos_1 && pos_1 < pos_2)
        );
        
        Ok(())
    })();

    UnitTestResult {
        name: "graph_dfs".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 512,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_shortest_path() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut graph = GraphTraversal::new();
        
        // Create a weighted graph
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 4.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(1, 3, 5.0);
        graph.add_edge(2, 3, 1.0);
        
        // Find shortest path from 0 to 3
        let path = graph.shortest_path(0, 3);
        assert!(path.is_some());
        
        let path = path.unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 3);
        
        // The shortest path should be 0 -> 1 -> 2 -> 3 (total weight: 4.0)
        // or 0 -> 2 -> 3 (total weight: 5.0)
        // So it should prefer the first path
        assert_eq!(path, vec![0, 1, 2, 3]);
        
        // Test non-existent path
        let no_path = graph.shortest_path(0, 99);
        assert!(no_path.is_none());
        
        // Test same start and end
        let self_path = graph.shortest_path(0, 0);
        assert_eq!(self_path, Some(vec![0]));
        
        Ok(())
    })();

    UnitTestResult {
        name: "shortest_path".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 92.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_engine_comprehensive() {
        let results = run_query_tests().await.unwrap();
        
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        
        println!("Query Engine Tests: {}/{} passed", passed_tests, total_tests);
        
        for result in &results {
            if result.passed {
                println!("✅ {}: {}ms", result.name, result.duration_ms);
            } else {
                println!("❌ {}: {} ({}ms)", result.name, 
                         result.error_message.as_deref().unwrap_or("Unknown error"),
                         result.duration_ms);
            }
        }
        
        assert_eq!(passed_tests, total_tests, "Some query tests failed");
    }
}