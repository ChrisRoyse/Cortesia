//! Query Pattern Generation
//! 
//! Provides generation of comprehensive query patterns for testing all aspects of the LLMKG system.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use crate::data_generation::graph_topologies::{TestGraph, TestEntity, TestEdge};
use std::collections::{HashMap, HashSet, VecDeque};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Graph traversal query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalQuery {
    pub query_id: u64,
    pub start_entity: u32,
    pub target_entity: Option<u32>,
    pub max_depth: u32,
    pub relationship_filter: Option<String>,
    pub expected_result_count: u64,
    pub expected_entities: Vec<u32>,
    pub expected_path: Option<Vec<u32>>,
    pub expected_distance: Option<u32>,
    pub traversal_type: TraversalType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraversalType {
    ShortestPath,
    DepthFirstSearch,
    BreadthFirstSearch,
    RandomWalk,
    BiDirectional,
}

/// RAG (Retrieval-Augmented Generation) query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQuery {
    pub query_id: u64,
    pub query_concept: u32,
    pub query_embedding: Vec<f32>,
    pub max_context_size: u32,
    pub max_depth: u32,
    pub similarity_threshold: f32,
    pub expected_context: Vec<u32>,
    pub expected_relevance_scores: Vec<f32>,
    pub expected_context_quality: f32,
    pub context_diversity_target: f32,
    pub rag_strategy: RagStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RagStrategy {
    EmbeddingOnly,       // Pure embedding similarity
    GraphOnly,          // Pure graph traversal
    Hybrid,             // Combined embedding + graph
    Adaptive,           // Dynamic strategy selection
}

/// Similarity search query specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityQuery {
    pub query_id: u64,
    pub query_entity: u32,
    pub query_embedding: Vec<f32>,
    pub k: usize,
    pub similarity_metric: SimilarityMetric,
    pub expected_neighbors: Vec<(u32, f32)>,
    pub expected_distances: Vec<f32>,
    pub expected_recall: f32,
    pub expected_precision: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Euclidean,
    Cosine,
    DotProduct,
    Manhattan,
    Hamming,
}

/// Complex multi-hop query combining multiple operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexQuery {
    pub query_id: u64,
    pub operations: Vec<QueryOperation>,
    pub expected_result: QueryResult,
    pub expected_performance: PerformanceExpectation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperation {
    SimilaritySearch { entity: u32, k: usize },
    GraphTraversal { start: u32, max_depth: u32 },
    Filter { attribute: String, value: String },
    Aggregate { operation: AggregateOp, field: String },
    Join { entities: Vec<u32>, relationship: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateOp {
    Count,
    Sum,
    Average,
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub entities: Vec<u32>,
    pub relationships: Vec<(u32, u32, String)>,
    pub aggregated_values: HashMap<String, f64>,
    pub result_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectation {
    pub max_latency_ms: f64,
    pub max_memory_mb: f64,
    pub expected_io_operations: u64,
    pub cache_hit_ratio: f32,
}

/// Query pattern generator
pub struct QueryPatternGenerator {
    graph: TestGraph,
    embeddings: HashMap<u32, Vec<f32>>,
    rng: DeterministicRng,
}

impl QueryPatternGenerator {
    /// Create a new query pattern generator
    pub fn new(seed: u64, graph: TestGraph, embeddings: HashMap<u32, Vec<f32>>) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("query_pattern_generator".to_string());
        
        Self { graph, embeddings, rng }
    }

    /// Generate traversal queries for testing graph navigation
    pub fn generate_traversal_queries(&mut self, count: u32) -> Result<Vec<TraversalQuery>> {
        if self.graph.entities.is_empty() {
            return Err(anyhow!("Graph must have entities to generate traversal queries"));
        }

        let mut queries = Vec::new();
        let entities = &self.graph.entities;

        for i in 0..count {
            let query_type = match i % 5 {
                0 => TraversalType::ShortestPath,
                1 => TraversalType::DepthFirstSearch,
                2 => TraversalType::BreadthFirstSearch,
                3 => TraversalType::RandomWalk,
                _ => TraversalType::BiDirectional,
            };

            let start_entity = entities[self.rng.next_usize(entities.len())].id;
            
            let query = match query_type {
                TraversalType::ShortestPath => {
                    let target_entity = entities[self.rng.next_usize(entities.len())].id;
                    let (path, distance) = self.compute_shortest_path(start_entity, target_entity)?;
                    
                    TraversalQuery {
                        query_id: i as u64,
                        start_entity,
                        target_entity: Some(target_entity),
                        max_depth: distance.unwrap_or(u32::MAX),
                        relationship_filter: None,
                        expected_result_count: if path.is_some() { 1 } else { 0 },
                        expected_entities: path.clone().unwrap_or_default(),
                        expected_path: path,
                        expected_distance: distance,
                        traversal_type: query_type,
                    }
                },
                
                TraversalType::BreadthFirstSearch => {
                    let max_depth = self.rng.range_i32(1, 4) as u32;
                    let reachable_entities = self.compute_reachable_entities(start_entity, max_depth)?;
                    
                    TraversalQuery {
                        query_id: i as u64,
                        start_entity,
                        target_entity: None,
                        max_depth,
                        relationship_filter: None,
                        expected_result_count: reachable_entities.len() as u64,
                        expected_entities: reachable_entities,
                        expected_path: None,
                        expected_distance: None,
                        traversal_type: query_type,
                    }
                },
                
                TraversalType::DepthFirstSearch => {
                    let max_depth = self.rng.range_i32(1, 3) as u32;
                    let dfs_entities = self.compute_dfs_entities(start_entity, max_depth)?;
                    
                    TraversalQuery {
                        query_id: i as u64,
                        start_entity,
                        target_entity: None,
                        max_depth,
                        relationship_filter: None,
                        expected_result_count: dfs_entities.len() as u64,
                        expected_entities: dfs_entities,
                        expected_path: None,
                        expected_distance: None,
                        traversal_type: query_type,
                    }
                },
                
                TraversalType::RandomWalk => {
                    let walk_length = self.rng.range_i32(5, 20) as u32;
                    let random_walk = self.compute_random_walk(start_entity, walk_length)?;
                    
                    TraversalQuery {
                        query_id: i as u64,
                        start_entity,
                        target_entity: None,
                        max_depth: walk_length,
                        relationship_filter: None,
                        expected_result_count: random_walk.len() as u64,
                        expected_entities: random_walk,
                        expected_path: None,
                        expected_distance: None,
                        traversal_type: query_type,
                    }
                },
                
                TraversalType::BiDirectional => {
                    let target_entity = entities[self.rng.next_usize(entities.len())].id;
                    let (path, distance) = self.compute_bidirectional_path(start_entity, target_entity)?;
                    
                    TraversalQuery {
                        query_id: i as u64,
                        start_entity,
                        target_entity: Some(target_entity),
                        max_depth: distance.unwrap_or(u32::MAX),
                        relationship_filter: None,
                        expected_result_count: if path.is_some() { 1 } else { 0 },
                        expected_entities: path.clone().unwrap_or_default(),
                        expected_path: path,
                        expected_distance: distance,
                        traversal_type: query_type,
                    }
                },
            };

            queries.push(query);
        }

        Ok(queries)
    }

    /// Generate RAG queries for testing context retrieval
    pub fn generate_rag_queries(&mut self, count: u32) -> Result<Vec<RagQuery>> {
        if self.embeddings.is_empty() {
            return Err(anyhow!("Embeddings required for RAG query generation"));
        }

        let mut queries = Vec::new();
        let entity_ids: Vec<u32> = self.embeddings.keys().cloned().collect();

        for i in 0..count {
            let query_concept = entity_ids[self.rng.next_usize(entity_ids.len())];
            let query_embedding = self.embeddings[&query_concept].clone();
            
            let strategy = match i % 4 {
                0 => RagStrategy::EmbeddingOnly,
                1 => RagStrategy::GraphOnly,
                2 => RagStrategy::Hybrid,
                _ => RagStrategy::Adaptive,
            };

            let max_context_size = self.rng.range_i32(5, 20) as u32;
            let max_depth = self.rng.range_i32(1, 3) as u32;
            let similarity_threshold = self.rng.range_f64(0.3, 0.8) as f32;

            let context_entities = self.compute_rag_context(
                query_concept,
                &query_embedding,
                max_context_size,
                max_depth,
                &strategy,
            )?;

            let relevance_scores = self.compute_relevance_scores(&context_entities, &query_embedding)?;
            let context_quality = self.assess_context_quality(&context_entities, query_concept)?;
            let diversity_target = self.rng.range_f64(0.4, 0.8) as f32;

            let query = RagQuery {
                query_id: i as u64,
                query_concept,
                query_embedding,
                max_context_size,
                max_depth,
                similarity_threshold,
                expected_context: context_entities,
                expected_relevance_scores: relevance_scores,
                expected_context_quality: context_quality,
                context_diversity_target: diversity_target,
                rag_strategy: strategy,
            };

            queries.push(query);
        }

        Ok(queries)
    }

    /// Generate similarity search queries
    pub fn generate_similarity_queries(&mut self, count: u32) -> Result<Vec<SimilarityQuery>> {
        if self.embeddings.is_empty() {
            return Err(anyhow!("Embeddings required for similarity query generation"));
        }

        let mut queries = Vec::new();
        let entity_ids: Vec<u32> = self.embeddings.keys().cloned().collect();

        for i in 0..count {
            let query_entity = entity_ids[self.rng.next_usize(entity_ids.len())];
            let query_embedding = self.embeddings[&query_entity].clone();
            
            let metric = match i % 5 {
                0 => SimilarityMetric::Euclidean,
                1 => SimilarityMetric::Cosine,
                2 => SimilarityMetric::DotProduct,
                3 => SimilarityMetric::Manhattan,
                _ => SimilarityMetric::Hamming,
            };

            let k = match i % 4 {
                0 => 5,
                1 => 10,
                2 => 20,
                _ => 50,
            };

            let (neighbors, distances) = self.compute_k_nearest_neighbors(
                query_entity,
                &query_embedding,
                k,
                &metric,
            )?;

            let expected_recall = self.estimate_recall(&neighbors, k)?;
            let expected_precision = self.estimate_precision(&neighbors, query_entity)?;

            let query = SimilarityQuery {
                query_id: i as u64,
                query_entity,
                query_embedding,
                k,
                similarity_metric: metric,
                expected_neighbors: neighbors,
                expected_distances: distances,
                expected_recall,
                expected_precision,
            };

            queries.push(query);
        }

        Ok(queries)
    }

    /// Generate complex multi-operation queries
    pub fn generate_complex_queries(&mut self, count: u32) -> Result<Vec<ComplexQuery>> {
        let mut queries = Vec::new();
        let entity_ids: Vec<u32> = self.graph.entities.iter().map(|e| e.id).collect();

        for i in 0..count {
            let operations = self.generate_operation_sequence(i)?;
            let expected_result = self.compute_complex_query_result(&operations)?;
            let expected_performance = self.estimate_complex_query_performance(&operations)?;

            let query = ComplexQuery {
                query_id: i as u64,
                operations,
                expected_result,
                expected_performance,
            };

            queries.push(query);
        }

        Ok(queries)
    }

    // Graph traversal computation methods

    /// Compute shortest path between two entities
    fn compute_shortest_path(&self, start: u32, target: u32) -> Result<(Option<Vec<u32>>, Option<u32>)> {
        if start == target {
            return Ok((Some(vec![start]), Some(0)));
        }

        let adj_list = self.build_adjacency_list();
        let mut distances = HashMap::new();
        let mut predecessors = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(start, 0u32);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if current == target {
                break;
            }

            let current_distance = distances[&current];

            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    if !distances.contains_key(&neighbor) {
                        distances.insert(neighbor, current_distance + 1);
                        predecessors.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if let Some(&distance) = distances.get(&target) {
            let path = self.reconstruct_path(&predecessors, start, target);
            Ok((Some(path), Some(distance)))
        } else {
            Ok((None, None))
        }
    }

    /// Compute reachable entities within max_depth
    fn compute_reachable_entities(&self, start: u32, max_depth: u32) -> Result<Vec<u32>> {
        let adj_list = self.build_adjacency_list();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut reachable = Vec::new();

        queue.push_back((start, 0u32));
        visited.insert(start);

        while let Some((current, depth)) = queue.pop_front() {
            reachable.push(current);

            if depth < max_depth {
                if let Some(neighbors) = adj_list.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                }
            }
        }

        Ok(reachable)
    }

    /// Compute DFS entities
    fn compute_dfs_entities(&self, start: u32, max_depth: u32) -> Result<Vec<u32>> {
        let adj_list = self.build_adjacency_list();
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        self.dfs_recursive(start, 0, max_depth, &adj_list, &mut visited, &mut result);
        Ok(result)
    }

    fn dfs_recursive(
        &self,
        current: u32,
        depth: u32,
        max_depth: u32,
        adj_list: &HashMap<u32, Vec<u32>>,
        visited: &mut HashSet<u32>,
        result: &mut Vec<u32>,
    ) {
        if visited.contains(&current) || depth > max_depth {
            return;
        }

        visited.insert(current);
        result.push(current);

        if let Some(neighbors) = adj_list.get(&current) {
            for &neighbor in neighbors {
                self.dfs_recursive(neighbor, depth + 1, max_depth, adj_list, visited, result);
            }
        }
    }

    /// Compute random walk
    fn compute_random_walk(&mut self, start: u32, walk_length: u32) -> Result<Vec<u32>> {
        let adj_list = self.build_adjacency_list();
        let mut walk = vec![start];
        let mut current = start;

        for _ in 1..walk_length {
            if let Some(neighbors) = adj_list.get(&current) {
                if !neighbors.is_empty() {
                    current = neighbors[self.rng.next_usize(neighbors.len())];
                    walk.push(current);
                } else {
                    break; // Dead end
                }
            } else {
                break; // No neighbors
            }
        }

        Ok(walk)
    }

    /// Compute bidirectional shortest path
    fn compute_bidirectional_path(&self, start: u32, target: u32) -> Result<(Option<Vec<u32>>, Option<u32>)> {
        // For simplicity, fall back to regular shortest path
        // In a full implementation, this would use bidirectional BFS
        self.compute_shortest_path(start, target)
    }

    // RAG computation methods

    /// Compute RAG context based on strategy
    fn compute_rag_context(
        &mut self,
        query_concept: u32,
        query_embedding: &[f32],
        max_context_size: u32,
        max_depth: u32,
        strategy: &RagStrategy,
    ) -> Result<Vec<u32>> {
        match strategy {
            RagStrategy::EmbeddingOnly => {
                self.compute_embedding_context(query_concept, query_embedding, max_context_size)
            },
            RagStrategy::GraphOnly => {
                self.compute_graph_context(query_concept, max_depth, max_context_size)
            },
            RagStrategy::Hybrid => {
                self.compute_hybrid_context(query_concept, query_embedding, max_depth, max_context_size)
            },
            RagStrategy::Adaptive => {
                // Choose strategy based on query characteristics
                if self.get_entity_degree(query_concept) > 10 {
                    self.compute_graph_context(query_concept, max_depth, max_context_size)
                } else {
                    self.compute_embedding_context(query_concept, query_embedding, max_context_size)
                }
            }
        }
    }

    fn compute_embedding_context(&self, query_concept: u32, query_embedding: &[f32], max_context_size: u32) -> Result<Vec<u32>> {
        let mut candidates = Vec::new();

        for (&entity_id, embedding) in &self.embeddings {
            if entity_id != query_concept {
                let similarity = self.cosine_similarity(query_embedding, embedding);
                candidates.push((entity_id, similarity));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(candidates.iter().take(max_context_size as usize).map(|(id, _)| *id).collect())
    }

    fn compute_graph_context(&self, query_concept: u32, max_depth: u32, max_context_size: u32) -> Result<Vec<u32>> {
        let reachable = self.compute_reachable_entities(query_concept, max_depth)?;
        Ok(reachable.into_iter().take(max_context_size as usize).collect())
    }

    fn compute_hybrid_context(&self, query_concept: u32, query_embedding: &[f32], max_depth: u32, max_context_size: u32) -> Result<Vec<u32>> {
        let embedding_context = self.compute_embedding_context(query_concept, query_embedding, max_context_size / 2)?;
        let graph_context = self.compute_graph_context(query_concept, max_depth, max_context_size / 2)?;
        
        let mut combined: HashSet<u32> = embedding_context.into_iter().collect();
        combined.extend(graph_context);
        
        Ok(combined.into_iter().take(max_context_size as usize).collect())
    }

    fn compute_relevance_scores(&self, context_entities: &[u32], query_embedding: &[f32]) -> Result<Vec<f32>> {
        let mut scores = Vec::new();
        
        for &entity_id in context_entities {
            let score = if let Some(embedding) = self.embeddings.get(&entity_id) {
                self.cosine_similarity(query_embedding, embedding)
            } else {
                0.5 // Default relevance for entities without embeddings
            };
            scores.push(score);
        }
        
        Ok(scores)
    }

    fn assess_context_quality(&self, context_entities: &[u32], query_concept: u32) -> Result<f32> {
        if context_entities.is_empty() {
            return Ok(0.0);
        }

        // Quality based on diversity and relevance
        let mut diversity_score = 0.0;
        let mut relevance_score = 0.0;

        // Calculate average pairwise distance for diversity
        if context_entities.len() > 1 {
            let mut total_distance = 0.0;
            let mut pair_count = 0;

            for i in 0..context_entities.len() {
                for j in (i + 1)..context_entities.len() {
                    if let (Some(emb1), Some(emb2)) = (
                        self.embeddings.get(&context_entities[i]),
                        self.embeddings.get(&context_entities[j]),
                    ) {
                        total_distance += 1.0 - self.cosine_similarity(emb1, emb2);
                        pair_count += 1;
                    }
                }
            }

            diversity_score = if pair_count > 0 {
                total_distance / pair_count as f32
            } else {
                0.0
            };
        }

        // Calculate average relevance to query
        if let Some(query_embedding) = self.embeddings.get(&query_concept) {
            let mut total_relevance = 0.0;
            let mut entity_count = 0;

            for &entity_id in context_entities {
                if let Some(embedding) = self.embeddings.get(&entity_id) {
                    total_relevance += self.cosine_similarity(query_embedding, embedding);
                    entity_count += 1;
                }
            }

            relevance_score = if entity_count > 0 {
                total_relevance / entity_count as f32
            } else {
                0.0
            };
        }

        // Combine diversity and relevance
        Ok((diversity_score * 0.4 + relevance_score * 0.6).min(1.0))
    }

    // Similarity computation methods

    fn compute_k_nearest_neighbors(
        &self,
        query_entity: u32,
        query_embedding: &[f32],
        k: usize,
        metric: &SimilarityMetric,
    ) -> Result<(Vec<(u32, f32)>, Vec<f32>)> {
        let mut candidates = Vec::new();

        for (&entity_id, embedding) in &self.embeddings {
            if entity_id != query_entity {
                let distance = match metric {
                    SimilarityMetric::Euclidean => self.euclidean_distance(query_embedding, embedding),
                    SimilarityMetric::Cosine => 1.0 - self.cosine_similarity(query_embedding, embedding),
                    SimilarityMetric::DotProduct => -self.dot_product(query_embedding, embedding),
                    SimilarityMetric::Manhattan => self.manhattan_distance(query_embedding, embedding),
                    SimilarityMetric::Hamming => self.hamming_distance(query_embedding, embedding),
                };
                candidates.push((entity_id, distance));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        let neighbors = candidates.clone();
        let distances = candidates.iter().map(|(_, d)| *d).collect();

        Ok((neighbors, distances))
    }

    fn estimate_recall(&self, neighbors: &[(u32, f32)], k: usize) -> Result<f32> {
        // Estimate recall based on distance distribution
        if neighbors.is_empty() {
            return Ok(0.0);
        }

        let avg_distance = neighbors.iter().map(|(_, d)| *d).sum::<f32>() / neighbors.len() as f32;
        
        // Simple heuristic: lower average distance suggests better recall
        let recall = (1.0 - avg_distance.min(1.0)).max(0.0);
        Ok(recall)
    }

    fn estimate_precision(&self, neighbors: &[(u32, f32)], query_entity: u32) -> Result<f32> {
        // Estimate precision based on neighbor consistency
        if neighbors.is_empty() {
            return Ok(0.0);
        }

        // Simple heuristic: all neighbors are considered relevant if distances are reasonable
        let reasonable_neighbors = neighbors.iter().filter(|(_, d)| *d < 1.0).count();
        let precision = reasonable_neighbors as f32 / neighbors.len() as f32;
        
        Ok(precision)
    }

    // Complex query methods

    fn generate_operation_sequence(&mut self, seed: u32) -> Result<Vec<QueryOperation>> {
        let mut operations = Vec::new();
        let operation_count = self.rng.range_i32(2, 6) as usize;
        let entity_ids: Vec<u32> = self.graph.entities.iter().map(|e| e.id).collect();

        for i in 0..operation_count {
            let operation = match (seed + i as u32) % 5 {
                0 => QueryOperation::SimilaritySearch {
                    entity: entity_ids[self.rng.next_usize(entity_ids.len())],
                    k: self.rng.range_i32(5, 20) as usize,
                },
                1 => QueryOperation::GraphTraversal {
                    start: entity_ids[self.rng.next_usize(entity_ids.len())],
                    max_depth: self.rng.range_i32(1, 4) as u32,
                },
                2 => QueryOperation::Filter {
                    attribute: "entity_type".to_string(),
                    value: "Node".to_string(),
                },
                3 => QueryOperation::Aggregate {
                    operation: AggregateOp::Count,
                    field: "id".to_string(),
                },
                _ => QueryOperation::Join {
                    entities: vec![
                        entity_ids[self.rng.next_usize(entity_ids.len())],
                        entity_ids[self.rng.next_usize(entity_ids.len())],
                    ],
                    relationship: "connected".to_string(),
                },
            };
            operations.push(operation);
        }

        Ok(operations)
    }

    fn compute_complex_query_result(&self, operations: &[QueryOperation]) -> Result<QueryResult> {
        let mut entities = HashSet::new();
        let mut relationships = Vec::new();
        let mut aggregated_values = HashMap::new();

        // Simulate executing operations in sequence
        for operation in operations {
            match operation {
                QueryOperation::SimilaritySearch { entity, k } => {
                    entities.insert(*entity);
                    // Add some neighbors
                    let adj_list = self.build_adjacency_list();
                    if let Some(neighbors) = adj_list.get(entity) {
                        for &neighbor in neighbors.iter().take(*k) {
                            entities.insert(neighbor);
                        }
                    }
                },
                QueryOperation::GraphTraversal { start, max_depth } => {
                    if let Ok(reachable) = self.compute_reachable_entities(*start, *max_depth) {
                        for entity in reachable {
                            entities.insert(entity);
                        }
                    }
                },
                QueryOperation::Filter { .. } => {
                    // Simulated filtering - keep all entities for simplicity
                },
                QueryOperation::Aggregate { operation, field } => {
                    let value = match operation {
                        AggregateOp::Count => entities.len() as f64,
                        AggregateOp::Sum => entities.iter().map(|&id| id as f64).sum(),
                        AggregateOp::Average => {
                            let sum: f64 = entities.iter().map(|&id| id as f64).sum();
                            sum / entities.len() as f64
                        },
                        AggregateOp::Min => entities.iter().map(|&id| id as f64).fold(f64::INFINITY, f64::min),
                        AggregateOp::Max => entities.iter().map(|&id| id as f64).fold(f64::NEG_INFINITY, f64::max),
                    };
                    aggregated_values.insert(field.clone(), value);
                },
                QueryOperation::Join { entities: join_entities, relationship } => {
                    for entity in join_entities {
                        entities.insert(*entity);
                        relationships.push((*entity, *entity, relationship.clone()));
                    }
                },
            }
        }

        Ok(QueryResult {
            entities: entities.into_iter().collect(),
            relationships,
            aggregated_values,
            result_count: entities.len() as u64,
        })
    }

    fn estimate_complex_query_performance(&self, operations: &[QueryOperation]) -> Result<PerformanceExpectation> {
        let mut total_latency = 0.0;
        let mut total_memory = 0.0;
        let mut total_io = 0u64;

        for operation in operations {
            match operation {
                QueryOperation::SimilaritySearch { k, .. } => {
                    total_latency += 1.0 + (*k as f64 * 0.1); // Base + k-dependent time
                    total_memory += *k as f64 * 0.1; // Memory for k results
                    total_io += *k as u64;
                },
                QueryOperation::GraphTraversal { max_depth, .. } => {
                    total_latency += *max_depth as f64 * 0.5;
                    total_memory += *max_depth as f64 * 0.2;
                    total_io += *max_depth as u64 * 10;
                },
                QueryOperation::Filter { .. } => {
                    total_latency += 0.5;
                    total_memory += 0.1;
                    total_io += 1;
                },
                QueryOperation::Aggregate { .. } => {
                    total_latency += 0.2;
                    total_memory += 0.05;
                    total_io += 1;
                },
                QueryOperation::Join { entities, .. } => {
                    total_latency += entities.len() as f64 * 0.3;
                    total_memory += entities.len() as f64 * 0.1;
                    total_io += entities.len() as u64;
                },
            }
        }

        Ok(PerformanceExpectation {
            max_latency_ms: total_latency,
            max_memory_mb: total_memory,
            expected_io_operations: total_io,
            cache_hit_ratio: 0.7, // Assume 70% cache hit rate
        })
    }

    // Utility methods

    fn build_adjacency_list(&self) -> HashMap<u32, Vec<u32>> {
        let mut adj_list = HashMap::new();

        // Initialize with all entities
        for entity in &self.graph.entities {
            adj_list.insert(entity.id, Vec::new());
        }

        // Add edges
        for edge in &self.graph.edges {
            adj_list.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
            adj_list.entry(edge.target).or_insert_with(Vec::new).push(edge.source);
        }

        adj_list
    }

    fn reconstruct_path(&self, predecessors: &HashMap<u32, u32>, start: u32, target: u32) -> Vec<u32> {
        let mut path = Vec::new();
        let mut current = target;

        while current != start {
            path.push(current);
            if let Some(&pred) = predecessors.get(&current) {
                current = pred;
            } else {
                break;
            }
        }

        path.push(start);
        path.reverse();
        path
    }

    fn get_entity_degree(&self, entity_id: u32) -> usize {
        self.graph.edges.iter()
            .filter(|e| e.source == entity_id || e.target == entity_id)
            .count()
    }

    // Distance/similarity metrics

    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return f32::INFINITY;
        }
        v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>().sqrt()
    }

    fn cosine_similarity(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }

        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2 = v2.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm1 > 1e-10 && norm2 > 1e-10 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    fn dot_product(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }
        v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn manhattan_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return f32::INFINITY;
        }
        v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b).abs()).sum()
    }

    fn hamming_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return f32::INFINITY;
        }
        v1.iter().zip(v2.iter()).map(|(&a, &b)| if (a - b).abs() > 1e-6 { 1.0 } else { 0.0 }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generation::graph_topologies::GraphTopologyGenerator;
    use crate::data_generation::embeddings::EmbeddingGenerator;

    fn create_test_setup() -> (TestGraph, HashMap<u32, Vec<f32>>) {
        let mut graph_gen = GraphTopologyGenerator::new(42);
        let graph = graph_gen.generate_erdos_renyi(20, 0.1).unwrap();
        
        let mut embedding_gen = EmbeddingGenerator::new(42, 64).unwrap();
        let mut embeddings = HashMap::new();
        
        for entity in &graph.entities {
            let embedding = embedding_gen.generate_random_unit_vector();
            embeddings.insert(entity.id, embedding);
        }
        
        (graph, embeddings)
    }

    #[test]
    fn test_traversal_query_generation() {
        let (graph, embeddings) = create_test_setup();
        let mut generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        let queries = generator.generate_traversal_queries(10).unwrap();
        
        assert_eq!(queries.len(), 10);
        
        for query in &queries {
            assert!(query.start_entity < 20); // Within entity range
            assert!(query.max_depth > 0);
        }
    }

    #[test]
    fn test_rag_query_generation() {
        let (graph, embeddings) = create_test_setup();
        let mut generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        let queries = generator.generate_rag_queries(5).unwrap();
        
        assert_eq!(queries.len(), 5);
        
        for query in &queries {
            assert!(!query.expected_context.is_empty());
            assert!(!query.expected_relevance_scores.is_empty());
            assert!(query.expected_context_quality >= 0.0 && query.expected_context_quality <= 1.0);
        }
    }

    #[test]
    fn test_similarity_query_generation() {
        let (graph, embeddings) = create_test_setup();
        let mut generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        let queries = generator.generate_similarity_queries(5).unwrap();
        
        assert_eq!(queries.len(), 5);
        
        for query in &queries {
            assert!(query.k > 0);
            assert!(!query.expected_neighbors.is_empty());
            assert!(!query.expected_distances.is_empty());
        }
    }

    #[test]
    fn test_complex_query_generation() {
        let (graph, embeddings) = create_test_setup();
        let mut generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        let queries = generator.generate_complex_queries(3).unwrap();
        
        assert_eq!(queries.len(), 3);
        
        for query in &queries {
            assert!(!query.operations.is_empty());
            assert!(query.expected_performance.max_latency_ms > 0.0);
        }
    }

    #[test]
    fn test_shortest_path_computation() {
        let (graph, embeddings) = create_test_setup();
        let generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        // Test path from entity to itself
        let (path, distance) = generator.compute_shortest_path(0, 0).unwrap();
        assert_eq!(path, Some(vec![0]));
        assert_eq!(distance, Some(0));
    }

    #[test]
    fn test_similarity_metrics() {
        let (graph, embeddings) = create_test_setup();
        let generator = QueryPatternGenerator::new(42, graph, embeddings);
        
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        
        // Test different similarity metrics
        assert!((generator.euclidean_distance(&v1, &v2) - 1.414).abs() < 0.01);
        assert!((generator.cosine_similarity(&v1, &v2) - 0.0).abs() < 1e-6);
        assert_eq!(generator.dot_product(&v1, &v2), 0.0);
        assert_eq!(generator.manhattan_distance(&v1, &v2), 2.0);
    }

    #[test]
    fn test_deterministic_generation() {
        let (graph, embeddings) = create_test_setup();
        
        let mut gen1 = QueryPatternGenerator::new(12345, graph.clone(), embeddings.clone());
        let mut gen2 = QueryPatternGenerator::new(12345, graph, embeddings);
        
        let queries1 = gen1.generate_traversal_queries(5).unwrap();
        let queries2 = gen2.generate_traversal_queries(5).unwrap();
        
        for (q1, q2) in queries1.iter().zip(queries2.iter()) {
            assert_eq!(q1.start_entity, q2.start_entity);
            assert_eq!(q1.max_depth, q2.max_depth);
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let empty_graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: crate::data_generation::graph_topologies::GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: crate::data_generation::graph_topologies::ConnectivityType::Random,
                expected_path_length: 0.0,
            },
        };
        
        let mut generator = QueryPatternGenerator::new(42, empty_graph, HashMap::new());
        
        // Should fail with empty graph
        assert!(generator.generate_traversal_queries(1).is_err());
        assert!(generator.generate_rag_queries(1).is_err());
        assert!(generator.generate_similarity_queries(1).is_err());
    }
}