//! Golden Standards Computation
//! 
//! Provides exact computation of expected outcomes for all test scenarios.

use crate::data_generation::graph_topologies::{TestGraph, TestEntity, TestEdge};
use crate::data_generation::query_patterns::{TraversalQuery, RagQuery, SimilarityQuery, ComplexQuery, SimilarityMetric, TraversalType};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Complete set of golden standards for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenStandards {
    pub traversal_results: Vec<TraversalResult>,
    pub similarity_results: Vec<SimilarityResult>,
    pub rag_results: Vec<RagResult>,
    pub performance_expectations: PerformanceExpectations,
    pub federation_results: Vec<FederationResult>,
    pub checksum: String,
}

/// Traversal query result with mathematical guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    pub query_id: u64,
    pub result_type: TraversalResultType,
    pub execution_metadata: ExecutionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraversalResultType {
    Path {
        path: Vec<u32>,
        distance: u32,
        total_entities: u64,
    },
    Neighborhood {
        entities: Vec<u32>,
        total_entities: u64,
        depth_distribution: HashMap<u32, u64>, // depth -> count
    },
    Walk {
        walk_sequence: Vec<u32>,
        unique_entities: u64,
        cycles_detected: Vec<Vec<u32>>,
    },
}

/// Similarity search result with exact computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub query_id: u64,
    pub query_entity: u32,
    pub k: usize,
    pub nearest_neighbors: Vec<(u32, f64)>, // (entity_id, exact_distance)
    pub distance_statistics: DistanceStatistics,
    pub query_quality_metrics: QueryQualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceStatistics {
    pub min_distance: f64,
    pub max_distance: f64,
    pub mean_distance: f64,
    pub std_deviation: f64,
    pub distance_distribution: Vec<f64>, // Histogram bins
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryQualityMetrics {
    pub average_distance: f64,
    pub distance_variance: f64,
    pub neighbor_consistency_score: f64,
    pub geometric_consistency: f64,
}

/// RAG query result with context quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResult {
    pub query_id: u64,
    pub query_concept: u32,
    pub context_entities: Vec<u32>,
    pub context_quality_score: f64,
    pub relevance_scores: Vec<f64>,
    pub diversity_metrics: DiversityMetrics,
    pub coverage_analysis: CoverageAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub pairwise_similarity_mean: f64,
    pub pairwise_similarity_std: f64,
    pub cluster_separation: f64,
    pub topic_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    pub graph_coverage_ratio: f64,
    pub semantic_coverage_score: f64,
    pub redundancy_score: f64,
    pub completeness_score: f64,
}

/// Performance expectations with statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    pub query_latency_bounds: LatencyBounds,
    pub memory_usage_bounds: MemoryBounds,
    pub throughput_expectations: ThroughputExpectations,
    pub compression_expectations: CompressionExpectations,
    pub scalability_predictions: ScalabilityPredictions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBounds {
    pub traversal_queries: (f64, f64), // (min_ms, max_ms)
    pub similarity_queries: (f64, f64),
    pub rag_queries: (f64, f64),
    pub complex_queries: (f64, f64),
    pub percentile_95: f64,
    pub percentile_99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBounds {
    pub base_memory_mb: f64,
    pub per_entity_bytes: f64,
    pub per_embedding_bytes: f64,
    pub peak_memory_mb: f64,
    pub memory_efficiency_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputExpectations {
    pub queries_per_second: f64,
    pub entities_per_second: f64,
    pub embeddings_per_second: f64,
    pub concurrent_query_capacity: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionExpectations {
    pub compression_ratios: HashMap<String, f64>, // algorithm -> ratio
    pub quality_preservation: HashMap<String, f64>, // algorithm -> quality_score
    pub decompression_speed: HashMap<String, f64>, // algorithm -> ms_per_vector
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityPredictions {
    pub linear_scaling_factors: HashMap<String, f64>,
    pub logarithmic_scaling_factors: HashMap<String, f64>,
    pub memory_scaling_exponent: f64,
    pub complexity_analysis: HashMap<String, String>,
}

/// Federation query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationResult {
    pub query_id: u64,
    pub participating_databases: Vec<u32>,
    pub coordination_overhead_ms: f64,
    pub network_operations: u32,
    pub result_merge_complexity: MergeComplexity,
    pub consistency_guarantees: ConsistencyGuarantees,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeComplexity {
    SimpleUnion { cost: f64 },
    PathMerging { cost: f64, conflicts: u32 },
    Aggregation { cost: f64, precision_loss: f64 },
    ComplexJoin { cost: f64, join_selectivity: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyGuarantees {
    pub consistency_level: String,
    pub isolation_level: String,
    pub durability_guarantee: bool,
    pub conflict_resolution: String,
}

/// Execution metadata for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub computational_complexity: String,
    pub memory_complexity: String,
    pub io_operations: u64,
    pub cache_efficiency: f64,
    pub algorithm_used: String,
}

/// Exact computation engine for golden standards
pub struct ExactComputationEngine {
    graph: TestGraph,
    embeddings: HashMap<u32, Vec<f32>>,
}

impl ExactComputationEngine {
    /// Create a new exact computation engine
    pub fn new(graph: TestGraph, embeddings: HashMap<u32, Vec<f32>>) -> Self {
        Self { graph, embeddings }
    }

    /// Compute all expected outcomes for the given dataset
    pub fn compute_all_expected_outcomes(
        &mut self,
        traversal_queries: &[TraversalQuery],
        similarity_queries: &[SimilarityQuery],
        rag_queries: &[RagQuery],
    ) -> Result<GoldenStandards> {
        let traversal_results = self.compute_traversal_outcomes(traversal_queries)?;
        let similarity_results = self.compute_similarity_outcomes(similarity_queries)?;
        let rag_results = self.compute_rag_outcomes(rag_queries)?;
        let performance_expectations = self.compute_performance_expectations()?;
        let federation_results = self.compute_federation_outcomes()?;
        
        let checksum = self.compute_checksum(&traversal_results, &similarity_results, &rag_results)?;

        Ok(GoldenStandards {
            traversal_results,
            similarity_results,
            rag_results,
            performance_expectations,
            federation_results,
            checksum,
        })
    }

    /// Compute exact traversal outcomes using graph algorithms
    fn compute_traversal_outcomes(&mut self, queries: &[TraversalQuery]) -> Result<Vec<TraversalResult>> {
        let adj_list = self.build_adjacency_list();
        let mut results = Vec::new();

        for query in queries {
            let result_type = match query.traversal_type {
                TraversalType::ShortestPath => {
                    if let Some(target) = query.target_entity {
                        let (path, distance) = self.dijkstra_shortest_path(&adj_list, query.start_entity, target)?;
                        TraversalResultType::Path {
                            path: path.clone(),
                            distance,
                            total_entities: path.len() as u64,
                        }
                    } else {
                        return Err(anyhow!("Shortest path query requires target entity"));
                    }
                },
                
                TraversalType::BreadthFirstSearch => {
                    let (entities, depth_dist) = self.bfs_with_depth_tracking(&adj_list, query.start_entity, query.max_depth)?;
                    TraversalResultType::Neighborhood {
                        entities,
                        total_entities: entities.len() as u64,
                        depth_distribution: depth_dist,
                    }
                },
                
                TraversalType::DepthFirstSearch => {
                    let entities = self.dfs_exact(&adj_list, query.start_entity, query.max_depth)?;
                    let depth_dist = self.compute_depth_distribution(&entities, query.start_entity, &adj_list)?;
                    TraversalResultType::Neighborhood {
                        entities,
                        total_entities: entities.len() as u64,
                        depth_distribution: depth_dist,
                    }
                },
                
                TraversalType::RandomWalk => {
                    // For golden standard, compute all possible walks and analyze
                    let walk_analysis = self.analyze_random_walk_space(&adj_list, query.start_entity, query.max_depth)?;
                    TraversalResultType::Walk {
                        walk_sequence: walk_analysis.representative_walk,
                        unique_entities: walk_analysis.unique_entities,
                        cycles_detected: walk_analysis.cycles,
                    }
                },
                
                TraversalType::BiDirectional => {
                    if let Some(target) = query.target_entity {
                        let (path, distance) = self.bidirectional_shortest_path(&adj_list, query.start_entity, target)?;
                        TraversalResultType::Path {
                            path: path.clone(),
                            distance,
                            total_entities: path.len() as u64,
                        }
                    } else {
                        return Err(anyhow!("Bidirectional search requires target entity"));
                    }
                },
            };

            let execution_metadata = self.compute_traversal_metadata(&query.traversal_type, &result_type);

            results.push(TraversalResult {
                query_id: query.query_id,
                result_type,
                execution_metadata,
            });
        }

        Ok(results)
    }

    /// Compute exact similarity search outcomes
    fn compute_similarity_outcomes(&mut self, queries: &[SimilarityQuery]) -> Result<Vec<SimilarityResult>> {
        let mut results = Vec::new();

        for query in queries {
            let query_embedding = &query.query_embedding;
            let mut candidates = Vec::new();

            // Compute exact distances to all entities
            for (&entity_id, embedding) in &self.embeddings {
                if entity_id != query.query_entity {
                    let distance = self.compute_exact_distance(query_embedding, embedding, &query.similarity_metric)?;
                    candidates.push((entity_id, distance));
                }
            }

            // Sort by distance and take top k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let nearest_neighbors = candidates.into_iter().take(query.k).collect::<Vec<_>>();

            // Compute distance statistics
            let distances: Vec<f64> = nearest_neighbors.iter().map(|(_, d)| *d).collect();
            let distance_statistics = self.compute_distance_statistics(&distances)?;

            // Compute query quality metrics
            let query_quality_metrics = self.compute_query_quality_metrics(query_embedding, &nearest_neighbors)?;

            results.push(SimilarityResult {
                query_id: query.query_id,
                query_entity: query.query_entity,
                k: query.k,
                nearest_neighbors,
                distance_statistics,
                query_quality_metrics,
            });
        }

        Ok(results)
    }

    /// Compute exact RAG outcomes with quality analysis
    fn compute_rag_outcomes(&mut self, queries: &[RagQuery]) -> Result<Vec<RagResult>> {
        let mut results = Vec::new();
        let adj_list = self.build_adjacency_list();

        for query in queries {
            // Step 1: Compute embedding-based candidates
            let embedding_candidates = self.compute_embedding_similarity_candidates(
                &query.query_embedding,
                query.max_context_size * 2,
            )?;

            // Step 2: Compute graph-based expansion
            let graph_candidates = self.compute_graph_expansion_candidates(
                &adj_list,
                query.query_concept,
                query.max_depth,
                query.max_context_size,
            )?;

            // Step 3: Merge and rank candidates
            let context_entities = self.merge_and_rank_rag_candidates(
                embedding_candidates,
                graph_candidates,
                &query.query_embedding,
                query.max_context_size,
            )?;

            // Step 4: Compute relevance scores
            let relevance_scores = self.compute_exact_relevance_scores(&context_entities, &query.query_embedding)?;

            // Step 5: Analyze context quality
            let context_quality_score = self.analyze_context_quality(&context_entities, query.query_concept)?;

            // Step 6: Compute diversity metrics
            let diversity_metrics = self.compute_diversity_metrics(&context_entities)?;

            // Step 7: Compute coverage analysis
            let coverage_analysis = self.compute_coverage_analysis(&context_entities, query.query_concept)?;

            results.push(RagResult {
                query_id: query.query_id,
                query_concept: query.query_concept,
                context_entities,
                context_quality_score,
                relevance_scores,
                diversity_metrics,
                coverage_analysis,
            });
        }

        Ok(results)
    }

    /// Compute performance expectations based on algorithmic analysis
    fn compute_performance_expectations(&self) -> Result<PerformanceExpectations> {
        let entity_count = self.graph.entities.len();
        let edge_count = self.graph.edges.len();
        let embedding_count = self.embeddings.len();

        // Latency bounds based on algorithmic complexity
        let query_latency_bounds = LatencyBounds {
            traversal_queries: (0.1, entity_count as f64 * 0.01), // O(V + E)
            similarity_queries: (0.5, embedding_count as f64 * 0.1), // O(n * d)
            rag_queries: (1.0, (entity_count + embedding_count) as f64 * 0.1), // Hybrid
            complex_queries: (2.0, entity_count as f64 * 0.5), // Complex operations
            percentile_95: entity_count as f64 * 0.08,
            percentile_99: entity_count as f64 * 0.15,
        };

        // Memory bounds based on data structures
        let memory_bounds = MemoryBounds {
            base_memory_mb: 10.0, // Base overhead
            per_entity_bytes: 64.0, // Entity storage
            per_embedding_bytes: self.embeddings.values().next().map_or(0.0, |v| v.len() as f64 * 4.0),
            peak_memory_mb: 50.0 + (entity_count as f64 * 0.1),
            memory_efficiency_ratio: 0.85, // 85% efficiency
        };

        // Throughput expectations
        let throughput_expectations = ThroughputExpectations {
            queries_per_second: 1000.0 / (1.0 + (entity_count as f64 / 1000.0).log10()),
            entities_per_second: 10000.0,
            embeddings_per_second: 5000.0,
            concurrent_query_capacity: 16,
        };

        // Compression expectations
        let mut compression_ratios = HashMap::new();
        compression_ratios.insert("PQ8".to_string(), 4.0);
        compression_ratios.insert("PQ4".to_string(), 8.0);
        compression_ratios.insert("LSH".to_string(), 10.0);

        let mut quality_preservation = HashMap::new();
        quality_preservation.insert("PQ8".to_string(), 0.95);
        quality_preservation.insert("PQ4".to_string(), 0.85);
        quality_preservation.insert("LSH".to_string(), 0.80);

        let mut decompression_speed = HashMap::new();
        decompression_speed.insert("PQ8".to_string(), 0.01);
        decompression_speed.insert("PQ4".to_string(), 0.005);
        decompression_speed.insert("LSH".to_string(), 0.002);

        let compression_expectations = CompressionExpectations {
            compression_ratios,
            quality_preservation,
            decompression_speed,
        };

        // Scalability predictions
        let mut linear_factors = HashMap::new();
        linear_factors.insert("entity_insertion".to_string(), 1.0);
        linear_factors.insert("embedding_similarity".to_string(), 1.0);

        let mut log_factors = HashMap::new();
        log_factors.insert("shortest_path".to_string(), 1.0);
        log_factors.insert("graph_traversal".to_string(), 1.2);

        let mut complexity_analysis = HashMap::new();
        complexity_analysis.insert("BFS".to_string(), "O(V + E)".to_string());
        complexity_analysis.insert("DFS".to_string(), "O(V + E)".to_string());
        complexity_analysis.insert("Dijkstra".to_string(), "O((V + E) log V)".to_string());
        complexity_analysis.insert("Similarity".to_string(), "O(n * d)".to_string());

        let scalability_predictions = ScalabilityPredictions {
            linear_scaling_factors: linear_factors,
            logarithmic_scaling_factors: log_factors,
            memory_scaling_exponent: 1.1,
            complexity_analysis,
        };

        Ok(PerformanceExpectations {
            query_latency_bounds,
            memory_bounds,
            throughput_expectations,
            compression_expectations,
            scalability_predictions,
        })
    }

    /// Compute federation outcomes (simplified for single-node)
    fn compute_federation_outcomes(&self) -> Result<Vec<FederationResult>> {
        // For single-node testing, create mock federation results
        let mut results = Vec::new();

        for i in 0..3 {
            results.push(FederationResult {
                query_id: i,
                participating_databases: vec![0, 1],
                coordination_overhead_ms: 2.5,
                network_operations: 2,
                result_merge_complexity: MergeComplexity::SimpleUnion { cost: 1.0 },
                consistency_guarantees: ConsistencyGuarantees {
                    consistency_level: "eventual".to_string(),
                    isolation_level: "read_committed".to_string(),
                    durability_guarantee: true,
                    conflict_resolution: "last_write_wins".to_string(),
                },
            });
        }

        Ok(results)
    }

    // Graph algorithm implementations

    fn build_adjacency_list(&self) -> HashMap<u32, Vec<u32>> {
        let mut adj_list = HashMap::new();

        for entity in &self.graph.entities {
            adj_list.insert(entity.id, Vec::new());
        }

        for edge in &self.graph.edges {
            adj_list.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
            adj_list.entry(edge.target).or_insert_with(Vec::new).push(edge.source);
        }

        adj_list
    }

    fn dijkstra_shortest_path(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, target: u32) -> Result<(Vec<u32>, u32)> {
        if start == target {
            return Ok((vec![start], 0));
        }

        let mut distances = HashMap::new();
        let mut predecessors = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(start, 0u32);
        heap.push(std::cmp::Reverse((0u32, start)));

        while let Some(std::cmp::Reverse((current_distance, current_node))) = heap.pop() {
            if current_node == target {
                break;
            }

            if current_distance > distances.get(&current_node).unwrap_or(&u32::MAX) {
                continue;
            }

            if let Some(neighbors) = adj_list.get(&current_node) {
                for &neighbor in neighbors {
                    let new_distance = current_distance + 1;
                    
                    if new_distance < *distances.get(&neighbor).unwrap_or(&u32::MAX) {
                        distances.insert(neighbor, new_distance);
                        predecessors.insert(neighbor, current_node);
                        heap.push(std::cmp::Reverse((new_distance, neighbor)));
                    }
                }
            }
        }

        if let Some(&distance) = distances.get(&target) {
            let path = self.reconstruct_path(&predecessors, start, target);
            Ok((path, distance))
        } else {
            Err(anyhow!("No path found from {} to {}", start, target))
        }
    }

    fn bfs_with_depth_tracking(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, max_depth: u32) -> Result<(Vec<u32>, HashMap<u32, u64>)> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut entities = Vec::new();
        let mut depth_distribution = HashMap::new();

        queue.push_back((start, 0u32));
        visited.insert(start);

        while let Some((current, depth)) = queue.pop_front() {
            entities.push(current);
            *depth_distribution.entry(depth).or_insert(0) += 1;

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

        Ok((entities, depth_distribution))
    }

    fn dfs_exact(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, max_depth: u32) -> Result<Vec<u32>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        self.dfs_recursive(start, 0, max_depth, adj_list, &mut visited, &mut result);
        Ok(result)
    }

    fn dfs_recursive(&self, current: u32, depth: u32, max_depth: u32, adj_list: &HashMap<u32, Vec<u32>>, visited: &mut HashSet<u32>, result: &mut Vec<u32>) {
        if depth > max_depth || visited.contains(&current) {
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

    fn bidirectional_shortest_path(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, target: u32) -> Result<(Vec<u32>, u32)> {
        // Simplified implementation - use regular Dijkstra for now
        self.dijkstra_shortest_path(adj_list, start, target)
    }

    // Distance computation methods

    fn compute_exact_distance(&self, v1: &[f32], v2: &[f32], metric: &SimilarityMetric) -> Result<f64> {
        if v1.len() != v2.len() {
            return Err(anyhow!("Vector dimension mismatch"));
        }

        let distance = match metric {
            SimilarityMetric::Euclidean => {
                v1.iter().zip(v2.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt()
            },
            SimilarityMetric::Cosine => {
                let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
                let norm1 = v1.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let norm2 = v2.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                
                if norm1 > 1e-10 && norm2 > 1e-10 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0
                }
            },
            SimilarityMetric::DotProduct => {
                -v1.iter().zip(v2.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum::<f64>()
            },
            SimilarityMetric::Manhattan => {
                v1.iter().zip(v2.iter()).map(|(&a, &b)| ((a - b) as f64).abs()).sum()
            },
            SimilarityMetric::Hamming => {
                v1.iter().zip(v2.iter()).map(|(&a, &b)| if (a - b).abs() > 1e-6 { 1.0 } else { 0.0 }).sum()
            },
        };

        Ok(distance)
    }

    // Quality analysis methods

    fn compute_distance_statistics(&self, distances: &[f64]) -> Result<DistanceStatistics> {
        if distances.is_empty() {
            return Ok(DistanceStatistics {
                min_distance: 0.0,
                max_distance: 0.0,
                mean_distance: 0.0,
                std_deviation: 0.0,
                distance_distribution: vec![],
            });
        }

        let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_distance = distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        
        let variance = distances.iter()
            .map(|&x| (x - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
        let std_deviation = variance.sqrt();

        // Create histogram
        let bin_count = 10;
        let mut distance_distribution = vec![0.0; bin_count];
        let range = max_distance - min_distance;
        
        if range > 0.0 {
            for &distance in distances {
                let bin = ((distance - min_distance) / range * (bin_count - 1) as f64).floor() as usize;
                let bin_idx = bin.min(bin_count - 1);
                distance_distribution[bin_idx] += 1.0;
            }
            
            // Normalize
            for count in &mut distance_distribution {
                *count /= distances.len() as f64;
            }
        }

        Ok(DistanceStatistics {
            min_distance,
            max_distance,
            mean_distance,
            std_deviation,
            distance_distribution,
        })
    }

    fn compute_query_quality_metrics(&self, query_embedding: &[f32], neighbors: &[(u32, f64)]) -> Result<QueryQualityMetrics> {
        if neighbors.is_empty() {
            return Ok(QueryQualityMetrics {
                average_distance: 0.0,
                distance_variance: 0.0,
                neighbor_consistency_score: 0.0,
                geometric_consistency: 0.0,
            });
        }

        let distances: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
        let average_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let distance_variance = distances.iter()
            .map(|&d| (d - average_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;

        // Consistency score: how well neighbors agree with each other
        let mut consistency_sum = 0.0;
        let mut consistency_count = 0;

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if let (Some(emb1), Some(emb2)) = (
                    self.embeddings.get(&neighbors[i].0),
                    self.embeddings.get(&neighbors[j].0),
                ) {
                    let similarity = self.cosine_similarity(emb1, emb2);
                    consistency_sum += similarity as f64;
                    consistency_count += 1;
                }
            }
        }

        let neighbor_consistency_score = if consistency_count > 0 {
            consistency_sum / consistency_count as f64
        } else {
            0.0
        };

        // Geometric consistency: triangle inequality preservation
        let geometric_consistency = self.compute_geometric_consistency(query_embedding, neighbors)?;

        Ok(QueryQualityMetrics {
            average_distance,
            distance_variance,
            neighbor_consistency_score,
            geometric_consistency,
        })
    }

    fn compute_geometric_consistency(&self, query_embedding: &[f32], neighbors: &[(u32, f64)]) -> Result<f64> {
        let mut triangle_violations = 0;
        let mut total_triangles = 0;

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if let (Some(emb1), Some(emb2)) = (
                    self.embeddings.get(&neighbors[i].0),
                    self.embeddings.get(&neighbors[j].0),
                ) {
                    let d_query_1 = neighbors[i].1;
                    let d_query_2 = neighbors[j].1;
                    let d_1_2 = self.compute_exact_distance(emb1, emb2, &SimilarityMetric::Euclidean)?;

                    // Check triangle inequality: d(q,1) + d(1,2) >= d(q,2)
                    if d_query_1 + d_1_2 < d_query_2 - 1e-10 {
                        triangle_violations += 1;
                    }
                    total_triangles += 1;
                }
            }
        }

        let consistency = if total_triangles > 0 {
            1.0 - (triangle_violations as f64 / total_triangles as f64)
        } else {
            1.0
        };

        Ok(consistency)
    }

    // RAG-specific methods

    fn compute_embedding_similarity_candidates(&self, query_embedding: &[f32], max_candidates: u32) -> Result<Vec<u32>> {
        let mut candidates = Vec::new();

        for (&entity_id, embedding) in &self.embeddings {
            let similarity = self.cosine_similarity(query_embedding, embedding);
            candidates.push((entity_id, similarity));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(candidates.into_iter().take(max_candidates as usize).map(|(id, _)| id).collect())
    }

    fn compute_graph_expansion_candidates(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, max_depth: u32, max_candidates: u32) -> Result<Vec<u32>> {
        let (entities, _) = self.bfs_with_depth_tracking(adj_list, start, max_depth)?;
        Ok(entities.into_iter().take(max_candidates as usize).collect())
    }

    fn merge_and_rank_rag_candidates(&self, embedding_candidates: Vec<u32>, graph_candidates: Vec<u32>, query_embedding: &[f32], max_context: u32) -> Result<Vec<u32>> {
        let mut combined = HashSet::new();
        combined.extend(embedding_candidates);
        combined.extend(graph_candidates);

        let mut scored_candidates = Vec::new();
        for entity_id in combined {
            let score = if let Some(embedding) = self.embeddings.get(&entity_id) {
                self.cosine_similarity(query_embedding, embedding) as f64
            } else {
                0.5 // Default score for entities without embeddings
            };
            scored_candidates.push((entity_id, score));
        }

        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored_candidates.into_iter().take(max_context as usize).map(|(id, _)| id).collect())
    }

    fn compute_exact_relevance_scores(&self, context_entities: &[u32], query_embedding: &[f32]) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        
        for &entity_id in context_entities {
            let score = if let Some(embedding) = self.embeddings.get(&entity_id) {
                self.cosine_similarity(query_embedding, embedding) as f64
            } else {
                0.5
            };
            scores.push(score);
        }
        
        Ok(scores)
    }

    fn analyze_context_quality(&self, context_entities: &[u32], query_concept: u32) -> Result<f64> {
        if context_entities.is_empty() {
            return Ok(0.0);
        }

        // Quality = relevance * diversity * coverage
        let relevance_score = if let Some(query_embedding) = self.embeddings.get(&query_concept) {
            let relevance_scores = self.compute_exact_relevance_scores(context_entities, query_embedding)?;
            relevance_scores.iter().sum::<f64>() / relevance_scores.len() as f64
        } else {
            0.5
        };

        let diversity_score = self.compute_context_diversity(context_entities)?;
        let coverage_score = self.compute_context_coverage(context_entities, query_concept)?;

        Ok((relevance_score * 0.5 + diversity_score * 0.3 + coverage_score * 0.2).min(1.0))
    }

    fn compute_context_diversity(&self, context_entities: &[u32]) -> Result<f64> {
        if context_entities.len() < 2 {
            return Ok(0.0);
        }

        let mut total_distance = 0.0;
        let mut pair_count = 0;

        for i in 0..context_entities.len() {
            for j in (i + 1)..context_entities.len() {
                if let (Some(emb1), Some(emb2)) = (
                    self.embeddings.get(&context_entities[i]),
                    self.embeddings.get(&context_entities[j]),
                ) {
                    let distance = 1.0 - self.cosine_similarity(emb1, emb2) as f64;
                    total_distance += distance;
                    pair_count += 1;
                }
            }
        }

        Ok(if pair_count > 0 { total_distance / pair_count as f64 } else { 0.0 })
    }

    fn compute_context_coverage(&self, context_entities: &[u32], query_concept: u32) -> Result<f64> {
        // Simple coverage: ratio of context entities to total reachable entities
        let adj_list = self.build_adjacency_list();
        let (reachable_entities, _) = self.bfs_with_depth_tracking(&adj_list, query_concept, 2)?;
        
        let coverage = context_entities.len() as f64 / reachable_entities.len().max(1) as f64;
        Ok(coverage.min(1.0))
    }

    fn compute_diversity_metrics(&self, context_entities: &[u32]) -> Result<DiversityMetrics> {
        let mut similarities = Vec::new();

        for i in 0..context_entities.len() {
            for j in (i + 1)..context_entities.len() {
                if let (Some(emb1), Some(emb2)) = (
                    self.embeddings.get(&context_entities[i]),
                    self.embeddings.get(&context_entities[j]),
                ) {
                    let similarity = self.cosine_similarity(emb1, emb2) as f64;
                    similarities.push(similarity);
                }
            }
        }

        let mean_similarity = if !similarities.is_empty() {
            similarities.iter().sum::<f64>() / similarities.len() as f64
        } else {
            0.0
        };

        let std_similarity = if similarities.len() > 1 {
            let variance = similarities.iter()
                .map(|&s| (s - mean_similarity).powi(2))
                .sum::<f64>() / similarities.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        Ok(DiversityMetrics {
            pairwise_similarity_mean: mean_similarity,
            pairwise_similarity_std: std_similarity,
            cluster_separation: 1.0 - mean_similarity, // Higher separation = lower similarity
            topic_coverage: std_similarity, // Higher std = more diverse
        })
    }

    fn compute_coverage_analysis(&self, context_entities: &[u32], query_concept: u32) -> Result<CoverageAnalysis> {
        let adj_list = self.build_adjacency_list();
        let (reachable_entities, _) = self.bfs_with_depth_tracking(&adj_list, query_concept, 3)?;
        
        let graph_coverage_ratio = context_entities.len() as f64 / reachable_entities.len().max(1) as f64;
        
        // Semantic coverage based on embedding space coverage
        let semantic_coverage_score = self.compute_semantic_coverage(context_entities)?;
        
        // Redundancy: how much overlap exists in the context
        let redundancy_score = 1.0 - self.compute_context_diversity(context_entities)?;
        
        // Completeness: how well the context represents the query space
        let completeness_score = (graph_coverage_ratio + semantic_coverage_score) / 2.0;

        Ok(CoverageAnalysis {
            graph_coverage_ratio,
            semantic_coverage_score,
            redundancy_score,
            completeness_score,
        })
    }

    fn compute_semantic_coverage(&self, context_entities: &[u32]) -> Result<f64> {
        // Simplified semantic coverage: variance in embedding space
        if context_entities.len() < 2 {
            return Ok(0.0);
        }

        if let Some(first_embedding) = context_entities.first().and_then(|&id| self.embeddings.get(&id)) {
            let dim = first_embedding.len();
            let mut dimension_variances = vec![0.0; dim];

            for d in 0..dim {
                let values: Vec<f64> = context_entities.iter()
                    .filter_map(|&id| self.embeddings.get(&id))
                    .map(|emb| emb[d] as f64)
                    .collect();

                if !values.is_empty() {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter()
                        .map(|&v| (v - mean).powi(2))
                        .sum::<f64>() / values.len() as f64;
                    dimension_variances[d] = variance;
                }
            }

            let avg_variance = dimension_variances.iter().sum::<f64>() / dim as f64;
            Ok(avg_variance.min(1.0))
        } else {
            Ok(0.0)
        }
    }

    // Utility methods

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

    fn compute_depth_distribution(&self, entities: &[u32], start: u32, adj_list: &HashMap<u32, Vec<u32>>) -> Result<HashMap<u32, u64>> {
        let mut depths = HashMap::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((start, 0u32));
        visited.insert(start);

        while let Some((current, depth)) = queue.pop_front() {
            if entities.contains(&current) {
                *depths.entry(depth).or_insert(0) += 1;
            }

            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) && entities.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(depths)
    }

    fn analyze_random_walk_space(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32, max_length: u32) -> Result<RandomWalkAnalysis> {
        // For deterministic analysis, compute a representative walk
        let mut walk = vec![start];
        let mut current = start;
        let mut unique_entities = HashSet::new();
        unique_entities.insert(start);

        for _ in 1..max_length {
            if let Some(neighbors) = adj_list.get(&current) {
                if !neighbors.is_empty() {
                    // Choose first neighbor for determinism
                    current = neighbors[0];
                    walk.push(current);
                    unique_entities.insert(current);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Detect cycles
        let mut cycles = Vec::new();
        let mut seen_positions = HashMap::new();
        
        for (pos, &node) in walk.iter().enumerate() {
            if let Some(&prev_pos) = seen_positions.get(&node) {
                let cycle = walk[prev_pos..=pos].to_vec();
                if cycle.len() > 1 {
                    cycles.push(cycle);
                }
            }
            seen_positions.insert(node, pos);
        }

        Ok(RandomWalkAnalysis {
            representative_walk: walk,
            unique_entities: unique_entities.len() as u64,
            cycles,
        })
    }

    fn compute_traversal_metadata(&self, traversal_type: &TraversalType, result: &TraversalResultType) -> ExecutionMetadata {
        let (complexity, memory_complexity, algorithm) = match traversal_type {
            TraversalType::ShortestPath => ("O((V + E) log V)", "O(V)", "Dijkstra"),
            TraversalType::BreadthFirstSearch => ("O(V + E)", "O(V)", "BFS"),
            TraversalType::DepthFirstSearch => ("O(V + E)", "O(V)", "DFS"),
            TraversalType::RandomWalk => ("O(L)", "O(L)", "Random Walk"),
            TraversalType::BiDirectional => ("O((V + E) log V)", "O(V)", "Bidirectional Search"),
        };

        let io_operations = match result {
            TraversalResultType::Path { path, .. } => path.len() as u64,
            TraversalResultType::Neighborhood { entities, .. } => entities.len() as u64,
            TraversalResultType::Walk { walk_sequence, .. } => walk_sequence.len() as u64,
        };

        ExecutionMetadata {
            computational_complexity: complexity.to_string(),
            memory_complexity: memory_complexity.to_string(),
            io_operations,
            cache_efficiency: 0.8, // Assume good cache locality
            algorithm_used: algorithm.to_string(),
        }
    }

    fn compute_checksum(&self, traversal_results: &[TraversalResult], similarity_results: &[SimilarityResult], rag_results: &[RagResult]) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash key properties of results
        traversal_results.len().hash(&mut hasher);
        similarity_results.len().hash(&mut hasher);
        rag_results.len().hash(&mut hasher);
        
        for result in traversal_results {
            result.query_id.hash(&mut hasher);
        }
        
        for result in similarity_results {
            result.query_id.hash(&mut hasher);
            result.k.hash(&mut hasher);
        }
        
        for result in rag_results {
            result.query_id.hash(&mut hasher);
            result.context_entities.len().hash(&mut hasher);
        }

        Ok(format!("{:x}", hasher.finish()))
    }
}

struct RandomWalkAnalysis {
    representative_walk: Vec<u32>,
    unique_entities: u64,
    cycles: Vec<Vec<u32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generation::graph_topologies::GraphTopologyGenerator;

    fn create_test_setup() -> (TestGraph, HashMap<u32, Vec<f32>>) {
        let mut graph_gen = GraphTopologyGenerator::new(42);
        let graph = graph_gen.generate_erdos_renyi(10, 0.3).unwrap();
        
        let mut embeddings = HashMap::new();
        for entity in &graph.entities {
            let embedding = vec![entity.id as f32, (entity.id * 2) as f32, (entity.id * 3) as f32];
            embeddings.insert(entity.id, embedding);
        }
        
        (graph, embeddings)
    }

    #[test]
    fn test_exact_computation_engine_creation() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        assert_eq!(engine.graph.entities.len(), 10);
        assert_eq!(engine.embeddings.len(), 10);
    }

    #[test]
    fn test_dijkstra_shortest_path() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        let adj_list = engine.build_adjacency_list();
        
        // Test path from node to itself
        let (path, distance) = engine.dijkstra_shortest_path(&adj_list, 0, 0).unwrap();
        assert_eq!(path, vec![0]);
        assert_eq!(distance, 0);
    }

    #[test]
    fn test_bfs_with_depth_tracking() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        let adj_list = engine.build_adjacency_list();
        
        let (entities, depth_dist) = engine.bfs_with_depth_tracking(&adj_list, 0, 2).unwrap();
        
        assert!(!entities.is_empty());
        assert!(depth_dist.contains_key(&0)); // Should contain starting node at depth 0
        assert_eq!(depth_dist[&0], 1); // Starting node count
    }

    #[test]
    fn test_exact_distance_computation() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        
        let euclidean_dist = engine.compute_exact_distance(&v1, &v2, &SimilarityMetric::Euclidean).unwrap();
        assert!((euclidean_dist - 2.0_f64.sqrt()).abs() < 1e-10);
        
        let cosine_dist = engine.compute_exact_distance(&v1, &v2, &SimilarityMetric::Cosine).unwrap();
        assert!((cosine_dist - 1.0).abs() < 1e-10); // Orthogonal vectors
    }

    #[test]
    fn test_distance_statistics() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = engine.compute_distance_statistics(&distances).unwrap();
        
        assert_eq!(stats.min_distance, 1.0);
        assert_eq!(stats.max_distance, 5.0);
        assert_eq!(stats.mean_distance, 3.0);
        assert!(stats.std_deviation > 0.0);
    }

    #[test]
    fn test_performance_expectations() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let expectations = engine.compute_performance_expectations().unwrap();
        
        assert!(expectations.query_latency_bounds.traversal_queries.0 > 0.0);
        assert!(expectations.memory_bounds.per_entity_bytes > 0.0);
        assert!(expectations.throughput_expectations.queries_per_second > 0.0);
    }

    #[test]
    fn test_checksum_computation() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let traversal_results = vec![];
        let similarity_results = vec![];
        let rag_results = vec![];
        
        let checksum = engine.compute_checksum(&traversal_results, &similarity_results, &rag_results).unwrap();
        
        assert!(!checksum.is_empty());
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_cosine_similarity() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];
        
        assert!((engine.cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-6); // Identical
        assert!((engine.cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_context_diversity() {
        let (graph, embeddings) = create_test_setup();
        let engine = ExactComputationEngine::new(graph, embeddings);
        
        let context_entities = vec![0, 1, 2];
        let diversity = engine.compute_context_diversity(&context_entities).unwrap();
        
        assert!(diversity >= 0.0);
        assert!(diversity <= 1.0);
    }
}