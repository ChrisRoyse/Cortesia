# Phase 2: Synthetic Data Generation Framework

## Overview

Phase 2 creates a comprehensive synthetic data generation system that produces controlled, deterministic datasets with known properties for testing all aspects of the LLMKG system. This framework ensures that every test has predictable, verifiable outcomes.

## Objectives

1. **Deterministic Data Generation**: Create reproducible datasets with known mathematical properties
2. **Comprehensive Coverage**: Generate data for all LLMKG features and use cases
3. **Scalable Generation**: Support datasets from tiny (100 entities) to massive (1M+ entities)
4. **Property Validation**: Ensure generated data has expected characteristics
5. **Performance Optimization**: Generate large datasets efficiently
6. **Golden Standard Creation**: Precompute expected outcomes for all test scenarios

## Detailed Implementation Plan

### 1. Graph Structure Generation

#### 1.1 Basic Graph Topologies
**File**: `tests/data_generation/graph_topologies.rs`

```rust
pub struct GraphTopologyGenerator {
    rng: DeterministicRng,
    properties: GraphProperties,
}

pub struct GraphProperties {
    pub entity_count: u64,
    pub average_degree: f64,
    pub clustering_coefficient: f64,
    pub diameter: u32,
    pub density: f64,
    pub connectivity: ConnectivityType,
}

pub enum ConnectivityType {
    Connected,
    Forest,
    Complete,
    Random,
    SmallWorld,
    ScaleFree,
    Bipartite,
}

impl GraphTopologyGenerator {
    pub fn generate_erdos_renyi(&mut self, n: u64, p: f64) -> Graph {
        // Erdős–Rényi random graph with predictable properties
        let mut graph = Graph::new();
        
        // Add entities with deterministic IDs
        for i in 0..n {
            let entity_id = EntityKey::from_hash(format!("entity_{}", i));
            graph.add_entity(entity_id, format!("Entity {}", i));
        }
        
        // Add edges with probability p
        for i in 0..n {
            for j in (i + 1)..n {
                if self.rng.gen::<f64>() < p {
                    let rel = Relationship::new(
                        format!("connects_{}_{}", i, j),
                        1.0,
                        RelationshipType::Undirected
                    );
                    graph.add_relationship(
                        EntityKey::from_hash(format!("entity_{}", i)),
                        EntityKey::from_hash(format!("entity_{}", j)),
                        rel
                    );
                }
            }
        }
        
        graph
    }
    
    pub fn generate_barabasi_albert(&mut self, n: u64, m: u32) -> Graph {
        // Preferential attachment model for scale-free networks
        // Mathematical guarantee: degree distribution follows power law
        // Expected: P(degree = k) ∝ k^(-3)
    }
    
    pub fn generate_watts_strogatz(&mut self, n: u64, k: u32, beta: f64) -> Graph {
        // Small-world network with known clustering and path length
        // Expected clustering: 3/4 * (1 - beta)
        // Expected path length: n/(2k) for beta = 0, ln(n)/ln(k) for beta = 1
    }
    
    pub fn generate_complete_graph(&mut self, n: u64) -> Graph {
        // Complete graph with all possible edges
        // Guaranteed properties:
        // - Degree of each node: n-1
        // - Total edges: n(n-1)/2
        // - Diameter: 1
        // - Clustering coefficient: 1.0
    }
    
    pub fn generate_tree(&mut self, n: u64, branching_factor: u32) -> Graph {
        // Tree structure with guaranteed properties:
        // - Exactly n-1 edges
        // - Diameter: 2 * log_b(n) where b is branching factor
        // - No cycles
        // - Clustering coefficient: 0.0
    }
}
```

#### 1.2 Specialized Knowledge Graph Structures
**File**: `tests/data_generation/knowledge_graphs.rs`

```rust
pub struct KnowledgeGraphGenerator {
    rng: DeterministicRng,
    ontology: Ontology,
}

pub struct Ontology {
    pub entity_types: Vec<EntityType>,
    pub relationship_types: Vec<RelationshipType>,
    pub hierarchies: Vec<Hierarchy>,
    pub constraints: Vec<Constraint>,
}

impl KnowledgeGraphGenerator {
    pub fn generate_academic_papers(&mut self, paper_count: u64) -> Graph {
        // Generate academic paper citation network
        // Entities: Papers, Authors, Venues, Topics
        // Relationships: Cites, AuthoredBy, PublishedIn, HasTopic
        // Expected properties:
        // - Citation power law distribution
        // - Temporal ordering (papers cite older papers)
        // - Co-authorship clustering
        
        let mut graph = Graph::new();
        
        // Generate papers with temporal ordering
        for i in 0..paper_count {
            let paper_id = EntityKey::from_hash(format!("paper_{}", i));
            let paper = Entity::new(
                paper_id,
                format!("Paper {}", i),
                EntityType::Paper
            );
            
            // Add metadata with known values
            paper.add_attribute("year", (2000 + (i % 24)).to_string());
            paper.add_attribute("citation_count", self.calculate_expected_citations(i));
            
            graph.add_entity(paper);
        }
        
        // Generate authors with realistic distribution
        let author_count = (paper_count as f64 * 0.3) as u64; // 30% unique authors
        for i in 0..author_count {
            let author_id = EntityKey::from_hash(format!("author_{}", i));
            let productivity = self.generate_author_productivity(i);
            
            let author = Entity::new(
                author_id,
                format!("Author {}", i),
                EntityType::Author
            );
            author.add_attribute("h_index", productivity.h_index.to_string());
            author.add_attribute("paper_count", productivity.paper_count.to_string());
            
            graph.add_entity(author);
        }
        
        // Generate citations with power law distribution
        self.generate_citation_network(&mut graph, paper_count);
        
        graph
    }
    
    pub fn generate_social_network(&mut self, user_count: u64) -> Graph {
        // Generate social network with known community structure
        // Expected properties:
        // - High clustering coefficient (0.3-0.7)
        // - Small world property (avg path length ~ log(n))
        // - Community structure with modularity > 0.3
        // - Degree distribution follows power law
    }
    
    pub fn generate_biological_pathway(&mut self, protein_count: u64) -> Graph {
        // Generate biological pathway network
        // Entities: Proteins, Genes, Pathways, Compounds
        // Relationships: Interacts, Regulates, Catalyzes, PartOf
        // Expected properties:
        // - Scale-free degree distribution
        // - High clustering (biological modules)
        // - Hierarchical organization
    }
}
```

#### 1.3 Multi-Scale Graph Generation
**File**: `tests/data_generation/multi_scale.rs`

```rust
pub struct MultiScaleGenerator {
    rng: DeterministicRng,
}

impl MultiScaleGenerator {
    pub fn generate_hierarchical_graph(&mut self, levels: u32, nodes_per_level: Vec<u64>) -> Graph {
        // Generate hierarchical graph with known structure
        // Level 0: Individual entities
        // Level 1: Local clusters
        // Level 2: Regional groups
        // Level 3: Global communities
        
        let mut graph = Graph::new();
        let mut level_nodes = Vec::new();
        
        for (level, &count) in nodes_per_level.iter().enumerate() {
            let mut nodes = Vec::new();
            
            for i in 0..count {
                let node_id = EntityKey::from_hash(format!("L{}_N{}", level, i));
                let node = Entity::new(
                    node_id,
                    format!("Level {} Node {}", level, i),
                    EntityType::Hierarchical
                );
                node.add_attribute("level", level.to_string());
                node.add_attribute("position", i.to_string());
                
                graph.add_entity(node);
                nodes.push(node_id);
            }
            
            // Connect to previous level if exists
            if level > 0 {
                self.connect_hierarchical_levels(
                    &mut graph,
                    &level_nodes[level - 1],
                    &nodes
                );
            }
            
            level_nodes.push(nodes);
        }
        
        graph
    }
    
    pub fn generate_fractal_graph(&mut self, iterations: u32, base_pattern: Graph) -> Graph {
        // Generate self-similar fractal graph structure
        // Each iteration replaces nodes with scaled copies of base pattern
        // Expected properties:
        // - Self-similarity at multiple scales
        // - Predictable scaling laws
        // - Known fractal dimension
    }
}
```

### 2. Vector Embedding Generation

#### 2.1 Controlled Similarity Embeddings
**File**: `tests/data_generation/embeddings.rs`

```rust
pub struct EmbeddingGenerator {
    rng: DeterministicRng,
    dimension: usize,
}

impl EmbeddingGenerator {
    pub fn generate_clustered_embeddings(&mut self, 
        cluster_specs: Vec<ClusterSpec>) -> Vec<(EntityKey, Vec<f32>)> {
        // Generate embeddings with known cluster structure
        let mut embeddings = Vec::new();
        
        for cluster_spec in cluster_specs {
            let cluster_center = self.generate_random_unit_vector();
            
            for i in 0..cluster_spec.size {
                let entity_id = EntityKey::from_hash(
                    format!("cluster_{}_entity_{}", cluster_spec.id, i)
                );
                
                // Generate point within cluster radius
                let embedding = self.generate_point_in_sphere(
                    &cluster_center,
                    cluster_spec.radius
                );
                
                embeddings.push((entity_id, embedding));
            }
        }
        
        embeddings
    }
    
    pub fn generate_distance_controlled_embeddings(&mut self, 
        pairs: Vec<(EntityKey, EntityKey, f32)>) -> HashMap<EntityKey, Vec<f32>> {
        // Generate embeddings where specific pairs have exact distances
        let mut embeddings = HashMap::new();
        
        for (entity1, entity2, target_distance) in pairs {
            if !embeddings.contains_key(&entity1) {
                embeddings.insert(entity1, self.generate_random_unit_vector());
            }
            
            let embedding1 = embeddings[&entity1].clone();
            let embedding2 = self.generate_point_at_distance(&embedding1, target_distance);
            embeddings.insert(entity2, embedding2);
        }
        
        embeddings
    }
    
    pub fn generate_hierarchical_embeddings(&mut self, 
        hierarchy: &HierarchicalStructure) -> HashMap<EntityKey, Vec<f32>> {
        // Generate embeddings that reflect hierarchical relationships
        // Parent embeddings are weighted averages of children
        // Sibling embeddings are clustered together
        // Expected property: TreeDistance(A,B) ∝ EuclideanDistance(emb(A), emb(B))
    }
    
    fn generate_point_at_distance(&mut self, reference: &[f32], distance: f32) -> Vec<f32> {
        // Generate a point at exactly the specified distance from reference
        let random_direction = self.generate_random_unit_vector();
        let mut result = vec![0.0; self.dimension];
        
        for i in 0..self.dimension {
            result[i] = reference[i] + distance * random_direction[i];
        }
        
        // Verify distance is correct (within floating point precision)
        let actual_distance = self.euclidean_distance(reference, &result);
        assert!((actual_distance - distance).abs() < 1e-6);
        
        result
    }
}

pub struct ClusterSpec {
    pub id: u32,
    pub size: u64,
    pub radius: f32,
    pub center: Option<Vec<f32>>,
}
```

#### 2.2 Quantization Test Data
**File**: `tests/data_generation/quantization_data.rs`

```rust
pub struct QuantizationDataGenerator {
    rng: DeterministicRng,
}

impl QuantizationDataGenerator {
    pub fn generate_product_quantization_test_data(&mut self, 
        vector_count: u64, 
        dimension: usize,
        codebook_size: usize) -> QuantizationTestSet {
        
        // Generate vectors specifically for testing PQ accuracy
        let mut original_vectors = Vec::new();
        let mut expected_approximations = Vec::new();
        let mut expected_distances = Vec::new();
        
        for i in 0..vector_count {
            let vector = self.generate_structured_vector(dimension, i);
            let approximation = self.compute_expected_pq_approximation(
                &vector, dimension, codebook_size
            );
            
            original_vectors.push(vector.clone());
            expected_approximations.push(approximation);
            
            // Precompute expected distances for similarity search validation
            for j in 0..i {
                let distance = self.euclidean_distance(&vector, &original_vectors[j as usize]);
                let approx_distance = self.euclidean_distance(
                    &approximation, 
                    &expected_approximations[j as usize]
                );
                
                expected_distances.push(DistanceComparison {
                    vector1: i,
                    vector2: j,
                    original_distance: distance,
                    approximated_distance: approx_distance,
                    compression_error: (approx_distance - distance).abs(),
                });
            }
        }
        
        QuantizationTestSet {
            original_vectors,
            expected_approximations,
            expected_distances,
            compression_ratio: self.calculate_compression_ratio(dimension, codebook_size),
        }
    }
    
    pub fn generate_simd_test_vectors(&mut self, count: u64) -> SimdTestSet {
        // Generate vectors specifically for testing SIMD operations
        // Include edge cases: zeros, infinities, NaN handling
        // Include alignment test cases for different SIMD widths
    }
}

pub struct QuantizationTestSet {
    pub original_vectors: Vec<Vec<f32>>,
    pub expected_approximations: Vec<Vec<f32>>,
    pub expected_distances: Vec<DistanceComparison>,
    pub compression_ratio: f64,
}

pub struct DistanceComparison {
    pub vector1: u64,
    pub vector2: u64,
    pub original_distance: f32,
    pub approximated_distance: f32,
    pub compression_error: f32,
}
```

### 3. Query Pattern Generation

#### 3.1 Graph Traversal Queries
**File**: `tests/data_generation/query_patterns.rs`

```rust
pub struct QueryPatternGenerator {
    graph: Graph,
    rng: DeterministicRng,
}

impl QueryPatternGenerator {
    pub fn generate_traversal_queries(&mut self) -> Vec<TraversalQuery> {
        let mut queries = Vec::new();
        
        // Single-hop queries
        for entity in self.graph.entities().take(100) {
            let query = TraversalQuery {
                start_entity: entity.key(),
                max_depth: 1,
                relationship_filter: None,
                expected_result_count: self.count_neighbors(entity.key()),
                expected_entities: self.get_neighbors(entity.key()),
            };
            queries.push(query);
        }
        
        // Multi-hop queries with known paths
        for (start, end) in self.select_entity_pairs_with_known_paths(50) {
            let shortest_path = self.compute_shortest_path(start, end);
            let query = TraversalQuery {
                start_entity: start,
                target_entity: Some(end),
                max_depth: shortest_path.len() as u32,
                expected_path: Some(shortest_path),
                expected_distance: Some(shortest_path.len() as u32),
            };
            queries.push(query);
        }
        
        queries
    }
    
    pub fn generate_rag_queries(&mut self) -> Vec<RagQuery> {
        // Generate Graph RAG queries with known expected contexts
        let mut queries = Vec::new();
        
        for concept in self.select_representative_concepts(20) {
            let context_entities = self.compute_expected_context(concept, 2, 10);
            let relevance_scores = self.compute_relevance_scores(&context_entities, concept);
            
            let query = RagQuery {
                query_concept: concept,
                max_context_size: 10,
                max_depth: 2,
                expected_context: context_entities,
                expected_relevance_scores: relevance_scores,
                expected_context_quality: self.compute_context_quality(&context_entities),
            };
            queries.push(query);
        }
        
        queries
    }
    
    pub fn generate_similarity_queries(&mut self, embeddings: &HashMap<EntityKey, Vec<f32>>) 
        -> Vec<SimilarityQuery> {
        let mut queries = Vec::new();
        
        for (entity, embedding) in embeddings.iter().take(50) {
            // Find k nearest neighbors using brute force (ground truth)
            let mut neighbors = Vec::new();
            for (other_entity, other_embedding) in embeddings.iter() {
                if entity != other_entity {
                    let distance = self.euclidean_distance(embedding, other_embedding);
                    neighbors.push((other_entity.clone(), distance));
                }
            }
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            for k in [1, 5, 10, 20] {
                let query = SimilarityQuery {
                    query_entity: entity.clone(),
                    k,
                    expected_neighbors: neighbors.iter().take(k).cloned().collect(),
                    expected_distances: neighbors.iter().take(k).map(|(_, d)| *d).collect(),
                };
                queries.push(query);
            }
        }
        
        queries
    }
}

pub struct TraversalQuery {
    pub start_entity: EntityKey,
    pub target_entity: Option<EntityKey>,
    pub max_depth: u32,
    pub relationship_filter: Option<String>,
    pub expected_result_count: u64,
    pub expected_entities: Vec<EntityKey>,
    pub expected_path: Option<Vec<EntityKey>>,
    pub expected_distance: Option<u32>,
}

pub struct RagQuery {
    pub query_concept: EntityKey,
    pub max_context_size: u32,
    pub max_depth: u32,
    pub expected_context: Vec<EntityKey>,
    pub expected_relevance_scores: Vec<f32>,
    pub expected_context_quality: f32,
}

pub struct SimilarityQuery {
    pub query_entity: EntityKey,
    pub k: usize,
    pub expected_neighbors: Vec<(EntityKey, f32)>,
    pub expected_distances: Vec<f32>,
}
```

### 4. Streaming Data Generation

#### 4.1 Temporal Data Streams
**File**: `tests/data_generation/streaming_data.rs`

```rust
pub struct StreamingDataGenerator {
    rng: DeterministicRng,
    base_graph: Graph,
}

impl StreamingDataGenerator {
    pub fn generate_temporal_updates(&mut self, duration: Duration, updates_per_second: f64) 
        -> Vec<TimestampedUpdate> {
        let mut updates = Vec::new();
        let total_updates = (duration.as_secs_f64() * updates_per_second) as u64;
        
        for i in 0..total_updates {
            let timestamp = Duration::from_secs_f64(i as f64 / updates_per_second);
            let update = self.generate_realistic_update(timestamp, i);
            updates.push(TimestampedUpdate { timestamp, update });
        }
        
        // Ensure updates have expected properties:
        // - Temporal locality (related updates cluster in time)
        // - Entity popularity follows power law
        // - Update types have realistic distributions
        
        updates
    }
    
    pub fn generate_batch_updates(&mut self, batch_count: u32, batch_size: u32) 
        -> Vec<UpdateBatch> {
        let mut batches = Vec::new();
        
        for batch_id in 0..batch_count {
            let mut updates = Vec::new();
            
            for update_id in 0..batch_size {
                let update = self.generate_deterministic_update(batch_id, update_id);
                updates.push(update);
            }
            
            let batch = UpdateBatch {
                id: batch_id,
                updates,
                expected_processing_time: self.estimate_processing_time(batch_size),
                expected_memory_impact: self.estimate_memory_impact(&updates),
            };
            batches.push(batch);
        }
        
        batches
    }
    
    fn generate_realistic_update(&mut self, timestamp: Duration, sequence: u64) -> GraphUpdate {
        // Generate updates that follow realistic patterns:
        // - 70% entity updates (attribute changes)
        // - 20% relationship additions
        // - 10% entity/relationship deletions
        
        let update_type = match self.rng.gen_range(0..100) {
            0..70 => UpdateType::EntityUpdate,
            70..90 => UpdateType::RelationshipAdd,
            _ => UpdateType::Delete,
        };
        
        match update_type {
            UpdateType::EntityUpdate => {
                let entity = self.select_entity_by_popularity();
                GraphUpdate::UpdateEntity {
                    entity_key: entity,
                    new_attributes: self.generate_attribute_updates(),
                    expected_index_updates: self.compute_expected_index_changes(&entity),
                }
            },
            UpdateType::RelationshipAdd => {
                let (source, target) = self.select_entity_pair_for_relationship();
                GraphUpdate::AddRelationship {
                    source,
                    target,
                    relationship: self.generate_realistic_relationship(),
                    expected_graph_changes: self.compute_expected_graph_changes(source, target),
                }
            },
            UpdateType::Delete => {
                // Generate deletion with known cascading effects
                let entity = self.select_entity_for_deletion();
                GraphUpdate::DeleteEntity {
                    entity_key: entity,
                    expected_cascade_deletes: self.compute_cascade_effects(entity),
                    expected_index_removals: self.compute_index_removals(entity),
                }
            }
        }
    }
}

pub struct TimestampedUpdate {
    pub timestamp: Duration,
    pub update: GraphUpdate,
}

pub struct UpdateBatch {
    pub id: u32,
    pub updates: Vec<GraphUpdate>,
    pub expected_processing_time: Duration,
    pub expected_memory_impact: i64, // Bytes change (can be negative)
}

pub enum GraphUpdate {
    AddEntity {
        entity: Entity,
        expected_memory_usage: u64,
    },
    UpdateEntity {
        entity_key: EntityKey,
        new_attributes: HashMap<String, String>,
        expected_index_updates: Vec<IndexUpdate>,
    },
    DeleteEntity {
        entity_key: EntityKey,
        expected_cascade_deletes: Vec<EntityKey>,
        expected_index_removals: Vec<IndexRemoval>,
    },
    AddRelationship {
        source: EntityKey,
        target: EntityKey,
        relationship: Relationship,
        expected_graph_changes: GraphChangeSet,
    },
}
```

### 5. Federation Test Data

#### 5.1 Multi-Database Scenarios
**File**: `tests/data_generation/federation_data.rs`

```rust
pub struct FederationDataGenerator {
    rng: DeterministicRng,
}

impl FederationDataGenerator {
    pub fn generate_distributed_graph(&mut self, 
        database_count: u32, 
        overlap_percentage: f64) -> Vec<DatabaseShard> {
        let mut shards = Vec::new();
        let total_entities = 10000u64;
        let entities_per_db = total_entities / database_count as u64;
        let overlap_entities = (entities_per_db as f64 * overlap_percentage) as u64;
        
        for db_id in 0..database_count {
            let mut shard = DatabaseShard {
                id: db_id,
                entities: HashSet::new(),
                relationships: Vec::new(),
                metadata: DatabaseMetadata::new(),
            };
            
            // Add unique entities
            let start_id = db_id as u64 * entities_per_db;
            for i in start_id..(start_id + entities_per_db - overlap_entities) {
                let entity_key = EntityKey::from_hash(format!("unique_entity_{}", i));
                shard.entities.insert(entity_key);
            }
            
            // Add overlapping entities with previous database
            if db_id > 0 {
                let prev_start = (db_id - 1) as u64 * entities_per_db;
                for i in 0..overlap_entities {
                    let entity_key = EntityKey::from_hash(
                        format!("overlap_entity_{}", prev_start + entities_per_db - overlap_entities + i)
                    );
                    shard.entities.insert(entity_key);
                }
            }
            
            // Generate cross-database relationships
            shard.relationships = self.generate_cross_db_relationships(&shard.entities, db_id);
            
            // Set expected query routing information
            shard.metadata.expected_query_routes = self.compute_query_routes(db_id, database_count);
            
            shards.push(shard);
        }
        
        shards
    }
    
    pub fn generate_federation_queries(&mut self, shards: &[DatabaseShard]) 
        -> Vec<FederationQuery> {
        let mut queries = Vec::new();
        
        // Single-database queries
        for shard in shards {
            for entity in shard.entities.iter().take(5) {
                queries.push(FederationQuery {
                    query_type: FederationQueryType::SingleDatabase,
                    target_entity: *entity,
                    expected_databases: vec![shard.id],
                    expected_network_calls: 1,
                    expected_result_merge_complexity: MergeComplexity::None,
                });
            }
        }
        
        // Cross-database queries
        for i in 0..shards.len() {
            for j in (i + 1)..shards.len() {
                let entity1 = shards[i].entities.iter().next().unwrap();
                let entity2 = shards[j].entities.iter().next().unwrap();
                
                queries.push(FederationQuery {
                    query_type: FederationQueryType::CrossDatabase,
                    start_entity: *entity1,
                    target_entity: *entity2,
                    expected_databases: vec![shards[i].id, shards[j].id],
                    expected_network_calls: 2,
                    expected_result_merge_complexity: MergeComplexity::PathMerging,
                    expected_coordination_overhead: self.estimate_coordination_overhead(2),
                });
            }
        }
        
        // Multi-database aggregation queries
        queries.push(FederationQuery {
            query_type: FederationQueryType::Aggregation,
            aggregation_type: AggregationType::EntityCount,
            expected_databases: shards.iter().map(|s| s.id).collect(),
            expected_network_calls: shards.len(),
            expected_result_merge_complexity: MergeComplexity::Aggregation,
            expected_final_result: self.compute_expected_aggregation(shards),
        });
        
        queries
    }
}

pub struct DatabaseShard {
    pub id: u32,
    pub entities: HashSet<EntityKey>,
    pub relationships: Vec<CrossDbRelationship>,
    pub metadata: DatabaseMetadata,
}

pub struct DatabaseMetadata {
    pub entity_count: u64,
    pub relationship_count: u64,
    pub expected_query_routes: HashMap<EntityKey, Vec<u32>>,
    pub performance_characteristics: PerformanceProfile,
}

pub struct FederationQuery {
    pub query_type: FederationQueryType,
    pub start_entity: Option<EntityKey>,
    pub target_entity: EntityKey,
    pub expected_databases: Vec<u32>,
    pub expected_network_calls: u32,
    pub expected_result_merge_complexity: MergeComplexity,
    pub expected_coordination_overhead: Duration,
    pub expected_final_result: Option<QueryResult>,
}

pub enum FederationQueryType {
    SingleDatabase,
    CrossDatabase,
    Aggregation,
    DistributedRAG,
}

pub enum MergeComplexity {
    None,
    SimpleUnion,
    PathMerging,
    Aggregation,
    ComplexJoin,
}
```

### 6. Expected Outcome Computation

#### 6.1 Golden Standard Generation
**File**: `tests/data_generation/golden_standards.rs`

```rust
pub struct GoldenStandardGenerator {
    computation_engine: ExactComputationEngine,
}

impl GoldenStandardGenerator {
    pub fn compute_all_expected_outcomes(&mut self, 
        dataset: &TestDataset) -> GoldenStandards {
        
        let mut standards = GoldenStandards::new();
        
        // Graph traversal outcomes
        standards.traversal_results = self.compute_traversal_outcomes(&dataset.graph, &dataset.traversal_queries);
        
        // Similarity search outcomes
        standards.similarity_results = self.compute_similarity_outcomes(&dataset.embeddings, &dataset.similarity_queries);
        
        // RAG outcomes
        standards.rag_results = self.compute_rag_outcomes(&dataset.graph, &dataset.embeddings, &dataset.rag_queries);
        
        // Performance outcomes
        standards.performance_expectations = self.compute_performance_expectations(&dataset);
        
        // Federation outcomes
        standards.federation_results = self.compute_federation_outcomes(&dataset.federation_shards, &dataset.federation_queries);
        
        standards
    }
    
    fn compute_traversal_outcomes(&mut self, 
        graph: &Graph, 
        queries: &[TraversalQuery]) -> Vec<TraversalResult> {
        
        queries.iter().map(|query| {
            let result = match query.target_entity {
                Some(target) => {
                    // Shortest path computation
                    let path = self.dijkstra_shortest_path(graph, query.start_entity, target);
                    TraversalResult::Path {
                        path: path.clone(),
                        distance: path.len() as u32,
                        total_entities: path.len() as u64,
                    }
                },
                None => {
                    // Neighborhood exploration
                    let entities = self.breadth_first_search(graph, query.start_entity, query.max_depth);
                    TraversalResult::Neighborhood {
                        entities: entities.clone(),
                        total_entities: entities.len() as u64,
                        depth_distribution: self.compute_depth_distribution(&entities, query.start_entity),
                    }
                }
            };
            
            result
        }).collect()
    }
    
    fn compute_similarity_outcomes(&mut self, 
        embeddings: &HashMap<EntityKey, Vec<f32>>, 
        queries: &[SimilarityQuery]) -> Vec<SimilarityResult> {
        
        queries.iter().map(|query| {
            let query_embedding = &embeddings[&query.query_entity];
            let mut candidates = Vec::new();
            
            // Compute exact distances to all other entities
            for (entity_key, embedding) in embeddings.iter() {
                if *entity_key != query.query_entity {
                    let distance = self.exact_euclidean_distance(query_embedding, embedding);
                    candidates.push((*entity_key, distance));
                }
            }
            
            // Sort by distance and take top k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let top_k = candidates.into_iter().take(query.k).collect();
            
            SimilarityResult {
                query_entity: query.query_entity,
                k: query.k,
                nearest_neighbors: top_k,
                average_distance: self.compute_average_distance(&query_embedding, embeddings),
                query_quality_score: self.compute_query_quality(&query_embedding, embeddings),
            }
        }).collect()
    }
    
    fn compute_rag_outcomes(&mut self, 
        graph: &Graph, 
        embeddings: &HashMap<EntityKey, Vec<f32>>, 
        queries: &[RagQuery]) -> Vec<RagResult> {
        
        queries.iter().map(|query| {
            // Step 1: Embedding-based retrieval
            let similarity_candidates = self.similarity_search(
                embeddings, 
                &query.query_concept, 
                query.max_context_size * 2
            );
            
            // Step 2: Graph-based expansion
            let mut context_entities = HashSet::new();
            for (entity, _score) in similarity_candidates.iter().take(query.max_context_size as usize / 2) {
                context_entities.insert(*entity);
                
                // Expand through graph relationships
                let neighbors = self.get_k_hop_neighbors(graph, *entity, query.max_depth);
                context_entities.extend(neighbors);
            }
            
            // Step 3: Context ranking and filtering
            let ranked_context = self.rank_context_entities(
                &context_entities, 
                &query.query_concept, 
                embeddings, 
                graph
            );
            
            let final_context = ranked_context.into_iter()
                .take(query.max_context_size as usize)
                .collect();
            
            // Step 4: Context quality assessment
            let context_quality = self.assess_context_quality(&final_context, &query.query_concept, graph);
            
            RagResult {
                query_concept: query.query_concept,
                context_entities: final_context,
                context_quality,
                relevance_scores: self.compute_relevance_scores(&final_context, &query.query_concept),
                coverage_score: self.compute_coverage_score(&final_context, &query.query_concept, graph),
                diversity_score: self.compute_diversity_score(&final_context, embeddings),
            }
        }).collect()
    }
}

pub struct GoldenStandards {
    pub traversal_results: Vec<TraversalResult>,
    pub similarity_results: Vec<SimilarityResult>,
    pub rag_results: Vec<RagResult>,
    pub performance_expectations: PerformanceExpectations,
    pub federation_results: Vec<FederationResult>,
    pub checksum: String, // Overall checksum for integrity verification
}

pub struct PerformanceExpectations {
    pub query_latency_bounds: (Duration, Duration), // (min, max)
    pub memory_usage_bounds: (u64, u64),
    pub throughput_expectations: ThroughputExpectations,
    pub compression_expectations: CompressionExpectations,
}
```

## Implementation Strategy

### Week 1: Core Generators
**Days 1-2**: Graph topology generators (Erdős–Rényi, Barabási–Albert, Watts–Strogatz)
**Days 3-4**: Vector embedding generators with controlled similarity
**Days 5**: Query pattern generators for basic scenarios
**Weekend**: Integration and validation of basic generators

### Week 2: Advanced Scenarios  
**Days 6-7**: Knowledge graph generators (academic, social, biological)
**Days 8-9**: Streaming data and federation scenarios
**Days 10**: Golden standard computation system
**Weekend**: Comprehensive testing and documentation

## Validation Framework

### Data Quality Validation
Each generated dataset must pass rigorous validation:

1. **Mathematical Property Verification**: 
   - Graph properties match theoretical expectations
   - Embedding distances are geometrically consistent
   - Statistical distributions follow specified models

2. **Determinism Verification**:
   - Identical inputs produce identical outputs
   - Cross-platform consistency
   - Reproducibility across multiple runs

3. **Expected Outcome Accuracy**:
   - Golden standards are mathematically correct
   - All precomputed results are verifiable
   - Error bounds are clearly defined

### Performance Benchmarks
- **Generation Speed**: All datasets generate within estimated time
- **Memory Efficiency**: Generation uses minimal memory overhead
- **Storage Optimization**: Generated data compresses efficiently

## Success Criteria

### Functional Requirements
- ✅ Generate all required dataset types and sizes
- ✅ Produce deterministic, reproducible results
- ✅ Generate accurate golden standards for all queries
- ✅ Support scalable generation from tiny to massive datasets

### Quality Requirements
- ✅ All generated data has expected mathematical properties
- ✅ Golden standards are verifiably correct
- ✅ Data generation is well-documented and maintainable
- ✅ Generation framework is extensible for new scenarios

### Performance Requirements
- ✅ Generate datasets within time and memory constraints
- ✅ Support parallel generation for large datasets
- ✅ Efficient caching and reuse of generated data

## Integration with Testing Pipeline

This synthetic data generation framework provides the foundation for all subsequent testing phases:
- **Phase 3 (Unit Tests)**: Small, focused datasets for individual component testing
- **Phase 4 (Integration Tests)**: Medium-scale datasets for multi-component validation  
- **Phase 5 (E2E Simulation)**: Large, realistic datasets for complete workflow testing
- **Phase 6 (Performance Testing)**: Massive datasets for scalability and stress testing

The deterministic nature ensures that all testing phases can rely on predictable, verifiable outcomes.