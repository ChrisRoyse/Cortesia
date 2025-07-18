//! Federation Test Data Generation
//! 
//! Generates test data for multi-database federation scenarios with cross-DB queries.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use crate::data_generation::{
    TestGraph, TestEntity, TestEdge, GraphProperties,
    EmbeddingTestSet, TraversalQuery, RagQuery, SimilarityQuery,
    ComprehensiveDataGenerator, GenerationParameters
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeSet};

/// Federation data generator for multi-database scenarios
pub struct FederationDataGenerator {
    rng: DeterministicRng,
    base_generator: ComprehensiveDataGenerator,
    federation_config: FederationConfig,
    database_schemas: HashMap<String, DatabaseSchema>,
}

/// Configuration for federation scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    pub database_count: u32,
    pub entities_per_database: u64,
    pub cross_db_connection_probability: f64,
    pub schema_overlap_ratio: f64,
    pub data_distribution_strategy: DataDistributionStrategy,
    pub consistency_requirements: ConsistencyRequirements,
    pub replication_factor: f64,
}

/// Strategy for distributing data across databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataDistributionStrategy {
    Random,
    Geographic { regions: Vec<String> },
    Hierarchical { levels: u32 },
    Semantic { domains: Vec<String> },
    Temporal { time_partitions: u32 },
    Hybrid { strategies: Vec<DataDistributionStrategy> },
}

/// Consistency requirements across federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyRequirements {
    pub eventual_consistency_delay: u64, // milliseconds
    pub strong_consistency_entities: Vec<String>,
    pub conflict_resolution_strategy: ConflictResolutionStrategy,
    pub replication_consistency: ReplicationConsistency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    LastWriteWins,
    FirstWriteWins,
    MergeConflicts,
    ManualResolution,
    VectorClocks,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationConsistency {
    Strong,
    Eventual,
    Causal,
    Session,
}

/// Schema definition for a federated database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    pub database_id: String,
    pub entity_types: HashSet<String>,
    pub relationship_types: HashSet<String>,
    pub unique_attributes: HashSet<String>,
    pub shared_attributes: HashSet<String>,
    pub embedding_dimensions: Vec<usize>,
    pub indexing_strategy: IndexingStrategy,
    pub sharding_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStrategy {
    BTree,
    Hash,
    Vector { algorithm: VectorIndexAlgorithm },
    Composite { strategies: Vec<IndexingStrategy> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorIndexAlgorithm {
    HNSW { m: u32, ef_construction: u32 },
    IVF { clusters: u32 },
    LSH { hash_tables: u32, hash_length: u32 },
    Product { subvectors: u32 },
}

/// Complete federation test dataset
#[derive(Debug, Clone)]
pub struct FederationTestDataset {
    pub databases: HashMap<String, DatabaseInstance>,
    pub cross_db_references: Vec<CrossDatabaseReference>,
    pub federation_queries: Vec<FederationQuery>,
    pub consistency_test_cases: Vec<ConsistencyTestCase>,
    pub performance_expectations: FederationPerformanceExpectations,
    pub metadata: FederationMetadata,
}

/// Individual database instance in the federation
#[derive(Debug, Clone)]
pub struct DatabaseInstance {
    pub database_id: String,
    pub graph: TestGraph,
    pub embeddings: HashMap<String, EmbeddingTestSet>,
    pub local_queries: Vec<LocalQuery>,
    pub replication_targets: Vec<String>,
    pub partition_info: PartitionInfo,
    pub performance_characteristics: DatabasePerformanceCharacteristics,
}

/// Reference between entities in different databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseReference {
    pub reference_id: u64,
    pub source_db: String,
    pub source_entity: u32,
    pub target_db: String,
    pub target_entity: u32,
    pub reference_type: CrossReferenceType,
    pub consistency_requirements: ConsistencyLevel,
    pub resolution_strategy: ResolutionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossReferenceType {
    OwnershipLink,
    DependencyLink,
    EquivalenceLink,
    HierarchyLink,
    TemporalLink,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Weak,
    Session,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    Eager,
    Lazy,
    OnDemand,
    Cached,
}

/// Query that spans multiple databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationQuery {
    pub query_id: u64,
    pub query_type: FederationQueryType,
    pub participating_databases: Vec<String>,
    pub expected_result_distribution: HashMap<String, u64>, // db_id -> expected_count
    pub performance_expectations: QueryPerformanceExpectations,
    pub consistency_requirements: QueryConsistencyRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationQueryType {
    CrossDatabaseJoin {
        join_conditions: Vec<JoinCondition>,
        join_strategy: JoinStrategy,
    },
    DistributedAggregation {
        aggregation_functions: Vec<AggregationFunction>,
        grouping_keys: Vec<String>,
    },
    GlobalSearch {
        search_criteria: SearchCriteria,
        ranking_strategy: RankingStrategy,
    },
    ConsistencyCheck {
        entity_references: Vec<CrossDatabaseReference>,
        consistency_level: ConsistencyLevel,
    },
    FederatedRAG {
        query_concept: u32,
        databases_to_search: Vec<String>,
        result_merging_strategy: ResultMergingStrategy,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinCondition {
    pub left_db: String,
    pub left_attribute: String,
    pub right_db: String,
    pub right_attribute: String,
    pub join_type: JoinType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    Cross,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinStrategy {
    NestedLoop,
    HashJoin,
    SortMerge,
    Broadcast,
    Shuffle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationFunction {
    pub function_type: AggregationType,
    pub attribute: String,
    pub database_scope: AggregationScope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Min,
    Max,
    StandardDeviation,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationScope {
    Global,
    PerDatabase,
    PerPartition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCriteria {
    pub entity_types: Vec<String>,
    pub attribute_filters: HashMap<String, String>,
    pub embedding_similarity: Option<EmbeddingSearchCriteria>,
    pub result_limit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSearchCriteria {
    pub query_vector: Vec<f64>,
    pub similarity_threshold: f64,
    pub metric: SimilarityMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Jaccard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankingStrategy {
    RelevanceScore,
    DatabasePriority,
    ResponseTime,
    DataFreshness,
    Composite(Vec<RankingStrategy>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultMergingStrategy {
    Union,
    Intersection,
    RankedMerge,
    WeightedCombination,
}

/// Test case for consistency verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyTestCase {
    pub test_id: u64,
    pub test_type: ConsistencyTestType,
    pub affected_databases: Vec<String>,
    pub test_scenario: ConsistencyScenario,
    pub expected_behavior: ExpectedConsistencyBehavior,
    pub verification_method: ConsistencyVerificationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyTestType {
    EventualConsistency,
    StrongConsistency,
    CausalConsistency,
    SessionConsistency,
    MonotonicRead,
    MonotonicWrite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyScenario {
    pub initial_state: HashMap<String, Vec<TestEntity>>,
    pub operations: Vec<FederationOperation>,
    pub timing_constraints: TimingConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationOperation {
    pub operation_id: u64,
    pub operation_type: OperationType,
    pub target_database: String,
    pub timestamp: u64,
    pub dependencies: Vec<u64>, // operation_ids this depends on
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Create { entity: TestEntity },
    Update { entity_id: u32, changes: HashMap<String, String> },
    Delete { entity_id: u32 },
    Link { source_entity: u32, target_entity: u32, target_db: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    pub max_propagation_delay: u64,
    pub operation_ordering: OrderingRequirement,
    pub conflict_window: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingRequirement {
    Sequential,
    Concurrent,
    PartialOrder(Vec<(u64, u64)>), // operation dependencies
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedConsistencyBehavior {
    pub convergence_time: u64,
    pub intermediate_states: Vec<IntermediateState>,
    pub final_state: HashMap<String, Vec<TestEntity>>,
    pub invariants: Vec<ConsistencyInvariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateState {
    pub timestamp: u64,
    pub database_states: HashMap<String, Vec<TestEntity>>,
    pub consistency_level: ConsistencyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyInvariant {
    pub invariant_type: InvariantType,
    pub description: String,
    pub validation_rule: ValidationRule,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantType {
    ReferentialIntegrity,
    DataIntegrity,
    CausalOrdering,
    MonotonicProperty,
    UniqueConstraint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    AllDatabasesMatch,
    NoOrphanedReferences,
    MonotonicIncrement,
    CausallyOrdered,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyVerificationMethod {
    DirectComparison,
    StatisticalSampling,
    HashComparison,
    InvariantChecking,
    EventualVerification { timeout: u64 },
}

/// Performance expectations for federation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationPerformanceExpectations {
    pub query_latency_bounds: LatencyBounds,
    pub throughput_expectations: ThroughputExpectations,
    pub scalability_characteristics: ScalabilityCharacteristics,
    pub resource_usage: ResourceUsageExpectations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBounds {
    pub local_query_max_ms: u64,
    pub cross_db_query_max_ms: u64,
    pub consistency_propagation_max_ms: u64,
    pub join_operation_max_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputExpectations {
    pub queries_per_second: f64,
    pub cross_db_operations_per_second: f64,
    pub consistency_updates_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityCharacteristics {
    pub linear_scaling_databases: u32,
    pub degradation_point: u32,
    pub bottleneck_operations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageExpectations {
    pub memory_per_database_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub cpu_utilization_percent: f64,
}

/// Additional supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalQuery {
    pub query_id: u64,
    pub query_type: String,
    pub expected_result_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    pub partition_key: String,
    pub partition_value_range: (String, String),
    pub related_partitions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabasePerformanceCharacteristics {
    pub max_query_latency_ms: u64,
    pub max_throughput_qps: f64,
    pub memory_usage_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceExpectations {
    pub expected_latency_ms: u64,
    pub expected_network_overhead_mb: f64,
    pub expected_cpu_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConsistencyRequirements {
    pub read_consistency: ConsistencyLevel,
    pub write_consistency: ConsistencyLevel,
    pub isolation_level: IsolationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationMetadata {
    pub total_databases: u32,
    pub total_entities: u64,
    pub total_cross_references: u64,
    pub total_federation_queries: u64,
    pub generation_seed: u64,
    pub generation_timestamp: String,
    pub complexity_score: f64,
}

impl FederationDataGenerator {
    /// Create a new federation data generator
    pub fn new(seed: u64, config: FederationConfig) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("federation_generator".to_string());
        
        let base_generator = ComprehensiveDataGenerator::new(seed + 1);
        
        Self {
            rng,
            base_generator,
            federation_config: config,
            database_schemas: HashMap::new(),
        }
    }

    /// Generate complete federation test dataset
    pub fn generate_federation_dataset(&mut self) -> Result<FederationTestDataset> {
        // Generate database schemas
        self.generate_database_schemas()?;
        
        // Generate individual database instances
        let databases = self.generate_database_instances()?;
        
        // Generate cross-database references
        let cross_db_references = self.generate_cross_database_references(&databases)?;
        
        // Generate federation queries
        let federation_queries = self.generate_federation_queries(&databases, &cross_db_references)?;
        
        // Generate consistency test cases
        let consistency_test_cases = self.generate_consistency_test_cases(&databases)?;
        
        // Calculate performance expectations
        let performance_expectations = self.calculate_performance_expectations(&databases)?;
        
        // Generate metadata
        let metadata = self.generate_metadata(&databases, &cross_db_references, &federation_queries)?;

        Ok(FederationTestDataset {
            databases,
            cross_db_references,
            federation_queries,
            consistency_test_cases,
            performance_expectations,
            metadata,
        })
    }

    /// Generate schemas for each database in the federation
    fn generate_database_schemas(&mut self) -> Result<()> {
        for i in 0..self.federation_config.database_count {
            let database_id = format!("db_{}", i);
            
            let schema = self.generate_single_database_schema(&database_id, i)?;
            self.database_schemas.insert(database_id, schema);
        }
        Ok(())
    }

    fn generate_single_database_schema(&mut self, database_id: &str, index: u32) -> Result<DatabaseSchema> {
        let mut entity_types = HashSet::new();
        let mut relationship_types = HashSet::new();
        let mut unique_attributes = HashSet::new();
        let mut shared_attributes = HashSet::new();

        // Base entity types for all databases
        let base_types = vec!["Person", "Document", "Organization", "Event", "Location"];
        for base_type in &base_types {
            if self.rng.probability(0.8) {
                entity_types.insert(base_type.to_string());
            }
        }

        // Add database-specific entity types
        match &self.federation_config.data_distribution_strategy {
            DataDistributionStrategy::Geographic { regions } => {
                if let Some(region) = regions.get(index as usize % regions.len()) {
                    entity_types.insert(format!("{}Specific", region));
                }
            }
            DataDistributionStrategy::Semantic { domains } => {
                if let Some(domain) = domains.get(index as usize % domains.len()) {
                    entity_types.insert(format!("{}Entity", domain));
                }
            }
            _ => {
                entity_types.insert(format!("DB{}Specific", index));
            }
        }

        // Generate relationship types
        let base_relationships = vec!["knows", "contains", "related_to", "parent_of", "member_of"];
        for rel_type in &base_relationships {
            if self.rng.probability(0.7) {
                relationship_types.insert(rel_type.to_string());
            }
        }

        // Generate attributes
        let common_attributes = vec!["id", "name", "created_at", "updated_at"];
        for attr in &common_attributes {
            shared_attributes.insert(attr.to_string());
        }

        // Database-specific unique attributes
        for j in 0..self.rng.range(2, 6) {
            unique_attributes.insert(format!("{}_{}_attr_{}", database_id, "unique", j));
        }

        // Apply schema overlap
        if index > 0 && self.rng.probability(self.federation_config.schema_overlap_ratio) {
            // Add some shared attributes with previous databases
            let prev_db_id = format!("db_{}", index - 1);
            if let Some(prev_schema) = self.database_schemas.get(&prev_db_id) {
                for attr in prev_schema.shared_attributes.iter().take(2) {
                    shared_attributes.insert(attr.clone());
                }
            }
        }

        let indexing_strategy = match index % 4 {
            0 => IndexingStrategy::BTree,
            1 => IndexingStrategy::Hash,
            2 => IndexingStrategy::Vector { 
                algorithm: VectorIndexAlgorithm::HNSW { m: 16, ef_construction: 200 } 
            },
            _ => IndexingStrategy::Composite { 
                strategies: vec![IndexingStrategy::BTree, IndexingStrategy::Hash] 
            },
        };

        Ok(DatabaseSchema {
            database_id: database_id.to_string(),
            entity_types,
            relationship_types,
            unique_attributes,
            shared_attributes,
            embedding_dimensions: vec![64, 128, 256],
            indexing_strategy,
            sharding_key: Some("id".to_string()),
        })
    }

    fn generate_database_instances(&mut self) -> Result<HashMap<String, DatabaseInstance>> {
        let mut databases = HashMap::new();

        for (db_id, schema) in &self.database_schemas.clone() {
            let instance = self.generate_single_database_instance(db_id, schema)?;
            databases.insert(db_id.clone(), instance);
        }

        Ok(databases)
    }

    fn generate_single_database_instance(&mut self, db_id: &str, schema: &DatabaseSchema) -> Result<DatabaseInstance> {
        // Use base generator to create graph structure
        let params = GenerationParameters {
            graph_sizes: vec![self.federation_config.entities_per_database],
            embedding_dimensions: schema.embedding_dimensions.clone(),
            query_counts: {
                let mut counts = HashMap::new();
                counts.insert("local".to_string(), 20);
                counts
            },
            quantization_settings: vec![],
            validation_enabled: false,
        };

        let dataset = self.base_generator.generate_complete_dataset(params)?;
        let graph = dataset.graph_topologies.into_iter().next()
            .ok_or_else(|| anyhow!("Failed to generate graph for database {}", db_id))?;

        // Generate local queries
        let local_queries = self.generate_local_queries(db_id, &graph)?;

        // Determine replication targets
        let replication_targets = self.determine_replication_targets(db_id)?;

        // Generate partition info
        let partition_info = PartitionInfo {
            partition_key: "entity_type".to_string(),
            partition_value_range: ("0".to_string(), "999".to_string()),
            related_partitions: replication_targets.clone(),
        };

        // Set performance characteristics
        let performance_characteristics = DatabasePerformanceCharacteristics {
            max_query_latency_ms: 100 + (self.rng.range(0, 200) as u64),
            max_throughput_qps: 1000.0 + (self.rng.range(0, 5000) as f64),
            memory_usage_mb: 512 + (self.rng.range(0, 2048) as u64),
        };

        Ok(DatabaseInstance {
            database_id: db_id.to_string(),
            graph,
            embeddings: dataset.embeddings,
            local_queries,
            replication_targets,
            partition_info,
            performance_characteristics,
        })
    }

    fn generate_local_queries(&mut self, db_id: &str, graph: &TestGraph) -> Result<Vec<LocalQuery>> {
        let mut queries = Vec::new();

        for i in 0..10 {
            let query = LocalQuery {
                query_id: self.rng.next() as u64,
                query_type: format!("local_query_{}_type_{}", db_id, i % 3),
                expected_result_count: self.rng.range(1, graph.entities.len().min(100)) as u64,
            };
            queries.push(query);
        }

        Ok(queries)
    }

    fn determine_replication_targets(&mut self, db_id: &str) -> Result<Vec<String>> {
        let mut targets = Vec::new();
        let target_count = (self.federation_config.replication_factor * 
                           self.federation_config.database_count as f64) as u32;

        for _ in 0..target_count.min(self.federation_config.database_count - 1) {
            let target_index = self.rng.range(0, self.federation_config.database_count);
            let target_id = format!("db_{}", target_index);
            
            if target_id != db_id && !targets.contains(&target_id) {
                targets.push(target_id);
            }
        }

        Ok(targets)
    }

    fn generate_cross_database_references(&mut self, databases: &HashMap<String, DatabaseInstance>) -> Result<Vec<CrossDatabaseReference>> {
        let mut references = Vec::new();
        let mut reference_id = 0;

        let db_ids: Vec<_> = databases.keys().cloned().collect();
        
        for source_db in &db_ids {
            if let Some(source_instance) = databases.get(source_db) {
                let reference_count = (source_instance.graph.entities.len() as f64 * 
                                     self.federation_config.cross_db_connection_probability) as usize;

                for _ in 0..reference_count {
                    let target_db = &db_ids[self.rng.range(0, db_ids.len())];
                    
                    if source_db != target_db {
                        if let Some(target_instance) = databases.get(target_db) {
                            let source_entity = source_instance.graph.entities[
                                self.rng.range(0, source_instance.graph.entities.len())
                            ].id;
                            
                            let target_entity = target_instance.graph.entities[
                                self.rng.range(0, target_instance.graph.entities.len())
                            ].id;

                            let reference = CrossDatabaseReference {
                                reference_id,
                                source_db: source_db.clone(),
                                source_entity,
                                target_db: target_db.clone(),
                                target_entity,
                                reference_type: self.select_cross_reference_type(),
                                consistency_requirements: self.select_consistency_level(),
                                resolution_strategy: self.select_resolution_strategy(),
                            };

                            references.push(reference);
                            reference_id += 1;
                        }
                    }
                }
            }
        }

        Ok(references)
    }

    fn generate_federation_queries(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                  references: &[CrossDatabaseReference]) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();
        let mut query_id = 0;

        // Generate cross-database join queries
        queries.extend(self.generate_join_queries(databases, &mut query_id)?);
        
        // Generate distributed aggregation queries
        queries.extend(self.generate_aggregation_queries(databases, &mut query_id)?);
        
        // Generate global search queries
        queries.extend(self.generate_global_search_queries(databases, &mut query_id)?);
        
        // Generate consistency check queries
        queries.extend(self.generate_consistency_queries(references, &mut query_id)?);
        
        // Generate federated RAG queries
        queries.extend(self.generate_federated_rag_queries(databases, &mut query_id)?);

        Ok(queries)
    }

    fn generate_join_queries(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                           query_id: &mut u64) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        for _ in 0..5 {
            if db_ids.len() >= 2 {
                let db1 = &db_ids[self.rng.range(0, db_ids.len())];
                let db2 = &db_ids[self.rng.range(0, db_ids.len())];

                if db1 != db2 {
                    let join_condition = JoinCondition {
                        left_db: db1.clone(),
                        left_attribute: "id".to_string(),
                        right_db: db2.clone(),
                        right_attribute: "related_id".to_string(),
                        join_type: JoinType::Inner,
                    };

                    let mut expected_distribution = HashMap::new();
                    expected_distribution.insert(db1.clone(), self.rng.range(10, 100) as u64);
                    expected_distribution.insert(db2.clone(), self.rng.range(10, 100) as u64);

                    let query = FederationQuery {
                        query_id: *query_id,
                        query_type: FederationQueryType::CrossDatabaseJoin {
                            join_conditions: vec![join_condition],
                            join_strategy: JoinStrategy::HashJoin,
                        },
                        participating_databases: vec![db1.clone(), db2.clone()],
                        expected_result_distribution: expected_distribution,
                        performance_expectations: QueryPerformanceExpectations {
                            expected_latency_ms: 500,
                            expected_network_overhead_mb: 10.0,
                            expected_cpu_usage_percent: 25.0,
                        },
                        consistency_requirements: QueryConsistencyRequirements {
                            read_consistency: ConsistencyLevel::Eventual,
                            write_consistency: ConsistencyLevel::Strong,
                            isolation_level: IsolationLevel::ReadCommitted,
                        },
                    };

                    queries.push(query);
                    *query_id += 1;
                }
            }
        }

        Ok(queries)
    }

    fn generate_aggregation_queries(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                  query_id: &mut u64) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        for _ in 0..3 {
            let participating_dbs = if db_ids.len() >= 3 {
                db_ids[0..3].to_vec()
            } else {
                db_ids.clone()
            };

            let aggregation_function = AggregationFunction {
                function_type: AggregationType::Count,
                attribute: "entity_type".to_string(),
                database_scope: AggregationScope::Global,
            };

            let mut expected_distribution = HashMap::new();
            for db in &participating_dbs {
                expected_distribution.insert(db.clone(), self.rng.range(50, 200) as u64);
            }

            let query = FederationQuery {
                query_id: *query_id,
                query_type: FederationQueryType::DistributedAggregation {
                    aggregation_functions: vec![aggregation_function],
                    grouping_keys: vec!["entity_type".to_string()],
                },
                participating_databases: participating_dbs,
                expected_result_distribution: expected_distribution,
                performance_expectations: QueryPerformanceExpectations {
                    expected_latency_ms: 300,
                    expected_network_overhead_mb: 5.0,
                    expected_cpu_usage_percent: 15.0,
                },
                consistency_requirements: QueryConsistencyRequirements {
                    read_consistency: ConsistencyLevel::Eventual,
                    write_consistency: ConsistencyLevel::Strong,
                    isolation_level: IsolationLevel::ReadCommitted,
                },
            };

            queries.push(query);
            *query_id += 1;
        }

        Ok(queries)
    }

    fn generate_global_search_queries(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                    query_id: &mut u64) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        for _ in 0..4 {
            let search_criteria = SearchCriteria {
                entity_types: vec!["Person".to_string(), "Document".to_string()],
                attribute_filters: {
                    let mut filters = HashMap::new();
                    filters.insert("status".to_string(), "active".to_string());
                    filters
                },
                embedding_similarity: Some(EmbeddingSearchCriteria {
                    query_vector: vec![0.1; 64],
                    similarity_threshold: 0.8,
                    metric: SimilarityMetric::Cosine,
                }),
                result_limit: 100,
            };

            let mut expected_distribution = HashMap::new();
            for db in &db_ids {
                expected_distribution.insert(db.clone(), self.rng.range(5, 25) as u64);
            }

            let query = FederationQuery {
                query_id: *query_id,
                query_type: FederationQueryType::GlobalSearch {
                    search_criteria,
                    ranking_strategy: RankingStrategy::RelevanceScore,
                },
                participating_databases: db_ids.clone(),
                expected_result_distribution: expected_distribution,
                performance_expectations: QueryPerformanceExpectations {
                    expected_latency_ms: 800,
                    expected_network_overhead_mb: 15.0,
                    expected_cpu_usage_percent: 40.0,
                },
                consistency_requirements: QueryConsistencyRequirements {
                    read_consistency: ConsistencyLevel::Weak,
                    write_consistency: ConsistencyLevel::Strong,
                    isolation_level: IsolationLevel::ReadCommitted,
                },
            };

            queries.push(query);
            *query_id += 1;
        }

        Ok(queries)
    }

    fn generate_consistency_queries(&mut self, references: &[CrossDatabaseReference], 
                                  query_id: &mut u64) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();

        for chunk in references.chunks(5) {
            let participating_dbs: BTreeSet<String> = chunk.iter()
                .flat_map(|r| vec![r.source_db.clone(), r.target_db.clone()])
                .collect();

            let mut expected_distribution = HashMap::new();
            for db in &participating_dbs {
                expected_distribution.insert(db.clone(), chunk.len() as u64);
            }

            let query = FederationQuery {
                query_id: *query_id,
                query_type: FederationQueryType::ConsistencyCheck {
                    entity_references: chunk.to_vec(),
                    consistency_level: ConsistencyLevel::Strong,
                },
                participating_databases: participating_dbs.into_iter().collect(),
                expected_result_distribution: expected_distribution,
                performance_expectations: QueryPerformanceExpectations {
                    expected_latency_ms: 200,
                    expected_network_overhead_mb: 2.0,
                    expected_cpu_usage_percent: 10.0,
                },
                consistency_requirements: QueryConsistencyRequirements {
                    read_consistency: ConsistencyLevel::Strong,
                    write_consistency: ConsistencyLevel::Strong,
                    isolation_level: IsolationLevel::Serializable,
                },
            };

            queries.push(query);
            *query_id += 1;
        }

        Ok(queries)
    }

    fn generate_federated_rag_queries(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                    query_id: &mut u64) -> Result<Vec<FederationQuery>> {
        let mut queries = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        for _ in 0..3 {
            let databases_to_search = if db_ids.len() >= 2 {
                db_ids[0..2].to_vec()
            } else {
                db_ids.clone()
            };

            let query_concept = self.rng.range(1, 1000) as u32;

            let mut expected_distribution = HashMap::new();
            for db in &databases_to_search {
                expected_distribution.insert(db.clone(), self.rng.range(10, 50) as u64);
            }

            let query = FederationQuery {
                query_id: *query_id,
                query_type: FederationQueryType::FederatedRAG {
                    query_concept,
                    databases_to_search: databases_to_search.clone(),
                    result_merging_strategy: ResultMergingStrategy::RankedMerge,
                },
                participating_databases: databases_to_search,
                expected_result_distribution: expected_distribution,
                performance_expectations: QueryPerformanceExpectations {
                    expected_latency_ms: 600,
                    expected_network_overhead_mb: 12.0,
                    expected_cpu_usage_percent: 30.0,
                },
                consistency_requirements: QueryConsistencyRequirements {
                    read_consistency: ConsistencyLevel::Eventual,
                    write_consistency: ConsistencyLevel::Strong,
                    isolation_level: IsolationLevel::ReadCommitted,
                },
            };

            queries.push(query);
            *query_id += 1;
        }

        Ok(queries)
    }

    fn generate_consistency_test_cases(&mut self, databases: &HashMap<String, DatabaseInstance>) -> Result<Vec<ConsistencyTestCase>> {
        let mut test_cases = Vec::new();
        let mut test_id = 0;

        // Generate eventual consistency test cases
        test_cases.extend(self.generate_eventual_consistency_tests(databases, &mut test_id)?);
        
        // Generate strong consistency test cases
        test_cases.extend(self.generate_strong_consistency_tests(databases, &mut test_id)?);
        
        // Generate causal consistency test cases
        test_cases.extend(self.generate_causal_consistency_tests(databases, &mut test_id)?);

        Ok(test_cases)
    }

    fn generate_eventual_consistency_tests(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                         test_id: &mut u64) -> Result<Vec<ConsistencyTestCase>> {
        let mut test_cases = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        if db_ids.len() >= 2 {
            let affected_databases = vec![db_ids[0].clone(), db_ids[1].clone()];
            
            let initial_state = self.create_initial_state(&affected_databases, databases)?;
            let operations = self.create_eventual_consistency_operations(&affected_databases)?;
            
            let test_case = ConsistencyTestCase {
                test_id: *test_id,
                test_type: ConsistencyTestType::EventualConsistency,
                affected_databases: affected_databases.clone(),
                test_scenario: ConsistencyScenario {
                    initial_state,
                    operations,
                    timing_constraints: TimingConstraints {
                        max_propagation_delay: self.federation_config.consistency_requirements.eventual_consistency_delay,
                        operation_ordering: OrderingRequirement::Concurrent,
                        conflict_window: 1000,
                    },
                },
                expected_behavior: ExpectedConsistencyBehavior {
                    convergence_time: self.federation_config.consistency_requirements.eventual_consistency_delay * 2,
                    intermediate_states: vec![],
                    final_state: HashMap::new(),
                    invariants: vec![
                        ConsistencyInvariant {
                            invariant_type: InvariantType::ReferentialIntegrity,
                            description: "All references must eventually be consistent".to_string(),
                            validation_rule: ValidationRule::AllDatabasesMatch,
                        }
                    ],
                },
                verification_method: ConsistencyVerificationMethod::EventualVerification { 
                    timeout: self.federation_config.consistency_requirements.eventual_consistency_delay * 3 
                },
            };

            test_cases.push(test_case);
            *test_id += 1;
        }

        Ok(test_cases)
    }

    fn generate_strong_consistency_tests(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                       test_id: &mut u64) -> Result<Vec<ConsistencyTestCase>> {
        let mut test_cases = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        if db_ids.len() >= 2 {
            let affected_databases = vec![db_ids[0].clone(), db_ids[1].clone()];
            
            let initial_state = self.create_initial_state(&affected_databases, databases)?;
            let operations = self.create_strong_consistency_operations(&affected_databases)?;
            
            let test_case = ConsistencyTestCase {
                test_id: *test_id,
                test_type: ConsistencyTestType::StrongConsistency,
                affected_databases: affected_databases.clone(),
                test_scenario: ConsistencyScenario {
                    initial_state,
                    operations,
                    timing_constraints: TimingConstraints {
                        max_propagation_delay: 100, // Strong consistency requires immediate propagation
                        operation_ordering: OrderingRequirement::Sequential,
                        conflict_window: 0,
                    },
                },
                expected_behavior: ExpectedConsistencyBehavior {
                    convergence_time: 100,
                    intermediate_states: vec![],
                    final_state: HashMap::new(),
                    invariants: vec![
                        ConsistencyInvariant {
                            invariant_type: InvariantType::DataIntegrity,
                            description: "All databases must have identical state immediately".to_string(),
                            validation_rule: ValidationRule::AllDatabasesMatch,
                        }
                    ],
                },
                verification_method: ConsistencyVerificationMethod::DirectComparison,
            };

            test_cases.push(test_case);
            *test_id += 1;
        }

        Ok(test_cases)
    }

    fn generate_causal_consistency_tests(&mut self, databases: &HashMap<String, DatabaseInstance>, 
                                       test_id: &mut u64) -> Result<Vec<ConsistencyTestCase>> {
        let mut test_cases = Vec::new();
        let db_ids: Vec<_> = databases.keys().cloned().collect();

        if db_ids.len() >= 3 {
            let affected_databases = vec![db_ids[0].clone(), db_ids[1].clone(), db_ids[2].clone()];
            
            let initial_state = self.create_initial_state(&affected_databases, databases)?;
            let operations = self.create_causal_consistency_operations(&affected_databases)?;
            
            let test_case = ConsistencyTestCase {
                test_id: *test_id,
                test_type: ConsistencyTestType::CausalConsistency,
                affected_databases: affected_databases.clone(),
                test_scenario: ConsistencyScenario {
                    initial_state,
                    operations,
                    timing_constraints: TimingConstraints {
                        max_propagation_delay: 500,
                        operation_ordering: OrderingRequirement::PartialOrder(vec![(0, 1), (1, 2)]),
                        conflict_window: 200,
                    },
                },
                expected_behavior: ExpectedConsistencyBehavior {
                    convergence_time: 1000,
                    intermediate_states: vec![],
                    final_state: HashMap::new(),
                    invariants: vec![
                        ConsistencyInvariant {
                            invariant_type: InvariantType::CausalOrdering,
                            description: "Causally related operations must be observed in order".to_string(),
                            validation_rule: ValidationRule::CausallyOrdered,
                        }
                    ],
                },
                verification_method: ConsistencyVerificationMethod::InvariantChecking,
            };

            test_cases.push(test_case);
            *test_id += 1;
        }

        Ok(test_cases)
    }

    // Helper methods

    fn select_cross_reference_type(&mut self) -> CrossReferenceType {
        match self.rng.range(0, 5) {
            0 => CrossReferenceType::OwnershipLink,
            1 => CrossReferenceType::DependencyLink,
            2 => CrossReferenceType::EquivalenceLink,
            3 => CrossReferenceType::HierarchyLink,
            _ => CrossReferenceType::TemporalLink,
        }
    }

    fn select_consistency_level(&mut self) -> ConsistencyLevel {
        match self.rng.range(0, 4) {
            0 => ConsistencyLevel::Strong,
            1 => ConsistencyLevel::Eventual,
            2 => ConsistencyLevel::Weak,
            _ => ConsistencyLevel::Session,
        }
    }

    fn select_resolution_strategy(&mut self) -> ResolutionStrategy {
        match self.rng.range(0, 4) {
            0 => ResolutionStrategy::Eager,
            1 => ResolutionStrategy::Lazy,
            2 => ResolutionStrategy::OnDemand,
            _ => ResolutionStrategy::Cached,
        }
    }

    fn create_initial_state(&mut self, databases: &[String], 
                          all_databases: &HashMap<String, DatabaseInstance>) -> Result<HashMap<String, Vec<TestEntity>>> {
        let mut initial_state = HashMap::new();
        
        for db_id in databases {
            if let Some(db_instance) = all_databases.get(db_id) {
                let entities = db_instance.graph.entities[0..5.min(db_instance.graph.entities.len())].to_vec();
                initial_state.insert(db_id.clone(), entities);
            }
        }
        
        Ok(initial_state)
    }

    fn create_eventual_consistency_operations(&mut self, databases: &[String]) -> Result<Vec<FederationOperation>> {
        let mut operations = Vec::new();
        let mut operation_id = 0;

        for (i, db_id) in databases.iter().enumerate() {
            let operation = FederationOperation {
                operation_id,
                operation_type: OperationType::Create {
                    entity: TestEntity {
                        id: 9000 + i as u32,
                        entity_type: "TestEntity".to_string(),
                        properties: HashMap::new(),
                    }
                },
                target_database: db_id.clone(),
                timestamp: i as u64 * 100,
                dependencies: vec![],
            };
            operations.push(operation);
            operation_id += 1;
        }

        Ok(operations)
    }

    fn create_strong_consistency_operations(&mut self, databases: &[String]) -> Result<Vec<FederationOperation>> {
        let mut operations = Vec::new();
        let mut operation_id = 0;

        // Create a single entity that must be consistent across all databases
        for db_id in databases {
            let operation = FederationOperation {
                operation_id,
                operation_type: OperationType::Create {
                    entity: TestEntity {
                        id: 8000,
                        entity_type: "ConsistentEntity".to_string(),
                        properties: HashMap::new(),
                    }
                },
                target_database: db_id.clone(),
                timestamp: 0, // All operations happen simultaneously
                dependencies: if operation_id > 0 { vec![operation_id - 1] } else { vec![] },
            };
            operations.push(operation);
            operation_id += 1;
        }

        Ok(operations)
    }

    fn create_causal_consistency_operations(&mut self, databases: &[String]) -> Result<Vec<FederationOperation>> {
        let mut operations = Vec::new();
        let mut operation_id = 0;

        // Create causally related operations
        if databases.len() >= 3 {
            // Operation 1: Create entity in DB1
            operations.push(FederationOperation {
                operation_id,
                operation_type: OperationType::Create {
                    entity: TestEntity {
                        id: 7000,
                        entity_type: "CausalEntity".to_string(),
                        properties: HashMap::new(),
                    }
                },
                target_database: databases[0].clone(),
                timestamp: 0,
                dependencies: vec![],
            });
            operation_id += 1;

            // Operation 2: Update entity (depends on operation 1)
            operations.push(FederationOperation {
                operation_id,
                operation_type: OperationType::Update {
                    entity_id: 7000,
                    changes: {
                        let mut changes = HashMap::new();
                        changes.insert("status".to_string(), "updated".to_string());
                        changes
                    }
                },
                target_database: databases[1].clone(),
                timestamp: 100,
                dependencies: vec![0],
            });
            operation_id += 1;

            // Operation 3: Link entity (depends on operation 2)
            operations.push(FederationOperation {
                operation_id,
                operation_type: OperationType::Link {
                    source_entity: 7000,
                    target_entity: 7001,
                    target_db: databases[2].clone(),
                },
                target_database: databases[2].clone(),
                timestamp: 200,
                dependencies: vec![1],
            });
        }

        Ok(operations)
    }

    fn calculate_performance_expectations(&self, databases: &HashMap<String, DatabaseInstance>) -> Result<FederationPerformanceExpectations> {
        let avg_local_latency = databases.values()
            .map(|db| db.performance_characteristics.max_query_latency_ms)
            .sum::<u64>() / databases.len().max(1) as u64;

        let total_throughput = databases.values()
            .map(|db| db.performance_characteristics.max_throughput_qps)
            .sum::<f64>();

        let total_memory = databases.values()
            .map(|db| db.performance_characteristics.memory_usage_mb)
            .sum::<u64>();

        Ok(FederationPerformanceExpectations {
            query_latency_bounds: LatencyBounds {
                local_query_max_ms: avg_local_latency,
                cross_db_query_max_ms: avg_local_latency * 3,
                consistency_propagation_max_ms: self.federation_config.consistency_requirements.eventual_consistency_delay,
                join_operation_max_ms: avg_local_latency * 5,
            },
            throughput_expectations: ThroughputExpectations {
                queries_per_second: total_throughput * 0.8, // 80% of theoretical max
                cross_db_operations_per_second: total_throughput * 0.3,
                consistency_updates_per_second: total_throughput * 0.1,
            },
            scalability_characteristics: ScalabilityCharacteristics {
                linear_scaling_databases: self.federation_config.database_count / 2,
                degradation_point: self.federation_config.database_count,
                bottleneck_operations: vec!["cross_db_join".to_string(), "consistency_check".to_string()],
            },
            resource_usage: ResourceUsageExpectations {
                memory_per_database_mb: total_memory / databases.len().max(1) as u64,
                network_bandwidth_mbps: 100.0 * databases.len() as f64,
                cpu_utilization_percent: 70.0,
            },
        })
    }

    fn generate_metadata(&self, databases: &HashMap<String, DatabaseInstance>, 
                        references: &[CrossDatabaseReference], 
                        queries: &[FederationQuery]) -> Result<FederationMetadata> {
        let total_entities = databases.values()
            .map(|db| db.graph.entities.len() as u64)
            .sum();

        // Calculate complexity score based on various factors
        let complexity_score = (databases.len() as f64 * 0.3) +
                              (references.len() as f64 * 0.4) +
                              (queries.len() as f64 * 0.3);

        Ok(FederationMetadata {
            total_databases: databases.len() as u32,
            total_entities,
            total_cross_references: references.len() as u64,
            total_federation_queries: queries.len() as u64,
            generation_seed: self.rng.seed(),
            generation_timestamp: "2024-01-01T00:00:00Z".to_string(), // Simplified
            complexity_score,
        })
    }
}

/// Create default federation configuration
pub fn create_default_federation_config() -> FederationConfig {
    FederationConfig {
        database_count: 3,
        entities_per_database: 100,
        cross_db_connection_probability: 0.1,
        schema_overlap_ratio: 0.3,
        data_distribution_strategy: DataDistributionStrategy::Random,
        consistency_requirements: ConsistencyRequirements {
            eventual_consistency_delay: 1000,
            strong_consistency_entities: vec!["CriticalEntity".to_string()],
            conflict_resolution_strategy: ConflictResolutionStrategy::LastWriteWins,
            replication_consistency: ReplicationConsistency::Eventual,
        },
        replication_factor: 0.5,
    }
}

/// Create geographic federation configuration
pub fn create_geographic_federation_config() -> FederationConfig {
    let mut config = create_default_federation_config();
    config.data_distribution_strategy = DataDistributionStrategy::Geographic {
        regions: vec!["US".to_string(), "EU".to_string(), "ASIA".to_string()],
    };
    config.database_count = 3;
    config
}

/// Create high-scale federation configuration
pub fn create_high_scale_federation_config() -> FederationConfig {
    let mut config = create_default_federation_config();
    config.database_count = 10;
    config.entities_per_database = 1000;
    config.cross_db_connection_probability = 0.05;
    config.replication_factor = 0.8;
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_generator_creation() {
        let config = create_default_federation_config();
        let generator = FederationDataGenerator::new(42, config);
        
        assert_eq!(generator.federation_config.database_count, 3);
        assert!(generator.database_schemas.is_empty());
    }

    #[test]
    fn test_database_schema_generation() {
        let config = create_default_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        let result = generator.generate_database_schemas();
        assert!(result.is_ok());
        
        assert_eq!(generator.database_schemas.len(), 3);
        for (db_id, schema) in &generator.database_schemas {
            assert!(db_id.starts_with("db_"));
            assert!(!schema.entity_types.is_empty());
            assert!(!schema.shared_attributes.is_empty());
        }
    }

    #[test]
    fn test_federation_dataset_generation() {
        let config = create_default_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        let dataset = generator.generate_federation_dataset().unwrap();
        
        assert_eq!(dataset.databases.len(), 3);
        assert!(!dataset.cross_db_references.is_empty());
        assert!(!dataset.federation_queries.is_empty());
        assert!(!dataset.consistency_test_cases.is_empty());
        assert_eq!(dataset.metadata.total_databases, 3);
    }

    #[test]
    fn test_geographic_distribution() {
        let config = create_geographic_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        let result = generator.generate_database_schemas();
        assert!(result.is_ok());
        
        // Check that geographic-specific entity types are created
        let has_geographic_entities = generator.database_schemas.values()
            .any(|schema| schema.entity_types.iter()
                .any(|entity_type| entity_type.contains("Specific")));
        assert!(has_geographic_entities);
    }

    #[test]
    fn test_cross_database_references() {
        let config = create_default_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        // Generate databases first
        let dataset = generator.generate_federation_dataset().unwrap();
        
        // Verify cross-database references
        for reference in &dataset.cross_db_references {
            assert_ne!(reference.source_db, reference.target_db);
            assert!(dataset.databases.contains_key(&reference.source_db));
            assert!(dataset.databases.contains_key(&reference.target_db));
        }
    }

    #[test]
    fn test_consistency_test_cases() {
        let config = create_default_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        let dataset = generator.generate_federation_dataset().unwrap();
        
        // Verify consistency test cases are generated
        assert!(!dataset.consistency_test_cases.is_empty());
        
        for test_case in &dataset.consistency_test_cases {
            assert!(!test_case.affected_databases.is_empty());
            assert!(!test_case.test_scenario.operations.is_empty());
            assert!(!test_case.expected_behavior.invariants.is_empty());
        }
    }

    #[test]
    fn test_federation_queries() {
        let config = create_default_federation_config();
        let mut generator = FederationDataGenerator::new(42, config);
        
        let dataset = generator.generate_federation_dataset().unwrap();
        
        // Verify federation queries are generated
        assert!(!dataset.federation_queries.is_empty());
        
        for query in &dataset.federation_queries {
            assert!(!query.participating_databases.is_empty());
            assert!(!query.expected_result_distribution.is_empty());
            assert!(query.performance_expectations.expected_latency_ms > 0);
        }
    }
}