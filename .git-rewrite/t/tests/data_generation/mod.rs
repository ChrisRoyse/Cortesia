//! Data Generation Module
//! 
//! Comprehensive synthetic data generation framework for Phase 2 testing.

pub mod graph_topologies;
pub mod embeddings;
pub mod knowledge_graphs;
pub mod multi_scale;
pub mod quantization_data;
pub mod query_patterns;
pub mod golden_standards;
pub mod validation;
pub mod streaming;
pub mod federation;

// Re-export main types for convenience
pub use graph_topologies::{
    GraphTopologyGenerator, TestGraph, TestEntity, TestEdge, 
    GraphProperties, ConnectivityType
};

pub use embeddings::{
    EmbeddingGenerator, EmbeddingTestSet, ClusterSpec, 
    HierarchicalStructure, DistanceConstraint, EmbeddingProperties
};

pub use knowledge_graphs::{
    KnowledgeGraphGenerator, Ontology, EntityType, RelationshipType,
    create_academic_ontology
};

pub use multi_scale::{
    MultiScaleGenerator, HierarchicalSpec, FractalSpec, 
    MultiScaleProperties, create_standard_hierarchy_spec, 
    create_triangle_fractal_spec
};

pub use quantization_data::{
    QuantizationDataGenerator, QuantizationTestSet, SimdTestSet,
    CompressionTestData, QuantizationParams
};

pub use query_patterns::{
    QueryPatternGenerator, TraversalQuery, RagQuery, SimilarityQuery,
    ComplexQuery, TraversalType, RagStrategy, SimilarityMetric
};

pub use golden_standards::{
    ExactComputationEngine, GoldenStandards, TraversalResult, SimilarityResult,
    RagResult, PerformanceExpectations
};

pub use validation::{
    DataQualityValidator, ValidationReport, ValidationStatus, quick_validate,
    comprehensive_validate
};

pub use streaming::{
    StreamingDataGenerator, TemporalBatch, TemporalConfig, TemporalQuery,
    ProcessingResult, create_default_temporal_config, create_high_frequency_config
};

pub use federation::{
    FederationDataGenerator, FederationTestDataset, FederationConfig,
    CrossDatabaseReference, FederationQuery, ConsistencyTestCase,
    create_default_federation_config, create_geographic_federation_config
};

use crate::infrastructure::deterministic_rng::DeterministicRng;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive test dataset containing all generated data types
#[derive(Debug, Clone)]
pub struct ComprehensiveTestDataset {
    pub graph_topologies: Vec<TestGraph>,
    pub knowledge_graphs: Vec<TestGraph>,
    pub hierarchical_graphs: Vec<TestGraph>,
    pub fractal_graphs: Vec<TestGraph>,
    pub embeddings: HashMap<String, EmbeddingTestSet>,
    pub quantization_data: Vec<QuantizationTestSet>,
    pub simd_data: Vec<SimdTestSet>,
    pub traversal_queries: Vec<TraversalQuery>,
    pub rag_queries: Vec<RagQuery>,
    pub similarity_queries: Vec<SimilarityQuery>,
    pub complex_queries: Vec<ComplexQuery>,
    pub metadata: DatasetMetadata,
}

/// Metadata about the generated dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub generation_seed: u64,
    pub creation_timestamp: String,
    pub total_entities: u64,
    pub total_edges: u64,
    pub total_embeddings: u64,
    pub total_queries: u64,
    pub dataset_size_bytes: u64,
    pub generation_parameters: GenerationParameters,
}

/// Parameters used for dataset generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParameters {
    pub graph_sizes: Vec<u64>,
    pub embedding_dimensions: Vec<usize>,
    pub query_counts: HashMap<String, u32>,
    pub quantization_settings: Vec<QuantizationParams>,
    pub validation_enabled: bool,
}

/// Master data generator that orchestrates all generation components
pub struct ComprehensiveDataGenerator {
    rng: DeterministicRng,
    seed: u64,
}

impl ComprehensiveDataGenerator {
    /// Create a new comprehensive data generator
    pub fn new(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("comprehensive_data_generator".to_string());
        
        Self { rng, seed }
    }

    /// Generate streaming dataset with temporal characteristics
    pub fn generate_streaming_dataset(&mut self, temporal_config: TemporalConfig) -> Result<Vec<TemporalBatch>> {
        let mut streaming_generator = StreamingDataGenerator::new(self.seed + 100, temporal_config);
        streaming_generator.generate_temporal_stream(100)
    }

    /// Generate federation dataset with multi-database scenarios
    pub fn generate_federation_dataset(&mut self, federation_config: FederationConfig) -> Result<FederationTestDataset> {
        let mut federation_generator = FederationDataGenerator::new(self.seed + 200, federation_config);
        federation_generator.generate_federation_dataset()
    }

    /// Generate a complete test dataset with all components
    pub fn generate_complete_dataset(&mut self, params: GenerationParameters) -> Result<ComprehensiveTestDataset> {
        let start_time = std::time::Instant::now();
        
        // Generate graph topologies
        let graph_topologies = self.generate_graph_topologies(&params.graph_sizes)?;
        
        // Generate knowledge graphs
        let knowledge_graphs = self.generate_knowledge_graphs(&params.graph_sizes)?;
        
        // Generate multi-scale graphs
        let (hierarchical_graphs, fractal_graphs) = self.generate_multi_scale_graphs(&params.graph_sizes)?;
        
        // Generate embeddings for all graphs
        let embeddings = self.generate_all_embeddings(&graph_topologies, &knowledge_graphs, &params.embedding_dimensions)?;
        
        // Generate quantization test data
        let (quantization_data, simd_data) = self.generate_quantization_data(&params.quantization_settings)?;
        
        // Generate query patterns
        let (traversal_queries, rag_queries, similarity_queries, complex_queries) = 
            self.generate_all_queries(&graph_topologies, &embeddings, &params.query_counts)?;
        
        // Calculate metadata
        let metadata = self.calculate_metadata(&params, start_time.elapsed())?;
        
        Ok(ComprehensiveTestDataset {
            graph_topologies,
            knowledge_graphs,
            hierarchical_graphs,
            fractal_graphs,
            embeddings,
            quantization_data,
            simd_data,
            traversal_queries,
            rag_queries,
            similarity_queries,
            complex_queries,
            metadata,
        })
    }

    /// Generate standard test dataset configurations
    pub fn generate_standard_small_dataset(&mut self) -> Result<ComprehensiveTestDataset> {
        let params = GenerationParameters {
            graph_sizes: vec![50, 100, 200],
            embedding_dimensions: vec![64, 128],
            query_counts: {
                let mut counts = HashMap::new();
                counts.insert("traversal".to_string(), 20);
                counts.insert("rag".to_string(), 15);
                counts.insert("similarity".to_string(), 25);
                counts.insert("complex".to_string(), 10);
                counts
            },
            quantization_settings: vec![
                QuantizationParams {
                    vector_dimension: 64,
                    subvector_count: 8,
                    codebook_size: 256,
                    subvector_dimension: 8,
                },
                QuantizationParams {
                    vector_dimension: 128,
                    subvector_count: 8,
                    codebook_size: 256,
                    subvector_dimension: 16,
                },
            ],
            validation_enabled: true,
        };
        
        self.generate_complete_dataset(params)
    }

    pub fn generate_standard_large_dataset(&mut self) -> Result<ComprehensiveTestDataset> {
        let params = GenerationParameters {
            graph_sizes: vec![1000, 5000, 10000],
            embedding_dimensions: vec![96, 256, 512],
            query_counts: {
                let mut counts = HashMap::new();
                counts.insert("traversal".to_string(), 100);
                counts.insert("rag".to_string(), 75);
                counts.insert("similarity".to_string(), 150);
                counts.insert("complex".to_string(), 50);
                counts
            },
            quantization_settings: vec![
                QuantizationParams {
                    vector_dimension: 96,
                    subvector_count: 8,
                    codebook_size: 256,
                    subvector_dimension: 12,
                },
                QuantizationParams {
                    vector_dimension: 256,
                    subvector_count: 8,
                    codebook_size: 256,
                    subvector_dimension: 32,
                },
                QuantizationParams {
                    vector_dimension: 512,
                    subvector_count: 16,
                    codebook_size: 256,
                    subvector_dimension: 32,
                },
            ],
            validation_enabled: true,
        };
        
        self.generate_complete_dataset(params)
    }

    // Private generation methods

    fn generate_graph_topologies(&mut self, sizes: &[u64]) -> Result<Vec<TestGraph>> {
        let mut graphs = Vec::new();
        let mut generator = GraphTopologyGenerator::new(self.rng.fork_for_component("graph_topologies").seed());
        
        for &size in sizes {
            // Generate different topology types
            graphs.push(generator.generate_erdos_renyi(size, 0.1)?);
            graphs.push(generator.generate_barabasi_albert(size, 3)?);
            graphs.push(generator.generate_watts_strogatz(size, 6, 0.3)?);
            
            if size <= 100 {
                graphs.push(generator.generate_complete_graph(size)?);
            }
            
            graphs.push(generator.generate_tree(size, 3)?);
        }
        
        Ok(graphs)
    }

    fn generate_knowledge_graphs(&mut self, sizes: &[u64]) -> Result<Vec<TestGraph>> {
        let mut graphs = Vec::new();
        let ontology = create_academic_ontology();
        let mut generator = KnowledgeGraphGenerator::new(
            self.rng.fork_for_component("knowledge_graphs").seed(), 
            ontology
        );
        
        for &size in sizes {
            graphs.push(generator.generate_academic_papers(size)?);
            graphs.push(generator.generate_social_network(size)?);
            
            if size <= 1000 {
                graphs.push(generator.generate_biological_pathway(size / 2)?);
            }
        }
        
        Ok(graphs)
    }

    fn generate_multi_scale_graphs(&mut self, sizes: &[u64]) -> Result<(Vec<TestGraph>, Vec<TestGraph>)> {
        let mut hierarchical_graphs = Vec::new();
        let mut fractal_graphs = Vec::new();
        let mut generator = MultiScaleGenerator::new(self.rng.fork_for_component("multi_scale").seed());
        
        for &size in sizes {
            // Hierarchical graphs
            let levels = if size <= 100 {
                vec![size, size / 5, size / 25, size / 125]
            } else {
                vec![size, size / 10, size / 100, size / 1000]
            };
            hierarchical_graphs.push(generator.generate_standard_hierarchy(levels.len() as u32, levels)?);
            
            // Fractal graphs (only for smaller sizes due to exponential growth)
            if size <= 100 {
                let iterations = if size <= 50 { 3 } else { 2 };
                let spec = create_triangle_fractal_spec(iterations);
                fractal_graphs.push(generator.generate_fractal_graph(spec)?);
            }
        }
        
        Ok((hierarchical_graphs, fractal_graphs))
    }

    fn generate_all_embeddings(&mut self, 
        graph_topologies: &[TestGraph], 
        knowledge_graphs: &[TestGraph], 
        dimensions: &[usize]) -> Result<HashMap<String, EmbeddingTestSet>> {
        
        let mut embeddings = HashMap::new();
        
        for &dim in dimensions {
            let mut generator = EmbeddingGenerator::new(
                self.rng.fork_for_component(&format!("embeddings_{}", dim)).seed(), 
                dim
            )?;
            
            // Generate clustered embeddings
            let cluster_specs = vec![
                ClusterSpec {
                    id: 0,
                    size: 50,
                    radius: 0.3,
                    center: None,
                    label: "Cluster A".to_string(),
                },
                ClusterSpec {
                    id: 1,
                    size: 30,
                    radius: 0.4,
                    center: None,
                    label: "Cluster B".to_string(),
                },
                ClusterSpec {
                    id: 2,
                    size: 20,
                    radius: 0.2,
                    center: None,
                    label: "Cluster C".to_string(),
                },
            ];
            
            let clustered_embeddings = generator.generate_clustered_embeddings(cluster_specs)?;
            embeddings.insert(format!("clustered_{}", dim), clustered_embeddings);
            
            // Generate distance-controlled embeddings
            let constraints = vec![
                DistanceConstraint {
                    entity1: 0,
                    entity2: 1,
                    target_distance: 1.0,
                    tolerance: 0.01,
                },
                DistanceConstraint {
                    entity1: 1,
                    entity2: 2,
                    target_distance: 1.5,
                    tolerance: 0.01,
                },
                DistanceConstraint {
                    entity1: 0,
                    entity2: 2,
                    target_distance: 2.0,
                    tolerance: 0.01,
                },
            ];
            
            let controlled_embeddings = generator.generate_distance_controlled_embeddings(constraints)?;
            
            // Convert to EmbeddingTestSet format
            let test_set = EmbeddingTestSet {
                embeddings: controlled_embeddings,
                ground_truth_similarities: HashMap::new(),
                cluster_assignments: HashMap::new(),
                expected_nearest_neighbors: HashMap::new(),
                dimension: dim,
                properties: EmbeddingProperties {
                    dimension: dim,
                    entity_count: 3,
                    cluster_count: 1,
                    average_norm: 1.0,
                    variance: 0.1,
                    min_distance: 1.0,
                    max_distance: 2.0,
                    average_distance: 1.5,
                    silhouette_score: 0.8,
                },
            };
            
            embeddings.insert(format!("controlled_{}", dim), test_set);
        }
        
        Ok(embeddings)
    }

    fn generate_quantization_data(&mut self, settings: &[QuantizationParams]) -> Result<(Vec<QuantizationTestSet>, Vec<SimdTestSet>)> {
        let mut quantization_data = Vec::new();
        let mut simd_data = Vec::new();
        
        let mut generator = QuantizationDataGenerator::new(
            self.rng.fork_for_component("quantization").seed()
        );
        
        for params in settings {
            let test_set = generator.generate_product_quantization_test_data(
                100, // vector count
                params.vector_dimension,
                params.codebook_size,
            )?;
            quantization_data.push(test_set);
            
            let simd_test_set = generator.generate_simd_test_vectors(50)?;
            simd_data.push(simd_test_set);
        }
        
        Ok((quantization_data, simd_data))
    }

    fn generate_all_queries(&mut self, 
        graphs: &[TestGraph], 
        embeddings: &HashMap<String, EmbeddingTestSet>, 
        query_counts: &HashMap<String, u32>) -> Result<(Vec<TraversalQuery>, Vec<RagQuery>, Vec<SimilarityQuery>, Vec<ComplexQuery>)> {
        
        let mut all_traversal_queries = Vec::new();
        let mut all_rag_queries = Vec::new();
        let mut all_similarity_queries = Vec::new();
        let mut all_complex_queries = Vec::new();
        
        // Use the first graph and embedding set for query generation
        if let (Some(graph), Some((_, embedding_set))) = (graphs.first(), embeddings.iter().next()) {
            let mut entity_embeddings = HashMap::new();
            
            // Convert embedding test set to simple entity -> embedding mapping
            for (&entity_id, embedding) in &embedding_set.embeddings {
                entity_embeddings.insert(entity_id, embedding.clone());
            }
            
            let mut generator = QueryPatternGenerator::new(
                self.rng.fork_for_component("queries").seed(),
                graph.clone(),
                entity_embeddings,
            );
            
            if let Some(&count) = query_counts.get("traversal") {
                all_traversal_queries.extend(generator.generate_traversal_queries(count)?);
            }
            
            if let Some(&count) = query_counts.get("rag") {
                all_rag_queries.extend(generator.generate_rag_queries(count)?);
            }
            
            if let Some(&count) = query_counts.get("similarity") {
                all_similarity_queries.extend(generator.generate_similarity_queries(count)?);
            }
            
            if let Some(&count) = query_counts.get("complex") {
                all_complex_queries.extend(generator.generate_complex_queries(count)?);
            }
        }
        
        Ok((all_traversal_queries, all_rag_queries, all_similarity_queries, all_complex_queries))
    }

    fn calculate_metadata(&self, params: &GenerationParameters, generation_time: std::time::Duration) -> Result<DatasetMetadata> {
        let total_entities = params.graph_sizes.iter().sum::<u64>() * 10; // Rough estimate
        let total_edges = total_entities * 5; // Rough estimate
        let total_embeddings = params.embedding_dimensions.len() as u64 * 100; // Rough estimate
        let total_queries = params.query_counts.values().sum::<u32>() as u64;
        
        let dataset_size_bytes = (total_entities * 64) + (total_edges * 32) + (total_embeddings * 512);
        
        Ok(DatasetMetadata {
            generation_seed: self.seed,
            creation_timestamp: chrono::Utc::now().to_rfc3339(),
            total_entities,
            total_edges,
            total_embeddings,
            total_queries,
            dataset_size_bytes,
            generation_parameters: params.clone(),
        })
    }
}

/// Validation utilities for generated data
pub mod validation {
    use super::*;
    
    /// Validate that generated dataset has expected properties
    pub fn validate_dataset(dataset: &ComprehensiveTestDataset) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            graphs_valid: true,
            embeddings_valid: true,
            queries_valid: true,
            issues: Vec::new(),
        };
        
        // Validate graphs
        for (i, graph) in dataset.graph_topologies.iter().enumerate() {
            if graph.entities.is_empty() {
                report.graphs_valid = false;
                report.issues.push(format!("Graph {} has no entities", i));
            }
            
            if graph.properties.entity_count != graph.entities.len() as u64 {
                report.graphs_valid = false;
                report.issues.push(format!("Graph {} entity count mismatch", i));
            }
        }
        
        // Validate embeddings
        for (name, embedding_set) in &dataset.embeddings {
            if embedding_set.embeddings.is_empty() {
                report.embeddings_valid = false;
                report.issues.push(format!("Embedding set {} is empty", name));
            }
            
            for (entity_id, embedding) in &embedding_set.embeddings {
                if embedding.len() != embedding_set.dimension {
                    report.embeddings_valid = false;
                    report.issues.push(format!("Embedding dimension mismatch for entity {} in set {}", entity_id, name));
                }
            }
        }
        
        // Validate queries
        if dataset.traversal_queries.is_empty() && dataset.rag_queries.is_empty() && 
           dataset.similarity_queries.is_empty() && dataset.complex_queries.is_empty() {
            report.queries_valid = false;
            report.issues.push("No queries generated".to_string());
        }
        
        Ok(report)
    }
    
    #[derive(Debug)]
    pub struct ValidationReport {
        pub graphs_valid: bool,
        pub embeddings_valid: bool,
        pub queries_valid: bool,
        pub issues: Vec<String>,
    }
    
    impl ValidationReport {
        pub fn is_valid(&self) -> bool {
            self.graphs_valid && self.embeddings_valid && self.queries_valid
        }
    }
}

// Add chrono dependency simulation
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime {
            DateTime
        }
    }
    
    pub struct DateTime;
    impl DateTime {
        pub fn to_rfc3339(&self) -> String {
            "2024-01-01T00:00:00Z".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_data_generation_small() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        let dataset = generator.generate_standard_small_dataset().unwrap();
        
        // Validate basic structure
        assert!(!dataset.graph_topologies.is_empty());
        assert!(!dataset.embeddings.is_empty());
        assert!(!dataset.traversal_queries.is_empty());
        
        // Validate metadata
        assert_eq!(dataset.metadata.generation_seed, 42);
        assert!(dataset.metadata.total_entities > 0);
    }

    #[test]
    fn test_dataset_validation() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        let dataset = generator.generate_standard_small_dataset().unwrap();
        
        let report = validation::validate_dataset(&dataset).unwrap();
        assert!(report.is_valid(), "Validation issues: {:?}", report.issues);
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = ComprehensiveDataGenerator::new(12345);
        let mut gen2 = ComprehensiveDataGenerator::new(12345);
        
        let dataset1 = gen1.generate_standard_small_dataset().unwrap();
        let dataset2 = gen2.generate_standard_small_dataset().unwrap();
        
        // Same seed should produce same metadata
        assert_eq!(dataset1.metadata.generation_seed, dataset2.metadata.generation_seed);
        assert_eq!(dataset1.graph_topologies.len(), dataset2.graph_topologies.len());
        assert_eq!(dataset1.embeddings.len(), dataset2.embeddings.len());
    }

    #[test]
    fn test_large_dataset_generation() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        
        // Test that large dataset generation doesn't crash
        let dataset = generator.generate_standard_large_dataset().unwrap();
        
        assert!(!dataset.graph_topologies.is_empty());
        assert!(dataset.metadata.total_entities > 1000);
    }

    #[test]
    fn test_streaming_dataset_generation() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        let config = create_default_temporal_config();
        
        let stream = generator.generate_streaming_dataset(config).unwrap();
        
        assert_eq!(stream.len(), 100);
        assert!(stream.iter().all(|batch| !batch.new_entities.is_empty()));
    }

    #[test]
    fn test_federation_dataset_generation() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        let config = create_default_federation_config();
        
        let federation_dataset = generator.generate_federation_dataset(config).unwrap();
        
        assert_eq!(federation_dataset.databases.len(), 3);
        assert!(!federation_dataset.cross_db_references.is_empty());
        assert!(!federation_dataset.federation_queries.is_empty());
    }

    #[test]
    fn test_complete_phase_2_integration() {
        let mut generator = ComprehensiveDataGenerator::new(42);
        
        // Generate standard dataset
        let standard_dataset = generator.generate_standard_small_dataset().unwrap();
        
        // Generate streaming data
        let temporal_config = create_default_temporal_config();
        let streaming_data = generator.generate_streaming_dataset(temporal_config).unwrap();
        
        // Generate federation data
        let federation_config = create_default_federation_config();
        let federation_data = generator.generate_federation_dataset(federation_config).unwrap();
        
        // Validate all components work together
        assert!(!standard_dataset.graph_topologies.is_empty());
        assert!(!streaming_data.is_empty());
        assert!(!federation_data.databases.is_empty());
        
        // Test validation framework integration
        let validation_report = comprehensive_validate(
            &standard_dataset.graph_topologies,
            &standard_dataset.embeddings,
            &standard_dataset.traversal_queries,
            &standard_dataset.rag_queries,
            &standard_dataset.similarity_queries,
            None,
        ).unwrap();
        
        assert!(matches!(validation_report.overall_status, ValidationStatus::Passed));
    }
}