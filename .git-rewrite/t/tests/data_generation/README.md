# Phase 2 Synthetic Data Generation Framework

This module provides a comprehensive synthetic data generation framework for testing the LLMKG system with deterministic, reproducible datasets and known properties.

## Overview

The Phase 2 framework implements all requirements from the Phase 2 specification:

- **Deterministic data generation** - All generators use seeded RNGs for reproducibility
- **Comprehensive coverage** - Multiple graph topologies, embedding types, and query patterns
- **Scalable generation** - Supports datasets from tiny (testing) to huge (production scale)
- **Property validation** - Complete validation framework with detailed reporting
- **Performance optimization** - Efficient algorithms with known mathematical properties
- **Golden standard creation** - Exact computation engines for test validation
- **Full integration** - Seamlessly integrated with Phase 1 infrastructure

## Core Components

### 1. Graph Topology Generators (`graph_topologies.rs`)

Generates mathematically precise graph structures:

- **Erdős–Rényi graphs** - Random graphs with controlled edge probability
- **Barabási–Albert graphs** - Scale-free networks with preferential attachment
- **Watts–Strogatz graphs** - Small-world networks with adjustable clustering
- **Complete graphs** - Fully connected networks for baseline testing
- **Tree structures** - Hierarchical graphs with controlled branching

```rust
use crate::data_generation::GraphTopologyGenerator;

let mut generator = GraphTopologyGenerator::new(42);
let graph = generator.generate_erdos_renyi(1000, 0.1)?;
assert_eq!(graph.entities.len(), 1000);
```

### 2. Vector Embedding Generators (`embeddings.rs`)

Creates controlled vector embeddings with:

- **Clustering control** - Precise cluster specifications with centers and radii
- **Distance constraints** - Exact distance relationships between entities
- **Hierarchical structures** - Multi-level embedding hierarchies
- **Similarity preservation** - Guaranteed similarity relationships

```rust
use crate::data_generation::EmbeddingGenerator;

let mut generator = EmbeddingGenerator::new(42, 128)?;
let embeddings = generator.generate_clustered_embeddings(cluster_specs)?;
```

### 3. Knowledge Graph Generators (`knowledge_graphs.rs`)

Domain-specific knowledge graph generation:

- **Academic papers** - Citation networks with realistic publication patterns
- **Social networks** - User interaction graphs with community structure
- **Biological pathways** - Protein interaction and metabolic networks

```rust
use crate::data_generation::KnowledgeGraphGenerator;

let ontology = create_academic_ontology();
let mut generator = KnowledgeGraphGenerator::new(42, ontology);
let papers_graph = generator.generate_academic_papers(500)?;
```

### 4. Multi-Scale Graph Generation (`multi_scale.rs`)

Generates complex hierarchical and fractal structures:

- **Hierarchical graphs** - Multi-level organizational structures
- **Fractal graphs** - Self-similar recursive patterns
- **Scale-free properties** - Consistent behavior across scales

```rust
use crate::data_generation::MultiScaleGenerator;

let mut generator = MultiScaleGenerator::new(42);
let levels = vec![1000, 100, 10];
let hierarchy = generator.generate_standard_hierarchy(3, levels)?;
```

### 5. Quantization Test Data (`quantization_data.rs`)

Specialized data for compression testing:

- **Product Quantization** - PQ codebook and vector sets
- **SIMD test vectors** - Hardware-optimized test cases
- **Compression analysis** - Accuracy vs. compression trade-offs

```rust
use crate::data_generation::QuantizationDataGenerator;

let mut generator = QuantizationDataGenerator::new(42);
let test_set = generator.generate_product_quantization_test_data(1000, 128, 256)?;
```

### 6. Query Pattern Generators (`query_patterns.rs`)

Comprehensive query generation:

- **Traversal queries** - Path finding, neighborhood exploration, random walks
- **RAG queries** - Retrieval-augmented generation with context selection
- **Similarity queries** - Vector similarity search with various metrics
- **Complex queries** - Multi-step operations combining multiple techniques

```rust
use crate::data_generation::QueryPatternGenerator;

let mut generator = QueryPatternGenerator::new(42, graph, embeddings);
let traversal_queries = generator.generate_traversal_queries(50)?;
let rag_queries = generator.generate_rag_queries(30)?;
```

### 7. Golden Standards Computation (`golden_standards.rs`)

Exact computation of expected results:

- **Traversal results** - Exact paths, neighborhoods, and walk statistics
- **Similarity results** - Precise distance computations and rankings
- **RAG results** - Context quality and relevance scores
- **Performance expectations** - Mathematical bounds on computational complexity

```rust
use crate::data_generation::ExactComputationEngine;

let engine = ExactComputationEngine::new();
let golden_standards = engine.compute_all_standards(&graph, &queries)?;
```

### 8. Data Quality Validation (`validation.rs`)

Comprehensive validation framework:

- **Mathematical property validation** - Verifies theoretical guarantees
- **Structural integrity checks** - Ensures data consistency
- **Performance validation** - Confirms computational expectations
- **Cross-component consistency** - Validates integration correctness

```rust
use crate::data_generation::{comprehensive_validate, quick_validate};

// Quick validation for basic checks
let is_valid = quick_validate(&graphs, &embeddings)?;

// Comprehensive validation with detailed reporting
let report = comprehensive_validate(&graphs, &embeddings, &queries, &rag_queries, &similarity_queries, Some(&golden_standards))?;
```

### 9. Streaming Data Generation (`streaming.rs`)

Temporal data generation with real-time characteristics:

- **Temporal batches** - Time-ordered data generation
- **Entity lifecycle tracking** - Creation, updates, and deletion patterns
- **Seasonal patterns** - Configurable temporal variations
- **Burst events** - Sudden activity spikes
- **Real-time simulation** - Streaming processing simulation

```rust
use crate::data_generation::StreamingDataGenerator;

let config = create_default_temporal_config();
let mut generator = StreamingDataGenerator::new(42, config);
let stream = generator.generate_temporal_stream(100)?;
```

### 10. Federation Test Data (`federation.rs`)

Multi-database federation scenarios:

- **Cross-database references** - Links between entities in different databases
- **Federation queries** - Joins, aggregations, and searches across databases
- **Consistency test cases** - Eventual, strong, and causal consistency scenarios
- **Performance modeling** - Multi-database operation expectations

```rust
use crate::data_generation::FederationDataGenerator;

let config = create_default_federation_config();
let mut generator = FederationDataGenerator::new(42, config);
let federation_dataset = generator.generate_federation_dataset()?;
```

## Integration with Phase 1 Infrastructure

The Phase 2 framework seamlessly integrates with Phase 1 infrastructure through the `data_generation.rs` module in `tests/infrastructure/`:

```rust
use crate::infrastructure::{DataGenerator, GenerationParams, DataSize};

let config = TestConfig::default();
let mut generator = DataGenerator::new(config);

let params = GenerationParams {
    data_size: DataSize::Medium,
    scenario: GenerationScenario::Complete,
    validation_enabled: true,
    cache_results: true,
};

let dataset_id = generator.generate_data(&params)?;
```

## Usage Examples

### Basic Dataset Generation

```rust
use crate::data_generation::ComprehensiveDataGenerator;

let mut generator = ComprehensiveDataGenerator::new(42);

// Generate standard small dataset
let dataset = generator.generate_standard_small_dataset()?;

// Generate custom dataset
let params = GenerationParameters {
    graph_sizes: vec![100, 500, 1000],
    embedding_dimensions: vec![64, 128, 256],
    query_counts: {
        let mut counts = HashMap::new();
        counts.insert("traversal".to_string(), 50);
        counts.insert("rag".to_string(), 30);
        counts.insert("similarity".to_string(), 75);
        counts
    },
    quantization_settings: vec![/* ... */],
    validation_enabled: true,
};

let custom_dataset = generator.generate_complete_dataset(params)?;
```

### Streaming Data Generation

```rust
use crate::data_generation::{StreamingDataGenerator, create_high_frequency_config};

let config = create_high_frequency_config();
let mut generator = StreamingDataGenerator::new(42, config);

// Generate temporal stream
let batches = generator.generate_temporal_stream(1000)?;

// Real-time processing simulation
let results = generator.simulate_realtime_processing(100, |batch| {
    // Process batch
    Ok(ProcessingOutcome::Success { entities_processed: batch.new_entities.len() as u64 })
})?;
```

### Federation Testing

```rust
use crate::data_generation::{FederationDataGenerator, create_geographic_federation_config};

let config = create_geographic_federation_config();
let mut generator = FederationDataGenerator::new(42, config);

let federation_dataset = generator.generate_federation_dataset()?;

// Validate cross-database consistency
for test_case in &federation_dataset.consistency_test_cases {
    // Run consistency test
    // Verify expected behavior
}
```

### Validation and Quality Assurance

```rust
use crate::data_generation::{DataQualityValidator, ValidationStatus};

let validator = DataQualityValidator::new(1e-10, true); // High precision, strict mode

let report = validator.validate_dataset(
    &graphs,
    &embeddings,
    &queries,
    &rag_queries,
    &similarity_queries,
    Some(&golden_standards),
)?;

match report.overall_status {
    ValidationStatus::Passed => println!("Data quality validation passed"),
    ValidationStatus::Warning => println!("Validation completed with warnings: {:?}", report.issues),
    ValidationStatus::Failed => println!("Validation failed: {:?}", report.issues),
}

println!("Pass rate: {:.2}%", report.summary.pass_rate * 100.0);
```

## Performance Characteristics

The framework is designed for:

- **Deterministic generation** - Same seed produces identical results
- **Scalable performance** - Efficient algorithms for large datasets
- **Memory efficiency** - Streaming generation for large-scale data
- **Parallel processing** - Independent component generation
- **Caching support** - Reuse of expensive computations

## Configuration Options

### Dataset Sizes

- **Tiny** - For unit tests (10-50 entities)
- **Small** - For integration tests (50-200 entities)  
- **Medium** - For performance tests (500-2000 entities)
- **Large** - For stress tests (5000-10000 entities)
- **Huge** - For scalability tests (50000+ entities)

### Generation Scenarios

- **GraphTopologies** - Focus on graph structure generation
- **KnowledgeGraphs** - Domain-specific knowledge graphs
- **EmbeddingClusters** - Vector embedding generation
- **MultiScale** - Hierarchical and fractal structures
- **Quantization** - Compression testing data
- **QueryPatterns** - Query generation and golden standards
- **Complete** - All components with full integration

### Validation Levels

- **Quick validation** - Basic structural checks
- **Comprehensive validation** - Full mathematical verification
- **Performance validation** - Computational expectations
- **Consistency validation** - Cross-component integrity

## Testing and Verification

The framework includes extensive tests:

```bash
# Run all Phase 2 tests
cargo test data_generation

# Run specific component tests
cargo test graph_topologies
cargo test streaming
cargo test federation

# Run integration tests
cargo test test_complete_phase_2_integration
```

## Error Handling and Diagnostics

All generators provide detailed error reporting:

- **Generation errors** - Issues during data creation
- **Validation errors** - Quality assurance failures
- **Performance errors** - Computational bound violations
- **Consistency errors** - Cross-component integrity issues

## Future Extensions

The framework is designed for extensibility:

- **New graph types** - Additional topology generators
- **Domain-specific generators** - Specialized knowledge graphs
- **Advanced query patterns** - Complex multi-step operations
- **Real-time streaming** - Live data generation
- **Distributed generation** - Multi-node data creation

## Dependencies

- `anyhow` - Error handling
- `serde` - Serialization support
- `std::collections` - Data structures
- Phase 1 infrastructure - Base testing framework

## Documentation and Support

- API documentation: Generated with `cargo doc`
- Examples: See `tests/` directory
- Performance benchmarks: `cargo bench`
- Issues: Report via GitHub issues

This comprehensive framework provides everything needed for thorough testing of the LLMKG system with synthetic data that has known properties and expected behaviors.