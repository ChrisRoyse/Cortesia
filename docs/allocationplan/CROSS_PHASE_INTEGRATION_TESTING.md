# Cross-Phase Integration Testing Strategy

## Executive Summary

This document defines comprehensive integration testing strategies that span multiple phases of the CortexKG system. Following London School TDD principles, we start with fully mocked cross-phase interactions, progressively unmock components, and finally test with real connections and data flows.

## Core Integration Paths

### Critical Path 1: Concept Lifecycle (Phases 1→2→3→4→5→7)
**Journey**: Raw concept → Allocation → Storage → Inheritance → Versioning → Query

### Critical Path 2: Multi-Database Knowledge Emergence (Phases 6→10→11)
**Journey**: Cross-database patterns → Advanced algorithms → Production features

### Critical Path 3: MCP Intelligence Flow (Phases 8→9→11)
**Journey**: MCP tools → WASM interface → Production deployment

### Critical Path 4: Neural Processing Pipeline (Phases 0→1→2→3)
**Journey**: Foundation → Cortical columns → Allocation → Sparse storage

## Integration Test Suite Architecture

### 1. Mock-First Integration Tests

```rust
use mockall::*;
use cortexkg::test_framework::*;

#[cfg(test)]
mod phase_integration_mocks {
    use super::*;
    
    // Mock all phase interfaces first
    mock! {
        PhaseInterface {
            // Phase 0A: Parsing Quality Assurance (CRITICAL)
            fn validate_parsing_quality(&self, input: &str) -> Result<Vec<ValidatedFact>>;
            fn check_quality_gates(&self, facts: &[ValidatedFact]) -> Result<Vec<ValidatedFact>>;
            fn get_parsing_metrics(&self) -> Result<ParsingMetrics>;
            
            // Phase 1: Cortical Column
            fn create_cortical_column(&self, config: ColumnConfig) -> Result<ColumnId>;
            fn process_spike_pattern(&self, pattern: SpikePattern) -> Result<AllocationHint>;
            
            // Phase 2: Allocation Engine
            fn allocate_concept(&self, hint: AllocationHint) -> Result<ConceptAddress>;
            fn validate_allocation(&self, address: ConceptAddress) -> Result<bool>;
            
            // Phase 3: Sparse Storage
            fn store_sparse(&self, concept: Concept, address: ConceptAddress) -> Result<StorageId>;
            fn retrieve_sparse(&self, address: ConceptAddress) -> Result<Concept>;
            
            // Phase 4: Inheritance
            fn apply_inheritance(&self, concept: Concept) -> Result<InheritedConcept>;
            fn compress_via_inheritance(&self, concepts: Vec<Concept>) -> Result<CompressionRatio>;
            
            // Phase 5: Temporal Versioning
            fn create_version(&self, concept: InheritedConcept) -> Result<VersionedConcept>;
            fn branch_timeline(&self, at: Timestamp) -> Result<BranchId>;
            
            // Phase 7: Query Through Activation
            fn query_by_activation(&self, pattern: ActivationPattern) -> Result<QueryResult>;
            fn visualize_activation(&self) -> Result<ActivationMap>;
        }
    }
}

#[tokio::test]
async fn test_concept_lifecycle_fully_mocked() {
    // Stage 1: All components mocked
    let mut mock_system = MockPhaseInterface::new();
    
    // Phase 0A: Parsing Quality (CRITICAL - must pass first)
    mock_system
        .expect_validate_parsing_quality()
        .withf(|input| !input.is_empty())
        .returning(|input| Ok(vec![
            ValidatedFact {
                content: FactContent::from(input),
                quality_score: 0.95,
                confidence_components: ConfidenceComponents {
                    syntax_confidence: 0.98,
                    entity_confidence: 0.92,
                    semantic_confidence: 0.94,
                    ..Default::default()
                },
                validation_chain: vec!["syntax", "semantic", "logical"],
                ..Default::default()
            }
        ]));
    
    mock_system
        .expect_check_quality_gates()
        .returning(|facts| Ok(facts.iter()
            .filter(|f| f.quality_score > 0.8)
            .cloned()
            .collect()));
    
    // Define expected behavior for full concept lifecycle
    mock_system
        .expect_create_cortical_column()
        .times(4)  // 4 columns: semantic, structural, temporal, exception
        .returning(|_| Ok(ColumnId::new()));
    
    mock_system
        .expect_process_spike_pattern()
        .withf(|pattern| pattern.spike_count > 0)
        .returning(|_| Ok(AllocationHint {
            confidence: 0.95,
            suggested_address: ConceptAddress::random(),
            column_votes: vec![0.9, 0.85, 0.92, 0.88],
        }));
    
    mock_system
        .expect_allocate_concept()
        .returning(|hint| Ok(hint.suggested_address));
    
    mock_system
        .expect_store_sparse()
        .returning(|_, addr| Ok(StorageId::from_address(addr)));
    
    mock_system
        .expect_apply_inheritance()
        .returning(|concept| Ok(InheritedConcept {
            base: concept,
            compression_ratio: 0.3,
            inherited_properties: HashMap::new(),
        }));
    
    mock_system
        .expect_create_version()
        .returning(|inherited| Ok(VersionedConcept {
        
            inherited,
            version: Version::new(1, 0, 0),
            timestamp: Timestamp::now(),
        }));
    
    mock_system
        .expect_query_by_activation()
        .returning(|_| Ok(QueryResult {
            concepts: vec![],
            activation_strength: 0.87,
            response_time_ms: 42,
        }));
    
    // Execute full lifecycle with mocks
    let lifecycle_result = execute_concept_lifecycle(&mock_system, "test_concept").await;
    assert!(lifecycle_result.is_ok());
    assert!(lifecycle_result.unwrap().response_time_ms < 100);
}

#[tokio::test]
async fn test_phase_0a_quality_gate_integration() {
    // Test Phase 0A -> Phase 2 integration specifically
    let mut mock_system = MockPhaseInterface::new();
    
    // Test high-quality parsing
    let high_quality_input = "The CPU has 8 cores running at 3.2 GHz";
    mock_system
        .expect_validate_parsing_quality()
        .with(eq(high_quality_input))
        .returning(|_| Ok(vec![
            ValidatedFact {
                content: FactContent::new("CPU", "has_cores", "8"),
                quality_score: 0.96,
                validation_chain: vec!["syntax", "semantic", "logical"],
                ambiguity_flags: vec![],
                ..Default::default()
            },
            ValidatedFact {
                content: FactContent::new("CPU", "clock_speed", "3.2 GHz"),
                quality_score: 0.94,
                validation_chain: vec!["syntax", "semantic", "logical"],
                ambiguity_flags: vec![],
                ..Default::default()
            }
        ]));
    
    // Test low-quality parsing rejection
    let low_quality_input = "thing stuff whatever";
    mock_system
        .expect_validate_parsing_quality()
        .with(eq(low_quality_input))
        .returning(|_| Ok(vec![
            ValidatedFact {
                content: FactContent::new("thing", "unknown", "stuff"),
                quality_score: 0.3,  // Below threshold
                validation_chain: vec!["syntax"],  // Incomplete validation
                ambiguity_flags: vec!["unresolved_entity", "unclear_relation"],
                ..Default::default()
            }
        ]));
    
    mock_system
        .expect_check_quality_gates()
        .returning(|facts| Ok(facts.iter()
            .filter(|f| f.quality_score >= 0.8 && 
                       f.validation_chain.len() == 3 &&
                       f.ambiguity_flags.is_empty())
            .cloned()
            .collect()));
    
    // High-quality should pass through to allocation
    let high_quality_facts = mock_system.validate_parsing_quality(high_quality_input).await?;
    let passed_facts = mock_system.check_quality_gates(&high_quality_facts).await?;
    assert_eq!(passed_facts.len(), 2);
    
    // Low-quality should be rejected
    let low_quality_facts = mock_system.validate_parsing_quality(low_quality_input).await?;
    let rejected_facts = mock_system.check_quality_gates(&low_quality_facts).await?;
    assert_eq!(rejected_facts.len(), 0);
}
```

### 2. Progressive Unmocking Strategy

```rust
#[cfg(test)]
mod progressive_unmocking {
    use super::*;
    
    struct PartiallyMockedSystem {
        // Real components
        cortical_columns: RealCorticalColumns,
        allocation_engine: RealAllocationEngine,
        
        // Still mocked components
        sparse_storage: MockSparseStorage,
        inheritance_system: MockInheritanceSystem,
        temporal_versioning: MockTemporalVersioning,
        query_engine: MockQueryEngine,
    }
    
    #[tokio::test]
    async fn test_concept_lifecycle_partial_unmocking() {
        // Stage 2: Unmock early phases (1-2), keep later phases mocked
        let mut system = PartiallyMockedSystem::new();
        
        // Real cortical columns and allocation
        let columns = system.cortical_columns.initialize(
            CorticalConfig {
                num_columns: 4,
                neurons_per_column: 100,
                spike_threshold: 0.7,
                lateral_inhibition: true,
            }
        ).await?;
        
        let allocation_engine = system.allocation_engine.initialize(
            AllocationConfig {
                ttfs_precision_us: 10,
                refractory_period_ms: 2,
                voting_threshold: 0.75,
            }
        ).await?;
        
        // Mock storage and beyond
        system.sparse_storage
            .expect_store()
            .returning(|concept, address| Ok(StorageId::new()));
        
        system.inheritance_system
            .expect_apply()
            .returning(|concept| Ok(apply_simple_inheritance(concept)));
        
        // Execute with partial mocking
        let test_concept = Concept::new("integration_test", vec![0.1, 0.2, 0.3]);
        
        // Real neural processing
        let spike_pattern = encode_to_spikes(&test_concept);
        let allocation_hint = columns.process(spike_pattern).await?;
        let address = allocation_engine.allocate(allocation_hint).await?;
        
        // Mocked storage and inheritance
        let storage_id = system.sparse_storage.store(test_concept.clone(), address).await?;
        let inherited = system.inheritance_system.apply(test_concept).await?;
        
        assert!(address.is_valid());
        assert!(inherited.compression_ratio < 0.5);
    }
}
```

### 3. Full Integration Tests with Real Components

```rust
#[cfg(test)]
mod full_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_end_to_end_concept_lifecycle() {
        // Stage 3: All components real, no mocks
        let system = IntegratedCortexKG::new()
            .with_cpu_optimization()  // No GPU required
            .with_test_configuration()
            .await?;
        
        // Test 1: Simple concept flow
        let concept = Concept::new("email", vec![
            ("type", "communication"),
            ("protocol", "SMTP"),
            ("port", "587"),
        ]);
        
        // Full lifecycle execution
        let lifecycle_timer = Instant::now();
        
        // Phase 1-2: Neural allocation
        let allocation = system.allocate_neural(concept.clone()).await?;
        assert!(allocation.confidence > 0.8);
        
        // Phase 3: Sparse storage
        let storage_result = system.store_sparse(concept.clone(), allocation.address).await?;
        assert!(storage_result.sparsity_ratio > 0.7);
        
        // Phase 4: Inheritance application
        let inherited = system.apply_inheritance(concept.clone()).await?;
        assert!(inherited.shared_properties.len() > 0);
        
        // Phase 5: Temporal versioning
        let versioned = system.create_version(inherited).await?;
        assert_eq!(versioned.version, Version::new(1, 0, 0));
        
        // Phase 7: Query through activation
        let query_pattern = ActivationPattern::from_query("communication protocol");
        let results = system.query_by_activation(query_pattern).await?;
        assert!(results.contains_concept("email"));
        
        let total_time = lifecycle_timer.elapsed();
        println!("Full concept lifecycle time: {:?}", total_time);
        
        // Performance assertion - relative to baseline
        let baseline_time = get_baseline_kg_operation_time();
        assert!(total_time < baseline_time * 0.8); // 20% faster than baseline
    }
    
    #[tokio::test]
    async fn test_cross_database_knowledge_emergence() {
        // Test Critical Path 2: Multi-database integration
        let system = IntegratedCortexKG::new()
            .with_cpu_optimization()
            .await?;
        
        // Create multiple database branches
        let main_db = system.get_main_database();
        let branch_a = system.create_branch("feature-a").await?;
        let branch_b = system.create_branch("feature-b").await?;
        
        // Add concepts to different branches
        let concept_a = Concept::new("quantum_computer", vec![
            ("type", "computer"),
            ("technology", "quantum"),
            ("qubits", "1000"),
        ]);
        
        let concept_b = Concept::new("classical_computer", vec![
            ("type", "computer"),
            ("technology", "classical"),
            ("bits", "64"),
        ]);
        
        // Store in different branches
        branch_a.allocate_and_store(concept_a).await?;
        branch_b.allocate_and_store(concept_b).await?;
        
        // Phase 6: Cross-database pattern detection
        let patterns = system.detect_cross_db_patterns().await?;
        assert!(patterns.contains_pattern("shared_type:computer"));
        
        // Phase 10: Advanced algorithm application
        let reasoning_result = system.apply_analogical_reasoning(
            "quantum_computer",
            "classical_computer"
        ).await?;
        assert!(reasoning_result.analogy_strength > 0.7);
        
        // Phase 11: Production merge
        let merge_result = system.merge_branches_production(vec![branch_a, branch_b]).await?;
        assert!(merge_result.conflicts.is_empty());
        assert!(merge_result.emerged_concepts.contains("computer_taxonomy"));
    }
    
    #[tokio::test]
    async fn test_mcp_intelligence_pipeline() {
        // Test Critical Path 3: MCP → WASM → Production
        let system = IntegratedCortexKG::new()
            .with_cpu_optimization()
            .await?;
        
        // Phase 8: MCP tool interaction
        let mcp_request = MpcRequest {
            tool: "create_concept",
            params: json!({
                "name": "ai_assistant",
                "properties": {
                    "type": "software",
                    "capability": "conversation"
                }
            }),
        };
        
        let mcp_response = system.handle_mcp_request(mcp_request).await?;
        assert!(mcp_response.success);
        
        // Phase 9: WASM compilation and execution
        let wasm_module = system.compile_to_wasm().await?;
        let wasm_instance = WasmInstance::new(wasm_module)?;
        
        let wasm_result = wasm_instance.execute_query(
            "find similar to ai_assistant"
        ).await?;
        assert!(wasm_result.execution_time_ms < 50);
        
        // Phase 11: Production deployment verification
        let production_health = system.check_production_health().await?;
        assert!(production_health.uptime_percent > 99.0);
        assert!(production_health.active_connections > 0);
    }
}
```

### 4. Integration Failure Scenarios

```rust
#[cfg(test)]
mod integration_failure_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cascade_failure_recovery() {
        let system = IntegratedCortexKG::new()
            .with_failure_injection()
            .await?;
        
        // Inject failure at Phase 3 (storage)
        system.inject_failure_at_phase(3, FailureType::NetworkTimeout);
        
        let concept = Concept::new("test_failure", vec![]);
        let result = system.allocate_and_store(concept).await;
        
        // Should gracefully handle storage failure
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().phase, 3);
        
        // System should remain healthy
        let health = system.check_health().await?;
        assert!(health.is_healthy);
        
        // Retry should succeed after clearing failure
        system.clear_injected_failures();
        let retry_result = system.allocate_and_store(concept).await;
        assert!(retry_result.is_ok());
    }
    
    #[tokio::test]
    async fn test_phase_rollback_on_error() {
        let system = IntegratedCortexKG::new()
            .with_transaction_support()
            .await?;
        
        // Start transaction spanning multiple phases
        let tx = system.begin_transaction().await?;
        
        let concept = Concept::new("transactional_test", vec![]);
        
        // Execute phases 1-4
        let allocation = tx.allocate(concept.clone()).await?;
        let storage = tx.store(concept.clone(), allocation.address).await?;
        let inherited = tx.apply_inheritance(concept.clone()).await?;
        
        // Inject failure at phase 5
        system.inject_failure_at_phase(5, FailureType::ValidationError);
        
        // This should fail
        let version_result = tx.create_version(inherited).await;
        assert!(version_result.is_err());
        
        // Transaction should rollback all changes
        tx.rollback().await?;
        
        // Verify rollback succeeded
        let query_result = system.query_concept("transactional_test").await?;
        assert!(query_result.is_none());
    }
}
```

### 5. Performance Integration Tests

```rust
#[cfg(test)]
mod performance_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_comparative_performance() {
        // Compare against existing knowledge graph baselines
        let cortexkg = IntegratedCortexKG::new()
            .with_cpu_optimization()
            .await?;
        
        let baseline_kg = BaselineKnowledgeGraph::new();  // Mock of existing system
        
        // Test dataset
        let test_concepts: Vec<Concept> = generate_test_concepts(1000);
        
        // Measure CortexKG performance
        let cortex_start = Instant::now();
        for concept in &test_concepts {
            cortexkg.allocate_and_store(concept.clone()).await?;
        }
        let cortex_time = cortex_start.elapsed();
        
        // Measure baseline performance (simulated)
        let baseline_time = Duration::from_millis(5000); // Typical KG operation time
        
        // Assert we're faster (but don't require specific numbers)
        assert!(cortex_time < baseline_time);
        println!("Performance improvement: {:.2}x", 
                 baseline_time.as_secs_f64() / cortex_time.as_secs_f64());
    }
    
    #[tokio::test]
    async fn test_cpu_only_performance() {
        // Ensure no GPU dependency
        let system = IntegratedCortexKG::new()
            .with_cpu_optimization()
            .without_gpu()
            .await?;
        
        // Should work fine on CPU only
        let stress_test_result = system.run_stress_test(
            StressTestConfig {
                concurrent_operations: 100,
                duration_seconds: 10,
                operation_mix: OperationMix::Balanced,
            }
        ).await?;
        
        assert!(stress_test_result.errors.is_empty());
        assert!(stress_test_result.avg_latency_ms < 100.0);
        
        // No GPU should be detected/used
        assert_eq!(stress_test_result.gpu_utilization, 0.0);
        assert!(stress_test_result.cpu_utilization > 0.0);
    }
}
```

### 6. Data Flow Validation Tests

```rust
#[cfg(test)]
mod data_flow_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_transformation_through_phases() {
        let system = IntegratedCortexKG::new().await?;
        
        // Track data transformations
        let mut transformations = Vec::new();
        
        // Input data
        let raw_input = "The quick brown fox jumps over the lazy dog";
        transformations.push(("raw_input", format!("{:?}", raw_input)));
        
        // Phase 1: Neural encoding
        let neural_encoding = system.encode_to_neural(raw_input).await?;
        transformations.push(("neural_spikes", format!("spikes: {}", neural_encoding.spike_count)));
        
        // Phase 2: Allocation
        let allocation = system.allocate_from_neural(neural_encoding).await?;
        transformations.push(("allocation_address", format!("{:?}", allocation.address)));
        
        // Phase 3: Sparse representation
        let sparse = system.to_sparse_representation(allocation).await?;
        transformations.push(("sparse_ratio", format!("{:.2}", sparse.sparsity_ratio)));
        
        // Phase 4: Inheritance compression
        let inherited = system.apply_inheritance_compression(sparse).await?;
        transformations.push(("compression_ratio", format!("{:.2}", inherited.compression_ratio)));
        
        // Validate data integrity through transformations
        let reconstructed = system.reconstruct_from_inherited(inherited).await?;
        assert!(reconstructed.similarity_to_original > 0.9);
        
        // Log transformation pipeline
        println!("Data transformation pipeline:");
        for (phase, data) in transformations {
            println!("  {} -> {}", phase, data);
        }
    }
}
```

## Integration Test Execution Strategy

### 1. Test Execution Order

```yaml
integration_test_pipeline:
  stage_1_mocked:
    - test_concept_lifecycle_fully_mocked
    - test_cross_db_patterns_mocked
    - test_mcp_pipeline_mocked
    duration: ~5 minutes
    
  stage_2_partial:
    - test_concept_lifecycle_partial_unmocking
    - test_phase_boundaries_partial
    - test_error_propagation_partial
    duration: ~10 minutes
    
  stage_3_full:
    - test_end_to_end_concept_lifecycle
    - test_cross_database_knowledge_emergence
    - test_mcp_intelligence_pipeline
    duration: ~20 minutes
    
  stage_4_stress:
    - test_comparative_performance
    - test_cpu_only_performance
    - test_cascade_failure_recovery
    duration: ~30 minutes
```

### 2. Continuous Integration Configuration

```yaml
# .github/workflows/integration-tests.yml
name: Cross-Phase Integration Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Nightly full integration tests

jobs:
  quick_integration:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - name: Run Stage 1 (Mocked) Tests
        run: cargo test --features integration_mocked
        
  full_integration:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Run All Integration Stages
        run: |
          cargo test --features integration_stage_1
          cargo test --features integration_stage_2
          cargo test --features integration_stage_3
          cargo test --features integration_stage_4
```

### 3. Performance Baseline Tracking

```rust
// Baseline performance tracking for comparative testing
pub struct PerformanceBaselines {
    // Track performance relative to other systems
    pub neo4j_baseline: BaselineMetrics,
    pub dgraph_baseline: BaselineMetrics,
    pub traditional_sql_baseline: BaselineMetrics,
}

impl PerformanceBaselines {
    pub fn load() -> Self {
        Self {
            neo4j_baseline: BaselineMetrics {
                concept_creation_ms: 50.0,
                pattern_query_ms: 100.0,
                cross_db_join_ms: 500.0,
            },
            dgraph_baseline: BaselineMetrics {
                concept_creation_ms: 30.0,
                pattern_query_ms: 80.0,
                cross_db_join_ms: 400.0,
            },
            traditional_sql_baseline: BaselineMetrics {
                concept_creation_ms: 20.0,
                pattern_query_ms: 200.0,
                cross_db_join_ms: 1000.0,
            },
        }
    }
    
    pub fn assert_better_than_baseline(&self, actual: &ActualMetrics) {
        // We aim to beat the best baseline, but don't require specific numbers
        let best_creation = self.best_baseline_for("creation");
        let best_query = self.best_baseline_for("query");
        
        assert!(actual.concept_creation_ms < best_creation,
                "Should be faster than best baseline ({:.2}ms)", best_creation);
        assert!(actual.pattern_query_ms < best_query,
                "Should be faster than best baseline ({:.2}ms)", best_query);
    }
}
```

### 7. Sophisticated Test Data Generation

```rust
pub struct TestDataGenerator {
    quality_distribution: QualityDistribution,
    ambiguity_generator: AmbiguityGenerator,
    edge_case_factory: EdgeCaseFactory,
}

impl TestDataGenerator {
    pub fn generate_parsing_test_data(&self, count: usize) -> Vec<TestDocument> {
        let mut test_data = Vec::new();
        
        // 60% high-quality, well-formed documents
        let high_quality_count = (count as f32 * 0.6) as usize;
        for _ in 0..high_quality_count {
            test_data.push(self.generate_high_quality_document());
        }
        
        // 20% ambiguous documents requiring resolution
        let ambiguous_count = (count as f32 * 0.2) as usize;
        for _ in 0..ambiguous_count {
            test_data.push(self.generate_ambiguous_document());
        }
        
        // 15% edge cases (special characters, nested structures, etc.)
        let edge_case_count = (count as f32 * 0.15) as usize;
        for _ in 0..edge_case_count {
            test_data.push(self.edge_case_factory.generate_random_edge_case());
        }
        
        // 5% malformed/low-quality documents
        let low_quality_count = count - high_quality_count - ambiguous_count - edge_case_count;
        for _ in 0..low_quality_count {
            test_data.push(self.generate_low_quality_document());
        }
        
        test_data
    }
    
    fn generate_ambiguous_document(&self) -> TestDocument {
        let ambiguity_type = self.ambiguity_generator.random_type();
        match ambiguity_type {
            AmbiguityType::EntityAmbiguity => TestDocument {
                content: "Apple released a new product in California.",
                expected_ambiguities: vec!["Apple: Company or Fruit?"],
                expected_quality_score: 0.7..0.85,
                expected_resolution: "Company (context: product release)",
            },
            AmbiguityType::RelationAmbiguity => TestDocument {
                content: "The bank by the river bank handles bank accounts.",
                expected_ambiguities: vec!["bank: financial, geological, or action?"],
                expected_quality_score: 0.6..0.8,
                expected_resolution: "Multiple entities with different types",
            },
            // More ambiguity types...
        }
    }
}

// Edge case factory for comprehensive testing
pub struct EdgeCaseFactory {
    edge_cases: Vec<EdgeCaseTemplate>,
}

impl EdgeCaseFactory {
    pub fn new() -> Self {
        Self {
            edge_cases: vec![
                EdgeCaseTemplate::UnicodeEntities("🚀 rockets to the 🌙"),
                EdgeCaseTemplate::NestedStructures("((deeply (nested (concepts))))"),
                EdgeCaseTemplate::MixedLanguages("The 世界 is connected"),
                EdgeCaseTemplate::ScientificNotation("6.022e23 molecules"),
                EdgeCaseTemplate::Contradictions("All birds fly. Penguins don't fly."),
                EdgeCaseTemplate::TemporalConflicts("It's 2020. It's 2024."),
                EdgeCaseTemplate::CircularReferences("A is B. B is C. C is A."),
                EdgeCaseTemplate::ExtremeLengths(generate_very_long_text()),
                EdgeCaseTemplate::EmptyContent(""),
                EdgeCaseTemplate::PureNoise("@#$%^&*()_+{}|:<>?"),
            ],
        }
    }
}
```

## Success Criteria

### Integration Test Coverage
- ✅ All critical paths tested with mocks first
- ✅ Progressive unmocking demonstrated
- ✅ Full end-to-end tests without mocks
- ✅ Failure scenarios and recovery tested
- ✅ Performance relative to baselines validated
- ✅ CPU-only operation confirmed
- ✅ Phase 0A quality gates fully integrated
- ✅ Sophisticated test data generation with edge cases

### Performance Goals (Flexible)
- **Better than baseline**: Outperform existing knowledge graphs
- **CPU optimized**: No GPU dependency
- **Scalable**: Handle increasing load gracefully
- **Responsive**: User-perceptible improvements

### Reliability Requirements
- **Graceful degradation**: System remains operational during failures
- **Transaction support**: Rollback capability across phases
- **Health monitoring**: Continuous health checks during integration

## Conclusion

This comprehensive integration testing strategy ensures that all phases of CortexKG work together seamlessly while following London School TDD principles. The progressive approach from fully mocked to real implementations provides confidence at each stage while maintaining fast test execution in CI/CD pipelines.