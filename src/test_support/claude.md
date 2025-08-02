# Directory Overview: test_support

## 1. High-Level Summary

The `test_support` directory contains comprehensive testing utilities for the LLMKG (Large Language Model Knowledge Graph) project. This module provides builders, assertions, test data, fixtures, and predefined scenarios specifically designed to test cognitive patterns, attention management, knowledge graph operations, and the overall brain-inspired AI system. The directory serves as the foundation for all testing across the cognitive modules, ensuring consistent, reliable, and comprehensive test coverage.

## 2. Tech Stack

* **Language:** Rust
* **Frameworks:** Standard Rust testing framework with custom extensions
* **Libraries:** 
  - `std::collections::HashMap`
  - `std::sync::Arc` 
  - `std::time::Duration`
  - `slotmap::SlotMap`
  - `rand` (for test data generation)
* **Testing Patterns:** Builder pattern, Trait-based assertions, Scenario-driven testing
* **Async Support:** Async/await for cognitive component testing

## 3. Directory Structure

* **`mod.rs`** - Module declarations and convenient re-exports
* **`assertions.rs`** - Custom assertion traits and implementations for cognitive testing
* **`builders.rs`** - Builder patterns for creating test instances of cognitive components
* **`data.rs`** - Test data fixtures, providers, and generators
* **`fixtures.rs`** - Common test fixtures for graphs and entities
* **`scenarios.rs`** - Predefined test scenarios for comprehensive cognitive pattern testing

## 4. File Breakdown

### `mod.rs`

* **Purpose:** Main module file that declares all submodules and provides convenient re-exports for commonly used testing utilities.
* **Key Exports:**
  - **Builders:** `AttentionManagerBuilder`, `CognitivePatternBuilder`, `CognitiveOrchestratorBuilder`, `WorkingMemoryBuilder`, `QueryContextBuilder`, `PatternParametersBuilder`
  - **Assertions:** `CognitiveAssertions`, `PatternAssertions`
  - **Data:** `TestQueries`, `PerformanceTestData`, `create_standard_test_entities`
  - **Scenarios:** `TestScenario`, `AttentionScenario`, various scenario functions

### `assertions.rs`

* **Purpose:** Comprehensive custom assertion framework for validating cognitive module behavior and results.

#### Key Traits:

* **`CognitiveAssertions`**
  - `assert_attention_focused_on(target: &EntityKey, min_weight: f32)`: Validates attention focus on specific entities
  - `assert_total_weight_approximately(expected: f32, tolerance: f32)`: Checks total attention weight distribution
  - `assert_weights_distributed_evenly(tolerance: f32)`: Validates even distribution of attention weights

* **`PatternAssertions`**
  - `assert_confidence_in_range(min: f32, max: f32)`: Validates confidence scores within expected ranges
  - `assert_contains_content(expected: &str)`: Checks if results contain specific content
  - `assert_pattern_detected(pattern_type: &str, min_confidence: f32)`: Validates detected cognitive patterns
  - `assert_reasoning_depth(expected_depth: usize)`: Ensures adequate reasoning depth
  - `assert_execution_time_within(max_ms: u64)`: Performance validation

* **`OrchestratorAssertions`**
  - `assert_workflow_completed()`: Validates successful workflow completion
  - `assert_patterns_used(expected_patterns: &[CognitivePatternType])`: Checks expected pattern usage
  - `assert_reasonable_confidence(min_confidence: f32)`: Validates result confidence

* **`AttentionAssertions`**
  - `assert_attention_focuses_on(entities: &[EntityKey])`: Validates attention targeting
  - `assert_attention_distribution(expected_distribution: &[(EntityKey, f32)], tolerance: f32)`: Checks attention distribution patterns
  - `assert_attention_state(expected_state: &str)`: Validates attention state

#### Implementations:
- `PatternAssertions` for `crate::cognitive::PatternResult`
- `PatternAssertions` for `crate::cognitive::ConvergentResult`
- `PatternAssertions` for `crate::cognitive::DivergentResult`

#### Utility Macros:
- `assert_cognitive_pattern!`: Validates cognitive pattern detection
- `assert_attention_weights_sum_to_one!`: Ensures attention weights sum to 1.0
- `assert_performance_within_bounds!`: Performance boundary validation
- `assert_memory_usage_reasonable!`: Memory usage validation
- `assert_ok!`: Async result validation
- `assert_err_type!`: Specific error type validation

### `builders.rs`

* **Purpose:** Builder pattern implementations for creating properly configured test instances of cognitive components.

#### Key Builders:

* **`AttentionManagerBuilder`**
  - `new()`: Creates new builder with defaults
  - `with_graph(graph: Arc<BrainEnhancedKnowledgeGraph>)`: Sets custom graph
  - `with_embedding_dim(dim: usize)`: Sets embedding dimension (default: 96)
  - `build()`: Builds AttentionManager asynchronously
  - `build_with_deps()`: Builds with all dependencies exposed

* **`CognitivePatternBuilder`**
  - `with_graph(graph: Arc<BrainEnhancedKnowledgeGraph>)`: Sets custom graph
  - `build_convergent()`: Creates ConvergentThinking instance
  - `build_divergent()`: Creates DivergentThinking instance
  - `build_lateral()`: Creates LateralThinking instance
  - `build_systems()`: Creates SystemsThinking instance
  - `build_critical()`: Creates CriticalThinking instance
  - `build_abstract()`: Creates AbstractThinking instance
  - `build_adaptive()`: Creates AdaptiveThinking instance

* **`CognitiveOrchestratorBuilder`**
  - `with_graph(graph: Arc<BrainEnhancedKnowledgeGraph>)`: Sets custom graph
  - `with_config(config: CognitiveOrchestratorConfig)`: Sets configuration
  - `build()`: Builds CognitiveOrchestrator asynchronously

* **`WorkingMemoryBuilder`**
  - `with_graph(graph: Arc<BrainEnhancedKnowledgeGraph>)`: Sets custom graph
  - `with_activation_engine(engine: Arc<ActivationPropagationEngine>)`: Sets activation engine
  - `build()`: Builds WorkingMemorySystem asynchronously

* **`QueryContextBuilder`**
  - `with_domain(domain: String)`: Sets query domain
  - `with_confidence_threshold(threshold: f32)`: Sets confidence threshold (default: 0.7)
  - `with_max_depth(depth: usize)`: Sets max reasoning depth
  - `with_required_evidence(evidence: usize)`: Sets evidence requirements
  - `with_reasoning_trace(trace: bool)`: Enables reasoning trace
  - `build()`: Creates QueryContext

* **`PatternParametersBuilder`**
  - `with_max_depth(depth: usize)`: Sets maximum reasoning depth
  - `with_activation_threshold(threshold: f32)`: Sets activation threshold
  - `with_exploration_breadth(breadth: usize)`: Sets exploration breadth
  - `with_creativity_threshold(threshold: f32)`: Sets creativity threshold
  - `with_validation_level(level: ValidationLevel)`: Sets validation level
  - `with_pattern_type(pattern_type: PatternType)`: Sets pattern type
  - `with_reasoning_strategy(strategy: ReasoningStrategy)`: Sets reasoning strategy
  - `build()`: Creates PatternParameters

### `data.rs`

* **Purpose:** Comprehensive test data fixtures and generators for consistent testing across all cognitive patterns.

#### Key Data Structures:

* **`TestQueries`**
  - `factual`: Vec of factual queries for convergent thinking tests
  - `creative`: Vec of creative queries for divergent thinking tests
  - `analytical`: Vec of analytical queries for critical thinking tests
  - `relational`: Vec of relational queries for lateral thinking tests

* **`PerformanceTestData`**
  - `small_dataset`: 10 entities for lightweight tests
  - `medium_dataset`: 100 entities for standard tests
  - `large_dataset`: 1000 entities for performance tests
  - `stress_dataset`: 10000 entities for stress tests

* **`AttentionTestData`**
  - `entity_attention_scores`: HashMap of predefined attention scores
  - `query_weights`: HashMap of query type weights
  - `decay_factors`: Vec of decay factors for temporal testing

* **`PatternTestData`**
  - `pattern_sequences`: Vec of cognitive pattern sequences
  - `expected_transitions`: HashMap of pattern transition probabilities

* **`EdgeCaseTestData`**
  - `empty_queries`: Empty and whitespace-only queries
  - `malformed_queries`: Invalid query formats
  - `extreme_values`: Edge case numerical values (MIN, MAX, INFINITY, NAN)
  - `unicode_test_strings`: International and emoji test strings

* **`TemporalTestData`**
  - `timestamps`: Unix timestamps for temporal testing
  - `time_intervals`: Duration intervals for timing tests
  - `decay_curves`: Predefined decay curves for memory testing

* **`MemoryTestData`**
  - `memory_traces`: Tuples of (EntityKey, strength, timestamp)
  - `reinforcement_patterns`: Learning reinforcement patterns
  - `forgetting_curves`: Named forgetting curve patterns

* **`TestDataProvider`** (Master provider)
  - `get_queries_for_pattern(pattern: CognitivePatternType)`: Returns appropriate queries for pattern type
  - `get_entities_with_scores()`: Returns entities with pre-assigned attention scores
  - `get_performance_dataset(size: &str)`: Returns dataset of specified size

#### Generators Module:
- `generate_entities(count: usize, prefix: &str)`: Creates entity keys with consistent naming
- `generate_attention_scores(count: usize, min: f32, max: f32, seed: u64)`: Random attention scores
- `generate_temporal_sequence(start_time: u64, count: usize, interval_ms: u64)`: Temporal sequences
- `generate_decay_curve(initial_value: f32, decay_factor: f32, steps: usize)`: Decay curves

### `fixtures.rs`

* **Purpose:** Common test fixtures for creating knowledge graphs and entities consistently across tests.

#### Key Functions:

* **Graph Creation:**
  - `create_test_graph()`: Creates test graph with default configuration
  - `create_test_graph_with_dim(embedding_dim: usize)`: Creates graph with custom embedding dimension
  - `create_test_knowledge_graph()`: Alias for consistency with README examples
  - `create_minimal_test_graph()`: Lightweight test graph
  - `create_populated_test_graph()`: Graph populated with standard test entities (async)
  - `create_cognitive_test_graph()`: Graph configured for cognitive testing (async)
  - `create_configured_test_graph(config: TestGraphConfig)`: Graph with custom configuration (async)

* **Entity Creation:**
  - `create_standard_test_entities()`: 7 AI/ML domain entities with realistic embeddings
  - `create_cognitive_test_entities()`: 5 cognitive process entities
  - `create_diverse_test_entities()`: Combines standard + cognitive + domain-specific entities
  - `create_test_entity_keys(count: usize)`: Creates entity keys without full graph

* **Configuration:**
  - `TestGraphConfig` struct with fields:
    - `embedding_dim: usize` (default: 96)
    - `entity_count: usize` (default: 100)
    - `enable_caching: bool` (default: true)
    - `enable_indexing: bool` (default: true)

### `scenarios.rs`

* **Purpose:** Comprehensive predefined test scenarios for all cognitive patterns, performance testing, error handling, and complex reasoning.

#### Key Structures:

* **`TestScenario`**
  - `name`: Scenario identifier
  - `description`: Human-readable description
  - `query`: Test query string
  - `expected_pattern`: Expected CognitivePatternType
  - `expected_intent`: Expected QueryIntent
  - `expected_confidence_min/max`: Confidence range
  - `complexity_level`: Basic, Moderate, Complex, Extreme
  - `scenario_type`: Functional, Performance, EdgeCase, Stress, Integration, Regression
  - `timeout`: Optional timeout duration
  - `graph_setup`: Function to setup graph for scenario

* **`PerformanceScenario`**
  - `iterations`: Number of test iterations
  - `max_duration`: Maximum allowed duration
  - `memory_limit_mb`: Optional memory limit
  - `entity_count`: Number of entities to test with
  - `concurrent_queries`: Number of concurrent queries
  - `setup`: Graph setup function
  - `workload`: Workload generation function

* **`ErrorScenario`**
  - `invalid_input`: Input designed to trigger errors
  - `expected_error_type`: Expected error type
  - `recovery_expected`: Whether system should recover
  - `setup`: Graph setup function

* **`ComplexReasoningScenario`**
  - `multi_step_query`: Complex multi-step query
  - `required_patterns`: Vec of required cognitive patterns
  - `intermediate_steps`: Expected reasoning steps
  - `expected_synthesis`: Expected final synthesis
  - `min_reasoning_depth`: Minimum reasoning depth required

#### Enums:

* **`ComplexityLevel`**: Basic, Moderate, Complex, Extreme
* **`ScenarioType`**: Functional, Performance, EdgeCase, Stress, Integration, Regression
* **`ErrorType`**: InvalidQuery, MissingEntity, GraphCorruption, MemoryExhaustion, TimeoutError, ConcurrencyError, PatternMismatch

#### Scenario Collections:

* **`get_test_scenarios()`**: 27+ comprehensive test scenarios covering:
  - **Convergent thinking**: Factual queries, definitions
  - **Divergent thinking**: Creative exploration, brainstorming
  - **Lateral thinking**: Relationship queries, cross-domain connections
  - **Systems thinking**: Ecosystem analysis, hierarchical structures
  - **Critical thinking**: Analysis, argument evaluation
  - **Abstract thinking**: Pattern recognition
  - **Adaptive thinking**: Context-adaptive problem solving
  - **Multi-hop reasoning**: Complex chain reasoning
  - **Counterfactual reasoning**: What-if scenarios
  - **Temporal reasoning**: Chronological sequences
  - **Compositional understanding**: Component-whole relationships
  - **Edge cases**: Ambiguous queries, contradictions
  - **Performance testing**: Large-scale queries

* **`get_performance_scenarios()`**: 4 performance benchmark scenarios:
  - High volume queries (1000 iterations, 50 concurrent)
  - Complex reasoning benchmark (100 iterations, 10 concurrent)
  - Memory stress test (50 iterations, 100 concurrent, 50K entities)
  - Latency benchmark (500 iterations, single threaded)

* **`get_error_scenarios()`**: 6 error handling scenarios:
  - Invalid query syntax
  - Missing entity references
  - Memory exhaustion conditions
  - Timeout conditions
  - Concurrent access conflicts
  - Pattern mismatches

* **`get_complex_reasoning_scenarios()`**: 4 multi-pattern scenarios:
  - Multi-pattern synthesis (climate change → food security → solutions)
  - Cross-domain innovation (biology → software development)
  - Ethical dilemma resolution (AI impact analysis)
  - Technology convergence analysis (AI + quantum + biotech)

#### Validation and Execution:

* **Validation Functions:**
  - `validate_scenario_result()`: Validates scenario outcomes against expectations
  - `validate_performance_scenario()`: Validates performance metrics
  - `execute_scenario_with_validation()`: Full scenario execution with validation
  - `execute_scenarios_parallel()`: Parallel scenario execution

* **Utility Functions:**
  - `filter_scenarios()`: Filter scenarios by type, complexity, timeout
  - `generate_test_report()`: Comprehensive test reporting
  - `create_scenario_entities()`: Entity creation for scenarios
  - `create_scenario_entities_in_graph()`: Async entity creation in graphs

* **Graph Setup Functions** (40+ setup functions for different domains):
  - `basic_animal_setup`, `science_concepts_setup`, `object_creativity_setup`
  - `urban_planning_setup`, `environmental_relations_setup`, `cross_domain_setup`
  - `ecosystem_setup`, `organizational_setup`, `energy_analysis_setup`
  - `mathematical_patterns_setup`, `crisis_management_setup`
  - And many more domain-specific setups

## 5. Dependencies

### Internal Dependencies:
* **Core modules:**
  - `crate::core::types::{EntityKey, EntityData}`
  - `crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph`
  - `crate::core::activation_engine::ActivationPropagationEngine`
  - `crate::core::activation_config::ActivationConfig`
  
* **Cognitive modules:**
  - `crate::cognitive::attention_manager::AttentionManager`
  - `crate::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig}`
  - `crate::cognitive::working_memory::WorkingMemorySystem`
  - `crate::cognitive::types::CognitivePatternType`
  - `crate::cognitive::basic_query::QueryIntent`
  - Various cognitive pattern implementations (convergent, divergent, lateral, etc.)

* **Error handling:**
  - `crate::error::Result`

### External Dependencies:
* **Standard library:**
  - `std::collections::HashMap`
  - `std::sync::Arc`
  - `std::time::Duration`
  
* **Third-party crates:**
  - `slotmap::SlotMap` - for entity key management
  - `rand` - for test data generation (with seeded RNG)

## 6. Key Design Patterns

### Builder Pattern
All major components use the builder pattern for flexible, readable test setup:
```rust
let attention_manager = AttentionManagerBuilder::new()
    .with_embedding_dim(128)
    .with_graph(custom_graph)
    .build()
    .await?;
```

### Trait-Based Assertions
Custom assertion traits provide domain-specific validation:
```rust
result.assert_confidence_in_range(0.7, 0.9);
attention_weights.assert_attention_focused_on(&entity_key, 0.8);
```

### Scenario-Driven Testing
Comprehensive predefined scenarios enable consistent testing:
```rust
let scenarios = get_test_scenarios();
let filtered = filter_scenarios(&scenarios, Some(ScenarioType::Functional), Some(ComplexityLevel::Basic), None);
```

### Data Providers
Centralized test data providers ensure consistency:
```rust
let provider = TestDataProvider::new();
let queries = provider.get_queries_for_pattern(CognitivePatternType::Divergent);
```

## 7. Testing Architecture

### Layered Testing Approach:
1. **Unit Testing**: Individual component builders and assertions
2. **Integration Testing**: Component interaction scenarios
3. **Performance Testing**: Scalability and timing scenarios
4. **Edge Case Testing**: Error conditions and boundary cases
5. **Regression Testing**: Prevent functionality breakage
6. **Stress Testing**: High-load and resource exhaustion scenarios

### Test Data Management:
- **Fixtures**: Standard entity and graph setups
- **Generators**: Procedural test data with seeded randomness
- **Providers**: Centralized data access with pattern-specific queries
- **Scenarios**: Domain-specific knowledge graph setups

### Validation Framework:
- **Result Validation**: Confidence, content, pattern detection
- **Performance Validation**: Timing, memory usage, success rates
- **State Validation**: Attention distribution, reasoning depth
- **Error Validation**: Expected error types and recovery

## 8. Usage Guidelines

### For Test Authors:
1. Use builders for component setup instead of direct constructors
2. Apply appropriate assertion traits for domain-specific validation
3. Select or create scenarios that match your testing needs
4. Use data providers for consistent test data across tests
5. Validate both positive and negative cases

### For Scenario Creation:
1. Define clear expected outcomes (pattern, intent, confidence range)
2. Set appropriate complexity levels and timeouts
3. Provide domain-specific graph setup functions
4. Include both functional and edge case scenarios

### For Performance Testing:
1. Use predefined performance scenarios as baselines
2. Set realistic memory limits and concurrency levels
3. Validate both throughput and latency metrics
4. Include stress testing for resource exhaustion

## 9. Extension Points

### Adding New Cognitive Patterns:
1. Add builder methods to `CognitivePatternBuilder`
2. Implement `PatternAssertions` for the new pattern result type
3. Add appropriate test queries to `TestQueries`
4. Create scenarios in `get_test_scenarios()`

### Adding New Test Data Types:
1. Create data structures in `data.rs`
2. Add to `TestDataProvider` for centralized access
3. Include generators in the `generators` module
4. Create fixtures in `fixtures.rs` if needed

### Adding New Scenario Types:
1. Define scenario struct (similar to `TestScenario`)
2. Create collection function (similar to `get_test_scenarios()`)
3. Add validation function for the scenario type
4. Include in comprehensive test reporting

This comprehensive test support system enables thorough, consistent, and maintainable testing of the entire LLMKG cognitive system, from individual components to complex multi-pattern reasoning scenarios.