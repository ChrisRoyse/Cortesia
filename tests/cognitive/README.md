# Cognitive Module Test Suite

## Overview

This directory contains the complete test suite for the LLMKG cognitive module, following Rust best practices for test organization and visibility. The test suite is designed to comprehensively validate all cognitive functionality through unit tests, integration tests, property-based tests, and performance benchmarks.

## Test Organization

### Integration Tests (in tests/cognitive/ directory)
These test the public API and component interactions:

- **`attention_integration_tests.rs`** - Tests AttentionManager public API workflows and integration scenarios
- **`patterns_integration_tests.rs`** - Tests cognitive pattern coordination and cross-pattern interactions  
- **`orchestrator_integration_tests.rs`** - Tests CognitiveOrchestrator functionality and workflow orchestration

**Purpose**: Validate public API behavior, component interactions, and end-to-end workflows without access to private implementation details.

### Unit Tests (in tests/cognitive/ directory)
These provide comprehensive testing of individual cognitive components:

- **`test_attention_manager.rs`** - Comprehensive AttentionManager testing
- **`test_convergent.rs`** - Convergent thinking pattern tests
- **`test_divergent.rs`** - Divergent thinking pattern tests
- **`test_lateral.rs`** - Lateral thinking pattern tests
- **`test_adaptive.rs`** - Adaptive thinking pattern tests
- **`test_orchestrator.rs`** - CognitiveOrchestrator unit tests
- **`test_neural_bridge_finder.rs`** - Neural bridge finding tests
- **`test_neural_query.rs`** - Neural query processing tests
- **`test_utils.rs`** - Utility function tests

**Purpose**: Test individual component functionality, edge cases, and error conditions.

### Specialized Tests

- **`property_tests.rs`** - Property-based tests using proptest crate to validate mathematical invariants and behavioral properties
- **`performance_tests.rs`** - Performance benchmarks and timing validation for critical cognitive pathways

### Additional Unit Tests (in source files)
Located in `#[cfg(test)]` modules within each source file for testing private functions:
- `src/cognitive/attention_manager.rs` - Tests private attention methods
- `src/cognitive/convergent.rs` - Tests private convergent thinking functions
- `src/cognitive/divergent.rs` - Tests private divergent thinking functions  
- `src/cognitive/lateral.rs` - Tests private lateral thinking functions
- `src/cognitive/orchestrator.rs` - Tests private orchestrator functions
- `src/cognitive/neural_query.rs` - Tests private neural query functions

**Purpose**: Test private functions and implementation details that cannot be accessed from external test files.

### Test Support Library
Located in `src/test_support/`:

- **`fixtures.rs`** - Test entity and knowledge graph creation utilities
- **`builders.rs`** - Builder patterns for creating test instances (AttentionManagerBuilder, CognitivePatternBuilder)
- **`assertions.rs`** - Custom cognitive assertions (CognitiveAssertions, PatternAssertions)
- **`scenarios.rs`** - Predefined test scenarios (TestScenario, AttentionScenario)
- **`data.rs`** - Test data management and standard test entities

## Running Tests

### All Tests
```bash
# Run all tests
cargo test

# Run all tests with output
cargo test -- --nocapture
```

### By Test Type
```bash
# Run only unit tests (in source files)
cargo test --lib

# Run only integration tests (in tests/ directory)
cargo test --test '*'

# Run cognitive module tests specifically
cargo test cognitive
```

### Specific Test Files
```bash
# Run specific integration test
cargo test --test attention_integration_tests
cargo test --test patterns_integration_tests
cargo test --test orchestrator_integration_tests

# Run specific unit test files
cargo test test_attention_manager
cargo test test_convergent
cargo test test_divergent
cargo test test_lateral
cargo test test_orchestrator

# Run specialized tests
cargo test property_tests
cargo test performance_tests
```

### Filtered Testing
```bash
# Run tests matching a pattern
cargo test attention
cargo test convergent
cargo test orchestrator

# Run tests for specific functionality
cargo test neural_bridge
cargo test cognitive_pattern
cargo test thinking_patterns
```

## Test Design Principles

### 1. Comprehensive Coverage
- **Unit Tests**: Cover individual component functionality and edge cases
- **Integration Tests**: Validate component interactions and public API workflows
- **Property Tests**: Verify mathematical invariants and behavioral properties
- **Performance Tests**: Ensure cognitive operations meet timing requirements

### 2. Realistic Scenarios
- Tests use real-world cognitive workflows and use cases
- Test data reflects actual knowledge graph structures and queries
- Scenarios cover typical user interactions and complex reasoning chains

### 3. Clear Separation of Concerns
- **Public API Testing**: Integration tests focus on external interfaces
- **Implementation Testing**: Unit tests validate internal logic
- **Cross-cutting Concerns**: Property and performance tests validate system-wide behavior

### 4. Maintainable Test Structure
- Shared test utilities prevent code duplication
- Builder patterns simplify test setup
- Custom assertions provide clear failure messages
- Modular organization supports easy test discovery

### 5. Performance Validation
- Critical cognitive pathways have performance benchmarks
- Memory usage and allocation patterns are validated
- Concurrent operation safety is tested

## Test Support Utilities

### Fixtures (`src/test_support/fixtures.rs`)
```rust
use llmkg::test_support::fixtures::*;

// Create standard test entities
let entities = create_standard_test_entities();

// Create test knowledge graph
let graph = create_test_knowledge_graph();
```

### Builders (`src/test_support/builders.rs`)
```rust
use llmkg::test_support::builders::*;

// Build test AttentionManager
let attention_manager = AttentionManagerBuilder::new()
    .with_focus_threshold(0.8)
    .with_decay_rate(0.1)
    .build();

// Build test cognitive pattern
let pattern = CognitivePatternBuilder::new()
    .with_pattern_type(PatternType::Convergent)
    .with_confidence(0.9)
    .build();
```

### Assertions (`src/test_support/assertions.rs`)
```rust
use llmkg::test_support::assertions::*;

// Cognitive-specific assertions
result.assert_attention_focused_on(&entity);
pattern.assert_pattern_detected();
orchestrator.assert_workflow_completed();
```

### Scenarios (`src/test_support/scenarios.rs`)
```rust
use llmkg::test_support::scenarios::*;

// Use predefined test scenarios
let scenario = TestScenario::ComplexReasoning;
let entities = create_scenario_entities(&scenario);

// Attention-specific scenarios
let attention_scenario = AttentionScenario::MultiEntityFocus;
```

## Adding New Tests

### For New Cognitive Components
1. Create a new test file: `test_[component_name].rs`
2. Add comprehensive unit tests for all public methods
3. Include edge cases and error conditions
4. Add integration tests if the component interacts with others

### For Public API Testing
Add tests to the appropriate integration test file:
```rust
#[tokio::test]
async fn test_public_api_workflow() {
    use llmkg::test_support::*;
    
    let graph = fixtures::create_test_knowledge_graph();
    let entities = scenarios::create_scenario_entities(&TestScenario::BasicReasoning);
    
    // Test through public API only
    let result = cognitive_module.process(&graph, &entities).await;
    
    result.assert_workflow_completed();
}
```

### For Property-Based Testing
Add to `property_tests.rs`:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn cognitive_property_holds(
        input in any::<CognitiveInput>()
    ) {
        let result = cognitive_function(input);
        // Assert mathematical property
        prop_assert!(result.satisfies_invariant());
    }
}
```

### For Performance Testing
Add to `performance_tests.rs`:
```rust
#[tokio::test]
async fn test_cognitive_operation_performance() {
    let start = Instant::now();
    
    // Perform cognitive operation
    let result = perform_cognitive_operation().await;
    
    let duration = start.elapsed();
    assert!(duration < Duration::from_millis(100), 
           "Cognitive operation took too long: {:?}", duration);
}
```

## Test Coverage Goals

- **Unit Tests**: 100% coverage of public methods, 80%+ coverage of private methods
- **Integration Tests**: All major workflows and API endpoints covered
- **Edge Cases**: Error conditions, boundary values, and invalid inputs tested
- **Performance**: All critical paths have performance validation
- **Properties**: Mathematical invariants and behavioral properties verified

## Debugging Tests

### Running with Debug Output
```bash
# Run with full output
cargo test -- --nocapture

# Run specific test with output
cargo test test_attention_manager -- --nocapture

# Run with environment logging
RUST_LOG=debug cargo test
```

### Test Debugging Tips
- Use `dbg!()` macro for runtime debugging
- Enable logging in test support utilities
- Use `--test-threads=1` for sequential test execution
- Check test failure output for assertion details

## Future Test Enhancements

1. **Mutation Testing**: Validate test quality through mutation testing
2. **Fuzzing**: Add property-based fuzzing for input validation
3. **Stress Testing**: Extended performance tests under load
4. **Visual Testing**: Cognitive workflow visualization tests
5. **Documentation Testing**: Ensure all examples in docs are tested

This test suite provides comprehensive validation of the LLMKG cognitive module while maintaining clear organization and supporting future development needs.