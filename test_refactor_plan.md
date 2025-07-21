# Comprehensive Test Suite Refactoring Plan

## Executive Summary

This document provides a complete, function-level plan to refactor the LLMKG cognitive module test suite to properly work within Rust's visibility constraints while maintaining comprehensive test coverage.

## Core Principles

1. **Unit tests** that need private access stay in source files (`#[cfg(test)]` modules)
2. **Integration tests** in `/tests/` directory test only public APIs
3. **Test helpers** are shared via a `testlib` module when needed by both
4. **Clear naming** distinguishes unit tests from integration tests
5. **No pretending** - tests clearly state what they're testing

## Detailed Refactoring Plan

### Phase 1: Establish Test Infrastructure

#### 1.1 Create a Test Support Library
**File**: `src/test_support/mod.rs` (new file)
```rust
//! Test support utilities available to both unit and integration tests
#![cfg(test)]

pub mod fixtures;
pub mod assertions;
pub mod builders;
```

**Why**: This module is part of the main crate, so it can access `pub(crate)` items and be used by both unit tests (in src/) and integration tests (in tests/).

#### 1.2 Create Test Fixtures Module
**File**: `src/test_support/fixtures.rs`
```rust
use crate::cognitive::*;
use crate::core::*;

/// Creates test entity keys for attention tests
pub fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
    (0..count).map(|i| EntityKey::new(format!("entity_{}", i))).collect()
}

/// Creates a minimal test graph
pub fn create_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    let config = BrainEnhancedConfig::test_default();
    Arc::new(BrainEnhancedKnowledgeGraph::new(config))
}

/// Creates test memory items
pub fn create_test_memory_items(count: usize) -> Vec<MemoryItem> {
    // Implementation
}
```

#### 1.3 Create Test Builders Module
**File**: `src/test_support/builders.rs`
```rust
/// Builder for creating configured AttentionManager instances for testing
pub struct AttentionManagerBuilder {
    graph: Option<Arc<BrainEnhancedKnowledgeGraph>>,
    config: Option<AttentionConfig>,
}

impl AttentionManagerBuilder {
    pub fn new() -> Self { /* ... */ }
    pub fn with_graph(mut self, graph: Arc<BrainEnhancedKnowledgeGraph>) -> Self { /* ... */ }
    pub fn with_config(mut self, config: AttentionConfig) -> Self { /* ... */ }
    
    /// Builds just the AttentionManager
    pub fn build(self) -> AttentionManager { /* ... */ }
    
    /// Builds AttentionManager with all dependencies
    pub fn build_with_deps(self) -> (
        AttentionManager,
        Arc<CognitiveOrchestrator>,
        Arc<ActivationPropagationEngine>,
        Arc<WorkingMemorySystem>
    ) { /* ... */ }
}
```

#### 1.4 Create Test Assertions Module
**File**: `src/test_support/assertions.rs`
```rust
/// Custom assertions for cognitive tests
pub trait CognitiveAssertions {
    fn assert_attention_focused_on(&self, target: &EntityKey, min_weight: f32);
    fn assert_total_weight_approximately(&self, expected: f32, tolerance: f32);
}

impl CognitiveAssertions for HashMap<EntityKey, f32> {
    // Implementations
}
```

### Phase 2: Refactor Unit Tests (Keep in Source Files)

#### 2.1 AttentionManager Unit Tests
**File**: `src/cognitive/attention_manager.rs`

Add at the end of the file:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fixtures::*, builders::*, assertions::*};
    
    // Tests that NEED private access
    
    #[test]
    fn test_calculate_attention_weights_selective() {
        // This test needs access to private calculate_attention_weights method
        let attention_manager = AttentionManagerBuilder::new().build();
        let targets = create_test_entity_keys(3);
        
        // Direct test of private method
        let weights = attention_manager.calculate_attention_weights(
            &targets,
            1.0,
            &AttentionType::Selective
        );
        
        assert!(weights[&targets[0]] > 0.9);
        assert!(weights[&targets[1]] < 0.1);
        assert!(weights[&targets[2]] < 0.1);
        assert_total_weight_approximately(&weights, 1.0, 0.01);
    }
    
    #[test]
    fn test_calculate_attention_weights_divided() {
        // Tests private method with divided attention
        let attention_manager = AttentionManagerBuilder::new().build();
        let targets = create_test_entity_keys(4);
        
        let weights = attention_manager.calculate_attention_weights(
            &targets,
            1.0,
            &AttentionType::Divided
        );
        
        let expected_weight = 1.0 / 4.0;
        for (_, weight) in &weights {
            assert!((weight - expected_weight).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_attention_state_private_methods() {
        // Test private state manipulation methods
        let mut state = AttentionState::new();
        
        // Test private method update_cognitive_load
        state.update_cognitive_load(0.5);
        assert_eq!(state.cognitive_load, 0.5);
        assert_eq!(state.attention_capacity, 0.5);
    }
}
```

#### 2.2 Convergent Thinking Unit Tests
**File**: `src/cognitive/convergent.rs`

Add at the end:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fixtures::*;
    
    #[test]
    fn test_calculate_concept_relevance_private() {
        // Direct test of private function
        assert_eq!(calculate_concept_relevance("", "dog", "dog"), 1.0);
        assert!(calculate_concept_relevance("mammal", "dog", "canine") > 0.5);
    }
    
    #[test]
    fn test_levenshtein_distance_private() {
        // Direct test of private function
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }
    
    #[test]
    fn test_is_stop_word_private() {
        // Direct test of private function
        assert!(is_stop_word("the"));
        assert!(!is_stop_word("quantum"));
    }
}
```

#### 2.3 Repeat for Other Modules
For each cognitive module with private functions to test:
- `divergent.rs` - test `calculate_concept_similarity`, `infer_exploration_type`
- `lateral.rs` - test `calculate_concept_relevance`, `parse_lateral_query`
- `neural_query.rs` - test `identify_query_intent`, `extract_concepts`
- `orchestrator.rs` - test `calculate_pattern_weight`

### Phase 3: Create Proper Integration Tests

#### 3.1 Attention Manager Integration Tests
**File**: `/tests/cognitive/attention_integration_tests.rs`
```rust
//! Integration tests for AttentionManager public API

use llmkg::cognitive::{AttentionManager, AttentionType, ExecutiveCommand};
use llmkg::test_support::{fixtures::*, builders::*};

#[tokio::test]
async fn test_attention_focus_and_shift() {
    // Test PUBLIC API behavior
    let (manager, _, _, _) = AttentionManagerBuilder::new()
        .build_with_deps();
    
    let targets = create_test_entity_keys(3);
    
    // Test focusing attention
    manager.focus_attention(
        targets.clone(),
        1.0,
        AttentionType::Selective
    ).await.unwrap();
    
    // Verify through public API
    let snapshot = manager.get_attention_snapshot().await;
    assert_eq!(snapshot.current_focus.targets.len(), 3);
    assert_eq!(snapshot.attention_type, AttentionType::Selective);
}

#[tokio::test]
async fn test_executive_control_integration() {
    let (manager, _, _, _) = AttentionManagerBuilder::new()
        .build_with_deps();
    
    // Test executive commands through public API
    let result = manager.executive_control(
        ExecutiveCommand::SwitchFocus {
            new_targets: create_test_entity_keys(2),
            attention_type: AttentionType::Divided,
        }
    ).await.unwrap();
    
    assert!(result.success);
}

#[tokio::test]
async fn test_memory_integration_workflow() {
    let (manager, _, _, working_memory) = AttentionManagerBuilder::new()
        .build_with_deps();
    
    // Test the full integration between attention and memory
    let targets = create_test_entity_keys(2);
    
    // Focus attention
    manager.focus_attention(
        targets.clone(),
        1.0,
        AttentionType::Selective
    ).await.unwrap();
    
    // Verify memory was updated through public API
    let memory_contents = working_memory.get_all_items().await;
    assert!(memory_contents.len() > 0);
}
```

#### 3.2 Cognitive Pattern Integration Tests
**File**: `/tests/cognitive/patterns_integration_tests.rs`
```rust
//! Integration tests for cognitive patterns

use llmkg::cognitive::{
    ConvergentThinking, DivergentThinking, LateralThinking,
    AdaptiveThinking, CognitivePattern
};
use llmkg::test_support::fixtures::*;

#[tokio::test]
async fn test_convergent_pattern_end_to_end() {
    let graph = create_test_graph();
    // Add test data to graph
    
    let convergent = ConvergentThinking::new(graph.clone());
    let result = convergent.execute("What is a dog?").await.unwrap();
    
    assert!(result.confidence > 0.5);
    assert!(result.answer.contains("dog"));
}

#[tokio::test]
async fn test_adaptive_pattern_strategy_selection() {
    let graph = create_test_graph();
    let adaptive = AdaptiveThinking::new(graph.clone());
    
    // Test factual query routes to convergent
    let result = adaptive.execute("What is quantum computing?").await.unwrap();
    assert_eq!(result.strategy_used, "convergent");
    
    // Test creative query routes to divergent
    let result = adaptive.execute("Give me examples of creative uses").await.unwrap();
    assert_eq!(result.strategy_used, "divergent");
}
```

#### 3.3 Orchestrator Integration Tests
**File**: `/tests/cognitive/orchestrator_integration_tests.rs`
```rust
//! Integration tests for CognitiveOrchestrator

use llmkg::cognitive::{CognitiveOrchestrator, QueryType};
use llmkg::test_support::fixtures::*;

#[tokio::test]
async fn test_orchestrator_pipeline_execution() {
    let graph = create_test_graph();
    let orchestrator = CognitiveOrchestrator::new(graph);
    
    // Test query type detection and pipeline building
    let query = "What are the connections between art and science?";
    let pipeline = orchestrator.build_pipeline(query).await.unwrap();
    
    assert!(pipeline.patterns.contains(&PatternType::Lateral));
}
```

### Phase 4: Specialized Test Types

#### 4.1 Property-Based Tests
**File**: `/tests/cognitive/property_tests.rs`
```rust
use proptest::prelude::*;
use llmkg::cognitive::*;

proptest! {
    #[test]
    fn test_attention_weights_sum_to_one(
        target_count in 1..10usize,
        focus_strength in 0.1..1.0f32,
    ) {
        // Property: attention weights should always sum to ~1.0
        let manager = create_test_attention_manager();
        let targets = create_test_entity_keys(target_count);
        
        manager.focus_attention(targets, focus_strength, AttentionType::Divided)
            .await.unwrap();
            
        let snapshot = manager.get_attention_snapshot().await;
        let sum: f32 = snapshot.attention_weights.values().sum();
        
        prop_assert!((sum - 1.0).abs() < 0.01);
    }
}
```

#### 4.2 Performance Tests
**File**: `/tests/cognitive/performance_tests.rs`
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn attention_switching_benchmark(c: &mut Criterion) {
    c.bench_function("attention_switching", |b| {
        let manager = create_test_attention_manager();
        let targets1 = create_test_entity_keys(5);
        let targets2 = create_test_entity_keys(5);
        
        b.iter(|| {
            black_box(async {
                manager.focus_attention(targets1.clone(), 1.0, AttentionType::Selective).await;
                manager.focus_attention(targets2.clone(), 1.0, AttentionType::Selective).await;
            });
        });
    });
}
```

### Phase 5: Test Data Management

#### 5.1 Test Scenarios Module
**File**: `src/test_support/scenarios.rs`
```rust
/// Predefined test scenarios for cognitive patterns
pub struct TestScenario {
    pub name: &'static str,
    pub query: &'static str,
    pub expected_pattern: PatternType,
    pub expected_confidence_min: f32,
    pub graph_setup: Box<dyn Fn(&mut BrainEnhancedKnowledgeGraph)>,
}

pub fn get_test_scenarios() -> Vec<TestScenario> {
    vec![
        TestScenario {
            name: "simple_factual_query",
            query: "What is a dog?",
            expected_pattern: PatternType::Convergent,
            expected_confidence_min: 0.7,
            graph_setup: Box::new(|graph| {
                // Add dog entity with properties
            }),
        },
        // More scenarios...
    ]
}
```

### Phase 6: Migration Steps

#### 6.1 Step-by-Step Migration Process

1. **Create test_support module structure**
   ```bash
   mkdir src/test_support
   touch src/test_support/mod.rs
   touch src/test_support/fixtures.rs
   touch src/test_support/builders.rs
   touch src/test_support/assertions.rs
   touch src/test_support/scenarios.rs
   ```

2. **Update Cargo.toml**
   ```toml
   [dev-dependencies]
   proptest = "1.0"
   criterion = "0.5"
   
   [[bench]]
   name = "cognitive_benchmarks"
   harness = false
   ```

3. **For each cognitive module:**
   
   a. **Identify private functions that need testing**
   ```rust
   // In divergent.rs
   fn calculate_concept_similarity() // private, needs unit test
   pub async fn execute_divergent_exploration() // public, needs integration test
   ```
   
   b. **Move private function tests back to source file**
   ```rust
   // At end of divergent.rs
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_calculate_concept_similarity() {
           // Direct test of private function
       }
   }
   ```
   
   c. **Create integration tests for public API**
   ```rust
   // In /tests/cognitive/divergent_integration_tests.rs
   #[tokio::test]
   async fn test_divergent_exploration_public_api() {
       // Test through public interface only
   }
   ```

4. **Delete non-functional test infrastructure**
   - Remove `test_impls` module (it doesn't work from tests/)
   - Remove any traits pretending to give private access
   - Remove "adapted" tests that lost their original purpose

5. **Update test organization**
   ```
   tests/
   └── cognitive/
       ├── attention_integration_tests.rs
       ├── patterns_integration_tests.rs
       ├── orchestrator_integration_tests.rs
       ├── property_tests.rs
       ├── performance_tests.rs
       └── mod.rs
   ```

### Phase 7: Validation and Documentation

#### 7.1 Create Test Documentation
**File**: `tests/cognitive/README.md`
```markdown
# Cognitive Module Test Suite

## Test Organization

### Unit Tests (in source files)
- Test private functions and implementation details
- Located in `#[cfg(test)]` modules within each source file
- Have access to private items

### Integration Tests (in tests/ directory)
- Test public API behavior
- Test component interactions
- No access to private items

### Test Support Library
- Shared fixtures and utilities in `src/test_support/`
- Available to both unit and integration tests

## Running Tests

```bash
# Run all tests
cargo test

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test '*'

# Run specific test file
cargo test --test attention_integration_tests
```
```

#### 7.2 Validation Checklist

- [ ] All private function tests are in source files
- [ ] All integration tests only use public APIs
- [ ] No test infrastructure pretends to access private items
- [ ] Test names clearly indicate what they test
- [ ] Performance-critical paths have benchmarks
- [ ] Property tests validate invariants
- [ ] Documentation explains test organization

### Phase 8: Specific Function-Level Changes

#### 8.1 AttentionManager Test Migration

**Move back to `src/cognitive/attention_manager.rs`:**
```rust
#[cfg(test)]
mod tests {
    // Unit tests that need private access
    
    #[test]
    fn test_calculate_attention_weights_selective() {
        let manager = AttentionManager::new(/* deps */);
        let weights = manager.calculate_attention_weights(/* private method */);
        // assertions
    }
    
    #[test]
    fn test_update_cognitive_load() {
        let mut state = AttentionState::new();
        state.update_cognitive_load(0.5); // private method
        assert_eq!(state.cognitive_load, 0.5);
    }
}
```

**Create new `/tests/cognitive/attention_integration_tests.rs`:**
```rust
// Integration tests using only public API

#[tokio::test]
async fn test_attention_management_workflow() {
    let manager = /* create via public API */;
    
    // Test public methods only
    manager.focus_attention(/* params */).await?;
    let snapshot = manager.get_attention_snapshot().await;
    
    // Assertions on public data
}
```

#### 8.2 Pattern-Specific Migrations

For each cognitive pattern:

**Convergent Pattern:**
- Move `test_levenshtein_distance` → `src/cognitive/convergent.rs`
- Move `test_calculate_concept_relevance` → `src/cognitive/convergent.rs`
- Create `test_convergent_query_execution` → `/tests/cognitive/patterns_integration_tests.rs`

**Divergent Pattern:**
- Move `test_calculate_concept_similarity` → `src/cognitive/divergent.rs`
- Move `test_infer_exploration_type` → `src/cognitive/divergent.rs`
- Create `test_divergent_exploration` → `/tests/cognitive/patterns_integration_tests.rs`

**Lateral Pattern:**
- Move `test_parse_lateral_query` → `src/cognitive/lateral.rs`
- Create `test_lateral_connections` → `/tests/cognitive/patterns_integration_tests.rs`

### Phase 9: Final Cleanup

1. **Remove all test code from `/tests/` that attempts private access**
2. **Delete the non-functional `test_impls` module**
3. **Ensure no test is "adapted" - each test does what it claims**
4. **Run full test suite and verify everything passes**
5. **Document any breaking changes for team**

## Summary

This plan completely restructures the test suite to work properly with Rust's visibility rules:
- Unit tests stay with the code they test
- Integration tests test public behavior
- Shared test utilities are properly accessible
- No fake abstractions or non-functional code
- Clear separation of concerns
- Maintainable and understandable structure