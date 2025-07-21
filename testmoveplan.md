# Test Migration Plan - Detailed Instructions for AI Model

## Overview
This document contains precise instructions for moving embedded test code from source files in `src/cognitive/` to test files in `/tests/cognitive/`. Each instruction is self-contained with full context.

---

## File 1: src/cognitive/attention_manager.rs

### Helper Function 1: create_test_entity_keys
**Current Location:** `src/cognitive/attention_manager.rs` lines 872-889
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Test helper function
**Dependencies:** None
**Full Function:**
```rust
fn create_test_entity_keys(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("entity_{}", i)).collect()
}
```
**Instructions:** 
1. Copy this entire function from the source file
2. Add it to the test file inside the test module after the imports
3. Delete it from the source file

### Helper Function 2: create_test_attention_manager
**Current Location:** `src/cognitive/attention_manager.rs` lines 891-906
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Test helper function that creates test fixtures
**Dependencies:** Requires these imports:
```rust
use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, MemoryItem};
use crate::core::activation_engine::ActivationPropagationEngine;
use crate::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
use std::sync::Arc;
```
**Full Function:**
```rust
async fn create_test_attention_manager() -> (
    AttentionManager,
    Arc<CognitiveOrchestrator>,
    Arc<ActivationPropagationEngine>,
    Arc<WorkingMemorySystem>,
) {
    let config = BrainEnhancedConfig::default();
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(config.clone()));
    let orchestrator = Arc::new(CognitiveOrchestrator::new(graph.clone()));
    let activation_engine = Arc::new(ActivationPropagationEngine::new(graph.clone()));
    let working_memory = Arc::new(WorkingMemorySystem::new(config.working_memory));
    
    let attention_manager = AttentionManager::new(
        orchestrator.clone(),
        activation_engine.clone(),
        working_memory.clone(),
    );
    
    (attention_manager, orchestrator, activation_engine, working_memory)
}
```
**Instructions:**
1. Copy this entire function from the source file
2. Add it to the test file inside the test module after the imports
3. Ensure all the required imports listed above are added to the test file
4. Delete it from the source file

### Test Function 1: test_attention_state_new
**Current Location:** `src/cognitive/attention_manager.rs` lines 908-916
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_attention_state_new() {
    let state = AttentionState::new();
    assert!(state.current_focus.targets.is_empty());
    assert_eq!(state.attention_capacity, 1.0);
    assert_eq!(state.cognitive_load, 0.0);
    assert_eq!(state.attention_type, AttentionType::Selective);
    assert!(state.inhibited_targets.is_empty());
    assert!(state.history.is_empty());
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_attention_manager.rs`
3. Delete it from the source file

### Test Function 2: test_attention_state_update_cognitive_load
**Current Location:** `src/cognitive/attention_manager.rs` lines 918-943
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_attention_state_update_cognitive_load() {
    let mut state = AttentionState::new();
    
    // Test normal load
    state.update_cognitive_load(0.5);
    assert_eq!(state.cognitive_load, 0.5);
    assert_eq!(state.attention_capacity, 0.5); // 1.0 - 0.5
    
    // Test high load
    state.update_cognitive_load(0.8);
    assert_eq!(state.cognitive_load, 0.8);
    assert_eq!(state.attention_capacity, 0.2); // 1.0 - 0.8
    
    // Test overload (should clamp)
    state.update_cognitive_load(1.5);
    assert_eq!(state.cognitive_load, 1.0);
    assert_eq!(state.attention_capacity, 0.0);
    
    // Test negative load (should clamp)
    state.update_cognitive_load(-0.5);
    assert_eq!(state.cognitive_load, 0.0);
    assert_eq!(state.attention_capacity, 1.0);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_attention_manager.rs`
3. Delete it from the source file

### Test Function 3: test_calculate_attention_weights_selective
**Current Location:** `src/cognitive/attention_manager.rs` lines 945-962
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Async unit test
**Test Attribute:** `#[tokio::test]`
**Dependencies:** Requires the helper functions and test trait implementations
**Full Function:**
```rust
#[tokio::test]
async fn test_calculate_attention_weights_selective() {
    let (attention_manager, _, _, _) = create_test_attention_manager().await;
    let targets = create_test_entity_keys(3);
    
    // Use the test trait to access the private method
    use test_impls::AttentionCalculator;
    let weights = attention_manager.test_calculate_weights(&targets, AttentionType::Selective);
    
    // Selective attention should focus on first target
    assert_eq!(weights.len(), 3);
    assert!(weights[&targets[0]] > 0.9);
    assert!(weights[&targets[1]] < 0.1);
    assert!(weights[&targets[2]] < 0.1);
    
    // Weights should sum to approximately 1.0
    let sum: f32 = weights.values().sum();
    assert!((sum - 1.0).abs() < 0.01);
}
```
**Instructions:**
1. Copy this entire test function including the `#[tokio::test]` attribute
2. Add it to the test module in `/tests/cognitive/test_attention_manager.rs`
3. This test depends on the `test_impls` module and helper functions - ensure they are moved first
4. Delete it from the source file

### Test Function 4: test_calculate_attention_weights_divided
**Current Location:** `src/cognitive/attention_manager.rs` lines 964-990
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Async unit test
**Test Attribute:** `#[tokio::test]`
**Dependencies:** Requires the helper functions and test trait implementations
**Full Function:**
```rust
#[tokio::test]
async fn test_calculate_attention_weights_divided() {
    let (attention_manager, _, _, _) = create_test_attention_manager().await;
    
    // Test with 4 targets
    let targets = create_test_entity_keys(4);
    use test_impls::AttentionCalculator;
    let weights = attention_manager.test_calculate_weights(&targets, AttentionType::Divided);
    
    // Divided attention should distribute weights
    assert_eq!(weights.len(), 4);
    
    // Each target should get approximately equal weight
    let expected_weight = 1.0 / 4.0;
    for target in &targets {
        let weight = weights[target];
        assert!((weight - expected_weight).abs() < 0.01);
    }
    
    // With penalty, actual weights might be lower
    // but they should still sum to less than or equal to 1.0
    let sum: f32 = weights.values().sum();
    assert!(sum <= 1.0);
    
    // Test with just 1 target (no division needed)
    let single_target = create_test_entity_keys(1);
    let single_weights = attention_manager.test_calculate_weights(&single_target, AttentionType::Divided);
    assert_eq!(single_weights.len(), 1);
    assert_eq!(single_weights[&single_target[0]], 1.0);
}
```
**Instructions:**
1. Copy this entire test function including the `#[tokio::test]` attribute
2. Add it to the test module in `/tests/cognitive/test_attention_manager.rs`
3. This test depends on the `test_impls` module and helper functions - ensure they are moved first
4. Delete it from the source file

### Test Function 5: test_calculate_memory_load
**Current Location:** `src/cognitive/attention_manager.rs` lines 992-1024
**Move To:** `/tests/cognitive/test_attention_manager.rs`
**Function Type:** Async unit test
**Test Attribute:** `#[tokio::test]`
**Dependencies:** Requires the helper functions and WorkingMemorySystem types
**Full Function:**
```rust
#[tokio::test]
async fn test_calculate_memory_load() {
    let (attention_manager, _, _, working_memory) = create_test_attention_manager().await;
    
    // Start with empty memory
    let initial_load = attention_manager.calculate_memory_load(&working_memory).await;
    assert_eq!(initial_load, 0.0);
    
    // Add some items to memory
    let items = vec![
        MemoryItem {
            content: MemoryContent::Concept("test1".to_string()),
            activation: 0.8,
            timestamp: Instant::now(),
            access_count: 1,
            last_accessed: Instant::now(),
        },
        MemoryItem {
            content: MemoryContent::Relationship("test2".to_string(), "test3".to_string()),
            activation: 0.6,
            timestamp: Instant::now(),
            access_count: 2,
            last_accessed: Instant::now(),
        },
    ];
    
    // Store items in working memory
    for item in items {
        working_memory.store(item.content.clone(), item.activation).await.unwrap();
    }
    
    // Calculate load should be proportional to number of items and their activation
    let load = attention_manager.calculate_memory_load(&working_memory).await;
    assert!(load > 0.0);
    assert!(load < 1.0);
}
```
**Instructions:**
1. Copy this entire test function including the `#[tokio::test]` attribute
2. Add it to the test module in `/tests/cognitive/test_attention_manager.rs`
3. This test depends on the helper functions - ensure they are moved first
4. Add required import: `use std::time::Instant;`
5. Delete it from the source file

### Test Trait Implementations Module: test_impls
**Current Location:** `src/cognitive/attention_manager.rs` lines 1029-1063
**Move To:** `/tests/cognitive/test_attention_manager.rs` 
**Module Type:** Test-specific trait implementations for accessing private methods
**Full Module:**
```rust
mod test_impls {
    use super::*;
    
    pub trait AttentionCalculator {
        fn test_calculate_weights(&self, targets: &[String], attention_type: AttentionType) -> HashMap<String, f32>;
    }
    
    impl AttentionCalculator for AttentionManager {
        fn test_calculate_weights(&self, targets: &[String], attention_type: AttentionType) -> HashMap<String, f32> {
            self.calculate_attention_weights(targets, attention_type)
        }
    }
    
    pub trait AttentionStateManager {
        fn test_get_state(&self) -> &AttentionState;
    }
    
    impl AttentionStateManager for AttentionManager {
        fn test_get_state(&self) -> &AttentionState {
            &self.attention_state
        }
    }
}
```
**Instructions:**
1. Copy this entire module including the `mod test_impls {` wrapper
2. Add it to `/tests/cognitive/test_attention_manager.rs` at the module level (not inside a test function)
3. This module is used by test functions 3 and 4, so it must be available in the test file
4. Delete it from the source file

### Cleanup for attention_manager.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 870-1063
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## File 2: src/cognitive/lateral.rs

### Test Function 1: test_lateral_query_parsing
**Current Location:** `src/cognitive/lateral.rs` lines 732-741
**Move To:** `/tests/cognitive/test_lateral.rs`
**Function Type:** Async unit test
**Test Attribute:** `#[tokio::test]`
**Full Function:**
```rust
#[tokio::test]
async fn test_lateral_query_parsing() {
    let queries = vec![
        ("connections between art and science", ("art", "science")),
        ("relationship between music and mathematics", ("music", "mathematics")),
        ("how does psychology relate to neuroscience", ("psychology", "neuroscience")),
    ];
    
    for (query, expected) in queries {
        assert_eq!(parse_lateral_query(query).await.unwrap(), expected);
    }
}
```
**Instructions:**
1. Copy this entire test function including the `#[tokio::test]` attribute
2. Add it to the test module in `/tests/cognitive/test_lateral.rs`
3. Add import: `use llmkg::cognitive::lateral::parse_lateral_query;` (or appropriate import path)
4. Delete it from the source file

### Test Function 2: test_concept_relevance
**Current Location:** `src/cognitive/lateral.rs` lines 743-751
**Move To:** `/tests/cognitive/test_lateral.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_concept_relevance() {
    assert!(calculate_concept_relevance("dog", "animal") > 0.5);
    assert!(calculate_concept_relevance("computer", "technology") > 0.5);
    assert!(calculate_concept_relevance("ocean", "water") > 0.5);
    assert!(calculate_concept_relevance("book", "knowledge") > 0.3);
    assert!(calculate_concept_relevance("sun", "energy") > 0.3);
    assert!(calculate_concept_relevance("car", "fish") < 0.1);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_lateral.rs`
3. Add import: `use llmkg::cognitive::lateral::calculate_concept_relevance;` (or appropriate import path)
4. Delete it from the source file

### Cleanup for lateral.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 730-752
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## File 3: src/cognitive/divergent.rs

### Test Function 1: test_exploration_type_inference
**Current Location:** `src/cognitive/divergent.rs` lines 1079-1098
**Move To:** `/tests/cognitive/test_divergent.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_exploration_type_inference() {
    use super::*;
    
    // Example queries
    assert_eq!(
        infer_exploration_type("give me examples of machine learning algorithms"),
        ExplorationType::Examples
    );
    
    // Creative queries
    assert_eq!(
        infer_exploration_type("brainstorm innovative uses for blockchain"),
        ExplorationType::Creative
    );
    
    // Related queries
    assert_eq!(
        infer_exploration_type("what concepts are related to quantum computing"),
        ExplorationType::Related
    );
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_divergent.rs`
3. Add imports: 
   - `use llmkg::cognitive::divergent::{infer_exploration_type, ExplorationType};`
4. Remove the `use super::*;` line as it won't be needed in the test file
5. Delete it from the source file

### Test Function 2: test_seed_concept_extraction
**Current Location:** `src/cognitive/divergent.rs` lines 1100-1115
**Move To:** `/tests/cognitive/test_divergent.rs`
**Function Type:** Async unit test
**Test Attribute:** `#[tokio::test]`
**Full Function:**
```rust
#[tokio::test]
async fn test_seed_concept_extraction() {
    let test_cases = vec![
        ("examples of dogs", "dogs"),
        ("brainstorm about artificial intelligence", "artificial intelligence"),
        ("creative uses for plastic bottles", "plastic bottles"),
        ("things related to space exploration", "space exploration"),
        ("innovative applications of nanotechnology", "nanotechnology"),
    ];
    
    for (query, expected) in test_cases {
        let result = extract_seed_concept(query).await.unwrap();
        assert_eq!(result, expected);
    }
}
```
**Instructions:**
1. Copy this entire test function including the `#[tokio::test]` attribute
2. Add it to the test module in `/tests/cognitive/test_divergent.rs`
3. Add import: `use llmkg::cognitive::divergent::extract_seed_concept;`
4. Delete it from the source file

### Test Function 3: test_concept_similarity
**Current Location:** `src/cognitive/divergent.rs` lines 1117-1142
**Move To:** `/tests/cognitive/test_divergent.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_concept_similarity() {
    // High similarity
    assert!(calculate_concept_similarity("dog", "puppy") > 0.8);
    assert!(calculate_concept_similarity("car", "automobile") > 0.8);
    assert!(calculate_concept_similarity("happy", "joyful") > 0.7);
    
    // Medium similarity
    assert!(calculate_concept_similarity("dog", "cat") > 0.4);
    assert!(calculate_concept_similarity("computer", "laptop") > 0.6);
    assert!(calculate_concept_similarity("tree", "forest") > 0.5);
    
    // Low similarity
    assert!(calculate_concept_similarity("dog", "quantum") < 0.2);
    assert!(calculate_concept_similarity("music", "mathematics") < 0.4);
    assert!(calculate_concept_similarity("ocean", "desert") < 0.3);
    
    // Identical concepts
    assert_eq!(calculate_concept_similarity("test", "test"), 1.0);
    
    // Empty strings
    assert_eq!(calculate_concept_similarity("", "test"), 0.0);
    assert_eq!(calculate_concept_similarity("test", ""), 0.0);
    assert_eq!(calculate_concept_similarity("", ""), 0.0);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_divergent.rs`
3. Add import: `use llmkg::cognitive::divergent::calculate_concept_similarity;`
4. Delete it from the source file

### Cleanup for divergent.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 1077-1143
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## File 4: src/cognitive/convergent.rs

### Test Function 1: test_levenshtein_distance
**Current Location:** `src/cognitive/convergent.rs` lines 1019-1027
**Move To:** `/tests/cognitive/test_convergent.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_levenshtein_distance() {
    assert_eq!(levenshtein_distance("", ""), 0);
    assert_eq!(levenshtein_distance("hello", "hello"), 0);
    assert_eq!(levenshtein_distance("hello", "hallo"), 1);
    assert_eq!(levenshtein_distance("hello", ""), 5);
    assert_eq!(levenshtein_distance("", "world"), 5);
    assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_convergent.rs`
3. Add import: `use llmkg::cognitive::convergent::levenshtein_distance;`
4. Delete it from the source file

### Test Function 2: test_concept_relevance
**Current Location:** `src/cognitive/convergent.rs` lines 1029-1038
**Move To:** `/tests/cognitive/test_convergent.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_concept_relevance() {
    assert_eq!(calculate_concept_relevance("", "dog", "dog"), 1.0);
    assert!(calculate_concept_relevance("mammal", "dog", "canine") > 0.5);
    assert!(calculate_concept_relevance("furniture", "dog", "chair") < 0.3);
    assert_eq!(calculate_concept_relevance("", "", ""), 0.0);
    
    // Test with entity description
    assert!(calculate_concept_relevance("a four-legged pet", "dog", "pet") > 0.7);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_convergent.rs`
3. Add import: `use llmkg::cognitive::convergent::calculate_concept_relevance;`
4. Delete it from the source file

### Test Function 3: test_stop_words
**Current Location:** `src/cognitive/convergent.rs` lines 1040-1050
**Move To:** `/tests/cognitive/test_convergent.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_stop_words() {
    assert!(is_stop_word("the"));
    assert!(is_stop_word("is"));
    assert!(is_stop_word("and"));
    assert!(is_stop_word("of"));
    assert!(!is_stop_word("dog"));
    assert!(!is_stop_word("computer"));
    assert!(!is_stop_word("quantum"));
    assert!(!is_stop_word(""));
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_convergent.rs`
3. Add import: `use llmkg::cognitive::convergent::is_stop_word;`
4. Delete it from the source file

### Cleanup for convergent.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 1017-1051
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## File 5: src/cognitive/orchestrator.rs

### Test Function 1: test_orchestrator_creation
**Current Location:** `src/cognitive/orchestrator.rs` lines 607-612
**Move To:** `/tests/cognitive/test_orchestrator.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_orchestrator_creation() {
    let config = BrainEnhancedConfig::default();
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(config));
    let orchestrator = CognitiveOrchestrator::new(graph);
    assert!(orchestrator.active_patterns.is_empty());
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_orchestrator.rs`
3. Add imports:
   - `use llmkg::cognitive::orchestrator::CognitiveOrchestrator;`
   - `use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};`
   - `use std::sync::Arc;`
4. Delete it from the source file

### Test Function 2: test_pattern_weight_calculation
**Current Location:** `src/cognitive/orchestrator.rs` lines 614-620
**Move To:** `/tests/cognitive/test_orchestrator.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_pattern_weight_calculation() {
    assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.9, 5, false));
    assert!(calculate_pattern_weight(0.9, 5, true) > calculate_pattern_weight(0.7, 5, true));
    assert!(calculate_pattern_weight(0.9, 3, true) > calculate_pattern_weight(0.9, 10, true));
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it to the test module in `/tests/cognitive/test_orchestrator.rs`
3. Add import: `use llmkg::cognitive::orchestrator::calculate_pattern_weight;`
4. Delete it from the source file

### Cleanup for orchestrator.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 605-621
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## File 6: src/cognitive/neural_query.rs

### Create New Test File
**Action Required:** Create a new file `/tests/cognitive/test_neural_query.rs`
**File Header:**
```rust
//! Tests for the neural query processor

use llmkg::cognitive::neural_query::*;
use llmkg::cognitive::types::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    // Test functions will go here
}
```

### Test Function 1: test_query_intent_identification
**Current Location:** `src/cognitive/neural_query.rs` lines 659-682
**Move To:** `/tests/cognitive/test_neural_query.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_query_intent_identification() {
    let test_cases = vec![
        ("What is quantum computing?", QueryIntent::Factual),
        ("Give me examples of renewable energy", QueryIntent::Exploratory),
        ("How does photosynthesis relate to solar panels?", QueryIntent::Relational),
        ("Why is the sky blue?", QueryIntent::Causal),
        ("Analyze the impact of AI on employment", QueryIntent::Analytical),
        ("What are the steps to bake a cake?", QueryIntent::Procedural),
    ];
    
    for (query, expected_intent) in test_cases {
        let intent = identify_query_intent(query);
        assert_eq!(
            intent, expected_intent,
            "Failed for query: {}",
            query
        );
    }
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it inside the `mod tests` block in `/tests/cognitive/test_neural_query.rs`
3. Ensure the import `use llmkg::cognitive::neural_query::identify_query_intent;` is present
4. Delete it from the source file

### Test Function 2: test_concept_extraction
**Current Location:** `src/cognitive/neural_query.rs` lines 684-701
**Move To:** `/tests/cognitive/test_neural_query.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_concept_extraction() {
    let test_cases = vec![
        (
            "How does machine learning relate to artificial intelligence?",
            vec!["machine learning", "artificial intelligence"],
        ),
        (
            "What is the connection between DNA and heredity?",
            vec!["dna", "heredity", "connection"],
        ),
        (
            "Examples of sustainable energy sources",
            vec!["sustainable energy", "sources", "examples"],
        ),
    ];
    
    for (query, expected_concepts) in test_cases {
        let concepts = extract_concepts(query);
        for expected in expected_concepts {
            assert!(concepts.iter().any(|c| c.name.to_lowercase().contains(expected)));
        }
    }
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it inside the `mod tests` block in `/tests/cognitive/test_neural_query.rs`
3. Ensure the import `use llmkg::cognitive::neural_query::extract_concepts;` is present
4. Delete it from the source file

### Test Function 3: test_relationship_extraction
**Current Location:** `src/cognitive/neural_query.rs` lines 703-717
**Move To:** `/tests/cognitive/test_neural_query.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_relationship_extraction() {
    let relationships = extract_relationships(
        "How does the brain process visual information from the eyes?"
    );
    
    assert!(!relationships.is_empty());
    assert!(relationships.iter().any(|r| r.source.contains("brain")));
    assert!(relationships.iter().any(|r| r.target.contains("information") || r.target.contains("eyes")));
    assert!(relationships.iter().any(|r| r.relationship_type.contains("process")));
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it inside the `mod tests` block in `/tests/cognitive/test_neural_query.rs`
3. Ensure the import `use llmkg::cognitive::neural_query::extract_relationships;` is present
4. Delete it from the source file

### Test Function 4: test_constraint_extraction
**Current Location:** `src/cognitive/neural_query.rs` lines 719-730
**Move To:** `/tests/cognitive/test_neural_query.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_constraint_extraction() {
    let constraints = extract_constraints(
        "Show me recent advances in quantum computing from the last 5 years"
    );
    
    assert!(!constraints.is_empty());
    assert!(constraints.iter().any(|c| matches!(c, QueryConstraint::Temporal(_))));
    assert!(constraints.iter().any(|c| matches!(c, QueryConstraint::Domain(_))));
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it inside the `mod tests` block in `/tests/cognitive/test_neural_query.rs`
3. Ensure the import `use llmkg::cognitive::neural_query::extract_constraints;` is present
4. Delete it from the source file

### Test Function 5: test_domain_inference
**Current Location:** `src/cognitive/neural_query.rs` lines 732-742
**Move To:** `/tests/cognitive/test_neural_query.rs`
**Function Type:** Unit test
**Test Attribute:** `#[test]`
**Full Function:**
```rust
#[test]
fn test_domain_inference() {
    assert_eq!(infer_domain("quantum computing algorithms"), vec!["physics", "computer_science"]);
    assert_eq!(infer_domain("photosynthesis in plants"), vec!["biology"]);
    assert_eq!(infer_domain("machine learning models"), vec!["computer_science", "mathematics"]);
    assert_eq!(infer_domain("Renaissance art history"), vec!["art", "history"]);
    assert_eq!(infer_domain("general question"), vec!["general"]);
}
```
**Instructions:**
1. Copy this entire test function including the `#[test]` attribute
2. Add it inside the `mod tests` block in `/tests/cognitive/test_neural_query.rs`
3. Ensure the import `use llmkg::cognitive::neural_query::infer_domain;` is present
4. Delete it from the source file

### Update Module Declaration
**File:** `/tests/cognitive/mod.rs`
**Add Line:** `pub mod test_neural_query;`
**Instructions:**
1. Open `/tests/cognitive/mod.rs`
2. Add the line `pub mod test_neural_query;` after the other module declarations
3. Save the file

### Cleanup for neural_query.rs
**After moving all tests:**
1. Delete the entire `#[cfg(test)]` block from lines 657-743
2. Delete the line `mod tests {` and its closing `}`
3. Ensure no test-related code remains in the source file

---

## Final Verification Steps

After completing all migrations:

1. **Check source files:** Ensure no `#[cfg(test)]` blocks remain in any file under `src/cognitive/`
2. **Run tests:** Execute `cargo test --test '*' --package llmkg` to verify all tests still pass
3. **Count tests:** Verify that the total number of tests hasn't decreased
4. **Compile check:** Run `cargo check` to ensure no compilation errors

## Summary

Total items to move:
- 20 test functions
- 2 helper functions  
- 1 test trait module
- 1 new test file to create

All test code should be completely removed from source files and placed in the appropriate test files under `/tests/cognitive/`.