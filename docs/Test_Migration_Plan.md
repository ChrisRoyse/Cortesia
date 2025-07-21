# Test Migration Plan: Moving Embedded Tests to /tests/ Directory

## Overview
This document provides a detailed, function-level plan for migrating all embedded tests from `src/cognitive/` files to the proper `/tests/cognitive/` directory structure.

## Files Requiring Test Migration

### 1. **src/cognitive/attention_manager.rs**

#### Tests to Move (lines 870-1063):

**Helper Functions:**
```rust
// Lines 872-889
fn create_test_entity_keys(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("entity_{}", i)).collect()
}

// Lines 891-906
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

**Test Functions:**
1. `test_attention_state_new()` - Tests AttentionState initialization
2. `test_attention_state_update_cognitive_load()` - Tests cognitive load updates
3. `test_calculate_attention_weights_selective()` - Tests selective attention weight calculation
4. `test_calculate_attention_weights_divided()` - Tests divided attention weight calculation
5. `test_calculate_memory_load()` - Tests memory load calculation

**Test Trait Implementations (lines 1029-1063):**
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

**Action:** Move all to `/tests/cognitive/test_attention_manager.rs`

---

### 2. **src/cognitive/lateral.rs**

#### Tests to Move (lines 730-752):

**Test Functions:**
1. `test_lateral_query_parsing()` - Tests query parsing for two concepts
2. `test_concept_relevance()` - Tests calculate_concept_relevance function

**Action:** Move to `/tests/cognitive/test_lateral.rs`

---

### 3. **src/cognitive/divergent.rs**

#### Tests to Move (lines 1077-1143):

**Test Functions:**
1. `test_exploration_type_inference()` - Tests infer_exploration_type function
2. `test_seed_concept_extraction()` - Tests extract_seed_concept function
3. `test_concept_similarity()` - Tests calculate_concept_similarity function

**Action:** Move to `/tests/cognitive/test_divergent.rs`

---

### 4. **src/cognitive/convergent.rs**

#### Tests to Move (lines 1017-1051):

**Test Functions:**
1. `test_levenshtein_distance()` - Tests Levenshtein distance calculation
2. `test_concept_relevance()` - Tests calculate_concept_relevance function
3. `test_stop_words()` - Tests is_stop_word function

**Action:** Move to `/tests/cognitive/test_convergent.rs`

---

### 5. **src/cognitive/orchestrator.rs**

#### Tests to Move (lines 605-621):

**Test Functions:**
1. `test_orchestrator_creation()` - Tests CognitiveOrchestrator initialization
2. `test_pattern_weight_calculation()` - Tests calculate_pattern_weight function

**Action:** Move to `/tests/cognitive/test_orchestrator.rs`

---

### 6. **src/cognitive/neural_query.rs**

#### Tests to Move (lines 657-743):

**Test Functions:**
1. `test_query_intent_identification()` - Tests identify_query_intent function
2. `test_concept_extraction()` - Tests extract_concepts function
3. `test_relationship_extraction()` - Tests extract_relationships function
4. `test_constraint_extraction()` - Tests extract_constraints function
5. `test_domain_inference()` - Tests infer_domain function

**Action:** 
1. Create new file `/tests/cognitive/test_neural_query.rs`
2. Move all tests to the new file
3. Add module declaration to `/tests/cognitive/mod.rs`

---

## Migration Steps

### Step 1: Create Missing Test File
```bash
touch tests/cognitive/test_neural_query.rs
```

### Step 2: Update `/tests/cognitive/mod.rs`
Add the following line:
```rust
pub mod test_neural_query;
```

### Step 3: For Each Source File

1. **Copy the entire `#[cfg(test)]` module content** to the corresponding test file
2. **Remove test-specific imports** from the module level and add them to the test file
3. **Update imports** in the test file to properly reference the source module:
   ```rust
   use llmkg::cognitive::{attention_manager::*, types::*};
   // or appropriate imports based on what's needed
   ```
4. **Delete the entire `#[cfg(test)]` block** from the source file

### Step 4: Handle Test Utilities

For shared test utilities (like those in `attention_manager.rs`):

1. **Option A**: Move to `/tests/cognitive/test_utils.rs` if used by multiple test files
2. **Option B**: Keep in the specific test file if only used there

### Step 5: Verify Migration

After migration, run:
```bash
cargo test --test '*' --package llmkg
```

To ensure all tests still pass and are discovered properly.

## File-by-File Checklist

- [ ] `src/cognitive/attention_manager.rs`
  - [ ] Move 2 helper functions
  - [ ] Move 5 test functions
  - [ ] Move test trait implementations
  - [ ] Remove `#[cfg(test)]` block

- [ ] `src/cognitive/lateral.rs`
  - [ ] Move 2 test functions
  - [ ] Remove `#[cfg(test)]` block

- [ ] `src/cognitive/divergent.rs`
  - [ ] Move 3 test functions
  - [ ] Remove `#[cfg(test)]` block

- [ ] `src/cognitive/convergent.rs`
  - [ ] Move 3 test functions
  - [ ] Remove `#[cfg(test)]` block

- [ ] `src/cognitive/orchestrator.rs`
  - [ ] Move 2 test functions
  - [ ] Remove `#[cfg(test)]` block

- [ ] `src/cognitive/neural_query.rs`
  - [ ] Create new test file
  - [ ] Move 5 test functions
  - [ ] Remove `#[cfg(test)]` block

- [ ] Update `/tests/cognitive/mod.rs`
  - [ ] Add `test_neural_query` module

## Post-Migration Verification

1. **Ensure no `#[cfg(test)]` blocks remain** in any `src/cognitive/*.rs` files
2. **Run all tests** to verify they still pass
3. **Check test count** hasn't decreased
4. **Verify test coverage** remains the same or improves

## Benefits After Migration

1. **Cleaner source files** - Implementation code only
2. **Better test organization** - All tests in `/tests/`
3. **Faster compilation** - Source files compile without test code
4. **Easier maintenance** - Tests grouped by module in one location
5. **Consistent project structure** - Follows Rust best practices for larger projects