# Test Migration Fix Plan - Parallel Agent Tasks

## Overview
This document outlines parallel agent tasks to fix all issues from the test migration without losing test intent or context.

## Agent Task Groups

### Group 1: Source File Cleanup Agents (6 agents)
Each agent removes test code from one source file only.

**Agent 1.1: Clean attention_manager.rs**
- File: `src/cognitive/attention_manager.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 870-1063)
- Preserve: All non-test code

**Agent 1.2: Clean lateral.rs**
- File: `src/cognitive/lateral.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 730-752)
- Preserve: All non-test code

**Agent 1.3: Clean divergent.rs**
- File: `src/cognitive/divergent.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 1077-1143)
- Preserve: All non-test code

**Agent 1.4: Clean convergent.rs**
- File: `src/cognitive/convergent.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 1017-1051)
- Preserve: All non-test code

**Agent 1.5: Clean orchestrator.rs**
- File: `src/cognitive/orchestrator.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 605-621)
- Preserve: All non-test code

**Agent 1.6: Clean neural_query.rs**
- File: `src/cognitive/neural_query.rs`
- Task: Remove entire `#[cfg(test)]` block (lines 657-743)
- Preserve: All non-test code

### Group 2: Test Infrastructure Fix Agents (3 agents)

**Agent 2.1: Fix Helper Function Signatures**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Analyze all test functions to determine which need the tuple return vs which need just AttentionManager
  2. Create TWO helper functions:
     - `create_test_attention_manager() -> AttentionManager` (for simple tests)
     - `create_test_attention_manager_with_deps() -> (AttentionManager, Arc<CognitiveOrchestrator>, Arc<ActivationPropagationEngine>, Arc<WorkingMemorySystem>)` (for complex tests)
  3. Update all test calls to use the appropriate helper

**Agent 2.2: Restore test_impls Module**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Verify the test_impls module was correctly moved
  2. Ensure it provides access to private methods for testing:
     - `AttentionCalculator` trait with `test_calculate_weights`
     - `AttentionStateManager` trait with `test_get_state`
  3. Make sure it's accessible to all tests that need it

**Agent 2.3: Fix Import Paths**
- Files: All test files in `/tests/cognitive/`
- Tasks:
  1. Scan for duplicate imports and remove them
  2. Ensure all imports use correct paths (llmkg:: prefix for external imports)
  3. Add any missing imports identified by compilation errors

### Group 3: Test Intent Restoration Agents (5 agents)

**Agent 3.1: Restore test_calculate_attention_weights_selective**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Find the adapted version of this test
  2. Restore it to use `test_impls::AttentionCalculator` trait
  3. Ensure it tests the private `calculate_attention_weights` method directly
  4. Verify it tests selective attention focusing on first target

**Agent 3.2: Restore test_calculate_attention_weights_divided**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Find the completely rewritten version
  2. Replace with original that uses `test_impls::AttentionCalculator`
  3. Ensure it tests private method with divided attention logic
  4. Restore tests for weight distribution and single target edge case

**Agent 3.3: Restore test_calculate_memory_load**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Ensure it uses the tuple-returning helper function correctly
  2. Verify it tests the private calculate_memory_load method
  3. Restore any lost test logic for memory item creation and load calculation

**Agent 3.4: Organize Test Modules**
- File: `/tests/cognitive/test_attention_manager.rs`
- Tasks:
  1. Move edge case tests to a dedicated `edge_case_tests` module
  2. Move integration tests to `integration_tests` module
  3. Move unit tests for private methods to `unit_tests` module
  4. Ensure each test is in the appropriate module

**Agent 3.5: Add Missing Test Traits**
- File: `/tests/cognitive/test_utils.rs` or appropriate location
- Tasks:
  1. If any test traits are used across multiple test files, move them to a shared location
  2. Ensure all test files can access needed test traits
  3. Document which traits provide access to which private methods

### Group 4: Verification Agent (1 agent)

**Agent 4.1: Compile and Verify All Tests**
- Tasks:
  1. Run `cargo test --no-run` to verify compilation
  2. Document any remaining compilation errors
  3. Run `cargo test` to ensure all tests pass
  4. Create a summary report of:
     - Total tests before migration
     - Total tests after migration
     - Any tests that changed behavior
     - Confirmation that test intent is preserved

## Execution Order

1. **Phase 1**: Groups 1 & 2 run in parallel (cleanup source files while fixing test infrastructure)
2. **Phase 2**: Group 3 runs after Phase 1 completes (restore test intent with fixed infrastructure)
3. **Phase 3**: Group 4 runs last (verification)

## Success Criteria

- All source files have zero test code
- All tests compile without errors
- All tests pass
- Original test intent is preserved (testing private methods where intended)
- No duplicate imports or path conflicts
- Helper functions support both simple and complex test needs
- Test organization reflects their purpose (unit vs integration vs edge case)

## Notes for Agents

1. DO NOT adapt tests to "make them work" - preserve original testing intent
2. If a test needs private method access, use the test trait approach
3. Maintain separation between different types of tests
4. Document any changes that affect test behavior
5. Ensure backward compatibility where possible