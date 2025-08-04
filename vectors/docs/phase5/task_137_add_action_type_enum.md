# Task 137: Add ActionType Enum

## Prerequisites Check
- [ ] Task 136 completed: OptimizationAction struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the ActionType enum to categorize different optimization actions that can be performed.

## Task Objective
Define the ActionType enum to specify the types of optimization actions available.

## Steps
1. Add ActionType enum to vector_store.rs:
   ```rust
   /// Types of optimization actions
   #[derive(Debug, Clone, PartialEq)]
   pub enum ActionType {
       /// Switch search mode
       SwitchSearchMode,
       /// Adjust RRF parameters
       AdjustRrfParameters,
       /// Modify cache settings
       ModifyCacheSettings,
       /// Rewrite query
       RewriteQuery,
       /// Warm cache
       WarmCache,
   }
   ```

## Success Criteria
- [ ] ActionType enum added with all action variants
- [ ] Proper derives (Debug, Clone, PartialEq)
- [ ] Compiles without errors

## Time: 2 minutes