# Task 136: Add OptimizationAction Struct

## Prerequisites Check
- [ ] Task 135 completed: OptimizationCondition struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the OptimizationAction struct to define actions taken when optimization conditions are met.

## Task Objective
Define the OptimizationAction struct to specify what actions should be executed during optimization.

## Steps
1. Add OptimizationAction struct to vector_store.rs:
   ```rust
   /// Optimization action to take
   #[derive(Debug, Clone)]
   pub struct OptimizationAction {
       /// Action type
       pub action_type: ActionType,
       /// Parameters for the action
       pub parameters: HashMap<String, String>,
       /// Priority of this action
       pub priority: i32,
   }
   ```

## Success Criteria
- [ ] OptimizationAction struct added with all required fields
- [ ] Proper field types including HashMap for parameters
- [ ] Compiles without errors

## Time: 3 minutes