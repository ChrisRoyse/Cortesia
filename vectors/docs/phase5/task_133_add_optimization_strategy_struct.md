# Task 133: Add OptimizationStrategy Struct

## Prerequisites Check
- [ ] Task 132 completed: QueryPerformanceRecord struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the OptimizationStrategy struct to define optimization rules and conditions.

## Task Objective
Define the OptimizationStrategy struct to represent configurable optimization strategies with conditions and actions.

## Steps
1. Add OptimizationStrategy struct to vector_store.rs:
   ```rust
   /// Optimization strategy definition
   #[derive(Debug, Clone)]
   pub struct OptimizationStrategy {
       /// Strategy name
       pub name: String,
       /// Strategy type
       pub strategy_type: StrategyType,
       /// Conditions for applying this strategy
       pub conditions: Vec<OptimizationCondition>,
       /// Actions to take
       pub actions: Vec<OptimizationAction>,
       /// Expected improvement
       pub expected_improvement: f64,
   }
   ```

## Success Criteria
- [ ] OptimizationStrategy struct added with all required fields
- [ ] Proper field types including Vec for conditions and actions
- [ ] Compiles without errors

## Time: 3 minutes