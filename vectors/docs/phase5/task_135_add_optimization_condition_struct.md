# Task 135: Add OptimizationCondition Struct

## Prerequisites Check
- [ ] Task 134 completed: StrategyType enum added
- [ ] Run: `cargo check` (should pass)

## Context
Add the OptimizationCondition struct to define when optimization strategies should be applied.

## Task Objective
Define the OptimizationCondition struct to specify conditions that trigger optimization strategies.

## Steps
1. Add OptimizationCondition struct to vector_store.rs:
   ```rust
   /// Condition for applying optimization
   #[derive(Debug, Clone)]
   pub struct OptimizationCondition {
       /// Metric to check
       pub metric: MetricType,
       /// Threshold value
       pub threshold: f64,
       /// Comparison operator
       pub operator: ComparisonOperator,
   }
   ```

## Success Criteria
- [ ] OptimizationCondition struct added with all required fields
- [ ] Proper field types for metric evaluation
- [ ] Compiles without errors

## Time: 3 minutes