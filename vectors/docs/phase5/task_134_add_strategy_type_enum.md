# Task 134: Add StrategyType Enum

## Prerequisites Check
- [ ] Task 133 completed: OptimizationStrategy struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the StrategyType enum to categorize different types of optimization strategies.

## Task Objective
Define the StrategyType enum to classify optimization strategies by their approach and focus area.

## Steps
1. Add StrategyType enum to vector_store.rs:
   ```rust
   /// Types of optimization strategies
   #[derive(Debug, Clone, PartialEq)]
   pub enum StrategyType {
       /// Query rewriting
       QueryRewriting,
       /// Search mode selection
       ModeSelection,
       /// Parameter tuning
       ParameterTuning,
       /// Caching strategy
       CachingStrategy,
       /// Index optimization
       IndexOptimization,
   }
   ```

## Success Criteria
- [ ] StrategyType enum added with all strategy variants
- [ ] Proper derives (Debug, Clone, PartialEq)
- [ ] Compiles without errors

## Time: 2 minutes