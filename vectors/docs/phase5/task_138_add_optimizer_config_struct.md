# Task 138: Add OptimizerConfig Struct

## Prerequisites Check
- [ ] Task 137 completed: ActionType enum added
- [ ] Run: `cargo check` (should pass)

## Context
Add the OptimizerConfig struct to configure query optimization behavior and thresholds.

## Task Objective
Define the OptimizerConfig struct to control optimization settings and parameters.

## Steps
1. Add OptimizerConfig struct to vector_store.rs:
   ```rust
   /// Query optimizer configuration
   #[derive(Debug, Clone)]
   pub struct OptimizerConfig {
       /// Enable query optimization
       pub enabled: bool,
       /// Minimum queries before optimization
       pub min_queries_for_optimization: usize,
       /// Performance improvement threshold
       pub improvement_threshold: f64,
       /// Pattern analysis window in seconds
       pub analysis_window: u64,
       /// Auto-apply optimizations
       pub auto_apply: bool,
       /// Maximum optimization history
       pub max_history_size: usize,
   }
   ```

## Success Criteria
- [ ] OptimizerConfig struct added with all configuration fields
- [ ] Proper field types for optimization settings
- [ ] Compiles without errors

## Time: 3 minutes