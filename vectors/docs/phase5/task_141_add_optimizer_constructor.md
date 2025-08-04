# Task 141: Add QueryOptimizer Constructor

## Prerequisites Check
- [ ] Task 140 completed: QueryOptimizer struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add constructor method for QueryOptimizer to initialize the optimization system.

## Task Objective
Implement the new() method for QueryOptimizer to create instances with default strategies.

## Steps
1. Add constructor implementation for QueryOptimizer in vector_store.rs:
   ```rust
   impl QueryOptimizer {
       /// Create new query optimizer
       pub fn new(performance_monitor: Arc<PerformanceMonitor>, config: OptimizerConfig) -> Self {
           let strategies = Self::create_default_strategies();
           
           Self {
               performance_monitor,
               query_patterns: Arc::new(RwLock::new(HashMap::new())),
               strategies,
               config,
               performance_history: Arc::new(RwLock::new(Vec::new())),
           }
       }
   }
   ```

## Success Criteria
- [ ] Constructor method added with proper initialization
- [ ] All fields properly initialized
- [ ] Compiles without errors

## Time: 3 minutes