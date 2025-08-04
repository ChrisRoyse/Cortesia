# Task 131: Add OptimalSearchMode Enum

## Prerequisites Check
- [ ] Task 130 completed: QueryPattern struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the OptimalSearchMode enum to represent different search mode recommendations.

## Task Objective
Define the OptimalSearchMode enum to specify which search mode performs best for specific query patterns.

## Steps
1. Add OptimalSearchMode enum to vector_store.rs:
   ```rust
   /// Optimal search mode recommendation
   #[derive(Debug, Clone, PartialEq)]
   pub enum OptimalSearchMode {
       /// Text search performs best
       TextOptimal,
       /// Vector search performs best
       VectorOptimal,
       /// Hybrid search performs best
       HybridOptimal,
       /// Adaptive mode recommended
       AdaptiveOptimal,
       /// Insufficient data
       Unknown,
   }
   ```

## Success Criteria
- [ ] OptimalSearchMode enum added with all variants
- [ ] Proper derives (Debug, Clone, PartialEq)
- [ ] Compiles without errors

## Time: 2 minutes