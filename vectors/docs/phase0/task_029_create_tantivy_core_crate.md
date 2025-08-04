# Micro-Task 029: Create Tantivy Core Crate

## Objective
Create the tantivy-core crate with basic structure and dependencies.

## Context
This begins the Architecture Validation phase. The tantivy-core crate will contain the core text indexing and search functionality. This task creates the crate structure without implementation details.

## Prerequisites
- Task 028 completed (Environment setup finalized)
- Cargo workspace configured
- Ready for architecture validation

## Time Estimate
7 minutes

## Instructions
1. Create crate directory: `mkdir crates\tantivy-core`
2. Navigate to crate: `cd crates\tantivy-core`
3. Create `Cargo.toml`:
   ```toml
   [package]
   name = "tantivy-core"
   version = "0.1.0"
   edition = "2021"
   description = "Core text indexing and search functionality"
   
   [dependencies]
   tantivy = { workspace = true }
   tokio = { workspace = true }
   serde = { workspace = true }
   anyhow = { workspace = true }
   thiserror = { workspace = true }
   ```
4. Create basic source structure:
   - `mkdir src`
   - Create `src/lib.rs`:
     ```rust
     //! Tantivy core text indexing and search functionality
     
     pub mod indexer;
     pub mod searcher;
     pub mod schema;
     
     // Re-exports for convenience
     pub use indexer::*;
     pub use searcher::*; 
     pub use schema::*;
     
     #[cfg(test)]
     mod tests {
         #[test]
         fn test_crate_compiles() {
             // Basic compilation test
             assert!(true);
         }
     }
     ```
5. Create placeholder modules:
   - `echo // TODO: Indexer implementation > src\indexer.rs`
   - `echo // TODO: Searcher implementation > src\searcher.rs`
   - `echo // TODO: Schema implementation > src\schema.rs`
6. Test crate: `cargo check`
7. Return to root: `cd ..\..`
8. Commit: `git add crates\tantivy-core && git commit -m "Create tantivy-core crate structure"`

## Expected Output
- tantivy-core crate created with proper structure
- Basic library structure in place
- Crate compiles without errors
- Crate committed to repository

## Success Criteria
- [ ] Crate directory and Cargo.toml created
- [ ] Basic lib.rs with module structure
- [ ] Placeholder modules created
- [ ] `cargo check` passes in crate directory
- [ ] Crate structure committed to Git

## Next Task
task_030_create_lancedb_integration_crate.md