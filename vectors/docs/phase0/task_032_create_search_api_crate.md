# Micro-Task 032: Create Search API Crate

## Objective
Create the search-api crate that provides the public interface for search operations.

## Context
The search-api crate serves as the main entry point for external consumers. It provides a clean, high-level API that abstracts the complexity of hybrid text and vector search operations.

## Prerequisites
- Task 031 completed (vector-indexing crate created)
- All core crates (tantivy-core, lancedb-integration, vector-indexing) exist
- Understanding of API design requirements

## Time Estimate
9 minutes

## Instructions
1. Create crate directory: `mkdir crates\search-api`
2. Navigate to crate: `cd crates\search-api`
3. Create `Cargo.toml`:
   ```toml
   [package]
   name = "search-api"
   version = "0.1.0"
   edition = "2021"
   description = "Public API for hybrid text and vector search"
   
   [dependencies]
   tantivy-core = { path = "../tantivy-core" }
   lancedb-integration = { path = "../lancedb-integration" }
   vector-indexing = { path = "../vector-indexing" }
   tokio = { workspace = true }
   serde = { workspace = true }
   serde_json = { workspace = true }
   anyhow = { workspace = true }
   thiserror = { workspace = true }
   uuid = { workspace = true }
   ```
4. Create source structure:
   - `mkdir src`
   - Create `src/lib.rs`:
     ```rust
     //! Public API for hybrid text and vector search
     
     pub mod search_client;
     pub mod query_builder;
     pub mod search_results;
     pub mod config;
     
     // Re-exports for public API
     pub use search_client::SearchClient;
     pub use query_builder::{QueryBuilder, SearchQuery};
     pub use search_results::{SearchResult, SearchResults};
     pub use config::SearchConfig;
     
     /// Main search API error types
     #[derive(thiserror::Error, Debug)]
     pub enum SearchApiError {
         #[error("Search query failed: {0}")]
         QueryFailed(String),
         #[error("Invalid query format: {0}")]
         InvalidQuery(String),
         #[error("Configuration error: {0}")]
         ConfigurationError(String),
         #[error("Index not available: {0}")]
         IndexNotAvailable(String),
     }
     
     pub type Result<T> = std::result::Result<T, SearchApiError>;
     
     /// Search query parameters
     #[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
     pub struct SearchParams {
         pub text_query: Option<String>,
         pub vector_query: Option<Vec<f32>>,
         pub limit: Option<usize>,
         pub offset: Option<usize>,
         pub filters: std::collections::HashMap<String, String>,
     }
     
     impl Default for SearchParams {
         fn default() -> Self {
             Self {
                 text_query: None,
                 vector_query: None,
                 limit: Some(10),
                 offset: Some(0),
                 filters: std::collections::HashMap::new(),
             }
         }
     }
     
     #[cfg(test)]
     mod tests {
         use super::*;
         
         #[test]
         fn test_search_params_default() {
             let params = SearchParams::default();
             assert_eq!(params.limit, Some(10));
             assert_eq!(params.offset, Some(0));
         }
         
         #[test]
         fn test_error_types() {
             let error = SearchApiError::QueryFailed("test".to_string());
             assert!(error.to_string().contains("Search query failed"));
         }
     }
     ```
5. Create placeholder modules:
   - `echo // TODO: Search client implementation > src\search_client.rs`
   - `echo // TODO: Query builder implementation > src\query_builder.rs`
   - `echo // TODO: Search results implementation > src\search_results.rs`
   - `echo // TODO: Configuration implementation > src\config.rs`
6. Test crate: `cargo check`
7. Return to root: `cd ..\..`
8. Update workspace `Cargo.toml` to include all member crates:
   ```toml
   [workspace]
   members = [
       "crates/tantivy-core",
       "crates/lancedb-integration", 
       "crates/vector-indexing",
       "crates/search-api"
   ]
   resolver = "2"
   ```
9. Test entire workspace: `cargo check --workspace`
10. Commit: `git add crates\search-api Cargo.toml && git commit -m "Create search-api crate and update workspace"`

## Expected Output
- search-api crate created as public interface
- SearchParams and error types defined
- Workspace updated to include all 4 core crates
- Entire workspace compiles successfully

## Success Criteria
- [ ] search-api crate created with dependencies on all other crates
- [ ] SearchParams struct and SearchApiError enum defined
- [ ] Workspace Cargo.toml updated with all 4 member crates
- [ ] `cargo check --workspace` passes successfully
- [ ] Complete crate structure committed to Git

## Next Task
task_033_validate_crate_dependencies.md