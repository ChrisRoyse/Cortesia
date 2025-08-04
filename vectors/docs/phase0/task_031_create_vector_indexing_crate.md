# Micro-Task 031: Create Vector Indexing Crate

## Objective
Create the vector-indexing crate that bridges text and vector search capabilities.

## Context
The vector-indexing crate combines text indexing from Tantivy with vector operations from LanceDB. It provides the unified interface for hybrid search operations and indexing coordination.

## Prerequisites
- Task 030 completed (lancedb-integration crate created)
- Both tantivy-core and lancedb-integration crates exist
- Understanding of hybrid search architecture

## Time Estimate
9 minutes

## Instructions
1. Create crate directory: `mkdir crates\vector-indexing`
2. Navigate to crate: `cd crates\vector-indexing`
3. Create `Cargo.toml`:
   ```toml
   [package]
   name = "vector-indexing"
   version = "0.1.0"
   edition = "2021"
   description = "Hybrid text and vector indexing coordination"
   
   [dependencies]
   tantivy-core = { path = "../tantivy-core" }
   lancedb-integration = { path = "../lancedb-integration" }
   tokio = { workspace = true }
   serde = { workspace = true }
   anyhow = { workspace = true }
   thiserror = { workspace = true }
   uuid = { workspace = true }
   rayon = { workspace = true }
   ```
4. Create source structure:
   - `mkdir src`
   - Create `src/lib.rs`:
     ```rust
     //! Hybrid text and vector indexing coordination
     
     pub mod hybrid_index;
     pub mod document_processor;
     pub mod embedding_generator;
     pub mod index_coordinator;
     
     // Re-exports
     pub use hybrid_index::*;
     pub use document_processor::*;
     pub use embedding_generator::*;
     pub use index_coordinator::*;
     
     /// Indexing error types
     #[derive(thiserror::Error, Debug)]
     pub enum IndexingError {
         #[error("Text indexing failed: {0}")]
         TextIndexingFailed(String),
         #[error("Vector indexing failed: {0}")]
         VectorIndexingFailed(String),
         #[error("Document processing failed: {0}")]
         DocumentProcessingFailed(String),
         #[error("Coordination error: {0}")]
         CoordinationFailed(String),
     }
     
     pub type Result<T> = std::result::Result<T, IndexingError>;
     
     /// Document representation for hybrid indexing
     #[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
     pub struct Document {
         pub id: uuid::Uuid,
         pub content: String,
         pub metadata: std::collections::HashMap<String, String>,
         pub embedding: Option<Vec<f32>>,
     }
     
     #[cfg(test)]
     mod tests {
         use super::*;
         
         #[test]
         fn test_document_creation() {
             let doc = Document {
                 id: uuid::Uuid::new_v4(),
                 content: "test content".to_string(),
                 metadata: std::collections::HashMap::new(),
                 embedding: None,
             };
             assert_eq!(doc.content, "test content");
         }
     }
     ```
5. Create placeholder modules:
   - `echo // TODO: Hybrid index implementation > src\hybrid_index.rs`
   - `echo // TODO: Document processor implementation > src\document_processor.rs`
   - `echo // TODO: Embedding generator implementation > src\embedding_generator.rs`
   - `echo // TODO: Index coordinator implementation > src\index_coordinator.rs`
6. Test crate: `cargo check`
7. Return to root: `cd ..\..`
8. Commit: `git add crates\vector-indexing && git commit -m "Create vector-indexing crate structure"`

## Expected Output
- vector-indexing crate created with dependencies on other crates
- Document type and error handling defined
- Hybrid indexing architecture outlined
- Crate compiles and committed

## Success Criteria
- [ ] Crate created with dependencies on tantivy-core and lancedb-integration
- [ ] Document struct and IndexingError enum defined
- [ ] Placeholder modules for hybrid functionality
- [ ] `cargo check` passes with inter-crate dependencies
- [ ] Crate structure committed to Git

## Next Task
task_032_create_search_api_crate.md