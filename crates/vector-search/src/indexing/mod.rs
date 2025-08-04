//! Document indexing capabilities for vector search

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Tantivy error: {0}")]
    Tantivy(String),
}

/// Document indexer for creating searchable indices
pub struct DocumentIndexer {
    // Implementation in Task 00_4
}

impl DocumentIndexer {
    pub fn new() -> Result<Self, IndexingError> {
        todo!("Implementation in Task 00_4")
    }
}