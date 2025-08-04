//! Vector search system for the neuromorphic neural network
//! 
//! Provides full-text search capabilities with AST-based smart chunking
//! for integration with cortical column allocation and neural processing.

pub mod indexing;
pub mod search;
pub mod chunking;
pub mod boolean_logic;

#[cfg(test)]
pub mod test_utils;

pub use indexing::{DocumentIndexer, IndexingError};
pub use search::{SearchEngine, SearchError, SearchResult};
pub use chunking::{SmartChunker, ChunkingError, CodeChunk};
pub use boolean_logic::{BooleanSearchEngine, BooleanQuery, DocumentResult};

/// Vector search system errors
#[derive(thiserror::Error, Debug)]
pub enum VectorSearchError {
    #[error("Indexing error: {0}")]
    Indexing(#[from] IndexingError),
    
    #[error("Search error: {0}")]
    Search(#[from] SearchError),
    
    #[error("Chunking error: {0}")]
    Chunking(#[from] ChunkingError),
}

pub type Result<T> = std::result::Result<T, VectorSearchError>;