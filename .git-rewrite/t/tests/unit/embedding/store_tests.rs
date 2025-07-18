//! Embedding Store Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::embedding::store::EmbeddingStore;

#[cfg(test)]
mod store_tests {
    use super::*;

    #[test]
    fn test_embedding_store_operations() {
        let dimension = 128;
        let mut store = EmbeddingStore::new(dimension);
        
        // Test adding embeddings
        let embedding1 = vec![0.1; dimension];
        let embedding2 = vec![0.2; dimension];
        
        let id1 = store.add_embedding("entity1", embedding1.clone()).unwrap();
        let id2 = store.add_embedding("entity2", embedding2.clone()).unwrap();
        
        assert_ne!(id1, id2);
        
        // Test retrieval
        let retrieved1 = store.get_embedding(id1).unwrap();
        assert_eq!(retrieved1, &embedding1);
        
        // Test similarity search
        let query = vec![0.15; dimension];
        let results = store.find_similar(&query, 5, 0.5);
        assert!(!results.is_empty());
    }
}