//! Storage System Mocks
//! 
//! Mock implementations for storage components including hierarchical storage,
//! indexing systems, and data persistence layers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock hierarchical storage implementation
pub struct MockHierarchicalStorage {
    data: Arc<Mutex<HashMap<String, MockStorageEntry>>>,
    tier_assignments: Arc<Mutex<HashMap<String, StorageTier>>>,
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockHierarchicalStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            tier_assignments: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn store_entry(&self, key: String, entry: MockStorageEntry, tier: StorageTier) {
        self.call_log.lock().unwrap().push(format!("store_entry: {} to {:?}", key, tier));
        self.data.lock().unwrap().insert(key.clone(), entry);
        self.tier_assignments.lock().unwrap().insert(key, tier);
    }
    
    pub fn retrieve_entry(&self, key: &str) -> Option<MockStorageEntry> {
        self.call_log.lock().unwrap().push(format!("retrieve_entry: {}", key));
        self.data.lock().unwrap().get(key).cloned()
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
    
    pub fn clear_call_log(&self) {
        self.call_log.lock().unwrap().clear();
    }
}

/// Mock storage entry for testing
#[derive(Debug, Clone)]
pub struct MockStorageEntry {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub relationships: Vec<String>,
}

/// Storage tier enumeration for testing
#[derive(Debug, Clone, PartialEq)]
pub enum StorageTier {
    Hot,
    Warm,
    Cold,
    Archive,
}

/// Mock index implementation
pub struct MockIndex {
    entries: Arc<Mutex<HashMap<String, Vec<String>>>>,
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockIndex {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn add_entry(&self, key: String, values: Vec<String>) {
        self.call_log.lock().unwrap().push(format!("add_entry: {}", key));
        self.entries.lock().unwrap().insert(key, values);
    }
    
    pub fn search(&self, query: &str) -> Vec<String> {
        self.call_log.lock().unwrap().push(format!("search: {}", query));
        // Mock search implementation
        vec!["mock_result_1".to_string(), "mock_result_2".to_string()]
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Mock semantic store implementation
pub struct MockSemanticStore {
    embeddings: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    similarity_threshold: f32,
    call_log: Arc<Mutex<Vec<String>>>,
}

impl MockSemanticStore {
    pub fn new() -> Self {
        Self {
            embeddings: Arc::new(Mutex::new(HashMap::new())),
            similarity_threshold: 0.8,
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn store_embedding(&self, id: String, embedding: Vec<f32>) {
        self.call_log.lock().unwrap().push(format!("store_embedding: {}", id));
        self.embeddings.lock().unwrap().insert(id, embedding);
    }
    
    pub fn find_similar(&self, query_embedding: &[f32], limit: usize) -> Vec<SimilarityResult> {
        self.call_log.lock().unwrap().push(format!("find_similar: limit {}", limit));
        // Mock similarity search
        vec![
            SimilarityResult { id: "result_1".to_string(), score: 0.95 },
            SimilarityResult { id: "result_2".to_string(), score: 0.87 },
        ]
    }
    
    pub fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().unwrap().clone()
    }
}

/// Similarity search result for testing
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub id: String,
    pub score: f32,
}

/// Helper functions for creating mock storage components
pub fn create_mock_hierarchical_storage() -> MockHierarchicalStorage {
    MockHierarchicalStorage::new()
}

pub fn create_mock_index() -> MockIndex {
    MockIndex::new()
}

pub fn create_mock_semantic_store() -> MockSemanticStore {
    MockSemanticStore::new()
}

/// Test helper for setting up storage mocks with sample data
pub fn setup_storage_mocks_with_sample_data() -> (MockHierarchicalStorage, MockIndex, MockSemanticStore) {
    let storage = create_mock_hierarchical_storage();
    let index = create_mock_index();
    let semantic_store = create_mock_semantic_store();
    
    // Add some sample data
    storage.store_entry(
        "test_doc_1".to_string(),
        MockStorageEntry {
            id: "test_doc_1".to_string(),
            content: "Sample document content".to_string(),
            metadata: HashMap::new(),
            relationships: vec!["related_doc_1".to_string()],
        },
        StorageTier::Hot,
    );
    
    index.add_entry("test_key".to_string(), vec!["value1".to_string(), "value2".to_string()]);
    semantic_store.store_embedding("test_doc_1".to_string(), vec![0.1, 0.2, 0.3, 0.4]);
    
    (storage, index, semantic_store)
}