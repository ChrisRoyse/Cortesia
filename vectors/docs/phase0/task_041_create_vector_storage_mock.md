# Micro-Task 041a: Create Vector Storage Crate Structure

## Objective
Create the lancedb-integration crate structure with mock vector storage functionality.

## Prerequisites
- Task 040d completed (searcher implementation with ranking committed)
- lancedb-integration crate exists from task 030

## Time Estimate
6 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Create `src/mock_storage.rs` with basic structure:
   ```rust
   //! Mock vector storage for testing and development
   
   use std::collections::HashMap;
   use std::sync::{Arc, Mutex};
   use uuid::Uuid;
   ```
3. Add mock storage module to `src/lib.rs`:
   ```rust
   pub mod mock_storage;
   
   pub use mock_storage::{MockVectorStorage, VectorRecord, StorageError};
   ```
4. Add dependencies to `Cargo.toml`:
   ```toml
   [dependencies]
   uuid = { workspace = true }
   serde = { workspace = true }
   ```
5. Test compilation: `cargo check`
6. Return to root: `cd ..\..`

## Success Criteria
- [ ] mock_storage.rs file created with basic imports
- [ ] Module added to lib.rs
- [ ] Dependencies added to Cargo.toml
- [ ] Crate compiles successfully

## Next Task
task_041b_define_vector_record_struct.md

---

# Micro-Task 041b: Define VectorRecord Struct

## Objective
Define the VectorRecord struct to represent documents with vector embeddings.

## Prerequisites
- Task 041a completed (vector storage crate structure created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add to `src/mock_storage.rs`:
   ```rust
   use serde::{Deserialize, Serialize};
   
   /// A document record with vector embedding
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct VectorRecord {
       pub id: Uuid,
       pub content: String,
       pub title: String,
       pub file_path: Option<String>,
       pub extension: Option<String>,
       pub embedding: Vec<f32>,
       pub metadata: HashMap<String, String>,
   }
   
   impl VectorRecord {
       /// Create new vector record with generated embedding
       pub fn new(
           content: String,
           title: String,
           file_path: Option<String>,
           extension: Option<String>,
       ) -> Self {
           Self {
               id: Uuid::new_v4(),
               content,
               title,
               file_path,
               extension,
               embedding: Self::generate_mock_embedding(&content),
               metadata: HashMap::new(),
           }
       }
       
       /// Generate a mock embedding based on content hash
       pub fn generate_mock_embedding(content: &str) -> Vec<f32> {
           // Simple mock: use content characters to generate deterministic embedding
           let mut embedding = Vec::with_capacity(384); // Common embedding size
           let bytes = content.as_bytes();
           
           for i in 0..384 {
               let byte_idx = i % bytes.len();
               let normalized = (bytes[byte_idx] as f32) / 255.0;
               embedding.push(normalized);
           }
           
           embedding
       }
       
       pub fn with_metadata(mut self, key: String, value: String) -> Self {
           self.metadata.insert(key, value);
           self
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] VectorRecord struct defined with all required fields
- [ ] Mock embedding generation implemented
- [ ] Metadata support added
- [ ] Code compiles successfully

## Next Task
task_041c_create_storage_error_types.md

---

# Micro-Task 041c: Create Storage Error Types

## Objective
Define error types for vector storage operations.

## Prerequisites
- Task 041b completed (VectorRecord struct defined)

## Time Estimate
7 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add error types to `src/mock_storage.rs`:
   ```rust
   use std::fmt;
   
   /// Storage operation errors
   #[derive(Debug, Clone)]
   pub enum StorageError {
       RecordNotFound(Uuid),
       InvalidEmbedding(String),
       SerializationError(String),
       StorageCorrupted(String),
       InternalError(String),
   }
   
   impl fmt::Display for StorageError {
       fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
           match self {
               StorageError::RecordNotFound(id) => write!(f, "Record not found: {}", id),
               StorageError::InvalidEmbedding(msg) => write!(f, "Invalid embedding: {}", msg),
               StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
               StorageError::StorageCorrupted(msg) => write!(f, "Storage corrupted: {}", msg),
               StorageError::InternalError(msg) => write!(f, "Internal error: {}", msg),
           }
       }
   }
   
   impl std::error::Error for StorageError {}
   
   /// Result type for storage operations
   pub type StorageResult<T> = Result<T, StorageError>;
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] StorageError enum defined with common error cases
- [ ] Error trait implementations added
- [ ] StorageResult type alias created
- [ ] Error types compile successfully

## Next Task
task_041d_implement_mock_vector_storage.md

---

# Micro-Task 041d: Implement MockVectorStorage

## Objective
Implement the main MockVectorStorage struct with basic CRUD operations.

## Prerequisites
- Task 041c completed (storage error types created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add MockVectorStorage implementation to `src/mock_storage.rs`:
   ```rust
   /// Mock vector storage for testing
   pub struct MockVectorStorage {
       records: Arc<Mutex<HashMap<Uuid, VectorRecord>>>,
       dimension: usize,
   }
   
   impl MockVectorStorage {
       /// Create new mock storage with specified embedding dimension
       pub fn new(dimension: usize) -> Self {
           Self {
               records: Arc::new(Mutex::new(HashMap::new())),
               dimension,
           }
       }
       
       /// Create default storage with 384-dimensional embeddings
       pub fn default() -> Self {
           Self::new(384)
       }
       
       /// Insert a vector record
       pub fn insert(&self, record: VectorRecord) -> StorageResult<()> {
           if record.embedding.len() != self.dimension {
               return Err(StorageError::InvalidEmbedding(
                   format!("Expected {} dimensions, got {}", self.dimension, record.embedding.len())
               ));
           }
           
           let mut records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           records.insert(record.id, record);
           Ok(())
       }
       
       /// Get a record by ID
       pub fn get(&self, id: &Uuid) -> StorageResult<Option<VectorRecord>> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           Ok(records.get(id).cloned())
       }
       
       /// Delete a record by ID
       pub fn delete(&self, id: &Uuid) -> StorageResult<bool> {
           let mut records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           Ok(records.remove(id).is_some())
       }
       
       /// Get total number of records
       pub fn count(&self) -> StorageResult<usize> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           Ok(records.len())
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] MockVectorStorage struct implemented with CRUD operations
- [ ] Thread-safe storage using Arc<Mutex<_>>
- [ ] Embedding dimension validation
- [ ] Code compiles successfully

## Next Task
task_041e_add_storage_tests_and_commit.md

---

# Micro-Task 041e: Add Storage Tests and Commit

## Objective
Write tests for the mock vector storage and commit the implementation.

## Prerequisites
- Task 041d completed (MockVectorStorage implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add tests to `src/mock_storage.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_storage_creation() {
           let storage = MockVectorStorage::new(384);
           assert_eq!(storage.dimension, 384);
           assert_eq!(storage.count().unwrap(), 0);
       }
       
       #[test]
       fn test_record_insertion() {
           let storage = MockVectorStorage::new(384);
           
           let record = VectorRecord::new(
               "Test content".to_string(),
               "Test title".to_string(),
               Some("test.txt".to_string()),
               Some("txt".to_string()),
           );
           
           let record_id = record.id;
           assert!(storage.insert(record).is_ok());
           assert_eq!(storage.count().unwrap(), 1);
           
           let retrieved = storage.get(&record_id).unwrap();
           assert!(retrieved.is_some());
           assert_eq!(retrieved.unwrap().title, "Test title");
       }
       
       #[test]
       fn test_record_deletion() {
           let storage = MockVectorStorage::new(384);
           
           let record = VectorRecord::new(
               "Test content".to_string(),
               "Test title".to_string(),
               None,
               None,
           );
           
           let record_id = record.id;
           storage.insert(record).unwrap();
           
           assert!(storage.delete(&record_id).unwrap());
           assert_eq!(storage.count().unwrap(), 0);
           assert!(storage.get(&record_id).unwrap().is_none());
       }
       
       #[test]
       fn test_embedding_validation() {
           let storage = MockVectorStorage::new(384);
           
           let mut record = VectorRecord::new(
               "Test content".to_string(),
               "Test title".to_string(),
               None,
               None,
           );
           
           // Corrupt the embedding size
           record.embedding = vec![0.0; 100]; // Wrong size
           
           assert!(storage.insert(record).is_err());
       }
       
       #[test]
       fn test_mock_embedding_generation() {
           let content = "Hello, world!";
           let embedding = VectorRecord::generate_mock_embedding(content);
           
           assert_eq!(embedding.len(), 384);
           
           // Should be deterministic
           let embedding2 = VectorRecord::generate_mock_embedding(content);
           assert_eq!(embedding, embedding2);
           
           // Different content should produce different embeddings
           let embedding3 = VectorRecord::generate_mock_embedding("Different content");
           assert_ne!(embedding, embedding3);
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement mock vector storage with CRUD operations and tests"`

## Success Criteria
- [ ] Storage creation tests implemented and passing
- [ ] Record insertion/retrieval tests implemented and passing
- [ ] Record deletion tests implemented and passing
- [ ] Embedding validation tests implemented and passing
- [ ] Mock vector storage committed to Git

## Next Task
task_042_implement_similarity_search_mock.md