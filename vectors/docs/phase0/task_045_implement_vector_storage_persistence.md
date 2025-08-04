# Micro-Task 045a: Add Storage Persistence Structure

## Objective
Add basic persistence capabilities to save and load vector storage state.

## Prerequisites
- Task 044e completed (advanced vector operations tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add persistence structures to `src/mock_storage.rs`:
   ```rust
   use std::path::{Path, PathBuf};
   use std::fs;
   use std::io::{Read, Write};
   
   /// Serializable storage state for persistence
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct StorageSnapshot {
       pub version: String,
       pub dimension: usize,
       pub records: Vec<VectorRecord>,
       pub index_config: Option<IndexConfig>,
       pub created_at: String,
   }
   
   impl StorageSnapshot {
       pub fn new(dimension: usize, records: Vec<VectorRecord>) -> Self {
           Self {
               version: "1.0.0".to_string(),
               dimension,
               records,
               index_config: None,
               created_at: chrono::Utc::now().to_rfc3339(),
           }
       }
       
       pub fn with_index_config(mut self, config: IndexConfig) -> Self {
           self.index_config = Some(config);
           self
       }
   }
   
   /// Persistence configuration
   #[derive(Debug, Clone)]
   pub struct PersistenceConfig {
       pub storage_path: PathBuf,
       pub auto_save_threshold: usize, // Save after N operations
       pub compression_enabled: bool,
   }
   
   impl Default for PersistenceConfig {
       fn default() -> Self {
           Self {
               storage_path: PathBuf::from("vector_storage.json"),
               auto_save_threshold: 100,
               compression_enabled: false,
           }
       }
   }
   ```
3. Add chrono dependency to `Cargo.toml`:
   ```toml
   [dependencies]
   chrono = { version = "0.4", features = ["serde"] }
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] StorageSnapshot structure for serializable state
- [ ] PersistenceConfig with file path and options
- [ ] Version tracking and timestamp support
- [ ] Chrono dependency added for timestamps

## Next Task
task_045b_implement_save_functionality.md

---

# Micro-Task 045b: Implement Save Functionality

## Objective
Implement methods to save vector storage state to disk.

## Prerequisites
- Task 045a completed (storage persistence structure added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add save functionality to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Save storage state to file
       pub fn save_to_file<P: AsRef<Path>>(&self, file_path: P) -> StorageResult<()> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let records_vec: Vec<VectorRecord> = records.values().cloned().collect();
           let snapshot = StorageSnapshot::new(self.dimension, records_vec);
           
           let json_data = serde_json::to_string_pretty(&snapshot)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           // Ensure parent directory exists
           if let Some(parent) = file_path.as_ref().parent() {
               fs::create_dir_all(parent)
                   .map_err(|e| StorageError::InternalError(format!("Failed to create directory: {}", e)))?;
           }
           
           fs::write(file_path, json_data)
               .map_err(|e| StorageError::InternalError(format!("Failed to write file: {}", e)))?;
           
           Ok(())
       }
       
       /// Save with index configuration
       pub fn save_with_index<P: AsRef<Path>>(&self, file_path: P, index_config: &IndexConfig) -> StorageResult<()> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let records_vec: Vec<VectorRecord> = records.values().cloned().collect();
           let snapshot = StorageSnapshot::new(self.dimension, records_vec)
               .with_index_config(index_config.clone());
           
           let json_data = serde_json::to_string_pretty(&snapshot)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           if let Some(parent) = file_path.as_ref().parent() {
               fs::create_dir_all(parent)
                   .map_err(|e| StorageError::InternalError(format!("Failed to create directory: {}", e)))?;
           }
           
           fs::write(file_path, json_data)
               .map_err(|e| StorageError::InternalError(format!("Failed to write file: {}", e)))?;
           
           Ok(())
       }
       
       /// Save storage state as binary (more compact)
       pub fn save_to_binary<P: AsRef<Path>>(&self, file_path: P) -> StorageResult<()> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let records_vec: Vec<VectorRecord> = records.values().cloned().collect();
           let snapshot = StorageSnapshot::new(self.dimension, records_vec);
           
           let binary_data = bincode::serialize(&snapshot)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           if let Some(parent) = file_path.as_ref().parent() {
               fs::create_dir_all(parent)
                   .map_err(|e| StorageError::InternalError(format!("Failed to create directory: {}", e)))?;
           }
           
           fs::write(file_path, binary_data)
               .map_err(|e| StorageError::InternalError(format!("Failed to write file: {}", e)))?;
           
           Ok(())
       }
       
       /// Get storage statistics for saving decision
       pub fn should_auto_save(&self, threshold: usize) -> bool {
           if let Ok(records) = self.records.lock() {
               records.len() >= threshold
           } else {
               false
           }
       }
   }
   ```
3. Add bincode dependency to `Cargo.toml`:
   ```toml
   [dependencies]
   bincode = "1.3"
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] JSON save functionality with pretty formatting
- [ ] Binary save functionality for compact storage
- [ ] Index configuration preservation
- [ ] Auto-save threshold checking

## Next Task
task_045c_implement_load_functionality.md

---

# Micro-Task 045c: Implement Load Functionality

## Objective
Implement methods to load vector storage state from disk.

## Prerequisites
- Task 045b completed (save functionality implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add load functionality to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Load storage state from JSON file
       pub fn load_from_file<P: AsRef<Path>>(file_path: P) -> StorageResult<Self> {
           let json_data = fs::read_to_string(file_path)
               .map_err(|e| StorageError::InternalError(format!("Failed to read file: {}", e)))?;
           
           let snapshot: StorageSnapshot = serde_json::from_str(&json_data)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           let storage = Self::new(snapshot.dimension);
           
           // Load all records
           for record in snapshot.records {
               storage.insert(record)?;
           }
           
           // Rebuild index if configuration was saved
           if let Some(index_config) = snapshot.index_config {
               storage.build_index(&index_config)?;
           }
           
           Ok(storage)
       }
       
       /// Load storage state from binary file
       pub fn load_from_binary<P: AsRef<Path>>(file_path: P) -> StorageResult<Self> {
           let binary_data = fs::read(file_path)
               .map_err(|e| StorageError::InternalError(format!("Failed to read file: {}", e)))?;
           
           let snapshot: StorageSnapshot = bincode::deserialize(&binary_data)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           let storage = Self::new(snapshot.dimension);
           
           for record in snapshot.records {
               storage.insert(record)?;
           }
           
           if let Some(index_config) = snapshot.index_config {
               storage.build_index(&index_config)?;
           }
           
           Ok(storage)
       }
       
       /// Check if storage file exists and is valid
       pub fn validate_storage_file<P: AsRef<Path>>(file_path: P) -> StorageResult<StorageSnapshot> {
           if !file_path.as_ref().exists() {
               return Err(StorageError::InternalError("Storage file does not exist".to_string()));
           }
           
           let json_data = fs::read_to_string(file_path)
               .map_err(|e| StorageError::InternalError(format!("Failed to read file: {}", e)))?;
           
           let snapshot: StorageSnapshot = serde_json::from_str(&json_data)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           // Basic validation
           if snapshot.records.is_empty() {
               return Err(StorageError::StorageCorrupted("No records found in storage".to_string()));
           }
           
           // Check dimension consistency
           for record in &snapshot.records {
               if record.embedding.len() != snapshot.dimension {
                   return Err(StorageError::StorageCorrupted(
                       format!("Record {} has incorrect embedding dimension", record.id)
                   ));
               }
           }
           
           Ok(snapshot)
       }
       
       /// Merge another storage into this one
       pub fn merge_from_file<P: AsRef<Path>>(&self, file_path: P) -> StorageResult<usize> {
           let json_data = fs::read_to_string(file_path)
               .map_err(|e| StorageError::InternalError(format!("Failed to read file: {}", e)))?;
           
           let snapshot: StorageSnapshot = serde_json::from_str(&json_data)
               .map_err(|e| StorageError::SerializationError(e.to_string()))?;
           
           if snapshot.dimension != self.dimension {
               return Err(StorageError::InvalidEmbedding(
                   format!("Dimension mismatch: storage has {}, file has {}", 
                           self.dimension, snapshot.dimension)
               ));
           }
           
           let mut merged_count = 0;
           let mut records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           for record in snapshot.records {
               if !records.contains_key(&record.id) {
                   records.insert(record.id, record);
                   merged_count += 1;
               }
           }
           
           Ok(merged_count)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] JSON and binary load functionality
- [ ] Index reconstruction after loading
- [ ] Storage file validation with corruption detection
- [ ] Merge functionality for combining storages

## Next Task
task_045d_add_auto_persistence_wrapper.md

---

# Micro-Task 045d: Add Auto-Persistence Wrapper

## Objective
Create a wrapper that automatically saves storage changes at regular intervals.

## Prerequisites
- Task 045c completed (load functionality implemented)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add auto-persistence wrapper to `src/mock_storage.rs`:
   ```rust
   /// Auto-persisting wrapper for MockVectorStorage
   pub struct PersistentVectorStorage {
       storage: MockVectorStorage,
       config: PersistenceConfig,
       operation_count: Arc<Mutex<usize>>,
   }
   
   impl PersistentVectorStorage {
       /// Create new persistent storage
       pub fn new(dimension: usize, config: PersistenceConfig) -> Self {
           Self {
               storage: MockVectorStorage::new(dimension),
               config,
               operation_count: Arc::new(Mutex::new(0)),
           }
       }
       
       /// Load from existing file
       pub fn load_from_config(config: PersistenceConfig) -> StorageResult<Self> {
           let storage = if config.storage_path.exists() {
               MockVectorStorage::load_from_file(&config.storage_path)?
           } else {
               MockVectorStorage::new(384) // Default dimension
           };
           
           Ok(Self {
               storage,
               config,
               operation_count: Arc::new(Mutex::new(0)),
           })
       }
       
       /// Insert with auto-save
       pub fn insert(&self, record: VectorRecord) -> StorageResult<()> {
           self.storage.insert(record)?;
           self.increment_operation_count()?;
           Ok(())
       }
       
       /// Batch insert with auto-save
       pub fn batch_insert(&self, records: Vec<VectorRecord>) -> StorageResult<usize> {
           let count = self.storage.batch_insert(records)?;
           self.increment_operation_count_by(count)?;
           Ok(count)
       }
       
       /// Delete with auto-save
       pub fn delete(&self, id: &Uuid) -> StorageResult<bool> {
           let deleted = self.storage.delete(id)?;
           if deleted {
               self.increment_operation_count()?;
           }
           Ok(deleted)
       }
       
       /// Increment operation count and auto-save if needed
       fn increment_operation_count(&self) -> StorageResult<()> {
           self.increment_operation_count_by(1)
       }
       
       fn increment_operation_count_by(&self, count: usize) -> StorageResult<()> {
           let mut op_count = self.operation_count.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           *op_count += count;
           
           if *op_count >= self.config.auto_save_threshold {
               drop(op_count); // Release lock before saving
               self.force_save()?;
               
               // Reset counter
               let mut op_count = self.operation_count.lock()
                   .map_err(|e| StorageError::InternalError(e.to_string()))?;
               *op_count = 0;
           }
           
           Ok(())
       }
       
       /// Force save regardless of threshold
       pub fn force_save(&self) -> StorageResult<()> {
           if self.config.compression_enabled {
               self.storage.save_to_binary(&self.config.storage_path)
           } else {
               self.storage.save_to_file(&self.config.storage_path)
           }
       }
       
       /// Delegate read operations to underlying storage
       pub fn get(&self, id: &Uuid) -> StorageResult<Option<VectorRecord>> {
           self.storage.get(id)
       }
       
       pub fn similarity_search(&self, query_embedding: &[f32], config: &SearchConfig) -> StorageResult<Vec<SimilarityResult>> {
           self.storage.similarity_search(query_embedding, config)
       }
       
       pub fn search_by_content(&self, query_content: &str, config: &SearchConfig) -> StorageResult<Vec<SimilarityResult>> {
           self.storage.search_by_content(query_content, config)
       }
       
       pub fn filtered_similarity_search(&self, query_embedding: &[f32], filter: &VectorSearchFilter) -> StorageResult<Vec<SimilarityResult>> {
           self.storage.filtered_similarity_search(query_embedding, filter)
       }
       
       pub fn count(&self) -> StorageResult<usize> {
           self.storage.count()
       }
       
       pub fn build_index(&self, config: &IndexConfig) -> StorageResult<()> {
           self.storage.build_index(config)
       }
       
       pub fn has_index(&self) -> bool {
           self.storage.has_index()
       }
       
       /// Get operation count since last save
       pub fn operations_since_save(&self) -> usize {
           self.operation_count.lock().map(|count| *count).unwrap_or(0)
       }
   }
   
   /// Automatic cleanup on drop
   impl Drop for PersistentVectorStorage {
       fn drop(&mut self) {
           // Try to save on drop, but don't panic if it fails
           let _ = self.force_save();
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] PersistentVectorStorage wrapper with auto-save
- [ ] Operation counting and threshold-based saving
- [ ] Force save and automatic cleanup on drop
- [ ] Delegation of read operations to underlying storage

## Next Task
task_045e_add_persistence_tests_and_commit.md

---

# Micro-Task 045e: Add Persistence Tests and Commit

## Objective
Write comprehensive tests for storage persistence and commit the implementation.

## Prerequisites
- Task 045d completed (auto-persistence wrapper added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add persistence tests to `src/mock_storage.rs`:
   ```rust
   #[cfg(test)]
   mod persistence_tests {
       use super::*;
       use tempfile::TempDir;
       
       #[test]
       fn test_save_and_load_json() {
           let temp_dir = TempDir::new().unwrap();
           let file_path = temp_dir.path().join("test_storage.json");
           
           // Create storage and add records
           let storage = MockVectorStorage::new(384);
           
           for i in 0..3 {
               let record = VectorRecord::new(
                   format!("Test content {}", i),
                   format!("Test title {}", i),
                   Some(format!("test{}.txt", i)),
                   Some("txt".to_string()),
               );
               storage.insert(record).unwrap();
           }
           
           let original_count = storage.count().unwrap();
           
           // Save to file
           storage.save_to_file(&file_path).unwrap();
           assert!(file_path.exists());
           
           // Load from file
           let loaded_storage = MockVectorStorage::load_from_file(&file_path).unwrap();
           
           assert_eq!(loaded_storage.count().unwrap(), original_count);
           assert_eq!(loaded_storage.dimension, 384);
       }
       
       #[test]
       fn test_save_and_load_binary() {
           let temp_dir = TempDir::new().unwrap();
           let file_path = temp_dir.path().join("test_storage.bin");
           
           let storage = MockVectorStorage::new(384);
           
           let record = VectorRecord::new(
               "Binary test content".to_string(),
               "Binary test".to_string(),
               None,
               None,
           );
           
           storage.insert(record.clone()).unwrap();
           
           // Save as binary
           storage.save_to_binary(&file_path).unwrap();
           assert!(file_path.exists());
           
           // Load from binary
           let loaded_storage = MockVectorStorage::load_from_binary(&file_path).unwrap();
           
           assert_eq!(loaded_storage.count().unwrap(), 1);
           
           let loaded_record = loaded_storage.get(&record.id).unwrap().unwrap();
           assert_eq!(loaded_record.content, record.content);
           assert_eq!(loaded_record.title, record.title);
       }
       
       #[test]
       fn test_save_and_load_with_index() {
           let temp_dir = TempDir::new().unwrap();
           let file_path = temp_dir.path().join("test_storage_indexed.json");
           
           let storage = MockVectorStorage::new(384);
           
           // Add multiple records
           for i in 0..10 {
               let record = VectorRecord::new(
                   format!("Indexed content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           let index_config = IndexConfig::default();
           storage.build_index(&index_config).unwrap();
           
           // Save with index config
           storage.save_with_index(&file_path, &index_config).unwrap();
           
           // Load and verify index is rebuilt
           let loaded_storage = MockVectorStorage::load_from_file(&file_path).unwrap();
           
           assert_eq!(loaded_storage.count().unwrap(), 10);
           assert!(loaded_storage.has_index());
       }
       
       #[test]
       fn test_storage_validation() {
           let temp_dir = TempDir::new().unwrap();
           let file_path = temp_dir.path().join("test_validation.json");
           
           let storage = MockVectorStorage::new(384);
           
           let record = VectorRecord::new(
               "Validation test".to_string(),
               "Validation".to_string(),
               None,
               None,
           );
           
           storage.insert(record).unwrap();
           storage.save_to_file(&file_path).unwrap();
           
           // Validate storage file
           let snapshot = MockVectorStorage::validate_storage_file(&file_path).unwrap();
           
           assert_eq!(snapshot.version, "1.0.0");
           assert_eq!(snapshot.dimension, 384);
           assert_eq!(snapshot.records.len(), 1);
           
           // Test non-existent file
           let bad_path = temp_dir.path().join("nonexistent.json");
           assert!(MockVectorStorage::validate_storage_file(&bad_path).is_err());
       }
       
       #[test]
       fn test_persistent_storage_auto_save() {
           let temp_dir = TempDir::new().unwrap();
           let file_path = temp_dir.path().join("auto_save.json");
           
           let config = PersistenceConfig {
               storage_path: file_path.clone(),
               auto_save_threshold: 3, // Save after 3 operations
               compression_enabled: false,
           };
           
           let persistent = PersistentVectorStorage::new(384, config);
           
           // Add records - should trigger auto-save at 3rd record
           for i in 0..5 {
               let record = VectorRecord::new(
                   format!("Auto save content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               persistent.insert(record).unwrap();
           }
           
           // File should exist due to auto-save
           assert!(file_path.exists());
           
           // Load and verify
           let loaded = MockVectorStorage::load_from_file(&file_path).unwrap();
           
           // Should have at least 3 records (auto-saved after threshold)
           assert!(loaded.count().unwrap() >= 3);
       }
       
       #[test]
       fn test_storage_merge() {
           let temp_dir = TempDir::new().unwrap();
           let file1_path = temp_dir.path().join("storage1.json");
           let file2_path = temp_dir.path().join("storage2.json");
           
           // Create first storage
           let storage1 = MockVectorStorage::new(384);
           for i in 0..3 {
               let record = VectorRecord::new(
                   format!("Storage1 content {}", i),
                   format!("S1 Title {}", i),
                   None,
                   None,
               );
               storage1.insert(record).unwrap();
           }
           storage1.save_to_file(&file1_path).unwrap();
           
           // Create second storage
           let storage2 = MockVectorStorage::new(384);
           for i in 0..2 {
               let record = VectorRecord::new(
                   format!("Storage2 content {}", i),
                   format!("S2 Title {}", i),
                   None,
                   None,
               );
               storage2.insert(record).unwrap();
           }
           storage2.save_to_file(&file2_path).unwrap();
           
           // Merge storage2 into storage1
           let merged_count = storage1.merge_from_file(&file2_path).unwrap();
           
           assert_eq!(merged_count, 2);
           assert_eq!(storage1.count().unwrap(), 5); // 3 + 2
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Complete vector storage mock with persistence and auto-save functionality"`

## Success Criteria
- [ ] JSON save/load tests implemented and passing
- [ ] Binary save/load tests implemented and passing
- [ ] Index preservation tests implemented and passing
- [ ] Storage validation tests implemented and passing
- [ ] Auto-save and merge tests implemented and passing
- [ ] Complete vector storage mock committed to Git

## Next Task
task_046_validate_similarity_search_functionality.md