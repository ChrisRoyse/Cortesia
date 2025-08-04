# Micro-Task 042a: Add Similarity Search Structure

## Objective
Add similarity search functionality to the mock vector storage.

## Prerequisites
- Task 041e completed (mock vector storage tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add similarity search structures to `src/mock_storage.rs`:
   ```rust
   /// Similarity search result with score
   #[derive(Debug, Clone)]
   pub struct SimilarityResult {
       pub record: VectorRecord,
       pub similarity_score: f32,
       pub distance: f32,
   }
   
   impl SimilarityResult {
       pub fn new(record: VectorRecord, similarity_score: f32) -> Self {
           Self {
               record,
               similarity_score,
               distance: 1.0 - similarity_score, // Convert similarity to distance
           }
       }
   }
   
   /// Search configuration for similarity queries
   #[derive(Debug, Clone)]
   pub struct SearchConfig {
       pub limit: usize,
       pub threshold: f32,        // Minimum similarity score
       pub include_metadata: bool,
   }
   
   impl Default for SearchConfig {
       fn default() -> Self {
           Self {
               limit: 10,
               threshold: 0.0,
               include_metadata: true,
           }
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] SimilarityResult struct defined with score and distance
- [ ] SearchConfig struct implemented with sensible defaults
- [ ] Structures compile successfully

## Next Task
task_042b_implement_vector_similarity_functions.md

---

# Micro-Task 042b: Implement Vector Similarity Functions

## Objective
Implement cosine similarity and Euclidean distance calculations for vector comparison.

## Prerequisites
- Task 042a completed (similarity search structure added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add similarity calculation functions to `src/mock_storage.rs`:
   ```rust
   /// Vector similarity calculation utilities
   pub struct VectorMath;
   
   impl VectorMath {
       /// Calculate cosine similarity between two vectors
       pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
           if a.len() != b.len() {
               return Err(format!("Vector dimension mismatch: {} vs {}", a.len(), b.len()));
           }
           
           let mut dot_product = 0.0;
           let mut norm_a = 0.0;
           let mut norm_b = 0.0;
           
           for i in 0..a.len() {
               dot_product += a[i] * b[i];
               norm_a += a[i] * a[i];
               norm_b += b[i] * b[i];
           }
           
           norm_a = norm_a.sqrt();
           norm_b = norm_b.sqrt();
           
           if norm_a == 0.0 || norm_b == 0.0 {
               return Ok(0.0); // Handle zero vectors
           }
           
           Ok(dot_product / (norm_a * norm_b))
       }
       
       /// Calculate Euclidean distance between two vectors
       pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, String> {
           if a.len() != b.len() {
               return Err(format!("Vector dimension mismatch: {} vs {}", a.len(), b.len()));
           }
           
           let mut sum_squared_diff = 0.0;
           
           for i in 0..a.len() {
               let diff = a[i] - b[i];
               sum_squared_diff += diff * diff;
           }
           
           Ok(sum_squared_diff.sqrt())
       }
       
       /// Calculate L2 (Euclidean) norm of a vector
       pub fn l2_norm(vector: &[f32]) -> f32 {
           vector.iter().map(|x| x * x).sum::<f32>().sqrt()
       }
       
       /// Normalize a vector to unit length
       pub fn normalize(vector: &mut [f32]) {
           let norm = Self::l2_norm(vector);
           if norm > 0.0 {
               for component in vector {
                   *component /= norm;
               }
           }
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Cosine similarity calculation implemented
- [ ] Euclidean distance calculation implemented
- [ ] Vector normalization utilities added
- [ ] Error handling for dimension mismatches

## Next Task
task_042c_add_similarity_search_to_storage.md

---

# Micro-Task 042c: Add Similarity Search to Storage

## Objective
Add similarity search methods to the MockVectorStorage implementation.

## Prerequisites
- Task 042b completed (vector similarity functions implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add similarity search methods to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Search for similar vectors using cosine similarity
       pub fn similarity_search(
           &self, 
           query_embedding: &[f32], 
           config: &SearchConfig
       ) -> StorageResult<Vec<SimilarityResult>> {
           if query_embedding.len() != self.dimension {
               return Err(StorageError::InvalidEmbedding(
                   format!("Query embedding has {} dimensions, expected {}", 
                           query_embedding.len(), self.dimension)
               ));
           }
           
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let mut results = Vec::new();
           
           for record in records.values() {
               match VectorMath::cosine_similarity(query_embedding, &record.embedding) {
                   Ok(similarity) => {
                       if similarity >= config.threshold {
                           results.push(SimilarityResult::new(record.clone(), similarity));
                       }
                   }
                   Err(e) => {
                       return Err(StorageError::InternalError(e));
                   }
               }
           }
           
           // Sort by similarity score (descending)
           results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score)
               .unwrap_or(std::cmp::Ordering::Equal));
           
           // Apply limit
           results.truncate(config.limit);
           
           Ok(results)
       }
       
       /// Search for similar vectors using content-based query
       pub fn search_by_content(
           &self, 
           query_content: &str, 
           config: &SearchConfig
       ) -> StorageResult<Vec<SimilarityResult>> {
           let query_embedding = VectorRecord::generate_mock_embedding(query_content);
           self.similarity_search(&query_embedding, config)
       }
       
       /// Get k-nearest neighbors
       pub fn knn_search(
           &self, 
           query_embedding: &[f32], 
           k: usize
       ) -> StorageResult<Vec<SimilarityResult>> {
           let config = SearchConfig {
               limit: k,
               threshold: 0.0, // No threshold for k-NN
               include_metadata: true,
           };
           
           self.similarity_search(query_embedding, &config)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Similarity search method implemented with configurable parameters
- [ ] Content-based search method added
- [ ] K-nearest neighbors search implemented
- [ ] Proper sorting by similarity scores

## Next Task
task_042d_add_batch_operations.md

---

# Micro-Task 042d: Add Batch Operations

## Objective
Add batch insertion and search operations for better performance.

## Prerequisites
- Task 042c completed (similarity search added to storage)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add batch operations to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Insert multiple records in batch
       pub fn batch_insert(&self, records: Vec<VectorRecord>) -> StorageResult<usize> {
           let mut inserted_count = 0;
           
           // Validate all embeddings first
           for record in &records {
               if record.embedding.len() != self.dimension {
                   return Err(StorageError::InvalidEmbedding(
                       format!("Record {} has {} dimensions, expected {}", 
                               record.id, record.embedding.len(), self.dimension)
                   ));
               }
           }
           
           let mut storage_records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           for record in records {
               storage_records.insert(record.id, record);
               inserted_count += 1;
           }
           
           Ok(inserted_count)
       }
       
       /// Delete multiple records by IDs
       pub fn batch_delete(&self, ids: &[Uuid]) -> StorageResult<usize> {
           let mut deleted_count = 0;
           
           let mut storage_records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           for id in ids {
               if storage_records.remove(id).is_some() {
                   deleted_count += 1;
               }
           }
           
           Ok(deleted_count)
       }
       
       /// Get multiple records by IDs
       pub fn batch_get(&self, ids: &[Uuid]) -> StorageResult<Vec<Option<VectorRecord>>> {
           let storage_records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let mut results = Vec::with_capacity(ids.len());
           
           for id in ids {
               results.push(storage_records.get(id).cloned());
           }
           
           Ok(results)
       }
       
       /// Get all records (for testing/debugging)
       pub fn get_all(&self) -> StorageResult<Vec<VectorRecord>> {
           let storage_records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           Ok(storage_records.values().cloned().collect())
       }
       
       /// Clear all records
       pub fn clear(&self) -> StorageResult<()> {
           let mut storage_records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           storage_records.clear();
           Ok(())
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Batch insertion with validation implemented
- [ ] Batch deletion returning count
- [ ] Batch retrieval operations
- [ ] Utility methods for testing and debugging

## Next Task
task_042e_add_similarity_search_tests_and_commit.md

---

# Micro-Task 042e: Add Similarity Search Tests and Commit

## Objective
Write comprehensive tests for similarity search functionality and commit the implementation.

## Prerequisites
- Task 042d completed (batch operations added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add similarity search tests to `src/mock_storage.rs`:
   ```rust
   #[cfg(test)]
   mod similarity_tests {
       use super::*;
       
       #[test]
       fn test_cosine_similarity_calculation() {
           let vec_a = vec![1.0, 0.0, 0.0];
           let vec_b = vec![1.0, 0.0, 0.0];
           let vec_c = vec![0.0, 1.0, 0.0];
           
           // Identical vectors should have similarity 1.0
           let sim_identical = VectorMath::cosine_similarity(&vec_a, &vec_b).unwrap();
           assert!((sim_identical - 1.0).abs() < 1e-6);
           
           // Orthogonal vectors should have similarity 0.0
           let sim_orthogonal = VectorMath::cosine_similarity(&vec_a, &vec_c).unwrap();
           assert!(sim_orthogonal.abs() < 1e-6);
       }
       
       #[test]
       fn test_similarity_search() {
           let storage = MockVectorStorage::new(384);
           
           // Create test records
           let record1 = VectorRecord::new(
               "Hello world".to_string(),
               "greeting".to_string(),
               None,
               None,
           );
           
           let record2 = VectorRecord::new(
               "Hello universe".to_string(),
               "greeting2".to_string(),
               None,
               None,
           );
           
           let record3 = VectorRecord::new(
               "Goodbye world".to_string(),
               "farewell".to_string(),
               None,
               None,
           );
           
           storage.insert(record1).unwrap();
           storage.insert(record2).unwrap();
           storage.insert(record3).unwrap();
           
           // Search for content similar to "Hello world"
           let config = SearchConfig::default();
           let results = storage.search_by_content("Hello world", &config).unwrap();
           
           assert!(!results.is_empty());
           // The exact match should have the highest similarity
           assert_eq!(results[0].record.content, "Hello world");
           assert!(results[0].similarity_score > 0.9);
       }
       
       #[test]
       fn test_knn_search() {
           let storage = MockVectorStorage::new(384);
           
           // Insert multiple records
           for i in 0..5 {
               let record = VectorRecord::new(
                   format!("Content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           let query_embedding = VectorRecord::generate_mock_embedding("Content 0");
           let results = storage.knn_search(&query_embedding, 3).unwrap();
           
           assert_eq!(results.len(), 3);
           // Results should be sorted by similarity (descending)
           for i in 1..results.len() {
               assert!(results[i-1].similarity_score >= results[i].similarity_score);
           }
       }
       
       #[test]
       fn test_batch_operations() {
           let storage = MockVectorStorage::new(384);
           
           // Create batch of records
           let mut records = Vec::new();
           for i in 0..3 {
               records.push(VectorRecord::new(
                   format!("Batch content {}", i),
                   format!("Batch title {}", i),
                   None,
                   None,
               ));
           }
           
           let ids: Vec<Uuid> = records.iter().map(|r| r.id).collect();
           
           // Test batch insert
           let inserted = storage.batch_insert(records).unwrap();
           assert_eq!(inserted, 3);
           assert_eq!(storage.count().unwrap(), 3);
           
           // Test batch get
           let retrieved = storage.batch_get(&ids).unwrap();
           assert_eq!(retrieved.len(), 3);
           assert!(retrieved.iter().all(|r| r.is_some()));
           
           // Test batch delete
           let deleted = storage.batch_delete(&ids).unwrap();
           assert_eq!(deleted, 3);
           assert_eq!(storage.count().unwrap(), 0);
       }
       
       #[test]
       fn test_search_threshold() {
           let storage = MockVectorStorage::new(384);
           
           let record = VectorRecord::new(
               "Test content".to_string(),
               "Test title".to_string(),
               None,
               None,
           );
           
           storage.insert(record).unwrap();
           
           // Search with high threshold - should find the exact match
           let config_high = SearchConfig {
               limit: 10,
               threshold: 0.9,
               include_metadata: true,
           };
           
           let results_high = storage.search_by_content("Test content", &config_high).unwrap();
           assert!(!results_high.is_empty());
           
           // Search with very high threshold - might not find anything
           let config_very_high = SearchConfig {
               limit: 10,
               threshold: 0.99,
               include_metadata: true,
           };
           
           let results_very_high = storage.search_by_content("Different content", &config_very_high).unwrap();
           assert!(results_very_high.is_empty() || results_very_high[0].similarity_score >= 0.99);
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement similarity search with cosine similarity and batch operations"`

## Success Criteria
- [ ] Cosine similarity calculation tests implemented and passing
- [ ] Similarity search tests implemented and passing
- [ ] K-NN search tests implemented and passing
- [ ] Batch operations tests implemented and passing
- [ ] Similarity search implementation committed to Git

## Next Task
task_043_implement_vector_indexing_operations.md