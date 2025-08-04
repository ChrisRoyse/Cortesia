# Micro-Task 043a: Create Vector Index Management

## Objective
Add vector indexing capabilities to improve search performance in the mock storage.

## Prerequisites
- Task 042e completed (similarity search tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add vector index structures to `src/mock_storage.rs`:
   ```rust
   /// Vector index for faster similarity search
   #[derive(Debug, Clone)]
   pub struct VectorIndex {
       pub records: Vec<Uuid>,
       pub centroids: Vec<Vec<f32>>,
       pub clusters: HashMap<usize, Vec<Uuid>>,
       pub dimension: usize,
   }
   
   impl VectorIndex {
       pub fn new(dimension: usize) -> Self {
           Self {
               records: Vec::new(),
               centroids: Vec::new(),
               clusters: HashMap::new(),
               dimension,
           }
       }
   }
   
   /// Index configuration for vector clustering
   #[derive(Debug, Clone)]
   pub struct IndexConfig {
       pub num_clusters: usize,
       pub max_iterations: usize,
       pub convergence_threshold: f32,
   }
   
   impl Default for IndexConfig {
       fn default() -> Self {
           Self {
               num_clusters: 10,
               max_iterations: 100,
               convergence_threshold: 0.001,
           }
       }
   }
   ```
3. Update MockVectorStorage to include index:
   ```rust
   pub struct MockVectorStorage {
       records: Arc<Mutex<HashMap<Uuid, VectorRecord>>>,
       dimension: usize,
       index: Arc<Mutex<Option<VectorIndex>>>,
   }
   
   impl MockVectorStorage {
       pub fn new(dimension: usize) -> Self {
           Self {
               records: Arc::new(Mutex::new(HashMap::new())),
               dimension,
               index: Arc::new(Mutex::new(None)),
           }
       }
   }
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] VectorIndex structure defined with clustering support
- [ ] IndexConfig with reasonable defaults
- [ ] MockVectorStorage updated to include index
- [ ] Code compiles successfully

## Next Task
task_043b_implement_simple_clustering.md

---

# Micro-Task 043b: Implement Simple Clustering

## Objective
Implement a simple k-means clustering algorithm for vector indexing.

## Prerequisites
- Task 043a completed (vector index management created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add clustering implementation to `src/mock_storage.rs`:
   ```rust
   impl VectorIndex {
       /// Build index using simple k-means clustering
       pub fn build_from_records(
           records: &HashMap<Uuid, VectorRecord>, 
           config: &IndexConfig
       ) -> Result<Self, String> {
           if records.is_empty() {
               return Ok(Self::new(384)); // Default dimension
           }
           
           let dimension = records.values().next().unwrap().embedding.len();
           let mut index = Self::new(dimension);
           
           // Collect all embeddings and IDs
           let mut embeddings = Vec::new();
           let mut ids = Vec::new();
           
           for (id, record) in records {
               embeddings.push(record.embedding.clone());
               ids.push(*id);
               index.records.push(*id);
           }
           
           // Initialize centroids randomly
           let num_clusters = config.num_clusters.min(embeddings.len());
           index.centroids = Self::initialize_centroids(&embeddings, num_clusters);
           
           // Perform k-means iterations
           for _iteration in 0..config.max_iterations {
               let mut new_clusters: HashMap<usize, Vec<Uuid>> = HashMap::new();
               
               // Assign each point to nearest centroid
               for (i, embedding) in embeddings.iter().enumerate() {
                   let cluster_id = Self::find_nearest_centroid(embedding, &index.centroids)?;
                   new_clusters.entry(cluster_id).or_insert_with(Vec::new).push(ids[i]);
               }
               
               // Update centroids
               let mut converged = true;
               for (cluster_id, cluster_points) in &new_clusters {
                   if let Some(new_centroid) = Self::calculate_centroid(&embeddings, &ids, cluster_points) {
                       if let Some(old_centroid) = index.centroids.get_mut(*cluster_id) {
                           let distance = VectorMath::euclidean_distance(old_centroid, &new_centroid)
                               .unwrap_or(f32::MAX);
                           if distance > config.convergence_threshold {
                               converged = false;
                           }
                           *old_centroid = new_centroid;
                       }
                   }
               }
               
               index.clusters = new_clusters;
               
               if converged {
                   break;
               }
           }
           
           Ok(index)
       }
       
       /// Initialize centroids using random selection
       fn initialize_centroids(embeddings: &[Vec<f32>], num_clusters: usize) -> Vec<Vec<f32>> {
           let mut centroids = Vec::new();
           let step = embeddings.len() / num_clusters.max(1);
           
           for i in 0..num_clusters {
               let idx = (i * step).min(embeddings.len() - 1);
               centroids.push(embeddings[idx].clone());
           }
           
           centroids
       }
       
       /// Find the nearest centroid for a given embedding
       fn find_nearest_centroid(embedding: &[f32], centroids: &[Vec<f32>]) -> Result<usize, String> {
           let mut best_cluster = 0;
           let mut best_distance = f32::MAX;
           
           for (i, centroid) in centroids.iter().enumerate() {
               let distance = VectorMath::euclidean_distance(embedding, centroid)?;
               if distance < best_distance {
                   best_distance = distance;
                   best_cluster = i;
               }
           }
           
           Ok(best_cluster)
       }
       
       /// Calculate centroid for a cluster of points
       fn calculate_centroid(
           embeddings: &[Vec<f32>], 
           ids: &[Uuid], 
           cluster_points: &[Uuid]
       ) -> Option<Vec<f32>> {
           if cluster_points.is_empty() {
               return None;
           }
           
           let dimension = embeddings[0].len();
           let mut centroid = vec![0.0; dimension];
           let mut count = 0;
           
           for point_id in cluster_points {
               if let Some(pos) = ids.iter().position(|id| id == point_id) {
                   for (i, &value) in embeddings[pos].iter().enumerate() {
                       centroid[i] += value;
                   }
                   count += 1;
               }
           }
           
           if count > 0 {
               for component in &mut centroid {
                   *component /= count as f32;
               }
           }
           
           Some(centroid)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] K-means clustering algorithm implemented
- [ ] Centroid initialization and update logic working
- [ ] Cluster assignment based on nearest centroid
- [ ] Convergence detection implemented

## Next Task
task_043c_add_indexed_search_methods.md

---

# Micro-Task 043c: Add Indexed Search Methods

## Objective
Add search methods that utilize the vector index for improved performance.

## Prerequisites
- Task 043b completed (simple clustering implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add indexed search methods to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Build vector index for faster searching
       pub fn build_index(&self, config: &IndexConfig) -> StorageResult<()> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let new_index = VectorIndex::build_from_records(&records, config)
               .map_err(|e| StorageError::InternalError(e))?;
           
           let mut index = self.index.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           *index = Some(new_index);
           Ok(())
       }
       
       /// Indexed similarity search (faster for large datasets)
       pub fn indexed_similarity_search(
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
           
           let index_guard = self.index.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           match index_guard.as_ref() {
               Some(index) => {
                   // Find nearest clusters
                   let nearest_clusters = self.find_nearest_clusters(query_embedding, index, 3)?;
                   
                   // Search only in nearest clusters
                   let candidate_ids = self.get_cluster_candidates(&nearest_clusters, index);
                   
                   drop(index_guard);
                   
                   self.search_candidates(query_embedding, &candidate_ids, config)
               }
               None => {
                   drop(index_guard);
                   // Fall back to brute force search
                   self.similarity_search(query_embedding, config)
               }
           }
       }
       
       /// Find nearest clusters to query embedding
       fn find_nearest_clusters(
           &self,
           query_embedding: &[f32],
           index: &VectorIndex,
           num_clusters: usize
       ) -> StorageResult<Vec<usize>> {
           let mut cluster_distances = Vec::new();
           
           for (i, centroid) in index.centroids.iter().enumerate() {
               match VectorMath::euclidean_distance(query_embedding, centroid) {
                   Ok(distance) => cluster_distances.push((i, distance)),
                   Err(e) => return Err(StorageError::InternalError(e)),
               }
           }
           
           // Sort by distance and take nearest clusters
           cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
           
           Ok(cluster_distances.into_iter()
               .take(num_clusters)
               .map(|(cluster_id, _)| cluster_id)
               .collect())
       }
       
       /// Get candidate record IDs from specified clusters
       fn get_cluster_candidates(&self, cluster_ids: &[usize], index: &VectorIndex) -> Vec<Uuid> {
           let mut candidates = Vec::new();
           
           for &cluster_id in cluster_ids {
               if let Some(cluster_records) = index.clusters.get(&cluster_id) {
                   candidates.extend(cluster_records.iter().copied());
               }
           }
           
           candidates
       }
       
       /// Search only among candidate records
       fn search_candidates(
           &self,
           query_embedding: &[f32],
           candidate_ids: &[Uuid],
           config: &SearchConfig
       ) -> StorageResult<Vec<SimilarityResult>> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let mut results = Vec::new();
           
           for &candidate_id in candidate_ids {
               if let Some(record) = records.get(&candidate_id) {
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
           }
           
           // Sort by similarity score (descending)
           results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score)
               .unwrap_or(std::cmp::Ordering::Equal));
           
           // Apply limit
           results.truncate(config.limit);
           
           Ok(results)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Index building method implemented
- [ ] Indexed similarity search with cluster pruning
- [ ] Fallback to brute force when no index exists
- [ ] Candidate filtering based on nearest clusters

## Next Task
task_043d_add_index_maintenance_methods.md

---

# Micro-Task 043d: Add Index Maintenance Methods

## Objective
Add methods for index maintenance including updates and rebuilds.

## Prerequisites
- Task 043c completed (indexed search methods added)

## Time Estimate
8 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add index maintenance methods to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Check if index exists and is valid
       pub fn has_index(&self) -> bool {
           if let Ok(index) = self.index.lock() {
               index.is_some()
           } else {
               false
           }
       }
       
       /// Get index statistics
       pub fn index_stats(&self) -> StorageResult<Option<IndexStats>> {
           let index_guard = self.index.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           match index_guard.as_ref() {
               Some(index) => {
                   let mut cluster_sizes = Vec::new();
                   let mut total_points = 0;
                   
                   for cluster_records in index.clusters.values() {
                       cluster_sizes.push(cluster_records.len());
                       total_points += cluster_records.len();
                   }
                   
                   Ok(Some(IndexStats {
                       num_clusters: index.centroids.len(),
                       total_records: total_points,
                       cluster_sizes,
                       dimension: index.dimension,
                   }))
               }
               None => Ok(None),
           }
       }
       
       /// Rebuild index (should be called after significant data changes)
       pub fn rebuild_index(&self, config: &IndexConfig) -> StorageResult<()> {
           self.build_index(config)
       }
       
       /// Clear the index
       pub fn clear_index(&self) -> StorageResult<()> {
           let mut index = self.index.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           *index = None;
           Ok(())
       }
       
       /// Insert record and update index incrementally (simplified approach)
       pub fn insert_with_index_update(&self, record: VectorRecord, index_config: &IndexConfig) -> StorageResult<()> {
           // Insert the record first
           self.insert(record)?;
           
           // If we have many records, rebuild the index
           let record_count = self.count()?;
           if record_count % 100 == 0 && record_count > 0 {
               self.rebuild_index(index_config)?;
           }
           
           Ok(())
       }
   }
   
   /// Index statistics for monitoring
   #[derive(Debug, Clone)]
   pub struct IndexStats {
       pub num_clusters: usize,
       pub total_records: usize,
       pub cluster_sizes: Vec<usize>,
       pub dimension: usize,
   }
   
   impl IndexStats {
       pub fn average_cluster_size(&self) -> f32 {
           if self.num_clusters > 0 {
               self.total_records as f32 / self.num_clusters as f32
           } else {
               0.0
           }
       }
       
       pub fn largest_cluster_size(&self) -> usize {
           self.cluster_sizes.iter().max().copied().unwrap_or(0)
       }
       
       pub fn smallest_cluster_size(&self) -> usize {
           self.cluster_sizes.iter().min().copied().unwrap_or(0)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Index existence checking implemented
- [ ] Index statistics collection working
- [ ] Index rebuild and clear methods added
- [ ] Incremental index update strategy

## Next Task
task_043e_add_vector_indexing_tests_and_commit.md

---

# Micro-Task 043e: Add Vector Indexing Tests and Commit

## Objective
Write tests for vector indexing functionality and commit the implementation.

## Prerequisites
- Task 043d completed (index maintenance methods added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add indexing tests to `src/mock_storage.rs`:
   ```rust
   #[cfg(test)]
   mod indexing_tests {
       use super::*;
       
       #[test]
       fn test_index_building() {
           let storage = MockVectorStorage::new(384);
           
           // Add some test records
           for i in 0..20 {
               let record = VectorRecord::new(
                   format!("Content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           assert!(!storage.has_index());
           
           let config = IndexConfig::default();
           storage.build_index(&config).unwrap();
           
           assert!(storage.has_index());
           
           let stats = storage.index_stats().unwrap().unwrap();
           assert_eq!(stats.total_records, 20);
           assert!(stats.num_clusters <= config.num_clusters);
           assert_eq!(stats.dimension, 384);
       }
       
       #[test]
       fn test_indexed_vs_regular_search() {
           let storage = MockVectorStorage::new(384);
           
           // Add test records
           let mut test_records = Vec::new();
           for i in 0..15 {
               let record = VectorRecord::new(
                   format!("Test content number {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               test_records.push(record.clone());
               storage.insert(record).unwrap();
           }
           
           let query_embedding = &test_records[0].embedding;
           let config = SearchConfig {
               limit: 5,
               threshold: 0.0,
               include_metadata: true,
           };
           
           // Search without index
           let results_no_index = storage.similarity_search(query_embedding, &config).unwrap();
           
           // Build index and search with index
           let index_config = IndexConfig::default();
           storage.build_index(&index_config).unwrap();
           let results_with_index = storage.indexed_similarity_search(query_embedding, &config).unwrap();
           
           // Both should find results (though order might differ slightly)
           assert!(!results_no_index.is_empty());
           assert!(!results_with_index.is_empty());
           
           // The top result should be the same (exact match)
           assert_eq!(results_no_index[0].record.id, results_with_index[0].record.id);
       }
       
       #[test]
       fn test_index_statistics() {
           let storage = MockVectorStorage::new(384);
           
           // Initially no index
           assert!(storage.index_stats().unwrap().is_none());
           
           // Add records and build index
           for i in 0..25 {
               let record = VectorRecord::new(
                   format!("Content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           let config = IndexConfig {
               num_clusters: 5,
               max_iterations: 50,
               convergence_threshold: 0.001,
           };
           
           storage.build_index(&config).unwrap();
           
           let stats = storage.index_stats().unwrap().unwrap();
           assert_eq!(stats.total_records, 25);
           assert!(stats.num_clusters <= 5);
           assert!(stats.average_cluster_size() > 0.0);
           assert!(stats.largest_cluster_size() >= stats.smallest_cluster_size());
       }
       
       #[test]
       fn test_index_maintenance() {
           let storage = MockVectorStorage::new(384);
           
           // Add initial records
           for i in 0..10 {
               let record = VectorRecord::new(
                   format!("Initial content {}", i),
                   format!("Initial title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           let config = IndexConfig::default();
           storage.build_index(&config).unwrap();
           
           let initial_stats = storage.index_stats().unwrap().unwrap();
           assert_eq!(initial_stats.total_records, 10);
           
           // Clear index
           storage.clear_index().unwrap();
           assert!(!storage.has_index());
           
           // Rebuild
           storage.rebuild_index(&config).unwrap();
           assert!(storage.has_index());
           
           let rebuilt_stats = storage.index_stats().unwrap().unwrap();
           assert_eq!(rebuilt_stats.total_records, 10);
       }
       
       #[test]
       fn test_clustering_with_few_records() {
           let storage = MockVectorStorage::new(384);
           
           // Add only 3 records but request 10 clusters
           for i in 0..3 {
               let record = VectorRecord::new(
                   format!("Content {}", i),
                   format!("Title {}", i),
                   None,
                   None,
               );
               storage.insert(record).unwrap();
           }
           
           let config = IndexConfig {
               num_clusters: 10, // More clusters than records
               max_iterations: 50,
               convergence_threshold: 0.001,
           };
           
           storage.build_index(&config).unwrap();
           
           let stats = storage.index_stats().unwrap().unwrap();
           assert_eq!(stats.total_records, 3);
           // Should create at most as many clusters as records
           assert!(stats.num_clusters <= 3);
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement vector indexing with k-means clustering and indexed search"`

## Success Criteria
- [ ] Index building tests implemented and passing
- [ ] Indexed vs regular search comparison tests passing
- [ ] Index statistics tests implemented and passing
- [ ] Index maintenance tests implemented and passing
- [ ] Vector indexing implementation committed to Git

## Next Task
task_044_implement_advanced_vector_operations.md