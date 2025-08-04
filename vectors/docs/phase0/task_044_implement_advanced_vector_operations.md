# Micro-Task 044a: Add Vector Filtering Operations

## Objective
Implement advanced vector filtering operations with metadata-based queries.

## Prerequisites
- Task 043e completed (vector indexing tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add vector filtering structures to `src/mock_storage.rs`:
   ```rust
   /// Vector search filter combining similarity and metadata criteria
   #[derive(Debug, Clone)]
   pub struct VectorSearchFilter {
       pub similarity_threshold: f32,
       pub metadata_filters: HashMap<String, MetadataFilter>,
       pub content_filters: ContentFilter,
       pub limit: usize,
   }
   
   #[derive(Debug, Clone)]
   pub enum MetadataFilter {
       Equals(String),
       Contains(String),
       StartsWith(String),
       EndsWith(String),
       OneOf(Vec<String>),
   }
   
   #[derive(Debug, Clone, Default)]
   pub struct ContentFilter {
       pub min_length: Option<usize>,
       pub max_length: Option<usize>,
       pub extensions: Vec<String>,
       pub exclude_extensions: Vec<String>,
       pub title_contains: Option<String>,
   }
   
   impl Default for VectorSearchFilter {
       fn default() -> Self {
           Self {
               similarity_threshold: 0.0,
               metadata_filters: HashMap::new(),
               content_filters: ContentFilter::default(),
               limit: 10,
           }
       }
   }
   
   impl VectorSearchFilter {
       pub fn new() -> Self {
           Self::default()
       }
       
       pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
           self.similarity_threshold = threshold;
           self
       }
       
       pub fn with_metadata_filter(mut self, key: String, filter: MetadataFilter) -> Self {
           self.metadata_filters.insert(key, filter);
           self
       }
       
       pub fn with_content_filter(mut self, filter: ContentFilter) -> Self {
           self.content_filters = filter;
           self
       }
       
       pub fn with_limit(mut self, limit: usize) -> Self {
           self.limit = limit;
           self
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] VectorSearchFilter with metadata and content filtering
- [ ] MetadataFilter enum with common string operations
- [ ] ContentFilter for document-specific criteria
- [ ] Builder pattern for easy filter construction

## Next Task
task_044b_implement_filter_matching_logic.md

---

# Micro-Task 044b: Implement Filter Matching Logic

## Objective
Implement logic to apply vector search filters to records.

## Prerequisites
- Task 044a completed (vector filtering operations created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add filter matching logic to `src/mock_storage.rs`:
   ```rust
   impl MetadataFilter {
       /// Check if a metadata value matches this filter
       pub fn matches(&self, value: &str) -> bool {
           match self {
               MetadataFilter::Equals(expected) => value == expected,
               MetadataFilter::Contains(substr) => value.to_lowercase().contains(&substr.to_lowercase()),
               MetadataFilter::StartsWith(prefix) => value.to_lowercase().starts_with(&prefix.to_lowercase()),
               MetadataFilter::EndsWith(suffix) => value.to_lowercase().ends_with(&suffix.to_lowercase()),
               MetadataFilter::OneOf(options) => {
                   let value_lower = value.to_lowercase();
                   options.iter().any(|opt| opt.to_lowercase() == value_lower)
               }
           }
       }
   }
   
   impl ContentFilter {
       /// Check if a record matches the content filter criteria
       pub fn matches(&self, record: &VectorRecord) -> bool {
           // Check content length
           let content_len = record.content.len();
           if let Some(min_len) = self.min_length {
               if content_len < min_len {
                   return false;
               }
           }
           if let Some(max_len) = self.max_length {
               if content_len > max_len {
                   return false;
               }
           }
           
           // Check extension filters
           if let Some(ref ext) = record.extension {
               if !self.extensions.is_empty() {
                   if !self.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                       return false;
                   }
               }
               
               if self.exclude_extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                   return false;
               }
           } else if !self.extensions.is_empty() {
               return false; // No extension but extensions required
           }
           
           // Check title contains
           if let Some(ref title_filter) = self.title_contains {
               if !record.title.to_lowercase().contains(&title_filter.to_lowercase()) {
                   return false;
               }
           }
           
           true
       }
   }
   
   impl VectorSearchFilter {
       /// Check if a record and its similarity score match all filter criteria
       pub fn matches(&self, record: &VectorRecord, similarity_score: f32) -> bool {
           // Check similarity threshold
           if similarity_score < self.similarity_threshold {
               return false;
           }
           
           // Check metadata filters
           for (key, filter) in &self.metadata_filters {
               if let Some(value) = record.metadata.get(key) {
                   if !filter.matches(value) {
                       return false;
                   }
               } else {
                   return false; // Required metadata key not found
               }
           }
           
           // Check content filters
           if !self.content_filters.matches(record) {
               return false;
           }
           
           true
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] MetadataFilter matching logic with case-insensitive operations
- [ ] ContentFilter matching for length, extension, and title criteria
- [ ] VectorSearchFilter combining all filter types
- [ ] Proper handling of missing metadata keys

## Next Task
task_044c_add_filtered_vector_search_methods.md

---

# Micro-Task 044c: Add Filtered Vector Search Methods

## Objective
Add search methods that apply vector search filters.

## Prerequisites
- Task 044b completed (filter matching logic implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add filtered search methods to MockVectorStorage impl in `src/mock_storage.rs`:
   ```rust
   impl MockVectorStorage {
       /// Advanced similarity search with filtering
       pub fn filtered_similarity_search(
           &self, 
           query_embedding: &[f32], 
           filter: &VectorSearchFilter
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
               // Calculate similarity
               match VectorMath::cosine_similarity(query_embedding, &record.embedding) {
                   Ok(similarity) => {
                       // Apply all filters
                       if filter.matches(record, similarity) {
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
           results.truncate(filter.limit);
           
           Ok(results)
       }
       
       /// Filtered search using content as query
       pub fn filtered_search_by_content(
           &self, 
           query_content: &str, 
           filter: &VectorSearchFilter
       ) -> StorageResult<Vec<SimilarityResult>> {
           let query_embedding = VectorRecord::generate_mock_embedding(query_content);
           self.filtered_similarity_search(&query_embedding, filter)
       }
       
       /// Advanced indexed search with filtering
       pub fn filtered_indexed_search(
           &self, 
           query_embedding: &[f32], 
           filter: &VectorSearchFilter
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
                   let nearest_clusters = self.find_nearest_clusters(query_embedding, index, 5)?;
                   let candidate_ids = self.get_cluster_candidates(&nearest_clusters, index);
                   
                   drop(index_guard);
                   
                   // Apply filtering to candidates
                   self.search_filtered_candidates(query_embedding, &candidate_ids, filter)
               }
               None => {
                   drop(index_guard);
                   // Fall back to regular filtered search
                   self.filtered_similarity_search(query_embedding, filter)
               }
           }
       }
       
       /// Search filtered candidates from index
       fn search_filtered_candidates(
           &self,
           query_embedding: &[f32],
           candidate_ids: &[Uuid],
           filter: &VectorSearchFilter
       ) -> StorageResult<Vec<SimilarityResult>> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           let mut results = Vec::new();
           
           for &candidate_id in candidate_ids {
               if let Some(record) = records.get(&candidate_id) {
                   match VectorMath::cosine_similarity(query_embedding, &record.embedding) {
                       Ok(similarity) => {
                           if filter.matches(record, similarity) {
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
           results.truncate(filter.limit);
           
           Ok(results)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Filtered similarity search with full filter application
- [ ] Content-based filtered search method
- [ ] Indexed filtered search with cluster pruning
- [ ] Proper result sorting and limiting

## Next Task
task_044d_add_vector_aggregation_operations.md

---

# Micro-Task 044d: Add Vector Aggregation Operations

## Objective
Implement vector aggregation operations for analytics and insights.

## Prerequisites
- Task 044c completed (filtered vector search methods added)

## Time Estimate
9 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add vector aggregation structures to `src/mock_storage.rs`:
   ```rust
   /// Vector collection statistics and aggregations
   #[derive(Debug, Clone)]
   pub struct VectorAggregations {
       pub count: usize,
       pub centroid: Vec<f32>,
       pub average_similarity_to_centroid: f32,
       pub min_similarity_to_centroid: f32,
       pub max_similarity_to_centroid: f32,
       pub content_length_stats: ContentLengthStats,
   }
   
   #[derive(Debug, Clone)]
   pub struct ContentLengthStats {
       pub min_length: usize,
       pub max_length: usize,
       pub average_length: f32,
       pub total_length: usize,
   }
   
   impl MockVectorStorage {
       /// Calculate aggregations for all records
       pub fn calculate_aggregations(&self) -> StorageResult<VectorAggregations> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           if records.is_empty() {
               return Ok(VectorAggregations {
                   count: 0,
                   centroid: vec![0.0; self.dimension],
                   average_similarity_to_centroid: 0.0,
                   min_similarity_to_centroid: 0.0,
                   max_similarity_to_centroid: 0.0,
                   content_length_stats: ContentLengthStats {
                       min_length: 0,
                       max_length: 0,
                       average_length: 0.0,
                       total_length: 0,
                   },
               });
           }
           
           // Calculate centroid
           let mut centroid = vec![0.0; self.dimension];
           let mut content_lengths = Vec::new();
           
           for record in records.values() {
               for (i, &value) in record.embedding.iter().enumerate() {
                   centroid[i] += value;
               }
               content_lengths.push(record.content.len());
           }
           
           let count = records.len() as f32;
           for component in &mut centroid {
               *component /= count;
           }
           
           // Calculate similarity statistics to centroid
           let mut similarities = Vec::new();
           for record in records.values() {
               match VectorMath::cosine_similarity(&centroid, &record.embedding) {
                   Ok(sim) => similarities.push(sim),
                   Err(e) => return Err(StorageError::InternalError(e)),
               }
           }
           
           let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
           let min_similarity = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
           let max_similarity = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
           
           // Calculate content length statistics
           let min_length = content_lengths.iter().min().copied().unwrap_or(0);
           let max_length = content_lengths.iter().max().copied().unwrap_or(0);
           let total_length: usize = content_lengths.iter().sum();
           let average_length = total_length as f32 / content_lengths.len() as f32;
           
           Ok(VectorAggregations {
               count: records.len(),
               centroid,
               average_similarity_to_centroid: avg_similarity,
               min_similarity_to_centroid: min_similarity,
               max_similarity_to_centroid: max_similarity,
               content_length_stats: ContentLengthStats {
                   min_length,
                   max_length,
                   average_length,
                   total_length,
               },
           })
       }
       
       /// Calculate aggregations for filtered subset of records
       pub fn calculate_filtered_aggregations(
           &self, 
           filter: &VectorSearchFilter
       ) -> StorageResult<VectorAggregations> {
           let records = self.records.lock()
               .map_err(|e| StorageError::InternalError(e.to_string()))?;
           
           // Filter records first
           let filtered_records: Vec<&VectorRecord> = records.values()
               .filter(|record| {
                   // For aggregation, we check filter without similarity threshold
                   filter.content_filters.matches(record) && 
                   filter.metadata_filters.iter().all(|(key, metadata_filter)| {
                       record.metadata.get(key)
                           .map(|value| metadata_filter.matches(value))
                           .unwrap_or(false)
                   })
               })
               .collect();
           
           if filtered_records.is_empty() {
               return Ok(VectorAggregations {
                   count: 0,
                   centroid: vec![0.0; self.dimension],
                   average_similarity_to_centroid: 0.0,
                   min_similarity_to_centroid: 0.0,
                   max_similarity_to_centroid: 0.0,
                   content_length_stats: ContentLengthStats {
                       min_length: 0,
                       max_length: 0,
                       average_length: 0.0,
                       total_length: 0,
                   },
               });
           }
           
           // Calculate centroid for filtered records
           let mut centroid = vec![0.0; self.dimension];
           let mut content_lengths = Vec::new();
           
           for record in &filtered_records {
               for (i, &value) in record.embedding.iter().enumerate() {
                   centroid[i] += value;
               }
               content_lengths.push(record.content.len());
           }
           
           let count = filtered_records.len() as f32;
           for component in &mut centroid {
               *component /= count;
           }
           
           // Calculate similarity statistics
           let mut similarities = Vec::new();
           for record in &filtered_records {
               match VectorMath::cosine_similarity(&centroid, &record.embedding) {
                   Ok(sim) => similarities.push(sim),
                   Err(e) => return Err(StorageError::InternalError(e)),
               }
           }
           
           let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
           let min_similarity = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
           let max_similarity = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
           
           // Content length statistics
           let min_length = content_lengths.iter().min().copied().unwrap_or(0);
           let max_length = content_lengths.iter().max().copied().unwrap_or(0);
           let total_length: usize = content_lengths.iter().sum();
           let average_length = total_length as f32 / content_lengths.len() as f32;
           
           Ok(VectorAggregations {
               count: filtered_records.len(),
               centroid,
               average_similarity_to_centroid: avg_similarity,
               min_similarity_to_centroid: min_similarity,
               max_similarity_to_centroid: max_similarity,
               content_length_stats: ContentLengthStats {
                   min_length,
                   max_length,
                   average_length,
                   total_length,
               },
           })
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Vector aggregation structures with comprehensive statistics
- [ ] Centroid calculation for record collections
- [ ] Similarity statistics to centroid
- [ ] Content length statistics and filtered aggregations

## Next Task
task_044e_add_advanced_operations_tests_and_commit.md

---

# Micro-Task 044e: Add Advanced Operations Tests and Commit

## Objective
Write tests for advanced vector operations and commit the implementation.

## Prerequisites
- Task 044d completed (vector aggregation operations added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add advanced operations tests to `src/mock_storage.rs`:
   ```rust
   #[cfg(test)]
   mod advanced_operations_tests {
       use super::*;
       
       #[test]
       fn test_metadata_filtering() {
           let storage = MockVectorStorage::new(384);
           
           // Create records with metadata
           let record1 = VectorRecord::new(
               "Content about programming".to_string(),
               "Programming Guide".to_string(),
               None,
               Some("md".to_string()),
           ).with_metadata("category".to_string(), "tutorial".to_string())
            .with_metadata("language".to_string(), "rust".to_string());
           
           let record2 = VectorRecord::new(
               "Content about algorithms".to_string(),
               "Algorithm Guide".to_string(),
               None,
               Some("md".to_string()),
           ).with_metadata("category".to_string(), "reference".to_string())
            .with_metadata("language".to_string(), "python".to_string());
           
           storage.insert(record1).unwrap();
           storage.insert(record2).unwrap();
           
           // Filter by category
           let filter = VectorSearchFilter::new()
               .with_metadata_filter("category".to_string(), MetadataFilter::Equals("tutorial".to_string()))
               .with_limit(10);
           
           let results = storage.filtered_search_by_content("programming", &filter).unwrap();
           
           assert_eq!(results.len(), 1);
           assert_eq!(results[0].record.title, "Programming Guide");
           assert_eq!(results[0].record.metadata.get("category").unwrap(), "tutorial");
       }
       
       #[test]
       fn test_content_filtering() {
           let storage = MockVectorStorage::new(384);
           
           let short_record = VectorRecord::new(
               "Short".to_string(),
               "Short Content".to_string(),
               None,
               Some("txt".to_string()),
           );
           
           let long_record = VectorRecord::new(
               "This is a much longer piece of content that should be filtered based on length criteria".to_string(),
               "Long Content".to_string(),
               None,
               Some("md".to_string()),
           );
           
           storage.insert(short_record).unwrap();
           storage.insert(long_record).unwrap();
           
           // Filter by content length and extension
           let content_filter = ContentFilter {
               min_length: Some(50),
               max_length: None,
               extensions: vec!["md".to_string()],
               exclude_extensions: vec![],
               title_contains: None,
           };
           
           let filter = VectorSearchFilter::new()
               .with_content_filter(content_filter)
               .with_limit(10);
           
           let results = storage.filtered_search_by_content("content", &filter).unwrap();
           
           assert_eq!(results.len(), 1);
           assert_eq!(results[0].record.title, "Long Content");
           assert_eq!(results[0].record.extension.as_ref().unwrap(), "md");
       }
       
       #[test]
       fn test_combined_filtering() {
           let storage = MockVectorStorage::new(384);
           
           let record1 = VectorRecord::new(
               "Rust programming tutorial content with sufficient length".to_string(),
               "Rust Tutorial".to_string(),
               None,
               Some("md".to_string()),
           ).with_metadata("language".to_string(), "rust".to_string())
            .with_metadata("difficulty".to_string(), "beginner".to_string());
           
           let record2 = VectorRecord::new(
               "Python programming tutorial".to_string(),
               "Python Tutorial".to_string(),
               None,
               Some("md".to_string()),
           ).with_metadata("language".to_string(), "python".to_string())
            .with_metadata("difficulty".to_string(), "beginner".to_string());
           
           storage.insert(record1).unwrap();
           storage.insert(record2).unwrap();
           
           // Complex filter: Rust language, beginner difficulty, minimum length, title contains "rust"
           let content_filter = ContentFilter {
               min_length: Some(30),
               max_length: None,
               extensions: vec!["md".to_string()],
               exclude_extensions: vec![],
               title_contains: Some("rust".to_string()),
           };
           
           let filter = VectorSearchFilter::new()
               .with_metadata_filter("language".to_string(), MetadataFilter::Equals("rust".to_string()))
               .with_metadata_filter("difficulty".to_string(), MetadataFilter::Equals("beginner".to_string()))
               .with_content_filter(content_filter)
               .with_similarity_threshold(0.0)
               .with_limit(10);
           
           let results = storage.filtered_search_by_content("programming", &filter).unwrap();
           
           assert_eq!(results.len(), 1);
           assert_eq!(results[0].record.title, "Rust Tutorial");
       }
       
       #[test]
       fn test_vector_aggregations() {
           let storage = MockVectorStorage::new(384);
           
           // Add records with varying content lengths
           for i in 0..5 {
               let content = "x".repeat((i + 1) * 10); // Lengths: 10, 20, 30, 40, 50
               let record = VectorRecord::new(
                   content,
                   format!("Title {}", i),
                   None,
                   Some("txt".to_string()),
               );
               storage.insert(record).unwrap();
           }
           
           let aggregations = storage.calculate_aggregations().unwrap();
           
           assert_eq!(aggregations.count, 5);
           assert_eq!(aggregations.centroid.len(), 384);
           assert_eq!(aggregations.content_length_stats.min_length, 10);
           assert_eq!(aggregations.content_length_stats.max_length, 50);
           assert_eq!(aggregations.content_length_stats.average_length, 30.0);
           assert_eq!(aggregations.content_length_stats.total_length, 150);
           
           // Similarity stats should be valid
           assert!(aggregations.average_similarity_to_centroid >= 0.0);
           assert!(aggregations.average_similarity_to_centroid <= 1.0);
           assert!(aggregations.min_similarity_to_centroid <= aggregations.average_similarity_to_centroid);
           assert!(aggregations.max_similarity_to_centroid >= aggregations.average_similarity_to_centroid);
       }
       
       #[test]
       fn test_filtered_aggregations() {
           let storage = MockVectorStorage::new(384);
           
           // Add records with different extensions
           for i in 0..6 {
               let ext = if i < 3 { "rs" } else { "py" };
               let record = VectorRecord::new(
                   format!("Content {}", i),
                   format!("Title {}", i),
                   None,
                   Some(ext.to_string()),
               );
               storage.insert(record).unwrap();
           }
           
           // Filter for Rust files only
           let content_filter = ContentFilter {
               min_length: None,
               max_length: None,
               extensions: vec!["rs".to_string()],
               exclude_extensions: vec![],
               title_contains: None,
           };
           
           let filter = VectorSearchFilter::new().with_content_filter(content_filter);
           
           let filtered_aggs = storage.calculate_filtered_aggregations(&filter).unwrap();
           
           assert_eq!(filtered_aggs.count, 3); // Should only include Rust files
           assert_eq!(filtered_aggs.centroid.len(), 384);
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement advanced vector operations with filtering and aggregations"`

## Success Criteria
- [ ] Metadata filtering tests implemented and passing
- [ ] Content filtering tests implemented and passing
- [ ] Combined filtering tests implemented and passing
- [ ] Vector aggregations tests implemented and passing
- [ ] Advanced vector operations committed to Git

## Next Task
task_045_implement_vector_storage_persistence.md