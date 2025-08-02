# Micro-Phase 9.13: Implement High-Performance Concept Storage

## Objective
Implement high-performance concept CRUD operations with advanced indexing, batch processing, and optimized retrieval for browser-based CortexKG storage.

## Prerequisites
- Completed micro-phase 9.12 (Storage schema)
- Understanding of IndexedDB transaction management
- Knowledge of concept metadata structures

## Task Description
Create comprehensive concept storage operations including batch processing, semantic queries, and performance-optimized retrieval with caching and index utilization.

## Specific Actions

1. **Create concept storage manager**:
   ```rust
   // src/storage/concept_storage.rs
   use wasm_bindgen::prelude::*;
   use wasm_bindgen_futures::JsFuture;
   use web_sys::{IdbDatabase, IdbObjectStore, IdbTransaction, IdbRequest, IdbCursorWithValue};
   use serde::{Serialize, Deserialize};
   use std::collections::HashMap;
   use crate::storage::schema::{ConceptSchema, ConceptMetadata, ConnectionType};
   
   #[wasm_bindgen]
   pub struct ConceptStorage {
       db: IdbDatabase,
       cache: HashMap<u32, ConceptSchema>,
       cache_size_limit: usize,
       cache_hits: u32,
       cache_misses: u32,
   }
   
   #[wasm_bindgen]
   impl ConceptStorage {
       #[wasm_bindgen(constructor)]
       pub fn new(db: IdbDatabase, cache_size: usize) -> Self {
           Self {
               db,
               cache: HashMap::new(),
               cache_size_limit: cache_size,
               cache_hits: 0,
               cache_misses: 0,
           }
       }
       
       #[wasm_bindgen]
       pub async fn store_concept(&mut self, concept: &ConceptSchema) -> Result<(), JsValue> {
           // Update cache
           self.update_cache(concept.id, concept.clone());
           
           // Store in IndexedDB
           let transaction = self.db.transaction_with_str_and_mode(
               "concepts",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("concepts")?;
           let value = serde_wasm_bindgen::to_value(concept)?;
           let key = JsValue::from(concept.id);
           
           let request = store.put_with_key(&value, &key)?;
           JsFuture::from(request).await?;
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub async fn get_concept(&mut self, id: u32) -> Result<Option<ConceptSchema>, JsValue> {
           // Check cache first
           if let Some(concept) = self.cache.get(&id) {
               self.cache_hits += 1;
               return Ok(Some(concept.clone()));
           }
           
           self.cache_misses += 1;
           
           // Fetch from IndexedDB
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           
           let request = store.get(&JsValue::from(id))?;
           let result = JsFuture::from(request).await?;
           
           if result.is_undefined() {
               Ok(None)
           } else {
               let concept: ConceptSchema = serde_wasm_bindgen::from_value(result)?;
               
               // Update cache
               self.update_cache(id, concept.clone());
               
               Ok(Some(concept))
           }
       }
   }
   ```

2. **Implement batch operations**:
   ```rust
   impl ConceptStorage {
       #[wasm_bindgen]
       pub async fn store_concepts_batch(&mut self, concepts: Vec<ConceptSchema>) -> Result<u32, JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "concepts",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("concepts")?;
           let mut stored_count = 0u32;
           
           // Process in chunks for better performance
           for chunk in concepts.chunks(50) {
               let mut requests = Vec::new();
               
               for concept in chunk {
                   // Update cache
                   self.update_cache(concept.id, concept.clone());
                   
                   let value = serde_wasm_bindgen::to_value(concept)?;
                   let key = JsValue::from(concept.id);
                   let request = store.put_with_key(&value, &key)?;
                   requests.push(JsFuture::from(request));
               }
               
               // Wait for all requests in chunk
               for request in requests {
                   request.await?;
                   stored_count += 1;
               }
           }
           
           Ok(stored_count)
       }
       
       #[wasm_bindgen]
       pub async fn get_concepts_batch(&mut self, ids: Vec<u32>) -> Result<Vec<ConceptSchema>, JsValue> {
           let mut concepts = Vec::new();
           let mut missing_ids = Vec::new();
           
           // Check cache first
           for id in ids {
               if let Some(concept) = self.cache.get(&id) {
                   concepts.push(concept.clone());
                   self.cache_hits += 1;
               } else {
                   missing_ids.push(id);
                   self.cache_misses += 1;
               }
           }
           
           // Fetch missing concepts from IndexedDB
           if !missing_ids.is_empty() {
               let transaction = self.db.transaction_with_str("concepts")?;
               let store = transaction.object_store("concepts")?;
               
               for id in missing_ids {
                   let request = store.get(&JsValue::from(id))?;
                   let result = JsFuture::from(request).await?;
                   
                   if !result.is_undefined() {
                       let concept: ConceptSchema = serde_wasm_bindgen::from_value(result)?;
                       self.update_cache(id, concept.clone());
                       concepts.push(concept);
                   }
               }
           }
           
           Ok(concepts)
       }
       
       #[wasm_bindgen]
       pub async fn delete_concepts_batch(&mut self, ids: Vec<u32>) -> Result<u32, JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "concepts",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("concepts")?;
           let mut deleted_count = 0u32;
           
           for id in ids {
               // Remove from cache
               self.cache.remove(&id);
               
               let request = store.delete(&JsValue::from(id))?;
               JsFuture::from(request).await?;
               deleted_count += 1;
           }
           
           Ok(deleted_count)
       }
   }
   ```

3. **Implement semantic queries**:
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct ConceptQuery {
       pub category: Option<String>,
       pub complexity_range: Option<(f32, f32)>,
       pub content_pattern: Option<String>,
       pub tags: Vec<String>,
       pub min_connections: Option<u32>,
       pub created_after: Option<f64>,
       pub last_accessed_after: Option<f64>,
       pub limit: Option<u32>,
       pub offset: Option<u32>,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct QueryResult {
       pub concepts: Vec<ConceptSchema>,
       pub total_count: u32,
       pub query_time_ms: f64,
       pub cache_utilized: bool,
   }
   
   impl ConceptStorage {
       #[wasm_bindgen]
       pub async fn query_concepts(&self, query: ConceptQuery) -> Result<QueryResult, JsValue> {
           let start_time = js_sys::Date::now();
           let mut concepts = Vec::new();
           let mut total_count = 0u32;
           
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           
           // Choose optimal index based on query
           let cursor_request = if let Some(category) = &query.category {
               // Use category index
               let index = store.index("category")?;
               index.open_cursor_with_range(&JsValue::from_str(category))?
           } else if query.complexity_range.is_some() {
               // Use complexity index
               let index = store.index("complexity")?;
               index.open_cursor()?
           } else if query.last_accessed_after.is_some() {
               // Use last_accessed index
               let index = store.index("last_accessed")?;
               let range = js_sys::IdbKeyRange::lower_bound(&JsValue::from(
                   query.last_accessed_after.unwrap()
               ))?;
               index.open_cursor_with_range(&range)?
           } else {
               // Full table scan
               store.open_cursor()?
           };
           
           let cursor_future = JsFuture::from(cursor_request).await?;
           let mut cursor: Option<IdbCursorWithValue> = 
               if cursor_future.is_null() { None } else { Some(cursor_future.dyn_into()?) };
           
           let mut current_offset = 0u32;
           let offset = query.offset.unwrap_or(0);
           let limit = query.limit.unwrap_or(1000);
           
           while let Some(cursor_ref) = cursor.as_ref() {
               let value = cursor_ref.value()?;
               let concept: ConceptSchema = serde_wasm_bindgen::from_value(value)?;
               
               // Apply filters
               if self.matches_query(&concept, &query) {
                   total_count += 1;
                   
                   // Apply pagination
                   if current_offset >= offset && concepts.len() < limit as usize {
                       concepts.push(concept);
                   }
                   current_offset += 1;
               }
               
               // Move to next
               let continue_request = cursor_ref.continue_()?;
               let continue_future = JsFuture::from(continue_request).await;
               cursor = if continue_future.is_ok() && !continue_future.unwrap().is_null() {
                   Some(continue_future.unwrap().dyn_into()?)
               } else {
                   None
               };
           }
           
           let query_time = js_sys::Date::now() - start_time;
           
           Ok(QueryResult {
               concepts,
               total_count,
               query_time_ms: query_time,
               cache_utilized: false, // TODO: Implement query cache
           })
       }
       
       fn matches_query(&self, concept: &ConceptSchema, query: &ConceptQuery) -> bool {
           // Category filter
           if let Some(category) = &query.category {
               if &concept.metadata.category != category {
                   return false;
               }
           }
           
           // Complexity range filter
           if let Some((min_complexity, max_complexity)) = query.complexity_range {
               let complexity = concept.metadata.complexity;
               if complexity < min_complexity || complexity > max_complexity {
                   return false;
               }
           }
           
           // Content pattern filter
           if let Some(pattern) = &query.content_pattern {
               if !concept.content.to_lowercase().contains(&pattern.to_lowercase()) {
                   return false;
               }
           }
           
           // Tags filter (all must match)
           if !query.tags.is_empty() {
               for required_tag in &query.tags {
                   if !concept.metadata.tags.contains(required_tag) {
                       return false;
                   }
               }
           }
           
           // Connection count filter
           if let Some(min_connections) = query.min_connections {
               if (concept.metadata.connections.len() as u32) < min_connections {
                   return false;
               }
           }
           
           // Timestamp filters
           if let Some(created_after) = query.created_after {
               if concept.created_timestamp < created_after {
                   return false;
               }
           }
           
           if let Some(accessed_after) = query.last_accessed_after {
               if concept.last_accessed < accessed_after {
                   return false;
               }
           }
           
           true
       }
   }
   ```

4. **Implement cache management**:
   ```rust
   impl ConceptStorage {
       fn update_cache(&mut self, id: u32, concept: ConceptSchema) {
           // Implement LRU cache eviction
           if self.cache.len() >= self.cache_size_limit && !self.cache.contains_key(&id) {
               self.evict_lru_cache_entry();
           }
           
           self.cache.insert(id, concept);
       }
       
       fn evict_lru_cache_entry(&mut self) {
           // Find least recently accessed concept
           let mut oldest_id = None;
           let mut oldest_time = f64::INFINITY;
           
           for (id, concept) in &self.cache {
               if concept.last_accessed < oldest_time {
                   oldest_time = concept.last_accessed;
                   oldest_id = Some(*id);
               }
           }
           
           if let Some(id) = oldest_id {
               self.cache.remove(&id);
           }
       }
       
       #[wasm_bindgen]
       pub fn clear_cache(&mut self) {
           self.cache.clear();
           self.cache_hits = 0;
           self.cache_misses = 0;
       }
       
       #[wasm_bindgen]
       pub fn get_cache_stats(&self) -> js_sys::Object {
           let stats = js_sys::Object::new();
           js_sys::Reflect::set(&stats, &"size".into(), &JsValue::from(self.cache.len()))?;
           js_sys::Reflect::set(&stats, &"hits".into(), &JsValue::from(self.cache_hits))?;
           js_sys::Reflect::set(&stats, &"misses".into(), &JsValue::from(self.cache_misses))?;
           
           let hit_rate = if (self.cache_hits + self.cache_misses) > 0 {
               (self.cache_hits as f64) / ((self.cache_hits + self.cache_misses) as f64)
           } else {
               0.0
           };
           js_sys::Reflect::set(&stats, &"hit_rate".into(), &JsValue::from(hit_rate))?;
           
           stats
       }
   }
   ```

5. **Implement similarity search**:
   ```rust
   impl ConceptStorage {
       #[wasm_bindgen]
       pub async fn find_similar_concepts(
           &self, 
           reference_id: u32, 
           similarity_threshold: f32,
           max_results: u32
       ) -> Result<Vec<SimilarConcept>, JsValue> {
           let reference_concept = self.get_concept(reference_id).await?;
           if reference_concept.is_none() {
               return Ok(Vec::new());
           }
           
           let reference = reference_concept.unwrap();
           let mut similar_concepts = Vec::new();
           
           // Query all concepts for similarity comparison
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           let cursor_request = store.open_cursor()?;
           
           let cursor_future = JsFuture::from(cursor_request).await?;
           let mut cursor: Option<IdbCursorWithValue> = 
               if cursor_future.is_null() { None } else { Some(cursor_future.dyn_into()?) };
           
           while let Some(cursor_ref) = cursor.as_ref() {
               let value = cursor_ref.value()?;
               let concept: ConceptSchema = serde_wasm_bindgen::from_value(value)?;
               
               if concept.id != reference_id {
                   let similarity = self.calculate_concept_similarity(&reference, &concept);
                   
                   if similarity >= similarity_threshold {
                       similar_concepts.push(SimilarConcept {
                           concept,
                           similarity_score: similarity,
                       });
                   }
               }
               
               // Move to next
               let continue_request = cursor_ref.continue_()?;
               let continue_future = JsFuture::from(continue_request).await;
               cursor = if continue_future.is_ok() && !continue_future.unwrap().is_null() {
                   Some(continue_future.unwrap().dyn_into()?)
               } else {
                   None
               };
           }
           
           // Sort by similarity and limit results
           similar_concepts.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
           similar_concepts.truncate(max_results as usize);
           
           Ok(similar_concepts)
       }
       
       fn calculate_concept_similarity(&self, a: &ConceptSchema, b: &ConceptSchema) -> f32 {
           let mut similarity = 0.0f32;
           let mut factors = 0;
           
           // Category similarity
           if a.metadata.category == b.metadata.category {
               similarity += 0.3;
           }
           factors += 1;
           
           // Complexity similarity
           let complexity_diff = (a.metadata.complexity - b.metadata.complexity).abs();
           similarity += (1.0 - complexity_diff).max(0.0) * 0.2;
           factors += 1;
           
           // Tag overlap
           let common_tags = a.metadata.tags.iter()
               .filter(|tag| b.metadata.tags.contains(tag))
               .count();
           let total_tags = (a.metadata.tags.len() + b.metadata.tags.len()).max(1);
           similarity += (common_tags as f32 / total_tags as f32) * 0.3;
           factors += 1;
           
           // Content similarity (simple Jaccard similarity on words)
           let words_a: std::collections::HashSet<&str> = a.content.split_whitespace().collect();
           let words_b: std::collections::HashSet<&str> = b.content.split_whitespace().collect();
           let intersection = words_a.intersection(&words_b).count();
           let union = words_a.union(&words_b).count().max(1);
           similarity += (intersection as f32 / union as f32) * 0.2;
           factors += 1;
           
           similarity / factors as f32
       }
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct SimilarConcept {
       pub concept: ConceptSchema,
       pub similarity_score: f32,
   }
   ```

## Expected Outputs
- High-performance concept CRUD operations
- Batch processing capabilities for bulk operations
- Advanced semantic query engine with indexing
- LRU cache management with statistics
- Similarity search with configurable thresholds
- Optimized transaction handling

## Validation
1. Single and batch operations work correctly
2. Cache hit rate exceeds 80% for typical workflows
3. Semantic queries utilize appropriate indexes
4. Similarity search produces relevant results
5. All operations handle errors gracefully

## Next Steps
- Create offline sync queue (micro-phase 9.14)
- Add storage persistence layer (micro-phase 9.15)