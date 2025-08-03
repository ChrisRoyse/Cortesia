# Micro-Phase 9.08: Port Query Processing Methods to WASM

## Objective
Port the complete query processing system with caching to WASM bindings for high-performance concept querying and semantic search.

## Prerequisites
- Completed micro-phases 9.01-9.07 (allocation methods ready)
- QueryProcessor struct foundation in place
- IndexedDB storage integration complete

## Task Description
Implement comprehensive query processing functionality with multi-level caching, semantic search, and real-time concept relationship queries optimized for JavaScript access patterns.

## Specific Actions

1. **Implement core QueryProcessor with WASM bindings**:
   ```rust
   // src/query/processor.rs
   use wasm_bindgen::prelude::*;
   use std::collections::{HashMap, VecDeque};
   use serde::{Serialize, Deserialize};
   
   #[wasm_bindgen]
   pub struct QueryProcessor {
       // Multi-level cache system
       l1_cache: HashMap<String, CachedResult>,
       l2_cache: VecDeque<CachedQuery>,
       cache_size_mb: u32,
       current_cache_size: usize,
       
       // Query optimization
       query_history: Vec<QueryMetrics>,
       semantic_index: HashMap<u32, Vec<f32>>, // concept_id -> embedding
       
       // Performance tracking
       cache_hits: u32,
       cache_misses: u32,
       total_queries: u32,
   }
   
   #[derive(Clone, Serialize, Deserialize)]
   struct CachedResult {
       result: QueryResult,
       timestamp: f64,
       access_count: u32,
       ttl_ms: u32,
   }
   
   #[derive(Clone, Serialize, Deserialize)]
   struct CachedQuery {
       query_hash: String,
       result: QueryResult,
       size_bytes: usize,
   }
   
   #[derive(Clone, Serialize, Deserialize)]
   #[wasm_bindgen]
   pub struct QueryResult {
       concept_ids: Vec<u32>,
       scores: Vec<f32>,
       execution_time_ms: f64,
       total_results: u32,
   }
   
   #[derive(Clone)]
   struct QueryMetrics {
       query_type: String,
       execution_time: f64,
       result_count: u32,
       cache_hit: bool,
   }
   
   #[wasm_bindgen]
   impl QueryProcessor {
       #[wasm_bindgen(constructor)]
       pub fn new(cache_size_mb: u32) -> Self {
           Self {
               l1_cache: HashMap::new(),
               l2_cache: VecDeque::new(),
               cache_size_mb,
               current_cache_size: 0,
               query_history: Vec::new(),
               semantic_index: HashMap::new(),
               cache_hits: 0,
               cache_misses: 0,
               total_queries: 0,
           }
       }
   }
   ```

2. **Implement semantic search and concept queries**:
   ```rust
   #[wasm_bindgen]
   impl QueryProcessor {
       #[wasm_bindgen]
       pub async fn semantic_search(&mut self, 
           query_embedding: Vec<f32>,
           top_k: u32,
           threshold: f32
       ) -> Result<QueryResult, JsValue> {
           let start_time = js_sys::performance::now();
           self.total_queries += 1;
           
           // Generate cache key
           let cache_key = self.generate_semantic_cache_key(&query_embedding, top_k, threshold);
           
           // Check L1 cache first
           if let Some(cached) = self.check_l1_cache(&cache_key) {
               self.cache_hits += 1;
               return Ok(cached.result);
           }
           
           // Check L2 cache
           if let Some(cached) = self.check_l2_cache(&cache_key) {
               self.cache_hits += 1;
               self.promote_to_l1(cache_key, cached.result.clone())?;
               return Ok(cached.result);
           }
           
           self.cache_misses += 1;
           
           // Perform semantic search
           let mut similarities: Vec<(u32, f32)> = Vec::new();
           
           for (&concept_id, embedding) in &self.semantic_index {
               let similarity = self.compute_cosine_similarity(&query_embedding, embedding);
               if similarity >= threshold {
                   similarities.push((concept_id, similarity));
               }
           }
           
           // Sort by similarity and take top-k
           similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
           similarities.truncate(top_k as usize);
           
           let concept_ids: Vec<u32> = similarities.iter().map(|(id, _)| *id).collect();
           let scores: Vec<f32> = similarities.iter().map(|(_, score)| *score).collect();
           
           let execution_time = js_sys::performance::now() - start_time;
           
           let result = QueryResult {
               concept_ids,
               scores,
               execution_time_ms: execution_time,
               total_results: similarities.len() as u32,
           };
           
           // Cache the result
           self.cache_result(cache_key, result.clone()).await?;
           
           // Record metrics
           self.record_query_metrics("semantic_search", execution_time, result.total_results, false);
           
           Ok(result)
       }
       
       #[wasm_bindgen]
       pub async fn find_related_concepts(&mut self,
           concept_id: u32,
           max_distance: u32,
           min_strength: f32
       ) -> Result<QueryResult, JsValue> {
           let start_time = js_sys::performance::now();
           self.total_queries += 1;
           
           let cache_key = format!("related_{}_{}_{}",concept_id, max_distance, min_strength);
           
           // Check cache
           if let Some(cached) = self.check_l1_cache(&cache_key) {
               self.cache_hits += 1;
               return Ok(cached.result);
           }
           
           self.cache_misses += 1;
           
           // Perform graph traversal to find related concepts
           let mut related_concepts: Vec<(u32, f32)> = Vec::new();
           let mut visited = std::collections::HashSet::new();
           let mut queue = VecDeque::new();
           
           queue.push_back((concept_id, 0u32, 1.0f32)); // (id, distance, strength)
           
           while let Some((current_id, distance, strength)) = queue.pop_front() {
               if distance > max_distance || strength < min_strength || visited.contains(&current_id) {
                   continue;
               }
               
               visited.insert(current_id);
               
               if current_id != concept_id {
                   related_concepts.push((current_id, strength));
               }
               
               // Add connected concepts to queue (simplified - would use actual graph)
               // This would integrate with the sparse connections system
               for connected_id in self.get_connected_concepts(current_id).await? {
                   let connection_strength = self.get_connection_strength(current_id, connected_id).await?;
                   if !visited.contains(&connected_id) {
                       queue.push_back((connected_id, distance + 1, strength * connection_strength));
                   }
               }
           }
           
           // Sort by strength
           related_concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
           
           let concept_ids: Vec<u32> = related_concepts.iter().map(|(id, _)| *id).collect();
           let scores: Vec<f32> = related_concepts.iter().map(|(_, score)| *score).collect();
           
           let execution_time = js_sys::performance::now() - start_time;
           
           let result = QueryResult {
               concept_ids,
               scores,
               execution_time_ms: execution_time,
               total_results: related_concepts.len() as u32,
           };
           
           self.cache_result(cache_key, result.clone()).await?;
           self.record_query_metrics("related_concepts", execution_time, result.total_results, false);
           
           Ok(result)
       }
       
       fn compute_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
           if a.len() != b.len() {
               return 0.0;
           }
           
           let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
           let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
           let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
           
           if norm_a == 0.0 || norm_b == 0.0 {
               0.0
           } else {
               dot_product / (norm_a * norm_b)
           }
       }
       
       async fn get_connected_concepts(&self, _concept_id: u32) -> Result<Vec<u32>, JsValue> {
           // Placeholder - would integrate with sparse connections
           Ok(vec![])
       }
       
       async fn get_connection_strength(&self, _from: u32, _to: u32) -> Result<f32, JsValue> {
           // Placeholder - would get actual connection strength
           Ok(0.5)
       }
   }
   ```

3. **Implement advanced caching with LRU eviction**:
   ```rust
   #[wasm_bindgen]
   impl QueryProcessor {
       fn check_l1_cache(&mut self, cache_key: &str) -> Option<CachedResult> {
           if let Some(cached) = self.l1_cache.get_mut(cache_key) {
               // Check TTL
               let now = js_sys::Date::now();
               if now - cached.timestamp > cached.ttl_ms as f64 {
                   self.l1_cache.remove(cache_key);
                   return None;
               }
               
               cached.access_count += 1;
               Some(cached.clone())
           } else {
               None
           }
       }
       
       fn check_l2_cache(&self, cache_key: &str) -> Option<CachedQuery> {
           self.l2_cache.iter()
               .find(|cached| cached.query_hash == cache_key)
               .cloned()
       }
       
       async fn cache_result(&mut self, cache_key: String, result: QueryResult) -> Result<(), JsValue> {
           let cached_result = CachedResult {
               result: result.clone(),
               timestamp: js_sys::Date::now(),
               access_count: 1,
               ttl_ms: 300000, // 5 minutes
           };
           
           // Add to L1 cache
           self.l1_cache.insert(cache_key.clone(), cached_result);
           
           // Estimate size and add to L2 if L1 is full
           let estimated_size = std::mem::size_of::<QueryResult>() + 
                               result.concept_ids.len() * 4 + 
                               result.scores.len() * 4;
           
           self.current_cache_size += estimated_size;
           
           // Evict if cache is full
           if self.current_cache_size > (self.cache_size_mb as usize * 1024 * 1024) {
               self.evict_lru_entries().await?;
           }
           
           Ok(())
       }
       
       fn promote_to_l1(&mut self, cache_key: String, result: QueryResult) -> Result<(), JsValue> {
           let cached_result = CachedResult {
               result,
               timestamp: js_sys::Date::now(),
               access_count: 1,
               ttl_ms: 300000,
           };
           
           self.l1_cache.insert(cache_key, cached_result);
           Ok(())
       }
       
       async fn evict_lru_entries(&mut self) -> Result<(), JsValue> {
           // Sort L1 cache entries by access count and timestamp
           let mut entries: Vec<_> = self.l1_cache.iter().collect();
           entries.sort_by(|a, b| {
               let access_cmp = a.1.access_count.cmp(&b.1.access_count);
               if access_cmp == std::cmp::Ordering::Equal {
                   a.1.timestamp.partial_cmp(&b.1.timestamp).unwrap()
               } else {
                   access_cmp
               }
           });
           
           // Remove least recently used entries
           let target_size = (self.cache_size_mb as usize * 1024 * 1024) / 2; // Keep 50%
           let mut current_size = self.current_cache_size;
           
           for (key, _) in entries {
               if current_size <= target_size {
                   break;
               }
               
               if let Some(removed) = self.l1_cache.remove(*key) {
                   let size = std::mem::size_of::<QueryResult>() +
                            removed.result.concept_ids.len() * 4 +
                            removed.result.scores.len() * 4;
                   current_size -= size;
               }
           }
           
           self.current_cache_size = current_size;
           Ok(())
       }
       
       fn generate_semantic_cache_key(&self, embedding: &[f32], top_k: u32, threshold: f32) -> String {
           // Create a hash of the embedding for cache key
           let hash = embedding.iter()
               .map(|&f| (f * 10000.0) as i32)
               .fold(0u64, |acc, i| acc.wrapping_mul(31).wrapping_add(i as u64));
           
           format!("semantic_{}_{}_{}", hash, top_k, (threshold * 1000.0) as i32)
       }
       
       fn record_query_metrics(&mut self, query_type: &str, execution_time: f64, result_count: u32, cache_hit: bool) {
           let metrics = QueryMetrics {
               query_type: query_type.to_string(),
               execution_time,
               result_count,
               cache_hit,
           };
           
           self.query_history.push(metrics);
           
           // Keep only recent metrics
           if self.query_history.len() > 10000 {
               self.query_history.remove(0);
           }
       }
   }
   ```

4. **Implement concept indexing and batch operations**:
   ```rust
   #[wasm_bindgen]
   impl QueryProcessor {
       #[wasm_bindgen]
       pub fn add_concept_embedding(&mut self, concept_id: u32, embedding: Vec<f32>) -> Result<(), JsValue> {
           if embedding.is_empty() {
               return Err(JsValue::from_str("Embedding cannot be empty"));
           }
           
           self.semantic_index.insert(concept_id, embedding);
           Ok(())
       }
       
       #[wasm_bindgen]
       pub fn remove_concept_embedding(&mut self, concept_id: u32) -> bool {
           self.semantic_index.remove(&concept_id).is_some()
       }
       
       #[wasm_bindgen]
       pub fn clear_cache(&mut self) {
           self.l1_cache.clear();
           self.l2_cache.clear();
           self.current_cache_size = 0;
           
           web_sys::console::log_1(&JsValue::from_str("Query cache cleared"));
       }
       
       #[wasm_bindgen]
       pub async fn batch_semantic_search(&mut self,
           queries: js_sys::Array,
           top_k: u32,
           threshold: f32
       ) -> Result<js_sys::Array, JsValue> {
           let results = js_sys::Array::new();
           
           for i in 0..queries.length() {
               let query_obj = queries.get(i);
               let embedding_array = js_sys::Reflect::get(&query_obj, &"embedding".into())?;
               
               if let Ok(embedding_vec) = embedding_array.dyn_into::<js_sys::Array>() {
                   let mut embedding = Vec::new();
                   for j in 0..embedding_vec.length() {
                       if let Ok(val) = embedding_vec.get(j).as_f64() {
                           embedding.push(val as f32);
                       }
                   }
                   
                   let result = self.semantic_search(embedding, top_k, threshold).await?;
                   results.push(&JsValue::from(result));
               }
           }
           
           Ok(results)
       }
   }
   ```

5. **Implement query statistics and performance monitoring**:
   ```rust
   #[wasm_bindgen]
   impl QueryProcessor {
       #[wasm_bindgen]
       pub fn get_cache_hit_rate(&self) -> f32 {
           if self.total_queries == 0 {
               0.0
           } else {
               (self.cache_hits as f32) / (self.total_queries as f32)
           }
       }
       
       #[wasm_bindgen]
       pub fn get_cache_stats(&self) -> js_sys::Object {
           let stats = js_sys::Object::new();
           
           js_sys::Reflect::set(&stats, &"totalQueries".into(), &self.total_queries.into()).unwrap();
           js_sys::Reflect::set(&stats, &"cacheHits".into(), &self.cache_hits.into()).unwrap();
           js_sys::Reflect::set(&stats, &"cacheMisses".into(), &self.cache_misses.into()).unwrap();
           js_sys::Reflect::set(&stats, &"hitRate".into(), &self.get_cache_hit_rate().into()).unwrap();
           js_sys::Reflect::set(&stats, &"l1CacheSize".into(), &self.l1_cache.len().into()).unwrap();
           js_sys::Reflect::set(&stats, &"l2CacheSize".into(), &self.l2_cache.len().into()).unwrap();
           js_sys::Reflect::set(&stats, &"currentCacheSizeMB".into(), &((self.current_cache_size / 1024 / 1024) as u32).into()).unwrap();
           js_sys::Reflect::set(&stats, &"maxCacheSizeMB".into(), &self.cache_size_mb.into()).unwrap();
           
           stats
       }
       
       #[wasm_bindgen]
       pub fn get_query_performance_metrics(&self) -> js_sys::Array {
           let metrics = js_sys::Array::new();
           
           for metric in &self.query_history {
               let obj = js_sys::Object::new();
               js_sys::Reflect::set(&obj, &"queryType".into(), &metric.query_type.clone().into()).unwrap();
               js_sys::Reflect::set(&obj, &"executionTime".into(), &metric.execution_time.into()).unwrap();
               js_sys::Reflect::set(&obj, &"resultCount".into(), &metric.result_count.into()).unwrap();
               js_sys::Reflect::set(&obj, &"cacheHit".into(), &metric.cache_hit.into()).unwrap();
               metrics.push(&obj);
           }
           
           metrics
       }
       
       #[wasm_bindgen]
       pub fn get_semantic_index_size(&self) -> u32 {
           self.semantic_index.len() as u32
       }
   }
   ```

## Expected Outputs
- Complete QueryProcessor with multi-level caching system
- Semantic search with cosine similarity computation
- Graph-based concept relationship queries
- Batch query processing capabilities
- Comprehensive performance monitoring and statistics

## Validation
1. Semantic search returns relevant results with proper scoring
2. Cache system achieves high hit rates and proper LRU eviction
3. Related concept queries traverse graph relationships correctly
4. Batch operations process multiple queries efficiently
5. Performance metrics accurately track query execution

## Next Steps
- Setup SIMD intrinsics for query optimization (micro-phase 9.09)
- Integrate neural processing optimizations (micro-phase 9.10)