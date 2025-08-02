# Task 06: TTFS Encoding Integration

**Estimated Time**: 25-30 minutes  
**Dependencies**: 05_basic_crud_operations.md, Phase 2 TTFS components  
**Stage**: Neural Integration  

## Objective
Integrate Time-to-First-Spike (TTFS) encoding from Phase 2 with the knowledge graph to enable neural-guided node placement and retrieval.

## Specific Requirements

### 1. TTFS Integration Points
- Connect to Phase 2 TTFS encoder components
- Store TTFS encodings in concept nodes
- Use TTFS for similarity calculations
- Support TTFS-based queries and clustering

### 2. Neural-Guided Operations
- Convert text content to TTFS encodings
- Use TTFS for node placement decisions
- Implement TTFS similarity metrics
- Support TTFS-based retrieval

### 3. Performance Optimization
- Cache TTFS encodings for reuse
- Batch TTFS encoding operations
- Optimize TTFS similarity calculations

## Implementation Steps

### 1. Create TTFS Integration Service
```rust
// src/integration/ttfs_integration.rs
use crate::phase2::ttfs::{TTFSEncoder, TTFSSpikePattern};

pub struct TTFSIntegrationService {
    ttfs_encoder: Arc<TTFSEncoder>,
    encoding_cache: Arc<RwLock<LRUCache<String, f32>>>,
    similarity_cache: Arc<RwLock<LRUCache<(String, String), f32>>>,
    performance_monitor: Arc<PerformanceMonitor>,
}

impl TTFSIntegrationService {
    pub async fn new(ttfs_encoder: Arc<TTFSEncoder>) -> Result<Self, IntegrationError> {
        Ok(Self {
            ttfs_encoder,
            encoding_cache: Arc::new(RwLock::new(LRUCache::new(10000))),
            similarity_cache: Arc::new(RwLock::new(LRUCache::new(50000))),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        })
    }
    
    pub async fn encode_content(
        &self,
        content: &str,
    ) -> Result<f32, TTFSIntegrationError> {
        let encoding_start = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_content_hash(content);
        if let Some(cached_encoding) = self.encoding_cache.read().await.get(&cache_key) {
            return Ok(*cached_encoding);
        }
        
        // Generate TTFS encoding using Phase 2 encoder
        let spike_pattern = self.ttfs_encoder.encode_text(content).await?;
        let ttfs_encoding = spike_pattern.first_spike_time;
        
        // Cache the result
        self.encoding_cache.write().await.put(cache_key, ttfs_encoding);
        
        // Record performance metrics
        let encoding_time = encoding_start.elapsed();
        self.performance_monitor.record_encoding_time(encoding_time).await;
        
        Ok(ttfs_encoding)
    }
    
    pub async fn batch_encode_content(
        &self,
        content_items: Vec<&str>,
    ) -> Result<Vec<f32>, TTFSIntegrationError> {
        let batch_start = Instant::now();
        let mut encodings = Vec::with_capacity(content_items.len());
        
        // Check cache for all items first
        let mut uncached_items = Vec::new();
        let mut uncached_indices = Vec::new();
        
        for (i, content) in content_items.iter().enumerate() {
            let cache_key = self.generate_content_hash(content);
            if let Some(cached_encoding) = self.encoding_cache.read().await.get(&cache_key) {
                encodings.push(*cached_encoding);
            } else {
                uncached_items.push(*content);
                uncached_indices.push(i);
                encodings.push(0.0); // Placeholder
            }
        }
        
        // Batch encode uncached items
        if !uncached_items.is_empty() {
            let batch_encodings = self.ttfs_encoder.batch_encode_text(uncached_items).await?;
            
            // Update cache and results
            for (i, encoding) in batch_encodings.iter().enumerate() {
                let original_index = uncached_indices[i];
                let cache_key = self.generate_content_hash(content_items[original_index]);
                
                self.encoding_cache.write().await.put(cache_key, *encoding);
                encodings[original_index] = *encoding;
            }
        }
        
        let batch_time = batch_start.elapsed();
        self.performance_monitor.record_batch_encoding_time(batch_time, content_items.len()).await;
        
        Ok(encodings)
    }
    
    pub async fn calculate_ttfs_similarity(
        &self,
        encoding1: f32,
        encoding2: f32,
    ) -> Result<f32, TTFSIntegrationError> {
        // Use inverse of time difference for similarity
        // Closer spike times = higher similarity
        let time_diff = (encoding1 - encoding2).abs();
        let similarity = 1.0 / (1.0 + time_diff);
        
        Ok(similarity)
    }
    
    pub async fn find_ttfs_similar_concepts(
        &self,
        target_encoding: f32,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<TTFSSimilarConcept>, TTFSIntegrationError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept)
            WHERE c.ttfs_encoding IS NOT NULL
            WITH c, abs(c.ttfs_encoding - $target_encoding) as time_diff
            WHERE 1.0 / (1.0 + time_diff) >= $threshold
            RETURN c.id as concept_id, 
                   c.name as concept_name,
                   c.ttfs_encoding as encoding,
                   1.0 / (1.0 + time_diff) as similarity
            ORDER BY similarity DESC
            LIMIT $limit
        "#;
        
        let parameters = hashmap![
            "target_encoding".to_string() => target_encoding.into(),
            "threshold".to_string() => threshold.into(),
            "limit".to_string() => (limit as i64).into()
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut similar_concepts = Vec::new();
        for record in result {
            similar_concepts.push(TTFSSimilarConcept {
                concept_id: record.get("concept_id")?,
                concept_name: record.get("concept_name")?,
                encoding: record.get("encoding")?,
                similarity: record.get("similarity")?,
            });
        }
        
        Ok(similar_concepts)
    }
}
```

### 2. Integrate TTFS with Concept Creation
```rust
// src/integration/ttfs_concept_integration.rs
pub struct TTFSConceptIntegration {
    ttfs_service: Arc<TTFSIntegrationService>,
    concept_crud: Arc<NodeCrudService<ConceptNode>>,
}

impl TTFSConceptIntegration {
    pub async fn create_concept_with_ttfs(
        &self,
        name: &str,
        content: &str,
        concept_type: &str,
    ) -> Result<ConceptNode, ConceptCreationError> {
        // Generate TTFS encoding for the content
        let ttfs_encoding = self.ttfs_service.encode_content(content).await?;
        
        // Create concept with TTFS encoding
        let concept = ConceptNodeBuilder::new(name, concept_type)
            .with_ttfs_encoding(ttfs_encoding)
            .with_content_hash(self.generate_content_hash(content))
            .build()?;
        
        // Store in knowledge graph
        let concept_id = self.concept_crud.create(&concept).await?;
        
        // Update the concept with the actual ID
        let mut final_concept = concept;
        final_concept.id = concept_id;
        
        Ok(final_concept)
    }
    
    pub async fn update_concept_ttfs(
        &self,
        concept_id: &str,
        new_content: &str,
    ) -> Result<(), ConceptUpdateError> {
        // Generate new TTFS encoding
        let new_ttfs_encoding = self.ttfs_service.encode_content(new_content).await?;
        
        // Retrieve existing concept
        let mut concept = self.concept_crud.read(concept_id).await?
            .ok_or(ConceptUpdateError::ConceptNotFound(concept_id.to_string()))?;
        
        // Update TTFS encoding
        concept.ttfs_encoding = Some(new_ttfs_encoding);
        concept.last_accessed = Utc::now();
        
        // Store updated concept
        self.concept_crud.update(concept_id, &concept).await?;
        
        Ok(())
    }
}
```

### 3. Implement TTFS-Based Queries
```rust
// src/integration/ttfs_queries.rs
pub struct TTFSQueryService {
    ttfs_service: Arc<TTFSIntegrationService>,
    connection_manager: Arc<Neo4jConnectionManager>,
}

impl TTFSQueryService {
    pub async fn query_by_ttfs_similarity(
        &self,
        query_content: &str,
        similarity_threshold: f32,
        limit: usize,
    ) -> Result<Vec<TTFSQueryResult>, TTFSQueryError> {
        // Generate TTFS encoding for query
        let query_encoding = self.ttfs_service.encode_content(query_content).await?;
        
        // Find similar concepts
        let similar_concepts = self.ttfs_service.find_ttfs_similar_concepts(
            query_encoding,
            similarity_threshold,
            limit,
        ).await?;
        
        // Enrich results with full concept data
        let mut results = Vec::new();
        for similar in similar_concepts {
            if let Some(concept) = self.concept_crud.read(&similar.concept_id).await? {
                results.push(TTFSQueryResult {
                    concept,
                    ttfs_similarity: similar.similarity,
                    encoding_difference: (query_encoding - similar.encoding).abs(),
                });
            }
        }
        
        Ok(results)
    }
    
    pub async fn cluster_concepts_by_ttfs(
        &self,
        concept_ids: Vec<String>,
        cluster_threshold: f32,
    ) -> Result<Vec<TTFSCluster>, TTFSQueryError> {
        // Retrieve TTFS encodings for all concepts
        let mut concept_encodings = Vec::new();
        for concept_id in &concept_ids {
            if let Some(concept) = self.concept_crud.read(concept_id).await? {
                if let Some(encoding) = concept.ttfs_encoding {
                    concept_encodings.push((concept_id.clone(), encoding));
                }
            }
        }
        
        // Perform clustering based on TTFS similarity
        let clusters = self.perform_ttfs_clustering(concept_encodings, cluster_threshold).await?;
        
        Ok(clusters)
    }
    
    async fn perform_ttfs_clustering(
        &self,
        encodings: Vec<(String, f32)>,
        threshold: f32,
    ) -> Result<Vec<TTFSCluster>, TTFSQueryError> {
        let mut clusters = Vec::new();
        let mut assigned = vec![false; encodings.len()];
        
        for i in 0..encodings.len() {
            if assigned[i] {
                continue;
            }
            
            let mut cluster = TTFSCluster {
                center_encoding: encodings[i].1,
                concept_ids: vec![encodings[i].0.clone()],
                average_similarity: 1.0,
            };
            
            assigned[i] = true;
            
            // Find similar concepts for this cluster
            for j in (i + 1)..encodings.len() {
                if assigned[j] {
                    continue;
                }
                
                let similarity = self.ttfs_service.calculate_ttfs_similarity(
                    encodings[i].1,
                    encodings[j].1,
                ).await?;
                
                if similarity >= threshold {
                    cluster.concept_ids.push(encodings[j].0.clone());
                    assigned[j] = true;
                }
            }
            
            clusters.push(cluster);
        }
        
        Ok(clusters)
    }
}
```

### 4. Add Performance Monitoring
```rust
// src/integration/ttfs_performance.rs
pub struct TTFSPerformanceMonitor {
    encoding_times: Arc<RwLock<Vec<Duration>>>,
    similarity_times: Arc<RwLock<Vec<Duration>>>,
    query_times: Arc<RwLock<Vec<Duration>>>,
    cache_stats: Arc<RwLock<CacheStats>>,
}

impl TTFSPerformanceMonitor {
    pub async fn record_encoding_time(&self, duration: Duration) {
        self.encoding_times.write().await.push(duration);
    }
    
    pub async fn get_encoding_stats(&self) -> EncodingStats {
        let times = self.encoding_times.read().await;
        
        if times.is_empty() {
            return EncodingStats::default();
        }
        
        let total: Duration = times.iter().sum();
        let average = total / times.len() as u32;
        let min = *times.iter().min().unwrap();
        let max = *times.iter().max().unwrap();
        
        EncodingStats {
            average_time: average,
            min_time: min,
            max_time: max,
            total_encodings: times.len(),
        }
    }
    
    pub async fn get_cache_efficiency(&self) -> f32 {
        let stats = self.cache_stats.read().await;
        if stats.total_requests == 0 {
            return 0.0;
        }
        
        stats.cache_hits as f32 / stats.total_requests as f32
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] TTFS encoding integration works with Phase 2 components
- [ ] Concept creation includes TTFS encoding generation
- [ ] TTFS similarity calculations are accurate
- [ ] TTFS-based queries return relevant results
- [ ] Batch encoding operations work efficiently

### Performance Requirements
- [ ] Single TTFS encoding generation < 10ms
- [ ] TTFS similarity calculation < 1ms
- [ ] Batch encoding throughput > 100 items/second
- [ ] Cache hit rate > 70% for repeated content

### Testing Requirements
- [ ] Unit tests for TTFS integration components
- [ ] Integration tests with Phase 2 TTFS encoder
- [ ] Performance tests for encoding and similarity
- [ ] Accuracy tests for similarity calculations

## Validation Steps

1. **Test TTFS encoding integration**:
   ```rust
   let ttfs_service = TTFSIntegrationService::new(ttfs_encoder).await?;
   let encoding = ttfs_service.encode_content("test content").await?;
   ```

2. **Test concept creation with TTFS**:
   ```rust
   let concept = ttfs_concept_integration.create_concept_with_ttfs(
       "Test Concept", "test content", "Entity"
   ).await?;
   ```

3. **Run integration tests**:
   ```bash
   cargo test ttfs_integration_tests
   ```

## Files to Create/Modify
- `src/integration/ttfs_integration.rs` - Main TTFS integration service
- `src/integration/ttfs_concept_integration.rs` - Concept-TTFS integration
- `src/integration/ttfs_queries.rs` - TTFS-based query service
- `src/integration/ttfs_performance.rs` - Performance monitoring
- `tests/integration/ttfs_tests.rs` - Integration test suite

## Error Handling
- Phase 2 component connectivity issues
- TTFS encoding failures
- Cache inconsistencies
- Performance degradation detection
- Invalid similarity calculations

## Success Metrics
- TTFS encoding success rate: 100%
- Integration with Phase 2: Seamless
- Performance requirements met
- Cache efficiency > 70%

## Next Task
Upon completion, proceed to **07_cortical_column_integration.md** to connect to Phase 2 cortical columns.