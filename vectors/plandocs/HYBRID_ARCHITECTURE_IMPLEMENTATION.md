# Hybrid Architecture Implementation Guide

## Overview

This guide details the implementation of a multi-tiered hybrid search system that combines exact matching, fuzzy search, local embeddings, and remote APIs to achieve maximum accuracy while maintaining performance and cost efficiency.

## Architecture Layers

### Layer 1: Lightning-Fast Exact Match Cache

```rust
pub struct ExactMatchLayer {
    // In-memory hash table for O(1) lookups
    content_hash_map: DashMap<u64, Vec<DocumentId>>,
    
    // Bloom filter for quick non-existence checks
    bloom_filter: BloomFilter,
    
    // Content normalization for better matching
    normalizer: ContentNormalizer,
}

impl ExactMatchLayer {
    pub fn search(&self, query: &str) -> Option<Vec<SearchResult>> {
        // Normalize query (lowercase, remove whitespace variations)
        let normalized = self.normalizer.normalize(query);
        
        // Quick bloom filter check
        if !self.bloom_filter.contains(&normalized) {
            return None;
        }
        
        // Hash lookup
        let hash = xxhash64(&normalized);
        self.content_hash_map.get(&hash)
            .map(|doc_ids| {
                doc_ids.iter()
                    .map(|id| SearchResult {
                        document_id: *id,
                        score: 1.0, // Perfect match
                        match_type: MatchType::Exact,
                    })
                    .collect()
            })
    }
}
```

### Layer 2: Fuzzy String Matching

```rust
pub struct FuzzyMatchLayer {
    // Trigram index for fast fuzzy search
    trigram_index: TrigramIndex,
    
    // Edit distance calculator
    edit_distance: DamerauLevenshtein,
    
    // Fuzzy matching thresholds
    config: FuzzyConfig,
}

impl FuzzyMatchLayer {
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        // Get candidates using trigram similarity
        let candidates = self.trigram_index.find_similar(query, self.config.trigram_threshold);
        
        // Rank by edit distance
        let mut results = candidates.into_iter()
            .map(|candidate| {
                let distance = self.edit_distance.distance(query, &candidate.content);
                let score = 1.0 - (distance as f32 / query.len().max(candidate.content.len()) as f32);
                
                SearchResult {
                    document_id: candidate.id,
                    score: score * 0.9, // Slightly lower than exact match
                    match_type: MatchType::Fuzzy(distance),
                }
            })
            .filter(|r| r.score >= self.config.min_score)
            .collect::<Vec<_>>();
            
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(self.config.max_results);
        results
    }
}
```

### Layer 3: Local Embedding Models

```rust
pub struct LocalEmbeddingLayer {
    // Multiple local models for different content types
    models: HashMap<ContentType, Box<dyn LocalModel>>,
    
    // Unified vector index
    vector_index: HNSWIndex,
    
    // Model selector
    selector: ModelSelector,
}

// Recommended local models
pub fn create_local_models() -> HashMap<ContentType, Box<dyn LocalModel>> {
    let mut models = HashMap::new();
    
    // MiniLM-L6-v2: Fast, general purpose (384 dims, 22M params)
    models.insert(
        ContentType::General,
        Box::new(MiniLMModel::load("sentence-transformers/all-MiniLM-L6-v2"))
    );
    
    // CodeBERT-small: Code understanding (512 dims, 84M params)
    models.insert(
        ContentType::Code,
        Box::new(CodeBERTSmall::load("microsoft/codebert-base"))
    );
    
    // MPNet-base: High quality general embeddings (768 dims, 110M params)
    models.insert(
        ContentType::Documentation,
        Box::new(MPNetModel::load("sentence-transformers/all-mpnet-base-v2"))
    );
    
    models
}

impl LocalEmbeddingLayer {
    pub async fn search(&self, query: &str, content_type: ContentType) -> Vec<SearchResult> {
        // Select appropriate model
        let model = self.selector.select_model(query, content_type, &self.models);
        
        // Generate query embedding
        let query_embedding = model.encode(query).await;
        
        // Project to unified space
        let unified_embedding = self.project_to_unified_space(query_embedding, model.dimensions());
        
        // Search in vector index
        let neighbors = self.vector_index.search(&unified_embedding, 50);
        
        // Convert to search results
        neighbors.into_iter()
            .map(|neighbor| SearchResult {
                document_id: neighbor.id,
                score: neighbor.similarity * 0.8, // Local models slightly less accurate
                match_type: MatchType::Semantic(model.name()),
            })
            .collect()
    }
}
```

### Layer 4: Remote API Models

```rust
pub struct RemoteAPILayer {
    // API clients with built-in retry and circuit breakers
    clients: HashMap<ModelType, Box<dyn APIClient>>,
    
    // Cost tracking
    cost_tracker: CostTracker,
    
    // Rate limiting
    rate_limiters: HashMap<ModelType, RateLimiter>,
    
    // Cache for expensive API calls
    api_cache: TTLCache<String, Vec<f32>>,
}

impl RemoteAPILayer {
    pub async fn search(&self, query: &str, content_type: ContentType, budget: f32) -> Option<Vec<SearchResult>> {
        // Check budget
        if !self.cost_tracker.has_budget(budget) {
            return None;
        }
        
        // Check cache first
        if let Some(cached_embedding) = self.api_cache.get(query) {
            return Some(self.search_with_embedding(cached_embedding).await);
        }
        
        // Select best API for content type
        let api_client = self.select_api(content_type, budget);
        
        // Check rate limit
        if !self.rate_limiters[&api_client.model_type()].try_acquire() {
            return None;
        }
        
        // Make API call with circuit breaker
        match api_client.generate_embedding(query).await {
            Ok(embedding) => {
                // Cache the expensive result
                self.api_cache.insert(query.to_string(), embedding.clone(), Duration::from_secs(3600));
                
                // Track cost
                self.cost_tracker.record_usage(api_client.model_type(), query.len());
                
                // Search with embedding
                Some(self.search_with_embedding(&embedding).await)
            }
            Err(e) => {
                log::warn!("API call failed: {}", e);
                None
            }
        }
    }
}
```

### Layer 5: Result Fusion

```rust
pub struct ResultFusion {
    // Fusion algorithm
    fusion_method: FusionMethod,
    
    // Learning to rank model
    ranker: LearningToRankModel,
    
    // Feature extractors
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
}

pub enum FusionMethod {
    ReciprocalRankFusion { k: f32 },
    WeightedCombination { weights: HashMap<MatchType, f32> },
    LearnedFusion { model: Box<dyn FusionModel> },
}

impl ResultFusion {
    pub fn fuse(&self, results_by_layer: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
        match &self.fusion_method {
            FusionMethod::ReciprocalRankFusion { k } => {
                self.reciprocal_rank_fusion(results_by_layer, *k)
            }
            FusionMethod::WeightedCombination { weights } => {
                self.weighted_combination(results_by_layer, weights)
            }
            FusionMethod::LearnedFusion { model } => {
                self.learned_fusion(results_by_layer, model)
            }
        }
    }
    
    fn reciprocal_rank_fusion(&self, results_by_layer: Vec<Vec<SearchResult>>, k: f32) -> Vec<SearchResult> {
        let mut doc_scores: HashMap<DocumentId, f32> = HashMap::new();
        
        for layer_results in results_by_layer {
            for (rank, result) in layer_results.iter().enumerate() {
                let score = 1.0 / (k + rank as f32 + 1.0);
                *doc_scores.entry(result.document_id).or_insert(0.0) += score;
            }
        }
        
        let mut fused_results: Vec<_> = doc_scores.into_iter()
            .map(|(doc_id, score)| SearchResult {
                document_id: doc_id,
                score,
                match_type: MatchType::Fused,
            })
            .collect();
            
        fused_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        fused_results
    }
}
```

## Implementation Strategy

### Phase 1: Core Search Infrastructure (Days 1-5)

```rust
// Day 1-2: Exact and fuzzy matching
pub fn implement_basic_search() {
    // 1. Create content indexing pipeline
    let indexer = ContentIndexer::new()
        .with_exact_match_index()
        .with_trigram_index()
        .with_bloom_filter(expected_items: 1_000_000);
    
    // 2. Index existing codebase
    indexer.index_directory("./src", ProgressBar::new());
    
    // 3. Implement search layers
    let exact_layer = ExactMatchLayer::new(&indexer.exact_index());
    let fuzzy_layer = FuzzyMatchLayer::new(&indexer.trigram_index());
}

// Day 3-5: Local embedding models
pub async fn implement_embedding_search() {
    // 1. Download and load models
    let models = download_local_models(&[
        "sentence-transformers/all-MiniLM-L6-v2",
        "microsoft/codebert-base",
    ]).await;
    
    // 2. Create vector index
    let vector_index = HNSWIndex::new(HNSWConfig {
        dimensions: 512, // Unified space
        ef_construction: 200,
        m: 16,
    });
    
    // 3. Index codebase with embeddings
    let embedding_indexer = EmbeddingIndexer::new(models, vector_index);
    embedding_indexer.index_directory("./src").await;
}
```

### Phase 2: Advanced Features (Days 6-10)

```rust
// Day 6-7: Query understanding
pub struct QueryProcessor {
    intent_classifier: IntentClassifier,
    entity_extractor: EntityExtractor,
    query_expander: QueryExpander,
    
    pub fn process(&self, query: &str) -> ProcessedQuery {
        let intent = self.intent_classifier.classify(query);
        let entities = self.entity_extractor.extract(query);
        
        // Expand query based on intent
        let expanded = match intent {
            QueryIntent::FindFunction => self.expand_function_query(query, &entities),
            QueryIntent::FindUsage => self.expand_usage_query(query, &entities),
            QueryIntent::FindSimilar => self.expand_similarity_query(query, &entities),
            _ => vec![query.to_string()],
        };
        
        ProcessedQuery {
            original: query.to_string(),
            intent,
            entities,
            expanded_queries: expanded,
        }
    }
}

// Day 8-10: Result ranking and learning
pub struct RankingOptimizer {
    click_log: ClickLog,
    ranker: LGBMRanker,
    
    pub fn optimize_ranking(&mut self) {
        // Collect training data from click logs
        let training_data = self.click_log.get_training_examples();
        
        // Extract features
        let features = training_data.iter()
            .map(|example| self.extract_features(&example.query, &example.result))
            .collect();
            
        // Train ranker
        self.ranker.fit(&features, &training_data.labels());
    }
}
```

### Phase 3: Production Features (Days 11-15)

```rust
// Day 11-12: Monitoring and observability
pub struct SearchMonitoring {
    metrics: MetricsCollector,
    
    pub fn track_search(&self, query: &str, results: &[SearchResult], latency: Duration) {
        self.metrics.histogram("search.latency", latency.as_millis() as f64);
        self.metrics.gauge("search.results_count", results.len() as f64);
        
        // Track search success (results with score > 0.7)
        let high_quality_results = results.iter().filter(|r| r.score > 0.7).count();
        self.metrics.gauge("search.high_quality_ratio", 
            high_quality_results as f64 / results.len().max(1) as f64);
    }
}

// Day 13-15: API integration with fallbacks
pub struct SmartAPIClient {
    primary: Box<dyn APIClient>,
    fallback: Box<dyn APIClient>,
    local_fallback: Box<dyn LocalModel>,
    circuit_breaker: CircuitBreaker,
    
    pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Try primary API
        if self.circuit_breaker.is_open() {
            return self.use_fallback(text).await;
        }
        
        match timeout(Duration::from_millis(200), self.primary.embed(text)).await {
            Ok(Ok(embedding)) => {
                self.circuit_breaker.record_success();
                Ok(embedding)
            }
            Ok(Err(e)) | Err(_) => {
                self.circuit_breaker.record_failure();
                self.use_fallback(text).await
            }
        }
    }
    
    async fn use_fallback(&self, text: &str) -> Result<Vec<f32>> {
        // Try fallback API
        if let Ok(embedding) = timeout(Duration::from_millis(200), self.fallback.embed(text)).await {
            return embedding;
        }
        
        // Fall back to local model
        Ok(self.local_fallback.embed(text).await)
    }
}
```

## Performance Optimizations

### 1. Intelligent Caching

```rust
pub struct MultiTierCache {
    // L1: In-process LRU cache (microsecond access)
    l1_cache: Arc<Mutex<LruCache<String, CachedResult>>>,
    
    // L2: Shared memory cache (sub-millisecond access)  
    l2_cache: SharedMemoryCache,
    
    // L3: Redis cache (millisecond access)
    l3_cache: RedisCache,
    
    pub async fn get(&self, key: &str) -> Option<CachedResult> {
        // Check L1
        if let Some(result) = self.l1_cache.lock().unwrap().get(key) {
            return Some(result.clone());
        }
        
        // Check L2
        if let Some(result) = self.l2_cache.get(key) {
            self.l1_cache.lock().unwrap().put(key.to_string(), result.clone());
            return Some(result);
        }
        
        // Check L3
        if let Some(result) = self.l3_cache.get(key).await {
            self.promote_to_upper_tiers(key, &result);
            return Some(result);
        }
        
        None
    }
}
```

### 2. Query Batching

```rust
pub struct QueryBatcher {
    pending: Arc<Mutex<Vec<PendingQuery>>>,
    batch_size: usize,
    max_wait: Duration,
    
    pub async fn search(&self, query: String) -> oneshot::Receiver<Vec<SearchResult>> {
        let (tx, rx) = oneshot::channel();
        
        let mut pending = self.pending.lock().unwrap();
        pending.push(PendingQuery { query, sender: tx });
        
        if pending.len() >= self.batch_size {
            let batch = pending.drain(..).collect();
            drop(pending);
            tokio::spawn(self.process_batch(batch));
        } else if pending.len() == 1 {
            // Start timer for first query in batch
            let pending = Arc::clone(&self.pending);
            let max_wait = self.max_wait;
            tokio::spawn(async move {
                tokio::time::sleep(max_wait).await;
                let batch = pending.lock().unwrap().drain(..).collect::<Vec<_>>();
                if !batch.is_empty() {
                    self.process_batch(batch).await;
                }
            });
        }
        
        rx
    }
}
```

## Testing Strategy

### 1. Accuracy Testing

```rust
#[cfg(test)]
mod accuracy_tests {
    use super::*;
    
    #[test]
    fn test_exact_match_accuracy() {
        let search = create_test_search_system();
        
        // Test exact matches
        let results = search.search("fn calculate_sum(a: i32, b: i32)");
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[0].match_type, MatchType::Exact);
    }
    
    #[test]
    fn test_fuzzy_match_accuracy() {
        let search = create_test_search_system();
        
        // Test fuzzy matches with typos
        let results = search.search("calculte_sum"); // typo
        assert!(results[0].score > 0.8);
        assert!(matches!(results[0].match_type, MatchType::Fuzzy(_)));
    }
    
    #[test]
    fn test_semantic_search_accuracy() {
        let search = create_test_search_system();
        
        // Test semantic understanding
        let results = search.search("function that adds two numbers");
        assert!(results.iter().any(|r| r.document_id == "calculate_sum"));
    }
}
```

### 2. Performance Testing

```rust
#[bench]
fn bench_search_performance(b: &mut Bencher) {
    let search = create_production_search_system();
    let queries = load_benchmark_queries();
    
    b.iter(|| {
        for query in &queries {
            black_box(search.search(query));
        }
    });
}

#[test]
fn test_search_latency_requirements() {
    let search = create_production_search_system();
    let mut latencies = Vec::new();
    
    for _ in 0..1000 {
        let start = Instant::now();
        search.search("test query");
        latencies.push(start.elapsed());
    }
    
    latencies.sort();
    let p50 = latencies[500];
    let p95 = latencies[950];
    let p99 = latencies[990];
    
    assert!(p50 < Duration::from_millis(50));
    assert!(p95 < Duration::from_millis(200));
    assert!(p99 < Duration::from_millis(500));
}
```

## Deployment Configuration

```yaml
# config/search_system.yaml
search:
  layers:
    exact_match:
      enabled: true
      bloom_filter_size: 10_000_000
      
    fuzzy_match:
      enabled: true
      min_similarity: 0.7
      max_edit_distance: 3
      
    local_embeddings:
      enabled: true
      models:
        - name: "all-MiniLM-L6-v2"
          max_sequence_length: 256
          batch_size: 32
        - name: "codebert-base"
          max_sequence_length: 512
          batch_size: 16
          
    remote_apis:
      enabled: true
      budget_per_hour: 10.0
      fallback_on_failure: true
      
  performance:
    cache_size: 10000
    batch_size: 10
    max_batch_wait_ms: 50
    num_workers: 4
    
  monitoring:
    metrics_port: 9090
    log_level: "info"
    sample_rate: 0.1
```

This hybrid architecture provides the best balance of accuracy, performance, and cost-efficiency while maintaining high availability through multiple fallback mechanisms.