# Task 28: Memory Retrieval Service Implementation

**Estimated Time**: 15-20 minutes  
**Dependencies**: 27_allocation_service.md  
**Stage**: Service Layer  

## Objective
Implement a high-performance memory retrieval service that provides semantic search, spreading activation, hierarchical traversal, and TTFS-based similarity matching with intelligent caching and sub-50ms response times.

## Specific Requirements

### 1. Multi-Modal Retrieval Engine
- Semantic similarity search using embeddings
- TTFS-based neural pattern matching
- Spreading activation search across knowledge graph
- Hierarchical traversal with inheritance resolution
- Hybrid search combining multiple retrieval methods

### 2. Performance-Optimized Retrieval
- Intelligent query caching with invalidation strategies
- Parallel search execution for complex queries
- Result ranking and relevance scoring
- Lazy loading for large result sets

### 3. Advanced Search Features
- Query analysis and optimization
- Result clustering and deduplication
- Contextual search with user preferences
- Real-time search analytics and monitoring

## Implementation Steps

### 1. Create Core Retrieval Service
```rust
// src/services/retrieval_service.rs
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub struct MemoryRetrievalService {
    // Core retrieval components
    semantic_search: Arc<SemanticSearchEngine>,
    ttfs_matcher: Arc<TTFSSimilarityMatcher>,
    spreading_activation: Arc<SpreadingActivationEngine>,
    hierarchical_traversal: Arc<HierarchicalTraversalEngine>,
    hybrid_searcher: Arc<HybridSearchEngine>,
    
    // Query processing
    query_analyzer: Arc<QueryAnalyzer>,
    query_optimizer: Arc<QueryOptimizer>,
    result_ranker: Arc<ResultRanker>,
    result_clusterer: Arc<ResultClusterer>,
    
    // Performance optimization
    query_cache: Arc<RwLock<LRUCache<String, CachedQueryResult>>>,
    result_cache: Arc<RwLock<LRUCache<String, RetrievalResult>>>,
    parallel_executor: Arc<ParallelSearchExecutor>,
    
    // Monitoring and analytics
    search_analytics: Arc<SearchAnalytics>,
    performance_monitor: Arc<RetrievalPerformanceMonitor>,
    
    config: RetrievalServiceConfig,
}

impl MemoryRetrievalService {
    pub async fn new(
        knowledge_graph_service: Arc<KnowledgeGraphService>,
        ttfs_integration: Arc<TTFSIntegrationService>,
        inheritance_engine: Arc<PropertyInheritanceEngine>,
        config: RetrievalServiceConfig,
    ) -> Result<Self, RetrievalServiceError> {
        // Initialize search engines
        let semantic_search = Arc::new(
            SemanticSearchEngine::new(knowledge_graph_service.clone()).await?
        );
        
        let ttfs_matcher = Arc::new(
            TTFSSimilarityMatcher::new(ttfs_integration.clone()).await?
        );
        
        let spreading_activation = Arc::new(
            SpreadingActivationEngine::new(knowledge_graph_service.clone()).await?
        );
        
        let hierarchical_traversal = Arc::new(
            HierarchicalTraversalEngine::new(
                knowledge_graph_service.clone(),
                inheritance_engine.clone(),
            ).await?
        );
        
        let hybrid_searcher = Arc::new(
            HybridSearchEngine::new(
                semantic_search.clone(),
                ttfs_matcher.clone(),
                spreading_activation.clone(),
                hierarchical_traversal.clone(),
            ).await?
        );
        
        // Initialize query processing
        let query_analyzer = Arc::new(QueryAnalyzer::new());
        let query_optimizer = Arc::new(QueryOptimizer::new());
        let result_ranker = Arc::new(ResultRanker::new(config.ranking_config.clone()));
        let result_clusterer = Arc::new(ResultClusterer::new());
        
        // Initialize caching
        let query_cache = Arc::new(RwLock::new(
            LRUCache::new(config.query_cache_size)
        ));
        let result_cache = Arc::new(RwLock::new(
            LRUCache::new(config.result_cache_size)
        ));
        
        // Initialize performance components
        let parallel_executor = Arc::new(ParallelSearchExecutor::new(config.max_parallel_searches));
        let search_analytics = Arc::new(SearchAnalytics::new());
        let performance_monitor = Arc::new(RetrievalPerformanceMonitor::new());
        
        Ok(Self {
            semantic_search,
            ttfs_matcher,
            spreading_activation,
            hierarchical_traversal,
            hybrid_searcher,
            query_analyzer,
            query_optimizer,
            result_ranker,
            result_clusterer,
            query_cache,
            result_cache,
            parallel_executor,
            search_analytics,
            performance_monitor,
            config,
        })
    }
    
    pub async fn search_memory(
        &self,
        request: SearchRequest,
    ) -> Result<SearchResult, RetrievalError> {
        let search_start = Instant::now();
        
        // Analyze and optimize query
        let query_analysis = self.query_analyzer.analyze_query(&request).await?;
        let optimized_query = self.query_optimizer.optimize_query(&request, &query_analysis).await?;
        
        // Check cache
        let cache_key = self.generate_cache_key(&optimized_query);
        if let Some(cached_result) = self.query_cache.read().await.get(&cache_key) {
            if cached_result.is_valid() {
                self.search_analytics.record_cache_hit().await;
                return Ok(cached_result.result.clone());
            }
        }
        
        // Execute search
        let search_result = self.execute_search(optimized_query, query_analysis, search_start).await?;
        
        // Cache result
        let cached_query_result = CachedQueryResult {
            result: search_result.clone(),
            timestamp: Utc::now(),
            ttl: Duration::from_secs(self.config.cache_ttl_seconds),
        };
        self.query_cache.write().await.put(cache_key, cached_query_result);
        
        // Record analytics
        let search_time = search_start.elapsed();
        self.search_analytics.record_search(
            &request,
            &search_result,
            search_time,
        ).await;
        
        Ok(search_result)
    }
    
    async fn execute_search(
        &self,
        request: OptimizedSearchRequest,
        analysis: QueryAnalysis,
        search_start: Instant,
    ) -> Result<SearchResult, RetrievalError> {
        let raw_results = match request.search_type {
            SearchType::Semantic => {
                self.execute_semantic_search(request).await?
            },
            SearchType::TTFSSimilarity => {
                self.execute_ttfs_search(request).await?
            },
            SearchType::SpreadingActivation => {
                self.execute_spreading_activation_search(request).await?
            },
            SearchType::HierarchicalTraversal => {
                self.execute_hierarchical_search(request).await?
            },
            SearchType::Hybrid => {
                self.execute_hybrid_search(request).await?
            },
        };
        
        // Post-process results
        let processed_results = self.post_process_results(
            raw_results,
            &request,
            &analysis,
        ).await?;
        
        Ok(SearchResult {
            results: processed_results.results,
            total_matches: processed_results.total_matches,
            search_time_ms: search_start.elapsed().as_millis() as u64,
            query_analysis: analysis,
            result_metadata: processed_results.metadata,
            cache_hit: false,
        })
    }
    
    async fn execute_semantic_search(
        &self,
        request: OptimizedSearchRequest,
    ) -> Result<RawSearchResults, RetrievalError> {
        let semantic_results = self.semantic_search
            .search_by_similarity(
                &request.query_text,
                request.similarity_threshold.unwrap_or(0.7),
                request.limit.unwrap_or(10),
            )
            .await?;
        
        Ok(RawSearchResults {
            results: semantic_results,
            search_method: SearchMethod::Semantic,
            processing_time: semantic_results.processing_time,
        })
    }
    
    async fn execute_ttfs_search(
        &self,
        request: OptimizedSearchRequest,
    ) -> Result<RawSearchResults, RetrievalError> {
        // Encode query with TTFS
        let query_encoding = self.ttfs_matcher
            .encode_query(&request.query_text)
            .await?;
        
        // Find similar TTFS patterns
        let ttfs_results = self.ttfs_matcher
            .find_similar_patterns(
                &query_encoding,
                request.similarity_threshold.unwrap_or(0.8),
                request.limit.unwrap_or(10),
            )
            .await?;
        
        Ok(RawSearchResults {
            results: ttfs_results,
            search_method: SearchMethod::TTFSSimilarity,
            processing_time: Duration::from_millis(5), // TTFS is very fast
        })
    }
    
    async fn execute_spreading_activation_search(
        &self,
        request: OptimizedSearchRequest,
    ) -> Result<RawSearchResults, RetrievalError> {
        // Parse seed concepts from query
        let seed_concepts = self.query_analyzer
            .extract_seed_concepts(&request.query_text)
            .await?;
        
        // Execute spreading activation
        let activation_results = self.spreading_activation
            .activate_from_seeds(
                &seed_concepts,
                request.max_hops.unwrap_or(6),
                request.activation_threshold.unwrap_or(0.3),
                request.limit.unwrap_or(20),
            )
            .await?;
        
        Ok(RawSearchResults {
            results: activation_results,
            search_method: SearchMethod::SpreadingActivation,
            processing_time: Duration::from_millis(45), // More complex
        })
    }
    
    async fn execute_hierarchical_search(
        &self,
        request: OptimizedSearchRequest,
    ) -> Result<RawSearchResults, RetrievalError> {
        // Identify root concepts in hierarchy
        let root_concepts = self.query_analyzer
            .identify_hierarchical_roots(&request.query_text)
            .await?;
        
        // Traverse hierarchy with inheritance resolution
        let hierarchical_results = self.hierarchical_traversal
            .traverse_hierarchy(
                &root_concepts,
                request.traversal_depth.unwrap_or(5),
                request.include_inherited_properties.unwrap_or(true),
                request.limit.unwrap_or(15),
            )
            .await?;
        
        Ok(RawSearchResults {
            results: hierarchical_results,
            search_method: SearchMethod::HierarchicalTraversal,
            processing_time: Duration::from_millis(30),
        })
    }
    
    async fn execute_hybrid_search(
        &self,
        request: OptimizedSearchRequest,
    ) -> Result<RawSearchResults, RetrievalError> {
        // Execute multiple search methods in parallel
        let search_futures = vec![
            self.parallel_executor.execute_semantic_search(request.clone()),
            self.parallel_executor.execute_ttfs_search(request.clone()),
            self.parallel_executor.execute_spreading_activation_search(request.clone()),
        ];
        
        let parallel_results = futures::future::try_join_all(search_futures).await?;
        
        // Combine and rank results
        let combined_results = self.hybrid_searcher
            .combine_results(parallel_results, &request)
            .await?;
        
        Ok(RawSearchResults {
            results: combined_results,
            search_method: SearchMethod::Hybrid,
            processing_time: Duration::from_millis(60), // Parallel execution overhead
        })
    }
    
    async fn post_process_results(
        &self,
        raw_results: RawSearchResults,
        request: &OptimizedSearchRequest,
        analysis: &QueryAnalysis,
    ) -> Result<ProcessedSearchResults, RetrievalError> {
        // Rank results
        let ranked_results = self.result_ranker
            .rank_results(raw_results.results, request, analysis)
            .await?;
        
        // Deduplicate if requested
        let deduplicated_results = if request.deduplicate_results.unwrap_or(true) {
            self.result_clusterer.deduplicate_results(ranked_results).await?
        } else {
            ranked_results
        };
        
        // Cluster similar results if requested
        let final_results = if request.cluster_results.unwrap_or(false) {
            self.result_clusterer.cluster_results(deduplicated_results).await?
        } else {
            deduplicated_results
        };
        
        Ok(ProcessedSearchResults {
            results: final_results.into_iter().take(request.limit.unwrap_or(10)).collect(),
            total_matches: final_results.len(),
            metadata: ResultMetadata {
                search_method: raw_results.search_method,
                processing_time: raw_results.processing_time,
                ranking_applied: true,
                deduplication_applied: request.deduplicate_results.unwrap_or(true),
                clustering_applied: request.cluster_results.unwrap_or(false),
            },
        })
    }
    
    pub async fn get_search_analytics(&self) -> SearchAnalytics {
        self.search_analytics.get_current_analytics().await
    }
    
    pub async fn get_retrieval_metrics(&self) -> RetrievalMetrics {
        RetrievalMetrics {
            total_searches: self.search_analytics.get_total_searches().await,
            average_search_time: self.performance_monitor.get_average_search_time().await,
            cache_hit_rate: self.search_analytics.get_cache_hit_rate().await,
            search_method_distribution: self.search_analytics.get_method_distribution().await,
            query_complexity_distribution: self.search_analytics.get_complexity_distribution().await,
            result_relevance_scores: self.performance_monitor.get_relevance_scores().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query_text: String,
    pub search_type: SearchType,
    pub similarity_threshold: Option<f64>,
    pub max_hops: Option<usize>,
    pub activation_threshold: Option<f64>,
    pub traversal_depth: Option<usize>,
    pub include_inherited_properties: Option<bool>,
    pub limit: Option<usize>,
    pub deduplicate_results: Option<bool>,
    pub cluster_results: Option<bool>,
    pub user_context: Option<UserContext>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub results: Vec<RetrievalMatch>,
    pub total_matches: usize,
    pub search_time_ms: u64,
    pub query_analysis: QueryAnalysis,
    pub result_metadata: ResultMetadata,
    pub cache_hit: bool,
}
```

### 2. Implement Search Analytics
```rust
// src/services/retrieval_service/analytics.rs
#[derive(Debug, Clone)]
pub struct SearchAnalytics {
    search_history: Arc<RwLock<Vec<SearchRecord>>>,
    method_stats: Arc<RwLock<HashMap<SearchMethod, MethodStats>>>,
    query_patterns: Arc<RwLock<QueryPatternAnalyzer>>,
    performance_tracker: Arc<PerformanceTracker>,
}

impl SearchAnalytics {
    pub async fn record_search(
        &self,
        request: &SearchRequest,
        result: &SearchResult,
        duration: Duration,
    ) {
        let search_record = SearchRecord {
            timestamp: Utc::now(),
            query_text: request.query_text.clone(),
            search_type: request.search_type.clone(),
            result_count: result.results.len(),
            search_duration: duration,
            cache_hit: result.cache_hit,
            relevance_score: self.calculate_average_relevance(&result.results),
        };
        
        self.search_history.write().await.push(search_record);
        self.update_method_stats(&request.search_type, &result, duration).await;
        self.query_patterns.write().await.analyze_pattern(&request.query_text).await;
    }
    
    pub async fn get_search_insights(&self) -> SearchInsights {
        SearchInsights {
            popular_queries: self.get_popular_queries().await,
            optimal_search_methods: self.get_optimal_methods().await,
            query_complexity_trends: self.get_complexity_trends().await,
            performance_bottlenecks: self.identify_bottlenecks().await,
        }
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All search types (semantic, TTFS, spreading activation, hierarchical) work correctly
- [ ] Hybrid search combines results effectively
- [ ] Query caching improves performance
- [ ] Result ranking and clustering work properly
- [ ] Search analytics capture meaningful metrics

### Performance Requirements
- [ ] Search operations < 50ms for 6-hop queries
- [ ] Cache hit rate > 60% for repeated queries
- [ ] Parallel searches execute correctly
- [ ] Memory usage stays within bounds

### Testing Requirements
- [ ] Unit tests for each search method
- [ ] Performance tests for search speed
- [ ] Cache effectiveness tests
- [ ] Result relevance quality tests

## Validation Steps

1. **Test retrieval service creation**:
   ```rust
   let service = MemoryRetrievalService::new(kg_service, ttfs, inheritance, config).await?;
   let metrics = service.get_retrieval_metrics().await;
   ```

2. **Test search operations**:
   ```rust
   let result = service.search_memory(search_request).await?;
   assert!(result.search_time_ms < 50);
   ```

3. **Run retrieval service tests**:
   ```bash
   cargo test retrieval_service_tests
   ```

## Files to Create/Modify
- `src/services/retrieval_service.rs` - Main retrieval service
- `src/services/retrieval_service/search_engines.rs` - Individual search engines
- `src/services/retrieval_service/analytics.rs` - Search analytics
- `tests/services/retrieval_service_tests.rs` - Test suite

## Success Metrics
- Search latency: < 50ms (95th percentile)
- Cache hit rate: > 60% for repeated queries
- Result relevance: > 0.8 average score
- Concurrent searches: 500+ operations/second

## Next Task
Upon completion, proceed to **29_error_handling.md** to add comprehensive error handling.