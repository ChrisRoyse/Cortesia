# Task 17: Semantic Similarity Search Implementation
**Estimated Time**: 15-20 minutes
**Dependencies**: 16_query_optimization.md
**Stage**: Performance Optimization

## Objective
Implement high-performance semantic similarity search capabilities for knowledge graph concepts using neural embeddings, vector indexing, and approximate nearest neighbor algorithms for sub-5ms search responses.

## Specific Requirements

### 1. Vector Embedding Integration
- Integrate concept embeddings with graph nodes
- Implement embedding update mechanisms for concept changes
- Add embedding versioning and consistency management
- Build embedding quality monitoring and validation

### 2. Similarity Search Engine
- Implement approximate nearest neighbor (ANN) search
- Add semantic similarity scoring with multiple metrics
- Build hybrid search combining graph structure and semantics
- Add search result ranking and relevance tuning

### 3. Performance Optimization
- Implement vector index caching and preloading
- Add batch similarity computation for efficiency
- Build search result caching with semantic invalidation
- Add parallel search execution for complex queries

## Implementation Steps

### 1. Create Semantic Search Framework
```rust
// src/inheritance/semantic/semantic_search_engine.rs
use std::collections::HashMap;
use candle_core::{Tensor, Device};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct SemanticSearchEngine {
    vector_index: Arc<RwLock<VectorIndex>>,
    embedding_model: Arc<EmbeddingModel>,
    similarity_calculator: Arc<SimilarityCalculator>,
    search_cache: Arc<RwLock<HashMap<String, CachedSearchResult>>>,
    performance_monitor: Arc<SearchPerformanceMonitor>,
    config: SemanticSearchConfig,
}

#[derive(Debug, Clone)]
pub struct VectorIndex {
    concept_embeddings: HashMap<String, ConceptEmbedding>,
    index_metadata: IndexMetadata,
    ann_index: Option<HnswIndex>, // Hierarchical Navigable Small World index
    last_updated: DateTime<Utc>,
    consistency_hash: u64,
}

#[derive(Debug, Clone)]
pub struct ConceptEmbedding {
    pub concept_id: String,
    pub embedding_vector: Vec<f32>,
    pub embedding_model: String,
    pub embedding_version: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub quality_score: f32,
}

#[derive(Debug, Clone)]
pub struct SemanticSearchRequest {
    pub query_text: Option<String>,
    pub query_embedding: Option<Vec<f32>>,
    pub concept_id: Option<String>,
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub include_graph_context: bool,
    pub search_strategy: SearchStrategy,
}

#[derive(Debug, Clone)]
pub enum SearchStrategy {
    PureSemantics,
    GraphStructureWeighted,
    HybridOptimal,
    InheritanceAware,
}

impl SemanticSearchEngine {
    pub async fn new(
        embedding_model: Arc<EmbeddingModel>,
        config: SemanticSearchConfig,
    ) -> Result<Self, SemanticSearchError> {
        let vector_index = Arc::new(RwLock::new(VectorIndex::new()));
        let similarity_calculator = Arc::new(SimilarityCalculator::new());
        let search_cache = Arc::new(RwLock::new(HashMap::new()));
        let performance_monitor = Arc::new(SearchPerformanceMonitor::new());
        
        let engine = Self {
            vector_index,
            embedding_model,
            similarity_calculator,
            search_cache,
            performance_monitor,
            config,
        };
        
        // Initialize vector index
        engine.initialize_vector_index().await?;
        
        Ok(engine)
    }
    
    pub async fn semantic_search(
        &self,
        request: SemanticSearchRequest,
    ) -> Result<SemanticSearchResult, SemanticSearchError> {
        let search_start = Instant::now();
        let search_id = uuid::Uuid::new_v4().to_string();
        
        // Check cache first
        let cache_key = self.generate_cache_key(&request);
        if let Some(cached_result) = self.get_cached_result(&cache_key).await {
            self.performance_monitor.record_cache_hit(&search_id, search_start.elapsed()).await;
            return Ok(cached_result);
        }
        
        // Get query embedding
        let query_embedding = match (&request.query_text, &request.query_embedding) {
            (Some(text), _) => {
                self.embedding_model.encode_text(text).await?
            },
            (None, Some(embedding)) => embedding.clone(),
            (None, None) if request.concept_id.is_some() => {
                self.get_concept_embedding(request.concept_id.as_ref().unwrap()).await?
            },
            _ => return Err(SemanticSearchError::InvalidRequest("No query provided".to_string())),
        };
        
        // Perform similarity search based on strategy
        let search_results = match request.search_strategy {
            SearchStrategy::PureSemantics => {
                self.pure_semantic_search(&query_embedding, &request).await?
            },
            SearchStrategy::GraphStructureWeighted => {
                self.graph_weighted_search(&query_embedding, &request).await?
            },
            SearchStrategy::HybridOptimal => {
                self.hybrid_search(&query_embedding, &request).await?
            },
            SearchStrategy::InheritanceAware => {
                self.inheritance_aware_search(&query_embedding, &request).await?
            },
        };
        
        // Build result
        let result = SemanticSearchResult {
            search_id,
            request: request.clone(),
            results: search_results,
            search_duration: search_start.elapsed(),
            cache_status: CacheStatus::Miss,
            index_metadata: self.get_index_metadata().await,
        };
        
        // Cache result
        self.cache_search_result(cache_key, &result).await?;
        
        // Record performance metrics
        self.performance_monitor.record_search_completion(&result).await;
        
        Ok(result)
    }
    
    async fn pure_semantic_search(
        &self,
        query_embedding: &[f32],
        request: &SemanticSearchRequest,
    ) -> Result<Vec<ScoredConcept>, SearchError> {
        let index = self.vector_index.read().await;
        let mut scored_concepts = Vec::new();
        
        // Use ANN index if available, otherwise fall back to brute force
        if let Some(ann_index) = &index.ann_index {
            let candidates = ann_index.search(
                query_embedding,
                request.max_results * 2, // Get more candidates for better precision
            )?;
            
            for candidate in candidates {
                let similarity = self.similarity_calculator.cosine_similarity(
                    query_embedding,
                    &candidate.embedding,
                )?;
                
                if similarity >= request.similarity_threshold {
                    scored_concepts.push(ScoredConcept {
                        concept_id: candidate.concept_id,
                        similarity_score: similarity,
                        semantic_score: similarity,
                        graph_score: 0.0, // No graph component in pure semantic search
                        combined_score: similarity,
                        explanation: ScoringExplanation::pure_semantic(similarity),
                    });
                }
            }
        } else {
            // Brute force search for small datasets
            for (concept_id, embedding) in &index.concept_embeddings {
                let similarity = self.similarity_calculator.cosine_similarity(
                    query_embedding,
                    &embedding.embedding_vector,
                )?;
                
                if similarity >= request.similarity_threshold {
                    scored_concepts.push(ScoredConcept {
                        concept_id: concept_id.clone(),
                        similarity_score: similarity,
                        semantic_score: similarity,
                        graph_score: 0.0,
                        combined_score: similarity,
                        explanation: ScoringExplanation::pure_semantic(similarity),
                    });
                }
            }
        }
        
        // Sort by similarity score and limit results
        scored_concepts.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        scored_concepts.truncate(request.max_results);
        
        Ok(scored_concepts)
    }
    
    async fn hybrid_search(
        &self,
        query_embedding: &[f32],
        request: &SemanticSearchRequest,
    ) -> Result<Vec<ScoredConcept>, SearchError> {
        // Get semantic candidates
        let semantic_candidates = self.pure_semantic_search(query_embedding, request).await?;
        
        // Get graph structure information
        let graph_scores = if request.include_graph_context {
            self.calculate_graph_relevance_scores(&semantic_candidates, request).await?
        } else {
            HashMap::new()
        };
        
        // Combine semantic and graph scores
        let mut hybrid_results = Vec::new();
        for candidate in semantic_candidates {
            let graph_score = graph_scores.get(&candidate.concept_id).unwrap_or(&0.0);
            
            // Weighted combination of semantic and graph scores
            let combined_score = (candidate.semantic_score * self.config.semantic_weight) + 
                              (graph_score * self.config.graph_weight);
            
            hybrid_results.push(ScoredConcept {
                concept_id: candidate.concept_id,
                similarity_score: candidate.semantic_score,
                semantic_score: candidate.semantic_score,
                graph_score: *graph_score,
                combined_score,
                explanation: ScoringExplanation::hybrid(
                    candidate.semantic_score,
                    *graph_score,
                    self.config.semantic_weight,
                    self.config.graph_weight,
                ),
            });
        }
        
        // Re-sort by combined score
        hybrid_results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        hybrid_results.truncate(request.max_results);
        
        Ok(hybrid_results)
    }
    
    async fn inheritance_aware_search(
        &self,
        query_embedding: &[f32],
        request: &SemanticSearchRequest,
    ) -> Result<Vec<ScoredConcept>, SearchError> {
        // First get hybrid search results
        let base_results = self.hybrid_search(query_embedding, request).await?;
        
        // Enhance with inheritance context
        let mut inheritance_enhanced = Vec::new();
        
        for result in base_results {
            // Get inheritance chain for context
            let inheritance_boost = self.calculate_inheritance_relevance_boost(
                &result.concept_id,
                query_embedding,
            ).await?;
            
            let enhanced_score = result.combined_score * (1.0 + inheritance_boost);
            
            inheritance_enhanced.push(ScoredConcept {
                concept_id: result.concept_id,
                similarity_score: result.similarity_score,
                semantic_score: result.semantic_score,
                graph_score: result.graph_score + inheritance_boost,
                combined_score: enhanced_score,
                explanation: ScoringExplanation::inheritance_aware(
                    result.semantic_score,
                    result.graph_score,
                    inheritance_boost,
                ),
            });
        }
        
        // Re-sort with inheritance enhancement
        inheritance_enhanced.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        inheritance_enhanced.truncate(request.max_results);
        
        Ok(inheritance_enhanced)
    }
    
    pub async fn update_concept_embedding(
        &self,
        concept_id: &str,
        concept_text: &str,
    ) -> Result<(), EmbeddingUpdateError> {
        // Generate new embedding
        let new_embedding = self.embedding_model.encode_text(concept_text).await?;
        
        // Calculate quality score
        let quality_score = self.evaluate_embedding_quality(&new_embedding, concept_text).await?;
        
        let concept_embedding = ConceptEmbedding {
            concept_id: concept_id.to_string(),
            embedding_vector: new_embedding,
            embedding_model: self.embedding_model.model_name().to_string(),
            embedding_version: self.embedding_model.version().to_string(),
            created_at: if self.has_existing_embedding(concept_id).await {
                self.get_existing_embedding_creation_time(concept_id).await?
            } else {
                Utc::now()
            },
            updated_at: Utc::now(),
            quality_score,
        };
        
        // Update in vector index
        let mut index = self.vector_index.write().await;
        index.concept_embeddings.insert(concept_id.to_string(), concept_embedding);
        index.last_updated = Utc::now();
        
        // Invalidate related caches
        self.invalidate_search_caches_for_concept(concept_id).await?;
        
        // Update ANN index if threshold reached
        if index.concept_embeddings.len() % self.config.ann_rebuild_threshold == 0 {
            self.rebuild_ann_index().await?;
        }
        
        Ok(())
    }
}
```

### 2. Implement Vector Index Management
```rust
// src/inheritance/semantic/vector_index.rs
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::dist::DistCosine;

pub struct VectorIndexManager {
    hnsw_index: Option<Hnsw<f32, DistCosine>>,
    index_config: HnswConfig,
    rebuild_scheduler: Arc<RebuildScheduler>,
}

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub rebuild_threshold: usize,
    pub parallel_build: bool,
}

impl VectorIndexManager {
    pub async fn build_ann_index(
        &mut self,
        embeddings: &HashMap<String, ConceptEmbedding>,
    ) -> Result<(), IndexBuildError> {
        let build_start = Instant::now();
        
        if embeddings.is_empty() {
            return Ok(());
        }
        
        // Determine vector dimension from first embedding
        let vector_dim = embeddings.values().next().unwrap().embedding_vector.len();
        
        // Create HNSW index
        let mut hnsw = Hnsw::<f32, DistCosine>::new(
            self.index_config.max_connections,
            embeddings.len(),
            vector_dim,
            self.index_config.ef_construction,
            DistCosine,
        );
        
        // Insert embeddings into index
        let mut concept_id_map = HashMap::new();
        for (idx, (concept_id, embedding)) in embeddings.iter().enumerate() {
            hnsw.insert(&embedding.embedding_vector, idx)?;
            concept_id_map.insert(idx, concept_id.clone());
        }
        
        // Set search parameters
        hnsw.set_searching_mode(true);
        hnsw.set_ef(self.index_config.ef_search);
        
        self.hnsw_index = Some(hnsw);
        
        info!(
            "Built ANN index for {} concepts in {:?}",
            embeddings.len(),
            build_start.elapsed()
        );
        
        Ok(())
    }
    
    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<IndexCandidate>, SearchError> {
        let index = self.hnsw_index.as_ref()
            .ok_or(SearchError::IndexNotBuilt)?;
        
        let search_results = index.search(query_embedding, k, self.index_config.ef_search)?;
        
        let mut candidates = Vec::new();
        for (distance, node_id) in search_results {
            // Convert distance to similarity (cosine distance -> cosine similarity)
            let similarity = 1.0 - distance;
            
            candidates.push(IndexCandidate {
                node_id,
                similarity_score: similarity,
                distance,
            });
        }
        
        Ok(candidates)
    }
    
    pub async fn get_index_statistics(&self) -> IndexStatistics {
        if let Some(index) = &self.hnsw_index {
            IndexStatistics {
                total_nodes: index.get_nb_point(),
                index_size_bytes: self.estimate_index_memory_usage(),
                build_time: self.last_build_duration,
                search_performance: self.calculate_search_performance_metrics().await,
            }
        } else {
            IndexStatistics::empty()
        }
    }
}
```

### 3. Implement Similarity Calculators
```rust
// src/inheritance/semantic/similarity.rs
pub struct SimilarityCalculator {
    metrics_cache: Arc<RwLock<HashMap<String, f32>>>,
    calculation_stats: Arc<RwLock<CalculationStats>>,
}

impl SimilarityCalculator {
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
        if a.len() != b.len() {
            return Err(SimilarityError::DimensionMismatch);
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }
    
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
        if a.len() != b.len() {
            return Err(SimilarityError::DimensionMismatch);
        }
        
        let distance = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
            
        Ok(distance)
    }
    
    pub fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
        if a.len() != b.len() {
            return Err(SimilarityError::DimensionMismatch);
        }
        
        let distance = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>();
            
        Ok(distance)
    }
    
    pub async fn batch_cosine_similarity(
        &self,
        query: &[f32],
        candidates: &[&[f32]],
    ) -> Result<Vec<f32>, SimilarityError> {
        let mut similarities = Vec::with_capacity(candidates.len());
        
        // Precompute query norm for efficiency
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if query_norm == 0.0 {
            return Ok(vec![0.0; candidates.len()]);
        }
        
        for candidate in candidates {
            let dot_product: f32 = query.iter().zip(candidate.iter()).map(|(x, y)| x * y).sum();
            let candidate_norm: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            let similarity = if candidate_norm == 0.0 {
                0.0
            } else {
                dot_product / (query_norm * candidate_norm)
            };
            
            similarities.push(similarity);
        }
        
        Ok(similarities)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Semantic similarity search with multiple distance metrics
- [ ] ANN index integration for fast approximate search
- [ ] Hybrid search combining semantic and graph features
- [ ] Inheritance-aware search with context boosting
- [ ] Real-time embedding updates with consistency management

### Performance Requirements
- [ ] Semantic search response time < 5ms for queries
- [ ] ANN index search < 2ms for 1000+ concept datasets
- [ ] Batch similarity computation handles 100+ vectors in <10ms
- [ ] Embedding updates complete within 50ms
- [ ] Memory usage for vector index < 500MB for 10k concepts

### Testing Requirements
- [ ] Unit tests for similarity calculation algorithms
- [ ] Performance benchmarks for different search strategies
- [ ] Index build and search accuracy tests
- [ ] Embedding quality validation tests

## Validation Steps

1. **Test semantic search accuracy**:
   ```rust
   let search_engine = SemanticSearchEngine::new(embedding_model, config).await?;
   let results = search_engine.semantic_search(request).await?;
   assert!(results.results.len() > 0);
   assert!(results.results[0].similarity_score > 0.7);
   ```

2. **Benchmark search performance**:
   ```rust
   let start = Instant::now();
   let results = search_engine.semantic_search(search_request).await?;
   let duration = start.elapsed();
   assert!(duration < Duration::from_millis(5));
   ```

3. **Run semantic search tests**:
   ```bash
   cargo test semantic_search_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/semantic/semantic_search_engine.rs` - Core search engine
- `src/inheritance/semantic/vector_index.rs` - Vector index management
- `src/inheritance/semantic/similarity.rs` - Similarity calculators
- `src/inheritance/semantic/embedding_model.rs` - Embedding model interface
- `src/inheritance/semantic/mod.rs` - Module exports
- `tests/inheritance/semantic_tests.rs` - Semantic search test suite

## Success Metrics
- Search accuracy: >90% relevance for semantic queries
- Search speed: <5ms average response time
- Index efficiency: >95% memory utilization for vector storage
- Embedding quality: >0.85 average similarity correlation

## Next Task
Upon completion, proceed to **18_caching_system.md** to build comprehensive caching layer for the knowledge graph system.