# Find Facts Enhancement: Technical Architecture Design

## Core Architecture Philosophy

### Design Principles

1. **Layered Enhancement**: Each tier builds upon the previous, maintaining independent functionality
2. **Backward Compatibility**: Zero breaking changes to existing `find_facts` API
3. **Resource Efficiency**: Lazy loading, intelligent caching, configurable limits
4. **Performance Predictability**: Clear performance characteristics for each enhancement level
5. **Test-Driven Design**: Every component designed with mockability and testability first

### System Architecture Overview

```rust
// Core architectural components
pub struct EnhancedFindFactsSystem {
    // Existing core (unchanged)
    core_engine: Arc<RwLock<KnowledgeEngine>>,
    
    // Enhancement layers (optional, lazy-loaded)
    entity_linking: Option<Arc<EntityLinkingLayer>>,
    semantic_expansion: Option<Arc<SemanticExpansionLayer>>,
    research_grade: Option<Arc<ResearchGradeLayer>>,
    
    // Resource management
    model_manager: Arc<ModelManager>,
    cache_manager: Arc<CacheManager>,
    config: EnhancementConfig,
}
```

## Component Architecture

### Layer 1: Entity Linking Foundation

#### Core Components
```rust
pub struct EntityLinkingLayer {
    entity_linker: Arc<dyn EntityLinker>,
    embedding_cache: Arc<EmbeddingCache>,
    alias_resolver: Arc<AliasResolver>,
    normalization_engine: Arc<NormalizationEngine>,
}

#[async_trait]
pub trait EntityLinker: Send + Sync {
    async fn link_entity(&self, mention: &str) -> Result<Vec<LinkedEntity>>;
    async fn normalize_entity(&self, entity: &str) -> Result<String>;
    async fn find_aliases(&self, canonical: &str) -> Result<Vec<String>>;
}

pub struct MiniLMEntityLinker {
    model: Arc<dyn Model>,
    entity_embeddings: Arc<EntityEmbeddingIndex>,
    similarity_threshold: f32,
}
```

#### Integration with Existing System
```rust
impl EntityLinkingLayer {
    pub async fn enhance_query(&self, query: TripleQuery) -> Result<Vec<TripleQuery>> {
        let mut enhanced_queries = vec![query.clone()]; // Always include original
        
        // Enhance subject if present
        if let Some(subject) = &query.subject {
            let linked_entities = self.entity_linker.link_entity(subject).await?;
            for entity in linked_entities {
                if entity.confidence > self.config.min_confidence {
                    enhanced_queries.push(TripleQuery {
                        subject: Some(entity.canonical_name),
                        ..query.clone()
                    });
                }
            }
        }
        
        // Similar for object
        if let Some(object) = &query.object {
            // ... entity linking for object
        }
        
        Ok(enhanced_queries)
    }
}
```

### Layer 2: Semantic Expansion System

#### Core Components
```rust
pub struct SemanticExpansionLayer {
    query_expander: Arc<dyn QueryExpander>,
    predicate_expander: Arc<dyn PredicateExpander>,
    similarity_ranker: Arc<dyn SimilarityRanker>,
    vector_index: Arc<VectorIndex>,
}

#[async_trait]
pub trait QueryExpander: Send + Sync {
    async fn expand_predicate(&self, predicate: &str) -> Result<Vec<ExpandedPredicate>>;
    async fn expand_semantic_context(&self, query: &TripleQuery) -> Result<SemanticContext>;
}

pub struct SmolLMQueryExpander {
    model: Arc<dyn Model>,
    relation_ontology: Arc<RelationOntology>,
    expansion_cache: Arc<LRUCache<String, Vec<ExpandedPredicate>>>,
}
```

#### Semantic Enhancement Pipeline
```rust
impl SemanticExpansionLayer {
    pub async fn enhance_semantically(
        &self, 
        base_queries: Vec<TripleQuery>
    ) -> Result<Vec<ScoredTripleQuery>> {
        let mut semantic_queries = Vec::new();
        
        for query in base_queries {
            // Expand predicates semantically
            if let Some(predicate) = &query.predicate {
                let expanded = self.query_expander.expand_predicate(predicate).await?;
                for expansion in expanded {
                    semantic_queries.push(ScoredTripleQuery {
                        query: TripleQuery {
                            predicate: Some(expansion.relation),
                            ..query.clone()
                        },
                        semantic_score: expansion.confidence,
                        expansion_type: expansion.expansion_type,
                    });
                }
            }
            
            // Add original query with high score
            semantic_queries.push(ScoredTripleQuery {
                query: query.clone(),
                semantic_score: 1.0,
                expansion_type: ExpansionType::Exact,
            });
        }
        
        Ok(semantic_queries)
    }
}
```

### Layer 3: Research-Grade Multi-Model System

#### Architecture Components
```rust
pub struct ResearchGradeLayer {
    model_coordinator: Arc<MultiModelCoordinator>,
    reasoning_engine: Arc<ReasoningEngine>,
    context_analyzer: Arc<ContextAnalyzer>,
    inference_pipeline: Arc<InferencePipeline>,
}

pub struct MultiModelCoordinator {
    primary_reasoner: Arc<dyn Model>,    // SmolLM-1.7B
    efficiency_executor: Arc<dyn Model>, // OpenELM-1.1B
    coordination_strategy: CoordinationStrategy,
}

#[async_trait]
pub trait ReasoningEngine: Send + Sync {
    async fn analyze_query_intent(&self, query: &TripleQuery) -> Result<QueryIntent>;
    async fn generate_inference_plan(&self, intent: &QueryIntent) -> Result<InferencePlan>;
    async fn execute_multi_hop_reasoning(&self, plan: &InferencePlan) -> Result<ReasoningResult>;
}
```

## Model Management Architecture

### Model Loading and Lifecycle
```rust
pub struct ModelManager {
    loaded_models: Arc<RwLock<HashMap<ModelId, Arc<dyn Model>>>>,
    loading_queue: Arc<Mutex<VecDeque<ModelLoadRequest>>>,
    resource_monitor: Arc<ResourceMonitor>,
    config: ModelManagerConfig,
}

impl ModelManager {
    pub async fn get_or_load_model<T: Model + 'static>(
        &self, 
        model_id: &ModelId
    ) -> Result<Arc<T>> {
        // Check if already loaded
        if let Some(model) = self.get_cached_model::<T>(model_id) {
            return Ok(model);
        }
        
        // Check resource constraints
        self.ensure_sufficient_resources::<T>(model_id).await?;
        
        // Load model asynchronously
        let model = self.load_model_async::<T>(model_id).await?;
        
        // Cache and return
        self.cache_model(model_id, model.clone()).await?;
        Ok(model)
    }
    
    async fn ensure_sufficient_resources<T: Model>(
        &self, 
        model_id: &ModelId
    ) -> Result<()> {
        let required_memory = T::estimated_memory_usage();
        let available = self.resource_monitor.available_memory();
        
        if available < required_memory {
            self.evict_least_recently_used(required_memory - available).await?;
        }
        
        Ok(())
    }
}
```

### Resource Monitoring and Management
```rust
pub struct ResourceMonitor {
    memory_tracker: Arc<MemoryTracker>,
    performance_monitor: Arc<PerformanceMonitor>,
    threshold_config: ResourceThresholds,
}

pub struct ResourceThresholds {
    max_total_memory: usize,
    max_models_loaded: usize,
    memory_warning_threshold: f32,
    memory_critical_threshold: f32,
}

impl ResourceMonitor {
    pub fn check_resource_health(&self) -> ResourceHealth {
        let memory_usage = self.memory_tracker.current_usage();
        let memory_ratio = memory_usage as f32 / self.threshold_config.max_total_memory as f32;
        
        match memory_ratio {
            r if r > self.threshold_config.memory_critical_threshold => ResourceHealth::Critical,
            r if r > self.threshold_config.memory_warning_threshold => ResourceHealth::Warning,
            _ => ResourceHealth::Healthy,
        }
    }
}
```

## Caching Architecture

### Multi-Level Caching Strategy
```rust
pub struct CacheManager {
    // L1: Hot embeddings cache (in-memory, fast access)
    embedding_cache: Arc<LRUCache<String, Vec<f32>>>,
    
    // L2: Query results cache (recent queries)
    query_cache: Arc<LRUCache<QuerySignature, CachedQueryResult>>,
    
    // L3: Model inference cache (expensive operations)
    inference_cache: Arc<LRUCache<InferenceKey, InferenceResult>>,
    
    // Persistent cache for embeddings
    persistent_cache: Arc<dyn PersistentCache>,
}

impl CacheManager {
    pub async fn get_or_compute_embedding(
        &self,
        text: &str,
        model: &dyn Model,
    ) -> Result<Vec<f32>> {
        // L1 Cache check
        if let Some(embedding) = self.embedding_cache.get(text) {
            return Ok(embedding.clone());
        }
        
        // L3 Persistent cache check
        if let Some(embedding) = self.persistent_cache.get_embedding(text).await? {
            self.embedding_cache.put(text.to_string(), embedding.clone());
            return Ok(embedding);
        }
        
        // Compute and cache at all levels
        let embedding = model.generate_embedding(text).await?;
        self.embedding_cache.put(text.to_string(), embedding.clone());
        self.persistent_cache.store_embedding(text, &embedding).await?;
        
        Ok(embedding)
    }
}
```

## Query Processing Pipeline

### Enhanced Query Execution Flow
```rust
pub struct EnhancedQueryProcessor {
    layers: EnhancementLayers,
    execution_coordinator: Arc<ExecutionCoordinator>,
    result_merger: Arc<ResultMerger>,
}

impl EnhancedQueryProcessor {
    pub async fn process_query(
        &self,
        query: TripleQuery,
        mode: FindFactsMode,
    ) -> Result<EnhancedQueryResult> {
        let execution_plan = self.create_execution_plan(&query, mode).await?;
        let layer_results = self.execute_layers(execution_plan).await?;
        let merged_result = self.result_merger.merge_results(layer_results).await?;
        
        Ok(merged_result)
    }
    
    async fn create_execution_plan(
        &self,
        query: &TripleQuery,
        mode: FindFactsMode,
    ) -> Result<ExecutionPlan> {
        let mut plan = ExecutionPlan::new();
        
        // Always include exact matching (fastest path)
        plan.add_stage(ExecutionStage::ExactMatching {
            query: query.clone(),
            priority: Priority::Highest,
        });
        
        match mode {
            FindFactsMode::Exact => {}, // Only exact matching
            FindFactsMode::EntityLinked => {
                plan.add_stage(ExecutionStage::EntityLinking {
                    query: query.clone(),
                    priority: Priority::High,
                });
            },
            FindFactsMode::SemanticExpanded => {
                plan.add_stage(ExecutionStage::EntityLinking { /* ... */ });
                plan.add_stage(ExecutionStage::SemanticExpansion {
                    query: query.clone(),
                    priority: Priority::Medium,
                });
            },
            // ... other modes
        }
        
        Ok(plan)
    }
}
```

## Integration Points with Existing System

### Handler Layer Integration
```rust
// Enhanced handler that wraps existing functionality
pub async fn handle_find_facts_enhanced(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    
    // Extract enhancement parameters (backward compatible)
    let enhancement_mode = extract_enhancement_mode(&params)?;
    let base_query = extract_base_query(&params)?;
    
    // Get or create enhanced processor
    let processor = get_enhanced_processor(knowledge_engine.clone()).await?;
    
    // Process with appropriate enhancement level
    let result = processor.process_query(base_query, enhancement_mode).await?;
    
    // Format response (compatible with existing format)
    let response = format_enhanced_response(result)?;
    
    // Update usage statistics
    update_enhanced_usage_stats(usage_stats, &enhancement_mode, &result).await?;
    
    Ok(response)
}

// Fallback to existing implementation for exact mode
async fn fallback_to_existing(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    // Call original handle_find_facts implementation
    crate::mcp::llm_friendly_server::handlers::query::handle_find_facts(
        knowledge_engine,
        usage_stats,
        params,
    ).await
}
```

### Configuration Integration
```rust
pub struct EnhancementConfig {
    // Feature toggles
    pub enable_entity_linking: bool,
    pub enable_semantic_expansion: bool,
    pub enable_research_grade: bool,
    
    // Resource limits
    pub max_memory_usage: usize,
    pub max_models_loaded: usize,
    pub model_loading_timeout: Duration,
    
    // Performance tuning
    pub cache_sizes: CacheSizes,
    pub similarity_thresholds: SimilarityThresholds,
    pub batch_sizes: BatchSizes,
    
    // Model selections
    pub entity_linking_model: ModelId,
    pub semantic_expansion_model: ModelId,
    pub research_grade_models: Vec<ModelId>,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            // Conservative defaults
            enable_entity_linking: true,
            enable_semantic_expansion: false,
            enable_research_grade: false,
            
            // Resource constraints
            max_memory_usage: 2_000_000_000, // 2GB
            max_models_loaded: 3,
            model_loading_timeout: Duration::from_secs(30),
            
            // Performance defaults
            cache_sizes: CacheSizes::default(),
            similarity_thresholds: SimilarityThresholds::default(),
            batch_sizes: BatchSizes::default(),
            
            // Model defaults (most efficient)
            entity_linking_model: ModelId::new("sentence-transformers/all-MiniLM-L6-v2"),
            semantic_expansion_model: ModelId::new("HuggingFaceTB/SmolLM-135M-Instruct"),
            research_grade_models: vec![
                ModelId::new("HuggingFaceTB/SmolLM-360M-Instruct"),
            ],
        }
    }
}
```

## Error Handling and Resilience

### Graceful Degradation Strategy
```rust
pub struct GracefulDegradationHandler {
    fallback_chain: Vec<FallbackStrategy>,
    error_tracker: Arc<ErrorTracker>,
    health_monitor: Arc<HealthMonitor>,
}

#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    UseExactMatching,
    UseCachedResults,
    ReduceEnhancementLevel,
    RetryWithTimeout,
    EmergencyFallback,
}

impl GracefulDegradationHandler {
    pub async fn handle_enhancement_failure(
        &self,
        error: EnhancementError,
        original_query: &TripleQuery,
    ) -> Result<QueryResult> {
        for strategy in &self.fallback_chain {
            match strategy {
                FallbackStrategy::UseExactMatching => {
                    log::warn!("Enhancement failed, falling back to exact matching: {}", error);
                    return self.execute_exact_matching(original_query).await;
                },
                FallbackStrategy::UseCachedResults => {
                    if let Some(cached) = self.try_cached_results(original_query).await? {
                        log::info!("Using cached results for failed enhancement");
                        return Ok(cached);
                    }
                },
                // ... other strategies
            }
        }
        
        // Final fallback
        self.emergency_fallback(original_query).await
    }
}
```

## Performance Characteristics

### Latency Profiles by Enhancement Level
```rust
pub struct PerformanceProfile {
    pub mode: FindFactsMode,
    pub expected_latency: Duration,
    pub memory_overhead: usize,
    pub accuracy_improvement: f32,
    pub resource_requirements: ResourceRequirements,
}

impl PerformanceProfile {
    pub fn get_profiles() -> Vec<PerformanceProfile> {
        vec![
            PerformanceProfile {
                mode: FindFactsMode::Exact,
                expected_latency: Duration::from_millis(5),
                memory_overhead: 0,
                accuracy_improvement: 0.0,
                resource_requirements: ResourceRequirements::minimal(),
            },
            PerformanceProfile {
                mode: FindFactsMode::EntityLinked,
                expected_latency: Duration::from_millis(15),
                memory_overhead: 100_000_000, // 100MB
                accuracy_improvement: 0.35,
                resource_requirements: ResourceRequirements::low(),
            },
            PerformanceProfile {
                mode: FindFactsMode::SemanticExpanded,
                expected_latency: Duration::from_millis(80),
                memory_overhead: 800_000_000, // 800MB
                accuracy_improvement: 0.70,
                resource_requirements: ResourceRequirements::medium(),
            },
            PerformanceProfile {
                mode: FindFactsMode::ResearchGrade,
                expected_latency: Duration::from_millis(350),
                memory_overhead: 3_000_000_000, // 3GB
                accuracy_improvement: 0.90,
                resource_requirements: ResourceRequirements::high(),
            },
        ]
    }
}
```

This architecture provides a solid foundation for the three-tier enhancement system while maintaining the existing system's performance and reliability. The next documents will detail the specific implementation of each tier using the London School TDD approach.