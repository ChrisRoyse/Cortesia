# Phase 2: Semantic Intelligence

## Overview
**Duration**: 4 weeks  
**Goal**: Implement deep semantic understanding with read-optimized database  
**Priority**: HIGH  
**Dependencies**: Phase 1 completion  
**Target Performance**: <3ms per embedding on Intel i9 processor

## Multi-Database Architecture - Phase 2
**New in Phase 2**: Implement CQRS pattern with read-optimized database
- **Read-Optimized Graph**: Denormalized views for fast queries
- **Materialized Projections**: Event-driven updates from event store
- **Cached Embeddings**: Pre-computed embeddings for similarity search
- **Query Router**: Intelligent routing between write and read models

## Infrastructure Integration Strategy
**Leverage Existing Systems**:
- **EmbeddingStore Integration**: Use existing `src/embedding/store.rs` with ProductQuantizer for memory efficiency
- **String Interner Integration**: Leverage `src/storage/string_interner.rs` for entity name deduplication
- **Zero-Copy Operations**: Use `src/core/zero_copy_types.rs` patterns for high-performance access
- **Batch Processing**: Extend existing `src/streaming/update_handler.rs` for semantic updates
- **MCP Enhancement**: Update existing MCP handlers in `src/mcp/llm_friendly_server/handlers/` for semantic capabilities
- **CQRS Pattern**: Maintain existing read/write separation with semantic layer on top

## AI Model Integration (Rust/Candle)
**Available Models in src/models**:
- **all-MiniLM-L6-v2** (22M params) - Primary embedding model
- **DistilBERT-NER** (66M params) - Can be used for high-quality embeddings
- **T5-Small** (60M params) - For advanced text understanding
- **Dependency Parser** - For syntactic analysis
- **Intent Classifier** - For query intent understanding
- All models ported to Rust using Candle framework

## Week 5: Semantic Understanding Framework

### Task 5.1: Implement AI-Powered Embeddings with Infrastructure Integration
**File**: `src/semantic/embeddings.rs` (new file)
```rust
use candle_core::{Device, Tensor};
use crate::models::{AllMiniLM, DistilBertNER};
use crate::embedding::store::{EmbeddingStore, ProductQuantizer};
use crate::storage::string_interner::{StringInterner, InternedString};
use crate::core::zero_copy_types::ZeroCopyEntityInfo;

pub struct EmbeddingEngine {
    // Primary model: MiniLM optimized for speed (Rust/Candle)
    primary_model: AllMiniLM,
    // Quality model: DistilBERT for important embeddings
    quality_model: DistilBertNER,  // Reuse NER model for embeddings
    
    // INTEGRATION: Use existing EmbeddingStore for quantized storage
    embedding_store: Arc<RwLock<EmbeddingStore>>,
    
    // INTEGRATION: Use StringInterner for memory-efficient text storage
    string_interner: Arc<StringInterner>,
    
    // Multi-level cache (no static embeddings in Rust port)
    l1_cache: DashMap<InternedString, Vec<f32>>,      // Hot cache with interned keys
    l2_cache: Arc<Mutex<LruCache<InternedString, Vec<f32>>>>,  // Warm cache
    
    // INTEGRATION: Use existing BatchProcessor patterns
    batch_processor: Arc<BatchProcessor>,
    
    // Device for computation
    device: Device,
}

impl EmbeddingEngine {
    pub fn new(
        embedding_store: Arc<RwLock<EmbeddingStore>>,
        string_interner: Arc<StringInterner>,
        batch_processor: Arc<BatchProcessor>
    ) -> Result<Self> {
        // Initialize Candle models with i9 optimizations
        let primary_model = AllMiniLM::load(
            "models/all-MiniLM-L6-v2-int8.safetensors",
            &Device::Cpu, // Use CPU with SIMD optimizations
        )?;
        
        let quality_model = DistilBertNER::load(
            "models/distilbert-ner-int8.safetensors",
            &Device::Cpu,
        )?;
        
        Ok(Self {
            primary_model,
            quality_model,
            embedding_store,
            string_interner,
            l1_cache: DashMap::with_capacity(10_000),
            l2_cache: Arc::new(Mutex::new(LruCache::new(100_000))),
            batch_processor,
            device: Device::Cpu,
        })
    }
    
    pub async fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // INTEGRATION: Use StringInterner for efficient text storage
        let text_id = self.string_interner.intern(text);
        
        // Check L1 cache (fastest)
        if let Some(cached) = self.l1_cache.get(&text_id) {
            return Ok(cached.clone());
        }
        
        // Check L2 cache
        if let Some(cached) = self.l2_cache.lock().get(&text_id) {
            self.l1_cache.insert(text_id, cached.clone());
            return Ok(cached);
        }
        
        // INTEGRATION: Check if already stored in EmbeddingStore
        if let Ok(stored_embedding) = self.embedding_store.read().get_embedding_by_text_id(text_id) {
            self.l1_cache.insert(text_id, stored_embedding.clone());
            return Ok(stored_embedding);
        }
        
        // Use appropriate model based on text length
        let embedding = if text.len() < 100 {
            // Fast model for short texts
            self.primary_model.encode(text).await?
        } else {
            // Quality model for longer texts
            self.quality_model.encode(text).await?
        };
        
        // INTEGRATION: Store in EmbeddingStore with quantization
        let embedding_offset = self.embedding_store.write().store_embedding(&embedding)?;
        
        // Update caches with zero-copy patterns
        self.update_caches_zero_copy(text_id, &embedding);
        
        Ok(embedding)
    }
    
    pub async fn batch_encode(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        // INTEGRATION: Use existing BatchProcessor for efficient batch handling
        let batch_request = BatchRequest {
            texts: texts.clone(),
            priority: BatchPriority::Normal,
            callback: None,
        };
        
        // Process through batch processor with integrated caching
        let results = self.batch_processor.process_embedding_batch(batch_request).await?;
        
        Ok(results)
    }
    
    // INTEGRATION: Zero-copy cache update method
    fn update_caches_zero_copy(&self, text_id: InternedString, embedding: &[f32]) {
        // Use zero-copy patterns for efficient cache updates
        let embedding_vec = embedding.to_vec();
        self.l1_cache.insert(text_id, embedding_vec.clone());
        
        // Update L2 cache with LRU eviction
        self.l2_cache.lock().put(text_id, embedding_vec);
    }
    
    pub fn semantic_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        // Use SIMD-accelerated cosine similarity
        simd_cosine_similarity(emb1, emb2)
    }
}

// SIMD acceleration for i9
#[cfg(target_arch = "x86_64")]
fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    
    unsafe {
        let mut dot_product = _mm256_setzero_ps();
        let mut norm_a = _mm256_setzero_ps();
        let mut norm_b = _mm256_setzero_ps();
        
        for i in (0..a.len()).step_by(8) {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            
            dot_product = _mm256_fmadd_ps(va, vb, dot_product);
            norm_a = _mm256_fmadd_ps(va, va, norm_a);
            norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
        }
        
        let dot = hsum_ps_avx(dot_product);
        let na = hsum_ps_avx(norm_a).sqrt();
        let nb = hsum_ps_avx(norm_b).sqrt();
        
        dot / (na * nb)
    }
}
```

### Task 5.2: Build Concept Hierarchy
**File**: `src/semantic/concept_hierarchy.rs` (new file)
```rust
pub struct ConceptHierarchy {
    root: ConceptNode,
    index: HashMap<String, NodeId>,
}

pub struct ConceptNode {
    id: NodeId,
    name: String,
    embedding: Vec<f32>,
    hypernyms: Vec<NodeId>,  // parent concepts
    hyponyms: Vec<NodeId>,   // child concepts
    properties: HashMap<String, Value>,
}

impl ConceptHierarchy {
    pub fn find_common_ancestor(&self, concept1: &str, concept2: &str) -> Option<String> {
        // Walk up hierarchy to find common parent
    }
    
    pub fn inherit_properties(&self, concept: &str) -> HashMap<String, Value> {
        // Collect properties from ancestors
    }
}
```

### Task 5.3: AI-Optimized Semantic Index with Infrastructure Integration
**File**: `src/semantic/semantic_index.rs` (new file)
```rust
use crate::embedding::store::EmbeddingStore;
use crate::embedding::simd_search::SIMDSearch;
use crate::storage::lsh::LSHIndex;
use crate::core::zero_copy_types::{ZeroCopyEntityInfo, ZeroCopySearchResult};

pub struct SemanticIndex {
    // INTEGRATION: Use existing EmbeddingStore for quantized storage
    embedding_store: Arc<RwLock<EmbeddingStore>>,
    
    // INTEGRATION: Use existing SIMD search capabilities
    simd_search: Arc<SIMDSearch>,
    
    // INTEGRATION: Use existing LSH index
    lsh_index: Arc<LSHIndex>,
    
    // Embedding engine with integrated infrastructure
    embedding_engine: Arc<EmbeddingEngine>,
    
    // INTEGRATION: Use existing zero-copy types for results
    entity_info_cache: DashMap<EntityId, ZeroCopyEntityInfo>,
    
    // Index statistics
    stats: IndexStats,
}

impl SemanticIndex {
    pub fn new(
        embedding_engine: Arc<EmbeddingEngine>,
        embedding_store: Arc<RwLock<EmbeddingStore>>,
        simd_search: Arc<SIMDSearch>,
        lsh_index: Arc<LSHIndex>
    ) -> Result<Self> {
        Ok(Self {
            embedding_store,
            simd_search,
            lsh_index,
            embedding_engine,
            entity_info_cache: DashMap::with_capacity(1_000_000),
            stats: IndexStats::default(),
        })
    }
    
    pub async fn add_entity(&self, entity: &Entity) -> Result<()> {
        // Get or compute embedding
        let embedding = if let Some(emb) = &entity.embedding {
            emb.clone()
        } else {
            self.embedding_engine.encode_text(&entity.description).await?
        };
        
        // INTEGRATION: Store in EmbeddingStore with quantization
        let embedding_offset = self.embedding_store.write().store_embedding(&embedding)?;
        
        // INTEGRATION: Create ZeroCopyEntityInfo for efficient access
        let entity_info = ZeroCopyEntityInfo {
            id: entity.id,
            type_id: entity.type_id,
            degree: entity.degree,
            properties: entity.properties.clone(),
            embedding: embedding.clone(),
        };
        self.entity_info_cache.insert(entity.id, entity_info);
        
        // INTEGRATION: Add to LSH index for fast approximate search
        self.lsh_index.add_vector(entity.id, &embedding).await?;
        
        // INTEGRATION: Update SIMD search index
        self.simd_search.add_vector(entity.id, &embedding).await?;
        
        self.stats.increment_entities();
        Ok(())
    }
    
    pub async fn semantic_search(
        &self, 
        query: &str, 
        k: usize,
        search_mode: SearchMode
    ) -> Result<Vec<ZeroCopySearchResult>> {
        // Encode query
        let query_embedding = self.embedding_engine.encode_text(query).await?;
        
        let search_results = match search_mode {
            SearchMode::Fast => {
                // INTEGRATION: Use existing LSH for ultra-fast approximate search
                self.lsh_index.search(&query_embedding, k * 2).await?
                    .into_iter()
                    .take(k)
                    .collect()
            },
            SearchMode::Accurate => {
                // INTEGRATION: Use SIMD search for accurate high-performance search
                self.simd_search.search(&query_embedding, k).await?
            },
            SearchMode::Hybrid => {
                // Use LSH to get candidates, then refine with SIMD similarity
                let candidates = self.lsh_index.search(&query_embedding, k * 5).await?;
                self.refine_with_simd_similarity(candidates, &query_embedding, k).await?
            },
        };
        
        // INTEGRATION: Convert to ZeroCopySearchResult for efficient transfer
        let zero_copy_results = search_results.into_iter()
            .filter_map(|(entity_id, score)| {
                self.entity_info_cache.get(&entity_id).map(|info| {
                    ZeroCopySearchResult {
                        entity_id,
                        score,
                        entity_info: info.clone(),
                    }
                })
            })
            .collect();
            
        Ok(zero_copy_results)
    }
    
    pub async fn batch_search(
        &self,
        queries: Vec<&str>,
        k: usize
    ) -> Result<Vec<Vec<ZeroCopySearchResult>>> {
        // INTEGRATION: Use BatchProcessor for efficient batch encoding
        let embeddings = self.embedding_engine.batch_encode(queries).await?;
        
        // INTEGRATION: Use SIMD search for parallel processing
        let results = self.simd_search.batch_search(&embeddings, k).await?;
        
        // Convert to zero-copy results
        let zero_copy_results: Vec<_> = results.into_iter()
            .map(|batch_result| {
                batch_result.into_iter()
                    .filter_map(|(entity_id, score)| {
                        self.entity_info_cache.get(&entity_id).map(|info| {
                            ZeroCopySearchResult {
                                entity_id,
                                score,
                                entity_info: info.clone(),
                            }
                        })
                    })
                    .collect()
            })
            .collect();
            
        Ok(zero_copy_results)
    }
    
    // INTEGRATION: Use SIMD for refinement instead of exact similarity
    async fn refine_with_simd_similarity(
        &self,
        candidates: Vec<(EntityId, f32)>,
        query_embedding: &[f32],
        k: usize
    ) -> Result<Vec<(EntityId, f32)>> {
        // Extract candidate IDs
        let candidate_ids: Vec<_> = candidates.into_iter().map(|(id, _)| id).collect();
        
        // Use SIMD search for refinement
        let refined_results = self.simd_search.search_within_candidates(
            query_embedding,
            &candidate_ids,
            k
        ).await?;
        
        Ok(refined_results)
    }
}

pub enum SearchMode {
    Fast,      // LSH - <1ms, 85% recall
    Accurate,  // HNSW - <5ms, 99% recall
    Hybrid,    // LSH + reranking - <3ms, 95% recall
}
```

## Week 6: Advanced Query Understanding

### Task 6.1: AI-Powered Natural Language Query Parser
**File**: `src/semantic/query_parser.rs` (new file)
```rust
use crate::models::{DistilBertNER, DependencyParser, IntentClassifier};\n\npub struct QueryParser {
    // Query understanding model (Rust/Candle)
    query_model: DistilBertNER,  // Reuse NER model for query embeddings
    // Dependency parser
    dependency_parser: DependencyParser,  // From src/models
    // Intent classifier
    intent_classifier: IntentClassifier,  // From src/models
    // Entity extractor (from Phase 1)
    entity_extractor: Arc<EntityExtractor>,
    // Constraint parser
    constraint_parser: ConstraintParser,
    // Device for computation
    device: Device,
}

pub struct ParsedQuery {
    intent: QueryIntent,
    entities: Vec<IdentifiedEntity>,
    constraints: Vec<Constraint>,
    aggregations: Vec<Aggregation>,
    temporal_scope: Option<TimeRange>,
    semantic_embedding: Vec<f32>,
    complexity_score: f32,
    confidence: f32,
}

impl QueryParser {
    pub async fn parse_natural_query(&self, query: &str) -> ParsedQuery {
        // Parallel processing for all components
        let (
            intent,
            entities,
            dependencies,
            embedding,
            constraints
        ) = tokio::join!(
            self.classify_intent(query),
            self.entity_extractor.extract_entities(query),
            self.dependency_parser.parse(query),
            self.query_model.encode(query),
            self.extract_constraints(query)
        );
        
        // Analyze query complexity
        let complexity_score = self.calculate_complexity(&intent, &entities, &constraints);
        
        // Build parsed query
        ParsedQuery {
            intent: intent.intent_type,
            entities: self.identify_entity_roles(entities, &dependencies),
            constraints,
            aggregations: self.detect_aggregations(query, &dependencies),
            temporal_scope: self.extract_temporal_scope(query),
            semantic_embedding: embedding,
            complexity_score,
            confidence: intent.confidence,
        }
    }
    
    async fn classify_intent(&self, query: &str) -> IntentResult {
        // Use fine-tuned model for intent classification
        let encoding = self.query_model.encode(query).await?;
        let intent = self.intent_classifier.classify(encoding).await?;
        
        // Boost confidence with pattern matching
        let pattern_intent = self.pattern_based_intent(query);
        if pattern_intent == intent.intent_type {
            IntentResult {
                intent_type: intent.intent_type,
                confidence: (intent.confidence + 0.1).min(1.0),
                sub_intents: intent.sub_intents,
            }
        } else {
            intent
        }
    }
    
    fn identify_entity_roles(
        &self,
        entities: Vec<Entity>,
        dependencies: &DependencyTree
    ) -> Vec<IdentifiedEntity> {
        entities.into_iter()
            .map(|entity| {
                let role = dependencies.get_syntactic_role(&entity.name);
                let semantic_role = self.infer_semantic_role(&entity, role);
                
                IdentifiedEntity {
                    entity,
                    syntactic_role: role,
                    semantic_role,
                    importance: self.calculate_entity_importance(&entity, dependencies),
                }
            })
            .collect()
    }
}

pub enum QueryIntent {
    FactRetrieval {
        fact_type: FactType,
        confidence: f32,
    },
    Relationship {
        relationship_type: Option<RelationshipType>,
        direction: RelationshipDirection,
    },
    Causal {
        cause_entity: Option<String>,
        effect_entity: Option<String>,
    },
    Temporal {
        temporal_type: TemporalQueryType,
        reference_time: Option<DateTime>,
    },
    Comparison {
        comparison_type: ComparisonType,
        aspects: Vec<String>,
    },
    Aggregation {
        aggregation_type: AggregationType,
        group_by: Option<String>,
    },
    Process {
        process_type: ProcessType,
        steps_requested: bool,
    },
    Complex {
        sub_intents: Vec<QueryIntent>,
        logical_operators: Vec<LogicalOperator>,
    },
}

// Enhanced constraint detection with AI
impl QueryParser {
    async fn extract_constraints(&self, query: &str) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        
        // Numerical constraints
        if let Some(num_constraints) = self.extract_numerical_constraints(query).await {
            constraints.extend(num_constraints);
        }
        
        // Temporal constraints
        if let Some(temp_constraints) = self.extract_temporal_constraints(query).await {
            constraints.extend(temp_constraints);
        }
        
        // Categorical constraints
        if let Some(cat_constraints) = self.extract_categorical_constraints(query).await {
            constraints.extend(cat_constraints);
        }
        
        // Logical constraints
        if let Some(log_constraints) = self.extract_logical_constraints(query).await {
            constraints.extend(log_constraints);
        }
        
        constraints
    }
}
```

### Task 6.2: Semantic Query Expansion
**File**: `src/semantic/query_expansion.rs` (new file)
```rust
pub struct QueryExpander {
    thesaurus: Thesaurus,
    concept_net: ConceptNet,
}

impl QueryExpander {
    pub fn expand_query(&self, query: &ParsedQuery) -> ExpandedQuery {
        // Add synonyms
        // Include related concepts
        // Add domain-specific terms
        // Consider context
    }
    
    pub fn generate_subqueries(&self, query: &ParsedQuery) -> Vec<ParsedQuery> {
        // Break complex queries into simpler parts
        // Handle multi-hop reasoning
    }
}
```

### Task 6.3: Intelligent Answer Synthesis
**File**: `src/semantic/answer_synthesis.rs` (new file)
```rust
pub struct AnswerSynthesizer {
    template_engine: TemplateEngine,
    summarizer: TextSummarizer,
}

impl AnswerSynthesizer {
    pub fn synthesize_answer(&self, 
        query: &ParsedQuery, 
        facts: Vec<Fact>,
        context: &Context
    ) -> Answer {
        match query.intent {
            QueryIntent::Comparison => self.generate_comparison(facts),
            QueryIntent::Process => self.generate_process_explanation(facts),
            QueryIntent::Causal => self.generate_causal_chain(facts),
            _ => self.generate_factual_answer(facts),
        }
    }
    
    fn generate_comparison(&self, facts: Vec<Fact>) -> Answer {
        // Structure comparative analysis
        // Highlight similarities and differences
        // Use parallel structure
    }
}
```

## Week 7: Semantic Relationship Network

### Task 7.1: Relationship Type Inference
**File**: `src/semantic/relationship_inference.rs` (new file)
```rust
pub struct RelationshipInferencer {
    patterns: Vec<InferenceRule>,
    ml_model: RelationshipClassifier,
}

impl RelationshipInferencer {
    pub fn infer_relationships(&self, entity1: &Entity, entity2: &Entity) -> Vec<InferredRelationship> {
        // Use rules and ML to infer relationships
        // Consider transitive properties
        // Handle uncertainty
    }
    
    pub fn validate_relationship(&self, rel: &Relationship) -> ValidationResult {
        // Check semantic consistency
        // Verify against ontology
        // Flag contradictions
    }
}

pub struct InferenceRule {
    pattern: Pattern,
    inferred_relation: RelationshipType,
    confidence: f32,
}
```

### Task 7.2: Semantic Network Navigation
**File**: `src/semantic/network_navigator.rs` (new file)
```rust
pub struct NetworkNavigator {
    graph: SemanticGraph,
    pathfinder: PathFinder,
}

impl NetworkNavigator {
    pub fn find_semantic_path(&self, 
        start: &str, 
        end: &str, 
        max_hops: usize
    ) -> Option<SemanticPath> {
        // Find meaningful connection paths
        // Weight by semantic similarity
        // Prefer stronger relationships
    }
    
    pub fn explore_neighborhood(&self, 
        entity: &str, 
        radius: usize, 
        filter: Option<RelationshipFilter>
    ) -> SemanticNeighborhood {
        // Get semantically related entities
        // Apply relationship filters
        // Rank by relevance
    }
}
```

### Task 7.3: Semantic Consistency Checker
**File**: `src/semantic/consistency_checker.rs` (new file)
```rust
pub struct ConsistencyChecker {
    rules: Vec<ConsistencyRule>,
    contradiction_detector: ContradictionDetector,
}

impl ConsistencyChecker {
    pub fn check_consistency(&self, graph: &KnowledgeGraph) -> ConsistencyReport {
        // Find logical contradictions
        // Detect circular definitions
        // Identify missing relationships
        // Suggest fixes
    }
    
    pub fn validate_new_fact(&self, fact: &Fact, graph: &KnowledgeGraph) -> ValidationResult {
        // Check against existing knowledge
        // Identify conflicts
        // Suggest integration approach
    }
}
```

## Week 8: Integration and Optimization

### Task 8.1: Semantic Cache Implementation
**File**: `src/semantic/semantic_cache.rs` (new file)
```rust
pub struct SemanticCache {
    embedding_cache: LruCache<String, Vec<f32>>,
    query_cache: LruCache<QueryHash, Answer>,
    similarity_cache: LruCache<(String, String), f32>,
}

impl SemanticCache {
    pub fn get_similar_query(&self, query: &str) -> Option<&Answer> {
        // Find semantically similar cached queries
        // Return if similarity > threshold
    }
    
    pub fn invalidate_related(&mut self, entity: &str) {
        // Remove cache entries related to entity
        // Use semantic similarity for fuzzy matching
    }
}
```

### Task 8.2: Batch Processing Pipeline
**File**: `src/semantic/batch_processor.rs` (new file)
```rust
pub struct BatchProcessor {
    embedding_engine: EmbeddingEngine,
    relationship_extractor: RelationshipExtractor,
}

impl BatchProcessor {
    pub async fn process_documents(&self, docs: Vec<Document>) -> ProcessingResult {
        // Batch encode all documents
        // Extract entities and relationships in parallel
        // Deduplicate and merge
        // Update indices
    }
}
```

### Task 8.3: MCP Handler Integration and Enhancement
**File**: `src/mcp/llm_friendly_server/handlers/semantic.rs` (new file)
```rust
use crate::semantic::{SemanticIndex, EmbeddingEngine, QueryParser};
use crate::core::zero_copy_types::ZeroCopySearchResult;

/// INTEGRATION: Enhanced ask_question handler with semantic understanding
pub async fn handle_ask_question_semantic(params: Value) -> Result<Value> {
    let question = params["question"].as_str().unwrap();
    let max_results = params["max_results"].as_u64().unwrap_or(5) as usize;
    let context = params["context"].as_str();
    
    // Parse query with AI-powered understanding
    let parsed_query = query_parser.parse_natural_query(question).await?;
    
    // Perform semantic search with appropriate mode
    let search_mode = match parsed_query.complexity_score {
        score if score < 0.3 => SearchMode::Fast,
        score if score < 0.7 => SearchMode::Hybrid,
        _ => SearchMode::Accurate,
    };
    
    let semantic_results = semantic_index.semantic_search(
        question,
        max_results * 2, // Get extra for filtering
        search_mode
    ).await?;
    
    // Filter and rank results based on query intent
    let filtered_results = filter_by_intent(&semantic_results, &parsed_query.intent);
    
    // Synthesize answer using semantic understanding
    let answer = answer_synthesizer.synthesize_answer(
        &parsed_query,
        filtered_results,
        context
    ).await?;
    
    Ok(json!({
        "question": question,
        "answer": answer.text,
        "confidence": answer.confidence,
        "sources": answer.sources,
        "semantic_expansion": parsed_query.expansion_terms,
        "query_intent": parsed_query.intent,
    }))
}

/// INTEGRATION: Enhanced hybrid_search with semantic capabilities
pub async fn handle_hybrid_search_semantic(params: Value) -> Result<Value> {
    let query = params["query"].as_str().unwrap();
    let limit = params["limit"].as_u64().unwrap_or(10) as usize;
    let search_type = params["search_type"].as_str().unwrap_or("hybrid");
    let performance_mode = params["performance_mode"].as_str().unwrap_or("standard");
    
    // Parse query semantically
    let parsed_query = query_parser.parse_natural_query(query).await?;
    
    // Choose search strategy based on performance mode and query complexity
    let search_mode = match (performance_mode, parsed_query.complexity_score) {
        ("lsh", _) => SearchMode::Fast,
        ("simd", _) => SearchMode::Accurate,
        (_, score) if score < 0.3 => SearchMode::Fast,
        _ => SearchMode::Hybrid,
    };
    
    // Perform semantic search
    let semantic_results = semantic_index.semantic_search(
        query,
        limit,
        search_mode
    ).await?;
    
    // Expand query for related concepts
    let expanded_query = query_expander.expand_query(&parsed_query);
    let expanded_results = if expanded_query.expansion_terms.len() > 0 {
        let mut all_results = semantic_results;
        for expansion_term in &expanded_query.expansion_terms {
            let expansion_results = semantic_index.semantic_search(
                expansion_term,
                limit / 2,
                SearchMode::Fast
            ).await?;
            all_results.extend(expansion_results);
        }
        // Deduplicate and re-rank
        deduplicate_and_rank(all_results, limit)
    } else {
        semantic_results
    };
    
    Ok(json!({
        "query": query,
        "results": expanded_results.into_iter().map(|r| json!({
            "entity_id": r.entity_id,
            "score": r.score,
            "entity_info": r.entity_info,
        })).collect::<Vec<_>>(),
        "query_analysis": {
            "intent": parsed_query.intent,
            "complexity": parsed_query.complexity_score,
            "entities": parsed_query.entities,
            "constraints": parsed_query.constraints,
        },
        "expansion_terms": expanded_query.expansion_terms,
        "performance_stats": {
            "search_mode": format!("{:?}", search_mode),
            "processing_time_ms": "calculated_in_actual_implementation",
        }
    }))
}

pub async fn handle_concept_hierarchy(params: Value) -> Result<Value> {
    let concept = params["concept"].as_str().unwrap();
    
    let hierarchy = concept_hierarchy.get_hierarchy(concept);
    
    Ok(json!({
        "concept": concept,
        "parents": hierarchy.parents,
        "children": hierarchy.children,
        "properties": hierarchy.inherited_properties,
    }))
}

// INTEGRATION: Helper functions using existing infrastructure
fn filter_by_intent(
    results: &[ZeroCopySearchResult], 
    intent: &QueryIntent
) -> Vec<ZeroCopySearchResult> {
    // Filter results based on query intent using zero-copy patterns
    results.iter()
        .filter(|result| match intent {
            QueryIntent::FactRetrieval { fact_type, .. } => {
                // Filter by fact type
                result.entity_info.type_id == fact_type.to_type_id()
            },
            QueryIntent::Relationship { relationship_type, .. } => {
                // Filter by relationship relevance
                has_relationship_relevance(&result.entity_info, relationship_type)
            },
            _ => true, // Include all for other intent types
        })
        .cloned()
        .collect()
}

fn deduplicate_and_rank(
    mut results: Vec<ZeroCopySearchResult>, 
    limit: usize
) -> Vec<ZeroCopySearchResult> {
    // Use existing deduplication patterns with zero-copy efficiency
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.dedup_by(|a, b| a.entity_id == b.entity_id);
    results.truncate(limit);
    results
}
```

### Task 8.4: Performance Optimization with Infrastructure Integration
```rust
// INTEGRATION-FOCUSED Optimizations to implement:

1. **EmbeddingStore Integration Optimizations**:
   - Use existing ProductQuantizer for 75% memory reduction
   - Implement batch quantization for 50% faster storage
   - Add compression-aware similarity search

2. **StringInterner Memory Optimizations**:
   - Cache frequently accessed entity names (90% hit rate target)
   - Use interned strings for all semantic operations
   - Reduce string allocation overhead by 80%

3. **Zero-Copy Performance Enhancements**:
   - Use ZeroCopyEntityInfo for result transfers (no serialization overhead)
   - Implement zero-copy embedding access patterns
   - Reduce memory copies in search results by 95%

4. **BatchProcessor Integration**:
   - Extend existing batch processing for semantic operations
   - Use StreamingUpdateHandler for real-time embedding updates
   - Implement priority-based batch scheduling

5. **SIMD Search Integration**:
   - Leverage existing SIMD similarity calculations
   - Use hardware-accelerated distance computations
   - Implement SIMD-optimized query expansion

6. **LSH Index Optimizations**:
   - Tune existing LSH parameters for semantic search
   - Implement adaptive hash table sizing
   - Use LSH for fast query pre-filtering

7. **Caching Strategy with Existing Infrastructure**:
   - Integrate with existing cache hierarchies
   - Use memory-mapped storage for large embeddings
   - Implement semantic-aware cache eviction policies

8. **Model Loading Optimizations**:
   - Use existing model loading infrastructure from src/models
   - Implement lazy loading for secondary models
   - Add model-specific optimization flags

// Performance Targets with Integration:
// - Embedding generation: <2ms (vs 3ms baseline) due to quantization
// - Semantic search: <3ms (vs 5ms baseline) due to SIMD + LSH
// - Memory usage: 60% reduction due to StringInterner + ProductQuantizer
// - Batch processing: 50% faster due to existing BatchProcessor patterns
```

## Deliverables
1. **Integrated AI-powered embedding system** using existing EmbeddingStore with ProductQuantizer
2. **Multi-level semantic index** with existing LSH/SIMD search integration
3. **Neural query parser** with intent classification using existing model infrastructure
4. **Intelligent query expansion** with semantic understanding
5. **Answer synthesis** with T5-Small generation from src/models
6. **Enhanced MCP handlers** with semantic capabilities for ask_question and hybrid_search
7. **Batch processing pipeline** using existing StreamingUpdateHandler patterns
8. **Zero-copy semantic operations** with ZeroCopyEntityInfo integration

## Success Criteria with Infrastructure Integration
- [ ] Embedding generation: <2ms per text (improved from 3ms due to quantization)
- [ ] Semantic search accuracy > 95% (using SIMD + LSH hybrid approach)
- [ ] Query parsing handles 20+ intent types with 90% accuracy
- [ ] Semantic similarity computation: <0.3ms with existing SIMD optimizations
- [ ] Batch processing: 15,000+ embeddings/second (50% improvement with existing BatchProcessor)
- [ ] End-to-end query response: <35ms (improved through zero-copy patterns)
- [ ] Memory usage: <2.5GB for 1M embeddings (60% reduction via StringInterner + ProductQuantizer)
- [ ] MCP handler enhancement: ask_question and hybrid_search support semantic understanding
- [ ] Zero-copy operations: 95% reduction in memory copies for search results

## Performance Benchmarks (Intel i9) with Infrastructure Integration
- Single embedding (MiniLM + ProductQuantizer): 1.5-2ms (improved)
- Batch embedding (64 texts + BatchProcessor): 20ms (improved)
- Semantic search (SIMD + LSH hybrid, 1M vectors): 2-3ms (improved)
- Semantic search (LSH only, 1M vectors): <0.5ms (improved)
- Query parsing (full pipeline + StringInterner): 6ms (improved)
- Answer synthesis (template + zero-copy): 1ms (improved)
- Answer synthesis (T5-Small from src/models): 18ms (improved)
- MCP handler response (with semantic enhancement): 35ms total
- Zero-copy result transfer: <0.1ms (vs 2ms with serialization)

## Dependencies with Infrastructure Integration
- **Existing Infrastructure**:
  - `src/embedding/store.rs` - EmbeddingStore with ProductQuantizer
  - `src/storage/string_interner.rs` - StringInterner for memory efficiency
  - `src/embedding/simd_search.rs` - SIMD-accelerated similarity search
  - `src/storage/lsh.rs` - LSH index for fast approximate search
  - `src/streaming/update_handler.rs` - BatchProcessor patterns
  - `src/core/zero_copy_types.rs` - Zero-copy result types
  - `src/mcp/llm_friendly_server/handlers/` - Existing MCP handlers to enhance

- **AI Models from src/models** (all ported to Rust/Candle):
  - all-MiniLM-L6-v2 (22M params) - Primary embedding model
  - DistilBERT-NER (66M params) - For quality embeddings and NER
  - T5-Small (60M params) - For answer synthesis
  - Dependency Parser - For syntactic analysis
  - Intent Classifier - For query understanding

- **New Semantic Components**:
  - Candle framework for model inference
  - Semantic query parser and expander
  - Answer synthesis engine
  - Concept hierarchy builder

## Risks & Mitigations with Infrastructure Integration
1. **Embedding quality vs speed tradeoff**
   - Mitigation: Dual model approach with existing ProductQuantizer, quality tiers based on text length
   - Integration: Use existing EmbeddingStore compression to maintain quality while reducing memory usage

2. **Index memory growth**
   - Mitigation: Leverage existing StringInterner for 80% string deduplication, use LSH for approximate indexing
   - Integration: Use existing memory-mapped storage and periodic pruning patterns

3. **Query complexity explosion**
   - Mitigation: Query simplification with existing BatchProcessor, semantic caching with zero-copy patterns
   - Integration: Use existing streaming update handlers for real-time query optimization

4. **Integration complexity with existing systems**
   - Mitigation: Gradual integration approach, maintain backward compatibility with existing MCP handlers
   - Integration: Extensive testing with existing infrastructure, fallback to current implementations

5. **Performance regression from additional layers**
   - Mitigation: Zero-copy operations, SIMD optimizations, existing batch processing patterns
   - Integration: Performance monitoring with existing metrics system, A/B testing between old and new handlers

6. **Model loading and memory overhead**
   - Mitigation: Share models across semantic components, lazy loading from existing src/models infrastructure
   - Integration: Use existing model caching patterns, coordinate with other AI-powered phases