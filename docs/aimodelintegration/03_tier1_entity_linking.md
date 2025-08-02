# Tier 1 Implementation: Entity Linking Foundation
## MiniLM-Based Entity Normalization and Alias Resolution

### Tier 1 Overview

#### Objective
Implement entity linking and normalization using MiniLM-L6-v2 (22M parameters) to resolve entity name variations and improve query recall by 25-40% with minimal performance impact (+5-15ms, +100MB memory).

#### Core Capabilities
1. **Entity Normalization**: "Einstein" → "Albert Einstein"
2. **Alias Resolution**: Multiple name variations to canonical form
3. **Embedding-Based Similarity**: Vector similarity for entity matching
4. **Fast Inference**: 22M parameter model for sub-15ms response times
5. **Caching Strategy**: Aggressive caching to amortize inference costs

### Technical Architecture

#### Core Components
```rust
// src/enhanced_find_facts/entity_linking/mod.rs
pub mod entity_linker;
pub mod embedding_cache;
pub mod alias_resolver;
pub mod normalization_engine;

pub use entity_linker::{EntityLinker, MiniLMEntityLinker, LinkedEntity};
pub use embedding_cache::{EmbeddingCache, EntityEmbeddingIndex};
pub use alias_resolver::{AliasResolver, AliasResolutionResult};
pub use normalization_engine::{NormalizationEngine, NormalizationStrategy};
```

#### Entity Linking Layer
```rust
// src/enhanced_find_facts/entity_linking/entity_linker.rs

use crate::models::minilm::{all_minilm_l6_v2, MiniLMVariant};
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait EntityLinker: Send + Sync {
    async fn link_entity(&self, mention: &str) -> Result<Vec<LinkedEntity>>;
    async fn normalize_entity(&self, entity: &str) -> Result<String>;
    async fn find_aliases(&self, canonical: &str) -> Result<Vec<String>>;
    async fn compute_similarity(&self, entity1: &str, entity2: &str) -> Result<f32>;
}

#[derive(Debug, Clone)]
pub struct LinkedEntity {
    pub canonical_name: String,
    pub confidence: f32,
    pub aliases: Vec<String>,
    pub entity_type: Option<String>,
}

pub struct MiniLMEntityLinker {
    model: Arc<dyn Model>,
    embedding_cache: Arc<EmbeddingCache>,
    entity_index: Arc<EntityEmbeddingIndex>,
    config: EntityLinkingConfig,
}

impl MiniLMEntityLinker {
    pub async fn new(config: EntityLinkingConfig) -> Result<Self> {
        let model = all_minilm_l6_v2()
            .with_config(ModelConfig {
                device: config.device.clone(),
                max_sequence_length: 256,
                ..Default::default()
            })
            .build()?;
        
        let embedding_cache = Arc::new(EmbeddingCache::new(config.cache_size));
        let entity_index = Arc::new(EntityEmbeddingIndex::new(config.index_config.clone())?);
        
        Ok(Self {
            model: Arc::new(model),
            embedding_cache,
            entity_index,
            config,
        })
    }
    
    pub async fn load_entity_knowledge(&self, knowledge_engine: &KnowledgeEngine) -> Result<()> {
        // Build entity index from existing knowledge base
        let entities = knowledge_engine.get_all_entities().await?;
        
        for entity in entities {
            let embedding = self.compute_embedding_uncached(&entity.name).await?;
            self.entity_index.add_entity(EntityEmbedding {
                name: entity.name.clone(),
                embedding,
                aliases: entity.aliases.clone(),
                entity_type: entity.entity_type.clone(),
            }).await?;
        }
        
        self.entity_index.build_index().await?;
        Ok(())
    }
}

#[async_trait]
impl EntityLinker for MiniLMEntityLinker {
    async fn link_entity(&self, mention: &str) -> Result<Vec<LinkedEntity>> {
        // Generate embedding for the mention
        let mention_embedding = self.get_or_compute_embedding(mention).await?;
        
        // Find similar entities in the index
        let candidates = self.entity_index.similarity_search(
            &mention_embedding,
            self.config.max_candidates,
            self.config.similarity_threshold,
        ).await?;
        
        // Convert to LinkedEntity results
        let mut linked_entities = Vec::new();
        for candidate in candidates {
            linked_entities.push(LinkedEntity {
                canonical_name: candidate.entity.name.clone(),
                confidence: candidate.similarity_score,
                aliases: candidate.entity.aliases.clone(),
                entity_type: candidate.entity.entity_type.clone(),
            });
        }
        
        // Sort by confidence (highest first)
        linked_entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        Ok(linked_entities)
    }
    
    async fn normalize_entity(&self, entity: &str) -> Result<String> {
        let linked = self.link_entity(entity).await?;
        
        if let Some(best_match) = linked.first() {
            if best_match.confidence >= self.config.normalization_threshold {
                return Ok(best_match.canonical_name.clone());
            }
        }
        
        // Fallback to original entity name
        Ok(entity.to_string())
    }
    
    async fn find_aliases(&self, canonical: &str) -> Result<Vec<String>> {
        if let Some(entity) = self.entity_index.get_entity(canonical).await? {
            Ok(entity.aliases.clone())
        } else {
            Ok(Vec::new())
        }
    }
    
    async fn compute_similarity(&self, entity1: &str, entity2: &str) -> Result<f32> {
        let embedding1 = self.get_or_compute_embedding(entity1).await?;
        let embedding2 = self.get_or_compute_embedding(entity2).await?;
        
        Ok(cosine_similarity(&embedding1, &embedding2))
    }
}

impl MiniLMEntityLinker {
    async fn get_or_compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.embedding_cache.get(text).await {
            return Ok(cached);
        }
        
        let embedding = self.compute_embedding_uncached(text).await?;
        self.embedding_cache.put(text.to_string(), embedding.clone()).await;
        
        Ok(embedding)
    }
    
    async fn compute_embedding_uncached(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize and encode the text
        let tokens = self.model.tokenize(text)?;
        let input_ids = self.model.encode(&tokens)?;
        
        // Generate embedding using mean pooling
        let hidden_states = self.model.forward(&input_ids).await?;
        let embedding = mean_pooling(&hidden_states, &input_ids.attention_mask)?;
        
        // Normalize the embedding
        Ok(normalize_embedding(embedding))
    }
}
```

#### Entity Embedding Index
```rust
// src/enhanced_find_facts/entity_linking/embedding_cache.rs

use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct EntityEmbedding {
    pub name: String,
    pub embedding: Vec<f32>,
    pub aliases: Vec<String>,
    pub entity_type: Option<String>,
}

#[derive(Debug)]
pub struct SimilarityCandidate {
    pub entity: EntityEmbedding,
    pub similarity_score: f32,
}

pub struct EntityEmbeddingIndex {
    entities: Arc<RwLock<HashMap<String, EntityEmbedding>>>,
    embedding_matrix: Arc<RwLock<Option<ndarray::Array2<f32>>>>,
    entity_names: Arc<RwLock<Vec<String>>>,
    config: IndexConfig,
}

impl EntityEmbeddingIndex {
    pub fn new(config: IndexConfig) -> Result<Self> {
        Ok(Self {
            entities: Arc::new(RwLock::new(HashMap::new())),
            embedding_matrix: Arc::new(RwLock::new(None)),
            entity_names: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }
    
    pub async fn add_entity(&self, entity: EntityEmbedding) -> Result<()> {
        let mut entities = self.entities.write().await;
        let mut names = self.entity_names.write().await;
        
        if !entities.contains_key(&entity.name) {
            names.push(entity.name.clone());
        }
        
        entities.insert(entity.name.clone(), entity);
        
        // Mark index as needing rebuild
        *self.embedding_matrix.write().await = None;
        
        Ok(())
    }
    
    pub async fn build_index(&self) -> Result<()> {
        let entities = self.entities.read().await;
        let names = self.entity_names.read().await;
        
        if entities.is_empty() {
            return Ok(());
        }
        
        // Build embedding matrix for fast similarity search
        let embedding_dim = entities.values().next().unwrap().embedding.len();
        let mut matrix = ndarray::Array2::<f32>::zeros((names.len(), embedding_dim));
        
        for (i, name) in names.iter().enumerate() {
            let entity = entities.get(name).unwrap();
            for (j, &value) in entity.embedding.iter().enumerate() {
                matrix[[i, j]] = value;
            }
        }
        
        *self.embedding_matrix.write().await = Some(matrix);
        
        Ok(())
    }
    
    pub async fn similarity_search(
        &self,
        query_embedding: &[f32],
        max_results: usize,
        min_similarity: f32,
    ) -> Result<Vec<SimilarityCandidate>> {
        let embedding_matrix = self.embedding_matrix.read().await;
        let matrix = embedding_matrix.as_ref()
            .ok_or_else(|| EntityLinkingError::IndexNotBuilt)?;
        
        let entities = self.entities.read().await;
        let names = self.entity_names.read().await;
        
        // Compute similarities using vectorized operations
        let query_array = ndarray::Array1::from_vec(query_embedding.to_vec());
        let similarities = matrix.dot(&query_array);
        
        // Find top candidates
        let mut candidates: Vec<(usize, f32)> = similarities
            .indexed_iter()
            .map(|(idx, &sim)| (idx, sim))
            .filter(|(_, sim)| *sim >= min_similarity)
            .collect();
        
        // Sort by similarity (highest first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(max_results);
        
        // Convert to SimilarityCandidate
        let mut results = Vec::new();
        for (idx, similarity) in candidates {
            let entity_name = &names[idx];
            let entity = entities.get(entity_name).unwrap().clone();
            
            results.push(SimilarityCandidate {
                entity,
                similarity_score: similarity,
            });
        }
        
        Ok(results)
    }
    
    pub async fn get_entity(&self, name: &str) -> Result<Option<EntityEmbedding>> {
        let entities = self.entities.read().await;
        Ok(entities.get(name).cloned())
    }
}

pub struct EmbeddingCache {
    cache: Arc<RwLock<lru::LruCache<String, Vec<f32>>>>,
}

impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(lru::LruCache::new(capacity))),
        }
    }
    
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().await;
        cache.get(key).cloned()
    }
    
    pub async fn put(&self, key: String, value: Vec<f32>) {
        let mut cache = self.cache.write().await;
        cache.put(key, value);
    }
    
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}
```

#### Integration with Find Facts Handler
```rust
// src/enhanced_find_facts/tier1_integration.rs

use crate::mcp::llm_friendly_server::handlers::query::handle_find_facts;
use crate::enhanced_find_facts::entity_linking::{EntityLinker, MiniLMEntityLinker};

pub struct Tier1EnhancedHandler {
    core_engine: Arc<RwLock<KnowledgeEngine>>,
    entity_linker: Option<Arc<dyn EntityLinker>>,
    config: Tier1Config,
}

impl Tier1EnhancedHandler {
    pub async fn new(
        core_engine: Arc<RwLock<KnowledgeEngine>>,
        config: Tier1Config,
    ) -> Result<Self> {
        let entity_linker = if config.enable_entity_linking {
            let linker = MiniLMEntityLinker::new(config.entity_linking_config.clone()).await?;
            
            // Load entity knowledge from existing knowledge base
            linker.load_entity_knowledge(&*core_engine.read().await).await?;
            
            Some(Arc::new(linker) as Arc<dyn EntityLinker>)
        } else {
            None
        };
        
        Ok(Self {
            core_engine,
            entity_linker,
            config,
        })
    }
    
    pub async fn find_facts_enhanced(
        &self,
        query: TripleQuery,
        mode: FindFactsMode,
    ) -> Result<EnhancedFactsResult> {
        match mode {
            FindFactsMode::Exact => {
                // Use existing implementation for exact matching
                self.find_facts_exact(query).await
            },
            FindFactsMode::EntityLinked => {
                self.find_facts_with_entity_linking(query).await
            },
            _ => {
                // Fallback to exact for unsupported modes in Tier 1
                self.find_facts_exact(query).await
            }
        }
    }
    
    async fn find_facts_with_entity_linking(
        &self,
        query: TripleQuery,
    ) -> Result<EnhancedFactsResult> {
        let start_time = std::time::Instant::now();
        
        // Generate enhanced queries through entity linking
        let enhanced_queries = self.generate_entity_linked_queries(query.clone()).await?;
        
        // Execute all queries and merge results
        let mut all_results = Vec::new();
        let mut enhancement_metadata = EnhancementMetadata::default();
        
        for enhanced_query in enhanced_queries {
            let engine = self.core_engine.read().await;
            match engine.query_triples(enhanced_query.query.clone()) {
                Ok(result) => {
                    all_results.extend(result.triples);
                    if enhanced_query.is_enhanced {
                        enhancement_metadata.entity_linking_applied = true;
                        enhancement_metadata.entities_resolved.push(enhanced_query.resolution_info);
                    }
                },
                Err(_) => {
                    // Continue with other queries on individual failures
                    continue;
                }
            }
        }
        
        // Remove duplicates and limit results
        all_results.dedup_by(|a, b| {
            a.subject == b.subject && a.predicate == b.predicate && a.object == b.object
        });
        all_results.truncate(query.limit);
        
        let execution_time = start_time.elapsed();
        enhancement_metadata.execution_time_ms = execution_time.as_millis() as f64;
        
        Ok(EnhancedFactsResult {
            facts: all_results,
            count: all_results.len(),
            enhancement_metadata: Some(enhancement_metadata),
            semantic_scores: None, // Not used in Tier 1
        })
    }
    
    async fn generate_entity_linked_queries(
        &self,
        query: TripleQuery,
    ) -> Result<Vec<EnhancedQuery>> {
        let mut enhanced_queries = vec![EnhancedQuery {
            query: query.clone(),
            is_enhanced: false,
            resolution_info: String::new(),
        }];
        
        if let Some(ref entity_linker) = self.entity_linker {
            // Enhance subject if present
            if let Some(subject) = &query.subject {
                match entity_linker.link_entity(subject).await {
                    Ok(linked_entities) => {
                        for entity in linked_entities {
                            if entity.confidence >= self.config.min_confidence 
                                && entity.canonical_name != *subject {
                                enhanced_queries.push(EnhancedQuery {
                                    query: TripleQuery {
                                        subject: Some(entity.canonical_name.clone()),
                                        ..query.clone()
                                    },
                                    is_enhanced: true,
                                    resolution_info: format!("{} -> {}", subject, entity.canonical_name),
                                });
                            }
                        }
                    },
                    Err(e) => {
                        log::warn!("Entity linking failed for subject '{}': {}", subject, e);
                        // Continue without enhancement
                    }
                }
            }
            
            // Enhance object if present
            if let Some(object) = &query.object {
                match entity_linker.link_entity(object).await {
                    Ok(linked_entities) => {
                        for entity in linked_entities {
                            if entity.confidence >= self.config.min_confidence 
                                && entity.canonical_name != *object {
                                enhanced_queries.push(EnhancedQuery {
                                    query: TripleQuery {
                                        object: Some(entity.canonical_name.clone()),
                                        ..query.clone()
                                    },
                                    is_enhanced: true,
                                    resolution_info: format!("{} -> {}", object, entity.canonical_name),
                                });
                            }
                        }
                    },
                    Err(e) => {
                        log::warn!("Entity linking failed for object '{}': {}", object, e);
                        // Continue without enhancement
                    }
                }
            }
        }
        
        Ok(enhanced_queries)
    }
    
    async fn find_facts_exact(&self, query: TripleQuery) -> Result<EnhancedFactsResult> {
        let engine = self.core_engine.read().await;
        let result = engine.query_triples(query)?;
        
        Ok(EnhancedFactsResult {
            facts: result.triples,
            count: result.triples.len(),
            enhancement_metadata: None,
            semantic_scores: None,
        })
    }
}

#[derive(Debug, Clone)]
struct EnhancedQuery {
    query: TripleQuery,
    is_enhanced: bool,
    resolution_info: String,
}

#[derive(Debug, Default)]
pub struct EnhancementMetadata {
    pub entity_linking_applied: bool,
    pub entities_resolved: Vec<String>,
    pub execution_time_ms: f64,
    pub fallback_reason: Option<String>,
}

pub struct EnhancedFactsResult {
    pub facts: Vec<Triple>,
    pub count: usize,
    pub enhancement_metadata: Option<EnhancementMetadata>,
    pub semantic_scores: Option<Vec<f32>>,
}
```

### Configuration and Setup

#### Configuration Structure
```rust
// src/enhanced_find_facts/config.rs

#[derive(Debug, Clone)]
pub struct Tier1Config {
    pub enable_entity_linking: bool,
    pub min_confidence: f32,
    pub entity_linking_config: EntityLinkingConfig,
}

impl Default for Tier1Config {
    fn default() -> Self {
        Self {
            enable_entity_linking: true,
            min_confidence: 0.7,
            entity_linking_config: EntityLinkingConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EntityLinkingConfig {
    pub cache_size: usize,
    pub max_candidates: usize,
    pub similarity_threshold: f32,
    pub normalization_threshold: f32,
    pub device: Device,
    pub index_config: IndexConfig,
}

impl Default for EntityLinkingConfig {
    fn default() -> Self {
        Self {
            cache_size: 100_000,
            max_candidates: 10,
            similarity_threshold: 0.7,
            normalization_threshold: 0.8,
            device: Device::Cpu,
            index_config: IndexConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub rebuild_threshold: usize,
    pub max_entities: usize,
    pub embedding_dim: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            rebuild_threshold: 1000,
            max_entities: 1_000_000,
            embedding_dim: 384, // MiniLM-L6-v2 embedding dimension
        }
    }
}
```

### TDD Implementation Schedule

#### Week 1: Mock-First Foundation
**Days 1-2: Interface Design & Mock Setup**
```rust
// tests/enhanced_find_facts/unit/entity_linking/mock_tests.rs

#[tokio::test]
async fn test_entity_linker_interface_design() {
    let mut mock_linker = MockEntityLinkerModel::new();
    
    // Define the contract through mock expectations
    mock_linker.expect_link_entity()
        .with(eq("Einstein"))
        .times(1)
        .returning(|_| Ok(vec![
            LinkedEntity {
                canonical_name: "Albert Einstein".to_string(),
                confidence: 0.95,
                aliases: vec!["Einstein".to_string()],
                entity_type: Some("Person".to_string()),
            }
        ]));
    
    // Test the interface
    let result = mock_linker.link_entity("Einstein").await.unwrap();
    assert_eq!(result[0].canonical_name, "Albert Einstein");
    assert!(result[0].confidence > 0.9);
}
```

**Days 3-4: Core Component Mocks**
- Mock MiniLMEntityLinker implementation
- Mock EmbeddingCache behavior
- Mock EntityEmbeddingIndex functionality
- Mock integration with KnowledgeEngine

**Days 5-7: Integration Mock Testing**
- Mock-based integration tests
- Performance baseline establishment
- Error handling mock scenarios

#### Week 2: Real Implementation
**Days 1-3: Core Component Implementation**
- Implement MiniLMEntityLinker with real MiniLM model
- Implement EmbeddingCache with LRU eviction
- Implement EntityEmbeddingIndex with ndarray

**Days 4-5: Model Integration**
- MiniLM model loading and inference
- Embedding computation and caching
- Entity index building from knowledge base

**Days 6-7: Integration Testing**
- Replace mocks with real implementations progressively
- Validate contract compliance
- Performance benchmarking

#### Week 3: Production Readiness
**Days 1-2: Optimization**
- Performance tuning and caching optimization
- Memory usage optimization
- Concurrent access handling

**Days 3-4: Error Handling & Resilience**
- Graceful degradation implementation
- Resource constraint handling
- Recovery mechanisms

**Days 5-7: End-to-End Validation**
- Full integration with find_facts handler
- Acceptance testing
- Performance validation against SLA

### Performance Expectations

#### Latency Breakdown
- **Model Loading**: 2-5 seconds (one-time, cached)
- **Entity Embedding**: 2-8ms per entity (cached aggressively)
- **Similarity Search**: 1-3ms per query
- **Total Enhancement**: 5-15ms additional latency

#### Memory Usage
- **Model**: ~90MB (MiniLM-L6-v2)
- **Entity Index**: ~10MB (10K entities × 384 dims × 4 bytes)
- **Embedding Cache**: ~15MB (100K cached embeddings)
- **Total**: ~115MB additional memory

#### Accuracy Improvements
- **Entity Variation Recall**: +25-40% for queries with entity name variations
- **Alias Resolution**: 90%+ success rate for common aliases
- **False Positive Rate**: <5% due to conservative confidence thresholds

### Success Metrics

#### Functional Metrics
- **Entity Resolution Accuracy**: >90% for common entity variations
- **Query Enhancement Rate**: 15-25% of queries benefit from entity linking
- **Fallback Success**: 100% graceful degradation on model failures

#### Performance Metrics
- **P95 Latency**: <15ms additional latency
- **Memory Overhead**: <120MB total
- **Cache Hit Rate**: >85% for embedding cache

#### Quality Metrics
- **Test Coverage**: >95% unit, >90% integration
- **Bug Escape Rate**: <0.5% critical issues
- **Performance Regression**: Zero for exact mode

This Tier 1 implementation provides a solid foundation for entity linking while maintaining the high performance and reliability expectations of the `find_facts` tool.