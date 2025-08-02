# Tier 2 Implementation: Semantic Query Expansion
## SmolLM-Based Predicate Expansion and Fuzzy Matching

### Tier 2 Overview

#### Objective
Implement semantic query expansion using SmolLM-360M-Instruct to understand predicate relationships, expand query contexts, and provide fuzzy matching capabilities, improving semantic query success rate from 60% to 90% with controlled performance impact (+20-80ms, +800MB memory).

#### Core Capabilities
1. **Predicate Expansion**: "born_in" → ["birth_place", "birthplace", "place_of_birth"]
2. **Semantic Similarity Ranking**: Vector-based relevance scoring
3. **Query Context Understanding**: Natural language query interpretation
4. **Fuzzy Matching**: Approximate results with confidence scores
5. **REMATCH Integration**: State-of-the-art semantic similarity scoring

### Technical Architecture

#### Core Components
```rust
// src/enhanced_find_facts/semantic_expansion/mod.rs
pub mod query_expander;
pub mod predicate_expander;
pub mod similarity_ranker;
pub mod vector_index;
pub mod rematch_scorer;

pub use query_expander::{QueryExpander, SmolLMQueryExpander, SemanticContext};
pub use predicate_expander::{PredicateExpander, ExpandedPredicate, ExpansionType};
pub use similarity_ranker::{SimilarityRanker, ScoredTriple, RankingStrategy};
pub use vector_index::{VectorIndex, SemanticIndex};
pub use rematch_scorer::{RematchScorer, RematchScore};
```

#### Semantic Query Expander
```rust
// src/enhanced_find_facts/semantic_expansion/query_expander.rs

use crate::models::smollm::{smollm_360m_instruct, SmolLMVariant};
use async_trait::async_trait;

#[async_trait]
pub trait QueryExpander: Send + Sync {
    async fn expand_predicate(&self, predicate: &str) -> Result<Vec<ExpandedPredicate>>;
    async fn expand_semantic_context(&self, query: &TripleQuery) -> Result<SemanticContext>;
    async fn interpret_natural_query(&self, natural_query: &str) -> Result<StructuredQuery>;
    async fn generate_query_variations(&self, query: &TripleQuery) -> Result<Vec<QueryVariation>>;
}

pub struct SmolLMQueryExpander {
    model: Arc<dyn Model>,
    predicate_ontology: Arc<PredicateOntology>,
    expansion_cache: Arc<LRUCache<String, Vec<ExpandedPredicate>>>,
    context_cache: Arc<LRUCache<String, SemanticContext>>,
    config: SemanticExpansionConfig,
}

impl SmolLMQueryExpander {
    pub async fn new(config: SemanticExpansionConfig) -> Result<Self> {
        let model = smollm_360m_instruct()
            .with_config(ModelConfig {
                max_sequence_length: 512,
                temperature: 0.3, // Lower temperature for more consistent expansions
                top_p: 0.9,
                ..Default::default()
            })
            .build()?;
        
        let predicate_ontology = Arc::new(PredicateOntology::load_from_knowledge_base().await?);
        let expansion_cache = Arc::new(LRUCache::new(config.expansion_cache_size));
        let context_cache = Arc::new(LRUCache::new(config.context_cache_size));
        
        Ok(Self {
            model: Arc::new(model),
            predicate_ontology,
            expansion_cache,
            context_cache,
            config,
        })
    }
}

#[async_trait]
impl QueryExpander for SmolLMQueryExpander {
    async fn expand_predicate(&self, predicate: &str) -> Result<Vec<ExpandedPredicate>> {
        // Check cache first
        if let Some(cached) = self.expansion_cache.get(predicate).await {
            return Ok(cached);
        }
        
        // Generate expansion prompt for SmolLM
        let prompt = self.create_predicate_expansion_prompt(predicate);
        
        let response = self.model.generate_text(&prompt, Some(150)).await?;
        let parsed_expansions = self.parse_predicate_expansions(&response, predicate)?;
        
        // Enhance with ontology-based expansions
        let ontology_expansions = self.predicate_ontology
            .get_similar_predicates(predicate, self.config.ontology_similarity_threshold)
            .await?;
        
        // Combine and deduplicate
        let mut all_expansions = parsed_expansions;
        for ontology_expansion in ontology_expansions {
            if !all_expansions.iter().any(|e| e.relation == ontology_expansion.relation) {
                all_expansions.push(ontology_expansion);
            }
        }
        
        // Sort by confidence and limit
        all_expansions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        all_expansions.truncate(self.config.max_expansions_per_predicate);
        
        // Cache result
        self.expansion_cache.put(predicate.to_string(), all_expansions.clone()).await;
        
        Ok(all_expansions)
    }
    
    async fn expand_semantic_context(&self, query: &TripleQuery) -> Result<SemanticContext> {
        let context_key = self.create_context_cache_key(query);
        
        if let Some(cached) = self.context_cache.get(&context_key).await {
            return Ok(cached);
        }
        
        // Generate context understanding prompt
        let prompt = self.create_context_analysis_prompt(query);
        let response = self.model.generate_text(&prompt, Some(200)).await?;
        
        let semantic_context = self.parse_semantic_context(&response, query)?;
        
        // Cache result
        self.context_cache.put(context_key, semantic_context.clone()).await;
        
        Ok(semantic_context)
    }
    
    async fn interpret_natural_query(&self, natural_query: &str) -> Result<StructuredQuery> {
        let prompt = format!(
            "Convert this natural language query into structured subject-predicate-object format:\n\
            Query: \"{}\"\n\
            Extract:\n\
            - Subject: [entity or 'unknown']\n\
            - Predicate: [relationship or 'unknown']\n\
            - Object: [target entity/value or 'unknown']\n\
            - Intent: [what the user wants to find]\n\
            Format as JSON:",
            natural_query
        );
        
        let response = self.model.generate_text(&prompt, Some(100)).await?;
        let structured_query = self.parse_structured_query(&response)?;
        
        Ok(structured_query)
    }
    
    async fn generate_query_variations(&self, query: &TripleQuery) -> Result<Vec<QueryVariation>> {
        let prompt = self.create_variation_prompt(query);
        let response = self.model.generate_text(&prompt, Some(300)).await?;
        
        let variations = self.parse_query_variations(&response)?;
        Ok(variations)
    }
}

impl SmolLMQueryExpander {
    fn create_predicate_expansion_prompt(&self, predicate: &str) -> String {
        format!(
            "Given the relationship '{}', list semantically similar relationships that could be used \
            interchangeably in a knowledge graph. Focus on synonyms, alternate phrasings, and \
            conceptually equivalent relationships.\n\
            \n\
            Examples:\n\
            - 'born_in' → birth_place, birthplace, place_of_birth, native_of\n\
            - 'works_at' → employed_by, job_at, position_at, workplace\n\
            - 'created_by' → authored_by, made_by, developed_by, invented_by\n\
            \n\
            Relationship: '{}'\n\
            Similar relationships (comma-separated):",
            predicate, predicate
        )
    }
    
    fn parse_predicate_expansions(
        &self, 
        response: &str, 
        original_predicate: &str
    ) -> Result<Vec<ExpandedPredicate>> {
        let mut expansions = Vec::new();
        
        // Parse comma-separated list from model response
        let cleaned_response = response
            .lines()
            .last()
            .unwrap_or(response)
            .trim();
        
        for expansion in cleaned_response.split(',') {
            let expansion = expansion.trim();
            if !expansion.is_empty() && expansion != original_predicate {
                expansions.push(ExpandedPredicate {
                    relation: expansion.to_string(),
                    confidence: self.calculate_expansion_confidence(original_predicate, expansion),
                    expansion_type: ExpansionType::Semantic,
                    source: ExpansionSource::LanguageModel,
                });
            }
        }
        
        Ok(expansions)
    }
    
    fn create_context_analysis_prompt(&self, query: &TripleQuery) -> String {
        format!(
            "Analyze this knowledge graph query and provide semantic context:\n\
            \n\
            Query:\n\
            - Subject: {}\n\
            - Predicate: {}\n\
            - Object: {}\n\
            \n\
            Analyze:\n\
            1. Domain: What knowledge domain is this about? (science, history, geography, etc.)\n\
            2. Intent: What is the user trying to discover?\n\
            3. Entity Types: What types of entities are involved?\n\
            4. Relationship Nature: What kind of relationship is being queried?\n\
            5. Completeness: Is this a complete or partial query?\n\
            \n\
            Provide analysis:",
            query.subject.as_ref().map(|s| s.as_str()).unwrap_or("unknown"),
            query.predicate.as_ref().map(|p| p.as_str()).unwrap_or("unknown"),
            query.object.as_ref().map(|o| o.as_str()).unwrap_or("unknown")
        )
    }
    
    fn parse_semantic_context(&self, response: &str, query: &TripleQuery) -> Result<SemanticContext> {
        // Simple parsing - in production, might use more sophisticated NLP
        let domain = self.extract_field(response, "Domain:")?;
        let intent = self.extract_field(response, "Intent:")?;
        let entity_types = self.extract_field(response, "Entity Types:")?;
        let relationship_nature = self.extract_field(response, "Relationship Nature:")?;
        
        Ok(SemanticContext {
            domain: domain.unwrap_or_else(|| "general".to_string()),
            intent: intent.unwrap_or_else(|| "find_related".to_string()),
            entity_types: entity_types.map(|et| et.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default(),
            relationship_nature: relationship_nature.unwrap_or_else(|| "generic".to_string()),
            confidence: 0.8,
            query_completeness: self.assess_query_completeness(query),
        })
    }
    
    fn calculate_expansion_confidence(&self, original: &str, expansion: &str) -> f32 {
        // Simple heuristic - could be enhanced with more sophisticated similarity measures
        let similarity = string_similarity(original, expansion);
        
        // Base confidence from string similarity
        let mut confidence = 0.5 + (similarity * 0.3);
        
        // Boost confidence for common expansion patterns
        if self.is_common_expansion_pattern(original, expansion) {
            confidence += 0.2;
        }
        
        // Cap at reasonable maximum
        confidence.min(0.95)
    }
    
    fn is_common_expansion_pattern(&self, original: &str, expansion: &str) -> bool {
        // Common patterns like underscore variants, synonym patterns, etc.
        original.replace("_", "") == expansion.replace("_", "") ||
        original.contains("place") && expansion.contains("location") ||
        original.contains("born") && expansion.contains("birth")
    }
}

#[derive(Debug, Clone)]
pub struct ExpandedPredicate {
    pub relation: String,
    pub confidence: f32,
    pub expansion_type: ExpansionType,
    pub source: ExpansionSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionType {
    Exact,
    Semantic,
    Syntactic,
    Ontological,
}

#[derive(Debug, Clone)]
pub enum ExpansionSource {
    LanguageModel,
    Ontology,
    StatisticalAnalysis,
    UserFeedback,
}

#[derive(Debug, Clone)]
pub struct SemanticContext {
    pub domain: String,
    pub intent: String,
    pub entity_types: Vec<String>,
    pub relationship_nature: String,
    pub confidence: f32,
    pub query_completeness: QueryCompleteness,
}

#[derive(Debug, Clone)]
pub enum QueryCompleteness {
    Complete,      // All three components specified
    Partial,       // Two components specified
    Exploratory,   // One component specified
}

#[derive(Debug, Clone)]
pub struct QueryVariation {
    pub query: TripleQuery,
    pub variation_type: VariationType,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum VariationType {
    Paraphrase,
    Generalization,
    Specialization,
    RelatedConcept,
}
```

#### Semantic Expansion Layer Integration
```rust
// src/enhanced_find_facts/semantic_expansion/semantic_layer.rs

pub struct SemanticExpansionLayer {
    query_expander: Arc<dyn QueryExpander>,
    similarity_ranker: Arc<dyn SimilarityRanker>,
    vector_index: Arc<VectorIndex>,
    rematch_scorer: Arc<RematchScorer>,
    config: SemanticExpansionConfig,
}

impl SemanticExpansionLayer {
    pub async fn new(config: SemanticExpansionConfig) -> Result<Self> {
        let query_expander = Arc::new(SmolLMQueryExpander::new(config.clone()).await?);
        let similarity_ranker = Arc::new(VectorSimilarityRanker::new(config.clone()).await?);
        let vector_index = Arc::new(VectorIndex::new(config.vector_index_config.clone()).await?);
        let rematch_scorer = Arc::new(RematchScorer::new(config.rematch_config.clone())?);
        
        Ok(Self {
            query_expander,
            similarity_ranker,
            vector_index,
            rematch_scorer,
            config,
        })
    }
    
    pub async fn enhance_semantically(
        &self,
        base_queries: Vec<TripleQuery>,
    ) -> Result<Vec<ScoredTripleQuery>> {
        let mut all_semantic_queries = Vec::new();
        
        for query in base_queries {
            // Generate semantic context
            let semantic_context = self.query_expander.expand_semantic_context(&query).await?;
            
            // Expand predicates
            let predicate_expansions = if let Some(predicate) = &query.predicate {
                self.query_expander.expand_predicate(predicate).await?
            } else {
                Vec::new()
            };
            
            // Create expanded queries
            let mut query_expansions = self.create_expanded_queries(&query, &predicate_expansions)?;
            
            // Add original query with highest score
            query_expansions.insert(0, ScoredTripleQuery {
                query: query.clone(),
                semantic_score: 1.0,
                expansion_type: ExpansionType::Exact,
                semantic_context: semantic_context.clone(),
            });
            
            all_semantic_queries.extend(query_expansions);
        }
        
        // Sort by semantic score (highest first)
        all_semantic_queries.sort_by(|a, b| b.semantic_score.partial_cmp(&a.semantic_score).unwrap());
        
        Ok(all_semantic_queries)
    }
    
    pub async fn rank_results_semantically(
        &self,
        results: Vec<Triple>,
        original_query: &TripleQuery,
    ) -> Result<Vec<ScoredTriple>> {
        let mut scored_results = Vec::new();
        
        for triple in results {
            // Calculate semantic similarity using REMATCH
            let rematch_score = self.rematch_scorer.compute_similarity(original_query, &triple).await?;
            
            // Calculate vector similarity if available
            let vector_score = self.vector_index.compute_triple_similarity(original_query, &triple).await
                .unwrap_or(0.0);
            
            // Combine scores
            let combined_score = (rematch_score.structural_similarity * 0.4) +
                               (rematch_score.semantic_similarity * 0.4) +
                               (vector_score * 0.2);
            
            scored_results.push(ScoredTriple {
                triple,
                semantic_score: combined_score,
                structural_score: rematch_score.structural_similarity,
                vector_score,
                confidence: rematch_score.confidence,
            });
        }
        
        // Sort by combined score (highest first)
        scored_results.sort_by(|a, b| b.semantic_score.partial_cmp(&a.semantic_score).unwrap());
        
        Ok(scored_results)
    }
    
    fn create_expanded_queries(
        &self,
        base_query: &TripleQuery,
        predicate_expansions: &[ExpandedPredicate],
    ) -> Result<Vec<ScoredTripleQuery>> {
        let mut expanded_queries = Vec::new();
        
        for expansion in predicate_expansions {
            if expansion.confidence >= self.config.min_expansion_confidence {
                expanded_queries.push(ScoredTripleQuery {
                    query: TripleQuery {
                        predicate: Some(expansion.relation.clone()),
                        ..base_query.clone()
                    },
                    semantic_score: expansion.confidence,
                    expansion_type: expansion.expansion_type.clone(),
                    semantic_context: SemanticContext::default(), // Will be filled later
                });
            }
        }
        
        Ok(expanded_queries)
    }
}

#[derive(Debug, Clone)]
pub struct ScoredTripleQuery {
    pub query: TripleQuery,
    pub semantic_score: f32,
    pub expansion_type: ExpansionType,
    pub semantic_context: SemanticContext,
}

#[derive(Debug, Clone)]
pub struct ScoredTriple {
    pub triple: Triple,
    pub semantic_score: f32,
    pub structural_score: f32,
    pub vector_score: f32,
    pub confidence: f32,
}
```

#### REMATCH Scorer Integration
```rust
// src/enhanced_find_facts/semantic_expansion/rematch_scorer.rs

use crate::core::triple::Triple;
use crate::core::knowledge_types::TripleQuery;

pub struct RematchScorer {
    amr_parser: Arc<AMRParser>,
    structural_analyzer: Arc<StructuralAnalyzer>,
    semantic_analyzer: Arc<SemanticAnalyzer>,
    config: RematchConfig,
}

impl RematchScorer {
    pub fn new(config: RematchConfig) -> Result<Self> {
        Ok(Self {
            amr_parser: Arc::new(AMRParser::new()?),
            structural_analyzer: Arc::new(StructuralAnalyzer::new()),
            semantic_analyzer: Arc::new(SemanticAnalyzer::new(config.clone())?),
            config,
        })
    }
    
    pub async fn compute_similarity(
        &self,
        query: &TripleQuery,
        triple: &Triple,
    ) -> Result<RematchScore> {
        // Convert query and triple to AMR representations
        let query_amr = self.amr_parser.parse_query(query).await?;
        let triple_amr = self.amr_parser.parse_triple(triple).await?;
        
        // Compute structural similarity (RARE metric)
        let structural_similarity = self.structural_analyzer
            .compute_rare_similarity(&query_amr, &triple_amr)?;
        
        // Compute semantic similarity (REMATCH metric)
        let semantic_similarity = self.semantic_analyzer
            .compute_rematch_similarity(&query_amr, &triple_amr).await?;
        
        // Combine scores with configuration weights
        let combined_score = (structural_similarity * self.config.structural_weight) +
                           (semantic_similarity * self.config.semantic_weight);
        
        Ok(RematchScore {
            structural_similarity,
            semantic_similarity,
            combined_score,
            confidence: self.calculate_confidence(structural_similarity, semantic_similarity),
        })
    }
    
    fn calculate_confidence(&self, structural: f32, semantic: f32) -> f32 {
        // Higher confidence when both metrics agree
        let agreement = 1.0 - (structural - semantic).abs();
        let avg_score = (structural + semantic) / 2.0;
        
        (agreement * 0.3) + (avg_score * 0.7)
    }
}

#[derive(Debug, Clone)]
pub struct RematchScore {
    pub structural_similarity: f32,
    pub semantic_similarity: f32,
    pub combined_score: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct RematchConfig {
    pub structural_weight: f32,
    pub semantic_weight: f32,
    pub confidence_threshold: f32,
}

impl Default for RematchConfig {
    fn default() -> Self {
        Self {
            structural_weight: 0.4,
            semantic_weight: 0.6,
            confidence_threshold: 0.7,
        }
    }
}
```

### Integration with Tier 1

#### Tier 2 Enhanced Handler
```rust
// src/enhanced_find_facts/tier2_integration.rs

pub struct Tier2EnhancedHandler {
    tier1_handler: Tier1EnhancedHandler,
    semantic_layer: Arc<SemanticExpansionLayer>,
    config: Tier2Config,
}

impl Tier2EnhancedHandler {
    pub async fn new(
        core_engine: Arc<RwLock<KnowledgeEngine>>,
        config: Tier2Config,
    ) -> Result<Self> {
        let tier1_handler = Tier1EnhancedHandler::new(
            core_engine.clone(),
            config.tier1_config.clone(),
        ).await?;
        
        let semantic_layer = Arc::new(SemanticExpansionLayer::new(
            config.semantic_expansion_config.clone()
        ).await?);
        
        Ok(Self {
            tier1_handler,
            semantic_layer,
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
                self.tier1_handler.find_facts_enhanced(query, FindFactsMode::Exact).await
            },
            FindFactsMode::EntityLinked => {
                self.tier1_handler.find_facts_enhanced(query, FindFactsMode::EntityLinked).await
            },
            FindFactsMode::SemanticExpanded => {
                self.find_facts_with_semantic_expansion(query).await
            },
            FindFactsMode::FuzzyRanked => {
                self.find_facts_with_fuzzy_ranking(query).await
            },
            _ => {
                // Fallback to Tier 1 for unsupported modes
                self.tier1_handler.find_facts_enhanced(query, FindFactsMode::EntityLinked).await
            }
        }
    }
    
    async fn find_facts_with_semantic_expansion(
        &self,
        query: TripleQuery,
    ) -> Result<EnhancedFactsResult> {
        let start_time = std::time::Instant::now();
        
        // First, apply Tier 1 entity linking
        let tier1_result = self.tier1_handler
            .find_facts_enhanced(query.clone(), FindFactsMode::EntityLinked)
            .await?;
        
        // If Tier 1 found sufficient results, return them
        if tier1_result.facts.len() >= query.limit || 
           tier1_result.facts.len() >= self.config.semantic_threshold {
            return Ok(tier1_result);
        }
        
        // Apply semantic expansion
        let entity_linked_queries = self.extract_entity_linked_queries(&tier1_result)?;
        let semantic_queries = self.semantic_layer
            .enhance_semantically(entity_linked_queries)
            .await?;
        
        // Execute semantic queries
        let mut all_results = tier1_result.facts;
        let mut enhancement_metadata = tier1_result.enhancement_metadata.unwrap_or_default();
        
        for semantic_query in semantic_queries.iter().take(self.config.max_semantic_queries) {
            if semantic_query.semantic_score < self.config.min_semantic_confidence {
                break;
            }
            
            let engine = self.tier1_handler.core_engine.read().await;
            match engine.query_triples(semantic_query.query.clone()) {
                Ok(result) => {
                    all_results.extend(result.triples);
                    enhancement_metadata.semantic_expansion_applied = true;
                    enhancement_metadata.predicates_expanded.push(format!(
                        "{} -> {} (confidence: {:.2})",
                        semantic_query.query.predicate.as_ref().unwrap_or(&"unknown".to_string()),
                        semantic_query.expansion_type.to_string(),
                        semantic_query.semantic_score
                    ));
                },
                Err(e) => {
                    log::warn!("Semantic query failed: {}", e);
                    continue;
                }
            }
        }
        
        // Remove duplicates and limit
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
            semantic_scores: None,
        })
    }
    
    async fn find_facts_with_fuzzy_ranking(
        &self,
        query: TripleQuery,
    ) -> Result<EnhancedFactsResult> {
        // First get semantically expanded results
        let semantic_result = self.find_facts_with_semantic_expansion(query.clone()).await?;
        
        // Apply semantic ranking
        let scored_results = self.semantic_layer
            .rank_results_semantically(semantic_result.facts, &query)
            .await?;
        
        // Convert back to regular triples but include semantic scores
        let ranked_facts: Vec<Triple> = scored_results.iter()
            .map(|scored| scored.triple.clone())
            .collect();
        
        let semantic_scores: Vec<f32> = scored_results.iter()
            .map(|scored| scored.semantic_score)
            .collect();
        
        let mut enhancement_metadata = semantic_result.enhancement_metadata.unwrap_or_default();
        enhancement_metadata.semantic_ranking_applied = true;
        enhancement_metadata.avg_semantic_score = semantic_scores.iter().sum::<f32>() / semantic_scores.len() as f32;
        
        Ok(EnhancedFactsResult {
            facts: ranked_facts,
            count: ranked_facts.len(),
            enhancement_metadata: Some(enhancement_metadata),
            semantic_scores: Some(semantic_scores),
        })
    }
}

#[derive(Debug, Default)]
pub struct Tier2EnhancementMetadata {
    // Inherit from Tier 1
    pub entity_linking_applied: bool,
    pub entities_resolved: Vec<String>,
    
    // Tier 2 specific
    pub semantic_expansion_applied: bool,
    pub predicates_expanded: Vec<String>,
    pub semantic_ranking_applied: bool,
    pub avg_semantic_score: f32,
    
    // Common
    pub execution_time_ms: f64,
    pub fallback_reason: Option<String>,
}
```

### Performance Optimization

#### Caching Strategy
```rust
// src/enhanced_find_facts/semantic_expansion/cache_manager.rs

pub struct SemanticCacheManager {
    predicate_expansion_cache: Arc<LRUCache<String, Vec<ExpandedPredicate>>>,
    semantic_context_cache: Arc<LRUCache<String, SemanticContext>>>,
    similarity_score_cache: Arc<LRUCache<SimilarityKey, f32>>,
    query_result_cache: Arc<LRUCache<QuerySignature, CachedSemanticResult>>,
}

impl SemanticCacheManager {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            predicate_expansion_cache: Arc::new(LRUCache::new(config.predicate_cache_size)),
            semantic_context_cache: Arc::new(LRUCache::new(config.context_cache_size)),
            similarity_score_cache: Arc::new(LRUCache::new(config.similarity_cache_size)),
            query_result_cache: Arc::new(LRUCache::new(config.result_cache_size)),
        }
    }
    
    pub async fn get_or_compute_predicate_expansion<F>(
        &self,
        predicate: &str,
        compute_fn: F,
    ) -> Result<Vec<ExpandedPredicate>>
    where
        F: Future<Output = Result<Vec<ExpandedPredicate>>>,
    {
        if let Some(cached) = self.predicate_expansion_cache.get(predicate).await {
            return Ok(cached);
        }
        
        let result = compute_fn.await?;
        self.predicate_expansion_cache.put(predicate.to_string(), result.clone()).await;
        
        Ok(result)
    }
}
```

### TDD Implementation Schedule

#### Week 4: Mock-First Semantic Foundation
**Days 1-2: SmolLM Integration Mocks**
- Mock SmolLM-360M-Instruct model behavior
- Mock predicate expansion functionality
- Mock semantic context analysis

**Days 3-4: Semantic Layer Mocks**
- Mock SemanticExpansionLayer components
- Mock REMATCH scorer integration
- Mock vector similarity computation

**Days 5-7: Integration Mock Testing**
- Mock Tier 1 + Tier 2 integration
- Mock performance characteristics
- Mock error handling scenarios

#### Week 5: Real Implementation
**Days 1-3: Core Semantic Components**
- Implement SmolLMQueryExpander with real model
- Implement predicate expansion parsing
- Implement semantic context analysis

**Days 4-5: REMATCH Integration**
- Implement simplified REMATCH scoring
- Implement structural similarity metrics
- Implement semantic similarity computation

**Days 6-7: Integration Testing**
- Replace semantic mocks progressively
- Validate contract compliance
- Performance benchmarking

#### Week 6: Advanced Features & Optimization
**Days 1-3: Advanced Semantic Features**
- Implement query variation generation
- Implement natural language query interpretation
- Implement semantic ranking system

**Days 4-5: Performance Optimization**
- Implement comprehensive caching
- Optimize model inference patterns
- Implement batch processing

**Days 6-7: End-to-End Validation**
- Full Tier 2 integration testing
- Acceptance testing
- Performance validation against SLA

### Performance Expectations

#### Latency Breakdown
- **Model Loading**: 5-10 seconds (one-time, cached)
- **Predicate Expansion**: 15-40ms per expansion (cached)
- **Semantic Context**: 10-25ms per query (cached)
- **REMATCH Scoring**: 5-15ms per result
- **Total Enhancement**: 20-80ms additional latency

#### Memory Usage
- **SmolLM Model**: ~750MB (360M parameters)
- **Semantic Caches**: ~50MB (expansion + context caches)
- **Vector Index**: ~20MB (vector similarity data)
- **Total**: ~820MB additional memory

#### Accuracy Improvements
- **Semantic Query Success**: 60% → 90% for fuzzy queries
- **Predicate Expansion Coverage**: 85%+ for common relationships
- **False Positive Rate**: <10% with confidence thresholds

### Success Metrics

#### Functional Metrics
- **Semantic Query Success Rate**: >90% for natural language queries
- **Predicate Expansion Accuracy**: >85% for common relationships
- **Ranking Quality**: >80% precision at top-5 results

#### Performance Metrics
- **P95 Latency**: <80ms additional latency for semantic expansion
- **Memory Overhead**: <850MB total
- **Cache Hit Rate**: >75% for predicate expansions

#### Quality Metrics
- **Test Coverage**: >95% unit, >90% integration
- **Enhancement Success Rate**: >70% of queries benefit from semantic expansion
- **User Satisfaction**: Measurable improvement in query success satisfaction

This Tier 2 implementation significantly enhances the semantic capabilities of `find_facts` while building upon the solid foundation provided by Tier 1 entity linking.