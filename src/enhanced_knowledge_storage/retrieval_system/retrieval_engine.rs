//! Retrieval Engine
//! 
//! Main retrieval engine that coordinates all retrieval components to provide
//! intelligent, context-aware search with multi-hop reasoning capabilities.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, error, debug, instrument};
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    hierarchical_storage::{
        types::*,
        HierarchicalStorageEngine,
        IndexQuery,
        SearchResult,
    },
    retrieval_system::{
        types::*,
        QueryProcessor,
        MultiHopReasoner,
        ContextAggregator,
        multi_hop_reasoner::GraphContext,
    },
    logging::LogContext,
};

/// Main retrieval engine
pub struct RetrievalEngine {
    model_manager: Arc<ModelResourceManager>,
    storage_engine: Arc<HierarchicalStorageEngine>,
    query_processor: Arc<QueryProcessor>,
    multi_hop_reasoner: Arc<MultiHopReasoner>,
    context_aggregator: Arc<ContextAggregator>,
    search_cache: Arc<RwLock<HashMap<u64, SearchCacheEntry>>>,
    config: RetrievalConfig,
}

impl RetrievalEngine {
    /// Create new retrieval engine
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        storage_engine: Arc<HierarchicalStorageEngine>,
        config: RetrievalConfig,
    ) -> Self {
        let query_processor = Arc::new(QueryProcessor::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        let multi_hop_reasoner = Arc::new(MultiHopReasoner::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        let context_aggregator = Arc::new(ContextAggregator::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        Self {
            model_manager,
            storage_engine,
            query_processor,
            multi_hop_reasoner,
            context_aggregator,
            search_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Execute a retrieval query
    #[instrument(
        skip(self, query),
        fields(
            query = %query.natural_language_query,
            max_results = query.max_results,
            retrieval_mode = ?query.retrieval_mode,
            multi_hop = query.enable_multi_hop
        )
    )]
    pub async fn retrieve(
        &self,
        query: RetrievalQuery,
    ) -> RetrievalResult2<RetrievalResult> {
        let start_time = Instant::now();
        let query_id = self.generate_query_id(&query);
        
        let log_context = LogContext::new("retrieve", "retrieval_engine")
            .with_request_id(query_id.clone());
        
        info!(
            context = ?log_context,
            query_id = %query_id,
            query = %query.natural_language_query,
            max_results = query.max_results,
            retrieval_mode = ?query.retrieval_mode,
            enable_multi_hop = query.enable_multi_hop,
            context_window_size = query.context_window_size,
            "Starting retrieval query processing"
        );
        
        // Check cache if enabled
        if self.config.cache_search_results {
            debug!("Checking cache for results");
            if let Some(cached) = self.check_cache(&query).await {
                info!(
                    context = ?log_context,
                    cached_results_count = cached.len(),
                    "Returning cached results"
                );
                return Ok(self.create_cached_result(query_id, cached, start_time));
            }
            debug!("No cached results found");
        }
        
        // Step 1: Process and understand query
        info!("Step 1/8: Processing and understanding query");
        let query_start = Instant::now();
        let processed_query = self.query_processor.process_query(&query).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to process query"
                );
                e
            })?;
        debug!(
            query_processing_time_ms = query_start.elapsed().as_millis(),
            intent = ?processed_query.understanding.intent,
            entities_extracted = processed_query.understanding.extracted_entities.len(),
            concepts_extracted = processed_query.understanding.extracted_concepts.len(),
            complexity = ?processed_query.understanding.complexity_level,
            "Query processing completed"
        );
        
        // Step 2: Execute initial search
        info!("Step 2/8: Executing initial search");
        let search_start = Instant::now();
        let initial_results = self.execute_search(&processed_query).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to execute search"
                );
                e
            })?;
        debug!(
            initial_search_time_ms = search_start.elapsed().as_millis(),
            initial_results_count = initial_results.len(),
            avg_relevance_score = if initial_results.is_empty() { 0.0 } else { 
                initial_results.iter().map(|r| r.relevance_score).sum::<f32>() / initial_results.len() as f32
            },
            "Initial search completed"
        );
        
        // Step 3: Perform multi-hop reasoning if enabled
        let reasoning_chain = if query.enable_multi_hop && initial_results.len() < query.max_results {
            info!("Step 3/8: Performing multi-hop reasoning");
            let reasoning_start = Instant::now();
            
            let graph_context = self.build_graph_context(&initial_results).await
                .map_err(|e| {
                    error!(
                        context = ?log_context,
                        error = %e,
                        "Failed to build graph context for multi-hop reasoning"
                    );
                    e
                })?;
            
            let chain = self.multi_hop_reasoner.perform_reasoning(
                &processed_query,
                &initial_results,
                &graph_context,
                query.max_reasoning_hops,
            ).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to perform multi-hop reasoning"
                );
                e
            })?;
            
            debug!(
                reasoning_time_ms = reasoning_start.elapsed().as_millis(),
                reasoning_steps = chain.reasoning_steps.len(),
                reasoning_confidence = chain.confidence,
                max_hops = query.max_reasoning_hops,
                "Multi-hop reasoning completed"
            );
            
            Some(chain)
        } else {
            if !query.enable_multi_hop {
                debug!("Multi-hop reasoning disabled");
            } else {
                debug!(
                    initial_results = initial_results.len(),
                    max_results = query.max_results,
                    "Skipping multi-hop reasoning - sufficient initial results"
                );
            }
            None
        };
        
        // Step 4: Collect all results including multi-hop discoveries
        info!("Step 4/8: Collecting and merging results");
        let merge_start = Instant::now();
        let mut all_results = initial_results.clone();
        let mut multi_hop_results = 0;
        
        if let Some(ref chain) = reasoning_chain {
            let additional_results = self.extract_results_from_reasoning(chain).await
                .map_err(|e| {
                    error!(
                        context = ?log_context,
                        error = %e,
                        "Failed to extract results from reasoning chain"
                    );
                    e
                })?;
            multi_hop_results = additional_results.len();
            all_results.extend(additional_results);
        }
        debug!(
            merge_time_ms = merge_start.elapsed().as_millis(),
            initial_results = initial_results.len(),
            multi_hop_results = multi_hop_results,
            total_results = all_results.len(),
            "Result merging completed"
        );
        
        // Step 5: Re-rank results if enabled
        if self.config.enable_result_reranking {
            info!("Step 5/8: Re-ranking results");
            let rerank_start = Instant::now();
            let pre_rerank_count = all_results.len();
            
            all_results = self.rerank_results(all_results, &processed_query).await
                .map_err(|e| {
                    error!(
                        context = ?log_context,
                        error = %e,
                        "Failed to re-rank results"
                    );
                    e
                })?;
            
            debug!(
                rerank_time_ms = rerank_start.elapsed().as_millis(),
                results_before = pre_rerank_count,
                results_after = all_results.len(),
                "Result re-ranking completed"
            );
        } else {
            debug!("Result re-ranking disabled");
        }
        
        // Step 6: Apply context window and limit results
        info!("Step 6/8: Applying context window and result limits");
        let limit_start = Instant::now();
        let pre_limit_count = all_results.len();
        
        all_results = self.apply_context_window(all_results, query.context_window_size);
        all_results.truncate(query.max_results);
        
        debug!(
            limit_time_ms = limit_start.elapsed().as_millis(),
            results_before_limit = pre_limit_count,
            results_after_limit = all_results.len(),
            context_window_size = query.context_window_size,
            max_results = query.max_results,
            "Context window and result limiting completed"
        );
        
        // Step 7: Aggregate context
        info!("Step 7/8: Aggregating context");
        let context_start = Instant::now();
        let aggregated_context = self.context_aggregator.aggregate_context(
            &all_results,
            &processed_query,
            reasoning_chain.as_ref(),
        ).await
        .map_err(|e| {
            error!(
                context = ?log_context,
                error = %e,
                "Failed to aggregate context"
            );
            e
        })?;
        debug!(
            context_aggregation_time_ms = context_start.elapsed().as_millis(),
            coherence_score = aggregated_context.coherence_score,
            context_snippets = aggregated_context.supporting_contexts.len(),
            "Context aggregation completed"
        );
        
        // Step 8: Calculate confidence score
        info!("Step 8/8: Calculating confidence score");
        let confidence_start = Instant::now();
        let confidence_score = self.calculate_overall_confidence(
            &all_results,
            &reasoning_chain,
            &aggregated_context,
        );
        debug!(
            confidence_calculation_time_ms = confidence_start.elapsed().as_millis(),
            confidence_score = confidence_score,
            "Confidence calculation completed"
        );
        
        // Cache results if enabled
        if self.config.cache_search_results {
            debug!("Caching results for future queries");
            self.cache_results(&query, &all_results).await;
        }
        
        let retrieval_time_ms = start_time.elapsed().as_millis() as u64;
        let total_matches = all_results.len();
        
        let result = RetrievalResult {
            query_id: query_id.clone(),
            retrieved_items: all_results,
            reasoning_chain,
            query_understanding: processed_query.understanding,
            total_matches,
            retrieval_time_ms,
            confidence_score,
        };
        
        info!(
            context = ?log_context,
            query_id = %query_id,
            total_retrieval_time_ms = retrieval_time_ms,
            total_matches = total_matches,
            confidence_score = confidence_score,
            used_multi_hop = result.reasoning_chain.is_some(),
            reasoning_steps = result.reasoning_chain.as_ref().map(|c| c.reasoning_steps.len()).unwrap_or(0),
            "Retrieval query completed successfully"
        );
        
        Ok(result)
    }
    
    /// Execute search using processed query
    async fn execute_search(
        &self,
        processed_query: &ProcessedQuery,
    ) -> RetrievalResult2<Vec<RetrievedItem>> {
        // Convert to storage engine query
        let index_query = IndexQuery {
            keywords: Some(processed_query.search_components.keywords.clone()),
            entities: Some(processed_query.search_components.entities.clone()),
            concepts: Some(processed_query.search_components.concepts.clone()),
            relationship_types: if processed_query.search_components.relationships.is_empty() {
                None
            } else {
                Some(processed_query.search_components.relationships.clone())
            },
            layer_types: processed_query.original_query.structured_constraints
                .as_ref()
                .map(|c| c.layer_types.clone()),
            min_score: processed_query.original_query.min_relevance_score,
            max_results: processed_query.original_query.max_results * 2, // Get extra for filtering
            keyword_weight: 1.0,
            entity_weight: 1.5,
            concept_weight: 1.2,
            relationship_weight: 1.3,
            importance_boost: 0.5,
        };
        
        // Execute search
        let search_results = self.storage_engine.search(index_query).await
            .map_err(|e| RetrievalError::StorageAccessError(e.to_string()))?;
        
        // Execute semantic search if in semantic or hybrid mode
        let semantic_results = if matches!(
            processed_query.original_query.retrieval_mode,
            RetrievalMode::Semantic | RetrievalMode::Hybrid
        ) {
            self.storage_engine.find_similar(
                &processed_query.query_embedding,
                processed_query.original_query.max_results,
            ).await
            .map_err(|e| RetrievalError::StorageAccessError(e.to_string()))?
        } else {
            Vec::new()
        };
        
        // Convert and merge results
        let mut retrieved_items = Vec::new();
        
        // Add keyword/entity search results
        for result in search_results {
            let item = self.convert_search_result_to_item(result, processed_query).await?;
            retrieved_items.push(item);
        }
        
        // Add semantic search results
        for (layer_id, similarity) in semantic_results {
            if let Ok(layer) = self.storage_engine.retrieve_layer(&layer_id).await {
                let item = self.create_retrieved_item_from_layer(
                    layer,
                    similarity,
                    MatchType::SemanticSimilarity,
                    processed_query,
                ).await?;
                retrieved_items.push(item);
            }
        }
        
        // Remove duplicates
        self.deduplicate_results(&mut retrieved_items);
        
        Ok(retrieved_items)
    }
    
    /// Convert search result to retrieved item
    async fn convert_search_result_to_item(
        &self,
        result: SearchResult,
        processed_query: &ProcessedQuery,
    ) -> RetrievalResult2<RetrievedItem> {
        let layer = self.storage_engine.retrieve_layer(&result.layer_id).await
            .map_err(|e| RetrievalError::StorageAccessError(e.to_string()))?;
        
        let match_type = if !result.matched_keywords.is_empty() {
            MatchType::ExactKeyword
        } else if !result.matched_entities.is_empty() {
            MatchType::EntityReference
        } else if !result.matched_concepts.is_empty() {
            MatchType::ConceptMatch
        } else {
            MatchType::ContextualRelevance
        };
        
        self.create_retrieved_item_from_layer(
            layer,
            result.score,
            match_type,
            processed_query,
        ).await
    }
    
    /// Create retrieved item from layer
    async fn create_retrieved_item_from_layer(
        &self,
        layer: KnowledgeLayer,
        relevance_score: f32,
        match_type: MatchType,
        processed_query: &ProcessedQuery,
    ) -> RetrievalResult2<RetrievedItem> {
        // Extract document ID from layer ID
        let document_id = layer.layer_id
            .split('_')
            .next()
            .unwrap_or("unknown")
            .to_string();
        
        // Get context if available
        let (context_before, context_after) = if let Some(parent_id) = &layer.parent_layer_id {
            self.get_layer_context(parent_id, &layer.layer_id).await
        } else {
            (None, None)
        };
        
        // Extract semantic links
        let semantic_links = self.get_semantic_links_for_layer(&layer.layer_id).await;
        
        Ok(RetrievedItem {
            layer_id: layer.layer_id.clone(),
            document_id,
            content: layer.content.processed_text,
            relevance_score,
            match_explanation: MatchExplanation {
                matched_keywords: processed_query.search_components.keywords
                    .iter()
                    .filter(|kw| layer.content.raw_text.to_lowercase().contains(&kw.to_lowercase()))
                    .cloned()
                    .collect(),
                matched_entities: layer.entities
                    .iter()
                    .map(|e| e.name.clone())
                    .filter(|name| processed_query.search_components.entities.contains(name))
                    .collect(),
                matched_concepts: layer.content.key_phrases
                    .iter()
                    .filter(|phrase| processed_query.search_components.concepts
                        .iter()
                        .any(|concept| phrase.to_lowercase().contains(&concept.to_lowercase())))
                    .cloned()
                    .collect(),
                semantic_similarity: if match_type == MatchType::SemanticSimilarity {
                    Some(relevance_score)
                } else {
                    None
                },
                reasoning_steps: Vec::new(),
                match_type,
            },
            context_before,
            context_after,
            layer_type: layer.layer_type,
            importance_score: layer.importance_score,
            semantic_links,
        })
    }
    
    /// Build graph context for multi-hop reasoning
    async fn build_graph_context(
        &self,
        initial_results: &[RetrievedItem],
    ) -> RetrievalResult2<GraphContext> {
        let mut context = GraphContext::new();
        
        for item in initial_results {
            // Get full layer information
            if let Ok(layer) = self.storage_engine.retrieve_layer(&item.layer_id).await {
                context.add_layer(
                    layer.layer_id.clone(),
                    layer.content.processed_text,
                    layer.layer_type,
                );
                
                // Add connections from semantic links
                for link in &item.semantic_links {
                    context.add_connection(
                        item.layer_id.clone(),
                        link.layer_id.clone(),
                        vec![link.link_type.clone()],
                    );
                }
            }
        }
        
        Ok(context)
    }
    
    /// Extract additional results from reasoning chain
    async fn extract_results_from_reasoning(
        &self,
        chain: &ReasoningChain,
    ) -> RetrievalResult2<Vec<RetrievedItem>> {
        let mut results = Vec::new();
        
        for step in &chain.reasoning_steps {
            for layer_id in &step.layer_ids {
                if let Ok(layer) = self.storage_engine.retrieve_layer(layer_id).await {
                    let item = self.create_retrieved_item_from_layer(
                        layer,
                        step.confidence,
                        MatchType::MultiHopInference,
                        &ProcessedQuery {
                            original_query: RetrievalQuery::default(),
                            understanding: QueryUnderstanding {
                                intent: QueryIntent::FactualLookup,
                                extracted_entities: Vec::new(),
                                extracted_concepts: Vec::new(),
                                temporal_context: None,
                                complexity_level: ComplexityLevel::Medium,
                                suggested_expansions: Vec::new(),
                                ambiguities: Vec::new(),
                            },
                            expansion: None,
                            search_components: SearchComponents {
                                keywords: Vec::new(),
                                entities: Vec::new(),
                                concepts: Vec::new(),
                                relationships: Vec::new(),
                                boolean_queries: Vec::new(),
                                fuzzy_terms: Vec::new(),
                            },
                            query_embedding: Vec::new(),
                            temporal_context: None,
                            processing_metadata: ProcessingMetadata {
                                memory_used: 0,
                                cache_hit: false,
                                model_load_time: None,
                                inference_time: std::time::Duration::from_millis(0),
                            },
                        },
                    ).await?;
                    
                    results.push(item);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Re-rank results using advanced model
    async fn rerank_results(
        &self,
        mut results: Vec<RetrievedItem>,
        processed_query: &ProcessedQuery,
    ) -> RetrievalResult2<Vec<RetrievedItem>> {
        if results.is_empty() {
            return Ok(results);
        }
        
        // Use reranking model if configured
        if let Some(_reranking_model) = &self.config.reranking_model_id {
            // Create reranking prompt
            let prompt = format!(
                r#"Rerank these search results for the query:

Query: "{}"

Results to rank:
{}

Rank by relevance (1=most relevant). Consider:
- Direct answer to query
- Contextual relevance
- Information quality
- Completeness

Return ranking as: [index1, index2, index3, ...]

Ranking:"#,
                processed_query.original_query.natural_language_query,
                results.iter()
                    .enumerate()
                    .map(|(i, item)| format!("{}: {}", i, &item.content[..item.content.len().min(200)]))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            
            let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
            
            if let Ok(result) = self.model_manager.process_with_optimal_model(task).await {
                // Parse ranking and reorder
                if let Ok(ranking) = self.parse_ranking(&result.output, results.len()) {
                    let mut reranked = Vec::new();
                    for idx in ranking {
                        if idx < results.len() {
                            reranked.push(results[idx].clone());
                        }
                    }
                    return Ok(reranked);
                }
            }
        }
        
        // Fallback to score-based sorting
        results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(results)
    }
    
    /// Apply context window to results
    fn apply_context_window(
        &self,
        results: Vec<RetrievedItem>,
        window_size: usize,
    ) -> Vec<RetrievedItem> {
        results.into_iter()
            .map(|mut item| {
                // Truncate content if needed
                if item.content.len() > window_size {
                    item.content = item.content[..window_size].to_string();
                    if !item.content.ends_with('.') {
                        item.content.push_str("...");
                    }
                }
                item
            })
            .collect()
    }
    
    /// Calculate overall confidence score
    fn calculate_overall_confidence(
        &self,
        results: &[RetrievedItem],
        reasoning_chain: &Option<ReasoningChain>,
        aggregated_context: &AggregatedContext,
    ) -> f32 {
        let mut confidence = 0.0;
        
        // Base confidence from result relevance
        if !results.is_empty() {
            let avg_relevance = results.iter()
                .map(|r| r.relevance_score)
                .sum::<f32>() / results.len() as f32;
            confidence += avg_relevance * 0.4;
        }
        
        // Boost from reasoning chain
        if let Some(chain) = reasoning_chain {
            confidence += chain.confidence * 0.3;
        }
        
        // Boost from context coherence
        confidence += aggregated_context.coherence_score * 0.3;
        
        confidence.min(1.0)
    }
    
    // Helper methods
    
    /// Generate unique query ID
    fn generate_query_id(&self, query: &RetrievalQuery) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.natural_language_query.hash(&mut hasher);
        let hash = hasher.finish();
        
        format!("query_{:x}_{}", hash, std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos())
    }
    
    /// Check cache for results
    async fn check_cache(&self, query: &RetrievalQuery) -> Option<Vec<RetrievedItem>> {
        let cache = self.search_cache.read().await;
        let query_hash = self.calculate_query_hash(query);
        
        if let Some(entry) = cache.get(&query_hash) {
            let age_seconds = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() - entry.timestamp;
            
            if age_seconds < self.config.cache_ttl_seconds {
                return Some(entry.results.clone());
            }
        }
        
        None
    }
    
    /// Cache search results
    async fn cache_results(&self, query: &RetrievalQuery, results: &[RetrievedItem]) {
        let mut cache = self.search_cache.write().await;
        let query_hash = self.calculate_query_hash(query);
        
        cache.insert(query_hash, SearchCacheEntry {
            query_hash,
            results: results.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 1,
        });
        
        // Clean old entries if cache is too large
        if cache.len() > 1000 {
            let mut entries: Vec<_> = cache.iter().map(|(k, v)| (*k, v.timestamp)).collect();
            entries.sort_by_key(|e| e.1);
            
            // Remove oldest 20%
            let to_remove = entries.len() / 5;
            for (key, _) in entries.into_iter().take(to_remove) {
                cache.remove(&key);
            }
        }
    }
    
    /// Calculate query hash
    fn calculate_query_hash(&self, query: &RetrievalQuery) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.natural_language_query.hash(&mut hasher);
        format!("{:?}", query.retrieval_mode).hash(&mut hasher);
        hasher.finish()
    }
    
    /// Create cached result
    fn create_cached_result(
        &self,
        query_id: String,
        cached_items: Vec<RetrievedItem>,
        start_time: Instant,
    ) -> RetrievalResult {
        RetrievalResult {
            query_id,
            retrieved_items: cached_items.clone(),
            reasoning_chain: None,
            query_understanding: QueryUnderstanding {
                intent: QueryIntent::FactualLookup,
                extracted_entities: Vec::new(),
                extracted_concepts: Vec::new(),
                temporal_context: None,
                complexity_level: ComplexityLevel::Low,
                suggested_expansions: Vec::new(),
                ambiguities: Vec::new(),
            },
            total_matches: cached_items.len(),
            retrieval_time_ms: start_time.elapsed().as_millis() as u64,
            confidence_score: 0.9, // High confidence for cached results
        }
    }
    
    /// Get layer context
    async fn get_layer_context(
        &self,
        _parent_id: &str,
        _layer_id: &str,
    ) -> (Option<String>, Option<String>) {
        // Simplified - would get actual sibling layers
        (None, None)
    }
    
    /// Get semantic links for layer
    async fn get_semantic_links_for_layer(&self, _layer_id: &str) -> Vec<LinkedLayer> {
        // Simplified - would query actual semantic links
        Vec::new()
    }
    
    /// Deduplicate results
    fn deduplicate_results(&self, results: &mut Vec<RetrievedItem>) {
        let mut seen = std::collections::HashSet::new();
        results.retain(|item| seen.insert(item.layer_id.clone()));
    }
    
    /// Parse ranking from model output
    fn parse_ranking(&self, output: &str, max_len: usize) -> Result<Vec<usize>> {
        // Extract numbers from output like "[2, 0, 1, 3]"
        let numbers: std::result::Result<Vec<usize>, _> = output
            .chars()
            .filter(|c| c.is_numeric() || *c == ',')
            .collect::<String>()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().parse::<usize>())
            .collect();
        
        match numbers {
            Ok(ranking) if ranking.iter().all(|&idx| idx < max_len) => Ok(ranking),
            _ => Err(EnhancedStorageError::ProcessingFailed(
                "Failed to parse ranking from model output".to_string()
            )),
        }
    }
}

// Import necessary types
use crate::enhanced_knowledge_storage::retrieval_system::query_processor::{ProcessedQuery, SearchComponents};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::{
        model_management::ModelResourceManager,
        hierarchical_storage::HierarchicalStorageConfig,
    };
    
    #[tokio::test]
    async fn test_retrieval_engine_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        let storage_engine = Arc::new(HierarchicalStorageEngine::new(
            model_manager.clone(),
            storage_config,
        ));
        let retrieval_config = RetrievalConfig::default();
        
        let engine = RetrievalEngine::new(model_manager, storage_engine, retrieval_config);
        
        assert!(engine.config.cache_search_results);
        assert!(engine.config.enable_result_reranking);
    }
    
    #[test]
    fn test_query_id_generation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        let storage_engine = Arc::new(HierarchicalStorageEngine::new(
            model_manager.clone(),
            storage_config,
        ));
        let retrieval_config = RetrievalConfig::default();
        
        let engine = RetrievalEngine::new(model_manager, storage_engine, retrieval_config);
        
        let query = RetrievalQuery {
            natural_language_query: "test query".to_string(),
            ..Default::default()
        };
        
        let id1 = engine.generate_query_id(&query);
        let id2 = engine.generate_query_id(&query);
        
        assert!(id1.starts_with("query_"));
        assert!(id2.starts_with("query_"));
        assert_ne!(id1, id2); // Different timestamps
    }
    
    #[test]
    fn test_query_hash_calculation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        let storage_engine = Arc::new(HierarchicalStorageEngine::new(
            model_manager.clone(),
            storage_config,
        ));
        let retrieval_config = RetrievalConfig::default();
        
        let engine = RetrievalEngine::new(model_manager, storage_engine, retrieval_config);
        
        let query1 = RetrievalQuery {
            natural_language_query: "test query".to_string(),
            retrieval_mode: RetrievalMode::Hybrid,
            ..Default::default()
        };
        
        let query2 = RetrievalQuery {
            natural_language_query: "test query".to_string(),
            retrieval_mode: RetrievalMode::Semantic,
            ..Default::default()
        };
        
        let hash1 = engine.calculate_query_hash(&query1);
        let hash2 = engine.calculate_query_hash(&query2);
        
        assert_ne!(hash1, hash2); // Different modes should produce different hashes
    }
    
    #[test]
    fn test_ranking_parser() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        let storage_engine = Arc::new(HierarchicalStorageEngine::new(
            model_manager.clone(),
            storage_config,
        ));
        let retrieval_config = RetrievalConfig::default();
        
        let engine = RetrievalEngine::new(model_manager, storage_engine, retrieval_config);
        
        assert_eq!(engine.parse_ranking("[0, 2, 1]", 3).unwrap(), vec![0, 2, 1]);
        assert_eq!(engine.parse_ranking("2, 0, 1", 3).unwrap(), vec![2, 0, 1]);
        assert!(engine.parse_ranking("[0, 3, 1]", 3).is_err()); // Index out of bounds
        assert!(engine.parse_ranking("999999999999999999999", 3).is_err()); // Number too large to parse
    }
}