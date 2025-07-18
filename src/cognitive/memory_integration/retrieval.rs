//! Memory retrieval operations for unified memory system

use super::types::*;
use super::coordinator::MemoryCoordinator;
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryQuery, MemoryRetrievalResult, MemoryContent, MemoryItem};
use crate::core::sdr_storage::SDRStorage;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::{Result, GraphError};
use std::sync::Arc;
use std::time::Instant;

/// Memory retrieval handler
pub struct MemoryRetrieval {
    pub working_memory: Arc<WorkingMemorySystem>,
    pub sdr_storage: Arc<SDRStorage>,
    pub long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub coordinator: Arc<MemoryCoordinator>,
}

impl MemoryRetrieval {
    /// Create new memory retrieval handler
    pub fn new(
        working_memory: Arc<WorkingMemorySystem>,
        sdr_storage: Arc<SDRStorage>,
        long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
        coordinator: Arc<MemoryCoordinator>,
    ) -> Self {
        Self {
            working_memory,
            sdr_storage,
            long_term_graph,
            coordinator,
        }
    }
    
    /// Retrieve memories using integrated approach
    pub async fn retrieve_integrated(&self, query: &str, strategy_id: Option<&str>) -> Result<MemoryIntegrationResult> {
        let start_time = Instant::now();
        
        // Get retrieval strategy
        let strategy = if let Some(id) = strategy_id {
            self.coordinator.get_strategy(id).ok_or_else(|| GraphError::ConfigError(format!("Strategy not found: {}", id)))?
        } else {
            self.coordinator.get_best_strategy(query)
        };
        
        // Execute retrieval based on strategy type
        let (primary_results, secondary_results) = match strategy.strategy_type {
            RetrievalType::ParallelSearch => {
                self.parallel_memory_retrieval(query, strategy).await?
            }
            RetrievalType::HierarchicalSearch => {
                self.hierarchical_memory_retrieval(query, strategy).await?
            }
            RetrievalType::AdaptiveSearch => {
                self.adaptive_memory_retrieval(query, strategy).await?
            }
            RetrievalType::ContextualSearch => {
                self.contextual_memory_retrieval(query, strategy).await?
            }
        };
        
        // Fuse results
        let fused_results = self.fuse_results(&primary_results, &secondary_results, strategy)?;
        
        // Activate cross-memory links
        let activated_links = self.activate_cross_memory_links(&fused_results.0).await?;
        
        let _retrieval_time = start_time.elapsed();
        
        Ok(MemoryIntegrationResult {
            primary_results: fused_results.0,
            secondary_results: fused_results.1,
            fusion_confidence: fused_results.2,
            retrieval_strategy_used: strategy.strategy_id.clone(),
            cross_memory_links_activated: activated_links,
        })
    }
    
    /// Parallel memory retrieval
    async fn parallel_memory_retrieval(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
    ) -> Result<(Vec<MemoryRetrievalResult>, Vec<MemoryRetrievalResult>)> {
        let mut primary_results = Vec::new();
        let mut secondary_results = Vec::new();
        
        // Launch parallel queries
        let working_future = self.query_working_memory(query);
        let sdr_future = self.query_sdr_storage(query);
        let graph_future = self.query_long_term_graph(query);
        
        let (working_result, sdr_result, graph_result) = 
            tokio::join!(working_future, sdr_future, graph_future);
        
        // Organize results by priority
        for (index, memory_type) in strategy.memory_priority.iter().enumerate() {
            match memory_type {
                MemoryType::WorkingMemory => {
                    if let Ok(ref result) = working_result {
                        if index == 0 {
                            primary_results.push(result.clone());
                        } else {
                            secondary_results.push(result.clone());
                        }
                    }
                }
                MemoryType::SemanticMemory => {
                    if let Ok(ref result) = sdr_result {
                        if index == 0 {
                            primary_results.push(result.clone());
                        } else {
                            secondary_results.push(result.clone());
                        }
                    }
                }
                MemoryType::LongTermMemory => {
                    if let Ok(ref result) = graph_result {
                        if index == 0 {
                            primary_results.push(result.clone());
                        } else {
                            secondary_results.push(result.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok((primary_results, secondary_results))
    }
    
    /// Hierarchical memory retrieval
    async fn hierarchical_memory_retrieval(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
    ) -> Result<(Vec<MemoryRetrievalResult>, Vec<MemoryRetrievalResult>)> {
        let mut primary_results = Vec::new();
        let mut secondary_results = Vec::new();
        
        // Query memories hierarchically based on priority
        for (index, memory_type) in strategy.memory_priority.iter().enumerate() {
            let result = match memory_type {
                MemoryType::WorkingMemory => self.query_working_memory(query).await,
                MemoryType::SemanticMemory => self.query_sdr_storage(query).await,
                MemoryType::LongTermMemory => self.query_long_term_graph(query).await,
                _ => continue,
            };
            
            if let Ok(memory_result) = result {
                if index == 0 {
                    primary_results.push(memory_result);
                } else {
                    secondary_results.push(memory_result);
                }
                
                // Early termination if we found good results
                if index == 0 && !primary_results.is_empty() 
                    && primary_results[0].retrieval_confidence > 0.8 {
                    break;
                }
            }
        }
        
        Ok((primary_results, secondary_results))
    }
    
    /// Adaptive memory retrieval
    async fn adaptive_memory_retrieval(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
    ) -> Result<(Vec<MemoryRetrievalResult>, Vec<MemoryRetrievalResult>)> {
        // Start with hierarchical, then expand to parallel if needed
        let (mut primary_results, mut secondary_results) = 
            self.hierarchical_memory_retrieval(query, strategy).await?;
        
        // If results are insufficient, expand search
        if primary_results.is_empty() || 
           (primary_results.len() == 1 && primary_results[0].retrieval_confidence < 0.6) {
            let (additional_primary, additional_secondary) = 
                self.parallel_memory_retrieval(query, strategy).await?;
            
            primary_results.extend(additional_primary);
            secondary_results.extend(additional_secondary);
        }
        
        Ok((primary_results, secondary_results))
    }
    
    /// Contextual memory retrieval
    async fn contextual_memory_retrieval(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
    ) -> Result<(Vec<MemoryRetrievalResult>, Vec<MemoryRetrievalResult>)> {
        // Extract context from query
        let context_keywords = self.extract_context_keywords(query);
        
        // Query episodic memory first for contextual information
        let episodic_results = self.query_episodic_memory(query, &context_keywords).await?;
        
        // Use episodic results to inform other memory queries
        let enhanced_query = self.enhance_query_with_context(query, &episodic_results);
        
        // Query other memory systems with enhanced query
        let (primary_results, secondary_results) = 
            self.parallel_memory_retrieval(&enhanced_query, strategy).await?;
        
        Ok((primary_results, secondary_results))
    }
    
    /// Query working memory
    async fn query_working_memory(&self, query: &str) -> Result<MemoryRetrievalResult> {
        let memory_query = MemoryQuery {
            query_text: query.to_string(),
            search_buffers: vec![
                crate::cognitive::working_memory::BufferType::Phonological,
                crate::cognitive::working_memory::BufferType::Episodic,
                crate::cognitive::working_memory::BufferType::Visuospatial,
            ],
            apply_attention: true,
            importance_threshold: 0.3,
            recency_weight: 0.7,
        };
        
        self.working_memory.retrieve_from_working_memory(&memory_query).await
    }
    
    /// Query SDR storage
    async fn query_sdr_storage(&self, query: &str) -> Result<MemoryRetrievalResult> {
        let sdr_results = self.sdr_storage.similarity_search(query, 0.7).await?;
        
        let items = sdr_results.into_iter().map(|sdr_result| {
            MemoryItem {
                content: MemoryContent::Concept(sdr_result.content),
                activation_level: sdr_result.similarity,
                timestamp: Instant::now(),
                importance_score: sdr_result.similarity,
                access_count: 1,
                decay_factor: 1.0,
            }
        }).collect();
        
        Ok(MemoryRetrievalResult {
            items,
            retrieval_confidence: 0.9,
            buffer_states: Vec::new(),
        })
    }
    
    /// Query long-term graph
    async fn query_long_term_graph(&self, query: &str) -> Result<MemoryRetrievalResult> {
        // Convert query string to embedding (simplified - just use hash for now)
        let query_hash = query.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));
        let query_embedding = vec![query_hash as f32 / u32::MAX as f32; 96]; // Assuming 96 dim embeddings
        
        let query_result = self.long_term_graph.query(&query_embedding, &[], 10).await?;
        let graph_results: Vec<GraphSearchResult> = query_result.entities.into_iter().map(|entity| {
            GraphSearchResult {
                content: format!("entity_{:?}", entity.id),
                relevance_score: entity.similarity,
            }
        }).collect();
        
        let items = graph_results.into_iter().map(|graph_result| {
            MemoryItem {
                content: MemoryContent::Concept(graph_result.content),
                activation_level: graph_result.relevance_score,
                timestamp: Instant::now(),
                importance_score: graph_result.relevance_score,
                access_count: 1,
                decay_factor: 1.0,
            }
        }).collect();
        
        Ok(MemoryRetrievalResult {
            items,
            retrieval_confidence: 0.85,
            buffer_states: Vec::new(),
        })
    }
    
    /// Query episodic memory (simplified)
    async fn query_episodic_memory(&self, _query: &str, context: &[String]) -> Result<Vec<MemoryItem>> {
        // Simplified episodic memory query
        // In practice, this would query a dedicated episodic memory system
        let mut items = Vec::new();
        
        for context_item in context {
            items.push(MemoryItem {
                content: MemoryContent::Concept(format!("Context: {}", context_item)),
                activation_level: 0.7,
                timestamp: Instant::now(),
                importance_score: 0.6,
                access_count: 1,
                decay_factor: 1.0,
            });
        }
        
        Ok(items)
    }
    
    /// Extract context keywords from query
    fn extract_context_keywords(&self, query: &str) -> Vec<String> {
        // Simple keyword extraction
        query.split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect()
    }
    
    /// Enhance query with contextual information
    fn enhance_query_with_context(&self, query: &str, episodic_results: &[MemoryItem]) -> String {
        let mut enhanced_query = query.to_string();
        
        // Add context from episodic results
        for item in episodic_results.iter().take(3) {
            if let MemoryContent::Concept(concept) = &item.content {
                enhanced_query.push_str(&format!(" {}", concept));
            }
        }
        
        enhanced_query
    }
    
    /// Fuse results from multiple memory systems
    fn fuse_results(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
        strategy: &RetrievalStrategy,
    ) -> Result<(Vec<MemoryRetrievalResult>, Vec<MemoryRetrievalResult>, f32)> {
        let fusion_confidence = match strategy.fusion_method {
            FusionMethod::WeightedAverage => {
                self.weighted_average_fusion(primary_results, secondary_results, strategy)
            }
            FusionMethod::MaximumConfidence => {
                self.maximum_confidence_fusion(primary_results, secondary_results)
            }
            FusionMethod::MajorityVoting => {
                self.majority_voting_fusion(primary_results, secondary_results)
            }
            FusionMethod::RankFusion => {
                self.rank_fusion(primary_results, secondary_results)
            }
            FusionMethod::ContextualFusion => {
                self.contextual_fusion(primary_results, secondary_results)
            }
        };
        
        Ok((primary_results.to_vec(), secondary_results.to_vec(), fusion_confidence))
    }
    
    /// Weighted average fusion
    fn weighted_average_fusion(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
        _strategy: &RetrievalStrategy,
    ) -> f32 {
        let mut total_confidence = 0.0;
        let mut total_weight = 0.0;
        
        // Weight primary results
        for result in primary_results {
            total_confidence += result.retrieval_confidence * 0.7;
            total_weight += 0.7;
        }
        
        // Weight secondary results
        for result in secondary_results {
            total_confidence += result.retrieval_confidence * 0.3;
            total_weight += 0.3;
        }
        
        if total_weight > 0.0 {
            total_confidence / total_weight
        } else {
            0.0
        }
    }
    
    /// Maximum confidence fusion
    fn maximum_confidence_fusion(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
    ) -> f32 {
        let max_primary = primary_results.iter()
            .map(|r| r.retrieval_confidence)
            .fold(0.0, f32::max);
        
        let max_secondary = secondary_results.iter()
            .map(|r| r.retrieval_confidence)
            .fold(0.0, f32::max);
        
        max_primary.max(max_secondary)
    }
    
    /// Majority voting fusion
    fn majority_voting_fusion(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
    ) -> f32 {
        let total_results = primary_results.len() + secondary_results.len();
        if total_results == 0 {
            return 0.0;
        }
        
        let successful_results = primary_results.iter()
            .chain(secondary_results.iter())
            .filter(|r| r.retrieval_confidence > 0.5)
            .count();
        
        successful_results as f32 / total_results as f32
    }
    
    /// Rank fusion
    fn rank_fusion(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
    ) -> f32 {
        // Simple rank-based fusion
        let mut all_results: Vec<&MemoryRetrievalResult> = primary_results.iter()
            .chain(secondary_results.iter())
            .collect();
        
        all_results.sort_by(|a, b| b.retrieval_confidence.partial_cmp(&a.retrieval_confidence).unwrap());
        
        if let Some(best_result) = all_results.first() {
            best_result.retrieval_confidence
        } else {
            0.0
        }
    }
    
    /// Contextual fusion
    fn contextual_fusion(
        &self,
        primary_results: &[MemoryRetrievalResult],
        secondary_results: &[MemoryRetrievalResult],
    ) -> f32 {
        // Weight results based on contextual relevance
        let mut context_weighted_confidence = 0.0;
        let mut total_results = 0;
        
        for result in primary_results {
            // Primary results get higher contextual weight
            context_weighted_confidence += result.retrieval_confidence * 0.8;
            total_results += 1;
        }
        
        for result in secondary_results {
            // Secondary results get lower contextual weight
            context_weighted_confidence += result.retrieval_confidence * 0.4;
            total_results += 1;
        }
        
        if total_results > 0 {
            context_weighted_confidence / total_results as f32
        } else {
            0.0
        }
    }
    
    /// Activate cross-memory links
    async fn activate_cross_memory_links(&self, results: &[MemoryRetrievalResult]) -> Result<Vec<String>> {
        let mut activated_links = Vec::new();
        
        for result in results {
            for item in &result.items {
                if let MemoryContent::Concept(concept) = &item.content {
                    let links = self.coordinator.get_cross_memory_links(concept).await;
                    for link in links {
                        activated_links.push(link.link_id);
                    }
                }
            }
        }
        
        Ok(activated_links)
    }
}


/// Graph search result
#[derive(Debug, Clone)]
pub struct GraphSearchResult {
    pub content: String,
    pub relevance_score: f32,
}