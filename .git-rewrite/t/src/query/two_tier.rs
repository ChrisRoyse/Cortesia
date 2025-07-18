use crate::core::graph::KnowledgeGraph;
use crate::core::types::ContextEntity;
use crate::error::{GraphError, Result};
use crate::query::clustering::ClusterHierarchy;
use crate::query::summarization::CommunitySummarizer;
use crate::query::rag::GraphRAGEngine;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Two-tier query system for GraphRAG
pub struct TwoTierQueryEngine {
    graph: Arc<KnowledgeGraph>,
    hierarchy: Arc<RwLock<Option<ClusterHierarchy>>>,
    summarizer: Arc<CommunitySummarizer>,
    rag_engine: Arc<GraphRAGEngine>,
    query_cache: Arc<RwLock<HashMap<String, CachedQueryResult>>>,
}

impl TwoTierQueryEngine {
    pub fn new(
        graph: Arc<KnowledgeGraph>,
        summarizer: Arc<CommunitySummarizer>,
        rag_engine: Arc<GraphRAGEngine>,
    ) -> Self {
        Self {
            graph,
            hierarchy: Arc::new(RwLock::new(None)),
            summarizer,
            rag_engine,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn set_hierarchy(&self, hierarchy: ClusterHierarchy) {
        let mut h = self.hierarchy.write().await;
        *h = Some(hierarchy);
    }

    /// Main query entry point that determines query type and routes accordingly
    pub async fn query(&self, query: GraphRAGQuery) -> Result<GraphRAGResult> {
        let cache_key = self.generate_cache_key(&query);
        
        // Check cache first
        {
            let cache = self.query_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                if !cached.is_expired() {
                    return Ok(cached.result.clone());
                }
            }
        }

        let result = match query {
            GraphRAGQuery::GlobalSearch { 
                question, 
                use_community_summaries, 
                max_communities 
            } => {
                self.global_search(&question, use_community_summaries, max_communities).await?
            }
            GraphRAGQuery::LocalSearch { 
                entity, 
                max_hops, 
                include_neighbors 
            } => {
                self.local_search(&entity, max_hops, include_neighbors).await?
            }
            GraphRAGQuery::HybridSearch {
                question,
                target_entities,
                max_global_communities,
                max_local_hops,
            } => {
                self.hybrid_search(&question, target_entities, max_global_communities, max_local_hops).await?
            }
        };

        // Cache the result
        {
            let mut cache = self.query_cache.write().await;
            cache.insert(cache_key, CachedQueryResult::new(result.clone()));
        }

        Ok(result)
    }

    /// Global search using community summaries for broad knowledge retrieval
    async fn global_search(
        &self,
        question: &str,
        use_community_summaries: bool,
        max_communities: usize,
    ) -> Result<GraphRAGResult> {
        let hierarchy = self.hierarchy.read().await;
        let hierarchy = hierarchy.as_ref()
            .ok_or_else(|| GraphError::InvalidInput("Hierarchy not available".to_string()))?;

        if use_community_summaries {
            self.community_based_global_search(question, hierarchy, max_communities).await
        } else {
            self.traditional_global_search(question, max_communities).await
        }
    }

    /// Local search focusing on specific entities and their neighborhoods
    async fn local_search(
        &self,
        entity: &str,
        max_hops: u8,
        include_neighbors: bool,
    ) -> Result<GraphRAGResult> {
        // Find the entity in the graph
        let entity_id = self.find_entity_by_name(entity).await?;
        
        // Use the RAG engine for local exploration
        let query_embedding = self.generate_entity_embedding(entity_id).await?;
        let rag_context = self.rag_engine.retrieve_context(&query_embedding, 50, max_hops)?;
        
        // Filter to focus on the local neighborhood
        let local_entities = if include_neighbors {
            rag_context.entities
        } else {
            rag_context.entities.into_iter()
                .filter(|e| e.id == entity_id)
                .collect()
        };

        Ok(GraphRAGResult {
            query_type: QueryType::LocalSearch,
            entities: local_entities,
            relationships: rag_context.relationships,
            community_summaries: Vec::new(),
            global_context: None,
            confidence_score: 0.9,
            processing_time_ms: 0, // Would be measured
        })
    }

    /// Hybrid search combining global and local strategies
    async fn hybrid_search(
        &self,
        question: &str,
        target_entities: Vec<String>,
        max_global_communities: usize,
        max_local_hops: u8,
    ) -> Result<GraphRAGResult> {
        // Step 1: Global search for broad context
        let global_result = self.global_search(question, true, max_global_communities).await?;
        
        // Step 2: Local search for each target entity
        let mut local_entities = Vec::new();
        let mut local_relationships = Vec::new();
        
        for entity in target_entities {
            let local_result = self.local_search(&entity, max_local_hops, true).await?;
            local_entities.extend(local_result.entities);
            local_relationships.extend(local_result.relationships);
        }
        
        // Step 3: Merge and rank results
        let merged_entities = self.merge_and_rank_entities(global_result.entities, local_entities)?;
        let merged_relationships = self.merge_relationships(global_result.relationships, local_relationships)?;
        
        Ok(GraphRAGResult {
            query_type: QueryType::HybridSearch,
            entities: merged_entities,
            relationships: merged_relationships,
            community_summaries: global_result.community_summaries,
            global_context: global_result.global_context,
            confidence_score: (global_result.confidence_score + 0.9) / 2.0,
            processing_time_ms: 0,
        })
    }

    /// Community-based global search using hierarchical summaries
    async fn community_based_global_search(
        &self,
        question: &str,
        hierarchy: &ClusterHierarchy,
        max_communities: usize,
    ) -> Result<GraphRAGResult> {
        // Generate question embedding
        let question_embedding = self.generate_question_embedding(question).await?;
        
        // Find relevant communities at different levels
        let relevant_communities = self.find_relevant_communities(
            &question_embedding,
            hierarchy,
            max_communities,
        ).await?;
        
        // Generate summaries for relevant communities
        let mut community_summaries = Vec::new();
        for (level, community_id) in relevant_communities {
            if let Some(communities) = hierarchy.get_communities_at_level(level) {
                if let Some(community) = communities.get(&community_id) {
                    let summary = self.summarizer.summarize_community(community, &self.graph).await?;
                    community_summaries.push(summary);
                }
            }
        }
        
        // Extract key entities from communities
        let key_entities = self.extract_key_entities_from_communities(&community_summaries)?;
        
        // Generate global context
        let global_context = self.generate_global_context(question, &community_summaries).await?;
        
        Ok(GraphRAGResult {
            query_type: QueryType::GlobalSearch,
            entities: key_entities,
            relationships: Vec::new(),
            community_summaries,
            global_context: Some(global_context),
            confidence_score: 0.8,
            processing_time_ms: 0,
        })
    }

    /// Traditional global search without community summaries
    async fn traditional_global_search(
        &self,
        question: &str,
        max_entities: usize,
    ) -> Result<GraphRAGResult> {
        let question_embedding = self.generate_question_embedding(question).await?;
        let rag_context = self.rag_engine.retrieve_context(&question_embedding, max_entities, 2)?;
        
        Ok(GraphRAGResult {
            query_type: QueryType::GlobalSearch,
            entities: rag_context.entities,
            relationships: rag_context.relationships,
            community_summaries: Vec::new(),
            global_context: None,
            confidence_score: 0.7,
            processing_time_ms: 0,
        })
    }

    /// Find relevant communities based on question embedding
    async fn find_relevant_communities(
        &self,
        question_embedding: &[f32],
        hierarchy: &ClusterHierarchy,
        max_communities: usize,
    ) -> Result<Vec<(usize, u32)>> {
        let mut relevant_communities = Vec::new();
        
        // Search through each level of the hierarchy
        for (level_idx, level) in hierarchy.levels.iter().enumerate() {
            let mut community_scores = Vec::new();
            
            for (community_id, community) in &level.communities {
                let score = self.calculate_community_relevance(
                    question_embedding,
                    community,
                    &self.graph,
                ).await?;
                
                community_scores.push((level_idx, *community_id, score));
            }
            
            // Sort by relevance score
            community_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            // Take top communities from this level
            for (level, community_id, _) in community_scores.into_iter().take(max_communities / hierarchy.levels.len()) {
                relevant_communities.push((level, community_id));
            }
        }
        
        Ok(relevant_communities)
    }

    /// Calculate how relevant a community is to the question
    async fn calculate_community_relevance(
        &self,
        question_embedding: &[f32],
        community: &crate::query::clustering::Community,
        graph: &KnowledgeGraph,
    ) -> Result<f64> {
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        for &entity_id in &community.entities {
            if let Ok(entity_embedding) = self.get_entity_embedding(entity_id).await {
                let similarity = self.cosine_similarity(question_embedding, &entity_embedding);
                total_similarity += similarity as f64;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(total_similarity / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Helper functions
    async fn find_entity_by_name(&self, name: &str) -> Result<u32> {
        // This would search the graph for an entity by name
        // For now, return a placeholder
        Ok(1)
    }

    async fn generate_entity_embedding(&self, entity_id: u32) -> Result<Vec<f32>> {
        self.get_entity_embedding(entity_id).await
    }

    async fn get_entity_embedding(&self, entity_id: u32) -> Result<Vec<f32>> {
        // This would get the actual embedding from the graph
        Ok(vec![0.0; 384]) // Placeholder
    }

    async fn generate_question_embedding(&self, question: &str) -> Result<Vec<f32>> {
        // This would generate an embedding for the question
        Ok(vec![0.0; 384]) // Placeholder
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    fn extract_key_entities_from_communities(&self, summaries: &[crate::query::summarization::CommunitySummary]) -> Result<Vec<ContextEntity>> {
        let mut entities = Vec::new();
        
        for summary in summaries {
            for &entity_id in &summary.key_entities {
                entities.push(ContextEntity {
                    id: entity_id,
                    similarity: summary.confidence_score as f32,
                    neighbors: Vec::new(),
                    properties: format!("Key entity from community {}", summary.community_id),
                });
            }
        }
        
        Ok(entities)
    }

    async fn generate_global_context(&self, question: &str, summaries: &[crate::query::summarization::CommunitySummary]) -> Result<String> {
        let summaries_text = summaries.iter()
            .map(|s| s.to_llm_context())
            .collect::<Vec<_>>()
            .join("\n\n");
        
        Ok(format!(
            "Global context for question: '{}'\n\nRelevant communities:\n{}",
            question, summaries_text
        ))
    }

    fn merge_and_rank_entities(&self, global_entities: Vec<ContextEntity>, local_entities: Vec<ContextEntity>) -> Result<Vec<ContextEntity>> {
        let mut all_entities = global_entities;
        all_entities.extend(local_entities);
        
        // Remove duplicates and rank by similarity
        let mut unique_entities = std::collections::HashMap::new();
        for entity in all_entities {
            unique_entities.entry(entity.id)
                .and_modify(|e: &mut ContextEntity| {
                    if entity.similarity > e.similarity {
                        *e = entity.clone();
                    }
                })
                .or_insert(entity);
        }
        
        let mut result: Vec<ContextEntity> = unique_entities.into_values().collect();
        result.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        Ok(result)
    }

    fn merge_relationships(&self, global_rels: Vec<crate::core::types::Relationship>, local_rels: Vec<crate::core::types::Relationship>) -> Result<Vec<crate::core::types::Relationship>> {
        let mut all_rels = global_rels;
        all_rels.extend(local_rels);
        
        // Remove duplicates
        let mut unique_rels = std::collections::HashSet::new();
        let mut result = Vec::new();
        
        for rel in all_rels {
            let key = (rel.from, rel.to, rel.rel_type);
            if unique_rels.insert(key) {
                result.push(rel);
            }
        }
        
        Ok(result)
    }

    fn generate_cache_key(&self, query: &GraphRAGQuery) -> String {
        match query {
            GraphRAGQuery::GlobalSearch { question, use_community_summaries, max_communities } => {
                format!("global:{}:{}:{}", question, use_community_summaries, max_communities)
            }
            GraphRAGQuery::LocalSearch { entity, max_hops, include_neighbors } => {
                format!("local:{}:{}:{}", entity, max_hops, include_neighbors)
            }
            GraphRAGQuery::HybridSearch { question, target_entities, max_global_communities, max_local_hops } => {
                format!("hybrid:{}:{:?}:{}:{}", question, target_entities, max_global_communities, max_local_hops)
            }
        }
    }
}

/// Query types for the two-tier system
#[derive(Debug, Clone)]
pub enum GraphRAGQuery {
    GlobalSearch {
        question: String,
        use_community_summaries: bool,
        max_communities: usize,
    },
    LocalSearch {
        entity: String,
        max_hops: u8,
        include_neighbors: bool,
    },
    HybridSearch {
        question: String,
        target_entities: Vec<String>,
        max_global_communities: usize,
        max_local_hops: u8,
    },
}

/// Result type for GraphRAG queries
#[derive(Debug, Clone)]
pub struct GraphRAGResult {
    pub query_type: QueryType,
    pub entities: Vec<ContextEntity>,
    pub relationships: Vec<crate::core::types::Relationship>,
    pub community_summaries: Vec<crate::query::summarization::CommunitySummary>,
    pub global_context: Option<String>,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone)]
pub enum QueryType {
    GlobalSearch,
    LocalSearch,
    HybridSearch,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedQueryResult {
    result: GraphRAGResult,
    timestamp: chrono::DateTime<chrono::Utc>,
    ttl_seconds: i64,
}

impl CachedQueryResult {
    fn new(result: GraphRAGResult) -> Self {
        Self {
            result,
            timestamp: chrono::Utc::now(),
            ttl_seconds: 300, // 5 minutes
        }
    }

    fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() - self.timestamp.timestamp() > self.ttl_seconds
    }
}

impl GraphRAGResult {
    pub fn to_llm_context(&self) -> String {
        let mut context = String::new();
        
        context.push_str(&format!("Query Type: {:?}\n", self.query_type));
        context.push_str(&format!("Confidence: {:.2}\n\n", self.confidence_score));
        
        if let Some(global_context) = &self.global_context {
            context.push_str(&format!("Global Context:\n{}\n\n", global_context));
        }
        
        context.push_str("Entities:\n");
        for entity in &self.entities {
            context.push_str(&format!("- Entity {}: {} (similarity: {:.3})\n", 
                entity.id, entity.properties, entity.similarity));
        }
        
        context.push_str("\nRelationships:\n");
        for rel in &self.relationships {
            context.push_str(&format!("- {} → {} (type: {}, weight: {:.2})\n",
                rel.from, rel.to, rel.rel_type, rel.weight));
        }
        
        if !self.community_summaries.is_empty() {
            context.push_str("\nCommunity Summaries:\n");
            for summary in &self.community_summaries {
                context.push_str(&format!("- {}\n", summary.to_llm_context()));
            }
        }
        
        context
    }
}