use crate::core::graph::KnowledgeGraph;
use crate::error::{GraphError, Result};
use crate::query::clustering::{Community, ClusterHierarchy};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Community summarization using LLM integration
pub struct CommunitySummarizer {
    llm_client: Arc<dyn LLMClient>,
    summarization_prompts: HashMap<String, String>,
    cache: Arc<RwLock<HashMap<u32, CommunitySummary>>>,
}

impl CommunitySummarizer {
    pub fn new(llm_client: Arc<dyn LLMClient>) -> Self {
        let mut prompts = HashMap::new();
        
        // Entity-focused summarization
        prompts.insert("entity_summary".to_string(), 
            "Analyze the following entities and their relationships. Provide a concise summary that captures the main theme, key entities, and their significance:\n\n{entities}\n\nSummary:".to_string());
        
        // Topic-focused summarization
        prompts.insert("topic_summary".to_string(),
            "Given these related entities and facts, identify the main topic or domain they represent and explain their collective significance:\n\n{entities}\n\nTopic Summary:".to_string());
        
        // Relationship-focused summarization
        prompts.insert("relationship_summary".to_string(),
            "Examine the relationships between these entities. Describe the key connections and how they form a cohesive group:\n\n{entities}\n\nRelationship Summary:".to_string());

        Self {
            llm_client,
            summarization_prompts: prompts,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn summarize_community(&self, community: &Community, graph: &KnowledgeGraph) -> Result<CommunitySummary> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&community.id) {
                return Ok(cached.clone());
            }
        }

        // Extract entity information
        let entity_info = self.extract_entity_information(community, graph).await?;
        
        // Generate different types of summaries
        let entity_summary = self.generate_entity_summary(&entity_info).await?;
        let topic_summary = self.generate_topic_summary(&entity_info).await?;
        let relationship_summary = self.generate_relationship_summary(&entity_info).await?;
        
        // Create comprehensive summary
        let summary = CommunitySummary {
            community_id: community.id,
            entity_count: community.entities.len(),
            main_summary: entity_summary,
            topic_summary,
            relationship_summary,
            key_entities: self.identify_key_entities(&entity_info),
            confidence_score: self.calculate_confidence_score(&entity_info),
            creation_timestamp: chrono::Utc::now(),
        };

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(community.id, summary.clone());
        }

        Ok(summary)
    }

    pub async fn summarize_hierarchy(&self, hierarchy: &ClusterHierarchy, graph: &KnowledgeGraph) -> Result<HierarchySummary> {
        let mut level_summaries = Vec::new();
        
        for (level_idx, level) in hierarchy.levels.iter().enumerate() {
            let mut community_summaries = HashMap::new();
            
            for (community_id, community) in &level.communities {
                let summary = self.summarize_community(community, graph).await?;
                community_summaries.insert(*community_id, summary);
            }
            
            let level_overview = self.generate_level_overview(level_idx, &community_summaries).await?;
            let level_summary = LevelSummary {
                level: level_idx,
                resolution: level.resolution,
                community_summaries,
                level_overview,
            };
            
            level_summaries.push(level_summary);
        }

        let global_overview = self.generate_global_overview(&level_summaries).await?;
        Ok(HierarchySummary {
            total_levels: hierarchy.levels.len(),
            level_summaries,
            global_overview,
        })
    }

    async fn extract_entity_information(&self, community: &Community, graph: &KnowledgeGraph) -> Result<Vec<EntityInfo>> {
        let mut entity_info = Vec::new();
        
        for &entity_id in &community.entities {
            if let Ok((_metadata, data)) = graph.get_entity(entity_id) {
                let neighbors = graph.get_neighbors(entity_id).unwrap_or_default();
                
                let info = EntityInfo {
                    id: entity_id,
                    properties: data.properties.clone(),
                    neighbor_count: neighbors.len(),
                    internal_connections: neighbors.iter()
                        .filter(|&&n| community.entities.contains(&n))
                        .count(),
                    external_connections: neighbors.iter()
                        .filter(|&&n| !community.entities.contains(&n))
                        .count(),
                };
                
                entity_info.push(info);
            }
        }
        
        Ok(entity_info)
    }

    async fn generate_entity_summary(&self, entity_info: &[EntityInfo]) -> Result<String> {
        let entities_text = self.format_entities_for_prompt(entity_info);
        let prompt = self.summarization_prompts.get("entity_summary")
            .ok_or_else(|| GraphError::InvalidInput("Entity summary prompt not found".to_string()))?
            .replace("{entities}", &entities_text);
        
        self.llm_client.generate_text(&prompt).await
    }

    async fn generate_topic_summary(&self, entity_info: &[EntityInfo]) -> Result<String> {
        let entities_text = self.format_entities_for_prompt(entity_info);
        let prompt = self.summarization_prompts.get("topic_summary")
            .ok_or_else(|| GraphError::InvalidInput("Topic summary prompt not found".to_string()))?
            .replace("{entities}", &entities_text);
        
        self.llm_client.generate_text(&prompt).await
    }

    async fn generate_relationship_summary(&self, entity_info: &[EntityInfo]) -> Result<String> {
        let entities_text = self.format_entities_for_prompt(entity_info);
        let prompt = self.summarization_prompts.get("relationship_summary")
            .ok_or_else(|| GraphError::InvalidInput("Relationship summary prompt not found".to_string()))?
            .replace("{entities}", &entities_text);
        
        self.llm_client.generate_text(&prompt).await
    }

    async fn generate_level_overview(&self, level: usize, community_summaries: &HashMap<u32, CommunitySummary>) -> Result<String> {
        let summaries_text = community_summaries.values()
            .map(|s| format!("Community {}: {}", s.community_id, s.main_summary))
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Analyze the following community summaries from clustering level {}. Provide an overview of the main themes and patterns at this level:\n\n{}\n\nLevel Overview:",
            level, summaries_text
        );
        
        self.llm_client.generate_text(&prompt).await
    }

    async fn generate_global_overview(&self, level_summaries: &[LevelSummary]) -> Result<String> {
        let levels_text = level_summaries.iter()
            .map(|l| format!("Level {}: {}", l.level, l.level_overview))
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Analyze the following hierarchical clustering results. Provide a global overview of the knowledge graph structure and main themes:\n\n{}\n\nGlobal Overview:",
            levels_text
        );
        
        self.llm_client.generate_text(&prompt).await
    }

    fn format_entities_for_prompt(&self, entity_info: &[EntityInfo]) -> String {
        entity_info.iter()
            .map(|e| format!(
                "Entity {}: {} (neighbors: {}, internal: {}, external: {})",
                e.id, e.properties, e.neighbor_count, e.internal_connections, e.external_connections
            ))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn identify_key_entities(&self, entity_info: &[EntityInfo]) -> Vec<u32> {
        let mut scored_entities: Vec<(u32, f64)> = entity_info.iter()
            .map(|e| {
                let centrality_score = e.internal_connections as f64 / (e.neighbor_count as f64 + 1.0);
                let hub_score = e.external_connections as f64 / (e.neighbor_count as f64 + 1.0);
                let total_score = centrality_score + hub_score;
                (e.id, total_score)
            })
            .collect();
        
        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_entities.into_iter()
            .take(5)
            .map(|(id, _)| id)
            .collect()
    }

    fn calculate_confidence_score(&self, entity_info: &[EntityInfo]) -> f64 {
        if entity_info.is_empty() {
            return 0.0;
        }
        
        let total_connections: usize = entity_info.iter()
            .map(|e| e.internal_connections)
            .sum();
        
        let total_possible = entity_info.len() * (entity_info.len() - 1) / 2;
        
        if total_possible == 0 {
            1.0
        } else {
            (total_connections as f64 / total_possible as f64).min(1.0)
        }
    }

    pub async fn invalidate_cache(&self, community_id: u32) {
        let mut cache = self.cache.write().await;
        cache.remove(&community_id);
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Trait for LLM client abstraction
#[async_trait::async_trait]
pub trait LLMClient: Send + Sync {
    async fn generate_text(&self, prompt: &str) -> Result<String>;
}

/// Mock LLM client for testing
pub struct MockLLMClient;

#[async_trait::async_trait]
impl LLMClient for MockLLMClient {
    async fn generate_text(&self, prompt: &str) -> Result<String> {
        // Mock implementation for testing
        if prompt.contains("entity_summary") {
            Ok("This community represents a cluster of related entities with strong interconnections.".to_string())
        } else if prompt.contains("topic_summary") {
            Ok("The main topic of this community appears to be a specific domain or subject area.".to_string())
        } else if prompt.contains("relationship_summary") {
            Ok("These entities are connected through various relationships forming a cohesive group.".to_string())
        } else {
            Ok("Generated summary based on the provided information.".to_string())
        }
    }
}

/// Information about an entity for summarization
#[derive(Debug, Clone)]
struct EntityInfo {
    id: u32,
    properties: String,
    neighbor_count: usize,
    internal_connections: usize,
    external_connections: usize,
}

/// Complete summary of a community
#[derive(Debug, Clone)]
pub struct CommunitySummary {
    pub community_id: u32,
    pub entity_count: usize,
    pub main_summary: String,
    pub topic_summary: String,
    pub relationship_summary: String,
    pub key_entities: Vec<u32>,
    pub confidence_score: f64,
    pub creation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Summary of a clustering level
#[derive(Debug, Clone)]
pub struct LevelSummary {
    pub level: usize,
    pub resolution: f64,
    pub community_summaries: HashMap<u32, CommunitySummary>,
    pub level_overview: String,
}

/// Complete hierarchy summary
#[derive(Debug, Clone)]
pub struct HierarchySummary {
    pub total_levels: usize,
    pub level_summaries: Vec<LevelSummary>,
    pub global_overview: String,
}

impl CommunitySummary {
    pub fn to_llm_context(&self) -> String {
        format!(
            "Community {} ({} entities):\n- Main Theme: {}\n- Topic: {}\n- Relationships: {}\n- Key Entities: {:?}\n- Confidence: {:.2}",
            self.community_id,
            self.entity_count,
            self.main_summary,
            self.topic_summary,
            self.relationship_summary,
            self.key_entities,
            self.confidence_score
        )
    }
}

impl HierarchySummary {
    pub fn to_llm_context(&self) -> String {
        let mut context = format!("Knowledge Graph Hierarchy ({} levels):\n\n", self.total_levels);
        
        context.push_str(&format!("Global Overview: {}\n\n", self.global_overview));
        
        for level in &self.level_summaries {
            context.push_str(&format!("Level {} (Resolution: {:.2}):\n", level.level, level.resolution));
            context.push_str(&format!("  Overview: {}\n", level.level_overview));
            context.push_str(&format!("  Communities: {}\n", level.community_summaries.len()));
            
            for summary in level.community_summaries.values().take(3) {
                context.push_str(&format!("    - {}\n", summary.main_summary));
            }
            
            context.push_str("\n");
        }
        
        context
    }
}