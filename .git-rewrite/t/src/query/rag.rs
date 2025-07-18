use crate::core::graph::KnowledgeGraph;
use crate::core::types::{ContextEntity, QueryResult, Relationship};
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub struct GraphRAGEngine {
    graph: KnowledgeGraph,
    cache: HashMap<Vec<u8>, CachedResult>,
    cache_size_limit: usize,
}

#[derive(Clone)]
struct CachedResult {
    result: QueryResult,
    timestamp: std::time::Instant,
    access_count: u32,
}

impl GraphRAGEngine {
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Ok(Self {
            graph: KnowledgeGraph::new_with_dimension(embedding_dim)?,
            cache: HashMap::new(),
            cache_size_limit: 1000,
        })
    }
    
    /// Retrieve context using Graph RAG methodology
    /// Combines vector similarity search with graph traversal for comprehensive context
    pub fn retrieve_context(&self, query_embedding: &[f32], max_entities: usize, max_depth: u8) -> Result<GraphRAGContext> {
        let start_time = Instant::now();
        
        // For now, skip caching in the immutable version
        // In production, use interior mutability or Arc<Mutex<Cache>>
        
        // Step 1: Vector similarity search to find relevant entities
        let similar_entities = self.graph.similarity_search(query_embedding, max_entities * 2)?;
        
        // Step 2: Expand with graph context using multiple strategies
        let mut context_entities = Vec::new();
        let mut all_relationships = Vec::new();
        let mut visited_entities = HashSet::new();
        
        // Strategy 1: Direct neighbors of similar entities
        for (entity_id, similarity) in &similar_entities {
            if context_entities.len() >= max_entities {
                break;
            }
            
            if visited_entities.insert(*entity_id) {
                let neighbors = self.graph.get_neighbors(*entity_id)?;
                let (meta, data) = self.graph.get_entity(*entity_id)?;
                
                context_entities.push(ContextEntity {
                    id: *entity_id,
                    similarity: *similarity,
                    neighbors: neighbors.clone(),
                    properties: data.properties,
                });
                
                // Collect relationships
                for &neighbor_id in &neighbors {
                    if neighbor_id != *entity_id {
                        all_relationships.push(Relationship {
                            from: *entity_id,
                            to: neighbor_id,
                            rel_type: 0, // Would be determined from graph structure
                            weight: 1.0,
                        });
                    }
                }
            }
        }
        
        // Strategy 2: Multi-hop exploration for highly relevant entities
        let top_entities: Vec<_> = similar_entities.iter()
            .take(max_entities.min(5))
            .map(|(id, _)| *id)
            .collect();
        
        for &entity_id in &top_entities {
            self.expand_entity_context(entity_id, max_depth, &mut visited_entities, &mut context_entities, &mut all_relationships)?;
        }
        
        // Strategy 3: Bridge entities (entities that connect multiple similar entities)
        let bridge_entities = self.find_bridge_entities(&similar_entities, max_depth)?;
        for bridge_id in bridge_entities {
            if !visited_entities.contains(&bridge_id) && context_entities.len() < max_entities {
                if let Ok((meta, data)) = self.graph.get_entity(bridge_id) {
                    let neighbors = self.graph.get_neighbors(bridge_id)?;
                    
                    context_entities.push(ContextEntity {
                        id: bridge_id,
                        similarity: 0.5, // Bridge entities get moderate similarity
                        neighbors: neighbors.clone(),
                        properties: data.properties,
                    });
                    
                    visited_entities.insert(bridge_id);
                }
            }
        }
        
        // Step 3: Rank and filter context
        self.rank_context_entities(&mut context_entities, query_embedding)?;
        context_entities.truncate(max_entities);
        
        // Step 4: Generate explanatory text for relationships
        let relationship_explanations = self.generate_relationship_explanations(&all_relationships, &context_entities)?;
        
        let query_time = start_time.elapsed();
        
        let result = GraphRAGContext {
            entities: context_entities,
            relationships: all_relationships,
            relationship_explanations,
            query_metadata: QueryMetadata {
                query_time_ms: query_time.as_millis() as u64,
                entities_examined: visited_entities.len(),
                cache_hit: false,
                expansion_strategies: vec![
                    "vector_similarity".to_string(),
                    "neighbor_expansion".to_string(),
                    "bridge_detection".to_string(),
                ],
            },
        };
        
        // Would cache the result in mutable version
        
        Ok(result)
    }
    
    fn expand_entity_context(
        &self,
        entity_id: u32,
        max_depth: u8,
        visited: &mut HashSet<u32>,
        context_entities: &mut Vec<ContextEntity>,
        relationships: &mut Vec<Relationship>,
    ) -> Result<()> {
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((entity_id, 0u8));
        
        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth || visited.contains(&current_id) {
                continue;
            }
            
            visited.insert(current_id);
            let neighbors = self.graph.get_neighbors(current_id)?;
            
            for &neighbor_id in &neighbors {
                if depth + 1 < max_depth && !visited.contains(&neighbor_id) {
                    queue.push_back((neighbor_id, depth + 1));
                }
                
                relationships.push(Relationship {
                    from: current_id,
                    to: neighbor_id,
                    rel_type: 0,
                    weight: 1.0 / (depth as f32 + 1.0), // Decay weight by distance
                });
            }
        }
        
        Ok(())
    }
    
    fn find_bridge_entities(&self, similar_entities: &[(u32, f32)], max_depth: u8) -> Result<Vec<u32>> {
        let mut bridge_scores: HashMap<u32, u32> = HashMap::new();
        let entity_ids: Vec<u32> = similar_entities.iter().map(|(id, _)| *id).collect();
        
        // Find entities that appear in paths between multiple similar entities
        for i in 0..entity_ids.len() {
            for j in (i + 1)..entity_ids.len() {
                if let Ok(Some(path)) = self.graph.find_path(entity_ids[i], entity_ids[j], max_depth) {
                    for &path_entity in &path {
                        if !entity_ids.contains(&path_entity) {
                            *bridge_scores.entry(path_entity).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        
        // Return top bridge entities
        let mut bridges: Vec<(u32, u32)> = bridge_scores.into_iter().collect();
        bridges.sort_by(|a, b| b.1.cmp(&a.1));
        
        Ok(bridges.into_iter().take(10).map(|(id, _)| id).collect())
    }
    
    fn rank_context_entities(&self, entities: &mut Vec<ContextEntity>, query_embedding: &[f32]) -> Result<()> {
        // Enhanced ranking considering multiple factors
        entities.sort_by(|a, b| {
            let score_a = self.calculate_entity_score(a, query_embedding);
            let score_b = self.calculate_entity_score(b, query_embedding);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(())
    }
    
    fn calculate_entity_score(&self, entity: &ContextEntity, _query_embedding: &[f32]) -> f32 {
        // Multi-factor scoring
        let similarity_score = entity.similarity;
        let connectivity_score = (entity.neighbors.len() as f32).log10().min(1.0);
        let recency_score = 1.0; // Would be based on entity timestamps in production
        
        // Weighted combination
        0.6 * similarity_score + 0.3 * connectivity_score + 0.1 * recency_score
    }
    
    fn generate_relationship_explanations(&self, relationships: &[Relationship], _entities: &[ContextEntity]) -> Result<Vec<RelationshipExplanation>> {
        let mut explanations = Vec::new();
        
        for rel in relationships {
            let explanation = RelationshipExplanation {
                from_entity: rel.from,
                to_entity: rel.to,
                relationship_type: rel.rel_type,
                explanation: format!(
                    "Entity {} is connected to entity {} with relationship type {} (strength: {:.2})",
                    rel.from, rel.to, rel.rel_type, rel.weight
                ),
                confidence: rel.weight,
                evidence: vec![
                    format!("Direct graph connection with weight {:.2}", rel.weight),
                ],
            };
            explanations.push(explanation);
        }
        
        Ok(explanations)
    }
    
    fn create_cache_key(&self, query_embedding: &[f32], max_entities: usize, max_depth: u8) -> Vec<u8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &val in query_embedding {
            val.to_bits().hash(&mut hasher);
        }
        max_entities.hash(&mut hasher);
        max_depth.hash(&mut hasher);
        
        hasher.finish().to_le_bytes().to_vec()
    }
    
    fn cache_result(&mut self, key: Vec<u8>, context: &GraphRAGContext) {
        if self.cache.len() >= self.cache_size_limit {
            // Simple LRU eviction
            let oldest_key = self.cache.iter()
                .min_by_key(|(_, cached)| cached.timestamp)
                .map(|(k, _)| k.clone());
            
            if let Some(key_to_remove) = oldest_key {
                self.cache.remove(&key_to_remove);
            }
        }
        
        let query_result = QueryResult {
            entities: context.entities.clone(),
            relationships: context.relationships.clone(),
            confidence: 0.95,
            query_time_ms: context.query_metadata.query_time_ms,
        };
        
        self.cache.insert(key, CachedResult {
            result: query_result,
            timestamp: Instant::now(),
            access_count: 1,
        });
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.cache_size_limit,
            hit_rate: 0.0, // Would track this in production
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphRAGContext {
    pub entities: Vec<ContextEntity>,
    pub relationships: Vec<Relationship>,
    pub relationship_explanations: Vec<RelationshipExplanation>,
    pub query_metadata: QueryMetadata,
}

impl GraphRAGContext {
    fn from_query_result(result: &QueryResult) -> Self {
        Self {
            entities: result.entities.clone(),
            relationships: result.relationships.clone(),
            relationship_explanations: Vec::new(),
            query_metadata: QueryMetadata {
                query_time_ms: result.query_time_ms,
                entities_examined: result.entities.len(),
                cache_hit: true,
                expansion_strategies: vec!["cached".to_string()],
            },
        }
    }
    
    pub fn to_llm_context(&self) -> String {
        let mut context = String::new();
        
        context.push_str("# Knowledge Graph Context\n\n");
        
        // Entities section
        context.push_str("## Entities\n");
        for entity in &self.entities {
            context.push_str(&format!(
                "- Entity {}: {} (similarity: {:.3})\n",
                entity.id,
                entity.properties,
                entity.similarity
            ));
        }
        
        // Relationships section
        context.push_str("\n## Relationships\n");
        for rel in &self.relationships {
            context.push_str(&format!(
                "- {} → {} (type: {}, weight: {:.2})\n",
                rel.from, rel.to, rel.rel_type, rel.weight
            ));
        }
        
        // Explanations section
        if !self.relationship_explanations.is_empty() {
            context.push_str("\n## Relationship Explanations\n");
            for explanation in &self.relationship_explanations {
                context.push_str(&format!("- {}\n", explanation.explanation));
            }
        }
        
        context
    }
}

#[derive(Debug, Clone)]
pub struct RelationshipExplanation {
    pub from_entity: u32,
    pub to_entity: u32,
    pub relationship_type: u8,
    pub explanation: String,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QueryMetadata {
    pub query_time_ms: u64,
    pub entities_examined: usize,
    pub cache_hit: bool,
    pub expansion_strategies: Vec<String>,
}

#[derive(Debug)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f32,
}