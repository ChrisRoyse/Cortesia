/// Advanced pattern detection for abstract thinking
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use ahash::AHashMap;
use tokio::sync::RwLock;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::RelationType;
use crate::core::types::EntityKey;
// Neural server dependency removed - using pure graph operations
use crate::error::Result;

/// Neural pattern detector for identifying structural and semantic patterns
pub struct NeuralPatternDetector {
    graph: Arc<BrainEnhancedKnowledgeGraph>,
    pattern_cache: Arc<RwLock<HashMap<String, Vec<DetectedPattern>>>>,
    minimum_pattern_frequency: f32,
}

impl NeuralPatternDetector {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            minimum_pattern_frequency: 0.1,
        }
    }
    
    /// Detect patterns in the knowledge graph
    pub async fn detect_patterns(
        &self,
        analysis_scope: AnalysisScope,
        pattern_type: PatternType,
    ) -> Result<Vec<DetectedPattern>> {
        match pattern_type {
            PatternType::Structural => self.detect_structural_patterns(analysis_scope).await,
            PatternType::Temporal => self.detect_temporal_patterns(analysis_scope).await,
            PatternType::Semantic => self.detect_semantic_patterns(analysis_scope).await,
            PatternType::Usage => self.detect_usage_patterns(analysis_scope).await,
        }
    }
    
    /// Detect structural patterns in the graph topology
    async fn detect_structural_patterns(&self, scope: AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Get entities to analyze
        let entity_keys = self.get_entities_for_scope(scope).await?;
        let _all_entities = self.graph.get_all_entities().await;
        
        // Build relationships map from neighbors
        let mut relationships = AHashMap::new();
        for key in &entity_keys {
            let neighbors = self.graph.get_neighbors_with_weights(*key).await;
            for (neighbor, weight) in neighbors {
                // Create a BrainInspiredRelationship with the weight and default values
                let relationship = crate::core::brain_types::BrainInspiredRelationship {
                    source: *key,
                    target: neighbor,
                    source_key: *key,
                    target_key: neighbor,
                    relation_type: RelationType::RelatedTo, // Default to general relationship
                    weight,
                    strength: weight,
                    is_inhibitory: false,
                    temporal_decay: 0.1,
                    last_strengthened: std::time::SystemTime::now(),
                    last_update: std::time::SystemTime::now(),
                    activation_count: 0,
                    usage_count: 0,
                    creation_time: std::time::SystemTime::now(),
                    ingestion_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                };
                relationships.insert((*key, neighbor), relationship);
            }
        }
        
        // 1. Detect hub patterns (entities with many connections)
        let hub_pattern = self.detect_hub_pattern(&entity_keys, &relationships).await?;
        if let Some(pattern) = hub_pattern {
            patterns.push(pattern);
        }
        
        // 2. Detect chain patterns (linear sequences)
        let chain_patterns = self.detect_chain_patterns(&entity_keys, &relationships).await?;
        patterns.extend(chain_patterns);
        
        // 3. Detect cluster patterns (densely connected groups)
        let cluster_patterns = self.detect_cluster_patterns(&entity_keys, &relationships).await?;
        patterns.extend(cluster_patterns);
        
        // 4. Detect hierarchy patterns (tree-like structures)
        let hierarchy_patterns = self.detect_hierarchy_patterns(&entity_keys, &relationships).await?;
        patterns.extend(hierarchy_patterns);
        
        Ok(patterns)
    }
    
    /// Detect temporal patterns in knowledge evolution
    async fn detect_temporal_patterns(&self, scope: AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Get entities to analyze
        let entity_keys = self.get_entities_for_scope(scope).await?;
        let all_entities = self.graph.get_all_entities().await;
        
        // Analyze creation time patterns
        let mut creation_times = Vec::new();
        for (key, entity_data, _) in &all_entities {
            if entity_keys.contains(key) {
                // Use entity type_id as a proxy for creation order (entities with same type created together)
                creation_times.push((entity_data.type_id, *key));
            }
        }
        
        // Sort by type_id (creation order proxy)
        creation_times.sort_by_key(|&(type_id, _)| type_id);
        
        // Detect burst patterns (many entities created close together)
        let mut burst_count = 0;
        let mut current_burst = Vec::new();
        let mut last_version = 0;
        
        for (version, key) in creation_times {
            if version - last_version <= 2 {
                // Close together in version numbers
                current_burst.push(key);
            } else if !current_burst.is_empty() {
                // End of burst
                if current_burst.len() >= 3 {
                    burst_count += 1;
                    patterns.push(DetectedPattern {
                        pattern_id: format!("temporal_burst_{}", burst_count),
                        id: format!("temporal_burst_{}", burst_count),
                        pattern_type: PatternType::Temporal,
                        confidence: 0.7 + (current_burst.len() as f32 * 0.05).min(0.2),
                        entities_involved: current_burst.clone(),
                        affected_entities: current_burst.clone(),
                        description: format!("Temporal burst pattern with {} entities", current_burst.len()),
                        frequency: current_burst.len() as f32,
                    });
                }
                current_burst.clear();
                current_burst.push(key);
            }
            last_version = version;
        }
        
        // Check final burst
        if current_burst.len() >= 3 {
            burst_count += 1;
            patterns.push(DetectedPattern {
                pattern_id: format!("temporal_burst_{}", burst_count),
                id: format!("temporal_burst_{}", burst_count),
                pattern_type: PatternType::Temporal,
                confidence: 0.7 + (current_burst.len() as f32 * 0.05).min(0.2),
                entities_involved: current_burst.clone(),
                affected_entities: current_burst.clone(),
                description: format!("Temporal burst pattern with {} entities", current_burst.len()),
                frequency: current_burst.len() as f32,
            });
        }
        
        // Detect growth pattern
        if entity_keys.len() > 10 {
            patterns.push(DetectedPattern {
                pattern_id: "temporal_growth".to_string(),
                id: "temporal_growth".to_string(),
                pattern_type: PatternType::Temporal,
                confidence: 0.8,
                entities_involved: entity_keys.clone(),
                affected_entities: entity_keys,
                description: "Knowledge graph showing temporal growth patterns".to_string(),
                frequency: 0.7,
            });
        }
        
        Ok(patterns)
    }
    
    /// Detect semantic patterns in concepts and relationships
    async fn detect_semantic_patterns(&self, scope: AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        let entity_keys = self.get_entities_for_scope(scope).await?;
        let all_entities = self.graph.get_all_entities().await;
        
        // Group entities by semantic similarity
        let semantic_groups = self.group_entities_by_semantics(&entity_keys, &all_entities).await?;
        
        for (category, group_entities) in semantic_groups {
            if group_entities.len() >= 3 {
                patterns.push(DetectedPattern {
                    pattern_id: format!("semantic_cluster_{}", category),
                    id: format!("semantic_cluster_{}", category),
                    pattern_type: PatternType::Semantic,
                    confidence: 0.6 + (group_entities.len() as f32 * 0.05),
                    entities_involved: group_entities.clone(),
                    affected_entities: group_entities,
                    description: format!("Semantic cluster of {} concepts", category),
                    frequency: 0.5,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Detect usage patterns in entity access
    async fn detect_usage_patterns(&self, scope: AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Get entities to analyze
        let entity_keys = self.get_entities_for_scope(scope).await?;
        let all_entities = self.graph.get_all_entities().await;
        
        // Analyze activation levels as proxy for usage
        let mut high_usage_entities = Vec::new();
        let mut low_usage_entities = Vec::new();
        let mut activation_sum = 0.0;
        let mut count = 0;
        
        for (key, _, activation) in &all_entities {
            if entity_keys.contains(key) {
                activation_sum += activation;
                count += 1;
            }
        }
        
        let avg_activation = if count > 0 { activation_sum / count as f32 } else { 0.0 };
        
        // Categorize entities by usage (activation level)
        for (key, _, activation) in &all_entities {
            if entity_keys.contains(key) {
                if *activation > avg_activation * 1.5 {
                    high_usage_entities.push(*key);
                } else if *activation < avg_activation * 0.5 {
                    low_usage_entities.push(*key);
                }
            }
        }
        
        // High usage pattern
        if !high_usage_entities.is_empty() {
            patterns.push(DetectedPattern {
                pattern_id: "high_usage_entities".to_string(),
                id: "high_usage_entities".to_string(),
                pattern_type: PatternType::Usage,
                confidence: 0.8,
                entities_involved: high_usage_entities.clone(),
                affected_entities: high_usage_entities.clone(),
                description: format!("Entities with high usage frequency ({} entities)", high_usage_entities.len()),
                frequency: high_usage_entities.len() as f32 / entity_keys.len().max(1) as f32,
            });
        }
        
        // Low usage pattern (potential for removal)
        if !low_usage_entities.is_empty() && low_usage_entities.len() >= 5 {
            patterns.push(DetectedPattern {
                pattern_id: "low_usage_entities".to_string(),
                id: "low_usage_entities".to_string(),
                pattern_type: PatternType::Usage,
                confidence: 0.7,
                entities_involved: low_usage_entities.clone(),
                affected_entities: low_usage_entities.clone(),
                description: format!("Entities with low usage frequency ({} entities)", low_usage_entities.len()),
                frequency: low_usage_entities.len() as f32 / entity_keys.len().max(1) as f32,
            });
        }
        
        // Access pattern based on connectivity
        let mut connection_counts = HashMap::new();
        for &key in &entity_keys {
            let neighbors = self.graph.get_neighbors_with_weights(key).await;
            connection_counts.insert(key, neighbors.len());
        }
        
        // Find hub entities (frequently accessed)
        let avg_connections = connection_counts.values().sum::<usize>() as f32 / connection_counts.len().max(1) as f32;
        let hub_entities: Vec<EntityKey> = connection_counts.iter()
            .filter(|(_, &count)| count as f32 > avg_connections * 2.0)
            .map(|(&key, _)| key)
            .collect();
        
        if !hub_entities.is_empty() {
            patterns.push(DetectedPattern {
                pattern_id: "access_hub_pattern".to_string(),
                id: "access_hub_pattern".to_string(),
                pattern_type: PatternType::Usage,
                confidence: 0.75,
                entities_involved: hub_entities.clone(),
                affected_entities: hub_entities,
                description: "Hub entities frequently accessed in queries".to_string(),
                frequency: 0.8,
            });
        }
        
        Ok(patterns)
    }
    
    /// Detect hub patterns (highly connected entities)
    async fn detect_hub_pattern(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Option<DetectedPattern>> {
        let mut connection_counts = HashMap::new();
        
        // Count connections for each entity
        for relationship in relationships.values() {
            *connection_counts.entry(relationship.source).or_insert(0) += 1;
            *connection_counts.entry(relationship.target).or_insert(0) += 1;
        }
        
        // Find entities with significantly high connection counts
        let average_connections = if entity_keys.is_empty() {
            0.0
        } else {
            connection_counts.values().sum::<usize>() as f32 / entity_keys.len() as f32
        };
        
        let hub_entities: Vec<EntityKey> = entity_keys.iter()
            .filter(|&&key| {
                let count = connection_counts.get(&key).unwrap_or(&0);
                *count as f32 > average_connections * 2.0 && *count > 5
            })
            .cloned()
            .collect();
        
        if !hub_entities.is_empty() {
            Ok(Some(DetectedPattern {
                pattern_id: "hub_pattern".to_string(),
                id: "hub_pattern".to_string(),
                pattern_type: PatternType::Structural,
                confidence: 0.8,
                entities_involved: hub_entities.clone(),
                affected_entities: hub_entities,
                description: "Hub entities with high connectivity".to_string(),
                frequency: 0.6,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Detect chain patterns (linear sequences)
    async fn detect_chain_patterns(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        let mut visited = HashSet::new();
        
        for &start_entity in entity_keys {
            if visited.contains(&start_entity) {
                continue;
            }
            
            let chain = self.find_chain_from_entity(start_entity, relationships).await?;
            if chain.len() >= 4 {  // Minimum chain length
                for &entity in &chain {
                    visited.insert(entity);
                }
                
                patterns.push(DetectedPattern {
                    pattern_id: format!("chain_{}", chain.len()),
                    id: format!("chain_{}", chain.len()),
                    pattern_type: PatternType::Structural,
                    confidence: 0.7,
                    entities_involved: chain.clone(),
                    affected_entities: chain,
                    description: "Linear chain pattern in knowledge structure".to_string(),
                    frequency: 0.4,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Detect cluster patterns (densely connected groups)
    async fn detect_cluster_patterns(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Simple clustering based on relationship density
        let clusters = self.find_dense_clusters(entity_keys, relationships).await?;
        
        for (i, cluster) in clusters.into_iter().enumerate() {
            if cluster.len() >= 3 {
                patterns.push(DetectedPattern {
                    pattern_id: format!("cluster_{}", i),
                    id: format!("cluster_{}", i),
                    pattern_type: PatternType::Structural,
                    confidence: 0.6,
                    entities_involved: cluster.clone(),
                    affected_entities: cluster,
                    description: "Dense cluster of interconnected entities".to_string(),
                    frequency: 0.5,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Detect hierarchy patterns (tree-like structures)
    async fn detect_hierarchy_patterns(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Find hierarchical structures based on IsA relationships
        let hierarchies = self.find_hierarchical_structures(entity_keys, relationships).await?;
        
        for (i, hierarchy) in hierarchies.into_iter().enumerate() {
            if hierarchy.len() >= 3 {
                patterns.push(DetectedPattern {
                    pattern_id: format!("is_a_hierarchy_{}", i),
                    id: format!("is_a_hierarchy_{}", i),
                    pattern_type: PatternType::Structural,
                    confidence: 0.8,
                    entities_involved: hierarchy.clone(),
                    affected_entities: hierarchy,
                    description: "Hierarchical classification structure using IsA relationships".to_string(),
                    frequency: 0.7,
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Get entities for the specified analysis scope
    async fn get_entities_for_scope(&self, scope: AnalysisScope) -> Result<Vec<EntityKey>> {
        match scope {
            AnalysisScope::Local(entity_key) => Ok(vec![entity_key]),
            AnalysisScope::Regional(entity_keys) => Ok(entity_keys),
            AnalysisScope::Global => {
                let all_entities = self.graph.get_all_entities().await;
                Ok(all_entities.iter().map(|(key, _, _)| *key).collect())
            }
        }
    }
    
    /// Group entities by semantic categories
    async fn group_entities_by_semantics(
        &self,
        entity_keys: &[EntityKey],
        entities: &Vec<(EntityKey, crate::core::types::EntityData, f32)>,
    ) -> Result<HashMap<String, Vec<EntityKey>>> {
        let mut groups = HashMap::new();
        
        for &entity_key in entity_keys {
            if let Some((_, entity_data, _)) = entities.iter().find(|(k, _, _)| k == &entity_key) {
                let category = self.extract_semantic_category(&entity_data.properties);
                groups.entry(category).or_insert(Vec::new()).push(entity_key);
            }
        }
        
        Ok(groups)
    }
    
    /// Extract semantic category from concept ID
    fn extract_semantic_category(&self, concept_id: &str) -> String {
        let concept_lower = concept_id.to_lowercase();
        
        // Simple heuristic categorization
        if concept_lower.contains("animal") || concept_lower.contains("dog") || concept_lower.contains("cat") || concept_lower.contains("mammal") {
            "animals".to_string()
        } else if concept_lower.contains("color") || concept_lower.contains("red") || concept_lower.contains("blue") || concept_lower.contains("green") {
            "colors".to_string()
        } else if concept_lower.contains("number") || concept_lower.chars().all(|c| c.is_ascii_digit()) {
            "numbers".to_string()
        } else if concept_lower.contains("food") || concept_lower.contains("eat") || concept_lower.contains("fruit") {
            "food".to_string()
        } else {
            "general".to_string()
        }
    }
    
    /// Find a linear chain starting from an entity
    async fn find_chain_from_entity(
        &self,
        start_entity: EntityKey,
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<EntityKey>> {
        let mut chain = vec![start_entity];
        let mut current = start_entity;
        let mut visited = HashSet::new();
        visited.insert(current);
        
        // Follow the chain forward
        loop {
            let next_entity;
            let mut candidates = Vec::new();
            
            for relationship in relationships.values() {
                if relationship.source == current && !visited.contains(&relationship.target) {
                    candidates.push(relationship.target);
                }
            }
            
            // If exactly one candidate, continue the chain
            if candidates.len() == 1 {
                next_entity = Some(candidates[0]);
            } else {
                break;
            }
            
            if let Some(next) = next_entity {
                chain.push(next);
                visited.insert(next);
                current = next;
            } else {
                break;
            }
        }
        
        Ok(chain)
    }
    
    /// Find dense clusters using simple connectivity analysis
    async fn find_dense_clusters(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<Vec<EntityKey>>> {
        let mut clusters = Vec::new();
        let mut visited = HashSet::new();
        
        for &entity_key in entity_keys {
            if visited.contains(&entity_key) {
                continue;
            }
            
            let cluster = self.find_connected_component(entity_key, relationships, &mut visited).await?;
            if cluster.len() >= 3 {
                clusters.push(cluster);
            }
        }
        
        Ok(clusters)
    }
    
    /// Find hierarchical structures using IsA relationships
    async fn find_hierarchical_structures(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<Vec<EntityKey>>> {
        let mut hierarchies = Vec::new();
        let mut visited = HashSet::new();
        
        // Find root entities (entities with no IsA parents)
        let roots = self.find_hierarchy_roots(entity_keys, relationships).await?;
        
        for root in roots {
            if visited.contains(&root) {
                continue;
            }
            
            let hierarchy = self.traverse_hierarchy(root, relationships, &mut visited).await?;
            if hierarchy.len() >= 3 {
                hierarchies.push(hierarchy);
            }
        }
        
        Ok(hierarchies)
    }
    
    /// Find connected component starting from an entity
    async fn find_connected_component(
        &self,
        start_entity: EntityKey,
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
        visited: &mut HashSet<EntityKey>,
    ) -> Result<Vec<EntityKey>> {
        let mut component = Vec::new();
        let mut stack = vec![start_entity];
        
        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            
            visited.insert(current);
            component.push(current);
            
            // Add neighbors to stack
            for relationship in relationships.values() {
                if relationship.source == current && !visited.contains(&relationship.target) {
                    stack.push(relationship.target);
                } else if relationship.target == current && !visited.contains(&relationship.source) {
                    stack.push(relationship.source);
                }
            }
        }
        
        Ok(component)
    }
    
    /// Find root entities in hierarchies
    async fn find_hierarchy_roots(
        &self,
        entity_keys: &[EntityKey],
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
    ) -> Result<Vec<EntityKey>> {
        let mut has_parent = HashSet::new();
        
        // Mark entities that have IsA parents
        for relationship in relationships.values() {
            if relationship.relation_type == RelationType::IsA {
                has_parent.insert(relationship.source);
            }
        }
        
        // Return entities without IsA parents
        Ok(entity_keys.iter()
            .filter(|&&key| !has_parent.contains(&key))
            .cloned()
            .collect())
    }
    
    /// Traverse hierarchy depth-first
    async fn traverse_hierarchy(
        &self,
        root: EntityKey,
        relationships: &ahash::AHashMap<(crate::core::types::EntityKey, crate::core::types::EntityKey), crate::core::brain_types::BrainInspiredRelationship>,
        visited: &mut HashSet<EntityKey>,
    ) -> Result<Vec<EntityKey>> {
        let mut hierarchy = Vec::new();
        let mut stack = vec![root];
        
        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            
            visited.insert(current);
            hierarchy.push(current);
            
            // Add children (entities that have IsA relationship to current)
            for relationship in relationships.values() {
                if relationship.relation_type == RelationType::IsA &&
                   relationship.target == current &&
                   !visited.contains(&relationship.source) {
                    stack.push(relationship.source);
                }
            }
        }
        
        Ok(hierarchy)
    }
}