/// Advanced neural bridge finding for lateral thinking
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::types::EntityKey;
use crate::core::brain_types::RelationType;
// Neural server dependency removed - using pure graph operations
use crate::error::Result;

/// Neural bridge finder for creative connections between disparate concepts
pub struct NeuralBridgeFinder {
    graph: Arc<BrainEnhancedKnowledgeGraph>,
    max_bridge_length: usize,
    creativity_threshold: f32,
    embedding_cache: Arc<RwLock<HashMap<EntityKey, Vec<f32>>>>,
}

impl NeuralBridgeFinder {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            max_bridge_length: 5,
            creativity_threshold: 0.3,
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Find creative bridges between two concepts using neural pathfinding
    pub async fn find_creative_bridges(
        &self,
        concept_a: &str,
        concept_b: &str,
    ) -> Result<Vec<BridgePath>> {
        self.find_creative_bridges_with_length(concept_a, concept_b, self.max_bridge_length).await
    }
    
    /// Find creative bridges between two concepts using neural pathfinding with custom length limit
    pub async fn find_creative_bridges_with_length(
        &self,
        concept_a: &str,
        concept_b: &str,
        max_length: usize,
    ) -> Result<Vec<BridgePath>> {
        // 1. Find entities for both concepts
        let entities_a = self.find_concept_entities(concept_a).await?;
        let entities_b = self.find_concept_entities(concept_b).await?;
        
        if entities_a.is_empty() || entities_b.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut bridges = Vec::new();
        
        // 2. For each pair of entities, find potential bridges
        for &entity_a in &entities_a {
            for &entity_b in &entities_b {
                let paths = self.neural_pathfinding_with_length(entity_a, entity_b, max_length).await?;
                for path in paths {
                    if let Some(bridge) = self.evaluate_bridge_creativity(path, concept_a, concept_b).await? {
                        bridges.push(bridge);
                    }
                }
            }
        }
        
        // 3. Sort by novelty and return top bridges
        bridges.sort_by(|a, b| b.novelty_score.partial_cmp(&a.novelty_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(bridges.into_iter().take(10).collect())
    }
    
    /// Neural pathfinding using breadth-first search with embedding similarity
    async fn neural_pathfinding(
        &self,
        start: EntityKey,
        end: EntityKey,
    ) -> Result<Vec<Vec<EntityKey>>> {
        self.neural_pathfinding_with_length(start, end, self.max_bridge_length).await
    }
    
    /// Neural pathfinding using breadth-first search with embedding similarity with custom length limit
    async fn neural_pathfinding_with_length(
        &self,
        start: EntityKey,
        end: EntityKey,
        max_length: usize,
    ) -> Result<Vec<Vec<EntityKey>>> {
        let mut paths = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = HashMap::new();
        
        // Initialize with start entity
        queue.push_back(vec![start]);
        visited.insert(start, 0);
        
        while let Some(current_path) = queue.pop_front() {
            let current_entity = *current_path.last().unwrap();
            
            // If we reached the target, add this path
            if current_entity == end {
                paths.push(current_path.clone());
                continue;
            }
            
            // If path is too long, skip
            if current_path.len() >= max_length {
                continue;
            }
            
            // Find neighboring entities
            let neighbors = self.graph.get_neighbors_with_weights(current_entity).await;
            for (neighbor, _weight) in neighbors {
                // Avoid cycles
                if current_path.contains(&neighbor) {
                    continue;
                }
                
                // Check if we've seen this entity at a shorter or equal distance
                let distance = current_path.len();
                if let Some(&prev_distance) = visited.get(&neighbor) {
                    if distance > prev_distance {
                        continue;
                    }
                }
                visited.insert(neighbor, distance);
                
                // Add to queue
                let mut new_path = current_path.clone();
                new_path.push(neighbor);
                queue.push_back(new_path);
            }
        }
        
        Ok(paths)
    }
    
    /// Evaluate the creativity and plausibility of a bridge path
    async fn evaluate_bridge_creativity(
        &self,
        path: Vec<EntityKey>,
        concept_a: &str,
        concept_b: &str,
    ) -> Result<Option<BridgePath>> {
        if path.len() < 2 {
            return Ok(None);
        }
        
        let all_entities = self.graph.get_all_entities().await;
        
        // Get intermediate concepts
        let mut intermediate_concepts = Vec::new();
        let mut concept_names = Vec::new();
        
        for &entity_key in &path {
            if let Some((_, _entity_data, _)) = all_entities.iter().find(|(k, _, _)| k == &entity_key) {
                let concept_id = format!("entity_{:?}", entity_key);
                intermediate_concepts.push(concept_id.clone());
                concept_names.push(concept_id);
            }
        }
        
        if intermediate_concepts.len() < 2 {
            return Ok(None);
        }
        
        // Calculate novelty score based on path diversity and length
        let novelty_score = self.calculate_path_novelty(&path, &intermediate_concepts).await?;
        let plausibility_score = self.calculate_path_plausibility(&path).await?;
        
        // Only return bridges above creativity threshold
        if novelty_score >= self.creativity_threshold {
            let explanation = self.generate_bridge_explanation(
                concept_a,
                concept_b,
                &intermediate_concepts,
            );
            
            Ok(Some(BridgePath {
                path,
                intermediate_concepts,
                novelty_score,
                plausibility_score,
                explanation,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate novelty score for a path
    async fn calculate_path_novelty(
        &self,
        path: &[EntityKey],
        concepts: &[String],
    ) -> Result<f32> {
        // Base novelty on path length (longer = more novel)
        let length_novelty = (path.len() as f32 - 2.0) / (self.max_bridge_length as f32 - 2.0);
        
        // Calculate concept diversity
        let mut diversity_score = 0.0;
        for i in 0..concepts.len() - 1 {
            let similarity = self.calculate_concept_similarity(&concepts[i], &concepts[i + 1]);
            diversity_score += 1.0 - similarity;
        }
        diversity_score /= (concepts.len() - 1) as f32;
        
        // Combine scores
        Ok((length_novelty + diversity_score) / 2.0)
    }
    
    /// Calculate plausibility score for a path
    async fn calculate_path_plausibility(&self, path: &[EntityKey]) -> Result<f32> {
        let mut plausibility = 1.0;
        
        // Check relationship strengths along the path using neighbors
        for i in 0..path.len() - 1 {
            let neighbors = self.graph.get_neighbors_with_weights(path[i]).await;
            let mut found_relationship = false;
            let mut max_weight: f32 = 0.0;
            
            for (neighbor, weight) in neighbors {
                if neighbor == path[i + 1] {
                    found_relationship = true;
                    max_weight = max_weight.max(weight);
                }
            }
            
            if found_relationship {
                plausibility *= max_weight;
            } else {
                plausibility *= 0.1; // Penalty for missing direct relationship
            }
        }
        
        Ok(plausibility)
    }
    
    /// Generate human-readable explanation for a bridge
    fn generate_bridge_explanation(
        &self,
        concept_a: &str,
        concept_b: &str,
        intermediate_concepts: &[String],
    ) -> String {
        if intermediate_concepts.len() < 3 {
            format!("{} connects to {} through a direct relationship", concept_a, concept_b)
        } else {
            let middle_concepts = &intermediate_concepts[1..intermediate_concepts.len()-1];
            format!(
                "{} connects to {} through: {}",
                concept_a,
                concept_b,
                middle_concepts.join(" → ")
            )
        }
    }
    
    /// Find entities that represent a concept
    async fn find_concept_entities(&self, concept: &str) -> Result<Vec<EntityKey>> {
        let all_entities = self.graph.get_all_entities().await;
        let mut matches = Vec::new();
        
        let concept_norm = concept.to_lowercase();
        
        for (key, entity_data, _) in &all_entities {
            let entity_norm = entity_data.properties.to_lowercase();
            if entity_norm.contains(&concept_norm) || concept_norm.contains(&entity_norm) {
                matches.push(*key);
            }
        }
        
        Ok(matches)
    }
    
    /// Calculate similarity between two concepts
    fn calculate_concept_similarity(&self, concept_a: &str, concept_b: &str) -> f32 {
        let a_norm = concept_a.to_lowercase();
        let b_norm = concept_b.to_lowercase();
        
        // Simple string similarity for now
        if a_norm == b_norm {
            1.0
        } else if a_norm.contains(&b_norm) || b_norm.contains(&a_norm) {
            0.7
        } else {
            // Use Levenshtein-based similarity
            let max_len = a_norm.len().max(b_norm.len());
            if max_len == 0 { return 1.0; }
            
            let distance = levenshtein_distance(&a_norm, &b_norm);
            1.0 - (distance as f32 / max_len as f32)
        }
    }
}

/// Extension trait for RelationType to check directionality
trait RelationTypeExt {
    fn is_directional(&self) -> bool;
}

impl RelationTypeExt for RelationType {
    fn is_directional(&self) -> bool {
        matches!(self, RelationType::IsA | RelationType::HasInstance | RelationType::PartOf)
    }
}

/// Calculate Levenshtein distance between two strings
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[len1][len2]
}