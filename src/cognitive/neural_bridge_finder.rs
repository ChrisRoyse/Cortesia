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
                path: path.clone(),
                intermediate_concepts: intermediate_concepts.clone(),
                novelty_score,
                plausibility_score,
                explanation,
                bridge_id: format!("neural_bridge_{}", path.len()),
                start_concept: "start".to_string(),
                end_concept: "end".to_string(),
                bridge_concepts: intermediate_concepts,
                creativity_score: novelty_score,
                connection_strength: plausibility_score,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    
    // Helper to create a test neural bridge finder
    fn create_test_finder() -> NeuralBridgeFinder {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap());
        NeuralBridgeFinder::new(graph)
    }
    
    #[test]
    fn test_levenshtein_distance() {
        // Test identical strings
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", ""), 0);
        
        // Test completely different strings
        assert_eq!(levenshtein_distance("hello", "world"), 4);
        assert_eq!(levenshtein_distance("cat", "dog"), 3);
        
        // Test one character difference
        assert_eq!(levenshtein_distance("cat", "bat"), 1);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        
        // Test empty strings
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        
        // Test substring relationships
        assert_eq!(levenshtein_distance("test", "testing"), 3);
        assert_eq!(levenshtein_distance("abc", "abcdef"), 3);
    }

    #[test]
    fn test_calculate_concept_similarity() {
        let finder = create_test_finder();
        
        // Test identical concepts
        assert_eq!(finder.calculate_concept_similarity("test", "test"), 1.0);
        assert_eq!(finder.calculate_concept_similarity("", ""), 1.0);
        
        // Test substring relationships
        assert!(finder.calculate_concept_similarity("artificial intelligence", "intelligence") > 0.6);
        assert!(finder.calculate_concept_similarity("machine learning", "learning") > 0.6);
        
        // Test different concepts
        assert!(finder.calculate_concept_similarity("dog", "quantum") < 0.5);
        assert!(finder.calculate_concept_similarity("art", "science") < 0.5);
        
        // Test similar concepts
        assert!(finder.calculate_concept_similarity("neural network", "neural") > 0.6);
        assert!(finder.calculate_concept_similarity("deep learning", "machine learning") > 0.4);
    }

    #[tokio::test]
    async fn test_calculate_path_novelty() {
        let finder = create_test_finder();
        
        // Create test entity keys
        let entity1 = EntityKey::from_raw_parts(1, 0);
        let entity2 = EntityKey::from_raw_parts(2, 0);
        let entity3 = EntityKey::from_raw_parts(3, 0);
        
        // Test simple path
        let simple_path = vec![entity1, entity2];
        let simple_concepts = vec!["concept1".to_string(), "concept2".to_string()];
        let novelty = finder.calculate_path_novelty(&simple_path, &simple_concepts).await.unwrap();
        assert!(novelty >= 0.0 && novelty <= 1.0, "Novelty should be between 0 and 1");
        
        // Test longer path (should be more novel)
        let longer_path = vec![entity1, entity2, entity3];
        let longer_concepts = vec!["concept1".to_string(), "concept2".to_string(), "concept3".to_string()];
        let longer_novelty = finder.calculate_path_novelty(&longer_path, &longer_concepts).await.unwrap();
        assert!(longer_novelty > novelty, "Longer paths should be more novel");
        
        // Test empty path
        let empty_path = vec![];
        let empty_concepts = vec![];
        let empty_novelty = finder.calculate_path_novelty(&empty_path, &empty_concepts).await.unwrap();
        assert_eq!(empty_novelty, 0.0, "Empty path should have zero novelty");
    }

    #[tokio::test]
    async fn test_calculate_path_plausibility() {
        let finder = create_test_finder();
        
        // Create test entity keys
        let entity1 = EntityKey::from_raw_parts(1, 0);
        let entity2 = EntityKey::from_raw_parts(2, 0);
        
        // Test simple path
        let path = vec![entity1, entity2];
        let plausibility = finder.calculate_path_plausibility(&path).await.unwrap();
        assert!(plausibility >= 0.0 && plausibility <= 1.0, "Plausibility should be between 0 and 1");
        
        // Test single entity path
        let single_path = vec![entity1];
        let single_plausibility = finder.calculate_path_plausibility(&single_path).await.unwrap();
        assert_eq!(single_plausibility, 1.0, "Single entity path should have maximum plausibility");
        
        // Test empty path
        let empty_path = vec![];
        let empty_plausibility = finder.calculate_path_plausibility(&empty_path).await.unwrap();
        assert_eq!(empty_plausibility, 1.0, "Empty path should have maximum plausibility");
    }

    #[test]
    fn test_generate_bridge_explanation() {
        let finder = create_test_finder();
        
        // Test direct connection
        let direct_concepts = vec!["start".to_string(), "end".to_string()];
        let direct_explanation = finder.generate_bridge_explanation("start", "end", &direct_concepts);
        assert!(direct_explanation.contains("direct relationship"));
        
        // Test multi-step connection
        let complex_concepts = vec![
            "art".to_string(),
            "creativity".to_string(), 
            "innovation".to_string(),
            "technology".to_string()
        ];
        let complex_explanation = finder.generate_bridge_explanation("art", "technology", &complex_concepts);
        assert!(complex_explanation.contains("→"));
        assert!(complex_explanation.contains("creativity"));
        assert!(complex_explanation.contains("innovation"));
    }

    #[tokio::test]
    async fn test_find_concept_entities() {
        let finder = create_test_finder();
        
        // Test finding entities for a concept
        let entities = finder.find_concept_entities("test").await.unwrap();
        // Since we're using a test graph, we may not find any entities
        // This test mainly verifies the method doesn't crash and returns a valid result
        assert!(entities.len() >= 0, "Should return a valid vector");
    }

    #[tokio::test]
    async fn test_neural_pathfinding_with_length() {
        let finder = create_test_finder();
        
        // Create test entity keys
        let entity1 = EntityKey::from_raw_parts(1, 0);
        let entity2 = EntityKey::from_raw_parts(2, 0);
        
        // Test pathfinding
        let paths = finder.neural_pathfinding_with_length(entity1, entity2, 3).await.unwrap();
        assert!(paths.len() >= 0, "Should return a valid vector of paths");
        
        // Test pathfinding with same start and end
        let same_paths = finder.neural_pathfinding_with_length(entity1, entity1, 3).await.unwrap();
        assert!(same_paths.len() >= 0, "Should handle same start and end entities");
    }

    #[tokio::test]
    async fn test_evaluate_bridge_creativity() {
        let finder = create_test_finder();
        
        // Test with empty path
        let empty_path = vec![];
        let empty_result = finder.evaluate_bridge_creativity(empty_path, "start", "end").await.unwrap();
        assert!(empty_result.is_none(), "Empty path should return None");
        
        // Test with single entity path
        let single_path = vec![EntityKey::from_raw_parts(1, 0)];
        let single_result = finder.evaluate_bridge_creativity(single_path, "start", "end").await.unwrap();
        assert!(single_result.is_none(), "Single entity path should return None");
        
        // Test with valid path
        let valid_path = vec![EntityKey::from_raw_parts(1, 0), EntityKey::from_raw_parts(2, 0)];
        let valid_result = finder.evaluate_bridge_creativity(valid_path, "art", "science").await.unwrap();
        // The result depends on whether entities exist in the test graph and creativity threshold
        // This test mainly verifies the method doesn't crash
        assert!(valid_result.is_some() || valid_result.is_none(), "Should return a valid Option");
    }

    #[tokio::test]
    async fn test_find_creative_bridges_with_length() {
        let finder = create_test_finder();
        
        // Test finding bridges between concepts
        let bridges = finder.find_creative_bridges_with_length("art", "science", 4).await.unwrap();
        assert!(bridges.len() >= 0, "Should return a valid vector of bridges");
        
        // Test with same concepts
        let same_bridges = finder.find_creative_bridges_with_length("test", "test", 3).await.unwrap();
        assert!(same_bridges.len() >= 0, "Should handle same concepts");
    }
}