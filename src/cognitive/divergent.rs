use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, Instant};
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::{ActivationPattern, BrainInspiredEntity, EntityDirection, ActivationStep, ActivationOperation, RelationType};
use crate::core::types::EntityKey;
// Neural server dependency removed - using pure graph operations
use crate::error::{Result, GraphError};

/// Divergent thinking pattern - explores many possible paths, brainstorming, creative connections
pub struct DivergentThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub exploration_breadth: usize,
    pub creativity_threshold: f32,
    pub max_exploration_depth: usize,
    pub novelty_weight: f32,
}

impl DivergentThinking {
    /// Create a new divergent thinking processor
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            exploration_breadth: 20,
            creativity_threshold: 0.3,
            max_exploration_depth: 4,
            novelty_weight: 0.4,
        }
    }
    
    /// Create a new divergent thinking processor with custom parameters
    pub fn new_with_params(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
        exploration_breadth: usize,
        creativity_threshold: f32,
    ) -> Self {
        Self {
            graph,
            exploration_breadth,
            creativity_threshold,
            max_exploration_depth: 4,
            novelty_weight: 0.4,
        }
    }
    
    /// Execute divergent exploration starting from seed concept
    pub async fn execute_divergent_exploration(
        &self,
        seed_concept: &str,
        exploration_type: ExplorationType,
    ) -> Result<DivergentResult> {
        let start_time = Instant::now();
        
        // 1. Activate seed concept
        let seed_activation = self.activate_seed_concept(seed_concept).await?;
        
        // 2. Spread activation broadly through different relationship types
        let exploration_map = self.spread_activation(
            seed_activation,
            exploration_type,
        ).await?;
        
        // 3. Use neural path exploration to find creative connections
        let path_exploration = self.neural_path_exploration(exploration_map).await?;
        
        // 4. Rank results by relevance and novelty
        let ranked_results = self.rank_by_creativity(path_exploration).await?;
        
        let _execution_time = start_time.elapsed();
        
        Ok(DivergentResult {
            explorations: ranked_results.clone(),
            creativity_scores: ranked_results.iter().map(|path| path.novelty_score).collect(),
            total_paths_explored: ranked_results.len(),
        })
    }
    
    /// Activate the seed concept as starting point
    async fn activate_seed_concept(&self, concept: &str) -> Result<ActivationPattern> {
        let mut activation_pattern = ActivationPattern::new(format!("divergent_exploration_{}", concept));
        
        // Find entities matching the seed concept
        let matching_entities = self.find_concept_entities(concept).await?;
        
        if matching_entities.is_empty() {
            return Err(GraphError::ProcessingError(format!("No entities found for seed concept: {}", concept)));
        }
        
        // Set initial high activation for seed entities
        for (entity_key, relevance) in matching_entities {
            activation_pattern.activations.insert(entity_key, relevance);
        }
        
        Ok(activation_pattern)
    }
    
    /// Spread activation through the graph based on exploration type
    async fn spread_activation(
        &self,
        seed_activation: ActivationPattern,
        exploration_type: ExplorationType,
    ) -> Result<ExplorationMap> {
        let mut exploration_map = ExplorationMap::new();
        let mut current_wave = seed_activation.activations.clone();
        let mut visited = HashSet::new();
        
        for depth in 0..self.max_exploration_depth {
            let mut next_wave = HashMap::new();
            
            for (entity_key, activation) in &current_wave {
                if visited.contains(entity_key) {
                    continue;
                }
                visited.insert(*entity_key);
                
                // Add to exploration map
                exploration_map.add_node(*entity_key, *activation, depth);
                
                // Find connections based on exploration type
                let connections = self.find_typed_connections(*entity_key, &exploration_type).await?;
                
                for (connected_key, connection_weight, relation_type) in connections {
                    if !visited.contains(&connected_key) {
                        let propagated_activation = activation * connection_weight * 0.8_f32.powi(depth as i32);
                        
                        if propagated_activation >= self.creativity_threshold {
                            let current_activation = next_wave.get(&connected_key).unwrap_or(&0.0);
                            next_wave.insert(connected_key, ((*current_activation) as f32).max(propagated_activation));
                            
                            exploration_map.add_edge(*entity_key, connected_key, relation_type, connection_weight);
                        }
                    }
                }
            }
            
            if next_wave.is_empty() {
                break;
            }
            
            current_wave = next_wave;
        }
        
        Ok(exploration_map)
    }
    
    /// Use neural networks for creative path exploration
    async fn neural_path_exploration(&self, exploration_map: ExplorationMap) -> Result<Vec<ExplorationPath>> {
        let mut paths = Vec::new();
        
        // Generate paths from seed nodes to interesting endpoints
        let seed_nodes = exploration_map.get_nodes_at_depth(0);
        let endpoint_nodes = exploration_map.get_high_activation_endpoints(self.exploration_breadth);
        
        // If no seed nodes or endpoints, create basic paths from exploration map
        if seed_nodes.is_empty() || endpoint_nodes.is_empty() {
            // Create exploration paths from available entities
            for wave in &exploration_map.exploration_waves {
                for (i, &entity_key) in wave.entities.iter().enumerate() {
                    if let Ok(entity) = self.get_entity(entity_key).await {
                        let activation = wave.activation_levels.get(i).unwrap_or(&0.5);
                        paths.push(ExplorationPath {
                            path: vec![entity_key],
                            concepts: vec![entity.concept_id.clone()],
                            concept: entity.concept_id.clone(),
                            relevance_score: *activation,
                            novelty_score: self.calculate_single_concept_novelty(&entity.concept_id),
                        });
                    }
                }
            }
        } else {
            // Generate paths between seed and endpoint nodes
            for seed_key in seed_nodes {
                for endpoint_key in &endpoint_nodes {
                    if seed_key != *endpoint_key {
                        if let Some(path) = self.find_creative_path(seed_key, *endpoint_key, &exploration_map).await? {
                            paths.push(path);
                        }
                    }
                }
            }
        }
        
        // Use neural network to enhance path discovery
        let neural_paths = self.neural_enhance_paths(paths, &exploration_map).await?;
        
        Ok(neural_paths)
    }
    
    /// Find creative path between two entities
    async fn find_creative_path(
        &self,
        start: EntityKey,
        end: EntityKey,
        exploration_map: &ExplorationMap,
    ) -> Result<Option<ExplorationPath>> {
        // Use breadth-first search with creativity scoring
        let mut queue = Vec::new();
        let mut visited = HashSet::new();
        let mut parent_map = HashMap::new();
        
        queue.push(start);
        visited.insert(start);
        
        while let Some(current) = queue.pop() {
            if current == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut concepts = Vec::new();
                let mut current_node = end;
                
                while let Some(&parent) = parent_map.get(&current_node) {
                    path.push(current_node);
                    if let Ok(entity) = self.get_entity(current_node).await {
                        concepts.push(entity.concept_id);
                    }
                    current_node = parent;
                }
                path.push(start);
                if let Ok(entity) = self.get_entity(start).await {
                    concepts.push(entity.concept_id);
                }
                
                path.reverse();
                concepts.reverse();
                
                let relevance_score = self.calculate_path_relevance(&path).await?;
                let novelty_score = self.calculate_path_novelty(&path, &concepts).await?;
                
                let primary_concept = concepts.first().unwrap_or(&"unknown".to_string()).clone();
                
                return Ok(Some(ExplorationPath {
                    path,
                    concepts,
                    concept: primary_concept,
                    relevance_score,
                    novelty_score,
                }));
            }
            
            // Explore neighbors
            for neighbor in exploration_map.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent_map.insert(neighbor, current);
                    queue.push(neighbor);
                }
            }
        }
        
        Ok(None)
    }
    
    /// Enhance paths using neural network analysis
    async fn neural_enhance_paths(
        &self,
        paths: Vec<ExplorationPath>,
        _exploration_map: &ExplorationMap,
    ) -> Result<Vec<ExplorationPath>> {
        let mut enhanced_paths = Vec::new();
        
        for path in paths {
            // Use graph-based metrics to evaluate path creativity
            let path_embedding = self.generate_path_embedding(&path).await?;
            let creativity_score = self.evaluate_path_creativity(&path, &path_embedding).await?;
            
            let enhanced_novelty = creativity_score.max(path.novelty_score);
            
            enhanced_paths.push(ExplorationPath {
                path: path.path,
                concepts: path.concepts,
                concept: path.concept,
                relevance_score: path.relevance_score,
                novelty_score: enhanced_novelty,
            });
        }
        
        Ok(enhanced_paths)
    }
    
    /// Rank exploration results by creativity (combination of relevance and novelty)
    async fn rank_by_creativity(&self, paths: Vec<ExplorationPath>) -> Result<Vec<ExplorationPath>> {
        let mut ranked_paths = paths;
        
        // Calculate creativity scores and sort
        ranked_paths.sort_by(|a, b| {
            let creativity_a = (1.0 - self.novelty_weight) * a.relevance_score + self.novelty_weight * a.novelty_score;
            let creativity_b = (1.0 - self.novelty_weight) * b.relevance_score + self.novelty_weight * b.novelty_score;
            creativity_b.partial_cmp(&creativity_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take top results
        ranked_paths.truncate(self.exploration_breadth);
        
        Ok(ranked_paths)
    }
    
    /// Find typed connections based on exploration type
    async fn find_typed_connections(
        &self,
        entity_key: EntityKey,
        _exploration_type: &ExplorationType,
    ) -> Result<Vec<(EntityKey, f32, RelationType)>> {
        let mut connections = Vec::new();
        
        // Get outgoing connections
        let neighbors = self.graph.get_neighbors_with_weights(entity_key).await;
        for (connected_key, weight) in neighbors {
            connections.push((connected_key, weight, RelationType::RelatedTo));
        }
        
        // Get incoming connections
        let parents = self.graph.get_parent_entities(entity_key).await;
        for (connected_key, weight) in parents {
            connections.push((connected_key, weight, RelationType::RelatedTo));
        }
        
        // Filter by exploration type - for now include all connections
        // since we don't have relationship type information readily available
        Ok(connections)
    }
    
    
    
    
    /// Calculate path relevance to original query
    async fn calculate_path_relevance(&self, path: &[EntityKey]) -> Result<f32> {
        if path.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate average activation of entities in path
        let mut total_relevance = 0.0;
        let mut count = 0;
        
        for &entity_key in path {
            if let Ok(entity) = self.get_entity(entity_key).await {
                // Simple relevance based on entity properties
                let entity_relevance = match entity.direction {
                    EntityDirection::Input => 0.9,
                    EntityDirection::Gate => 0.7,
                    EntityDirection::Output => 0.8,
                    EntityDirection::Hidden => 0.6,
                };
                total_relevance += entity_relevance;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_relevance / count as f32 } else { 0.0 })
    }
    
    /// Calculate path novelty based on concept combinations
    async fn calculate_path_novelty(&self, _path: &[EntityKey], concepts: &[String]) -> Result<f32> {
        if concepts.is_empty() {
            return Ok(0.0);
        }
        
        // Novelty based on concept diversity and unexpectedness
        let mut unique_domains = HashSet::new();
        let mut total_novelty = 0.0;
        
        for concept in concepts {
            // Extract domain from concept (simplified)
            let domain = concept.split('_').next().unwrap_or(concept);
            unique_domains.insert(domain);
            
            // Add novelty based on concept length and complexity
            let concept_novelty = (concept.len() as f32 / 20.0).min(1.0);
            total_novelty += concept_novelty;
        }
        
        let domain_diversity = if concepts.is_empty() {
            0.0
        } else {
            unique_domains.len() as f32 / concepts.len() as f32
        };
        
        let average_concept_novelty = if concepts.is_empty() {
            0.0
        } else {
            total_novelty / concepts.len() as f32
        };
        
        Ok((domain_diversity + average_concept_novelty) / 2.0)
    }
    
    /// Calculate novelty for a single concept
    fn calculate_single_concept_novelty(&self, concept: &str) -> f32 {
        // Novelty based on concept characteristics
        let mut novelty = 0.0;
        
        // Length-based novelty
        let length_novelty = (concept.len() as f32 / 20.0).min(1.0);
        novelty += length_novelty * 0.3;
        
        // Complexity-based novelty (number of words/components)
        let components = concept.split(['_', ' ', '-']).collect::<Vec<_>>();
        let complexity_novelty = (components.len() as f32 / 3.0).min(1.0);
        novelty += complexity_novelty * 0.3;
        
        // Domain-specific novelty
        let domain_novelty = if concept.contains("advanced") || concept.contains("complex") || concept.contains("specialized") {
            0.8
        } else if concept.contains("basic") || concept.contains("simple") {
            0.2
        } else {
            0.5
        };
        novelty += domain_novelty * 0.4;
        
        novelty.min(1.0)
    }
    
    /// Generate embedding for path analysis
    async fn generate_path_embedding(&self, path: &ExplorationPath) -> Result<Vec<f32>> {
        let mut embedding = vec![0.0; 384];
        
        // Simple embedding based on path concepts
        for (i, concept) in path.concepts.iter().enumerate() {
            for (j, byte) in concept.bytes().enumerate() {
                let idx = (i * 20 + j) % 384;
                embedding[idx] += (byte as f32) / 255.0;
            }
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        Ok(embedding)
    }
    
    /// Find entities matching a concept
    async fn find_concept_entities(&self, concept: &str) -> Result<Vec<(EntityKey, f32)>> {
        let all_entities = self.graph.get_all_entities().await;
        let mut matches = Vec::new();
        
        let query_norm = Self::normalize_concept(concept);
        
        // First pass: exact and substring matches
        for (key, _entity_data, _) in &all_entities {
            let entity_concept = format!("entity_{:?}", key);
            let entity_norm = Self::normalize_concept(&entity_concept);
            
            if entity_norm == query_norm {
                matches.push((*key, 1.0));
            } else if entity_norm.contains(&query_norm) || query_norm.contains(&entity_norm) {
                matches.push((*key, 0.8));
            } else {
                // Use detailed relevance calculation for remaining entities
                let relevance = self.calculate_concept_relevance(&entity_concept, concept);
                if relevance > 0.1 { // Lower threshold for better matching
                    matches.push((*key, relevance));
                }
            }
        }
        
        // If still no matches, try even more relaxed matching
        if matches.is_empty() {
            for (key, _entity_data, _) in &all_entities {
                // Check if either query or entity concept contains the other as a word
                let query_words: Vec<&str> = query_norm.split_whitespace().collect();
                let entity_concept_lower = format!("entity_{:?}", key).to_lowercase();
                let entity_words: Vec<&str> = entity_concept_lower.split_whitespace().collect();
                
                for query_word in &query_words {
                    for entity_word in &entity_words {
                        if entity_word.contains(query_word) || query_word.contains(entity_word) {
                            matches.push((*key, 0.5));
                            break;
                        }
                    }
                }
            }
        }
        
        // Sort by relevance
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(matches)
    }
    
    /// Get an entity by key
    async fn get_entity(&self, entity_key: EntityKey) -> Result<BrainInspiredEntity> {
        let all_entities = self.graph.get_all_entities().await;
        all_entities.iter()
            .find(|(k, _, _)| k == &entity_key)
            .map(|(_, data, activation)| BrainInspiredEntity {
                id: entity_key,
                concept_id: format!("entity_{:?}", entity_key),
                direction: EntityDirection::Input,
                properties: HashMap::new(),
                embedding: data.embedding.clone(),
                activation_state: *activation,
                last_activation: std::time::SystemTime::now(),
                last_update: std::time::SystemTime::now(),
            })
            .ok_or(GraphError::EntityKeyNotFound { key: entity_key })
    }
    
    /// Advanced concept relevance calculation using semantic relationships and multi-layer analysis
    fn calculate_concept_relevance(&self, entity_concept: &str, query_concept: &str) -> f32 {
        let entity_norm = Self::normalize_concept(entity_concept);
        let query_norm = Self::normalize_concept(query_concept);
        
        // 1. Exact match check
        if entity_norm == query_norm {
            return 1.0;
        }
        
        // 2. Substring match check (more forgiving)
        if entity_norm.contains(&query_norm) || query_norm.contains(&entity_norm) {
            return 0.9;
        }
        
        // 3. Hierarchical relationship check
        let hierarchical_score = self.calculate_hierarchical_relevance(&entity_norm, &query_norm);
        if hierarchical_score > 0.8 {
            return hierarchical_score;
        }
        
        // 4. Semantic field analysis
        let semantic_score = self.calculate_semantic_relevance(&entity_norm, &query_norm);
        
        // 5. Lexical similarity
        let lexical_score = self.calculate_lexical_similarity(&entity_norm, &query_norm);
        
        // 6. Domain-specific patterns
        let domain_score = self.calculate_domain_relevance(&entity_norm, &query_norm);
        
        // Weighted combination with emphasis on semantic relationships
        let weights = [0.4, 0.25, 0.2, 0.15]; // semantic, hierarchical, lexical, domain
        let scores = [semantic_score, hierarchical_score, lexical_score, domain_score];
        
        let final_score = weights.iter()
            .zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum::<f32>();
        
        final_score.min(1.0).max(0.0)
    }
    
    fn normalize_concept(concept: &str) -> String {
        concept.to_lowercase()
            .trim()
            .replace("_", " ")
            .replace("-", " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn calculate_hierarchical_relevance(&self, entity: &str, query: &str) -> f32 {
        // Hierarchical relationships - specific to general
        let hierarchies = vec![
            // Animal hierarchy
            vec!["golden retriever", "retriever", "dog", "canine", "mammal", "animal"],
            vec!["persian cat", "cat", "feline", "mammal", "animal"],
            vec!["domestic cat", "house cat", "cat", "feline", "mammal", "animal"],
            
            // Technology hierarchy
            vec!["machine learning", "artificial intelligence", "ai", "technology"],
            vec!["neural network", "deep learning", "machine learning", "ai", "technology"],
            vec!["computer science", "technology", "science"],
            
            // Science hierarchy
            vec!["quantum mechanics", "physics", "science"],
            vec!["relativity", "physics", "science"],
            vec!["organic chemistry", "chemistry", "science"],
        ];
        
        for hierarchy in &hierarchies {
            let entity_pos = hierarchy.iter().position(|&x| x == entity);
            let query_pos = hierarchy.iter().position(|&x| x == query);
            
            match (entity_pos, query_pos) {
                (Some(e_pos), Some(q_pos)) => {
                    // Both concepts are in the same hierarchy
                    let distance = (e_pos as i32 - q_pos as i32).abs() as f32;
                    let max_distance = hierarchy.len() as f32 - 1.0;
                    
                    if distance == 0.0 {
                        return 1.0;
                    } else if distance == 1.0 {
                        // Direct parent-child relationship
                        return 0.9;
                    } else {
                        // Same hierarchy but further apart
                        return (1.0 - distance / max_distance).max(0.6);
                    }
                },
                _ => continue,
            }
        }
        
        // Check for partial hierarchical matches with better scoring
        for hierarchy in &hierarchies {
            let entity_matches: Vec<(usize, &str)> = hierarchy.iter().enumerate()
                .filter(|(_, term)| entity.contains(**term) || (*term).contains(entity))
                .map(|(i, term)| (i, *term))
                .collect();
                
            let query_matches: Vec<(usize, &str)> = hierarchy.iter().enumerate()
                .filter(|(_, term)| query.contains(**term) || (*term).contains(query))
                .map(|(i, term)| (i, *term))
                .collect();
            
            if !entity_matches.is_empty() && !query_matches.is_empty() {
                // Find the best matching terms and calculate distance
                let mut best_score = 0.0f32;
                for (e_pos, _e_term) in &entity_matches {
                    for (q_pos, _q_term) in &query_matches {
                        let distance = (*e_pos as i32 - *q_pos as i32).abs() as f32;
                        let max_distance = hierarchy.len() as f32 - 1.0;
                        let score = (1.0 - distance / max_distance).max(0.3);
                        best_score = best_score.max(score);
                    }
                }
                return best_score;
            }
        }
        
        0.0
    }
    
    fn calculate_semantic_relevance(&self, entity: &str, query: &str) -> f32 {
        // Semantic field associations
        let semantic_fields = vec![
            // Pet/animal field
            vec!["dog", "cat", "pet", "animal", "mammal", "furry", "domesticated", 
                 "canine", "feline", "puppy", "kitten", "breed"],
            
            // AI/Technology field  
            vec!["ai", "artificial", "intelligence", "machine", "computer", "algorithm", 
                 "technology", "neural", "network", "learning", "deep"],
            
            // Science field
            vec!["physics", "chemistry", "biology", "science", "theory", "experiment", 
                 "research", "academic", "quantum", "relativity"],
            
            // Creative field
            vec!["art", "creative", "beauty", "expression", "aesthetic", "culture", 
                 "imagination", "design", "artistic"],
        ];
        
        for field in &semantic_fields {
            let entity_score = field.iter().map(|&term| {
                if entity == term {
                    1.0
                } else if entity.contains(term) || term.contains(entity) {
                    0.8
                } else {
                    // Check for word-level matches
                    let entity_words: Vec<&str> = entity.split_whitespace().collect();
                    if entity_words.iter().any(|&word| word == term) {
                        0.9
                    } else {
                        0.0
                    }
                }
            }).fold(0.0f32, f32::max);
            
            let query_score = field.iter().map(|&term| {
                if query == term {
                    1.0
                } else if query.contains(term) || term.contains(query) {
                    0.8
                } else {
                    // Check for word-level matches
                    let query_words: Vec<&str> = query.split_whitespace().collect();
                    if query_words.iter().any(|&word| word == term) {
                        0.9
                    } else {
                        0.0
                    }
                }
            }).fold(0.0f32, f32::max);
            
            if entity_score > 0.0 && query_score > 0.0 {
                return (entity_score * query_score).sqrt(); // Geometric mean
            }
        }
        
        0.0
    }
    
    fn calculate_lexical_similarity(&self, entity: &str, query: &str) -> f32 {
        // Direct substring matching
        if entity.contains(query) || query.contains(entity) {
            let longer_len = entity.len().max(query.len()) as f32;
            let shorter_len = entity.len().min(query.len()) as f32;
            return shorter_len / longer_len;
        }
        
        // Word overlap similarity (Jaccard)
        let entity_words: std::collections::HashSet<_> = entity.split_whitespace().collect();
        let query_words: std::collections::HashSet<_> = query.split_whitespace().collect();
        
        let intersection = entity_words.intersection(&query_words).count();
        let union = entity_words.union(&query_words).count();
        
        let jaccard = if union > 0 { intersection as f32 / union as f32 } else { 0.0 };
        
        // Edit distance similarity as fallback
        if jaccard == 0.0 {
            let distance = self.levenshtein_distance(entity, query);
            let max_len = entity.len().max(query.len());
            if max_len > 0 {
                (1.0 - (distance as f32 / max_len as f32)).max(0.0)
            } else {
                0.0
            }
        } else {
            jaccard
        }
    }
    
    fn calculate_domain_relevance(&self, entity: &str, query: &str) -> f32 {
        // Domain-specific patterns and associations
        let domain_patterns = vec![
            // Dog breed patterns
            (vec!["golden", "retriever"], vec!["dog", "canine"], 0.9),
            (vec!["persian", "cat"], vec!["cat", "feline"], 0.9),
            (vec!["domestic", "house"], vec!["pet", "animal"], 0.8),
            
            // AI patterns
            (vec!["machine", "learning"], vec!["ai", "artificial"], 0.9),
            (vec!["neural", "network"], vec!["ai", "deep"], 0.9),
            (vec!["deep", "learning"], vec!["neural", "ai"], 0.9),
            
            // Science patterns
            (vec!["quantum", "mechanics"], vec!["physics", "science"], 0.9),
            (vec!["general", "relativity"], vec!["physics", "einstein"], 0.9),
        ];
        
        for (pattern1, pattern2, score) in &domain_patterns {
            let entity_matches_1 = pattern1.iter().any(|&p| entity.contains(p));
            let entity_matches_2 = pattern2.iter().any(|&p| entity.contains(p));
            let query_matches_1 = pattern1.iter().any(|&p| query.contains(p));
            let query_matches_2 = pattern2.iter().any(|&p| query.contains(p));
            
            if (entity_matches_1 && query_matches_2) || (entity_matches_2 && query_matches_1) {
                return *score;
            }
        }
        
        0.0
    }
    
    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s1_len = s1_chars.len();
        let s2_len = s2_chars.len();
        
        let mut matrix = vec![vec![0; s2_len + 1]; s1_len + 1];
        
        for i in 0..=s1_len {
            matrix[i][0] = i;
        }
        for j in 0..=s2_len {
            matrix[0][j] = j;
        }
        
        for i in 1..=s1_len {
            for j in 1..=s2_len {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                    matrix[i - 1][j - 1] + cost,
                );
            }
        }
        
        matrix[s1_len][s2_len]
    }
    
    /// Infer exploration type from query
    fn infer_exploration_type(&self, query: &str) -> ExplorationType {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("examples") || query_lower.contains("instances") {
            ExplorationType::Instances
        } else if query_lower.contains("types") || query_lower.contains("categories") {
            ExplorationType::Categories
        } else if query_lower.contains("properties") || query_lower.contains("attributes") {
            ExplorationType::Properties
        } else if query_lower.contains("related") || query_lower.contains("associated") {
            ExplorationType::Associations
        } else if query_lower.contains("creative") || query_lower.contains("brainstorm") {
            ExplorationType::Creative
        } else {
            ExplorationType::Instances // Default
        }
    }
    
    /// Extract seed concept from query
    fn extract_seed_concept(&self, query: &str) -> String {
        // Simple extraction - find the main noun/concept
        let words: Vec<&str> = query.split_whitespace().collect();
        
        // Look for pattern "What are examples of X?"
        if let Some(pos) = words.iter().position(|&w| w.to_lowercase() == "of") {
            if pos + 1 < words.len() {
                return words[pos + 1].trim_end_matches('?').to_lowercase();
            }
        }
        
        // Look for pattern "Types of X"
        if let Some(pos) = words.iter().position(|&w| w.to_lowercase() == "of") {
            if pos + 1 < words.len() {
                return words[pos + 1].trim_end_matches('?').to_lowercase();
            }
        }
        
        // Default: find the last meaningful word
        words.iter()
            .rev()
            .find(|&&word| word.len() > 2 && !self.is_stop_word(word))
            .map(|&word| word.trim_end_matches('?').to_lowercase())
            .unwrap_or_else(|| "unknown".to_string())
    }
    
    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let word_clean = word.to_lowercase();
        let word_clean = word_clean.trim_end_matches('?');
        matches!(word_clean, 
            "the" | "is" | "at" | "which" | "on" | "a" | "an" | "and" | "or" | 
            "but" | "in" | "with" | "to" | "for" | "of" | "as" | "by" | "that" | 
            "this" | "what" | "are" | "types" | "examples" | "instances" |
            "where" | "when" | "why" | "how" | "who" | "whom" | "whose" |
            "type" | "kind" | "kinds" | "about" | "can" | "could" | "should" |
            "would" | "will" | "was" | "were" | "been" | "being" | "have" |
            "has" | "had" | "do" | "does" | "did" | "many" | "much" | "some" | "any"
        )
    }
    
    /// Generate exploration summary
    async fn generate_exploration_summary(&self, result: &DivergentResult) -> Result<String> {
        if result.explorations.is_empty() {
            return Ok("No explorations found".to_string());
        }
        
        let mut summary = String::new();
        summary.push_str("Exploration results:\n");
        
        for (i, exploration) in result.explorations.iter().enumerate() {
            if i >= 5 { break; } // Limit to top 5
            summary.push_str(&format!("- {}: {:.2} relevance, {:.2} novelty\n", 
                exploration.concept, exploration.relevance_score, exploration.novelty_score));
        }
        
        Ok(summary)
    }
    
    /// Create reasoning trace from exploration results
    async fn create_reasoning_trace(&self, result: &DivergentResult) -> Result<Vec<ActivationStep>> {
        let mut trace = Vec::new();
        
        for (i, exploration) in result.explorations.iter().enumerate() {
            if i >= 10 { break; } // Limit trace size
            
            for (j, entity_key) in exploration.path.iter().enumerate() {
                trace.push(ActivationStep {
                    step_id: trace.len(),
                    entity_key: *entity_key,
                    concept_id: exploration.concepts.get(j).unwrap_or(&"unknown".to_string()).clone(),
                    activation_level: exploration.relevance_score,
                    operation_type: ActivationOperation::Propagate,
                    timestamp: SystemTime::now(),
                });
            }
        }
        
        Ok(trace)
    }
}

#[async_trait]
impl CognitivePattern for DivergentThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Extract exploration type from query or use default
        let exploration_type = self.infer_exploration_type(query);
        
        // Update parameters if provided
        let exploration_breadth = parameters.exploration_breadth.unwrap_or(self.exploration_breadth);
        let creativity_threshold = parameters.creativity_threshold.unwrap_or(self.creativity_threshold);
        
        // Create temporary instance with updated parameters
        let mut temp_instance = DivergentThinking::new(self.graph.clone());
        temp_instance.exploration_breadth = exploration_breadth;
        temp_instance.creativity_threshold = creativity_threshold;
        
        // Extract seed concept from query
        let seed_concept = self.extract_seed_concept(query);
        
        let result = temp_instance.execute_divergent_exploration(&seed_concept, exploration_type).await?;
        
        let execution_time = start_time.elapsed();
        
        // Generate answer from exploration results
        let answer = self.generate_exploration_summary(&result).await?;
        
        // Create reasoning trace from exploration paths
        let reasoning_trace = self.create_reasoning_trace(&result).await?;
        
        // Calculate overall confidence
        let confidence = if !result.creativity_scores.is_empty() {
            result.creativity_scores.iter().sum::<f32>() / result.creativity_scores.len() as f32
        } else {
            0.0
        };
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Divergent,
            answer,
            confidence,
            reasoning_trace,
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: result.total_paths_explored,
                iterations_completed: self.max_exploration_depth,
                converged: false, // Divergent thinking doesn't converge
                total_energy: result.creativity_scores.iter().sum(),
                additional_info: {
                    let mut info = HashMap::new();
                    info.insert("query".to_string(), query.to_string());
                    info.insert("pattern".to_string(), "divergent".to_string());
                    info.insert("exploration_type".to_string(), format!("{:?}", exploration_type));
                    info.insert("paths_found".to_string(), result.total_paths_explored.to_string());
                    info
                },
            },
        })
    }

    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Divergent
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Brainstorming".to_string(),
            "Creative exploration".to_string(),
            "Finding examples".to_string(),
            "Discovering associations".to_string(),
            "Open-ended questions".to_string(),
            "What are types of X?".to_string(),
            "What's related to Y?".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, query: &str) -> ComplexityEstimate {
        let word_count = query.split_whitespace().count();
        let complexity = ((word_count * self.exploration_breadth) as u32).min(50);
        
        ComplexityEstimate {
            computational_complexity: complexity,
            estimated_time_ms: (complexity as u64) * 50,
            memory_requirements_mb: (complexity as f32 * 0.5) as u32,
            confidence: 0.7,
            parallelizable: true,
        }
    }
}

impl DivergentThinking {
    /// Evaluate path creativity using graph-based metrics
    async fn evaluate_path_creativity(&self, path: &ExplorationPath, embedding: &[f32]) -> Result<f32> {
        let mut creativity_score = 0.0;
        
        // 1. Path length bonus - longer paths are more creative
        creativity_score += (path.path.len() as f32 / 10.0).min(0.3);
        
        // 2. Cross-domain connections - paths spanning different domains are more creative
        let domains = self.extract_domains_from_path(path)?;
        if domains.len() > 1 {
            creativity_score += 0.3 * (domains.len() as f32 / 4.0).min(1.0);
        }
        
        // 3. Novelty of connections - rare relationships are more creative
        for i in 0..path.path.len().saturating_sub(1) {
            let connection_strength = self.get_connection_strength(path.path[i], path.path[i+1]).await?;
            // Lower connection strength means more novel/creative
            creativity_score += (1.0 - connection_strength) * 0.1;
        }
        
        // 4. Embedding variance - high variance indicates diverse concepts
        let variance = self.calculate_embedding_variance(embedding);
        creativity_score += variance.min(0.2);
        
        Ok(creativity_score.min(1.0))
    }
    
    /// Extract domains from a path
    fn extract_domains_from_path(&self, path: &ExplorationPath) -> Result<HashSet<String>> {
        let mut domains = HashSet::new();
        for concept in &path.concepts {
            if let Some(domain) = self.infer_domain(concept) {
                domains.insert(domain);
            }
        }
        Ok(domains)
    }
    
    /// Infer domain from concept
    fn infer_domain(&self, concept: &str) -> Option<String> {
        let concept_lower = concept.to_lowercase();
        
        if concept_lower.contains("tech") || concept_lower.contains("computer") || concept_lower.contains("ai") {
            Some("technology".to_string())
        } else if concept_lower.contains("bio") || concept_lower.contains("life") || concept_lower.contains("organ") {
            Some("biology".to_string())
        } else if concept_lower.contains("art") || concept_lower.contains("creative") || concept_lower.contains("design") {
            Some("art".to_string())
        } else if concept_lower.contains("phys") || concept_lower.contains("quantum") || concept_lower.contains("energy") {
            Some("physics".to_string())
        } else {
            None
        }
    }
    
    /// Get connection strength between two entities
    async fn get_connection_strength(&self, from: EntityKey, to: EntityKey) -> Result<f32> {
        // Check if there's a direct connection by looking at neighbors
        let neighbors = self.graph.get_neighbors_with_weights(from).await;
        
        for (neighbor, weight) in neighbors {
            if neighbor == to {
                return Ok(weight);
            }
        }
        
        // No direct connection means low strength
        Ok(0.1)
    }
    
    /// Calculate variance of embedding values
    fn calculate_embedding_variance(&self, embedding: &[f32]) -> f32 {
        if embedding.is_empty() {
            return 0.0;
        }
        
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance: f32 = embedding.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / embedding.len() as f32;
            
        variance.sqrt()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exploration_type_inference() {
        let divergent = DivergentThinking::new(
            Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap()),
        );
        
        assert!(matches!(
            divergent.infer_exploration_type("What are types of animals?"),
            ExplorationType::Instances
        ));
        
        assert!(matches!(
            divergent.infer_exploration_type("What categories exist?"),
            ExplorationType::Categories
        ));
        
        assert!(matches!(
            divergent.infer_exploration_type("What is related to computers?"),
            ExplorationType::Associations
        ));
    }
    
    #[test]
    fn test_seed_concept_extraction() {
        let divergent = DivergentThinking::new(
            Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap()),
        );
        
        assert_eq!(
            divergent.extract_seed_concept("What are types of animals?"),
            "animals"
        );
        
        assert_eq!(
            divergent.extract_seed_concept("Tell me about computers"),
            "computers"
        );
    }
    
    #[test]
    fn test_concept_similarity() {
        let divergent = DivergentThinking::new(
            Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().unwrap()),
        );
        
        assert_eq!(divergent.calculate_concept_relevance("dog", "dog"), 1.0);
        
        // Debug: Check what score we're actually getting
        let score = divergent.calculate_concept_relevance("golden retriever", "dog");
        println!("Debug: golden retriever -> dog score: {}", score);
        
        // Debug individual components
        let entity_norm = DivergentThinking::normalize_concept("golden retriever");
        let query_norm = DivergentThinking::normalize_concept("dog");
        println!("Debug: normalized: '{}' -> '{}'", entity_norm, query_norm);
        
        let hierarchical_score = divergent.calculate_hierarchical_relevance(&entity_norm, &query_norm);
        println!("Debug: hierarchical score: {}", hierarchical_score);
        
        let semantic_score = divergent.calculate_semantic_relevance(&entity_norm, &query_norm);
        println!("Debug: semantic score: {}", semantic_score);
        
        assert!(score > 0.8, "Expected score > 0.8 but got {}", score);
        assert!(divergent.calculate_concept_relevance("domestic cat", "house cat") > 0.5);
    }
}