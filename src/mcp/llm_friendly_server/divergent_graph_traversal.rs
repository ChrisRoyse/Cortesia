//! Graph traversal algorithms for divergent thinking
//! Pure algorithmic implementation - no AI models

use crate::core::knowledge_types::TripleQuery;
use crate::core::knowledge_engine::KnowledgeEngine;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use rand::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPath {
    pub steps: Vec<ExplorationStep>,
    pub total_distance: f32,
    pub uniqueness_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStep {
    pub entity: String,
    pub relationship: String,
    pub step_number: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergentExplorationResult {
    pub paths: Vec<ExplorationPath>,
    pub discovered_entities: HashSet<String>,
    pub discovered_relationships: HashSet<String>,
    pub cross_domain_connections: Vec<CrossDomainConnection>,
    pub exploration_stats: ExplorationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainConnection {
    pub from_entity: String,
    pub to_entity: String,
    pub from_domain: String,
    pub to_domain: String,
    pub connection_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStats {
    pub total_entities_explored: usize,
    pub unique_paths_found: usize,
    pub average_path_length: f32,
    pub max_depth_reached: usize,
}

/// Perform divergent graph traversal from a seed entity
pub async fn explore_divergent_paths(
    engine: &Arc<RwLock<KnowledgeEngine>>,
    seed_entity: &str,
    max_depth: usize,
    creativity_factor: f32,
    max_branches: usize,
) -> DivergentExplorationResult {
    let mut paths = Vec::new();
    let mut discovered_entities = HashSet::new();
    let mut discovered_relationships = HashSet::new();
    
    // Track visited nodes to avoid cycles
    let mut global_visited = HashSet::new();
    global_visited.insert(seed_entity.to_string());
    
    // Generate multiple exploration branches
    for branch_id in 0..max_branches {
        let path = explore_single_branch(
            engine,
            seed_entity,
            max_depth,
            creativity_factor,
            &mut global_visited,
            branch_id,
        ).await;
        
        // Collect discovered entities and relationships
        for step in &path.steps {
            discovered_entities.insert(step.entity.clone());
            discovered_relationships.insert(step.relationship.clone());
        }
        
        if !path.steps.is_empty() {
            paths.push(path);
        }
    }
    
    // Find cross-domain connections
    let cross_domain_connections = find_cross_domain_connections(&discovered_entities, engine).await;
    
    // Calculate statistics
    let stats = calculate_exploration_stats(&paths, discovered_entities.len());
    
    DivergentExplorationResult {
        paths,
        discovered_entities,
        discovered_relationships,
        cross_domain_connections,
        exploration_stats: stats,
    }
}

/// Explore a single branch using weighted random walk
async fn explore_single_branch(
    engine: &Arc<RwLock<KnowledgeEngine>>,
    seed_entity: &str,
    max_depth: usize,
    creativity_factor: f32,
    global_visited: &mut HashSet<String>,
    _branch_id: usize,
) -> ExplorationPath {
    let mut steps = Vec::new();
    let mut current_entity = seed_entity.to_string();
    let mut total_distance = 0.0;
    let mut local_visited = HashSet::new();
    local_visited.insert(current_entity.clone());
    
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_entropy();
    
    for depth in 0..max_depth {
        // Query all relationships from current entity
        let engine_lock = engine.read().await;
        let neighbors = engine_lock.query_triples(TripleQuery {
            subject: Some(current_entity.clone()),
            predicate: None,
            object: None,
            limit: 50,
            min_confidence: 0.0,
            include_chunks: false,
        }).unwrap_or_default();
        drop(engine_lock);
        
        if neighbors.is_empty() {
            break;
        }
        
        // Score each neighbor based on creativity factor
        let mut scored_neighbors: Vec<(String, String, f32)> = Vec::new();
        
        for triple in &neighbors.triples {
            let neighbor = &triple.object;
            
            // Skip if already visited locally
            if local_visited.contains(neighbor) {
                continue;
            }
            
            // Calculate score based on:
            // 1. Global novelty (not visited by other branches)
            // 2. Semantic distance (approximated by string difference)
            // 3. Random factor based on creativity
            
            let novelty_score = if global_visited.contains(neighbor) { 0.5 } else { 1.0 };
            let distance_score = calculate_semantic_distance(&current_entity, neighbor);
            let random_score = rng.gen::<f32>();
            
            let total_score = (novelty_score * 0.3) + 
                            (distance_score * 0.4) + 
                            (random_score * creativity_factor * 0.3);
            
            scored_neighbors.push((neighbor.clone(), triple.predicate.clone(), total_score));
        }
        
        // Sort by score and select based on creativity
        scored_neighbors.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        if scored_neighbors.is_empty() {
            break;
        }
        
        // Select next node - higher creativity means more randomness
        let selection_index = if creativity_factor > 0.7 {
            // High creativity: pick from top 50% randomly
            let range = (scored_neighbors.len() as f32 * 0.5).ceil() as usize;
            rng.gen_range(0..range.min(scored_neighbors.len()))
        } else {
            // Low creativity: pick the best
            0
        };
        
        let (next_entity, relationship, score) = &scored_neighbors[selection_index];
        
        // Add step to path
        steps.push(ExplorationStep {
            entity: next_entity.clone(),
            relationship: relationship.clone(),
            step_number: depth + 1,
        });
        
        // Update tracking
        current_entity = next_entity.clone();
        local_visited.insert(current_entity.clone());
        global_visited.insert(current_entity.clone());
        total_distance += score;
    }
    
    // Calculate uniqueness score
    let uniqueness_score = calculate_path_uniqueness(&steps);
    
    ExplorationPath {
        steps,
        total_distance,
        uniqueness_score,
    }
}

/// Calculate semantic distance between two entities (simplified)
fn calculate_semantic_distance(entity1: &str, entity2: &str) -> f32 {
    // Simple heuristic based on string similarity
    let chars1: HashSet<char> = entity1.chars().collect();
    let chars2: HashSet<char> = entity2.chars().collect();
    
    let intersection = chars1.intersection(&chars2).count() as f32;
    let union = chars1.union(&chars2).count() as f32;
    
    if union > 0.0 {
        1.0 - (intersection / union)
    } else {
        1.0
    }
}

/// Calculate how unique a path is based on its entities
fn calculate_path_uniqueness(steps: &[ExplorationStep]) -> f32 {
    if steps.is_empty() {
        return 0.0;
    }
    
    let unique_entities: HashSet<_> = steps.iter().map(|s| &s.entity).collect();
    let unique_relationships: HashSet<_> = steps.iter().map(|s| &s.relationship).collect();
    
    let entity_ratio = unique_entities.len() as f32 / steps.len() as f32;
    let relationship_ratio = unique_relationships.len() as f32 / steps.len() as f32;
    
    (entity_ratio + relationship_ratio) / 2.0
}

/// Find connections between different domains
async fn find_cross_domain_connections(
    entities: &HashSet<String>,
    engine: &Arc<RwLock<KnowledgeEngine>>,
) -> Vec<CrossDomainConnection> {
    let mut connections = Vec::new();
    
    // Group entities by domain (simplified - based on keywords)
    let mut domains: HashMap<String, Vec<String>> = HashMap::new();
    
    for entity in entities {
        let domain = classify_entity_domain(entity);
        domains.entry(domain).or_default().push(entity.clone());
    }
    
    // Find connections between different domains
    let domain_keys: Vec<String> = domains.keys().cloned().collect();
    
    for i in 0..domain_keys.len() {
        for j in i+1..domain_keys.len() {
            let domain1 = &domain_keys[i];
            let domain2 = &domain_keys[j];
            
            // Check for connections between entities in different domains
            for entity1 in &domains[domain1] {
                for entity2 in &domains[domain2] {
                    // Query if there's a path between them
                    let engine_lock = engine.read().await;
                    let connections_exist = engine_lock.query_triples(TripleQuery {
                        subject: Some(entity1.clone()),
                        predicate: None,
                        object: Some(entity2.clone()),
                        limit: 1,
                        min_confidence: 0.0,
                        include_chunks: false,
                    }).map(|r| !r.is_empty()).unwrap_or(false);
                    drop(engine_lock);
                    
                    if connections_exist {
                        connections.push(CrossDomainConnection {
                            from_entity: entity1.clone(),
                            to_entity: entity2.clone(),
                            from_domain: domain1.clone(),
                            to_domain: domain2.clone(),
                            connection_strength: 0.8, // Simplified
                        });
                    }
                }
            }
        }
    }
    
    connections
}

/// Classify entity into a domain based on keywords
fn classify_entity_domain(entity: &str) -> String {
    let entity_lower = entity.to_lowercase();
    
    if entity_lower.contains("science") || entity_lower.contains("physics") || 
       entity_lower.contains("chemistry") || entity_lower.contains("biology") {
        "science".to_string()
    } else if entity_lower.contains("art") || entity_lower.contains("music") || 
              entity_lower.contains("painting") || entity_lower.contains("sculpture") {
        "art".to_string()
    } else if entity_lower.contains("tech") || entity_lower.contains("computer") || 
              entity_lower.contains("software") || entity_lower.contains("data") {
        "technology".to_string()
    } else if entity_lower.contains("history") || entity_lower.contains("ancient") || 
              entity_lower.contains("war") || entity_lower.contains("civilization") {
        "history".to_string()
    } else {
        "general".to_string()
    }
}

/// Calculate exploration statistics
fn calculate_exploration_stats(paths: &[ExplorationPath], total_entities: usize) -> ExplorationStats {
    if paths.is_empty() {
        return ExplorationStats {
            total_entities_explored: total_entities,
            unique_paths_found: 0,
            average_path_length: 0.0,
            max_depth_reached: 0,
        };
    }
    
    let total_steps: usize = paths.iter().map(|p| p.steps.len()).sum();
    let max_depth = paths.iter().map(|p| p.steps.len()).max().unwrap_or(0);
    let average_length = total_steps as f32 / paths.len() as f32;
    
    ExplorationStats {
        total_entities_explored: total_entities,
        unique_paths_found: paths.len(),
        average_path_length: average_length,
        max_depth_reached: max_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_semantic_distance() {
        assert_eq!(calculate_semantic_distance("Einstein", "Einstein"), 0.0);
        assert!(calculate_semantic_distance("Einstein", "Newton") > 0.5);
        assert!(calculate_semantic_distance("cat", "dog") > 0.7);
    }
    
    #[test]
    fn test_path_uniqueness() {
        let steps = vec![
            ExplorationStep { entity: "A".to_string(), relationship: "knows".to_string(), step_number: 1 },
            ExplorationStep { entity: "B".to_string(), relationship: "likes".to_string(), step_number: 2 },
            ExplorationStep { entity: "C".to_string(), relationship: "knows".to_string(), step_number: 3 },
        ];
        
        let uniqueness = calculate_path_uniqueness(&steps);
        assert!(uniqueness > 0.8); // High uniqueness as all entities are different
    }
    
    #[test]
    fn test_domain_classification() {
        assert_eq!(classify_entity_domain("Einstein"), "general");
        assert_eq!(classify_entity_domain("Quantum Physics"), "science");
        assert_eq!(classify_entity_domain("Computer Science"), "technology");
        assert_eq!(classify_entity_domain("Ancient Rome"), "history");
        assert_eq!(classify_entity_domain("Classical Music"), "art");
    }
}