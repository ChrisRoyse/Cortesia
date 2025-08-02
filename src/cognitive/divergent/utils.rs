//! Utility functions for divergent thinking

use super::constants::*;
use crate::cognitive::types::{ExplorationPath, ExplorationMap, ExplorationContext};
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::collections::{HashMap, HashSet};

/// Find entities matching a concept
pub async fn find_concept_entities(graph: &BrainEnhancedKnowledgeGraph, concept: &str) -> Result<Vec<EntityKey>> {
    // Generate embedding for concept
    let concept_embedding = generate_concept_embedding(concept);
    
    // Search for similar entities
    let similar_entities = graph.core_graph.similarity_search(&concept_embedding, 10)?;
    
    // Extract entity keys
    let entity_keys: Vec<EntityKey> = similar_entities.into_iter().map(|(key, _)| key).collect();
    
    Ok(entity_keys)
}

/// Generate embedding for concept (simplified)
pub fn generate_concept_embedding(concept: &str) -> Vec<f32> {
    // This is a simplified embedding generation
    // In practice, you'd use a proper embedding model
    let mut embedding = vec![0.0; 96]; // Standard embedding dimension
    
    // Hash-based embedding generation
    let hash = simple_hash(concept);
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((hash as u64).wrapping_mul(i as u64 + 1) % 1000) as f32 / 1000.0;
    }
    
    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in embedding.iter_mut() {
            *val /= magnitude;
        }
    }
    
    embedding
}

/// Simple hash function for concept
pub fn simple_hash(s: &str) -> u32 {
    let mut hash = 0u32;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
    }
    hash
}

/// Calculate path creativity
pub async fn calculate_path_creativity(
    graph: &BrainEnhancedKnowledgeGraph,
    path: &ExplorationPath,
    exploration_type: &str,
) -> Result<f32> {
    let mut creativity_score = 0.0;
    
    // Base creativity from path length
    creativity_score += (path.depth as f32) * 0.1;
    
    // Creativity from activation level
    creativity_score += path.activation_level * 0.3;
    
    // Creativity from connection weights
    let avg_weight = if path.weights.is_empty() {
        0.0
    } else {
        path.weights.iter().sum::<f32>() / path.weights.len() as f32
    };
    creativity_score += avg_weight * 0.2;
    
    // Boost for creative exploration type
    if exploration_type == "creative" {
        creativity_score *= 1.2;
    }
    
    Ok(creativity_score.clamp(0.0, 1.0))
}

/// Calculate path novelty
pub async fn calculate_path_novelty(
    graph: &BrainEnhancedKnowledgeGraph,
    path: &ExplorationPath,
) -> Result<f32> {
    let mut novelty_score = 0.0;
    
    // Novelty from path length (longer paths are more novel)
    novelty_score += (path.depth as f32 / DEFAULT_MAX_PATH_LENGTH as f32) * 0.4;
    
    // Novelty from weak connections (unusual connections are more novel)
    let avg_weight = if path.weights.is_empty() {
        0.0
    } else {
        path.weights.iter().sum::<f32>() / path.weights.len() as f32
    };
    novelty_score += (1.0 - avg_weight) * 0.3;
    
    // Novelty from activation level (lower activation can be more novel)
    novelty_score += (1.0 - path.activation_level) * 0.3;
    
    Ok(novelty_score.clamp(0.0, 1.0))
}

/// Calculate path relevance
pub async fn calculate_path_relevance(
    graph: &BrainEnhancedKnowledgeGraph,
    path: &ExplorationPath,
    exploration_type: &str,
) -> Result<f32> {
    let mut relevance_score = 0.0;
    
    // Relevance from activation level
    relevance_score += path.activation_level * 0.4;
    
    // Relevance from connection strength
    let avg_weight = if path.weights.is_empty() {
        0.0
    } else {
        path.weights.iter().sum::<f32>() / path.weights.len() as f32
    };
    relevance_score += avg_weight * 0.3;
    
    // Relevance penalty for very long paths
    if path.depth > 3 {
        relevance_score *= 0.8;
    }
    
    // Boost for analytical exploration type
    if exploration_type == "analytical" {
        relevance_score *= 1.1;
    }
    
    Ok(relevance_score.clamp(0.0, 1.0))
}

/// Infer exploration type from query
pub fn infer_exploration_type(query: &str) -> String {
    let query_lower = query.to_lowercase();
    let keywords = get_exploration_type_keywords();
    
    let mut type_scores = HashMap::new();
    
    for (exploration_type, type_keywords) in keywords {
        let mut score = 0;
        for keyword in type_keywords {
            if query_lower.contains(&keyword) {
                score += 1;
            }
        }
        type_scores.insert(exploration_type, score);
    }
    
    // Find the type with highest score
    let best_type = type_scores.iter()
        .max_by_key(|(_, &score)| score)
        .map(|(type_name, _)| type_name.clone())
        .unwrap_or_else(|| "exploratory".to_string());
    
    best_type
}

/// Extract seed concept from query
pub fn extract_seed_concept(query: &str) -> String {
    let stop_words = get_stop_words();
    let words: Vec<&str> = query.split_whitespace().collect();
    
    // Remove stop words and find meaningful concept
    let meaningful_words: Vec<&str> = words.into_iter()
        .filter(|&word| !stop_words.contains(&word.to_lowercase()))
        .collect();
    
    if meaningful_words.is_empty() {
        query.to_string()
    } else {
        meaningful_words[0].to_string()
    }
}