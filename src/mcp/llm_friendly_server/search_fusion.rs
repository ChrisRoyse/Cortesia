//! Search result fusion for hybrid search

use crate::core::knowledge_types::KnowledgeResult;
use crate::error::Result;
use std::collections::HashMap;

/// Fuse search results from multiple search strategies
pub async fn fuse_search_results(
    semantic_results: Vec<KnowledgeResult>,
    structural_results: Vec<KnowledgeResult>,
    keyword_results: Vec<KnowledgeResult>,
    fusion_weights: Option<FusionWeights>,
) -> Result<Vec<FusedResult>> {
    let weights = fusion_weights.unwrap_or_default();
    let mut fusion_map: HashMap<String, FusedResult> = HashMap::new();
    
    // Process semantic results
    for (rank, result) in semantic_results.into_iter().enumerate() {
        let id = generate_result_id(&result);
        let score = calculate_rank_score(rank) * weights.semantic;
        
        fusion_map.entry(id.clone()).or_insert_with(|| FusedResult {
            result: result.clone(),
            fusion_score: 0.0,
            semantic_score: 0.0,
            structural_score: 0.0,
            keyword_score: 0.0,
            source_types: vec![],
        }).semantic_score = score;
        
        fusion_map.get_mut(&id).unwrap().source_types.push(SearchType::Semantic);
    }
    
    // Process structural results
    for (rank, result) in structural_results.into_iter().enumerate() {
        let id = generate_result_id(&result);
        let score = calculate_rank_score(rank) * weights.structural;
        
        fusion_map.entry(id.clone()).or_insert_with(|| FusedResult {
            result: result.clone(),
            fusion_score: 0.0,
            semantic_score: 0.0,
            structural_score: 0.0,
            keyword_score: 0.0,
            source_types: vec![],
        }).structural_score = score;
        
        fusion_map.get_mut(&id).unwrap().source_types.push(SearchType::Structural);
    }
    
    // Process keyword results
    for (rank, result) in keyword_results.into_iter().enumerate() {
        let id = generate_result_id(&result);
        let score = calculate_rank_score(rank) * weights.keyword;
        
        fusion_map.entry(id.clone()).or_insert_with(|| FusedResult {
            result: result.clone(),
            fusion_score: 0.0,
            semantic_score: 0.0,
            structural_score: 0.0,
            keyword_score: 0.0,
            source_types: vec![],
        }).keyword_score = score;
        
        fusion_map.get_mut(&id).unwrap().source_types.push(SearchType::Keyword);
    }
    
    // Calculate fusion scores
    for fused in fusion_map.values_mut() {
        fused.fusion_score = fused.semantic_score + fused.structural_score + fused.keyword_score;
        
        // Boost results that appear in multiple search types
        let appearance_boost = (fused.source_types.len() as f32 - 1.0) * weights.multi_source_boost;
        fused.fusion_score *= 1.0 + appearance_boost;
    }
    
    // Sort by fusion score
    let mut results: Vec<_> = fusion_map.into_values().collect();
    results.sort_by(|a, b| b.fusion_score.partial_cmp(&a.fusion_score).unwrap());
    
    Ok(results)
}

/// Get fusion weights based on search type
pub fn get_fusion_weights(search_type: &str) -> FusionWeights {
    match search_type {
        "semantic" => FusionWeights {
            semantic: 0.8,
            structural: 0.1,
            keyword: 0.1,
            multi_source_boost: 0.2,
        },
        "structural" => FusionWeights {
            semantic: 0.1,
            structural: 0.8,
            keyword: 0.1,
            multi_source_boost: 0.2,
        },
        "keyword" => FusionWeights {
            semantic: 0.1,
            structural: 0.1,
            keyword: 0.8,
            multi_source_boost: 0.2,
        },
        "hybrid" | _ => FusionWeights {
            semantic: 0.4,
            structural: 0.3,
            keyword: 0.3,
            multi_source_boost: 0.3,
        },
    }
}

/// Calculate score based on rank position
fn calculate_rank_score(rank: usize) -> f32 {
    // Reciprocal rank scoring with smoothing
    1.0 / (rank as f32 + 1.0).sqrt()
}

/// Generate unique ID for a result
fn generate_result_id(result: &KnowledgeResult) -> String {
    // Generate ID based on the first triple in the result or a hash of all triples
    if let Some(first_triple) = result.triples.first() {
        format!("triple:{}:{}:{}", first_triple.subject, first_triple.predicate, first_triple.object)
    } else if let Some(first_node) = result.nodes.first() {
        format!("node:{}", first_node.id)
    } else {
        // Fallback to a generic ID based on the result metadata
        format!("result:found_{}_in_{}ms", result.total_found, result.query_time_ms)
    }
}

#[derive(Debug, Clone)]
pub struct FusedResult {
    pub result: KnowledgeResult,
    pub fusion_score: f32,
    pub semantic_score: f32,
    pub structural_score: f32,
    pub keyword_score: f32,
    pub source_types: Vec<SearchType>,
}

#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    pub semantic: f32,
    pub structural: f32,
    pub keyword: f32,
    pub multi_source_boost: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            semantic: 0.4,
            structural: 0.3,
            keyword: 0.3,
            multi_source_boost: 0.3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchType {
    Semantic,
    Structural,
    Keyword,
}