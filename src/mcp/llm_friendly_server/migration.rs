//! Migration utilities for tool consolidation

use serde_json::{json, Value};

/// Map old tool calls to new consolidated tools
pub fn migrate_tool_call(method: &str, params: Value) -> Option<(String, Value)> {
    match method {
        "simd_ultra_fast_search" => {
            // Convert SIMD search parameters to new format
            let mut new_params = json!({
                "query": params.get("query_text").or(params.get("query")).unwrap_or(&json!("")),
                "search_type": "semantic", // SIMD was primarily semantic
                "performance_mode": "simd",
                "limit": params.get("top_k").or(params.get("limit")).unwrap_or(&json!(10)),
            });
            
            // Add SIMD-specific config
            if let Some(threshold) = params.get("distance_threshold") {
                new_params["simd_config"] = json!({
                    "distance_threshold": threshold,
                    "use_simd": params.get("use_simd").unwrap_or(&json!(true))
                });
            }
            
            Some(("hybrid_search".to_string(), new_params))
        },
        
        "approximate_similarity_search" => {
            // Convert LSH search parameters to new format
            let mut new_params = json!({
                "query": params.get("query_text").or(params.get("query")).unwrap_or(&json!("")),
                "search_type": "semantic",
                "performance_mode": "lsh",
                "limit": params.get("max_results").or(params.get("limit")).unwrap_or(&json!(20)),
            });
            
            // Add LSH-specific config
            new_params["lsh_config"] = json!({
                "hash_functions": params.get("hash_functions").unwrap_or(&json!(64)),
                "hash_tables": params.get("hash_tables").unwrap_or(&json!(8)),
                "similarity_threshold": params.get("similarity_threshold").unwrap_or(&json!(0.7))
            });
            
            Some(("hybrid_search".to_string(), new_params))
        },
        
        "explore_connections" => {
            // Convert to analyze_graph with connections mode
            let config = json!({
                "start_entity": params.get("start_entity"),
                "end_entity": params.get("end_entity"),
                "max_depth": params.get("max_depth").unwrap_or(&json!(2)),
                "relationship_types": params.get("relationship_types")
            });
            
            let new_params = json!({
                "analysis_type": "connections",
                "config": config
            });
            
            Some(("analyze_graph".to_string(), new_params))
        },
        
        "analyze_graph_centrality" => {
            // Convert to analyze_graph with centrality mode
            let config = json!({
                "centrality_types": params.get("centrality_types").unwrap_or(&json!(["pagerank", "betweenness", "closeness"])),
                "top_n": params.get("top_n").unwrap_or(&json!(20)),
                "include_scores": params.get("include_scores").unwrap_or(&json!(true)),
                "entity_filter": params.get("entity_filter")
            });
            
            let new_params = json!({
                "analysis_type": "centrality",
                "config": config
            });
            
            Some(("analyze_graph".to_string(), new_params))
        },
        
        "hierarchical_clustering" => {
            // Convert to analyze_graph with clustering mode
            let config = json!({
                "algorithm": params.get("algorithm").unwrap_or(&json!("leiden")),
                "resolution": params.get("resolution").unwrap_or(&json!(1.0)),
                "min_cluster_size": params.get("min_cluster_size").unwrap_or(&json!(3)),
                "max_clusters": params.get("max_clusters").unwrap_or(&json!(50)),
                "include_metadata": params.get("include_metadata").unwrap_or(&json!(true))
            });
            
            let new_params = json!({
                "analysis_type": "clustering",
                "config": config
            });
            
            Some(("analyze_graph".to_string(), new_params))
        },
        
        "predict_graph_structure" => {
            // Convert to analyze_graph with prediction mode
            let config = json!({
                "prediction_type": params.get("prediction_type").unwrap_or(&json!("missing_links")),
                "confidence_threshold": params.get("confidence_threshold").unwrap_or(&json!(0.7)),
                "max_predictions": params.get("max_predictions").unwrap_or(&json!(20)),
                "entity_filter": params.get("entity_filter"),
                "use_neural_features": params.get("use_neural_features").unwrap_or(&json!(true))
            });
            
            let new_params = json!({
                "analysis_type": "prediction",
                "config": config
            });
            
            Some(("analyze_graph".to_string(), new_params))
        },
        
        "knowledge_quality_metrics" => {
            // Convert to enhanced validate_knowledge
            let mut new_params = json!({
                "scope": "comprehensive",
                "validation_type": params.get("assessment_scope").unwrap_or(&json!("all")),
                "include_metrics": true
            });
            
            // Map assessment parameters
            if let Some(threshold) = params.get("quality_threshold") {
                new_params["quality_threshold"] = threshold.clone();
            }
            
            Some(("validate_knowledge".to_string(), new_params))
        },
        
        _ => None
    }
}

/// Generate deprecation warning for old tools
pub fn deprecation_warning(old_tool: &str) -> String {
    match old_tool {
        "simd_ultra_fast_search" => {
            "Tool 'simd_ultra_fast_search' is deprecated. Use 'hybrid_search' with performance_mode='simd' instead.".to_string()
        },
        "approximate_similarity_search" => {
            "Tool 'approximate_similarity_search' is deprecated. Use 'hybrid_search' with performance_mode='lsh' instead.".to_string()
        },
        "explore_connections" => {
            "Tool 'explore_connections' is deprecated. Use 'analyze_graph' with analysis_type='connections' instead.".to_string()
        },
        "analyze_graph_centrality" => {
            "Tool 'analyze_graph_centrality' is deprecated. Use 'analyze_graph' with analysis_type='centrality' instead.".to_string()
        },
        "hierarchical_clustering" => {
            "Tool 'hierarchical_clustering' is deprecated. Use 'analyze_graph' with analysis_type='clustering' instead.".to_string()
        },
        "predict_graph_structure" => {
            "Tool 'predict_graph_structure' is deprecated. Use 'analyze_graph' with analysis_type='prediction' instead.".to_string()
        },
        "knowledge_quality_metrics" => {
            "Tool 'knowledge_quality_metrics' is deprecated. Use 'validate_knowledge' with scope='comprehensive' instead.".to_string()
        },
        _ => format!("Tool '{}' may be deprecated.", old_tool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_migration() {
        let old_params = json!({
            "query_text": "test query",
            "top_k": 5,
            "distance_threshold": 0.7,
            "use_simd": true
        });
        
        let result = migrate_tool_call("simd_ultra_fast_search", old_params);
        assert!(result.is_some());
        
        let (new_tool, new_params) = result.unwrap();
        assert_eq!(new_tool, "hybrid_search");
        assert_eq!(new_params["performance_mode"], "simd");
        assert_eq!(new_params["query"], "test query");
        assert_eq!(new_params["limit"], 5);
        assert_eq!(new_params["simd_config"]["distance_threshold"], 0.7);
    }
    
    #[test]
    fn test_explore_connections_migration() {
        let old_params = json!({
            "start_entity": "Einstein",
            "end_entity": "Nobel_Prize",
            "max_depth": 3
        });
        
        let result = migrate_tool_call("explore_connections", old_params);
        assert!(result.is_some());
        
        let (new_tool, new_params) = result.unwrap();
        assert_eq!(new_tool, "analyze_graph");
        assert_eq!(new_params["analysis_type"], "connections");
        assert_eq!(new_params["config"]["start_entity"], "Einstein");
        assert_eq!(new_params["config"]["max_depth"], 3);
    }
}