//! Tests for graph analysis consolidation functionality
//! Following TDD: Write failing tests first

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::handlers::graph_analysis::handle_analyze_graph;
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::test_support::test_utils::{create_test_engine, create_test_stats};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;

#[cfg(test)]
mod graph_analysis_tests {
    use super::*;

    // Test helper functions are now imported from test_utils

    #[tokio::test]
    async fn test_analyze_graph_connections_mode() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "connections",
            "config": {
                "start_entity": "Einstein",
                "end_entity": "Nobel_Prize",
                "max_depth": 3,
                "relationship_types": ["won", "nominated_for", "colleague_of"]
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        assert_eq!(data["analysis_type"], "connections");
        assert!(data["results"]["paths"].is_array());
        assert!(data["results"]["total_paths"].as_u64().is_some());
        assert!(message.contains("connections"));
    }

    #[tokio::test]
    async fn test_analyze_graph_centrality_mode() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "centrality",
            "config": {
                "centrality_types": ["pagerank", "betweenness"],
                "top_n": 10,
                "include_scores": true,
                "entity_filter": "scientist"
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        assert_eq!(data["analysis_type"], "centrality");
        assert!(data["results"]["centrality_measures"].is_object());
        assert!(data["results"]["centrality_measures"]["pagerank"].is_array());
        assert!(data["results"]["centrality_measures"]["betweenness"].is_array());
        assert!(message.contains("centrality"));
    }

    #[tokio::test]
    async fn test_analyze_graph_clustering_mode() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "clustering",
            "config": {
                "algorithm": "leiden",
                "resolution": 1.2,
                "min_cluster_size": 3,
                "max_clusters": 20,
                "include_metadata": true
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        assert_eq!(data["analysis_type"], "clustering");
        assert!(data["results"]["clusters"].is_array());
        assert!(data["results"]["clustering_metrics"]["modularity"].as_f64().is_some());
        assert!(data["results"]["cluster_metadata"].is_object());
        assert!(message.contains("clustering"));
    }

    #[tokio::test]
    async fn test_analyze_graph_prediction_mode() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "prediction",
            "config": {
                "prediction_type": "missing_links",
                "confidence_threshold": 0.7,
                "max_predictions": 15,
                "entity_filter": "physicist",
                "use_neural_features": true
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        assert_eq!(data["analysis_type"], "prediction");
        assert!(data["results"]["predictions"].is_array());
        assert!(data["results"]["confidence_distribution"].is_object());
        assert!(data["results"]["validation_score"].as_f64().is_some());
        assert!(message.contains("prediction"));
    }

    #[tokio::test]
    async fn test_analyze_graph_invalid_analysis_type() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "invalid_type",
            "config": {}
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Invalid analysis_type"));
    }

    #[tokio::test]
    async fn test_analyze_graph_missing_required_config() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "connections",
            "config": {
                // Missing start_entity
                "end_entity": "Nobel_Prize"
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Missing required config field: start_entity"));
    }

    #[tokio::test]
    async fn test_backwards_compatibility_explore_connections() {
        // This test ensures the migration from explore_connections works
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        
        // Old explore_connections parameters
        let old_params = json!({
            "start_entity": "Einstein",
            "end_entity": "Relativity",
            "max_depth": 2
        });
        
        // Expected new format
        let expected_params = json!({
            "analysis_type": "connections",
            "config": old_params
        });
        
        // Test that the new format works
        let result = handle_analyze_graph(&engine, &stats, expected_params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_metrics_tracking() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "centrality",
            "config": {
                "centrality_types": ["pagerank"],
                "top_n": 5
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert!(data["performance_metrics"]["execution_time_ms"].as_f64().is_some());
        assert!(data["performance_metrics"]["nodes_processed"].as_u64().is_some());
        assert!(data["performance_metrics"]["edges_processed"].as_u64().is_some());
    }

    #[tokio::test]
    async fn test_connections_with_no_path_found() {
        // Arrange
        let engine = create_test_engine(true).await.expect("Failed to create test engine");
        let stats = create_test_stats();
        let params = json!({
            "analysis_type": "connections",
            "config": {
                "start_entity": "UnknownEntity1",
                "end_entity": "UnknownEntity2",
                "max_depth": 2
            }
        });

        // Act
        let result = handle_analyze_graph(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        assert_eq!(data["results"]["total_paths"], 0);
        assert!(message.contains("No paths found"));
    }

    #[tokio::test]
    async fn test_clustering_algorithm_selection() {
        // Test different clustering algorithms
        let algorithms = vec!["leiden", "louvain", "hierarchical"];
        
        for algorithm in algorithms {
            let engine = create_test_engine(true).await.expect("Failed to create test engine");
            let stats = create_test_stats();
            let params = json!({
                "analysis_type": "clustering",
                "config": {
                    "algorithm": algorithm,
                    "min_cluster_size": 2
                }
            });

            let result = handle_analyze_graph(&engine, &stats, params).await;
            assert!(result.is_ok(), "Algorithm {} should be supported", algorithm);
            
            let (data, _, _) = result.unwrap();
            assert_eq!(data["results"]["algorithm_used"], algorithm);
        }
    }
}