//! Tests for search consolidation functionality
//! Following TDD: Write failing tests first

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::handlers::advanced::handle_hybrid_search;
use crate::mcp::llm_friendly_server::types::UsageStats;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;

#[cfg(test)]
mod search_consolidation_tests {
    use super::*;

    /// Helper to create test knowledge engine
    async fn create_test_engine() -> Arc<RwLock<KnowledgeEngine>> {
        let engine = KnowledgeEngine::new(384, 100_000).unwrap();
        Arc::new(RwLock::new(engine))
    }

    /// Helper to create test usage stats
    fn create_test_stats() -> Arc<RwLock<UsageStats>> {
        Arc::new(RwLock::new(UsageStats::default()))
    }

    #[tokio::test]
    async fn test_hybrid_search_with_standard_performance_mode() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid",
            "performance_mode": "standard",
            "limit": 10
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_mode"], "standard");
        assert!(data["performance_metrics"]["mode"].as_str().unwrap() == "standard");
    }

    #[tokio::test]
    async fn test_hybrid_search_with_simd_performance_mode() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid",
            "performance_mode": "simd",
            "limit": 10
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_mode"], "simd");
        assert!(data["performance_metrics"]["mode"].as_str().unwrap() == "simd");
        assert!(data["performance_metrics"]["simd_acceleration"].as_bool().unwrap());
        assert!(data["performance_metrics"]["throughput_mvps"].as_f64().unwrap() > 1.0);
    }

    #[tokio::test]
    async fn test_hybrid_search_with_lsh_performance_mode() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid",
            "performance_mode": "lsh",
            "limit": 10,
            "lsh_config": {
                "hash_functions": 32,
                "hash_tables": 8
            }
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_mode"], "lsh");
        assert!(data["performance_metrics"]["mode"].as_str().unwrap() == "lsh");
        assert!(data["performance_metrics"]["recall_estimate"].as_f64().unwrap() > 0.8);
        assert!(data["performance_metrics"]["speedup_factor"].as_f64().unwrap() > 5.0);
    }

    #[tokio::test]
    async fn test_hybrid_search_defaults_to_standard_mode() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid"
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_mode"], "standard");
    }

    #[tokio::test]
    async fn test_hybrid_search_with_invalid_performance_mode() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid",
            "performance_mode": "invalid_mode"
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid performance_mode"));
    }

    #[tokio::test]
    async fn test_simd_mode_configuration_passthrough() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "semantic",
            "performance_mode": "simd",
            "simd_config": {
                "use_simd": true,
                "distance_threshold": 0.9
            }
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_metrics"]["distance_threshold"], 0.9);
    }

    #[tokio::test]
    async fn test_lsh_mode_configuration_validation() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let params = json!({
            "query": "test query",
            "search_type": "hybrid",
            "performance_mode": "lsh",
            "lsh_config": {
                "hash_functions": 200, // Invalid: exceeds max of 128
                "hash_tables": 8
            }
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, params).await;

        // Assert
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hash_functions must be between"));
    }

    #[tokio::test]
    async fn test_performance_metrics_tracking() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        
        // Test all modes
        let modes = vec!["standard", "simd", "lsh"];
        
        for mode in modes {
            let params = json!({
                "query": "test query",
                "search_type": "hybrid",
                "performance_mode": mode
            });

            // Act
            let result = handle_hybrid_search(&engine, &stats, params).await;

            // Assert
            assert!(result.is_ok());
            let (data, _, _) = result.unwrap();
            
            // All modes should report performance metrics
            assert!(data["performance_metrics"]["execution_time_ms"].as_f64().is_some());
            assert!(data["performance_metrics"]["mode"].as_str().is_some());
            assert!(data["performance_metrics"]["vectors_processed"].as_u64().is_some());
        }
    }

    #[tokio::test]
    async fn test_backwards_compatibility_for_old_simd_tool() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        
        // Old simd_ultra_fast_search parameters
        let old_params = json!({
            "query_text": "test query",
            "top_k": 5,
            "distance_threshold": 0.7,
            "use_simd": true
        });

        // Convert to new format
        let new_params = json!({
            "query": old_params["query_text"],
            "search_type": "semantic",
            "performance_mode": "simd",
            "limit": old_params["top_k"],
            "simd_config": {
                "distance_threshold": old_params["distance_threshold"],
                "use_simd": old_params["use_simd"]
            }
        });

        // Act
        let result = handle_hybrid_search(&engine, &stats, new_params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        assert_eq!(data["performance_mode"], "simd");
        assert_eq!(data["result_count"], 5);
    }

    #[tokio::test]
    async fn test_performance_mode_with_different_search_types() {
        // Arrange
        let engine = create_test_engine().await;
        let stats = create_test_stats();
        let search_types = vec!["semantic", "structural", "keyword", "hybrid"];
        
        for search_type in search_types {
            let params = json!({
                "query": "test query",
                "search_type": search_type,
                "performance_mode": "simd" // Should work with all search types
            });

            // Act
            let result = handle_hybrid_search(&engine, &stats, params).await;

            // Assert
            assert!(result.is_ok(), "Failed for search_type: {}", search_type);
            let (data, _, _) = result.unwrap();
            assert_eq!(data["search_type"], search_type);
            assert_eq!(data["performance_mode"], "simd");
        }
    }
}