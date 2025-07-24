//! TDD tests for enhanced generate_graph_query implementation
//! Red-Green-Refactor cycle: Write failing tests first

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::handlers::advanced::handle_generate_graph_query;
use crate::mcp::llm_friendly_server::types::UsageStats;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
mod generate_graph_query_tests {
    use super::*;

    // Helper function to create test params
    fn create_test_params(query: &str, language: &str) -> serde_json::Value {
        json!({
            "natural_query": query,
            "query_language": language,
            "include_explanation": true
        })
    }

    #[tokio::test]
    async fn test_extract_single_entity_about_pattern() {
        // Arrange
        let params = create_test_params("Find all facts about Einstein", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, message, _) = result.unwrap();
        
        // Check extracted entities
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].as_str().unwrap(), "Einstein");
        
        // Check generated query contains Einstein
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("Einstein"));
        assert!(query.contains("MATCH"));
    }

    #[tokio::test]
    async fn test_extract_multiple_entities_between_pattern() {
        // Arrange
        let params = create_test_params("Show relationships between Einstein and Newton", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&json!("Einstein")));
        assert!(entities.contains(&json!("Newton")));
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("Einstein"));
        assert!(query.contains("Newton"));
        assert!(query.contains("path") || query.contains("PATH"));
    }

    #[tokio::test]
    async fn test_extract_possessive_pattern() {
        // Arrange
        let params = create_test_params("What are Einstein's discoveries?", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert!(entities.contains(&json!("Einstein")));
    }

    #[tokio::test]
    async fn test_extract_quoted_entities() {
        // Arrange
        let params = create_test_params("Find information about \"Theory of Relativity\"", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert!(entities.contains(&json!("Theory of Relativity")));
    }

    #[tokio::test]
    async fn test_complex_query_with_multiple_patterns() {
        // Arrange
        let params = create_test_params(
            "Show connections between Einstein and Tesla related to electricity", 
            "cypher"
        );
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert!(entities.contains(&json!("Einstein")));
        assert!(entities.contains(&json!("Tesla")));
        
        // electricity might be extracted as lowercase, so check both
        assert!(entities.contains(&json!("electricity")) || 
                entities.iter().any(|e| e.as_str().unwrap().to_lowercase() == "electricity"));
    }

    #[tokio::test]
    async fn test_sparql_query_generation() {
        // Arrange
        let params = create_test_params("Find all facts about Einstein", "sparql");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("SELECT"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("Einstein"));
    }

    #[tokio::test]
    async fn test_gremlin_query_generation() {
        // Arrange
        let params = create_test_params("Find all facts about Einstein", "gremlin");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("g.V()"));
        assert!(query.contains("Einstein"));
    }

    #[tokio::test]
    async fn test_shortest_path_query() {
        // Arrange
        let params = create_test_params("Find the shortest path between Einstein and Tesla", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("shortestPath") || query.contains("shortest"));
        assert!(query.contains("Einstein"));
        assert!(query.contains("Tesla"));
    }

    #[tokio::test]
    async fn test_count_query() {
        // Arrange
        let params = create_test_params("How many scientists are there?", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("COUNT") || query.contains("count"));
    }

    #[tokio::test]
    async fn test_who_what_questions() {
        // Arrange
        let params = create_test_params("Who invented the telephone?", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("invented") || query.contains("INVENTED"));
    }

    #[tokio::test]
    async fn test_empty_query_fallback() {
        // Arrange
        let params = create_test_params("", "cypher");
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert_eq!(entities.len(), 0);
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("LIMIT")); // Fallback query should have a limit
    }

    #[tokio::test]
    async fn test_explanation_included() {
        // Arrange
        let params = json!({
            "natural_query": "Find all facts about Einstein",
            "query_language": "cypher",
            "include_explanation": true
        });
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        assert!(data.get("explanation").is_some());
        let explanation = data.get("explanation").unwrap().as_str().unwrap();
        assert!(!explanation.is_empty());
    }

    #[tokio::test]
    async fn test_explanation_excluded() {
        // Arrange
        let params = json!({
            "natural_query": "Find all facts about Einstein",
            "query_language": "cypher",
            "include_explanation": false
        });
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        // When include_explanation is false, explanation should be null or empty
        if let Some(explanation) = data.get("explanation") {
            assert!(explanation.is_null() || explanation.as_str().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn test_complex_filter_query() {
        // Arrange
        let params = create_test_params(
            "Find all scientists that are from Germany with Nobel prizes", 
            "cypher"
        );
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;

        // Assert
        assert!(result.is_ok());
        let (data, _, _) = result.unwrap();
        
        let entities = data.get("extracted_entities").unwrap().as_array().unwrap();
        assert!(entities.contains(&json!("Germany")));
        assert!(entities.contains(&json!("Nobel")));
        
        let query = data.get("generated_query").unwrap().as_str().unwrap();
        assert!(query.contains("WHERE") || query.contains("where"));
    }

    #[tokio::test]
    async fn test_performance_under_100ms() {
        // Arrange
        let params = create_test_params(
            "Find all relationships between Einstein, Tesla, and Newton related to physics", 
            "cypher"
        );
        let knowledge_engine = Arc::new(RwLock::new(KnowledgeEngine::new(256, 10000).unwrap()));
        let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

        // Act
        let start = std::time::Instant::now();
        let result = handle_generate_graph_query(&knowledge_engine, &usage_stats, params).await;
        let duration = start.elapsed();

        // Assert
        assert!(result.is_ok());
        assert!(duration.as_millis() < 100, "Query generation took {}ms, should be under 100ms", duration.as_millis());
    }
}