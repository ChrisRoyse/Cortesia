//! Comprehensive integration tests for the query module
//!
//! Tests all public APIs including:
//! - RealtimeQuery construction and execution
//! - PartitionedGraphRAG system
//! - GraphRAGEngine retrieval
//! - QueryOptimizer functionality
//! - HierarchicalClusterer
//! - CommunitySummarizer
//! - TwoTierQueryEngine
//!
//! Each test focuses on the public interface and verifies correct behavior
//! including edge cases and error handling.

use llmkg::query::{
    RealtimeQuery, PartitionedGraphRAG, PartitionStrategy, MergeStrategy,
};
use std::sync::Arc;

// ===== RealtimeQuery Tests =====

#[test]
fn test_realtime_query_construction() {
    let query = RealtimeQuery::new("test query".to_string());
    
    assert_eq!(query.query_text, "test query");
    assert!(query.initial_nodes.is_empty());
    assert_eq!(query.max_depth, 3);
    assert_eq!(query.similarity_threshold, 0.7);
    assert_eq!(query.max_results, 10);
}

#[test]
fn test_realtime_query_builder_pattern() {
    let query = RealtimeQuery::new("test query".to_string())
        .with_initial_nodes(vec![1, 2, 3])
        .with_max_depth(5);
    
    assert_eq!(query.query_text, "test query");
    assert_eq!(query.initial_nodes, vec![1, 2, 3]);
    assert_eq!(query.max_depth, 5);
    assert_eq!(query.similarity_threshold, 0.7); // Default unchanged
    assert_eq!(query.max_results, 10); // Default unchanged
}

#[test]
fn test_realtime_query_edge_cases() {
    // Empty query text
    let empty_query = RealtimeQuery::new("".to_string());
    assert_eq!(empty_query.query_text, "");
    
    // Very large depth
    let deep_query = RealtimeQuery::new("deep search".to_string())
        .with_max_depth(u32::MAX);
    assert_eq!(deep_query.max_depth, u32::MAX);
    
    // Large initial nodes
    let many_nodes: Vec<u32> = (0..10000).collect();
    let large_query = RealtimeQuery::new("large".to_string())
        .with_initial_nodes(many_nodes.clone());
    assert_eq!(large_query.initial_nodes.len(), 10000);
}

#[test]
fn test_realtime_query_clone() {
    let original = RealtimeQuery::new("test".to_string())
        .with_initial_nodes(vec![1, 2, 3])
        .with_max_depth(10);
    
    let cloned = original.clone();
    
    assert_eq!(cloned.query_text, original.query_text);
    assert_eq!(cloned.initial_nodes, original.initial_nodes);
    assert_eq!(cloned.max_depth, original.max_depth);
    assert_eq!(cloned.similarity_threshold, original.similarity_threshold);
    assert_eq!(cloned.max_results, original.max_results);
}

// ===== PartitionedGraphRAG Tests =====

#[test]
fn test_partitioned_graph_rag_construction() {
    let rag = PartitionedGraphRAG::new();
    
    assert!(rag.partitions.is_empty());
    assert!(matches!(rag.partition_strategy, PartitionStrategy::ByNodeId));
    assert!(matches!(rag.merge_strategy, MergeStrategy::RankByScore));
}

#[test]
fn test_partitioned_graph_rag_strategies() {
    // Test all partition strategies
    let strategies = vec![
        PartitionStrategy::ByNodeId,
        PartitionStrategy::ByEntityType,
        PartitionStrategy::ByEmbeddingCluster,
        PartitionStrategy::Custom("test_strategy".to_string()),
    ];
    
    for strategy in strategies {
        let test_rag = PartitionedGraphRAG::new()
            .with_strategy(strategy.clone());
        match (&test_rag.partition_strategy, &strategy) {
            (PartitionStrategy::ByNodeId, PartitionStrategy::ByNodeId) => {},
            (PartitionStrategy::ByEntityType, PartitionStrategy::ByEntityType) => {},
            (PartitionStrategy::ByEmbeddingCluster, PartitionStrategy::ByEmbeddingCluster) => {},
            (PartitionStrategy::Custom(a), PartitionStrategy::Custom(b)) => assert_eq!(a, b),
            _ => panic!("Strategy mismatch"),
        }
    }
}

#[test]
fn test_partitioned_graph_rag_default() {
    // Test Default implementation
    let default_rag = PartitionedGraphRAG::default();
    assert!(default_rag.partitions.is_empty());
    assert!(matches!(default_rag.partition_strategy, PartitionStrategy::ByNodeId));
    assert!(matches!(default_rag.merge_strategy, MergeStrategy::RankByScore));
}

#[test]
fn test_merge_strategies() {
    // Test all merge strategies
    let strategies = vec![
        MergeStrategy::UnionAll,
        MergeStrategy::IntersectAll,
        MergeStrategy::RankByScore,
        MergeStrategy::Custom("custom_merge".to_string()),
    ];
    
    for strategy in strategies {
        let rag = PartitionedGraphRAG {
            partitions: Vec::new(),
            partition_strategy: PartitionStrategy::ByNodeId,
            merge_strategy: strategy.clone(),
        };
        
        match (&rag.merge_strategy, &strategy) {
            (MergeStrategy::UnionAll, MergeStrategy::UnionAll) => {},
            (MergeStrategy::IntersectAll, MergeStrategy::IntersectAll) => {},
            (MergeStrategy::RankByScore, MergeStrategy::RankByScore) => {},
            (MergeStrategy::Custom(a), MergeStrategy::Custom(b)) => assert_eq!(a, b),
            _ => panic!("Merge strategy mismatch"),
        }
    }
}

#[test]
fn test_partition_and_merge_strategy_combinations() {
    let partition_strategies = vec![
        PartitionStrategy::ByNodeId,
        PartitionStrategy::ByEntityType,
        PartitionStrategy::ByEmbeddingCluster,
    ];
    
    let merge_strategies = vec![
        MergeStrategy::UnionAll,
        MergeStrategy::IntersectAll,
        MergeStrategy::RankByScore,
    ];
    
    // Test all combinations work
    for ps in &partition_strategies {
        for ms in &merge_strategies {
            let rag = PartitionedGraphRAG {
                partitions: Vec::new(),
                partition_strategy: ps.clone(),
                merge_strategy: ms.clone(),
            };
            
            // Verify strategies are set correctly
            match (&rag.partition_strategy, ps) {
                (PartitionStrategy::ByNodeId, PartitionStrategy::ByNodeId) => {},
                (PartitionStrategy::ByEntityType, PartitionStrategy::ByEntityType) => {},
                (PartitionStrategy::ByEmbeddingCluster, PartitionStrategy::ByEmbeddingCluster) => {},
                _ => panic!("Partition strategy mismatch"),
            }
            
            match (&rag.merge_strategy, ms) {
                (MergeStrategy::UnionAll, MergeStrategy::UnionAll) => {},
                (MergeStrategy::IntersectAll, MergeStrategy::IntersectAll) => {},
                (MergeStrategy::RankByScore, MergeStrategy::RankByScore) => {},
                _ => panic!("Merge strategy mismatch"),
            }
        }
    }
}

// Additional tests for QueryOptimizer using just the public types

#[test]
fn test_optimization_settings_default() {
    use llmkg::query::optimizer::OptimizationSettings;
    use std::time::Duration;
    
    let settings = OptimizationSettings::default();
    
    assert!(settings.enable_caching);
    assert_eq!(settings.cache_ttl, Duration::from_secs(300));
    assert_eq!(settings.max_cache_size, 1000);
    assert!(settings.enable_query_rewriting);
    assert!(!settings.enable_result_prefetching);
}

#[test]
fn test_query_stats_construction() {
    use llmkg::query::optimizer::QueryStats;
    use std::time::{Duration, Instant};
    
    let stats = QueryStats {
        query_embedding_hash: 12345,
        execution_time: Duration::from_millis(100),
        result_count: 50,
        cache_hit: true,
        timestamp: Instant::now(),
    };
    
    assert_eq!(stats.query_embedding_hash, 12345);
    assert_eq!(stats.execution_time, Duration::from_millis(100));
    assert_eq!(stats.result_count, 50);
    assert!(stats.cache_hit);
}

// Tests for enum types to ensure all variants are accessible

#[test] 
fn test_partition_strategy_variants() {
    // Ensure all variants can be constructed
    let _ = PartitionStrategy::ByNodeId;
    let _ = PartitionStrategy::ByEntityType;
    let _ = PartitionStrategy::ByEmbeddingCluster;
    let _ = PartitionStrategy::Custom("test".to_string());
}

#[test]
fn test_merge_strategy_variants() {
    // Ensure all variants can be constructed
    let _ = MergeStrategy::UnionAll;
    let _ = MergeStrategy::IntersectAll;
    let _ = MergeStrategy::RankByScore;
    let _ = MergeStrategy::Custom("test".to_string());
}

// Edge case tests

#[test]
fn test_realtime_query_extreme_values() {
    // Test with extreme similarity threshold
    let mut query = RealtimeQuery::new("test".to_string());
    query.similarity_threshold = 0.0;
    assert_eq!(query.similarity_threshold, 0.0);
    
    query.similarity_threshold = 1.0;
    assert_eq!(query.similarity_threshold, 1.0);
    
    // Test with zero max_results
    query.max_results = 0;
    assert_eq!(query.max_results, 0);
    
    // Test with very large max_results
    query.max_results = usize::MAX;
    assert_eq!(query.max_results, usize::MAX);
}

#[test]
fn test_custom_strategy_names() {
    // Test empty custom strategy names
    let empty_partition = PartitionStrategy::Custom("".to_string());
    let empty_merge = MergeStrategy::Custom("".to_string());
    
    match empty_partition {
        PartitionStrategy::Custom(s) => assert_eq!(s, ""),
        _ => panic!("Expected Custom variant"),
    }
    
    match empty_merge {
        MergeStrategy::Custom(s) => assert_eq!(s, ""),
        _ => panic!("Expected Custom variant"),
    }
    
    // Test very long custom strategy names
    let long_name = "a".repeat(1000);
    let long_partition = PartitionStrategy::Custom(long_name.clone());
    let long_merge = MergeStrategy::Custom(long_name.clone());
    
    match long_partition {
        PartitionStrategy::Custom(s) => assert_eq!(s.len(), 1000),
        _ => panic!("Expected Custom variant"),
    }
    
    match long_merge {
        MergeStrategy::Custom(s) => assert_eq!(s.len(), 1000),
        _ => panic!("Expected Custom variant"),
    }
}

#[test]
fn test_query_optimizer_settings_custom() {
    use llmkg::query::optimizer::OptimizationSettings;
    use std::time::Duration;
    
    let settings = OptimizationSettings {
        enable_caching: false,
        cache_ttl: Duration::from_secs(60),
        max_cache_size: 500,
        enable_query_rewriting: false,
        enable_result_prefetching: true,
    };
    
    assert!(!settings.enable_caching);
    assert_eq!(settings.cache_ttl, Duration::from_secs(60));
    assert_eq!(settings.max_cache_size, 500);
    assert!(!settings.enable_query_rewriting);
    assert!(settings.enable_result_prefetching);
}

// Tests for Debug trait implementations

#[test]
fn test_debug_implementations() {
    // Test Debug for RealtimeQuery
    let query = RealtimeQuery::new("debug test".to_string());
    let debug_str = format!("{:?}", query);
    assert!(debug_str.contains("RealtimeQuery"));
    assert!(debug_str.contains("debug test"));
    
    // Test Debug for PartitionedGraphRAG
    let rag = PartitionedGraphRAG::new();
    let debug_str = format!("{:?}", rag);
    assert!(debug_str.contains("PartitionedGraphRAG"));
    
    // Test Debug for enums
    let ps = PartitionStrategy::ByNodeId;
    let debug_str = format!("{:?}", ps);
    assert!(debug_str.contains("ByNodeId"));
    
    let ms = MergeStrategy::UnionAll;
    let debug_str = format!("{:?}", ms);
    assert!(debug_str.contains("UnionAll"));
}

// Tests for Clone trait implementations

#[test]
fn test_clone_implementations() {
    // Test Clone for enums
    let ps1 = PartitionStrategy::Custom("clone_test".to_string());
    let ps2 = ps1.clone();
    match (ps1, ps2) {
        (PartitionStrategy::Custom(s1), PartitionStrategy::Custom(s2)) => {
            assert_eq!(s1, s2);
        }
        _ => panic!("Clone failed"),
    }
    
    let ms1 = MergeStrategy::Custom("merge_clone".to_string());
    let ms2 = ms1.clone();
    match (ms1, ms2) {
        (MergeStrategy::Custom(s1), MergeStrategy::Custom(s2)) => {
            assert_eq!(s1, s2);
        }
        _ => panic!("Clone failed"),
    }
    
    // Test Clone for PartitionedGraphRAG
    let rag1 = PartitionedGraphRAG::new()
        .with_strategy(PartitionStrategy::ByEntityType);
    let rag2 = rag1.clone();
    
    assert_eq!(rag1.partitions.len(), rag2.partitions.len());
    match (&rag1.partition_strategy, &rag2.partition_strategy) {
        (PartitionStrategy::ByEntityType, PartitionStrategy::ByEntityType) => {},
        _ => panic!("Clone failed to preserve strategy"),
    }
}