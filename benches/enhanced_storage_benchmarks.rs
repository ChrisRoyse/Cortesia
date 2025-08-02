//! Enhanced Knowledge Storage System Benchmarks
//!
//! Comprehensive performance benchmarks measuring model loading, knowledge processing,
//! retrieval performance, and improvements over baseline implementations.

use criterion::{criterion_group, criterion_main};

mod enhanced_knowledge_storage;

use enhanced_knowledge_storage::{
    model_loading::*,
    knowledge_processing::*,
    retrieval_performance::*,
    baseline_comparison::*,
};

criterion_group!(
    model_management_benches,
    benchmark_model_loading_small,
    benchmark_model_loading_medium,
    benchmark_concurrent_model_loading,
    benchmark_lru_eviction_performance,
    benchmark_model_cache_operations,
    benchmark_resource_monitoring
);

criterion_group!(
    knowledge_processing_benches,
    benchmark_document_processing_1kb,
    benchmark_document_processing_10kb,
    benchmark_document_processing_100kb,
    benchmark_document_processing_1mb,
    benchmark_entity_extraction_rate,
    benchmark_semantic_chunking_throughput,
    benchmark_relationship_mapping,
    benchmark_context_analysis
);

criterion_group!(
    retrieval_benches,
    benchmark_simple_query_latency,
    benchmark_multi_hop_reasoning_2hops,
    benchmark_multi_hop_reasoning_5hops,
    benchmark_graph_scaling_1k,
    benchmark_graph_scaling_10k,
    benchmark_graph_scaling_100k,
    benchmark_context_aggregation,
    benchmark_query_optimization
);

criterion_group!(
    comparison_benches,
    benchmark_baseline_vs_enhanced_chunking,
    benchmark_baseline_vs_enhanced_extraction,
    benchmark_fixed_vs_semantic_chunking,
    benchmark_pattern_vs_ai_extraction,
    benchmark_memory_usage_comparison,
    benchmark_processing_speed_comparison
);

criterion_main!(
    model_management_benches,
    knowledge_processing_benches,
    retrieval_benches,
    comparison_benches
);