//! Knowledge Processing Performance Benchmarks
//!
//! Measures the performance of document processing, entity extraction, semantic chunking,
//! relationship mapping, and context analysis across different document sizes and types.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use llmkg::enhanced_knowledge_storage::{
    knowledge_processing::{IntelligentKnowledgeProcessor, KnowledgeProcessingConfig},
    model_management::{ModelResourceManager, ModelResourceConfig},
    types::*,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create test documents of various sizes
fn create_test_document(size_bytes: usize) -> String {
    let base_content = "This is a test document with various entities like OpenAI, artificial intelligence, machine learning algorithms, and neural networks. \
                      The document discusses the implementation of transformer models, attention mechanisms, and their applications in natural language processing. \
                      Key concepts include embeddings, tokenization, fine-tuning, and model deployment strategies. \
                      Organizations mentioned include Google, Microsoft, Meta, and various research institutions. \
                      Technical terms encompass BERT, GPT, T5, and other state-of-the-art architectures.";
    
    let mut document = String::with_capacity(size_bytes);
    while document.len() < size_bytes {
        document.push_str(base_content);
        document.push(' ');
    }
    
    document.truncate(size_bytes);
    document
}

/// Create a test configuration optimized for benchmarking
fn benchmark_config() -> ModelResourceConfig {
    ModelResourceConfig {
        max_memory_usage: 8_000_000_000, // 8GB for benchmarks
        max_concurrent_models: 3,
        idle_timeout: Duration::from_secs(600),
        min_memory_threshold: 200_000_000,
    }
}

/// Create intelligent processor for benchmarking
fn create_test_processor() -> IntelligentKnowledgeProcessor {
    let config = benchmark_config();
    let resource_manager = Arc::new(ModelResourceManager::new(config));
    let processing_config = KnowledgeProcessingConfig::default();
    IntelligentKnowledgeProcessor::new(resource_manager, processing_config)
}

/// Benchmark document processing with 1KB documents
pub fn benchmark_document_processing_1kb(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let document = create_test_document(1_024);
    
    c.benchmark_group("document_processing")
        .throughput(Throughput::Bytes(1_024))
        .bench_function("process_1kb_document", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&document, "benchmark_1kb").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark document processing with 10KB documents
pub fn benchmark_document_processing_10kb(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let document = create_test_document(10_240);
    
    c.benchmark_group("document_processing")
        .throughput(Throughput::Bytes(10_240))
        .bench_function("process_10kb_document", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&document, "benchmark_10kb").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark document processing with 100KB documents
pub fn benchmark_document_processing_100kb(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let document = create_test_document(102_400);
    
    c.benchmark_group("document_processing")
        .throughput(Throughput::Bytes(102_400))
        .bench_function("process_100kb_document", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&document, "benchmark_100kb").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark document processing with 1MB documents
pub fn benchmark_document_processing_1mb(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let document = create_test_document(1_048_576);
    
    c.benchmark_group("document_processing")
        .throughput(Throughput::Bytes(1_048_576))
        .bench_function("process_1mb_document", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&document, "benchmark_1mb").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark entity extraction rate across different document complexities
pub fn benchmark_entity_extraction_rate(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("entity_extraction");
    
    // Test documents with different entity densities
    let sparse_doc = "This is a simple document with few entities like OpenAI.".repeat(100);
    let dense_doc = "OpenAI developed GPT-4 with Microsoft Azure. Google created BERT and PaLM. Meta built LLaMA. Stanford Research Institute, MIT, and Carnegie Mellon contributed significantly.".repeat(50);
    
    group.bench_function("sparse_entities", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            // Process the document which includes entity extraction
            let result = processor.process_knowledge(&sparse_doc, "sparse_entities").await;
            black_box(result.unwrap());
        });
    });
    
    group.bench_function("dense_entities", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            // Process the document which includes entity extraction
            let result = processor.process_knowledge(&dense_doc, "dense_entities").await;
            black_box(result.unwrap());
        });
    });
    
    group.finish();
}

/// Benchmark semantic chunking throughput
pub fn benchmark_semantic_chunking_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("semantic_chunking");
    
    // Test different content types
    let narrative_content = "Once upon a time, in a land far away, there lived a wise old wizard. The wizard possessed great knowledge of artificial intelligence and machine learning. He taught young apprentices about neural networks, transformers, and the mysteries of deep learning. Each day brought new discoveries and insights into the world of AI.".repeat(20);
    
    let technical_content = "The transformer architecture consists of an encoder-decoder structure with self-attention mechanisms. The multi-head attention allows the model to focus on different positions simultaneously. Layer normalization and residual connections stabilize training. The feed-forward networks process representations at each position independently.".repeat(20);
    
    let mixed_content = format!("{} {} {}", narrative_content, technical_content, "Additional context about implementation details, performance metrics, and optimization strategies.");
    
    for (name, content) in [
        ("narrative", &narrative_content),
        ("technical", &technical_content),
        ("mixed", &mixed_content),
    ] {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(BenchmarkId::new("chunking_type", name), content, |b, content| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                // Process the document which includes semantic chunking
                let result = processor.process_knowledge(content, &format!("chunking_{}", name)).await;
                black_box(result.unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark relationship mapping performance
pub fn benchmark_relationship_mapping(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create test entities for relationship mapping
    let entities = vec![
        ("OpenAI", "organization"),
        ("GPT-4", "model"),
        ("ChatGPT", "product"),
        ("artificial intelligence", "concept"),
        ("machine learning", "concept"),
        ("neural networks", "concept"),
        ("San Francisco", "location"),
        ("2022", "date"),
    ];
    
    let context = "OpenAI, based in San Francisco, developed GPT-4 in 2022. This model powers ChatGPT and represents a significant advancement in artificial intelligence and machine learning, particularly in neural networks.";
    
    c.bench_function("relationship_mapping", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            // Process the context which includes relationship mapping
            let result = processor.process_knowledge(context, "relationship_mapping").await;
            black_box(result.unwrap());
        });
    });
}

/// Benchmark context analysis performance
pub fn benchmark_context_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("context_analysis");
    
    // Different complexity contexts
    let simple_context = "This is a basic document about AI.";
    let medium_context = "This document explores the relationship between artificial intelligence, machine learning, and neural networks in modern applications.";
    let complex_context = "This comprehensive analysis examines the intricate relationships between transformer architectures, attention mechanisms, and their applications in large language models, considering both technical implementations and societal implications across various domains including healthcare, education, and scientific research.";
    
    for (complexity, context) in [
        (ComplexityLevel::Low, simple_context),
        (ComplexityLevel::Medium, medium_context),
        (ComplexityLevel::High, complex_context),
    ] {
        group.bench_with_input(
            BenchmarkId::new("complexity", format!("{:?}", complexity)),
            &(complexity, context),
            |b, &(complexity, context)| {
                b.to_async(&rt).iter(|| async {
                    let processor = create_test_processor();
                    // Process the context which includes context analysis
                    let result = processor.process_knowledge(context, &format!("context_{:?}", complexity)).await;
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_creation() {
        let doc_1kb = create_test_document(1024);
        assert_eq!(doc_1kb.len(), 1024);
        
        let doc_10kb = create_test_document(10240);
        assert_eq!(doc_10kb.len(), 10240);
        
        assert!(doc_1kb.contains("artificial intelligence"));
        assert!(doc_10kb.contains("transformer models"));
    }
    
    #[tokio::test]
    async fn test_processor_creation() {
        let processor = create_test_processor();
        
        // Test basic functionality
        let simple_doc = "This is a test document about AI and machine learning.";
        let result = processor.process_knowledge(simple_doc, "test").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_entity_extraction_benchmark_setup() {
        let processor = create_test_processor();
        
        let test_text = "OpenAI developed GPT-4 with Microsoft Azure support.";
        let result = processor.process_knowledge(test_text, "entity_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.global_entities.is_empty());
    }
    
    #[tokio::test]
    async fn test_semantic_chunking_benchmark_setup() {
        let processor = create_test_processor();
        
        let test_content = create_test_document(2048);
        let result = processor.process_knowledge(&test_content, "chunking_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.chunks.is_empty());
    }
    
    #[tokio::test]
    async fn test_relationship_mapping_benchmark_setup() {
        let processor = create_test_processor();
        
        let context = "OpenAI developed GPT-4 in 2023.";
        let result = processor.process_knowledge(context, "relationship_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should have extracted relationships
        assert!(!result.global_relationships.is_empty() || !result.global_entities.is_empty());
    }
    
    #[tokio::test]
    async fn test_context_analysis_benchmark_setup() {
        let processor = create_test_processor();
        
        let context = "This document discusses AI and ML applications.";
        let result = processor.process_knowledge(context, "context_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should have processed the context successfully
        assert!(!result.chunks.is_empty());
    }
}