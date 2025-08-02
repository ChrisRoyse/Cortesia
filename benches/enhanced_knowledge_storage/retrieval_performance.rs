//! Retrieval System Performance Benchmarks
//!
//! Measures query processing performance and scaling behavior using the 
//! knowledge processing system as a proxy for retrieval performance.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use llmkg::enhanced_knowledge_storage::{
    knowledge_processing::{IntelligentKnowledgeProcessor, KnowledgeProcessingConfig},
    model_management::{ModelResourceManager, ModelResourceConfig},
    types::*,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create a test configuration for retrieval benchmarks
fn retrieval_benchmark_config() -> ModelResourceConfig {
    ModelResourceConfig {
        max_memory_usage: 6_000_000_000, // 6GB for retrieval benchmarks
        max_concurrent_models: 4,
        idle_timeout: Duration::from_secs(900),
        min_memory_threshold: 300_000_000,
    }
}

/// Create a test processor for retrieval benchmarks
fn create_test_processor() -> IntelligentKnowledgeProcessor {
    let config = retrieval_benchmark_config();
    let resource_manager = Arc::new(ModelResourceManager::new(config));
    let processing_config = KnowledgeProcessingConfig::default();
    IntelligentKnowledgeProcessor::new(resource_manager, processing_config)
}

/// Create test documents of different sizes for scaling tests
fn create_test_knowledge_base(size: usize) -> String {
    let base_entities = vec![
        "OpenAI develops artificial intelligence",
        "Microsoft collaborates with OpenAI",
        "Google Research creates machine learning models",
        "Meta focuses on social media AI",
        "Apple integrates AI into consumer products",
        "Stanford University conducts AI research",
        "MIT develops robotics and AI",
        "Carnegie Mellon advances computer vision",
        "Berkeley studies natural language processing",
        "DeepMind creates game-playing AI",
    ];
    
    let mut knowledge_base = String::new();
    for i in 0..size {
        let entity_info = &base_entities[i % base_entities.len()];
        knowledge_base.push_str(&format!(
            "Entity {}: {}. This represents important knowledge about technological advancement and research collaboration in the field of artificial intelligence. ",
            i, entity_info
        ));
    }
    
    knowledge_base
}

/// Benchmark simple query processing (represented by document processing)
pub fn benchmark_simple_query_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let test_document = create_test_knowledge_base(100);
    
    c.bench_function("simple_query_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            let result = processor.process_knowledge(&test_document, "simple_query").await;
            black_box(result.unwrap());
        });
    });
}

/// Benchmark multi-hop reasoning (simulated through complex document processing)
pub fn benchmark_multi_hop_reasoning_2hops(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create a document that requires understanding relationships between entities
    let complex_document = "OpenAI was founded by Sam Altman and others. Microsoft invested in OpenAI. OpenAI created GPT-4. GPT-4 powers ChatGPT. ChatGPT is used by millions of users. Therefore, Microsoft's investment indirectly benefits millions of users through the AI technology chain.";
    
    c.bench_function("multi_hop_reasoning_2hops", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            let result = processor.process_knowledge(complex_document, "multi_hop_2").await;
            black_box(result.unwrap());
        });
    });
}

/// Benchmark multi-hop reasoning with more complex chains
pub fn benchmark_multi_hop_reasoning_5hops(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create a document with a longer chain of relationships
    let very_complex_document = "
        Research institutions like Stanford and MIT conduct fundamental AI research.
        Their research leads to publications and PhD graduates.
        PhD graduates join companies like OpenAI, Google, and Microsoft.
        These companies develop AI products and services.
        AI products are integrated into consumer applications.
        Consumer applications are used by businesses worldwide.
        Business adoption drives economic growth and productivity gains.
        This creates a chain from academic research to global economic impact.
    ";
    
    c.bench_function("multi_hop_reasoning_5hops", |b| {
        b.to_async(&rt).iter(|| async {
            let processor = create_test_processor();
            let result = processor.process_knowledge(very_complex_document, "multi_hop_5").await;
            black_box(result.unwrap());
        });
    });
}

/// Benchmark scaling with 1K entities
pub fn benchmark_graph_scaling_1k(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let knowledge_base = create_test_knowledge_base(1_000);
    
    c.benchmark_group("graph_scaling")
        .throughput(Throughput::Elements(1_000))
        .bench_function("retrieval_1k_nodes", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&knowledge_base, "scaling_1k").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark scaling with 10K entities
pub fn benchmark_graph_scaling_10k(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let knowledge_base = create_test_knowledge_base(10_000);
    
    c.benchmark_group("graph_scaling")
        .throughput(Throughput::Elements(10_000))
        .bench_function("retrieval_10k_nodes", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&knowledge_base, "scaling_10k").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark scaling with 100K entities
pub fn benchmark_graph_scaling_100k(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let knowledge_base = create_test_knowledge_base(100_000);
    
    c.benchmark_group("graph_scaling")
        .throughput(Throughput::Elements(100_000))
        .bench_function("retrieval_100k_nodes", |b| {
            b.to_async(&rt).iter(|| async {
                let processor = create_test_processor();
                let result = processor.process_knowledge(&knowledge_base, "scaling_100k").await;
                black_box(result.unwrap());
            });
        });
}

/// Benchmark context aggregation through document processing
pub fn benchmark_context_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("context_aggregation");
    
    // Test different amounts of contextual information
    for &context_count in &[5, 20, 50, 100] {
        let context_document = create_test_knowledge_base(context_count);
        
        group.bench_with_input(
            BenchmarkId::new("context_pieces", context_count),
            &context_document,
            |b, document| {
                b.to_async(&rt).iter(|| async {
                    let processor = create_test_processor();
                    let result = processor.process_knowledge(document, &format!("context_{}", context_count)).await;
                    black_box(result.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark query optimization through different processing approaches
pub fn benchmark_query_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("query_optimization");
    
    let queries = vec![
        ("simple", "What is OpenAI and what do they do?"),
        ("complex", "How do research institutions contribute to AI development and what is the economic impact?"),
        ("analytical", "Compare the relationships between different technology companies and their AI strategies"),
        ("multi_constraint", "Find connections between academic research, commercial development, and consumer applications"),
    ];
    
    // Create documents that would answer these queries
    let simple_doc = "OpenAI is an artificial intelligence research company that develops advanced AI systems.";
    let complex_doc = create_test_knowledge_base(1000);
    let analytical_doc = "Google focuses on search and advertising AI. Microsoft emphasizes productivity AI. Meta develops social media AI. Apple integrates AI into consumer devices. Each has different strategic approaches.";
    let multi_constraint_doc = "Universities publish research papers. Companies hire PhD graduates. Products use research insights. Consumers benefit from AI applications. This creates an ecosystem of knowledge transfer.";
    
    let documents = vec![simple_doc, &complex_doc, analytical_doc, multi_constraint_doc];
    
    for ((query_type, _query_text), document) in queries.iter().zip(documents.iter()) {
        group.bench_with_input(
            BenchmarkId::new("query_type", query_type),
            document,
            |b, &document| {
                b.to_async(&rt).iter(|| async {
                    let processor = create_test_processor();
                    let result = processor.process_knowledge(document, &format!("opt_{}", query_type)).await;
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
    
    #[tokio::test]
    async fn test_processor_creation() {
        let processor = create_test_processor();
        
        let simple_doc = "Test document for processor creation.";
        let result = processor.process_knowledge(simple_doc, "test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.chunks.is_empty());
    }
    
    #[test]
    fn test_knowledge_base_creation() {
        let kb_small = create_test_knowledge_base(10);
        let kb_large = create_test_knowledge_base(1000);
        
        assert!(!kb_small.is_empty());
        assert!(!kb_large.is_empty());
        assert!(kb_large.len() > kb_small.len());
        
        // Should contain expected entities
        assert!(kb_small.contains("OpenAI"));
        assert!(kb_large.contains("artificial intelligence"));
    }
    
    #[tokio::test]
    async fn test_multi_hop_reasoning_setup() {
        let processor = create_test_processor();
        
        let complex_doc = "OpenAI created GPT-4. GPT-4 powers ChatGPT. ChatGPT serves millions of users.";
        let result = processor.process_knowledge(complex_doc, "multi_hop_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        
        // Should extract entities and relationships
        assert!(!result.global_entities.is_empty());
        assert!(!result.global_relationships.is_empty());
    }
    
    #[tokio::test]
    async fn test_scaling_benchmark_setup() {
        let processor = create_test_processor();
        
        // Test different scales
        let small_kb = create_test_knowledge_base(50);
        let medium_kb = create_test_knowledge_base(500);
        
        let small_result = processor.process_knowledge(&small_kb, "small_scale").await;
        let medium_result = processor.process_knowledge(&medium_kb, "medium_scale").await;
        
        assert!(small_result.is_ok());
        assert!(medium_result.is_ok());
        
        let small_result = small_result.unwrap();
        let medium_result = medium_result.unwrap();
        
        // Larger document should produce more entities/chunks
        assert!(medium_result.global_entities.len() >= small_result.global_entities.len());
        assert!(medium_result.chunks.len() >= small_result.chunks.len());
    }
    
    #[tokio::test]
    async fn test_context_aggregation_setup() {
        let processor = create_test_processor();
        
        let context_doc = create_test_knowledge_base(20);
        let result = processor.process_knowledge(&context_doc, "context_test").await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        
        // Should have processed context successfully
        assert!(!result.chunks.is_empty());
        assert!(!result.global_entities.is_empty());
        
        // Should have reasonable quality metrics
        assert!(result.quality_metrics.overall_quality > 0.0);
    }
    
    #[tokio::test]
    async fn test_query_optimization_setup() {
        let processor = create_test_processor();
        
        let test_queries = vec![
            "Simple test query document",
            "Complex document with multiple entities like OpenAI, Google, and Microsoft working on AI",
        ];
        
        for (i, doc) in test_queries.iter().enumerate() {
            let result = processor.process_knowledge(doc, &format!("query_opt_{}", i)).await;
            assert!(result.is_ok());
            
            let result = result.unwrap();
            assert!(!result.chunks.is_empty());
        }
    }
}