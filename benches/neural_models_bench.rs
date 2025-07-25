//! Benchmarks for real neural model performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use llmkg::models::model_loader::ModelLoader;
use llmkg::neural::neural_server::NeuralProcessingServer;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn benchmark_entity_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Initialize models once
    let model_loader = Arc::new(ModelLoader::new());
    let distilbert = rt.block_on(model_loader.load_distilbert_ner()).ok();
    let tinybert = rt.block_on(model_loader.load_tinybert_ner()).ok();
    
    let test_texts = vec![
        ("short", "Albert Einstein won the Nobel Prize."),
        ("medium", "Marie Curie discovered radium and polonium. She won two Nobel Prizes in Physics and Chemistry."),
        ("long", "The Theory of Relativity, developed by Albert Einstein in 1905, revolutionized our understanding of space and time. Einstein's work at the University of Berlin led to groundbreaking discoveries. He later moved to Princeton University in the United States."),
    ];
    
    let mut group = c.benchmark_group("entity_extraction");
    
    for (name, text) in test_texts {
        // Benchmark DistilBERT-NER (66M params)
        if let Some(model) = &distilbert {
            group.bench_with_input(
                BenchmarkId::new("DistilBERT-NER", name),
                &text,
                |b, &text| {
                    b.iter(|| {
                        let tokenized = model.tokenizer.encode(black_box(text), true);
                        let entities = model.predict(&tokenized.input_ids);
                        black_box(entities);
                    });
                },
            );
        }
        
        // Benchmark TinyBERT-NER (14.5M params)
        if let Some(model) = &tinybert {
            group.bench_with_input(
                BenchmarkId::new("TinyBERT-NER", name),
                &text,
                |b, &text| {
                    b.iter(|| {
                        let entities = model.predict(black_box(text));
                        black_box(entities);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_embedding_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Initialize MiniLM
    let model_loader = Arc::new(ModelLoader::new());
    let minilm = rt.block_on(model_loader.load_minilm()).ok();
    
    let test_texts = vec![
        ("word", "intelligence"),
        ("phrase", "artificial intelligence"),
        ("sentence", "Deep learning models process natural language efficiently."),
        ("paragraph", "Natural language processing has evolved significantly with the advent of transformer models. These models can understand context and generate human-like text with remarkable accuracy."),
    ];
    
    let mut group = c.benchmark_group("embedding_generation");
    
    if let Some(model) = &minilm {
        for (name, text) in test_texts {
            group.bench_with_input(
                BenchmarkId::new("MiniLM-L6-v2", name),
                &text,
                |b, &text| {
                    b.iter(|| {
                        let embedding = model.encode(black_box(text));
                        black_box(embedding);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Initialize models
    let model_loader = Arc::new(ModelLoader::new());
    let tinybert = rt.block_on(model_loader.load_tinybert_ner()).ok();
    
    // Create batches of different sizes
    let base_sentence = "Marie Curie discovered radium in her laboratory.";
    let batch_sizes = vec![1, 10, 50, 100];
    
    let mut group = c.benchmark_group("batch_processing");
    
    if let Some(model) = &tinybert {
        for size in batch_sizes {
            let batch_text = vec![base_sentence; size].join(" ");
            
            group.bench_with_input(
                BenchmarkId::new("TinyBERT-batch", size),
                &batch_text,
                |b, text| {
                    b.iter(|| {
                        let entities = model.predict(black_box(text));
                        black_box(entities);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_model_loading(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("model_loading");
    group.sample_size(10); // Reduce sample size for loading benchmarks
    
    // Benchmark DistilBERT loading
    group.bench_function("load_distilbert", |b| {
        b.iter(|| {
            let model_loader = ModelLoader::new();
            rt.block_on(async {
                let _model = model_loader.load_distilbert_ner().await;
            });
        });
    });
    
    // Benchmark TinyBERT loading
    group.bench_function("load_tinybert", |b| {
        b.iter(|| {
            let model_loader = ModelLoader::new();
            rt.block_on(async {
                let _model = model_loader.load_tinybert_ner().await;
            });
        });
    });
    
    // Benchmark MiniLM loading
    group.bench_function("load_minilm", |b| {
        b.iter(|| {
            let model_loader = ModelLoader::new();
            rt.block_on(async {
                let _model = model_loader.load_minilm().await;
            });
        });
    });
    
    group.finish();
}

fn benchmark_neural_server_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Initialize neural server with models
    let neural_server = rt.block_on(async {
        let server = Arc::new(NeuralProcessingServer::new("localhost:8080".to_string()).await.unwrap());
        server.initialize_models().await.unwrap();
        server
    });
    
    let test_text = "Einstein's theory of relativity changed physics forever.";
    
    let mut group = c.benchmark_group("neural_server");
    
    // Benchmark embedding generation through neural server
    group.bench_function("get_embedding", |b| {
        b.iter(|| {
            rt.block_on(async {
                let embedding = neural_server.get_embedding(black_box(test_text)).await.unwrap();
                black_box(embedding);
            });
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_entity_extraction,
    benchmark_embedding_generation,
    benchmark_batch_processing,
    benchmark_model_loading,
    benchmark_neural_server_integration
);
criterion_main!(benches);