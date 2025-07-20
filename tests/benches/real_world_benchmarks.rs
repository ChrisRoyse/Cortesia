/*!
Phase 5.3: Real-World Benchmarks
Realistic benchmarks that reflect actual LLMKG usage patterns and workflows
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// Real-world entity patterns
#[derive(Clone)]
struct RealWorldEntity {
    id: String,
    entity_type: String,
    content: String,
    metadata: HashMap<String, String>,
    embeddings: Vec<f32>,
    relationships: Vec<String>,
}

impl RealWorldEntity {
    fn new_document(id: &str, content: &str, domain: &str) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("domain".to_string(), domain.to_string());
        metadata.insert("created_at".to_string(), "2024-01-01T00:00:00Z".to_string());
        metadata.insert("source".to_string(), "synthetic".to_string());
        
        // Generate realistic embeddings (384 dimensions for sentence transformers)
        let embeddings: Vec<f32> = (0..384).map(|i| (i as f32 * 0.001).sin()).collect();
        
        Self {
            id: id.to_string(),
            entity_type: "document".to_string(),
            content: content.to_string(),
            metadata,
            embeddings,
            relationships: vec![],
        }
    }
    
    fn new_user(id: &str, preferences: &[&str]) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("user_type".to_string(), "standard".to_string());
        metadata.insert("preferences".to_string(), preferences.join(","));
        
        // User preference embeddings
        let embeddings: Vec<f32> = (0..384).map(|i| 
            (i as f32 * 0.002 + preferences.len() as f32).cos()
        ).collect();
        
        Self {
            id: id.to_string(),
            entity_type: "user".to_string(),
            content: format!("User profile for {}", id),
            metadata,
            embeddings,
            relationships: vec![],
        }
    }
    
    fn new_product(id: &str, category: &str, price: f32) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), category.to_string());
        metadata.insert("price".to_string(), price.to_string());
        metadata.insert("currency".to_string(), "USD".to_string());
        
        // Product embeddings based on category and price
        let embeddings: Vec<f32> = (0..384).map(|i| 
            (i as f32 * 0.003 + price * 0.01).tan().abs()
        ).collect();
        
        Self {
            id: id.to_string(),
            entity_type: "product".to_string(),
            content: format!("{} product in {} category", id, category),
            metadata,
            embeddings,
            relationships: vec![],
        }
    }
}

// Realistic data generators
fn generate_document_corpus(size: usize) -> Vec<RealWorldEntity> {
    let domains = ["technology", "healthcare", "finance", "education", "entertainment"];
    let content_templates = [
        "This document discusses advanced techniques in {}",
        "Recent developments in {} have shown promising results",
        "A comprehensive analysis of {} trends and opportunities",
        "Best practices for implementing {} solutions",
        "The future of {} technology and its applications",
    ];
    
    (0..size).map(|i| {
        let domain = domains[i % domains.len()];
        let template = content_templates[i % content_templates.len()];
        let content = format!("{} with detailed analysis and examples.", template.replace("{}", domain));
        RealWorldEntity::new_document(&format!("doc_{}", i), &content, domain)
    }).collect()
}

fn generate_user_profiles(size: usize) -> Vec<RealWorldEntity> {
    let preference_sets = [
        vec!["technology", "programming", "AI"],
        vec!["healthcare", "research", "biology"],
        vec!["finance", "investing", "economics"],
        vec!["education", "teaching", "learning"],
        vec!["entertainment", "movies", "gaming"],
    ];
    
    (0..size).map(|i| {
        let preferences = &preference_sets[i % preference_sets.len()];
        RealWorldEntity::new_user(&format!("user_{}", i), preferences)
    }).collect()
}

fn generate_product_catalog(size: usize) -> Vec<RealWorldEntity> {
    let categories = ["electronics", "books", "clothing", "home", "sports"];
    let base_prices = [99.99, 19.99, 49.99, 149.99, 79.99];
    
    (0..size).map(|i| {
        let category = categories[i % categories.len()];
        let base_price = base_prices[i % base_prices.len()];
        let price = base_price + (i as f32 * 0.1) % 50.0;
        RealWorldEntity::new_product(&format!("product_{}", i), category, price)
    }).collect()
}

// Real-world workflow benchmarks
fn benchmark_document_retrieval_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_retrieval_workflow");
    
    for size in [1000, 5000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("semantic_search", size), &size, |b, &size| {
            let documents = generate_document_corpus(size);
            let temp_dir = TempDir::new().unwrap();
            
            // Setup LLMKG with documents
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            // Index documents
            for doc in &documents {
                let key = EntityKey::from_hash(&doc.id);
                let content_id = interner.insert(&doc.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: doc.embeddings.clone(),
                    metadata: doc.metadata.clone(),
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark semantic search
            b.iter(|| {
                let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.004).sin()).collect();
                let results = black_box(graph.find_similar_entities(&query_embedding, 10));
                black_box(results)
            });
        });
    }
    
    group.finish();
}

fn benchmark_recommendation_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("recommendation_system");
    
    for size in [1000, 5000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("user_product_matching", size), &size, |b, &size| {
            let users = generate_user_profiles(size / 10);
            let products = generate_product_catalog(size);
            let temp_dir = TempDir::new().unwrap();
            
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            // Index users and products
            for user in &users {
                let key = EntityKey::from_hash(&user.id);
                let content_id = interner.insert(&user.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: user.embeddings.clone(),
                    metadata: user.metadata.clone(),
                };
                
                graph.add_entity(entity);
            }
            
            for product in &products {
                let key = EntityKey::from_hash(&product.id);
                let content_id = interner.insert(&product.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: product.embeddings.clone(),
                    metadata: product.metadata.clone(),
                };
                
                graph.add_entity(entity);
            }
            
            // Benchmark recommendation generation
            b.iter(|| {
                let user_embedding = &users[0].embeddings;
                let recommendations = black_box(graph.find_similar_entities(user_embedding, 20));
                black_box(recommendations)
            });
        });
    }
    
    group.finish();
}

fn benchmark_real_time_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_time_indexing");
    
    for batch_size in [10, 50, 100] {
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(BenchmarkId::new("streaming_ingestion", batch_size), &batch_size, |b, &batch_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = Arc::new(Mutex::new(EntityGraph::new()));
            let quantizer = Arc::new(ProductQuantizer::new(384, 8, 48).unwrap());
            let interner = Arc::new(Mutex::new(StringInterner::new()));
            let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
            
            b.iter(|| {
                let documents = generate_document_corpus(batch_size);
                
                let start = Instant::now();
                for doc in documents {
                    let graph = graph.clone();
                    let interner = interner.clone();
                    
                    let key = EntityKey::from_hash(&doc.id);
                    let content_id = {
                        let mut interner_guard = interner.lock().unwrap();
                        interner_guard.insert(&doc.content)
                    };
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding: doc.embeddings,
                        metadata: doc.metadata,
                    };
                    
                    {
                        let mut graph_guard = graph.lock().unwrap();
                        graph_guard.add_entity(entity);
                    }
                }
                black_box(start.elapsed())
            });
        });
    }
    
    group.finish();
}

fn benchmark_concurrent_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access_patterns");
    
    for thread_count in [2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("read_heavy_workload", thread_count), &thread_count, |b, &thread_count| {
            let documents = generate_document_corpus(10000);
            let temp_dir = TempDir::new().unwrap();
            
            let graph = Arc::new(Mutex::new(EntityGraph::new()));
            let quantizer = Arc::new(ProductQuantizer::new(384, 8, 48).unwrap());
            let interner = Arc::new(Mutex::new(StringInterner::new()));
            let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
            
            // Pre-populate with data
            {
                let mut graph_guard = graph.lock().unwrap();
                let mut interner_guard = interner.lock().unwrap();
                
                for doc in &documents[..1000] {
                    let key = EntityKey::from_hash(&doc.id);
                    let content_id = interner_guard.insert(&doc.content);
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding: doc.embeddings.clone(),
                        metadata: doc.metadata.clone(),
                    };
                    
                    graph_guard.add_entity(entity);
                }
            }
            
            b.iter(|| {
                let handles: Vec<_> = (0..thread_count).map(|i| {
                    let graph = graph.clone();
                    let query_embedding: Vec<f32> = (0..384).map(|j| ((i + j) as f32 * 0.001).sin()).collect();
                    
                    thread::spawn(move || {
                        for _ in 0..10 {
                            let graph_guard = graph.lock().unwrap();
                            let results = graph_guard.find_similar_entities(&query_embedding, 5);
                            black_box(results);
                        }
                    })
                }).collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
    }
    
    group.finish();
}

fn benchmark_analytics_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("analytics_workload");
    
    for data_size in [5000, 10000, 20000] {
        group.throughput(Throughput::Elements(data_size as u64));
        
        group.bench_with_input(BenchmarkId::new("aggregate_analysis", data_size), &data_size, |b, &data_size| {
            let documents = generate_document_corpus(data_size);
            let temp_dir = TempDir::new().unwrap();
            
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            // Index all documents
            for doc in &documents {
                let key = EntityKey::from_hash(&doc.id);
                let content_id = interner.insert(&doc.content);
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding: doc.embeddings.clone(),
                    metadata: doc.metadata.clone(),
                };
                
                graph.add_entity(entity);
            }
            
            b.iter(|| {
                // Simulate analytics queries
                let mut category_counts = HashMap::new();
                let mut total_entities = 0;
                let mut embedding_sums = vec![0.0f32; 384];
                
                // Aggregate analysis - count by domain and compute embedding centroids
                for entity in graph.get_all_entities() {
                    total_entities += 1;
                    
                    if let Some(domain) = entity.metadata.get("domain") {
                        *category_counts.entry(domain.clone()).or_insert(0) += 1;
                    }
                    
                    for (i, &val) in entity.embedding.iter().enumerate() {
                        embedding_sums[i] += val;
                    }
                }
                
                // Compute centroid
                let centroid: Vec<f32> = embedding_sums.iter()
                    .map(|&sum| sum / total_entities as f32)
                    .collect();
                
                black_box((category_counts, centroid))
            });
        });
    }
    
    group.finish();
}

fn benchmark_memory_efficiency_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency_patterns");
    
    for entity_count in [10000, 50000, 100000] {
        group.throughput(Throughput::Elements(entity_count as u64));
        
        group.bench_with_input(BenchmarkId::new("large_dataset_handling", entity_count), &entity_count, |b, &entity_count| {
            b.iter(|| {
                let temp_dir = TempDir::new().unwrap();
                let graph = EntityGraph::new();
                let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
                let mut interner = StringInterner::new();
                let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
                
                let start_memory = std::process::Command::new("ps")
                    .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                    .output()
                    .ok()
                    .and_then(|output| String::from_utf8(output.stdout).ok())
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                
                // Load entities in batches to simulate real-world streaming
                for batch in 0..(entity_count / 1000) {
                    let batch_documents = generate_document_corpus(1000);
                    
                    for doc in batch_documents {
                        let key = EntityKey::from_hash(&doc.id);
                        let content_id = interner.insert(&doc.content);
                        
                        let entity = Entity {
                            key,
                            content: content_id,
                            embedding: doc.embeddings,
                            metadata: doc.metadata,
                        };
                        
                        graph.add_entity(entity);
                    }
                    
                    // Perform periodic compaction simulation
                    if batch % 10 == 0 {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                
                let end_memory = std::process::Command::new("ps")
                    .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                    .output()
                    .ok()
                    .and_then(|output| String::from_utf8(output.stdout).ok())
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                
                black_box((graph.entity_count(), end_memory - start_memory))
            });
        });
    }
    
    group.finish();
}

fn benchmark_persistence_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence_patterns");
    
    for checkpoint_interval in [1000, 5000, 10000] {
        group.bench_with_input(BenchmarkId::new("incremental_persistence", checkpoint_interval), &checkpoint_interval, |b, &checkpoint_interval| {
            let temp_dir = TempDir::new().unwrap();
            
            b.iter(|| {
                let graph = EntityGraph::new();
                let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
                let mut interner = StringInterner::new();
                let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
                
                let documents = generate_document_corpus(*checkpoint_interval);
                
                let start = Instant::now();
                
                for (i, doc) in documents.iter().enumerate() {
                    let key = EntityKey::from_hash(&doc.id);
                    let content_id = interner.insert(&doc.content);
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding: doc.embeddings.clone(),
                        metadata: doc.metadata.clone(),
                    };
                    
                    graph.add_entity(entity);
                    
                    // Simulate periodic persistence checkpoints
                    if (i + 1) % 1000 == 0 {
                        // Force a checkpoint (in real implementation)
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
                
                black_box(start.elapsed())
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    real_world_benches,
    benchmark_document_retrieval_workflow,
    benchmark_recommendation_system,
    benchmark_real_time_indexing,
    benchmark_concurrent_access_patterns,
    benchmark_analytics_workload,
    benchmark_memory_efficiency_patterns,
    benchmark_persistence_patterns
);

criterion_main!(real_world_benches);