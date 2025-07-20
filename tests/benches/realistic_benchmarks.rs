use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::storage::zero_copy::ZeroCopySerializer;
use llmkg::core::types::{EntityData, AttributeValue};
use llmkg::embedding::store::EmbeddingStore;
use llmkg::embedding::quantizer::ProductQuantizer;
use llmkg::storage::string_interner::{StringInterner, intern_string, clear_interner};
use llmkg::core::interned_entity::{InternedEntityData, InternedEntityCollection};
use llmkg::core::graph::KnowledgeGraph;
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Generate realistic embeddings that simulate actual text embeddings
fn generate_realistic_embedding(rng: &mut StdRng, dim: usize, semantic_base: f32) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dim);
    
    // Generate embedding with semantic structure similar to real text embeddings
    for i in 0..dim {
        // Create clusters of related dimensions (common in real embeddings)
        let cluster = i / 16; // 16 dimensions per semantic cluster
        let cluster_base = (cluster as f32 * 0.1 + semantic_base).sin();
        
        // Add dimension-specific variation
        let dim_variation = ((i as f32 * 0.05).cos() + (i as f32 * 0.03).sin()) * 0.3;
        
        // Add Gaussian-like noise
        let noise = rng.gen_range(-0.1..0.1);
        
        let value = cluster_base + dim_variation + noise;
        embedding.push(value.clamp(-1.0, 1.0));
    }
    
    // Normalize to unit length (standard for text embeddings)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }
    
    embedding
}

/// Generate realistic test entities with varied properties
fn generate_realistic_test_entities(count: usize, embedding_dim: usize) -> Vec<EntityData> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let categories = ["person", "organization", "location", "concept", "event", "product"];
    let domains = ["technology", "science", "business", "education", "healthcare", "finance"];
    
    (0..count)
        .map(|i| {
            let category = categories[i % categories.len()];
            let domain = domains[(i / categories.len()) % domains.len()];
            
            // Generate realistic JSON properties
            let properties = serde_json::json!({
                "id": format!("entity_{}", i),
                "name": format!("{} Entity {}", category.chars().next().unwrap().to_uppercase(), i),
                "category": category,
                "domain": domain,
                "confidence": rng.gen_range(0.7..1.0),
                "timestamp": chrono::Utc::now().timestamp() - (i as i64 * 3600),
                "metadata": {
                    "source": ["web", "database", "api", "manual"][rng.gen_range(0..4)],
                    "version": rng.gen_range(1..5),
                    "tags": (0..rng.gen_range(1..5))
                        .map(|_| ["verified", "important", "archived", "pending", "reviewed"][rng.gen_range(0..5)])
                        .collect::<Vec<_>>()
                },
                "attributes": match category {
                    "person" => serde_json::json!({
                        "age": rng.gen_range(18..80),
                        "occupation": ["engineer", "scientist", "manager", "analyst"][rng.gen_range(0..4)]
                    }),
                    "organization" => serde_json::json!({
                        "size": ["small", "medium", "large", "enterprise"][rng.gen_range(0..4)],
                        "industry": domain
                    }),
                    _ => serde_json::json!({})
                }
            });
            
            EntityData {
                type_id: (i % 100) as u16, // More varied type IDs
                properties: properties.to_string(),
                embedding: generate_realistic_embedding(&mut rng, embedding_dim, i as f32 * 0.01),
            }
        })
        .collect()
}

/// Benchmark zero-copy serialization with realistic data
fn benchmark_realistic_zero_copy_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_zero_copy_serialization");
    group.sample_size(50);
    
    for &entity_count in &[1_000, 10_000, 50_000] {
        let entities = generate_realistic_test_entities(entity_count, 384); // 384-dim like many text models
        
        group.throughput(Throughput::Elements(entity_count as u64));
        group.throughput(Throughput::Bytes(
            entities.iter()
                .map(|e| e.properties.len() + e.embedding.len() * 4)
                .sum::<usize>() as u64
        ));
        
        // Warm-up phase
        let mut warmup_serializer = ZeroCopySerializer::new();
        for entity in entities.iter().take(100) {
            warmup_serializer.add_entity(entity, 384).unwrap();
        }
        
        group.bench_with_input(
            BenchmarkId::new("serialize", entity_count),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let mut serializer = ZeroCopySerializer::new();
                    for entity in entities {
                        serializer.add_entity(black_box(entity), 384).unwrap();
                    }
                    black_box(serializer.finalize())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark embedding operations with realistic data
fn benchmark_realistic_embedding_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_embedding_operations");
    
    // Test with different embedding dimensions used in practice
    for &dim in &[128, 384, 768, 1536] {
        let entity_count = 10_000;
        let entities = generate_realistic_test_entities(entity_count, dim);
        let embeddings: Vec<_> = entities.iter().map(|e| e.embedding.clone()).collect();
        
        // Create embedding store
        let mut store = EmbeddingStore::new(dim);
        for (i, embedding) in embeddings.iter().enumerate() {
            store.add_embedding(llmkg::core::types::EntityKey::default(), embedding.clone()).unwrap();
        }
        
        group.throughput(Throughput::Elements(1));
        
        // Benchmark similarity search
        let query_embedding = generate_realistic_embedding(&mut StdRng::seed_from_u64(123), dim, 0.5);
        
        group.bench_with_input(
            BenchmarkId::new("similarity_search", format!("{}d", dim)),
            &(&store, &query_embedding),
            |b, (store, query)| {
                b.iter(|| {
                    let results = store.find_similar(black_box(query), 10).unwrap();
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark graph operations with realistic data patterns
fn benchmark_realistic_graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_graph_operations");
    
    for &node_count in &[1_000, 5_000, 10_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = Arc::new(KnowledgeGraph::new(384).unwrap());
        let entities = generate_realistic_test_entities(node_count, 384);
        
        // Add entities to graph
        let mut entity_keys = Vec::new();
        for entity in &entities {
            let key = graph.add_entity(entity.clone()).unwrap();
            entity_keys.push(key);
        }
        
        // Create realistic relationship patterns
        // Power-law distribution (few nodes with many connections, many with few)
        for (i, &source_key) in entity_keys.iter().enumerate() {
            let connection_count = if i < node_count / 100 {
                rng.gen_range(50..100) // Hub nodes
            } else if i < node_count / 10 {
                rng.gen_range(10..50) // Well-connected nodes
            } else {
                rng.gen_range(1..10) // Regular nodes
            };
            
            for _ in 0..connection_count {
                let target_idx = rng.gen_range(0..entity_keys.len());
                if target_idx != i {
                    let weight = rng.gen_range(0.1..1.0);
                    graph.add_edge(source_key, entity_keys[target_idx], weight).ok();
                }
            }
        }
        
        group.throughput(Throughput::Elements(1));
        
        // Benchmark path finding
        let start_idx = rng.gen_range(0..entity_keys.len());
        let end_idx = rng.gen_range(0..entity_keys.len());
        
        group.bench_with_input(
            BenchmarkId::new("pathfinding", node_count),
            &(&graph, entity_keys[start_idx], entity_keys[end_idx]),
            |b, (graph, start, end)| {
                b.iter(|| {
                    let path = graph.find_path(black_box(*start), black_box(*end), 5);
                    black_box(path)
                });
            },
        );
        
        // Benchmark neighborhood queries
        group.bench_with_input(
            BenchmarkId::new("neighborhood_query", node_count),
            &(&graph, entity_keys[start_idx]),
            |b, (graph, node)| {
                b.iter(|| {
                    let neighbors = graph.get_neighbors(black_box(*node), 2);
                    black_box(neighbors)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark with statistical analysis
fn benchmark_with_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_analysis");
    group.sample_size(100); // More samples for better statistics
    group.measurement_time(std::time::Duration::from_secs(10)); // Longer measurement
    
    let entities = generate_realistic_test_entities(10_000, 384);
    
    group.bench_function("entity_processing_stats", |b| {
        let mut times = Vec::new();
        
        b.iter_custom(|iters| {
            let mut total_time = std::time::Duration::ZERO;
            
            for _ in 0..iters {
                let start = Instant::now();
                
                // Process entities
                let mut serializer = ZeroCopySerializer::new();
                for entity in &entities {
                    serializer.add_entity(black_box(entity), 384).unwrap();
                }
                let _ = black_box(serializer.finalize());
                
                let elapsed = start.elapsed();
                times.push(elapsed.as_nanos() as f64);
                total_time += elapsed;
            }
            
            // Calculate statistics
            if !times.is_empty() {
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = times[times.len() / 2];
                let p95 = times[(times.len() as f64 * 0.95) as usize];
                let p99 = times[(times.len() as f64 * 0.99) as usize];
                
                println!("Median: {:.2}µs, P95: {:.2}µs, P99: {:.2}µs", 
                         median / 1000.0, p95 / 1000.0, p99 / 1000.0);
            }
            
            total_time
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_realistic_zero_copy_serialization,
    benchmark_realistic_embedding_operations,
    benchmark_realistic_graph_operations,
    benchmark_with_statistics
);
criterion_main!(benches);