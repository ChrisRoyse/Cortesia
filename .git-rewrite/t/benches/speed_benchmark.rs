use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship};
use llmkg::storage::mmap_storage::MMapStorage;
use llmkg::embedding::simd_search::BatchProcessor;
use llmkg::query::rag::GraphRAGEngine;
use std::time::Duration;

fn setup_knowledge_graph(num_entities: usize, embedding_dim: usize) -> KnowledgeGraph {
    let kg = KnowledgeGraph::new(embedding_dim).unwrap();
    
    // Insert entities with random embeddings
    for i in 0..num_entities {
        let embedding: Vec<f32> = (0..embedding_dim)
            .map(|j| ((i * 31 + j * 17) as f32 / 1000.0).sin())
            .collect();
        
        let entity_data = EntityData {
            type_id: (i % 10) as u16,
            properties: format!("Entity {} with properties", i),
            embedding,
        };
        
        kg.insert_entity(i as u32, entity_data).unwrap();
    }
    
    // Insert relationships to create graph structure
    for i in 0..num_entities {
        for j in 1..=3 {
            if i + j < num_entities {
                let rel = Relationship {
                    from: i as u32,
                    to: (i + j) as u32,
                    rel_type: (j % 5) as u8,
                    weight: 1.0 / j as f32,
                };
                kg.insert_relationship(rel).unwrap();
            }
        }
    }
    
    kg
}

fn benchmark_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_search");
    
    for &num_entities in &[1_000, 10_000, 100_000] {
        let kg = setup_knowledge_graph(num_entities, 96);
        let query_embedding: Vec<f32> = (0..96).map(|i| (i as f32 / 96.0).sin()).collect();
        
        group.throughput(Throughput::Elements(num_entities as u64));
        group.bench_with_input(
            BenchmarkId::new("entities", num_entities),
            &num_entities,
            |b, _| {
                b.iter(|| {
                    kg.similarity_search(black_box(&query_embedding), black_box(20))
                        .unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_neighbor_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_retrieval");
    
    for &num_entities in &[1_000, 10_000, 100_000] {
        let kg = setup_knowledge_graph(num_entities, 96);
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("entities", num_entities),
            &num_entities,
            |b, _| {
                b.iter(|| {
                    kg.get_neighbors(black_box(500)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_path_finding(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_finding");
    
    for &num_entities in &[1_000, 10_000, 50_000] {
        let kg = setup_knowledge_graph(num_entities, 96);
        
        group.bench_with_input(
            BenchmarkId::new("entities", num_entities),
            &num_entities,
            |b, _| {
                b.iter(|| {
                    kg.find_path(black_box(10), black_box(num_entities as u32 - 10), black_box(6))
                        .unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_graph_rag_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_rag_query");
    group.sample_size(20); // Reduce sample size for complex operations
    group.measurement_time(Duration::from_secs(30));
    
    for &num_entities in &[1_000, 10_000] {
        let kg = setup_knowledge_graph(num_entities, 96);
        let query_embedding: Vec<f32> = (0..96).map(|i| (i as f32 / 96.0).cos()).collect();
        
        group.throughput(Throughput::Elements(num_entities as u64));
        group.bench_with_input(
            BenchmarkId::new("entities", num_entities),
            &num_entities,
            |b, _| {
                b.iter(|| {
                    kg.query(black_box(&query_embedding), black_box(25), black_box(3))
                        .unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    for &num_entities in &[10_000, 100_000, 1_000_000] {
        let kg = setup_knowledge_graph(num_entities, 96);
        let memory_usage = kg.memory_usage();
        
        let bytes_per_entity = memory_usage.bytes_per_entity(num_entities);
        
        group.bench_function(
            &format!("memory_footprint_{}_entities", num_entities),
            |b| {
                b.iter(|| {
                    // Simulate memory access patterns
                    for i in (0..num_entities).step_by(100) {
                        kg.get_neighbors(i as u32).unwrap();
                    }
                })
            },
        );
        
        println!(
            "Memory efficiency for {} entities: {} bytes per entity, total: {} MB",
            num_entities,
            bytes_per_entity,
            memory_usage.total_bytes() / 1_048_576
        );
    }
    
    group.finish();
}

fn benchmark_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    
    let kg = setup_knowledge_graph(50_000, 96);
    let query_embeddings: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            (0..96).map(|j| ((i * 7 + j * 13) as f32 / 1000.0).sin()).collect()
        })
        .collect();
    
    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_similarity_search", |b| {
        b.iter(|| {
            for embedding in black_box(&query_embeddings) {
                kg.similarity_search(embedding, 10).unwrap();
            }
        })
    });
    
    group.finish();
}

fn benchmark_mmap_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmap_storage");
    
    for &num_entities in &[10_000, 100_000] {
        let storage = MMapStorage::new(num_entities, num_entities * 3, 96).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("entity_lookup", num_entities),
            &num_entities,
            |b, _| {
                b.iter(|| {
                    // Simulate random entity lookups
                    for i in (0..1000).step_by(10) {
                        storage.get_entity(black_box(i));
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    let embedding_dim = 96;
    let subvector_count = 8;
    let batch_size = 64;
    
    let mut processor = BatchProcessor::new(embedding_dim, subvector_count, batch_size);
    let query: Vec<f32> = (0..embedding_dim).map(|i| (i as f32 / 96.0).sin()).collect();
    let codebooks: Vec<Vec<f32>> = (0..subvector_count)
        .map(|_| vec![0.0; 256 * (embedding_dim / subvector_count)])
        .collect();
    
    processor.precompute_distances(&query, &codebooks).unwrap();
    
    // Create test data
    let test_codes: Vec<Vec<u8>> = (0..10000)
        .map(|i| (0..subvector_count).map(|j| ((i + j) % 256) as u8).collect())
        .collect();
    let entity_ids: Vec<u32> = (0..10000).collect();
    
    group.throughput(Throughput::Elements(10000));
    group.bench_function("batch_distance_computation", |b| {
        b.iter(|| {
            processor.process_batched_search(
                black_box(&test_codes),
                black_box(&entity_ids),
                black_box(20)
            ).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets");
    group.sample_size(100);
    
    let kg = setup_knowledge_graph(100_000, 96);
    let query_embedding: Vec<f32> = (0..96).map(|i| (i as f32 / 96.0).sin()).collect();
    
    // Target: <1ms for single entity lookup
    group.bench_function("single_entity_lookup", |b| {
        b.iter(|| {
            kg.get_neighbors(black_box(50000)).unwrap()
        })
    });
    
    // Target: <5ms for similarity search
    group.bench_function("similarity_search_target", |b| {
        b.iter(|| {
            kg.similarity_search(black_box(&query_embedding), black_box(20)).unwrap()
        })
    });
    
    // Target: <10ms for context retrieval
    group.bench_function("context_retrieval_target", |b| {
        b.iter(|| {
            kg.query(black_box(&query_embedding), black_box(20), black_box(2)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_similarity_search,
    benchmark_neighbor_retrieval,
    benchmark_path_finding,
    benchmark_graph_rag_query,
    benchmark_memory_efficiency,
    benchmark_batch_operations,
    benchmark_mmap_storage,
    benchmark_simd_operations,
    benchmark_performance_targets
);

criterion_main!(benches);