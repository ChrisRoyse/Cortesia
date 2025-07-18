//! Performance Benchmarks for Unit Testing Framework
//!
//! Comprehensive benchmarks for all LLMKG components to establish
//! performance baselines and detect performance regressions.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg_tests::*;
use std::time::Duration;

// Benchmark core entity operations
fn bench_entity_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_operations");
    
    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("creation", size), size, |b, &size| {
            b.iter(|| {
                let mut entities = Vec::new();
                for i in 0..size {
                    let entity = create_test_entity(&format!("entity_{}", i), &format!("Entity {}", i));
                    entities.push(entity);
                }
                entities
            });
        });
        
        group.bench_with_input(BenchmarkId::new("attribute_access", size), size, |b, &size| {
            let mut entity = create_test_entity("benchmark_entity", "Benchmark Entity");
            for i in 0..size {
                entity.add_attribute(&format!("attr_{}", i), &format!("value_{}", i));
            }
            
            b.iter(|| {
                for i in 0..size {
                    let _ = entity.get_attribute(&format!("attr_{}", i));
                }
            });
        });
    }
    
    group.finish();
}

// Benchmark graph operations
fn bench_graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_operations");
    
    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("graph_creation", size), size, |b, &size| {
            b.iter(|| {
                create_test_graph(*size, *size * 2)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("entity_lookup", size), size, |b, &size| {
            let graph = create_test_graph(*size, *size * 2);
            let entity_keys: Vec<_> = (0..*size)
                .map(|i| EntityKey::from_hash(&format!("test_entity_{}", i)))
                .collect();
            
            b.iter(|| {
                for key in &entity_keys {
                    let _ = graph.get_entity(*key);
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("relationship_traversal", size), size, |b, &size| {
            let graph = create_test_graph(*size, *size * 2);
            let entity_keys: Vec<_> = (0..*size)
                .map(|i| EntityKey::from_hash(&format!("test_entity_{}", i)))
                .collect();
            
            b.iter(|| {
                for key in &entity_keys {
                    let _ = graph.get_relationships(*key);
                }
            });
        });
    }
    
    group.finish();
}

// Benchmark CSR storage operations
fn bench_csr_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_operations");
    
    for size in [500, 1000, 2000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("csr_construction", size), size, |b, &size| {
            let mut rng = DeterministicRng::new(CSR_TEST_SEED);
            let matrix = generate_random_adjacency_matrix(&mut rng, *size, 0.05);
            
            b.iter(|| {
                CompressedSparseRow::from_adjacency_matrix(&matrix)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("matrix_vector_multiply", size), size, |b, &size| {
            let mut rng = DeterministicRng::new(CSR_TEST_SEED);
            let matrix = generate_random_adjacency_matrix(&mut rng, *size, 0.05);
            let csr = CompressedSparseRow::from_adjacency_matrix(&matrix);
            let vector: Vec<f32> = (0..*size).map(|i| i as f32 * 0.1).collect();
            
            b.iter(|| {
                csr.multiply_vector(&vector)
            });
        });
    }
    
    group.finish();
}

// Benchmark bloom filter operations
fn bench_bloom_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_operations");
    
    for capacity in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*capacity as u64));
        
        group.bench_with_input(BenchmarkId::new("bloom_insertion", capacity), capacity, |b, &capacity| {
            let mut bloom = BloomFilter::new(capacity, 0.01);
            let items: Vec<String> = (0..capacity).map(|i| format!("item_{}", i)).collect();
            
            b.iter(|| {
                for item in &items {
                    bloom.insert(item);
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("bloom_lookup", capacity), capacity, |b, &capacity| {
            let mut bloom = BloomFilter::new(capacity, 0.01);
            let items: Vec<String> = (0..capacity).map(|i| format!("item_{}", i)).collect();
            
            // Pre-populate bloom filter
            for item in &items {
                bloom.insert(item);
            }
            
            b.iter(|| {
                for item in &items {
                    let _ = bloom.contains(item);
                }
            });
        });
    }
    
    group.finish();
}

// Benchmark quantization operations
fn bench_quantization_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_operations");
    
    for dim in [64, 128, 256].iter() {
        group.throughput(Throughput::Elements(*dim as u64));
        
        group.bench_with_input(BenchmarkId::new("quantizer_training", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(PQ_TEST_SEED);
            let training_data = generate_random_vectors(&mut rng, 1000, dim);
            
            b.iter(|| {
                let mut quantizer = ProductQuantizer::new(dim, 256);
                quantizer.train(&training_data).unwrap();
                quantizer
            });
        });
        
        group.bench_with_input(BenchmarkId::new("vector_quantization", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(PQ_TEST_SEED);
            let training_data = generate_random_vectors(&mut rng, 1000, dim);
            let test_vectors = generate_random_vectors(&mut rng, 100, dim);
            
            let mut quantizer = ProductQuantizer::new(dim, 256);
            quantizer.train(&training_data).unwrap();
            
            b.iter(|| {
                for vector in &test_vectors {
                    let _ = quantizer.quantize(vector);
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("vector_reconstruction", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(PQ_TEST_SEED);
            let training_data = generate_random_vectors(&mut rng, 1000, dim);
            let test_vectors = generate_random_vectors(&mut rng, 100, dim);
            
            let mut quantizer = ProductQuantizer::new(dim, 256);
            quantizer.train(&training_data).unwrap();
            
            let quantized_vectors: Vec<_> = test_vectors.iter()
                .map(|v| quantizer.quantize(v))
                .collect();
            
            b.iter(|| {
                for quantized in &quantized_vectors {
                    let _ = quantizer.reconstruct(quantized);
                }
            });
        });
    }
    
    group.finish();
}

// Benchmark SIMD operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    for dim in [128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*dim as u64));
        
        group.bench_with_input(BenchmarkId::new("simd_distance", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
            let vec1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vec2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            b.iter(|| {
                simd_euclidean_distance(&vec1, &vec2)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("scalar_distance", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
            let vec1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vec2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            
            b.iter(|| {
                scalar_euclidean_distance(&vec1, &vec2)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("simd_batch_distance", dim), dim, |b, &dim| {
            let mut rng = DeterministicRng::new(SIMD_TEST_SEED);
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let database: Vec<Vec<f32>> = (0..1000)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            
            b.iter(|| {
                simd_batch_distances(&query, &database)
            });
        });
    }
    
    group.finish();
}

// Benchmark RAG operations
fn bench_rag_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("rag_operations");
    
    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("context_assembly", size), size, |b, &size| {
            let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(*size, *size * 2);
            let rag_engine = GraphRagEngine::new(graph, embeddings);
            let query_entity = EntityKey::from_hash("test_entity_0");
            let params = RagParameters::default();
            
            b.iter(|| {
                rag_engine.assemble_context(query_entity, &params)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("similarity_search", size), size, |b, &size| {
            let (graph, embeddings) = create_test_knowledge_graph_with_embeddings(*size, *size * 2);
            let rag_engine = GraphRagEngine::new(graph, embeddings);
            let query_embedding = vec![0.1; 64]; // Assuming 64-dim embeddings
            
            b.iter(|| {
                rag_engine.similarity_search(&query_embedding, 20)
            });
        });
    }
    
    group.finish();
}

// Benchmark memory management
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    for size in [1024, 4096, 16384].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("memory_allocation", size), size, |b, &size| {
            let mut memory_manager = MemoryManager::new();
            
            b.iter(|| {
                let ptr = memory_manager.allocate(*size).unwrap();
                memory_manager.deallocate(ptr, *size).unwrap();
            });
        });
        
        group.bench_with_input(BenchmarkId::new("memory_pool_allocation", size), size, |b, &size| {
            let pool_size = 1024 * 1024; // 1MB pool
            let mut memory_manager = MemoryManager::with_pool(pool_size);
            
            b.iter(|| {
                let ptr = memory_manager.allocate(*size).unwrap();
                memory_manager.deallocate(ptr, *size).unwrap();
            });
        });
    }
    
    group.finish();
}

// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(10));
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(BenchmarkId::new("concurrent_graph_access", thread_count), thread_count, |b, &thread_count| {
            use std::sync::{Arc, Mutex};
            use std::thread;
            
            let graph = Arc::new(Mutex::new(create_test_graph(1000, 2000)));
            
            b.iter(|| {
                let mut handles = Vec::new();
                
                for _ in 0..thread_count {
                    let graph_clone = Arc::clone(&graph);
                    let handle = thread::spawn(move || {
                        for i in 0..100 {
                            let entity_key = EntityKey::from_hash(&format!("test_entity_{}", i % 100));
                            let graph = graph_clone.lock().unwrap();
                            let _ = graph.get_entity(entity_key);
                        }
                    });
                    handles.push(handle);
                }
                
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
    }
    
    group.finish();
}

// Helper functions for benchmarks
fn generate_random_adjacency_matrix(rng: &mut DeterministicRng, size: usize, density: f64) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0f32; size]; size];
    
    for i in 0..size {
        for j in 0..size {
            if rng.gen::<f64>() < density {
                matrix[i][j] = rng.gen_range(0.1..1.0);
            }
        }
    }
    
    matrix
}

fn generate_random_vectors(rng: &mut DeterministicRng, count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn scalar_euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn create_test_knowledge_graph_with_embeddings(entity_count: usize, relationship_count: usize) -> (KnowledgeGraph, EmbeddingStore) {
    let graph = create_test_graph(entity_count, relationship_count);
    let embedding_dim = 64;
    let mut embeddings = EmbeddingStore::new(embedding_dim);
    let mut rng = DeterministicRng::new(RAG_TEST_SEED);
    
    // Add embeddings for all entities
    for i in 0..entity_count {
        let embedding: Vec<f32> = (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        embeddings.add_embedding(&format!("test_entity_{}", i), embedding).unwrap();
    }
    
    (graph, embeddings)
}

criterion_group!(
    benches,
    bench_entity_operations,
    bench_graph_operations,
    bench_csr_operations,
    bench_bloom_operations,
    bench_quantization_operations,
    bench_simd_operations,
    bench_rag_operations,
    bench_memory_operations,
    bench_concurrent_operations
);

criterion_main!(benches);