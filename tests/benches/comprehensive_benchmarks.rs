// Phase 5.1: Comprehensive Criterion Benchmarks
// Enhanced benchmark suite covering all major LLMKG performance aspects

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer};
use llmkg::storage::string_interner::StringInterner;
use llmkg::embedding::quantizer::ProductQuantizer;
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use std::sync::Arc;

/// Generate realistic test entities for benchmarking
fn generate_test_entities(count: usize, embedding_dim: usize) -> Vec<EntityData> {
    (0..count)
        .map(|i| EntityData {
            type_id: (i % 20) as u16,
            properties: format!(
                r#"{{"id": {}, "name": "Entity {}", "category": "type_{}", "metadata": "additional data for entity {}", "timestamp": {}}}"#,
                i, i, i % 10, i, i * 1000
            ),
            embedding: (0..embedding_dim)
                .map(|j| ((i * 31 + j * 17) as f32 / 1000.0).sin())
                .collect(),
        })
        .collect()
}

/// Benchmark zero-copy serialization performance
fn benchmark_zero_copy_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_serialization");
    group.sample_size(50);
    
    for &entity_count in &[1_000, 10_000, 100_000] {
        let entities = generate_test_entities(entity_count, 96);
        
        group.throughput(Throughput::Elements(entity_count as u64));
        group.bench_with_input(
            BenchmarkId::new("serialize", entity_count),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let mut serializer = ZeroCopySerializer::new();
                    for entity in entities {
                        serializer.add_entity(black_box(entity), 96).unwrap();
                    }
                    black_box(serializer.finalize().unwrap())
                })
            },
        );
        
        // Benchmark deserialization
        let mut serializer = ZeroCopySerializer::new();
        for entity in &entities {
            serializer.add_entity(entity, 96).unwrap();
        }
        let serialized_data = serializer.finalize().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("deserialize", entity_count),
            &serialized_data,
            |b, data| {
                b.iter(|| {
                    black_box(unsafe { ZeroCopyDeserializer::new(black_box(data)).unwrap() })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark zero-copy access patterns
fn benchmark_zero_copy_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_access");
    
    let entity_count = 50_000;
    let entities = generate_test_entities(entity_count, 96);
    
    // Prepare serialized data
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, 96).unwrap();
    }
    let serialized_data = serializer.finalize().unwrap();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data).unwrap() };
    
    // Benchmark sequential access
    group.throughput(Throughput::Elements(entity_count as u64));
    group.bench_function("sequential_access", |b| {
        b.iter(|| {
            for i in 0..entity_count {
                black_box(deserializer.get_entity(i as u32));
            }
        })
    });
    
    // Benchmark random access
    let random_indices: Vec<u32> = (0..1000)
        .map(|i| ((i * 47) % entity_count) as u32)
        .collect();
    
    group.throughput(Throughput::Elements(1000));
    group.bench_function("random_access", |b| {
        b.iter(|| {
            for &index in &random_indices {
                black_box(deserializer.get_entity(black_box(index)));
            }
        })
    });
    
    // Benchmark iteration
    group.throughput(Throughput::Elements(entity_count as u64));
    group.bench_function("iteration", |b| {
        b.iter(|| {
            black_box(deserializer.iter_entities().count())
        })
    });
    
    group.finish();
}

/// Benchmark product quantization performance
fn benchmark_product_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_quantization");
    group.sample_size(30);
    
    let embedding_dim = 96;
    let subvector_count = 8;
    let quantizer = ProductQuantizer::new(embedding_dim, subvector_count).unwrap();
    
    // Test different batch sizes
    for &batch_size in &[100, 1000, 10000] {
        let embeddings: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                (0..embedding_dim)
                    .map(|j| ((i * 13 + j * 29) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("quantize_batch", batch_size),
            &embeddings,
            |b, embeddings| {
                b.iter(|| {
                    for embedding in embeddings {
                        black_box(quantizer.encode(black_box(embedding)).unwrap());
                    }
                })
            },
        );
        
        // Benchmark dequantization
        let quantized: Vec<Vec<u8>> = embeddings
            .iter()
            .map(|e| quantizer.encode(e).unwrap())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("dequantize_batch", batch_size),
            &quantized,
            |b, quantized| {
                b.iter(|| {
                    for codes in quantized {
                        black_box(quantizer.decode(black_box(codes)).unwrap());
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark string interning performance
fn benchmark_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");
    
    let interner = StringInterner::new();
    
    // Generate test strings with different patterns
    let unique_strings: Vec<String> = (0..10000)
        .map(|i| format!("unique_string_{}", i))
        .collect();
    
    let repeated_strings: Vec<String> = (0..10000)
        .map(|i| format!("repeated_string_{}", i % 100)) // Only 100 unique values
        .collect();
    
    // Benchmark unique string interning
    group.throughput(Throughput::Elements(10000));
    group.bench_function("intern_unique_strings", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            for string in &unique_strings {
                black_box(interner.intern(black_box(string)));
            }
        })
    });
    
    // Benchmark repeated string interning (should be much faster due to caching)
    group.throughput(Throughput::Elements(10000));
    group.bench_function("intern_repeated_strings", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            for string in &repeated_strings {
                black_box(interner.intern(black_box(string)));
            }
        })
    });
    
    // Benchmark string retrieval
    let interned_ids: Vec<_> = unique_strings[..1000]
        .iter()
        .map(|s| interner.intern(s))
        .collect();
    
    group.throughput(Throughput::Elements(1000));
    group.bench_function("retrieve_strings", |b| {
        b.iter(|| {
            for &id in &interned_ids {
                black_box(interner.get(black_box(id)));
            }
        })
    });
    
    group.finish();
}

/// Benchmark memory-mapped storage performance
fn benchmark_persistent_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistent_mmap");
    group.sample_size(20);
    
    let embedding_dim = 96;
    let subvector_count = 8;
    let quantizer = Arc::new(ProductQuantizer::new(embedding_dim, subvector_count).unwrap());
    
    for &entity_count in &[1000, 10000] {
        let entities = generate_test_entities(entity_count, embedding_dim);
        
        group.throughput(Throughput::Elements(entity_count as u64));
        group.bench_with_input(
            BenchmarkId::new("write_entities", entity_count),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let temp_path = std::env::temp_dir().join(format!("bench_mmap_{}.db", rand::random::<u32>()));
                    let mut storage = PersistentMMapStorage::new(Some(&temp_path), embedding_dim).unwrap();
                    
                    for (i, entity) in entities.iter().enumerate() {
                        let entity_key = llmkg::core::types::EntityKey::from_hash(&format!("entity_{}", i));
                        storage.add_entity(entity_key, black_box(entity), &entity.embedding).unwrap();
                    }
                    
                    // Storage automatically syncs on drop
                    
                    // Cleanup
                    let _ = std::fs::remove_file(temp_path);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark computation-heavy operations
fn benchmark_computational_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("computational_workloads");
    
    let embedding_dim = 384; // Larger dimension for stress testing
    let query_embedding: Vec<f32> = (0..embedding_dim)
        .map(|i| (i as f32 / embedding_dim as f32 * std::f32::consts::PI).sin())
        .collect();
    
    // Test different entity corpus sizes
    for &corpus_size in &[1000, 10000, 50000] {
        let entities = generate_test_entities(corpus_size, embedding_dim);
        
        // Benchmark brute-force similarity computation
        group.throughput(Throughput::Elements(corpus_size as u64));
        group.bench_with_input(
            BenchmarkId::new("brute_force_similarity", corpus_size),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let mut similarities = Vec::new();
                    for entity in entities {
                        let similarity = compute_cosine_similarity(
                            black_box(&query_embedding),
                            black_box(&entity.embedding)
                        );
                        similarities.push((similarity, entity.type_id));
                    }
                    // Get top 20 results
                    similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    black_box(&similarities[..20.min(similarities.len())]);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    let entity_count = 10000;
    let embedding_dim = 96;
    
    // Benchmark stack vs heap allocation patterns
    group.bench_function("stack_allocation_pattern", |b| {
        b.iter(|| {
            for _i in 0..entity_count {
                let embedding: [f32; 96] = [0.0; 96]; // Stack allocated
                black_box(embedding);
            }
        })
    });
    
    group.bench_function("heap_allocation_pattern", |b| {
        b.iter(|| {
            for _i in 0..entity_count {
                let embedding: Vec<f32> = vec![0.0; embedding_dim]; // Heap allocated
                black_box(embedding);
            }
        })
    });
    
    // Benchmark arena allocation simulation
    group.bench_function("arena_allocation_pattern", |b| {
        b.iter(|| {
            let mut arena = Vec::with_capacity(entity_count * embedding_dim);
            for i in 0..entity_count {
                arena.extend(vec![i as f32; embedding_dim]);
            }
            black_box(arena);
        })
    });
    
    group.finish();
}

/// Benchmark concurrency patterns
fn benchmark_concurrency(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency");
    group.sample_size(20);
    
    let entity_count = 10000;
    let embedding_dim = 96;
    let entities = generate_test_entities(entity_count, embedding_dim);
    
    // Prepare zero-copy data for concurrent access
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, embedding_dim).unwrap();
    }
    let serialized_data = Arc::new(serializer.finalize().unwrap());
    
    // Benchmark sequential vs parallel processing
    group.bench_function("sequential_processing", |b| {
        b.iter(|| {
            let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data).unwrap() };
            let mut total_properties_len = 0;
            for entity in deserializer.iter_entities() {
                let properties = deserializer.get_entity_properties(entity);
                total_properties_len += properties.len();
            }
            black_box(total_properties_len);
        })
    });
    
    group.bench_function("parallel_processing", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data).unwrap() };
            let total_properties_len: usize = (0..entity_count)
                .into_par_iter()
                .map(|i| {
                    if let Some(entity) = deserializer.get_entity(i as u32) {
                        deserializer.get_entity_properties(entity).len()
                    } else {
                        0
                    }
                })
                .sum();
            black_box(total_properties_len);
        })
    });
    
    group.finish();
}

/// Benchmark edge cases and stress scenarios
fn benchmark_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");
    
    // Large embedding dimensions
    for &embedding_dim in &[512, 1024, 2048] {
        let entities = generate_test_entities(1000, embedding_dim);
        let mut serializer = ZeroCopySerializer::new();
        for entity in &entities {
            serializer.add_entity(entity, embedding_dim).unwrap();
        }
        
        group.bench_with_input(
            BenchmarkId::new("large_embeddings", embedding_dim),
            &embedding_dim,
            |b, &dim| {
                b.iter(|| {
                    let mut serializer = ZeroCopySerializer::new();
                    let entity = &entities[0];
                    black_box(serializer.add_entity(black_box(entity), dim).unwrap());
                })
            },
        );
    }
    
    // Very long property strings
    let long_properties = vec![
        "short".to_string(),
        "a".repeat(1000),
        "b".repeat(10000),
        "c".repeat(100000),
    ];
    
    for (_i, properties) in long_properties.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("long_properties", properties.len()),
            properties,
            |b, props| {
                b.iter(|| {
                    let entity = EntityData {
                        type_id: 1,
                        properties: black_box(props.clone()),
                        embedding: vec![0.5; 96],
                    };
                    let mut serializer = ZeroCopySerializer::new();
                    black_box(serializer.add_entity(&entity, 96).unwrap());
                })
            },
        );
    }
    
    group.finish();
}

/// Helper function for similarity computation
fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

criterion_group!(
    comprehensive_benches,
    benchmark_zero_copy_serialization,
    benchmark_zero_copy_access,
    benchmark_product_quantization,
    benchmark_string_interning,
    benchmark_persistent_mmap,
    benchmark_computational_workloads,
    benchmark_memory_patterns,
    benchmark_concurrency,
    benchmark_edge_cases
);

criterion_main!(comprehensive_benches);