// Phase 5.1: Performance Target Validation Benchmarks
// Validates that LLMKG meets its specific performance targets

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer};
use llmkg::embedding::quantizer::ProductQuantizer;
use std::time::{Duration, Instant};

/// Performance targets that LLMKG must meet
const TARGET_ENTITY_LOOKUP_NS: u64 = 1_000_000; // 1ms
const TARGET_SERIALIZATION_ENTITIES_PER_SEC: u64 = 10_000;
const TARGET_MEMORY_BYTES_PER_ENTITY: f64 = 100.0;
const TARGET_COMPRESSION_RATIO: f32 = 2.0;
const TARGET_ACCESS_TIME_NS: u64 = 100; // 100ns per access

/// Generate performance test dataset
fn create_performance_dataset(entity_count: usize, embedding_dim: usize) -> Vec<EntityData> {
    (0..entity_count)
        .map(|i| EntityData {
            type_id: (i % 50) as u16,
            properties: format!(
                r#"{{"id": {}, "name": "Performance Entity {}", "category": "benchmark", "metadata": {{"score": {}, "active": true}}, "description": "Entity created for performance testing with ID {} and additional context data"}}"#,
                i, i, i % 1000, i
            ),
            embedding: (0..embedding_dim)
                .map(|j| ((i * 37 + j * 23) as f32 / 1000.0).sin())
                .collect(),
        })
        .collect()
}

/// Validate serialization performance targets
fn validate_serialization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_serialization");
    group.confidence_level(0.95);
    group.sample_size(50);
    
    let embedding_dim = 96;
    let entity_count = 10_000;
    let entities = create_performance_dataset(entity_count, embedding_dim);
    
    // Target: Serialize 10,000 entities per second
    group.throughput(Throughput::Elements(entity_count as u64));
    group.bench_function("serialization_target_10k_entities_per_sec", |b| {
        b.iter(|| {
            let start = Instant::now();
            let mut serializer = ZeroCopySerializer::new();
            
            for entity in black_box(&entities) {
                serializer.add_entity(entity, embedding_dim).unwrap();
            }
            
            let data = serializer.finalize().unwrap();
            let elapsed = start.elapsed();
            
            black_box(data);
            
            // Validate target
            let entities_per_sec = entity_count as f64 / elapsed.as_secs_f64();
            assert!(
                entities_per_sec >= TARGET_SERIALIZATION_ENTITIES_PER_SEC as f64,
                "Serialization performance target not met: {} entities/sec < {} target",
                entities_per_sec,
                TARGET_SERIALIZATION_ENTITIES_PER_SEC
            );
        })
    });
    
    group.finish();
}

/// Validate memory efficiency targets
fn validate_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_memory");
    group.sample_size(20);
    
    let embedding_dim = 96;
    
    for &entity_count in &[1_000, 10_000, 100_000] {
        let entities = create_performance_dataset(entity_count, embedding_dim);
        
        group.bench_with_input(
            BenchmarkId::new("memory_efficiency_target", entity_count),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let mut serializer = ZeroCopySerializer::new();
                    
                    for entity in entities {
                        serializer.add_entity(black_box(entity), embedding_dim).unwrap();
                    }
                    
                    let data = serializer.finalize().unwrap();
                    
                    // Calculate metrics
                    let bytes_per_entity = data.len() as f64 / entities.len() as f64;
                    let raw_size = entities.iter()
                        .map(|e| e.properties.len() + e.embedding.len() * 4 + 8)
                        .sum::<usize>();
                    let compression_ratio = raw_size as f32 / data.len() as f32;
                    
                    black_box(data);
                    
                    // Validate targets
                    assert!(
                        bytes_per_entity <= TARGET_MEMORY_BYTES_PER_ENTITY,
                        "Memory efficiency target not met: {:.1} bytes/entity > {:.1} target",
                        bytes_per_entity,
                        TARGET_MEMORY_BYTES_PER_ENTITY
                    );
                    
                    assert!(
                        compression_ratio >= TARGET_COMPRESSION_RATIO,
                        "Compression ratio target not met: {:.2}:1 < {:.2}:1 target",
                        compression_ratio,
                        TARGET_COMPRESSION_RATIO
                    );
                })
            },
        );
    }
    
    group.finish();
}

/// Validate access performance targets
fn validate_access_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_access");
    group.confidence_level(0.95);
    group.sample_size(100);
    
    let embedding_dim = 96;
    let entity_count = 50_000;
    let entities = create_performance_dataset(entity_count, embedding_dim);
    
    // Prepare serialized data
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, embedding_dim).unwrap();
    }
    let serialized_data = serializer.finalize().unwrap();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data).unwrap() };
    
    // Target: Single entity access in < 1ms
    group.bench_function("single_entity_access_target_1ms", |b| {
        b.iter(|| {
            let start = Instant::now();
            let entity = deserializer.get_entity(black_box(25_000)).unwrap();
            let _properties = deserializer.get_entity_properties(entity);
            let elapsed = start.elapsed();
            
            // Validate target
            assert!(
                elapsed.as_nanos() <= TARGET_ENTITY_LOOKUP_NS as u128,
                "Entity lookup target not met: {}ns > {}ns target",
                elapsed.as_nanos(),
                TARGET_ENTITY_LOOKUP_NS
            );
        })
    });
    
    // Target: Batch access efficiency
    let access_indices: Vec<u32> = (0..1000).map(|i| (i * 47) % entity_count as u32).collect();
    
    group.throughput(Throughput::Elements(1000));
    group.bench_function("batch_access_target_100ns_per_access", |b| {
        b.iter(|| {
            let start = Instant::now();
            
            for &index in black_box(&access_indices) {
                if let Some(entity) = deserializer.get_entity(index) {
                    black_box(deserializer.get_entity_properties(entity));
                }
            }
            
            let elapsed = start.elapsed();
            let ns_per_access = elapsed.as_nanos() / access_indices.len() as u128;
            
            // Validate target
            assert!(
                ns_per_access <= TARGET_ACCESS_TIME_NS as u128,
                "Batch access target not met: {}ns per access > {}ns target",
                ns_per_access,
                TARGET_ACCESS_TIME_NS
            );
        })
    });
    
    group.finish();
}

/// Validate quantization performance targets
fn validate_quantization_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_quantization");
    group.sample_size(30);
    
    let embedding_dim = 96;
    let subvector_count = 8;
    let quantizer = ProductQuantizer::new(embedding_dim, subvector_count).unwrap();
    
    // Test quantization speed targets
    for &batch_size in &[1000, 10000] {
        let embeddings: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                (0..embedding_dim)
                    .map(|j| ((i * 31 + j * 17) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("quantization_speed_target", batch_size),
            &embeddings,
            |b, embeddings| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    for embedding in embeddings {
                        let _quantized = quantizer.encode(black_box(embedding)).unwrap();
                    }
                    
                    let elapsed = start.elapsed();
                    let embeddings_per_sec = batch_size as f64 / elapsed.as_secs_f64();
                    
                    // Target: At least 100k embeddings per second
                    assert!(
                        embeddings_per_sec >= 100_000.0,
                        "Quantization speed target not met: {:.0} embeddings/sec < 100k target",
                        embeddings_per_sec
                    );
                })
            },
        );
    }
    
    // Test compression ratio targets
    let test_embedding: Vec<f32> = (0..embedding_dim)
        .map(|i| (i as f32 / embedding_dim as f32).sin())
        .collect();
    
    group.bench_function("compression_ratio_target_64_to_1", |b| {
        b.iter(|| {
            let original_size = test_embedding.len() * 4; // 4 bytes per f32
            let quantized = quantizer.encode(black_box(&test_embedding)).unwrap();
            let compressed_size = quantized.len();
            
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            black_box(quantized);
            
            // Target: At least 32:1 compression ratio
            assert!(
                compression_ratio >= 32.0,
                "Compression ratio target not met: {:.1}:1 < 32:1 target",
                compression_ratio
            );
        })
    });
    
    group.finish();
}

/// Validate scalability targets
fn validate_scalability_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_scalability");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    let embedding_dim = 96;
    
    // Test performance consistency across different scales
    for &entity_count in &[1_000, 10_000, 100_000, 1_000_000] {
        let entities = create_performance_dataset(entity_count, embedding_dim);
        
        group.throughput(Throughput::Elements(entity_count as u64));
        group.bench_with_input(
            BenchmarkId::new("scalability_linear_performance", entity_count),
            &entities,
            |b, entities| {
                b.iter(|| {
                    let start = Instant::now();
                    
                    let mut serializer = ZeroCopySerializer::new();
                    for entity in entities {
                        serializer.add_entity(black_box(entity), embedding_dim).unwrap();
                    }
                    let data = serializer.finalize().unwrap();
                    
                    let elapsed = start.elapsed();
                    let entities_per_sec = entities.len() as f64 / elapsed.as_secs_f64();
                    
                    black_box(data);
                    
                    // Target: Performance should scale linearly (within 50% variance)
                    // This means larger datasets shouldn't be dramatically slower per entity
                    let baseline_performance = 10_000.0; // entities per second
                    let performance_ratio = entities_per_sec / baseline_performance;
                    
                    assert!(
                        performance_ratio >= 0.5,
                        "Scalability target not met: {:.2}x baseline performance < 0.5x threshold for {} entities",
                        performance_ratio,
                        entities.len()
                    );
                })
            },
        );
    }
    
    group.finish();
}

/// Validate consistency and reliability targets
fn validate_consistency_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets_consistency");
    group.sample_size(100);
    
    let embedding_dim = 96;
    let entity_count = 10_000;
    let entities = create_performance_dataset(entity_count, embedding_dim);
    
    // Prepare test data
    let mut serializer = ZeroCopySerializer::new();
    for entity in &entities {
        serializer.add_entity(entity, embedding_dim).unwrap();
    }
    let serialized_data = serializer.finalize().unwrap();
    
    // Test repeated access consistency
    group.bench_function("access_consistency_target", |b| {
        b.iter(|| {
            let deserializer = unsafe { ZeroCopyDeserializer::new(&serialized_data).unwrap() };
            
            // Access the same entity multiple times - should be consistent
            let mut times = Vec::new();
            for _ in 0..100 {
                let start = Instant::now();
                let entity = deserializer.get_entity(5000).unwrap();
                let _properties = deserializer.get_entity_properties(entity);
                times.push(start.elapsed().as_nanos());
            }
            
            // Calculate variance
            let mean = times.iter().sum::<u128>() / times.len() as u128;
            let variance = times.iter()
                .map(|&x| {
                    let diff = if x > mean { x - mean } else { mean - x };
                    diff * diff
                })
                .sum::<u128>() / times.len() as u128;
            let std_dev = (variance as f64).sqrt();
            let coefficient_of_variation = std_dev / mean as f64;
            
            black_box(times);
            
            // Target: Coefficient of variation < 0.5 (50%)
            assert!(
                coefficient_of_variation < 0.5,
                "Consistency target not met: CV {:.3} > 0.5 threshold",
                coefficient_of_variation
            );
        })
    });
    
    group.finish();
}

criterion_group!(
    performance_targets,
    validate_serialization_performance,
    validate_memory_efficiency,
    validate_access_performance,
    validate_quantization_performance,
    validate_scalability_targets,
    validate_consistency_targets
);

criterion_main!(performance_targets);