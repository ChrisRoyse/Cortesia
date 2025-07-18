/*!
Phase 5.3: Scaling and Stress Test Benchmarks
Tests system behavior under extreme loads and scaling conditions
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// High-volume data generators for stress testing
fn generate_high_volume_entities(count: usize, embedding_dim: usize) -> Vec<(String, Vec<f32>, HashMap<String, String>)> {
    let mut rng = StdRng::seed_from_u64(42);
    
    (0..count).map(|i| {
        let id = format!("entity_{:08}", i);
        
        // Generate varied embedding patterns
        let embedding: Vec<f32> = (0..embedding_dim).map(|j| {
            let base = (i as f32 * 0.001 + j as f32 * 0.0001).sin();
            let noise = rng.gen_range(-0.1..0.1);
            base + noise
        }).collect();
        
        let mut metadata = HashMap::new();
        metadata.insert("batch".to_string(), (i / 1000).to_string());
        metadata.insert("category".to_string(), format!("cat_{}", i % 100));
        metadata.insert("priority".to_string(), (i % 10).to_string());
        metadata.insert("timestamp".to_string(), format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1));
        
        (id, embedding, metadata)
    }).collect()
}

fn benchmark_massive_entity_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("massive_entity_ingestion");
    group.measurement_time(Duration::from_secs(60)); // Longer measurement time for large datasets
    
    for entity_count in [100_000, 500_000, 1_000_000] {
        group.throughput(Throughput::Elements(entity_count as u64));
        
        group.bench_with_input(BenchmarkId::new("bulk_insert", entity_count), &entity_count, |b, &entity_count| {
            b.iter(|| {
                let temp_dir = TempDir::new().unwrap();
                let graph = EntityGraph::new();
                let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
                let mut interner = StringInterner::new();
                let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
                
                let start_time = Instant::now();
                
                // Generate and insert entities in batches
                let batch_size = 10_000;
                let mut total_inserted = 0;
                
                for batch_start in (0..*entity_count).step_by(batch_size) {
                    let batch_end = std::cmp::min(batch_start + batch_size, *entity_count);
                    let batch_entities = generate_high_volume_entities(batch_end - batch_start, 384);
                    
                    for (id, embedding, metadata) in batch_entities {
                        let key = EntityKey::from_hash(&id);
                        let content_id = interner.insert(&format!("Content for {}", id));
                        
                        let entity = Entity {
                            key,
                            content: content_id,
                            embedding,
                            metadata,
                        };
                        
                        graph.add_entity(entity);
                        total_inserted += 1;
                    }
                    
                    // Simulate periodic maintenance
                    if batch_start % 50_000 == 0 && batch_start > 0 {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                
                let duration = start_time.elapsed();
                black_box((total_inserted, duration))
            });
        });
    }
    
    group.finish();
}

fn benchmark_concurrent_read_write_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_read_write_stress");
    
    for thread_count in [4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::new("mixed_workload", thread_count), &thread_count, |b, &thread_count| {
            let temp_dir = TempDir::new().unwrap();
            let graph = Arc::new(RwLock::new(EntityGraph::new()));
            let quantizer = Arc::new(ProductQuantizer::new(256, 4, 64).unwrap());
            let interner = Arc::new(Mutex::new(StringInterner::new()));
            let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
            
            // Pre-populate with some data
            {
                let mut graph_guard = graph.write().unwrap();
                let mut interner_guard = interner.lock().unwrap();
                
                let initial_entities = generate_high_volume_entities(10_000, 256);
                for (id, embedding, metadata) in initial_entities {
                    let key = EntityKey::from_hash(&id);
                    let content_id = interner_guard.insert(&format!("Content for {}", id));
                    
                    let entity = Entity {
                        key,
                        content: content_id,
                        embedding,
                        metadata,
                    };
                    
                    graph_guard.add_entity(entity);
                }
            }
            
            b.iter(|| {
                let handles: Vec<_> = (0..*thread_count).map(|thread_id| {
                    let graph = graph.clone();
                    let interner = interner.clone();
                    let mut rng = StdRng::seed_from_u64(thread_id as u64);
                    
                    thread::spawn(move || {
                        let operations_per_thread = 1000;
                        let mut read_count = 0;
                        let mut write_count = 0;
                        
                        for i in 0..operations_per_thread {
                            let operation_type = rng.gen_range(0..100);
                            
                            if operation_type < 70 { // 70% reads
                                // Perform read operation
                                let query_embedding: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
                                
                                {
                                    let graph_guard = graph.read().unwrap();
                                    let results = graph_guard.find_similar_entities(&query_embedding, 5);
                                    black_box(results);
                                }
                                read_count += 1;
                            } else { // 30% writes
                                // Perform write operation
                                let new_id = format!("thread_{}_entity_{}", thread_id, i);
                                let embedding: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
                                
                                let mut metadata = HashMap::new();
                                metadata.insert("thread_id".to_string(), thread_id.to_string());
                                metadata.insert("operation".to_string(), i.to_string());
                                
                                let key = EntityKey::from_hash(&new_id);
                                let content_id = {
                                    let mut interner_guard = interner.lock().unwrap();
                                    interner_guard.insert(&format!("Content for {}", new_id))
                                };
                                
                                let entity = Entity {
                                    key,
                                    content: content_id,
                                    embedding,
                                    metadata,
                                };
                                
                                {
                                    let mut graph_guard = graph.write().unwrap();
                                    graph_guard.add_entity(entity);
                                }
                                write_count += 1;
                            }
                        }
                        
                        (read_count, write_count)
                    })
                }).collect();
                
                let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                black_box(results)
            });
        });
    }
    
    group.finish();
}

fn benchmark_memory_pressure_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure_handling");
    group.sample_size(10); // Fewer samples for memory-intensive tests
    
    for memory_target_mb in [500, 1000, 2000] {
        group.bench_with_input(BenchmarkId::new("memory_constrained_operations", memory_target_mb), &memory_target_mb, |b, &memory_target_mb| {
            b.iter(|| {
                let temp_dir = TempDir::new().unwrap();
                let graph = EntityGraph::new();
                let quantizer = ProductQuantizer::new(512, 8, 64).unwrap();
                let mut interner = StringInterner::new();
                let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
                
                // Calculate approximate entity count to reach target memory usage
                // Rough estimate: ~2KB per entity (embedding + metadata + overhead)
                let target_entity_count = (memory_target_mb * 1024) / 2;
                
                let mut inserted_count = 0;
                let mut peak_memory_kb = 0;
                
                for batch in 0..(target_entity_count / 1000) {
                    let batch_entities = generate_high_volume_entities(1000, 512);
                    
                    for (id, embedding, metadata) in batch_entities {
                        let key = EntityKey::from_hash(&id);
                        let content = format!("Large content block for entity {} with extensive metadata and detailed description", id);
                        let content_id = interner.insert(&content);
                        
                        let entity = Entity {
                            key,
                            content: content_id,
                            embedding,
                            metadata,
                        };
                        
                        graph.add_entity(entity);
                        inserted_count += 1;
                    }
                    
                    // Check memory usage periodically
                    if batch % 10 == 0 {
                        if let Ok(output) = std::process::Command::new("ps")
                            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                            .output()
                        {
                            if let Ok(memory_str) = String::from_utf8(output.stdout) {
                                if let Ok(memory_kb) = memory_str.trim().parse::<usize>() {
                                    peak_memory_kb = peak_memory_kb.max(memory_kb);
                                }
                            }
                        }
                        
                        // Perform some queries to test performance under memory pressure
                        let query_embedding: Vec<f32> = (0..512).map(|i| (i as f32 * 0.001).sin()).collect();
                        let results = graph.find_similar_entities(&query_embedding, 10);
                        black_box(results);
                    }
                }
                
                black_box((inserted_count, peak_memory_kb))
            });
        });
    }
    
    group.finish();
}

fn benchmark_query_scalability_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scalability_stress");
    
    for dataset_size in [50_000, 100_000, 500_000] {
        group.throughput(Throughput::Elements(dataset_size as u64));
        
        group.bench_with_input(BenchmarkId::new("high_frequency_queries", dataset_size), &dataset_size, |b, &dataset_size| {
            let temp_dir = TempDir::new().unwrap();
            let graph = EntityGraph::new();
            let quantizer = ProductQuantizer::new(384, 8, 48).unwrap();
            let mut interner = StringInterner::new();
            let storage = PersistentMMapStorage::new(Some(temp_dir.path())).unwrap();
            
            // Populate with varied data
            let entities = generate_high_volume_entities(*dataset_size, 384);
            for (id, embedding, metadata) in entities {
                let key = EntityKey::from_hash(&id);
                let content_id = interner.insert(&format!("Entity content: {}", id));
                
                let entity = Entity {
                    key,
                    content: content_id,
                    embedding,
                    metadata,
                };
                
                graph.add_entity(entity);
            }
            
            // Pre-generate query patterns
            let mut rng = StdRng::seed_from_u64(123);
            let query_patterns: Vec<Vec<f32>> = (0..100).map(|_| {
                (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect()
            }).collect();
            
            b.iter(|| {
                let start_time = Instant::now();
                let mut total_results = 0;
                let queries_per_batch = 1000;
                
                for batch in 0..10 { // 10 batches of 1000 queries each
                    for query_idx in 0..queries_per_batch {
                        let pattern_idx = (batch * queries_per_batch + query_idx) % query_patterns.len();
                        let query_embedding = &query_patterns[pattern_idx];
                        
                        let k = match batch {
                            0..=3 => 5,   // Small result sets
                            4..=6 => 20,  // Medium result sets
                            _ => 50,      // Large result sets
                        };
                        
                        let results = graph.find_similar_entities(query_embedding, k);
                        total_results += results.len();
                    }
                }
                
                let duration = start_time.elapsed();
                black_box((total_results, duration))
            });
        });
    }
    
    group.finish();
}

fn benchmark_persistence_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence_under_load");
    
    for write_intensity in [1000, 5000, 10000] {
        group.bench_with_input(BenchmarkId::new("heavy_write_persistence", write_intensity), &write_intensity, |b, &write_intensity| {
            b.iter(|| {
                let temp_dir = TempDir::new().unwrap();
                let graph = Arc::new(Mutex::new(EntityGraph::new()));
                let quantizer = Arc::new(ProductQuantizer::new(256, 4, 64).unwrap());
                let interner = Arc::new(Mutex::new(StringInterner::new()));
                let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
                
                let start_time = Instant::now();
                let mut write_operations = 0;
                
                // Simulate continuous write load with periodic persistence
                for batch in 0..(*write_intensity / 100) {
                    let batch_entities = generate_high_volume_entities(100, 256);
                    
                    for (id, embedding, metadata) in batch_entities {
                        let key = EntityKey::from_hash(&id);
                        let content_id = {
                            let mut interner_guard = interner.lock().unwrap();
                            interner_guard.insert(&format!("Persistent content for {}", id))
                        };
                        
                        let entity = Entity {
                            key,
                            content: content_id,
                            embedding,
                            metadata,
                        };
                        
                        {
                            let mut graph_guard = graph.lock().unwrap();
                            graph_guard.add_entity(entity);
                        }
                        
                        write_operations += 1;
                    }
                    
                    // Simulate persistence checkpoint every 1000 writes
                    if batch % 10 == 0 {
                        std::thread::sleep(Duration::from_millis(5)); // Simulate sync overhead
                    }
                }
                
                let duration = start_time.elapsed();
                black_box((write_operations, duration))
            });
        });
    }
    
    group.finish();
}

fn benchmark_degradation_under_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("degradation_under_stress");
    
    for stress_level in [1, 2, 4] {
        group.bench_with_input(BenchmarkId::new("system_stress_response", stress_level), &stress_level, |b, &stress_level| {
            b.iter(|| {
                let temp_dir = TempDir::new().unwrap();
                let graph = Arc::new(RwLock::new(EntityGraph::new()));
                let quantizer = Arc::new(ProductQuantizer::new(384, 8, 48).unwrap());
                let interner = Arc::new(Mutex::new(StringInterner::new()));
                let storage = Arc::new(PersistentMMapStorage::new(Some(temp_dir.path())).unwrap());
                
                // Pre-populate with substantial data
                {
                    let mut graph_guard = graph.write().unwrap();
                    let mut interner_guard = interner.lock().unwrap();
                    
                    let initial_entities = generate_high_volume_entities(20_000, 384);
                    for (id, embedding, metadata) in initial_entities {
                        let key = EntityKey::from_hash(&id);
                        let content_id = interner_guard.insert(&format!("Content for {}", id));
                        
                        let entity = Entity {
                            key,
                            content: content_id,
                            embedding,
                            metadata,
                        };
                        
                        graph_guard.add_entity(entity);
                    }
                }
                
                // Create stress conditions
                let num_stress_threads = *stress_level * 4;
                let query_latencies = Arc::new(Mutex::new(Vec::new()));
                
                let handles: Vec<_> = (0..num_stress_threads).map(|thread_id| {
                    let graph = graph.clone();
                    let interner = interner.clone();
                    let latencies = query_latencies.clone();
                    let mut rng = StdRng::seed_from_u64(thread_id as u64);
                    
                    thread::spawn(move || {
                        for _ in 0..500 {
                            let operation_type = rng.gen_range(0..100);
                            let start_op = Instant::now();
                            
                            if operation_type < 80 { // 80% queries
                                let query_embedding: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect();
                                
                                {
                                    let graph_guard = graph.read().unwrap();
                                    let results = graph_guard.find_similar_entities(&query_embedding, 10);
                                    black_box(results);
                                }
                            } else { // 20% writes with contention
                                let new_id = format!("stress_{}_{}", thread_id, rng.gen::<u32>());
                                let embedding: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect();
                                
                                let key = EntityKey::from_hash(&new_id);
                                let content_id = {
                                    let mut interner_guard = interner.lock().unwrap();
                                    interner_guard.insert(&format!("Stress content for {}", new_id))
                                };
                                
                                let entity = Entity {
                                    key,
                                    content: content_id,
                                    embedding,
                                    metadata: HashMap::new(),
                                };
                                
                                {
                                    let mut graph_guard = graph.write().unwrap();
                                    graph_guard.add_entity(entity);
                                }
                            }
                            
                            let op_latency = start_op.elapsed();
                            latencies.lock().unwrap().push(op_latency);
                        }
                    })
                }).collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
                
                // Analyze performance degradation
                let latencies = query_latencies.lock().unwrap();
                let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
                let max_latency = latencies.iter().max().unwrap_or(&Duration::ZERO);
                
                black_box((avg_latency, *max_latency, latencies.len()))
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    scaling_stress_benches,
    benchmark_massive_entity_ingestion,
    benchmark_concurrent_read_write_stress,
    benchmark_memory_pressure_handling,
    benchmark_query_scalability_stress,
    benchmark_persistence_under_load,
    benchmark_degradation_under_stress
);

criterion_main!(scaling_stress_benches);