use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use llmkg::core::{
    activation_engine::{ActivationPropagationEngine, ActivationConfig},
    brain_types::{
        BrainInspiredEntity, BrainInspiredRelationship, LogicGate, LogicGateType,
        EntityDirection, RelationType, ActivationPattern,
    },
    types::EntityKey,
};
use std::collections::HashMap;
use std::time::SystemTime;

/// Helper function to create a simple entity
fn create_entity(id: EntityKey, activation: f32, direction: EntityDirection) -> BrainInspiredEntity {
    BrainInspiredEntity {
        id,
        concept_id: format!("concept_{}", id.0),
        direction,
        properties: HashMap::new(),
        embedding: vec![0.0; 128],
        activation_state: activation,
        last_activation: SystemTime::now(),
        last_update: SystemTime::now(),
    }
}

/// Helper function to create a relationship
fn create_relationship(
    source: EntityKey,
    target: EntityKey,
    weight: f32,
    is_inhibitory: bool,
) -> BrainInspiredRelationship {
    BrainInspiredRelationship {
        source,
        target,
        source_key: source,
        target_key: target,
        relation_type: RelationType::RelatedTo,
        weight,
        strength: weight,
        is_inhibitory,
        temporal_decay: 0.05,
        last_strengthened: SystemTime::now(),
        last_update: SystemTime::now(),
        activation_count: 0,
        usage_count: 0,
        creation_time: SystemTime::now(),
        ingestion_time: SystemTime::now(),
        metadata: HashMap::new(),
    }
}

/// Create a small network (10 nodes)
async fn create_small_network(engine: &ActivationPropagationEngine) {
    // Create 10 entities
    for i in 0..10 {
        let direction = match i {
            0..=2 => EntityDirection::Input,
            7..=9 => EntityDirection::Output,
            _ => EntityDirection::Hidden,
        };
        engine.add_entity(create_entity(EntityKey(i), 0.0, direction)).await.unwrap();
    }

    // Create simple chain connections
    for i in 0..9 {
        let is_inhibitory = i % 3 == 0; // Every 3rd connection is inhibitory
        engine.add_relationship(
            create_relationship(EntityKey(i), EntityKey(i + 1), 0.8, is_inhibitory)
        ).await.unwrap();
    }
}

/// Create a medium network (1000 nodes)
async fn create_medium_network(engine: &ActivationPropagationEngine) {
    // Create 1000 entities
    for i in 0..1000 {
        let direction = match i {
            0..=50 => EntityDirection::Input,
            950..=999 => EntityDirection::Output,
            _ => EntityDirection::Hidden,
        };
        engine.add_entity(create_entity(EntityKey(i), 0.0, direction)).await.unwrap();
    }

    // Create random connections (average 5 per node)
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    for i in 0..1000 {
        for _ in 0..5 {
            let target = rng.gen_range(0..1000);
            if target != i {
                let weight = rng.gen_range(0.1..1.0);
                let is_inhibitory = rng.gen_bool(0.2); // 20% inhibitory
                engine.add_relationship(
                    create_relationship(EntityKey(i), EntityKey(target), weight, is_inhibitory)
                ).await.unwrap();
            }
        }
    }
}

/// Create a large network (100k nodes) - for memory/scaling tests
async fn create_large_network(engine: &ActivationPropagationEngine) {
    // Create 100k entities
    for i in 0..100_000 {
        let direction = match i {
            0..=1000 => EntityDirection::Input,
            99_000..=99_999 => EntityDirection::Output,
            _ => EntityDirection::Hidden,
        };
        engine.add_entity(create_entity(EntityKey(i), 0.0, direction)).await.unwrap();
    }

    // Create sparse connections (average 3 per node)
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    for i in 0..100_000 {
        for _ in 0..3 {
            let target = rng.gen_range(0..100_000);
            if target != i {
                let weight = rng.gen_range(0.1..1.0);
                let is_inhibitory = rng.gen_bool(0.15);
                engine.add_relationship(
                    create_relationship(EntityKey(i), EntityKey(target), weight, is_inhibitory)
                ).await.unwrap();
            }
        }
    }
}

/// Create a deep network (100 layers, 10 nodes per layer)
async fn create_deep_network(engine: &ActivationPropagationEngine) {
    let layers = 100;
    let nodes_per_layer = 10;
    
    // Create entities layer by layer
    for layer in 0..layers {
        for node in 0..nodes_per_layer {
            let id = layer * nodes_per_layer + node;
            let direction = match layer {
                0 => EntityDirection::Input,
                l if l == layers - 1 => EntityDirection::Output,
                _ => EntityDirection::Hidden,
            };
            engine.add_entity(create_entity(EntityKey(id), 0.0, direction)).await.unwrap();
        }
    }

    // Connect adjacent layers
    for layer in 0..layers - 1 {
        for from_node in 0..nodes_per_layer {
            for to_node in 0..nodes_per_layer {
                let from_id = layer * nodes_per_layer + from_node;
                let to_id = (layer + 1) * nodes_per_layer + to_node;
                let weight = 0.7 + (from_node as f32 * 0.01);
                let is_inhibitory = (from_node + to_node) % 5 == 0;
                engine.add_relationship(
                    create_relationship(EntityKey(from_id), EntityKey(to_id), weight, is_inhibitory)
                ).await.unwrap();
            }
        }
    }
}

/// Create a wide network (10 layers, 1000 nodes per layer)
async fn create_wide_network(engine: &ActivationPropagationEngine) {
    let layers = 10;
    let nodes_per_layer = 1000;
    
    // Create entities layer by layer
    for layer in 0..layers {
        for node in 0..nodes_per_layer {
            let id = layer * nodes_per_layer + node;
            let direction = match layer {
                0 => EntityDirection::Input,
                l if l == layers - 1 => EntityDirection::Output,
                _ => EntityDirection::Hidden,
            };
            engine.add_entity(create_entity(EntityKey(id), 0.0, direction)).await.unwrap();
        }
    }

    // Connect adjacent layers sparsely (10% connectivity)
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    for layer in 0..layers - 1 {
        for from_node in 0..nodes_per_layer {
            // Each node connects to ~100 nodes in next layer
            for _ in 0..100 {
                let to_node = rng.gen_range(0..nodes_per_layer);
                let from_id = layer * nodes_per_layer + from_node;
                let to_id = (layer + 1) * nodes_per_layer + to_node;
                let weight = rng.gen_range(0.3..0.9);
                let is_inhibitory = rng.gen_bool(0.2);
                engine.add_relationship(
                    create_relationship(EntityKey(from_id), EntityKey(to_id), weight, is_inhibitory)
                ).await.unwrap();
            }
        }
    }
}

/// Create initial activation pattern
fn create_activation_pattern(num_active: usize, max_id: usize) -> ActivationPattern {
    let mut activations = HashMap::new();
    for i in 0..num_active {
        activations.insert(EntityKey(i % max_id), 1.0);
    }
    
    ActivationPattern {
        activations,
        timestamp: SystemTime::now(),
        query: "benchmark_query".to_string(),
    }
}

fn bench_propagation_small(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("activation_small_network", |b| {
        b.to_async(&runtime).iter_batched(
            || {
                let config = ActivationConfig::default();
                let engine = ActivationPropagationEngine::new(config);
                let setup = async {
                    create_small_network(&engine).await;
                    (engine, create_activation_pattern(3, 10))
                };
                runtime.block_on(setup)
            },
            |(engine, pattern)| async move {
                let result = engine.propagate_activation(&pattern).await.unwrap();
                black_box(result)
            },
            BatchSize::SmallInput
        )
    });
}

fn bench_propagation_scaling(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("activation_scaling");
    
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&runtime).iter_batched(
                || {
                    let config = ActivationConfig::default();
                    let engine = ActivationPropagationEngine::new(config);
                    let setup = async {
                        // Create network of given size
                        for i in 0..size {
                            let direction = match i {
                                0 => EntityDirection::Input,
                                n if n == size - 1 => EntityDirection::Output,
                                _ => EntityDirection::Hidden,
                            };
                            engine.add_entity(create_entity(EntityKey(i), 0.0, direction)).await.unwrap();
                        }
                        
                        // Create chain connections
                        for i in 0..size - 1 {
                            engine.add_relationship(
                                create_relationship(EntityKey(i), EntityKey(i + 1), 0.8, false)
                            ).await.unwrap();
                        }
                        
                        (engine, create_activation_pattern(1, size))
                    };
                    runtime.block_on(setup)
                },
                |(engine, pattern)| async move {
                    let result = engine.propagate_activation(&pattern).await.unwrap();
                    black_box(result)
                },
                BatchSize::SmallInput
            )
        });
    }
    group.finish();
}

fn bench_inhibition_performance(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("inhibition_overhead");
    
    // Benchmark with different percentages of inhibitory connections
    for inhibition_percent in [0.0, 0.2, 0.5, 0.8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", (inhibition_percent * 100.0) as u32)),
            inhibition_percent,
            |b, &inhibition_percent| {
                b.to_async(&runtime).iter_batched(
                    || {
                        let config = ActivationConfig::default();
                        let engine = ActivationPropagationEngine::new(config);
                        let setup = async {
                            // Create medium network with varying inhibition
                            use rand::{Rng, SeedableRng};
                            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                            
                            for i in 0..1000 {
                                engine.add_entity(create_entity(EntityKey(i), 0.0, EntityDirection::Hidden)).await.unwrap();
                            }
                            
                            for i in 0..1000 {
                                for _ in 0..5 {
                                    let target = rng.gen_range(0..1000);
                                    if target != i {
                                        let is_inhibitory = rng.gen_bool(inhibition_percent);
                                        engine.add_relationship(
                                            create_relationship(EntityKey(i), EntityKey(target), 0.7, is_inhibitory)
                                        ).await.unwrap();
                                    }
                                }
                            }
                            
                            (engine, create_activation_pattern(50, 1000))
                        };
                        runtime.block_on(setup)
                    },
                    |(engine, pattern)| async move {
                        let result = engine.propagate_activation(&pattern).await.unwrap();
                        black_box(result)
                    },
                    BatchSize::SmallInput
                )
            }
        );
    }
    group.finish();
}

fn bench_convergence_time(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("convergence_vs_size");
    
    // Test convergence time for different network sizes
    for size in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&runtime).iter_batched(
                || {
                    let mut config = ActivationConfig::default();
                    config.max_iterations = 1000; // Allow more iterations to test convergence
                    let engine = ActivationPropagationEngine::new(config);
                    let setup = async {
                        // Create random network
                        use rand::{Rng, SeedableRng};
                        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                        
                        for i in 0..size {
                            engine.add_entity(create_entity(EntityKey(i), 0.0, EntityDirection::Hidden)).await.unwrap();
                        }
                        
                        // Create random connections
                        for i in 0..size {
                            let num_connections = rng.gen_range(3..8);
                            for _ in 0..num_connections {
                                let target = rng.gen_range(0..size);
                                if target != i {
                                    let weight = rng.gen_range(0.1..0.9);
                                    let is_inhibitory = rng.gen_bool(0.2);
                                    engine.add_relationship(
                                        create_relationship(EntityKey(i), EntityKey(target), weight, is_inhibitory)
                                    ).await.unwrap();
                                }
                            }
                        }
                        
                        (engine, create_activation_pattern(size / 10, size))
                    };
                    runtime.block_on(setup)
                },
                |(engine, pattern)| async move {
                    let result = engine.propagate_activation(&pattern).await.unwrap();
                    black_box((result.converged, result.iterations_completed))
                },
                BatchSize::SmallInput
            )
        });
    }
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10); // Fewer samples for memory-intensive benchmarks
    
    // Benchmark memory usage for different network sizes
    for size in [1000, 5000, 10000, 20000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&runtime).iter_batched(
                || {
                    let config = ActivationConfig::default();
                    let engine = ActivationPropagationEngine::new(config);
                    let setup = async {
                        // Create network with realistic density
                        use rand::{Rng, SeedableRng};
                        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                        
                        for i in 0..size {
                            let entity = BrainInspiredEntity {
                                id: EntityKey(i),
                                concept_id: format!("concept_{}", i),
                                direction: EntityDirection::Hidden,
                                properties: HashMap::new(),
                                embedding: vec![0.0; 384], // Realistic embedding size
                                activation_state: 0.0,
                                last_activation: SystemTime::now(),
                                last_update: SystemTime::now(),
                            };
                            engine.add_entity(entity).await.unwrap();
                        }
                        
                        // Average 5 connections per node
                        for i in 0..size {
                            for _ in 0..5 {
                                let target = rng.gen_range(0..size);
                                if target != i {
                                    let weight = rng.gen_range(0.1..0.9);
                                    let is_inhibitory = rng.gen_bool(0.2);
                                    engine.add_relationship(
                                        create_relationship(EntityKey(i), EntityKey(target), weight, is_inhibitory)
                                    ).await.unwrap();
                                }
                            }
                        }
                        
                        // Add some logic gates
                        for i in 0..size / 100 {
                            let gate = LogicGate {
                                gate_id: EntityKey(size + i),
                                gate_type: LogicGateType::And,
                                input_nodes: vec![EntityKey(i * 2), EntityKey(i * 2 + 1)],
                                output_nodes: vec![EntityKey(i * 3)],
                                threshold: 0.5,
                                weight_matrix: vec![1.0, 1.0],
                            };
                            engine.add_logic_gate(gate).await.unwrap();
                        }
                        
                        (engine, create_activation_pattern(size / 20, size))
                    };
                    runtime.block_on(setup)
                },
                |(engine, pattern)| async move {
                    let result = engine.propagate_activation(&pattern).await.unwrap();
                    let stats = engine.get_activation_statistics().await.unwrap();
                    black_box((result.total_energy, stats))
                },
                BatchSize::SmallInput
            )
        });
    }
    group.finish();
}

// Additional benchmarks for specific scenarios

fn bench_deep_network_propagation(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("activation_deep_network", |b| {
        b.to_async(&runtime).iter_batched(
            || {
                let mut config = ActivationConfig::default();
                config.max_iterations = 200; // Deep networks need more iterations
                let engine = ActivationPropagationEngine::new(config);
                let setup = async {
                    create_deep_network(&engine).await;
                    // Activate first layer
                    let mut activations = HashMap::new();
                    for i in 0..10 {
                        activations.insert(EntityKey(i), 1.0);
                    }
                    let pattern = ActivationPattern {
                        activations,
                        timestamp: SystemTime::now(),
                        query: "deep_network_test".to_string(),
                    };
                    (engine, pattern)
                };
                runtime.block_on(setup)
            },
            |(engine, pattern)| async move {
                let result = engine.propagate_activation(&pattern).await.unwrap();
                black_box(result)
            },
            BatchSize::SmallInput
        )
    });
}

fn bench_wide_network_propagation(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("activation_wide_network", |b| {
        b.to_async(&runtime).iter_batched(
            || {
                let config = ActivationConfig::default();
                let engine = ActivationPropagationEngine::new(config);
                let setup = async {
                    create_wide_network(&engine).await;
                    // Activate 10% of first layer
                    let mut activations = HashMap::new();
                    for i in 0..100 {
                        activations.insert(EntityKey(i), 1.0);
                    }
                    let pattern = ActivationPattern {
                        activations,
                        timestamp: SystemTime::now(),
                        query: "wide_network_test".to_string(),
                    };
                    (engine, pattern)
                };
                runtime.block_on(setup)
            },
            |(engine, pattern)| async move {
                let result = engine.propagate_activation(&pattern).await.unwrap();
                black_box(result)
            },
            BatchSize::SmallInput
        )
    });
}

fn bench_logic_gates_performance(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("logic_gates");
    
    // Benchmark different gate types
    for gate_type in [LogicGateType::And, LogicGateType::Or, LogicGateType::Xor, LogicGateType::Threshold].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", gate_type)),
            gate_type,
            |b, &gate_type| {
                b.to_async(&runtime).iter_batched(
                    || {
                        let config = ActivationConfig::default();
                        let engine = ActivationPropagationEngine::new(config);
                        let setup = async {
                            // Create network with many logic gates
                            let num_gates = 100;
                            let inputs_per_gate = 4;
                            
                            // Create entities
                            for i in 0..num_gates * (inputs_per_gate + 1) {
                                engine.add_entity(create_entity(EntityKey(i), 0.0, EntityDirection::Hidden)).await.unwrap();
                            }
                            
                            // Create logic gates
                            for i in 0..num_gates {
                                let base_idx = i * (inputs_per_gate + 1);
                                let inputs: Vec<EntityKey> = (0..inputs_per_gate)
                                    .map(|j| EntityKey(base_idx + j))
                                    .collect();
                                let output = EntityKey(base_idx + inputs_per_gate);
                                
                                let gate = LogicGate {
                                    gate_id: EntityKey(1000000 + i), // High ID to avoid conflicts
                                    gate_type,
                                    input_nodes: inputs,
                                    output_nodes: vec![output],
                                    threshold: 0.5,
                                    weight_matrix: vec![1.0; inputs_per_gate],
                                };
                                engine.add_logic_gate(gate).await.unwrap();
                            }
                            
                            // Activate half the inputs
                            let mut activations = HashMap::new();
                            for i in 0..num_gates {
                                let base_idx = i * (inputs_per_gate + 1);
                                for j in 0..inputs_per_gate / 2 {
                                    activations.insert(EntityKey(base_idx + j), 1.0);
                                }
                            }
                            
                            let pattern = ActivationPattern {
                                activations,
                                timestamp: SystemTime::now(),
                                query: "logic_gate_test".to_string(),
                            };
                            (engine, pattern)
                        };
                        runtime.block_on(setup)
                    },
                    |(engine, pattern)| async move {
                        let result = engine.propagate_activation(&pattern).await.unwrap();
                        black_box(result)
                    },
                    BatchSize::SmallInput
                )
            }
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_propagation_small,
    bench_propagation_scaling,
    bench_inhibition_performance,
    bench_convergence_time,
    bench_memory_usage,
    bench_deep_network_propagation,
    bench_wide_network_propagation,
    bench_logic_gates_performance
);

criterion_main!(benches);