use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use llmkg::cognitive::attention_manager::{AttentionManager, AttentionState, AttentionFocus, AttentionType, ExecutiveCommand};
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, MemoryItem};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::types::{EntityKey, EntityData};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use slotmap::SlotMap;

/// Creates test entity keys for benchmarking
fn create_entity_keys(count: usize) -> Vec<EntityKey> {
    let mut sm: SlotMap<EntityKey, EntityData> = SlotMap::with_key();
    (0..count)
        .map(|i| {
            sm.insert(EntityData {
                type_id: 1,
                properties: format!("entity_{}", i),
                embedding: vec![0.0; 64],
            })
        })
        .collect()
}

/// Creates a test attention manager
async fn create_attention_manager() -> AttentionManager {
    let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(Default::default()).unwrap());
    let orchestrator = Arc::new(
        CognitiveOrchestrator::new(brain_graph.clone(), CognitiveOrchestratorConfig::default())
            .await
            .unwrap()
    );
    let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
    let sdr_storage = Arc::new(SDRStorage::new(Default::default()));
    let working_memory = Arc::new(
        WorkingMemorySystem::new(activation_engine.clone(), sdr_storage)
            .await
            .unwrap()
    );
    
    AttentionManager::new(orchestrator, activation_engine, working_memory)
        .await
        .unwrap()
}

fn benchmark_attention_focusing(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("attention_focusing");
    
    for target_count in [1, 5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*target_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(target_count),
            target_count,
            |b, &target_count| {
                let targets = create_entity_keys(target_count);
                let mut manager = runtime.block_on(create_attention_manager());
                
                b.iter(|| {
                    runtime.block_on(async {
                        for target in &targets {
                            let focus = AttentionFocus {
                                target: *target,
                                intensity: 0.7,
                                duration: Duration::from_millis(10),
                            };
                            manager.focus_attention(focus).await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn benchmark_attention_switching(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("attention_switching");
    
    let targets = create_entity_keys(10);
    let mut manager = runtime.block_on(create_attention_manager());
    
    group.bench_function("executive_switch", |b| {
        b.iter(|| {
            runtime.block_on(async {
                for i in 0..9 {
                    let command = ExecutiveCommand::SwitchFocus {
                        from: targets[i],
                        to: targets[i + 1],
                        urgency: 0.8,
                    };
                    manager.execute_executive_command(command).await.unwrap();
                }
            });
        });
    });
    
    group.finish();
}

fn benchmark_attention_modes(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("attention_modes");
    
    let modes = vec![
        ("selective", AttentionType::Selective),
        ("divided", AttentionType::Divided),
        ("sustained", AttentionType::Sustained),
        ("executive", AttentionType::Executive),
        ("alternating", AttentionType::Alternating),
    ];
    
    for (name, mode) in modes {
        group.bench_function(name, |b| {
            let mut manager = runtime.block_on(create_attention_manager());
            let targets = create_entity_keys(5);
            
            b.iter(|| {
                runtime.block_on(async {
                    manager.set_attention_mode(mode.clone()).await.unwrap();
                    
                    for target in &targets {
                        let focus = AttentionFocus {
                            target: *target,
                            intensity: 0.6,
                            duration: Duration::from_millis(5),
                        };
                        manager.focus_attention(focus).await.unwrap();
                    }
                });
            });
        });
    }
    
    group.finish();
}

fn benchmark_snapshot_generation(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut manager = runtime.block_on(create_attention_manager());
    let targets = create_entity_keys(20);
    
    // Pre-populate attention state
    runtime.block_on(async {
        for target in &targets[..10] {
            let focus = AttentionFocus {
                target: *target,
                intensity: 0.5,
                duration: Duration::from_millis(20),
            };
            manager.focus_attention(focus).await.unwrap();
        }
    });
    
    c.bench_function("snapshot_generation", |b| {
        b.iter(|| {
            runtime.block_on(async {
                black_box(manager.get_attention_snapshot().await.unwrap());
            });
        });
    });
}

fn benchmark_cognitive_load_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_load");
    
    group.bench_function("update_load", |b| {
        let mut state = AttentionState::new();
        
        b.iter(|| {
            for load in [0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1].iter() {
                state.update_cognitive_load(black_box(*load));
            }
        });
    });
    
    group.finish();
}

fn benchmark_memory_load_calculation(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let manager = runtime.block_on(create_attention_manager());
    
    let mut group = c.benchmark_group("memory_load");
    
    for item_count in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*item_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(item_count),
            item_count,
            |b, &item_count| {
                let memory_items: Vec<MemoryItem> = (0..item_count)
                    .map(|i| MemoryItem {
                        content: MemoryContent::Concept(format!("concept_{}", i)),
                        activation_level: 0.5 + (i as f32 / item_count as f32) * 0.5,
                        timestamp: Instant::now(),
                        importance_score: 0.7,
                        access_count: 1,
                        decay_factor: 0.1,
                    })
                    .collect();
                
                b.iter(|| {
                    black_box(manager.calculate_memory_load(&memory_items));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_attention_capacity_under_load(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut manager = runtime.block_on(create_attention_manager());
    let targets = create_entity_keys(100);
    
    let mut group = c.benchmark_group("capacity_under_load");
    
    group.bench_function("increasing_load", |b| {
        b.iter(|| {
            runtime.block_on(async {
                // Gradually increase load by focusing on more targets
                for chunk in targets.chunks(10) {
                    for target in chunk {
                        let focus = AttentionFocus {
                            target: *target,
                            intensity: 0.6,
                            duration: Duration::from_millis(2),
                        };
                        manager.focus_attention(focus).await.unwrap();
                    }
                    
                    // Check capacity after each chunk
                    let snapshot = manager.get_attention_snapshot().await.unwrap();
                    black_box(snapshot.attention_capacity);
                }
            });
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_attention_focusing,
    benchmark_attention_switching,
    benchmark_attention_modes,
    benchmark_snapshot_generation,
    benchmark_cognitive_load_update,
    benchmark_memory_load_calculation,
    benchmark_attention_capacity_under_load
);

criterion_main!(benches);