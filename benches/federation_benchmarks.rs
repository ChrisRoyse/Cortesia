// Benchmarks for federation coordinator with 2PC

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use llmkg::federation::{
    FederationCoordinator,
    coordinator::{
        TransactionId, TransactionMetadata, TransactionPriority,
        IsolationLevel, ConsistencyMode, TransactionOperation, OperationType
    },
    types::DatabaseId,
    registry::{DatabaseRegistry, DatabaseDescriptor, DatabaseType, DatabaseStatus, DatabaseMetadata},
};
use llmkg::federation::types::DatabaseCapabilities;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use tokio::runtime::Runtime;

fn create_test_registry() -> Arc<DatabaseRegistry> {
    let registry = Arc::new(DatabaseRegistry::new().expect("Failed to create registry"));
    
    let rt = Runtime::new().unwrap();
    
    // Register test databases
    for i in 0..3 {
        let db = DatabaseDescriptor {
            id: DatabaseId::new(format!("db_{}", i)),
            name: format!("Test Database {}", i),
            description: Some(format!("Test database {}", i)),
            connection_string: ":memory:".to_string(),
            database_type: DatabaseType::InMemory,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("bench".to_string()),
                tags: vec![format!("bench_{}", i)],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        rt.block_on(registry.register(db)).expect("Failed to register database");
    }
    
    registry
}

fn bench_transaction_lifecycle(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = create_test_registry();
    let coordinator = rt.block_on(async {
        Arc::new(FederationCoordinator::new(registry).await.expect("Failed to create coordinator"))
    });
    
    c.bench_function("transaction_begin", |b| {
        b.iter(|| {
            rt.block_on(async {
                let metadata = TransactionMetadata {
                    initiator: Some("bench".to_string()),
                    description: Some("Benchmark transaction".to_string()),
                    priority: TransactionPriority::Normal,
                    isolation_level: IsolationLevel::ReadCommitted,
                    consistency_mode: ConsistencyMode::Eventual,
                };
                
                let databases = vec![DatabaseId::new("db_0".to_string())];
                black_box(coordinator.begin_transaction(databases, metadata).await.unwrap())
            })
        })
    });
}

fn bench_2pc_phases(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = create_test_registry();
    let coordinator = rt.block_on(async {
        Arc::new(FederationCoordinator::new(registry).await.expect("Failed to create coordinator"))
    });
    
    let mut group = c.benchmark_group("2pc_phases");
    
    for num_databases in [1, 2, 3].iter() {
        group.bench_with_input(
            BenchmarkId::new("prepare_commit", num_databases),
            num_databases,
            |b, &num_dbs| {
                b.iter(|| {
                    rt.block_on(async {
                        let metadata = TransactionMetadata {
                            initiator: Some("bench".to_string()),
                            description: Some("Benchmark 2PC".to_string()),
                            priority: TransactionPriority::Normal,
                            isolation_level: IsolationLevel::ReadCommitted,
                            consistency_mode: ConsistencyMode::Eventual,
                        };
                        
                        let databases: Vec<DatabaseId> = (0..num_dbs)
                            .map(|i| DatabaseId::new(format!("db_{}", i)))
                            .collect();
                        
                        let tx_id = coordinator.begin_transaction(databases, metadata).await.unwrap();
                        
                        // Add operations
                        for i in 0..num_dbs {
                            let mut data = HashMap::new();
                            data.insert("id".to_string(), serde_json::json!(i));
                            
                            let op = TransactionOperation {
                                operation_id: format!("op_{}", i),
                                database_id: DatabaseId::new(format!("db_{}", i)),
                                operation_type: OperationType::CreateEntity {
                                    entity_id: format!("entity_{}", i),
                                    entity_data: data,
                                },
                                parameters: HashMap::new(),
                                dependencies: vec![],
                                status: llmkg::federation::coordinator::OperationStatus::Pending,
                            };
                            
                            coordinator.add_operation(&tx_id, op).await.unwrap();
                        }
                        
                        // Execute 2PC
                        let prepared = coordinator.prepare_transaction(&tx_id).await.unwrap();
                        black_box(prepared)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_transactions(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = create_test_registry();
    let coordinator = rt.block_on(async {
        Arc::new(FederationCoordinator::new(registry).await.expect("Failed to create coordinator"))
    });
    
    let mut group = c.benchmark_group("concurrent_transactions");
    
    for num_concurrent in [1, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent", num_concurrent),
            num_concurrent,
            |b, &num| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = vec![];
                        
                        for i in 0..num {
                            let coordinator_clone = coordinator.clone();
                            let handle = tokio::spawn(async move {
                                let metadata = TransactionMetadata {
                                    initiator: Some(format!("bench_{}", i)),
                                    description: Some("Concurrent bench".to_string()),
                                    priority: TransactionPriority::Normal,
                                    isolation_level: IsolationLevel::ReadCommitted,
                                    consistency_mode: ConsistencyMode::Eventual,
                                };
                                
                                let databases = vec![DatabaseId::new("db_0".to_string())];
                                let tx_id = coordinator_clone.begin_transaction(databases, metadata).await.unwrap();
                                
                                let mut data = HashMap::new();
                                data.insert("concurrent".to_string(), serde_json::json!(i));
                                
                                let op = TransactionOperation {
                                    operation_id: format!("op_concurrent_{}", i),
                                    database_id: DatabaseId::new("db_0".to_string()),
                                    operation_type: OperationType::CreateEntity {
                                        entity_id: format!("concurrent_entity_{}", i),
                                        entity_data: data,
                                    },
                                    parameters: HashMap::new(),
                                    dependencies: vec![],
                                    status: llmkg::federation::coordinator::OperationStatus::Pending,
                                };
                                
                                coordinator_clone.add_operation(&tx_id, op).await.unwrap();
                                coordinator_clone.prepare_transaction(&tx_id).await.unwrap()
                            });
                            
                            handles.push(handle);
                        }
                        
                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn bench_operation_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = create_test_registry();
    let coordinator = rt.block_on(async {
        Arc::new(FederationCoordinator::new(registry).await.expect("Failed to create coordinator"))
    });
    
    let mut group = c.benchmark_group("operation_throughput");
    
    for num_operations in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("operations", num_operations),
            num_operations,
            |b, &num_ops| {
                b.iter(|| {
                    rt.block_on(async {
                        let metadata = TransactionMetadata {
                            initiator: Some("bench".to_string()),
                            description: Some("Operation throughput bench".to_string()),
                            priority: TransactionPriority::High,
                            isolation_level: IsolationLevel::ReadCommitted,
                            consistency_mode: ConsistencyMode::Eventual,
                        };
                        
                        let databases = vec![
                            DatabaseId::new("db_0".to_string()),
                            DatabaseId::new("db_1".to_string()),
                        ];
                        
                        let tx_id = coordinator.begin_transaction(databases, metadata).await.unwrap();
                        
                        // Add many operations
                        for i in 0..num_ops {
                            let mut data = HashMap::new();
                            data.insert("index".to_string(), serde_json::json!(i));
                            data.insert("data".to_string(), serde_json::json!(format!("data_{}", i)));
                            
                            let op = TransactionOperation {
                                operation_id: format!("op_{}", i),
                                database_id: DatabaseId::new(format!("db_{}", i % 2)),
                                operation_type: OperationType::CreateEntity {
                                    entity_id: format!("entity_{}", i),
                                    entity_data: data,
                                },
                                parameters: HashMap::new(),
                                dependencies: if i > 0 { vec![format!("op_{}", i - 1)] } else { vec![] },
                                status: llmkg::federation::coordinator::OperationStatus::Pending,
                            };
                            
                            coordinator.add_operation(&tx_id, op).await.unwrap();
                        }
                        
                        // Execute transaction
                        let result = coordinator.prepare_transaction(&tx_id).await.unwrap();
                        black_box(result)
                    })
                })
            },
        );
    }
    
    group.finish();
}

fn bench_recovery_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("transaction_recovery", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Create new coordinator each time to simulate recovery
                let registry = create_test_registry();
                let coordinator = FederationCoordinator::new(registry).await.unwrap();
                
                // Create some transactions
                for i in 0..5 {
                    let metadata = TransactionMetadata {
                        initiator: Some("recovery_bench".to_string()),
                        description: Some(format!("Recovery test {}", i)),
                        priority: TransactionPriority::Normal,
                        isolation_level: IsolationLevel::ReadCommitted,
                        consistency_mode: ConsistencyMode::Eventual,
                    };
                    
                    let databases = vec![DatabaseId::new("db_0".to_string())];
                    let _ = coordinator.begin_transaction(databases, metadata).await.unwrap();
                }
                
                // Cleanup expired transactions (simulating recovery)
                let cleaned = coordinator.cleanup_expired_transactions().await.unwrap();
                black_box(cleaned)
            })
        })
    });
}

criterion_group!(
    benches,
    bench_transaction_lifecycle,
    bench_2pc_phases,
    bench_concurrent_transactions,
    bench_operation_throughput,
    bench_recovery_operations
);
criterion_main!(benches);