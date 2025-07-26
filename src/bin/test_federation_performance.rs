// Federation Performance Test
// Tests that federation operations complete in <3ms as required

use llmkg::federation::coordinator::{
    FederationCoordinator, TransactionOperation, TransactionMetadata,
    TransactionPriority, IsolationLevel, ConsistencyMode, OperationType, OperationStatus
};
use llmkg::federation::registry::{DatabaseRegistry, DatabaseDescriptor, DatabaseType, DatabaseStatus, DatabaseMetadata};
use llmkg::federation::types::{DatabaseId, DatabaseCapabilities};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, Instant};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏃‍♂️ Federation Performance Test");
    println!("==============================");
    println!("Target: <3ms federation operations");
    
    // Setup
    let registry = Arc::new(DatabaseRegistry::new()?);
    
    // Register databases
    for i in 1..=3 {
        let db_id = DatabaseId::new(format!("perf_db_{}", i));
        let descriptor = DatabaseDescriptor {
            id: db_id.clone(),
            name: format!("Performance DB {}", i),
            description: Some(format!("Performance test database {}", i)),
            connection_string: format!("perf_test_{}.db", i),
            database_type: DatabaseType::SQLite,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("perf_test".to_string()),
                tags: vec!["performance".to_string(), "test".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        registry.register(descriptor).await?;
    }
    
    let coordinator = FederationCoordinator::new(registry).await?;
    
    // Run performance tests
    let mut measurements = Vec::new();
    let num_tests = 10;
    
    println!("\n🔄 Running {} performance tests...", num_tests);
    
    for test_num in 1..=num_tests {
        let start = Instant::now();
        
        // Begin transaction
        let database_ids = vec![
            DatabaseId::new("perf_db_1".to_string()),
            DatabaseId::new("perf_db_2".to_string()),
            DatabaseId::new("perf_db_3".to_string()),
        ];
        
        let metadata = TransactionMetadata {
            initiator: Some("perf_test".to_string()),
            description: Some(format!("Performance test {}", test_num)),
            priority: TransactionPriority::High,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Strong,
        };
        
        let tx_id = coordinator.begin_transaction(database_ids.clone(), metadata).await?;
        
        // Add a simple entity operation
        let mut entity_data = HashMap::new();
        entity_data.insert("name".to_string(), serde_json::Value::String(format!("Test Entity {}", test_num)));
        entity_data.insert("type".to_string(), serde_json::Value::String("test".to_string()));
        entity_data.insert("test_number".to_string(), serde_json::Value::Number(test_num.into()));
        
        let operation = TransactionOperation {
            operation_id: format!("op_test_{}", test_num),
            database_id: DatabaseId::new("perf_db_1".to_string()),
            operation_type: OperationType::CreateEntity {
                entity_id: format!("entity_{}", test_num),
                entity_data,
            },
            parameters: HashMap::new(),
            dependencies: vec![],
            status: OperationStatus::Pending,
        };
        
        coordinator.add_operation(&tx_id, operation).await?;
        
        // Execute 2PC
        let prepare_result = coordinator.prepare_transaction(&tx_id).await?;
        
        if prepare_result {
            let _commit_result = coordinator.commit_transaction(&tx_id).await?;
        } else {
            let _abort_result = coordinator.abort_transaction(&tx_id).await?;
        }
        
        let duration = start.elapsed();
        measurements.push(duration);
        
        print!(".");
        if test_num % 10 == 0 {
            println!();
        }
    }
    
    // Analyze results
    println!("\n\n📊 Performance Analysis:");
    println!("========================");
    
    let total_time: std::time::Duration = measurements.iter().sum();
    let avg_time = total_time / measurements.len() as u32;
    let min_time = measurements.iter().min().unwrap();
    let max_time = measurements.iter().max().unwrap();
    
    println!("Number of tests: {}", measurements.len());
    println!("Average time: {:.2}ms", avg_time.as_millis());
    println!("Minimum time: {:.2}ms", min_time.as_millis());
    println!("Maximum time: {:.2}ms", max_time.as_millis());
    
    // Count how many are under 3ms
    let under_3ms = measurements.iter().filter(|d| d.as_millis() < 3).count();
    let success_rate = (under_3ms as f64 / measurements.len() as f64) * 100.0;
    
    println!("\nPerformance Target Analysis:");
    println!("< 3ms operations: {}/{} ({:.1}%)", under_3ms, measurements.len(), success_rate);
    
    // Detailed breakdown
    let under_1ms = measurements.iter().filter(|d| d.as_millis() < 1).count();
    let under_2ms = measurements.iter().filter(|d| d.as_millis() < 2).count();
    let under_5ms = measurements.iter().filter(|d| d.as_millis() < 5).count();
    let under_10ms = measurements.iter().filter(|d| d.as_millis() < 10).count();
    
    println!("\nTime Distribution:");
    println!("< 1ms:  {} operations ({:.1}%)", under_1ms, (under_1ms as f64 / measurements.len() as f64) * 100.0);
    println!("< 2ms:  {} operations ({:.1}%)", under_2ms, (under_2ms as f64 / measurements.len() as f64) * 100.0);
    println!("< 3ms:  {} operations ({:.1}%)", under_3ms, (under_3ms as f64 / measurements.len() as f64) * 100.0);
    println!("< 5ms:  {} operations ({:.1}%)", under_5ms, (under_5ms as f64 / measurements.len() as f64) * 100.0);
    println!("< 10ms: {} operations ({:.1}%)", under_10ms, (under_10ms as f64 / measurements.len() as f64) * 100.0);
    
    // Result assessment
    println!("\n🎯 Performance Assessment:");
    if success_rate >= 90.0 {
        println!("✅ EXCELLENT: {:.1}% of operations met the <3ms target", success_rate);
    } else if success_rate >= 75.0 {
        println!("✅ GOOD: {:.1}% of operations met the <3ms target", success_rate);
    } else if success_rate >= 50.0 {
        println!("⚠️ FAIR: {:.1}% of operations met the <3ms target", success_rate);
    } else {
        println!("❌ POOR: Only {:.1}% of operations met the <3ms target", success_rate);
    }
    
    if avg_time.as_millis() < 3 {
        println!("✅ Average execution time ({:.2}ms) meets target", avg_time.as_millis());
    } else {
        println!("❌ Average execution time ({:.2}ms) exceeds target", avg_time.as_millis());
    }
    
    // Check database files were created
    println!("\n🗃️ Verifying Real Database Creation:");
    for i in 1..=3 {
        let db_file = format!("perf_test_{}.db", i);
        if std::path::Path::new(&db_file).exists() {
            let metadata = std::fs::metadata(&db_file)?;
            println!("✅ {} - Size: {} bytes", db_file, metadata.len());
        } else {
            println!("❌ {} - Not found", db_file);
        }
    }
    
    // Check transaction logs
    if std::path::Path::new("federation_logs").exists() {
        let entries = std::fs::read_dir("federation_logs")?;
        let log_count = entries.count();
        println!("✅ Transaction logs: {} files in federation_logs/", log_count);
    } else {
        println!("❌ Transaction log directory not found");
    }
    
    println!("\n🏁 Performance Test Complete!");
    
    Ok(())
}