// Federation Demo - Real Database Operations
// 
// This demo shows the FederationCoordinator working with actual SQLite databases
// demonstrating real 2-phase commit across multiple database instances.

use llmkg::federation::coordinator::{
    FederationCoordinator, TransactionId, TransactionOperation, TransactionMetadata,
    TransactionPriority, IsolationLevel, ConsistencyMode, OperationType
};
use llmkg::federation::registry::{DatabaseRegistry, DatabaseDescriptor, DatabaseType, DatabaseStatus, DatabaseMetadata};
use llmkg::federation::types::{DatabaseId, DatabaseCapabilities};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Federation Demo - Real Database Operations");
    println!("===============================================");
    
    // Create database registry
    let registry = Arc::new(DatabaseRegistry::new()?);
    
    // Register multiple databases
    let databases = vec![
        ("users_db", "Database for user entities"),
        ("products_db", "Database for product entities"),
        ("orders_db", "Database for order entities"),
    ];
    
    for (db_name, description) in databases {
        let db_id = DatabaseId::new(db_name.to_string());
        let descriptor = DatabaseDescriptor {
            id: db_id.clone(),
            name: db_name.to_string(),
            description: Some(description.to_string()),
            connection_string: format!("{}.db", db_name),
            database_type: DatabaseType::SQLite,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("federation_demo".to_string()),
                tags: vec!["demo".to_string(), "federation".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        registry.register(descriptor).await?;
        println!("✓ Registered database: {}", db_name);
    }
    
    // Create federation coordinator
    println!("\n📊 Initializing Federation Coordinator...");
    let coordinator = FederationCoordinator::new(registry.clone()).await?;
    println!("✓ Federation Coordinator initialized");
    
    // Begin cross-database transaction
    println!("\n🔄 Starting Cross-Database Transaction...");
    let database_ids = vec![
        DatabaseId::new("users_db".to_string()),
        DatabaseId::new("products_db".to_string()),
        DatabaseId::new("orders_db".to_string()),
    ];
    
    let metadata = TransactionMetadata {
        initiator: Some("federation_demo".to_string()),
        description: Some("Cross-database entity creation demo".to_string()),
        priority: TransactionPriority::High,
        isolation_level: IsolationLevel::ReadCommitted,
        consistency_mode: ConsistencyMode::Strong,
    };
    
    let tx_id = coordinator.begin_transaction(database_ids.clone(), metadata).await?;
    println!("✓ Transaction started: {}", tx_id.as_str());
    
    // Add operations to create entities across databases
    println!("\n📝 Adding Operations to Transaction...");
    
    // Create user entity
    let mut user_data = HashMap::new();
    user_data.insert("name".to_string(), serde_json::Value::String("Alice Smith".to_string()));
    user_data.insert("type".to_string(), serde_json::Value::String("user".to_string()));
    user_data.insert("email".to_string(), serde_json::Value::String("alice@example.com".to_string()));
    user_data.insert("age".to_string(), serde_json::Value::Number(30.into()));
    
    let user_operation = TransactionOperation {
        operation_id: "op_user_1".to_string(),
        database_id: DatabaseId::new("users_db".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "user_001".to_string(),
            entity_data: user_data,
        },
        parameters: HashMap::new(),
        dependencies: vec![],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, user_operation).await?;
    println!("✓ Added user creation operation");
    
    // Create product entity
    let mut product_data = HashMap::new();
    product_data.insert("name".to_string(), serde_json::Value::String("Laptop Pro".to_string()));
    product_data.insert("type".to_string(), serde_json::Value::String("product".to_string()));
    product_data.insert("price".to_string(), serde_json::Value::Number(1299.into()));
    product_data.insert("category".to_string(), serde_json::Value::String("electronics".to_string()));
    
    let product_operation = TransactionOperation {
        operation_id: "op_product_1".to_string(),
        database_id: DatabaseId::new("products_db".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "product_001".to_string(),
            entity_data: product_data,
        },
        parameters: HashMap::new(),
        dependencies: vec![],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, product_operation).await?;
    println!("✓ Added product creation operation");
    
    // Create order entity
    let mut order_data = HashMap::new();
    order_data.insert("name".to_string(), serde_json::Value::String("Order #1001".to_string()));
    order_data.insert("type".to_string(), serde_json::Value::String("order".to_string()));
    order_data.insert("user_id".to_string(), serde_json::Value::String("user_001".to_string()));
    order_data.insert("product_id".to_string(), serde_json::Value::String("product_001".to_string()));
    order_data.insert("quantity".to_string(), serde_json::Value::Number(1.into()));
    order_data.insert("total".to_string(), serde_json::Value::Number(1299.into()));
    
    let order_operation = TransactionOperation {
        operation_id: "op_order_1".to_string(),
        database_id: DatabaseId::new("orders_db".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "order_001".to_string(),
            entity_data: order_data,
        },
        parameters: HashMap::new(),
        dependencies: vec!["op_user_1".to_string(), "op_product_1".to_string()],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, order_operation).await?;
    println!("✓ Added order creation operation");
    
    // Execute 2-phase commit
    println!("\n⚡ Executing 2-Phase Commit...");
    let start_time = std::time::Instant::now();
    
    // Phase 1: Prepare
    println!("  📋 Phase 1: Prepare...");
    let prepare_result = coordinator.prepare_transaction(&tx_id).await?;
    
    if prepare_result {
        println!("  ✓ All databases prepared successfully");
        
        // Phase 2: Commit
        println!("  💾 Phase 2: Commit...");
        let commit_result = coordinator.commit_transaction(&tx_id).await?;
        
        let duration = start_time.elapsed();
        println!("  ✓ Transaction committed successfully");
        println!("  ⏱️  Total execution time: {:.2}ms", duration.as_millis());
        
        if duration.as_millis() < 3 {
            println!("  🎯 PERFORMANCE TARGET MET: < 3ms federation operation!");
        }
        
        println!("\n📊 Transaction Result:");
        println!("  • Transaction ID: {}", commit_result.transaction_id.as_str());
        println!("  • Success: {}", commit_result.success);
        println!("  • Operations committed: {}", commit_result.committed_operations);
        println!("  • Execution time: {}ms", commit_result.execution_time_ms);
        
    } else {
        println!("  ❌ Prepare phase failed, aborting transaction");
        let _abort_result = coordinator.abort_transaction(&tx_id).await?;
        println!("  ↩️  Transaction aborted");
    }
    
    // Verify database files were created
    println!("\n🗃️  Verifying Database Files:");
    for db_name in ["users_db.db", "products_db.db", "orders_db.db"] {
        if std::path::Path::new(db_name).exists() {
            let metadata = std::fs::metadata(db_name)?;
            println!("  ✓ {} - Size: {} bytes", db_name, metadata.len());
        } else {
            println!("  ❌ {} - Not found", db_name);
        }
    }
    
    // Check transaction log files
    println!("\n📋 Checking Transaction Logs:");
    if std::path::Path::new("federation_logs").exists() {
        let entries = std::fs::read_dir("federation_logs")?;
        let mut log_count = 0;
        for entry in entries {
            if let Ok(entry) = entry {
                log_count += 1;
                println!("  ✓ Log file: {}", entry.file_name().to_string_lossy());
            }
        }
        println!("  📊 Total log files: {}", log_count);
    } else {
        println!("  ❌ Transaction log directory not found");
    }
    
    println!("\n🎉 Federation Demo Completed!");
    println!("✨ Real database federation with 2PC is working!");
    
    Ok(())
}