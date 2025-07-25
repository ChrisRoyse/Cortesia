// Example demonstrating real federation coordinator with 2-phase commit

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
use std::time::SystemTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Federation Coordinator with 2-Phase Commit Example ===\n");
    
    // Step 1: Create registry and register databases
    println!("1. Setting up database registry...");
    let registry = Arc::new(DatabaseRegistry::new()?);
    
    // Register primary database
    let primary_db = DatabaseDescriptor {
        id: DatabaseId::new("primary".to_string()),
        name: "Primary Knowledge Store".to_string(),
        description: Some("Main knowledge graph database".to_string()),
        connection_string: "primary.db".to_string(),
        database_type: DatabaseType::SQLite,
        capabilities: DatabaseCapabilities::default(),
        metadata: DatabaseMetadata {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            owner: Some("admin".to_string()),
            tags: vec!["primary".to_string(), "main".to_string()],
            entity_count: Some(0),
            relationship_count: Some(0),
            storage_size_bytes: Some(0),
        },
        status: DatabaseStatus::Online,
    };
    
    // Register cognitive database
    let cognitive_db = DatabaseDescriptor {
        id: DatabaseId::new("cognitive".to_string()),
        name: "Cognitive Patterns Store".to_string(),
        description: Some("Database for cognitive patterns and neural structures".to_string()),
        connection_string: "cognitive.db".to_string(),
        database_type: DatabaseType::SQLite,
        capabilities: DatabaseCapabilities::default(),
        metadata: DatabaseMetadata {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            owner: Some("admin".to_string()),
            tags: vec!["cognitive".to_string(), "patterns".to_string()],
            entity_count: Some(0),
            relationship_count: Some(0),
            storage_size_bytes: Some(0),
        },
        status: DatabaseStatus::Online,
    };
    
    registry.register(primary_db).await?;
    registry.register(cognitive_db).await?;
    println!("✓ Registered 2 databases\n");
    
    // Step 2: Create federation coordinator
    println!("2. Creating federation coordinator with 2PC support...");
    let coordinator = Arc::new(FederationCoordinator::new(registry.clone()).await?);
    println!("✓ Coordinator initialized with transaction log and 2PC protocol\n");
    
    // Step 3: Begin a cross-database transaction
    println!("3. Beginning cross-database transaction...");
    let metadata = TransactionMetadata {
        initiator: Some("example_app".to_string()),
        description: Some("Create related entities across databases".to_string()),
        priority: TransactionPriority::Normal,
        isolation_level: IsolationLevel::ReadCommitted,
        consistency_mode: ConsistencyMode::Strong,
    };
    
    let databases = vec![
        DatabaseId::new("primary".to_string()),
        DatabaseId::new("cognitive".to_string()),
    ];
    
    let tx_id = coordinator.begin_transaction(databases, metadata).await?;
    println!("✓ Transaction started: {}\n", tx_id.as_str());
    
    // Step 4: Add operations to the transaction
    println!("4. Adding operations to transaction...");
    
    // Create a person entity in primary database
    let mut person_data = HashMap::new();
    person_data.insert("name".to_string(), serde_json::json!("Marie Curie"));
    person_data.insert("type".to_string(), serde_json::json!("Person"));
    person_data.insert("profession".to_string(), serde_json::json!("Scientist"));
    person_data.insert("born".to_string(), serde_json::json!(1867));
    
    let person_op = TransactionOperation {
        operation_id: format!("op_{}", uuid::Uuid::new_v4()),
        database_id: DatabaseId::new("primary".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "marie_curie_001".to_string(),
            entity_data: person_data,
        },
        parameters: HashMap::new(),
        dependencies: vec![],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, person_op).await?;
    println!("✓ Added entity creation in primary database");
    
    // Create a cognitive pattern in cognitive database
    let mut pattern_data = HashMap::new();
    pattern_data.insert("pattern_type".to_string(), serde_json::json!("Discovery"));
    pattern_data.insert("subject".to_string(), serde_json::json!("marie_curie_001"));
    pattern_data.insert("discovery".to_string(), serde_json::json!("Radioactivity"));
    pattern_data.insert("year".to_string(), serde_json::json!(1898));
    pattern_data.insert("significance".to_string(), serde_json::json!("Nobel Prize"));
    
    let pattern_op = TransactionOperation {
        operation_id: format!("op_{}", uuid::Uuid::new_v4()),
        database_id: DatabaseId::new("cognitive".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "discovery_pattern_001".to_string(),
            entity_data: pattern_data,
        },
        parameters: HashMap::new(),
        dependencies: vec![person_op.operation_id.clone()],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, pattern_op).await?;
    println!("✓ Added pattern creation in cognitive database");
    
    // Create a relationship between entities
    let mut rel_props = HashMap::new();
    rel_props.insert("confidence".to_string(), serde_json::json!(0.95));
    rel_props.insert("source".to_string(), serde_json::json!("historical_record"));
    
    let relationship_op = TransactionOperation {
        operation_id: format!("op_{}", uuid::Uuid::new_v4()),
        database_id: DatabaseId::new("primary".to_string()),
        operation_type: OperationType::CreateRelationship {
            from_entity: "marie_curie_001".to_string(),
            to_entity: "discovery_pattern_001".to_string(),
            relationship_type: "MADE_DISCOVERY".to_string(),
            properties: rel_props,
        },
        parameters: HashMap::new(),
        dependencies: vec![],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&tx_id, relationship_op).await?;
    println!("✓ Added relationship creation\n");
    
    // Step 5: Execute 2-phase commit
    println!("5. Executing 2-phase commit protocol...");
    println!("   Phase 1: Prepare");
    
    let prepared = coordinator.prepare_transaction(&tx_id).await?;
    
    if prepared {
        println!("   ✓ All databases voted YES - transaction prepared");
        println!("   Phase 2: Commit");
        println!("   ✓ Transaction committed successfully");
        
        // Check transaction status
        if let Some(status) = coordinator.get_transaction_status(&tx_id).await {
            println!("   Final status: {:?}", status);
        }
    } else {
        println!("   ✗ One or more databases voted NO - transaction aborted");
    }
    
    println!("\n6. Transaction complete!");
    
    // Step 6: List active transactions
    println!("\n7. Active transactions:");
    let active = coordinator.list_active_transactions().await;
    if active.is_empty() {
        println!("   No active transactions (all completed)");
    } else {
        for tx in active {
            println!("   - {} ({:?})", tx.transaction_id.as_str(), tx.status);
        }
    }
    
    // Step 7: Demonstrate error handling
    println!("\n8. Demonstrating rollback scenario...");
    
    let rollback_metadata = TransactionMetadata {
        initiator: Some("example_app".to_string()),
        description: Some("Transaction that will be rolled back".to_string()),
        priority: TransactionPriority::Low,
        isolation_level: IsolationLevel::Serializable,
        consistency_mode: ConsistencyMode::Strong,
    };
    
    let rollback_tx_id = coordinator.begin_transaction(
        vec![DatabaseId::new("primary".to_string())],
        rollback_metadata
    ).await?;
    
    // Add an operation
    let mut error_data = HashMap::new();
    error_data.insert("will_fail".to_string(), serde_json::json!(true));
    
    let error_op = TransactionOperation {
        operation_id: format!("op_{}", uuid::Uuid::new_v4()),
        database_id: DatabaseId::new("primary".to_string()),
        operation_type: OperationType::CreateEntity {
            entity_id: "error_entity".to_string(),
            entity_data: error_data,
        },
        parameters: HashMap::new(),
        dependencies: vec![],
        status: llmkg::federation::coordinator::OperationStatus::Pending,
    };
    
    coordinator.add_operation(&rollback_tx_id, error_op).await?;
    
    // Abort the transaction
    let abort_result = coordinator.abort_transaction(&rollback_tx_id).await?;
    println!("✓ Transaction aborted: {} operations rolled back", abort_result.failed_operations);
    
    println!("\n=== Example Complete ===");
    
    // Performance note
    println!("\nPerformance Notes:");
    println!("- 2PC prepare phase timeout: 30 seconds");
    println!("- 2PC commit phase timeout: 60 seconds");
    println!("- Target federation operation time: <3ms");
    println!("- Connection pool size per database: 10");
    println!("- Transaction log retention: Configurable");
    
    Ok(())
}