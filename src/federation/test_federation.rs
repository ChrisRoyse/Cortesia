// Comprehensive tests for real federation coordination with 2-phase commit
use crate::federation::coordinator::*;
use crate::federation::registry::{DatabaseRegistry, DatabaseDescriptor, DatabaseType, DatabaseStatus, DatabaseMetadata};
use crate::federation::types::{DatabaseId, DatabaseCapabilities};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, Duration};

pub async fn test_basic_federation() -> crate::error::Result<()> {
    // Create registry
    let registry = Arc::new(DatabaseRegistry::new()?);
    
    // Register a test database
    let db_id = DatabaseId::new("test_db".to_string());
    let descriptor = DatabaseDescriptor {
        id: db_id.clone(),
        name: "Test Database".to_string(),
        description: Some("Test database for federation".to_string()),
        connection_string: "test_federation.db".to_string(),
        database_type: DatabaseType::SQLite,
        capabilities: DatabaseCapabilities::default(),
        metadata: DatabaseMetadata {
            version: "1.0.0".to_string(),
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            owner: Some("test".to_string()),
            tags: vec!["test".to_string()],
            entity_count: Some(0),
            relationship_count: Some(0),
            storage_size_bytes: Some(0),
        },
        status: DatabaseStatus::Online,
    };
    
    registry.register(descriptor).await?;
    
    // Create coordinator
    let coordinator = FederationCoordinator::new(registry).await?;
    
    // Begin transaction
    let metadata = TransactionMetadata {
        initiator: Some("test".to_string()),
        description: Some("Basic federation test".to_string()),
        priority: TransactionPriority::Normal,
        isolation_level: IsolationLevel::ReadCommitted,
        consistency_mode: ConsistencyMode::Strong,
    };
    
    let tx_id = coordinator.begin_transaction(vec![db_id], metadata).await?;
    
    println!("✓ Federation test transaction created: {}", tx_id.as_str());
    
    // Test prepare
    let prepare_result = coordinator.prepare_transaction(&tx_id).await?;
    
    if prepare_result {
        println!("✓ Transaction prepared successfully");
        
        // Test commit
        let commit_result = coordinator.commit_transaction(&tx_id).await?;
        println!("✓ Transaction committed: success={}", commit_result.success);
        
        return Ok(());
    } else {
        println!("❌ Transaction prepare failed");
        let _abort = coordinator.abort_transaction(&tx_id).await?;
        return Err(crate::error::GraphError::TransactionError("Prepare failed".to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    async fn create_test_registry_with_dbs() -> Arc<DatabaseRegistry> {
        let registry = DatabaseRegistry::new().expect("Failed to create registry");
        
        // Register primary database
        let primary_db = DatabaseDescriptor {
            id: DatabaseId::new("primary".to_string()),
            name: "Primary Database".to_string(),
            description: Some("Primary knowledge storage".to_string()),
            connection_string: "primary.db".to_string(),
            database_type: DatabaseType::InMemory,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("system".to_string()),
                tags: vec!["primary".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        // Register semantic database
        let semantic_db = DatabaseDescriptor {
            id: DatabaseId::new("semantic".to_string()),
            name: "Semantic Database".to_string(),
            description: Some("Relationship and concept storage".to_string()),
            connection_string: "semantic.db".to_string(),
            database_type: DatabaseType::InMemory,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("system".to_string()),
                tags: vec!["semantic".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        // Register temporal database
        let temporal_db = DatabaseDescriptor {
            id: DatabaseId::new("temporal".to_string()),
            name: "Temporal Database".to_string(),
            description: Some("Time-series and versioning storage".to_string()),
            connection_string: "temporal.db".to_string(),
            database_type: DatabaseType::InMemory,
            capabilities: DatabaseCapabilities::default(),
            metadata: DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                owner: Some("system".to_string()),
                tags: vec!["temporal".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: DatabaseStatus::Online,
        };
        
        registry.register(primary_db).await.expect("Failed to register primary DB");
        registry.register(semantic_db).await.expect("Failed to register semantic DB");
        registry.register(temporal_db).await.expect("Failed to register temporal DB");
        
        Arc::new(registry)
    }
    
    #[tokio::test]
    async fn test_federation_basic() {
        let result = test_basic_federation().await;
        assert!(result.is_ok(), "Federation test failed: {:?}", result);
    }

    #[tokio::test]
    async fn test_real_2pc_success_scenario() {
        let registry = create_test_registry_with_dbs().await;
        let coordinator = FederationCoordinator::new(registry.clone()).await.expect("Failed to create coordinator");
        
        let databases = vec![
            DatabaseId::new("primary".to_string()),
            DatabaseId::new("semantic".to_string()),
        ];
        
        let metadata = TransactionMetadata {
            initiator: Some("test_user".to_string()),
            description: Some("Real 2PC success test".to_string()),
            priority: TransactionPriority::Normal,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Strong,
        };
        
        // Begin transaction
        let tx_id = coordinator.begin_transaction(databases.clone(), metadata).await
            .expect("Failed to begin transaction");
        
        // Add operations
        let operation = TransactionOperation {
            operation_id: format!("op_{}", uuid::Uuid::new_v4()),
            database_id: DatabaseId::new("primary".to_string()),
            operation_type: OperationType::CreateEntity {
                entity_id: "test_entity_1".to_string(),
                entity_data: {
                    let mut data = HashMap::new();
                    data.insert("name".to_string(), serde_json::Value::String("Test Entity".to_string()));
                    data.insert("type".to_string(), serde_json::Value::String("TestType".to_string()));
                    data
                },
            },
            parameters: HashMap::new(),
            dependencies: vec![],
            status: OperationStatus::Pending,
        };
        
        coordinator.add_operation(&tx_id, operation).await
            .expect("Failed to add operation");
        
        // Prepare transaction (should succeed)
        let prepare_result = coordinator.prepare_transaction(&tx_id).await
            .expect("Failed to prepare transaction");
        assert!(prepare_result, "Prepare phase should succeed");
        
        // Commit transaction
        let commit_result = coordinator.commit_transaction(&tx_id).await
            .expect("Failed to commit transaction");
        assert!(commit_result.success, "Commit should succeed");
        assert_eq!(commit_result.committed_operations, 1);
        assert_eq!(commit_result.failed_operations, 0);
        
        // Verify transaction is no longer active
        assert!(coordinator.get_transaction_status(&tx_id).await.is_none());
    }

    #[tokio::test]
    async fn test_cross_database_query_execution() {
        let registry = create_test_registry_with_dbs().await;
        let coordinator = FederationCoordinator::new(registry.clone()).await.expect("Failed to create coordinator");
        
        let databases = vec![
            DatabaseId::new("primary".to_string()),
            DatabaseId::new("semantic".to_string()),
        ];
        
        let query_metadata = QueryMetadata {
            initiator: Some("test_user".to_string()),
            query_type: QueryType::Read,
            priority: QueryPriority::Normal,
            timeout_ms: 30000,
            require_consistency: false,
        };
        
        // Execute cross-database query
        let query_result = coordinator.execute_cross_database_query(
            databases,
            "SELECT * FROM entities LIMIT 10",
            vec![],
            query_metadata,
        ).await.expect("Failed to execute cross-database query");
        
        // Verify query executed on both databases
        assert_eq!(query_result.databases_queried.len(), 2);
        assert!(query_result.results.contains_key(&DatabaseId::new("primary".to_string())));
        assert!(query_result.results.contains_key(&DatabaseId::new("semantic".to_string())));
        
        // All databases should be accessible (even if they return empty results)
        for (_, result) in &query_result.results {
            assert!(result.success, "Query should succeed on accessible database");
        }
    }

    #[tokio::test]
    async fn test_transaction_timeout_and_cleanup() {
        let registry = create_test_registry_with_dbs().await;
        let coordinator = FederationCoordinator::new(registry.clone()).await.expect("Failed to create coordinator");
        
        let databases = vec![DatabaseId::new("primary".to_string())];
        let metadata = TransactionMetadata {
            initiator: Some("test_user".to_string()),
            description: Some("Timeout test transaction".to_string()),
            priority: TransactionPriority::Normal,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Strong,
        };
        
        // Begin transaction
        let tx_id = coordinator.begin_transaction(databases, metadata).await
            .expect("Failed to begin transaction");
        
        // Verify transaction is active
        assert!(coordinator.get_transaction_status(&tx_id).await.is_some());
        
        // Manually expire the transaction by modifying timeout
        {
            let mut active_transactions = coordinator.active_transactions.write().await;
            if let Some(transaction) = active_transactions.get_mut(&tx_id) {
                transaction.timeout_at = SystemTime::now() - Duration::from_secs(1);
            }
        }
        
        // Clean up expired transactions
        let cleaned_count = coordinator.cleanup_expired_transactions().await
            .expect("Failed to cleanup expired transactions");
        assert_eq!(cleaned_count, 1);
        
        // Verify transaction is no longer active
        assert!(coordinator.get_transaction_status(&tx_id).await.is_none());
    }

    #[tokio::test]
    async fn test_coordinator_failure_recovery() {
        let registry = create_test_registry_with_dbs().await;
        let coordinator = FederationCoordinator::new(registry.clone()).await.expect("Failed to create coordinator");
        
        let databases = vec![DatabaseId::new("primary".to_string())];
        let metadata = TransactionMetadata {
            initiator: Some("test_user".to_string()),
            description: Some("Recovery test transaction".to_string()),
            priority: TransactionPriority::Normal,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Strong,
        };
        
        // Begin transaction but don't complete it
        let tx_id = coordinator.begin_transaction(databases, metadata).await
            .expect("Failed to begin transaction");
        
        // Simulate coordinator failure recovery
        let recovery_report = coordinator.recover_from_coordinator_failure().await
            .expect("Failed to perform recovery");
        
        // Recovery should handle pending transactions
        assert!(recovery_report.recovered_transactions >= 0);
        assert!(recovery_report.recovery_time_ms > 0);
    }
}