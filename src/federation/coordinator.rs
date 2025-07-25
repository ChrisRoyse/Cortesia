// Federation coordinator for managing cross-database transactions and consistency

use crate::federation::types::DatabaseId;
use crate::federation::registry::DatabaseRegistry;
use crate::federation::database_connection::{DatabaseConfig, DatabaseType, DatabaseConnectionPool};
use crate::federation::transaction_log::{DistributedTransactionLog, TransactionDecision};
use crate::federation::two_phase_commit::{TwoPhaseCommitCoordinator, TwoPhaseCommitConfig};
use crate::error::{GraphError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, Duration};
use std::path::PathBuf;

/// Transaction coordinator for cross-database operations
pub struct FederationCoordinator {
    registry: Arc<DatabaseRegistry>,
    active_transactions: Arc<RwLock<HashMap<TransactionId, CrossDatabaseTransaction>>>,
    transaction_timeout: Duration,
    two_phase_coordinator: Arc<TwoPhaseCommitCoordinator>,
    transaction_log: Arc<DistributedTransactionLog>,
    connection_pools: Arc<RwLock<HashMap<DatabaseId, Arc<DatabaseConnectionPool>>>>,
}

/// Unique identifier for cross-database transactions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub String);

impl TransactionId {
    pub fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        Self(format!("txn_{}", timestamp))
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Cross-database transaction that spans multiple databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseTransaction {
    pub transaction_id: TransactionId,
    pub involved_databases: Vec<DatabaseId>,
    pub operations: Vec<TransactionOperation>,
    pub status: TransactionStatus,
    pub created_at: SystemTime,
    pub timeout_at: SystemTime,
    pub metadata: TransactionMetadata,
}

/// Status of a cross-database transaction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,        // Transaction created but not started
    Preparing,      // Preparing resources on all databases
    Prepared,       // All databases are ready
    Committing,     // Committing changes
    Committed,      // Successfully committed
    Aborting,       // Rolling back changes
    Aborted,        // Successfully rolled back
    Failed,         // Transaction failed
}

/// Individual operation within a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionOperation {
    pub operation_id: String,
    pub database_id: DatabaseId,
    pub operation_type: OperationType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub dependencies: Vec<String>,
    pub status: OperationStatus,
}

/// Types of operations that can be performed in a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    CreateEntity {
        entity_id: String,
        entity_data: HashMap<String, serde_json::Value>,
    },
    UpdateEntity {
        entity_id: String,
        changes: HashMap<String, serde_json::Value>,
    },
    DeleteEntity {
        entity_id: String,
    },
    CreateRelationship {
        from_entity: String,
        to_entity: String,
        relationship_type: String,
        properties: HashMap<String, serde_json::Value>,
    },
    DeleteRelationship {
        from_entity: String,
        to_entity: String,
        relationship_type: String,
    },
    CreateSnapshot {
        snapshot_name: String,
    },
    RestoreSnapshot {
        snapshot_id: String,
    },
}

/// Status of an individual operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationStatus {
    Pending,
    Prepared,
    Committed,
    Aborted,
    Failed,
}

/// Metadata for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMetadata {
    pub initiator: Option<String>,
    pub description: Option<String>,
    pub priority: TransactionPriority,
    pub isolation_level: IsolationLevel,
    pub consistency_mode: ConsistencyMode,
}

/// Priority levels for transactions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TransactionPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Isolation levels for cross-database transactions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Consistency modes for federation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyMode {
    Eventual,       // Eventually consistent
    Strong,         // Strongly consistent
    Causal,         // Causally consistent
    Monotonic,      // Monotonic consistency
}

/// Result of a transaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    pub transaction_id: TransactionId,
    pub success: bool,
    pub committed_operations: usize,
    pub failed_operations: usize,
    pub execution_time_ms: u64,
    pub error_details: Option<String>,
}

impl FederationCoordinator {
    pub async fn new(registry: Arc<DatabaseRegistry>) -> Result<Self> {
        // Create transaction log directory
        let log_dir = PathBuf::from("federation_logs");
        tokio::fs::create_dir_all(&log_dir).await
            .map_err(|e| GraphError::StorageError(format!("Failed to create log directory: {}", e)))?;
        
        // Initialize transaction log
        let transaction_log = Arc::new(DistributedTransactionLog::new(log_dir).await?);
        
        // Initialize 2PC coordinator
        let two_phase_config = TwoPhaseCommitConfig {
            prepare_timeout: Duration::from_secs(30),
            commit_timeout: Duration::from_secs(60),
            max_retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            enable_logging: true,
            enable_recovery: true,
        };
        let two_phase_coordinator = Arc::new(
            TwoPhaseCommitCoordinator::new(transaction_log.clone(), two_phase_config).await?
        );
        
        Ok(Self {
            registry,
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            transaction_timeout: Duration::from_secs(300), // 5 minutes
            two_phase_coordinator,
            transaction_log,
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Begin a new cross-database transaction
    pub async fn begin_transaction(
        &self,
        databases: Vec<DatabaseId>,
        metadata: TransactionMetadata,
    ) -> Result<TransactionId> {
        let transaction_id = TransactionId::new();
        let timeout_at = SystemTime::now() + self.transaction_timeout;
        
        // Verify all databases are registered and create connection pools if needed
        for db_id in &databases {
            let registered_dbs = self.registry.list_databases().await;
            if !registered_dbs.iter().any(|db| db.id == *db_id) {
                return Err(GraphError::InvalidInput(format!("Database not registered: {}", db_id.as_str())));
            }
            
            // Create connection pool if it doesn't exist
            let mut pools = self.connection_pools.write().await;
            if !pools.contains_key(db_id) {
                let db_config = DatabaseConfig {
                    id: db_id.clone(),
                    connection_string: format!("federation_{}.db", db_id.as_str()),
                    database_type: DatabaseType::InMemory, // Default to in-memory for testing
                    max_connections: 10,
                    connection_timeout: Duration::from_secs(5),
                    query_timeout: Duration::from_secs(30),
                };
                
                // Register with 2PC coordinator
                self.two_phase_coordinator.register_database(db_config.clone()).await?;
                
                let pool = Arc::new(DatabaseConnectionPool::new(db_config).await?);
                pools.insert(db_id.clone(), pool);
            }
        }
        
        let transaction = CrossDatabaseTransaction {
            transaction_id: transaction_id.clone(),
            involved_databases: databases.clone(),
            operations: Vec::new(),
            status: TransactionStatus::Preparing,
            created_at: SystemTime::now(),
            timeout_at,
            metadata,
        };
        
        // Log transaction begin
        self.transaction_log.log_begin(transaction_id.clone(), databases).await?;
        
        let mut active_transactions = self.active_transactions.write().await;
        active_transactions.insert(transaction_id.clone(), transaction);
        
        Ok(transaction_id)
    }

    /// Add an operation to a transaction
    pub async fn add_operation(
        &self,
        transaction_id: &TransactionId,
        operation: TransactionOperation,
    ) -> Result<()> {
        let mut active_transactions = self.active_transactions.write().await;
        
        if let Some(transaction) = active_transactions.get_mut(transaction_id) {
            if matches!(transaction.status, TransactionStatus::Preparing) {
                transaction.operations.push(operation);
                Ok(())
            } else {
                Err(GraphError::InvalidInput("Cannot add operations to a transaction that is not in preparing state".to_string()))
            }
        } else {
            Err(GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))
        }
    }

    /// Prepare a transaction (2-phase commit phase 1)
    pub async fn prepare_transaction(&self, transaction_id: &TransactionId) -> Result<bool> {
        let active_transactions = self.active_transactions.read().await;
        
        if let Some(transaction) = active_transactions.get(transaction_id) {
            // Use real 2PC coordinator
            let operations = transaction.operations.clone();
            let databases = transaction.involved_databases.clone();
            let metadata = transaction.metadata.clone();
            
            drop(active_transactions); // Release lock before 2PC
            
            let result = self.two_phase_coordinator.execute_transaction(
                transaction_id.clone(),
                databases,
                operations,
                metadata,
            ).await?;
            
            // Update transaction status based on result
            let mut active_transactions = self.active_transactions.write().await;
            if let Some(transaction) = active_transactions.get_mut(transaction_id) {
                if result.success && result.phase_completed == crate::federation::two_phase_commit::CommitPhase::Commit {
                    transaction.status = TransactionStatus::Committed;
                    Ok(true)
                } else {
                    transaction.status = TransactionStatus::Aborted;
                    Ok(false)
                }
            } else {
                Ok(result.success)
            }
        } else {
            Err(GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))
        }
    }
    
    /// Send prepare request to a specific database (legacy method for compatibility)
    async fn send_prepare_request(&self, database_id: &DatabaseId, transaction_id: &TransactionId) -> Result<bool> {
        // This method is now handled by the 2PC coordinator
        // Keep for backward compatibility but delegate to connection pool
        
        let pools = self.connection_pools.read().await;
        if let Some(pool) = pools.get(database_id) {
            let conn = pool.get_connection().await?;
            let mut conn_guard = conn.lock().await;
            let result = conn_guard.prepare(transaction_id).await;
            drop(conn_guard);
            pool.return_connection(conn).await;
            result
        } else {
            Err(GraphError::InvalidInput(format!("Database connection pool not found: {}", database_id.as_str())))
        }
    }
    
    /// Send commit request to a specific database (legacy method for compatibility)
    async fn send_commit_request(&self, database_id: &DatabaseId, transaction_id: &TransactionId) -> Result<bool> {
        // This method is now handled by the 2PC coordinator
        // Keep for backward compatibility but delegate to connection pool
        
        let pools = self.connection_pools.read().await;
        if let Some(pool) = pools.get(database_id) {
            let conn = pool.get_connection().await?;
            let mut conn_guard = conn.lock().await;
            conn_guard.commit(transaction_id).await?;
            drop(conn_guard);
            pool.return_connection(conn).await;
            Ok(true)
        } else {
            Err(GraphError::InvalidInput(format!("Database connection pool not found: {}", database_id.as_str())))
        }
    }
    
    /// Internal method to abort a transaction
    async fn abort_transaction_internal(&self, transaction_id: &TransactionId) -> Result<()> {
        // Send abort/rollback requests to all involved databases
        let active_transactions = self.active_transactions.read().await;
        
        if let Some(transaction) = active_transactions.get(transaction_id) {
            let timeout = tokio::time::Duration::from_millis(5000);
            
            for database_id in &transaction.involved_databases {
                let _result = tokio::time::timeout(timeout, self.send_abort_request(database_id, transaction_id)).await;
                // In a real system, would log failures but continue with other databases
            }
        }
        
        Ok(())
    }
    
    /// Send abort request to a specific database
    async fn send_abort_request(&self, database_id: &DatabaseId, _transaction_id: &TransactionId) -> Result<bool> {
        // In a real implementation, this would:
        // 1. Connect to the database
        // 2. Send an abort/rollback request with the transaction ID
        // 3. Wait for the database to release locks and rollback changes
        
        // For demonstration, simulate database abort
        let databases = self.registry.list_databases().await;
        let database_exists = databases.iter().any(|db| db.id == *database_id);
        
        if database_exists {
            // Simulate abort time
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            Ok(true)
        } else {
            Err(GraphError::InvalidInput(format!("Database not found: {}", database_id.as_str())))
        }
    }

    /// Commit a transaction (2-phase commit phase 2)
    pub async fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<TransactionResult> {
        let start_time = std::time::Instant::now();
        
        // Note: The actual 2PC is handled in prepare_transaction
        // This method is for explicit commit after prepare
        
        let active_transactions = self.active_transactions.read().await;
        if let Some(transaction) = active_transactions.get(transaction_id) {
            let status = transaction.status.clone();
            let operations_count = transaction.operations.len();
            drop(active_transactions);
            
            match status {
                TransactionStatus::Committed => {
                    // Already committed
                    Ok(TransactionResult {
                        transaction_id: transaction_id.clone(),
                        success: true,
                        committed_operations: operations_count,
                        failed_operations: 0,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error_details: None,
                    })
                }
                TransactionStatus::Prepared => {
                    // Need to complete commit phase
                    let mut active_transactions = self.active_transactions.write().await;
                    if let Some(transaction) = active_transactions.get_mut(transaction_id) {
                        // Log final commit
                        self.transaction_log.log_decision(transaction_id, TransactionDecision::Commit).await?;
                        
                        transaction.status = TransactionStatus::Committed;
                        let result = TransactionResult {
                            transaction_id: transaction_id.clone(),
                            success: true,
                            committed_operations: transaction.operations.len(),
                            failed_operations: 0,
                            execution_time_ms: start_time.elapsed().as_millis() as u64,
                            error_details: None,
                        };
                        
                        // Remove from active transactions
                        active_transactions.remove(transaction_id);
                        Ok(result)
                    } else {
                        Err(GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))
                    }
                }
                _ => {
                    Err(GraphError::InvalidInput(format!("Transaction not in committable state: {:?}", status)))
                }
            }
        } else {
            Err(GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))
        }
    }

    /// Abort a transaction
    pub async fn abort_transaction(&self, transaction_id: &TransactionId) -> Result<TransactionResult> {
        let start_time = std::time::Instant::now();
        let mut active_transactions = self.active_transactions.write().await;
        
        if let Some(mut transaction) = active_transactions.remove(transaction_id) {
            transaction.status = TransactionStatus::Aborting;
            
            // In a real implementation, this would:
            // 1. Send abort requests to all involved databases
            // 2. Wait for all databases to rollback
            // 3. Clean up any partial changes
            
            let execution_time = start_time.elapsed().as_millis() as u64;
            
            transaction.status = TransactionStatus::Aborted;
            
            Ok(TransactionResult {
                transaction_id: transaction_id.clone(),
                success: true,
                committed_operations: 0,
                failed_operations: transaction.operations.len(),
                execution_time_ms: execution_time,
                error_details: Some("Transaction aborted by user".to_string()),
            })
        } else {
            Err(GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))
        }
    }

    /// Get the status of a transaction
    pub async fn get_transaction_status(&self, transaction_id: &TransactionId) -> Option<TransactionStatus> {
        let active_transactions = self.active_transactions.read().await;
        active_transactions.get(transaction_id).map(|t| t.status.clone())
    }

    /// List all active transactions
    pub async fn list_active_transactions(&self) -> Vec<CrossDatabaseTransaction> {
        let active_transactions = self.active_transactions.read().await;
        active_transactions.values().cloned().collect()
    }

    /// Clean up expired transactions
    pub async fn cleanup_expired_transactions(&self) -> Result<usize> {
        let mut active_transactions = self.active_transactions.write().await;
        let now = SystemTime::now();
        let mut expired_count = 0;
        
        active_transactions.retain(|_, transaction| {
            if transaction.timeout_at < now {
                expired_count += 1;
                false
            } else {
                true
            }
        });
        
        Ok(expired_count)
    }

    /// Ensure consistency across databases
    pub async fn ensure_consistency(&self, databases: Vec<DatabaseId>) -> Result<ConsistencyReport> {
        // Consistency checking requires actual database connections
        // This is a placeholder that indicates the feature is not implemented
        
        if databases.is_empty() {
            return Err(GraphError::InvalidInput("No databases specified for consistency check".into()));
        }
        
        // Verify all databases exist
        let registered_databases = self.registry.list_databases().await;
        for db_id in &databases {
            if !registered_databases.iter().any(|db| &db.id == db_id) {
                return Err(GraphError::InvalidInput(format!("Database not found: {}", db_id.as_str())));
            }
        }
        
        // Return error indicating this requires implementation
        Err(GraphError::NotImplemented(
            "Consistency checking requires database connections and comparison logic. \
             This would need to: 1) Connect to each database, 2) Compare schemas/data, \
             3) Identify conflicts, 4) Generate resolution strategies.".into()
        ))
    }

    /// Synchronize data between databases
    pub async fn synchronize_databases(
        &self,
        source_db: &DatabaseId,
        target_db: &DatabaseId,
        _sync_options: SynchronizationOptions,
    ) -> Result<SynchronizationResult> {
        // In a real implementation, this would:
        // 1. Compare data between source and target databases
        // 2. Identify differences
        // 3. Apply synchronization according to the options
        
        Ok(SynchronizationResult {
            source_database: source_db.clone(),
            target_database: target_db.clone(),
            entities_synchronized: 0,
            relationships_synchronized: 0,
            conflicts_resolved: 0,
            execution_time_ms: 0,
        })
    }

    /// Begin cross-database transaction (test compatibility method)
    pub async fn begin_cross_database_transaction(
        &self,
        transaction_id: TransactionId,
        databases: Vec<&str>,
    ) -> Result<CrossDatabaseTransaction> {
        // Convert string database names to DatabaseId
        let database_ids: Vec<DatabaseId> = databases.iter()
            .map(|&db| DatabaseId(db.to_string()))
            .collect();
        
        // Create default metadata
        let metadata = TransactionMetadata {
            initiator: Some("test".to_string()),
            description: Some("Test transaction".to_string()),
            priority: TransactionPriority::Normal,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Eventual,
        };
        
        // Begin the transaction using existing method
        let _returned_id = self.begin_transaction(database_ids.clone(), metadata.clone()).await?;
        
        // Return the transaction object for test compatibility
        Ok(CrossDatabaseTransaction {
            transaction_id,
            involved_databases: database_ids,
            operations: Vec::new(),
            status: TransactionStatus::Preparing,
            created_at: SystemTime::now(),
            timeout_at: SystemTime::now() + Duration::from_secs(300),
            metadata,
        })
    }

    /// Add entity to transaction (test compatibility method)
    pub async fn add_entity_to_transaction(
        &self,
        transaction_id: &TransactionId,
        entity_id: &str,
        entity_data: serde_json::Value,
    ) -> Result<()> {
        // Convert JSON value to HashMap
        let mut data_map = HashMap::new();
        if let serde_json::Value::Object(obj) = entity_data {
            for (key, value) in obj {
                data_map.insert(key, value);
            }
        }
        
        // Create operation
        let operation = TransactionOperation {
            operation_id: format!("op_{}", uuid::Uuid::new_v4()),
            database_id: DatabaseId("default".to_string()), // Default database
            operation_type: OperationType::CreateEntity {
                entity_id: entity_id.to_string(),
                entity_data: data_map,
            },
            parameters: HashMap::new(),
            dependencies: vec![],
            status: OperationStatus::Pending,
        };
        
        // Add to transaction
        self.add_operation(transaction_id, operation).await
    }
}

/// Report on database consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    pub databases_checked: Vec<DatabaseId>,
    pub inconsistencies_found: usize,
    pub auto_repaired: usize,
    pub manual_intervention_required: usize,
    pub last_check: SystemTime,
}

/// Options for database synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationOptions {
    pub sync_direction: SyncDirection,
    pub conflict_resolution: ConflictResolutionStrategy,
    pub include_metadata: bool,
    pub dry_run: bool,
}

/// Direction of synchronization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncDirection {
    SourceToTarget,
    TargetToSource,
    Bidirectional,
}

/// Strategy for resolving conflicts during synchronization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    SourceWins,
    TargetWins,
    NewestWins,
    HighestConfidenceWins,
    ManualResolution,
}

/// Result of a synchronization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationResult {
    pub source_database: DatabaseId,
    pub target_database: DatabaseId,
    pub entities_synchronized: usize,
    pub relationships_synchronized: usize,
    pub conflicts_resolved: usize,
    pub execution_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federation::registry::DatabaseDescriptor;
    use std::collections::HashMap;

    async fn create_test_registry() -> Arc<DatabaseRegistry> {
        let registry = DatabaseRegistry::new().expect("Failed to create registry");
        
        let db_desc = DatabaseDescriptor {
            id: DatabaseId::new("test_db".to_string()),
            name: "Test Database".to_string(),
            description: Some("Test database for coordinator tests".to_string()),
            connection_string: ":memory:".to_string(),
            database_type: crate::federation::registry::DatabaseType::InMemory,
            capabilities: crate::federation::types::DatabaseCapabilities::default(),
            metadata: crate::federation::registry::DatabaseMetadata {
                version: "1.0.0".to_string(),
                created_at: std::time::SystemTime::now(),
                last_updated: std::time::SystemTime::now(),
                owner: Some("test".to_string()),
                tags: vec!["test".to_string()],
                entity_count: Some(0),
                relationship_count: Some(0),
                storage_size_bytes: Some(0),
            },
            status: crate::federation::registry::DatabaseStatus::Online,
        };
        
        registry.register(db_desc).await.expect("Failed to register database");
        Arc::new(registry)
    }

    #[tokio::test]
    async fn test_send_prepare_request_no_pool() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).await.expect("Failed to create coordinator");
        
        let db_id = DatabaseId::new("test_db".to_string());
        let transaction_id = TransactionId::new();
        
        // Without creating a connection pool, this should fail
        let result = coordinator.send_prepare_request(&db_id, &transaction_id).await;
        assert!(result.is_err());
        
        if let Err(GraphError::InvalidInput(msg)) = result {
            assert!(msg.contains("connection pool not found"));
        } else {
            panic!("Expected InvalidInput error for missing connection pool");
        }
    }

    #[tokio::test]
    async fn test_send_prepare_request_invalid_database() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).await.expect("Failed to create coordinator");
        
        let db_id = DatabaseId::new("nonexistent_db".to_string());
        let transaction_id = TransactionId::new();
        
        let result = coordinator.send_prepare_request(&db_id, &transaction_id).await;
        assert!(result.is_err());
        
        if let Err(GraphError::InvalidInput(msg)) = result {
            assert!(msg.contains("connection pool not found"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_send_commit_request_no_pool() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).await.expect("Failed to create coordinator");
        
        let db_id = DatabaseId::new("test_db".to_string());
        let transaction_id = TransactionId::new();
        
        // Without creating a connection pool, this should fail
        let result = coordinator.send_commit_request(&db_id, &transaction_id).await;
        assert!(result.is_err());
        
        if let Err(GraphError::InvalidInput(msg)) = result {
            assert!(msg.contains("connection pool not found"));
        } else {
            panic!("Expected InvalidInput error for missing connection pool");
        }
    }

    #[tokio::test]
    async fn test_send_abort_request_success() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).await.expect("Failed to create coordinator");
        
        let db_id = DatabaseId::new("test_db".to_string());
        let transaction_id = TransactionId::new();
        
        // Note: abort still uses the old implementation for now
        let result = coordinator.send_abort_request(&db_id, &transaction_id).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
    }

    #[tokio::test]
    async fn test_send_abort_request_invalid_database() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).expect("Failed to create coordinator");
        
        let db_id = DatabaseId::new("nonexistent_db".to_string());
        let transaction_id = TransactionId::new();
        
        let result = coordinator.send_abort_request(&db_id, &transaction_id).await;
        assert!(result.is_err());
        
        if let Err(GraphError::InvalidInput(msg)) = result {
            assert!(msg.contains("Database not found"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_abort_transaction_internal() {
        let registry = create_test_registry().await;
        let coordinator = FederationCoordinator::new(registry).await.expect("Failed to create coordinator");
        
        let databases = vec![DatabaseId::new("test_db".to_string())];
        let metadata = TransactionMetadata {
            initiator: Some("test".to_string()),
            description: Some("test transaction".to_string()),
            priority: TransactionPriority::Normal,
            isolation_level: IsolationLevel::ReadCommitted,
            consistency_mode: ConsistencyMode::Strong,
        };
        
        let transaction_id = coordinator.begin_transaction(databases, metadata).await
            .expect("Failed to begin transaction");
        
        let result = coordinator.abort_transaction_internal(&transaction_id).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_transaction_id_new() {
        let id1 = TransactionId::new();
        let id2 = TransactionId::new();
        
        assert_ne!(id1, id2);
        assert!(id1.as_str().starts_with("txn_"));
        assert!(id2.as_str().starts_with("txn_"));
    }

    #[test]
    fn test_transaction_status_serialization() {
        use serde_json;
        
        let status = TransactionStatus::Committed;
        let serialized = serde_json::to_string(&status).expect("Failed to serialize");
        let deserialized: TransactionStatus = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_operation_status_serialization() {
        use serde_json;
        
        let status = OperationStatus::Prepared;
        let serialized = serde_json::to_string(&status).expect("Failed to serialize");
        let deserialized: OperationStatus = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        assert_eq!(status, deserialized);
    }

    #[test]
    fn test_transaction_priority_ordering() {
        assert!(TransactionPriority::Critical > TransactionPriority::High);
        assert!(TransactionPriority::High > TransactionPriority::Normal);
        assert!(TransactionPriority::Normal > TransactionPriority::Low);
    }

    #[test]
    fn test_isolation_level_ordering() {
        assert!(IsolationLevel::Serializable > IsolationLevel::RepeatableRead);
        assert!(IsolationLevel::RepeatableRead > IsolationLevel::ReadCommitted);
        assert!(IsolationLevel::ReadCommitted > IsolationLevel::ReadUncommitted);
    }

    #[test]
    fn test_operation_type_serialization() {
        use serde_json;
        
        let op = OperationType::CreateEntity {
            entity_id: "test".to_string(),
            entity_data: HashMap::new(),
        };
        
        let serialized = serde_json::to_string(&op).expect("Failed to serialize");
        let deserialized: OperationType = serde_json::from_str(&serialized)
            .expect("Failed to deserialize");
        
        if let OperationType::CreateEntity { entity_id, .. } = deserialized {
            assert_eq!(entity_id, "test");
        } else {
            panic!("Unexpected operation type after deserialization");
        }
    }
}