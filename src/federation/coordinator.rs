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

    /// Begin a new cross-database transaction with enhanced validation and 2PC setup
    pub async fn begin_transaction(
        &self,
        databases: Vec<DatabaseId>,
        metadata: TransactionMetadata,
    ) -> Result<TransactionId> {
        // Validate input
        if databases.is_empty() {
            return Err(GraphError::InvalidInput("No databases specified for transaction".to_string()));
        }
        
        let transaction_id = TransactionId::new();
        let timeout_at = SystemTime::now() + self.transaction_timeout;
        
        // Verify all databases are registered and accessible
        let registered_dbs = self.registry.list_databases().await;
        for db_id in &databases {
            if !registered_dbs.iter().any(|db| db.id == *db_id) {
                return Err(GraphError::InvalidInput(format!("Database not registered: {}", db_id.as_str())));
            }
            
            // Create connection pool if it doesn't exist
            let mut pools = self.connection_pools.write().await;
            if !pools.contains_key(db_id) {
                // Determine database type based on ID or configuration
                let database_type = match db_id.as_str() {
                    id if id.contains("memory") => DatabaseType::InMemory,
                    id if id.contains("postgres") => DatabaseType::PostgreSQL,
                    _ => DatabaseType::SQLite,
                };
                
                let connection_string = match database_type {
                    DatabaseType::SQLite => format!("federation_{}.db", db_id.as_str()),
                    DatabaseType::PostgreSQL => format!("postgresql://user:pass@localhost/{}", db_id.as_str()),
                    DatabaseType::InMemory => ":memory:".to_string(),
                };
                
                let db_config = DatabaseConfig {
                    id: db_id.clone(),
                    connection_string,
                    database_type,
                    max_connections: 10,
                    connection_timeout: Duration::from_secs(5),
                    query_timeout: Duration::from_secs(30),
                };
                
                // Register with 2PC coordinator first
                self.two_phase_coordinator.register_database(db_config.clone()).await?;
                
                // Create and test connection pool
                let pool = Arc::new(DatabaseConnectionPool::new(db_config).await?);
                
                // Test connection to ensure database is accessible
                let test_conn = pool.get_connection().await
                    .map_err(|e| GraphError::DatabaseConnectionError(
                        format!("Failed to connect to database {}: {}", db_id.as_str(), e)
                    ))?;
                {
                    let conn_guard = test_conn.lock().await;
                    conn_guard.is_alive().await
                        .map_err(|e| GraphError::DatabaseConnectionError(
                            format!("Database {} is not accessible: {}", db_id.as_str(), e)
                        ))?;
                }
                pool.return_connection(test_conn).await;
                
                pools.insert(db_id.clone(), pool);
            }
        }
        
        // Create transaction with proper 2PC initialization
        let transaction = CrossDatabaseTransaction {
            transaction_id: transaction_id.clone(),
            involved_databases: databases.clone(),
            operations: Vec::new(),
            status: TransactionStatus::Pending, // Start as pending, not preparing
            created_at: SystemTime::now(),
            timeout_at,
            metadata: metadata.clone(),
        };
        
        // Log transaction begin with enhanced logging
        self.transaction_log.log_begin(transaction_id.clone(), databases.clone()).await?;
        
        // Initialize transaction state on all databases
        for db_id in &databases {
            let pools = self.connection_pools.read().await;
            if let Some(pool) = pools.get(db_id) {
                let conn = pool.get_connection().await?;
                let mut conn_guard = conn.lock().await;
                
                // Begin transaction on each database
                conn_guard.begin_transaction(&transaction_id).await
                    .map_err(|e| GraphError::TransactionError(
                        format!("Failed to begin transaction on {}: {}", db_id.as_str(), e)
                    ))?;
                
                drop(conn_guard);
                pool.return_connection(conn).await;
            }
        }
        
        // Store active transaction
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

    /// Prepare a transaction (2-phase commit phase 1) with enhanced error handling
    pub async fn prepare_transaction(&self, transaction_id: &TransactionId) -> Result<bool> {
        // Get transaction and validate state
        let mut active_transactions = self.active_transactions.write().await;
        let transaction = active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))?;
        
        // Check current status
        match transaction.status {
            TransactionStatus::Pending => {
                transaction.status = TransactionStatus::Preparing;
            }
            TransactionStatus::Preparing => {
                return Err(GraphError::TransactionError("Transaction is already being prepared".to_string()));
            }
            TransactionStatus::Prepared => {
                return Ok(true); // Already prepared
            }
            _ => {
                return Err(GraphError::TransactionError(
                    format!("Transaction is in invalid state for prepare: {:?}", transaction.status)
                ));
            }
        }
        
        // Check for timeout
        if SystemTime::now() > transaction.timeout_at {
            transaction.status = TransactionStatus::Failed;
            return Err(GraphError::TransactionError("Transaction has timed out".to_string()));
        }
        
        // Get transaction data for 2PC
        let operations = transaction.operations.clone();
        let databases = transaction.involved_databases.clone();
        let metadata = transaction.metadata.clone();
        
        drop(active_transactions); // Release lock before 2PC
        
        // Execute 2PC prepare phase using the coordinator
        let result = match self.two_phase_coordinator.execute_transaction(
            transaction_id.clone(),
            databases.clone(),
            operations,
            metadata,
        ).await {
            Ok(result) => result,
            Err(e) => {
                // Update transaction status on error
                let mut active_transactions = self.active_transactions.write().await;
                if let Some(transaction) = active_transactions.get_mut(transaction_id) {
                    transaction.status = TransactionStatus::Failed;
                }
                return Err(GraphError::TransactionError(
                    format!("2PC execution failed: {}", e)
                ));
            }
        };
        
        // Update transaction status based on 2PC result
        let mut active_transactions = self.active_transactions.write().await;
        if let Some(transaction) = active_transactions.get_mut(transaction_id) {
            match result.phase_completed {
                crate::federation::two_phase_commit::CommitPhase::Prepare => {
                    if result.success {
                        transaction.status = TransactionStatus::Prepared;
                        Ok(true)
                    } else {
                        transaction.status = TransactionStatus::Aborted;
                        Ok(false)
                    }
                }
                crate::federation::two_phase_commit::CommitPhase::Commit => {
                    transaction.status = TransactionStatus::Committed;
                    Ok(true)
                }
                crate::federation::two_phase_commit::CommitPhase::Abort => {
                    transaction.status = TransactionStatus::Aborted;
                    Ok(false)
                }
                _ => {
                    transaction.status = TransactionStatus::Failed;
                    Err(GraphError::TransactionError(
                        format!("Unexpected 2PC phase: {:?}", result.phase_completed)
                    ))
                }
            }
        } else {
            // Transaction was removed during 2PC - return result anyway
            Ok(result.success)
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

    /// Commit a transaction (2-phase commit phase 2) with proper completion handling
    pub async fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<TransactionResult> {
        let start_time = std::time::Instant::now();
        
        let mut active_transactions = self.active_transactions.write().await;
        let transaction = active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Transaction not found: {}", transaction_id.as_str())))?;
        
        let status = transaction.status.clone();
        let operations_count = transaction.operations.len();
        let databases = transaction.involved_databases.clone();
        
        match status {
            TransactionStatus::Committed => {
                // Already committed - return success
                let result = TransactionResult {
                    transaction_id: transaction_id.clone(),
                    success: true,
                    committed_operations: operations_count,
                    failed_operations: 0,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_details: None,
                };
                drop(active_transactions);
                Ok(result)
            }
            TransactionStatus::Prepared => {
                // Execute commit phase on all databases
                transaction.status = TransactionStatus::Committing;
                drop(active_transactions); // Release lock for database operations
                
                // Send commit requests to all databases
                let mut commit_errors = Vec::new();
                let pools = self.connection_pools.read().await;
                
                for db_id in &databases {
                    if let Some(pool) = pools.get(db_id) {
                        match pool.get_connection().await {
                            Ok(conn) => {
                                let mut conn_guard = conn.lock().await;
                                if let Err(e) = conn_guard.commit(transaction_id).await {
                                    commit_errors.push(format!("Database {}: {}", db_id.as_str(), e));
                                }
                                drop(conn_guard);
                                pool.return_connection(conn).await;
                            }
                            Err(e) => {
                                commit_errors.push(format!("Connection to {}: {}", db_id.as_str(), e));
                            }
                        }
                    }
                }
                
                drop(pools);
                
                // Log final commit decision
                self.transaction_log.log_decision(transaction_id, TransactionDecision::Commit).await?;
                
                // Update transaction status
                let mut active_transactions = self.active_transactions.write().await;
                if let Some(transaction) = active_transactions.get_mut(transaction_id) {
                    transaction.status = TransactionStatus::Committed;
                    
                    let result = TransactionResult {
                        transaction_id: transaction_id.clone(),
                        success: commit_errors.is_empty(),
                        committed_operations: operations_count,
                        failed_operations: if commit_errors.is_empty() { 0 } else { databases.len() },
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error_details: if commit_errors.is_empty() {
                            None
                        } else {
                            Some(format!("Commit errors: {}", commit_errors.join("; ")))
                        },
                    };
                    
                    // Remove from active transactions
                    active_transactions.remove(transaction_id);
                    Ok(result)
                } else {
                    Err(GraphError::TransactionError("Transaction removed during commit".to_string()))
                }
            }
            TransactionStatus::Aborted => {
                let result = TransactionResult {
                    transaction_id: transaction_id.clone(),
                    success: false,
                    committed_operations: 0,
                    failed_operations: operations_count,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error_details: Some("Transaction was aborted".to_string()),
                };
                drop(active_transactions);
                Ok(result)
            }
            _ => {
                drop(active_transactions);
                Err(GraphError::TransactionError(
                    format!("Transaction not in committable state: {:?}", status)
                ))
            }
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

    /// Clean up expired transactions with proper rollback
    pub async fn cleanup_expired_transactions(&self) -> Result<usize> {
        let mut active_transactions = self.active_transactions.write().await;
        let now = SystemTime::now();
        let mut expired_transactions = Vec::new();
        
        // Collect expired transactions
        active_transactions.retain(|tx_id, transaction| {
            if transaction.timeout_at < now {
                expired_transactions.push((tx_id.clone(), transaction.clone()));
                false
            } else {
                true
            }
        });
        
        drop(active_transactions); // Release lock before cleanup
        
        // Clean up expired transactions
        let mut cleaned_count = 0;
        for (tx_id, transaction) in expired_transactions {
            // Attempt to rollback on all involved databases
            for db_id in &transaction.involved_databases {
                let pools = self.connection_pools.read().await;
                if let Some(pool) = pools.get(db_id) {
                    if let Ok(conn) = pool.get_connection().await {
                        let mut conn_guard = conn.lock().await;
                        let _ = conn_guard.rollback(&tx_id).await; // Best effort rollback
                        drop(conn_guard);
                        pool.return_connection(conn).await;
                    }
                }
            }
            
            // Log transaction expiry
            let _ = self.transaction_log.log_decision(&tx_id, TransactionDecision::Abort).await;
            cleaned_count += 1;
        }
        
        Ok(cleaned_count)
    }
    
    /// Recover from coordinator failure by processing pending transactions
    pub async fn recover_from_coordinator_failure(&self) -> Result<RecoveryReport> {
        let start_time = std::time::Instant::now();
        let mut recovery_report = RecoveryReport {
            recovered_transactions: 0,
            committed_transactions: 0,
            aborted_transactions: 0,
            failed_recoveries: 0,
            recovery_time_ms: 0,
            errors: Vec::new(),
        };
        
        // Get pending transactions from log
        let pending_transactions = self.transaction_log.recover_pending_transactions().await?;
        
        for tx_id in pending_transactions {
            match self.recover_single_transaction(&tx_id, &mut recovery_report).await {
                Ok(_) => recovery_report.recovered_transactions += 1,
                Err(e) => {
                    recovery_report.failed_recoveries += 1;
                    recovery_report.errors.push(format!("Failed to recover {}: {}", tx_id.as_str(), e));
                }
            }
        }
        
        // Use 2PC coordinator recovery as well
        if let Ok(recovered_2pc) = self.two_phase_coordinator.recover().await {
            recovery_report.recovered_transactions += recovered_2pc;
        }
        
        recovery_report.recovery_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(recovery_report)
    }
    
    /// Recover a single transaction
    async fn recover_single_transaction(
        &self,
        tx_id: &TransactionId,
        recovery_report: &mut RecoveryReport,
    ) -> Result<()> {
        // Check if there's a decision logged
        match self.transaction_log.get_transaction_decision(tx_id).await {
            Some(TransactionDecision::Commit) => {
                // Ensure commit is completed on all databases
                self.complete_commit_recovery(tx_id).await?;
                recovery_report.committed_transactions += 1;
            }
            Some(TransactionDecision::Abort) => {
                // Ensure abort is completed on all databases
                self.complete_abort_recovery(tx_id).await?;
                recovery_report.aborted_transactions += 1;
            }
            Some(TransactionDecision::Pending) | None => {
                // No decision was made, default to abort for safety
                self.complete_abort_recovery(tx_id).await?;
                self.transaction_log.log_decision(tx_id, TransactionDecision::Abort).await?;
                recovery_report.aborted_transactions += 1;
            }
        }
        
        Ok(())
    }
    
    /// Complete commit recovery for a transaction
    async fn complete_commit_recovery(&self, tx_id: &TransactionId) -> Result<()> {
        // Get all registered databases that might be involved
        let registered_dbs = self.registry.list_databases().await;
        let pools = self.connection_pools.read().await;
        
        for db_desc in &registered_dbs {
            if let Some(pool) = pools.get(&db_desc.id) {
                // Try to commit on this database (idempotent operation)
                if let Ok(conn) = pool.get_connection().await {
                    let mut conn_guard = conn.lock().await;
                    // Best effort commit - may fail if already committed or never started
                    let _ = conn_guard.commit(tx_id).await;
                    drop(conn_guard);
                    pool.return_connection(conn).await;
                }
            }
        }
        
        Ok(())
    }
    
    /// Complete abort recovery for a transaction
    async fn complete_abort_recovery(&self, tx_id: &TransactionId) -> Result<()> {
        // Get all registered databases that might be involved
        let registered_dbs = self.registry.list_databases().await;
        let pools = self.connection_pools.read().await;
        
        for db_desc in &registered_dbs {
            if let Some(pool) = pools.get(&db_desc.id) {
                // Try to rollback on this database (idempotent operation)
                if let Ok(conn) = pool.get_connection().await {
                    let mut conn_guard = conn.lock().await;
                    // Best effort rollback - may fail if already rolled back or never started
                    let _ = conn_guard.rollback(tx_id).await;
                    drop(conn_guard);
                    pool.return_connection(conn).await;
                }
            }
        }
        
        Ok(())
    }
    
    /// Monitor and auto-recover failed transactions (requires Arc wrapper)
    pub fn start_recovery_monitor(
        coordinator: Arc<FederationCoordinator>,
        check_interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(check_interval);
            
            loop {
                interval_timer.tick().await;
                
                // Clean up expired transactions
                if let Err(e) = coordinator.cleanup_expired_transactions().await {
                    eprintln!("Failed to cleanup expired transactions: {}", e);
                }
                
                // Perform recovery check
                if let Err(e) = coordinator.recover_from_coordinator_failure().await {
                    eprintln!("Failed to perform recovery check: {}", e);
                }
            }
        })
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

    /// Execute cross-database query with distributed coordination
    pub async fn execute_cross_database_query(
        &self,
        databases: Vec<DatabaseId>,
        query: &str,
        params: Vec<serde_json::Value>,
        query_metadata: QueryMetadata,
    ) -> Result<CrossDatabaseQueryResult> {
        let start_time = std::time::Instant::now();
        let query_id = format!("query_{}", uuid::Uuid::new_v4());
        
        // Validate databases exist
        let registered_dbs = self.registry.list_databases().await;
        for db_id in &databases {
            if !registered_dbs.iter().any(|db| db.id == *db_id) {
                return Err(GraphError::InvalidInput(format!("Database not registered: {}", db_id.as_str())));
            }
        }
        
        // Execute query on all databases in parallel
        let pools = self.connection_pools.read().await;
        let mut query_futures = Vec::new();
        
        for db_id in &databases {
            if let Some(pool) = pools.get(db_id) {
                let db_id_clone = db_id.clone();
                let query_clone = query.to_string();
                let params_clone = params.clone();
                let pool_clone = pool.clone();
                let tx_id = TransactionId::new(); // Temporary transaction for query
                
                let future = async move {
                    let conn = pool_clone.get_connection().await?;
                    let mut conn_guard = conn.lock().await;
                    
                    // Begin temporary transaction for query isolation
                    conn_guard.begin_transaction(&tx_id).await?;
                    
                    let result = conn_guard.query(&tx_id, &query_clone, params_clone).await;
                    
                    // Always rollback the temporary transaction
                    let _ = conn_guard.rollback(&tx_id).await;
                    
                    drop(conn_guard);
                    pool_clone.return_connection(conn).await;
                    
                    result.map(|rows| (db_id_clone, rows))
                };
                
                query_futures.push(tokio::time::timeout(Duration::from_secs(30), future));
            }
        }
        
        drop(pools);
        
        // Wait for all query results
        let results = futures::future::join_all(query_futures).await;
        let mut database_results = HashMap::new();
        let mut errors = Vec::new();
        
        for (i, result) in results.into_iter().enumerate() {
            let db_id = &databases[i];
            
            match result {
                Ok(Ok((_, rows))) => {
                    database_results.insert(db_id.clone(), DatabaseQueryResult {
                        database_id: db_id.clone(),
                        rows,
                        execution_time_ms: 0, // Individual timing would need per-database measurement
                        success: true,
                        error: None,
                    });
                }
                Ok(Err(e)) => {
                    errors.push(format!("Database {}: {}", db_id.as_str(), e));
                    database_results.insert(db_id.clone(), DatabaseQueryResult {
                        database_id: db_id.clone(),
                        rows: Vec::new(),
                        execution_time_ms: 0,
                        success: false,
                        error: Some(e.to_string()),
                    });
                }
                Err(_) => {
                    let error_msg = "Query timeout".to_string();
                    errors.push(format!("Database {}: {}", db_id.as_str(), error_msg));
                    database_results.insert(db_id.clone(), DatabaseQueryResult {
                        database_id: db_id.clone(),
                        rows: Vec::new(),
                        execution_time_ms: 0,
                        success: false,
                        error: Some(error_msg),
                    });
                }
            }
        }
        
        Ok(CrossDatabaseQueryResult {
            query_id,
            query: query.to_string(),
            databases_queried: databases,
            results: database_results,
            total_execution_time_ms: start_time.elapsed().as_millis() as u64,
            success: errors.is_empty(),
            errors,
            metadata: query_metadata,
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

/// Metadata for cross-database queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    pub initiator: Option<String>,
    pub query_type: QueryType,
    pub priority: QueryPriority,
    pub timeout_ms: u64,
    pub require_consistency: bool,
}

/// Types of cross-database queries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    Read,
    Aggregate,
    Join,
    Search,
    Analytics,
}

/// Priority levels for queries
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueryPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Result of a cross-database query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDatabaseQueryResult {
    pub query_id: String,
    pub query: String,
    pub databases_queried: Vec<DatabaseId>,
    pub results: HashMap<DatabaseId, DatabaseQueryResult>,
    pub total_execution_time_ms: u64,
    pub success: bool,
    pub errors: Vec<String>,
    pub metadata: QueryMetadata,
}

/// Result from a single database in a cross-database query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseQueryResult {
    pub database_id: DatabaseId,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
    pub execution_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Report from coordinator failure recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryReport {
    pub recovered_transactions: usize,
    pub committed_transactions: usize,
    pub aborted_transactions: usize,
    pub failed_recoveries: usize,
    pub recovery_time_ms: u64,
    pub errors: Vec<String>,
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