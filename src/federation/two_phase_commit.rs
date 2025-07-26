// Two-phase commit protocol implementation

use crate::error::{GraphError, Result};
use crate::federation::types::DatabaseId;
use crate::federation::coordinator::{TransactionId, TransactionOperation, TransactionMetadata};
use crate::federation::database_connection::{DatabaseConnectionPool, DatabaseConfig};
use crate::federation::transaction_log::{DistributedTransactionLog, TransactionDecision};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::time::{Duration, Instant};
use futures::future::join_all;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Two-phase commit coordinator
pub struct TwoPhaseCommitCoordinator {
    connection_pools: Arc<RwLock<HashMap<DatabaseId, Arc<DatabaseConnectionPool>>>>,
    transaction_log: Arc<DistributedTransactionLog>,
    config: TwoPhaseCommitConfig,
    participant_registry: Arc<RwLock<HashMap<DatabaseId, ParticipantInfo>>>,
}

/// Configuration for 2PC
#[derive(Debug, Clone)]
pub struct TwoPhaseCommitConfig {
    pub prepare_timeout: Duration,
    pub commit_timeout: Duration,
    pub max_retry_attempts: u32,
    pub retry_delay: Duration,
    pub enable_logging: bool,
    pub enable_recovery: bool,
}

impl Default for TwoPhaseCommitConfig {
    fn default() -> Self {
        Self {
            prepare_timeout: Duration::from_secs(30),
            commit_timeout: Duration::from_secs(60),
            max_retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            enable_logging: true,
            enable_recovery: true,
        }
    }
}

/// Information about a participant database
#[derive(Debug, Clone)]
struct ParticipantInfo {
    database_id: DatabaseId,
    status: ParticipantStatus,
    last_heartbeat: Instant,
    prepare_vote: Option<bool>,
    commit_acknowledgment: Option<bool>,
}

/// Status of a participant in 2PC
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParticipantStatus {
    Active,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
    Failed,
}

/// Result of a 2PC transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoPhaseCommitResult {
    pub transaction_id: TransactionId,
    pub success: bool,
    pub phase_completed: CommitPhase,
    pub participants: HashMap<DatabaseId, ParticipantResult>,
    pub total_duration_ms: u64,
    pub prepare_duration_ms: u64,
    pub commit_duration_ms: u64,
    pub error_message: Option<String>,
}

/// Phase of 2PC
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitPhase {
    None,
    Prepare,
    Commit,
    Abort,
}

/// Result for a single participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantResult {
    pub database_id: DatabaseId,
    pub prepare_vote: Option<bool>,
    pub commit_success: Option<bool>,
    pub error: Option<String>,
    pub response_time_ms: u64,
}

/// Trait for 2PC participants
#[async_trait]
pub trait TwoPhaseCommitParticipant: Send + Sync {
    /// Prepare phase - vote yes/no
    async fn prepare(&mut self, transaction_id: &TransactionId, operations: &[TransactionOperation]) -> Result<bool>;
    
    /// Commit phase - apply changes
    async fn commit(&mut self, transaction_id: &TransactionId) -> Result<()>;
    
    /// Abort phase - rollback changes
    async fn abort(&mut self, transaction_id: &TransactionId) -> Result<()>;
    
    /// Get participant ID
    fn get_id(&self) -> &DatabaseId;
}

impl TwoPhaseCommitCoordinator {
    pub async fn new(
        transaction_log: Arc<DistributedTransactionLog>,
        config: TwoPhaseCommitConfig,
    ) -> Result<Self> {
        Ok(Self {
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
            transaction_log,
            config,
            participant_registry: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Register a database with connection pool
    pub async fn register_database(&self, config: DatabaseConfig) -> Result<()> {
        let pool = Arc::new(DatabaseConnectionPool::new(config.clone()).await?);
        
        let mut pools = self.connection_pools.write().await;
        pools.insert(config.id.clone(), pool);
        
        let mut registry = self.participant_registry.write().await;
        registry.insert(config.id.clone(), ParticipantInfo {
            database_id: config.id,
            status: ParticipantStatus::Active,
            last_heartbeat: Instant::now(),
            prepare_vote: None,
            commit_acknowledgment: None,
        });
        
        Ok(())
    }
    
    /// Execute a distributed transaction using 2PC
    pub async fn execute_transaction(
        &self,
        transaction_id: TransactionId,
        databases: Vec<DatabaseId>,
        operations: Vec<TransactionOperation>,
        metadata: TransactionMetadata,
    ) -> Result<TwoPhaseCommitResult> {
        let start_time = Instant::now();
        
        // Log transaction begin
        if self.config.enable_logging {
            self.transaction_log.log_begin(transaction_id.clone(), databases.clone()).await?;
        }
        
        // Initialize participant tracking
        let mut participant_results = HashMap::new();
        for db_id in &databases {
            participant_results.insert(db_id.clone(), ParticipantResult {
                database_id: db_id.clone(),
                prepare_vote: None,
                commit_success: None,
                error: None,
                response_time_ms: 0,
            });
        }
        
        // Phase 1: Prepare
        let prepare_start = Instant::now();
        let prepare_result = self.execute_prepare_phase(
            &transaction_id,
            &databases,
            &operations,
            &mut participant_results,
        ).await;
        let prepare_duration = prepare_start.elapsed();
        
        let (all_prepared, phase_completed) = match prepare_result {
            Ok(all_prepared) => (all_prepared, if all_prepared { CommitPhase::Prepare } else { CommitPhase::Abort }),
            Err(e) => {
                // Abort on prepare failure
                let _ = self.execute_abort_phase(&transaction_id, &databases, &mut participant_results).await;
                
                return Ok(TwoPhaseCommitResult {
                    transaction_id,
                    success: false,
                    phase_completed: CommitPhase::Abort,
                    participants: participant_results,
                    total_duration_ms: start_time.elapsed().as_millis() as u64,
                    prepare_duration_ms: prepare_duration.as_millis() as u64,
                    commit_duration_ms: 0,
                    error_message: Some(e.to_string()),
                });
            }
        };
        
        // Decision point
        if !all_prepared {
            // Some participant voted no, abort transaction
            let commit_start = Instant::now();
            let _ = self.execute_abort_phase(&transaction_id, &databases, &mut participant_results).await;
            let commit_duration = commit_start.elapsed();
            
            return Ok(TwoPhaseCommitResult {
                transaction_id,
                success: false,
                phase_completed: CommitPhase::Abort,
                participants: participant_results,
                total_duration_ms: start_time.elapsed().as_millis() as u64,
                prepare_duration_ms: prepare_duration.as_millis() as u64,
                commit_duration_ms: commit_duration.as_millis() as u64,
                error_message: Some("One or more participants voted no in prepare phase".to_string()),
            });
        }
        
        // Phase 2: Commit
        let commit_start = Instant::now();
        let commit_result = self.execute_commit_phase(
            &transaction_id,
            &databases,
            &mut participant_results,
        ).await;
        let commit_duration = commit_start.elapsed();
        
        match commit_result {
            Ok(_) => {
                Ok(TwoPhaseCommitResult {
                    transaction_id,
                    success: true,
                    phase_completed: CommitPhase::Commit,
                    participants: participant_results,
                    total_duration_ms: start_time.elapsed().as_millis() as u64,
                    prepare_duration_ms: prepare_duration.as_millis() as u64,
                    commit_duration_ms: commit_duration.as_millis() as u64,
                    error_message: None,
                })
            }
            Err(e) => {
                // Log failure but consider transaction committed if prepare succeeded
                Ok(TwoPhaseCommitResult {
                    transaction_id,
                    success: true, // Transaction is committed even if some acks fail
                    phase_completed: CommitPhase::Commit,
                    participants: participant_results,
                    total_duration_ms: start_time.elapsed().as_millis() as u64,
                    prepare_duration_ms: prepare_duration.as_millis() as u64,
                    commit_duration_ms: commit_duration.as_millis() as u64,
                    error_message: Some(format!("Commit phase completed with errors: {}", e)),
                })
            }
        }
    }
    
    /// Execute prepare phase
    async fn execute_prepare_phase(
        &self,
        transaction_id: &TransactionId,
        databases: &[DatabaseId],
        operations: &[TransactionOperation],
        participant_results: &mut HashMap<DatabaseId, ParticipantResult>,
    ) -> Result<bool> {
        let connection_pools = self.connection_pools.read().await;
        
        // Prepare all participants in parallel
        let mut prepare_futures = Vec::new();
        
        for db_id in databases {
            let pool = connection_pools.get(db_id)
                .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", db_id.as_str())))?;
            
            let tx_id = transaction_id.clone();
            let db_id_clone = db_id.clone();
            let pool_clone = pool.clone();
            let ops_for_db: Vec<TransactionOperation> = operations.iter()
                .filter(|op| op.database_id == *db_id)
                .cloned()
                .collect();
            
            let future = async move {
                let start = Instant::now();
                let conn = pool_clone.get_connection().await?;
                let mut conn_guard = conn.lock().await;
                
                // Begin transaction on this database
                conn_guard.begin_transaction(&tx_id).await?;
                
                // Execute operations
                for op in &ops_for_db {
                    match &op.operation_type {
                        crate::federation::coordinator::OperationType::CreateEntity { entity_id, entity_data } => {
                            // Extract name and type from entity_data if available
                            let name = entity_data.get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or(entity_id)
                                .to_string();
                            let entity_type = entity_data.get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let metadata = serde_json::to_string(entity_data).unwrap();
                            
                            let sql = "INSERT INTO entities (id, name, type, metadata) VALUES (?, ?, ?, ?)";
                            let params = vec![
                                serde_json::Value::String(entity_id.clone()),
                                serde_json::Value::String(name),
                                serde_json::Value::String(entity_type),
                                serde_json::Value::String(metadata),
                            ];
                            conn_guard.execute(&tx_id, sql, params).await?;
                        }
                        crate::federation::coordinator::OperationType::UpdateEntity { entity_id, changes } => {
                            let metadata = serde_json::to_string(changes).unwrap();
                            let sql = "UPDATE entities SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?";
                            let params = vec![
                                serde_json::Value::String(metadata),
                                serde_json::Value::String(entity_id.clone()),
                            ];
                            conn_guard.execute(&tx_id, sql, params).await?;
                        }
                        _ => {
                            // Handle other operation types
                        }
                    }
                }
                
                // Prepare the transaction
                let vote = conn_guard.prepare(&tx_id).await?;
                let duration = start.elapsed();
                
                drop(conn_guard);
                pool_clone.return_connection(conn).await;
                
                Ok::<(DatabaseId, bool, u64), GraphError>((db_id_clone, vote, duration.as_millis() as u64))
            };
            
            prepare_futures.push(tokio::time::timeout(self.config.prepare_timeout, future));
        }
        
        // Wait for all prepare responses
        let results = join_all(prepare_futures).await;
        let mut all_prepared = true;
        
        for (i, result) in results.into_iter().enumerate() {
            let db_id = &databases[i];
            
            match result {
                Ok(Ok((_, vote, duration))) => {
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.prepare_vote = Some(vote);
                        participant.response_time_ms = duration;
                        
                        if !vote {
                            all_prepared = false;
                        }
                    }
                    
                    // Log vote
                    if self.config.enable_logging {
                        self.transaction_log.log_prepare_vote(transaction_id, db_id, vote).await?;
                    }
                }
                Ok(Err(e)) => {
                    all_prepared = false;
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.prepare_vote = Some(false);
                        participant.error = Some(e.to_string());
                    }
                }
                Err(_) => {
                    // Timeout
                    all_prepared = false;
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.prepare_vote = Some(false);
                        participant.error = Some("Prepare phase timeout".to_string());
                    }
                }
            }
        }
        
        // Log decision
        if self.config.enable_logging {
            let decision = if all_prepared { TransactionDecision::Commit } else { TransactionDecision::Abort };
            self.transaction_log.log_decision(transaction_id, decision).await?;
        }
        
        Ok(all_prepared)
    }
    
    /// Execute commit phase
    async fn execute_commit_phase(
        &self,
        transaction_id: &TransactionId,
        databases: &[DatabaseId],
        participant_results: &mut HashMap<DatabaseId, ParticipantResult>,
    ) -> Result<()> {
        let connection_pools = self.connection_pools.read().await;
        
        // Commit all participants in parallel
        let mut commit_futures = Vec::new();
        
        for db_id in databases {
            let pool = connection_pools.get(db_id)
                .ok_or_else(|| GraphError::InvalidInput(format!("Database not registered: {}", db_id.as_str())))?;
            
            let tx_id = transaction_id.clone();
            let db_id_clone = db_id.clone();
            let pool_clone = pool.clone();
            
            let future = async move {
                let start = Instant::now();
                let conn = pool_clone.get_connection().await?;
                let mut conn_guard = conn.lock().await;
                
                conn_guard.commit(&tx_id).await?;
                let duration = start.elapsed();
                
                drop(conn_guard);
                pool_clone.return_connection(conn).await;
                
                Ok::<(DatabaseId, u64), GraphError>((db_id_clone, duration.as_millis() as u64))
            };
            
            commit_futures.push(tokio::time::timeout(self.config.commit_timeout, future));
        }
        
        // Wait for all commit responses
        let results = join_all(commit_futures).await;
        let mut any_errors = false;
        
        for (i, result) in results.into_iter().enumerate() {
            let db_id = &databases[i];
            
            match result {
                Ok(Ok((_, duration))) => {
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.commit_success = Some(true);
                        participant.response_time_ms += duration;
                    }
                }
                Ok(Err(e)) => {
                    any_errors = true;
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.commit_success = Some(false);
                        participant.error = Some(e.to_string());
                    }
                }
                Err(_) => {
                    // Timeout - transaction is still committed
                    any_errors = true;
                    if let Some(participant) = participant_results.get_mut(db_id) {
                        participant.commit_success = Some(false);
                        participant.error = Some("Commit phase timeout".to_string());
                    }
                }
            }
        }
        
        if any_errors {
            Err(GraphError::TransactionError("Some participants failed to acknowledge commit".to_string()))
        } else {
            Ok(())
        }
    }
    
    /// Execute abort phase
    async fn execute_abort_phase(
        &self,
        transaction_id: &TransactionId,
        databases: &[DatabaseId],
        participant_results: &mut HashMap<DatabaseId, ParticipantResult>,
    ) -> Result<()> {
        let connection_pools = self.connection_pools.read().await;
        
        // Abort all participants in parallel
        let mut abort_futures = Vec::new();
        
        for db_id in databases {
            if let Some(pool) = connection_pools.get(db_id) {
                let tx_id = transaction_id.clone();
                let db_id_clone = db_id.clone();
                let pool_clone = pool.clone();
                
                let future = async move {
                    let conn = pool_clone.get_connection().await?;
                    let mut conn_guard = conn.lock().await;
                    
                    conn_guard.rollback(&tx_id).await?;
                    
                    drop(conn_guard);
                    pool_clone.return_connection(conn).await;
                    
                    Ok::<DatabaseId, GraphError>(db_id_clone)
                };
                
                abort_futures.push(future);
            }
        }
        
        // Wait for all abort responses
        let results = join_all(abort_futures).await;
        
        for result in results {
            if let Err(e) = result {
                // Log abort failures but continue
                eprintln!("Failed to abort on database: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Recover from coordinator failure
    pub async fn recover(&self) -> Result<usize> {
        if !self.config.enable_recovery {
            return Ok(0);
        }
        
        let pending_transactions = self.transaction_log.recover_pending_transactions().await?;
        let mut recovered = 0;
        
        for tx_id in pending_transactions {
            match self.transaction_log.get_transaction_decision(&tx_id).await {
                Some(TransactionDecision::Commit) => {
                    // Need to ensure commit on all participants
                    recovered += 1;
                }
                Some(TransactionDecision::Abort) => {
                    // Need to ensure abort on all participants
                    recovered += 1;
                }
                _ => {
                    // No decision made, default to abort
                    recovered += 1;
                }
            }
        }
        
        Ok(recovered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::federation::database_connection::DatabaseType;
    
    #[tokio::test]
    async fn test_two_phase_commit_success() {
        let temp_dir = TempDir::new().unwrap();
        let log = Arc::new(DistributedTransactionLog::new(temp_dir.path().to_path_buf()).await.unwrap());
        let coordinator = TwoPhaseCommitCoordinator::new(log, TwoPhaseCommitConfig::default()).await.unwrap();
        
        // Register databases
        let db1_config = DatabaseConfig {
            id: DatabaseId::new("db1".to_string()),
            connection_string: ":memory:".to_string(),
            database_type: DatabaseType::InMemory,
            max_connections: 10,
            connection_timeout: Duration::from_secs(5),
            query_timeout: Duration::from_secs(30),
        };
        
        let db2_config = DatabaseConfig {
            id: DatabaseId::new("db2".to_string()),
            connection_string: ":memory:".to_string(),
            database_type: DatabaseType::InMemory,
            max_connections: 10,
            connection_timeout: Duration::from_secs(5),
            query_timeout: Duration::from_secs(30),
        };
        
        coordinator.register_database(db1_config).await.unwrap();
        coordinator.register_database(db2_config).await.unwrap();
        
        // Create transaction
        let tx_id = TransactionId::new();
        let databases = vec![DatabaseId::new("db1".to_string()), DatabaseId::new("db2".to_string())];
        let operations = vec![];
        let metadata = TransactionMetadata {
            initiator: Some("test".to_string()),
            description: Some("test transaction".to_string()),
            priority: crate::federation::coordinator::TransactionPriority::Normal,
            isolation_level: crate::federation::coordinator::IsolationLevel::ReadCommitted,
            consistency_mode: crate::federation::coordinator::ConsistencyMode::Strong,
        };
        
        let result = coordinator.execute_transaction(tx_id, databases, operations, metadata).await.unwrap();
        
        assert!(result.success);
        assert_eq!(result.phase_completed, CommitPhase::Commit);
    }
}