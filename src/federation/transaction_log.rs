// Distributed transaction log for federation recovery

use crate::error::{GraphError, Result};
use crate::federation::types::DatabaseId;
use crate::federation::coordinator::{TransactionId, TransactionStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, Duration};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

/// Distributed transaction log for recovery and audit
pub struct DistributedTransactionLog {
    log_dir: PathBuf,
    active_logs: Arc<RwLock<HashMap<TransactionId, TransactionLogRecord>>>,
    write_ahead_log: Arc<RwLock<Vec<LogEntry>>>,
    checkpoint_interval: Duration,
}

/// Complete transaction log record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLogRecord {
    pub transaction_id: TransactionId,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub involved_databases: Vec<DatabaseId>,
    pub status: TransactionStatus,
    pub log_entries: Vec<LogEntry>,
    pub decision: Option<TransactionDecision>,
    pub recovery_info: Option<RecoveryInfo>,
}

/// Individual log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub transaction_id: TransactionId,
    pub database_id: DatabaseId,
    pub entry_type: LogEntryType,
    pub details: HashMap<String, serde_json::Value>,
}

/// Types of log entries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogEntryType {
    TransactionBegin,
    TransactionPrepare,
    PrepareVote { vote: bool },
    TransactionCommit,
    TransactionAbort,
    OperationExecute { operation_id: String },
    CheckpointStart,
    CheckpointComplete,
    RecoveryStart,
    RecoveryComplete,
}

/// Transaction decision for 2PC
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionDecision {
    Commit,
    Abort,
    Pending,
}

/// Recovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryInfo {
    pub recovery_time: SystemTime,
    pub recovered_from: RecoverySource,
    pub operations_replayed: usize,
    pub conflicts_resolved: usize,
}

/// Source of recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoverySource {
    LocalLog,
    RemoteCoordinator { coordinator_id: String },
    ConsensusProtocol,
}

impl DistributedTransactionLog {
    pub async fn new(log_dir: PathBuf) -> Result<Self> {
        // Create log directory if it doesn't exist
        fs::create_dir_all(&log_dir).await
            .map_err(|e| GraphError::StorageError(format!("Failed to create log directory: {}", e)))?;
        
        let log = Self {
            log_dir,
            active_logs: Arc::new(RwLock::new(HashMap::new())),
            write_ahead_log: Arc::new(RwLock::new(Vec::new())),
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
        };
        
        // Load existing logs
        log.load_existing_logs().await?;
        
        Ok(log)
    }
    
    /// Record transaction begin
    pub async fn log_begin(&self, transaction_id: TransactionId, databases: Vec<DatabaseId>) -> Result<()> {
        let record = TransactionLogRecord {
            transaction_id: transaction_id.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            involved_databases: databases.clone(),
            status: TransactionStatus::Pending,
            log_entries: vec![],
            decision: Some(TransactionDecision::Pending),
            recovery_info: None,
        };
        
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            transaction_id: transaction_id.clone(),
            database_id: DatabaseId::new("coordinator".to_string()),
            entry_type: LogEntryType::TransactionBegin,
            details: HashMap::new(),
        };
        
        // Add to active logs
        let mut active_logs = self.active_logs.write().await;
        active_logs.insert(transaction_id.clone(), record);
        
        // Add to write-ahead log
        let mut wal = self.write_ahead_log.write().await;
        wal.push(entry.clone());
        
        // Persist to disk
        self.write_log_entry(&entry).await?;
        
        Ok(())
    }
    
    /// Record prepare vote from a database
    pub async fn log_prepare_vote(
        &self,
        transaction_id: &TransactionId,
        database_id: &DatabaseId,
        vote: bool,
    ) -> Result<()> {
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            transaction_id: transaction_id.clone(),
            database_id: database_id.clone(),
            entry_type: LogEntryType::PrepareVote { vote },
            details: HashMap::new(),
        };
        
        // Update active log
        let mut active_logs = self.active_logs.write().await;
        if let Some(record) = active_logs.get_mut(transaction_id) {
            record.log_entries.push(entry.clone());
            if !vote {
                record.decision = Some(TransactionDecision::Abort);
            }
        }
        
        // Add to write-ahead log
        let mut wal = self.write_ahead_log.write().await;
        wal.push(entry.clone());
        
        // Persist to disk
        self.write_log_entry(&entry).await?;
        
        Ok(())
    }
    
    /// Record transaction decision
    pub async fn log_decision(
        &self,
        transaction_id: &TransactionId,
        decision: TransactionDecision,
    ) -> Result<()> {
        let entry_type = match decision {
            TransactionDecision::Commit => LogEntryType::TransactionCommit,
            TransactionDecision::Abort => LogEntryType::TransactionAbort,
            TransactionDecision::Pending => return Ok(()),
        };
        
        let entry = LogEntry {
            timestamp: SystemTime::now(),
            transaction_id: transaction_id.clone(),
            database_id: DatabaseId::new("coordinator".to_string()),
            entry_type,
            details: HashMap::new(),
        };
        
        // Update active log
        let mut active_logs = self.active_logs.write().await;
        if let Some(record) = active_logs.get_mut(transaction_id) {
            record.log_entries.push(entry.clone());
            record.decision = Some(decision);
            record.end_time = Some(SystemTime::now());
            record.status = match decision {
                TransactionDecision::Commit => TransactionStatus::Committed,
                TransactionDecision::Abort => TransactionStatus::Aborted,
                TransactionDecision::Pending => TransactionStatus::Pending,
            };
        }
        
        // Add to write-ahead log
        let mut wal = self.write_ahead_log.write().await;
        wal.push(entry.clone());
        
        // Persist to disk
        self.write_log_entry(&entry).await?;
        
        Ok(())
    }
    
    /// Get transaction decision for recovery
    pub async fn get_transaction_decision(&self, transaction_id: &TransactionId) -> Option<TransactionDecision> {
        let active_logs = self.active_logs.read().await;
        active_logs.get(transaction_id)
            .and_then(|record| record.decision.clone())
    }
    
    /// Recover pending transactions
    pub async fn recover_pending_transactions(&self) -> Result<Vec<TransactionId>> {
        let mut pending = Vec::new();
        let active_logs = self.active_logs.read().await;
        
        for (tx_id, record) in active_logs.iter() {
            if record.decision == Some(TransactionDecision::Pending) {
                pending.push(tx_id.clone());
            }
        }
        
        Ok(pending)
    }
    
    /// Write log entry to disk
    async fn write_log_entry(&self, entry: &LogEntry) -> Result<()> {
        let filename = self.log_dir.join(format!("{}.log", entry.transaction_id.as_str()));
        
        let serialized = serde_json::to_string(entry)
            .map_err(|e| GraphError::SerializationError(format!("Failed to serialize log entry: {}", e)))?;
        
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)
            .await
            .map_err(|e| GraphError::StorageError(format!("Failed to open log file: {}", e)))?;
        
        file.write_all(format!("{}\n", serialized).as_bytes()).await
            .map_err(|e| GraphError::StorageError(format!("Failed to write log entry: {}", e)))?;
        
        file.flush().await
            .map_err(|e| GraphError::StorageError(format!("Failed to flush log file: {}", e)))?;
        
        Ok(())
    }
    
    /// Load existing logs from disk
    async fn load_existing_logs(&self) -> Result<()> {
        let mut entries = fs::read_dir(&self.log_dir).await
            .map_err(|e| GraphError::StorageError(format!("Failed to read log directory: {}", e)))?;
        
        let mut logs: HashMap<TransactionId, Vec<LogEntry>> = HashMap::new();
        
        while let Some(entry) = entries.next_entry().await
            .map_err(|e| GraphError::StorageError(format!("Failed to read directory entry: {}", e)))? {
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("log") {
                let mut file = fs::File::open(&path).await
                    .map_err(|e| GraphError::StorageError(format!("Failed to open log file: {}", e)))?;
                
                let mut contents = String::new();
                file.read_to_string(&mut contents).await
                    .map_err(|e| GraphError::StorageError(format!("Failed to read log file: {}", e)))?;
                
                for line in contents.lines() {
                    if let Ok(entry) = serde_json::from_str::<LogEntry>(line) {
                        logs.entry(entry.transaction_id.clone())
                            .or_insert_with(Vec::new)
                            .push(entry);
                    }
                }
            }
        }
        
        // Reconstruct transaction records
        let mut active_logs = self.active_logs.write().await;
        for (tx_id, entries) in logs {
            let record = self.reconstruct_transaction_record(tx_id, entries)?;
            active_logs.insert(record.transaction_id.clone(), record);
        }
        
        Ok(())
    }
    
    /// Reconstruct transaction record from log entries
    fn reconstruct_transaction_record(
        &self,
        transaction_id: TransactionId,
        entries: Vec<LogEntry>,
    ) -> Result<TransactionLogRecord> {
        let mut record = TransactionLogRecord {
            transaction_id,
            start_time: SystemTime::now(),
            end_time: None,
            involved_databases: Vec::new(),
            status: TransactionStatus::Pending,
            log_entries: entries.clone(),
            decision: Some(TransactionDecision::Pending),
            recovery_info: None,
        };
        
        // Process entries to reconstruct state
        for entry in &entries {
            match &entry.entry_type {
                LogEntryType::TransactionBegin => {
                    record.start_time = entry.timestamp;
                }
                LogEntryType::TransactionCommit => {
                    record.decision = Some(TransactionDecision::Commit);
                    record.status = TransactionStatus::Committed;
                    record.end_time = Some(entry.timestamp);
                }
                LogEntryType::TransactionAbort => {
                    record.decision = Some(TransactionDecision::Abort);
                    record.status = TransactionStatus::Aborted;
                    record.end_time = Some(entry.timestamp);
                }
                LogEntryType::PrepareVote { vote } => {
                    if !vote {
                        record.decision = Some(TransactionDecision::Abort);
                    }
                }
                _ => {}
            }
        }
        
        Ok(record)
    }
    
    /// Create checkpoint
    pub async fn create_checkpoint(&self) -> Result<()> {
        let checkpoint_file = self.log_dir.join("checkpoint.json");
        
        let active_logs = self.active_logs.read().await;
        let checkpoint_data = serde_json::to_string(&*active_logs)
            .map_err(|e| GraphError::SerializationError(format!("Failed to serialize checkpoint: {}", e)))?;
        
        fs::write(&checkpoint_file, checkpoint_data).await
            .map_err(|e| GraphError::StorageError(format!("Failed to write checkpoint: {}", e)))?;
        
        Ok(())
    }
    
    /// Cleanup old logs
    pub async fn cleanup_old_logs(&self, retention_period: Duration) -> Result<usize> {
        let cutoff_time = SystemTime::now() - retention_period;
        let mut cleaned = 0;
        
        let mut active_logs = self.active_logs.write().await;
        active_logs.retain(|_, record| {
            if let Some(end_time) = record.end_time {
                if end_time < cutoff_time {
                    cleaned += 1;
                    return false;
                }
            }
            true
        });
        
        Ok(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_transaction_log_basic() {
        let temp_dir = TempDir::new().unwrap();
        let log = DistributedTransactionLog::new(temp_dir.path().to_path_buf()).await.unwrap();
        
        let tx_id = TransactionId::new();
        let databases = vec![DatabaseId::new("db1".to_string()), DatabaseId::new("db2".to_string())];
        
        // Log transaction begin
        log.log_begin(tx_id.clone(), databases).await.unwrap();
        
        // Log prepare votes
        log.log_prepare_vote(&tx_id, &DatabaseId::new("db1".to_string()), true).await.unwrap();
        log.log_prepare_vote(&tx_id, &DatabaseId::new("db2".to_string()), true).await.unwrap();
        
        // Log commit decision
        log.log_decision(&tx_id, TransactionDecision::Commit).await.unwrap();
        
        // Check decision
        let decision = log.get_transaction_decision(&tx_id).await;
        assert_eq!(decision, Some(TransactionDecision::Commit));
    }
    
    #[tokio::test]
    async fn test_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let log = DistributedTransactionLog::new(temp_dir.path().to_path_buf()).await.unwrap();
        
        let tx_id = TransactionId::new();
        let databases = vec![DatabaseId::new("db1".to_string())];
        
        // Log transaction begin but don't complete
        log.log_begin(tx_id.clone(), databases).await.unwrap();
        
        // Check pending transactions
        let pending = log.recover_pending_transactions().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0], tx_id);
    }
}