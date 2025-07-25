// Database connection management for federation

use crate::error::{GraphError, Result};
use crate::federation::types::DatabaseId;
use crate::federation::coordinator::TransactionId;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub id: DatabaseId,
    pub connection_string: String,
    pub database_type: DatabaseType,
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub query_timeout: Duration,
}

/// Supported database types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    SQLite,
    PostgreSQL,
    InMemory,
}

/// Database connection pool
pub struct DatabaseConnectionPool {
    config: DatabaseConfig,
    connections: Arc<RwLock<Vec<Arc<Mutex<Box<dyn DatabaseConnection>>>>>>,
    active_connections: Arc<RwLock<usize>>,
}

/// Trait for database connections
#[async_trait]
pub trait DatabaseConnection: Send + Sync {
    /// Check if connection is alive
    async fn is_alive(&self) -> Result<bool>;
    
    /// Begin a transaction
    async fn begin_transaction(&mut self, transaction_id: &TransactionId) -> Result<()>;
    
    /// Prepare a transaction (2PC phase 1)
    async fn prepare(&mut self, transaction_id: &TransactionId) -> Result<bool>;
    
    /// Commit a transaction (2PC phase 2)
    async fn commit(&mut self, transaction_id: &TransactionId) -> Result<()>;
    
    /// Rollback/abort a transaction
    async fn rollback(&mut self, transaction_id: &TransactionId) -> Result<()>;
    
    /// Execute a SQL statement within a transaction
    async fn execute(&mut self, transaction_id: &TransactionId, sql: &str, params: Vec<serde_json::Value>) -> Result<()>;
    
    /// Query data within a transaction
    async fn query(&mut self, transaction_id: &TransactionId, sql: &str, params: Vec<serde_json::Value>) -> Result<Vec<HashMap<String, serde_json::Value>>>;
}

/// SQLite database connection
pub struct SQLiteConnection {
    #[cfg(feature = "native")]
    connection: Option<rusqlite::Connection>,
    database_id: DatabaseId,
    active_transactions: HashMap<TransactionId, TransactionState>,
}

/// PostgreSQL database connection (stub for now)
pub struct PostgreSQLConnection {
    database_id: DatabaseId,
    connection_string: String,
    active_transactions: HashMap<TransactionId, TransactionState>,
}

/// In-memory database connection
pub struct InMemoryConnection {
    database_id: DatabaseId,
    data: HashMap<String, Vec<HashMap<String, serde_json::Value>>>,
    active_transactions: HashMap<TransactionId, TransactionState>,
    transaction_log: Vec<TransactionLogEntry>,
}

/// Transaction state for a connection
#[derive(Debug, Clone)]
struct TransactionState {
    transaction_id: TransactionId,
    status: TransactionPhase,
    operations: Vec<Operation>,
    savepoint_name: Option<String>,
}

/// Transaction phases in 2PC
#[derive(Debug, Clone, PartialEq, Eq)]
enum TransactionPhase {
    Active,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
}

/// Operation within a transaction
#[derive(Debug, Clone)]
struct Operation {
    sql: String,
    params: Vec<serde_json::Value>,
    operation_type: OperationType,
}

#[derive(Debug, Clone)]
enum OperationType {
    Query,
    Insert,
    Update,
    Delete,
    CreateTable,
    DropTable,
}

/// Transaction log entry for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransactionLogEntry {
    transaction_id: TransactionId,
    timestamp: std::time::SystemTime,
    phase: String,
    database_id: DatabaseId,
    success: bool,
    error_message: Option<String>,
}

impl DatabaseConnectionPool {
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        let pool = Self {
            config: config.clone(),
            connections: Arc::new(RwLock::new(Vec::new())),
            active_connections: Arc::new(RwLock::new(0)),
        };
        
        // Pre-create minimum connections
        let min_connections = config.max_connections / 2;
        for _ in 0..min_connections {
            pool.create_connection().await?;
        }
        
        Ok(pool)
    }
    
    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<Arc<Mutex<Box<dyn DatabaseConnection>>>> {
        let mut connections = self.connections.write().await;
        
        // Try to find an available connection
        if let Some(conn) = connections.pop() {
            *self.active_connections.write().await += 1;
            return Ok(conn);
        }
        
        // Create new connection if under limit
        let active = *self.active_connections.read().await;
        if active < self.config.max_connections {
            drop(connections); // Release lock before creating connection
            let conn = self.create_connection().await?;
            *self.active_connections.write().await += 1;
            return Ok(conn);
        }
        
        Err(GraphError::ResourceExhausted {
            resource: format!("Database connections for {}", self.config.id.as_str()),
        })
    }
    
    /// Return a connection to the pool
    pub async fn return_connection(&self, connection: Arc<Mutex<Box<dyn DatabaseConnection>>>) {
        let mut connections = self.connections.write().await;
        connections.push(connection);
        *self.active_connections.write().await -= 1;
    }
    
    /// Create a new connection
    async fn create_connection(&self) -> Result<Arc<Mutex<Box<dyn DatabaseConnection>>>> {
        let connection: Box<dyn DatabaseConnection> = match self.config.database_type {
            DatabaseType::SQLite => {
                Box::new(SQLiteConnection::new(
                    self.config.id.clone(),
                    &self.config.connection_string,
                )?)
            }
            DatabaseType::PostgreSQL => {
                Box::new(PostgreSQLConnection::new(
                    self.config.id.clone(),
                    &self.config.connection_string,
                )?)
            }
            DatabaseType::InMemory => {
                Box::new(InMemoryConnection::new(self.config.id.clone()))
            }
        };
        
        Ok(Arc::new(Mutex::new(connection)))
    }
}

impl SQLiteConnection {
    pub fn new(database_id: DatabaseId, connection_string: &str) -> Result<Self> {
        #[cfg(feature = "native")]
        {
            let conn = rusqlite::Connection::open(connection_string)
                .map_err(|e| GraphError::DatabaseConnectionError(format!("Failed to open SQLite database: {}", e)))?;
            
            // Enable foreign keys and WAL mode for better concurrency
            conn.execute("PRAGMA foreign_keys = ON", [])
                .map_err(|e| GraphError::DatabaseConnectionError(format!("Failed to enable foreign keys: {}", e)))?;
            conn.execute("PRAGMA journal_mode = WAL", [])
                .map_err(|e| GraphError::DatabaseConnectionError(format!("Failed to set WAL mode: {}", e)))?;
            
            Ok(Self {
                connection: Some(conn),
                database_id,
                active_transactions: HashMap::new(),
            })
        }
        
        #[cfg(not(feature = "native"))]
        {
            Ok(Self {
                connection: None,
                database_id,
                active_transactions: HashMap::new(),
            })
        }
    }
}

#[async_trait]
impl DatabaseConnection for SQLiteConnection {
    async fn is_alive(&self) -> Result<bool> {
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                conn.execute("SELECT 1", [])
                    .map(|_| true)
                    .map_err(|e| GraphError::DatabaseConnectionError(format!("Connection check failed: {}", e)))
            } else {
                Ok(false)
            }
        }
        
        #[cfg(not(feature = "native"))]
        Ok(false)
    }
    
    async fn begin_transaction(&mut self, transaction_id: &TransactionId) -> Result<()> {
        if self.active_transactions.contains_key(transaction_id) {
            return Err(GraphError::TransactionError(
                format!("Transaction {} already exists", transaction_id.as_str())
            ));
        }
        
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                let savepoint_name = format!("sp_{}", transaction_id.as_str());
                conn.execute(&format!("SAVEPOINT {}", savepoint_name), [])
                    .map_err(|e| GraphError::TransactionError(format!("Failed to begin transaction: {}", e)))?;
                
                self.active_transactions.insert(
                    transaction_id.clone(),
                    TransactionState {
                        transaction_id: transaction_id.clone(),
                        status: TransactionPhase::Active,
                        operations: Vec::new(),
                        savepoint_name: Some(savepoint_name),
                    },
                );
            }
        }
        
        Ok(())
    }
    
    async fn prepare(&mut self, transaction_id: &TransactionId) -> Result<bool> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        if state.status != TransactionPhase::Active {
            return Err(GraphError::TransactionError(
                format!("Transaction {} is not in active state", transaction_id.as_str())
            ));
        }
        
        state.status = TransactionPhase::Preparing;
        
        // SQLite doesn't have explicit prepare phase, but we can validate operations
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                // Check if all operations are valid
                for op in &state.operations {
                    // Try to prepare statement to validate SQL
                    let _ = conn.prepare(&op.sql)
                        .map_err(|e| GraphError::TransactionError(
                            format!("Invalid SQL in transaction: {}", e)
                        ))?;
                }
            }
        }
        
        state.status = TransactionPhase::Prepared;
        Ok(true)
    }
    
    async fn commit(&mut self, transaction_id: &TransactionId) -> Result<()> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        if state.status != TransactionPhase::Prepared {
            return Err(GraphError::TransactionError(
                format!("Transaction {} is not prepared", transaction_id.as_str())
            ));
        }
        
        state.status = TransactionPhase::Committing;
        
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                if let Some(savepoint_name) = &state.savepoint_name {
                    conn.execute(&format!("RELEASE SAVEPOINT {}", savepoint_name), [])
                        .map_err(|e| GraphError::TransactionError(
                            format!("Failed to commit transaction: {}", e)
                        ))?;
                }
            }
        }
        
        state.status = TransactionPhase::Committed;
        self.active_transactions.remove(transaction_id);
        Ok(())
    }
    
    async fn rollback(&mut self, transaction_id: &TransactionId) -> Result<()> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        state.status = TransactionPhase::Aborting;
        
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                if let Some(savepoint_name) = &state.savepoint_name {
                    conn.execute(&format!("ROLLBACK TO SAVEPOINT {}", savepoint_name), [])
                        .map_err(|e| GraphError::TransactionError(
                            format!("Failed to rollback transaction: {}", e)
                        ))?;
                    conn.execute(&format!("RELEASE SAVEPOINT {}", savepoint_name), [])
                        .map_err(|e| GraphError::TransactionError(
                            format!("Failed to release savepoint: {}", e)
                        ))?;
                }
            }
        }
        
        state.status = TransactionPhase::Aborted;
        self.active_transactions.remove(transaction_id);
        Ok(())
    }
    
    async fn execute(&mut self, transaction_id: &TransactionId, sql: &str, params: Vec<serde_json::Value>) -> Result<()> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        if state.status != TransactionPhase::Active {
            return Err(GraphError::TransactionError(
                format!("Transaction {} is not active", transaction_id.as_str())
            ));
        }
        
        // Determine operation type
        let operation_type = if sql.trim().to_uppercase().starts_with("SELECT") {
            OperationType::Query
        } else if sql.trim().to_uppercase().starts_with("INSERT") {
            OperationType::Insert
        } else if sql.trim().to_uppercase().starts_with("UPDATE") {
            OperationType::Update
        } else if sql.trim().to_uppercase().starts_with("DELETE") {
            OperationType::Delete
        } else if sql.trim().to_uppercase().starts_with("CREATE TABLE") {
            OperationType::CreateTable
        } else if sql.trim().to_uppercase().starts_with("DROP TABLE") {
            OperationType::DropTable
        } else {
            OperationType::Query
        };
        
        state.operations.push(Operation {
            sql: sql.to_string(),
            params: params.clone(),
            operation_type,
        });
        
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                // Convert JSON values to rusqlite params
                let rusqlite_params: Vec<rusqlite::types::Value> = params.into_iter()
                    .map(|v| match v {
                        serde_json::Value::Null => rusqlite::types::Value::Null,
                        serde_json::Value::Bool(b) => rusqlite::types::Value::Integer(if b { 1 } else { 0 }),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                rusqlite::types::Value::Integer(i)
                            } else if let Some(f) = n.as_f64() {
                                rusqlite::types::Value::Real(f)
                            } else {
                                rusqlite::types::Value::Null
                            }
                        }
                        serde_json::Value::String(s) => rusqlite::types::Value::Text(s),
                        _ => rusqlite::types::Value::Text(v.to_string()),
                    })
                    .collect();
                
                conn.execute(sql, rusqlite::params_from_iter(rusqlite_params))
                    .map_err(|e| GraphError::DatabaseConnectionError(
                        format!("Failed to execute SQL: {}", e)
                    ))?;
            }
        }
        
        Ok(())
    }
    
    async fn query(&mut self, transaction_id: &TransactionId, sql: &str, params: Vec<serde_json::Value>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        if !self.active_transactions.contains_key(transaction_id) {
            return Err(GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ));
        }
        
        #[cfg(feature = "native")]
        {
            if let Some(conn) = &self.connection {
                // Convert JSON values to rusqlite params
                let rusqlite_params: Vec<rusqlite::types::Value> = params.into_iter()
                    .map(|v| match v {
                        serde_json::Value::Null => rusqlite::types::Value::Null,
                        serde_json::Value::Bool(b) => rusqlite::types::Value::Integer(if b { 1 } else { 0 }),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                rusqlite::types::Value::Integer(i)
                            } else if let Some(f) = n.as_f64() {
                                rusqlite::types::Value::Real(f)
                            } else {
                                rusqlite::types::Value::Null
                            }
                        }
                        serde_json::Value::String(s) => rusqlite::types::Value::Text(s),
                        _ => rusqlite::types::Value::Text(v.to_string()),
                    })
                    .collect();
                
                let mut stmt = conn.prepare(sql)
                    .map_err(|e| GraphError::DatabaseConnectionError(
                        format!("Failed to prepare query: {}", e)
                    ))?;
                
                let column_names: Vec<String> = stmt.column_names()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect();
                
                let rows = stmt.query_map(rusqlite::params_from_iter(rusqlite_params), |row| {
                    let mut result = HashMap::new();
                    for (i, name) in column_names.iter().enumerate() {
                        let value: rusqlite::types::Value = row.get(i)?;
                        let json_value = match value {
                            rusqlite::types::Value::Null => serde_json::Value::Null,
                            rusqlite::types::Value::Integer(i) => serde_json::Value::Number(i.into()),
                            rusqlite::types::Value::Real(f) => serde_json::Value::Number(
                                serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0))
                            ),
                            rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
                            rusqlite::types::Value::Blob(b) => serde_json::Value::String(
                                base64::encode(b)
                            ),
                        };
                        result.insert(name.clone(), json_value);
                    }
                    Ok(result)
                }).map_err(|e| GraphError::DatabaseConnectionError(
                    format!("Failed to execute query: {}", e)
                ))?;
                
                let mut results = Vec::new();
                for row in rows {
                    results.push(row.map_err(|e| GraphError::DatabaseConnectionError(
                        format!("Failed to fetch row: {}", e)
                    ))?);
                }
                
                return Ok(results);
            }
        }
        
        Ok(Vec::new())
    }
}

impl PostgreSQLConnection {
    pub fn new(database_id: DatabaseId, connection_string: &str) -> Result<Self> {
        // PostgreSQL support would require tokio-postgres
        Ok(Self {
            database_id,
            connection_string: connection_string.to_string(),
            active_transactions: HashMap::new(),
        })
    }
}

#[async_trait]
impl DatabaseConnection for PostgreSQLConnection {
    async fn is_alive(&self) -> Result<bool> {
        // Stub implementation
        Err(GraphError::NotImplemented(
            "PostgreSQL connections require tokio-postgres dependency".into()
        ))
    }
    
    async fn begin_transaction(&mut self, _transaction_id: &TransactionId) -> Result<()> {
        Err(GraphError::NotImplemented(
            "PostgreSQL transactions require tokio-postgres dependency".into()
        ))
    }
    
    async fn prepare(&mut self, _transaction_id: &TransactionId) -> Result<bool> {
        Err(GraphError::NotImplemented(
            "PostgreSQL prepare requires tokio-postgres dependency".into()
        ))
    }
    
    async fn commit(&mut self, _transaction_id: &TransactionId) -> Result<()> {
        Err(GraphError::NotImplemented(
            "PostgreSQL commit requires tokio-postgres dependency".into()
        ))
    }
    
    async fn rollback(&mut self, _transaction_id: &TransactionId) -> Result<()> {
        Err(GraphError::NotImplemented(
            "PostgreSQL rollback requires tokio-postgres dependency".into()
        ))
    }
    
    async fn execute(&mut self, _transaction_id: &TransactionId, _sql: &str, _params: Vec<serde_json::Value>) -> Result<()> {
        Err(GraphError::NotImplemented(
            "PostgreSQL execute requires tokio-postgres dependency".into()
        ))
    }
    
    async fn query(&mut self, _transaction_id: &TransactionId, _sql: &str, _params: Vec<serde_json::Value>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        Err(GraphError::NotImplemented(
            "PostgreSQL query requires tokio-postgres dependency".into()
        ))
    }
}

impl InMemoryConnection {
    pub fn new(database_id: DatabaseId) -> Self {
        Self {
            database_id,
            data: HashMap::new(),
            active_transactions: HashMap::new(),
            transaction_log: Vec::new(),
        }
    }
}

#[async_trait]
impl DatabaseConnection for InMemoryConnection {
    async fn is_alive(&self) -> Result<bool> {
        Ok(true)
    }
    
    async fn begin_transaction(&mut self, transaction_id: &TransactionId) -> Result<()> {
        if self.active_transactions.contains_key(transaction_id) {
            return Err(GraphError::TransactionError(
                format!("Transaction {} already exists", transaction_id.as_str())
            ));
        }
        
        self.active_transactions.insert(
            transaction_id.clone(),
            TransactionState {
                transaction_id: transaction_id.clone(),
                status: TransactionPhase::Active,
                operations: Vec::new(),
                savepoint_name: None,
            },
        );
        
        self.transaction_log.push(TransactionLogEntry {
            transaction_id: transaction_id.clone(),
            timestamp: std::time::SystemTime::now(),
            phase: "BEGIN".to_string(),
            database_id: self.database_id.clone(),
            success: true,
            error_message: None,
        });
        
        Ok(())
    }
    
    async fn prepare(&mut self, transaction_id: &TransactionId) -> Result<bool> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        state.status = TransactionPhase::Prepared;
        
        self.transaction_log.push(TransactionLogEntry {
            transaction_id: transaction_id.clone(),
            timestamp: std::time::SystemTime::now(),
            phase: "PREPARE".to_string(),
            database_id: self.database_id.clone(),
            success: true,
            error_message: None,
        });
        
        Ok(true)
    }
    
    async fn commit(&mut self, transaction_id: &TransactionId) -> Result<()> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        state.status = TransactionPhase::Committed;
        
        self.transaction_log.push(TransactionLogEntry {
            transaction_id: transaction_id.clone(),
            timestamp: std::time::SystemTime::now(),
            phase: "COMMIT".to_string(),
            database_id: self.database_id.clone(),
            success: true,
            error_message: None,
        });
        
        self.active_transactions.remove(transaction_id);
        Ok(())
    }
    
    async fn rollback(&mut self, transaction_id: &TransactionId) -> Result<()> {
        self.active_transactions.remove(transaction_id);
        
        self.transaction_log.push(TransactionLogEntry {
            transaction_id: transaction_id.clone(),
            timestamp: std::time::SystemTime::now(),
            phase: "ROLLBACK".to_string(),
            database_id: self.database_id.clone(),
            success: true,
            error_message: None,
        });
        
        Ok(())
    }
    
    async fn execute(&mut self, transaction_id: &TransactionId, sql: &str, params: Vec<serde_json::Value>) -> Result<()> {
        let state = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| GraphError::TransactionError(
                format!("Transaction {} not found", transaction_id.as_str())
            ))?;
        
        state.operations.push(Operation {
            sql: sql.to_string(),
            params,
            operation_type: OperationType::Insert,
        });
        
        Ok(())
    }
    
    async fn query(&mut self, _transaction_id: &TransactionId, _sql: &str, _params: Vec<serde_json::Value>) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        // Simple in-memory query implementation
        Ok(Vec::new())
    }
}

// Add base64 encoding support for blob data
#[cfg(feature = "native")]
mod base64 {
    pub fn encode(data: Vec<u8>) -> String {
        use std::fmt::Write;
        let mut encoded = String::new();
        for byte in data {
            write!(&mut encoded, "{:02x}", byte).unwrap();
        }
        encoded
    }
}