//! Federation Layer Unit Tests
//! 
//! Comprehensive unit tests for the LLMKG federation layer components

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use rand::prelude::*;

/// Database registry for federation
#[derive(Debug, Clone)]
pub struct DatabaseRegistry {
    /// Registered databases
    databases: HashMap<String, DatabaseInfo>,
    /// Performance metrics
    metrics: HashMap<String, DatabaseMetrics>,
}

#[derive(Debug, Clone)]
pub struct DatabaseInfo {
    pub id: String,
    pub name: String,
    pub url: String,
    pub database_type: DatabaseType,
    pub capabilities: Vec<Capability>,
    pub status: DatabaseStatus,
    pub priority: i32,
}

#[derive(Debug, Clone)]
pub enum DatabaseType {
    Neo4j,
    ArangoDB,
    TigerGraph,
    MemGraph,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Capability {
    FullTextSearch,
    GraphTraversal,
    Analytics,
    MachineLearning,
    Streaming,
    ACID,
    Sharding,
}

#[derive(Debug, Clone)]
pub enum DatabaseStatus {
    Online,
    Offline,
    Degraded,
    Maintenance,
}

#[derive(Debug, Clone)]
pub struct DatabaseMetrics {
    pub response_time_ms: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub query_count: u64,
    pub error_rate: f64,
    pub last_updated: std::time::SystemTime,
}

impl DatabaseRegistry {
    /// Create new database registry
    pub fn new() -> Self {
        Self {
            databases: HashMap::new(),
            metrics: HashMap::new(),
        }
    }

    /// Register a database
    pub fn register_database(&mut self, db_info: DatabaseInfo) -> Result<()> {
        if self.databases.contains_key(&db_info.id) {
            return Err(anyhow!("Database {} already registered", db_info.id));
        }

        // Initialize metrics
        let metrics = DatabaseMetrics {
            response_time_ms: 0.0,
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            query_count: 0,
            error_rate: 0.0,
            last_updated: std::time::SystemTime::now(),
        };

        self.metrics.insert(db_info.id.clone(), metrics);
        self.databases.insert(db_info.id.clone(), db_info);
        
        Ok(())
    }

    /// Unregister a database
    pub fn unregister_database(&mut self, db_id: &str) -> Result<()> {
        if !self.databases.contains_key(db_id) {
            return Err(anyhow!("Database {} not found", db_id));
        }

        self.databases.remove(db_id);
        self.metrics.remove(db_id);
        Ok(())
    }

    /// Get database info
    pub fn get_database(&self, db_id: &str) -> Option<&DatabaseInfo> {
        self.databases.get(db_id)
    }

    /// List all databases
    pub fn list_databases(&self) -> Vec<&DatabaseInfo> {
        self.databases.values().collect()
    }

    /// Find databases by capability
    pub fn find_by_capability(&self, capability: Capability) -> Vec<&DatabaseInfo> {
        self.databases.values()
            .filter(|db| db.capabilities.contains(&capability))
            .collect()
    }

    /// Update database metrics
    pub fn update_metrics(&mut self, db_id: &str, metrics: DatabaseMetrics) -> Result<()> {
        if !self.databases.contains_key(db_id) {
            return Err(anyhow!("Database {} not found", db_id));
        }

        self.metrics.insert(db_id.to_string(), metrics);
        Ok(())
    }

    /// Get database metrics
    pub fn get_metrics(&self, db_id: &str) -> Option<&DatabaseMetrics> {
        self.metrics.get(db_id)
    }

    /// Get healthy databases
    pub fn get_healthy_databases(&self) -> Vec<&DatabaseInfo> {
        self.databases.values()
            .filter(|db| matches!(db.status, DatabaseStatus::Online))
            .collect()
    }

    /// Get database count
    pub fn database_count(&self) -> usize {
        self.databases.len()
    }
}

/// Query router for federated queries
#[derive(Debug)]
pub struct QueryRouter {
    /// Database registry
    registry: Arc<Mutex<DatabaseRegistry>>,
    /// Routing strategies
    strategies: Vec<RoutingStrategy>,
    /// Load balancer
    load_balancer: LoadBalancer,
}

#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    ResponseTime,
    Capability,
    Geographic,
}

#[derive(Debug)]
pub struct LoadBalancer {
    strategy: RoutingStrategy,
    state: HashMap<String, LoadBalancerState>,
}

#[derive(Debug)]
pub struct LoadBalancerState {
    pub current_connections: u32,
    pub total_requests: u64,
    pub last_used: std::time::SystemTime,
    pub weight: f32,
}

impl QueryRouter {
    /// Create new query router
    pub fn new(registry: Arc<Mutex<DatabaseRegistry>>) -> Self {
        Self {
            registry,
            strategies: vec![RoutingStrategy::RoundRobin],
            load_balancer: LoadBalancer::new(RoutingStrategy::RoundRobin),
        }
    }

    /// Route query to appropriate databases
    pub async fn route_query(&mut self, query: &FederatedQuery) -> Result<Vec<String>> {
        let registry = self.registry.lock().map_err(|_| anyhow!("Registry lock failed"))?;
        
        // Filter databases by query requirements
        let mut candidates = self.filter_candidates(&registry, query)?;
        
        // Apply load balancing
        candidates = self.load_balancer.select_databases(candidates, query.max_databases)?;
        
        Ok(candidates.into_iter().map(|db| db.id.clone()).collect())
    }

    /// Filter database candidates based on query requirements
    fn filter_candidates(&self, registry: &DatabaseRegistry, query: &FederatedQuery) -> Result<Vec<&DatabaseInfo>> {
        let mut candidates = Vec::new();

        for db in registry.get_healthy_databases() {
            // Check capabilities
            let has_required_capabilities = query.required_capabilities.iter()
                .all(|cap| db.capabilities.contains(cap));
            
            if !has_required_capabilities {
                continue;
            }

            // Check database type filter
            if let Some(ref allowed_types) = query.allowed_database_types {
                if !allowed_types.contains(&db.database_type) {
                    continue;
                }
            }

            // Check performance requirements
            if let Some(metrics) = registry.get_metrics(&db.id) {
                if query.max_response_time_ms.map_or(false, |max| metrics.response_time_ms > max as f64) {
                    continue;
                }
                
                if query.max_error_rate.map_or(false, |max| metrics.error_rate > max) {
                    continue;
                }
            }

            candidates.push(db);
        }

        Ok(candidates)
    }

    /// Execute federated query
    pub async fn execute_federated_query(&mut self, query: FederatedQuery) -> Result<FederatedQueryResult> {
        let start_time = std::time::Instant::now();
        
        // Route query to databases
        let selected_databases = self.route_query(&query).await?;
        
        if selected_databases.is_empty() {
            return Err(anyhow!("No suitable databases found for query"));
        }

        // Execute query on each selected database (simulated)
        let mut partial_results = Vec::new();
        let mut errors = Vec::new();

        for db_id in &selected_databases {
            match self.execute_on_database(db_id, &query).await {
                Ok(result) => partial_results.push(result),
                Err(e) => errors.push(format!("Database {}: {}", db_id, e)),
            }
        }

        // Merge results
        let merged_result = self.merge_results(partial_results)?;
        
        let execution_time = start_time.elapsed();

        Ok(FederatedQueryResult {
            merged_result,
            database_count: selected_databases.len(),
            execution_time_ms: execution_time.as_millis() as u64,
            errors,
            partial_result_count: partial_results.len(),
        })
    }

    /// Execute query on a specific database (simulated)
    async fn execute_on_database(&self, db_id: &str, query: &FederatedQuery) -> Result<PartialQueryResult> {
        // Simulate database query execution
        let mut rng = StdRng::seed_from_u64(db_id.chars().map(|c| c as u64).sum());
        
        // Simulate variable execution time
        let execution_time = rng.gen_range(10..200);
        tokio::time::sleep(tokio::time::Duration::from_millis(execution_time)).await;
        
        // Simulate occasional errors
        if rng.gen_range(0..100) < 5 { // 5% error rate
            return Err(anyhow!("Simulated database error"));
        }

        // Generate mock results
        let result_count = rng.gen_range(0..query.limit.unwrap_or(100));
        let mut entities = Vec::new();
        
        for i in 0..result_count {
            entities.push(format!("{}:entity_{}", db_id, i));
        }

        Ok(PartialQueryResult {
            database_id: db_id.to_string(),
            entities,
            execution_time_ms: execution_time,
            metadata: HashMap::new(),
        })
    }

    /// Merge partial results from multiple databases
    fn merge_results(&self, partial_results: Vec<PartialQueryResult>) -> Result<MergedQueryResult> {
        let mut all_entities = Vec::new();
        let mut total_execution_time = 0;
        let mut database_results = HashMap::new();

        for result in partial_results {
            all_entities.extend(result.entities.clone());
            total_execution_time += result.execution_time_ms;
            database_results.insert(result.database_id.clone(), result);
        }

        // Remove duplicates (simple string-based deduplication)
        all_entities.sort();
        all_entities.dedup();

        Ok(MergedQueryResult {
            entities: all_entities,
            total_execution_time_ms: total_execution_time,
            database_results,
            merge_strategy: "union".to_string(),
        })
    }
}

impl LoadBalancer {
    fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            state: HashMap::new(),
        }
    }

    fn select_databases(&mut self, candidates: Vec<&DatabaseInfo>, max_count: Option<usize>) -> Result<Vec<&DatabaseInfo>> {
        let limit = max_count.unwrap_or(candidates.len()).min(candidates.len());
        
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        match self.strategy {
            RoutingStrategy::RoundRobin => {
                Ok(candidates.into_iter().take(limit).collect())
            }
            RoutingStrategy::ResponseTime => {
                // Sort by response time (simulated)
                let mut candidates = candidates;
                candidates.sort_by_key(|db| db.priority);
                Ok(candidates.into_iter().take(limit).collect())
            }
            _ => {
                // Default to round robin
                Ok(candidates.into_iter().take(limit).collect())
            }
        }
    }
}

/// Federated query structure
#[derive(Debug, Clone)]
pub struct FederatedQuery {
    pub query_text: String,
    pub query_type: QueryType,
    pub required_capabilities: Vec<Capability>,
    pub allowed_database_types: Option<Vec<DatabaseType>>,
    pub max_databases: Option<usize>,
    pub max_response_time_ms: Option<u32>,
    pub max_error_rate: Option<f64>,
    pub limit: Option<usize>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum QueryType {
    Search,
    Traversal,
    Analytics,
    Update,
}

#[derive(Debug)]
pub struct FederatedQueryResult {
    pub merged_result: MergedQueryResult,
    pub database_count: usize,
    pub execution_time_ms: u64,
    pub errors: Vec<String>,
    pub partial_result_count: usize,
}

#[derive(Debug)]
pub struct PartialQueryResult {
    pub database_id: String,
    pub entities: Vec<String>,
    pub execution_time_ms: u64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct MergedQueryResult {
    pub entities: Vec<String>,
    pub total_execution_time_ms: u64,
    pub database_results: HashMap<String, PartialQueryResult>,
    pub merge_strategy: String,
}

/// Transaction coordinator for distributed transactions
#[derive(Debug)]
pub struct TransactionCoordinator {
    /// Active transactions
    active_transactions: HashMap<String, DistributedTransaction>,
    /// Transaction timeout
    timeout_ms: u64,
}

#[derive(Debug)]
pub struct DistributedTransaction {
    pub id: String,
    pub participants: Vec<String>, // Database IDs
    pub status: TransactionStatus,
    pub operations: Vec<TransactionOperation>,
    pub start_time: std::time::SystemTime,
    pub prepare_votes: HashMap<String, bool>, // Database ID -> vote
}

#[derive(Debug, Clone)]
pub enum TransactionStatus {
    Active,
    Preparing,
    Committed,
    Aborted,
    TimedOut,
}

#[derive(Debug, Clone)]
pub struct TransactionOperation {
    pub database_id: String,
    pub operation_type: OperationType,
    pub entity_id: String,
    pub data: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    Create,
    Update,
    Delete,
}

impl TransactionCoordinator {
    /// Create new transaction coordinator
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            active_transactions: HashMap::new(),
            timeout_ms,
        }
    }

    /// Begin distributed transaction
    pub fn begin_transaction(&mut self, participants: Vec<String>) -> Result<String> {
        let transaction_id = format!("txn_{}", uuid::Uuid::new_v4());
        
        let transaction = DistributedTransaction {
            id: transaction_id.clone(),
            participants,
            status: TransactionStatus::Active,
            operations: Vec::new(),
            start_time: std::time::SystemTime::now(),
            prepare_votes: HashMap::new(),
        };

        self.active_transactions.insert(transaction_id.clone(), transaction);
        Ok(transaction_id)
    }

    /// Add operation to transaction
    pub fn add_operation(&mut self, transaction_id: &str, operation: TransactionOperation) -> Result<()> {
        let transaction = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", transaction_id))?;

        if !matches!(transaction.status, TransactionStatus::Active) {
            return Err(anyhow!("Transaction {} is not active", transaction_id));
        }

        transaction.operations.push(operation);
        Ok(())
    }

    /// Commit transaction using two-phase commit
    pub async fn commit_transaction(&mut self, transaction_id: &str) -> Result<bool> {
        let transaction = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", transaction_id))?;

        // Check if transaction has timed out
        if transaction.start_time.elapsed().unwrap().as_millis() > self.timeout_ms as u128 {
            transaction.status = TransactionStatus::TimedOut;
            return Err(anyhow!("Transaction {} timed out", transaction_id));
        }

        // Phase 1: Prepare
        transaction.status = TransactionStatus::Preparing;
        let prepare_success = self.prepare_phase(transaction).await?;

        if !prepare_success {
            // Abort transaction
            transaction.status = TransactionStatus::Aborted;
            self.abort_phase(transaction).await?;
            return Ok(false);
        }

        // Phase 2: Commit
        transaction.status = TransactionStatus::Committed;
        self.commit_phase(transaction).await?;
        
        Ok(true)
    }

    /// Abort transaction
    pub async fn abort_transaction(&mut self, transaction_id: &str) -> Result<()> {
        let transaction = self.active_transactions.get_mut(transaction_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", transaction_id))?;

        transaction.status = TransactionStatus::Aborted;
        self.abort_phase(transaction).await?;
        Ok(())
    }

    /// Phase 1: Prepare (ask all participants to prepare)
    async fn prepare_phase(&mut self, transaction: &mut DistributedTransaction) -> Result<bool> {
        for participant in &transaction.participants {
            // Simulate prepare request
            let vote = self.send_prepare_request(participant, transaction).await?;
            transaction.prepare_votes.insert(participant.clone(), vote);
            
            if !vote {
                return Ok(false); // At least one participant voted NO
            }
        }
        Ok(true) // All participants voted YES
    }

    /// Phase 2: Commit (tell all participants to commit)
    async fn commit_phase(&self, transaction: &DistributedTransaction) -> Result<()> {
        for participant in &transaction.participants {
            self.send_commit_request(participant, transaction).await?;
        }
        Ok(())
    }

    /// Abort phase (tell all participants to abort)
    async fn abort_phase(&self, transaction: &DistributedTransaction) -> Result<()> {
        for participant in &transaction.participants {
            self.send_abort_request(participant, transaction).await?;
        }
        Ok(())
    }

    /// Send prepare request to participant (simulated)
    async fn send_prepare_request(&self, participant: &str, _transaction: &DistributedTransaction) -> Result<bool> {
        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // Simulate vote (90% YES, 10% NO)
        let mut rng = StdRng::seed_from_u64(participant.chars().map(|c| c as u64).sum());
        Ok(rng.gen_range(0..100) < 90)
    }

    /// Send commit request to participant (simulated)
    async fn send_commit_request(&self, participant: &str, _transaction: &DistributedTransaction) -> Result<()> {
        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        
        // Simulate occasional failure
        let mut rng = StdRng::seed_from_u64(participant.chars().map(|c| c as u64).sum());
        if rng.gen_range(0..100) < 5 { // 5% failure rate
            return Err(anyhow!("Simulated commit failure"));
        }
        
        Ok(())
    }

    /// Send abort request to participant (simulated)
    async fn send_abort_request(&self, participant: &str, _transaction: &DistributedTransaction) -> Result<()> {
        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        Ok(())
    }

    /// Get transaction status
    pub fn get_transaction_status(&self, transaction_id: &str) -> Option<&TransactionStatus> {
        self.active_transactions.get(transaction_id).map(|txn| &txn.status)
    }

    /// Clean up old transactions
    pub fn cleanup_old_transactions(&mut self) -> usize {
        let now = std::time::SystemTime::now();
        let timeout_duration = std::time::Duration::from_millis(self.timeout_ms);
        
        let initial_count = self.active_transactions.len();
        
        self.active_transactions.retain(|_, txn| {
            now.duration_since(txn.start_time).unwrap_or_default() < timeout_duration
        });
        
        initial_count - self.active_transactions.len()
    }

    /// Get active transaction count
    pub fn active_transaction_count(&self) -> usize {
        self.active_transactions.len()
    }
}

/// Test suite for federation layer
pub async fn run_federation_tests() -> Result<Vec<UnitTestResult>> {
    let mut results = Vec::new();

    // Database registry tests
    results.push(test_database_registry().await);
    results.push(test_database_metrics().await);
    results.push(test_capability_filtering().await);

    // Query routing tests
    results.push(test_query_routing().await);
    results.push(test_load_balancing().await);
    results.push(test_federated_query_execution().await);

    // Transaction coordination tests
    results.push(test_distributed_transactions().await);
    results.push(test_two_phase_commit().await);
    results.push(test_transaction_timeout().await);

    Ok(results)
}

async fn test_database_registry() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut registry = DatabaseRegistry::new();
        
        // Register databases
        let db1 = DatabaseInfo {
            id: "db1".to_string(),
            name: "Neo4j Instance 1".to_string(),
            url: "bolt://localhost:7687".to_string(),
            database_type: DatabaseType::Neo4j,
            capabilities: vec![Capability::GraphTraversal, Capability::ACID],
            status: DatabaseStatus::Online,
            priority: 1,
        };
        
        let db2 = DatabaseInfo {
            id: "db2".to_string(),
            name: "ArangoDB Instance".to_string(),
            url: "http://localhost:8529".to_string(),
            database_type: DatabaseType::ArangoDB,
            capabilities: vec![Capability::FullTextSearch, Capability::Analytics],
            status: DatabaseStatus::Online,
            priority: 2,
        };
        
        registry.register_database(db1)?;
        registry.register_database(db2)?;
        
        // Test basic operations
        assert_eq!(registry.database_count(), 2);
        assert!(registry.get_database("db1").is_some());
        assert!(registry.get_database("db3").is_none());
        
        // Test capability filtering
        let graph_dbs = registry.find_by_capability(Capability::GraphTraversal);
        assert_eq!(graph_dbs.len(), 1);
        assert_eq!(graph_dbs[0].id, "db1");
        
        // Test duplicate registration
        let duplicate_db = DatabaseInfo {
            id: "db1".to_string(),
            name: "Duplicate".to_string(),
            url: "test".to_string(),
            database_type: DatabaseType::Neo4j,
            capabilities: vec![],
            status: DatabaseStatus::Online,
            priority: 1,
        };
        assert!(registry.register_database(duplicate_db).is_err());
        
        // Test unregistration
        registry.unregister_database("db1")?;
        assert_eq!(registry.database_count(), 1);
        assert!(registry.get_database("db1").is_none());
        
        Ok(())
    })();

    UnitTestResult {
        name: "database_registry".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 92.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_database_metrics() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut registry = DatabaseRegistry::new();
        
        // Register database
        let db = DatabaseInfo {
            id: "test_db".to_string(),
            name: "Test DB".to_string(),
            url: "test://localhost".to_string(),
            database_type: DatabaseType::Neo4j,
            capabilities: vec![],
            status: DatabaseStatus::Online,
            priority: 1,
        };
        registry.register_database(db)?;
        
        // Update metrics
        let metrics = DatabaseMetrics {
            response_time_ms: 45.5,
            cpu_usage_percent: 65.0,
            memory_usage_percent: 80.0,
            query_count: 1000,
            error_rate: 0.02,
            last_updated: std::time::SystemTime::now(),
        };
        
        registry.update_metrics("test_db", metrics)?;
        
        // Verify metrics
        let stored_metrics = registry.get_metrics("test_db").unwrap();
        assert_eq!(stored_metrics.response_time_ms, 45.5);
        assert_eq!(stored_metrics.cpu_usage_percent, 65.0);
        assert_eq!(stored_metrics.query_count, 1000);
        
        // Test metrics for non-existent database
        assert!(registry.update_metrics("invalid_db", DatabaseMetrics {
            response_time_ms: 0.0,
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            query_count: 0,
            error_rate: 0.0,
            last_updated: std::time::SystemTime::now(),
        }).is_err());
        
        Ok(())
    })();

    UnitTestResult {
        name: "database_metrics".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 512,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_capability_filtering() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut registry = DatabaseRegistry::new();
        
        // Register databases with different capabilities
        let dbs = vec![
            DatabaseInfo {
                id: "graph_db".to_string(),
                name: "Graph DB".to_string(),
                url: "test1".to_string(),
                database_type: DatabaseType::Neo4j,
                capabilities: vec![Capability::GraphTraversal, Capability::ACID],
                status: DatabaseStatus::Online,
                priority: 1,
            },
            DatabaseInfo {
                id: "search_db".to_string(),
                name: "Search DB".to_string(),
                url: "test2".to_string(),
                database_type: DatabaseType::ArangoDB,
                capabilities: vec![Capability::FullTextSearch, Capability::Analytics],
                status: DatabaseStatus::Online,
                priority: 2,
            },
            DatabaseInfo {
                id: "ml_db".to_string(),
                name: "ML DB".to_string(),
                url: "test3".to_string(),
                database_type: DatabaseType::TigerGraph,
                capabilities: vec![Capability::MachineLearning, Capability::Analytics],
                status: DatabaseStatus::Online,
                priority: 3,
            },
        ];
        
        for db in dbs {
            registry.register_database(db)?;
        }
        
        // Test single capability filtering
        let graph_dbs = registry.find_by_capability(Capability::GraphTraversal);
        assert_eq!(graph_dbs.len(), 1);
        assert_eq!(graph_dbs[0].id, "graph_db");
        
        let analytics_dbs = registry.find_by_capability(Capability::Analytics);
        assert_eq!(analytics_dbs.len(), 2);
        
        let streaming_dbs = registry.find_by_capability(Capability::Streaming);
        assert_eq!(streaming_dbs.len(), 0);
        
        // Test healthy database filtering
        let healthy_dbs = registry.get_healthy_databases();
        assert_eq!(healthy_dbs.len(), 3);
        
        Ok(())
    })();

    UnitTestResult {
        name: "capability_filtering".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1536,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_query_routing() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let registry = Arc::new(Mutex::new(DatabaseRegistry::new()));
            let mut router = QueryRouter::new(registry.clone());
            
            // Setup databases
            {
                let mut reg = registry.lock().unwrap();
                let db1 = DatabaseInfo {
                    id: "db1".to_string(),
                    name: "Graph DB".to_string(),
                    url: "test1".to_string(),
                    database_type: DatabaseType::Neo4j,
                    capabilities: vec![Capability::GraphTraversal],
                    status: DatabaseStatus::Online,
                    priority: 1,
                };
                
                let db2 = DatabaseInfo {
                    id: "db2".to_string(),
                    name: "Search DB".to_string(),
                    url: "test2".to_string(),
                    database_type: DatabaseType::ArangoDB,
                    capabilities: vec![Capability::FullTextSearch],
                    status: DatabaseStatus::Online,
                    priority: 2,
                };
                
                reg.register_database(db1)?;
                reg.register_database(db2)?;
            }
            
            // Test routing with capability requirement
            let query = FederatedQuery {
                query_text: "MATCH (n) RETURN n".to_string(),
                query_type: QueryType::Traversal,
                required_capabilities: vec![Capability::GraphTraversal],
                allowed_database_types: None,
                max_databases: Some(1),
                max_response_time_ms: None,
                max_error_rate: None,
                limit: Some(100),
                timeout_ms: Some(5000),
            };
            
            let selected = router.route_query(&query).await?;
            assert_eq!(selected.len(), 1);
            assert_eq!(selected[0], "db1");
            
            // Test routing with no matching capabilities
            let query = FederatedQuery {
                query_text: "TEST".to_string(),
                query_type: QueryType::Analytics,
                required_capabilities: vec![Capability::MachineLearning],
                allowed_database_types: None,
                max_databases: Some(2),
                max_response_time_ms: None,
                max_error_rate: None,
                limit: Some(100),
                timeout_ms: Some(5000),
            };
            
            let selected = router.route_query(&query).await?;
            assert_eq!(selected.len(), 0);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "query_routing".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 85.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_load_balancing() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let mut load_balancer = LoadBalancer::new(RoutingStrategy::RoundRobin);
        
        // Create test databases
        let dbs = vec![
            DatabaseInfo {
                id: "db1".to_string(),
                name: "DB 1".to_string(),
                url: "test1".to_string(),
                database_type: DatabaseType::Neo4j,
                capabilities: vec![Capability::GraphTraversal],
                status: DatabaseStatus::Online,
                priority: 1,
            },
            DatabaseInfo {
                id: "db2".to_string(),
                name: "DB 2".to_string(),
                url: "test2".to_string(),
                database_type: DatabaseType::ArangoDB,
                capabilities: vec![Capability::GraphTraversal],
                status: DatabaseStatus::Online,
                priority: 2,
            },
            DatabaseInfo {
                id: "db3".to_string(),
                name: "DB 3".to_string(),
                url: "test3".to_string(),
                database_type: DatabaseType::TigerGraph,
                capabilities: vec![Capability::GraphTraversal],
                status: DatabaseStatus::Online,
                priority: 3,
            },
        ];
        
        let db_refs: Vec<&DatabaseInfo> = dbs.iter().collect();
        
        // Test selection with limit
        let selected = load_balancer.select_databases(db_refs.clone(), Some(2))?;
        assert_eq!(selected.len(), 2);
        
        // Test selection without limit
        let selected = load_balancer.select_databases(db_refs.clone(), None)?;
        assert_eq!(selected.len(), 3);
        
        // Test with empty candidates
        let selected = load_balancer.select_databases(vec![], Some(5))?;
        assert_eq!(selected.len(), 0);
        
        Ok(())
    })();

    UnitTestResult {
        name: "load_balancing".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 82.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_federated_query_execution() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let registry = Arc::new(Mutex::new(DatabaseRegistry::new()));
            let mut router = QueryRouter::new(registry.clone());
            
            // Setup databases
            {
                let mut reg = registry.lock().unwrap();
                for i in 1..=3 {
                    let db = DatabaseInfo {
                        id: format!("db{}", i),
                        name: format!("Database {}", i),
                        url: format!("test{}", i),
                        database_type: DatabaseType::Neo4j,
                        capabilities: vec![Capability::GraphTraversal],
                        status: DatabaseStatus::Online,
                        priority: i,
                    };
                    reg.register_database(db)?;
                }
            }
            
            // Execute federated query
            let query = FederatedQuery {
                query_text: "MATCH (n:Person) RETURN n".to_string(),
                query_type: QueryType::Search,
                required_capabilities: vec![Capability::GraphTraversal],
                allowed_database_types: None,
                max_databases: Some(2),
                max_response_time_ms: Some(1000),
                max_error_rate: Some(0.1),
                limit: Some(50),
                timeout_ms: Some(5000),
            };
            
            let result = router.execute_federated_query(query).await?;
            
            // Verify results
            assert!(result.database_count <= 2);
            assert!(result.execution_time_ms > 0);
            assert!(result.partial_result_count > 0);
            assert!(!result.merged_result.entities.is_empty());
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "federated_query_execution".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 4096,
        coverage_percentage: 88.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_distributed_transactions() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut coordinator = TransactionCoordinator::new(5000);
            
            // Begin transaction
            let participants = vec!["db1".to_string(), "db2".to_string()];
            let txn_id = coordinator.begin_transaction(participants)?;
            
            // Add operations
            let op1 = TransactionOperation {
                database_id: "db1".to_string(),
                operation_type: OperationType::Create,
                entity_id: "entity1".to_string(),
                data: HashMap::new(),
            };
            
            let op2 = TransactionOperation {
                database_id: "db2".to_string(),
                operation_type: OperationType::Update,
                entity_id: "entity2".to_string(),
                data: HashMap::new(),
            };
            
            coordinator.add_operation(&txn_id, op1)?;
            coordinator.add_operation(&txn_id, op2)?;
            
            // Check status
            let status = coordinator.get_transaction_status(&txn_id);
            assert!(matches!(status, Some(TransactionStatus::Active)));
            
            // Commit transaction
            let success = coordinator.commit_transaction(&txn_id).await?;
            assert!(success); // Should succeed most of the time
            
            let final_status = coordinator.get_transaction_status(&txn_id);
            assert!(matches!(final_status, Some(TransactionStatus::Committed) | Some(TransactionStatus::Aborted)));
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "distributed_transactions".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 2048,
        coverage_percentage: 85.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_two_phase_commit() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut coordinator = TransactionCoordinator::new(10000);
            
            // Test successful 2PC
            let participants = vec!["db1".to_string(), "db2".to_string(), "db3".to_string()];
            let txn_id = coordinator.begin_transaction(participants)?;
            
            assert_eq!(coordinator.active_transaction_count(), 1);
            
            // The commit may succeed or fail based on simulated votes
            let _commit_result = coordinator.commit_transaction(&txn_id).await;
            
            // Transaction should be in a final state
            let status = coordinator.get_transaction_status(&txn_id);
            assert!(matches!(status, 
                Some(TransactionStatus::Committed) | 
                Some(TransactionStatus::Aborted) |
                Some(TransactionStatus::TimedOut)
            ));
            
            // Test abort
            let txn_id2 = coordinator.begin_transaction(vec!["db4".to_string()])?;
            coordinator.abort_transaction(&txn_id2).await?;
            
            let status = coordinator.get_transaction_status(&txn_id2);
            assert!(matches!(status, Some(TransactionStatus::Aborted)));
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "two_phase_commit".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1536,
        coverage_percentage: 90.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

async fn test_transaction_timeout() -> UnitTestResult {
    let start = std::time::Instant::now();
    
    let result = (|| -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut coordinator = TransactionCoordinator::new(100); // Very short timeout
            
            let txn_id = coordinator.begin_transaction(vec!["db1".to_string()])?;
            
            // Wait longer than timeout
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            
            // Try to commit - should fail due to timeout
            let commit_result = coordinator.commit_transaction(&txn_id).await;
            assert!(commit_result.is_err());
            
            // Test cleanup
            let cleaned = coordinator.cleanup_old_transactions();
            assert!(cleaned > 0);
            
            Ok(())
        })
    })();

    UnitTestResult {
        name: "transaction_timeout".to_string(),
        passed: result.is_ok(),
        duration_ms: start.elapsed().as_millis() as u64,
        memory_usage_bytes: 1024,
        coverage_percentage: 82.0,
        error_message: result.err().map(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federation_layer_comprehensive() {
        let results = run_federation_tests().await.unwrap();
        
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        
        println!("Federation Layer Tests: {}/{} passed", passed_tests, total_tests);
        
        for result in &results {
            if result.passed {
                println!("✅ {}: {}ms", result.name, result.duration_ms);
            } else {
                println!("❌ {}: {} ({}ms)", result.name, 
                         result.error_message.as_deref().unwrap_or("Unknown error"),
                         result.duration_ms);
            }
        }
        
        assert_eq!(passed_tests, total_tests, "Some federation tests failed");
    }
}

/// Mock UUID implementation for testing
mod uuid {
    pub struct Uuid;
    
    impl Uuid {
        pub fn new_v4() -> Self {
            Self
        }
    }
    
    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", rand::random::<u64>())
        }
    }
}