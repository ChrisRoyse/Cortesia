// Database registry and discovery for federation

use crate::federation::types::{DatabaseId, DatabaseCapabilities, DatabaseHealth};
use crate::error::{GraphError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, Instant};

/// Descriptor for a database in the federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseDescriptor {
    pub id: DatabaseId,
    pub name: String,
    pub description: Option<String>,
    pub connection_string: String,
    pub database_type: DatabaseType,
    pub capabilities: DatabaseCapabilities,
    pub metadata: DatabaseMetadata,
    pub status: DatabaseStatus,
}

/// Types of databases supported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    KnowledgeGraph,  // Native LLMKG format
    SQLite,
    PostgreSQL,
    Neo4j,
    MongoDB,
    InMemory,
}

/// Metadata about a database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    pub version: String,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
    pub owner: Option<String>,
    pub tags: Vec<String>,
    pub entity_count: Option<usize>,
    pub relationship_count: Option<usize>,
    pub storage_size_bytes: Option<u64>,
}

/// Status of a database
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseStatus {
    Online,
    Offline,
    Maintenance,
    ReadOnly,
    Synchronizing,
}

/// Discovery manager for automatic database detection
pub struct DiscoveryManager {
    discovery_sources: Vec<DiscoverySource>,
    auto_discovery_enabled: bool,
}

/// Sources for database discovery
#[derive(Debug, Clone)]
pub enum DiscoverySource {
    ConfigFile(String),
    Environment,
    Network(String),
    Registry(String),
}

/// Main database registry
#[derive(Clone)]
pub struct DatabaseRegistry {
    databases: Arc<RwLock<HashMap<DatabaseId, DatabaseDescriptor>>>,
    discovery_manager: Arc<DiscoveryManager>,
    health_cache: Arc<RwLock<HashMap<DatabaseId, (DatabaseHealth, Instant)>>>,
    cache_ttl_seconds: u64,
}

impl DatabaseRegistry {
    pub fn new() -> Result<Self> {
        Ok(Self {
            databases: Arc::new(RwLock::new(HashMap::new())),
            discovery_manager: Arc::new(DiscoveryManager::new()),
            health_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl_seconds: 300, // 5 minutes
        })
    }

    /// Register a new database
    pub async fn register(&self, descriptor: DatabaseDescriptor) -> Result<()> {
        let mut databases = self.databases.write().await;
        databases.insert(descriptor.id.clone(), descriptor);
        Ok(())
    }

    /// Unregister a database
    pub async fn unregister(&self, database_id: &DatabaseId) -> Result<()> {
        let mut databases = self.databases.write().await;
        databases.remove(database_id);
        
        // Clear health cache
        let mut health_cache = self.health_cache.write().await;
        health_cache.remove(database_id);
        
        Ok(())
    }

    /// Get a database descriptor
    pub async fn get(&self, database_id: &DatabaseId) -> Option<DatabaseDescriptor> {
        let databases = self.databases.read().await;
        databases.get(database_id).cloned()
    }

    /// List all registered databases
    pub async fn list_databases(&self) -> Vec<DatabaseDescriptor> {
        let databases = self.databases.read().await;
        databases.values().cloned().collect()
    }

    /// Get databases with specific capabilities
    pub async fn get_databases_with_capabilities(&self, required_capabilities: &DatabaseCapabilities) -> Vec<DatabaseDescriptor> {
        let databases = self.databases.read().await;
        databases.values()
            .filter(|db| self.has_required_capabilities(&db.capabilities, required_capabilities))
            .cloned()
            .collect()
    }

    /// Check if a database has the required capabilities
    fn has_required_capabilities(&self, db_capabilities: &DatabaseCapabilities, required: &DatabaseCapabilities) -> bool {
        if required.supports_versioning && !db_capabilities.supports_versioning {
            return false;
        }
        if required.supports_vector_similarity && !db_capabilities.supports_vector_similarity {
            return false;
        }
        if required.supports_temporal_queries && !db_capabilities.supports_temporal_queries {
            return false;
        }
        if required.supports_graph_algorithms && !db_capabilities.supports_graph_algorithms {
            return false;
        }
        
        // Check if all required math operations are supported
        for required_op in &required.supported_math_operations {
            if !db_capabilities.supported_math_operations.contains(required_op) {
                return false;
            }
        }
        
        true
    }

    /// Perform health check on a specific database
    pub async fn health_check(&self, database_id: &DatabaseId) -> Result<DatabaseHealth> {
        // Check cache first
        {
            let health_cache = self.health_cache.read().await;
            if let Some((health, cached_at)) = health_cache.get(database_id) {
                if cached_at.elapsed().as_secs() < self.cache_ttl_seconds {
                    return Ok(health.clone());
                }
            }
        }

        // Perform actual health check
        let _start_time = Instant::now();
        let health = self.perform_health_check(database_id).await?;
        
        // Update cache
        {
            let mut health_cache = self.health_cache.write().await;
            health_cache.insert(database_id.clone(), (health.clone(), Instant::now()));
        }
        
        Ok(health)
    }

    /// Perform health check on all databases
    pub async fn health_check_all(&self) -> Vec<DatabaseHealth> {
        let databases = self.databases.read().await;
        let mut results = Vec::new();
        
        for database_id in databases.keys() {
            if let Ok(health) = self.health_check(database_id).await {
                results.push(health);
            }
        }
        
        results
    }

    /// Discover new databases automatically
    pub async fn discover_databases(&self) -> Result<Vec<DatabaseDescriptor>> {
        self.discovery_manager.discover().await
    }

    /// Update database metadata
    pub async fn update_metadata(&self, database_id: &DatabaseId, metadata: DatabaseMetadata) -> Result<()> {
        let mut databases = self.databases.write().await;
        if let Some(descriptor) = databases.get_mut(database_id) {
            descriptor.metadata = metadata;
            Ok(())
        } else {
            Err(GraphError::InvalidInput(format!("Database not found: {}", database_id.as_str())))
        }
    }

    /// Get databases by status
    pub async fn get_databases_by_status(&self, status: DatabaseStatus) -> Vec<DatabaseDescriptor> {
        let databases = self.databases.read().await;
        databases.values()
            .filter(|db| db.status == status)
            .cloned()
            .collect()
    }

    /// Get federation statistics
    pub async fn get_federation_stats(&self) -> FederationStats {
        let databases = self.databases.read().await;
        let health_cache = self.health_cache.read().await;
        
        let total_databases = databases.len();
        let online_databases = databases.values()
            .filter(|db| db.status == DatabaseStatus::Online)
            .count();
        let healthy_databases = health_cache.values()
            .filter(|(health, _)| health.is_healthy)
            .count();
        
        let total_entities = databases.values()
            .filter_map(|db| db.metadata.entity_count)
            .sum();
        let total_relationships = databases.values()
            .filter_map(|db| db.metadata.relationship_count)
            .sum();
        let total_storage = databases.values()
            .filter_map(|db| db.metadata.storage_size_bytes)
            .sum();

        FederationStats {
            total_databases,
            online_databases,
            healthy_databases,
            total_entities,
            total_relationships,
            total_storage_bytes: total_storage,
            average_response_time_ms: self.calculate_average_response_time(&health_cache),
        }
    }

    /// Calculate average response time from health cache
    fn calculate_average_response_time(&self, health_cache: &HashMap<DatabaseId, (DatabaseHealth, Instant)>) -> f64 {
        let response_times: Vec<u64> = health_cache.values()
            .filter_map(|(health, _)| health.response_time_ms)
            .collect();
        
        if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
        }
    }

    /// Perform the actual health check implementation
    async fn perform_health_check(&self, database_id: &DatabaseId) -> Result<DatabaseHealth> {
        let databases = self.databases.read().await;
        let descriptor = databases.get(database_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Database not found: {}", database_id.as_str())))?;

        let start_time = Instant::now();
        
        // This would be implemented based on the actual database type
        // For now, return a mock health check
        let is_healthy = matches!(descriptor.status, DatabaseStatus::Online);
        let response_time = start_time.elapsed().as_millis() as u64;
        
        Ok(DatabaseHealth {
            database_id: database_id.clone(),
            is_healthy,
            response_time_ms: Some(response_time),
            last_error: None,
            capabilities: descriptor.capabilities.clone(),
            entity_count: descriptor.metadata.entity_count,
            memory_usage_mb: None, // Would be retrieved from actual database
        })
    }
}

impl Default for DiscoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoveryManager {
    pub fn new() -> Self {
        Self {
            discovery_sources: vec![
                DiscoverySource::Environment,
                DiscoverySource::ConfigFile("federation_config.json".to_string()),
            ],
            auto_discovery_enabled: true,
        }
    }

    /// Discover databases from all configured sources
    pub async fn discover(&self) -> Result<Vec<DatabaseDescriptor>> {
        let mut discovered = Vec::new();
        
        for source in &self.discovery_sources {
            match source {
                DiscoverySource::Environment => {
                    discovered.extend(self.discover_from_environment().await?);
                },
                DiscoverySource::ConfigFile(path) => {
                    discovered.extend(self.discover_from_config_file(path).await?);
                },
                _ => {
                    // Other discovery sources would be implemented here
                }
            }
        }
        
        Ok(discovered)
    }

    /// Discover databases from environment variables
    async fn discover_from_environment(&self) -> Result<Vec<DatabaseDescriptor>> {
        // Implementation would read environment variables like:
        // LLMKG_DB_1=sqlite://./db1.sqlite
        // LLMKG_DB_2=postgres://localhost:5432/kg
        Ok(Vec::new())
    }

    /// Discover databases from configuration file
    async fn discover_from_config_file(&self, _path: &str) -> Result<Vec<DatabaseDescriptor>> {
        // Implementation would read JSON/YAML configuration file
        Ok(Vec::new())
    }
}

/// Federation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    pub total_databases: usize,
    pub online_databases: usize,
    pub healthy_databases: usize,
    pub total_entities: usize,
    pub total_relationships: usize,
    pub total_storage_bytes: u64,
    pub average_response_time_ms: f64,
}