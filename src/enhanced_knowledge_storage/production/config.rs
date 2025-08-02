//! Production Configuration Management
//! 
//! Comprehensive configuration system for production deployment with environment
//! variable support, validation, and dynamic reloading capabilities.

use std::time::Duration;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use std::env;
use std::fs;

/// Main production configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    // System configuration
    pub system: SystemConfig,
    
    // AI component configurations
    pub model_config: ModelConfig,
    pub local_model_config: LocalModelConfig,
    pub entity_extraction_config: EntityExtractionConfig,
    pub chunking_config: ChunkingConfig,
    pub relationship_config: RelationshipConfig,
    
    // Storage configurations
    pub storage_config: StorageConfig,
    pub retrieval_config: RetrievalConfig,
    pub reasoning_config: ReasoningConfig,
    
    // Infrastructure configurations
    pub caching_config: CachingConfig,
    pub monitoring_config: MonitoringConfig,
    pub error_handling_config: ErrorHandlingConfig,
    
    // API and networking
    pub api_config: ApiConfig,
    pub security_config: SecurityConfig,
    
    // Performance and scaling
    pub performance_config: PerformanceConfig,
    pub scaling_config: ScalingConfig,
    
    // Feature flags
    pub features: FeatureFlags,
}

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub version: String,
    pub environment: Environment,
    pub log_level: LogLevel,
    pub max_concurrent_requests: usize,
    pub request_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub memory_limit: usize,
    pub cpu_limit: f32,
}

/// Model configuration for AI components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub device: DeviceType,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub model_cache_size: usize,
    pub preload_models: bool,
}

/// Local Model Backend configuration for local-only AI components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelConfig {
    pub model_weights_path: PathBuf,
    pub max_loaded_models: usize,
    pub memory_threshold: u64,
    pub load_timeout: Duration,
    pub cache_dir: Option<String>,
    pub model_registry: Vec<LocalModelRegistryEntry>,
}

/// Local Model registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelRegistryEntry {
    pub model_id: String,
    pub model_name: String,
    pub model_type: AIModelType,
    pub local_path: PathBuf,
    pub memory_requirement: u64,
    pub max_sequence_length: usize,
    pub device_preference: ModelDevicePreference,
}

/// AI Model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIModelType {
    LanguageModel,
    EmbeddingModel,
    ClassificationModel,
    GenerativeModel,
}

/// Quantization types for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    None,
    Int8,
    Int4,
    Float16,
}

/// Model device preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelDevicePreference {
    CPU,
    CUDA,
    Metal,
    Auto,
}

/// Entity extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    pub model_name: String,
    pub confidence_threshold: f32,
    pub max_entities_per_chunk: usize,
    pub entity_types: Vec<String>,
    pub use_coreference_resolution: bool,
    pub parallel_processing: bool,
    pub batch_size: usize,
}

/// Semantic chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub strategy: ChunkingStrategy,
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub overlap_size: usize,
    pub semantic_similarity_threshold: f32,
    pub use_sentence_boundaries: bool,
    pub preserve_structure: bool,
}

/// Relationship mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipConfig {
    pub max_relationships_per_entity: usize,
    pub confidence_threshold: f32,
    pub relationship_types: Vec<String>,
    pub use_graph_validation: bool,
    pub enable_temporal_relationships: bool,
}

/// Storage system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub primary_storage: StorageBackend,
    pub backup_storage: Option<StorageBackend>,
    pub neo4j_config: Neo4jConfig,
    pub elasticsearch_config: ElasticsearchConfig,
    pub postgres_config: PostgresConfig,
    pub replication_factor: usize,
    pub backup_interval: Duration,
    pub compression_enabled: bool,
}

/// Retrieval system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub use_reranking: bool,
    pub reranking_model: Option<String>,
    pub enable_query_expansion: bool,
    pub query_expansion_terms: usize,
    pub use_hybrid_search: bool,
    pub semantic_weight: f32,
    pub lexical_weight: f32,
}

/// Reasoning system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub max_reasoning_steps: usize,
    pub confidence_threshold: f32,
    pub reasoning_timeout: Duration,
    pub enable_explanation_generation: bool,
    pub use_chain_of_thought: bool,
    pub parallel_reasoning: bool,
}

/// Caching system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub cache_type: CacheType,
    pub max_memory_usage: usize,
    pub max_entries: usize,
    pub ttl: Duration,
    pub enable_compression: bool,
    pub eviction_policy: EvictionPolicy,
    pub redis_config: Option<RedisConfig>,
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics_port: u16,
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub alert_thresholds: AlertThresholds,
    pub exporters: Vec<MetricExporter>,
    pub dashboard_config: DashboardConfig,
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: Duration,
    pub enable_fallback_mode: bool,
    pub error_logging_level: LogLevel,
    pub enable_error_recovery: bool,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub tls_enabled: bool,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub cors_enabled: bool,
    pub cors_origins: Vec<String>,
    pub api_version: String,
    pub max_request_size: usize,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub authentication_enabled: bool,
    pub jwt_secret: String,
    pub jwt_expiration: Duration,
    pub api_key_enabled: bool,
    pub rate_limiting: RateLimitConfig,
    pub encryption_enabled: bool,
    pub encryption_key: String,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub thread_pool_size: usize,
    pub async_runtime_threads: usize,
    pub memory_pool_size: usize,
    pub gc_interval: Duration,
    pub optimization_level: OptimizationLevel,
    pub enable_profiling: bool,
    pub performance_targets: PerformanceTargets,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub enable_auto_scaling: bool,
    pub min_instances: usize,
    pub max_instances: usize,
    pub scale_up_threshold: f32,
    pub scale_down_threshold: f32,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
    pub target_cpu_utilization: f32,
    pub target_memory_utilization: f32,
}

/// Feature flags for enabling/disabling functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    pub enable_gpu_acceleration: bool,
    pub enable_advanced_caching: bool,
    pub enable_real_time_monitoring: bool,
    pub enable_auto_scaling: bool,
    pub enable_distributed_processing: bool,
    pub enable_experimental_features: bool,
    pub enable_debug_logging: bool,
    pub enable_performance_profiling: bool,
}

// Supporting configuration structures

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
    OpenCL,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    FixedSize,
    Sentence,
    Paragraph,
    Semantic,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Neo4j,
    Elasticsearch,
    PostgreSQL,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
    pub max_connections: usize,
    pub connection_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticsearchConfig {
    pub urls: Vec<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub index_prefix: String,
    pub max_connections: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: usize,
    pub connection_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheType {
    Memory,
    Redis,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub password: Option<String>,
    pub database: usize,
    pub max_connections: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub response_time: Duration,
    pub error_rate: f32,
    pub queue_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricExporter {
    Prometheus,
    Grafana,
    CloudWatch,
    DataDog,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub enabled: bool,
    pub port: u16,
    pub refresh_interval: Duration,
    pub metrics_history: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
    pub burst_size: usize,
    pub enable_per_user_limits: bool,
    pub per_user_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub max_response_time: Duration,
    pub min_throughput: f32,
    pub max_error_rate: f32,
    pub min_availability: f32,
}

impl ProductionConfig {
    /// Load configuration from multiple sources with priority order
    pub fn load() -> Result<Self, ConfigError> {
        // Start with default configuration
        let mut config = Self::default();
        
        // Load from configuration file if it exists
        if let Ok(file_config) = Self::from_file("config/production.toml") {
            config = config.merge(file_config);
        }
        
        // Override with environment-specific file
        let env_file = format!("config/{}.toml", config.system.environment.as_str());
        if let Ok(env_config) = Self::from_file(&env_file) {
            config = config.merge(env_config);
        }
        
        // Override with environment variables
        config = config.merge_from_env()?;
        
        // Validate the final configuration
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration from a TOML file
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ConfigError::FileError(format!("Failed to read {}: {}", path, e)))?;
        
        toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(format!("Failed to parse {}: {}", path, e)))
    }
    
    /// Merge configuration with another configuration
    pub fn merge(mut self, other: Self) -> Self {
        // System config
        if other.system.name != self.system.name && !other.system.name.is_empty() {
            self.system.name = other.system.name;
        }
        if other.system.max_concurrent_requests > 0 {
            self.system.max_concurrent_requests = other.system.max_concurrent_requests;
        }
        
        // AI component configs
        if other.model_config.batch_size > 0 {
            self.model_config.batch_size = other.model_config.batch_size;
        }
        if other.entity_extraction_config.confidence_threshold > 0.0 {
            self.entity_extraction_config.confidence_threshold = other.entity_extraction_config.confidence_threshold;
        }
        
        // Feature flags
        self.features.enable_gpu_acceleration = other.features.enable_gpu_acceleration || self.features.enable_gpu_acceleration;
        self.features.enable_advanced_caching = other.features.enable_advanced_caching || self.features.enable_advanced_caching;
        
        self
    }
    
    /// Merge configuration from environment variables
    pub fn merge_from_env(mut self) -> Result<Self, ConfigError> {
        // System configuration
        if let Ok(val) = env::var("LLMKG_MAX_CONCURRENT_REQUESTS") {
            self.system.max_concurrent_requests = val.parse()
                .map_err(|e| ConfigError::EnvError(format!("Invalid MAX_CONCURRENT_REQUESTS: {}", e)))?;
        }
        
        if let Ok(val) = env::var("LLMKG_REQUEST_TIMEOUT") {
            let seconds: u64 = val.parse()
                .map_err(|e| ConfigError::EnvError(format!("Invalid REQUEST_TIMEOUT: {}", e)))?;
            self.system.request_timeout = Duration::from_secs(seconds);
        }
        
        if let Ok(val) = env::var("LLMKG_MEMORY_LIMIT") {
            self.system.memory_limit = val.parse()
                .map_err(|e| ConfigError::EnvError(format!("Invalid MEMORY_LIMIT: {}", e)))?;
        }
        
        // API configuration
        if let Ok(val) = env::var("LLMKG_API_HOST") {
            self.api_config.host = val;
        }
        
        if let Ok(val) = env::var("LLMKG_API_PORT") {
            self.api_config.port = val.parse()
                .map_err(|e| ConfigError::EnvError(format!("Invalid API_PORT: {}", e)))?;
        }
        
        // Database configurations
        if let Ok(val) = env::var("NEO4J_URI") {
            self.storage_config.neo4j_config.uri = val;
        }
        
        if let Ok(val) = env::var("NEO4J_USERNAME") {
            self.storage_config.neo4j_config.username = val;
        }
        
        if let Ok(val) = env::var("NEO4J_PASSWORD") {
            self.storage_config.neo4j_config.password = val;
        }
        
        if let Ok(val) = env::var("ELASTICSEARCH_URL") {
            self.storage_config.elasticsearch_config.urls = vec![val];
        }
        
        if let Ok(val) = env::var("POSTGRES_HOST") {
            self.storage_config.postgres_config.host = val;
        }
        
        if let Ok(val) = env::var("POSTGRES_PORT") {
            self.storage_config.postgres_config.port = val.parse()
                .map_err(|e| ConfigError::EnvError(format!("Invalid POSTGRES_PORT: {}", e)))?;
        }
        
        if let Ok(val) = env::var("POSTGRES_DATABASE") {
            self.storage_config.postgres_config.database = val;
        }
        
        if let Ok(val) = env::var("POSTGRES_USERNAME") {
            self.storage_config.postgres_config.username = val;
        }
        
        if let Ok(val) = env::var("POSTGRES_PASSWORD") {
            self.storage_config.postgres_config.password = val;
        }
        
        // Security configuration
        if let Ok(val) = env::var("JWT_SECRET") {
            self.security_config.jwt_secret = val;
        }
        
        if let Ok(val) = env::var("ENCRYPTION_KEY") {
            self.security_config.encryption_key = val;
        }
        
        // Feature flags
        if let Ok(val) = env::var("ENABLE_GPU_ACCELERATION") {
            self.features.enable_gpu_acceleration = val.parse().unwrap_or(false);
        }
        
        if let Ok(val) = env::var("ENABLE_AUTO_SCALING") {
            self.features.enable_auto_scaling = val.parse().unwrap_or(false);
        }
        
        Ok(self)
    }
    
    /// Validate configuration for consistency and completeness
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate system configuration
        if self.system.max_concurrent_requests == 0 {
            return Err(ConfigError::ValidationError("max_concurrent_requests must be > 0".to_string()));
        }
        
        if self.system.memory_limit < 1_000_000_000 { // 1GB minimum
            return Err(ConfigError::ValidationError("memory_limit must be at least 1GB".to_string()));
        }
        
        // Validate API configuration
        if self.api_config.port == 0 {
            return Err(ConfigError::ValidationError("API port must be specified".to_string()));
        }
        
        if self.api_config.tls_enabled {
            if self.api_config.cert_path.is_none() || self.api_config.key_path.is_none() {
                return Err(ConfigError::ValidationError("TLS cert and key paths required when TLS enabled".to_string()));
            }
        }
        
        // Validate storage configuration
        if self.storage_config.neo4j_config.uri.is_empty() {
            return Err(ConfigError::ValidationError("Neo4j URI must be specified".to_string()));
        }
        
        if !self.storage_config.neo4j_config.uri.starts_with("neo4j://") && 
           !self.storage_config.neo4j_config.uri.starts_with("bolt://") {
            return Err(ConfigError::ValidationError("Neo4j URI must use neo4j:// or bolt:// scheme".to_string()));
        }
        
        // Validate model configuration
        if self.model_config.batch_size == 0 {
            return Err(ConfigError::ValidationError("Model batch_size must be > 0".to_string()));
        }
        
        if self.model_config.max_sequence_length == 0 {
            return Err(ConfigError::ValidationError("Model max_sequence_length must be > 0".to_string()));
        }
        
        // Validate AI component configurations
        if self.entity_extraction_config.confidence_threshold < 0.0 || self.entity_extraction_config.confidence_threshold > 1.0 {
            return Err(ConfigError::ValidationError("Entity extraction confidence_threshold must be between 0.0 and 1.0".to_string()));
        }
        
        if self.chunking_config.max_chunk_size <= self.chunking_config.min_chunk_size {
            return Err(ConfigError::ValidationError("max_chunk_size must be greater than min_chunk_size".to_string()));
        }
        
        // Validate performance configuration
        if self.performance_config.thread_pool_size == 0 {
            return Err(ConfigError::ValidationError("thread_pool_size must be > 0".to_string()));
        }
        
        // Validate security configuration
        if self.security_config.authentication_enabled && self.security_config.jwt_secret.is_empty() {
            return Err(ConfigError::ValidationError("JWT secret required when authentication enabled".to_string()));
        }
        
        if self.security_config.encryption_enabled && self.security_config.encryption_key.is_empty() {
            return Err(ConfigError::ValidationError("Encryption key required when encryption enabled".to_string()));
        }
        
        Ok(())
    }
    
    /// Get configuration for specific environment
    pub fn for_environment(env: Environment) -> Result<Self, ConfigError> {
        let mut config = Self::default();
        config.system.environment = env;
        
        match config.system.environment {
            Environment::Development => {
                config.system.log_level = LogLevel::Debug;
                config.features.enable_debug_logging = true;
                config.api_config.cors_enabled = true;
                config.monitoring_config.enabled = false;
            },
            Environment::Testing => {
                config.system.log_level = LogLevel::Info;
                config.features.enable_performance_profiling = true;
                config.monitoring_config.enabled = true;
            },
            Environment::Staging => {
                config.system.log_level = LogLevel::Info;
                config.monitoring_config.enabled = true;
                config.security_config.authentication_enabled = true;
            },
            Environment::Production => {
                config.system.log_level = LogLevel::Warn;
                config.monitoring_config.enabled = true;
                config.security_config.authentication_enabled = true;
                config.security_config.encryption_enabled = true;
                config.features.enable_auto_scaling = true;
            },
        }
        
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::SerializationError(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path, content)
            .map_err(|e| ConfigError::FileError(format!("Failed to write config to {}: {}", path, e)))?;
        
        Ok(())
    }
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig {
                name: "LLMKG Production System".to_string(),
                version: "1.0.0".to_string(),
                environment: Environment::Development,
                log_level: LogLevel::Info,
                max_concurrent_requests: 1000,
                request_timeout: Duration::from_secs(30),
                shutdown_timeout: Duration::from_secs(10),
                memory_limit: 8_000_000_000, // 8GB
                cpu_limit: 4.0,
            },
            model_config: ModelConfig {
                model_path: PathBuf::from("models/"),
                device: DeviceType::CPU,
                batch_size: 32,
                max_sequence_length: 512,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                model_cache_size: 3,
                preload_models: true,
            },
            local_model_config: LocalModelConfig {
                model_weights_path: PathBuf::from("./model_weights"),
                max_loaded_models: 3,
                memory_threshold: 8 * 1024 * 1024 * 1024, // 8GB
                load_timeout: Duration::from_secs(300),
                cache_dir: Some("./model_cache".to_string()),
                model_registry: vec![
                    LocalModelRegistryEntry {
                        model_id: "bert-base-uncased".to_string(),
                        model_name: "BERT Base Uncased".to_string(),
                        model_type: AIModelType::LanguageModel,
                        local_path: PathBuf::from("./model_weights/bert-base-uncased"),
                        memory_requirement: 450_000_000, // ~450MB
                        max_sequence_length: 512,
                        device_preference: ModelDevicePreference::Auto,
                    },
                    LocalModelRegistryEntry {
                        model_id: "minilm-l6-v2".to_string(),
                        model_name: "All MiniLM L6 v2".to_string(),
                        model_type: AIModelType::EmbeddingModel,
                        local_path: PathBuf::from("./model_weights/minilm-l6-v2"),
                        memory_requirement: 90_000_000, // ~90MB
                        max_sequence_length: 256,
                        device_preference: ModelDevicePreference::Auto,
                    },
                    LocalModelRegistryEntry {
                        model_id: "bert-large-ner".to_string(),
                        model_name: "BERT Large NER".to_string(),
                        model_type: AIModelType::ClassificationModel,
                        local_path: PathBuf::from("./model_weights/bert-large-ner"),
                        memory_requirement: 1_360_000_000, // ~1.36GB
                        max_sequence_length: 512,
                        device_preference: ModelDevicePreference::Auto,
                    },
                ],
            },
            entity_extraction_config: EntityExtractionConfig {
                model_name: "distilbert-base-cased".to_string(),
                confidence_threshold: 0.85,
                max_entities_per_chunk: 50,
                entity_types: vec![
                    "PERSON".to_string(),
                    "ORG".to_string(),
                    "LOCATION".to_string(),
                    "MISC".to_string(),
                ],
                use_coreference_resolution: true,
                parallel_processing: true,
                batch_size: 16,
            },
            chunking_config: ChunkingConfig {
                strategy: ChunkingStrategy::Semantic,
                max_chunk_size: 1000,
                min_chunk_size: 100,
                overlap_size: 50,
                semantic_similarity_threshold: 0.75,
                use_sentence_boundaries: true,
                preserve_structure: true,
            },
            relationship_config: RelationshipConfig {
                max_relationships_per_entity: 10,
                confidence_threshold: 0.7,
                relationship_types: vec![
                    "RELATED_TO".to_string(),
                    "PART_OF".to_string(),
                    "CAUSES".to_string(),
                    "TEMPORAL".to_string(),
                ],
                use_graph_validation: true,
                enable_temporal_relationships: true,
            },
            storage_config: StorageConfig {
                primary_storage: StorageBackend::Hybrid,
                backup_storage: Some(StorageBackend::PostgreSQL),
                neo4j_config: Neo4jConfig {
                    uri: "neo4j://localhost:7687".to_string(),
                    username: "neo4j".to_string(),
                    password: "password".to_string(),
                    database: "neo4j".to_string(),
                    max_connections: 100,
                    connection_timeout: Duration::from_secs(5),
                },
                elasticsearch_config: ElasticsearchConfig {
                    urls: vec!["http://localhost:9200".to_string()],
                    username: None,
                    password: None,
                    index_prefix: "llmkg".to_string(),
                    max_connections: 50,
                    timeout: Duration::from_secs(10),
                },
                postgres_config: PostgresConfig {
                    host: "localhost".to_string(),
                    port: 5432,
                    database: "llmkg".to_string(),
                    username: "postgres".to_string(),
                    password: "password".to_string(),
                    max_connections: 50,
                    connection_timeout: Duration::from_secs(5),
                },
                replication_factor: 2,
                backup_interval: Duration::from_secs(3600), // 1 hour
                compression_enabled: true,
            },
            retrieval_config: RetrievalConfig {
                max_results: 100,
                similarity_threshold: 0.7,
                use_reranking: true,
                reranking_model: Some("cross-encoder/ms-marco-MiniLM-L-6-v2".to_string()),
                enable_query_expansion: true,
                query_expansion_terms: 5,
                use_hybrid_search: true,
                semantic_weight: 0.7,
                lexical_weight: 0.3,
            },
            reasoning_config: ReasoningConfig {
                max_reasoning_steps: 10,
                confidence_threshold: 0.6,
                reasoning_timeout: Duration::from_secs(30),
                enable_explanation_generation: true,
                use_chain_of_thought: true,
                parallel_reasoning: false,
            },
            caching_config: CachingConfig {
                cache_type: CacheType::Hybrid,
                max_memory_usage: 2_000_000_000, // 2GB
                max_entries: 100_000,
                ttl: Duration::from_secs(3600), // 1 hour
                enable_compression: true,
                eviction_policy: EvictionPolicy::LRU,
                redis_config: Some(RedisConfig {
                    url: "redis://localhost:6379".to_string(),
                    password: None,
                    database: 0,
                    max_connections: 20,
                    timeout: Duration::from_secs(5),
                }),
            },
            monitoring_config: MonitoringConfig {
                enabled: true,
                metrics_port: 9090,
                collection_interval: Duration::from_secs(10),
                retention_period: Duration::from_secs(604800), // 1 week
                alert_thresholds: AlertThresholds {
                    cpu_usage: 80.0,
                    memory_usage: 85.0,
                    disk_usage: 90.0,
                    response_time: Duration::from_millis(1000),
                    error_rate: 5.0,
                    queue_size: 1000,
                },
                exporters: vec![MetricExporter::Prometheus],
                dashboard_config: DashboardConfig {
                    enabled: true,
                    port: 3000,
                    refresh_interval: Duration::from_secs(5),
                    metrics_history: Duration::from_secs(86400), // 24 hours
                },
            },
            error_handling_config: ErrorHandlingConfig {
                max_retries: 3,
                retry_delay: Duration::from_millis(1000),
                circuit_breaker_threshold: 5,
                circuit_breaker_timeout: Duration::from_secs(60),
                enable_fallback_mode: true,
                error_logging_level: LogLevel::Error,
                enable_error_recovery: true,
            },
            api_config: ApiConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                tls_enabled: false,
                cert_path: None,
                key_path: None,
                cors_enabled: false,
                cors_origins: vec!["*".to_string()],
                api_version: "v1".to_string(),
                max_request_size: 10_000_000, // 10MB
            },
            security_config: SecurityConfig {
                authentication_enabled: false,
                jwt_secret: "change-me-in-production".to_string(),
                jwt_expiration: Duration::from_secs(86400), // 24 hours
                api_key_enabled: false,
                rate_limiting: RateLimitConfig {
                    requests_per_minute: 1000,
                    burst_size: 100,
                    enable_per_user_limits: false,
                    per_user_limit: 100,
                },
                encryption_enabled: false,
                encryption_key: "change-me-in-production".to_string(),
            },
            performance_config: PerformanceConfig {
                thread_pool_size: 8,
                async_runtime_threads: 4,
                memory_pool_size: 1_000_000_000, // 1GB
                gc_interval: Duration::from_secs(60),
                optimization_level: OptimizationLevel::Basic,
                enable_profiling: false,
                performance_targets: PerformanceTargets {
                    max_response_time: Duration::from_millis(500),
                    min_throughput: 100.0,
                    max_error_rate: 1.0,
                    min_availability: 99.9,
                },
            },
            scaling_config: ScalingConfig {
                enable_auto_scaling: false,
                min_instances: 1,
                max_instances: 10,
                scale_up_threshold: 70.0,
                scale_down_threshold: 30.0,
                scale_up_cooldown: Duration::from_secs(300),
                scale_down_cooldown: Duration::from_secs(600),
                target_cpu_utilization: 70.0,
                target_memory_utilization: 80.0,
            },
            features: FeatureFlags {
                enable_gpu_acceleration: false,
                enable_advanced_caching: true,
                enable_real_time_monitoring: true,
                enable_auto_scaling: false,
                enable_distributed_processing: false,
                enable_experimental_features: false,
                enable_debug_logging: false,
                enable_performance_profiling: false,
            },
        }
    }
}

impl Environment {
    pub fn as_str(&self) -> &'static str {
        match self {
            Environment::Development => "development",
            Environment::Testing => "testing",
            Environment::Staging => "staging",
            Environment::Production => "production",
        }
    }
}

/// Configuration error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("File error: {0}")]
    FileError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Environment variable error: {0}")]
    EnvError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config_validation() {
        let config = ProductionConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_environment_specific_config() {
        let prod_config = ProductionConfig::for_environment(Environment::Production).unwrap();
        assert_eq!(prod_config.system.environment, Environment::Production);
        assert!(prod_config.security_config.authentication_enabled);
        assert!(prod_config.security_config.encryption_enabled);
    }
    
    #[test]
    fn test_config_merge() {
        let mut base_config = ProductionConfig::default();
        base_config.system.max_concurrent_requests = 500;
        
        let mut override_config = ProductionConfig::default();
        override_config.system.max_concurrent_requests = 1000;
        override_config.features.enable_gpu_acceleration = true;
        
        let merged = base_config.merge(override_config);
        assert_eq!(merged.system.max_concurrent_requests, 1000);
        assert!(merged.features.enable_gpu_acceleration);
    }
}