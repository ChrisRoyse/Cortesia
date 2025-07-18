use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum GraphError {
    #[error("Entity not found: {id}")]
    EntityNotFound { id: u32 },
    
    #[error("Entity key not found: {key:?}")]
    EntityKeyNotFound { key: crate::core::types::EntityKey },
    
    #[error("Memory allocation failed")]
    OutOfMemory,
    
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },
    
    #[error("Query timeout after {timeout_ms}ms")]
    QueryTimeout { timeout_ms: u64 },
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Invalid entity type: {type_id}")]
    InvalidEntityType { type_id: u16 },
    
    #[error("Relationship not found between entities {from} and {to}")]
    RelationshipNotFound { from: u32, to: u32 },
    
    #[error("Index corruption detected")]
    IndexCorruption,
    
    #[error("Operation not supported in WASM: {operation}")]
    WasmUnsupported { operation: String },
    
    #[error("Operation timeout: {0}")]
    OperationTimeout(String),
    
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    #[error("Federation error: {0}")]
    FederationError(String),
    
    #[error("Database connection error: {0}")]
    DatabaseConnectionError(String),
    
    #[error("Consistency violation: {0}")]
    ConsistencyViolation(String),
    
    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    
    #[error("Validation timeout: {0}")]
    ValidationTimeout(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Invalid weight value: {0}")]
    InvalidWeight(f32),
    
    #[error("Invalid relationship weight: {0}")]
    InvalidRelationshipWeight(f32),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    // Brain-inspired graph specific errors
    #[error("Neural server error: {0}")]
    NeuralServerError(String),
    
    #[error("Activation propagation failed: {0}")]
    ActivationPropagationFailed(String),
    
    #[error("Logic gate error: {0}")]
    LogicGateError(String),
    
    #[error("SDR encoding error: {0}")]
    SDREncodingError(String),
    
    #[error("Temporal consistency violation: {0}")]
    TemporalConsistencyViolation(String),
    
    #[error("Brain entity not found: {0}")]
    BrainEntityNotFound(String),
    
    #[error("Canonicalization failed: {0}")]
    CanonicalizationFailed(String),
    
    #[error("Structure prediction failed: {0}")]
    StructurePredictionFailed(String),
    
    // Cognitive pattern specific errors
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Pattern not found: {0}")]
    PatternNotFound(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<anyhow::Error> for GraphError {
    fn from(err: anyhow::Error) -> Self {
        GraphError::ProcessingError(err.to_string())
    }
}

impl From<GraphError> for String {
    fn from(err: GraphError) -> Self {
        err.to_string()
    }
}

pub type Result<T> = std::result::Result<T, GraphError>;