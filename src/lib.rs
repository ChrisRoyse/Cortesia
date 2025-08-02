#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]
#![allow(clippy::wildcard_in_or_patterns)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::needless_late_init)]
#![doc = include_str!("../docs/enhanced_knowledge_storage/README.md")]

#[cfg(target_arch = "wasm32")]
extern crate alloc;


pub mod core;
pub mod embedding;
pub mod storage;
pub mod query;
pub mod error;
pub mod error_recovery;
pub mod federation;
pub mod versioning;
pub mod math;
pub mod extraction;
pub mod streaming;
pub mod gpu;
pub mod validation;
pub mod monitoring;
pub mod text;
pub mod cognitive;
pub mod learning;
pub mod graph;
pub mod api;
pub mod models;
pub mod cli;
pub mod enhanced_knowledge_storage;

#[cfg(feature = "native")]
pub mod production;

pub mod test_support;

#[cfg(test)]
pub mod test_utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "native")]
pub mod mcp;

pub use crate::core::graph::KnowledgeGraph;
pub use crate::core::entity_compat::{Entity as CompatEntity, Relationship as CompatRelationship, SimilarityResult};
pub use crate::core::types::{EntityKey, EntityData, EntityMeta, Relationship, AttributeValue, RelationshipType, Weight};
pub use crate::embedding::store_compat::EmbeddingStore;
pub use crate::embedding::quantizer::ProductQuantizer;
pub use crate::storage::persistent_mmap::{PersistentMMapStorage, StorageStats};
pub use crate::storage::string_interner::{StringInterner, InternedString, InternedProperties, InternerStats, GlobalStringInterner, intern_string, get_string, interner_stats, clear_interner};
pub use crate::core::interned_entity::{InternedEntityData, InternedRelationship, InternedEntityCollection, InternedDataStats};
pub use crate::error::{GraphError, Result};
pub use crate::federation::{FederationManager, DatabaseRegistry, FederatedQuery};
pub use crate::versioning::{MultiDatabaseVersionManager, VersionId, TemporalQuery};
pub use crate::math::{MathEngine, SimilarityMetric, MathematicalResult};
pub use crate::extraction::{AdvancedEntityExtractor, Entity as ExtractionEntity, Relation};
pub use crate::streaming::{StreamingUpdateHandler, IncrementalIndexer, UpdateStream};
pub use crate::gpu::GpuAccelerator;
pub use crate::validation::{HumanValidationInterface, ValidationItem, ValidationResult};
pub use crate::monitoring::{PerformanceMonitor, ObservabilityEngine, AlertManager};
pub use crate::models::{
    Model, ModelMetadata, ModelCapabilities, ModelSize, ModelConfig, LoadingConfig, DeviceConfig, QuantizationConfig,
    ModelLoader, AdvancedModelLoader, ModelRegistry, RecommendedModels, RegistryStatistics,
    smollm, tinyllama, openelm, minilm, utils as model_utils
};

// Production system exports
#[cfg(feature = "native")]
pub use crate::production::{
    ProductionSystem, ProductionConfig, OperationConfig,
    ErrorRecoveryManager, RetryConfig, CircuitBreakerConfig, CircuitState,
    ProductionMonitor, MonitoringConfig, LogLevel, MetricType, AlertSeverity,
    RateLimitingManager, RateLimitConfig, ResourceLimits, OperationPermit, DatabaseConnection,
    HealthCheckSystem, HealthCheckConfig, HealthStatus, SystemHealthReport,
    GracefulShutdownManager, ShutdownConfig, ShutdownPhase, ActiveRequestGuard,
    create_production_system, create_production_system_with_config
};

// Brain-Inspired Exports
pub use crate::core::brain_types::{
    BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType,
    BrainInspiredRelationship, ActivationPattern, ActivationStep, GraphOperation, TrainingExample
};
pub use crate::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange, TemporalEntity, TemporalRelationship};
pub use crate::streaming::temporal_updates::{IncrementalTemporalProcessor, TemporalUpdate, UpdateOperation, UpdateSource, UpdateStatistics, TemporalUpdateBuilder};

#[cfg(feature = "native")]
pub use crate::mcp::production_server::ProductionMCPServer;
#[cfg(feature = "native")]
pub use crate::mcp::shared_types::{MCPTool, MCPRequest, MCPResponse, MCPContent};

// Text processing exports  
pub use crate::text::{
    TextCompressor, utils as text_utils, TextChunk, Chunker, SlidingWindowChunker, SemanticChunker, AdaptiveChunker,
    StringNormalizer, HeuristicImportanceScorer, GraphMetrics, GraphStructurePredictor
    // GraphOperation re-exported from brain_types to avoid conflict
};

// Brain-Enhanced Knowledge Graph Exports
pub use crate::core::brain_enhanced_graph::{
    BrainEnhancedKnowledgeGraph, BrainMemoryUsage, BrainQueryResult, ConceptStructure,
    BrainStatistics, BrainEnhancedConfig, ActivationPropagationResult,
    EntityStatistics, QueryStatistics, RelationshipStatistics, RelationshipPattern,
    EntityRole, SplitCriteria, OptimizationResult, ConceptUsageStats, GraphPatternAnalysis
};

// Cognitive Pattern Engine Exports
pub use crate::cognitive::{
    CognitiveOrchestrator, CognitiveOrchestratorConfig, OrchestratorStatistics,
    CognitivePattern, CognitivePatternType, PatternParameters, PatternResult, 
    ReasoningStrategy, ReasoningResult, ComplexityEstimate, QualityMetrics,
    ConvergentThinking, ConvergentResult,
    DivergentThinking, DivergentResult, ExplorationType, ExplorationPath,
    LateralThinking, LateralResult, BridgePath, NoveltyAnalysis,
    SystemsThinking, SystemsResult, SystemsReasoningType,
    CriticalThinking, CriticalResult, ValidationLevel,
    AbstractThinking, AbstractResult, PatternType, AnalysisScope,
    AdaptiveThinking, AdaptiveResult, QueryCharacteristics, StrategySelection
};

// Advanced Reasoning System Exports  
pub use crate::cognitive::{
    WorkingMemorySystem, MemoryQuery, MemoryRetrievalResult, MemoryContent, BufferType,
    AttentionManager, AttentionState, AttentionFocus, AttentionType, AttentionTarget, AttentionTargetType, ExecutiveCommand,
    CompetitiveInhibitionSystem, InhibitionResult, CompetitionGroup, CompetitionType, InhibitionPerformanceMetrics,
    UnifiedMemorySystem, UnifiedRetrievalResult, RetrievalStrategy,
};

// Learning System Exports
pub use crate::learning::{
    HebbianLearningEngine, SynapticHomeostasis, AdaptiveLearningSystem,
    ActivationEvent, LearningContext, WeightChange, LearningUpdate,
    STDPResult, PlasticityType, HomeostasisUpdate, LearningResult,
    LearningGoal, LearningGoalType, ResourceRequirement
};

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::wasm32::unreachable();
    }
}

