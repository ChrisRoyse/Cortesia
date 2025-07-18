//! Phase 4: Self-Organization & Learning Systems
//! 
//! This module implements adaptive learning mechanisms that enhance the existing
//! cognitive architecture with biological learning principles.

pub mod hebbian;
pub mod homeostasis;
pub mod optimization_agent;
pub mod adaptive_learning;
pub mod phase4_integration;
pub mod types;
pub mod neural_pattern_detection;
pub mod parameter_tuning;
pub mod meta_learning;

// Re-export specific items to avoid conflicts
pub use hebbian::HebbianLearningEngine;
pub use homeostasis::{SynapticHomeostasis, HomeostasisUpdate};

// From optimization_agent - rename conflicting types
pub use optimization_agent::{
    GraphOptimizationAgent,
    OptimizationOpportunity,
    PatternDetector as OptimizationPatternDetector,
    AnalysisScope as OptimizationAnalysisScope,
    PerformanceBottleneck as OptimizationPerformanceBottleneck
};

// From adaptive_learning - rename conflicting types
pub use adaptive_learning::{
    AdaptiveLearningSystem,
    AdaptiveLearningConfig,
    PerformanceBottleneck as AdaptivePerformanceBottleneck,
    QueryMetrics,
    CognitiveMetrics,
    SystemMetrics,
    ResourceRequirement
};

pub use phase4_integration::{
    Phase4LearningSystem,
    ComprehensiveLearningResult
};

// From types - rename conflicting types
pub use types::{
    ActivationEvent,
    LearningContext,
    WeightChange,
    LearningUpdate,
    STDPResult,
    PlasticityType,
    LearningResult,
    LearningGoal,
    LearningGoalType,
    PerformanceBottleneck as CorePerformanceBottleneck
};

// From neural_pattern_detection - rename conflicting types
pub use neural_pattern_detection::{
    NeuralPatternDetectionSystem,
    PatternDetector as NeuralPatternDetector,
    AnalysisScope as NeuralAnalysisScope
};

pub use parameter_tuning::{
    ParameterTuner
};

pub use meta_learning::{
    MetaLearningSystem
};