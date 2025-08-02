//! Learning Systems
//! 
//! This module implements adaptive learning mechanisms that enhance the existing
//! cognitive architecture with biological learning principles.

pub mod hebbian;
pub mod homeostasis;
pub mod adaptive_learning;
pub mod types;
pub mod parameter_tuning;
pub mod meta_learning;

// Re-export specific items to avoid conflicts
pub use hebbian::HebbianLearningEngine;
pub use homeostasis::{SynapticHomeostasis, HomeostasisUpdate};


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


// From types - rename conflicting types
pub use types::{
    ActivationEvent,
    ActivationContext,
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


pub use parameter_tuning::{
    ParameterTuner
};

pub use meta_learning::{
    MetaLearningSystem
};