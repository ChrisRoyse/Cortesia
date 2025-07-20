// Phase 1 library exports - excluding modules with missing dependencies

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
pub mod neural;
pub mod streaming;
pub mod gpu;
pub mod validation;
//pub mod monitoring; // Excluded due to missing dependencies
pub mod text;
pub mod mcp;

pub use crate::core::graph::KnowledgeGraph;
pub use crate::core::brain_types::{
    BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType,
    BrainInspiredRelationship, ActivationPattern
};
pub use crate::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
pub use crate::neural::neural_server::NeuralProcessingServer;
pub use crate::neural::structure_predictor::GraphStructurePredictor;
pub use crate::mcp::brain_inspired_server::BrainInspiredMCPServer;
pub use crate::core::phase1_integration::{Phase1IntegrationLayer, Phase1Config};