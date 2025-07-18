#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(dead_code)]

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
pub mod neural;
pub mod streaming;
pub mod gpu;
pub mod validation;
pub mod monitoring;
pub mod agents;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "native")]
pub mod mcp;

pub use crate::core::graph::KnowledgeGraph;
pub use crate::core::entity_compat::{Entity as CompatEntity, Relationship as CompatRelationship, SimilarityResult};
pub use crate::core::types::{EntityKey, EntityData, EntityMeta, Relationship};
pub use crate::embedding::store_compat::EmbeddingStore;
pub use crate::embedding::quantizer::ProductQuantizer;
pub use crate::error::{GraphError, Result};
pub use crate::federation::{FederationManager, DatabaseRegistry, FederatedQuery};
pub use crate::versioning::{MultiDatabaseVersionManager, VersionId, TemporalQuery};
pub use crate::math::{MathEngine, SimilarityMetric, MathematicalResult};
pub use crate::extraction::{AdvancedEntityExtractor, Entity as ExtractionEntity, Relation};
pub use crate::neural::{NeuralSummarizer, NeuralCanonicalizer, NeuralSalienceModel};
pub use crate::streaming::{StreamingUpdateHandler, IncrementalIndexer, UpdateStream};
pub use crate::gpu::GpuAccelerator;
pub use crate::validation::{HumanValidationInterface, ValidationItem, ValidationResult};
pub use crate::monitoring::{PerformanceMonitor, ObservabilityEngine, AlertManager};
pub use crate::agents::{MultiAgentCoordinator, KnowledgeAgent, ConstructionTask, ConstructionResult, ConsensusProtocol, MergeStrategy};

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