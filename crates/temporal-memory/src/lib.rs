//! Temporal memory management
//!
//! Handles memory branches and consolidation states for
//! neuromorphic temporal versioning.

pub mod branch;
pub mod consolidation;

pub use branch::{MemoryBranch, BranchId, BranchMetadata, BranchRelationship, ConsolidationState};
pub use consolidation::{ConsolidationEngine, ConsolidationConfig, ConsolidationResult};
