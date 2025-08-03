//! Temporal memory management
//! 
//! Handles memory branches and consolidation states.

pub mod branch;
pub mod consolidation;

pub use branch::{
    MemoryBranch, BranchMetadata, BranchRelationship, ConsolidationState,
    BranchManager, BranchEvent, BranchError, BranchStats, BranchConfig
};
pub use neuromorphic_core::BranchId;
pub use consolidation::{ConsolidationEngine, ConsolidationConfig, ConsolidationResult, ConflictType, Conflict};
