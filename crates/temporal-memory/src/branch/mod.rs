//! Memory branch management for temporal versioning

pub mod types;
pub mod state;

pub use types::{MemoryBranch, BranchMetadata, BranchRelationship};
pub use state::{ConsolidationState, StateTransition};

use std::time::Duration;

/// Configuration for memory branches
#[derive(Debug, Clone)]
pub struct BranchConfig {
    /// Maximum age before consolidation required
    pub max_age: Duration,
    
    /// Maximum divergence from parent
    pub max_divergence: f32,
    
    /// Auto-consolidation threshold
    pub auto_consolidate_threshold: f32,
    
    /// Enable conflict detection
    pub detect_conflicts: bool,
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(86400), // 24 hours
            max_divergence: 0.3,
            auto_consolidate_threshold: 0.8,
            detect_conflicts: true,
        }
    }
}