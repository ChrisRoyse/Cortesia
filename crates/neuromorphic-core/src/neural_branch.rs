//! Neural memory branch implementation
//!
//! This module provides temporal versioning through neuromorphic
//! memory branches with consolidation states.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Unique identifier for a memory branch
pub type BranchId = String;

/// Represents different consolidation states of memory
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConsolidationState {
    /// Working memory (< 30 seconds)
    WorkingMemory,
    /// Short-term memory (< 1 hour)
    ShortTerm,
    /// Consolidating (1-24 hours)
    Consolidating,
    /// Long-term memory (> 24 hours)
    LongTerm,
}

/// Represents a neuromorphic memory branch for temporal versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicMemoryBranch {
    id: BranchId,
    parent: Option<BranchId>,
    timestamp: DateTime<Utc>,
    consolidation_state: ConsolidationState,
}

impl NeuromorphicMemoryBranch {
    /// Creates a new memory branch
    pub fn new(id: impl Into<String>, parent: Option<impl Into<String>>) -> Self {
        Self {
            id: id.into(),
            parent: parent.map(Into::into),
            timestamp: Utc::now(),
            consolidation_state: ConsolidationState::WorkingMemory,
        }
    }

    /// Returns the branch ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the parent branch ID if it exists
    pub fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    /// Returns the branch timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    /// Returns the current consolidation state
    pub fn consolidation_state(&self) -> ConsolidationState {
        self.consolidation_state
    }
}
