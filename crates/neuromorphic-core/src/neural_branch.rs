//! Neural memory branch implementation
//!
//! This module provides temporal versioning through neuromorphic
//! memory branches with consolidation states.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Unique identifier for a memory branch
pub type BranchId = String;

/// Represents different consolidation states of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationState {
    /// Working memory (< 30 seconds)
    WorkingMemory,
    /// Short-term memory (< 1 hour)
    ShortTerm,
    /// Consolidating (1-24 hours)
    Consolidating,
    /// Long-term memory (> 24 hours)
    LongTerm,
    /// Memory branch has conflicts that need resolution
    Conflicted,
}

impl ConsolidationState {
    /// Determine state based on memory age
    pub fn from_age(age: chrono::Duration) -> Self {
        match age.num_seconds() {
            0..=30 => Self::WorkingMemory,
            31..=3600 => Self::ShortTerm,
            3601..=86400 => Self::Consolidating,
            _ => Self::LongTerm,
        }
    }
    
    /// Check if this state represents active memory (can receive new data)
    pub fn is_active(&self) -> bool {
        matches!(self, Self::WorkingMemory | Self::ShortTerm)
    }
    
    /// Check if this state represents stable memory
    pub fn is_stable(&self) -> bool {
        matches!(self, Self::LongTerm)
    }
    
    /// Check if this state needs attention
    pub fn needs_attention(&self) -> bool {
        matches!(self, Self::Conflicted)
    }
    
    /// Check if transition to new state is valid for temporal processing
    /// This implements the temporal-memory logic for branch consolidation
    pub fn can_transition_to(&self, new_state: ConsolidationState) -> bool {
        use ConsolidationState::*;
        
        match (*self, new_state) {
            // From WorkingMemory (equivalent to old Active)
            (WorkingMemory, Consolidating) => true,
            (WorkingMemory, Conflicted) => true,
            (WorkingMemory, WorkingMemory) => true,
            (WorkingMemory, ShortTerm) => true, // Natural aging
            
            // From ShortTerm 
            (ShortTerm, Consolidating) => true,
            (ShortTerm, Conflicted) => true,
            (ShortTerm, ShortTerm) => true,
            (ShortTerm, WorkingMemory) => true, // Can reset
            
            // From Consolidating
            (Consolidating, LongTerm) => true, // Success
            (Consolidating, Conflicted) => true, // Failure
            (Consolidating, WorkingMemory) => true, // Cancel
            (Consolidating, ShortTerm) => true, // Cancel
            
            // From LongTerm (equivalent to old Consolidated)
            (LongTerm, WorkingMemory) => true, // Can reactivate
            (LongTerm, ShortTerm) => true, // Can reactivate
            (LongTerm, LongTerm) => true,
            
            // From Conflicted
            (Conflicted, WorkingMemory) => true, // After resolution
            (Conflicted, ShortTerm) => true, // After resolution  
            (Conflicted, Consolidating) => true, // Retry
            (Conflicted, Conflicted) => true,
            
            // All other transitions invalid
            _ => false,
        }
    }
}

impl std::fmt::Display for ConsolidationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkingMemory => write!(f, "WorkingMemory"),
            Self::ShortTerm => write!(f, "ShortTerm"),
            Self::Consolidating => write!(f, "Consolidating"),
            Self::LongTerm => write!(f, "LongTerm"),
            Self::Conflicted => write!(f, "Conflicted"),
        }
    }
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

    /// Set the consolidation state (for state transitions)
    pub fn set_consolidation_state(&mut self, state: ConsolidationState) {
        self.consolidation_state = state;
    }

    /// Update timestamp to current time
    pub fn update_timestamp(&mut self) {
        self.timestamp = Utc::now();
    }

    /// Calculate the age of this branch
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.timestamp
    }

    /// Set a custom timestamp
    pub fn set_timestamp(&mut self, timestamp: DateTime<Utc>) {
        self.timestamp = timestamp;
    }

    /// Create a new branch with explicit parent (primarily for testing)
    pub fn new_with_parent(id: impl Into<String>, parent: Option<impl Into<String>>) -> Self {
        Self {
            id: id.into(),
            parent: parent.map(Into::into),
            timestamp: Utc::now(),
            consolidation_state: ConsolidationState::WorkingMemory,
        }
    }
}
