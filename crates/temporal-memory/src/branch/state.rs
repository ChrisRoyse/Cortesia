//! Consolidation state management for memory branches

use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-export ConsolidationState from neuromorphic-core
pub use neuromorphic_core::ConsolidationState;

/// Manages state transitions with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    current: ConsolidationState,
    history: Vec<StateChange>,
}

/// Record of a state change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChange {
    pub from: ConsolidationState,
    pub to: ConsolidationState,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub reason: String,
}

/// Errors that can occur during state transitions
#[derive(Error, Debug)]
pub enum StateError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: ConsolidationState,
        to: ConsolidationState,
    },
    
    #[error("Cannot perform operation in {0:?} state")]
    InvalidOperation(ConsolidationState),
    
    #[error("State validation failed: {0}")]
    ValidationError(String),
}

impl StateTransition {
    /// Create new state manager
    pub fn new(initial: ConsolidationState) -> Self {
        Self {
            current: initial,
            history: Vec::new(),
        }
    }
    
    /// Get current state
    pub fn current(&self) -> ConsolidationState {
        self.current
    }
    
    /// Attempt to transition to new state
    pub fn transition_to(&mut self, 
                        new_state: ConsolidationState, 
                        reason: String) -> Result<(), StateError> {
        if !self.current.can_transition_to(new_state) {
            return Err(StateError::InvalidTransition {
                from: self.current,
                to: new_state,
            });
        }
        
        let change = StateChange {
            from: self.current,
            to: new_state,
            timestamp: chrono::Utc::now(),
            reason,
        };
        
        self.history.push(change);
        self.current = new_state;
        
        Ok(())
    }
    
    /// Get state history
    pub fn history(&self) -> &[StateChange] {
        &self.history
    }
    
    /// Check if in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self.current, ConsolidationState::LongTerm)
    }
    
    /// Check if can accept new allocations
    pub fn can_allocate(&self) -> bool {
        self.current.is_active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_transitions() {
        assert!(ConsolidationState::WorkingMemory.can_transition_to(ConsolidationState::Consolidating));
        assert!(ConsolidationState::Consolidating.can_transition_to(ConsolidationState::LongTerm));
        assert!(ConsolidationState::Conflicted.can_transition_to(ConsolidationState::WorkingMemory));
    }
    
    #[test]
    fn test_invalid_transitions() {
        assert!(!ConsolidationState::LongTerm.can_transition_to(ConsolidationState::Conflicted));
        assert!(!ConsolidationState::WorkingMemory.can_transition_to(ConsolidationState::LongTerm));
    }
    
    #[test]
    fn test_state_manager() {
        let mut sm = StateTransition::new(ConsolidationState::WorkingMemory);
        
        // Valid transition
        assert!(sm.transition_to(
            ConsolidationState::Consolidating, 
            "Starting consolidation".to_string()
        ).is_ok());
        
        assert_eq!(sm.current(), ConsolidationState::Consolidating);
        assert_eq!(sm.history().len(), 1);
        
        // Complete consolidation
        assert!(sm.transition_to(
            ConsolidationState::LongTerm,
            "Completed".to_string()
        ).is_ok());
        
        assert_eq!(sm.current(), ConsolidationState::LongTerm);
        assert!(sm.is_terminal());
    }
}