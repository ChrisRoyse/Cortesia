//! Consolidation state management for memory branches

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// States a memory branch can be in during its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationState {
    /// Branch is actively receiving new allocations
    Active,
    
    /// Branch is in the process of consolidating
    Consolidating,
    
    /// Branch has been successfully consolidated
    Consolidated,
    
    /// Branch has conflicts that need resolution
    Conflicted,
}

impl ConsolidationState {
    /// Check if transition to new state is valid
    pub fn can_transition_to(&self, new_state: ConsolidationState) -> bool {
        use ConsolidationState::*;
        
        match (*self, new_state) {
            // From Active
            (Active, Consolidating) => true,
            (Active, Conflicted) => true,
            (Active, Active) => true,
            
            // From Consolidating
            (Consolidating, Consolidated) => true,
            (Consolidating, Conflicted) => true,
            (Consolidating, Active) => true, // Can cancel
            
            // From Consolidated
            (Consolidated, Active) => true, // Can reactivate
            (Consolidated, Consolidated) => true,
            
            // From Conflicted
            (Conflicted, Active) => true, // After resolution
            (Conflicted, Consolidating) => true, // Retry
            (Conflicted, Conflicted) => true,
            
            // All other transitions invalid
            _ => false,
        }
    }
    
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Active => "Branch is actively receiving allocations",
            Self::Consolidating => "Branch is being merged with parent",
            Self::Consolidated => "Branch has been successfully merged",
            Self::Conflicted => "Branch has conflicts requiring resolution",
        }
    }
}

impl fmt::Display for ConsolidationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Consolidating => write!(f, "Consolidating"),
            Self::Consolidated => write!(f, "Consolidated"),
            Self::Conflicted => write!(f, "Conflicted"),
        }
    }
}

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
    #[error("Invalid transition from {from} to {to}")]
    InvalidTransition {
        from: ConsolidationState,
        to: ConsolidationState,
    },
    
    #[error("Cannot perform operation in {0} state")]
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
        matches!(self.current, ConsolidationState::Consolidated)
    }
    
    /// Check if can accept new allocations
    pub fn can_allocate(&self) -> bool {
        matches!(self.current, ConsolidationState::Active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_transitions() {
        assert!(ConsolidationState::Active.can_transition_to(ConsolidationState::Consolidating));
        assert!(ConsolidationState::Consolidating.can_transition_to(ConsolidationState::Consolidated));
        assert!(ConsolidationState::Conflicted.can_transition_to(ConsolidationState::Active));
    }
    
    #[test]
    fn test_invalid_transitions() {
        assert!(!ConsolidationState::Consolidated.can_transition_to(ConsolidationState::Conflicted));
        assert!(!ConsolidationState::Active.can_transition_to(ConsolidationState::Consolidated));
    }
    
    #[test]
    fn test_state_manager() {
        let mut sm = StateTransition::new(ConsolidationState::Active);
        
        // Valid transition
        assert!(sm.transition_to(
            ConsolidationState::Consolidating, 
            "Starting consolidation".to_string()
        ).is_ok());
        
        assert_eq!(sm.current(), ConsolidationState::Consolidating);
        assert_eq!(sm.history().len(), 1);
        
        // Invalid transition
        assert!(sm.transition_to(
            ConsolidationState::Conflicted,
            "Invalid".to_string()
        ).is_ok()); // This should actually be valid according to the state machine
        
        // Test invalid transition
        let mut sm2 = StateTransition::new(ConsolidationState::Active);
        assert!(sm2.transition_to(
            ConsolidationState::Consolidated,
            "Invalid direct transition".to_string()
        ).is_err());
    }
}