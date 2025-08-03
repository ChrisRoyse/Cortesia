//! Tests for spiking column state machine
//!
//! Following TDD principles - tests written first

#[cfg(test)]
mod state_tests {
    use super::super::state::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_state_creation() {
        // Test that we can create states
        let state = ColumnState::Available;
        assert_eq!(state as u8, 0);
        
        let state = ColumnState::Activated;
        assert_eq!(state as u8, 1);
        
        let state = ColumnState::Competing;
        assert_eq!(state as u8, 2);
        
        let state = ColumnState::Allocated;
        assert_eq!(state as u8, 3);
        
        let state = ColumnState::Refractory;
        assert_eq!(state as u8, 4);
    }

    #[test]
    fn test_from_u8_conversion() {
        assert_eq!(ColumnState::from_u8(0), ColumnState::Available);
        assert_eq!(ColumnState::from_u8(1), ColumnState::Activated);
        assert_eq!(ColumnState::from_u8(2), ColumnState::Competing);
        assert_eq!(ColumnState::from_u8(3), ColumnState::Allocated);
        assert_eq!(ColumnState::from_u8(4), ColumnState::Refractory);
    }

    #[test]
    fn test_valid_transitions() {
        // Available transitions
        assert!(ColumnState::Available.can_transition_to(ColumnState::Activated));
        assert!(ColumnState::Available.can_transition_to(ColumnState::Available));
        
        // Activated transitions
        assert!(ColumnState::Activated.can_transition_to(ColumnState::Competing));
        assert!(ColumnState::Activated.can_transition_to(ColumnState::Available));
        
        // Competing transitions
        assert!(ColumnState::Competing.can_transition_to(ColumnState::Allocated));
        assert!(ColumnState::Competing.can_transition_to(ColumnState::Available));
        assert!(ColumnState::Competing.can_transition_to(ColumnState::Refractory));
        
        // Allocated transitions
        assert!(ColumnState::Allocated.can_transition_to(ColumnState::Refractory));
        assert!(ColumnState::Allocated.can_transition_to(ColumnState::Allocated));
        
        // Refractory transitions
        assert!(ColumnState::Refractory.can_transition_to(ColumnState::Available));
        assert!(ColumnState::Refractory.can_transition_to(ColumnState::Refractory));
    }

    #[test]
    fn test_invalid_transitions() {
        // Available cannot go directly to Allocated
        assert!(!ColumnState::Available.can_transition_to(ColumnState::Allocated));
        assert!(!ColumnState::Available.can_transition_to(ColumnState::Refractory));
        assert!(!ColumnState::Available.can_transition_to(ColumnState::Competing));
        
        // Activated cannot go directly to Allocated
        assert!(!ColumnState::Activated.can_transition_to(ColumnState::Allocated));
        assert!(!ColumnState::Activated.can_transition_to(ColumnState::Refractory));
        
        // Competing cannot go to Activated
        assert!(!ColumnState::Competing.can_transition_to(ColumnState::Activated));
        
        // Allocated cannot go backwards
        assert!(!ColumnState::Allocated.can_transition_to(ColumnState::Available));
        assert!(!ColumnState::Allocated.can_transition_to(ColumnState::Activated));
        assert!(!ColumnState::Allocated.can_transition_to(ColumnState::Competing));
        
        // Refractory cannot go to intermediate states
        assert!(!ColumnState::Refractory.can_transition_to(ColumnState::Activated));
        assert!(!ColumnState::Refractory.can_transition_to(ColumnState::Competing));
        assert!(!ColumnState::Refractory.can_transition_to(ColumnState::Allocated));
    }

    #[test]
    fn test_atomic_state_creation() {
        let state = AtomicState::new(ColumnState::Available);
        assert_eq!(state.load(), ColumnState::Available);
    }

    #[test]
    fn test_atomic_state_store_load() {
        let state = AtomicState::new(ColumnState::Available);
        
        state.store(ColumnState::Activated);
        assert_eq!(state.load(), ColumnState::Activated);
        
        state.store(ColumnState::Competing);
        assert_eq!(state.load(), ColumnState::Competing);
    }

    #[test]
    fn test_atomic_compare_exchange_success() {
        let state = AtomicState::new(ColumnState::Available);
        
        let result = state.compare_exchange(
            ColumnState::Available,
            ColumnState::Activated
        );
        assert!(result.is_ok());
        assert_eq!(state.load(), ColumnState::Activated);
    }

    #[test]
    fn test_atomic_compare_exchange_wrong_current() {
        let state = AtomicState::new(ColumnState::Available);
        
        // Try to transition from wrong current state
        let result = state.compare_exchange(
            ColumnState::Activated, // Wrong current state
            ColumnState::Competing
        );
        assert!(result.is_err());
        assert_eq!(state.load(), ColumnState::Available); // State unchanged
    }

    #[test]
    fn test_atomic_compare_exchange_invalid_transition() {
        let state = AtomicState::new(ColumnState::Available);
        
        // Try invalid transition
        let result = state.compare_exchange(
            ColumnState::Available,
            ColumnState::Allocated // Invalid direct transition
        );
        assert!(result.is_err());
        assert_eq!(state.load(), ColumnState::Available);
    }

    #[test]
    fn test_try_transition_valid() {
        let state = AtomicState::new(ColumnState::Available);
        
        // Valid transition
        assert!(state.try_transition(ColumnState::Activated).is_ok());
        assert_eq!(state.load(), ColumnState::Activated);
        
        // Another valid transition
        assert!(state.try_transition(ColumnState::Competing).is_ok());
        assert_eq!(state.load(), ColumnState::Competing);
        
        // And another
        assert!(state.try_transition(ColumnState::Allocated).is_ok());
        assert_eq!(state.load(), ColumnState::Allocated);
    }

    #[test]
    fn test_try_transition_invalid() {
        let state = AtomicState::new(ColumnState::Activated);
        
        // Invalid transition
        assert!(state.try_transition(ColumnState::Allocated).is_err());
        assert_eq!(state.load(), ColumnState::Activated); // State unchanged
    }

    #[test]
    fn test_concurrent_state_access() {
        let state = Arc::new(AtomicState::new(ColumnState::Available));
        let mut handles = vec![];
        
        // Spawn 10 threads trying to activate
        for _ in 0..10 {
            let state_clone = state.clone();
            handles.push(thread::spawn(move || {
                state_clone.compare_exchange(
                    ColumnState::Available,
                    ColumnState::Activated
                )
            }));
        }
        
        let results: Vec<_> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // Exactly one should succeed
        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(successes, 1);
        assert_eq!(state.load(), ColumnState::Activated);
    }

    #[test]
    fn test_state_machine_completeness() {
        // Verify all states are reachable
        let state = AtomicState::new(ColumnState::Available);
        
        // Path to Allocated
        assert!(state.try_transition(ColumnState::Activated).is_ok());
        assert!(state.try_transition(ColumnState::Competing).is_ok());
        assert!(state.try_transition(ColumnState::Allocated).is_ok());
        
        // Path to Refractory and back
        assert!(state.try_transition(ColumnState::Refractory).is_ok());
        assert!(state.try_transition(ColumnState::Available).is_ok());
    }

    #[test]
    fn test_refractory_cycle() {
        let state = AtomicState::new(ColumnState::Available);
        
        // Full cycle through allocation
        assert!(state.try_transition(ColumnState::Activated).is_ok());
        assert!(state.try_transition(ColumnState::Competing).is_ok());
        assert!(state.try_transition(ColumnState::Allocated).is_ok());
        assert!(state.try_transition(ColumnState::Refractory).is_ok());
        assert!(state.try_transition(ColumnState::Available).is_ok());
        
        // Back to start, ready for another cycle
        assert_eq!(state.load(), ColumnState::Available);
    }

    #[test]
    fn test_competing_to_refractory_path() {
        // Test that losing columns can go to refractory
        let state = AtomicState::new(ColumnState::Available);
        
        assert!(state.try_transition(ColumnState::Activated).is_ok());
        assert!(state.try_transition(ColumnState::Competing).is_ok());
        
        // Lost competition, go to refractory
        assert!(state.try_transition(ColumnState::Refractory).is_ok());
        assert!(state.try_transition(ColumnState::Available).is_ok());
    }
}