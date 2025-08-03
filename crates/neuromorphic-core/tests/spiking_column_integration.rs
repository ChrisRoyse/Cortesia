//! Integration tests for spiking cortical columns

use neuromorphic_core::{ColumnError, ColumnState, SpikingCorticalColumn};
use std::sync::Arc;
use std::thread;

#[test]
fn test_column_creation_and_basic_operations() {
    let column = SpikingCorticalColumn::new(1);
    
    assert_eq!(column.id(), 1);
    assert_eq!(column.state(), ColumnState::Available);
    assert!(column.is_available());
    assert!(!column.is_allocated());
    assert!(!column.is_refractory());
}

#[test]
fn test_column_activation_flow() {
    let column = SpikingCorticalColumn::new(2);
    
    // Activate the column
    assert!(column.activate().is_ok());
    assert_eq!(column.state(), ColumnState::Activated);
    
    // Start competing
    assert!(column.start_competing().is_ok());
    assert_eq!(column.state(), ColumnState::Competing);
    
    // Allocate
    assert!(column.allocate().is_ok());
    assert_eq!(column.state(), ColumnState::Allocated);
    assert!(column.is_allocated());
    
    // Try to allocate again - should fail
    assert!(matches!(
        column.allocate(),
        Err(ColumnError::AlreadyAllocated)
    ));
    
    // Enter refractory
    assert!(column.enter_refractory().is_ok());
    assert_eq!(column.state(), ColumnState::Refractory);
    assert!(column.is_refractory());
    
    // Reset to available
    assert!(column.reset().is_ok());
    assert_eq!(column.state(), ColumnState::Available);
    assert!(column.is_available());
}

#[test]
fn test_invalid_transitions_through_column_methods() {
    let column = SpikingCorticalColumn::new(3);
    
    // Cannot allocate directly from Available
    assert!(column.allocate().is_err());
    
    // Activate first
    assert!(column.activate().is_ok());
    
    // Cannot allocate from Activated
    assert!(column.allocate().is_err());
    
    // Must compete first
    assert!(column.start_competing().is_ok());
    
    // Now can allocate
    assert!(column.allocate().is_ok());
    
    // Cannot reset from Allocated (must go through Refractory)
    assert!(column.reset().is_err());
    
    // Must enter refractory first
    assert!(column.enter_refractory().is_ok());
    
    // Now can reset
    assert!(column.reset().is_ok());
}

#[test]
fn test_concurrent_column_activation() {
    let column = Arc::new(SpikingCorticalColumn::new(4));
    let mut handles = vec![];
    
    // Try to activate from multiple threads
    for _ in 0..10 {
        let column_clone = column.clone();
        handles.push(thread::spawn(move || column_clone.activate()));
    }
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All should succeed (activation can be strengthened when already activated)
    let successes = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(successes, 10);
    assert_eq!(column.state(), ColumnState::Activated);
}

#[test]
fn test_competition_winner_and_losers() {
    let columns: Vec<_> = (0..5).map(|i| SpikingCorticalColumn::new(i)).collect();
    
    // All columns activate
    for column in &columns {
        assert!(column.activate().is_ok());
    }
    
    // All start competing
    for column in &columns {
        assert!(column.start_competing().is_ok());
    }
    
    // First column wins (gets allocated)
    assert!(columns[0].allocate().is_ok());
    
    // Others lose (go to refractory)
    for column in &columns[1..] {
        assert!(column.enter_refractory().is_ok());
    }
    
    // Winner is allocated
    assert!(columns[0].is_allocated());
    
    // Losers are in refractory
    for column in &columns[1..] {
        assert!(column.is_refractory());
    }
    
    // Losers can reset
    for column in &columns[1..] {
        assert!(column.reset().is_ok());
        assert!(column.is_available());
    }
}

#[test]
fn test_multiple_allocation_cycles() {
    let column = SpikingCorticalColumn::new(5);
    
    // First cycle
    assert!(column.activate().is_ok());
    assert!(column.start_competing().is_ok());
    assert!(column.allocate().is_ok());
    assert!(column.enter_refractory().is_ok());
    assert!(column.reset().is_ok());
    
    // Second cycle - should work the same
    assert!(column.activate().is_ok());
    assert!(column.start_competing().is_ok());
    assert!(column.allocate().is_ok());
    assert!(column.enter_refractory().is_ok());
    assert!(column.reset().is_ok());
    
    // State should be back to Available
    assert_eq!(column.state(), ColumnState::Available);
}