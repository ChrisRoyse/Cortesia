# Task 1.1: Basic Column State Machine

**Duration**: 2 hours  
**Complexity**: Low  
**Dependencies**: None (first task)  
**AI Assistant Suitability**: High - Clear state machine implementation  

## Objective

Implement the fundamental state machine for a cortical column with atomic state transitions and proper memory management.

## Specification

Create a `CorticalColumn` struct that manages five distinct states with thread-safe transitions:

**States**:
- `Available` (0): Ready for allocation
- `Activated` (1): Currently processing input
- `Competing` (2): In lateral inhibition competition  
- `Allocated` (3): Successfully allocated to a concept
- `Refractory` (4): Temporarily unavailable after firing

**Requirements**:
- Atomic state transitions using `AtomicU8`
- Compare-and-swap operations for thread safety
- Invalid transition prevention
- State transition logging for debugging

## Implementation Guide

### Step 1: Create the Core State Enum

```rust
// src/column_state.rs
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnState {
    Available = 0,
    Activated = 1,
    Competing = 2,
    Allocated = 3,
    Refractory = 4,
}

impl ColumnState {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Available),
            1 => Some(Self::Activated),
            2 => Some(Self::Competing),
            3 => Some(Self::Allocated),
            4 => Some(Self::Refractory),
            _ => None,
        }
    }
    
    pub fn is_valid_transition(&self, to: Self) -> bool {
        use ColumnState::*;
        matches!(
            (self, to),
            (Available, Activated) | 
            (Activated, Competing) |
            (Competing, Allocated) |
            (Competing, Available) |
            (Allocated, Refractory) |
            (Refractory, Available)
        )
    }
}
```

### Step 2: Implement Atomic State Wrapper

```rust
// src/atomic_state.rs
use std::sync::atomic::{AtomicU8, Ordering};

pub struct AtomicColumnState {
    state: AtomicU8,
}

impl AtomicColumnState {
    pub fn new(initial: ColumnState) -> Self {
        Self {
            state: AtomicU8::new(initial as u8),
        }
    }
    
    pub fn load(&self) -> ColumnState {
        let value = self.state.load(Ordering::Acquire);
        ColumnState::from_u8(value).expect("Invalid state in atomic storage")
    }
    
    pub fn try_transition(&self, from: ColumnState, to: ColumnState) -> Result<ColumnState, StateTransitionError> {
        if !from.is_valid_transition(to) {
            return Err(StateTransitionError::InvalidTransition { from, to });
        }
        
        match self.state.compare_exchange(
            from as u8,
            to as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(to),
            Err(actual) => {
                let actual_state = ColumnState::from_u8(actual)
                    .expect("Invalid state from compare_exchange");
                Err(StateTransitionError::StateMismatch { 
                    expected: from, 
                    actual: actual_state 
                })
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition { from: ColumnState, to: ColumnState },
    
    #[error("State mismatch: expected {expected:?}, found {actual:?}")]
    StateMismatch { expected: ColumnState, actual: ColumnState },
    
    #[error("Concurrent modification detected during transition")]
    ConcurrentModification,
    
    #[error("Transition timeout after {timeout_ms}ms")]
    TransitionTimeout { timeout_ms: u64 },
    
    #[error("Column {column_id} is in invalid state")]
    InvalidColumnState { column_id: u32 },
}
```

### Step 3: Create the CorticalColumn Struct

```rust
// src/cortical_column.rs
use crate::{AtomicColumnState, ColumnState, StateTransitionError};
use std::time::{SystemTime, Duration};
use parking_lot::RwLock;

pub type ColumnId = u32;

pub struct CorticalColumn {
    id: ColumnId,
    state: AtomicColumnState,
    created_at: SystemTime,
    last_transition: RwLock<SystemTime>,
    transition_count: AtomicU64,
}

impl CorticalColumn {
    pub fn new(id: ColumnId) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            state: AtomicColumnState::new(ColumnState::Available),
            created_at: now,
            last_transition: RwLock::new(now),
            transition_count: AtomicU64::new(0),
        }
    }
    
    pub fn id(&self) -> ColumnId {
        self.id
    }
    
    pub fn current_state(&self) -> ColumnState {
        self.state.load()
    }
    
    pub fn try_activate(&self) -> Result<(), StateTransitionError> {
        self.try_transition_to(ColumnState::Activated)
    }
    
    pub fn try_compete(&self) -> Result<(), StateTransitionError> {
        self.try_transition_to(ColumnState::Competing)
    }
    
    pub fn try_allocate(&self) -> Result<(), StateTransitionError> {
        self.try_transition_to(ColumnState::Allocated)
    }
    
    pub fn try_enter_refractory(&self) -> Result<(), StateTransitionError> {
        self.try_transition_to(ColumnState::Refractory)
    }
    
    pub fn try_reset(&self) -> Result<(), StateTransitionError> {
        self.try_transition_to(ColumnState::Available)
    }
    
    /// Try transition with retry mechanism for concurrent modifications
    pub fn try_transition_with_retry(&self, target: ColumnState, max_retries: u32) -> Result<(), StateTransitionError> {
        for attempt in 0..max_retries {
            match self.try_transition_to(target) {
                Ok(()) => return Ok(()),
                Err(StateTransitionError::ConcurrentModification) => {
                    // Brief exponential backoff
                    let backoff_ns = 100 * (1 << attempt.min(10));
                    std::thread::sleep(Duration::from_nanos(backoff_ns));
                    continue;
                },
                Err(e) => return Err(e), // Don't retry other errors
            }
        }
        Err(StateTransitionError::ConcurrentModification)
    }
    
    fn try_transition_to(&self, target: ColumnState) -> Result<(), StateTransitionError> {
        let current = self.current_state();
        self.state.try_transition(current, target)?;
        
        // Update metadata
        *self.last_transition.write() = SystemTime::now();
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    pub fn time_since_last_transition(&self) -> Duration {
        SystemTime::now()
            .duration_since(*self.last_transition.read())
            .unwrap_or(Duration::ZERO)
    }
    
    pub fn transition_count(&self) -> u64 {
        self.transition_count.load(Ordering::Relaxed)
    }
}
```

## AI-Executable Test Suite

Create this complete test file to verify your implementation:

```rust
// tests/column_state_test.rs
use llmkg::{CorticalColumn, ColumnState, StateTransitionError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_column_creation() {
    let column = CorticalColumn::new(42);
    
    assert_eq!(column.id(), 42);
    assert_eq!(column.current_state(), ColumnState::Available);
    assert_eq!(column.transition_count(), 0);
}

#[test]
fn test_valid_state_transitions() {
    let column = CorticalColumn::new(1);
    
    // Available -> Activated
    assert!(column.try_activate().is_ok());
    assert_eq!(column.current_state(), ColumnState::Activated);
    assert_eq!(column.transition_count(), 1);
    
    // Activated -> Competing
    assert!(column.try_compete().is_ok());
    assert_eq!(column.current_state(), ColumnState::Competing);
    
    // Competing -> Allocated
    assert!(column.try_allocate().is_ok());
    assert_eq!(column.current_state(), ColumnState::Allocated);
    
    // Allocated -> Refractory
    assert!(column.try_enter_refractory().is_ok());
    assert_eq!(column.current_state(), ColumnState::Refractory);
    
    // Refractory -> Available
    assert!(column.try_reset().is_ok());
    assert_eq!(column.current_state(), ColumnState::Available);
    
    assert_eq!(column.transition_count(), 5);
}

#[test]
fn test_invalid_state_transitions() {
    let column = CorticalColumn::new(1);
    
    // Available -> Allocated (skipping intermediate states)
    assert!(matches!(
        column.try_allocate(),
        Err(StateTransitionError::InvalidTransition { .. })
    ));
    
    // Available -> Refractory (invalid transition)
    assert!(matches!(
        column.try_enter_refractory(),
        Err(StateTransitionError::InvalidTransition { .. })
    ));
    
    // State should remain Available
    assert_eq!(column.current_state(), ColumnState::Available);
    assert_eq!(column.transition_count(), 0);
}

#[test]
fn test_concurrent_state_transitions() {
    let column = Arc::new(CorticalColumn::new(1));
    let mut handles = vec![];
    
    // Spawn 10 threads trying to activate simultaneously
    for _ in 0..10 {
        let col = column.clone();
        handles.push(thread::spawn(move || {
            col.try_activate()
        }));
    }
    
    // Collect results
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Exactly one should succeed
    let successes = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(successes, 1);
    
    // State should be Activated
    assert_eq!(column.current_state(), ColumnState::Activated);
    assert_eq!(column.transition_count(), 1);
}

#[test]
fn test_time_tracking() {
    let column = CorticalColumn::new(1);
    
    thread::sleep(Duration::from_millis(10));
    assert!(column.time_since_last_transition() >= Duration::from_millis(10));
    
    // Transition should update time
    column.try_activate().unwrap();
    assert!(column.time_since_last_transition() < Duration::from_millis(5));
}

#[test]
fn test_state_transition_validation() {
    use llmkg::ColumnState;
    
    // Test valid transitions
    assert!(ColumnState::Available.is_valid_transition(ColumnState::Activated));
    assert!(ColumnState::Activated.is_valid_transition(ColumnState::Competing));
    assert!(ColumnState::Competing.is_valid_transition(ColumnState::Allocated));
    assert!(ColumnState::Competing.is_valid_transition(ColumnState::Available));
    assert!(ColumnState::Allocated.is_valid_transition(ColumnState::Refractory));
    assert!(ColumnState::Refractory.is_valid_transition(ColumnState::Available));
    
    // Test invalid transitions
    assert!(!ColumnState::Available.is_valid_transition(ColumnState::Allocated));
    assert!(!ColumnState::Available.is_valid_transition(ColumnState::Refractory));
    assert!(!ColumnState::Activated.is_valid_transition(ColumnState::Refractory));
    assert!(!ColumnState::Allocated.is_valid_transition(ColumnState::Available));
}

#[test]
fn test_stress_concurrent_transitions() {
    let column = Arc::new(CorticalColumn::new(1));
    let mut handles = vec![];
    
    // 100 threads trying different transitions
    for i in 0..100 {
        let col = column.clone();
        handles.push(thread::spawn(move || {
            match i % 4 {
                0 => col.try_activate(),
                1 => col.try_compete(),
                2 => col.try_allocate(),
                _ => col.try_reset(),
            }
        }));
    }
    
    // All should complete without panic
    for handle in handles {
        let _ = handle.join().unwrap();
    }
    
    // Column should be in a valid state
    let final_state = column.current_state();
    assert!(matches!(
        final_state, 
        ColumnState::Available | 
        ColumnState::Activated | 
        ColumnState::Competing | 
        ColumnState::Allocated | 
        ColumnState::Refractory
    ));
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: Run `cargo test column_state_test` - must be 8/8 passing
2. **No clippy warnings**: Run `cargo clippy` - must be 0 warnings  
3. **Memory safety**: No `unsafe` code blocks
4. **Performance**: State transitions < 10ns (measured with criterion)
5. **Thread safety**: Concurrent test passes 100 times in a row

## Verification Commands

```bash
# Run tests
cargo test column_state_test --release

# Check performance  
cargo bench --bench state_transition_bench

# Memory verification
cargo test test_stress_concurrent_transitions --release -- --ignored

# Code quality
cargo clippy -- -D warnings
```

## Files to Create

1. `src/column_state.rs`
2. `src/atomic_state.rs` 
3. `src/cortical_column.rs`
4. `tests/column_state_test.rs`
5. Update `src/lib.rs` with exports

## Expected Completion Time

2 hours for an AI assistant with:
- 30 minutes: Basic enum and atomic wrapper
- 45 minutes: CorticalColumn implementation
- 30 minutes: Test implementation and debugging
- 15 minutes: Performance verification and documentation

## Next Task

Task 1.2: Atomic State Transitions (depends on this task being complete)