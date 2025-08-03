//! State machine for spiking cortical columns

use std::sync::atomic::{AtomicU8, Ordering};

/// States of a cortical column in the spiking neural network
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnState {
    /// Column is available for allocation
    Available = 0,
    /// Column has been activated by input
    Activated = 1,
    /// Column is competing via lateral inhibition
    Competing = 2,
    /// Column has been allocated to a concept
    Allocated = 3,
    /// Column is in refractory period
    Refractory = 4,
}

impl ColumnState {
    /// Convert from u8 representation
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Available,
            1 => Self::Activated,
            2 => Self::Competing,
            3 => Self::Allocated,
            4 => Self::Refractory,
            _ => unreachable!("Invalid column state: {}", value),
        }
    }

    /// Check if transition to new state is valid
    pub fn can_transition_to(&self, new_state: ColumnState) -> bool {
        use ColumnState::*;
        match (*self, new_state) {
            // From Available
            (Available, Activated) => true,
            (Available, Available) => true,

            // From Activated
            (Activated, Competing) => true,
            (Activated, Available) => true,

            // From Competing
            (Competing, Allocated) => true,
            (Competing, Available) => true,
            (Competing, Refractory) => true,

            // From Allocated
            (Allocated, Refractory) => true,
            (Allocated, Allocated) => true,

            // From Refractory
            (Refractory, Available) => true,
            (Refractory, Refractory) => true,

            // All other transitions invalid
            _ => false,
        }
    }
}

/// Thread-safe atomic state for cortical columns
#[derive(Debug)]
pub struct AtomicState(AtomicU8);

impl AtomicState {
    /// Create new atomic state
    pub fn new(state: ColumnState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    /// Load current state
    pub fn load(&self) -> ColumnState {
        ColumnState::from_u8(self.0.load(Ordering::Acquire))
    }

    /// Store new state
    pub fn store(&self, state: ColumnState) {
        self.0.store(state as u8, Ordering::Release);
    }

    /// Atomic compare and exchange
    pub fn compare_exchange(
        &self,
        current: ColumnState,
        new: ColumnState,
    ) -> Result<ColumnState, ColumnState> {
        // Validate transition
        if !current.can_transition_to(new) {
            return Err(self.load());
        }

        match self.0.compare_exchange(
            current as u8,
            new as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(v) => Ok(ColumnState::from_u8(v)),
            Err(v) => Err(ColumnState::from_u8(v)),
        }
    }

    /// Try to transition to new state
    pub fn try_transition(&self, new_state: ColumnState) -> Result<(), ColumnState> {
        let current = self.load();
        self.compare_exchange(current, new_state)?;
        Ok(())
    }
}