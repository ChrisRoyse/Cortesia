//! Main spiking cortical column implementation

use super::{AtomicState, ColumnError, ColumnId, ColumnState};

/// Represents a spiking cortical column with TTFS encoding
#[derive(Debug)]
pub struct SpikingCorticalColumn {
    id: ColumnId,
    state: AtomicState,
    // Placeholder for future implementation
    _phantom: std::marker::PhantomData<()>,
}

impl SpikingCorticalColumn {
    /// Creates a new spiking cortical column
    pub fn new(id: ColumnId) -> Self {
        Self {
            id,
            state: AtomicState::new(ColumnState::Available),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the column ID
    pub fn id(&self) -> ColumnId {
        self.id
    }

    /// Returns the current state
    pub fn state(&self) -> ColumnState {
        self.state.load()
    }

    /// Try to transition to a new state
    pub fn try_transition(&self, new_state: ColumnState) -> Result<(), ColumnError> {
        let current = self.state.load();
        self.state
            .try_transition(new_state)
            .map_err(|_| ColumnError::InvalidTransition(current, new_state))
    }

    /// Try to activate the column
    pub fn activate(&self) -> Result<(), ColumnError> {
        self.try_transition(ColumnState::Activated)
    }

    /// Start competing for allocation
    pub fn start_competing(&self) -> Result<(), ColumnError> {
        self.try_transition(ColumnState::Competing)
    }

    /// Allocate the column
    pub fn allocate(&self) -> Result<(), ColumnError> {
        let current = self.state();
        if current == ColumnState::Allocated {
            return Err(ColumnError::AlreadyAllocated);
        }
        self.try_transition(ColumnState::Allocated)
    }

    /// Enter refractory period
    pub fn enter_refractory(&self) -> Result<(), ColumnError> {
        self.try_transition(ColumnState::Refractory)
    }

    /// Reset to available state
    pub fn reset(&self) -> Result<(), ColumnError> {
        let current = self.state();
        if current != ColumnState::Refractory && current != ColumnState::Available {
            return Err(ColumnError::InvalidTransition(
                current,
                ColumnState::Available,
            ));
        }
        self.try_transition(ColumnState::Available)
    }

    /// Check if column is available for allocation
    pub fn is_available(&self) -> bool {
        self.state() == ColumnState::Available
    }

    /// Check if column is allocated
    pub fn is_allocated(&self) -> bool {
        self.state() == ColumnState::Allocated
    }

    /// Check if column is in refractory period
    pub fn is_refractory(&self) -> bool {
        self.state() == ColumnState::Refractory
    }
}