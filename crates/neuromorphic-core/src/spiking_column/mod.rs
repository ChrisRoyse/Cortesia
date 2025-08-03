//! Spiking cortical column implementation with TTFS dynamics

pub mod state;
pub mod activation; 
pub mod column;
pub mod inhibition;
pub mod grid;

pub use state::{ColumnState, AtomicState};
pub use activation::ActivationDynamics;
pub use column::SpikingCorticalColumn;
pub use inhibition::{LateralInhibitionNetwork, InhibitionConfig, CompetitionResult, InhibitionStats};
pub use grid::{CorticalGrid, GridConfig, GridPosition, GridStats};

use std::time::Duration;
use thiserror::Error;

pub type ColumnId = u32;
pub type SpikeTiming = Duration;
pub type InhibitoryWeight = f32;
pub type RefractoryPeriod = Duration;

/// Errors that can occur during column operations
#[derive(Error, Debug, Clone)]
pub enum ColumnError {
    #[error("Column already allocated")]
    AlreadyAllocated,

    #[error("Column in refractory period")]
    InRefractory,

    #[error("Invalid state transition from {0:?} to {1:?}")]
    InvalidTransition(ColumnState, ColumnState),

    #[error("Allocation blocked by lateral inhibition")]
    InhibitionBlocked,
}