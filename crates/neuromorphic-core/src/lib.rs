//! Neuromorphic core structures for CortexKG
//!
//! This crate provides spiking neural network primitives
//! with Time-to-First-Spike (TTFS) encoding for efficient
//! neuromorphic computing and allocation.

pub mod error;
pub mod neural_branch;
pub mod simd_backend;
pub mod spiking_column;
pub mod ttfs_concept;

// Re-export main types for convenience
pub use error::{NeuromorphicError, Result, ResultExt};
pub use neural_branch::{BranchId, NeuromorphicMemoryBranch};
pub use spiking_column::{ColumnError, ColumnId, ColumnState, SpikingCorticalColumn};
pub use ttfs_concept::{TTFSConcept, ConceptMetadata, SpikePattern, SpikeEvent, TTFSEncoder, EncodingConfig, EncodingError, ms_to_duration, duration_to_ms};

// Common types used across the neuromorphic system
use std::time::Duration;

/// Time-to-First-Spike timing representation
pub type SpikeTiming = Duration;

/// Inhibitory weight strength between columns
pub type InhibitoryWeight = f32;

/// Refractory period duration
pub type RefractoryPeriod = Duration;
