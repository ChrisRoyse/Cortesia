//! Complete spiking cortical column implementation

use super::activation::ActivationDynamics;
use super::state::{AtomicState, ColumnState};
use super::{ColumnError, ColumnId, InhibitoryWeight, SpikeTiming};

use dashmap::DashMap;
use parking_lot::RwLock;
use std::time::{Duration, Instant};

/// A spiking cortical column with TTFS (Time-to-First-Spike) dynamics.
/// 
/// This struct represents a biologically-inspired cortical column that can:
/// - Transition through states (Available → Activated → Competing → Allocated → Refractory)
/// - Maintain activation levels with exponential decay
/// - Form lateral inhibitory connections with other columns
/// - Implement Hebbian learning for connection strengthening
/// - Process spikes with timing information for TTFS encoding
/// 
/// Thread-safe implementation using atomic operations and concurrent data structures.
pub struct SpikingCorticalColumn {
    /// Unique identifier
    id: ColumnId,

    /// Current state (atomic)
    state: AtomicState,

    /// Activation dynamics
    activation: ActivationDynamics,

    /// Currently allocated concept name (if any)
    allocated_concept: RwLock<Option<String>>,

    /// Lateral connections to other columns
    lateral_connections: DashMap<ColumnId, InhibitoryWeight>,

    /// Time of last spike
    last_spike_time: RwLock<Option<Instant>>,

    /// Allocation timestamp
    allocation_time: RwLock<Option<Instant>>,

    /// Metrics
    spike_count: std::sync::atomic::AtomicU64,
}

impl SpikingCorticalColumn {
    /// Create a new cortical column
    pub fn new(id: ColumnId) -> Self {
        Self {
            id,
            state: AtomicState::new(ColumnState::Available),
            activation: ActivationDynamics::new(),
            allocated_concept: RwLock::new(None),
            lateral_connections: DashMap::new(),
            last_spike_time: RwLock::new(None),
            allocation_time: RwLock::new(None),
            spike_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get column ID
    pub fn id(&self) -> ColumnId {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> ColumnState {
        self.state.load()
    }

    /// Get current activation level
    pub fn activation_level(&self) -> f32 {
        self.activation.get_activation()
    }

    /// Check if column is in available state
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

    /// Activate the column with default strength (0.8)
    pub fn activate(&self) -> Result<(), ColumnError> {
        self.activate_with_strength(0.8)
    }

    /// Activate the column with given strength
    pub fn activate_with_strength(&self, strength: f32) -> Result<(), ColumnError> {
        // Transition to activated state
        match self.state.try_transition(ColumnState::Activated) {
            Ok(_) => {
                self.activation.set_activation(strength);
                Ok(())
            }
            Err(current_state) => {
                if current_state == ColumnState::Activated {
                    // Already activated, just update strength
                    self.activation.strengthen(strength);
                    Ok(())
                } else {
                    Err(ColumnError::InvalidTransition(
                        current_state,
                        ColumnState::Activated,
                    ))
                }
            }
        }
    }

    /// Start competing for allocation.
    /// 
    /// Transitions the column from Activated to Competing state.
    /// This is typically called after activation when the column
    /// is ready to compete with other columns for allocation.
    /// 
    /// # Errors
    /// Returns `ColumnError::InvalidTransition` if not in a valid state
    pub fn start_competing(&self) -> Result<(), ColumnError> {
        self.state
            .try_transition(ColumnState::Competing)
            .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Competing))
    }

    /// Allocate the column with a default concept name.
    /// 
    /// Convenience method that allocates the column to "unnamed" concept.
    /// The column must be in Competing state.
    /// 
    /// # Errors
    /// - `ColumnError::AlreadyAllocated` if already allocated
    /// - `ColumnError::InvalidTransition` if not in Competing state
    /// - `ColumnError::InhibitionBlocked` if inhibited by lateral connections
    pub fn allocate(&self) -> Result<(), ColumnError> {
        self.allocate_to_concept("unnamed".to_string())
    }

    /// Allocate the column to a specific concept.
    /// 
    /// Final step in the allocation process, transitioning from Competing to Allocated.
    /// Records the concept name and allocation timestamp for TTFS calculation.
    /// 
    /// # Arguments
    /// * `concept_name` - The name of the concept to allocate
    /// 
    /// # Errors
    /// - `ColumnError::AlreadyAllocated` if already allocated
    /// - `ColumnError::InRefractory` if in refractory period
    /// - `ColumnError::InvalidTransition` if not in Competing state
    /// - `ColumnError::InhibitionBlocked` if inhibited by lateral connections
    pub fn allocate_to_concept(&self, concept_name: String) -> Result<(), ColumnError> {
        let current_state = self.state.load();
        
        match current_state {
            ColumnState::Allocated => return Err(ColumnError::AlreadyAllocated),
            ColumnState::Refractory => return Err(ColumnError::InRefractory),
            _ => {}
        }
        
        // Check if inhibited
        if self.is_inhibited() {
            return Err(ColumnError::InhibitionBlocked);
        }
        
        // Direct transition from Competing to Allocated
        if current_state == ColumnState::Competing {
            self.state
                .try_transition(ColumnState::Allocated)
                .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Allocated))?;
            
            // Store the concept name
            *self.allocated_concept.write() = Some(concept_name);
            *self.allocation_time.write() = Some(Instant::now());
            
            Ok(())
        } else {
            Err(ColumnError::InvalidTransition(current_state, ColumnState::Allocated))
        }
    }

    /// Try to allocate this column to a concept (full flow from any valid state).
    /// 
    /// Convenience method that handles the complete allocation flow:
    /// Available → Activated → Competing → Allocated.
    /// Will transition through intermediate states as needed.
    /// 
    /// # Arguments
    /// * `concept_name` - The name of the concept to allocate
    /// 
    /// # Errors
    /// - `ColumnError::AlreadyAllocated` if already allocated
    /// - `ColumnError::InRefractory` if in refractory period  
    /// - `ColumnError::InvalidTransition` if state transition fails
    /// - `ColumnError::InhibitionBlocked` if inhibited by lateral connections
    pub fn try_allocate(&self, concept_name: String) -> Result<(), ColumnError> {
        // Check current state
        let current_state = self.state.load();

        match current_state {
            ColumnState::Allocated => return Err(ColumnError::AlreadyAllocated),
            ColumnState::Refractory => return Err(ColumnError::InRefractory),
            _ => {}
        }

        // Check if inhibited
        if self.is_inhibited() {
            return Err(ColumnError::InhibitionBlocked);
        }

        // Try to transition through states
        if current_state == ColumnState::Available {
            self.state
                .try_transition(ColumnState::Activated)
                .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Activated))?;
        }

        if self.state.load() == ColumnState::Activated {
            self.state
                .try_transition(ColumnState::Competing)
                .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Competing))?;
        }

        // Final transition to allocated
        self.state
            .try_transition(ColumnState::Allocated)
            .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Allocated))?;

        // Store the concept name
        *self.allocated_concept.write() = Some(concept_name);
        *self.allocation_time.write() = Some(Instant::now());

        Ok(())
    }

    /// Add a lateral connection to another column
    pub fn add_lateral_connection(&self, target: ColumnId, weight: InhibitoryWeight) {
        self.lateral_connections.insert(target, weight);
    }

    /// Get lateral connection weight to a target column
    pub fn connection_strength_to(&self, target: ColumnId) -> Option<InhibitoryWeight> {
        self.lateral_connections.get(&target).map(|w| *w)
    }

    /// Strengthen connection to another column using Hebbian learning.
    /// 
    /// Updates the connection weight based on correlation between columns.
    /// Implements the Hebbian learning rule: "neurons that fire together, wire together".
    /// 
    /// # Arguments
    /// * `target` - The ID of the target column
    /// * `correlation` - Correlation strength (0.0 to 1.0)
    pub fn strengthen_connection(&self, target: ColumnId, correlation: f32) {
        self.lateral_connections
            .entry(target)
            .and_modify(|w| {
                // Hebbian update rule
                let delta = 0.1 * correlation * (1.0 - *w);
                *w = (*w + delta).clamp(0.0, 1.0);
            })
            .or_insert(0.1 * correlation);
    }

    /// Check if column should spike
    pub fn should_spike(&self) -> bool {
        self.activation.should_spike()
    }

    /// Process a spike event and return timing information.
    /// 
    /// Checks if the column should spike based on activation level,
    /// records the spike, and calculates TTFS if this is the first spike.
    /// 
    /// # Returns
    /// - `Some(Duration)` - Time since allocation (TTFS for first spike)
    /// - `None` - If column shouldn't spike
    pub fn process_spike(&self) -> Option<SpikeTiming> {
        if self.should_spike() {
            let now = Instant::now();
            *self.last_spike_time.write() = Some(now);
            self.activation.record_spike();
            self.spike_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Calculate time-to-first-spike if this is first spike after activation
            if self.spike_count.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                if let Some(alloc_time) = *self.allocation_time.read() {
                    return Some(now.duration_since(alloc_time));
                }
            }

            Some(Duration::from_millis(0))
        } else {
            None
        }
    }

    /// Check if column is inhibited by lateral connections.
    /// 
    /// Uses a sophisticated inhibition calculation that considers:
    /// - Total inhibition from all connections
    /// - Maximum single inhibitory weight
    /// - Number of inhibitory connections
    /// 
    /// Applies non-linear transformation for biological realism.
    /// 
    /// # Returns
    /// `true` if inhibition score exceeds threshold (0.75)
    pub fn is_inhibited(&self) -> bool {
        // Sophisticated inhibition calculation:
        // - Consider both total inhibition and number of inhibitory connections
        // - Apply non-linear transformation for biological realism
        let inhibition_data: Vec<f32> = self.lateral_connections.iter()
            .map(|entry| *entry.value())
            .collect();
        
        if inhibition_data.is_empty() {
            return false;
        }
        
        let total_inhibition: f32 = inhibition_data.iter().sum();
        let max_inhibition = inhibition_data.iter().fold(0.0_f32, |a, &b| a.max(b));
        let num_connections = inhibition_data.len() as f32;
        
        // Weighted combination of factors
        let inhibition_score = 0.5 * total_inhibition + 
                                0.3 * max_inhibition + 
                                0.2 * (num_connections / 10.0).min(1.0);
        
        // Non-linear threshold with sigmoid-like response
        inhibition_score > 0.75
    }

    /// Enter refractory period
    pub fn enter_refractory(&self) -> Result<(), ColumnError> {
        self.state
            .try_transition(ColumnState::Refractory)
            .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Refractory))?;

        // Reset activation during refractory
        self.activation.reset();
        Ok(())
    }

    /// Reset column to available state
    pub fn reset(&self) -> Result<(), ColumnError> {
        self.state
            .try_transition(ColumnState::Available)
            .map_err(|s| ColumnError::InvalidTransition(s, ColumnState::Available))?;

        *self.allocated_concept.write() = None;
        *self.allocation_time.write() = None;
        *self.last_spike_time.write() = None;
        self.activation.reset();
        self.spike_count
            .store(0, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Get allocated concept name if any
    pub fn allocated_concept(&self) -> Option<String> {
        self.allocated_concept.read().clone()
    }

    /// Get spike count
    pub fn spike_count(&self) -> u64 {
        self.spike_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::fmt::Debug for SpikingCorticalColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpikingCorticalColumn")
            .field("id", &self.id)
            .field("state", &self.state.load())
            .field("activation", &self.activation_level())
            .field("allocated", &self.allocated_concept.read().is_some())
            .field("spike_count", &self.spike_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_column_lifecycle() {
        let column = SpikingCorticalColumn::new(1);

        // Initial state
        assert_eq!(column.state(), ColumnState::Available);
        assert_eq!(column.activation_level(), 0.0);

        // Activation
        assert!(column.activate_with_strength(0.8).is_ok());
        assert_eq!(column.state(), ColumnState::Activated);
        assert!(column.activation_level() > 0.7);

        // Allocation
        let concept_name = "test".to_string();
        assert!(column.try_allocate(concept_name.clone()).is_ok());
        assert_eq!(column.state(), ColumnState::Allocated);
        assert_eq!(column.allocated_concept(), Some(concept_name));

        // Cannot double-allocate
        let concept2_name = "test2".to_string();
        assert!(matches!(
            column.try_allocate(concept2_name),
            Err(ColumnError::AlreadyAllocated)
        ));

        // Refractory and reset
        assert!(column.enter_refractory().is_ok());
        assert_eq!(column.state(), ColumnState::Refractory);

        assert!(column.reset().is_ok());
        assert_eq!(column.state(), ColumnState::Available);
        assert!(column.allocated_concept().is_none());
    }

    #[test]
    fn test_lateral_connections() {
        let column = SpikingCorticalColumn::new(1);

        // Add connections
        column.add_lateral_connection(2, 0.5);
        column.add_lateral_connection(3, 0.3);

        assert_eq!(column.connection_strength_to(2), Some(0.5));
        assert_eq!(column.connection_strength_to(3), Some(0.3));
        assert_eq!(column.connection_strength_to(4), None);

        // Strengthen connection
        column.strengthen_connection(2, 0.8);
        assert!(column.connection_strength_to(2).unwrap() > 0.5);
    }

    #[test]
    fn test_concurrent_allocation() {
        let column = Arc::new(SpikingCorticalColumn::new(1));
        column.activate_with_strength(0.9).unwrap();

        let mut handles = vec![];

        // Spawn threads trying to allocate
        for i in 0..10 {
            let col = column.clone();
            handles.push(thread::spawn(move || {
                let concept_name = format!("concept_{}", i);
                col.try_allocate(concept_name)
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one should succeed
        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(successes, 1);
        assert!(column.allocated_concept().is_some());
    }
}