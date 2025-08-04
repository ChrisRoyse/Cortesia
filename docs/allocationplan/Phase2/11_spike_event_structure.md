# Task 11: Spike Event Structure

## Metadata
- **Micro-Phase**: 2.11
- **Duration**: 15 minutes
- **Dependencies**: None
- **Output**: `src/ttfs_encoding/spike_event.rs`

## Description
Create the fundamental SpikeEvent structure that represents individual neural spike events with nanosecond precision timing for biological accuracy.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_spike_event_creation() {
        let spike = SpikeEvent::new(
            NeuronId(42), 
            Duration::from_micros(500), 
            0.8
        );
        
        assert_eq!(spike.neuron_id, NeuronId(42));
        assert_eq!(spike.timing, Duration::from_micros(500));
        assert_eq!(spike.amplitude, 0.8);
        assert_eq!(spike.refractory_state, RefractoryState::Ready);
    }
    
    #[test]
    fn test_amplitude_clamping() {
        let spike1 = SpikeEvent::new(NeuronId(1), Duration::from_nanos(1000), 1.5);
        assert_eq!(spike1.amplitude, 1.0); // Clamped to max
        
        let spike2 = SpikeEvent::new(NeuronId(2), Duration::from_nanos(2000), -0.5);
        assert_eq!(spike2.amplitude, 0.0); // Clamped to min
    }
    
    #[test]
    fn test_biological_validation() {
        let valid_spike = SpikeEvent::new(
            NeuronId(1), 
            Duration::from_millis(10), 
            0.7
        );
        assert!(valid_spike.is_biologically_valid());
        
        let invalid_spike = SpikeEvent::new(
            NeuronId(2), 
            Duration::from_secs(1), // Too long
            0.5
        );
        assert!(!invalid_spike.is_biologically_valid());
    }
}
```

## Implementation
```rust
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Unique identifier for neurons in the spiking network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronId(pub usize);

/// Refractory state of a neuron after spiking
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RefractoryState {
    /// Neuron is ready to spike
    Ready,
    /// Neuron is in absolute refractory period (cannot spike)
    Absolute,
    /// Neuron is in relative refractory period (reduced likelihood)
    Relative,
}

/// Individual spike event with biological timing precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Unique identifier of the neuron that spiked
    pub neuron_id: NeuronId,
    
    /// Precise timing of the spike (nanosecond resolution)
    pub timing: Duration,
    
    /// Spike amplitude (0.0-1.0, represents action potential strength)
    pub amplitude: f32,
    
    /// Current refractory state of the neuron
    pub refractory_state: RefractoryState,
}

impl SpikeEvent {
    /// Create a new spike event
    pub fn new(neuron_id: NeuronId, timing: Duration, amplitude: f32) -> Self {
        Self {
            neuron_id,
            timing,
            amplitude: amplitude.clamp(0.0, 1.0),
            refractory_state: RefractoryState::Ready,
        }
    }
    
    /// Check if this spike is within biological timing constraints
    pub fn is_biologically_valid(&self) -> bool {
        // Spikes should occur within reasonable timeframes (< 100ms)
        self.timing < Duration::from_millis(100) && 
        self.amplitude > 0.0 && 
        self.amplitude <= 1.0
    }
    
    /// Get timing in nanoseconds for precise calculations
    pub fn timing_ns(&self) -> u128 {
        self.timing.as_nanos()
    }
    
    /// Set refractory state after spike
    pub fn set_refractory_state(&mut self, state: RefractoryState) {
        self.refractory_state = state;
    }
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.neuron_id == other.neuron_id && 
        self.timing == other.timing &&
        (self.amplitude - other.amplitude).abs() < f32::EPSILON
    }
}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.timing.partial_cmp(&other.timing)
    }
}
```

## Verification Steps
1. Create NeuronId and RefractoryState types
2. Implement SpikeEvent structure with all fields
3. Add biological validation logic
4. Implement timing precision methods
5. Ensure all tests pass

## Success Criteria
- [ ] SpikeEvent struct compiles
- [ ] Nanosecond timing precision maintained
- [ ] Amplitude clamping works correctly
- [ ] Biological validation functional
- [ ] All tests pass