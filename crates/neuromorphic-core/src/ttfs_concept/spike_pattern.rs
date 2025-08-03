//! Spike pattern representation for TTFS encoding

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A spike event in the TTFS pattern
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Neuron that spiked
    pub neuron_id: u32,
    
    /// Time since pattern start
    pub timestamp: Duration,
    
    /// Spike amplitude (normalized 0.0 to 1.0)
    pub amplitude: f32,
    
    /// Frequency component (Hz)
    pub frequency: f32,
}

/// Pattern of spikes representing a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePattern {
    /// Ordered sequence of spike events
    pub events: Vec<SpikeEvent>,
    
    /// Total pattern duration
    pub duration: Duration,
    
    /// Pattern complexity metric
    pub complexity: f32,
    
    /// Spike density (spikes per millisecond)
    pub density: f32,
}

impl SpikePattern {
    /// Create a new spike pattern
    pub fn new(events: Vec<SpikeEvent>) -> Self {
        let duration = events.iter()
            .map(|e| e.timestamp)
            .max()
            .unwrap_or(Duration::ZERO);
        
        let density = if duration.as_millis() > 0 {
            events.len() as f32 / duration.as_millis() as f32
        } else {
            0.0
        };
        
        let complexity = Self::calculate_complexity(&events);
        
        Self {
            events,
            duration,
            complexity,
            density,
        }
    }
    
    /// Get time of first spike
    pub fn first_spike_time(&self) -> Option<Duration> {
        self.events.iter()
            .map(|e| e.timestamp)
            .min()
    }
    
    /// Get time of last spike
    pub fn last_spike_time(&self) -> Option<Duration> {
        self.events.iter()
            .map(|e| e.timestamp)
            .max()
    }
    
    /// Calculate inter-spike intervals
    pub fn inter_spike_intervals(&self) -> Vec<Duration> {
        let mut sorted_events = self.events.clone();
        sorted_events.sort_by_key(|e| e.timestamp);
        
        sorted_events.windows(2)
            .map(|pair| pair[1].timestamp - pair[0].timestamp)
            .collect()
    }
    
    /// Calculate pattern complexity based on temporal structure
    fn calculate_complexity(events: &[SpikeEvent]) -> f32 {
        if events.len() < 2 {
            return 0.0;
        }
        
        // Complexity based on:
        // 1. Number of unique neurons
        // 2. Temporal distribution
        // 3. Frequency diversity
        
        let unique_neurons = events.iter()
            .map(|e| e.neuron_id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        let frequency_variance = Self::calculate_frequency_variance(events);
        let temporal_entropy = Self::calculate_temporal_entropy(events);
        
        let complexity = (unique_neurons as f32 / events.len() as f32) 
            * frequency_variance 
            * temporal_entropy;
        
        complexity.clamp(0.0, 1.0)
    }
    
    fn calculate_frequency_variance(events: &[SpikeEvent]) -> f32 {
        if events.is_empty() {
            return 0.0;
        }
        
        let mean_freq = events.iter().map(|e| e.frequency).sum::<f32>() / events.len() as f32;
        let variance = events.iter()
            .map(|e| (e.frequency - mean_freq).powi(2))
            .sum::<f32>() / events.len() as f32;
        
        variance.sqrt() / 100.0 // Normalize assuming max frequency ~100Hz
    }
    
    fn calculate_temporal_entropy(events: &[SpikeEvent]) -> f32 {
        // Simple entropy calculation based on spike timing distribution
        let total_duration = events.iter()
            .map(|e| e.timestamp.as_millis())
            .max()
            .unwrap_or(1) as f32;
        
        let mut bins = vec![0u32; 10]; // 10 time bins
        
        for event in events {
            let bin = ((event.timestamp.as_millis() as f32 / total_duration) * 9.0) as usize;
            bins[bin.min(9)] += 1;
        }
        
        let total = events.len() as f32;
        let entropy = bins.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total;
                -p * p.ln()
            })
            .sum::<f32>();
        
        entropy / 2.3 // Normalize by ln(10)
    }
}

impl Default for SpikePattern {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            duration: Duration::ZERO,
            complexity: 0.0,
            density: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spike_pattern_creation() {
        let events = vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(10),
                amplitude: 0.8,
                frequency: 40.0,
            },
            SpikeEvent {
                neuron_id: 2,
                timestamp: Duration::from_millis(25),
                amplitude: 0.6,
                frequency: 60.0,
            },
        ];
        
        let pattern = SpikePattern::new(events);
        
        assert_eq!(pattern.events.len(), 2);
        assert_eq!(pattern.duration, Duration::from_millis(25));
        assert!(pattern.density > 0.0);
        assert_eq!(pattern.first_spike_time(), Some(Duration::from_millis(10)));
    }
    
    #[test]
    fn test_inter_spike_intervals() {
        let events = vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(0),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 2,
                timestamp: Duration::from_millis(10),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 3,
                timestamp: Duration::from_millis(30),
                amplitude: 1.0,
                frequency: 50.0,
            },
        ];
        
        let pattern = SpikePattern::new(events);
        let isis = pattern.inter_spike_intervals();
        
        assert_eq!(isis.len(), 2);
        assert_eq!(isis[0], Duration::from_millis(10));
        assert_eq!(isis[1], Duration::from_millis(20));
    }
}