//! TTFS encoding utilities

use super::spike_pattern::{SpikePattern, SpikeEvent};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct EncodingConfig {
    pub max_spike_time: Duration,
    pub num_neurons: u32,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            max_spike_time: Duration::from_millis(100),
            num_neurons: 100,
        }
    }
}

pub struct TTFSEncoder {
    config: EncodingConfig,
}

impl Default for TTFSEncoder {
    fn default() -> Self {
        Self {
            config: EncodingConfig::default(),
        }
    }
}

impl TTFSEncoder {
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }
    
    pub fn encode(&self, features: &[f32]) -> SpikePattern {
        let max_time_ms = self.config.max_spike_time.as_millis() as f32;
        
        let events = features.iter()
            .enumerate()
            .take(self.config.num_neurons as usize)
            .filter_map(|(i, &feature)| {
                if feature > 0.1 {
                    // TTFS encoding: stronger features spike earlier
                    let spike_time = max_time_ms * (1.0 - feature.clamp(0.0, 1.0));
                    
                    Some(SpikeEvent {
                        neuron_id: i as u32,
                        timestamp: Duration::from_millis(spike_time as u64),
                        amplitude: feature.clamp(0.0, 1.0),
                        frequency: 25.0 + (75.0 * feature.clamp(0.0, 1.0)),
                    })
                } else {
                    None
                }
            })
            .collect();
        
        SpikePattern::new(events)
    }
}