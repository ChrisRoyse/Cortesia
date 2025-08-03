# Task 10g: Create Spike Operations Mapper

**Estimated Time**: 6 minutes  
**Dependencies**: 10f_implement_spike_analysis_method.md  
**Stage**: Neural Integration - Operation Mapping

## Objective
Create spike operation mapper to convert patterns to graph operations.

## Implementation

Create `src/spike_processing/spike_operations.rs`:
```rust
use crate::spike_processing::spike_processor::{ProcessedSpikePattern, GraphOperation, SpikeProcessingError};

pub struct SpikeOperationMapper {
    frequency_thresholds: FrequencyThresholds,
}

#[derive(Debug, Clone)]
struct FrequencyThresholds {
    low_frequency: f32,
    medium_frequency: f32,
    high_frequency: f32,
}

impl Default for FrequencyThresholds {
    fn default() -> Self {
        Self {
            low_frequency: 10.0,    // < 10 Hz
            medium_frequency: 50.0, // 10-50 Hz
            high_frequency: 100.0,  // > 50 Hz
        }
    }
}

impl SpikeOperationMapper {
    pub fn new() -> Self {
        Self {
            frequency_thresholds: FrequencyThresholds::default(),
        }
    }
    
    pub async fn map_pattern_to_operations(
        &self,
        processed_pattern: &ProcessedSpikePattern,
    ) -> Result<Vec<GraphOperation>, SpikeProcessingError> {
        let mut operations = Vec::new();
        
        // Map based on spike frequency
        match processed_pattern.spike_frequency {
            freq if freq < self.frequency_thresholds.low_frequency => {
                // Low frequency -> weak node activation
                operations.push(GraphOperation::NodeActivation {
                    concept_id: format!("concept_{}", processed_pattern.pattern_id),
                    activation_strength: 0.3,
                });
            },
            freq if freq < self.frequency_thresholds.medium_frequency => {
                // Medium frequency -> moderate activation + relationship traversal
                operations.push(GraphOperation::NodeActivation {
                    concept_id: format!("concept_{}", processed_pattern.pattern_id),
                    activation_strength: 0.6,
                });
                operations.push(GraphOperation::RelationshipTraversal {
                    source_id: format!("concept_{}", processed_pattern.pattern_id),
                    target_id: "related_concept".to_string(),
                    traversal_weight: 0.5,
                });
            },
            _ => {
                // High frequency -> strong activation + cluster activation
                operations.push(GraphOperation::NodeActivation {
                    concept_id: format!("concept_{}", processed_pattern.pattern_id),
                    activation_strength: 0.9,
                });
                operations.push(GraphOperation::ClusterActivation {
                    cluster_id: "active_cluster".to_string(),
                    activation_pattern: processed_pattern.temporal_features.clone(),
                });
            }
        }
        
        Ok(operations)
    }
}
```

## Acceptance Criteria
- [ ] Operation mapper compiles
- [ ] Frequency-based mapping implemented
- [ ] Different operation types generated
- [ ] Threshold-based logic working

## Validation Steps
```bash
cargo check
```

## Next Task
Proceed to **10h_implement_main_processing_method.md**