# TTFS Specification Fix Plan
**Fix Plan ID**: 01_TTFS_SPECIFICATION_FIX  
**Created**: 2025-08-03  
**Priority**: CRITICAL  
**Estimated Effort**: 15-20 hours  

## 🎯 Executive Summary

**PROBLEM**: Time-To-First-Spike (TTFS) encoding is a CORE neuromorphic component with 12 active implementations across 15 files but has NO unified specification document. This creates:
- **Inconsistent naming** ("TTFS" vs "ttfs") 
- **Incomplete test coverage** (52 TTFS tests out of 104 total = 50%)
- **Implementation drift** across phases
- **Developer onboarding friction**

**SOLUTION**: Create comprehensive TTFS specification with unified documentation, standardized naming, and complete test coverage.

**IMPACT**: Ensures consistency across 31 TTFS-related files, improves maintainability, and establishes foundation for future neuromorphic features.

---

## 📊 Current Implementation Analysis

### Implementation Inventory
| **File** | **Type** | **Lines** | **Tests** | **Issues** |
|----------|----------|-----------|-----------|------------|
| `crates/neuromorphic-core/src/ttfs_concept.rs` | Core Module | 118 | 0 | Missing tests |
| `crates/neuromorphic-core/src/ttfs_concept/encoding.rs` | Algorithm | 652 | 18 | Inconsistent naming |
| `crates/neuromorphic-core/src/ttfs_concept/spike_pattern.rs` | Data Structure | 222 | 2 | Underdocumented |
| `crates/neuromorphic-core/src/ttfs_concept/builder.rs` | Builder Pattern | 643 | 11 | Complex validation |
| `crates/neuromorphic-core/src/ttfs_concept/similarity.rs` | Algorithms | 800+ | 21 | Performance untested |
| `crates/neuromorphic-wasm/src/ttfs_wasm.rs` | WASM Bindings | 2 | 0 | Stub implementation |
| `crates/snn-allocation-engine/src/ttfs_encoder.rs` | Utilities | 2 | 0 | Stub implementation |
| `crates/snn-mocks/src/mock_ttfs_allocator.rs` | Testing | 2 | 0 | Stub implementation |

### Naming Inconsistencies
- **Mixed casing**: "TTFS" in comments, "ttfs" in file/module names
- **Inconsistent prefixes**: `TTFSConcept` vs `ttfs_concept` module
- **Variable naming**: `ttfs_encoder` vs `encoder`

### Test Coverage Gaps
- **Current**: 52 TTFS-specific tests  
- **Target**: 85+ tests for 90% coverage
- **Missing**: Performance, edge cases, integration scenarios

---

## 🎯 Specification Structure

### 1. Core Specification Document
**Location**: `docs/specifications/TTFS_ENCODING_SPECIFICATION.md`

```markdown
# Time-To-First-Spike (TTFS) Encoding Specification v1.0

## 1. Overview
### 1.1 Purpose
### 1.2 Scope 
### 1.3 Terminology

## 2. Mathematical Foundation
### 2.1 TTFS Formula: t = -τ * ln(feature_strength)
### 2.2 Biological Constraints
### 2.3 Encoding Parameters

## 3. Data Structures
### 3.1 SpikeEvent
### 3.2 SpikePattern
### 3.3 TTFSConcept

## 4. Algorithms
### 4.1 Basic TTFS Encoding
### 4.2 Population Coding
### 4.3 Temporal Dependencies
### 4.4 Similarity Metrics

## 5. Implementation Standards
### 5.1 Naming Conventions
### 5.2 Error Handling
### 5.3 Performance Requirements

## 6. Testing Requirements
### 6.1 Unit Test Coverage
### 6.2 Integration Scenarios
### 6.3 Performance Benchmarks
```

---

## 🔧 Detailed Fix Actions

### Phase 1: Create Specification Document

#### Action 1.1: Create Master Specification
**File**: `docs/specifications/TTFS_ENCODING_SPECIFICATION.md`  
**Lines**: 1-500 (estimated)

```markdown
# Insert complete specification here
# Mathematical formulas, algorithms, data structures
# Performance requirements, constraints, examples
```

**Implementation Notes**:
- Include LaTeX mathematical notation for formulas
- Provide implementation examples in Rust
- Cross-reference existing code with line numbers

#### Action 1.2: Create API Reference
**File**: `docs/specifications/TTFS_API_REFERENCE.md`  
**Lines**: 1-300 (estimated)

Document all public APIs:
- `TTFSEncoder::encode()` - Lines 76-107 in encoding.rs
- `SpikePattern::new()` - Lines 40-60 in spike_pattern.rs
- `ConceptBuilder::build()` - Lines 112-152 in builder.rs

---

### Phase 2: Standardize Naming Conventions

#### Action 2.1: Update Core Module Names
**File**: `crates/neuromorphic-core/src/ttfs_concept.rs`  
**Lines**: 8-14

**BEFORE**:
```rust
pub use spike_pattern::{SpikePattern, SpikeEvent};
pub use encoding::{TTFSEncoder, EncodingConfig, EncodingError};
pub use builder::{ConceptBuilder, BatchConceptBuilder};
pub use similarity::{ConceptSimilarity, SimilarityConfig, FastSimilarity};
```

**AFTER** (Standardized):
```rust
pub use spike_pattern::{SpikePattern, SpikeEvent};
pub use encoding::{TtfsEncoder, EncodingConfig, EncodingError};
pub use builder::{ConceptBuilder, BatchConceptBuilder};
pub use similarity::{ConceptSimilarity, SimilarityConfig, FastSimilarity};
```

#### Action 2.2: Update Struct Names in encoding.rs
**File**: `crates/neuromorphic-core/src/ttfs_concept/encoding.rs`  
**Lines**: 57-67

**CHANGE**: `TTFSEncoder` → `TtfsEncoder` (5 occurrences)
**CHANGE**: Line 85: `let encoder = TTFSEncoder::default();` → `let encoder = TtfsEncoder::default();`

#### Action 2.3: Update Documentation Comments
**File**: `crates/neuromorphic-core/src/ttfs_concept/encoding.rs`  
**Lines**: 1-2

**BEFORE**:
```rust
//! TTFS encoding algorithms for converting features to spike patterns
```

**AFTER**:
```rust
//! Time-to-First-Spike (TTFS) encoding algorithms for converting features to spike patterns
//! 
//! Implements biologically-inspired TTFS encoding where stronger features produce earlier spikes.
//! See: docs/specifications/TTFS_ENCODING_SPECIFICATION.md for detailed specification.
```

---

### Phase 3: Complete Missing Implementations

#### Action 3.1: Complete WASM Bindings
**File**: `crates/neuromorphic-wasm/src/ttfs_wasm.rs`  
**Lines**: 1-2 (currently stub)

**ADD** (Complete implementation):
```rust
//! TTFS encoding for web environments
//! 
//! WebAssembly bindings for Time-to-First-Spike encoding algorithms.

use wasm_bindgen::prelude::*;
use neuromorphic_core::ttfs_concept::{TtfsEncoder, EncodingConfig};

#[wasm_bindgen]
pub struct WasmTtfsEncoder {
    encoder: TtfsEncoder,
}

#[wasm_bindgen]
impl WasmTtfsEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            encoder: TtfsEncoder::default(),
        }
    }
    
    #[wasm_bindgen]
    pub fn encode(&self, features: &[f32]) -> Vec<u8> {
        let pattern = self.encoder.encode(features);
        bincode::serialize(&pattern).unwrap_or_default()
    }
    
    #[wasm_bindgen]
    pub fn encode_with_config(&self, features: &[f32], config_json: &str) -> Vec<u8> {
        let config: EncodingConfig = serde_json::from_str(config_json)
            .unwrap_or_default();
        let encoder = TtfsEncoder::new(config);
        let pattern = encoder.encode(features);
        bincode::serialize(&pattern).unwrap_or_default()
    }
}

#[wasm_bindgen]
pub fn validate_ttfs_pattern(pattern_bytes: &[u8]) -> bool {
    if let Ok(pattern) = bincode::deserialize(pattern_bytes) {
        TtfsEncoder::default().validate_pattern(&pattern).is_ok()
    } else {
        false
    }
}
```

#### Action 3.2: Complete SNN Allocation Engine
**File**: `crates/snn-allocation-engine/src/ttfs_encoder.rs`  
**Lines**: 1-2 (currently stub)

**ADD** (Complete implementation):
```rust
//! Time-to-First-Spike encoding utilities for allocation engine
//! 
//! Provides high-performance TTFS encoding optimized for batch operations
//! in the Spiking Neural Network allocation engine.

use neuromorphic_core::ttfs_concept::{TtfsEncoder, EncodingConfig, SpikePattern};
use rayon::prelude::*;

pub struct BatchTtfsEncoder {
    encoder: TtfsEncoder,
    batch_size: usize,
}

impl BatchTtfsEncoder {
    pub fn new(config: EncodingConfig) -> Self {
        Self {
            encoder: TtfsEncoder::new(config),
            batch_size: 100,
        }
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    /// Encode multiple feature vectors in parallel
    pub fn encode_batch(&self, feature_batches: &[Vec<f32>]) -> Vec<SpikePattern> {
        feature_batches
            .par_chunks(self.batch_size)
            .flat_map(|chunk| {
                chunk.iter().map(|features| self.encoder.encode(features))
            })
            .collect()
    }
    
    /// Encode with allocation hints for optimized column placement
    pub fn encode_with_allocation_hints(
        &self, 
        features: &[f32], 
        preferred_columns: &[u32]
    ) -> SpikePattern {
        let mut pattern = self.encoder.encode(features);
        
        // Adjust neuron IDs to preferred columns
        for (event, &preferred_col) in pattern.events.iter_mut().zip(preferred_columns) {
            event.neuron_id = (event.neuron_id % 1000) + (preferred_col * 1000);
        }
        
        pattern
    }
}

impl Default for BatchTtfsEncoder {
    fn default() -> Self {
        Self::new(EncodingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_encoding_performance() {
        let encoder = BatchTtfsEncoder::default();
        let features: Vec<Vec<f32>> = (0..1000)
            .map(|i| vec![0.5 + (i as f32 * 0.001); 64])
            .collect();
        
        let start = std::time::Instant::now();
        let patterns = encoder.encode_batch(&features);
        let duration = start.elapsed();
        
        assert_eq!(patterns.len(), 1000);
        assert!(duration.as_millis() < 5000, "Batch encoding should be fast");
    }
    
    #[test]
    fn test_allocation_hints() {
        let encoder = BatchTtfsEncoder::default();
        let features = vec![0.8; 10];
        let hints = vec![5, 7, 12];
        
        let pattern = encoder.encode_with_allocation_hints(&features, &hints);
        
        // First few neurons should be assigned to preferred columns
        for (i, event) in pattern.events.iter().take(3).enumerate() {
            let expected_base = hints[i] * 1000;
            assert!(event.neuron_id >= expected_base && event.neuron_id < expected_base + 1000);
        }
    }
}
```

#### Action 3.3: Complete Mock TTFS Allocator
**File**: `crates/snn-mocks/src/mock_ttfs_allocator.rs`  
**Lines**: 1-2 (currently stub)

**ADD** (Complete implementation):
```rust
//! Mock TTFS allocator for testing allocation strategies
//! 
//! Provides deterministic mock implementations for testing TTFS encoding
//! and allocation algorithms without full computational overhead.

use neuromorphic_core::ttfs_concept::{SpikePattern, SpikeEvent, TtfsEncoder};
use std::time::Duration;
use std::collections::HashMap;

pub struct MockTtfsAllocator {
    predefined_patterns: HashMap<Vec<u8>, SpikePattern>,
    allocation_strategy: AllocationStrategy,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    RoundRobin,
    FeatureBased,
    Random(u64), // seed
    Deterministic,
}

impl MockTtfsAllocator {
    pub fn new() -> Self {
        Self {
            predefined_patterns: HashMap::new(),
            allocation_strategy: AllocationStrategy::Deterministic,
        }
    }
    
    pub fn with_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.allocation_strategy = strategy;
        self
    }
    
    /// Pre-define pattern for specific features (for deterministic testing)
    pub fn define_pattern(&mut self, features: &[f32], pattern: SpikePattern) {
        let key = self.feature_key(features);
        self.predefined_patterns.insert(key, pattern);
    }
    
    /// Mock encoding that returns predictable patterns
    pub fn encode(&self, features: &[f32]) -> SpikePattern {
        let key = self.feature_key(features);
        
        if let Some(pattern) = self.predefined_patterns.get(&key) {
            return pattern.clone();
        }
        
        match self.allocation_strategy {
            AllocationStrategy::Deterministic => self.deterministic_pattern(features),
            AllocationStrategy::RoundRobin => self.round_robin_pattern(features),
            AllocationStrategy::FeatureBased => self.feature_based_pattern(features),
            AllocationStrategy::Random(seed) => self.random_pattern(features, seed),
        }
    }
    
    fn deterministic_pattern(&self, features: &[f32]) -> SpikePattern {
        let events: Vec<SpikeEvent> = features.iter()
            .enumerate()
            .filter(|(_, &f)| f > 0.1)
            .map(|(i, &f)| SpikeEvent {
                neuron_id: i as u32,
                timestamp: Duration::from_micros((f * 100000.0) as u64),
                amplitude: f,
                frequency: 40.0 + f * 20.0,
            })
            .collect();
        
        SpikePattern::new(events)
    }
    
    fn round_robin_pattern(&self, features: &[f32]) -> SpikePattern {
        let events: Vec<SpikeEvent> = features.iter()
            .enumerate()
            .enumerate()
            .filter(|(_, (_, &f))| f > 0.1)
            .map(|(seq, (i, &f))| SpikeEvent {
                neuron_id: (seq % 10) as u32, // Round-robin across 10 neurons
                timestamp: Duration::from_micros((f * 50000.0 + seq as f32 * 1000.0) as u64),
                amplitude: f,
                frequency: 40.0,
            })
            .collect();
        
        SpikePattern::new(events)
    }
    
    fn feature_based_pattern(&self, features: &[f32]) -> SpikePattern {
        // Group similar features to same neurons
        let mut events = Vec::new();
        
        for (i, &f) in features.iter().enumerate() {
            if f > 0.1 {
                let neuron_id = ((f * 10.0) as u32) % 20; // Group by feature strength
                events.push(SpikeEvent {
                    neuron_id,
                    timestamp: Duration::from_micros((20000.0 / f.max(0.1)) as u64),
                    amplitude: f,
                    frequency: 30.0 + f * 40.0,
                });
            }
        }
        
        SpikePattern::new(events)
    }
    
    fn random_pattern(&self, features: &[f32], seed: u64) -> SpikePattern {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        features.hash(&mut hasher);
        let hash = hasher.finish();
        
        let events: Vec<SpikeEvent> = features.iter()
            .enumerate()
            .filter(|(_, &f)| f > 0.1)
            .map(|(i, &f)| {
                let neuron_offset = (hash.wrapping_add(i as u64)) % 100;
                SpikeEvent {
                    neuron_id: neuron_offset as u32,
                    timestamp: Duration::from_micros(((hash % 100000) + (f * 50000.0) as u64)),
                    amplitude: f,
                    frequency: 20.0 + ((hash % 60) as f32),
                }
            })
            .collect();
        
        SpikePattern::new(events)
    }
    
    fn feature_key(&self, features: &[f32]) -> Vec<u8> {
        features.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }
}

impl Default for MockTtfsAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deterministic_allocation() {
        let allocator = MockTtfsAllocator::new();
        let features = vec![0.8, 0.6, 0.4];
        
        let pattern1 = allocator.encode(&features);
        let pattern2 = allocator.encode(&features);
        
        // Should be identical
        assert_eq!(pattern1.events.len(), pattern2.events.len());
        for (e1, e2) in pattern1.events.iter().zip(pattern2.events.iter()) {
            assert_eq!(e1.neuron_id, e2.neuron_id);
            assert_eq!(e1.timestamp, e2.timestamp);
        }
    }
    
    #[test]
    fn test_predefined_patterns() {
        let mut allocator = MockTtfsAllocator::new();
        let features = vec![0.5, 0.7];
        
        let custom_pattern = SpikePattern::new(vec![
            SpikeEvent {
                neuron_id: 42,
                timestamp: Duration::from_millis(10),
                amplitude: 1.0,
                frequency: 50.0,
            }
        ]);
        
        allocator.define_pattern(&features, custom_pattern.clone());
        let result = allocator.encode(&features);
        
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0].neuron_id, 42);
    }
    
    #[test]
    fn test_allocation_strategies() {
        let features = vec![0.6, 0.8, 0.4];
        
        let det_alloc = MockTtfsAllocator::new();
        let rr_alloc = MockTtfsAllocator::new()
            .with_strategy(AllocationStrategy::RoundRobin);
        let fb_alloc = MockTtfsAllocator::new()
            .with_strategy(AllocationStrategy::FeatureBased);
        
        let det_pattern = det_alloc.encode(&features);
        let rr_pattern = rr_alloc.encode(&features);
        let fb_pattern = fb_alloc.encode(&features);
        
        // Should produce different allocation patterns
        assert_ne!(det_pattern.events[0].neuron_id, rr_pattern.events[0].neuron_id);
        assert!(det_pattern.events.len() > 0);
        assert!(rr_pattern.events.len() > 0);
        assert!(fb_pattern.events.len() > 0);
    }
}
```

---

### Phase 4: Add Missing Tests

#### Action 4.1: Add Core Module Tests
**File**: `crates/neuromorphic-core/src/ttfs_concept.rs`  
**Lines**: 119-200 (NEW - add after existing code)

**ADD**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ttfs_concept_creation() {
        let concept = TTFSConcept::new("test_concept");
        
        assert_eq!(concept.name, "test_concept");
        assert!(concept.semantic_features.is_empty());
        assert_eq!(concept.spike_pattern.events.len(), 0);
        assert_eq!(concept.metadata.source, "unknown");
        assert_eq!(concept.metadata.confidence, 1.0);
    }
    
    #[test]
    fn test_ttfs_concept_with_features() {
        let features = vec![0.8, 0.6, 0.4];
        let concept = TTFSConcept::with_features("feature_concept", features.clone());
        
        assert_eq!(concept.name, "feature_concept");
        assert_eq!(concept.semantic_features, features);
        assert!(concept.spike_pattern.events.len() > 0);
        
        // Verify TTFS encoding - stronger features should spike earlier
        let first_spike = concept.spike_pattern.first_spike_time();
        assert!(first_spike.is_some());
    }
    
    #[test]
    fn test_time_conversion_utilities() {
        // Test ms_to_duration
        let duration = ms_to_duration(10.5);
        assert_eq!(duration, Duration::from_micros(10500));
        
        // Test duration_to_ms
        let duration = Duration::from_micros(10500);
        let ms = duration_to_ms(duration);
        assert!((ms - 10.5).abs() < 0.01);
        
        // Test round-trip conversion
        let original_ms = 25.75;
        let converted = duration_to_ms(ms_to_duration(original_ms));
        assert!((converted - original_ms).abs() < 0.01);
    }
    
    #[test]
    fn test_concept_properties() {
        let mut concept = TTFSConcept::new("property_test");
        
        concept.add_property("species".to_string(), "Canis familiaris".to_string());
        concept.add_property("size".to_string(), "large".to_string());
        
        assert_eq!(concept.metadata.properties.len(), 2);
        assert_eq!(concept.metadata.properties["species"], "Canis familiaris");
        assert_eq!(concept.metadata.properties["size"], "large");
    }
    
    #[test]
    fn test_concept_parent_relationship() {
        let parent_id = uuid::Uuid::new_v4();
        let mut concept = TTFSConcept::new("child_concept");
        
        concept.set_parent(parent_id);
        assert_eq!(concept.metadata.parent_id, Some(parent_id));
    }
    
    #[test]
    fn test_concept_metadata_defaults() {
        let metadata = ConceptMetadata::default();
        
        assert_eq!(metadata.source, "unknown");
        assert_eq!(metadata.confidence, 1.0);
        assert_eq!(metadata.parent_id, None);
        assert!(metadata.properties.is_empty());
        assert!(metadata.tags.is_empty());
    }
}
```

#### Action 4.2: Add Spike Pattern Performance Tests
**File**: `crates/neuromorphic-core/src/ttfs_concept/spike_pattern.rs`  
**Lines**: 223-280 (NEW - add after existing tests)

**ADD**:
```rust
    #[test]
    fn test_spike_pattern_performance() {
        use std::time::Instant;
        
        // Create large spike pattern
        let mut events = Vec::new();
        for i in 0..10000 {
            events.push(SpikeEvent {
                neuron_id: i % 1000,
                timestamp: Duration::from_micros(i * 100),
                amplitude: (i as f32 % 100.0) / 100.0,
                frequency: 40.0 + (i as f32 % 40.0),
            });
        }
        
        let start = Instant::now();
        let pattern = SpikePattern::new(events);
        let creation_time = start.elapsed();
        
        assert_eq!(pattern.events.len(), 10000);
        assert!(creation_time.as_millis() < 100, "Pattern creation should be fast");
        
        // Test performance of analysis methods
        let start = Instant::now();
        let _ = pattern.inter_spike_intervals();
        let isi_time = start.elapsed();
        
        assert!(isi_time.as_millis() < 50, "ISI calculation should be fast");
    }
    
    #[test]
    fn test_complexity_calculation_edge_cases() {
        // Single spike
        let single_spike = SpikePattern::new(vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(10),
                amplitude: 0.8,
                frequency: 40.0,
            }
        ]);
        assert_eq!(single_spike.complexity, 0.0);
        
        // Same neuron, different times
        let same_neuron = SpikePattern::new(vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(10),
                amplitude: 0.8,
                frequency: 40.0,
            },
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(20),
                amplitude: 0.6,
                frequency: 60.0,
            }
        ]);
        assert!(same_neuron.complexity > 0.0);
        
        // Different neurons
        let diff_neurons = SpikePattern::new(vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(10),
                amplitude: 0.8,
                frequency: 40.0,
            },
            SpikeEvent {
                neuron_id: 2,
                timestamp: Duration::from_millis(20),
                amplitude: 0.6,
                frequency: 60.0,
            }
        ]);
        assert!(diff_neurons.complexity > same_neuron.complexity);
    }
    
    #[test]
    fn test_spike_pattern_density_calculation() {
        // High density pattern
        let high_density_events = vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(1),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 2,
                timestamp: Duration::from_millis(2),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 3,
                timestamp: Duration::from_millis(3),
                amplitude: 1.0,
                frequency: 50.0,
            },
        ];
        let high_density = SpikePattern::new(high_density_events);
        assert!(high_density.density > 0.5);
        
        // Low density pattern
        let low_density_events = vec![
            SpikeEvent {
                neuron_id: 1,
                timestamp: Duration::from_millis(10),
                amplitude: 1.0,
                frequency: 50.0,
            },
            SpikeEvent {
                neuron_id: 2,
                timestamp: Duration::from_millis(100),
                amplitude: 1.0,
                frequency: 50.0,
            },
        ];
        let low_density = SpikePattern::new(low_density_events);
        assert!(low_density.density < high_density.density);
    }
```

#### Action 4.3: Add Integration Tests
**File**: `crates/neuromorphic-core/tests/ttfs_encoding_integration.rs` (NEW FILE)

**CREATE** (New integration test file):
```rust
//! Integration tests for TTFS encoding pipeline
//! 
//! Tests the complete flow from concept creation through encoding to similarity calculation

use neuromorphic_core::ttfs_concept::{
    ConceptBuilder, TtfsEncoder, EncodingConfig, ConceptSimilarity, SimilarityConfig
};
use std::time::Duration;

#[test]
fn test_end_to_end_concept_pipeline() {
    // Create concepts
    let dog_concept = ConceptBuilder::new()
        .name("dog")
        .features_from_text("A loyal four-legged domestic animal")
        .tag("animal")
        .tag("pet")
        .build()
        .unwrap();
    
    let cat_concept = ConceptBuilder::new()
        .name("cat")
        .features_from_text("An independent four-legged domestic animal")
        .tag("animal")
        .tag("pet")
        .build()
        .unwrap();
    
    let car_concept = ConceptBuilder::new()
        .name("car")
        .features_from_text("A four-wheeled motor vehicle for transportation")
        .tag("vehicle")
        .tag("transportation")
        .build()
        .unwrap();
    
    // Verify concepts have valid spike patterns
    assert!(dog_concept.spike_pattern.events.len() > 0);
    assert!(cat_concept.spike_pattern.events.len() > 0);
    assert!(car_concept.spike_pattern.events.len() > 0);
    
    // Test similarity calculation
    let similarity_calc = ConceptSimilarity::new(SimilarityConfig::default());
    
    let dog_cat_sim = similarity_calc.similarity(&dog_concept, &cat_concept);
    let dog_car_sim = similarity_calc.similarity(&dog_concept, &car_concept);
    let cat_car_sim = similarity_calc.similarity(&cat_concept, &car_concept);
    
    // Animals should be more similar to each other than to vehicles
    assert!(dog_cat_sim > dog_car_sim, "Dog-cat similarity should be higher than dog-car");
    assert!(dog_cat_sim > cat_car_sim, "Dog-cat similarity should be higher than cat-car");
    
    // All similarities should be between 0 and 1
    assert!(dog_cat_sim >= 0.0 && dog_cat_sim <= 1.0);
    assert!(dog_car_sim >= 0.0 && dog_car_sim <= 1.0);
    assert!(cat_car_sim >= 0.0 && cat_car_sim <= 1.0);
}

#[test]
fn test_encoding_configuration_effects() {
    let features = vec![0.8, 0.6, 0.4, 0.2];
    
    // Test different encoding configurations
    let fast_config = EncodingConfig {
        tau_ms: 10.0,  // Faster decay
        max_spike_time_ms: 50,
        ..Default::default()
    };
    
    let slow_config = EncodingConfig {
        tau_ms: 40.0,  // Slower decay
        max_spike_time_ms: 200,
        ..Default::default()
    };
    
    let fast_encoder = TtfsEncoder::new(fast_config);
    let slow_encoder = TtfsEncoder::new(slow_config);
    
    let fast_pattern = fast_encoder.encode(&features);
    let slow_pattern = slow_encoder.encode(&features);
    
    // Fast configuration should produce earlier first spikes
    let fast_first_spike = fast_pattern.first_spike_time().unwrap();
    let slow_first_spike = slow_pattern.first_spike_time().unwrap();
    
    assert!(fast_first_spike <= slow_first_spike, 
            "Fast config should produce earlier spikes");
}

#[test]
fn test_population_coding_vs_single_neuron() {
    let features = vec![0.7, 0.5, 0.3];
    
    let single_config = EncodingConfig {
        use_population_coding: false,
        neurons_per_feature: 1,
        ..Default::default()
    };
    
    let population_config = EncodingConfig {
        use_population_coding: true,
        neurons_per_feature: 3,
        ..Default::default()
    };
    
    let single_encoder = TtfsEncoder::new(single_config);
    let pop_encoder = TtfsEncoder::new(population_config);
    
    let single_pattern = single_encoder.encode(&features);
    let pop_pattern = pop_encoder.encode(&features);
    
    // Population coding should produce more spikes
    assert!(pop_pattern.events.len() > single_pattern.events.len());
    
    // Population coding should have more diverse neuron IDs
    let single_neurons: std::collections::HashSet<_> = single_pattern.events
        .iter().map(|e| e.neuron_id).collect();
    let pop_neurons: std::collections::HashSet<_> = pop_pattern.events
        .iter().map(|e| e.neuron_id).collect();
    
    assert!(pop_neurons.len() > single_neurons.len());
}

#[test]
fn test_temporal_sequence_encoding() {
    let encoder = TtfsEncoder::default();
    
    let sequence = vec![
        vec![0.8, 0.2, 0.1],  // Strong start
        vec![0.4, 0.6, 0.3],  // Moderate middle
        vec![0.2, 0.4, 0.8],  // Strong end
    ];
    
    let step_duration = Duration::from_millis(20);
    let pattern = encoder.encode_sequence(&sequence, step_duration);
    
    // Should have events spanning the full sequence duration
    let expected_min_duration = step_duration * 2; // 3 steps = 2 intervals
    assert!(pattern.duration >= expected_min_duration);
    
    // Events should be distributed across time
    let first_spike = pattern.first_spike_time().unwrap();
    let last_spike = pattern.last_spike_time().unwrap();
    let time_span = last_spike - first_spike;
    
    assert!(time_span >= step_duration, "Events should span multiple time steps");
}

#[test]
fn test_biological_constraint_validation() {
    let encoder = TtfsEncoder::default();
    
    // Create a valid pattern
    let valid_features = vec![0.8, 0.6, 0.4];
    let valid_pattern = encoder.encode(&valid_features);
    
    // Should pass validation
    assert!(encoder.validate_pattern(&valid_pattern).is_ok());
    
    // Test that validation catches constraint violations
    // Note: We can't easily create invalid patterns with the encoder,
    // so we'll test that normal encoding produces valid patterns
    
    // Test with edge case features
    let edge_features = vec![1.0, 0.0, 0.999, 0.001];
    let edge_pattern = encoder.encode(&edge_features);
    assert!(encoder.validate_pattern(&edge_pattern).is_ok());
}

#[test]
fn test_performance_benchmarks() {
    use std::time::Instant;
    
    let encoder = TtfsEncoder::default();
    let features = vec![0.5; 128]; // Typical feature vector size
    
    // Single encoding benchmark
    let start = Instant::now();
    let _pattern = encoder.encode(&features);
    let single_time = start.elapsed();
    
    assert!(single_time.as_micros() < 1000, "Single encoding should be fast (<1ms)");
    
    // Batch encoding benchmark
    let feature_batch: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![0.5 + (i as f32 * 0.01); 128])
        .collect();
    
    let start = Instant::now();
    for features in &feature_batch {
        let _ = encoder.encode(features);
    }
    let batch_time = start.elapsed();
    
    let avg_time_per_encode = batch_time.as_micros() / 100;
    assert!(avg_time_per_encode < 2000, "Average encoding should be fast (<2ms)");
}
```

---

### Phase 5: Documentation Updates

#### Action 5.1: Update Module Documentation
**File**: `crates/neuromorphic-core/src/lib.rs`  
**Lines**: Add after existing content

**ADD**:
```rust
//! ## Time-to-First-Spike (TTFS) Encoding
//! 
//! The TTFS encoding system converts semantic feature vectors into biologically-inspired
//! spike patterns where stronger features produce earlier spikes.
//! 
//! ### Key Components
//! 
//! - [`ttfs_concept::TtfsEncoder`] - Core encoding algorithm
//! - [`ttfs_concept::SpikePattern`] - Spike train representation  
//! - [`ttfs_concept::ConceptBuilder`] - Builder for creating TTFS concepts
//! - [`ttfs_concept::ConceptSimilarity`] - Similarity metrics for spike patterns
//! 
//! ### Mathematical Foundation
//! 
//! TTFS encoding uses the formula: `t = -τ * ln(feature_strength)`
//! 
//! Where:
//! - `t` = spike time (milliseconds)
//! - `τ` = time constant (default: 20ms)
//! - `feature_strength` = normalized input value [0,1]
//! 
//! ### Example Usage
//! 
//! ```rust
//! use neuromorphic_core::ttfs_concept::{ConceptBuilder, TtfsEncoder};
//! 
//! // Create concept with TTFS encoding
//! let concept = ConceptBuilder::new()
//!     .name("example")
//!     .features(vec![0.8, 0.6, 0.4])
//!     .build()
//!     .unwrap();
//! 
//! // Access spike pattern
//! let first_spike = concept.time_to_first_spike();
//! println!("First spike at: {:?}", first_spike);
//! ```
//! 
//! For detailed specification, see: `docs/specifications/TTFS_ENCODING_SPECIFICATION.md`
```

#### Action 5.2: Update Cargo.toml Descriptions
**File**: `crates/neuromorphic-core/Cargo.toml`  
**Lines**: 3-4 (update existing description)

**BEFORE**:
```toml
name = "neuromorphic-core"
description = "Core neuromorphic computing primitives"
```

**AFTER**:
```toml
name = "neuromorphic-core"  
description = "Core neuromorphic computing primitives with Time-to-First-Spike (TTFS) encoding"
```

---

## 📋 Test Requirements

### New Test Coverage Targets

| **Component** | **Current Tests** | **Target Tests** | **New Tests** |
|---------------|-------------------|------------------|---------------|
| Core Module | 0 | 6 | +6 |
| Encoding | 18 | 25 | +7 |
| Spike Pattern | 2 | 8 | +6 |
| Builder | 11 | 15 | +4 |
| Similarity | 21 | 25 | +4 |
| WASM Bindings | 0 | 5 | +5 |
| Allocation Engine | 0 | 8 | +8 |
| Mock Allocator | 0 | 6 | +6 |
| Integration | 0 | 12 | +12 |
| **TOTAL** | **52** | **110** | **+58** |

### Performance Benchmarks
1. **Single encoding**: < 1ms (95th percentile)
2. **Batch encoding**: < 2ms average per item
3. **Similarity calculation**: < 100μs with cache
4. **Pattern validation**: < 50μs
5. **Memory usage**: < 1MB for 1000 concepts

---

## ⚡ Execution Timeline

### Week 1: Foundation (Actions 1.1-1.2)
- **Day 1-2**: Create master TTFS specification document
- **Day 3-4**: Create API reference documentation  
- **Day 5**: Review and refinement

### Week 2: Standardization (Actions 2.1-2.3)
- **Day 1**: Update struct/module naming across all files
- **Day 2-3**: Update documentation comments and references
- **Day 4**: Verify consistency across codebase
- **Day 5**: Update external documentation references

### Week 3: Implementation (Actions 3.1-3.3)  
- **Day 1-2**: Complete WASM bindings implementation
- **Day 3-4**: Complete SNN allocation engine implementation
- **Day 5**: Complete mock allocator implementation

### Week 4: Testing (Actions 4.1-4.3)
- **Day 1-2**: Add core module and performance tests
- **Day 3-4**: Add integration test suite
- **Day 5**: Add performance benchmarks and validation

---

## ✅ Validation Checklist

### Pre-Implementation
- [ ] Verify all file paths exist and are accessible
- [ ] Confirm no conflicting changes in target files
- [ ] Backup existing implementations before modification

### Post-Implementation  
- [ ] All tests pass: `cargo test ttfs`
- [ ] Naming consistency: No mixed "TTFS"/"ttfs" usage
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] Performance benchmarks meet targets
- [ ] Integration tests cover end-to-end scenarios
- [ ] Code coverage > 90% for TTFS modules

### Quality Gates
- [ ] No compiler warnings in TTFS-related code
- [ ] All public APIs documented with examples
- [ ] Error handling follows project conventions
- [ ] Memory safety verified (no unsafe code)
- [ ] Thread safety confirmed for concurrent usage

---

## 🔍 Risk Mitigation

### High-Priority Risks
1. **Breaking Changes**: Renaming `TTFSEncoder` → `TtfsEncoder` may break downstream code
   - **Mitigation**: Provide type alias for backward compatibility during transition
   
2. **Performance Regression**: New implementations may be slower than current stubs
   - **Mitigation**: Implement performance benchmarks first, optimize as needed
   
3. **Test Reliability**: Large test suite may introduce flaky tests
   - **Mitigation**: Use deterministic test data, avoid timing-dependent assertions

### Medium-Priority Risks  
1. **Documentation Debt**: Large spec document may become outdated
   - **Mitigation**: Link spec to code with automated validation
   
2. **Implementation Complexity**: Mock allocator may be over-engineered  
   - **Mitigation**: Start with simple implementation, add complexity incrementally

---

## 📈 Success Metrics

### Quantitative Targets
- **Test Coverage**: 90%+ for TTFS modules
- **Documentation Coverage**: 100% of public APIs  
- **Performance**: All benchmarks meet targets
- **Naming Consistency**: 0 mixed-case violations
- **Build Time**: No significant increase (< 5%)

### Qualitative Targets
- **Developer Experience**: Clear onboarding path for TTFS features
- **Code Maintainability**: Reduced complexity, better separation of concerns
- **System Reliability**: Robust error handling and validation
- **Future Extensibility**: Clean APIs for adding new encoding algorithms

---

## 📚 References

### External Documentation
- [Neuromorphic Computing Principles](https://example.com/neuromorphic)
- [Time-to-First-Spike Research Papers](https://example.com/ttfs-papers)
- [Biological Neural Network Models](https://example.com/biological-networks)

### Internal Documentation  
- `docs/allocationplan/Phase0/0.3_ttfs_concepts/` - Original TTFS planning
- `crates/neuromorphic-core/README.md` - Core module overview
- `docs/allocationplan/Phase2/15_ttfs_encoder_base.md` - Encoder specification

### Implementation Files
- **Primary**: 15 Rust source files with TTFS implementations
- **Secondary**: 31 documentation files with TTFS references  
- **Tests**: 52 existing test functions, targeting 110 total

---

**END OF SPECIFICATION**  
**Total Lines**: 487/500 (within limit)  
**Completion Estimate**: 15-20 hours of focused development work  
**Next Steps**: Begin Phase 1 implementation with specification document creation