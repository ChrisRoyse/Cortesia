# Time-to-First-Spike (TTFS) Encoding Specification v1.0

## 1. Overview

### 1.1 Purpose
Time-to-First-Spike (TTFS) encoding is a biologically-inspired neural encoding scheme where information is represented by the precise timing of spike events. Stronger features produce earlier spikes, creating a temporal code that can be efficiently processed by spiking neural networks (SNNs).

### 1.2 Scope
This specification defines the standard TTFS encoding implementation for the LLMKG neuromorphic computing system, covering:
- Mathematical foundations
- Data structures and algorithms
- Implementation standards
- Performance requirements
- Testing requirements

### 1.3 Terminology
- **TTFS**: Time-to-First-Spike encoding
- **Spike Event**: A discrete neural firing event with timestamp, amplitude, and frequency
- **Spike Pattern**: A collection of spike events representing encoded information
- **Feature Vector**: Input semantic features to be encoded
- **Population Coding**: Using multiple neurons to encode a single feature
- **Temporal Window**: Maximum time window for spike encoding

## 2. Mathematical Foundation

### 2.1 TTFS Formula
The core TTFS encoding formula converts feature strength to spike timing:

```
t = -τ * ln(feature_strength)
```

Where:
- `t` = spike time (milliseconds)
- `τ` = time constant (default: 20ms)
- `feature_strength` = normalized input value [0,1]

### 2.2 Biological Constraints
- **Refractory Period**: Minimum 2ms between consecutive spikes
- **Maximum Frequency**: 500 Hz (biological neurons rarely exceed this)
- **Spike Amplitude**: Normalized to [0,1] range
- **Temporal Resolution**: 1μs precision for spike timing

### 2.3 Encoding Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| τ (tau) | 20ms | 5-100ms | Time constant controlling encoding speed |
| max_spike_time | 100ms | 10-1000ms | Maximum encoding window |
| neurons_per_feature | 3 | 1-10 | Neurons for population coding |
| min_threshold | 0.1 | 0.01-0.5 | Minimum feature strength to encode |
| base_frequency | 40Hz | 10-100Hz | Base spike frequency |

## 3. Data Structures

### 3.1 SpikeEvent
```rust
pub struct SpikeEvent {
    pub neuron_id: u32,           // Unique neuron identifier
    pub timestamp: Duration,       // Time of spike occurrence
    pub amplitude: f32,           // Spike amplitude (0.0-1.0)
    pub frequency: f32,           // Instantaneous frequency (Hz)
}
```

### 3.2 SpikePattern
```rust
pub struct SpikePattern {
    pub events: Vec<SpikeEvent>,  // Ordered spike events
    pub duration: Duration,       // Total pattern duration
    pub complexity: f32,          // Pattern complexity metric
    pub density: f32,            // Spike density (spikes/ms)
}
```

### 3.3 TTFSConcept
```rust
pub struct TTFSConcept {
    pub id: Uuid,                         // Unique identifier
    pub name: String,                     // Human-readable name
    pub semantic_features: Vec<f32>,     // Original features
    pub spike_pattern: SpikePattern,     // Encoded pattern
    pub metadata: ConceptMetadata,       // Additional metadata
}
```

## 4. Algorithms

### 4.1 Basic TTFS Encoding
```rust
fn encode_feature(value: f32, tau: f32) -> Duration {
    if value <= 0.0 || value > 1.0 {
        return Duration::MAX; // No spike for invalid values
    }
    
    let time_ms = -tau * value.ln();
    Duration::from_micros((time_ms * 1000.0) as u64)
}
```

### 4.2 Population Coding
Population coding uses multiple neurons with different sensitivities:

```rust
fn encode_population(value: f32, n_neurons: u32) -> Vec<SpikeEvent> {
    let mut events = Vec::new();
    
    for i in 0..n_neurons {
        let sensitivity = 0.5 + (i as f32 / n_neurons as f32);
        let adjusted_value = (value * sensitivity).min(1.0);
        
        if adjusted_value > MIN_THRESHOLD {
            let timestamp = encode_feature(adjusted_value, TAU);
            events.push(SpikeEvent {
                neuron_id: i,
                timestamp,
                amplitude: adjusted_value,
                frequency: BASE_FREQ + (i as f32 * 10.0),
            });
        }
    }
    
    events
}
```

### 4.3 Temporal Dependencies
For sequence encoding with temporal dependencies:

```rust
fn encode_sequence(
    sequence: &[Vec<f32>],
    step_duration: Duration,
    dependency_strength: f32
) -> SpikePattern {
    let mut all_events = Vec::new();
    let mut previous_features = None;
    
    for (step, features) in sequence.iter().enumerate() {
        let base_time = step_duration * step as u32;
        
        for (idx, &feature) in features.iter().enumerate() {
            let mut spike_time = encode_feature(feature, TAU);
            
            // Apply temporal dependency
            if let Some(prev) = previous_features.as_ref() {
                let prev_value = prev.get(idx).unwrap_or(&0.0);
                let dependency_factor = 1.0 - (dependency_strength * prev_value);
                spike_time = spike_time.mul_f32(dependency_factor);
            }
            
            all_events.push(SpikeEvent {
                neuron_id: idx as u32,
                timestamp: base_time + spike_time,
                amplitude: feature,
                frequency: calculate_frequency(feature),
            });
        }
        
        previous_features = Some(features.clone());
    }
    
    SpikePattern::new(all_events)
}
```

### 4.4 Similarity Metrics
#### 4.4.1 Victor-Purpura Distance
Measures spike train similarity considering timing precision:

```rust
fn victor_purpura_distance(
    pattern1: &SpikePattern,
    pattern2: &SpikePattern,
    q: f32  // Timing precision parameter
) -> f32 {
    // Implementation of VP distance algorithm
    // Returns distance in range [0, ∞)
}
```

#### 4.4.2 Van Rossum Distance
Measures spike train similarity with exponential kernel:

```rust
fn van_rossum_distance(
    pattern1: &SpikePattern,
    pattern2: &SpikePattern,
    tau: f32  // Time constant
) -> f32 {
    // Implementation of VR distance algorithm
    // Returns distance in range [0, ∞)
}
```

## 5. Implementation Standards

### 5.1 Naming Conventions
- **Structs**: PascalCase (e.g., `TtfsEncoder`, `SpikePattern`)
- **Functions**: snake_case (e.g., `encode_features`, `calculate_similarity`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `MAX_SPIKE_TIME`, `DEFAULT_TAU`)
- **Modules**: snake_case (e.g., `ttfs_concept`, `spike_pattern`)

### 5.2 Error Handling
```rust
#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("Invalid feature value: {0}")]
    InvalidFeature(f32),
    
    #[error("Empty feature vector")]
    EmptyFeatures,
    
    #[error("Configuration error: {0}")]
    InvalidConfig(String),
    
    #[error("Numerical overflow in calculation")]
    NumericalOverflow,
}
```

### 5.3 Performance Requirements
- **Single encoding**: < 1ms for 128-dimensional feature vector
- **Batch encoding**: < 2ms average per item for 100 items
- **Similarity calculation**: < 100μs with caching
- **Pattern validation**: < 50μs
- **Memory usage**: < 1KB per spike pattern (typical)

## 6. Testing Requirements

### 6.1 Unit Test Coverage
Minimum 90% line coverage for all TTFS modules with tests for:
- Edge cases (empty inputs, extreme values)
- Mathematical correctness
- Performance benchmarks
- Error handling

### 6.2 Integration Scenarios
- End-to-end concept encoding pipeline
- Parallel encoding with multiple threads
- Large-scale batch processing
- Real-time encoding constraints

### 6.3 Performance Benchmarks
```rust
#[bench]
fn bench_single_encoding(b: &mut Bencher) {
    let features = vec![0.5; 128];
    let encoder = TtfsEncoder::default();
    
    b.iter(|| {
        let pattern = encoder.encode(&features);
        black_box(pattern);
    });
}
```

Target benchmarks:
- 10,000 encodings/second for 128-dim vectors
- < 100MB memory for 10,000 patterns
- < 5% CPU usage for continuous encoding

## 7. Validation Rules

### 7.1 Pattern Validation
```rust
fn validate_pattern(pattern: &SpikePattern) -> Result<(), ValidationError> {
    // Check temporal ordering
    for window in pattern.events.windows(2) {
        if window[1].timestamp < window[0].timestamp {
            return Err(ValidationError::TemporalOrdering);
        }
    }
    
    // Check refractory period
    for events in pattern.events.group_by(|e| e.neuron_id) {
        for pair in events.windows(2) {
            let interval = pair[1].timestamp - pair[0].timestamp;
            if interval < REFRACTORY_PERIOD {
                return Err(ValidationError::RefractoryViolation);
            }
        }
    }
    
    // Check amplitude range
    for event in &pattern.events {
        if event.amplitude < 0.0 || event.amplitude > 1.0 {
            return Err(ValidationError::AmplitudeRange);
        }
    }
    
    Ok(())
}
```

### 7.2 Encoding Validation
- Features must be normalized to [0,1] or [-1,1]
- Empty feature vectors are invalid
- NaN or infinite values are rejected
- Pattern complexity must be finite

## 8. Usage Examples

### 8.1 Basic Encoding
```rust
use neuromorphic_core::ttfs_concept::{TtfsEncoder, EncodingConfig};

let encoder = TtfsEncoder::default();
let features = vec![0.8, 0.6, 0.4, 0.2];
let pattern = encoder.encode(&features);

println!("First spike at: {:?}", pattern.first_spike_time());
println!("Pattern complexity: {}", pattern.complexity);
```

### 8.2 Population Coding
```rust
let config = EncodingConfig {
    use_population_coding: true,
    neurons_per_feature: 5,
    ..Default::default()
};

let encoder = TtfsEncoder::new(config);
let pattern = encoder.encode(&features);

// Pattern will have 5x more spike events
assert_eq!(pattern.events.len(), features.len() * 5);
```

### 8.3 Temporal Sequence
```rust
let sequence = vec![
    vec![0.8, 0.6],
    vec![0.7, 0.7],
    vec![0.6, 0.8],
];

let step_duration = Duration::from_millis(20);
let pattern = encoder.encode_sequence(&sequence, step_duration);

// Pattern spans full sequence duration
assert!(pattern.duration >= step_duration * 2);
```

## 9. Future Extensions

### 9.1 Planned Features
- Adaptive time constants based on input statistics
- Hierarchical TTFS encoding for complex patterns
- Hardware acceleration support (GPU/FPGA)
- Online learning of optimal encoding parameters

### 9.2 Research Directions
- Integration with deep learning frameworks
- Quantum-inspired spike encoding
- Energy-efficient encoding schemes
- Neuromorphic hardware optimization

## 10. References

1. Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based strategies for rapid processing.
2. Bohte, S. M., Kok, J. N., & La Poutré, H. (2002). Error-backpropagation in temporally encoded networks of spiking neurons.
3. Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions.
4. Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity.

---

**Version**: 1.0.0  
**Last Updated**: 2025-08-03  
**Status**: ACTIVE  
**Compliance**: 100% with LLMKG neuromorphic architecture