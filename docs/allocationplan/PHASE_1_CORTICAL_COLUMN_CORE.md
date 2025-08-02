# Phase 1: Spiking Neural Cortical Column Core

**Duration**: 1 week  
**Team Size**: 2-3 neuromorphic developers  
**Methodology**: SPARC + London School TDD + SNN Validation  
**Goal**: Implement production-ready spiking neural cortical columns with TTFS encoding and lateral inhibition  

## AI-Verifiable Success Criteria

### Neuromorphic Performance Metrics
- [ ] TTFS allocation: < 1ms (p99) using time-to-first-spike encoding
- [ ] Lateral inhibition convergence: < 500μs (p99) with winner-take-all
- [ ] Memory overhead per spiking column: < 512 bytes
- [ ] SIMD-accelerated allocations: > 10,000/second
- [ ] Spike timing precision: ±10μs accuracy
- [ ] Refractory period compliance: 100% enforcement

### Neuromorphic Functional Requirements
- [ ] 100% of TTFS allocations succeed or return neurobiological error
- [ ] Lateral inhibition prevents 100% of duplicate spike allocations
- [ ] Spike timing follows exact TTFS coding principles
- [ ] Refractory periods prevent temporal conflicts
- [ ] SIMD operations achieve 4x parallel processing speedup
- [ ] All spike state transitions are atomic and verifiable

## SPARC Methodology Application

### Specification

**Objective**: Create a neuromorphically-accurate spiking neural cortical column system that allocates concepts through time-to-first-spike competitive dynamics.

**Spiking Neural Model**:
```
Real Cortex → SNN Implementation
- 110 neurons/column → 1 SpikingCorticalColumn with 256 virtual neurons
- Spike timing → TTFS encoding (sub-millisecond precision)
- Lateral connections → Inhibitory synaptic networks
- Refractory periods → Temporal competition prevention
- Membrane potentials → SIMD-accelerated activation states
- Neural adaptation → Spike-timing-dependent plasticity (STDP)
```

**Neuromorphic Core Requirements**:
1. Each spiking column can encode exactly one TTFS concept
2. Columns compete through spike timing (earlier spike wins)
3. Lateral inhibition enforces winner-take-all through synaptic suppression
4. Spike patterns decay following neurobiological rules
5. Strong spike correlations strengthen inhibitory connections
6. Refractory periods prevent temporal conflicts
7. SIMD acceleration processes multiple spikes in parallel

### Pseudocode

```
SPIKING_NEURAL_ALLOCATION:
    INPUT: ttfs_concept, spiking_cortical_grid
    OUTPUT: allocated_column_id OR neuromorphic_error
    
    // Phase 1: TTFS Encoding
    spike_pattern = encode_concept_to_ttfs(ttfs_concept)
    candidate_columns = find_receptive_columns(spike_pattern, grid)
    
    // Phase 2: Parallel Spike Competition
    PARALLEL FOR EACH column IN candidate_columns:
        spike_response = column.process_spike_input(spike_pattern)
        column.set_membrane_potential(spike_response.potential)
        IF spike_response.exceeds_threshold():
            column.queue_spike(spike_response.spike_time)
    
    // Phase 3: Temporal Winner-Take-All
    first_spike_column = find_earliest_spike(candidate_columns)
    
    // Phase 4: Lateral Inhibition Propagation
    SIMD_PARALLEL inhibit_competing_columns(first_spike_column.neighbors)
    
    // Phase 5: Refractory Allocation
    IF first_spike_column.allocate_during_spike(ttfs_concept):
        first_spike_column.enter_refractory_period()
        strengthen_inhibitory_synapses(first_spike_column)
        RETURN first_spike_column.id
    ELSE:
        RETURN NeuromorphicError::RefractoryConflict
```

### Architecture

```
neuromorphic-core/
├── src/
│   ├── spiking_column/
│   │   ├── mod.rs
│   │   ├── neural_state.rs      # Spike state machine
│   │   ├── ttfs_dynamics.rs     # Time-to-First-Spike processing
│   │   ├── membrane.rs          # Membrane potential simulation
│   │   ├── refractory.rs        # Refractory period management
│   │   └── simd_metrics.rs      # SIMD performance tracking
│   ├── cortical_grid/
│   │   ├── mod.rs
│   │   ├── topology.rs          # 3D neuromorphic grid
│   │   ├── receptive_fields.rs  # Spatial spike reception
│   │   ├── simd_parallel.rs     # SIMD concurrent operations
│   │   └── spike_routing.rs     # Spike propagation paths
│   ├── lateral_inhibition/
│   │   ├── mod.rs
│   │   ├── inhibitory_synapses.rs  # Synaptic inhibition
│   │   ├── winner_take_all.rs      # Temporal competition
│   │   ├── spike_suppression.rs    # Spike blocking mechanisms
│   │   └── stdp_learning.rs        # Synaptic plasticity
│   ├── ttfs_allocation/
│   │   ├── mod.rs
│   │   ├── snn_engine.rs        # Spiking neural allocation
│   │   ├── temporal_coding.rs   # TTFS encoding/decoding
│   │   ├── simd_processing.rs   # WASM SIMD acceleration
│   │   └── neuromorphic_strategies.rs # Bio-inspired allocation
│   └── wasm_simd/
│       ├── mod.rs
│       ├── spike_simd.rs        # SIMD spike operations
│       ├── vector_ops.rs        # 128-bit vector processing
│       └── parallel_inhibition.rs # Parallel lateral inhibition
```

### Refinement

Iterative improvements based on benchmarks:
1. Start with naive O(n) search
2. Add spatial indexing for O(log n)
3. Implement SIMD for activation calculations
4. Add lock-free data structures
5. Profile and optimize hot paths

### Completion

Phase complete when ALL metrics pass:
- Performance benchmarks green
- Stress tests pass without errors
- API documentation complete
- Integration tests cover all paths

## Task Breakdown

### Task 1.1: Column State Machine (Day 1)

**Specification**: Implement thread-safe state transitions

**Test-Driven Development**:

```rust
// tests/column_state_test.rs
#[test]
fn test_column_state_transitions() {
    let column = CorticalColumn::new(1);
    
    // Verify initial state
    assert_eq!(column.state(), ColumnState::Available);
    assert_eq!(column.activation_level(), 0.0);
    
    // Test activation
    column.activate(0.8);
    assert_eq!(column.state(), ColumnState::Activated);
    assert_eq!(column.activation_level(), 0.8);
    
    // Test allocation
    let concept = Concept::new("dog");
    assert!(column.try_allocate(concept.clone()).is_ok());
    assert_eq!(column.state(), ColumnState::Allocated);
    
    // Test double allocation prevention
    let concept2 = Concept::new("cat");
    assert!(matches!(
        column.try_allocate(concept2),
        Err(AllocationError::AlreadyAllocated)
    ));
}

#[test]
fn test_concurrent_state_safety() {
    let column = Arc::new(CorticalColumn::new(1));
    let mut handles = vec![];
    
    // Spawn 100 threads trying to allocate
    for i in 0..100 {
        let col = column.clone();
        handles.push(thread::spawn(move || {
            let concept = Concept::new(format!("concept_{}", i));
            col.try_allocate(concept)
        }));
    }
    
    // Collect results
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Exactly one should succeed
    let successes = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(successes, 1);
}
```

**Implementation**:

```rust
// src/column/state.rs
use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnState {
    Available = 0,
    Activated = 1,
    Competing = 2,
    Allocated = 3,
    Refractory = 4,
}

impl ColumnState {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Available,
            1 => Self::Activated,
            2 => Self::Competing,
            3 => Self::Allocated,
            4 => Self::Refractory,
            _ => unreachable!(),
        }
    }
}

pub struct AtomicState(AtomicU8);

impl AtomicState {
    pub fn new(state: ColumnState) -> Self {
        Self(AtomicU8::new(state as u8))
    }
    
    pub fn load(&self) -> ColumnState {
        ColumnState::from_u8(self.0.load(Ordering::Acquire))
    }
    
    pub fn compare_exchange(
        &self,
        current: ColumnState,
        new: ColumnState,
    ) -> Result<ColumnState, ColumnState> {
        match self.0.compare_exchange(
            current as u8,
            new as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(v) => Ok(ColumnState::from_u8(v)),
            Err(v) => Err(ColumnState::from_u8(v)),
        }
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] All state transitions pass atomicity tests
- [ ] No invalid state transitions possible
- [ ] Concurrent access test passes 100/100 times
- [ ] State machine diagram generated and validated

### Task 1.2: Activation Dynamics (Day 2)

**Specification**: Implement biologically-inspired activation

**Test First**:

```rust
#[test]
fn test_activation_decay() {
    let column = CorticalColumn::new(1);
    
    // Set initial activation
    column.set_activation(1.0);
    assert!((column.activation_level() - 1.0).abs() < f32::EPSILON);
    
    // Apply decay (tau = 100ms)
    column.apply_decay(Duration::from_millis(50));
    let expected = 1.0 * (0.5_f32).exp(); // e^(-t/tau)
    assert!((column.activation_level() - expected).abs() < 0.01);
    
    // Full decay after 5 tau
    column.apply_decay(Duration::from_millis(450));
    assert!(column.activation_level() < 0.01);
}

#[test]
fn test_hebbian_strengthening() {
    let mut column = CorticalColumn::new(1);
    let neighbor_id = 2;
    
    // Initial connection
    column.add_lateral_connection(neighbor_id, 0.5);
    
    // Co-activation strengthens connection
    column.strengthen_connection(neighbor_id, 0.8);
    
    let strength = column.connection_strength_to(neighbor_id).unwrap();
    assert!(strength > 0.5);
    assert!(strength <= 1.0);
}
```

**Implementation**:

```rust
// src/column/activation.rs
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub struct ActivationDynamics {
    level: AtomicU32,
    last_update: RwLock<Instant>,
    tau_ms: f32,
}

impl ActivationDynamics {
    pub fn new(tau_ms: f32) -> Self {
        Self {
            level: AtomicU32::new(0),
            last_update: RwLock::new(Instant::now()),
            tau_ms,
        }
    }
    
    pub fn set_activation(&self, level: f32) {
        debug_assert!(level >= 0.0 && level <= 1.0);
        self.level.store(level.to_bits(), Ordering::Release);
        *self.last_update.write() = Instant::now();
    }
    
    pub fn get_activation(&self) -> f32 {
        self.apply_decay();
        f32::from_bits(self.level.load(Ordering::Acquire))
    }
    
    fn apply_decay(&self) {
        let now = Instant::now();
        let last = *self.last_update.read();
        let dt = now.duration_since(last).as_secs_f32() * 1000.0;
        
        if dt > 0.1 {
            let current = f32::from_bits(self.level.load(Ordering::Acquire));
            let decayed = current * (-dt / self.tau_ms).exp();
            
            self.level.store(decayed.to_bits(), Ordering::Release);
            *self.last_update.write() = now;
        }
    }
    
    pub fn strengthen(&self, amount: f32) {
        let current = self.get_activation();
        let new = (current + amount * (1.0 - current)).min(1.0);
        self.set_activation(new);
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Decay follows exponential curve within 1% error
- [ ] Activation stays in [0, 1] range for all operations
- [ ] Strengthening converges to 1.0 asymptotically
- [ ] 1M activation updates complete in < 100ms

### Task 1.3: Lateral Inhibition Network (Day 3)

**Specification**: Implement winner-take-all dynamics

**Test-Driven Approach**:

```rust
#[test]
fn test_lateral_inhibition_winner() {
    let mut network = LateralInhibitionNetwork::new();
    
    // Create competing columns
    let activations = vec![
        (ColumnId(1), 0.6),
        (ColumnId(2), 0.8), // Winner
        (ColumnId(3), 0.7),
    ];
    
    let winner = network.compete(activations.clone());
    assert_eq!(winner, ColumnId(2));
    
    // Verify inhibition applied
    for (id, original) in activations {
        if id == winner {
            assert_eq!(network.get_activation(id), original);
        } else {
            assert!(network.get_activation(id) < original);
        }
    }
}

#[test]
fn test_inhibition_prevents_duplicates() {
    let network = LateralInhibitionNetwork::new();
    let concept = "test_concept";
    
    // First allocation succeeds
    let col1 = network.allocate_for_concept(concept);
    assert!(col1.is_ok());
    
    // Second allocation fails due to inhibition
    let col2 = network.allocate_for_concept(concept);
    assert!(matches!(col2, Err(AllocationError::ConceptExists)));
}
```

**Implementation**:

```rust
// src/inhibition/lateral.rs
use dashmap::DashMap;

pub struct LateralInhibitionNetwork {
    connections: DashMap<ColumnId, Vec<(ColumnId, f32)>>,
    concept_map: DashMap<String, ColumnId>,
    inhibition_radius: usize,
    inhibition_strength: f32,
}

impl LateralInhibitionNetwork {
    pub fn compete(&self, candidates: Vec<(ColumnId, f32)>) -> ColumnId {
        // Find winner
        let winner = candidates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        
        // Apply lateral inhibition
        for (id, activation) in &candidates {
            if *id != winner.0 {
                self.inhibit_column(*id, self.inhibition_strength);
            }
        }
        
        winner.0
    }
    
    pub fn allocate_for_concept(&self, concept: &str) -> Result<ColumnId, AllocationError> {
        // Check if concept already allocated
        if let Some(existing) = self.concept_map.get(concept) {
            return Err(AllocationError::ConceptExists);
        }
        
        // Find best column through competition
        let candidates = self.find_candidate_columns(concept);
        let winner = self.compete(candidates);
        
        // Record allocation
        self.concept_map.insert(concept.to_string(), winner);
        Ok(winner)
    }
    
    fn inhibit_column(&self, column: ColumnId, strength: f32) {
        // Implement inhibition logic
        // This affects the column's future activation potential
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Winner-take-all selects highest activation 100% of time
- [ ] Inhibition reduces loser activation by exactly 0.3
- [ ] No concept allocated to multiple columns (0 duplicates in 10K tests)
- [ ] Inhibition radius affects exactly N neighbors

### Task 1.4: Spatial Grid Topology (Day 4)

**Specification**: Create 3D cortical grid with efficient search

**Tests First**:

```rust
#[test]
fn test_grid_topology() {
    let grid = CorticalGrid::new(10, 10, 6); // 10x10x6 grid
    
    assert_eq!(grid.total_columns(), 600);
    assert_eq!(grid.dimensions(), (10, 10, 6));
    
    // Test neighbor finding
    let pos = GridPosition::new(5, 5, 3);
    let neighbors = grid.get_neighbors(pos, 1);
    assert_eq!(neighbors.len(), 26); // 3x3x3 - 1 (self)
}

#[test]
fn test_spatial_search_performance() {
    let grid = CorticalGrid::new(100, 100, 6);
    let concept = Concept::new("test");
    
    let start = Instant::now();
    let candidates = grid.find_candidate_columns(&concept, 10);
    let elapsed = start.elapsed();
    
    assert_eq!(candidates.len(), 10);
    assert!(elapsed < Duration::from_millis(1)); // Sub-millisecond
}
```

**Implementation**:

```rust
// src/grid/topology.rs
pub struct CorticalGrid {
    dimensions: (usize, usize, usize),
    columns: Vec<Vec<Vec<CorticalColumn>>>,
    spatial_index: KdTree<ColumnId>,
}

impl CorticalGrid {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        let mut columns = Vec::with_capacity(x);
        let mut spatial_index = KdTree::new();
        
        for i in 0..x {
            let mut y_vec = Vec::with_capacity(y);
            for j in 0..y {
                let mut z_vec = Vec::with_capacity(z);
                for k in 0..z {
                    let id = Self::position_to_id(i, j, k, y, z);
                    let column = CorticalColumn::new(id);
                    z_vec.push(column);
                    
                    // Add to spatial index
                    let point = [i as f32, j as f32, k as f32];
                    spatial_index.add(point, id);
                }
                y_vec.push(z_vec);
            }
            columns.push(y_vec);
        }
        
        Self {
            dimensions: (x, y, z),
            columns,
            spatial_index,
        }
    }
    
    pub fn find_nearest_available(&self, position: GridPosition, radius: usize) -> Vec<ColumnId> {
        let point = [position.x as f32, position.y as f32, position.z as f32];
        let nearby = self.spatial_index.within_radius(&point, radius as f32);
        
        nearby.into_iter()
            .filter(|(_, id)| {
                let col = self.get_column(*id);
                col.state() == ColumnState::Available
            })
            .map(|(_, id)| id)
            .collect()
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Grid initialization < 10ms for 1M columns
- [ ] Spatial search < 1ms for radius 10
- [ ] Memory usage = columns * 1KB ± 5%
- [ ] All columns reachable via position mapping

### Task 1.5: Parallel Allocation Engine (Day 5)

**Specification**: Handle concurrent allocations efficiently

**Test-Driven Development**:

```rust
#[test]
fn test_parallel_allocation_throughput() {
    let engine = AllocationEngine::new();
    let concepts: Vec<_> = (0..10000)
        .map(|i| Concept::new(format!("concept_{}", i)))
        .collect();
    
    let start = Instant::now();
    let results: Vec<_> = concepts.par_iter()
        .map(|c| engine.allocate(c.clone()))
        .collect();
    let elapsed = start.elapsed();
    
    // All succeed
    assert!(results.iter().all(|r| r.is_ok()));
    
    // Meet throughput target
    let throughput = 10000.0 / elapsed.as_secs_f32();
    assert!(throughput > 1000.0); // >1000 allocations/second
}

#[test]
fn test_no_race_conditions() {
    let engine = Arc::new(AllocationEngine::new());
    let barrier = Arc::new(Barrier::new(100));
    
    let handles: Vec<_> = (0..100).map(|i| {
        let eng = engine.clone();
        let bar = barrier.clone();
        
        thread::spawn(move || {
            bar.wait(); // All threads start together
            eng.allocate(Concept::new("same_concept"))
        })
    }).collect();
    
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Exactly one success
    let successes = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(successes, 1);
}
```

**Implementation**:

```rust
// src/allocation/engine.rs
use rayon::prelude::*;
use crossbeam::queue::SegQueue;

pub struct AllocationEngine {
    grid: Arc<CorticalGrid>,
    inhibition: Arc<LateralInhibitionNetwork>,
    pending_queue: SegQueue<AllocationRequest>,
    metrics: AllocationMetrics,
}

impl AllocationEngine {
    pub fn allocate(&self, concept: Concept) -> Result<ColumnId, AllocationError> {
        let start = Instant::now();
        
        // Phase 1: Find candidates
        let candidates = self.find_candidates(&concept)?;
        
        // Phase 2: Score candidates
        let scored = candidates.par_iter()
            .map(|&col_id| {
                let score = self.score_candidate(col_id, &concept);
                (col_id, score)
            })
            .collect::<Vec<_>>();
        
        // Phase 3: Competition via lateral inhibition
        let winner = self.inhibition.compete(scored);
        
        // Phase 4: Atomic allocation
        let column = self.grid.get_column_mut(winner);
        match column.try_allocate(concept) {
            Ok(_) => {
                self.metrics.record_success(start.elapsed());
                Ok(winner)
            }
            Err(e) => {
                self.metrics.record_failure();
                Err(e)
            }
        }
    }
    
    fn score_candidate(&self, column_id: ColumnId, concept: &Concept) -> f32 {
        // Scoring based on:
        // 1. Semantic similarity
        // 2. Spatial locality
        // 3. Current activation
        // 4. Connection strength
        
        let column = self.grid.get_column(column_id);
        let base_score = self.calculate_semantic_similarity(column, concept);
        let activation_bonus = column.activation_level() * 0.2;
        let locality_bonus = self.calculate_spatial_locality(column_id, concept) * 0.3;
        
        (base_score + activation_bonus + locality_bonus).min(1.0)
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] 1000+ allocations/second sustained
- [ ] P99 latency < 5ms
- [ ] Zero race conditions in 100K operations
- [ ] Memory stable under continuous load

### Task 1.6: Integration and Benchmarks (Day 5)

**Specification**: Verify complete system meets all targets

**Comprehensive Tests**:

```rust
// benches/cortical_core_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn allocation_benchmark(c: &mut Criterion) {
    let engine = AllocationEngine::new();
    
    c.bench_function("single_allocation_p99", |b| {
        b.iter(|| {
            let concept = Concept::new(black_box("test_concept"));
            engine.allocate(concept)
        });
    });
    
    c.bench_function("parallel_allocation_1k", |b| {
        let concepts: Vec<_> = (0..1000)
            .map(|i| Concept::new(format!("concept_{}", i)))
            .collect();
        
        b.iter(|| {
            concepts.par_iter()
                .map(|c| engine.allocate(c.clone()))
                .collect::<Vec<_>>()
        });
    });
    
    c.bench_function("lateral_inhibition", |b| {
        let network = LateralInhibitionNetwork::new();
        let activations = vec![
            (ColumnId(1), 0.5),
            (ColumnId(2), 0.7),
            (ColumnId(3), 0.6),
        ];
        
        b.iter(|| {
            network.compete(black_box(activations.clone()))
        });
    });
}

criterion_group!(benches, allocation_benchmark);
criterion_main!(benches);
```

**AI-Verifiable Outcomes**:
- [ ] All benchmarks pass performance targets
- [ ] Memory usage within 10% of calculated
- [ ] API documentation coverage 100%
- [ ] Integration tests pass 100%

## Phase 1 Deliverables

### Code Artifacts
1. **Cortical Column Implementation**
   - State machine with atomic transitions
   - Activation dynamics with decay
   - Thread-safe allocation methods

2. **Lateral Inhibition Network**
   - Winner-take-all competition
   - Concept deduplication
   - Configurable inhibition radius

3. **Spatial Grid System**
   - 3D topology with efficient indexing
   - KD-tree for spatial queries
   - Parallel-safe operations

4. **Allocation Engine**
   - Multi-phase allocation process
   - Semantic + spatial scoring
   - Performance metrics tracking

### Performance Verification
```bash
cargo bench --bench cortical_core_bench

# Expected output:
# single_allocation_p99   time:   [4.2 ms 4.5 ms 4.8 ms]
# parallel_allocation_1k  time:   [892 ms 910 ms 925 ms]
# lateral_inhibition      time:   [1.8 ms 1.9 ms 2.0 ms]
```

### Documentation
- Architecture diagrams with neuroscience mappings
- API reference with examples
- Performance analysis report
- Integration guide for Phase 2

## Success Checklist

**All items must be checked for phase completion:**

- [ ] Single allocation < 5ms (p99) ✓
- [ ] Lateral inhibition < 2ms ✓
- [ ] 1000+ allocations/second ✓
- [ ] Zero race conditions ✓
- [ ] 100% test coverage ✓
- [ ] Benchmarks automated ✓
- [ ] Documentation complete ✓
- [ ] Code review passed ✓
- [ ] Integration tests green ✓
- [ ] Phase 2 ready to start ✓

## Next Phase Preview

Phase 2 will build the allocation engine with:
- Semantic analysis using small LLMs
- Inheritance hierarchy detection
- Exception handling mechanisms
- Advanced scoring strategies