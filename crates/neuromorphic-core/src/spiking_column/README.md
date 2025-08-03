# Spiking Column Module

## Overview
The spiking_column module implements biologically-inspired cortical columns with Time-to-First-Spike (TTFS) dynamics, lateral inhibition for winner-take-all competition, and 3D spatial organization.

## Components

### 1. SpikingCorticalColumn (`column.rs`)
- **State Machine**: 5 states (Available → Activated → Competing → Allocated → Refractory)
- **Thread-Safe**: Atomic operations, DashMap, RwLock for concurrent access
- **Key Features**:
  - Activation with configurable strength
  - State transitions with validation
  - TTFS spike processing
  - Hebbian learning for connections
  - Multi-factor inhibition checking

### 2. ActivationDynamics (`activation.rs`)
- **Biological Modeling**: Exponential decay (τ = 100ms)
- **Spike Generation**: Threshold at 0.7, 2ms refractory
- **Performance**: Lock-free CAS loops
- **Hebbian Strengthening**: Asymptotic toward 1.0

### 3. LateralInhibitionNetwork (`inhibition.rs`)
- **Winner-Take-All**: Deterministic competition
- **Spatial Inhibition**: Distance-based decay
- **Performance**: <2ms for 100 columns, <10ms for 1000
- **Statistics**: Competition history tracking

### 4. CorticalGrid (`grid.rs`)
- **3D Organization**: Default 10x10x6 grid (600 columns)
- **Spatial Indexing**: O(1) position/ID mapping
- **Neighbor Finding**: Manhattan/Euclidean distance
- **Region Queries**: Rectangular region extraction
- **Performance**: <100ms for 1000 neighbor queries

### 5. State Management (`state.rs`)
- **ColumnState Enum**: 5 states with atomic representation
- **Valid Transitions**: Enforced state machine rules
- **Thread-Safe**: Atomic compare-exchange operations

## Performance Characteristics

### Lateral Inhibition
- 100 columns: <2ms competition
- 400 columns: <10ms competition
- 1000 columns: <10ms competition

### Cortical Grid
- 4000 column initialization: <100ms
- 1000 neighbor queries: <100ms
- 100 region queries (large): <100ms
- Concurrent access: Thread-safe

## Usage Examples

### Basic Column Usage
```rust
let column = SpikingCorticalColumn::new(1);
column.activate_with_strength(0.8)?;
column.start_competing()?;
column.allocate_to_concept("concept".to_string())?;
```

### Grid with Competition
```rust
let grid = CorticalGrid::new(GridConfig::default());
let network = LateralInhibitionNetwork::new(InhibitionConfig::default());

// Setup spatial organization
grid.setup_lateral_connections(2, 0.5);

// Find neighbors
let pos = GridPosition::new(5, 5, 3);
let neighbors = grid.get_neighbors(pos, 2);

// Run competition
let candidates = /* collect active columns */;
let result = network.compete(candidates);
```

## Testing Coverage
- **43 Total Tests**
- Unit tests: 20
- Integration tests: 23
- Performance validated
- Thread-safety verified

## Implementation Status
✅ **Complete**:
- Core spiking column
- Activation dynamics
- Lateral inhibition
- Winner-take-all
- 3D cortical grid
- Spatial queries
- Performance optimization
- Thread safety