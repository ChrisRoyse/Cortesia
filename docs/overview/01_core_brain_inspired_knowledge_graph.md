# Core Brain-Inspired Knowledge Graph System

## Overview

The LLMKG (Lightning-fast Knowledge Graph) core system implements a revolutionary brain-inspired approach to knowledge representation and processing. This system combines traditional knowledge graph concepts with neural network principles to create a highly efficient, biologically-inspired knowledge storage and retrieval system optimized for LLM integration.

## Architecture Components

### 1. Brain-Inspired Entity System (`brain_types.rs`)

The core entity system models knowledge as biological neurons with activation states, temporal tracking, and directional properties.

#### Entity Direction Types
- **Input Entities**: Represent concept input nodes (sensory/input layer)
- **Output Entities**: Represent concept output nodes (motor/output layer)
- **Gate Entities**: Represent logic processing nodes (hidden layer)

#### BrainInspiredEntity Structure
```rust
pub struct BrainInspiredEntity {
    pub id: EntityKey,
    pub concept_id: String,          // Canonical concept identifier
    pub direction: EntityDirection,   // in/out/gate classification
    pub properties: HashMap<String, AttributeValue>,
    pub embedding: Vec<f32>,
    pub activation_state: f32,       // Current activation level (0.0-1.0)
    pub last_activation: SystemTime, // Temporal decay tracking
}
```

**Key Features:**
- **Temporal Decay**: Entities naturally decay over time unless reinforced
- **Activation States**: Dynamic activation levels that propagate through the network
- **Directional Processing**: Input/output/gate classification for structured computation
- **Embedding Integration**: High-dimensional vector representations for similarity

### 2. Logic Gate System

The system implements biological-inspired logic gates for neural computation:

#### Gate Types
- **AND Gate**: All inputs must exceed threshold (conjunction)
- **OR Gate**: Any input exceeding threshold activates (disjunction)
- **NOT Gate**: Inverts single input (negation)
- **Inhibitory Gate**: Primary input minus inhibitory inputs
- **Weighted Gate**: Weighted sum with configurable threshold

#### LogicGate Structure
```rust
pub struct LogicGate {
    pub gate_id: EntityKey,
    pub gate_type: LogicGateType,
    pub input_nodes: Vec<EntityKey>,
    pub output_nodes: Vec<EntityKey>,
    pub threshold: f32,
    pub weight_matrix: Vec<f32>,
}
```

**Computational Model:**
- Dynamic threshold-based activation
- Weighted input processing
- Multi-input/multi-output connections
- Configurable gate behaviors

### 3. Brain-Inspired Relationship System

Relationships model synaptic connections between entities with advanced properties:

#### BrainInspiredRelationship Structure
```rust
pub struct BrainInspiredRelationship {
    pub source: EntityKey,
    pub target: EntityKey,
    pub relation_type: RelationType,
    pub weight: f32,                    // Dynamic weight (0.0-1.0)
    pub is_inhibitory: bool,            // Inhibitory connection flag
    pub temporal_decay: f32,            // Decay rate (0.0-1.0)
    pub last_strengthened: SystemTime,  // Hebbian learning timestamp
    pub activation_count: u64,          // Usage frequency
    pub creation_time: SystemTime,      // Bi-temporal tracking
    pub ingestion_time: SystemTime,     // When added to system
}
```

**Advanced Features:**
- **Hebbian Learning**: Connections strengthen with use
- **Temporal Decay**: Unused connections weaken over time
- **Inhibitory Connections**: Negative reinforcement pathways
- **Bi-temporal Tracking**: Creation vs. ingestion time separation
- **Usage Statistics**: Frequency-based optimization

### 4. Relation Types

The system supports structured relationship types for semantic understanding:

```rust
pub enum RelationType {
    IsA,            // Inheritance relationship
    HasInstance,    // Instance relationship
    HasProperty,    // Property relationship
    RelatedTo,      // General association
    PartOf,         // Part-whole relationship
    Similar,        // Similarity relationship
    Opposite,       // Opposition relationship
    Temporal,       // Temporal relationship
}
```

### 5. Brain-Enhanced Knowledge Graph (`brain_enhanced_graph.rs`)

The main brain-enhanced graph provides a unified interface combining traditional knowledge graphs with brain-inspired computation:

#### Core Architecture
```rust
pub struct BrainEnhancedKnowledgeGraph {
    pub base_graph: Arc<RwLock<KnowledgeGraph>>,
    pub temporal_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
    pub brain_entities: Arc<RwLock<SlotMap<EntityKey, BrainInspiredEntity>>>,
    pub logic_gates: Arc<RwLock<SlotMap<EntityKey, LogicGate>>>,
    pub brain_relationships: Arc<RwLock<AHashMap<(EntityKey, EntityKey), BrainInspiredRelationship>>>,
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub sdr_storage: Arc<RwLock<SDRStorage>>,
    pub entity_mapping: Arc<RwLock<AHashMap<EntityKey, EntityKey>>>,
    pub config: BrainEnhancedConfig,
}
```

#### Key Capabilities

**1. Concept Structure Creation**
- Automatically creates input/output node pairs
- Connects them via logic gates
- Embeds semantic vectors
- Establishes neural pathways

**2. Neural Query Processing**
- Converts text queries to activation patterns
- Propagates activation through the network
- Finds inherited properties through relationships
- Returns structured reasoning results

**3. Temporal Integration**
- Bi-temporal storage (creation time vs. ingestion time)
- Temporal decay modeling
- Time-based query capabilities
- Historical state tracking

**4. Backward Compatibility**
- Maintains traditional knowledge graph interface
- Automatic entity mapping
- Seamless integration with existing systems
- Progressive enhancement capabilities

### 6. Activation Propagation Engine (`activation_engine.rs`)

The activation engine simulates neural network propagation through the knowledge graph:

#### Core Process
1. **Entity Activation**: Update entity states based on incoming connections
2. **Logic Gate Processing**: Calculate gate outputs and propagate to connected entities
3. **Inhibitory Processing**: Apply negative reinforcement through inhibitory connections
4. **Temporal Decay**: Apply time-based decay to all activations

#### Activation Configuration
```rust
pub struct ActivationConfig {
    pub max_iterations: usize,          // Maximum propagation cycles
    pub convergence_threshold: f32,     // Stability threshold
    pub decay_rate: f32,                // Temporal decay speed
    pub inhibition_strength: f32,       // Inhibitory connection strength
    pub default_threshold: f32,         // Default gate threshold
}
```

#### Propagation Results
```rust
pub struct PropagationResult {
    pub final_activations: HashMap<EntityKey, f32>,
    pub iterations_completed: usize,
    pub converged: bool,
    pub activation_trace: Vec<ActivationStep>,
    pub total_energy: f32,
}
```

### 7. Knowledge Engine (`knowledge_engine.rs`)

The knowledge engine provides ultra-fast triple storage and retrieval optimized for LLM integration:

#### Core Features
- **Triple Storage**: Subject-Predicate-Object storage with indexing
- **Semantic Search**: Embedding-based similarity search
- **Entity Relationships**: Multi-hop relationship traversal
- **Memory Management**: Automatic eviction of low-quality nodes
- **LLM Optimization**: Predicate suggestions and entity context

#### Triple Processing
- Automatic embedding generation
- Predicate normalization
- Quality scoring
- Index maintenance

#### Search Capabilities
- SPO pattern matching
- Semantic similarity search
- Entity relationship exploration
- Context-aware results

## System Integration

### Memory Architecture
The system uses multiple storage layers:
1. **SlotMap Storage**: Efficient entity/gate storage with stable keys
2. **Hash-based Indexing**: Fast SPO lookup tables
3. **Temporal Storage**: Bi-temporal knowledge tracking
4. **SDR Storage**: Sparse distributed representation for similarity

### Performance Optimizations
- **Zero-copy Operations**: Minimizes memory allocations
- **Concurrent Processing**: Async/await throughout
- **Efficient Indexing**: Multiple index structures for fast lookup
- **Memory Limits**: Configurable memory bounds with automatic eviction
- **Batch Processing**: Optimized for bulk operations

### Configuration Management
```rust
pub struct BrainEnhancedConfig {
    pub embedding_dim: usize,
    pub activation_config: ActivationConfig,
    pub sdr_config: SDRConfig,
    pub enable_temporal_tracking: bool,
    pub enable_sdr_storage: bool,
}
```

## Usage Patterns

### Basic Entity Creation
```rust
// Create input entity
let input_entity = BrainInspiredEntity::new("dog".to_string(), EntityDirection::Input);
let entity_key = graph.insert_brain_entity(input_entity).await?;

// Create relationship
let relationship = BrainInspiredRelationship::new(
    source_key, target_key, RelationType::IsA
);
graph.insert_brain_relationship(relationship).await?;
```

### Neural Query Processing
```rust
// Query with neural activation
let result = graph.neural_query("What is a dog?").await?;
// Returns activated concepts and reasoning trace
```

### Concept Structure Creation
```rust
// Create structured concept with gates
let structure = graph.create_concept_structure(
    "animal".to_string(),
    embedding_vector,
).await?;
```

## Advanced Features

### Hebbian Learning
- Connections strengthen with repeated activation
- Temporal decay for unused connections
- Adaptive weight adjustment
- Learning rate configuration

### Inhibitory Networks
- Negative reinforcement pathways
- Competitive activation
- Attention mechanisms
- Conflict resolution

### Temporal Processing
- Time-based decay modeling
- Historical state queries
- Temporal relationship tracking
- Bi-temporal data management

## Performance Characteristics

### Memory Usage
- Target: <60 bytes per entity
- Configurable memory limits
- Automatic garbage collection
- Quality-based eviction

### Query Performance
- Sub-millisecond entity lookup
- Parallel activation propagation
- Efficient indexing structures
- Batched processing optimization

### Scalability
- Concurrent read/write operations
- Distributed storage capability
- Horizontal scaling support
- Memory-efficient data structures

## Integration Points

### LLM Integration
- Semantic embedding support
- Natural language query processing
- Context-aware result formatting
- Predicate suggestion system

### Traditional Knowledge Graphs
- Backward compatibility layer
- Automatic entity mapping
- Progressive enhancement
- Seamless migration path

### Temporal Systems
- Bi-temporal data support
- Time-based queries
- Historical state tracking
- Temporal relationship modeling

This brain-inspired knowledge graph system represents a significant advancement in knowledge representation, combining the efficiency of traditional graphs with the dynamic processing capabilities of neural networks, all optimized for modern LLM integration requirements.