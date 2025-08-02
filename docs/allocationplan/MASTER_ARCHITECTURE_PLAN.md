# Master Architecture Plan: CortexKG - Neuromorphic Brain-Inspired Knowledge Graph

**Project Name**: CortexKG (Neuromorphic Cortical Knowledge Graph)  
**Duration**: 12 weeks  
**Team Size**: 4-6 neuromorphic developers  
**Core Principle**: "Where does this memory belong?" - Allocation-first with spiking neural networks
**Neuromorphic Foundation**: Time-to-First-Spike encoding with lateral inhibition  

## Vision

Build a neuromorphic knowledge graph system that truly mimics how the human brain stores, versions, and recalls memories through:
- **Spiking Neural Cortical Columns**: TTFS-encoded hierarchical memory with lateral inhibition
- **Neuromorphic Temporal Versioning**: Memory consolidation with STDP learning
- **Multi-Column Parallel Processing**: Semantic, structural, temporal, and exception columns
- **Cascade Correlation Growth**: Dynamic neural network adaptation
- **SIMD-Accelerated Processing**: 4x parallel speedup with WASM optimization
- **Neuromorphic Hardware Readiness**: Intel Loihi and IBM TrueNorth preparation

## Core Architecture Principles

### 1. Neuromorphic Memory Systems
```
Human Brain → Neuromorphic CortexKG Mapping:
- Cortical Columns → Spiking Neural Allocation Units with TTFS
- Lateral Inhibition → Winner-take-all competition circuits
- Synaptic Plasticity → STDP learning rules and cascade correlation
- Refractory Periods → Temporal conflict prevention
- Hippocampus → STDP-enhanced temporal versioning
- Corpus Callosum → Multi-column parallel processing bridge
- Memory Consolidation → Neural inheritance compression
- Spike Timing → Sub-millisecond temporal precision
```

### 2. Neuromorphic Multi-Database Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│           Neuromorphic CortexKG Master System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Main DB │  │Branch A │  │Branch B │  │Branch C │  ...      │
│  │ +SNN    │  │ +SNN    │  │ +SNN    │  │ +SNN    │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│  ┌────┴────────────┴────────────┴────────────┴────┐            │
│  │        Multi-Column Neuromorphic Bridge          │            │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────┐   │            │
│  │  │Semantic  │ │Structural│ │Temporal  │ │Exc │   │            │
│  │  │Column    │ │Column    │ │Column    │ │Col │   │            │
│  │  └──────────┘ └──────────┘ └──────────┘ └────┘   │            │
│  │           Cortical Voting & Lateral Inhibition    │            │
│  └─────────────────────────────────────────────────┘            │
│  ┌─────────────────────────────────────────────────┐            │
│  │        SIMD-Accelerated WASM Processing          │            │
│  │     (4x Parallel Spike Pattern Processing)       │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Neuromorphic Temporal Memory Layers
```
Working Memory (Spike Patterns & TTFS Encoding)
    ↓ (Sub-second spike timing precision)
Short-term Memory (Active Column States + Refractory Periods)
    ↓ (STDP learning window: minutes-hours)
Long-term Memory (Consolidated Neural Weights + Inheritance)
    ↓ (Cascade correlation adaptations: permanent)
Archived Memory (Compressed Neural Networks + Sparse Distributed Memory)
    ↓ (Neuromorphic hardware optimization)
Neuromorphic Hardware (Intel Loihi / IBM TrueNorth acceleration)
```

## System Components

### Core Modules

1. **Neuromorphic Cortical Allocation Engine**
   - <1ms TTFS fact allocation with spike timing precision
   - Multi-column parallel processing (semantic, structural, temporal, exception)
   - Lateral inhibition circuits with winner-take-all dynamics
   - Cascade correlation network growth and adaptation
   - SIMD-accelerated spike pattern processing

2. **Neuromorphic Temporal Versioning System**
   - Git-like branching with STDP learning integration
   - Time-travel queries with spike timing analysis
   - Memory consolidation tracking with neural adaptation
   - Refractory period management across time branches
   - Neural weight evolution over temporal versions

3. **Multi-Column Cross-Database Neural Bridge**
   - Cross-database spike pattern correlation
   - Cortical voting consensus across databases
   - Emergent knowledge through neural network growth
   - SIMD-accelerated pattern matching
   - Lateral inhibition for cross-database conflict resolution

4. **Neuromorphic Intelligence Layer**
   - Small LLMs integrated with spiking neural networks
   - TTFS-guided allocation hint generation
   - Neural semantic compression with inheritance
   - Multi-column parallel concept processing
   - Circuit breakers for fault-tolerant operation

5. **Neuromorphic Memory Architecture**
   - Sparse Distributed Memory (Kanerva SDM) integration
   - WASM shared memory for high-performance processing
   - Neuromorphic data structures optimized for spike timing
   - Content-addressable memory with temporal codes
   - Hardware-accelerated memory access patterns

6. **Advanced Learning Systems**
   - Spike-Timing-Dependent Plasticity (STDP) learning rules
   - Continuous adaptation and performance monitoring
   - Real-time neural network topology optimization
   - Biological error correction mechanisms
   - Neuromorphic hardware preparation and optimization
   - Episodic: Event-based storage
   - Semantic: Fact-based inheritance
   - Procedural: Pattern storage
   - Working: Active allocations

## Phase Overview

### Phase 0: Foundation (Week 1)
- Project setup and core structures
- WASM build configuration
- Test framework with mocks

### Phase 1: Cortical Core (Week 2)
- Cortical columns implementation
- Lateral inhibition networks
- Basic allocation engine

### Phase 2: Allocation System (Week 3)
- Parallel allocation processing
- Concept analysis with LLMs
- Synaptic strengthening

### Phase 3: Sparse Storage (Week 4)
- Graph storage (<5% connectivity)
- Memory-mapped persistence
- Compression algorithms

### Phase 4: Inheritance (Week 5)
- Hierarchical memory organization
- Exception handling
- 10x compression achievement

### Phase 5: Temporal Versioning (Week 6)
- Branch management system
- Time-travel capabilities
- Memory consolidation

### Phase 6: Multi-Database Bridge (Week 7)
- Cross-database connections
- Pattern recognition
- Emergent knowledge detection

### Phase 7: Query Through Activation (Week 8)
- Spreading activation queries
- Structural navigation
- Memory recall patterns

### Phase 8: MCP with Intelligence (Week 9)
- Enhanced MCP tools
- Embedded LLM integration
- Intelligent allocation hints

### Phase 9: WASM & Web Interface (Week 10)
- WASM compilation
- Web-based visualization
- Real-time interaction

### Phase 10: Advanced Algorithms (Week 11)
- Knowledge recovery strategies
- Cross-database mining
- Pattern emergence

### Phase 11: Production Features (Week 12)
- Performance optimization
- Monitoring and observability
- Documentation

## Key Innovations

### 1. Temporal Branches as Memory Consolidation
```rust
// Each branch represents a memory state
pub struct MemoryBranch {
    id: BranchId,
    parent: Option<BranchId>,
    timestamp: DateTime,
    consolidation_state: ConsolidationState,
    active_columns: HashSet<ColumnId>,
}

// Memory consolidation over time
pub enum ConsolidationState {
    WorkingMemory,      // < 30 seconds
    ShortTerm,          // < 1 hour
    Consolidating,      // 1-24 hours
    LongTerm,           // > 24 hours
}
```

### 2. Cross-Database Neural Bridges
```rust
// Discover patterns across databases without explicit queries
pub struct NeuralBridge {
    databases: Vec<DatabaseConnection>,
    pattern_detector: PatternDetector,
    
    pub async fn discover_emergent_knowledge(&self) -> Vec<EmergentPattern> {
        // Automatic pattern recognition across branches
        // No embeddings needed - pure structural analysis
    }
}
```

### 3. LLM-Enhanced Allocation
```rust
// Small LLM guides allocation decisions
pub struct IntelligentAllocator {
    cortical_engine: CorticalEngine,
    llm_advisor: SmallLanguageModel,
    
    pub async fn allocate_with_intelligence(&self, fact: &Fact) -> AllocationResult {
        // LLM suggests optimal allocation
        let hint = self.llm_advisor.suggest_allocation(fact).await?;
        
        // Cortical engine executes with hint
        self.cortical_engine.allocate_with_hint(fact, hint).await
    }
}
```

### 4. Episodic vs Semantic Storage
```rust
// Different memory types stored differently
pub enum MemoryType {
    Episodic {
        event_time: DateTime,
        context: EventContext,
        decay_rate: f32,
    },
    Semantic {
        inheritance_chain: Vec<ConceptId>,
        compression_level: u8,
        permanence: f32,
    },
}
```

## Performance Targets

| Operation | Target | Reasoning |
|-----------|--------|-----------|
| Fact Allocation | <5ms | Cortical column activation |
| Document Processing | <50ms | Parallel scene processing |
| Cross-DB Pattern | <100ms | Neural bridge activation |
| Time-Travel Query | <20ms | Indexed temporal states |
| Branch Creation | <10ms | Copy-on-write columns |
| LLM Enhancement | <30ms | Cached small model |

## SPARC Methodology

Each phase follows strict SPARC:

1. **Specification**: Define biological inspiration and requirements
2. **Pseudocode**: Algorithm design mimicking brain processes
3. **Architecture**: Component structure following neural patterns
4. **Refinement**: Iterative improvement with performance tests
5. **Completion**: Full implementation with comprehensive tests

## Test-Driven Development

Following London School TDD:

1. **Mock First**: Create mock brain components
2. **Behavior Tests**: Test memory patterns, not implementation
3. **Integration**: Gradually replace mocks with real neurons
4. **Performance**: Continuous benchmarking against brain metrics

## Success Criteria

### Functional Requirements
- [ ] 5ms cortical allocation
- [ ] 10x inheritance compression
- [ ] Temporal versioning with branches
- [ ] Cross-database pattern detection
- [ ] LLM-enhanced allocation
- [ ] WASM compilation

### Performance Requirements
- [ ] 10,000 facts/second throughput
- [ ] <5% memory connectivity
- [ ] <100ms cross-database analysis
- [ ] <1GB memory for 1M facts

### Advanced Features
- [ ] Automatic knowledge emergence
- [ ] Memory consolidation simulation
- [ ] Episodic memory replay
- [ ] Semantic compression

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM latency | High | Local small models, caching |
| Cross-DB complexity | High | Start with 2 DBs, scale gradually |
| Memory usage | Medium | Aggressive pruning, compression |
| WASM performance | Medium | Native fallback, optimization |

## Development Philosophy

"We're not building a database - we're building a brain. Every decision should ask: How does biological memory do this?"

Key principles:
- Allocation over validation
- Structure over statistics  
- Emergence over explicit programming
- Compression through inheritance
- Time as a first-class concept

## Next Steps

1. Review this master plan
2. Set up development environment
3. Begin Phase 0: Foundation
4. Daily standups focused on biological accuracy

The following phase documents will detail each component with:
- Specific neural algorithms
- Test-driven examples
- Performance benchmarks
- Integration patterns
- Biological justifications