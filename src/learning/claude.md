# Directory Overview: Learning Systems

## 1. High-Level Summary

The `src/learning` directory implements a comprehensive adaptive learning system that enhances the knowledge graph with biological learning principles. It provides multiple interconnected learning mechanisms including Hebbian learning, synaptic homeostasis, meta-learning, parameter tuning, and adaptive feedback-driven optimization. The system is designed to continuously improve performance based on user feedback, system monitoring, and cognitive pattern analysis.

## 2. Tech Stack

- **Languages:** Rust
- **Key Dependencies:** 
  - `tokio` for async operations
  - `uuid` for unique identifiers
  - `serde` for serialization
  - `anyhow` for error handling
  - `async-trait` for async traits
- **Internal Dependencies:** Core brain graph, cognitive systems, activation engines
- **Architecture:** Component-based with Arc/RwLock for thread-safe shared state

## 3. Directory Structure

```
src/learning/
├── mod.rs                    # Main module with re-exports
├── types.rs                  # Core learning type definitions
├── hebbian.rs               # Hebbian learning with STDP
├── homeostasis.rs           # Synaptic homeostasis system
├── meta_learning.rs         # Meta-learning algorithms
├── parameter_tuning.rs      # Automated parameter optimization
└── adaptive_learning/       # Adaptive learning subsystem
    ├── mod.rs              # Adaptive learning module
    ├── types.rs            # Adaptive learning types
    ├── system.rs           # Main adaptive learning system
    ├── feedback.rs         # Feedback aggregation
    ├── monitoring.rs       # Performance monitoring
    └── scheduler.rs        # Learning task scheduling
```

## 4. File Breakdown

### `mod.rs`
- **Purpose:** Main module definition and public API exports
- **Key Exports:**
  - `HebbianLearningEngine` - Hebbian learning implementation
  - `SynapticHomeostasis`, `HomeostasisUpdate` - Homeostasis system
  - `AdaptiveLearningSystem` - Adaptive learning coordination
  - `ParameterTuner` - Parameter optimization interface
  - `MetaLearningSystem` - Meta-learning capabilities

### `types.rs`
- **Purpose:** Core type definitions for learning systems
- **Key Types:**
  - `ActivationEvent` - Neural activation event with context
  - `LearningContext` - Learning environment and goals
  - `LearningUpdate` - Results of learning operations
  - `WeightChange` - Synaptic weight modifications
  - `STDPResult` - Spike-timing dependent plasticity results
  - `CoactivationTracker` - Tracks entity co-activation patterns
  - `LearningStatistics` - Learning performance metrics

### `hebbian.rs`
- **Purpose:** Implements Hebbian learning with biological realism
- **Classes:**
  - `HebbianLearningEngine`
    - **Description:** Core Hebbian learning implementation with STDP
    - **Methods:**
      - `new()` - Creates engine with brain graph and cognitive systems
      - `apply_hebbian_learning()` - Main learning cycle execution
      - `spike_timing_dependent_plasticity()` - STDP implementation
      - `update_coactivation_tracking()` - Tracks entity co-activations
      - `apply_synaptic_weight_changes()` - Updates connection weights
      - `apply_temporal_decay()` - Implements synaptic decay
- **Key Features:**
  - Biologically plausible learning rates and thresholds
  - Temporal correlation tracking with sliding windows
  - Integration with competitive inhibition systems
  - Comprehensive test suite validating biological constraints

### `homeostasis.rs`
- **Purpose:** Synaptic homeostasis and metaplasticity implementation
- **Classes:**
  - `SynapticHomeostasis`
    - **Description:** Maintains stable activity levels through homeostatic scaling
    - **Methods:**
      - `apply_homeostatic_scaling()` - Main homeostasis cycle
      - `implement_metaplasticity()` - Plasticity of plasticity
      - `calculate_integrated_activity_levels()` - Activity assessment
      - `emergency_stabilization()` - Crisis response mechanism
- **Key Features:**
  - Activity-dependent synaptic scaling
  - Metaplasticity with threshold adaptation
  - Integration with attention and memory systems
  - Emergency stabilization for system failures

### `meta_learning.rs`
- **Purpose:** Meta-learning system for learning to learn better
- **Classes:**
  - `MetaLearningSystem`
    - **Description:** Coordinates multiple learning algorithms and strategies
    - **Methods:**
      - `learn_to_learn()` - Main meta-learning function
      - `adapt_learning_strategy()` - Context-aware strategy adaptation
      - `transfer_knowledge()` - Cross-domain knowledge transfer
      - `analyze_learning_patterns()` - Pattern extraction from learning history
- **Key Features:**
  - Algorithm selection and recommendation
  - Transfer learning between domains
  - Learning strategy optimization
  - Performance prediction and analysis

### `parameter_tuning.rs`
- **Purpose:** Automated parameter optimization system
- **Classes:**
  - `ParameterTuningSystem`
    - **Description:** Optimizes system parameters using various algorithms
    - **Methods:**
      - `tune_hebbian_parameters()` - Optimizes Hebbian learning rates
      - `tune_attention_parameters()` - Adjusts attention mechanisms
      - `tune_memory_parameters()` - Optimizes memory systems
      - `auto_tune_system()` - Comprehensive system optimization
- **Key Features:**
  - Multiple optimization strategies (Bayesian, Grid Search, etc.)
  - Resource-aware optimization with constraints
  - Component-specific parameter spaces
  - Safety mechanisms and rollback capabilities

### `adaptive_learning/system.rs`
- **Purpose:** Main adaptive learning system coordinating all components
- **Classes:**
  - `AdaptiveLearningSystem`
    - **Description:** Orchestrates learning based on performance feedback
    - **Methods:**
      - `execute_learning_cycle()` - Main adaptive learning loop
      - `handle_emergency()` - Emergency adaptation response
      - `identify_learning_targets()` - Target identification from analysis
      - `execute_adaptations()` - Learning target execution
- **Key Features:**
  - Performance bottleneck detection
  - User satisfaction analysis
  - Coordinated multi-system adaptation
  - Emergency response capabilities

### `adaptive_learning/feedback.rs`
- **Purpose:** Feedback aggregation and analysis
- **Classes:**
  - `FeedbackAggregator`
    - **Description:** Collects and processes user and system feedback
    - **Methods:**
      - `add_user_feedback()` - Records user satisfaction data
      - `add_system_feedback()` - Records system performance data
      - `aggregate_feedback()` - Combines feedback sources
      - `analyze_user_satisfaction()` - Identifies satisfaction patterns
      - `correlate_performance_outcomes()` - Links performance to outcomes
- **Key Features:**
  - Multi-source feedback integration (explicit, implicit, system)
  - Temporal decay for feedback relevance
  - Correlation analysis between metrics
  - Problem area identification

### `adaptive_learning/monitoring.rs`
- **Purpose:** Performance monitoring and anomaly detection
- **Classes:**
  - `PerformanceMonitor`
    - **Description:** Monitors system performance and detects issues
    - **Methods:**
      - `get_current_snapshot()` - Current system performance state
      - `detect_anomalies()` - Identifies performance bottlenecks
      - `get_performance_trend()` - Historical performance analysis
      - `should_trigger_alert()` - Alert threshold checking
- **Key Features:**
  - Multi-dimensional performance scoring
  - Bottleneck categorization (Memory, CPU, Cognitive, User)
  - Historical trend analysis
  - Configurable alert thresholds

### `adaptive_learning/scheduler.rs`
- **Purpose:** Learning task scheduling and resource management
- **Classes:**
  - `LearningScheduler`
    - **Description:** Schedules and manages learning task execution
    - **Methods:**
      - `schedule_task()` - Schedules regular learning tasks
      - `schedule_emergency_task()` - High-priority emergency scheduling
      - `get_next_task()` - Task queue management
      - `adaptive_schedule()` - Dynamic scheduling based on system state
- **Key Features:**
  - Priority-based task scheduling
  - Resource constraint checking
  - Adaptive scheduling based on system load
  - Task completion tracking and statistics

## 5. Key Learning Algorithms

### Hebbian Learning
- **Implementation:** `hebbian.rs:HebbianLearningEngine`
- **Features:** 
  - Spike-timing dependent plasticity (STDP)
  - Temporal correlation tracking
  - Competitive inhibition integration
  - Biologically plausible parameters

### Synaptic Homeostasis
- **Implementation:** `homeostasis.rs:SynapticHomeostasis`
- **Features:**
  - Activity-dependent scaling
  - Metaplasticity mechanisms
  - Emergency stabilization
  - Integration with cognitive systems

### Meta-Learning
- **Implementation:** `meta_learning.rs:MetaLearningSystem`
- **Features:**
  - Algorithm selection optimization
  - Cross-domain knowledge transfer
  - Learning strategy adaptation
  - Performance prediction

### Parameter Optimization
- **Implementation:** `parameter_tuning.rs:ParameterTuningSystem`
- **Features:**
  - Multiple optimization algorithms
  - Component-specific tuning
  - Resource-aware optimization
  - Safety mechanisms

## 6. Data Flow and Integration

### Learning Cycle Flow
1. **Performance Monitoring** → Collect metrics from cognitive systems
2. **Feedback Aggregation** → Process user and system feedback  
3. **Bottleneck Detection** → Identify performance issues
4. **Target Identification** → Determine learning priorities
5. **Adaptation Execution** → Apply learning algorithms
6. **Impact Assessment** → Measure improvement results
7. **Task Scheduling** → Plan future learning activities

### Inter-System Communication
- **Brain Graph Integration:** Direct connection weight updates
- **Cognitive System Integration:** Parameter adjustments for attention, memory
- **Activation Engine Integration:** Propagation parameter tuning
- **Inhibition System Integration:** Competition strength adjustments

## 7. Key Variables and Logic

### Critical Parameters
- **Learning Rates:** Typically 0.001-0.1 range for biological plausibility
- **Decay Constants:** Usually < learning_rate for stability
- **Activity Thresholds:** 0.3-0.7 range for strengthening/weakening
- **Temporal Windows:** 60-300 seconds for correlation tracking
- **Emergency Thresholds:** 0.8+ severity for immediate intervention

### Learning Logic
- **Hebbian Rule:** "Entities that activate together, strengthen together"
- **STDP Rule:** Pre-before-post strengthens, post-before-pre weakens
- **Homeostatic Scaling:** Maintain target activity levels through weight scaling
- **Metaplasticity:** Adjust learning rates based on recent plasticity history

## 8. Dependencies

### Internal Dependencies
- `crate::core::brain_enhanced_graph` - Knowledge graph with brain-inspired features
- `crate::cognitive::*` - Attention, memory, orchestration, inhibition systems
- `crate::core::activation_engine` - Neural activation propagation
- `crate::core::types` - Core entity and relationship types

### External Dependencies
- `tokio` - Async runtime for concurrent learning operations
- `uuid` - Unique identifiers for learning sessions and tasks
- `serde` - Serialization for learning parameters and results
- `anyhow` - Error handling throughout learning systems

## 9. Performance Characteristics

### Computational Complexity
- **Hebbian Learning:** O(n²) for n active entities (correlation matrix)
- **Homeostasis:** O(n) for n entities with activity tracking
- **Parameter Tuning:** Varies by algorithm (Bayesian: O(d²), Grid: O(k^d))
- **Meta-Learning:** O(m·p) for m tasks and p patterns

### Memory Usage
- **Activation History:** Sliding window limits (typically 1000 entries)
- **Learning Statistics:** Bounded collections with automatic cleanup
- **Feedback Storage:** Time-based retention (default 1 week)
- **Task History:** Limited to recent 1000 tasks

### Scalability Considerations
- **Parallel Processing:** Multiple concurrent learning tasks supported
- **Resource Constraints:** Configurable memory and CPU limits
- **Adaptive Scheduling:** Dynamic load balancing based on system state
- **Emergency Throttling:** Automatic scaling during high-load periods

## 10. Testing and Validation

### Test Coverage
- **Unit Tests:** Each learning algorithm has comprehensive test suite
- **Integration Tests:** Cross-system interaction validation
- **Biological Plausibility Tests:** Parameter constraint validation
- **Performance Tests:** Efficiency and resource usage validation

### Key Test Scenarios
- **Learning Convergence:** Validates stable learning under various conditions
- **Emergency Response:** Tests crisis handling and recovery mechanisms
- **Parameter Optimization:** Validates improvement in target metrics
- **Feedback Integration:** Tests correlation between feedback and adaptations

This learning system represents a sophisticated implementation of biologically-inspired adaptive learning, providing both immediate performance optimization and long-term system evolution capabilities.