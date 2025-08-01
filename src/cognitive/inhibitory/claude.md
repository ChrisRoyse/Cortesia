# Directory Overview: cognitive/inhibitory

## 1. High-Level Summary

The `cognitive/inhibitory` directory implements a sophisticated competitive inhibition system inspired by biological neural processing. This system manages competition between activated entities in the knowledge graph, implementing lateral inhibition, hierarchical competition, and temporal dynamics to create sparse, meaningful activation patterns. The system includes adaptive learning mechanisms to optimize inhibition parameters over time and integrates with other cognitive patterns for context-aware processing.

## 2. Tech Stack

*   **Languages:** Rust
*   **Frameworks:** None (pure Rust implementation)
*   **Libraries:** 
    *   `tokio` - Async runtime for concurrent operations
    *   `uuid` - Unique identifier generation
    *   `slotmap` - Entity key management
*   **Testing:** Rust's built-in testing framework with tokio::test for async tests

## 3. Directory Structure

The directory contains 10 Rust modules organized by functionality:

*   Core system files (`mod.rs`, `types.rs`)
*   Competition mechanisms (`competition.rs`, `hierarchical.rs`)
*   Dynamic behavior (`dynamics.rs`, `learning.rs`)
*   Exception handling (`exceptions.rs`)
*   Integration and utilities (`integration.rs`, `matrix.rs`, `metrics.rs`)

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Main module file that defines the `CompetitiveInhibitionSystem` struct and orchestrates the overall inhibition process.
*   **Classes:**
    *   `CompetitiveInhibitionSystem`
        *   **Description:** Core system that manages competitive inhibition between activated entities.
        *   **Fields:**
            *   `activation_engine: Arc<ActivationPropagationEngine>` - Engine for activation propagation
            *   `critical_thinking: Arc<CriticalThinking>` - Critical thinking module integration
            *   `inhibition_matrix: Arc<RwLock<InhibitionMatrix>>` - Matrix storing inhibition relationships
            *   `competition_groups: Arc<RwLock<Vec<CompetitionGroup>>>` - Groups of competing entities
            *   `inhibition_config: InhibitionConfig` - Configuration parameters
        *   **Methods:**
            *   `new(activation_engine, critical_thinking)`: Creates new inhibition system
            *   `apply_competitive_inhibition(activation_pattern, domain_context)`: Main entry point for applying inhibition
            *   `add_competition_group(group)`: Adds new competition group
            *   `would_compete(entity_a, entity_b)`: Checks if two entities compete
            *   `update_competition_strength(entity_a, entity_b, strength_change)`: Updates competition strength
            *   `create_learned_competition_groups(activation_history, correlation_threshold)`: Creates groups from activation history
            *   `check_learning_status()`: Returns current learning status

### `types.rs`

*   **Purpose:** Type definitions for the inhibition system, providing all data structures used throughout the module.
*   **Structs:**
    *   `InhibitionMatrix` - Stores different types of inhibition relationships
        *   `lateral_inhibition: HashMap<(EntityKey, EntityKey), f32>`
        *   `hierarchical_inhibition: HashMap<(EntityKey, EntityKey), f32>`
        *   `contextual_inhibition: HashMap<(EntityKey, EntityKey), f32>`
        *   `temporal_inhibition: HashMap<(EntityKey, EntityKey), f32>`
    *   `CompetitionGroup` - Defines a group of competing entities
        *   `group_id: String` - Unique identifier
        *   `competing_entities: Vec<EntityKey>` - List of competing entities
        *   `competition_type: CompetitionType` - Type of competition
        *   `winner_takes_all: bool` - Whether to use winner-takes-all strategy
        *   `inhibition_strength: f32` - Strength of inhibition
        *   `priority: f32` - Processing priority
        *   `temporal_dynamics: TemporalDynamics` - Temporal behavior
    *   `TemporalDynamics` - Temporal behavior configuration
        *   `onset_delay: Duration` - Delay before activation
        *   `peak_time: Duration` - Time to peak activation
        *   `decay_time: Duration` - Time to decay
        *   `oscillation_frequency: Option<f32>` - Optional oscillation
    *   `InhibitionConfig` - System configuration
        *   `global_inhibition_strength: f32` - Overall inhibition strength
        *   `lateral_inhibition_strength: f32` - Lateral inhibition strength
        *   `hierarchical_inhibition_strength: f32` - Hierarchical inhibition strength
        *   `contextual_inhibition_strength: f32` - Contextual inhibition strength
        *   `winner_takes_all_threshold: f32` - Threshold for winner-takes-all
        *   `soft_competition_factor: f32` - Factor for soft competition
        *   `temporal_integration_window: Duration` - Time window for integration
        *   `enable_learning: bool` - Whether to enable adaptive learning
    *   `InhibitionResult` - Result of inhibition process
    *   `GroupCompetitionResult` - Result of group competition
    *   `HierarchicalInhibitionResult` - Result of hierarchical inhibition
    *   `ExceptionHandlingResult` - Result of exception handling
    *   `InhibitionPerformanceMetrics` - Performance metrics
    *   `LearningStatus` - Current learning status
*   **Enums:**
    *   `CompetitionType` - Types of competition (Semantic, Temporal, Hierarchical, Contextual, Spatial, Causal)
    *   `InhibitionException` - Types of exceptions that can occur
    *   `ResolutionStrategy` - Strategies for resolving exceptions
    *   `ParameterAdjustmentType` - Types of parameter adjustments
    *   `AdaptationType` - Types of adaptations

### `competition.rs`

*   **Purpose:** Implements various competition strategies and group-based competition logic.
*   **Functions:**
    *   `apply_group_competition(system, working_pattern, competition_groups, config)`: Applies group-based competition
    *   `apply_semantic_competition(pattern, group, config)`: Semantic competition implementation
    *   `apply_temporal_competition(pattern, group, config)`: Temporal competition with phase-based inhibition
    *   `apply_hierarchical_competition(pattern, group, config)`: Higher-level concepts inhibit lower-level ones
    *   `apply_contextual_competition(pattern, group, config)`: Context-aware competition
    *   `apply_spatial_competition(pattern, group, config)`: Spatial relationship-based competition
    *   `apply_causal_competition(pattern, group, config)`: Causal relationship-based competition
    *   `apply_soft_competition(pattern, entity_pairs, inhibition_strength)`: Soft mutual inhibition

### `dynamics.rs`

*   **Purpose:** Manages temporal dynamics and time-based modulation of inhibition.
*   **Functions:**
    *   `apply_temporal_dynamics(pattern, competition_results, config)`: Applies temporal modulation
    *   `calculate_temporal_factor(dynamics, elapsed)`: Calculates temporal modulation factor
    *   `calculate_decay_factor(integration_window)`: Calculates exponential decay
    *   `apply_oscillation(strength, frequency, elapsed)`: Applies oscillatory dynamics

### `exceptions.rs`

*   **Purpose:** Handles special cases and exceptions in the inhibition process, detecting and resolving conflicts.
*   **Functions:**
    *   `handle_inhibition_exceptions(system, pattern, competition_results, hierarchical_result)`: Main exception handler
    *   `detect_mutual_exclusions(pattern, exceptions)`: Detects mutually exclusive entity activations
    *   `detect_temporal_conflicts(pattern, competition_results, exceptions)`: Detects temporal ordering violations
    *   `detect_hierarchical_inconsistencies(hierarchical_result, exceptions)`: Detects hierarchy violations
    *   `detect_resource_contentions(pattern, exceptions)`: Detects resource limitation violations
    *   `resolve_exception(exception, pattern, system)`: Resolves detected exceptions
    *   `apply_resolution(pattern, resolution)`: Applies resolution to activation pattern
    *   `identify_unresolved_conflicts(exceptions, resolutions)`: Identifies unresolved issues

### `hierarchical.rs`

*   **Purpose:** Implements hierarchical inhibition based on abstraction levels.
*   **Functions:**
    *   `apply_hierarchical_inhibition(system, pattern, inhibition_matrix, config)`: Main hierarchical inhibition
    *   `assign_abstraction_levels(activations)`: Assigns abstraction levels to entities
    *   `create_hierarchical_layers(abstraction_levels, pattern)`: Creates layer structures
    *   `apply_top_down_inhibition(layers, matrix, config)`: Top-down inhibition from higher to lower layers
    *   `apply_within_layer_inhibition(layers, matrix, config)`: Lateral inhibition within layers
    *   `update_pattern_from_layers(pattern, layers)`: Updates activation pattern
    *   `identify_specificity_winners(layers)`: Identifies entities that won due to specificity
    *   `identify_generality_suppressed(layers)`: Identifies entities suppressed for being too general

### `integration.rs`

*   **Purpose:** Integrates inhibition with cognitive patterns for pattern-specific modulation.
*   **Functions:**
    *   `integrate_with_cognitive_patterns(system, pattern, active_cognitive_patterns)`: Main integration function
    *   `apply_pattern_specific_inhibition(pattern, cognitive_pattern)`: Applies pattern-specific inhibition
    *   `get_inhibition_profile(pattern_type)`: Gets inhibition profile for cognitive pattern
    *   `apply_convergent_inhibition(pattern, profile, affected)`: Strong lateral inhibition for focus
    *   `apply_divergent_inhibition(pattern, profile, affected)`: Weak inhibition for multiple ideas
    *   `apply_lateral_inhibition(pattern, profile, affected)`: Asymmetric inhibition for creativity
    *   `apply_critical_inhibition(pattern, profile, affected)`: Targeted inhibition of weak arguments
    *   `apply_systems_inhibition(pattern, profile, affected)`: Hierarchical inhibition preserving relationships
    *   `apply_abstract_inhibition(pattern, profile, affected)`: Inhibit concrete details, enhance patterns
    *   `apply_adaptive_inhibition(pattern, profile, affected)`: Dynamic context-based inhibition
    *   `detect_cross_pattern_conflicts(inhibitions, conflicts)`: Detects conflicts between patterns

### `learning.rs`

*   **Purpose:** Implements adaptive learning mechanisms to optimize inhibition performance over time.
*   **Functions:**
    *   `apply_adaptive_learning(system, activation_pattern, inhibition_results)`: Applies adaptive improvements
    *   `apply_learning_mechanisms(system, pattern, inhibition_results, performance_history)`: Main learning function
    *   `learn_inhibition_strength_adjustment(pattern, results, history)`: Learns optimal inhibition strength
    *   `learn_competition_group_optimization(results, history)`: Optimizes competition groups
    *   `learn_temporal_dynamics_optimization(pattern, history)`: Optimizes temporal parameters
    *   `apply_parameter_adjustment(system, adjustment)`: Applies parameter changes
    *   `calculate_performance_metrics(activation_pattern, inhibition_results)`: Calculates performance metrics
    *   `generate_adaptation_suggestions(metrics)`: Generates improvement suggestions
    *   `apply_adaptation_suggestion(system, suggestion)`: Applies adaptation
    *   `calculate_learning_confidence(history)`: Calculates confidence in learning
    *   `calculate_variance(values)`: Utility function for variance calculation

### `matrix.rs`

*   **Purpose:** Operations and management for the inhibition matrix.
*   **Traits:**
    *   `InhibitionMatrixOps`
        *   **Description:** Trait defining operations on inhibition matrices.
        *   **Methods:**
            *   `get_inhibition_strength(source, target, inhibition_type)`: Gets inhibition strength
            *   `set_inhibition_strength(source, target, strength, inhibition_type)`: Sets inhibition strength
            *   `update_inhibition_strength(source, target, delta, inhibition_type)`: Updates by delta
            *   `get_total_inhibition(source, target)`: Gets combined inhibition
*   **Enums:**
    *   `InhibitionType` - Types of inhibition (Lateral, Hierarchical, Contextual, Temporal)

### `metrics.rs`

*   **Purpose:** Performance metrics calculation and analysis for the inhibition system.
*   **Functions:**
    *   `calculate_comprehensive_metrics(pattern, competition_results, processing_time)`: Calculates all metrics
    *   `calculate_efficiency_score(competition_results)`: Measures competition resolution efficiency
    *   `calculate_effectiveness_score(pattern, competition_results)`: Measures activation pattern quality
    *   `calculate_variance(values)`: Statistical variance calculation
    *   `analyze_metric_trends(history, window_size)`: Analyzes trends over time
*   **Structs:**
    *   `MetricTrends` - Trend analysis results
        *   `avg_efficiency: f32` - Average efficiency
        *   `avg_effectiveness: f32` - Average effectiveness
        *   `avg_processing_time_ms: f64` - Average processing time
        *   `efficiency_trend: f32` - Efficiency trend (positive = improving)
        *   `effectiveness_trend: f32` - Effectiveness trend
        *   `processing_time_trend: f64` - Processing time trend

## 5. Key Variables and Logic

### Competition Resolution Process
1. Groups are sorted by priority
2. Competition type determines specific strategy
3. Winner-takes-all vs soft competition based on configuration
4. Results tracked for learning and metrics

### Hierarchical Processing
1. Entities assigned abstraction levels based on activation strength
2. Top-down inhibition from general to specific
3. Within-layer lateral competition
4. Specificity bias in final activation

### Learning Mechanisms
1. Performance metrics calculated after each cycle
2. Adaptation suggestions generated based on metrics
3. Parameter adjustments applied if confidence threshold met
4. History maintained for trend analysis

### Exception Handling
1. Multiple exception types detected (mutual exclusion, temporal conflicts, etc.)
2. Resolution strategies applied based on exception type
3. Unresolved conflicts tracked and reported

## 6. Dependencies

*   **Internal:**
    *   `crate::core::activation_engine::ActivationPropagationEngine` - Activation propagation
    *   `crate::core::brain_types::ActivationPattern` - Activation pattern structure
    *   `crate::core::types::EntityKey` - Entity identification
    *   `crate::cognitive::critical::CriticalThinking` - Critical thinking integration
    *   `crate::cognitive::types::CognitivePatternType` - Cognitive pattern types
    *   `crate::error::Result` - Error handling
*   **External:**
    *   `std::collections::HashMap` - Hash map data structure
    *   `std::sync::Arc` - Atomic reference counting
    *   `tokio::sync::RwLock` - Async read-write lock
    *   `uuid` - UUID generation
    *   `std::time::{Duration, SystemTime}` - Time handling

## 7. Testing Strategy

Each module includes comprehensive unit tests using Rust's testing framework:
*   Async tests use `#[tokio::test]`
*   Test utilities create mock systems and patterns
*   Edge cases tested (empty patterns, single entities, conflicts)
*   Performance and learning mechanisms verified
*   Competition strategies tested individually

## 8. API Integration Points

The system integrates with the broader cognitive architecture through:
*   `ActivationPropagationEngine` - Receives activation patterns
*   `CriticalThinking` - Coordinates with critical thinking processes
*   Cognitive pattern types - Adapts behavior based on active patterns
*   Learning results - Feeds back into system optimization

## 9. Key Algorithms

### Winner-Takes-All Competition
```rust
if group.winner_takes_all && max_strength > config.winner_takes_all_threshold {
    // Suppress all but winner
} else {
    // Soft competition with proportional inhibition
}
```

### Hierarchical Abstraction Assignment
- High activation (>0.8) = Most specific (level 0)
- Medium activation (>0.5) = Mid-level (level 1)  
- Low activation = Most general (level 2)

### Learning Confidence Calculation
- Based on history length and performance consistency
- More history + consistent performance = higher confidence
- Maximum confidence capped at 0.95

### Performance Metrics
- Efficiency: Clear winners, appropriate suppression ratio, optimal intensity
- Effectiveness: Optimal sparsity (20-40%), high differentiation, information preservation