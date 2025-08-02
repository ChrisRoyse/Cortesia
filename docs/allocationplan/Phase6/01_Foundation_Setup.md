# Phase 6.1: Foundation Setup for Truth Maintenance System

**Duration**: 2-3 hours  
**Complexity**: Low  
**Dependencies**: Core neuromorphic system from Phase 5

## Micro-Tasks Overview

This phase establishes the foundational structure for the Truth Maintenance System (TMS) integration with the neuromorphic knowledge graph.

---

## Task 6.1.1: Create TMS Core Module Structure

**Estimated Time**: 30 minutes  
**Complexity**: Low  
**AI Task**: Create the basic module structure and trait definitions

**Prompt for AI:**
```
Create a new module `src/truth_maintenance/mod.rs` with the following structure:
1. Define the main TruthMaintenanceSystem struct
2. Create error types (TMSError, RevisionError, ConflictError)
3. Define basic traits: RevisionStrategy, ResolutionStrategy
4. Add module declarations for sub-modules
5. Include comprehensive documentation with examples

Requirements:
- Follow existing code patterns in the codebase
- Use proper error handling with anyhow
- Include async support with tokio
- Add logging with tracing
- Include comprehensive rustdoc documentation

Code Example from existing codebase pattern (src/lib.rs):
```rust
// src/lib.rs shows this modular structure:
pub mod core;
pub mod scalable;

pub use core::{AllocationEngine, Fact, AllocationResult, NodeId};
pub use scalable::{ScalableAllocationEngine, ScalabilityConfig};

pub mod prelude {
    pub use crate::core::{AllocationEngine, Fact, AllocationResult, NodeId};
    pub use crate::scalable::{ScalableAllocationEngine, ScalabilityConfig};
}
```

Expected implementation for TMS with neuromorphic spike pattern integration:
```rust
// src/truth_maintenance/mod.rs
pub mod jtms;
pub mod atms;
pub mod belief_revision;
pub mod conflict_detection;
pub mod config;
pub mod errors;
pub mod types;
pub mod metrics;

use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

pub use config::TMSConfig;
pub use errors::{TMSError, RevisionError, ConflictError};
pub use types::{BeliefNode, BeliefStatus, Justification, ContextId};

/// Main Truth Maintenance System integrating JTMS and ATMS with neuromorphic processing
#[derive(Debug)]
pub struct TruthMaintenanceSystem {
    config: Arc<TMSConfig>,
    jtms_layer: Arc<RwLock<jtms::JustificationBasedTMS>>,
    atms_layer: Arc<RwLock<atms::AssumptionBasedTMS>>,
    spike_encoder: Arc<TTFSSpikeEncoder>,
    cortical_processor: Arc<CorticalColumnProcessor>,
    metrics: Arc<metrics::TMSHealthMetrics>,
}

/// TTFS (Time-To-First-Spike) encoder for belief confidence
#[derive(Debug)]
pub struct TTFSSpikeEncoder {
    max_spike_time: Duration,  // Maximum spike time for 0 confidence
    min_spike_time: Duration,  // Minimum spike time for max confidence
}

impl TTFSSpikeEncoder {
    /// Encode belief confidence as TTFS pattern
    /// Higher confidence = earlier spike (lower time-to-first-spike)
    pub fn encode_belief_confidence(&self, confidence: f64) -> TTFSSpikePattern {
        let spike_time = self.min_spike_time + 
            (self.max_spike_time - self.min_spike_time) * (1.0 - confidence);
        
        TTFSSpikePattern {
            first_spike_time: spike_time,
            confidence: confidence,
            intensity: (confidence * 255.0) as u8,
            duration: Duration::from_millis(10), // Standard spike duration
        }
    }
    
    /// Decode spike pattern back to confidence level
    pub fn decode_spike_confidence(&self, pattern: &TTFSSpikePattern) -> f64 {
        let normalized = (pattern.first_spike_time - self.min_spike_time).as_nanos() as f64 /
                        (self.max_spike_time - self.min_spike_time).as_nanos() as f64;
        1.0 - normalized.clamp(0.0, 1.0)
    }
}

/// Spike pattern for TTFS encoding
#[derive(Debug, Clone)]
pub struct TTFSSpikePattern {
    pub first_spike_time: Duration,
    pub confidence: f64,
    pub intensity: u8,
    pub duration: Duration,
}

/// Cortical column processor for parallel belief processing
#[derive(Debug)]
pub struct CorticalColumnProcessor {
    columns: Arc<RwLock<HashMap<ColumnId, SpikingCorticalColumn>>>,
    inhibition_network: Arc<LateralInhibitionNetwork>,
}

impl CorticalColumnProcessor {
    /// Process belief through cortical columns with lateral inhibition
    pub async fn process_belief_set(
        &self,
        beliefs: &[BeliefNode],
    ) -> Result<ConsensusResult, TMSError> {
        let mut column_assignments = HashMap::new();
        
        // Assign beliefs to cortical columns based on confidence
        for belief in beliefs {
            let column_id = self.assign_column_by_confidence(belief.confidence).await?;
            column_assignments.insert(belief.id, column_id);
        }
        
        // Apply lateral inhibition for conflicting beliefs
        let conflicts = self.detect_belief_conflicts(beliefs).await?;
        for (belief_a, belief_b) in conflicts {
            if let (Some(&col_a), Some(&col_b)) = (
                column_assignments.get(&belief_a),
                column_assignments.get(&belief_b)
            ) {
                self.apply_lateral_inhibition(col_a, col_b, 0.8).await?;
            }
        }
        
        // Process through columns and gather consensus
        let results = self.gather_cortical_consensus(&column_assignments).await?;
        Ok(results)
    }
    
    /// Apply lateral inhibition between competing columns
    async fn apply_lateral_inhibition(
        &self,
        column_a: ColumnId,
        column_b: ColumnId,
        strength: f64,
    ) -> Result<(), TMSError> {
        let inhibition = InhibitoryConnection {
            source: column_a,
            target: column_b,
            strength,
            delay: Duration::from_micros(100), // Synaptic delay
        };
        
        self.inhibition_network.add_connection(inhibition).await?;
        Ok(())
    }
}

/// Lateral inhibition connection between cortical columns
#[derive(Debug, Clone)]
pub struct InhibitoryConnection {
    pub source: ColumnId,
    pub target: ColumnId,
    pub strength: f64,
    pub delay: Duration,
}

/// Network managing lateral inhibition between columns
#[derive(Debug)]
pub struct LateralInhibitionNetwork {
    connections: Arc<RwLock<Vec<InhibitoryConnection>>>,
}

impl LateralInhibitionNetwork {
    pub async fn add_connection(&self, connection: InhibitoryConnection) -> Result<(), TMSError> {
        let mut connections = self.connections.write().await;
        connections.push(connection);
        Ok(())
    }
    
    /// Apply winner-take-all dynamics for conflict resolution
    pub async fn apply_winner_take_all(
        &self,
        competing_columns: &[ColumnId],
    ) -> Result<ColumnId, TMSError> {
        // Find column with highest activation
        let winner = competing_columns.iter()
            .max_by(|&a, &b| {
                self.get_column_activation(*a)
                    .partial_cmp(&self.get_column_activation(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .ok_or_else(|| TMSError::Integration("No competing columns".to_string()))?;
        
        // Suppress all other columns
        for &column_id in competing_columns {
            if column_id != winner {
                self.suppress_column(column_id).await?;
            }
        }
        
        Ok(winner)
    }
    
    fn get_column_activation(&self, _column_id: ColumnId) -> f64 {
        // Placeholder - would get actual activation from column
        0.5
    }
    
    async fn suppress_column(&self, _column_id: ColumnId) -> Result<(), TMSError> {
        // Placeholder - would apply inhibitory signal to column
        Ok(())
    }
}

/// Result of cortical consensus processing
#[derive(Debug)]
pub struct ConsensusResult {
    pub winning_beliefs: Vec<BeliefId>,
    pub confidence_scores: HashMap<BeliefId, f64>,
    pub processing_time: Duration,
}

impl TruthMaintenanceSystem {
    /// Create a new TMS instance with the given configuration
    pub async fn new(config: TMSConfig) -> Result<Self> {
        info!("Initializing Truth Maintenance System with config: {:?}", config);
        
        let config = Arc::new(config);
        let jtms_layer = Arc::new(RwLock::new(
            jtms::JustificationBasedTMS::new(config.clone()).await
                .context("Failed to initialize JTMS layer")?
        ));
        let atms_layer = Arc::new(RwLock::new(
            atms::AssumptionBasedTMS::new(config.clone()).await
                .context("Failed to initialize ATMS layer")?
        ));
        let metrics = Arc::new(metrics::TMSHealthMetrics::new());

        Ok(Self {
            config,
            jtms_layer,
            atms_layer,
            metrics,
        })
    }
}

/// Strategy for belief revision operations
#[async_trait::async_trait]
pub trait RevisionStrategy: Send + Sync {
    async fn revise_beliefs(
        &self,
        current: &BeliefSet,
        new_belief: &BeliefNode,
    ) -> Result<BeliefSet, RevisionError>;
}

/// Strategy for conflict resolution
#[async_trait::async_trait]
pub trait ResolutionStrategy: Send + Sync {
    async fn resolve_conflict(
        &self,
        conflict: &ConflictSet,
        context: &ContextId,
    ) -> Result<ResolutionResult, ConflictError>;
}

/// Re-exports for convenient usage
pub mod prelude {
    pub use super::{
        TruthMaintenanceSystem, TMSConfig, TMSError,
        RevisionStrategy, ResolutionStrategy,
        BeliefNode, BeliefStatus,
    };
}
```

Integration example with existing core module:
```rust
// src/lib.rs (updated to include TMS)
pub mod core;
pub mod scalable;
pub mod truth_maintenance;  // New TMS module

pub use core::{AllocationEngine, Fact, AllocationResult, NodeId};
pub use scalable::{ScalableAllocationEngine, ScalabilityConfig};
pub use truth_maintenance::{TruthMaintenanceSystem, TMSConfig};

pub mod prelude {
    pub use crate::core::{AllocationEngine, Fact, AllocationResult, NodeId};
    pub use crate::scalable::{ScalableAllocationEngine, ScalabilityConfig};
    pub use crate::truth_maintenance::prelude::*;
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 5 modules, 20 types, 10 traits
- Medium scale: 15 modules, 100 types, 50 traits  
- Large scale: 50 modules, 500 types, 200 traits
- Stress test: 200 modules, 2000 types, 1000 traits

**Validation Scenarios:**
1. Module structure validation: 50 valid module configurations with proper hierarchies
2. Type safety scenarios: 100 type definitions with complex trait bounds
3. Error handling scenarios: 200 error propagation chains with nested contexts
4. Documentation completeness: 500 public items requiring documentation coverage

**Synthetic Data Generation:**
```rust
// Reproducible module structure generator
pub fn generate_module_structure(complexity: ModuleComplexity, seed: u64) -> ModuleLayout {
    let mut rng = StdRng::seed_from_u64(seed);
    ModuleLayout {
        modules: generate_modules(complexity.module_count, &mut rng),
        types: generate_types(complexity.type_count, &mut rng),
        traits: generate_traits(complexity.trait_count, &mut rng),
        dependencies: generate_dependencies(complexity.dependency_count, &mut rng),
    }
}

pub fn generate_error_scenarios(count: usize, seed: u64) -> Vec<ErrorTestCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| ErrorTestCase {
        error_type: generate_error_type(&mut rng),
        context_chain: generate_context_chain(&mut rng),
        expected_recovery: generate_recovery_strategy(&mut rng),
        test_id: format!("error_test_{}", i),
    }).collect()
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with statistical analysis
- Target metrics: <30s compilation, zero warnings/errors
- Test duration: Full compilation cycle per benchmark
- Repetitions: 10 compilation cycles for statistical significance

**Success Criteria:**
- Module compiles within 30 seconds with zero errors and zero warnings
- All error types implement Debug, Clone, and Send traits with 100% trait coverage
- Documentation includes >5 usage examples with complete code samples covering all major functions
- Code style checker passes with 100% compliance to existing codebase conventions (rustfmt + clippy clean)

**Error Recovery Procedures:**
1. **Compilation Failure**:
   - Detect: `cargo check` returns non-zero exit code or compilation errors
   - Action: Roll back to previous stable module structure, analyze error messages
   - Retry: Implement modules incrementally starting with basic traits, then add complexity

2. **Missing Dependencies**:
   - Detect: "unresolved import" or "failed to resolve" errors during compilation
   - Action: Create temporary stub implementations for missing dependencies
   - Retry: Add real implementations once dependency issues are resolved

3. **Integration Conflicts**:
   - Detect: Trait bound errors or type conflicts with existing modules
   - Action: Create compatibility layer or wrapper types to bridge differences
   - Retry: Implement gradual integration with feature flags for testing

**Rollback Procedure:**
- Time limit: 5 minutes maximum rollback time
- Steps: [1] git stash current changes [2] verify core system compiles [3] create minimal working module structure
- Validation: Run `cargo check` and verify zero compilation errors before proceeding

---

## Task 6.1.2: Define TMS Configuration Structure

**Estimated Time**: 20 minutes  
**Complexity**: Low  
**AI Task**: Create configuration management for TMS

**Prompt for AI:**
```
Create `src/truth_maintenance/config.rs` with TMS configuration:
1. TMSConfig struct with performance thresholds
2. Default implementations matching Phase 6 targets
3. Validation methods for configuration values
4. Serde support for loading from files
5. Environment variable support

Target values from specification:
- Belief revision latency: <5ms
- Context switch time: <1ms  
- Conflict detection: <2ms
- Resolution success rate: >95%
- Consistency maintenance: >99%
- Memory overhead: <10%

Code Example from existing codebase pattern (Phase 2 QualityGateConfig):
```rust
// Similar configuration pattern from existing codebase:
#[derive(Debug, Clone, Deserialize)]
pub struct QualityGateConfig {
    pub min_confidence_for_allocation: f32,  // Default: 0.8
    pub require_all_validations: bool,       // Default: true
    pub max_ambiguity_count: usize,          // Default: 0
    pub min_entity_confidence: f32,          // Default: 0.75
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            min_confidence_for_allocation: 0.8,
            require_all_validations: true,
            max_ambiguity_count: 0,
            min_entity_confidence: 0.75,
        }
    }
}
```

Expected implementation for TMS configuration with neuromorphic parameters:
```rust
// src/truth_maintenance/config.rs
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, bail};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TMSConfig {
    /// Maximum allowed belief revision latency in milliseconds
    pub max_revision_latency_ms: u64,
    
    /// Maximum allowed context switch time in milliseconds
    pub max_context_switch_ms: u64,
    
    /// Maximum allowed conflict detection time in milliseconds
    pub max_conflict_detection_ms: u64,
    
    /// Minimum required resolution success rate (0.0 to 1.0)
    pub min_resolution_success_rate: f64,
    
    /// Minimum required consistency maintenance rate (0.0 to 1.0)
    pub min_consistency_rate: f64,
    
    /// Maximum allowed memory overhead percentage (0.0 to 1.0)
    pub max_memory_overhead: f64,
    
    /// Maximum number of parallel contexts
    pub max_parallel_contexts: usize,
    
    /// Maximum belief set size for optimal performance
    pub max_belief_set_size: usize,
    
    /// Enable neuromorphic integration
    pub enable_neuromorphic_integration: bool,
    
    /// TTFS encoding parameters
    pub ttfs_min_spike_time_us: u64,
    pub ttfs_max_spike_time_us: u64,
    
    /// Lateral inhibition parameters
    pub lateral_inhibition_strength: f64,
    pub inhibition_decay_time_ms: u64,
    
    /// Cortical column parameters
    pub cortical_column_count: usize,
    pub column_refractory_period_ms: u64,
    
    /// Spike timing dependent plasticity (STDP) parameters
    pub stdp_learning_rate: f64,
    pub stdp_time_window_ms: u64,
    
    /// Logging level for TMS operations
    pub log_level: String,
}

impl Default for TMSConfig {
    fn default() -> Self {
        Self {
            max_revision_latency_ms: 5,      // <5ms target
            max_context_switch_ms: 1,        // <1ms target
            max_conflict_detection_ms: 2,    // <2ms target
            min_resolution_success_rate: 0.95, // >95% target
            min_consistency_rate: 0.99,      // >99% target
            max_memory_overhead: 0.10,       // <10% target
            max_parallel_contexts: 100,
            max_belief_set_size: 10000,
            enable_neuromorphic_integration: true,
            ttfs_min_spike_time_us: 100,      // 0.1ms for max confidence
            ttfs_max_spike_time_us: 10000,    // 10ms for zero confidence
            lateral_inhibition_strength: 0.8,  // Strong inhibition for conflicts
            inhibition_decay_time_ms: 50,     // 50ms decay time
            cortical_column_count: 1000,      // Number of available columns
            column_refractory_period_ms: 10,  // 10ms refractory period
            stdp_learning_rate: 0.01,         // Learning rate for STDP
            stdp_time_window_ms: 20,          // ±20ms STDP window
            log_level: "info".to_string(),
        }
    }
}

impl TMSConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        if let Ok(val) = std::env::var("TMS_MAX_REVISION_LATENCY_MS") {
            config.max_revision_latency_ms = val.parse()
                .context("Invalid TMS_MAX_REVISION_LATENCY_MS")?;
        }
        
        if let Ok(val) = std::env::var("TMS_MAX_CONTEXT_SWITCH_MS") {
            config.max_context_switch_ms = val.parse()
                .context("Invalid TMS_MAX_CONTEXT_SWITCH_MS")?;
        }
        
        if let Ok(val) = std::env::var("TMS_ENABLE_NEUROMORPHIC") {
            config.enable_neuromorphic_integration = val.parse()
                .context("Invalid TMS_ENABLE_NEUROMORPHIC")?;
        }
        
        if let Ok(val) = std::env::var("TMS_TTFS_MIN_SPIKE_TIME_US") {
            config.ttfs_min_spike_time_us = val.parse()
                .context("Invalid TMS_TTFS_MIN_SPIKE_TIME_US")?;
        }
        
        if let Ok(val) = std::env::var("TMS_LATERAL_INHIBITION_STRENGTH") {
            config.lateral_inhibition_strength = val.parse()
                .context("Invalid TMS_LATERAL_INHIBITION_STRENGTH")?;
        }
        
        if let Ok(val) = std::env::var("TMS_CORTICAL_COLUMN_COUNT") {
            config.cortical_column_count = val.parse()
                .context("Invalid TMS_CORTICAL_COLUMN_COUNT")?;
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .context("Failed to read TMS config file")?;
        let config: Self = toml::from_str(&contents)
            .context("Failed to parse TMS config file")?;
        config.validate()?;
        Ok(config)
    }
    
    /// Validate configuration constraints
    pub fn validate(&self) -> Result<()> {
        if self.min_resolution_success_rate < 0.0 || self.min_resolution_success_rate > 1.0 {
            bail!("min_resolution_success_rate must be between 0.0 and 1.0");
        }
        
        if self.min_consistency_rate < 0.0 || self.min_consistency_rate > 1.0 {
            bail!("min_consistency_rate must be between 0.0 and 1.0");
        }
        
        if self.max_memory_overhead < 0.0 || self.max_memory_overhead > 1.0 {
            bail!("max_memory_overhead must be between 0.0 and 1.0");
        }
        
        if self.max_parallel_contexts == 0 {
            bail!("max_parallel_contexts must be greater than 0");
        }
        
        if self.max_belief_set_size == 0 {
            bail!("max_belief_set_size must be greater than 0");
        }
        
        if self.ttfs_min_spike_time_us >= self.ttfs_max_spike_time_us {
            bail!("ttfs_min_spike_time_us must be less than ttfs_max_spike_time_us");
        }
        
        if self.lateral_inhibition_strength < 0.0 || self.lateral_inhibition_strength > 1.0 {
            bail!("lateral_inhibition_strength must be between 0.0 and 1.0");
        }
        
        if self.cortical_column_count == 0 {
            bail!("cortical_column_count must be greater than 0");
        }
        
        Ok(())
    }
    
    /// Get revision latency as Duration
    pub fn revision_latency(&self) -> Duration {
        Duration::from_millis(self.max_revision_latency_ms)
    }
    
    /// Get context switch time as Duration
    pub fn context_switch_time(&self) -> Duration {
        Duration::from_millis(self.max_context_switch_ms)
    }
    
    /// Get conflict detection time as Duration
    pub fn conflict_detection_time(&self) -> Duration {
        Duration::from_millis(self.max_conflict_detection_ms)
    }
    
    /// Get TTFS encoding range
    pub fn ttfs_range(&self) -> (Duration, Duration) {
        (
            Duration::from_micros(self.ttfs_min_spike_time_us),
            Duration::from_micros(self.ttfs_max_spike_time_us)
        )
    }
    
    /// Get STDP time window
    pub fn stdp_time_window(&self) -> Duration {
        Duration::from_millis(self.stdp_time_window_ms)
    }
    
    /// Get column refractory period
    pub fn refractory_period(&self) -> Duration {
        Duration::from_millis(self.column_refractory_period_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tms_config_defaults() {
        let config = TMSConfig::default();
        assert_eq!(config.max_revision_latency_ms, 5);
        assert_eq!(config.max_context_switch_ms, 1);
        assert_eq!(config.max_conflict_detection_ms, 2);
        assert_eq!(config.min_resolution_success_rate, 0.95);
        assert_eq!(config.min_consistency_rate, 0.99);
        assert_eq!(config.max_memory_overhead, 0.10);
        assert!(config.enable_neuromorphic_integration);
        assert_eq!(config.ttfs_min_spike_time_us, 100);
        assert_eq!(config.ttfs_max_spike_time_us, 10000);
        assert_eq!(config.lateral_inhibition_strength, 0.8);
        assert_eq!(config.cortical_column_count, 1000);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = TMSConfig::default();
        assert!(config.validate().is_ok());
        
        config.min_resolution_success_rate = 1.5;
        assert!(config.validate().is_err());
        
        config.min_resolution_success_rate = 0.95;
        config.max_parallel_contexts = 0;
        assert!(config.validate().is_err());
        
        config.max_parallel_contexts = 100;
        config.ttfs_min_spike_time_us = 10000;
        config.ttfs_max_spike_time_us = 100;  // Invalid: min > max
        assert!(config.validate().is_err());
    }
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 10 config parameters, 5 environments, 3 file formats
- Medium scale: 50 config parameters, 20 environments, 10 file formats  
- Large scale: 200 config parameters, 100 environments, 25 file formats
- Stress test: 1000 config parameters, 500 environments, 100 file formats

**Validation Scenarios:**
1. Configuration validity: 100 valid configurations with performance targets met
2. Invalid constraint scenarios: 200 configurations with known violations
3. Environment loading: 50 environment variable combinations
4. File format support: Configurations in TOML, JSON, YAML formats
5. Concurrent access: 100 simultaneous configuration load operations

**Synthetic Data Generation:**
```rust
// Reproducible configuration generator
pub fn generate_tms_config_set(size: usize, seed: u64) -> Vec<TMSConfig> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|i| TMSConfig {
        max_revision_latency_ms: rng.gen_range(1..20),
        max_context_switch_ms: rng.gen_range(1..10),
        max_conflict_detection_ms: rng.gen_range(1..15),
        min_resolution_success_rate: rng.gen_range(0.5..1.0),
        min_consistency_rate: rng.gen_range(0.8..1.0),
        max_memory_overhead: rng.gen_range(0.05..0.30),
        max_parallel_contexts: rng.gen_range(10..1000),
        max_belief_set_size: rng.gen_range(100..50000),
        enable_neuromorphic_integration: rng.gen_bool(0.8),
        log_level: ["trace", "debug", "info", "warn", "error"][rng.gen_range(0..5)].to_string(),
    }).collect()
}

pub fn generate_invalid_configs(count: usize, seed: u64) -> Vec<InvalidConfigTest> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| InvalidConfigTest {
        config: generate_invalid_config(&mut rng),
        expected_error: generate_expected_validation_error(&mut rng),
        test_scenario: format!("invalid_config_{}", i),
    }).collect()
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with concurrent access benchmarks
- Target metrics: <50ms loading, >10 concurrent operations, 100% validation accuracy
- Test duration: 30 seconds per configuration benchmark
- Repetitions: 100 load cycles for statistical significance

**Success Criteria:**
- Configuration validation catches 100% of invalid constraint values with specific error messages
- File and environment loading completes within 50ms and supports >10 concurrent load operations
- Default values meet all Phase 6 performance targets: <5ms revision latency, >95% resolution success rate, <10% memory overhead
- Field documentation achieves >90% coverage with rustdoc, including parameter ranges and default explanations

**Error Recovery Procedures:**
1. **Validation Logic Failure**:
   - Detect: Configuration validation accepts invalid values or panics on valid inputs
   - Action: Implement defensive validation with comprehensive bounds checking
   - Retry: Test with fuzzing techniques and edge case inputs to verify robustness

2. **Serialization Errors**:
   - Detect: Serde deserialization failures or toml parsing errors
   - Action: Add custom deserialize implementations with detailed error messages
   - Retry: Implement graceful degradation to default values for corrupted config files

3. **Environment Variable Parsing**:
   - Detect: Environment variable parsing failures or type conversion errors
   - Action: Implement robust string parsing with fallback to configuration file values
   - Retry: Add environment variable validation with clear error messages

**Rollback Procedure:**
- Time limit: 3 minutes maximum rollback time
- Steps: [1] revert to hardcoded default configuration [2] disable file/env loading [3] verify basic TMS initialization works
- Validation: Test TMS initialization with default config and verify all validation functions work correctly

---

## Task 6.1.3: Create TMS Error Hierarchy

**Estimated Time**: 25 minutes  
**Complexity**: Low  
**AI Task**: Implement comprehensive error handling

**Prompt for AI:**
```
Create `src/truth_maintenance/errors.rs` with error types:
1. TMSError as main error enum
2. Specific error types: RevisionError, ConflictError, EntrenchmentError
3. Context-aware error messages
4. Error conversion implementations
5. Integration with existing error handling patterns

Error categories needed:
- Consistency violations
- Revision failures
- Context switching errors
- Temporal reasoning errors
- Resource exhaustion errors

Code Example from existing codebase pattern (Phase 1 error handling):
```rust
// Similar error pattern from Phase 1:
#[derive(Debug, thiserror::Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition { from: ColumnState, to: ColumnState },
    
    #[error("State mismatch: expected {expected:?}, found {actual:?}")]
    StateMismatch { expected: ColumnState, actual: ColumnState },
}
```

Expected implementation for TMS errors:
```rust
// src/truth_maintenance/errors.rs
use thiserror::Error;
use std::fmt;

/// Main TMS error type covering all truth maintenance system failures
#[derive(Debug, Error)]
pub enum TMSError {
    #[error("Belief revision error: {0}")]
    Revision(#[from] RevisionError),
    
    #[error("Conflict handling error: {0}")]
    Conflict(#[from] ConflictError),
    
    #[error("Entrenchment calculation error: {0}")]
    Entrenchment(#[from] EntrenchmentError),
    
    #[error("Context switching error: {0}")]
    ContextSwitch(#[from] ContextSwitchError),
    
    #[error("Temporal reasoning error: {0}")]
    Temporal(#[from] TemporalError),
    
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),
    
    #[error("Integration error: {0}")]
    Integration(String),
    
    #[error("Performance limit exceeded: {operation} took {actual_ms}ms, limit is {limit_ms}ms")]
    PerformanceLimit {
        operation: String,
        actual_ms: u64,
        limit_ms: u64,
    },
    
    #[error("Resource exhaustion: {resource} usage {current}/{max}")]
    ResourceExhaustion {
        resource: String,
        current: usize,
        max: usize,
    },
}

/// Errors related to belief revision operations
#[derive(Debug, Error)]
pub enum RevisionError {
    #[error("AGM postulate violation: {postulate} failed for operation {operation}")]
    PostulateViolation {
        postulate: String,
        operation: String,
    },
    
    #[error("Inconsistent belief set after revision: {details}")]
    InconsistentResult { details: String },
    
    #[error("Invalid belief format: {belief_id} - {reason}")]
    InvalidBelief {
        belief_id: String,
        reason: String,
    },
    
    #[error("Minimal change calculation failed: {reason}")]
    MinimalChangeFailure { reason: String },
    
    #[error("Revision strategy '{strategy}' failed: {error}")]
    StrategyFailure {
        strategy: String,
        error: String,
    },
}

/// Errors related to conflict detection and resolution
#[derive(Debug, Error)]
pub enum ConflictError {
    #[error("Unresolvable conflict detected between beliefs {belief1} and {belief2}")]
    UnresolvableConflict {
        belief1: String,
        belief2: String,
    },
    
    #[error("Conflict detection timeout after {timeout_ms}ms")]
    DetectionTimeout { timeout_ms: u64 },
    
    #[error("Invalid conflict type: {conflict_type}")]
    InvalidConflictType { conflict_type: String },
    
    #[error("Resolution strategy '{strategy}' failed: {reason}")]
    ResolutionFailure {
        strategy: String,
        reason: String,
    },
    
    #[error("Circular dependency detected in conflict chain: {chain}")]
    CircularDependency { chain: String },
}

/// Errors related to epistemic entrenchment calculations
#[derive(Debug, Error)]
pub enum EntrenchmentError {
    #[error("Invalid entrenchment value {value}: must be in range [0.0, 1.0]")]
    InvalidValue { value: f64 },
    
    #[error("Entrenchment ordering inconsistency: {belief1} vs {belief2}")]
    OrderingInconsistency {
        belief1: String,
        belief2: String,
    },
    
    #[error("Missing entrenchment data for belief {belief_id}")]
    MissingData { belief_id: String },
    
    #[error("Entrenchment calculation failed: {reason}")]
    CalculationFailure { reason: String },
}

/// Errors related to context switching operations
#[derive(Debug, Error)]
pub enum ContextSwitchError {
    #[error("Context {context_id} not found")]
    ContextNotFound { context_id: String },
    
    #[error("Context switch timeout: {timeout_ms}ms exceeded")]
    SwitchTimeout { timeout_ms: u64 },
    
    #[error("Invalid context state: {context_id} is in state {state}")]
    InvalidContextState {
        context_id: String,
        state: String,
    },
    
    #[error("Maximum contexts exceeded: {current}/{max}")]
    MaxContextsExceeded { current: usize, max: usize },
}

/// Errors related to temporal reasoning
#[derive(Debug, Error)]
pub enum TemporalError {
    #[error("Temporal paradox detected: {description}")]
    Paradox { description: String },
    
    #[error("Invalid timestamp: {timestamp}")]
    InvalidTimestamp { timestamp: String },
    
    #[error("Time travel query failed: {reason}")]
    TimeTravelFailure { reason: String },
    
    #[error("Temporal consistency violation: {violation}")]
    ConsistencyViolation { violation: String },
}

/// Errors related to TMS configuration
#[derive(Debug, Error)]
pub enum ConfigurationError {
    #[error("Invalid configuration value for {parameter}: {value} - {reason}")]
    InvalidValue {
        parameter: String,
        value: String,
        reason: String,
    },
    
    #[error("Missing required configuration: {parameter}")]
    MissingParameter { parameter: String },
    
    #[error("Configuration validation failed: {details}")]
    ValidationFailure { details: String },
}

/// Result type alias for TMS operations
pub type TMSResult<T> = Result<T, TMSError>;

impl TMSError {
    /// Create a performance limit error
    pub fn performance_limit(operation: impl Into<String>, actual_ms: u64, limit_ms: u64) -> Self {
        Self::PerformanceLimit {
            operation: operation.into(),
            actual_ms,
            limit_ms,
        }
    }
    
    /// Create a resource exhaustion error
    pub fn resource_exhaustion(resource: impl Into<String>, current: usize, max: usize) -> Self {
        Self::ResourceExhaustion {
            resource: resource.into(),
            current,
            max,
        }
    }
    
    /// Check if this error represents a recoverable condition
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            TMSError::ContextSwitch(ContextSwitchError::SwitchTimeout { .. })
                | TMSError::Conflict(ConflictError::DetectionTimeout { .. })
                | TMSError::PerformanceLimit { .. }
        )
    }
    
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            TMSError::Configuration(_) => ErrorSeverity::Critical,
            TMSError::Temporal(TemporalError::Paradox { .. }) => ErrorSeverity::Critical,
            TMSError::Conflict(ConflictError::UnresolvableConflict { .. }) => ErrorSeverity::High,
            TMSError::ResourceExhaustion { .. } => ErrorSeverity::High,
            TMSError::PerformanceLimit { .. } => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversion() {
        let revision_err = RevisionError::InvalidBelief {
            belief_id: "test".to_string(),
            reason: "malformed".to_string(),
        };
        let tms_err: TMSError = revision_err.into();
        
        assert!(matches!(tms_err, TMSError::Revision(_)));
        assert!(tms_err.to_string().contains("Belief revision error"));
    }
    
    #[test]
    fn test_performance_limit_error() {
        let err = TMSError::performance_limit("revision", 10, 5);
        assert!(!err.is_recoverable());
        assert_eq!(err.severity(), ErrorSeverity::Medium);
    }
    
    #[test]
    fn test_error_severity_classification() {
        let config_err = TMSError::Configuration(ConfigurationError::MissingParameter {
            parameter: "test".to_string(),
        });
        assert_eq!(config_err.severity(), ErrorSeverity::Critical);
        
        let perf_err = TMSError::performance_limit("test", 10, 5);
        assert_eq!(perf_err.severity(), ErrorSeverity::Medium);
    }
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 20 error types, 10 error chains, 5 severity levels
- Medium scale: 100 error types, 50 error chains, 8 severity levels
- Large scale: 500 error types, 200 error chains, 12 severity levels
- Stress test: 2000 error types, 1000 error chains, 20 severity levels

**Validation Scenarios:**
1. Error type coverage: All possible TMS error conditions with proper categorization
2. Error chain validation: Complex nested error scenarios with context preservation
3. Recovery testing: Error handling and graceful degradation scenarios
4. Performance impact: Error handling overhead measurement across error types
5. Diagnostic scenarios: Error message clarity and actionable guidance

**Synthetic Data Generation:**
```rust
// Reproducible error scenario generator
pub fn generate_error_test_suite(size: usize, seed: u64) -> Vec<ErrorTestScenario> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|i| ErrorTestScenario {
        error_type: generate_tms_error_type(&mut rng),
        chain_depth: rng.gen_range(1..6),
        recovery_strategy: generate_recovery_strategy(&mut rng),
        performance_impact: rng.gen_range(0.0..0.02),
        test_id: format!("error_scenario_{}", i),
    }).collect()
}

pub fn generate_error_chains(max_depth: usize, seed: u64) -> Vec<ErrorChain> {
    let mut rng = StdRng::seed_from_u64(seed);
    (1..=max_depth).map(|depth| ErrorChain {
        depth,
        root_cause: generate_root_cause(&mut rng),
        intermediate_errors: generate_intermediate_errors(depth - 1, &mut rng),
        context_preservation: validate_context_preservation(depth),
    }).collect()
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with error handling benchmarks
- Target metrics: <1% overhead, <5 chain depth, >95% pattern match
- Test duration: Error handling performance over 1000 operations
- Repetitions: 50 error handling cycles for statistical significance

**Success Criteria:**
- 100% of error messages include specific corrective actions or diagnostic steps
- Error chain depth <5 levels with full context preservation through error stack
- Error handling pattern matches existing codebase patterns in >95% of cases (validated by code review)
- Error handling overhead <1% of total operation time measured via benchmarks

**Error Recovery Procedures:**
1. **Error Type Design Failures**:
   - Detect: Compiler errors about missing trait implementations or infinite recursion in error chains
   - Action: Implement minimal error types first, then add complexity incrementally
   - Retry: Use existing codebase error patterns as templates to ensure consistency

2. **Error Conversion Issues**:
   - Detect: Type conversion failures or missing From/Into implementations
   - Action: Create explicit conversion functions and validate all error paths
   - Retry: Test error conversion chain with unit tests for each error type

3. **Performance Overhead**:
   - Detect: Benchmarks show >1% overhead from error handling infrastructure
   - Action: Implement zero-cost abstractions and lazy error message generation
   - Retry: Profile error handling paths and optimize hot paths with minimal allocations

**Rollback Procedure:**
- Time limit: 4 minutes maximum rollback time
- Steps: [1] revert to basic Result<T, String> error handling [2] implement minimal TMSError enum [3] add specific error types incrementally
- Validation: Verify error handling compiles and integrates with existing anyhow usage patterns

---

## Task 6.1.4: Define Core TMS Data Structures

**Estimated Time**: 45 minutes  
**Complexity**: Medium  
**AI Task**: Create fundamental data structures

**Prompt for AI:**
```
Create `src/truth_maintenance/types.rs` with core types:
1. BeliefNode struct with neuromorphic integration
2. BeliefStatus enum (IN/OUT/UNKNOWN)
3. Justification structures with spike encoding
4. Context and ContextId types
5. Belief version tracking structures

Integration requirements:
- Must work with existing Entity and EntityData
- Support spike-based confidence measures
- Include temporal validity periods
- Provide efficient serialization
- Support concurrent access patterns

Code Example from existing codebase pattern (Phase 1 state types):
```rust
// Similar state enum pattern from Phase 1:
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnState {
    Available = 0,
    Activated = 1,
    Competing = 2,
    Allocated = 3,
    Refractory = 4,
}
```

Expected implementation for TMS types with neuromorphic spike pattern integration:
```rust
// src/truth_maintenance/types.rs
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for beliefs
pub type BeliefId = Uuid;

/// Unique identifier for contexts
pub type ContextId = Uuid;

/// Unique identifier for justifications
pub type JustificationId = Uuid;

/// Belief status in the TMS
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefStatus {
    IN = 1,      // Believed to be true
    OUT = 0,     // Believed to be false
    UNKNOWN = 2, // Status undetermined
}

/// Core belief node with neuromorphic integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefNode {
    pub id: BeliefId,
    pub content: String,
    pub status: BeliefStatus,
    pub confidence: f64,
    pub spike_pattern: SpikePattern,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
    pub justifications: Vec<JustificationId>,
    pub contexts: Vec<ContextId>,
    pub version: u64,
}

/// Spike pattern for neuromorphic encoding with TTFS (Time-To-First-Spike)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePattern {
    pub ttfs_values: Vec<f64>,     // Time-to-first-spike values in milliseconds
    pub frequency: f64,            // Spike frequency (Hz)
    pub strength: f64,             // Synaptic strength (0.0-1.0)
    pub pattern_type: SpikePatternType,
    pub temporal_window: Duration, // Time window for pattern
}

/// Types of spike patterns for different belief representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpikePatternType {
    /// Single spike for simple belief confidence
    TTFSSingle { spike_time: Duration },
    /// Burst pattern for complex belief structures
    TTFSBurst { first_spike: Duration, burst_count: u8, inter_spike_interval: Duration },
    /// Frequency coding for belief strength
    FrequencyCoded { rate: f64, duration: Duration },
    /// Population vector for multi-dimensional beliefs
    PopulationVector { spike_times: Vec<Duration>, neuron_ids: Vec<u32> },
}

impl SpikePattern {
    /// Create TTFS pattern for belief confidence
    /// Higher confidence = earlier first spike time
    pub fn ttfs_confidence(confidence: f64) -> Self {
        let spike_time = Duration::from_micros(
            (100 + (1.0 - confidence) * 9900.0) as u64 // 0.1ms to 10ms range
        );
        
        Self {
            ttfs_values: vec![spike_time.as_secs_f64() * 1000.0], // Convert to milliseconds
            frequency: 1.0 / spike_time.as_secs_f64(), // Inverse relationship
            strength: confidence,
            pattern_type: SpikePatternType::TTFSSingle { spike_time },
            temporal_window: Duration::from_millis(50),
        }
    }
    
    /// Create burst pattern for complex belief with multiple justifications
    pub fn ttfs_burst(confidence: f64, justification_count: u8) -> Self {
        let first_spike = Duration::from_micros(
            (100 + (1.0 - confidence) * 9900.0) as u64
        );
        let inter_spike = Duration::from_micros(500); // 0.5ms between spikes
        
        let mut ttfs_values = Vec::new();
        for i in 0..justification_count {
            let spike_time = first_spike + inter_spike * i as u32;
            ttfs_values.push(spike_time.as_secs_f64() * 1000.0);
        }
        
        Self {
            ttfs_values,
            frequency: justification_count as f64 / (first_spike.as_secs_f64() + 
                      inter_spike.as_secs_f64() * justification_count as f64),
            strength: confidence,
            pattern_type: SpikePatternType::TTFSBurst {
                first_spike,
                burst_count: justification_count,
                inter_spike_interval: inter_spike,
            },
            temporal_window: Duration::from_millis(50),
        }
    }
    
    /// Extract confidence from TTFS pattern
    pub fn extract_confidence(&self) -> f64 {
        match &self.pattern_type {
            SpikePatternType::TTFSSingle { spike_time } => {
                let normalized = (spike_time.as_micros() - 100) as f64 / 9900.0;
                1.0 - normalized.clamp(0.0, 1.0)
            },
            SpikePatternType::TTFSBurst { first_spike, .. } => {
                let normalized = (first_spike.as_micros() - 100) as f64 / 9900.0;
                1.0 - normalized.clamp(0.0, 1.0)
            },
            _ => self.strength,
        }
    }
    
    /// Check if this pattern conflicts with another (for lateral inhibition)
    pub fn conflicts_with(&self, other: &SpikePattern) -> bool {
        // Patterns conflict if they have overlapping temporal windows
        // and similar spike timing (representing competing beliefs)
        let self_first = self.ttfs_values.get(0).unwrap_or(&0.0);
        let other_first = other.ttfs_values.get(0).unwrap_or(&0.0);
        
        (self_first - other_first).abs() < 2.0 // Within 2ms = conflict
    }
}

/// Justification for belief support with spike-based strength encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Justification {
    pub id: JustificationId,
    pub antecedents: Vec<BeliefId>,
    pub consequent: BeliefId,
    pub rule_type: RuleType,
    pub strength: f64,
    pub spike_encoding: JustificationSpikeEncoding,
    pub created_at: SystemTime,
    pub plasticity_state: STDPState,
}

/// Spike-based encoding for justification strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JustificationSpikeEncoding {
    /// Synaptic weight (strength of connection)
    pub synaptic_weight: f64,
    /// Spike timing dependent plasticity parameters
    pub stdp_trace: f64,
    /// Propagation delay for spike transmission
    pub propagation_delay: Duration,
    /// Last spike time for STDP calculations
    pub last_spike_time: Option<SystemTime>,
}

/// STDP (Spike-Timing Dependent Plasticity) state for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPState {
    pub pre_synaptic_trace: f64,
    pub post_synaptic_trace: f64,
    pub last_update: SystemTime,
    pub potentiation_window: Duration,  // Window for LTP (Long-Term Potentiation)
    pub depression_window: Duration,    // Window for LTD (Long-Term Depression)
}

impl JustificationSpikeEncoding {
    /// Create new spike encoding for justification
    pub fn new(initial_strength: f64) -> Self {
        Self {
            synaptic_weight: initial_strength,
            stdp_trace: 0.0,
            propagation_delay: Duration::from_micros(100 + (initial_strength * 900.0) as u64),
            last_spike_time: None,
        }
    }
    
    /// Update synaptic weight based on STDP rule
    /// Pre-before-post: potentiation (strengthen)
    /// Post-before-pre: depression (weaken)
    pub fn update_stdp(
        &mut self,
        pre_spike_time: SystemTime,
        post_spike_time: SystemTime,
        learning_rate: f64,
    ) -> Result<(), TMSError> {
        let time_diff = if pre_spike_time < post_spike_time {
            post_spike_time.duration_since(pre_spike_time)
                .map_err(|e| TMSError::Integration(format!("Time calculation error: {}", e)))?
        } else {
            pre_spike_time.duration_since(post_spike_time)
                .map_err(|e| TMSError::Integration(format!("Time calculation error: {}", e)))?
        };
        
        let stdp_window = Duration::from_millis(20); // ±20ms STDP window
        
        if time_diff <= stdp_window {
            if pre_spike_time < post_spike_time {
                // Potentiation: strengthen connection
                let weight_increase = learning_rate * 
                    (-time_diff.as_secs_f64() / stdp_window.as_secs_f64()).exp();
                self.synaptic_weight = (self.synaptic_weight + weight_increase).min(1.0);
            } else {
                // Depression: weaken connection
                let weight_decrease = learning_rate * 
                    (-time_diff.as_secs_f64() / stdp_window.as_secs_f64()).exp();
                self.synaptic_weight = (self.synaptic_weight - weight_decrease).max(0.0);
            }
        }
        
        self.last_spike_time = Some(post_spike_time);
        Ok(())
    }
}

impl STDPState {
    pub fn new() -> Self {
        Self {
            pre_synaptic_trace: 0.0,
            post_synaptic_trace: 0.0,
            last_update: SystemTime::now(),
            potentiation_window: Duration::from_millis(20),
            depression_window: Duration::from_millis(20),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    Modus_Ponens,
    Abduction,
    Default,
    Assumption,
}

/// Context for assumption-based reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub id: ContextId,
    pub name: String,
    pub assumptions: Vec<BeliefId>,
    pub beliefs: Vec<BeliefId>,
    pub consistency_status: ConsistencyStatus,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyStatus {
    Consistent,
    Inconsistent,
    Unknown,
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 100 beliefs, 50 justifications, 10 contexts
- Medium scale: 1000 beliefs, 500 justifications, 50 contexts
- Large scale: 10000 beliefs, 5000 justifications, 200 contexts
- Stress test: 100000 beliefs, 50000 justifications, 1000 contexts

**Validation Scenarios:**
1. Type compatibility: Integration with existing entity system types
2. Concurrency testing: Multi-threaded access patterns with race condition detection
3. Memory efficiency: Memory usage profiling under various load conditions
4. Spike pattern validation: Neuromorphic spike encoding and timing preservation
5. Data structure integrity: Invariant checking across all data types

**Synthetic Data Generation:**
```rust
// Reproducible TMS data structure generator
pub fn generate_belief_node_set(size: usize, seed: u64) -> Vec<BeliefNode> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|i| BeliefNode {
        id: BeliefId::from_u128(rng.gen()),
        content: format!("test_belief_{}", i),
        status: generate_belief_status(&mut rng),
        confidence: rng.gen_range(0.0..1.0),
        spike_pattern: generate_spike_pattern(&mut rng),
        created_at: SystemTime::now(),
        last_updated: SystemTime::now(),
        justifications: generate_justification_ids(rng.gen_range(0..5), &mut rng),
        contexts: generate_context_ids(rng.gen_range(0..3), &mut rng),
        version: rng.gen_range(1..100),
    }).collect()
}

pub fn generate_spike_pattern(rng: &mut StdRng) -> SpikePattern {
    let confidence = rng.gen_range(0.0..1.0);
    let pattern_choice = rng.gen_range(0..3);
    
    match pattern_choice {
        0 => SpikePattern::ttfs_confidence(confidence),
        1 => SpikePattern::ttfs_burst(confidence, rng.gen_range(2..6)),
        _ => {
            let spike_count = rng.gen_range(1..10);
            SpikePattern {
                ttfs_values: (0..spike_count)
                    .map(|_| rng.gen_range(0.1..10.0))
                    .collect(),
                frequency: rng.gen_range(1.0..100.0),
                strength: confidence,
                pattern_type: SpikePatternType::FrequencyCoded {
                    rate: rng.gen_range(10.0..200.0),
                    duration: Duration::from_millis(rng.gen_range(10..100)),
                },
                temporal_window: Duration::from_millis(50),
            }
        }
    }
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with concurrent access and memory profiling
- Target metrics: <20% memory overhead, <5% performance degradation, zero data races
- Test duration: 60 seconds concurrent access with 50+ threads
- Repetitions: 100 cycles for thread safety validation

**Success Criteria:**
- Type integration compiles without conversion warnings and passes 100% of existing entity system tests
- Concurrent access by >50 threads shows <5% performance degradation with zero data races (validated by thread sanitizer)
- Memory usage <20% overhead compared to raw data size measured via memory profiler
- Validation logic covers 100% of invariants with property-based testing across >1000 random inputs

**Error Recovery Procedures:**
1. **Type Integration Conflicts**:
   - Detect: Compilation errors from incompatible types with existing entity system
   - Action: Create adapter types and conversion functions to bridge type differences
   - Retry: Implement gradual migration path with both old and new types supported

2. **Concurrency Issues**:
   - Detect: Thread sanitizer reports data races or deadlocks in concurrent tests
   - Action: Implement proper synchronization primitives and lock-free data structures where possible
   - Retry: Use atomic operations and message passing to eliminate shared mutable state

3. **Memory Performance Issues**:
   - Detect: Memory profiler shows >20% overhead or excessive allocations
   - Action: Implement custom serialization, use arena allocation, and optimize data layout
   - Retry: Profile memory usage patterns and implement zero-copy operations where feasible

**Rollback Procedure:**
- Time limit: 6 minutes maximum rollback time
- Steps: [1] revert to basic struct definitions without complex features [2] disable concurrent access temporarily [3] implement minimal viable types
- Validation: Run existing entity system tests to ensure no regressions and verify basic type operations work

---

## Task 6.1.5: Create TMS Health Metrics Framework

**Estimated Time**: 30 minutes  
**Complexity**: Low  
**AI Task**: Set up monitoring infrastructure

**Prompt for AI:**
```
Create `src/truth_maintenance/metrics.rs` with monitoring:
1. TMSHealthMetrics struct with key performance indicators
2. Metric collection methods with minimal overhead
3. Integration with existing monitoring system
4. Performance threshold validation
5. Alert generation for degraded performance

Key metrics to track:
- Belief consistency ratio
- Context switch latency
- Revisions per minute
- Resolution success rate
- Entrenchment stability

Expected implementation with neuromorphic spike pattern monitoring:
```rust
// src/truth_maintenance/metrics.rs
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct TMSHealthMetrics {
    // Performance counters
    belief_revisions: AtomicU64,
    context_switches: AtomicU64,
    conflicts_detected: AtomicU64,
    conflicts_resolved: AtomicU64,
    
    // Timing metrics (in nanoseconds)
    total_revision_time: AtomicU64,
    total_context_switch_time: AtomicU64,
    total_conflict_detection_time: AtomicU64,
    
    // Quality metrics
    consistency_ratio: AtomicF64,
    resolution_success_rate: AtomicF64,
    
    // Neuromorphic-specific metrics
    spike_pattern_accuracy: AtomicF64,
    ttfs_encoding_latency: AtomicU64,
    lateral_inhibition_events: AtomicU64,
    cortical_column_utilization: AtomicF64,
    stdp_weight_changes: AtomicU64,
    winner_take_all_decisions: AtomicU64,
    
    start_time: Instant,
    
    // Detailed spike timing metrics
    spike_timing_histogram: Arc<RwLock<HashMap<u64, u64>>>, // Microsecond buckets
    column_activation_patterns: Arc<RwLock<Vec<ColumnActivationRecord>>>,
}

impl TMSHealthMetrics {
    pub fn new() -> Self {
        Self {
            belief_revisions: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            conflicts_detected: AtomicU64::new(0),
            conflicts_resolved: AtomicU64::new(0),
            total_revision_time: AtomicU64::new(0),
            total_context_switch_time: AtomicU64::new(0),
            total_conflict_detection_time: AtomicU64::new(0),
            consistency_ratio: AtomicF64::new(1.0),
            resolution_success_rate: AtomicF64::new(0.0),
            spike_pattern_accuracy: AtomicF64::new(1.0),
            ttfs_encoding_latency: AtomicU64::new(0),
            lateral_inhibition_events: AtomicU64::new(0),
            cortical_column_utilization: AtomicF64::new(0.0),
            stdp_weight_changes: AtomicU64::new(0),
            winner_take_all_decisions: AtomicU64::new(0),
            start_time: Instant::now(),
            spike_timing_histogram: Arc::new(RwLock::new(HashMap::new())),
            column_activation_patterns: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub fn record_revision(&self, duration: Duration) {
        self.belief_revisions.fetch_add(1, Ordering::Relaxed);
        self.total_revision_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }
    
    pub fn average_revision_latency(&self) -> Duration {
        let count = self.belief_revisions.load(Ordering::Relaxed);
        if count == 0 { return Duration::ZERO; }
        
        let total_ns = self.total_revision_time.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / count)
    }
    
    pub fn revisions_per_minute(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64() / 60.0;
        if elapsed == 0.0 { return 0.0; }
        
        self.belief_revisions.load(Ordering::Relaxed) as f64 / elapsed
    }
    
    /// Record TTFS encoding performance
    pub fn record_ttfs_encoding(&self, duration: Duration, accuracy: f64) {
        self.ttfs_encoding_latency.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        
        // Update accuracy using exponential moving average
        let current_accuracy = f64::from_bits(
            self.spike_pattern_accuracy.load(Ordering::Relaxed)
        );
        let alpha = 0.1; // Smoothing factor
        let new_accuracy = alpha * accuracy + (1.0 - alpha) * current_accuracy;
        self.spike_pattern_accuracy.store(new_accuracy.to_bits(), Ordering::Relaxed);
        
        // Update spike timing histogram
        let bucket = (duration.as_micros() / 10) as u64; // 10μs buckets
        if let Ok(mut histogram) = self.spike_timing_histogram.try_write() {
            *histogram.entry(bucket).or_insert(0) += 1;
        }
    }
    
    /// Record lateral inhibition event
    pub fn record_lateral_inhibition(&self, winning_column: u32, suppressed_columns: &[u32]) {
        self.lateral_inhibition_events.fetch_add(1, Ordering::Relaxed);
        self.winner_take_all_decisions.fetch_add(1, Ordering::Relaxed);
        
        // Record column activation pattern
        let activation_record = ColumnActivationRecord {
            timestamp: SystemTime::now(),
            winner: winning_column,
            suppressed: suppressed_columns.to_vec(),
            inhibition_strength: 0.8, // From config
        };
        
        if let Ok(mut patterns) = self.column_activation_patterns.try_write() {
            patterns.push(activation_record);
            // Keep only last 1000 records to prevent unbounded growth
            if patterns.len() > 1000 {
                patterns.remove(0);
            }
        }
    }
    
    /// Record STDP weight change
    pub fn record_stdp_update(&self, weight_change: f64) {
        self.stdp_weight_changes.fetch_add(1, Ordering::Relaxed);
        // Could add more detailed STDP analytics here
    }
    
    /// Get average TTFS encoding latency
    pub fn average_ttfs_latency(&self) -> Duration {
        let revisions = self.belief_revisions.load(Ordering::Relaxed);
        if revisions == 0 { return Duration::ZERO; }
        
        let total_ns = self.ttfs_encoding_latency.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / revisions)
    }
    
    /// Get current spike pattern accuracy
    pub fn spike_pattern_accuracy(&self) -> f64 {
        f64::from_bits(self.spike_pattern_accuracy.load(Ordering::Relaxed))
    }
    
    /// Get cortical column utilization rate
    pub fn column_utilization(&self) -> f64 {
        f64::from_bits(self.cortical_column_utilization.load(Ordering::Relaxed))
    }
    
    /// Update cortical column utilization
    pub fn update_column_utilization(&self, active_columns: usize, total_columns: usize) {
        let utilization = if total_columns > 0 {
            active_columns as f64 / total_columns as f64
        } else {
            0.0
        };
        self.cortical_column_utilization.store(utilization.to_bits(), Ordering::Relaxed);
    }
    
    /// Get detailed neuromorphic metrics report
    pub fn neuromorphic_report(&self) -> NeuromorphicMetricsReport {
        NeuromorphicMetricsReport {
            ttfs_accuracy: self.spike_pattern_accuracy(),
            average_encoding_latency: self.average_ttfs_latency(),
            lateral_inhibition_events: self.lateral_inhibition_events.load(Ordering::Relaxed),
            column_utilization: self.column_utilization(),
            stdp_updates: self.stdp_weight_changes.load(Ordering::Relaxed),
            winner_take_all_decisions: self.winner_take_all_decisions.load(Ordering::Relaxed),
        }
    }
}

/// Record of cortical column activation for analysis
#[derive(Debug, Clone)]
pub struct ColumnActivationRecord {
    pub timestamp: SystemTime,
    pub winner: u32,
    pub suppressed: Vec<u32>,
    pub inhibition_strength: f64,
}

/// Detailed neuromorphic performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicMetricsReport {
    pub ttfs_accuracy: f64,
    pub average_encoding_latency: Duration,
    pub lateral_inhibition_events: u64,
    pub column_utilization: f64,
    pub stdp_updates: u64,
    pub winner_take_all_decisions: u64,
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 10 metrics, 1000 data points, 5 alert thresholds
- Medium scale: 50 metrics, 10000 data points, 20 alert thresholds
- Large scale: 200 metrics, 100000 data points, 100 alert thresholds
- Stress test: 1000 metrics, 1000000 data points, 500 alert thresholds

**Validation Scenarios:**
1. Metrics accuracy: TMS performance metrics correlation with actual system behavior
2. Collection overhead: Performance impact measurement with and without metrics
3. Alert system: Threshold violation detection and false positive analysis
4. Concurrent access: Multi-threaded metric updates and query performance
5. Real-time queries: Response time analysis for historical metric data

**Synthetic Data Generation:**
```rust
// Reproducible metrics test data generator
pub fn generate_tms_metrics_dataset(duration_hours: u64, seed: u64) -> MetricsDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let start_time = SystemTime::now() - Duration::from_secs(duration_hours * 3600);
    
    MetricsDataset {
        revision_times: generate_latency_series(duration_hours * 3600, 1.0, 10.0, &mut rng),
        context_switches: generate_latency_series(duration_hours * 3600, 0.1, 2.0, &mut rng),
        conflict_detections: generate_latency_series(duration_hours * 1800, 0.5, 5.0, &mut rng),
        resolution_success_rates: generate_success_rate_series(duration_hours * 60, &mut rng),
        memory_usage: generate_memory_usage_series(duration_hours * 60, &mut rng),
        start_time,
    }
}

pub fn generate_alert_scenarios(count: usize, seed: u64) -> Vec<AlertTestCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| AlertTestCase {
        metric_name: format!("test_metric_{}", i),
        threshold: rng.gen_range(0.1..10.0),
        violation_scenario: generate_violation_pattern(&mut rng),
        expected_trigger_time: Duration::from_millis(rng.gen_range(100..1000)),
    }).collect()
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with metrics overhead and alert latency benchmarks
- Target metrics: <1% overhead, <100ms integration, <500ms alert trigger, <50ms queries
- Test duration: Continuous metrics collection over 24 hours
- Repetitions: 1000 metric update cycles for overhead measurement

**Success Criteria:**
- Metrics collection overhead <1% measured via benchmark comparison (with/without metrics)
- Integration with existing monitoring completes in <100ms and supports >100 concurrent metric updates
- Alert system triggers within 500ms of threshold violation with <0.1% false positive rate
- Real-time queries return results within 50ms for >10,000 stored metric data points

**Error Recovery Procedures:**
1. **Metrics Collection Overhead**:
   - Detect: Benchmarks show >1% performance impact from metrics collection
   - Action: Implement sampling-based metrics and lazy aggregation techniques
   - Retry: Use lock-free atomic operations and batch metric updates to reduce contention

2. **Monitoring Integration Failures**:
   - Detect: Integration timeouts or connection failures with existing monitoring system
   - Action: Implement local metric storage with periodic sync and circuit breaker pattern
   - Retry: Add retry logic with exponential backoff and graceful degradation when monitoring unavailable

3. **Alert System False Positives**:
   - Detect: Alert system triggers >0.1% false positives during testing
   - Action: Implement statistical smoothing and threshold hysteresis to reduce noise
   - Retry: Add configurable sensitivity levels and multi-threshold alerting logic

**Rollback Procedure:**
- Time limit: 4 minutes maximum rollback time
- Steps: [1] disable all metric collection temporarily [2] implement basic counters only [3] add alerting incrementally
- Validation: Verify TMS operates normally without metrics and basic counters function correctly

---

## Task 6.1.6: Initialize TMS Integration Points

**Estimated Time**: 40 minutes  
**Complexity**: Medium  
**AI Task**: Set up integration with neuromorphic system

**Prompt for AI:**
```
Create integration points in existing modules:
1. Add TMS hooks to `src/core/brain_enhanced_graph/brain_graph_core.rs`
2. Extend MultiColumnProcessor for TMS validation
3. Add TMS validation to entity operations
4. Create TMS factory methods in main system
5. Update configuration loading to include TMS settings

Integration points needed:
- Entity insertion/update validation
- Query result validation
- Consensus verification
- Conflict notification
- Performance monitoring integration
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 5 integration points, 20 test modules, 100 existing tests
- Medium scale: 20 integration points, 50 test modules, 500 existing tests
- Large scale: 50 integration points, 100 test modules, 2000 existing tests
- Stress test: 200 integration points, 500 test modules, 10000 existing tests

**Validation Scenarios:**
1. Integration compatibility: TMS integration with existing neuromorphic components
2. Backward compatibility: Existing test suite validation with TMS enabled/disabled
3. Performance impact: Before/after benchmarks for core neuromorphic operations
4. Error propagation: Integration failure modes and graceful degradation testing
5. Code change minimization: Integration with minimal modifications to existing code

**Synthetic Data Generation:**
```rust
// Reproducible integration test generator
pub fn generate_integration_test_suite(seed: u64) -> IntegrationTestSuite {
    let mut rng = StdRng::seed_from_u64(seed);
    
    IntegrationTestSuite {
        entity_operations: generate_entity_operation_tests(100, &mut rng),
        query_validations: generate_query_validation_tests(50, &mut rng),
        consensus_integrations: generate_consensus_tests(25, &mut rng),
        performance_baselines: generate_performance_baselines(&mut rng),
        error_scenarios: generate_integration_error_scenarios(75, &mut rng),
    }
}

pub fn generate_backward_compatibility_tests(seed: u64) -> Vec<BackwardCompatTest> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..500).map(|i| BackwardCompatTest {
        test_name: format!("compat_test_{}", i),
        original_behavior: generate_original_behavior(&mut rng),
        with_tms_behavior: generate_tms_enhanced_behavior(&mut rng),
        compatibility_level: generate_compatibility_level(&mut rng),
    }).collect()
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with before/after integration benchmarks
- Target metrics: <5 lines per integration, 100% test pass rate, <3% performance impact
- Test duration: Complete test suite execution with integration enabled
- Repetitions: 10 full test cycles for statistical significance

**Success Criteria:**
- Integration points require <5 lines of code changes per integration site with zero breaking changes
- Backward compatibility verified by 100% pass rate of existing test suite (>500 tests)
- Performance impact <3% measured via before/after benchmarks on core operations
- Error handling covers 100% of integration failure modes with graceful degradation paths

**Error Recovery Procedures:**
1. **Integration Breaking Changes**:
   - Detect: Existing test suite failures or compilation errors after TMS integration
   - Action: Implement adapter pattern and feature flags to isolate TMS integration
   - Retry: Use decorator pattern to add TMS functionality without modifying existing code

2. **Performance Degradation**:
   - Detect: Benchmarks show >3% performance impact on core operations
   - Action: Implement async integration points and lazy validation strategies
   - Retry: Add caching layer and optimize hot paths identified through profiling

3. **Integration Cascade Failures**:
   - Detect: TMS failures cause entire system instability or cascading errors
   - Action: Implement circuit breaker pattern and graceful degradation modes
   - Retry: Add health checks and automatic TMS disable on repeated failures

**Rollback Procedure:**
- Time limit: 8 minutes maximum rollback time
- Steps: [1] disable all TMS integration hooks [2] revert to pre-integration codebase state [3] verify existing functionality intact
- Validation: Run complete existing test suite to ensure 100% pass rate and no performance regressions

---

## Task 6.1.7: Create TMS Test Infrastructure

**Estimated Time**: 35 minutes  
**Complexity**: Medium  
**AI Task**: Set up testing framework for TMS

**Prompt for AI:**
```
Create `tests/truth_maintenance/mod.rs` and infrastructure:
1. Test utilities for creating test beliefs and contexts
2. Mock implementations for testing components in isolation
3. Property-based test framework for consistency checks
4. Performance test harnesses matching target metrics
5. Integration test setup with neuromorphic system

Test categories needed:
- Unit tests for individual components
- Integration tests with neuromorphic system
- Performance benchmarks against targets
- Property-based consistency testing
- Error condition testing
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small scale: 50 test cases, 10 mock components, 5 integration scenarios
- Medium scale: 200 test cases, 50 mock components, 25 integration scenarios
- Large scale: 1000 test cases, 200 mock components, 100 integration scenarios
- Stress test: 5000 test cases, 1000 mock components, 500 integration scenarios

**Validation Scenarios:**
1. Unit test coverage: Comprehensive testing of all TMS module functions
2. Performance validation: Target metric verification across all TMS operations
3. Integration testing: TMS integration with neuromorphic system components
4. Mock validation: Isolated component testing with dependency mocking
5. CI/CD pipeline: Automated test execution and reproducibility validation

**Synthetic Data Generation:**
```rust
// Reproducible test infrastructure generator
pub fn generate_tms_test_suite(seed: u64) -> TMSTestSuite {
    let mut rng = StdRng::seed_from_u64(seed);
    
    TMSTestSuite {
        unit_tests: generate_unit_tests(1000, &mut rng),
        integration_tests: generate_integration_tests(100, &mut rng),
        performance_tests: generate_performance_tests(50, &mut rng),
        property_tests: generate_property_tests(200, &mut rng),
        stress_tests: generate_stress_tests(25, &mut rng),
    }
}

pub fn generate_mock_components(count: usize, seed: u64) -> Vec<MockComponent> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| MockComponent {
        component_type: generate_component_type(&mut rng),
        mock_behavior: generate_mock_behavior(&mut rng),
        performance_characteristics: generate_performance_mock(&mut rng),
        failure_modes: generate_failure_scenarios(&mut rng),
    }).collect()
}

pub fn generate_performance_benchmarks(seed: u64) -> PerformanceBenchmarkSuite {
    let mut rng = StdRng::seed_from_u64(seed);
    
    PerformanceBenchmarkSuite {
        revision_latency_tests: generate_latency_tests("revision", 5, &mut rng),
        context_switch_tests: generate_latency_tests("context_switch", 1, &mut rng),
        conflict_detection_tests: generate_latency_tests("conflict_detection", 2, &mut rng),
        resolution_success_tests: generate_success_rate_tests(95.0, &mut rng),
        memory_overhead_tests: generate_memory_tests(10.0, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Measurement tool: criterion.rs with comprehensive TMS performance validation
- Target metrics: >95% coverage, <5ms revision, <1ms context switch, >95% success rate
- Test duration: Complete CI/CD pipeline execution under 10 minutes
- Repetitions: 20 full test cycles for reproducibility validation

**Success Criteria:**
- Test infrastructure achieves >95% code coverage across all TMS modules with >100 unit tests
- Performance tests validate all target metrics: <5ms revision, <1ms context switch, >95% success rate
- CI/CD pipeline runs complete test suite in <10 minutes with 100% reproducible results
- Mock implementations cover 100% of external dependencies enabling isolated testing of each component

**Error Recovery Procedures:**
1. **Test Infrastructure Setup Failures**:
   - Detect: Test compilation failures or missing test dependencies
   - Action: Implement basic test structure first, then add complex test scenarios incrementally
   - Retry: Use existing test patterns from codebase as templates and copy proven testing approaches

2. **Performance Test Failures**:
   - Detect: Performance tests fail to meet target metrics during validation
   - Action: Implement performance test mocks that simulate target performance characteristics
   - Retry: Add performance profiling and optimization before re-running performance validation

3. **CI/CD Integration Issues**:
   - Detect: Test suite timeouts or non-reproducible test failures in CI environment
   - Action: Implement test isolation and deterministic test data generation
   - Retry: Add retry logic for flaky tests and parallel test execution controls

**Rollback Procedure:**
- Time limit: 6 minutes maximum rollback time
- Steps: [1] disable complex tests and keep only basic unit tests [2] remove performance requirements temporarily [3] ensure basic test infrastructure compiles
- Validation: Verify basic unit tests run successfully and provide foundation for future test expansion

---

## Validation Checklist

- [ ] All modules compile without warnings
- [ ] Error handling follows existing patterns
- [ ] Configuration supports all required parameters
- [ ] Data structures integrate with existing entity system
- [ ] Metrics framework provides real-time monitoring
- [ ] Integration points maintain backward compatibility
- [ ] Test infrastructure supports comprehensive validation
- [ ] Documentation includes examples and usage patterns
- [ ] Performance impact is measured and acceptable
- [ ] All code follows existing style and conventions

## Next Phase

Upon completion, proceed to **Phase 6.2: Core TMS Components** for implementing the hybrid JTMS-ATMS architecture.