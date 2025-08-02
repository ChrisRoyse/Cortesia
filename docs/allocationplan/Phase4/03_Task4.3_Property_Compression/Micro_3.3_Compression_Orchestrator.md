# Micro Phase 3.3: Compression Orchestrator

**Estimated Time**: 35 minutes
**Dependencies**: Micro 3.2 (Property Promotion Engine)
**Objective**: Coordinate the entire compression workflow from analysis through validation

## Task Description

Create a high-level orchestrator that manages the complete compression workflow, coordinating analysis, promotion, and validation phases while maintaining system consistency and providing comprehensive reporting.

## Deliverables

Create `src/compression/orchestrator.rs` with:

1. **CompressionOrchestrator struct**: Main workflow coordinator
2. **Workflow management**: Control analysis → promotion → validation cycle
3. **Progress tracking**: Real-time progress reporting and metrics
4. **Error recovery**: Handle partial failures gracefully
5. **Configuration management**: Centralized compression settings
6. **Metrics collection**: Detailed compression statistics

## Success Criteria

- [ ] Coordinates full compression workflow without manual intervention
- [ ] Provides real-time progress updates during long operations
- [ ] Handles errors gracefully with automatic recovery where possible
- [ ] Generates comprehensive compression reports
- [ ] Maintains system consistency throughout the process
- [ ] Achieves target 10x compression ratio when possible

## Implementation Requirements

```rust
pub struct CompressionOrchestrator {
    analyzer: PropertyAnalyzer,
    promoter: PropertyPromoter,
    validator: CompressionValidator,
    config: CompressionConfig,
    metrics_collector: MetricsCollector,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub min_frequency_threshold: f32,
    pub min_nodes_threshold: usize,
    pub promotion_strategy: PromotionStrategy,
    pub max_iterations: usize,
    pub target_compression_ratio: f32,
    pub enable_progress_reporting: bool,
    pub validation_level: ValidationLevel,
}

#[derive(Debug)]
pub struct CompressionWorkflow {
    pub phase: CompressionPhase,
    pub progress: f32,
    pub current_iteration: usize,
    pub estimated_time_remaining: Duration,
    pub metrics: WorkflowMetrics,
}

#[derive(Debug, Clone)]
pub enum CompressionPhase {
    Initialization,
    Analysis,
    Promotion,
    Validation,
    Finalization,
    Complete,
}

#[derive(Debug)]
pub struct CompressionReport {
    pub total_properties_analyzed: usize,
    pub properties_promoted: usize,
    pub exceptions_created: usize,
    pub total_bytes_saved: usize,
    pub compression_ratio: f32,
    pub iterations_performed: usize,
    pub total_execution_time: Duration,
    pub phase_timings: HashMap<CompressionPhase, Duration>,
    pub validation_results: ValidationSummary,
}

impl CompressionOrchestrator {
    pub fn new(config: CompressionConfig) -> Self;
    
    pub fn compress_hierarchy(
        &self,
        hierarchy: &mut InheritanceHierarchy
    ) -> Result<CompressionReport, CompressionError>;
    
    pub fn compress_with_progress<F>(
        &self,
        hierarchy: &mut InheritanceHierarchy,
        progress_callback: F
    ) -> Result<CompressionReport, CompressionError>
    where
        F: Fn(&CompressionWorkflow);
    
    pub fn estimate_compression_potential(
        &self,
        hierarchy: &InheritanceHierarchy
    ) -> CompressionEstimate;
    
    pub fn validate_configuration(&self, hierarchy: &InheritanceHierarchy) -> ConfigValidationResult;
    
    pub fn create_compression_plan(
        &self,
        hierarchy: &InheritanceHierarchy
    ) -> CompressionPlan;
}

#[derive(Debug)]
pub struct CompressionPlan {
    pub estimated_iterations: usize,
    pub estimated_total_time: Duration,
    pub estimated_compression_ratio: f32,
    pub potential_candidates: usize,
    pub risk_assessment: RiskAssessment,
}

#[derive(Debug)]
pub struct WorkflowMetrics {
    pub properties_processed: usize,
    pub bytes_saved_so_far: usize,
    pub current_compression_ratio: f32,
    pub errors_encountered: usize,
    pub warnings_generated: usize,
}
```

## Test Requirements

Must pass compression orchestrator tests:
```rust
#[test]
fn test_full_compression_workflow() {
    let mut hierarchy = create_complex_hierarchy(5000);
    let config = CompressionConfig {
        min_frequency_threshold: 0.7,
        min_nodes_threshold: 5,
        promotion_strategy: PromotionStrategy::Balanced,
        max_iterations: 10,
        target_compression_ratio: 10.0,
        enable_progress_reporting: true,
        validation_level: ValidationLevel::Comprehensive,
    };
    
    let orchestrator = CompressionOrchestrator::new(config);
    let report = orchestrator.compress_hierarchy(&mut hierarchy).unwrap();
    
    assert!(report.compression_ratio >= 5.0); // At least 5x compression
    assert!(report.total_execution_time < Duration::from_secs(30));
    assert_eq!(report.validation_results.errors, 0);
    assert!(report.properties_promoted > 0);
}

#[test]
fn test_progress_reporting() {
    let mut hierarchy = create_large_hierarchy(10000);
    let config = CompressionConfig::default();
    let orchestrator = CompressionOrchestrator::new(config);
    
    let mut progress_updates = Vec::new();
    
    let report = orchestrator.compress_with_progress(
        &mut hierarchy,
        |workflow| {
            progress_updates.push((workflow.phase.clone(), workflow.progress));
        }
    ).unwrap();
    
    // Should have progress updates for each phase
    assert!(progress_updates.len() >= 5);
    assert!(progress_updates.iter().any(|(phase, _)| matches!(phase, CompressionPhase::Analysis)));
    assert!(progress_updates.iter().any(|(phase, _)| matches!(phase, CompressionPhase::Promotion)));
    assert!(progress_updates.iter().any(|(phase, _)| matches!(phase, CompressionPhase::Validation)));
    
    // Progress should increase over time
    let final_progress = progress_updates.last().unwrap().1;
    assert_eq!(final_progress, 1.0);
}

#[test]
fn test_compression_estimation() {
    let hierarchy = create_animal_hierarchy();
    let config = CompressionConfig::default();
    let orchestrator = CompressionOrchestrator::new(config);
    
    let estimate = orchestrator.estimate_compression_potential(&hierarchy);
    
    assert!(estimate.estimated_compression_ratio > 1.0);
    assert!(estimate.estimated_total_time > Duration::ZERO);
    assert!(estimate.potential_candidates > 0);
}

#[test]
fn test_error_recovery() {
    let mut hierarchy = create_corrupted_hierarchy(); // Some invalid data
    let config = CompressionConfig {
        promotion_strategy: PromotionStrategy::Conservative,
        ..CompressionConfig::default()
    };
    
    let orchestrator = CompressionOrchestrator::new(config);
    let report = orchestrator.compress_hierarchy(&mut hierarchy).unwrap();
    
    // Should complete despite some errors
    assert!(report.validation_results.warnings > 0);
    assert!(report.compression_ratio > 1.0); // Still achieved some compression
}

#[test]
fn test_iterative_convergence() {
    let mut hierarchy = create_deep_hierarchy(1000);
    let config = CompressionConfig {
        max_iterations: 5,
        target_compression_ratio: 15.0,
        ..CompressionConfig::default()
    };
    
    let orchestrator = CompressionOrchestrator::new(config);
    let report = orchestrator.compress_hierarchy(&mut hierarchy).unwrap();
    
    // Should stop when convergence reached or max iterations hit
    assert!(report.iterations_performed <= 5);
    assert!(report.compression_ratio > 5.0);
    
    // Verify convergence detection
    if report.iterations_performed < 5 {
        // Converged early - should have high compression ratio
        assert!(report.compression_ratio >= 10.0);
    }
}

#[test]
fn test_configuration_validation() {
    let hierarchy = create_test_hierarchy();
    let invalid_config = CompressionConfig {
        min_frequency_threshold: 1.5, // Invalid: > 1.0
        min_nodes_threshold: 0, // Invalid: should be > 0
        max_iterations: 0, // Invalid: should be > 0
        target_compression_ratio: -1.0, // Invalid: should be > 1.0
        ..CompressionConfig::default()
    };
    
    let orchestrator = CompressionOrchestrator::new(invalid_config);
    let validation = orchestrator.validate_configuration(&hierarchy);
    
    assert!(!validation.is_valid);
    assert!(validation.errors.len() >= 4); // Should catch all invalid settings
}

#[test]
fn test_compression_plan_creation() {
    let hierarchy = create_realistic_hierarchy(2000);
    let config = CompressionConfig::default();
    let orchestrator = CompressionOrchestrator::new(config);
    
    let plan = orchestrator.create_compression_plan(&hierarchy);
    
    assert!(plan.estimated_iterations > 0);
    assert!(plan.estimated_total_time > Duration::ZERO);
    assert!(plan.estimated_compression_ratio > 1.0);
    assert!(plan.potential_candidates > 0);
    assert!(!plan.risk_assessment.high_risk_operations.is_empty() || plan.risk_assessment.overall_risk == RiskLevel::Low);
}
```

## File Location
`src/compression/orchestrator.rs`

## Next Micro Phase
After completion, proceed to Micro 3.4: Iterative Compression Algorithm