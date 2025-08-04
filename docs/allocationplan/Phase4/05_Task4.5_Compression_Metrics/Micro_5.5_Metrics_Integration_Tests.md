# Micro Phase 5.5: Metrics Integration Tests

**Estimated Time**: 20 minutes
**Dependencies**: Micro 5.4 Complete (Report Generator)
**Objective**: Implement comprehensive integration tests for the complete metrics and analysis pipeline

## Task Description

Create thorough integration tests that validate the end-to-end functionality of the compression metrics system, ensuring all components work together seamlessly and deliver accurate, actionable insights.

## Deliverables

Create `tests/integration/task_4_5_metrics.rs` with:

1. **Full pipeline integration tests**: End-to-end system validation
2. **Cross-component interaction tests**: Verify proper data flow between components
3. **Performance integration tests**: Ensure system meets performance requirements
4. **Error handling integration tests**: Validate graceful error handling across components
5. **Real-world scenario tests**: Test with realistic data and usage patterns

## Success Criteria

- [ ] Tests cover 100% of public API surface for all metrics components
- [ ] Integration tests pass for hierarchies up to 50,000 nodes
- [ ] Performance tests validate <200ms end-to-end processing for 10,000 nodes
- [ ] Error handling tests cover all failure scenarios and recovery paths
- [ ] Tests validate accuracy requirements (2% compression ratio accuracy)
- [ ] Cross-platform compatibility tests pass on all target platforms

## Implementation Requirements

```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;
use crate::compression::{
    metrics::{CompressionMetricsCalculator, CompressionMetrics},
    storage_analyzer::{StorageAnalyzer, StorageAnalysis},
    verifier::{CompressionVerifier, VerificationResult, VerificationLevel},
    reporter::{ReportGenerator, ComprehensiveReport, OutputFormat},
};
use crate::core::inheritance_hierarchy::InheritanceHierarchy;
use crate::test_utils::{
    create_test_hierarchy, create_large_hierarchy, create_complex_hierarchy,
    inject_test_data, create_hierarchies_with_known_properties,
};

#[cfg(test)]
mod metrics_integration_tests {
    use super::*;

    #[test]
    fn test_complete_metrics_pipeline() {
        // Create comprehensive test hierarchy
        let hierarchy = create_comprehensive_test_hierarchy(1000);
        
        // Initialize all components
        let metrics_calculator = CompressionMetricsCalculator::new();
        let storage_analyzer = StorageAnalyzer::new();
        let verifier = CompressionVerifier::new();
        let report_generator = ReportGenerator::new();
        
        // Execute complete pipeline
        let start = Instant::now();
        
        // Step 1: Calculate metrics
        let metrics = metrics_calculator.calculate_comprehensive_metrics(&hierarchy);
        assert!(metrics.compression_ratio > 1.0);
        assert!(metrics.total_nodes == 1000);
        
        // Step 2: Analyze storage
        let storage_analysis = storage_analyzer.analyze_storage(&hierarchy, &metrics);
        assert!(storage_analysis.allocation_efficiency > 0.0);
        assert!(!storage_analysis.optimization_recommendations.is_empty());
        
        // Step 3: Verify compression
        let verification_result = verifier.verify_compression(&hierarchy, None);
        assert!(matches!(verification_result.verification_status, 
                        crate::compression::verifier::VerificationStatus::Passed));
        
        // Step 4: Generate report
        let comprehensive_report = report_generator.generate_comprehensive_report(
            &metrics, &storage_analysis, &verification_result
        );
        assert!(!comprehensive_report.executive_summary.compression_effectiveness.is_empty());
        
        let total_elapsed = start.elapsed();
        
        // Performance requirement: Complete pipeline in < 200ms for 1k nodes
        assert!(total_elapsed < Duration::from_millis(200));
        
        // Validate cross-component data consistency
        validate_cross_component_consistency(&metrics, &storage_analysis, &verification_result, &comprehensive_report);
    }

    #[test]
    fn test_large_scale_integration() {
        // Test with large hierarchy (10k nodes)
        let large_hierarchy = create_large_hierarchy(10000);
        
        let metrics_calculator = CompressionMetricsCalculator::new();
        let storage_analyzer = StorageAnalyzer::new();
        let verifier = CompressionVerifier::with_level(VerificationLevel::Standard);
        let report_generator = ReportGenerator::new();
        
        let start = Instant::now();
        
        // Execute pipeline with performance monitoring
        let metrics = metrics_calculator.calculate_comprehensive_metrics(&large_hierarchy);
        let metrics_time = start.elapsed();
        
        let storage_analysis = storage_analyzer.analyze_storage(&large_hierarchy, &metrics);
        let storage_time = start.elapsed() - metrics_time;
        
        let verification_result = verifier.verify_compression(&large_hierarchy, None);
        let verification_time = start.elapsed() - metrics_time - storage_time;
        
        let report = report_generator.generate_comprehensive_report(
            &metrics, &storage_analysis, &verification_result
        );
        let total_time = start.elapsed();
        
        // Performance requirements for 10k nodes
        assert!(metrics_time < Duration::from_millis(50), "Metrics calculation too slow: {:?}", metrics_time);
        assert!(storage_time < Duration::from_millis(25), "Storage analysis too slow: {:?}", storage_time);
        assert!(verification_time < Duration::from_millis(100), "Verification too slow: {:?}", verification_time);
        assert!(total_time < Duration::from_millis(200), "Total pipeline too slow: {:?}", total_time);
        
        // Validate scalability characteristics
        assert_eq!(metrics.total_nodes, 10000);
        assert!(verification_result.nodes_verified >= 9500); // At least 95% coverage
        assert!(storage_analysis.component_breakdown.node_headers > 0);
        assert!(!report.optimization_recommendations.is_empty());
    }

    #[test]
    fn test_accuracy_validation_integration() {
        // Create hierarchy with known compression characteristics
        let test_scenarios = create_hierarchies_with_known_properties();
        
        for (scenario_name, hierarchy, expected_compression, expected_efficiency) in test_scenarios {
            let metrics_calculator = CompressionMetricsCalculator::new();
            let storage_analyzer = StorageAnalyzer::new();
            
            let metrics = metrics_calculator.calculate_comprehensive_metrics(&hierarchy);
            let storage_analysis = storage_analyzer.analyze_storage(&hierarchy, &metrics);
            
            // Validate accuracy requirements (within 2%)
            let compression_accuracy = (metrics.compression_ratio - expected_compression).abs() / expected_compression;
            assert!(compression_accuracy < 0.02, 
                   "Compression ratio accuracy failed for scenario '{}': expected {}, got {}, accuracy: {:.3}%",
                   scenario_name, expected_compression, metrics.compression_ratio, compression_accuracy * 100.0);
            
            let efficiency_accuracy = (storage_analysis.allocation_efficiency - expected_efficiency).abs() / expected_efficiency;
            assert!(efficiency_accuracy < 0.02,
                   "Storage efficiency accuracy failed for scenario '{}': expected {}, got {}, accuracy: {:.3}%",
                   scenario_name, expected_efficiency, storage_analysis.allocation_efficiency, efficiency_accuracy * 100.0);
            
            println!("✓ Scenario '{}' passed accuracy validation", scenario_name);
        }
    }

    #[test]
    fn test_error_handling_integration() {
        // Test various error scenarios across the pipeline
        let error_scenarios = create_error_test_scenarios();
        
        for (scenario_name, hierarchy, expected_error_type) in error_scenarios {
            let metrics_calculator = CompressionMetricsCalculator::new();
            let storage_analyzer = StorageAnalyzer::new();
            let verifier = CompressionVerifier::with_level(VerificationLevel::Exhaustive);
            let report_generator = ReportGenerator::new();
            
            // Pipeline should handle errors gracefully
            let metrics_result = std::panic::catch_unwind(|| {
                metrics_calculator.calculate_comprehensive_metrics(&hierarchy)
            });
            
            match expected_error_type {
                ErrorType::MetricsCalculation => {
                    assert!(metrics_result.is_err(), "Expected metrics calculation error for scenario '{}'", scenario_name);
                    continue; // Can't proceed with invalid metrics
                }
                _ => {
                    assert!(metrics_result.is_ok(), "Unexpected metrics calculation error for scenario '{}'", scenario_name);
                }
            }
            
            let metrics = metrics_result.unwrap();
            
            // Test storage analysis error handling
            let storage_result = std::panic::catch_unwind(|| {
                storage_analyzer.analyze_storage(&hierarchy, &metrics)
            });
            
            match expected_error_type {
                ErrorType::StorageAnalysis => {
                    assert!(storage_result.is_err(), "Expected storage analysis error for scenario '{}'", scenario_name);
                    continue;
                }
                _ => {
                    assert!(storage_result.is_ok(), "Unexpected storage analysis error for scenario '{}'", scenario_name);
                }
            }
            
            let storage_analysis = storage_result.unwrap();
            
            // Test verification error handling
            let verification_result = verifier.verify_compression(&hierarchy, None);
            
            match expected_error_type {
                ErrorType::Verification => {
                    assert!(matches!(verification_result.verification_status,
                                   crate::compression::verifier::VerificationStatus::Failed { .. }),
                           "Expected verification failure for scenario '{}'", scenario_name);
                }
                _ => {
                    // Verification might pass or have warnings, but shouldn't crash
                    assert!(!matches!(verification_result.verification_status,
                                    crate::compression::verifier::VerificationStatus::Incomplete { .. }),
                           "Verification incomplete for scenario '{}'", scenario_name);
                }
            }
            
            // Report generation should always succeed with valid inputs
            let report_result = std::panic::catch_unwind(|| {
                report_generator.generate_comprehensive_report(&metrics, &storage_analysis, &verification_result)
            });
            
            assert!(report_result.is_ok(), "Report generation failed for scenario '{}'", scenario_name);
            
            println!("✓ Error handling validated for scenario '{}'", scenario_name);
        }
    }

    #[test]
    fn test_multi_format_report_integration() {
        let hierarchy = create_test_hierarchy(500);
        
        // Generate complete analysis
        let metrics_calculator = CompressionMetricsCalculator::new();
        let storage_analyzer = StorageAnalyzer::new();
        let verifier = CompressionVerifier::new();
        let report_generator = ReportGenerator::new();
        
        let metrics = metrics_calculator.calculate_comprehensive_metrics(&hierarchy);
        let storage_analysis = storage_analyzer.analyze_storage(&hierarchy, &metrics);
        let verification_result = verifier.verify_compression(&hierarchy, None);
        let comprehensive_report = report_generator.generate_comprehensive_report(
            &metrics, &storage_analysis, &verification_result
        );
        
        // Test all output formats
        let formats = vec![
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::Html,
            OutputFormat::Csv,
            OutputFormat::Markdown,
        ];
        
        for format in formats {
            let formatted_output = report_generator.format_report(&comprehensive_report, format.clone());
            
            assert!(!formatted_output.is_empty(), "Empty output for format {:?}", format);
            
            // Format-specific validations
            match format {
                OutputFormat::Json => {
                    // Should be valid JSON
                    assert!(formatted_output.trim().starts_with('{'));
                    assert!(formatted_output.trim().ends_with('}'));
                }
                OutputFormat::Html => {
                    // Should have HTML structure
                    assert!(formatted_output.contains("<html"));
                    assert!(formatted_output.contains("</html>"));
                }
                OutputFormat::Csv => {
                    // Should have CSV structure
                    assert!(formatted_output.contains(','));
                    let lines: Vec<&str> = formatted_output.lines().collect();
                    assert!(lines.len() > 1, "CSV should have header and data rows");
                }
                _ => {
                    // Text and Markdown should have meaningful content
                    assert!(formatted_output.len() > 100, "Output too short for format {:?}", format);
                }
            }
            
            println!("✓ Format {:?} validation passed", format);
        }
    }

    #[test]
    fn test_concurrent_processing_integration() {
        use std::sync::Arc;
        use std::thread;
        
        // Test concurrent processing of multiple hierarchies
        let hierarchies: Vec<InheritanceHierarchy> = (0..5)
            .map(|i| create_test_hierarchy(1000 + i * 100))
            .collect();
        
        let metrics_calculator = Arc::new(CompressionMetricsCalculator::new());
        let storage_analyzer = Arc::new(StorageAnalyzer::new());
        let verifier = Arc::new(CompressionVerifier::new());
        let report_generator = Arc::new(ReportGenerator::new());
        
        let mut handles = vec![];
        
        for (idx, hierarchy) in hierarchies.into_iter().enumerate() {
            let calc = metrics_calculator.clone();
            let analyzer = storage_analyzer.clone();
            let ver = verifier.clone();
            let gen = report_generator.clone();
            
            let handle = thread::spawn(move || {
                let start = Instant::now();
                
                let metrics = calc.calculate_comprehensive_metrics(&hierarchy);
                let storage_analysis = analyzer.analyze_storage(&hierarchy, &metrics);
                let verification_result = ver.verify_compression(&hierarchy, None);
                let report = gen.generate_comprehensive_report(&metrics, &storage_analysis, &verification_result);
                
                let elapsed = start.elapsed();
                (idx, elapsed, metrics.compression_ratio, report.executive_summary.overall_health_score)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut results = vec![];
        for handle in handles {
            let result = handle.join().expect("Thread should complete successfully");
            results.push(result);
        }
        
        // Validate concurrent processing
        assert_eq!(results.len(), 5);
        
        for (idx, elapsed, compression_ratio, health_score) in results {
            assert!(elapsed < Duration::from_millis(300), "Thread {} took too long: {:?}", idx, elapsed);
            assert!(compression_ratio > 1.0, "Thread {} has invalid compression ratio: {}", idx, compression_ratio);
            assert!(health_score >= 0.0 && health_score <= 1.0, "Thread {} has invalid health score: {}", idx, health_score);
            
            println!("✓ Thread {} completed successfully in {:?}", idx, elapsed);
        }
    }

    #[test]
    fn test_memory_usage_integration() {
        // Test memory usage patterns during processing
        let large_hierarchy = create_large_hierarchy(5000);
        
        let initial_memory = get_memory_usage();
        
        {
            let metrics_calculator = CompressionMetricsCalculator::new();
            let storage_analyzer = StorageAnalyzer::new();
            let verifier = CompressionVerifier::new();
            let report_generator = ReportGenerator::new();
            
            let metrics = metrics_calculator.calculate_comprehensive_metrics(&large_hierarchy);
            let peak_after_metrics = get_memory_usage();
            
            let storage_analysis = storage_analyzer.analyze_storage(&large_hierarchy, &metrics);
            let peak_after_storage = get_memory_usage();
            
            let verification_result = verifier.verify_compression(&large_hierarchy, None);
            let peak_after_verification = get_memory_usage();
            
            let report = report_generator.generate_comprehensive_report(&metrics, &storage_analysis, &verification_result);
            let peak_after_report = get_memory_usage();
            
            // Memory usage should be reasonable
            let metrics_memory_delta = peak_after_metrics - initial_memory;
            let storage_memory_delta = peak_after_storage - peak_after_metrics;
            let verification_memory_delta = peak_after_verification - peak_after_storage;
            let report_memory_delta = peak_after_report - peak_after_verification;
            
            println!("Memory usage deltas - Metrics: {}KB, Storage: {}KB, Verification: {}KB, Report: {}KB",
                    metrics_memory_delta / 1024, storage_memory_delta / 1024, 
                    verification_memory_delta / 1024, report_memory_delta / 1024);
            
            // No component should use more than 50MB for 5k nodes
            assert!(metrics_memory_delta < 50 * 1024 * 1024, "Metrics calculation uses too much memory");
            assert!(storage_memory_delta < 50 * 1024 * 1024, "Storage analysis uses too much memory");
            assert!(verification_memory_delta < 50 * 1024 * 1024, "Verification uses too much memory");
            assert!(report_memory_delta < 50 * 1024 * 1024, "Report generation uses too much memory");
            
            // Keep references alive until here to prevent early cleanup
            let _ = (metrics, storage_analysis, verification_result, report);
        }
        
        // Force garbage collection and check for memory leaks
        std::thread::sleep(Duration::from_millis(100));
        let final_memory = get_memory_usage();
        let memory_leak = final_memory.saturating_sub(initial_memory);
        
        // Allow some reasonable overhead but detect significant leaks
        assert!(memory_leak < 10 * 1024 * 1024, "Potential memory leak detected: {}KB", memory_leak / 1024);
        
        println!("✓ Memory usage validation passed, final overhead: {}KB", memory_leak / 1024);
    }

    // Helper functions
    
    fn validate_cross_component_consistency(
        metrics: &CompressionMetrics,
        storage_analysis: &StorageAnalysis,
        verification_result: &VerificationResult,
        report: &ComprehensiveReport
    ) {
        // Node counts should be consistent
        assert_eq!(metrics.total_nodes, verification_result.nodes_verified);
        
        // Storage analysis should reference metrics data
        let calculated_storage = storage_analysis.component_breakdown.node_headers +
                                storage_analysis.component_breakdown.property_names +
                                storage_analysis.component_breakdown.property_values;
        assert!(calculated_storage > 0);
        
        // Report should reflect metrics values
        assert!((report.metrics_summary.compression_ratio - metrics.compression_ratio).abs() < 0.001);
        
        // Verification confidence should influence report health score
        if verification_result.overall_confidence > 0.95 {
            assert!(report.executive_summary.overall_health_score > 0.7);
        }
        
        println!("✓ Cross-component consistency validated");
    }
    
    fn create_comprehensive_test_hierarchy(node_count: usize) -> InheritanceHierarchy {
        let mut hierarchy = create_test_hierarchy(node_count);
        inject_test_data(&mut hierarchy, node_count / 10); // 10% complex properties
        hierarchy
    }
    
    fn create_hierarchies_with_known_properties() -> Vec<(String, InheritanceHierarchy, f64, f64)> {
        vec![
            ("High Compression Scenario".to_string(), create_high_compression_hierarchy(), 15.0, 0.95),
            ("Medium Compression Scenario".to_string(), create_medium_compression_hierarchy(), 8.0, 0.85),
            ("Low Compression Scenario".to_string(), create_low_compression_hierarchy(), 3.0, 0.70),
            ("Complex Hierarchy Scenario".to_string(), create_complex_hierarchy(500), 10.0, 0.90),
        ]
    }
    
    fn create_error_test_scenarios() -> Vec<(String, InheritanceHierarchy, ErrorType)> {
        vec![
            ("Corrupted Data".to_string(), create_corrupted_hierarchy(), ErrorType::Verification),
            ("Circular References".to_string(), create_circular_hierarchy(), ErrorType::Verification),
            ("Empty Hierarchy".to_string(), create_empty_hierarchy(), ErrorType::MetricsCalculation),
            ("Malformed Properties".to_string(), create_malformed_hierarchy(), ErrorType::StorageAnalysis),
        ]
    }
    
    #[derive(Debug, Clone)]
    enum ErrorType {
        MetricsCalculation,
        StorageAnalysis,
        Verification,
        ReportGeneration,
    }
    
    fn get_memory_usage() -> usize {
        // Platform-specific memory usage implementation
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback for unsupported platforms
        0
    }
    
    // Test hierarchy creation helpers
    fn create_high_compression_hierarchy() -> InheritanceHierarchy {
        create_test_hierarchy_with_inheritance_rate(1000, 0.9)
    }
    
    fn create_medium_compression_hierarchy() -> InheritanceHierarchy {
        create_test_hierarchy_with_inheritance_rate(1000, 0.7)
    }
    
    fn create_low_compression_hierarchy() -> InheritanceHierarchy {
        create_test_hierarchy_with_inheritance_rate(1000, 0.4)
    }
    
    fn create_corrupted_hierarchy() -> InheritanceHierarchy {
        let mut hierarchy = create_test_hierarchy(100);
        // Inject corruption for testing
        corrupt_random_data(&mut hierarchy);
        hierarchy
    }
    
    fn create_circular_hierarchy() -> InheritanceHierarchy {
        let mut hierarchy = create_test_hierarchy(50);
        // Create circular references for testing
        introduce_circular_references(&mut hierarchy);
        hierarchy
    }
    
    fn create_empty_hierarchy() -> InheritanceHierarchy {
        InheritanceHierarchy::new()
    }
    
    fn create_malformed_hierarchy() -> InheritanceHierarchy {
        let mut hierarchy = create_test_hierarchy(100);
        // Introduce malformed properties for testing
        introduce_malformed_properties(&mut hierarchy);
        hierarchy
    }
    
    // Additional test utility functions would be implemented here
    // These are placeholders for the actual implementation
    fn create_test_hierarchy_with_inheritance_rate(nodes: usize, rate: f64) -> InheritanceHierarchy {
        // Implementation would create hierarchy with specific inheritance characteristics
        create_test_hierarchy(nodes)
    }
    
    fn corrupt_random_data(hierarchy: &mut InheritanceHierarchy) {
        // Implementation would introduce controlled corruption for testing
    }
    
    fn introduce_circular_references(hierarchy: &mut InheritanceHierarchy) {
        // Implementation would create circular reference patterns
    }
    
    fn introduce_malformed_properties(hierarchy: &mut InheritanceHierarchy) {
        // Implementation would create properties with invalid data
    }
}
```

## Test Requirements

Integration tests must validate:
```rust
// Key integration test requirements that must be met:

#[test]
fn test_end_to_end_accuracy_requirement() {
    // Must validate 2% accuracy requirement for compression ratios
    // across all components working together
}

#[test] 
fn test_performance_integration_requirements() {
    // Must validate complete pipeline performance:
    // - Metrics calculation: < 50ms for 10k nodes
    // - Storage analysis: < 25ms for 10k nodes  
    // - Verification: < 100ms for 10k nodes
    // - Report generation: < 15ms for 10k nodes
    // - Total pipeline: < 200ms for 10k nodes
}

#[test]
fn test_scalability_integration() {
    // Must test hierarchies up to 50,000 nodes
    // and validate linear scaling characteristics
}

#[test]
fn test_cross_platform_compatibility() {
    // Must pass on Windows, Linux, and macOS
    // with consistent results across platforms
}

#[test]
fn test_memory_efficiency_integration() {
    // Must validate memory usage stays within bounds
    // and no memory leaks occur during processing
}
```

## File Location
`tests/integration/task_4_5_metrics.rs`

## Completion Criteria

This micro phase completes Task 4.5 when:
- [ ] All integration tests pass consistently
- [ ] Performance requirements are met for all test scenarios
- [ ] Cross-component data flow is validated
- [ ] Error handling works correctly across the pipeline
- [ ] Memory usage is within acceptable bounds
- [ ] Real-world scenario testing demonstrates system robustness

## Task 4.5 Summary

Upon completion of this micro phase, Task 4.5 (Compression Metrics and Analysis) will be complete with:

1. ✅ **Micro 5.1**: Compression Metrics Calculator (35min)
2. ✅ **Micro 5.2**: Storage Analyzer (30min) 
3. ✅ **Micro 5.3**: Compression Verifier (40min)
4. ✅ **Micro 5.4**: Report Generator (25min)
5. ✅ **Micro 5.5**: Metrics Integration Tests (20min)

**Total Time**: 150 minutes
**Total Estimated vs Actual**: 115 minutes (revised from original estimate)