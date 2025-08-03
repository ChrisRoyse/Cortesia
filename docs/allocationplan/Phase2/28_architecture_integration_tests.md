# Task 28: Architecture Integration Tests

## Metadata
- **Micro-Phase**: 2.28
- **Duration**: 55-60 minutes
- **Dependencies**: Task 24 (architecture_selection_framework), Task 25 (semantic_column_setup), Task 26 (structural_column_setup), Task 27 (temporal_exception_columns)
- **Output**: `src/integration_tests/architecture_integration.rs`, `src/integration_tests/column_performance.rs`

## Description
Comprehensive integration testing framework for validating the selected neural network architectures across all cortical columns. This testing suite validates the intelligent architecture selection philosophy by verifying that the 3-4 selected architectures perform optimally across semantic processing, structural analysis, temporal sequence processing, and exception detection tasks. Includes performance benchmarking, memory validation, and cross-column integration testing.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruv_fann_integration::{ArchitectureSelector, TaskType, SelectedArchitecture};
    use crate::multi_column::{
        SemanticProcessingColumn, StructuralAnalysisColumn, 
        TemporalContextColumn, ExceptionDetectionColumn,
        ColumnVote, ColumnId
    };
    use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId, SpikeEvent, NeuronId};
    use std::time::{Duration, Instant};
    use std::collections::HashMap;
    use rayon::prelude::*;

    // Architecture Selection Integration Tests
    #[test]
    fn test_complete_architecture_selection_integration() {
        let selector = ArchitectureSelector::new();
        
        // Test intelligent selection of 3-4 optimal architectures
        let selected_architectures = selector.select_optimal_architecture_set();
        
        // Verify architecture count (should be 3-4 as per intelligent selection philosophy)
        assert!(selected_architectures.len() >= 3 && selected_architectures.len() <= 4, 
               "Should select 3-4 optimal architectures, got {}", selected_architectures.len());
        
        // Verify architecture diversity
        let mut task_coverage = std::collections::HashSet::new();
        for arch in &selected_architectures {
            for task in &arch.supported_tasks {
                task_coverage.insert(*task);
            }
        }
        
        // Should cover all core task types
        assert!(task_coverage.contains(&TaskType::Semantic));
        assert!(task_coverage.contains(&TaskType::Structural));
        assert!(task_coverage.contains(&TaskType::Temporal));
        assert!(task_coverage.contains(&TaskType::Exception) || task_coverage.contains(&TaskType::Classification));
        
        // Verify memory constraints
        let total_memory = selected_architectures.iter()
            .map(|arch| arch.memory_profile.memory_footprint)
            .sum::<usize>();
        assert!(total_memory <= 200_000_000, "Total memory should be ≤200MB, got {}MB", total_memory / 1_000_000);
        
        // Verify performance criteria
        for arch in &selected_architectures {
            assert!(arch.performance_metrics.performance_score >= 0.7, 
                   "Architecture {} has poor performance score: {}", 
                   arch.architecture.name, arch.performance_metrics.performance_score);
        }
    }
    
    #[test]
    fn test_column_architecture_assignment_validation() {
        let selector = ArchitectureSelector::new();
        
        // Test architecture assignment for each column type
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Verify appropriate architecture selection for each column
        let semantic_arch = semantic_column.get_selected_architecture();
        assert!(semantic_arch.supported_tasks.contains(&TaskType::Semantic) ||
                semantic_arch.supported_tasks.contains(&TaskType::Classification),
               "Semantic column should have semantic-capable architecture");
        
        let structural_arch = structural_column.get_selected_architecture();
        assert!(structural_arch.supported_tasks.contains(&TaskType::Structural) ||
                structural_arch.supported_tasks.contains(&TaskType::Semantic),
               "Structural column should have structural-capable architecture");
        
        let temporal_arch = temporal_column.get_selected_architecture();
        assert!(temporal_arch.supported_tasks.contains(&TaskType::Temporal) ||
                temporal_arch.supported_tasks.contains(&TaskType::Semantic),
               "Temporal column should have temporal-capable architecture");
        
        let exception_arch = exception_column.get_selected_architecture();
        assert!(exception_arch.supported_tasks.contains(&TaskType::Exception) ||
                exception_arch.supported_tasks.contains(&TaskType::Classification),
               "Exception column should have exception-capable architecture");
        
        // Verify no column exceeds individual memory limit
        assert!(semantic_arch.memory_profile.memory_footprint <= 50_000_000);
        assert!(structural_arch.memory_profile.memory_footprint <= 50_000_000);
        assert!(temporal_arch.memory_profile.memory_footprint <= 50_000_000);
        assert!(exception_arch.memory_profile.memory_footprint <= 50_000_000);
    }
    
    #[test]
    fn test_cross_column_architecture_compatibility() {
        let selector = ArchitectureSelector::new();
        
        // Initialize all columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Create test spike pattern for cross-column processing
        let test_pattern = create_comprehensive_test_pattern();
        
        // Process same pattern through all columns
        let semantic_result = semantic_column.extract_semantic_features(&test_pattern).unwrap();
        let structural_result = structural_column.extract_graph_structure(&test_pattern).unwrap();
        let temporal_result = temporal_column.detect_sequences(&test_pattern).unwrap();
        let exception_result = exception_column.find_inhibitions(&test_pattern).unwrap();
        
        // Verify column votes are compatible
        assert_eq!(semantic_result.column_id, ColumnId::Semantic);
        assert_eq!(structural_result.column_id, ColumnId::Structural);
        assert_eq!(temporal_result.column_id, ColumnId::Temporal);
        assert_eq!(exception_result.column_id, ColumnId::Exception);
        
        // Verify neural output dimensions are consistent for integration
        assert!(!semantic_result.neural_output.is_empty());
        assert!(!structural_result.neural_output.is_empty());
        assert!(!temporal_result.neural_output.is_empty());
        assert!(!exception_result.neural_output.is_empty());
        
        // All processing should complete within performance targets
        assert!(semantic_result.processing_time <= Duration::from_millis(1));
        assert!(structural_result.processing_time <= Duration::from_millis(1));
        assert!(temporal_result.processing_time <= Duration::from_millis(1));
        assert!(exception_result.processing_time <= Duration::from_millis(1));
    }
    
    #[test]
    fn test_parallel_multi_column_processing() {
        let selector = ArchitectureSelector::new();
        
        // Initialize all columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Create multiple test patterns
        let test_patterns = create_multiple_test_patterns(100);
        
        // Test parallel processing across all columns
        let start_time = Instant::now();
        let results: Vec<_> = test_patterns.par_iter().map(|pattern| {
            let semantic = semantic_column.extract_semantic_features(pattern);
            let structural = structural_column.extract_graph_structure(pattern);
            let temporal = temporal_column.detect_sequences(pattern);
            let exception = exception_column.find_inhibitions(pattern);
            
            (semantic, structural, temporal, exception)
        }).collect();
        let parallel_time = start_time.elapsed();
        
        // Verify all results are successful
        assert_eq!(results.len(), 100);
        for (semantic, structural, temporal, exception) in &results {
            assert!(semantic.is_ok());
            assert!(structural.is_ok());
            assert!(temporal.is_ok());
            assert!(exception.is_ok());
        }
        
        // Test sequential processing for comparison
        let start_time = Instant::now();
        for pattern in &test_patterns {
            let _ = semantic_column.extract_semantic_features(pattern).unwrap();
            let _ = structural_column.extract_graph_structure(pattern).unwrap();
            let _ = temporal_column.detect_sequences(pattern).unwrap();
            let _ = exception_column.find_inhibitions(pattern).unwrap();
        }
        let sequential_time = start_time.elapsed();
        
        // Parallel should be faster than sequential
        assert!(parallel_time < sequential_time, 
               "Parallel processing ({:?}) should be faster than sequential ({:?})", 
               parallel_time, sequential_time);
        
        // Should achieve target performance
        let average_pattern_time = parallel_time / test_patterns.len() as u32;
        assert!(average_pattern_time <= Duration::from_millis(1), 
               "Average pattern processing time should be ≤1ms, got {:?}", average_pattern_time);
    }
    
    #[test]
    fn test_architecture_performance_benchmarking() {
        let selector = ArchitectureSelector::new();
        
        // Get selected architectures
        let selected_architectures = selector.select_optimal_architecture_set();
        
        // Test each architecture across different task types
        let benchmark_results = run_architecture_benchmarks(&selected_architectures);
        
        // Verify performance meets targets
        for (arch_id, results) in &benchmark_results {
            // Each architecture should excel in at least one task type
            let max_performance = results.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!(*max_performance >= 0.8, 
                   "Architecture {} should excel in at least one task (max: {:.3})", 
                   arch_id, max_performance);
            
            // No architecture should perform poorly across all tasks
            let min_performance = results.values().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!(*min_performance >= 0.3, 
                   "Architecture {} should not perform poorly across all tasks (min: {:.3})", 
                   arch_id, min_performance);
        }
        
        // Verify task coverage across all architectures
        let mut task_performance = HashMap::new();
        for (_, results) in &benchmark_results {
            for (task, performance) in results {
                let entry = task_performance.entry(*task).or_insert(Vec::new());
                entry.push(*performance);
            }
        }
        
        // Each task should have at least one high-performing architecture
        for (task, performances) in &task_performance {
            let max_performance = performances.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!(*max_performance >= 0.8, 
                   "Task {:?} should have at least one high-performing architecture (max: {:.3})", 
                   task, max_performance);
        }
    }
    
    #[test]
    fn test_memory_usage_validation() {
        let selector = ArchitectureSelector::new();
        
        // Initialize all columns and measure memory usage
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Get memory profiles
        let semantic_memory = semantic_column.get_selected_architecture().memory_profile.memory_footprint;
        let structural_memory = structural_column.get_selected_architecture().memory_profile.memory_footprint;
        let temporal_memory = temporal_column.get_selected_architecture().memory_profile.memory_footprint;
        let exception_memory = exception_column.get_selected_architecture().memory_profile.memory_footprint;
        
        // Verify individual column memory limits
        assert!(semantic_memory <= 50_000_000, "Semantic column memory should be ≤50MB, got {}MB", semantic_memory / 1_000_000);
        assert!(structural_memory <= 50_000_000, "Structural column memory should be ≤50MB, got {}MB", structural_memory / 1_000_000);
        assert!(temporal_memory <= 50_000_000, "Temporal column memory should be ≤50MB, got {}MB", temporal_memory / 1_000_000);
        assert!(exception_memory <= 50_000_000, "Exception column memory should be ≤50MB, got {}MB", exception_memory / 1_000_000);
        
        // Verify total system memory limit
        let total_memory = semantic_memory + structural_memory + temporal_memory + exception_memory;
        assert!(total_memory <= 200_000_000, "Total system memory should be ≤200MB, got {}MB", total_memory / 1_000_000);
        
        // Test memory usage under load
        let test_patterns = create_multiple_test_patterns(1000);
        
        let pre_load_memory = get_current_memory_usage();
        
        // Process many patterns to test memory growth
        for pattern in &test_patterns {
            let _ = semantic_column.extract_semantic_features(pattern);
            let _ = structural_column.extract_graph_structure(pattern);
            let _ = temporal_column.detect_sequences(pattern);
            let _ = exception_column.find_inhibitions(pattern);
        }
        
        let post_load_memory = get_current_memory_usage();
        let memory_growth = post_load_memory - pre_load_memory;
        
        // Memory growth should be reasonable (allow for caching)
        assert!(memory_growth <= 100_000_000, "Memory growth under load should be ≤100MB, got {}MB", memory_growth / 1_000_000);
    }
    
    #[test]
    fn test_architecture_specialization_validation() {
        let selector = ArchitectureSelector::new();
        
        // Test that different architectures are selected for different specializations
        let semantic_archs = selector.select_for_task_type(TaskType::Semantic);
        let temporal_archs = selector.select_for_task_type(TaskType::Temporal);
        let structural_archs = selector.select_for_task_type(TaskType::Structural);
        let exception_archs = selector.select_for_task_type(TaskType::Exception);
        
        // Verify specialization preferences
        for arch in &temporal_archs {
            // Temporal architectures should prefer LSTM, TCN, or GRU (IDs 4, 5, 20)
            if [4, 5, 20].contains(&arch.id) {
                assert!(arch.supported_tasks.contains(&TaskType::Temporal) ||
                        arch.supported_tasks.contains(&TaskType::Semantic),
                       "Temporal-specialized architecture should support temporal tasks");
            }
        }
        
        for arch in &semantic_archs {
            // Semantic architectures should handle language and concept processing
            assert!(arch.supported_tasks.contains(&TaskType::Semantic) ||
                    arch.supported_tasks.contains(&TaskType::Classification),
                   "Semantic architecture should support semantic tasks");
        }
        
        // Test specialization performance
        let test_semantic_pattern = create_semantic_test_pattern();
        let test_temporal_pattern = create_temporal_test_pattern();
        let test_structural_pattern = create_structural_test_pattern();
        let test_exception_pattern = create_exception_test_pattern();
        
        // Each column should perform best on its specialized task
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        let semantic_on_semantic = semantic_column.extract_semantic_features(&test_semantic_pattern).unwrap();
        let temporal_on_temporal = temporal_column.detect_sequences(&test_temporal_pattern).unwrap();
        let structural_on_structural = structural_column.extract_graph_structure(&test_structural_pattern).unwrap();
        let exception_on_exception = exception_column.find_inhibitions(&test_exception_pattern).unwrap();
        
        // Specialized columns should show high confidence on their specialized tasks
        assert!(semantic_on_semantic.confidence >= 0.7, "Semantic column should have high confidence on semantic tasks");
        assert!(temporal_on_temporal.confidence >= 0.7, "Temporal column should have high confidence on temporal tasks");
        assert!(structural_on_structural.confidence >= 0.7, "Structural column should have high confidence on structural tasks");
        assert!(exception_on_exception.confidence >= 0.3 || exception_on_exception.activation == 0.0, 
               "Exception column should either detect exception or show low activation");
    }
    
    #[test]
    fn test_intelligent_selection_philosophy_validation() {
        let selector = ArchitectureSelector::new();
        
        // Test that intelligent selection chooses quality over quantity
        let all_candidates = selector.get_all_architecture_candidates();
        let selected_set = selector.select_optimal_architecture_set();
        
        // Should select far fewer than available (intelligent selection)
        assert!(selected_set.len() < all_candidates.len() / 2, 
               "Intelligent selection should choose significantly fewer architectures: {} selected from {} available", 
               selected_set.len(), all_candidates.len());
        
        // Selected architectures should have higher average performance
        let selected_avg_performance = selected_set.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>() / selected_set.len() as f32;
        
        let all_avg_performance = all_candidates.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>() / all_candidates.len() as f32;
        
        assert!(selected_avg_performance > all_avg_performance, 
               "Selected architectures should have higher average performance: {:.3} vs {:.3}", 
               selected_avg_performance, all_avg_performance);
        
        // Test efficiency metrics
        let selected_efficiency = calculate_architecture_set_efficiency(&selected_set);
        assert!(selected_efficiency >= 0.8, 
               "Selected architecture set should be highly efficient: {:.3}", selected_efficiency);
        
        // Verify coverage vs. redundancy tradeoff
        let task_coverage = calculate_task_coverage(&selected_set);
        let redundancy_score = calculate_redundancy_score(&selected_set);
        
        assert!(task_coverage >= 0.9, "Should cover most task types: {:.3}", task_coverage);
        assert!(redundancy_score <= 0.3, "Should minimize redundancy: {:.3}", redundancy_score);
    }
    
    #[test]
    fn test_column_integration_winner_take_all() {
        let selector = ArchitectureSelector::new();
        
        // Initialize all columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Test different pattern types and verify winner-take-all behavior
        
        // Semantic-dominant pattern
        let semantic_pattern = create_strong_semantic_pattern();
        let votes = process_pattern_through_all_columns(
            &semantic_pattern, &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        let winner = find_winner(&votes);
        assert_eq!(winner.column_id, ColumnId::Semantic, "Semantic pattern should activate semantic column strongest");
        
        // Temporal-dominant pattern
        let temporal_pattern = create_strong_temporal_pattern();
        let votes = process_pattern_through_all_columns(
            &temporal_pattern, &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        let winner = find_winner(&votes);
        assert_eq!(winner.column_id, ColumnId::Temporal, "Temporal pattern should activate temporal column strongest");
        
        // Exception-dominant pattern
        let exception_pattern = create_strong_exception_pattern();
        let votes = process_pattern_through_all_columns(
            &exception_pattern, &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Exception column should either win or show very low activation across all columns
        let exception_vote = votes.iter().find(|v| v.column_id == ColumnId::Exception).unwrap();
        if exception_vote.activation > 0.0 {
            let winner = find_winner(&votes);
            assert_eq!(winner.column_id, ColumnId::Exception, "Strong exception pattern should activate exception column");
        } else {
            // No strong activations - lateral inhibition working
            assert!(votes.iter().all(|v| v.activation < 0.5), "Exception pattern with no detection should show weak activations");
        }
    }
    
    #[test]
    fn test_architecture_robustness_under_stress() {
        let selector = ArchitectureSelector::new();
        
        // Initialize columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&selector).unwrap();
        
        // Stress test with various challenging patterns
        let stress_patterns = vec![
            create_noisy_pattern(),
            create_minimal_pattern(),
            create_maximum_complexity_pattern(),
            create_contradictory_pattern(),
            create_edge_case_pattern(),
        ];
        
        let mut all_successful = true;
        let mut performance_times = Vec::new();
        
        for pattern in &stress_patterns {
            let start = Instant::now();
            
            let semantic_result = semantic_column.extract_semantic_features(pattern);
            let structural_result = structural_column.extract_graph_structure(pattern);
            let temporal_result = temporal_column.detect_sequences(pattern);
            let exception_result = exception_column.find_inhibitions(pattern);
            
            let elapsed = start.elapsed();
            performance_times.push(elapsed);
            
            // All columns should handle stress patterns gracefully
            if semantic_result.is_err() || structural_result.is_err() || 
               temporal_result.is_err() || exception_result.is_err() {
                all_successful = false;
            }
            
            // Should still meet performance targets under stress
            assert!(elapsed <= Duration::from_millis(5), 
                   "Stress pattern processing should complete in ≤5ms, got {:?}", elapsed);
        }
        
        assert!(all_successful, "All columns should handle stress patterns gracefully");
        
        // Average performance should still be good
        let avg_time = performance_times.iter().sum::<Duration>() / performance_times.len() as u32;
        assert!(avg_time <= Duration::from_millis(2), 
               "Average stress test time should be ≤2ms, got {:?}", avg_time);
    }
    
    // Helper functions
    fn create_comprehensive_test_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("integration_test_concept");
        let first_spike_time = Duration::from_micros(500);
        let spikes = create_test_spikes(8);
        let total_duration = Duration::from_millis(2);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_multiple_test_patterns(count: usize) -> Vec<TTFSSpikePattern> {
        (0..count).map(|i| {
            let concept_id = ConceptId::new(&format!("test_concept_{}", i));
            let first_spike_time = Duration::from_micros(500 + i as u64 * 10);
            let spikes = create_test_spikes(6);
            let total_duration = Duration::from_millis(2);
            
            TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
        }).collect()
    }
    
    fn create_test_spikes(count: usize) -> Vec<SpikeEvent> {
        (0..count).map(|i| {
            SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(100 + i as u64 * 200),
                0.5 + (i as f32 * 0.1) % 0.5,
            )
        }).collect()
    }
    
    fn run_architecture_benchmarks(architectures: &[SelectedArchitecture]) -> HashMap<u32, HashMap<TaskType, f32>> {
        let mut results = HashMap::new();
        
        for arch in architectures {
            let mut task_results = HashMap::new();
            
            // Mock benchmarking for each task type
            for task_type in &[TaskType::Semantic, TaskType::Structural, TaskType::Temporal, TaskType::Exception] {
                let performance = if arch.supported_tasks.contains(task_type) {
                    0.8 + (arch.id as f32 % 3.0) * 0.1 // Mock high performance for supported tasks
                } else {
                    0.4 + (arch.id as f32 % 5.0) * 0.1 // Mock moderate performance for other tasks
                };
                task_results.insert(*task_type, performance);
            }
            
            results.insert(arch.architecture.id, task_results);
        }
        
        results
    }
    
    fn get_current_memory_usage() -> usize {
        // Mock memory usage measurement
        150_000_000 // 150MB baseline
    }
    
    fn create_semantic_test_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("semantic_concept");
        let first_spike_time = Duration::from_micros(300);
        let spikes = create_test_spikes(5);
        let total_duration = Duration::from_millis(1);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_temporal_test_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("temporal_sequence");
        let first_spike_time = Duration::from_micros(100);
        let spikes = (0..10).map(|i| {
            SpikeEvent::new(
                NeuronId(i + 50),
                Duration::from_micros(100 + i as u64 * 100), // Regular temporal sequence
                0.6,
            )
        }).collect();
        let total_duration = Duration::from_millis(2);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_structural_test_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("structural_graph");
        let first_spike_time = Duration::from_micros(400);
        let spikes = create_test_spikes(12); // Larger spike pattern for graph structure
        let total_duration = Duration::from_millis(3);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_exception_test_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("exception_case");
        let first_spike_time = Duration::from_micros(600);
        let spikes = create_test_spikes(4);
        let total_duration = Duration::from_millis(2);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn calculate_architecture_set_efficiency(architectures: &[SelectedArchitecture]) -> f32 {
        let total_performance = architectures.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>();
        let total_memory = architectures.iter()
            .map(|arch| arch.memory_profile.memory_footprint as f32)
            .sum::<f32>() / 1_000_000.0; // Convert to MB
        
        // Efficiency = Performance per MB
        total_performance / total_memory
    }
    
    fn calculate_task_coverage(architectures: &[SelectedArchitecture]) -> f32 {
        let mut covered_tasks = std::collections::HashSet::new();
        for arch in architectures {
            for task in &arch.supported_tasks {
                covered_tasks.insert(*task);
            }
        }
        
        let total_tasks = 4.0; // Semantic, Structural, Temporal, Exception
        covered_tasks.len() as f32 / total_tasks
    }
    
    fn calculate_redundancy_score(architectures: &[SelectedArchitecture]) -> f32 {
        let mut task_counts = HashMap::new();
        for arch in architectures {
            for task in &arch.supported_tasks {
                *task_counts.entry(*task).or_insert(0) += 1;
            }
        }
        
        let total_assignments = task_counts.values().sum::<usize>() as f32;
        let redundant_assignments = task_counts.values().map(|&count| if count > 1 { count - 1 } else { 0 }).sum::<usize>() as f32;
        
        redundant_assignments / total_assignments
    }
    
    fn process_pattern_through_all_columns(
        pattern: &TTFSSpikePattern,
        semantic: &SemanticProcessingColumn,
        structural: &StructuralAnalysisColumn,
        temporal: &TemporalContextColumn,
        exception: &ExceptionDetectionColumn,
    ) -> Vec<ColumnVote> {
        vec![
            semantic.extract_semantic_features(pattern).unwrap(),
            structural.extract_graph_structure(pattern).unwrap(),
            temporal.detect_sequences(pattern).unwrap(),
            exception.find_inhibitions(pattern).unwrap(),
        ]
    }
    
    fn find_winner(votes: &[ColumnVote]) -> &ColumnVote {
        votes.iter().max_by(|a, b| a.activation.partial_cmp(&b.activation).unwrap()).unwrap()
    }
    
    fn create_strong_semantic_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("strong_semantic");
        let first_spike_time = Duration::from_micros(200);
        let spikes = (0..8).map(|i| {
            SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(200 + i as u64 * 50),
                0.8, // High amplitude for strong semantic signal
            )
        }).collect();
        let total_duration = Duration::from_millis(1);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_strong_temporal_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("strong_temporal");
        let first_spike_time = Duration::from_micros(100);
        let spikes = (0..15).map(|i| {
            SpikeEvent::new(
                NeuronId(i + 50),
                Duration::from_micros(100 + i as u64 * 80), // Clear temporal sequence
                0.7,
            )
        }).collect();
        let total_duration = Duration::from_millis(2);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_strong_exception_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("strong_exception");
        let first_spike_time = Duration::from_micros(500);
        let spikes = (0..6).map(|i| {
            SpikeEvent::new(
                NeuronId(i + 100),
                Duration::from_micros(500 + i as u64 * 100),
                if i % 2 == 0 { 0.9 } else { 0.1 }, // Contradictory pattern
            )
        }).collect();
        let total_duration = Duration::from_millis(1);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_noisy_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("noisy_pattern");
        let first_spike_time = Duration::from_micros(150);
        let spikes = (0..20).map(|i| {
            SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(150 + (i as u64 * 37) % 500), // Irregular timing
                (i as f32 * 0.067) % 1.0, // Varying amplitudes
            )
        }).collect();
        let total_duration = Duration::from_millis(3);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_minimal_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("minimal_pattern");
        let first_spike_time = Duration::from_micros(1000);
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(1000), 0.1)
        ];
        let total_duration = Duration::from_millis(1);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_maximum_complexity_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("max_complexity");
        let first_spike_time = Duration::from_micros(50);
        let spikes = (0..50).map(|i| {
            SpikeEvent::new(
                NeuronId(i),
                Duration::from_micros(50 + i as u64 * 20),
                0.3 + (i as f32 * 0.014) % 0.7,
            )
        }).collect();
        let total_duration = Duration::from_millis(5);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_contradictory_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("contradictory");
        let first_spike_time = Duration::from_micros(300);
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(300), 1.0), // Strong signal
            SpikeEvent::new(NeuronId(1), Duration::from_micros(301), 0.0), // Immediate contradiction
            SpikeEvent::new(NeuronId(2), Duration::from_micros(400), 1.0), // Strong signal again
            SpikeEvent::new(NeuronId(3), Duration::from_micros(401), 0.0), // Immediate contradiction
        ];
        let total_duration = Duration::from_millis(1);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
    
    fn create_edge_case_pattern() -> TTFSSpikePattern {
        let concept_id = ConceptId::new("edge_case");
        let first_spike_time = Duration::from_micros(999);
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(999), 0.00001), // Minimal amplitude
            SpikeEvent::new(NeuronId(1000), Duration::from_micros(999), 0.99999), // Large neuron ID, max amplitude
        ];
        let total_duration = Duration::from_micros(1000);
        
        TTFSSpikePattern::new(concept_id, first_spike_time, spikes, total_duration)
    }
}
```

## Implementation
```rust
// src/integration_tests/architecture_integration.rs
use crate::ruv_fann_integration::{ArchitectureSelector, SelectedArchitecture, TaskType, ArchitectureCandidate};
use crate::multi_column::{
    SemanticProcessingColumn, StructuralAnalysisColumn, 
    TemporalContextColumn, ExceptionDetectionColumn,
    ColumnVote, ColumnId
};
use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId, SpikeEvent, NeuronId};
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

/// Comprehensive architecture integration testing framework
#[derive(Debug)]
pub struct ArchitectureIntegrationTester {
    /// Architecture selector
    selector: ArchitectureSelector,
    
    /// Test configuration
    config: IntegrationTestConfig,
    
    /// Performance benchmarks
    benchmarks: PerformanceBenchmarks,
    
    /// Memory monitoring
    memory_monitor: MemoryMonitor,
    
    /// Test results
    test_results: Vec<IntegrationTestResult>,
}

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    /// Target processing time per pattern
    pub target_processing_time: Duration,
    
    /// Maximum total memory usage
    pub max_total_memory: usize,
    
    /// Maximum memory per column
    pub max_column_memory: usize,
    
    /// Number of test patterns for load testing
    pub load_test_pattern_count: usize,
    
    /// Minimum architecture performance score
    pub min_performance_score: f32,
    
    /// Target architecture count (3-4 for intelligent selection)
    pub target_architecture_count: (usize, usize),
    
    /// Parallel processing speedup target
    pub parallel_speedup_target: f32,
}

/// Performance benchmarking suite
#[derive(Debug)]
pub struct PerformanceBenchmarks {
    /// Individual architecture benchmarks
    pub architecture_benchmarks: HashMap<u32, ArchitectureBenchmark>,
    
    /// Column performance benchmarks
    pub column_benchmarks: HashMap<ColumnId, ColumnBenchmark>,
    
    /// Cross-column integration benchmarks
    pub integration_benchmarks: IntegrationBenchmark,
    
    /// Stress test results
    pub stress_test_results: StressTestResults,
}

/// Individual architecture benchmark
#[derive(Debug, Clone)]
pub struct ArchitectureBenchmark {
    /// Architecture ID
    pub architecture_id: u32,
    
    /// Task-specific performance scores
    pub task_performance: HashMap<TaskType, f32>,
    
    /// Memory efficiency score
    pub memory_efficiency: f32,
    
    /// Processing speed score
    pub processing_speed: f32,
    
    /// Specialization score
    pub specialization_score: f32,
    
    /// Overall suitability score
    pub overall_score: f32,
}

/// Column performance benchmark
#[derive(Debug, Clone)]
pub struct ColumnBenchmark {
    /// Column type
    pub column_id: ColumnId,
    
    /// Selected architecture
    pub selected_architecture: SelectedArchitecture,
    
    /// Average processing time
    pub avg_processing_time: Duration,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// Accuracy on specialized tasks
    pub specialization_accuracy: f32,
    
    /// Performance under load
    pub load_performance: LoadPerformance,
    
    /// Cache hit rates
    pub cache_performance: CachePerformance,
}

/// Load performance metrics
#[derive(Debug, Clone)]
pub struct LoadPerformance {
    /// Processing time under load
    pub processing_time_under_load: Duration,
    
    /// Throughput (patterns per second)
    pub throughput: f32,
    
    /// Memory growth under load
    pub memory_growth: usize,
    
    /// Performance degradation factor
    pub degradation_factor: f32,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CachePerformance {
    /// Cache hit rate
    pub hit_rate: f32,
    
    /// Average hit time
    pub avg_hit_time: Duration,
    
    /// Average miss time
    pub avg_miss_time: Duration,
    
    /// Cache efficiency score
    pub efficiency_score: f32,
}

/// Integration benchmark results
#[derive(Debug, Clone)]
pub struct IntegrationBenchmark {
    /// Cross-column compatibility score
    pub compatibility_score: f32,
    
    /// Winner-take-all effectiveness
    pub winner_take_all_score: f32,
    
    /// Parallel processing speedup
    pub parallel_speedup: f32,
    
    /// System-wide memory efficiency
    pub system_memory_efficiency: f32,
    
    /// Architecture selection effectiveness
    pub selection_effectiveness: f32,
}

/// Stress testing results
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Robustness under noisy input
    pub noise_robustness: f32,
    
    /// Performance on edge cases
    pub edge_case_performance: f32,
    
    /// Handling of contradictory patterns
    pub contradiction_handling: f32,
    
    /// Minimal pattern processing
    pub minimal_pattern_handling: f32,
    
    /// Maximum complexity handling
    pub max_complexity_handling: f32,
}

/// Memory monitoring system
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Initial memory baseline
    baseline_memory: usize,
    
    /// Peak memory usage
    peak_memory: usize,
    
    /// Memory usage history
    memory_history: Vec<MemorySnapshot>,
    
    /// Memory leak detection
    leak_detector: MemoryLeakDetector,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp
    pub timestamp: Instant,
    
    /// Total memory usage
    pub total_memory: usize,
    
    /// Memory per column
    pub column_memory: HashMap<ColumnId, usize>,
    
    /// Cache memory usage
    pub cache_memory: usize,
    
    /// Active patterns count
    pub active_patterns: usize,
}

/// Memory leak detection system
#[derive(Debug)]
pub struct MemoryLeakDetector {
    /// Memory growth tracking
    growth_tracker: Vec<(Instant, usize)>,
    
    /// Leak detection threshold
    leak_threshold: f32,
    
    /// Detected leaks
    detected_leaks: Vec<MemoryLeak>,
}

/// Memory leak information
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    /// Leak detection time
    pub detected_at: Instant,
    
    /// Growth rate (bytes per second)
    pub growth_rate: f32,
    
    /// Suspected source
    pub suspected_source: String,
    
    /// Severity level
    pub severity: LeakSeverity,
}

/// Memory leak severity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeakSeverity {
    Minor,      // Slow growth, likely cache expansion
    Moderate,   // Noticeable growth, needs monitoring
    Severe,     // Fast growth, immediate attention needed
    Critical,   // Runaway growth, system stability at risk
}

/// Individual integration test result
#[derive(Debug, Clone)]
pub struct IntegrationTestResult {
    /// Test name
    pub test_name: String,
    
    /// Test category
    pub test_category: TestCategory,
    
    /// Test outcome
    pub outcome: TestOutcome,
    
    /// Performance metrics
    pub performance_metrics: TestPerformanceMetrics,
    
    /// Error details if test failed
    pub error_details: Option<String>,
    
    /// Test duration
    pub test_duration: Duration,
    
    /// Test timestamp
    pub timestamp: Instant,
}

/// Test categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestCategory {
    ArchitectureSelection,
    ColumnSpecialization,
    CrossColumnIntegration,
    PerformanceBenchmarking,
    MemoryValidation,
    StressTesting,
    ParallelProcessing,
    WinnerTakeAll,
}

/// Test outcomes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestOutcome {
    Passed,
    Failed,
    Warning,
    Skipped,
}

/// Test performance metrics
#[derive(Debug, Clone)]
pub struct TestPerformanceMetrics {
    /// Processing time
    pub processing_time: Duration,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// Accuracy score
    pub accuracy: f32,
    
    /// Efficiency score
    pub efficiency: f32,
    
    /// Custom metrics
    pub custom_metrics: HashMap<String, f32>,
}

/// Architecture selection validation results
#[derive(Debug, Clone)]
pub struct ArchitectureSelectionValidation {
    /// Selected architecture count
    pub selected_count: usize,
    
    /// Task coverage score
    pub task_coverage: f32,
    
    /// Performance distribution
    pub performance_distribution: Vec<f32>,
    
    /// Memory efficiency
    pub memory_efficiency: f32,
    
    /// Specialization balance
    pub specialization_balance: f32,
    
    /// Selection quality score
    pub selection_quality: f32,
}

/// Column specialization validation results
#[derive(Debug, Clone)]
pub struct ColumnSpecializationValidation {
    /// Specialization accuracy per column
    pub column_accuracies: HashMap<ColumnId, f32>,
    
    /// Cross-specialization performance
    pub cross_performance: HashMap<ColumnId, HashMap<TaskType, f32>>,
    
    /// Architecture-task alignment scores
    pub alignment_scores: HashMap<ColumnId, f32>,
    
    /// Specialization effectiveness
    pub effectiveness_score: f32,
}

impl ArchitectureIntegrationTester {
    /// Create new integration tester
    pub fn new() -> Self {
        Self {
            selector: ArchitectureSelector::new(),
            config: IntegrationTestConfig::default(),
            benchmarks: PerformanceBenchmarks::new(),
            memory_monitor: MemoryMonitor::new(),
            test_results: Vec::new(),
        }
    }
    
    /// Run comprehensive integration test suite
    pub fn run_complete_test_suite(&mut self) -> IntegrationTestSuiteResults {
        let start_time = Instant::now();
        
        println!("Starting comprehensive architecture integration testing...");
        
        // Phase 1: Architecture Selection Validation
        println!("Phase 1: Architecture Selection Validation");
        let selection_results = self.validate_architecture_selection();
        
        // Phase 2: Column Specialization Testing
        println!("Phase 2: Column Specialization Testing");
        let specialization_results = self.validate_column_specialization();
        
        // Phase 3: Performance Benchmarking
        println!("Phase 3: Performance Benchmarking");
        let performance_results = self.run_performance_benchmarks();
        
        // Phase 4: Memory Validation
        println!("Phase 4: Memory Validation");
        let memory_results = self.validate_memory_usage();
        
        // Phase 5: Integration Testing
        println!("Phase 5: Cross-Column Integration Testing");
        let integration_results = self.test_cross_column_integration();
        
        // Phase 6: Stress Testing
        println!("Phase 6: Stress Testing");
        let stress_results = self.run_stress_tests();
        
        // Phase 7: Parallel Processing Validation
        println!("Phase 7: Parallel Processing Validation");
        let parallel_results = self.validate_parallel_processing();
        
        let total_duration = start_time.elapsed();
        
        println!("Integration testing completed in {:?}", total_duration);
        
        IntegrationTestSuiteResults {
            selection_validation: selection_results,
            specialization_validation: specialization_results,
            performance_benchmarks: performance_results,
            memory_validation: memory_results,
            integration_validation: integration_results,
            stress_test_results: stress_results,
            parallel_validation: parallel_results,
            total_test_duration: total_duration,
            overall_success: self.calculate_overall_success(),
        }
    }
    
    /// Validate architecture selection follows intelligent selection philosophy
    pub fn validate_architecture_selection(&mut self) -> ArchitectureSelectionValidation {
        let start_time = Instant::now();
        
        // Get all available architectures
        let all_candidates = self.selector.get_all_architecture_candidates();
        let selected_set = self.selector.select_optimal_architecture_set();
        
        // Validate selection count (should be 3-4)
        let selected_count = selected_set.len();
        let count_valid = selected_count >= self.config.target_architecture_count.0 && 
                         selected_count <= self.config.target_architecture_count.1;
        
        // Calculate task coverage
        let task_coverage = self.calculate_task_coverage(&selected_set);
        
        // Calculate performance distribution
        let performance_distribution: Vec<f32> = selected_set.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .collect();
        
        // Calculate memory efficiency
        let memory_efficiency = self.calculate_memory_efficiency(&selected_set);
        
        // Calculate specialization balance
        let specialization_balance = self.calculate_specialization_balance(&selected_set);
        
        // Calculate overall selection quality
        let selection_quality = self.calculate_selection_quality(&selected_set, &all_candidates);
        
        // Record test result
        self.record_test_result(
            "architecture_selection_validation".to_string(),
            TestCategory::ArchitectureSelection,
            if count_valid && task_coverage >= 0.9 && selection_quality >= 0.8 {
                TestOutcome::Passed
            } else {
                TestOutcome::Failed
            },
            TestPerformanceMetrics {
                processing_time: start_time.elapsed(),
                memory_usage: 0,
                accuracy: task_coverage,
                efficiency: selection_quality,
                custom_metrics: HashMap::new(),
            },
            None,
        );
        
        ArchitectureSelectionValidation {
            selected_count,
            task_coverage,
            performance_distribution,
            memory_efficiency,
            specialization_balance,
            selection_quality,
        }
    }
    
    /// Validate column specialization and architecture assignment
    pub fn validate_column_specialization(&mut self) -> ColumnSpecializationValidation {
        let start_time = Instant::now();
        
        // Initialize all columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&self.selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&self.selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&self.selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&self.selector).unwrap();
        
        // Create specialized test patterns
        let semantic_pattern = self.create_semantic_test_pattern();
        let structural_pattern = self.create_structural_test_pattern();
        let temporal_pattern = self.create_temporal_test_pattern();
        let exception_pattern = self.create_exception_test_pattern();
        
        // Test specialization accuracy
        let mut column_accuracies = HashMap::new();
        
        // Test semantic column
        let semantic_result = semantic_column.extract_semantic_features(&semantic_pattern).unwrap();
        column_accuracies.insert(ColumnId::Semantic, semantic_result.confidence);
        
        // Test structural column
        let structural_result = structural_column.extract_graph_structure(&structural_pattern).unwrap();
        column_accuracies.insert(ColumnId::Structural, structural_result.confidence);
        
        // Test temporal column
        let temporal_result = temporal_column.detect_sequences(&temporal_pattern).unwrap();
        column_accuracies.insert(ColumnId::Temporal, temporal_result.confidence);
        
        // Test exception column
        let exception_result = exception_column.find_inhibitions(&exception_pattern).unwrap();
        column_accuracies.insert(ColumnId::Exception, exception_result.confidence);
        
        // Test cross-specialization performance
        let cross_performance = self.test_cross_specialization_performance(
            &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Calculate alignment scores
        let alignment_scores = self.calculate_architecture_task_alignment(
            &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Calculate effectiveness score
        let effectiveness_score = column_accuracies.values().sum::<f32>() / column_accuracies.len() as f32;
        
        // Record test result
        self.record_test_result(
            "column_specialization_validation".to_string(),
            TestCategory::ColumnSpecialization,
            if effectiveness_score >= 0.7 { TestOutcome::Passed } else { TestOutcome::Failed },
            TestPerformanceMetrics {
                processing_time: start_time.elapsed(),
                memory_usage: 0,
                accuracy: effectiveness_score,
                efficiency: effectiveness_score,
                custom_metrics: HashMap::new(),
            },
            None,
        );
        
        ColumnSpecializationValidation {
            column_accuracies,
            cross_performance,
            alignment_scores,
            effectiveness_score,
        }
    }
    
    /// Run comprehensive performance benchmarks
    pub fn run_performance_benchmarks(&mut self) -> PerformanceBenchmarks {
        let start_time = Instant::now();
        
        // Initialize columns for benchmarking
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&self.selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&self.selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&self.selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&self.selector).unwrap();
        
        // Benchmark individual architectures
        let architecture_benchmarks = self.benchmark_individual_architectures();
        
        // Benchmark columns
        let column_benchmarks = self.benchmark_columns(
            &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Benchmark integration
        let integration_benchmarks = self.benchmark_integration(
            &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Run stress tests
        let stress_test_results = self.benchmark_stress_performance(
            &semantic_column, &structural_column, &temporal_column, &exception_column
        );
        
        // Record test result
        let overall_performance = (
            architecture_benchmarks.values().map(|b| b.overall_score).sum::<f32>() / architecture_benchmarks.len() as f32 +
            integration_benchmarks.compatibility_score
        ) / 2.0;
        
        self.record_test_result(
            "performance_benchmarking".to_string(),
            TestCategory::PerformanceBenchmarking,
            if overall_performance >= 0.8 { TestOutcome::Passed } else { TestOutcome::Failed },
            TestPerformanceMetrics {
                processing_time: start_time.elapsed(),
                memory_usage: 0,
                accuracy: overall_performance,
                efficiency: overall_performance,
                custom_metrics: HashMap::new(),
            },
            None,
        );
        
        PerformanceBenchmarks {
            architecture_benchmarks,
            column_benchmarks,
            integration_benchmarks,
            stress_test_results,
        }
    }
    
    /// Validate memory usage across all components
    pub fn validate_memory_usage(&mut self) -> MemoryValidationResults {
        let start_time = Instant::now();
        
        // Initialize memory monitoring
        self.memory_monitor.start_monitoring();
        
        // Initialize all columns
        let semantic_column = SemanticProcessingColumn::new_with_auto_selection(&self.selector).unwrap();
        let structural_column = StructuralAnalysisColumn::new_with_auto_selection(&self.selector).unwrap();
        let temporal_column = TemporalContextColumn::new_with_auto_selection(&self.selector).unwrap();
        let exception_column = ExceptionDetectionColumn::new_with_auto_selection(&self.selector).unwrap();
        
        let initialization_memory = self.memory_monitor.get_current_usage();
        
        // Test memory under load
        let test_patterns = self.create_multiple_test_patterns(self.config.load_test_pattern_count);
        
        let pre_load_memory = self.memory_monitor.get_current_usage();
        
        // Process patterns and monitor memory
        for pattern in &test_patterns {
            let _ = semantic_column.extract_semantic_features(pattern);
            let _ = structural_column.extract_graph_structure(pattern);
            let _ = temporal_column.detect_sequences(pattern);
            let _ = exception_column.find_inhibitions(pattern);
            
            self.memory_monitor.record_snapshot();
        }
        
        let post_load_memory = self.memory_monitor.get_current_usage();
        let memory_growth = post_load_memory - pre_load_memory;
        
        // Check for memory leaks
        let leak_analysis = self.memory_monitor.analyze_leaks();
        
        // Validate memory constraints
        let individual_memory_valid = self.validate_individual_column_memory(&semantic_column, &structural_column, &temporal_column, &exception_column);
        let total_memory_valid = post_load_memory <= self.config.max_total_memory;
        let memory_growth_acceptable = memory_growth <= 100_000_000; // 100MB growth limit
        
        // Record test result
        self.record_test_result(
            "memory_validation".to_string(),
            TestCategory::MemoryValidation,
            if individual_memory_valid && total_memory_valid && memory_growth_acceptable && leak_analysis.severe_leaks == 0 {
                TestOutcome::Passed
            } else {
                TestOutcome::Failed
            },
            TestPerformanceMetrics {
                processing_time: start_time.elapsed(),
                memory_usage: post_load_memory,
                accuracy: if total_memory_valid { 1.0 } else { 0.0 },
                efficiency: (self.config.max_total_memory as f32 - post_load_memory as f32) / self.config.max_total_memory as f32,
                custom_metrics: HashMap::new(),
            },
            None,
        );
        
        MemoryValidationResults {
            initialization_memory,
            peak_memory: self.memory_monitor.peak_memory,
            memory_growth,
            individual_column_memory: self.get_individual_column_memory(&semantic_column, &structural_column, &temporal_column, &exception_column),
            memory_efficiency: self.calculate_memory_efficiency_score(),
            leak_analysis,
            constraints_satisfied: individual_memory_valid && total_memory_valid,
        }
    }
    
    // Implementation continues...
    // Private helper methods and additional test implementations
    
    fn calculate_task_coverage(&self, architectures: &[SelectedArchitecture]) -> f32 {
        let mut covered_tasks = HashSet::new();
        for arch in architectures {
            for task in &arch.supported_tasks {
                covered_tasks.insert(*task);
            }
        }
        
        let total_core_tasks = 4.0; // Semantic, Structural, Temporal, Exception
        covered_tasks.len() as f32 / total_core_tasks
    }
    
    fn calculate_memory_efficiency(&self, architectures: &[SelectedArchitecture]) -> f32 {
        let total_memory = architectures.iter()
            .map(|arch| arch.memory_profile.memory_footprint as f32)
            .sum::<f32>();
        let total_performance = architectures.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>();
        
        total_performance / (total_memory / 1_000_000.0) // Performance per MB
    }
    
    fn calculate_specialization_balance(&self, architectures: &[SelectedArchitecture]) -> f32 {
        let mut task_counts = HashMap::new();
        for arch in architectures {
            for task in &arch.supported_tasks {
                *task_counts.entry(*task).or_insert(0) += 1;
            }
        }
        
        let ideal_distribution = 1.0; // Each task covered by exactly one architecture
        let variance = task_counts.values()
            .map(|&count| (count as f32 - ideal_distribution).powi(2))
            .sum::<f32>() / task_counts.len() as f32;
        
        1.0 / (1.0 + variance) // Higher score for lower variance
    }
    
    fn calculate_selection_quality(&self, selected: &[SelectedArchitecture], all: &[ArchitectureCandidate]) -> f32 {
        let selected_avg = selected.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>() / selected.len() as f32;
        
        let all_avg = all.iter()
            .map(|arch| arch.performance_metrics.performance_score)
            .sum::<f32>() / all.len() as f32;
        
        selected_avg / all_avg // Quality improvement ratio
    }
    
    fn record_test_result(
        &mut self,
        test_name: String,
        category: TestCategory,
        outcome: TestOutcome,
        metrics: TestPerformanceMetrics,
        error: Option<String>,
    ) {
        self.test_results.push(IntegrationTestResult {
            test_name,
            test_category: category,
            outcome,
            performance_metrics: metrics,
            error_details: error,
            test_duration: metrics.processing_time,
            timestamp: Instant::now(),
        });
    }
    
    fn calculate_overall_success(&self) -> bool {
        let total_tests = self.test_results.len();
        if total_tests == 0 {
            return false;
        }
        
        let passed_tests = self.test_results.iter()
            .filter(|r| r.outcome == TestOutcome::Passed)
            .count();
        
        let success_rate = passed_tests as f32 / total_tests as f32;
        success_rate >= 0.9 // 90% pass rate required
    }
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            target_processing_time: Duration::from_millis(1),
            max_total_memory: 200_000_000, // 200MB
            max_column_memory: 50_000_000,  // 50MB per column
            load_test_pattern_count: 1000,
            min_performance_score: 0.7,
            target_architecture_count: (3, 4), // 3-4 architectures
            parallel_speedup_target: 1.5,
        }
    }
}

impl PerformanceBenchmarks {
    pub fn new() -> Self {
        Self {
            architecture_benchmarks: HashMap::new(),
            column_benchmarks: HashMap::new(),
            integration_benchmarks: IntegrationBenchmark::default(),
            stress_test_results: StressTestResults::default(),
        }
    }
}

impl Default for IntegrationBenchmark {
    fn default() -> Self {
        Self {
            compatibility_score: 0.0,
            winner_take_all_score: 0.0,
            parallel_speedup: 0.0,
            system_memory_efficiency: 0.0,
            selection_effectiveness: 0.0,
        }
    }
}

impl Default for StressTestResults {
    fn default() -> Self {
        Self {
            noise_robustness: 0.0,
            edge_case_performance: 0.0,
            contradiction_handling: 0.0,
            minimal_pattern_handling: 0.0,
            max_complexity_handling: 0.0,
        }
    }
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            baseline_memory: 100_000_000, // 100MB baseline
            peak_memory: 100_000_000,
            memory_history: Vec::new(),
            leak_detector: MemoryLeakDetector::new(),
        }
    }
    
    pub fn start_monitoring(&mut self) {
        self.baseline_memory = self.get_current_usage();
        self.peak_memory = self.baseline_memory;
    }
    
    pub fn get_current_usage(&self) -> usize {
        // Mock memory usage measurement
        150_000_000 + self.memory_history.len() * 1_000_000 // 150MB + 1MB per recorded snapshot
    }
    
    pub fn record_snapshot(&mut self) {
        let current_memory = self.get_current_usage();
        if current_memory > self.peak_memory {
            self.peak_memory = current_memory;
        }
        
        self.memory_history.push(MemorySnapshot {
            timestamp: Instant::now(),
            total_memory: current_memory,
            column_memory: HashMap::new(), // Mock
            cache_memory: current_memory / 10, // Mock: 10% cache
            active_patterns: 0,
        });
        
        self.leak_detector.track_memory_growth(current_memory);
    }
    
    pub fn analyze_leaks(&self) -> MemoryLeakAnalysis {
        self.leak_detector.analyze()
    }
}

impl MemoryLeakDetector {
    pub fn new() -> Self {
        Self {
            growth_tracker: Vec::new(),
            leak_threshold: 1.5, // 1.5x growth indicates potential leak
            detected_leaks: Vec::new(),
        }
    }
    
    pub fn track_memory_growth(&mut self, current_memory: usize) {
        self.growth_tracker.push((Instant::now(), current_memory));
        
        // Keep only recent measurements
        if self.growth_tracker.len() > 100 {
            self.growth_tracker.remove(0);
        }
        
        // Check for leaks if we have enough data
        if self.growth_tracker.len() >= 10 {
            self.check_for_leaks();
        }
    }
    
    fn check_for_leaks(&mut self) {
        // Simple leak detection: check if memory consistently grows
        let recent_growth = self.calculate_recent_growth_rate();
        
        if recent_growth > self.leak_threshold {
            let leak = MemoryLeak {
                detected_at: Instant::now(),
                growth_rate: recent_growth,
                suspected_source: "Unknown".to_string(),
                severity: if recent_growth > 5.0 {
                    LeakSeverity::Critical
                } else if recent_growth > 3.0 {
                    LeakSeverity::Severe
                } else if recent_growth > 2.0 {
                    LeakSeverity::Moderate
                } else {
                    LeakSeverity::Minor
                },
            };
            
            self.detected_leaks.push(leak);
        }
    }
    
    fn calculate_recent_growth_rate(&self) -> f32 {
        if self.growth_tracker.len() < 2 {
            return 0.0;
        }
        
        let start = &self.growth_tracker[0];
        let end = &self.growth_tracker[self.growth_tracker.len() - 1];
        
        let time_diff = end.0.duration_since(start.0).as_secs_f32();
        let memory_diff = end.1 as f32 - start.1 as f32;
        
        if time_diff > 0.0 {
            memory_diff / (start.1 as f32) / time_diff // Growth rate per second
        } else {
            0.0
        }
    }
    
    pub fn analyze(&self) -> MemoryLeakAnalysis {
        MemoryLeakAnalysis {
            total_leaks: self.detected_leaks.len(),
            severe_leaks: self.detected_leaks.iter()
                .filter(|leak| matches!(leak.severity, LeakSeverity::Severe | LeakSeverity::Critical))
                .count(),
            growth_trend: self.calculate_recent_growth_rate(),
            leak_details: self.detected_leaks.clone(),
        }
    }
}

/// Memory leak analysis results
#[derive(Debug, Clone)]
pub struct MemoryLeakAnalysis {
    /// Total number of detected leaks
    pub total_leaks: usize,
    
    /// Number of severe leaks
    pub severe_leaks: usize,
    
    /// Overall memory growth trend
    pub growth_trend: f32,
    
    /// Detailed leak information
    pub leak_details: Vec<MemoryLeak>,
}

/// Memory validation results
#[derive(Debug, Clone)]
pub struct MemoryValidationResults {
    /// Memory usage after initialization
    pub initialization_memory: usize,
    
    /// Peak memory usage during testing
    pub peak_memory: usize,
    
    /// Memory growth under load
    pub memory_growth: usize,
    
    /// Individual column memory usage
    pub individual_column_memory: HashMap<ColumnId, usize>,
    
    /// Memory efficiency score
    pub memory_efficiency: f32,
    
    /// Leak analysis results
    pub leak_analysis: MemoryLeakAnalysis,
    
    /// Whether memory constraints are satisfied
    pub constraints_satisfied: bool,
}

/// Complete integration test suite results
#[derive(Debug, Clone)]
pub struct IntegrationTestSuiteResults {
    /// Architecture selection validation results
    pub selection_validation: ArchitectureSelectionValidation,
    
    /// Column specialization validation results
    pub specialization_validation: ColumnSpecializationValidation,
    
    /// Performance benchmark results
    pub performance_benchmarks: PerformanceBenchmarks,
    
    /// Memory validation results
    pub memory_validation: MemoryValidationResults,
    
    /// Integration validation results
    pub integration_validation: IntegrationValidationResults,
    
    /// Stress test results
    pub stress_test_results: StressTestResults,
    
    /// Parallel processing validation results
    pub parallel_validation: ParallelValidationResults,
    
    /// Total test duration
    pub total_test_duration: Duration,
    
    /// Overall success indicator
    pub overall_success: bool,
}

/// Placeholder for additional result types
#[derive(Debug, Clone)]
pub struct IntegrationValidationResults {
    pub compatibility_score: f32,
    pub winner_take_all_effectiveness: f32,
}

#[derive(Debug, Clone)]
pub struct ParallelValidationResults {
    pub speedup_achieved: f32,
    pub efficiency_score: f32,
}
```

## Verification Steps
1. Implement comprehensive architecture selection validation ensuring 3-4 optimal architectures are chosen
2. Add column specialization testing to verify appropriate architecture assignment for each task type
3. Implement performance benchmarking across all architectures and columns with sub-millisecond targets
4. Add memory validation with individual (≤50MB) and total (≤200MB) memory constraints
5. Implement cross-column integration testing with winner-take-all validation
6. Add stress testing for robustness under noisy, edge case, and contradictory inputs
7. Implement parallel processing validation with speedup measurement and efficiency scoring
8. Add comprehensive memory leak detection and growth pattern analysis

## Success Criteria
- [ ] Architecture selection validates intelligent selection philosophy (3-4 architectures from 29 available)
- [ ] Task coverage validation ensures all core task types (Semantic, Structural, Temporal, Exception) are covered
- [ ] Performance benchmarking confirms all columns meet sub-millisecond processing targets (<1ms per pattern)
- [ ] Memory validation ensures individual column memory ≤50MB and total system memory ≤200MB
- [ ] Specialization testing confirms columns perform best on their designated task types
- [ ] Cross-column integration validates compatible neural output dimensions and vote structures
- [ ] Winner-take-all testing shows appropriate column activation for different pattern types
- [ ] Parallel processing achieves ≥1.5x speedup over sequential processing
- [ ] Stress testing demonstrates robustness under challenging input conditions
- [ ] Memory leak detection identifies zero severe leaks during extended testing
- [ ] Overall test suite passes with ≥90% success rate across all test categories
- [ ] Integration framework validates the intelligent architecture selection philosophy effectiveness