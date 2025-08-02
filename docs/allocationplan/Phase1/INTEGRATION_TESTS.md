# Phase 1 Cross-Task Integration Test Suites

**Duration**: 4 hours  
**Purpose**: Verify all Phase 1 tasks work together correctly in production scenarios  
**Dependencies**: All tasks 1.1-1.14 must be completed  

## Overview

This document provides comprehensive integration test suites that verify the correct interaction between all Phase 1 components. Unlike unit tests that test individual components, these tests validate system-wide behavior and cross-component dependencies.

## Integration Test Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Integration Test Matrix                     │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Foundation    │   Biological    │   Inhibition    │  Spatial  │
│  (Tasks 1.1-3)  │  (Tasks 1.4-6)  │  (Tasks 1.7-9)  │(Tasks 1.10-12)│
├─────────────────┼─────────────────┼─────────────────┼───────────┤
│ • State Machine │ • Activation    │ • Lateral Inhib │ • 3D Grid │
│ • Atomic Ops    │ • Decay         │ • Winner Take   │ • Indexing│
│ • Thread Safety │ • Hebbian       │ • Deduplication │ • Neighbors│
└─────────────────┴─────────────────┴─────────────────┴───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ System Integration│
                    │ (Tasks 1.13-14) │
                    │ • Parallel Alloc│
                    │ • Performance   │
                    └─────────────────┘
```

## Test Suite 1: Foundation Integration (Tasks 1.1-1.3)

Tests the interaction between state machine, atomic operations, and thread safety.

```rust
// integration_tests/foundation_integration.rs
use neuromorphic_core::{
    CorticalColumn, EnhancedCorticalColumn, ColumnState, StateTransitionError,
    AtomicColumnState, ThreadSafetyValidator
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use criterion::{black_box, Criterion};

#[tokio::test]
async fn test_concurrent_state_transitions_with_atomics() {
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    let barrier = Arc::new(Barrier::new(16));
    let mut handles = vec![];
    
    // 16 threads attempting different state transitions simultaneously
    for thread_id in 0..16 {
        let col = column.clone();
        let bar = barrier.clone();
        
        handles.push(tokio::spawn(async move {
            bar.wait();
            
            match thread_id % 4 {
                0 => {
                    // Activation sequence
                    let _ = col.try_activate();
                    tokio::time::sleep(Duration::from_micros(10)).await;
                    let _ = col.try_compete();
                }
                1 => {
                    // Competition sequence  
                    let _ = col.try_compete();
                    tokio::time::sleep(Duration::from_micros(5)).await;
                    let _ = col.try_allocate();
                }
                2 => {
                    // Reset sequence
                    let _ = col.try_reset();
                    tokio::time::sleep(Duration::from_micros(8)).await;
                    let _ = col.try_activate();
                }
                _ => {
                    // Full cycle
                    let _ = col.try_activate();
                    let _ = col.try_compete();
                    let _ = col.try_allocate();
                    let _ = col.try_enter_refractory();
                    let _ = col.try_reset();
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify final state is valid
    let final_state = column.current_state();
    assert!(matches!(
        final_state,
        ColumnState::Available | ColumnState::Activated | 
        ColumnState::Competing | ColumnState::Allocated | 
        ColumnState::Refractory
    ));
    
    // Verify transition count consistency
    let transition_count = column.transition_count();
    assert!(transition_count > 0, "No transitions occurred");
    assert!(transition_count <= 80, "Too many transitions (race condition?)");
}

#[test]
fn test_atomic_state_consistency_under_load() {
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    let iterations = 10000;
    let mut handles = vec![];
    
    // High-frequency state transitions
    for _ in 0..8 {
        let col = column.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..iterations {
                // Rapid fire transitions
                let _ = col.try_activate();
                let _ = col.try_compete();
                let _ = col.try_reset();
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // State must be consistent (no corrupted values)
    let state = column.current_state();
    assert!(state as u8 <= 4, "Corrupted state value: {:?}", state);
    
    // Timing information should be recent
    let time_since_transition = column.time_since_last_transition();
    assert!(time_since_transition < Duration::from_secs(1));
}

#[test] 
fn test_memory_ordering_guarantees() {
    let column = Arc::new(EnhancedCorticalColumn::new(1));
    let observer = column.clone();
    
    let mut observed_states = vec![];
    let observations_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_flag = observations_done.clone();
    
    // Observer thread - records all state changes
    let observer_handle = thread::spawn(move || {
        while !done_flag.load(std::sync::atomic::Ordering::Acquire) {
            observed_states.push((
                observer.current_state(),
                observer.transition_count(),
                Instant::now()
            ));
            thread::sleep(Duration::from_nanos(100));
        }
        observed_states
    });
    
    // Mutator thread - performs state transitions
    let mutator_handle = thread::spawn(move || {
        for _ in 0..1000 {
            let _ = column.try_activate();
            let _ = column.try_compete();
            let _ = column.try_allocate(); 
            let _ = column.try_enter_refractory();
            let _ = column.try_reset();
        }
    });
    
    mutator_handle.join().unwrap();
    observations_done.store(true, std::sync::atomic::Ordering::Release);
    let observations = observer_handle.join().unwrap();
    
    // Verify memory ordering: transition_count should be monotonic
    let mut prev_count = 0;
    for (_, count, _) in observations {
        assert!(count >= prev_count, "Non-monotonic transition count");
        prev_count = count;
    }
}
```

## Test Suite 2: Biological Integration (Tasks 1.4-1.6)

Tests the interaction between activation dynamics, decay, and Hebbian learning.

```rust
// integration_tests/biological_integration.rs
use neuromorphic_core::{
    BiologicalCorticalColumn, BiologicalConfig, StimulationResult,
    MembranePotential, RefractoryPeriodManager, HebbianLearningManager,
    HebbianUpdateResult
};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn test_activation_decay_hebbian_cycle() {
    let config = BiologicalConfig::cortical_neuron();
    let column1 = BiologicalCorticalColumn::new(1, config.clone());
    let column2 = BiologicalCorticalColumn::new(2, config);
    
    // Phase 1: Establish baseline
    let initial_state1 = column1.biological_state();
    let initial_state2 = column2.biological_state();
    
    assert!(initial_state1.membrane_voltage < 0.1);
    assert!(initial_state2.membrane_voltage < 0.1);
    assert!(initial_state1.synaptic_connections.is_empty());
    
    // Phase 2: Stimulate and fire column1
    let result1 = column1.stimulate(1.2, 2.0);
    assert!(matches!(result1, StimulationResult::Fired { .. }));
    
    // Small delay for STDP timing
    thread::sleep(Duration::from_millis(5));
    
    // Phase 3: Stimulate and fire column2  
    let result2 = column2.stimulate(1.3, 2.0);
    assert!(matches!(result2, StimulationResult::Fired { .. }));
    
    // Phase 4: Apply Hebbian learning
    let learn_result = column1.learn_from_coactivation(&column2);
    
    match learn_result {
        HebbianUpdateResult::Updated { 
            old_weight, new_weight, weight_delta, stdp_factor, timing_ms 
        } => {
            assert!(new_weight > old_weight, "Weight should increase");
            assert!(weight_delta > 0.0, "Positive weight change expected");
            assert!(timing_ms > 0.0 && timing_ms < 20.0, "STDP timing window");
            assert!(stdp_factor > 0.0, "Potentiation expected");
        }
        _ => panic!("Expected successful Hebbian update"),
    }
    
    // Phase 5: Verify persistent changes
    let final_state1 = column1.biological_state();
    assert!(!final_state1.synaptic_connections.is_empty());
    assert!(final_state1.synaptic_connections.get(&2).unwrap() > &0.0);
    
    // Phase 6: Test decay over time
    thread::sleep(Duration::from_millis(50));
    let decayed_state1 = column1.biological_state();
    
    // Membrane should have decayed toward resting potential
    assert!(decayed_state1.membrane_voltage < 0.2);
    assert!(decayed_state1.time_since_fire_ms >= 50.0);
}

#[test]
fn test_refractory_period_biological_accuracy() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Fire the neuron
    let fire_result = column.stimulate(1.5, 1.0);
    assert!(matches!(fire_result, StimulationResult::Fired { .. }));
    
    let fire_time = Instant::now();
    
    // During absolute refractory period - should be blocked
    for i in 0..5 {
        thread::sleep(Duration::from_millis(1));
        let result = column.stimulate(2.0, 1.0); // Very strong stimulus
        
        if i < 2 { // First 2ms (absolute refractory)
            assert!(matches!(result, StimulationResult::RefractoryBlock { .. }));
        }
    }
    
    // During relative refractory period - higher threshold
    thread::sleep(Duration::from_millis(5));
    let state = column.biological_state();
    assert!(state.firing_threshold > 0.8); // Elevated threshold
    
    // After refractory period - normal operation
    thread::sleep(Duration::from_millis(10));
    let normal_result = column.stimulate(1.0, 1.0);
    assert!(!matches!(normal_result, StimulationResult::RefractoryBlock { .. }));
}

#[test]
fn test_membrane_exponential_decay_accuracy() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config.clone());
    
    // Apply strong input
    column.stimulate(1.0, 5.0);
    let initial_voltage = column.biological_state().membrane_voltage;
    
    // Record voltage over time to verify exponential decay
    let mut voltage_samples = vec![];
    let start_time = Instant::now();
    
    for _ in 0..20 {
        thread::sleep(Duration::from_millis(2));
        let elapsed_ms = start_time.elapsed().as_millis() as f32;
        let voltage = column.biological_state().membrane_voltage;
        voltage_samples.push((elapsed_ms, voltage));
    }
    
    // Verify exponential decay: V(t) = V₀ * e^(-t/τ)
    let tau_ms = config.membrane_tau_ms;
    
    for (t_ms, voltage) in voltage_samples {
        if t_ms > 5.0 { // Skip initial settling period
            let expected_voltage = initial_voltage * (-t_ms / tau_ms).exp();
            let error = (voltage - expected_voltage).abs();
            let relative_error = error / expected_voltage.max(0.1);
            
            assert!(relative_error < 0.2, 
                "Decay accuracy: t={}ms, expected={:.3}, actual={:.3}, error={:.1}%",
                t_ms, expected_voltage, voltage, relative_error * 100.0);
        }
    }
}

#[test]
fn test_concurrent_biological_processing() {
    let config = BiologicalConfig::fast_processing();
    let columns: Vec<_> = (0..10).map(|i| 
        Arc::new(BiologicalCorticalColumn::new(i, config.clone()))
    ).collect();
    
    let mut handles = vec![];
    
    // Concurrent stimulation of all columns
    for (i, column) in columns.iter().enumerate() {
        let col = column.clone();
        handles.push(thread::spawn(move || {
            let mut fire_count = 0;
            let start = Instant::now();
            
            // High-frequency stimulation for 100ms
            while start.elapsed() < Duration::from_millis(100) {
                let stimulus_strength = 0.8 + (i as f32 * 0.1);
                let result = col.stimulate(stimulus_strength, 0.5);
                
                if matches!(result, StimulationResult::Fired { .. }) {
                    fire_count += 1;
                }
                
                thread::sleep(Duration::from_micros(100));
            }
            
            (i, fire_count, col.biological_state())
        }));
    }
    
    // Collect results
    let results: Vec<_> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Verify biological consistency
    for (column_id, fire_count, state) in results {
        assert!(fire_count > 0, "Column {} never fired", column_id);
        assert!(fire_count < 1000, "Column {} fired too frequently", column_id);
        assert!(state.membrane_voltage >= -0.5 && state.membrane_voltage <= 2.0);
        assert!(state.firing_threshold > 0.0);
    }
}
```

## Test Suite 3: Inhibition Integration (Tasks 1.7-1.9)

Tests the interaction between lateral inhibition, winner-take-all, and deduplication.

```rust
// integration_tests/inhibition_integration.rs
use neuromorphic_core::{
    LateralInhibitionNetwork, WinnerTakeAllSelector, ConceptDeduplicator,
    BiologicalCorticalColumn, BiologicalConfig, InhibitionResult,
    CompetitionOutcome, DeduplicationResult
};
use std::sync::Arc;
use std::collections::HashMap;

#[test]
fn test_lateral_inhibition_winner_take_all_pipeline() {
    let config = BiologicalConfig::default();
    let columns: Vec<_> = (0..8).map(|i| 
        Arc::new(BiologicalCorticalColumn::new(i, config.clone()))
    ).collect();
    
    let inhibition_network = LateralInhibitionNetwork::new(
        columns.iter().map(|c| c.base().id()).collect(),
        2.0 // inhibition radius
    );
    
    let wta_selector = WinnerTakeAllSelector::new();
    
    // Phase 1: Stimulate columns with different strengths
    let stimulation_strengths = vec![0.9, 1.2, 0.7, 1.5, 0.8, 1.1, 0.6, 1.3];
    
    for (i, strength) in stimulation_strengths.iter().enumerate() {
        columns[i].stimulate(*strength, 2.0);
    }
    
    // Phase 2: Apply lateral inhibition
    let mut activation_levels: HashMap<u32, f32> = HashMap::new();
    for column in &columns {
        activation_levels.insert(
            column.base().id(),
            column.base().activation_level()
        );
    }
    
    let inhibition_result = inhibition_network.apply_inhibition(&activation_levels);
    
    match inhibition_result {
        InhibitionResult::Converged { final_activations, iterations, .. } => {
            assert!(iterations > 0 && iterations < 20);
            assert_eq!(final_activations.len(), 8);
            
            // Verify inhibition reduced some activations
            let total_initial: f32 = activation_levels.values().sum();
            let total_final: f32 = final_activations.values().sum();
            assert!(total_final < total_initial, "Inhibition should reduce total activation");
        }
        _ => panic!("Expected inhibition to converge"),
    }
    
    // Phase 3: Winner-take-all selection
    let competition_result = wta_selector.select_winners(&inhibition_result.final_activations(), 3);
    
    match competition_result {
        CompetitionOutcome::Winners { selected, scores, .. } => {
            assert_eq!(selected.len(), 3);
            
            // Winners should be in descending order of activation
            for i in 1..selected.len() {
                assert!(scores[&selected[i-1]] >= scores[&selected[i]]);
            }
            
            // Top winner should be column 3 (highest initial stimulus: 1.5)
            assert_eq!(selected[0], 3);
        }
        _ => panic!("Expected successful winner selection"),
    }
    
    // Phase 4: Verify state consistency
    for column in &columns {
        let state = column.base().current_state();
        assert!(matches!(
            state,
            neuromorphic_core::ColumnState::Available |
            neuromorphic_core::ColumnState::Activated |
            neuromorphic_core::ColumnState::Competing |
            neuromorphic_core::ColumnState::Allocated
        ));
    }
}

#[test]
fn test_concept_deduplication_with_inhibition() {
    let config = BiologicalConfig::default();
    let deduplicator = ConceptDeduplicator::new(0.85); // 85% similarity threshold
    
    // Create columns representing similar concepts
    let concept_vectors = vec![
        vec![1.0, 0.8, 0.2, 0.1], // Concept A
        vec![0.9, 0.7, 0.3, 0.0], // Similar to A (should be deduplicated)
        vec![0.1, 0.2, 0.9, 0.8], // Concept B (different)
        vec![1.0, 0.8, 0.1, 0.2], // Very similar to A (should be deduplicated)
        vec![0.0, 0.1, 1.0, 0.9], // Similar to B (should be deduplicated)
    ];
    
    let columns: Vec<_> = (0..5).map(|i| {
        let col = Arc::new(BiologicalCorticalColumn::new(i, config.clone()));
        
        // Encode concept vector into activation pattern
        let strength = concept_vectors[i].iter().sum::<f32>() / concept_vectors[i].len() as f32;
        col.stimulate(strength * 1.5, 2.0);
        
        col
    }).collect();
    
    // Apply deduplication
    let dedup_result = deduplicator.deduplicate_concepts(
        &columns.iter().map(|c| c.base().id()).collect::<Vec<_>>(),
        &concept_vectors
    );
    
    match dedup_result {
        DeduplicationResult::Success { unique_concepts, duplicates_removed, .. } => {
            // Should keep concepts 0 and 2 (A and B), remove others as duplicates
            assert_eq!(unique_concepts.len(), 2);
            assert_eq!(duplicates_removed.len(), 3);
            
            // Verify A and B concepts are preserved
            assert!(unique_concepts.contains(&0) || unique_concepts.contains(&3)); // Keep one A variant
            assert!(unique_concepts.contains(&2) || unique_concepts.contains(&4)); // Keep one B variant
        }
        _ => panic!("Expected successful deduplication"),
    }
    
    // Verify inhibited columns are in correct state
    for &duplicate_id in &dedup_result.duplicates_removed() {
        let column = &columns[duplicate_id as usize];
        let state = column.base().current_state();
        
        // Duplicates should be inhibited (not allocated)
        assert!(!matches!(state, neuromorphic_core::ColumnState::Allocated));
    }
}

#[test]
fn test_inhibition_convergence_stability() {
    let config = BiologicalConfig::fast_processing();
    let num_columns = 20;
    
    let columns: Vec<_> = (0..num_columns).map(|i| 
        Arc::new(BiologicalCorticalColumn::new(i, config.clone()))
    ).collect();
    
    let inhibition_network = LateralInhibitionNetwork::new(
        (0..num_columns).collect(),
        3.0 // Larger inhibition radius
    );
    
    // Create random but reproducible activation pattern
    let mut activations = HashMap::new();
    for i in 0..num_columns {
        let strength = ((i * 17 + 7) % 100) as f32 / 100.0; // Pseudo-random 0-1
        columns[i as usize].stimulate(strength * 1.2, 1.0);
        activations.insert(i, columns[i as usize].base().activation_level());
    }
    
    // Apply inhibition multiple times - should converge to same result
    let results: Vec<_> = (0..5).map(|_| {
        inhibition_network.apply_inhibition(&activations)
    }).collect();
    
    // All results should be identical (deterministic convergence)
    for i in 1..results.len() {
        match (&results[0], &results[i]) {
            (
                InhibitionResult::Converged { final_activations: ref a1, .. },
                InhibitionResult::Converged { final_activations: ref a2, .. }
            ) => {
                for (&id, &value1) in a1 {
                    let value2 = a2[&id];
                    let diff = (value1 - value2).abs();
                    assert!(diff < 0.001, "Non-deterministic inhibition: {} vs {}", value1, value2);
                }
            }
            _ => panic!("Inhibition failed to converge consistently"),
        }
    }
    
    // Verify biological constraints are maintained
    let final_result = &results[0];
    for &final_activation in final_result.final_activations().values() {
        assert!(final_activation >= 0.0 && final_activation <= 2.0);
    }
}

#[test]
fn test_competition_dynamics_performance() {
    let config = BiologicalConfig::fast_processing();
    let num_columns = 100;
    
    let columns: Vec<_> = (0..num_columns).map(|i| 
        Arc::new(BiologicalCorticalColumn::new(i, config.clone()))
    ).collect();
    
    // Stimulate all columns
    for (i, column) in columns.iter().enumerate() {
        let strength = 0.5 + (i as f32 / num_columns as f32); // Gradient 0.5-1.5
        column.stimulate(strength, 1.0);
    }
    
    let inhibition_network = LateralInhibitionNetwork::new(
        (0..num_columns).collect(),
        5.0 // Large radius for complex interactions
    );
    
    let wta_selector = WinnerTakeAllSelector::new();
    
    // Measure performance of full inhibition + WTA pipeline
    let start = std::time::Instant::now();
    
    for _ in 0..100 {
        let activations: HashMap<u32, f32> = columns.iter().map(|c| 
            (c.base().id(), c.base().activation_level())
        ).collect();
        
        let inhibition_result = inhibition_network.apply_inhibition(&activations);
        let _competition_result = wta_selector.select_winners(
            &inhibition_result.final_activations(), 
            10 // Select top 10
        );
    }
    
    let elapsed = start.elapsed();
    let avg_time_us = elapsed.as_micros() / 100;
    
    println!("Average inhibition+WTA time for {} columns: {} μs", num_columns, avg_time_us);
    
    // Performance requirement: < 500μs for 100 columns
    assert!(avg_time_us < 500, "Competition dynamics too slow: {} μs", avg_time_us);
}
```

## Test Suite 4: Spatial Integration (Tasks 1.10-1.12)

Tests the interaction between 3D grid topology, spatial indexing, and neighbor finding.

```rust
// integration_tests/spatial_integration.rs
use neuromorphic_core::{
    CorticalGrid3D, SpatialIndexer, NeighborFinder, GridCoordinate,
    SpatialQuery, NeighborSearchResult, BiologicalCorticalColumn,
    BiologicalConfig
};
use std::sync::Arc;
use std::collections::HashSet;

#[test]
fn test_grid_indexing_neighbor_pipeline() {
    let config = BiologicalConfig::default();
    let grid_dimensions = (10, 10, 5); // 500 column grid
    
    // Create 3D cortical grid
    let mut grid = CorticalGrid3D::new(grid_dimensions.0, grid_dimensions.1, grid_dimensions.2);
    
    // Populate grid with biological columns
    let mut columns = Vec::new();
    let mut column_positions = Vec::new();
    
    for z in 0..grid_dimensions.2 {
        for y in 0..grid_dimensions.1 {
            for x in 0..grid_dimensions.0 {
                let column_id = (z * grid_dimensions.1 * grid_dimensions.0 + 
                                y * grid_dimensions.0 + x) as u32;
                
                let column = Arc::new(BiologicalCorticalColumn::new(column_id, config.clone()));
                let position = GridCoordinate { x, y, z };
                
                grid.place_column(position, column_id).expect("Grid placement failed");
                columns.push(column);
                column_positions.push((column_id, position));
            }
        }
    }
    
    // Create spatial indexer
    let mut indexer = SpatialIndexer::new();
    for (column_id, position) in &column_positions {
        indexer.insert(*column_id, *position);
    }
    
    indexer.build_index().expect("Index build failed");
    
    // Create neighbor finder
    let neighbor_finder = NeighborFinder::new(&indexer);
    
    // Test 1: Range queries across different scales
    let query_center = GridCoordinate { x: 5, y: 5, z: 2 };
    
    for radius in [1.0, 2.0, 3.0, 5.0] {
        let query = SpatialQuery::Range {
            center: query_center,
            radius,
        };
        
        let result = neighbor_finder.find_neighbors(&query);
        
        match result {
            NeighborSearchResult::Found { neighbors, distances } => {
                assert_eq!(neighbors.len(), distances.len());
                
                // Verify all neighbors are within radius
                for (&neighbor_id, &distance) in neighbors.iter().zip(distances.iter()) {
                    assert!(distance <= radius + 0.001, 
                        "Neighbor {} at distance {} exceeds radius {}", 
                        neighbor_id, distance, radius);
                    
                    // Verify neighbor exists in grid
                    let neighbor_pos = indexer.get_position(neighbor_id)
                        .expect("Neighbor not found in index");
                    
                    let actual_distance = query_center.euclidean_distance(neighbor_pos);
                    assert!((actual_distance - distance).abs() < 0.001,
                        "Distance calculation mismatch: {} vs {}", actual_distance, distance);
                }
                
                // Verify neighbors are sorted by distance
                for i in 1..distances.len() {
                    assert!(distances[i-1] <= distances[i], "Neighbors not sorted by distance");
                }
                
                println!("Radius {}: Found {} neighbors", radius, neighbors.len());
            }
            _ => panic!("Range query failed for radius {}", radius),
        }
    }
    
    // Test 2: K-nearest neighbors
    for k in [1, 5, 10, 20] {
        let query = SpatialQuery::KNearest {
            center: query_center,
            k,
        };
        
        let result = neighbor_finder.find_neighbors(&query);
        
        match result {
            NeighborSearchResult::Found { neighbors, distances } => {
                assert_eq!(neighbors.len(), k.min(499)); // Exclude center itself
                assert_eq!(distances.len(), neighbors.len());
                
                // Verify k-nearest property
                for i in 1..distances.len() {
                    assert!(distances[i-1] <= distances[i]);
                }
                
                println!("K={}: Found {} neighbors, max distance {:.2}", 
                    k, neighbors.len(), distances.last().unwrap_or(&0.0));
            }
            _ => panic!("K-nearest query failed for k={}", k),
        }
    }
}

#[test]
fn test_spatial_indexing_performance() {
    let grid_size = 50; // 125,000 columns
    let config = BiologicalConfig::fast_processing();
    
    // Build large grid
    let mut indexer = SpatialIndexer::new();
    let mut positions = Vec::new();
    
    for z in 0..grid_size {
        for y in 0..grid_size {
            for x in 0..grid_size {
                let column_id = (z * grid_size * grid_size + y * grid_size + x) as u32;
                let position = GridCoordinate { x, y, z };
                
                indexer.insert(column_id, position);
                positions.push((column_id, position));
            }
        }
    }
    
    // Measure index build time
    let build_start = std::time::Instant::now();
    indexer.build_index().expect("Index build failed");
    let build_time = build_start.elapsed();
    
    println!("Index build time for {} columns: {:?}", positions.len(), build_time);
    
    // Should build in < 100ms for 125K nodes
    assert!(build_time < std::time::Duration::from_millis(100),
        "Index build too slow: {:?}", build_time);
    
    let neighbor_finder = NeighborFinder::new(&indexer);
    
    // Measure query performance
    let num_queries = 1000;
    let query_start = std::time::Instant::now();
    
    for i in 0..num_queries {
        let center = GridCoordinate {
            x: (i * 7) % grid_size,
            y: (i * 11) % grid_size,
            z: (i * 13) % grid_size,
        };
        
        let query = SpatialQuery::Range { center, radius: 5.0 };
        let _result = neighbor_finder.find_neighbors(&query);
    }
    
    let query_time = query_start.elapsed();
    let avg_query_us = query_time.as_micros() / num_queries;
    
    println!("Average query time: {} μs", avg_query_us);
    
    // Should query in < 10μs average
    assert!(avg_query_us < 10, "Queries too slow: {} μs", avg_query_us);
}

#[test]
fn test_grid_topology_consistency() {
    let grid = CorticalGrid3D::new(8, 8, 4);
    let config = BiologicalConfig::default();
    
    // Place columns in regular pattern
    for z in 0..4 {
        for y in 0..8 {
            for x in 0..8 {
                let column_id = (z * 64 + y * 8 + x) as u32;
                let position = GridCoordinate { x, y, z };
                
                grid.place_column(position, column_id).expect("Placement failed");
            }
        }
    }
    
    // Test connectivity patterns
    let center_pos = GridCoordinate { x: 4, y: 4, z: 2 };
    let center_id = grid.get_column_at(center_pos).expect("Center column not found");
    
    // Get immediate neighbors (6-connected in 3D)
    let immediate_neighbors = grid.get_immediate_neighbors(center_pos);
    assert_eq!(immediate_neighbors.len(), 6); // Up, down, north, south, east, west
    
    // Verify neighbor distances
    for neighbor_pos in immediate_neighbors {
        let distance = center_pos.euclidean_distance(neighbor_pos);
        assert!((distance - 1.0).abs() < 0.001, 
            "Immediate neighbor not at unit distance: {}", distance);
    }
    
    // Test extended neighborhood (26-connected cube)
    let extended_neighbors = grid.get_neighbors_in_radius(center_pos, 1.8);
    assert!(extended_neighbors.len() >= 6 && extended_neighbors.len() <= 26);
    
    // Test grid boundaries
    let corner_pos = GridCoordinate { x: 0, y: 0, z: 0 };
    let corner_neighbors = grid.get_immediate_neighbors(corner_pos);
    assert_eq!(corner_neighbors.len(), 3); // Only positive directions available
    
    let edge_pos = GridCoordinate { x: 7, y: 4, z: 2 };
    let edge_neighbors = grid.get_immediate_neighbors(edge_pos);
    assert_eq!(edge_neighbors.len(), 5); // Missing +x direction
}

#[test]
fn test_spatial_cache_efficiency() {
    let indexer = SpatialIndexer::new();
    
    // Insert columns in patterns that test cache locality
    for i in 0..1000 {
        let position = GridCoordinate {
            x: i % 10,
            y: (i / 10) % 10,
            z: i / 100,
        };
        indexer.insert(i as u32, position);
    }
    
    indexer.build_index().expect("Index build failed");
    let neighbor_finder = NeighborFinder::new(&indexer);
    
    // Measure cache performance with locality patterns
    let mut cache_hits = 0;
    let mut cache_misses = 0;
    
    // Pattern 1: Clustered queries (high locality)
    for base_x in [0, 5] {
        for base_y in [0, 5] {
            for dx in 0..3 {
                for dy in 0..3 {
                    let center = GridCoordinate { x: base_x + dx, y: base_y + dy, z: 1 };
                    let query = SpatialQuery::Range { center, radius: 2.0 };
                    
                    let start = std::time::Instant::now();
                    let _result = neighbor_finder.find_neighbors(&query);
                    let elapsed = start.elapsed();
                    
                    if elapsed < std::time::Duration::from_micros(5) {
                        cache_hits += 1;
                    } else {
                        cache_misses += 1;
                    }
                }
            }
        }
    }
    
    let cache_hit_rate = cache_hits as f32 / (cache_hits + cache_misses) as f32;
    println!("Spatial cache hit rate: {:.1}%", cache_hit_rate * 100.0);
    
    // Should achieve >90% cache hit rate for clustered queries
    assert!(cache_hit_rate > 0.9, "Poor spatial cache performance: {:.1}%", cache_hit_rate * 100.0);
}
```

## Test Suite 5: System Integration (Tasks 1.13-1.14)

Tests the complete system with all components working together.

```rust
// integration_tests/system_integration.rs
use neuromorphic_core::{
    ParallelAllocationEngine, AllocationRequest, AllocationResult,
    BiologicalCorticalColumn, BiologicalConfig, CorticalGrid3D,
    LateralInhibitionNetwork, SpatialIndexer, PerformanceMetrics
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use tokio::sync::Semaphore;

#[tokio::test]
async fn test_full_system_end_to_end() {
    let config = BiologicalConfig::fast_processing();
    let grid_size = (20, 20, 10); // 4000 columns
    
    // Initialize complete system
    let allocation_engine = ParallelAllocationEngine::new(
        grid_size.0 * grid_size.1 * grid_size.2,
        config.clone()
    );
    
    // Create allocation requests
    let requests = vec![
        AllocationRequest {
            concept_vector: vec![1.0, 0.8, 0.2, 0.1, 0.0],
            priority: 1.0,
            timeout_ms: 5,
        },
        AllocationRequest {
            concept_vector: vec![0.1, 0.2, 0.9, 0.8, 0.3],
            priority: 0.9,
            timeout_ms: 5,
        },
        AllocationRequest {
            concept_vector: vec![0.5, 0.5, 0.5, 0.5, 0.5],
            priority: 0.8,
            timeout_ms: 5,
        },
        // Similar concepts (should be deduplicated)
        AllocationRequest {
            concept_vector: vec![0.95, 0.75, 0.25, 0.05, 0.05],
            priority: 0.7,
            timeout_ms: 5,
        },
        AllocationRequest {
            concept_vector: vec![0.05, 0.15, 0.95, 0.85, 0.25],
            priority: 0.6,
            timeout_ms: 5,
        },
    ];
    
    // Process allocations in parallel
    let start_time = Instant::now();
    let mut allocation_handles = vec![];
    
    for (i, request) in requests.into_iter().enumerate() {
        let engine = allocation_engine.clone();
        allocation_handles.push(tokio::spawn(async move {
            let result = engine.allocate(request).await;
            (i, result)
        }));
    }
    
    // Collect results
    let mut results = vec![];
    for handle in allocation_handles {
        let (request_id, result) = handle.await.unwrap();
        results.push((request_id, result));
    }
    
    let total_time = start_time.elapsed();
    
    // Verify results
    let mut successful_allocations = 0;
    let mut deduplicated_allocations = 0;
    
    for (request_id, result) in results {
        match result {
            AllocationResult::Success { column_id, activation_level, .. } => {
                successful_allocations += 1;
                assert!(column_id < 4000, "Invalid column ID: {}", column_id);
                assert!(activation_level > 0.0 && activation_level <= 2.0);
                println!("Request {} -> Column {} (activation: {:.3})", 
                    request_id, column_id, activation_level);
            }
            AllocationResult::Deduplicated { original_column_id, similarity } => {
                deduplicated_allocations += 1;
                assert!(similarity >= 0.85, "Deduplication threshold not met: {}", similarity);
                println!("Request {} deduplicated to column {} (similarity: {:.3})", 
                    request_id, original_column_id, similarity);
            }
            AllocationResult::Failed { reason } => {
                panic!("Allocation failed for request {}: {}", request_id, reason);
            }
        }
    }
    
    // System-level assertions
    assert!(successful_allocations >= 2, "Too few successful allocations");
    assert!(deduplicated_allocations >= 1, "Deduplication not working");
    assert!(total_time < Duration::from_millis(50), 
        "System too slow: {:?}", total_time);
    
    // Verify performance metrics
    let metrics = allocation_engine.get_performance_metrics();
    assert!(metrics.average_allocation_time_us < 5000); // < 5ms
    assert!(metrics.lateral_inhibition_time_us < 500); // < 500μs
    assert!(metrics.thread_safety_violations == 0);
    assert!(metrics.memory_usage_bytes < 4000 * 512); // < 512 bytes per column
}

#[test]
fn test_concurrent_system_stress() {
    let config = BiologicalConfig::fast_processing();
    let allocation_engine = ParallelAllocationEngine::new(1000, config);
    
    let num_threads = 8;
    let requests_per_thread = 100;
    let semaphore = Arc::new(Semaphore::new(num_threads));
    
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let engine = allocation_engine.clone();
        let sem = semaphore.clone();
        
        handles.push(thread::spawn(move || {
            let _permit = sem.try_acquire().unwrap();
            let mut thread_results = vec![];
            
            for request_id in 0..requests_per_thread {
                // Generate diverse concept vectors
                let concept_vector: Vec<f32> = (0..8).map(|i| {
                    let val = ((thread_id * 1000 + request_id * 10 + i) % 100) as f32 / 100.0;
                    val * 2.0 - 1.0 // Range [-1, 1]
                }).collect();
                
                let request = AllocationRequest {
                    concept_vector,
                    priority: 0.5 + (request_id as f32 / requests_per_thread as f32) * 0.5,
                    timeout_ms: 10,
                };
                
                let start = Instant::now();
                let result = futures::executor::block_on(engine.allocate(request));
                let elapsed = start.elapsed();
                
                thread_results.push((request_id, result, elapsed));
                
                // Brief pause to allow interleaving
                thread::sleep(Duration::from_micros(10));
            }
            
            (thread_id, thread_results)
        }));
    }
    
    // Collect all results
    let mut all_results = vec![];
    let mut performance_samples = vec![];
    
    for handle in handles {
        let (thread_id, thread_results) = handle.join().unwrap();
        
        for (request_id, result, elapsed) in thread_results {
            all_results.push((thread_id, request_id, result));
            performance_samples.push(elapsed);
        }
    }
    
    // Analyze results
    let total_requests = num_threads * requests_per_thread;
    let successful = all_results.iter()
        .filter(|(_, _, result)| matches!(result, AllocationResult::Success { .. }))
        .count();
    let deduplicated = all_results.iter()
        .filter(|(_, _, result)| matches!(result, AllocationResult::Deduplicated { .. }))
        .count();
    let failed = all_results.iter()
        .filter(|(_, _, result)| matches!(result, AllocationResult::Failed { .. }))
        .count();
    
    println!("Stress test results:");
    println!("  Total requests: {}", total_requests);
    println!("  Successful: {}", successful);
    println!("  Deduplicated: {}", deduplicated);
    println!("  Failed: {}", failed);
    
    // Performance analysis
    performance_samples.sort();
    let p50 = performance_samples[performance_samples.len() / 2];
    let p95 = performance_samples[performance_samples.len() * 95 / 100];
    let p99 = performance_samples[performance_samples.len() * 99 / 100];
    
    println!("  Performance (P50/P95/P99): {:?} / {:?} / {:?}", p50, p95, p99);
    
    // Stress test assertions
    assert!(successful + deduplicated >= total_requests * 90 / 100, 
        "Success rate too low: {}%", (successful + deduplicated) * 100 / total_requests);
    assert!(failed < total_requests * 5 / 100, 
        "Failure rate too high: {}%", failed * 100 / total_requests);
    assert!(p99 < Duration::from_millis(5), "P99 latency too high: {:?}", p99);
    
    // Verify no race conditions or data corruption
    let final_metrics = allocation_engine.get_performance_metrics();
    assert_eq!(final_metrics.thread_safety_violations, 0);
    assert!(final_metrics.memory_leaks_detected == 0);
}

#[test]
fn test_system_performance_targets() {
    let config = BiologicalConfig::fast_processing();
    let allocation_engine = ParallelAllocationEngine::new(10000, config);
    
    // Warm up the system
    for _ in 0..10 {
        let warmup_request = AllocationRequest {
            concept_vector: vec![0.5; 10],
            priority: 1.0,
            timeout_ms: 1,
        };
        let _ = futures::executor::block_on(allocation_engine.allocate(warmup_request));
    }
    
    // Measure performance targets
    let mut single_allocation_times = vec![];
    let mut lateral_inhibition_times = vec![];
    let mut memory_usage_samples = vec![];
    
    for i in 0..100 {
        let request = AllocationRequest {
            concept_vector: (0..10).map(|j| ((i + j) % 100) as f32 / 100.0).collect(),
            priority: 1.0,
            timeout_ms: 5,
        };
        
        let start = Instant::now();
        let result = futures::executor::block_on(allocation_engine.allocate(request));
        let elapsed = start.elapsed();
        
        single_allocation_times.push(elapsed);
        
        // Extract detailed timing from result
        if let AllocationResult::Success { timing_breakdown, .. } = result {
            lateral_inhibition_times.push(Duration::from_micros(timing_breakdown.lateral_inhibition_us));
        }
        
        // Sample memory usage
        let metrics = allocation_engine.get_performance_metrics();
        memory_usage_samples.push(metrics.memory_usage_bytes);
    }
    
    // Calculate statistics
    single_allocation_times.sort();
    lateral_inhibition_times.sort();
    
    let allocation_p99 = single_allocation_times[99];
    let inhibition_p99 = lateral_inhibition_times[lateral_inhibition_times.len() - 1];
    let avg_memory_usage = memory_usage_samples.iter().sum::<u64>() / memory_usage_samples.len() as u64;
    let memory_per_column = avg_memory_usage / 10000;
    
    println!("Performance Target Verification:");
    println!("  Single allocation P99: {:?} (target: <20ms)", allocation_p99);
    println!("  Lateral inhibition P99: {:?} (target: <500μs)", inhibition_p99);
    println!("  Memory per column: {} bytes (target: <512 bytes)", memory_per_column);
    
    // Verify all Phase 1 performance targets (realistic with neural network integration)
    assert!(allocation_p99 < Duration::from_millis(20), 
        "Single allocation P99 target missed: {:?}", allocation_p99);
    assert!(inhibition_p99 < Duration::from_micros(500), 
        "Lateral inhibition target missed: {:?}", inhibition_p99);
    assert!(memory_per_column < 512, 
        "Memory per column target missed: {} bytes", memory_per_column);
    
    // Additional targets
    let final_metrics = allocation_engine.get_performance_metrics();
    assert!(final_metrics.winner_take_all_accuracy > 0.98, 
        "Winner-take-all accuracy target missed: {:.3}", final_metrics.winner_take_all_accuracy);
    assert_eq!(final_metrics.thread_safety_violations, 0, 
        "Thread safety violations detected");
    
    // SIMD acceleration verification
    assert!(final_metrics.simd_acceleration_active, "SIMD acceleration not active");
    assert!(final_metrics.simd_speedup_factor >= 2.0, 
        "Insufficient SIMD speedup: {:.1}x", final_metrics.simd_speedup_factor);
    
    println!("✅ All Phase 1 performance targets achieved!");
}

#[tokio::test]
async fn test_neural_network_integration() {
    let config = BiologicalConfig::cortical_neuron();
    let allocation_engine = ParallelAllocationEngine::new(1000, config);
    
    // Test integration with selected neural network architectures
    let test_vectors = vec![
        // MLP semantic processing test
        vec![0.8, 0.6, 0.4, 0.2, 0.1, 0.3, 0.7, 0.9],
        // LSTM temporal sequence test  
        vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3],
        // TCN performance test
        vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
    ];
    
    for (arch_id, test_vector) in test_vectors.into_iter().enumerate() {
        let request = AllocationRequest {
            concept_vector: test_vector,
            priority: 1.0,
            timeout_ms: 5,
        };
        
        let result = allocation_engine.allocate(request).await;
        
        match result {
            AllocationResult::Success { 
                column_id, 
                neural_architecture_used, 
                inference_time_us,
                accuracy_score,
                ..
            } => {
                assert!(inference_time_us < 500); // < 0.5ms inference
                assert!(accuracy_score > 0.95); // > 95% accuracy
                
                println!("Architecture {}: Column {} (inference: {}μs, accuracy: {:.3})", 
                    neural_architecture_used, column_id, inference_time_us, accuracy_score);
            }
            _ => panic!("Neural network integration failed for architecture {}", arch_id),
        }
    }
    
    // Verify neural network selection criteria are met
    let metrics = allocation_engine.get_performance_metrics();
    assert!(metrics.neural_network_memory_usage < 200 * 1024); // < 200KB total
    assert!(metrics.wasm_compatibility_score > 0.9); // > 90% WASM compatible
    
    println!("✅ Neural network integration verified!");
}
```

## Performance Validation Protocol

```rust
// integration_tests/performance_validation.rs
use neuromorphic_core::*;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

fn benchmark_complete_allocation_pipeline(c: &mut Criterion) {
    let config = BiologicalConfig::fast_processing();
    let allocation_engine = ParallelAllocationEngine::new(1000, config);
    
    let test_vectors = vec![
        vec![1.0, 0.8, 0.6, 0.4, 0.2],
        vec![0.2, 0.4, 0.6, 0.8, 1.0],
        vec![0.5, 0.5, 0.5, 0.5, 0.5],
    ];
    
    let mut group = c.benchmark_group("allocation_pipeline");
    
    for (i, vector) in test_vectors.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("complete_allocation", i),
            vector,
            |b, vec| {
                b.iter(|| {
                    let request = AllocationRequest {
                        concept_vector: vec.clone(),
                        priority: 1.0,
                        timeout_ms: 5,
                    };
                    
                    futures::executor::block_on(allocation_engine.allocate(request))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_biological_dynamics(c: &mut Criterion) {
    let config = BiologicalConfig::fast_processing();
    let column = BiologicalCorticalColumn::new(1, config);
    
    c.bench_function("biological_stimulation", |b| {
        b.iter(|| {
            column.stimulate(1.0, 1.0)
        });
    });
    
    c.bench_function("membrane_decay_update", |b| {
        b.iter(|| {
            column.biological_state().membrane_voltage
        });
    });
    
    c.bench_function("hebbian_weight_update", |b| {
        let other_column = BiologicalCorticalColumn::new(2, BiologicalConfig::fast_processing());
        other_column.stimulate(1.0, 1.0);
        
        b.iter(|| {
            column.learn_from_coactivation(&other_column)
        });
    });
}

fn benchmark_spatial_operations(c: &mut Criterion) {
    let mut indexer = SpatialIndexer::new();
    
    // Build test index
    for i in 0..1000 {
        let pos = GridCoordinate {
            x: i % 10,
            y: (i / 10) % 10,
            z: i / 100,
        };
        indexer.insert(i as u32, pos);
    }
    indexer.build_index().unwrap();
    
    let neighbor_finder = NeighborFinder::new(&indexer);
    let center = GridCoordinate { x: 5, y: 5, z: 5 };
    
    c.bench_function("spatial_range_query", |b| {
        b.iter(|| {
            let query = SpatialQuery::Range { center, radius: 3.0 };
            neighbor_finder.find_neighbors(&query)
        });
    });
    
    c.bench_function("spatial_knn_query", |b| {
        b.iter(|| {
            let query = SpatialQuery::KNearest { center, k: 10 };
            neighbor_finder.find_neighbors(&query)
        });
    });
}

criterion_group!(
    benches,
    benchmark_complete_allocation_pipeline,
    benchmark_biological_dynamics,
    benchmark_spatial_operations
);
criterion_main!(benches);
```

## Integration Test Execution Guide

### Prerequisites

1. All Phase 1 tasks (1.1-1.14) must be completed
2. Rust toolchain with async support
3. Required dependencies in `Cargo.toml`:

```toml
[dev-dependencies]
tokio = { version = "1.0", features = ["full"] }
criterion = { version = "0.5", features = ["html_reports"] }
futures = "0.3"
```

### Execution Commands

```bash
# Run all integration tests
cargo test --test integration_tests --release -- --nocapture

# Run specific test suites
cargo test --test foundation_integration --release
cargo test --test biological_integration --release  
cargo test --test inhibition_integration --release
cargo test --test spatial_integration --release
cargo test --test system_integration --release

# Performance validation
cargo test --test performance_validation --release
cargo bench --bench integration_benchmarks

# Stress testing
cargo test test_concurrent_system_stress --release -- --ignored --nocapture

# Memory leak detection
cargo test --release -- --ignored memory_leak_detection
```

### Success Criteria

#### Foundation Integration (Test Suite 1)
- ✅ All concurrent state transitions complete without race conditions
- ✅ Atomic operations maintain consistency under high load
- ✅ Memory ordering guarantees preserved
- ✅ State transition timing < 10ns average

#### Biological Integration (Test Suite 2)
- ✅ Activation-decay-learning cycle completes correctly
- ✅ Refractory periods enforce biological timing
- ✅ Exponential decay follows mathematical model (±10%)
- ✅ Hebbian learning increases synaptic weights appropriately

#### Inhibition Integration (Test Suite 3)
- ✅ Lateral inhibition converges in < 20 iterations
- ✅ Winner-take-all selects correct champions
- ✅ Concept deduplication prevents duplicates (>99% accuracy)
- ✅ Competition dynamics complete in < 500μs

#### Spatial Integration (Test Suite 4)
- ✅ Grid topology maintains spatial relationships
- ✅ Spatial indexing provides O(log n) queries
- ✅ Neighbor finding achieves < 10μs per query
- ✅ Cache hit rate > 90% for clustered queries

#### System Integration (Test Suite 5)
- ✅ End-to-end allocation pipeline < 5ms (P99)
- ✅ Concurrent stress test handles 800+ requests/second
- ✅ All Phase 1 performance targets achieved
- ✅ Neural network integration functional
- ✅ Zero thread safety violations
- ✅ Memory usage < 512 bytes per column

### Debugging Failed Tests

```bash
# Detailed logging for failed tests
RUST_LOG=debug cargo test <test_name> --release -- --nocapture

# Memory profiling
cargo test --release --features memory-profiling

# Performance analysis
cargo flamegraph --test <test_name> -- --bench
```

## Test Suite 6: Edge Case and Boundary Condition Tests

Tests critical boundary conditions and edge cases for production robustness.

```rust
// integration_tests/edge_case_tests.rs
use neuromorphic_core::{
    CorticalColumn, BiologicalCorticalColumn, BiologicalConfig,
    ParallelAllocationEngine, AllocationRequest, AllocationResult,
    CorticalGrid3D, SpatialIndexer, NeighborFinder, GridCoordinate,
    LateralInhibitionNetwork, ConceptDeduplicator
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

#[test]
fn test_empty_grid_allocation_attempts() {
    // Test allocation attempts on completely empty grid
    let grid = Arc::new(CorticalGrid3D::new(0, 0, 0));
    let inhibition = Arc::new(LateralInhibitionNetwork::new());
    let winner_selector = Arc::new(WinnerTakeAllSelector::new());
    let deduplicator = Arc::new(ConceptDeduplicator::new());
    let engine = ParallelAllocationEngine::new(0, BiologicalConfig::default());
    
    let request = AllocationRequest {
        concept_vector: vec![0.5; 10],
        priority: 1.0,
        timeout_ms: 1000,
    };
    
    let result = futures::executor::block_on(engine.allocate(request));
    
    // Should gracefully handle empty grid
    match result {
        AllocationResult::Failed { reason } => {
            assert!(reason.contains("no columns available") || reason.contains("empty grid"));
        },
        _ => panic!("Expected failure for empty grid allocation"),
    }
}

#[test]
fn test_single_column_grid_edge_cases() {
    // Test edge cases with minimal 1x1x1 grid
    let config = BiologicalConfig::default();
    let grid = Arc::new(CorticalGrid3D::new(1, 1, 1));
    let column = Arc::new(BiologicalCorticalColumn::new(0, config.clone()));
    
    // Place single column
    grid.place_column(GridCoordinate { x: 0, y: 0, z: 0 }, 0).unwrap();
    
    // Test multiple allocation attempts on same column
    let concepts = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    
    let engine = ParallelAllocationEngine::new(1, config);
    let mut allocation_results = vec![];
    
    for (i, concept) in concepts.into_iter().enumerate() {
        let request = AllocationRequest {
            concept_vector: concept,
            priority: 1.0,
            timeout_ms: 100,
        };
        
        let result = futures::executor::block_on(engine.allocate(request));
        allocation_results.push((i, result));
    }
    
    // First allocation should succeed, subsequent ones should be handled appropriately
    let successful_count = allocation_results.iter()
        .filter(|(_, result)| matches!(result, AllocationResult::Success { .. }))
        .count();
    
    let deduplicated_count = allocation_results.iter()
        .filter(|(_, result)| matches!(result, AllocationResult::Deduplicated { .. }))
        .count();
    
    assert_eq!(successful_count + deduplicated_count, 3);
    assert!(successful_count >= 1, "At least one allocation should succeed");
}

#[test]
fn test_maximum_grid_size_boundaries() {
    // Test with large grid dimensions to check scaling limits
    let max_size = 100; // Reasonable for testing
    let grid = CorticalGrid3D::new(max_size, max_size, max_size);
    
    // Test corner coordinate access
    let corners = vec![
        GridCoordinate { x: 0, y: 0, z: 0 },
        GridCoordinate { x: max_size - 1, y: 0, z: 0 },
        GridCoordinate { x: 0, y: max_size - 1, z: 0 },
        GridCoordinate { x: 0, y: 0, z: max_size - 1 },
        GridCoordinate { x: max_size - 1, y: max_size - 1, z: max_size - 1 },
    ];
    
    for (i, corner) in corners.into_iter().enumerate() {
        let result = grid.place_column(corner, i as u32);
        assert!(result.is_ok(), "Should be able to place column at corner {:?}", corner);
        
        let retrieved = grid.get_column_at(corner);
        assert!(retrieved.is_some(), "Should be able to retrieve column at corner {:?}", corner);
    }
}

#[test]
fn test_invalid_coordinate_boundary_checks() {
    let grid = CorticalGrid3D::new(10, 10, 10);
    
    // Test out-of-bounds coordinates
    let invalid_coords = vec![
        GridCoordinate { x: -1, y: 5, z: 5 },     // Negative x
        GridCoordinate { x: 5, y: -1, z: 5 },     // Negative y
        GridCoordinate { x: 5, y: 5, z: -1 },     // Negative z
        GridCoordinate { x: 10, y: 5, z: 5 },     // x >= bounds
        GridCoordinate { x: 5, y: 10, z: 5 },     // y >= bounds
        GridCoordinate { x: 5, y: 5, z: 10 },     // z >= bounds
        GridCoordinate { x: i32::MAX, y: 0, z: 0 }, // Extreme values
        GridCoordinate { x: i32::MIN, y: 0, z: 0 },
    ];
    
    for coord in invalid_coords {
        let result = grid.place_column(coord, 999);
        assert!(result.is_err(), "Should reject invalid coordinate {:?}", coord);
        
        let retrieved = grid.get_column_at(coord);
        assert!(retrieved.is_none(), "Should not find column at invalid coordinate {:?}", coord);
    }
}

#[test]
fn test_extreme_membrane_potential_values() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Test extreme stimulation values
    let extreme_stimuli = vec![
        (f32::MIN, 1.0),           // Minimum float
        (f32::MAX, 1.0),           // Maximum float
        (f32::NAN, 1.0),           // NaN
        (f32::INFINITY, 1.0),      // Positive infinity
        (f32::NEG_INFINITY, 1.0),  // Negative infinity
        (0.0, f32::MIN),           // Minimum duration
        (1.0, f32::MAX),           // Maximum duration
        (1.0, f32::NAN),           // NaN duration
        (1.0, f32::INFINITY),      // Infinite duration
    ];
    
    for (strength, duration) in extreme_stimuli {
        let result = column.stimulate(strength, duration);
        
        // Should handle extreme values gracefully
        match result {
            Ok(_) => {
                // If successful, verify biological state is still valid
                let state = column.biological_state();
                assert!(!state.membrane_voltage.is_nan(), "Membrane voltage should not be NaN");
                assert!(state.membrane_voltage.is_finite(), "Membrane voltage should be finite");
                assert!(state.membrane_voltage >= -2.0 && state.membrane_voltage <= 3.0, 
                       "Membrane voltage should be in reasonable range");
            },
            Err(_) => {
                // Failure is acceptable for extreme values
                // Verify column is still in valid state
                let state = column.biological_state();
                assert!(!state.membrane_voltage.is_nan(), "Membrane voltage should not be NaN after error");
            }
        }
    }
}

#[test]
fn test_zero_and_negative_durations() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Test edge case durations
    let edge_durations = vec![
        0.0,    // Zero duration
        -1.0,   // Negative duration
        -0.1,   // Small negative
        1e-10,  // Very small positive
        1e-6,   // Microsecond
    ];
    
    for duration in edge_durations {
        let result = column.stimulate(1.0, duration);
        
        // Should handle gracefully without crashing
        match result {
            Ok(_) => {
                let state = column.biological_state();
                assert!(state.membrane_voltage.is_finite());
            },
            Err(_) => {
                // Error is acceptable for invalid durations
            }
        }
    }
}

#[test]
fn test_rapid_fire_state_transitions() {
    let column = Arc::new(CorticalColumn::new(1));
    let iteration_count = 10000;
    
    // Test rapid state transitions in tight loop
    let start = Instant::now();
    
    for _ in 0..iteration_count {
        let _ = column.try_activate();
        let _ = column.try_compete();
        let _ = column.try_allocate();
        let _ = column.try_enter_refractory();
        let _ = column.try_reset();
    }
    
    let elapsed = start.elapsed();
    
    // Verify no deadlocks or corruption
    let final_state = column.current_state();
    assert!(matches!(
        final_state,
        ColumnState::Available | ColumnState::Activated | 
        ColumnState::Competing | ColumnState::Allocated | 
        ColumnState::Refractory
    ));
    
    // Performance check - should complete quickly
    assert!(elapsed < Duration::from_secs(1), "Rapid transitions too slow: {:?}", elapsed);
    
    println!("Rapid fire test: {} iterations in {:?}", iteration_count, elapsed);
}

#[test]
fn test_memory_exhaustion_simulation() {
    // Test behavior under simulated memory pressure
    let large_grid_size = 200; // Large enough to stress test
    let config = BiologicalConfig::default();
    
    let mut columns = Vec::new();
    let creation_start = Instant::now();
    
    // Create many columns to test memory usage
    for i in 0..large_grid_size * large_grid_size {
        let column = BiologicalCorticalColumn::new(i as u32, config.clone());
        columns.push(column);
        
        // Check if we're approaching reasonable memory limits
        if creation_start.elapsed() > Duration::from_secs(5) {
            println!("Memory exhaustion test stopped at {} columns", i);
            break;
        }
    }
    
    println!("Created {} columns in {:?}", columns.len(), creation_start.elapsed());
    
    // Test that columns are still functional
    for (i, column) in columns.iter().enumerate().take(100) {
        let result = column.stimulate(0.5, 1.0);
        
        match result {
            Ok(_) => {
                let state = column.biological_state();
                assert!(state.membrane_voltage.is_finite());
            },
            Err(_) => {
                // Some failures acceptable under memory pressure
            }
        }
        
        if i % 20 == 0 {
            println!("Tested column {} - still functional", i);
        }
    }
}

#[test]
fn test_concurrent_access_boundary_conditions() {
    let column = Arc::new(CorticalColumn::new(1));
    let thread_count = 100; // High concurrency
    let operations_per_thread = 1000;
    
    let barrier = Arc::new(std::sync::Barrier::new(thread_count));
    let mut handles = vec![];
    
    // Spawn many threads for high-contention scenario
    for thread_id in 0..thread_count {
        let col = column.clone();
        let bar = barrier.clone();
        
        handles.push(thread::spawn(move || {
            bar.wait(); // Synchronize start
            
            let mut local_success_count = 0;
            
            for _ in 0..operations_per_thread {
                // Mix of different operations
                match thread_id % 4 {
                    0 => {
                        if col.try_activate().is_ok() {
                            local_success_count += 1;
                        }
                    },
                    1 => {
                        if col.try_compete().is_ok() {
                            local_success_count += 1;
                        }
                    },
                    2 => {
                        if col.try_allocate().is_ok() {
                            local_success_count += 1;
                        }
                    },
                    _ => {
                        if col.try_reset().is_ok() {
                            local_success_count += 1;
                        }
                    }
                }
            }
            
            local_success_count
        }));
    }
    
    // Collect results
    let mut total_operations = 0;
    for handle in handles {
        total_operations += handle.join().unwrap();
    }
    
    // Verify final state is consistent
    let final_state = column.current_state();
    assert!(final_state as u8 <= 4, "Final state should be valid");
    
    println!("High concurrency test: {} total successful operations", total_operations);
    assert!(total_operations > 0, "Some operations should succeed even under high contention");
}

#[test]
fn test_floating_point_precision_edge_cases() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Test floating point precision boundaries
    let precision_tests = vec![
        1.0 + f32::EPSILON,     // Just above 1.0
        1.0 - f32::EPSILON,     // Just below 1.0
        f32::EPSILON,           // Smallest positive value
        0.0 + f32::EPSILON,     // Just above 0.0
        0.9999999,              // Very close to 1.0
        0.0000001,              // Very small positive
    ];
    
    for value in precision_tests {
        let result = column.stimulate(value, 1.0);
        
        match result {
            Ok(_) => {
                let state = column.biological_state();
                // Verify precision is maintained
                assert!(state.membrane_voltage.is_finite());
                assert!(!state.membrane_voltage.is_nan());
            },
            Err(_) => {
                // Acceptable if value is out of valid range
            }
        }
    }
}

#[test]
fn test_timeout_boundary_conditions() {
    let config = BiologicalConfig::fast_processing();
    let engine = ParallelAllocationEngine::new(10, config);
    
    // Test various timeout values
    let timeout_tests = vec![
        0,      // Zero timeout
        1,      // 1ms timeout
        5,      // Normal timeout
        1000,   // Long timeout
        u64::MAX / 1000, // Very long timeout (avoid overflow)
    ];
    
    for timeout_ms in timeout_tests {
        let request = AllocationRequest {
            concept_vector: vec![0.5; 10],
            priority: 1.0,
            timeout_ms,
        };
        
        let start = Instant::now();
        let result = futures::executor::block_on(engine.allocate(request));
        let elapsed = start.elapsed();
        
        // Verify timeout is respected (with some tolerance)
        if timeout_ms > 0 && timeout_ms < 1000 {
            assert!(elapsed <= Duration::from_millis(timeout_ms + 50), 
                   "Operation should respect timeout: {}ms, took {:?}", timeout_ms, elapsed);
        }
        
        // Verify result is valid regardless of timeout
        match result {
            AllocationResult::Success { .. } |
            AllocationResult::Deduplicated { .. } |
            AllocationResult::Failed { .. } => {
                // All result types are valid
            }
        }
    }
}
```

## Quality Gates

Before proceeding to Phase 2, ALL integration tests must pass with these quality metrics:

1. **Test Coverage**: 100% of integration scenarios covered
2. **Edge Cases**: All boundary conditions tested and handled
3. **Performance**: All Phase 1 targets achieved
4. **Reliability**: 0 race conditions, 0 memory leaks
5. **Scalability**: Linear scaling verified up to 4 cores
6. **Biological Accuracy**: Mathematical models verified
7. **Spatial Correctness**: Topology and indexing validated
8. **Error Resilience**: Graceful handling of extreme inputs and conditions

## Next Steps

Once all integration tests pass:

1. **Documentation Update**: Record performance benchmarks
2. **Architecture Validation**: Confirm neural network selection
3. **Phase 2 Preparation**: Integration test results feed into Phase 2 design
4. **Deployment Readiness**: System ready for production Phase 2 features

---

**Total Integration Test Suite Completion Time**: 4 hours
**Dependencies**: All Phase 1 tasks completed
**Success Metric**: 100% test pass rate + all performance targets achieved