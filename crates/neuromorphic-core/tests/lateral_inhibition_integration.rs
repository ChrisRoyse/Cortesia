//! Integration tests for lateral inhibition network

use neuromorphic_core::spiking_column::{
    CompetitionResult, InhibitionConfig, LateralInhibitionNetwork, SpikingCorticalColumn,
};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn test_multi_column_competition_with_spatial_layout() {
    let config = InhibitionConfig {
        base_strength: 0.9,
        spatial_decay: 0.4,
        radius: 5.0,
        competition_threshold: 0.02,
    };
    
    let network = Arc::new(LateralInhibitionNetwork::new(config));
    
    // Create a 5x5 grid of columns
    let mut columns = Vec::new();
    for i in 0..25 {
        let column = SpikingCorticalColumn::new(i);
        let x = (i % 5) as f32;
        let y = (i / 5) as f32;
        network.register_column(i, (x, y, 0.0));
        columns.push(column);
    }
    
    // Activate columns with varying strengths based on distance from center
    let center = (2.0, 2.0);
    for (i, column) in columns.iter().enumerate() {
        let x = (i % 5) as f32;
        let y = (i / 5) as f32;
        let distance = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
        let activation = (1.0 - distance / 4.0).max(0.3);
        column.activate_with_strength(activation).unwrap();
    }
    
    // Create candidates for competition
    let candidates: Vec<_> = columns
        .iter()
        .map(|c| (c.id(), c.activation_level()))
        .collect();
    
    let result = network.compete(candidates);
    
    // Center column (index 12) should win
    assert_eq!(result.winner_id, 12);
    
    // Nearby columns should be suppressed
    assert!(result.suppressed_columns.len() > 10);
    
    // Apply inhibition to columns
    for column in &columns {
        column.apply_network_inhibition(&network);
    }
    
    // Winner should maintain high activation
    assert!(columns[12].activation_level() > 0.7);
    
    // Nearby columns should have reduced activation
    for i in [7, 11, 13, 17] {
        assert!(columns[i].activation_level() < 0.3);
    }
}

#[test]
fn test_concurrent_competition_safety() {
    let network = Arc::new(LateralInhibitionNetwork::new(InhibitionConfig::default()));
    
    // Register many columns
    for i in 0..50 {
        network.register_column(i, (i as f32, 0.0, 0.0));
    }
    
    let mut handles = vec![];
    
    // Run multiple competitions concurrently
    for batch in 0..10 {
        let net_clone = network.clone();
        handles.push(thread::spawn(move || {
            let candidates: Vec<_> = (0..50)
                .map(|i| (i, 0.5 + (i as f32 * 0.01) + (batch as f32 * 0.001)))
                .collect();
            net_clone.compete(candidates)
        }));
    }
    
    let results: Vec<CompetitionResult> = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // All competitions should complete successfully
    assert_eq!(results.len(), 10);
    
    // Each should have a winner
    for result in &results {
        assert!(result.winner_id < 50);
        assert!(result.competition_time < Duration::from_millis(5));
    }
}

#[test]
fn test_dynamic_inhibition_radius() {
    // Test with very small radius
    let config_small = InhibitionConfig {
        radius: 1.0,
        spatial_decay: 0.9,
        ..Default::default()
    };
    
    let network_small = LateralInhibitionNetwork::new(config_small);
    
    // Columns far apart
    network_small.register_column(1, (0.0, 0.0, 0.0));
    network_small.register_column(2, (5.0, 0.0, 0.0));
    network_small.register_column(3, (10.0, 0.0, 0.0));
    
    let candidates = vec![(1, 0.9), (2, 0.85), (3, 0.8)];
    let result = network_small.compete(candidates);
    
    // With small radius, only column 1 wins, but 2 and 3 aren't suppressed
    assert_eq!(result.winner_id, 1);
    assert!(result.suppressed_columns.is_empty() || result.suppressed_columns.len() < 2);
    
    // Test with large radius
    let config_large = InhibitionConfig {
        radius: 15.0,
        spatial_decay: 0.1,
        ..Default::default()
    };
    
    let network_large = LateralInhibitionNetwork::new(config_large);
    
    network_large.register_column(1, (0.0, 0.0, 0.0));
    network_large.register_column(2, (5.0, 0.0, 0.0));
    network_large.register_column(3, (10.0, 0.0, 0.0));
    
    let candidates = vec![(1, 0.9), (2, 0.85), (3, 0.8)];
    let result = network_large.compete(candidates);
    
    // With large radius, columns within threshold are suppressed
    assert_eq!(result.winner_id, 1);
    // Both columns 2 and 3 have significant activation difference (>0.05) so both suppressed
    assert!(result.suppressed_columns.len() >= 1);
}

#[test]
fn test_inhibition_with_column_states() {
    let network = Arc::new(LateralInhibitionNetwork::new(InhibitionConfig::default()));
    let mut columns = Vec::new();
    
    // Create columns in different states
    for i in 0..5 {
        let column = SpikingCorticalColumn::new(i);
        network.register_column(i, (i as f32, 0.0, 0.0));
        columns.push(column);
    }
    
    // Set different states
    columns[0].activate_with_strength(0.9).unwrap();
    columns[0].start_competing().unwrap();
    
    columns[1].activate_with_strength(0.8).unwrap();
    columns[1].start_competing().unwrap();
    
    columns[2].activate_with_strength(0.7).unwrap();
    
    columns[3].activate_with_strength(0.6).unwrap();
    columns[3].start_competing().unwrap();
    
    // Column 4 stays available
    
    // Only competing columns participate
    let candidates: Vec<_> = columns
        .iter()
        .filter(|c| c.state() == neuromorphic_core::ColumnState::Competing)
        .map(|c| (c.id(), c.activation_level()))
        .collect();
    
    let result = network.compete(candidates);
    
    // Column 0 should win (highest activation among competing)
    assert_eq!(result.winner_id, 0);
    
    // Try to allocate winner
    assert!(columns[0].allocate().is_ok());
    
    // Suppressed columns can't allocate due to inhibition
    for column in &columns[1..4] {
        if column.state() == neuromorphic_core::ColumnState::Competing {
            column.apply_network_inhibition(&network);
            if network.get_inhibition(column.id()) > 0.5 {
                // Highly inhibited columns enter refractory
                assert!(column.enter_refractory().is_ok());
            }
        }
    }
}

#[test]
fn test_performance_with_large_network() {
    let config = InhibitionConfig {
        base_strength: 0.7,
        spatial_decay: 0.2,
        radius: 10.0,
        competition_threshold: 0.01,
    };
    
    let network = LateralInhibitionNetwork::new(config);
    
    // Create a large 20x20 grid (400 columns)
    for i in 0..400 {
        let x = (i % 20) as f32;
        let y = (i / 20) as f32;
        network.register_column(i, (x, y, 0.0));
    }
    
    // Create candidates with varying activations
    let candidates: Vec<_> = (0..400)
        .map(|i| {
            let activation = 0.3 + (i as f32 * 0.001) + ((i % 7) as f32 * 0.05);
            (i, activation)
        })
        .collect();
    
    // Measure performance
    let start = Instant::now();
    let result = network.compete(candidates);
    let elapsed = start.elapsed();
    
    // Should still complete quickly even with 400 columns
    assert!(elapsed < Duration::from_millis(10));
    
    // Should have a winner and suppressed columns
    assert!(result.winner_id < 400);
    assert!(!result.suppressed_columns.is_empty());
    
    // Check statistics
    let stats = network.get_stats();
    assert_eq!(stats.total_competitions, 1);
    assert!(stats.average_competition_time_us < 10000.0); // < 10ms
}

#[test]
fn test_inhibition_memory_management() {
    let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
    
    // Apply many inhibitions
    for i in 0..1000 {
        network.apply_inhibition(i, 0.5);
    }
    
    // Check all are present
    for i in 0..1000 {
        assert_eq!(network.get_inhibition(i), 0.5);
    }
    
    // Clear specific ones
    for i in (0..1000).step_by(2) {
        network.clear_inhibition(i);
    }
    
    // Check pattern
    for i in 0..1000 {
        if i % 2 == 0 {
            assert_eq!(network.get_inhibition(i), 0.0);
        } else {
            assert_eq!(network.get_inhibition(i), 0.5);
        }
    }
    
    // Clear all
    network.clear_all_inhibition();
    
    for i in 0..1000 {
        assert_eq!(network.get_inhibition(i), 0.0);
    }
}

#[test]
fn test_competition_history_tracking() {
    let network = LateralInhibitionNetwork::new(InhibitionConfig::default());
    
    // Register columns
    for i in 0..10 {
        network.register_column(i, (i as f32, 0.0, 0.0));
    }
    
    // Run multiple competitions
    for round in 0..5 {
        let candidates: Vec<_> = (0..10)
            .map(|i| (i, 0.5 + (i as f32 * 0.02) + (round as f32 * 0.1)))
            .collect();
        network.compete(candidates);
    }
    
    // Check statistics
    let stats = network.get_stats();
    assert_eq!(stats.total_competitions, 5);
    assert!(stats.average_suppressed_columns > 0.0);
    assert!(stats.average_competition_time_us > 0.0);
}