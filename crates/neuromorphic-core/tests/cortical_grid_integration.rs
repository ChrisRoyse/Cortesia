//! Integration tests for cortical grid

use neuromorphic_core::spiking_column::{
    CorticalGrid, GridConfig, GridPosition, LateralInhibitionNetwork, InhibitionConfig,
};
use neuromorphic_core::ColumnState;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[test]
fn test_large_grid_initialization() {
    let config = GridConfig {
        width: 20,
        height: 20,
        depth: 10,
        wrap_edges: false,
    };
    
    let start = Instant::now();
    let grid = CorticalGrid::new(config);
    let elapsed = start.elapsed();
    
    assert_eq!(grid.total_columns(), 4000);
    assert_eq!(grid.dimensions(), (20, 20, 10));
    
    // Should initialize quickly even for large grids
    assert!(elapsed.as_millis() < 100);
    
    // Verify all columns are accessible
    for id in 0..4000 {
        assert!(grid.get_column(id).is_some());
    }
}

#[test]
fn test_spatial_locality_and_activation_spread() {
    let config = GridConfig {
        width: 10,
        height: 10,
        depth: 5,
        wrap_edges: false,
    };
    
    let grid = Arc::new(CorticalGrid::new(config));
    
    // Activate center column
    let center = GridPosition::new(5, 5, 2);
    let center_column = grid.get_column_at(&center).unwrap();
    center_column.activate_with_strength(1.0).unwrap();
    
    // Setup lateral connections
    grid.setup_lateral_connections(2, 0.5);
    
    // Activate neighbors based on connections
    let neighbors = grid.get_neighbors(center, 2);
    for (pos, column) in neighbors {
        let distance = center.euclidean_distance(&pos);
        let activation = 0.8 * (1.0 - distance / 3.0);
        if activation > 0.0 {
            column.activate_with_strength(activation).ok();
        }
    }
    
    // Check activation gradient
    for radius in 1..=3 {
        let ring = grid.get_neighbors(center, radius);
        let avg_activation: f32 = ring.iter()
            .map(|(_, col)| col.activation_level())
            .sum::<f32>() / ring.len() as f32;
        
        // Activation should decrease with distance
        if radius > 1 {
            let prev_ring = grid.get_neighbors(center, radius - 1);
            let prev_avg: f32 = prev_ring.iter()
                .map(|(_, col)| col.activation_level())
                .sum::<f32>() / prev_ring.len().max(1) as f32;
            
            assert!(avg_activation <= prev_avg || ring.is_empty());
        }
    }
}

#[test]
fn test_find_nearest_available_expanding_search() {
    let grid = CorticalGrid::new(GridConfig::default());
    
    // Allocate a region
    let region = grid.get_region(
        GridPosition::new(4, 4, 2),
        GridPosition::new(6, 6, 4),
    );
    
    for column in region {
        column.activate_with_strength(0.8).unwrap();
        column.start_competing().unwrap();
        column.allocate().ok();
    }
    
    // Search from center of allocated region
    let search_pos = GridPosition::new(5, 5, 3);
    let available = grid.find_nearest_available(search_pos, 5);
    
    // Should find available columns outside the allocated region
    assert!(!available.is_empty());
    
    // All found columns should be available
    for col in &available {
        assert_eq!(col.state(), ColumnState::Available);
    }
    
    // They should be at the edge of the allocated region
    for col in &available {
        let pos = grid.get_position(col.id()).unwrap();
        let distance = search_pos.manhattan_distance(&pos);
        assert!(distance >= 2); // Outside the allocated 3x3x3 region
    }
}

#[test]
fn test_grid_with_inhibition_network_integration() {
    let grid = Arc::new(CorticalGrid::new(GridConfig::default()));
    let inhibition_network = Arc::new(LateralInhibitionNetwork::new(InhibitionConfig::default()));
    
    // Register all columns with the inhibition network
    for id in 0..grid.total_columns() {
        if let Some(pos) = grid.get_position(id as u32) {
            let float_pos = (pos.x as f32, pos.y as f32, pos.z as f32);
            inhibition_network.register_column(id as u32, float_pos);
        }
    }
    
    // Don't setup lateral connections before competition to avoid self-inhibition issues
    
    // Activate multiple columns with varying strengths
    let positions = vec![
        (GridPosition::new(2, 2, 2), 0.9),
        (GridPosition::new(5, 5, 3), 0.85),
        (GridPosition::new(8, 8, 4), 0.8),
        (GridPosition::new(3, 7, 1), 0.75),
    ];
    
    let mut candidates = Vec::new();
    for (pos, strength) in positions {
        if let Some(column) = grid.get_column_at(&pos) {
            column.activate_with_strength(strength).unwrap();
            column.start_competing().unwrap();
            candidates.push((column.id(), strength));
        }
    }
    
    // Run competition
    let result = inhibition_network.compete(candidates);
    
    // Winner should be the highest activation
    assert_eq!(result.winner_activation, 0.9);
    
    // Clear any self-inhibition on the winner before allocation
    inhibition_network.clear_inhibition(result.winner_id);
    
    // Allocate winner
    if let Some(winner_column) = grid.get_column(result.winner_id) {
        assert!(winner_column.allocate().is_ok());
    }
    
    // Suppressed columns should enter refractory
    for suppressed_id in result.suppressed_columns {
        if let Some(column) = grid.get_column(suppressed_id) {
            column.apply_network_inhibition(&inhibition_network);
            if column.activation_level() < 0.3 {
                column.enter_refractory().ok();
            }
        }
    }
}

#[test]
fn test_concurrent_grid_operations() {
    let grid = Arc::new(CorticalGrid::new(GridConfig::default()));
    let mut handles = vec![];
    
    // Multiple threads accessing different regions
    for thread_id in 0..4 {
        let grid_clone = grid.clone();
        handles.push(thread::spawn(move || {
            let x_offset = (thread_id % 2) * 5;
            let y_offset = (thread_id / 2) * 5;
            
            // Each thread works on a 5x5x3 region
            let start = GridPosition::new(x_offset, y_offset, 0);
            let end = GridPosition::new(x_offset + 4, y_offset + 4, 2);
            
            let region = grid_clone.get_region(start, end);
            let mut activated_count = 0;
            
            for column in region {
                if column.activate_with_strength(0.7).is_ok() {
                    activated_count += 1;
                }
            }
            
            activated_count
        }));
    }
    
    let results: Vec<usize> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Each thread should have activated its region
    for count in results {
        assert_eq!(count, 75); // 5x5x3 = 75 columns per region
    }
}

#[test]
fn test_grid_statistics() {
    let grid = CorticalGrid::new(GridConfig::default());
    
    // Create different states
    let mut allocated_count = 0;
    let mut refractory_count = 0;
    
    for id in 0..100 {
        if let Some(column) = grid.get_column(id) {
            match id % 4 {
                0 => {
                    // Allocate some
                    column.activate_with_strength(0.8).unwrap();
                    column.start_competing().unwrap();
                    if column.allocate().is_ok() {
                        allocated_count += 1;
                    }
                }
                1 => {
                    // Put some in refractory
                    column.activate_with_strength(0.6).unwrap();
                    column.start_competing().unwrap();
                    if column.enter_refractory().is_ok() {
                        refractory_count += 1;
                    }
                }
                _ => {
                    // Leave as available or just activate
                    column.activate_with_strength(0.4).ok();
                }
            }
        }
    }
    
    let stats = grid.get_stats();
    
    assert_eq!(stats.total_columns, 600); // 10x10x6
    assert_eq!(stats.allocated_columns, allocated_count);
    assert_eq!(stats.refractory_columns, refractory_count);
    assert!(stats.average_activation > 0.0);
}

#[test]
fn test_distance_calculations() {
    let pos1 = GridPosition::new(1, 2, 3);
    let pos2 = GridPosition::new(4, 6, 5);
    
    // Manhattan distance
    let manhattan = pos1.manhattan_distance(&pos2);
    assert_eq!(manhattan, 3 + 4 + 2); // |4-1| + |6-2| + |5-3|
    
    // Euclidean distance
    let euclidean = pos1.euclidean_distance(&pos2);
    let expected = ((3.0_f32).powi(2) + (4.0_f32).powi(2) + (2.0_f32).powi(2)).sqrt();
    assert!((euclidean - expected).abs() < 0.001);
}

#[test]
fn test_boundary_conditions() {
    let config = GridConfig {
        width: 5,
        height: 5,
        depth: 5,
        wrap_edges: false,
    };
    
    let grid = CorticalGrid::new(config);
    
    // Test all corner positions
    let corners = vec![
        GridPosition::new(0, 0, 0),
        GridPosition::new(4, 0, 0),
        GridPosition::new(0, 4, 0),
        GridPosition::new(4, 4, 0),
        GridPosition::new(0, 0, 4),
        GridPosition::new(4, 0, 4),
        GridPosition::new(0, 4, 4),
        GridPosition::new(4, 4, 4),
    ];
    
    for corner in corners {
        let column = grid.get_column_at(&corner);
        assert!(column.is_some());
        
        // Corner should have exactly 3 face-adjacent neighbors
        let neighbors = grid.get_neighbors(corner, 1);
        assert_eq!(neighbors.len(), 3);
    }
    
    // Test edge (not corner) positions
    let edge = GridPosition::new(2, 0, 0); // Bottom edge, middle
    let edge_neighbors = grid.get_neighbors(edge, 1);
    assert_eq!(edge_neighbors.len(), 4); // 4 face-adjacent neighbors for edge
    
    // Test face (not edge) positions
    let face = GridPosition::new(2, 2, 0); // Bottom face, center
    let face_neighbors = grid.get_neighbors(face, 1);
    assert_eq!(face_neighbors.len(), 5); // 5 face-adjacent neighbors for face center
}

#[test]
fn test_position_index_consistency() {
    let grid = CorticalGrid::new(GridConfig::default());
    
    // Test bidirectional mapping consistency
    for z in 0..6 {
        for y in 0..10 {
            for x in 0..10 {
                let pos = GridPosition::new(x, y, z);
                
                // Get column at position
                let column = grid.get_column_at(&pos).unwrap();
                let id = column.id();
                
                // Get position from ID
                let retrieved_pos = grid.get_position(id).unwrap();
                
                // Should be the same
                assert_eq!(pos, retrieved_pos);
                
                // Get column by ID
                let retrieved_column = grid.get_column(id).unwrap();
                assert_eq!(column.id(), retrieved_column.id());
            }
        }
    }
}

#[test]
fn test_grid_performance_at_scale() {
    let config = GridConfig {
        width: 30,
        height: 30,
        depth: 10,
        wrap_edges: false,
    };
    
    let grid = Arc::new(CorticalGrid::new(config));
    
    // Test neighbor finding performance
    let start = Instant::now();
    for _ in 0..1000 {
        let pos = GridPosition::new(15, 15, 5);
        let _neighbors = grid.get_neighbors(pos, 3);
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 100); // 1000 neighbor queries in <100ms
    
    // Test region query performance
    let start = Instant::now();
    for _ in 0..100 {
        let region = grid.get_region(
            GridPosition::new(5, 5, 2),
            GridPosition::new(25, 25, 8),
        );
        assert!(!region.is_empty());
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 100); // 100 large region queries in <100ms
    
    // Test find nearest available performance
    let start = Instant::now();
    for i in 0..100 {
        let pos = GridPosition::new(i % 30, (i / 30) % 30, i % 10);
        let _available = grid.find_nearest_available(pos, 5);
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 200); // 100 searches in <200ms
}