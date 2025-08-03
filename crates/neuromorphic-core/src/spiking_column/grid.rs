//! 3D cortical grid structure for spatial organization

use super::{SpikingCorticalColumn, ColumnId};
use dashmap::DashMap;
use std::sync::Arc;

/// Position in the 3D cortical grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPosition {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GridPosition {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }
    
    /// Calculate Manhattan distance to another position
    pub fn manhattan_distance(&self, other: &GridPosition) -> usize {
        let dx = (self.x as i32 - other.x as i32).abs() as usize;
        let dy = (self.y as i32 - other.y as i32).abs() as usize;
        let dz = (self.z as i32 - other.z as i32).abs() as usize;
        dx + dy + dz
    }
    
    /// Calculate Euclidean distance to another position
    pub fn euclidean_distance(&self, other: &GridPosition) -> f32 {
        let dx = self.x as f32 - other.x as f32;
        let dy = self.y as f32 - other.y as f32;
        let dz = self.z as f32 - other.z as f32;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Configuration for the cortical grid
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub wrap_edges: bool,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            width: 10,
            height: 10,
            depth: 6,
            wrap_edges: false,
        }
    }
}

/// 3D grid of spiking cortical columns
pub struct CorticalGrid {
    /// Grid configuration
    config: GridConfig,
    
    /// Flattened array of columns
    columns: Vec<Arc<SpikingCorticalColumn>>,
    
    /// Position to column ID mapping
    position_map: DashMap<GridPosition, ColumnId>,
    
    /// Column ID to position mapping
    id_to_position: DashMap<ColumnId, GridPosition>,
    
    /// Total number of columns
    total_columns: usize,
}

impl CorticalGrid {
    /// Create a new cortical grid
    pub fn new(config: GridConfig) -> Self {
        let total = config.width * config.height * config.depth;
        let mut columns = Vec::with_capacity(total);
        let position_map = DashMap::new();
        let id_to_position = DashMap::new();
        
        // Create columns in 3D grid
        let mut column_id = 0;
        for z in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let pos = GridPosition::new(x, y, z);
                    let column = Arc::new(SpikingCorticalColumn::new(column_id));
                    
                    columns.push(column);
                    position_map.insert(pos, column_id);
                    id_to_position.insert(column_id, pos);
                    
                    column_id += 1;
                }
            }
        }
        
        Self {
            config,
            columns,
            position_map,
            id_to_position,
            total_columns: total,
        }
    }
    
    /// Get total number of columns
    pub fn total_columns(&self) -> usize {
        self.total_columns
    }
    
    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.config.width, self.config.height, self.config.depth)
    }
    
    /// Convert 3D position to linear index
    fn position_to_index(&self, pos: &GridPosition) -> Option<usize> {
        if pos.x >= self.config.width || 
           pos.y >= self.config.height || 
           pos.z >= self.config.depth {
            return None;
        }
        
        Some(pos.z * self.config.width * self.config.height + 
             pos.y * self.config.width + 
             pos.x)
    }
    
    /// Get column at specific position
    pub fn get_column_at(&self, pos: &GridPosition) -> Option<Arc<SpikingCorticalColumn>> {
        self.position_to_index(pos)
            .and_then(|idx| self.columns.get(idx))
            .cloned()
    }
    
    /// Get column by ID
    pub fn get_column(&self, id: ColumnId) -> Option<Arc<SpikingCorticalColumn>> {
        self.columns.get(id as usize).cloned()
    }
    
    /// Get position of a column
    pub fn get_position(&self, id: ColumnId) -> Option<GridPosition> {
        self.id_to_position.get(&id).map(|pos| *pos)
    }
    
    /// Find all neighbors within a given radius
    pub fn get_neighbors(&self, pos: GridPosition, radius: usize) -> Vec<(GridPosition, Arc<SpikingCorticalColumn>)> {
        let mut neighbors = Vec::new();
        
        let x_start = pos.x.saturating_sub(radius);
        let x_end = (pos.x + radius + 1).min(self.config.width);
        let y_start = pos.y.saturating_sub(radius);
        let y_end = (pos.y + radius + 1).min(self.config.height);
        let z_start = pos.z.saturating_sub(radius);
        let z_end = (pos.z + radius + 1).min(self.config.depth);
        
        for z in z_start..z_end {
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let neighbor_pos = GridPosition::new(x, y, z);
                    
                    // Skip self
                    if neighbor_pos == pos {
                        continue;
                    }
                    
                    // Check if within radius (using Manhattan distance)
                    if neighbor_pos.manhattan_distance(&pos) <= radius {
                        if let Some(column) = self.get_column_at(&neighbor_pos) {
                            neighbors.push((neighbor_pos, column));
                        }
                    }
                }
            }
        }
        
        neighbors
    }
    
    /// Find nearest available columns
    pub fn find_nearest_available(&self, pos: GridPosition, max_radius: usize) -> Vec<Arc<SpikingCorticalColumn>> {
        use super::ColumnState;
        
        // Search in expanding radii
        for radius in 1..=max_radius {
            let neighbors = self.get_neighbors(pos, radius);
            let available: Vec<_> = neighbors.into_iter()
                .filter(|(_, col)| col.state() == ColumnState::Available)
                .map(|(_, col)| col)
                .collect();
            
            if !available.is_empty() {
                return available;
            }
        }
        
        Vec::new()
    }
    
    /// Get columns in a rectangular region
    pub fn get_region(&self, 
                     start: GridPosition, 
                     end: GridPosition) -> Vec<Arc<SpikingCorticalColumn>> {
        let mut columns = Vec::new();
        
        let x_start = start.x.min(end.x);
        let x_end = start.x.max(end.x).min(self.config.width - 1);
        let y_start = start.y.min(end.y);
        let y_end = start.y.max(end.y).min(self.config.height - 1);
        let z_start = start.z.min(end.z);
        let z_end = start.z.max(end.z).min(self.config.depth - 1);
        
        for z in z_start..=z_end {
            for y in y_start..=y_end {
                for x in x_start..=x_end {
                    let pos = GridPosition::new(x, y, z);
                    if let Some(column) = self.get_column_at(&pos) {
                        columns.push(column);
                    }
                }
            }
        }
        
        columns
    }
    
    /// Setup lateral connections based on spatial proximity
    pub fn setup_lateral_connections(&self, connection_radius: usize, base_weight: f32) {
        for column in &self.columns {
            let column_id = column.id();
            if let Some(pos) = self.get_position(column_id) {
                let neighbors = self.get_neighbors(pos, connection_radius);
                
                for (neighbor_pos, _) in neighbors {
                    if let Some(neighbor_id) = self.position_map.get(&neighbor_pos) {
                        // Weight decreases with distance
                        let distance = pos.euclidean_distance(&neighbor_pos);
                        // Ensure weight stays positive by using max radius as normalization factor
                        let max_distance = (connection_radius as f32) * 1.732; // sqrt(3) for 3D diagonal
                        let weight = base_weight * (1.0 - distance / max_distance).max(0.1);
                        
                        column.add_lateral_connection(*neighbor_id, weight);
                    }
                }
            }
        }
    }
    
    /// Get statistics about the grid
    pub fn get_stats(&self) -> GridStats {
        use super::ColumnState;
        
        let mut stats = GridStats::default();
        
        for column in &self.columns {
            stats.total_columns += 1;
            
            match column.state() {
                ColumnState::Available => stats.available_columns += 1,
                ColumnState::Allocated => stats.allocated_columns += 1,
                ColumnState::Refractory => stats.refractory_columns += 1,
                _ => {}
            }
            
            stats.total_activation += column.activation_level();
        }
        
        stats.average_activation = stats.total_activation / stats.total_columns as f32;
        
        stats
    }
}

/// Statistics about the cortical grid
#[derive(Debug, Default)]
pub struct GridStats {
    pub total_columns: usize,
    pub available_columns: usize,
    pub allocated_columns: usize,
    pub refractory_columns: usize,
    pub total_activation: f32,
    pub average_activation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grid_creation() {
        let config = GridConfig {
            width: 5,
            height: 5,
            depth: 3,
            wrap_edges: false,
        };
        
        let grid = CorticalGrid::new(config);
        
        assert_eq!(grid.total_columns(), 75);
        assert_eq!(grid.dimensions(), (5, 5, 3));
        
        // Check position mapping
        let pos = GridPosition::new(2, 3, 1);
        let column = grid.get_column_at(&pos).unwrap();
        assert_eq!(grid.get_position(column.id()), Some(pos));
    }
    
    #[test]
    fn test_neighbor_finding() {
        let grid = CorticalGrid::new(GridConfig::default());
        let center = GridPosition::new(5, 5, 3);
        
        // Radius 1 neighbors (26 in 3D, minus corners for Manhattan distance)
        let neighbors_r1 = grid.get_neighbors(center, 1);
        assert_eq!(neighbors_r1.len(), 6); // Only face-adjacent for Manhattan distance 1
        
        // Radius 2 neighbors
        let neighbors_r2 = grid.get_neighbors(center, 2);
        assert!(neighbors_r2.len() > neighbors_r1.len());
        
        // Verify distances
        for (pos, _) in &neighbors_r1 {
            assert!(pos.manhattan_distance(&center) <= 1);
        }
    }
    
    #[test]
    fn test_edge_cases() {
        let grid = CorticalGrid::new(GridConfig::default());
        
        // Corner position
        let corner = GridPosition::new(0, 0, 0);
        let neighbors = grid.get_neighbors(corner, 1);
        assert_eq!(neighbors.len(), 3); // Only 3 face-adjacent neighbors at corner
        
        // Out of bounds
        let oob = GridPosition::new(100, 100, 100);
        assert!(grid.get_column_at(&oob).is_none());
    }
    
    #[test]
    fn test_region_query() {
        let grid = CorticalGrid::new(GridConfig::default());
        
        let start = GridPosition::new(2, 2, 1);
        let end = GridPosition::new(4, 4, 2);
        
        let region = grid.get_region(start, end);
        assert_eq!(region.len(), 18); // 3x3x2 region
    }
    
    #[test]
    fn test_lateral_connections() {
        let config = GridConfig {
            width: 3,
            height: 3,
            depth: 1,
            wrap_edges: false,
        };
        
        let grid = CorticalGrid::new(config);
        grid.setup_lateral_connections(1, 0.5);
        
        // Center column should have 4 connections (cross pattern)
        let center_pos = GridPosition::new(1, 1, 0);
        let center_column = grid.get_column_at(&center_pos).unwrap();
        
        // Check connections exist
        let neighbors = grid.get_neighbors(center_pos, 1);
        for (neighbor_pos, _) in neighbors {
            if let Some(neighbor_id) = grid.position_map.get(&neighbor_pos) {
                let weight = center_column.connection_strength_to(*neighbor_id);
                assert!(weight.is_some());
                assert!(weight.unwrap() > 0.0);
            }
        }
    }
}