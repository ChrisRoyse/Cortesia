# Task 1.10: 3D Grid Topology

**Duration**: 3 hours  
**Complexity**: Medium-High  
**Dependencies**: Task 1.9 (Concept Deduplication)  
**AI Assistant Suitability**: High - Well-defined spatial data structures  

## Objective

Implement a 3D cortical grid topology system that efficiently organizes cortical columns in 3D space, providing O(1) neighbor finding, memory-efficient storage, and biologically-inspired connectivity patterns. This creates the spatial foundation for neuromorphic allocation with realistic cortical architecture.

## Specification

Create a high-performance 3D grid system for cortical columns with biological connectivity:

**Grid Properties**:
- 3D coordinate system (x, y, z) with configurable dimensions
- Memory-efficient columnar storage (≤1KB per column)
- O(1) average case neighbor finding
- Distance-based connectivity patterns
- Spatial locality preservation for cache efficiency

**Mathematical Models**:
- Grid coordinate: `pos(i, j, k) = (i × dx, j × dy, k × dz)`
- Linear index: `idx = i + j × width + k × width × height`
- Euclidean distance: `d = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]`
- Connectivity probability: `P(connection) = e^(-d²/2σ²)` (Gaussian falloff)

**Performance Requirements**:
- Grid initialization: < 10ms for 1M columns
- Neighbor query: < 1μs average case
- Memory usage: columns × 1KB ± 5%
- Spatial query: O(1) for local neighborhoods

## Implementation Guide

### Step 1: Core 3D Grid Structure

```rust
// src/cortical_grid.rs
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;

/// Configuration for 3D cortical grid
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Grid dimensions (width, height, depth)
    pub dimensions: (u32, u32, u32),
    
    /// Physical spacing between columns (micrometers)
    pub column_spacing_um: (f32, f32, f32),
    
    /// Maximum connection radius (micrometers)
    pub max_connection_radius_um: f32,
    
    /// Connectivity probability parameters
    pub connectivity_sigma: f32,
    
    /// Enable fast neighbor lookup caching
    pub enable_neighbor_cache: bool,
    
    /// Maximum cached neighbor sets per column
    pub max_cached_neighbors: usize,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            dimensions: (100, 100, 6),  // Typical cortical layer structure
            column_spacing_um: (50.0, 50.0, 200.0), // Realistic cortical spacing
            max_connection_radius_um: 300.0,
            connectivity_sigma: 150.0,
            enable_neighbor_cache: true,
            max_cached_neighbors: 1000,
        }
    }
}

impl GridConfig {
    /// Minicolumn-scale configuration (high resolution)
    pub fn minicolumn_scale() -> Self {
        Self {
            dimensions: (500, 500, 6),
            column_spacing_um: (30.0, 30.0, 200.0),
            max_connection_radius_um: 200.0,
            connectivity_sigma: 100.0,
            enable_neighbor_cache: true,
            max_cached_neighbors: 500,
        }
    }
    
    /// Hypercolumn-scale configuration (lower resolution, broader connectivity)
    pub fn hypercolumn_scale() -> Self {
        Self {
            dimensions: (50, 50, 6),
            column_spacing_um: (200.0, 200.0, 300.0),
            max_connection_radius_um: 1000.0,
            connectivity_sigma: 400.0,
            enable_neighbor_cache: true,
            max_cached_neighbors: 200,
        }
    }
    
    /// Fast processing configuration (optimized for speed)
    pub fn fast_processing() -> Self {
        Self {
            dimensions: (64, 64, 4), // Power-of-2 for optimization
            column_spacing_um: (100.0, 100.0, 250.0),
            max_connection_radius_um: 400.0,
            connectivity_sigma: 200.0,
            enable_neighbor_cache: true,
            max_cached_neighbors: 256,
        }
    }
    
    /// Total number of columns in grid
    pub fn total_columns(&self) -> u32 {
        self.dimensions.0 * self.dimensions.1 * self.dimensions.2
    }
    
    /// Convert grid coordinates to linear index
    pub fn coords_to_index(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        if x >= self.dimensions.0 || y >= self.dimensions.1 || z >= self.dimensions.2 {
            return None;
        }
        Some(x + y * self.dimensions.0 + z * self.dimensions.0 * self.dimensions.1)
    }
    
    /// Convert linear index to grid coordinates
    pub fn index_to_coords(&self, index: u32) -> Option<(u32, u32, u32)> {
        let total = self.total_columns();
        if index >= total {
            return None;
        }
        
        let z = index / (self.dimensions.0 * self.dimensions.1);
        let remainder = index % (self.dimensions.0 * self.dimensions.1);
        let y = remainder / self.dimensions.0;
        let x = remainder % self.dimensions.0;
        
        Some((x, y, z))
    }
    
    /// Convert grid coordinates to physical position (micrometers)
    pub fn coords_to_position(&self, x: u32, y: u32, z: u32) -> (f32, f32, f32) {
        (
            x as f32 * self.column_spacing_um.0,
            y as f32 * self.column_spacing_um.1,
            z as f32 * self.column_spacing_um.2,
        )
    }
}
```

### Step 2: Spatial Position and Distance Calculations

```rust
// src/spatial_topology.rs
use crate::GridConfig;
use std::f32;

/// 3D position in physical space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    /// Calculate Euclidean distance to another position
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Calculate squared distance (faster when only comparing)
    pub fn distance_squared_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
    
    /// Calculate Manhattan distance
    pub fn manhattan_distance_to(&self, other: &Position3D) -> f32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
    
    /// Check if position is within radius of another position
    pub fn is_within_radius(&self, other: &Position3D, radius: f32) -> bool {
        self.distance_squared_to(other) <= radius * radius
    }
}

/// Spatial topology manager for 3D cortical grid
pub struct SpatialTopology {
    /// Grid configuration
    config: GridConfig,
    
    /// Position cache for fast lookup
    position_cache: Vec<Position3D>,
    
    /// Neighbor cache for frequently accessed patterns
    neighbor_cache: parking_lot::RwLock<std::collections::HashMap<u32, Vec<u32>>>,
}

impl SpatialTopology {
    pub fn new(config: GridConfig) -> Self {
        let total_columns = config.total_columns() as usize;
        let mut position_cache = Vec::with_capacity(total_columns);
        
        // Pre-compute all positions for O(1) lookup
        for index in 0..config.total_columns() {
            if let Some((x, y, z)) = config.index_to_coords(index) {
                let (px, py, pz) = config.coords_to_position(x, y, z);
                position_cache.push(Position3D::new(px, py, pz));
            }
        }
        
        Self {
            config,
            position_cache,
            neighbor_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }
    
    /// Get physical position of column by index
    pub fn get_position(&self, column_index: u32) -> Option<Position3D> {
        self.position_cache.get(column_index as usize).copied()
    }
    
    /// Calculate distance between two columns
    pub fn distance_between(&self, index1: u32, index2: u32) -> Option<f32> {
        let pos1 = self.get_position(index1)?;
        let pos2 = self.get_position(index2)?;
        Some(pos1.distance_to(&pos2))
    }
    
    /// Calculate connection probability based on distance
    pub fn connection_probability(&self, index1: u32, index2: u32) -> Option<f32> {
        let distance = self.distance_between(index1, index2)?;
        
        if distance > self.config.max_connection_radius_um {
            return Some(0.0);
        }
        
        // Gaussian probability distribution
        let sigma = self.config.connectivity_sigma;
        let prob = (-distance * distance / (2.0 * sigma * sigma)).exp();
        Some(prob)
    }
    
    /// Check if two columns should be connected based on probability
    pub fn should_connect(&self, index1: u32, index2: u32, random_threshold: f32) -> Option<bool> {
        let prob = self.connection_probability(index1, index2)?;
        Some(prob >= random_threshold)
    }
    
    /// Get all columns within a radius
    pub fn columns_within_radius(&self, center_index: u32, radius: f32) -> Vec<u32> {
        let center_pos = match self.get_position(center_index) {
            Some(pos) => pos,
            None => return Vec::new(),
        };
        
        let mut neighbors = Vec::new();
        
        for (index, &position) in self.position_cache.iter().enumerate() {
            if index as u32 != center_index && center_pos.is_within_radius(&position, radius) {
                neighbors.push(index as u32);
            }
        }
        
        neighbors
    }
    
    /// Get cached neighbors or compute and cache them
    pub fn get_neighbors_cached(&self, column_index: u32) -> Vec<u32> {
        if self.config.enable_neighbor_cache {
            // Check cache first
            {
                let cache = self.neighbor_cache.read();
                if let Some(neighbors) = cache.get(&column_index) {
                    return neighbors.clone();
                }
            }
            
            // Compute neighbors
            let neighbors = self.columns_within_radius(
                column_index, 
                self.config.max_connection_radius_um
            );
            
            // Cache the result
            {
                let mut cache = self.neighbor_cache.write();
                if cache.len() < self.config.max_cached_neighbors {
                    cache.insert(column_index, neighbors.clone());
                }
            }
            
            neighbors
        } else {
            self.columns_within_radius(column_index, self.config.max_connection_radius_um)
        }
    }
    
    /// Clear neighbor cache
    pub fn clear_cache(&self) {
        let mut cache = self.neighbor_cache.write();
        cache.clear();
    }
    
    /// Get grid configuration
    pub fn config(&self) -> &GridConfig {
        &self.config
    }
    
    /// Calculate grid efficiency metrics
    pub fn efficiency_metrics(&self) -> GridEfficiencyMetrics {
        let total_columns = self.config.total_columns();
        let cache_size = self.neighbor_cache.read().len();
        let cache_hit_rate = if total_columns > 0 {
            cache_size as f32 / total_columns as f32
        } else {
            0.0
        };
        
        // Sample neighbor distribution
        let sample_size = (total_columns / 10).max(1).min(100);
        let mut total_neighbors = 0;
        let mut max_neighbors = 0;
        
        for i in 0..sample_size {
            let neighbors = self.get_neighbors_cached(i);
            total_neighbors += neighbors.len();
            max_neighbors = max_neighbors.max(neighbors.len());
        }
        
        let avg_neighbors = if sample_size > 0 {
            total_neighbors as f32 / sample_size as f32
        } else {
            0.0
        };
        
        GridEfficiencyMetrics {
            total_columns,
            cache_hit_rate,
            average_neighbors: avg_neighbors,
            max_neighbors,
            memory_usage_kb: self.estimate_memory_usage_kb(),
        }
    }
    
    /// Estimate memory usage in KB
    fn estimate_memory_usage_kb(&self) -> f32 {
        let position_cache_size = self.position_cache.len() * std::mem::size_of::<Position3D>();
        let neighbor_cache_size = {
            let cache = self.neighbor_cache.read();
            cache.len() * 32 + // HashMap overhead
            cache.values().map(|v| v.len() * 4).sum::<usize>() // Vec<u32> storage
        };
        
        (position_cache_size + neighbor_cache_size) as f32 / 1024.0
    }
}

#[derive(Debug, Clone)]
pub struct GridEfficiencyMetrics {
    pub total_columns: u32,
    pub cache_hit_rate: f32,
    pub average_neighbors: f32,
    pub max_neighbors: usize,
    pub memory_usage_kb: f32,
}
```

### Step 3: Advanced Grid Indexing

```rust
// src/grid_indexing.rs
use crate::{GridConfig, Position3D, SpatialTopology};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Advanced indexing system for fast spatial queries
pub struct GridIndexer {
    /// Spatial topology manager
    topology: SpatialTopology,
    
    /// Spatial hash buckets for O(1) region queries
    spatial_buckets: RwLock<HashMap<SpatialBucketKey, Vec<u32>>>,
    
    /// Bucket size in micrometers
    bucket_size_um: f32,
    
    /// Layer-wise indexing for efficient z-layer queries
    layer_indices: Vec<Vec<u32>>,
}

/// Spatial hash bucket key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpatialBucketKey {
    x: i32,
    y: i32,
    z: i32,
}

impl GridIndexer {
    pub fn new(config: GridConfig) -> Self {
        let topology = SpatialTopology::new(config.clone());
        let bucket_size_um = config.max_connection_radius_um / 2.0; // Overlap for better coverage
        
        // Build layer indices
        let mut layer_indices = vec![Vec::new(); config.dimensions.2 as usize];
        for index in 0..config.total_columns() {
            if let Some((_, _, z)) = config.index_to_coords(index) {
                layer_indices[z as usize].push(index);
            }
        }
        
        let mut indexer = Self {
            topology,
            spatial_buckets: RwLock::new(HashMap::new()),
            bucket_size_um,
            layer_indices,
        };
        
        // Build spatial hash buckets
        indexer.build_spatial_buckets();
        
        indexer
    }
    
    /// Build spatial hash buckets for fast region queries
    fn build_spatial_buckets(&self) {
        let mut buckets = self.spatial_buckets.write();
        buckets.clear();
        
        for index in 0..self.topology.config().total_columns() {
            if let Some(position) = self.topology.get_position(index) {
                let bucket_key = self.position_to_bucket_key(&position);
                buckets.entry(bucket_key).or_insert_with(Vec::new).push(index);
            }
        }
    }
    
    /// Convert position to spatial bucket key
    fn position_to_bucket_key(&self, position: &Position3D) -> SpatialBucketKey {
        SpatialBucketKey {
            x: (position.x / self.bucket_size_um).floor() as i32,
            y: (position.y / self.bucket_size_um).floor() as i32,
            z: (position.z / self.bucket_size_um).floor() as i32,
        }
    }
    
    /// Get all columns in a spatial region
    pub fn columns_in_region(&self, center: &Position3D, radius: f32) -> Vec<u32> {
        let mut result = Vec::new();
        let buckets = self.spatial_buckets.read();
        
        // Calculate bucket range to check
        let bucket_radius = (radius / self.bucket_size_um).ceil() as i32;
        let center_bucket = self.position_to_bucket_key(center);
        
        for dx in -bucket_radius..=bucket_radius {
            for dy in -bucket_radius..=bucket_radius {
                for dz in -bucket_radius..=bucket_radius {
                    let bucket_key = SpatialBucketKey {
                        x: center_bucket.x + dx,
                        y: center_bucket.y + dy,
                        z: center_bucket.z + dz,
                    };
                    
                    if let Some(columns) = buckets.get(&bucket_key) {
                        for &column_index in columns {
                            if let Some(column_pos) = self.topology.get_position(column_index) {
                                if center.is_within_radius(&column_pos, radius) {
                                    result.push(column_index);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        result
    }
    
    /// Fast neighbor finding using spatial buckets
    pub fn fast_neighbors(&self, column_index: u32, radius: f32) -> Vec<u32> {
        if let Some(center_pos) = self.topology.get_position(column_index) {
            let mut neighbors = self.columns_in_region(&center_pos, radius);
            // Remove self from neighbors
            neighbors.retain(|&idx| idx != column_index);
            neighbors
        } else {
            Vec::new()
        }
    }
    
    /// Get all columns in a specific layer
    pub fn columns_in_layer(&self, layer_z: u32) -> &Vec<u32> {
        self.layer_indices.get(layer_z as usize).unwrap_or(&Vec::new())
    }
    
    /// Find nearest neighbors (k-nearest)
    pub fn k_nearest_neighbors(&self, column_index: u32, k: usize) -> Vec<(u32, f32)> {
        let center_pos = match self.topology.get_position(column_index) {
            Some(pos) => pos,
            None => return Vec::new(),
        };
        
        // Start with a reasonable radius and expand if needed
        let mut radius = self.topology.config().max_connection_radius_um * 0.5;
        let mut neighbors;
        
        loop {
            neighbors = self.columns_in_region(&center_pos, radius);
            neighbors.retain(|&idx| idx != column_index);
            
            if neighbors.len() >= k || radius >= self.topology.config().max_connection_radius_um {
                break;
            }
            
            radius *= 1.5; // Expand search radius
        }
        
        // Calculate distances and sort
        let mut neighbor_distances: Vec<(u32, f32)> = neighbors
            .into_iter()
            .filter_map(|idx| {
                self.topology.get_position(idx)
                    .map(|pos| (idx, center_pos.distance_to(&pos)))
            })
            .collect();
        
        neighbor_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbor_distances.truncate(k);
        
        neighbor_distances
    }
    
    /// Get spatial topology reference
    pub fn topology(&self) -> &SpatialTopology {
        &self.topology
    }
    
    /// Get indexing performance metrics
    pub fn indexing_metrics(&self) -> IndexingMetrics {
        let buckets = self.spatial_buckets.read();
        let total_buckets = buckets.len();
        let total_entries: usize = buckets.values().map(|v| v.len()).sum();
        let avg_bucket_size = if total_buckets > 0 {
            total_entries as f32 / total_buckets as f32
        } else {
            0.0
        };
        
        let max_bucket_size = buckets.values().map(|v| v.len()).max().unwrap_or(0);
        
        IndexingMetrics {
            total_buckets,
            average_bucket_size: avg_bucket_size,
            max_bucket_size,
            bucket_size_um: self.bucket_size_um,
            memory_usage_kb: self.estimate_memory_usage_kb(),
        }
    }
    
    /// Estimate memory usage in KB
    fn estimate_memory_usage_kb(&self) -> f32 {
        let topology_memory = self.topology.efficiency_metrics().memory_usage_kb;
        
        let bucket_memory = {
            let buckets = self.spatial_buckets.read();
            let bucket_overhead = buckets.len() * 32; // HashMap overhead
            let bucket_data: usize = buckets.values().map(|v| v.len() * 4).sum();
            bucket_overhead + bucket_data
        };
        
        let layer_memory = self.layer_indices.iter()
            .map(|layer| layer.len() * 4)
            .sum::<usize>();
        
        topology_memory + (bucket_memory + layer_memory) as f32 / 1024.0
    }
}

#[derive(Debug, Clone)]
pub struct IndexingMetrics {
    pub total_buckets: usize,
    pub average_bucket_size: f32,
    pub max_bucket_size: usize,
    pub bucket_size_um: f32,
    pub memory_usage_kb: f32,
}
```

### Step 4: Integrated 3D Cortical Grid

```rust
// src/cortical_grid_complete.rs
use crate::{GridConfig, GridIndexer, SpatialTopology, Position3D, BiologicalCorticalColumn};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Complete 3D cortical grid with columns and spatial indexing
pub struct CorticalGrid {
    /// Grid configuration
    config: GridConfig,
    
    /// Spatial indexing system
    indexer: GridIndexer,
    
    /// Cortical columns indexed by position
    columns: RwLock<HashMap<u32, Arc<BiologicalCorticalColumn>>>,
    
    /// Connection matrix (sparse representation)
    connections: RwLock<HashMap<u32, Vec<(u32, f32)>>>, // (source -> [(target, strength)])
    
    /// Grid initialization timestamp
    initialization_time: std::time::Instant,
}

impl CorticalGrid {
    pub fn new(config: GridConfig) -> Self {
        let start_time = std::time::Instant::now();
        
        let indexer = GridIndexer::new(config.clone());
        
        Self {
            config,
            indexer,
            columns: RwLock::new(HashMap::new()),
            connections: RwLock::new(HashMap::new()),
            initialization_time: start_time,
        }
    }
    
    /// Initialize all cortical columns
    pub fn initialize_columns(&self, biological_config: crate::BiologicalConfig) -> GridInitializationResult {
        let start_time = std::time::Instant::now();
        let mut columns = self.columns.write();
        
        let total_columns = self.config.total_columns();
        for index in 0..total_columns {
            let column = Arc::new(BiologicalCorticalColumn::new(index, biological_config.clone()));
            columns.insert(index, column);
        }
        
        let initialization_time = start_time.elapsed();
        
        GridInitializationResult {
            total_columns,
            initialization_time_ms: initialization_time.as_millis() as f32,
            memory_usage_kb: self.estimate_total_memory_kb(),
            success: true,
        }
    }
    
    /// Build spatial connections between columns
    pub fn build_connections(&self, connection_probability_threshold: f32) -> ConnectionBuildResult {
        let start_time = std::time::Instant::now();
        let mut connections = self.connections.write();
        connections.clear();
        
        let total_columns = self.config.total_columns();
        let mut total_connections = 0;
        let mut connection_lengths = Vec::new();
        
        for source_index in 0..total_columns {
            let neighbors = self.indexer.fast_neighbors(
                source_index, 
                self.config.max_connection_radius_um
            );
            
            let mut source_connections = Vec::new();
            
            for &target_index in &neighbors {
                if let Some(connection_prob) = self.indexer.topology().connection_probability(source_index, target_index) {
                    if connection_prob >= connection_probability_threshold {
                        // Calculate connection strength based on distance
                        let distance = self.indexer.topology().distance_between(source_index, target_index).unwrap_or(0.0);
                        let strength = connection_prob; // Use probability as initial strength
                        
                        source_connections.push((target_index, strength));
                        total_connections += 1;
                        connection_lengths.push(distance);
                    }
                }
            }
            
            if !source_connections.is_empty() {
                connections.insert(source_index, source_connections);
            }
        }
        
        let build_time = start_time.elapsed();
        
        // Calculate statistics
        let avg_connection_length = if !connection_lengths.is_empty() {
            connection_lengths.iter().sum::<f32>() / connection_lengths.len() as f32
        } else {
            0.0
        };
        
        let max_connection_length = connection_lengths.iter().fold(0.0f32, |a, &b| a.max(b));
        
        ConnectionBuildResult {
            total_connections,
            average_connections_per_column: if total_columns > 0 {
                total_connections as f32 / total_columns as f32
            } else {
                0.0
            },
            average_connection_length_um: avg_connection_length,
            max_connection_length_um: max_connection_length,
            build_time_ms: build_time.as_millis() as f32,
            connection_density: if total_columns > 1 {
                total_connections as f32 / (total_columns * (total_columns - 1)) as f32
            } else {
                0.0
            },
        }
    }
    
    /// Get cortical column by index
    pub fn get_column(&self, index: u32) -> Option<Arc<BiologicalCorticalColumn>> {
        self.columns.read().get(&index).cloned()
    }
    
    /// Get connections from a column
    pub fn get_connections(&self, source_index: u32) -> Vec<(u32, f32)> {
        self.connections.read()
            .get(&source_index)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Find columns within radius of a position
    pub fn columns_near_position(&self, position: Position3D, radius: f32) -> Vec<u32> {
        self.indexer.columns_in_region(&position, radius)
    }
    
    /// Get k-nearest neighbor columns
    pub fn k_nearest_columns(&self, column_index: u32, k: usize) -> Vec<(u32, f32)> {
        self.indexer.k_nearest_neighbors(column_index, k)
    }
    
    /// Get all columns in a specific layer
    pub fn layer_columns(&self, layer: u32) -> &Vec<u32> {
        self.indexer.columns_in_layer(layer)
    }
    
    /// Get grid configuration
    pub fn config(&self) -> &GridConfig {
        &self.config
    }
    
    /// Get spatial indexing system
    pub fn indexer(&self) -> &GridIndexer {
        &self.indexer
    }
    
    /// Get comprehensive grid metrics
    pub fn grid_metrics(&self) -> GridMetrics {
        let indexing_metrics = self.indexer.indexing_metrics();
        let topology_metrics = self.indexer.topology().efficiency_metrics();
        
        let column_count = self.columns.read().len();
        let connection_count = self.connections.read().values().map(|v| v.len()).sum::<usize>();
        
        GridMetrics {
            total_columns: self.config.total_columns(),
            active_columns: column_count as u32,
            total_connections: connection_count,
            indexing_metrics,
            topology_metrics,
            total_memory_kb: self.estimate_total_memory_kb(),
            grid_age_ms: self.initialization_time.elapsed().as_millis() as f32,
        }
    }
    
    /// Estimate total memory usage
    fn estimate_total_memory_kb(&self) -> f32 {
        let indexing_memory = self.indexer.indexing_metrics().memory_usage_kb;
        
        let columns_memory = {
            let columns = self.columns.read();
            columns.len() * 1024 // Estimated 1KB per column
        };
        
        let connections_memory = {
            let connections = self.connections.read();
            let connection_overhead = connections.len() * 32; // HashMap overhead
            let connection_data: usize = connections.values().map(|v| v.len() * 8).sum(); // (u32, f32) pairs
            connection_overhead + connection_data
        };
        
        indexing_memory + (columns_memory + connections_memory) as f32 / 1024.0
    }
    
    /// Validate grid integrity
    pub fn validate_integrity(&self) -> GridValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check column indices are valid
        let columns = self.columns.read();
        for &index in columns.keys() {
            if index >= self.config.total_columns() {
                errors.push(format!("Invalid column index: {} >= {}", index, self.config.total_columns()));
            }
        }
        
        // Check connections point to valid columns
        let connections = self.connections.read();
        for (&source, targets) in connections.iter() {
            if !columns.contains_key(&source) {
                errors.push(format!("Connection source {} has no column", source));
            }
            
            for &(target, strength) in targets {
                if !columns.contains_key(&target) {
                    warnings.push(format!("Connection target {} has no column", target));
                }
                
                if strength < 0.0 || strength > 1.0 {
                    warnings.push(format!("Connection strength {} out of range [0,1]", strength));
                }
            }
        }
        
        // Check memory usage
        let memory_kb = self.estimate_total_memory_kb();
        let expected_memory_kb = self.config.total_columns() as f32 * 1.0; // 1KB per column target
        let memory_ratio = memory_kb / expected_memory_kb;
        
        if memory_ratio > 1.1 {
            warnings.push(format!("Memory usage {:.1}KB exceeds target {:.1}KB", memory_kb, expected_memory_kb));
        }
        
        GridValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            memory_usage_kb: memory_kb,
            expected_memory_kb,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridInitializationResult {
    pub total_columns: u32,
    pub initialization_time_ms: f32,
    pub memory_usage_kb: f32,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectionBuildResult {
    pub total_connections: usize,
    pub average_connections_per_column: f32,
    pub average_connection_length_um: f32,
    pub max_connection_length_um: f32,
    pub build_time_ms: f32,
    pub connection_density: f32,
}

#[derive(Debug, Clone)]
pub struct GridMetrics {
    pub total_columns: u32,
    pub active_columns: u32,
    pub total_connections: usize,
    pub indexing_metrics: crate::IndexingMetrics,
    pub topology_metrics: crate::GridEfficiencyMetrics,
    pub total_memory_kb: f32,
    pub grid_age_ms: f32,
}

#[derive(Debug, Clone)]
pub struct GridValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub memory_usage_kb: f32,
    pub expected_memory_kb: f32,
}
```

## AI-Executable Test Suite

```rust
// tests/grid_topology_test.rs
use llmkg::{
    CorticalGrid, GridConfig, Position3D, BiologicalConfig
};
use std::time::Instant;

#[test]
fn test_grid_initialization_performance() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config);
    
    let start = Instant::now();
    let result = grid.initialize_columns(BiologicalConfig::fast_processing());
    let elapsed = start.elapsed();
    
    assert!(result.success);
    assert!(result.initialization_time_ms < 50.0); // Should be very fast for test grid
    assert_eq!(result.total_columns, 64 * 64 * 4); // Expected grid size
    
    println!("Grid initialized in {:.2}ms", elapsed.as_millis());
}

#[test]
fn test_neighbor_finding_accuracy() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config.clone());
    grid.initialize_columns(BiologicalConfig::default());
    
    // Test neighbor finding for center column
    let center_index = config.coords_to_index(32, 32, 2).unwrap();
    let neighbors = grid.k_nearest_columns(center_index, 10);
    
    assert_eq!(neighbors.len(), 10);
    
    // Neighbors should be sorted by distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].1 >= neighbors[i-1].1);
    }
    
    // All neighbors should be within reasonable distance
    for (_, distance) in &neighbors {
        assert!(*distance < config.max_connection_radius_um);
    }
}

#[test]
fn test_spatial_indexing_performance() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config);
    grid.initialize_columns(BiologicalConfig::default());
    
    // Benchmark neighbor queries
    let test_indices = vec![0, 100, 1000, 5000];
    let start = Instant::now();
    
    for &index in &test_indices {
        let neighbors = grid.k_nearest_columns(index, 20);
        assert!(!neighbors.is_empty());
    }
    
    let elapsed = start.elapsed();
    let ns_per_query = elapsed.as_nanos() / test_indices.len() as u128;
    
    println!("Neighbor query: {} ns", ns_per_query);
    assert!(ns_per_query < 10_000); // Should be < 10μs
}

#[test]
fn test_connection_building() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config);
    grid.initialize_columns(BiologicalConfig::default());
    
    let result = grid.build_connections(0.1); // 10% probability threshold
    
    assert!(result.total_connections > 0);
    assert!(result.average_connections_per_column > 0.0);
    assert!(result.connection_density > 0.0 && result.connection_density < 1.0);
    assert!(result.build_time_ms < 1000.0); // Should complete quickly
    
    println!("Built {} connections in {:.2}ms", 
             result.total_connections, result.build_time_ms);
}

#[test]
fn test_memory_efficiency() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config.clone());
    grid.initialize_columns(BiologicalConfig::default());
    grid.build_connections(0.05);
    
    let metrics = grid.grid_metrics();
    let memory_per_column = metrics.total_memory_kb / config.total_columns() as f32;
    
    println!("Memory per column: {:.2} KB", memory_per_column);
    assert!(memory_per_column <= 1.1); // Within 10% of 1KB target
    
    // Memory should scale reasonably
    assert!(metrics.total_memory_kb > 0.0);
    assert!(metrics.total_memory_kb < config.total_columns() as f32 * 2.0); // Less than 2KB per column
}

#[test]
fn test_grid_validation() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config);
    grid.initialize_columns(BiologicalConfig::default());
    grid.build_connections(0.1);
    
    let validation = grid.validate_integrity();
    
    assert!(validation.is_valid, "Grid validation failed: {:?}", validation.errors);
    
    // Warnings are okay, but no errors
    for error in &validation.errors {
        println!("Error: {}", error);
    }
    
    for warning in &validation.warnings {
        println!("Warning: {}", warning);
    }
}

#[test]
fn test_coordinate_conversions() {
    let config = GridConfig::fast_processing();
    
    // Test coordinate round-trip conversions
    for x in 0..config.dimensions.0.min(10) {
        for y in 0..config.dimensions.1.min(10) {
            for z in 0..config.dimensions.2 {
                let index = config.coords_to_index(x, y, z).unwrap();
                let (x2, y2, z2) = config.index_to_coords(index).unwrap();
                
                assert_eq!((x, y, z), (x2, y2, z2));
                
                // Test position conversion
                let (px, py, pz) = config.coords_to_position(x, y, z);
                assert!(px >= 0.0 && py >= 0.0 && pz >= 0.0);
            }
        }
    }
}

#[test]
fn test_layer_organization() {
    let config = GridConfig::fast_processing();
    let grid = CorticalGrid::new(config.clone());
    grid.initialize_columns(BiologicalConfig::default());
    
    // Test layer-wise access
    for layer in 0..config.dimensions.2 {
        let layer_columns = grid.layer_columns(layer);
        
        assert_eq!(layer_columns.len(), (config.dimensions.0 * config.dimensions.1) as usize);
        
        // All columns in layer should have same z-coordinate
        for &column_index in layer_columns {
            let (_, _, z) = config.index_to_coords(column_index).unwrap();
            assert_eq!(z, layer);
        }
    }
}

#[test]
fn test_large_grid_performance() {
    let config = GridConfig {
        dimensions: (200, 200, 6), // 240K columns
        ..GridConfig::fast_processing()
    };
    
    let start = Instant::now();
    let grid = CorticalGrid::new(config.clone());
    let construction_time = start.elapsed();
    
    let start = Instant::now();
    let init_result = grid.initialize_columns(BiologicalConfig::fast_processing());
    let init_time = start.elapsed();
    
    assert!(init_result.success);
    assert!(construction_time < std::time::Duration::from_millis(100));
    assert!(init_time < std::time::Duration::from_millis(5000)); // 5 seconds max
    
    println!("Large grid ({}K columns): construction {:.2}ms, init {:.2}ms",
             config.total_columns() / 1000,
             construction_time.as_millis(),
             init_time.as_millis());
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 8/8 grid topology tests passing
2. **Performance targets met**:
   - Grid initialization < 10ms for 100K columns
   - Neighbor query < 1μs average case
   - Memory usage ≤ 1KB per column ± 5%
   - Spatial queries O(1) for local neighborhoods
3. **Spatial accuracy**:
   - Coordinate conversions are bijective
   - Distance calculations accurate to ±0.1%
   - Layer organization preserved
4. **Memory efficiency**: Total memory within target bounds

## Verification Commands

```bash
# Run topology tests
cargo test grid_topology_test --release -- --nocapture

# Performance verification
cargo test test_spatial_indexing_performance --release -- --nocapture

# Large grid test
cargo test test_large_grid_performance --release -- --nocapture

# Memory efficiency test
cargo test test_memory_efficiency --release -- --nocapture
```

## Files to Create

1. `src/cortical_grid.rs`
2. `src/spatial_topology.rs`
3. `src/grid_indexing.rs`
4. `src/cortical_grid_complete.rs`
5. `tests/grid_topology_test.rs`

## Expected Completion Time

3 hours for an AI assistant:
- 60 minutes: Core grid structure and coordinate systems
- 60 minutes: Spatial topology and distance calculations
- 45 minutes: Advanced indexing and optimization
- 15 minutes: Integration testing and validation

## Next Task

Task 1.11: Spatial Indexing (KD-tree implementation for efficient queries)