# Task 1.12: Neighbor Finding

**Duration**: 2 hours  
**Complexity**: Medium  
**Dependencies**: Task 1.11 (Spatial Indexing)  
**AI Assistant Suitability**: High - Optimization-focused algorithms  

## Objective

Implement highly optimized spatial neighbor finding with Euclidean distance calculations, batch query support, and connection strength modeling. This provides sub-microsecond neighbor queries with biologically-inspired connectivity patterns for neuromorphic allocation systems.

## Specification

Create ultra-fast neighbor finding with biological accuracy:

**Neighbor Finding Properties**:
- Optimized Euclidean distance calculations with SIMD acceleration
- Batch query processing for improved cache utilization
- Biological connection strength calculation based on distance
- Distance-based connectivity rules with probabilistic thresholds
- Memory-efficient neighbor list caching and management

**Mathematical Models**:
- Euclidean distance: `d = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]`
- Squared distance (optimization): `d² = (x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²`
- Connection strength: `S = e^(-d²/2σ²) × (1 - d/d_max)` for d ≤ d_max
- Batch efficiency: `batch_speedup = min(10.0, batch_size/cache_line_factor)`

**Performance Requirements**:
- Single neighbor query: < 1μs average case
- Batch queries: 10x faster than individual queries
- Distance accuracy: ±0.1% precision
- Connection strength: Biological curve matching within 5%
- Memory overhead: < 100 bytes per cached neighbor set

## Implementation Guide

### Step 1: Optimized Distance Calculations

```rust
// src/distance_optimization.rs
use crate::Position3D;
use std::arch::x86_64::*;

/// High-performance distance calculation utilities
pub struct DistanceCalculator {
    /// Enable SIMD optimization
    simd_enabled: bool,
    
    /// Distance calculation cache
    distance_cache: parking_lot::RwLock<std::collections::HashMap<(u32, u32), f32>>,
    
    /// Cache hit statistics
    cache_stats: parking_lot::RwLock<CacheStatistics>,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_calculation_time_ns: f32,
}

impl DistanceCalculator {
    pub fn new() -> Self {
        Self {
            simd_enabled: is_x86_feature_detected!("avx2"),
            distance_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            cache_stats: parking_lot::RwLock::new(CacheStatistics::default()),
        }
    }
    
    /// Calculate Euclidean distance between two positions
    pub fn distance(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        let start_time = std::time::Instant::now();
        
        let result = if self.simd_enabled {
            self.distance_simd(pos1, pos2)
        } else {
            self.distance_scalar(pos1, pos2)
        };
        
        let calculation_time = start_time.elapsed();
        self.update_cache_stats(calculation_time);
        
        result
    }
    
    /// Calculate squared distance (faster for comparisons)
    pub fn distance_squared(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        if self.simd_enabled {
            self.distance_squared_simd(pos1, pos2)
        } else {
            self.distance_squared_scalar(pos1, pos2)
        }
    }
    
    /// Calculate distance with caching
    pub fn distance_cached(&self, id1: u32, pos1: &Position3D, id2: u32, pos2: &Position3D) -> f32 {
        let cache_key = if id1 < id2 { (id1, id2) } else { (id2, id1) };
        
        // Check cache first
        {
            let cache = self.distance_cache.read();
            if let Some(&cached_distance) = cache.get(&cache_key) {
                self.record_cache_hit();
                return cached_distance;
            }
        }
        
        // Calculate and cache
        let distance = self.distance(pos1, pos2);
        
        {
            let mut cache = self.distance_cache.write();
            // Limit cache size to prevent memory bloat
            if cache.len() < 10000 {
                cache.insert(cache_key, distance);
            }
        }
        
        self.record_cache_miss();
        distance
    }
    
    /// SIMD-accelerated distance calculation
    #[target_feature(enable = "avx2")]
    unsafe fn distance_simd(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        // Load positions into SIMD registers
        let p1 = _mm_set_ps(0.0, pos1.z, pos1.y, pos1.x);
        let p2 = _mm_set_ps(0.0, pos2.z, pos2.y, pos2.x);
        
        // Calculate difference
        let diff = _mm_sub_ps(p1, p2);
        
        // Square differences
        let squared = _mm_mul_ps(diff, diff);
        
        // Extract components for summation
        let mut components = [0.0f32; 4];
        _mm_storeu_ps(components.as_mut_ptr(), squared);
        
        // Sum x², y², z² and take square root
        (components[0] + components[1] + components[2]).sqrt()
    }
    
    /// Scalar distance calculation (fallback)
    fn distance_scalar(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// SIMD-accelerated squared distance
    #[target_feature(enable = "avx2")]
    unsafe fn distance_squared_simd(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        let p1 = _mm_set_ps(0.0, pos1.z, pos1.y, pos1.x);
        let p2 = _mm_set_ps(0.0, pos2.z, pos2.y, pos2.x);
        
        let diff = _mm_sub_ps(p1, p2);
        let squared = _mm_mul_ps(diff, diff);
        
        let mut components = [0.0f32; 4];
        _mm_storeu_ps(components.as_mut_ptr(), squared);
        
        components[0] + components[1] + components[2]
    }
    
    /// Scalar squared distance calculation
    fn distance_squared_scalar(&self, pos1: &Position3D, pos2: &Position3D) -> f32 {
        let dx = pos1.x - pos2.x;
        let dy = pos1.y - pos2.y;
        let dz = pos1.z - pos2.z;
        dx * dx + dy * dy + dz * dz
    }
    
    /// Batch distance calculation for multiple pairs
    pub fn batch_distances(&self, pairs: &[(Position3D, Position3D)]) -> Vec<f32> {
        let mut results = Vec::with_capacity(pairs.len());
        
        if self.simd_enabled && pairs.len() >= 4 {
            self.batch_distances_simd(pairs, &mut results);
        } else {
            for (pos1, pos2) in pairs {
                results.push(self.distance_scalar(pos1, pos2));
            }
        }
        
        results
    }
    
    /// SIMD batch distance calculation
    #[target_feature(enable = "avx2")]
    unsafe fn batch_distances_simd(&self, pairs: &[(Position3D, Position3D)], results: &mut Vec<f32>) {
        let chunks = pairs.chunks(4);
        
        for chunk in chunks {
            if chunk.len() == 4 {
                // Process 4 distances at once
                let mut distances = [0.0f32; 4];
                
                for (i, (pos1, pos2)) in chunk.iter().enumerate() {
                    distances[i] = self.distance_simd(pos1, pos2);
                }
                
                results.extend_from_slice(&distances);
            } else {
                // Handle remaining pairs with scalar calculation
                for (pos1, pos2) in chunk {
                    results.push(self.distance_scalar(pos1, pos2));
                }
            }
        }
    }
    
    /// Update cache statistics
    fn update_cache_stats(&self, calculation_time: std::time::Duration) {
        let mut stats = self.cache_stats.write();
        stats.total_requests += 1;
        
        let total = stats.total_requests as f32;
        let new_time = calculation_time.as_nanos() as f32;
        stats.avg_calculation_time_ns = ((stats.avg_calculation_time_ns * (total - 1.0)) + new_time) / total;
    }
    
    /// Record cache hit
    fn record_cache_hit(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_hits += 1;
    }
    
    /// Record cache miss
    fn record_cache_miss(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_misses += 1;
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStatistics {
        self.cache_stats.read().clone()
    }
    
    /// Clear distance cache
    pub fn clear_cache(&self) {
        let mut cache = self.distance_cache.write();
        cache.clear();
        
        let mut stats = self.cache_stats.write();
        *stats = CacheStatistics::default();
    }
    
    /// Check if SIMD is enabled
    pub fn is_simd_enabled(&self) -> bool {
        self.simd_enabled
    }
}
```

### Step 2: Connection Strength Modeling

```rust
// src/connection_strength.rs
use crate::{Position3D, DistanceCalculator};
use std::f32::consts::E;

/// Configuration for biological connection strength modeling
#[derive(Debug, Clone)]
pub struct ConnectionStrengthConfig {
    /// Maximum connection distance (micrometers)
    pub max_distance_um: f32,
    
    /// Gaussian falloff parameter (sigma)
    pub falloff_sigma_um: f32,
    
    /// Minimum connection strength threshold
    pub min_strength_threshold: f32,
    
    /// Enable distance-based attenuation
    pub enable_distance_attenuation: bool,
    
    /// Linear attenuation factor
    pub linear_attenuation_factor: f32,
    
    /// Base connection strength
    pub base_strength: f32,
    
    /// Random variation factor (0.0 = no variation, 1.0 = full random)
    pub random_variation: f32,
}

impl Default for ConnectionStrengthConfig {
    fn default() -> Self {
        Self {
            max_distance_um: 300.0,
            falloff_sigma_um: 150.0,
            min_strength_threshold: 0.01,
            enable_distance_attenuation: true,
            linear_attenuation_factor: 0.5,
            base_strength: 1.0,
            random_variation: 0.1,
        }
    }
}

impl ConnectionStrengthConfig {
    /// Configuration for local connections (strong, short-range)
    pub fn local_connections() -> Self {
        Self {
            max_distance_um: 150.0,
            falloff_sigma_um: 75.0,
            min_strength_threshold: 0.05,
            enable_distance_attenuation: true,
            linear_attenuation_factor: 0.3,
            base_strength: 0.9,
            random_variation: 0.05,
        }
    }
    
    /// Configuration for long-range connections (weak, long-range)
    pub fn long_range_connections() -> Self {
        Self {
            max_distance_um: 800.0,
            falloff_sigma_um: 400.0,
            min_strength_threshold: 0.001,
            enable_distance_attenuation: true,
            linear_attenuation_factor: 0.8,
            base_strength: 0.3,
            random_variation: 0.2,
        }
    }
    
    /// Configuration optimized for speed
    pub fn fast_calculation() -> Self {
        Self {
            max_distance_um: 200.0,
            falloff_sigma_um: 100.0,
            min_strength_threshold: 0.02,
            enable_distance_attenuation: false,
            linear_attenuation_factor: 0.0,
            base_strength: 0.8,
            random_variation: 0.0,
        }
    }
}

/// Connection strength calculator with biological modeling
pub struct ConnectionStrengthCalculator {
    config: ConnectionStrengthConfig,
    distance_calc: DistanceCalculator,
    strength_cache: parking_lot::RwLock<std::collections::HashMap<(u32, u32), f32>>,
    rng: parking_lot::Mutex<fastrand::Rng>,
}

impl ConnectionStrengthCalculator {
    pub fn new(config: ConnectionStrengthConfig) -> Self {
        Self {
            config,
            distance_calc: DistanceCalculator::new(),
            strength_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            rng: parking_lot::Mutex::new(fastrand::Rng::new()),
        }
    }
    
    /// Calculate connection strength between two positions
    pub fn calculate_strength(
        &self,
        pos1: &Position3D,
        pos2: &Position3D,
        column_id1: Option<u32>,
        column_id2: Option<u32>,
    ) -> f32 {
        // Check cache if IDs are provided
        if let (Some(id1), Some(id2)) = (column_id1, column_id2) {
            let cache_key = if id1 < id2 { (id1, id2) } else { (id2, id1) };
            
            {
                let cache = self.strength_cache.read();
                if let Some(&cached_strength) = cache.get(&cache_key) {
                    return cached_strength;
                }
            }
        }
        
        let distance = self.distance_calc.distance(pos1, pos2);
        let strength = self.calculate_strength_from_distance(distance);
        
        // Cache result if IDs are provided
        if let (Some(id1), Some(id2)) = (column_id1, column_id2) {
            let cache_key = if id1 < id2 { (id1, id2) } else { (id2, id1) };
            
            let mut cache = self.strength_cache.write();
            if cache.len() < 50000 { // Limit cache size
                cache.insert(cache_key, strength);
            }
        }
        
        strength
    }
    
    /// Calculate strength from pre-computed distance
    pub fn calculate_strength_from_distance(&self, distance: f32) -> f32 {
        // Check if distance exceeds maximum
        if distance > self.config.max_distance_um {
            return 0.0;
        }
        
        // Gaussian falloff component
        let sigma = self.config.falloff_sigma_um;
        let gaussian_factor = E.powf(-distance * distance / (2.0 * sigma * sigma));
        
        // Linear attenuation component (optional)
        let attenuation_factor = if self.config.enable_distance_attenuation {
            let linear_factor = 1.0 - (distance / self.config.max_distance_um) * self.config.linear_attenuation_factor;
            linear_factor.max(0.0)
        } else {
            1.0
        };
        
        // Calculate base strength
        let base_strength = self.config.base_strength * gaussian_factor * attenuation_factor;
        
        // Apply random variation if enabled
        let final_strength = if self.config.random_variation > 0.0 {
            let variation = {
                let mut rng = self.rng.lock();
                (rng.f32() - 0.5) * 2.0 * self.config.random_variation
            };
            (base_strength * (1.0 + variation)).max(0.0)
        } else {
            base_strength
        };
        
        // Apply minimum threshold
        if final_strength < self.config.min_strength_threshold {
            0.0
        } else {
            final_strength.min(1.0) // Clamp to maximum of 1.0
        }
    }
    
    /// Batch calculation of connection strengths
    pub fn batch_calculate_strengths(
        &self,
        position_pairs: &[(Position3D, Position3D)],
        id_pairs: Option<&[(u32, u32)]>,
    ) -> Vec<f32> {
        let mut results = Vec::with_capacity(position_pairs.len());
        
        // Calculate distances in batch
        let distances = self.distance_calc.batch_distances(position_pairs);
        
        // Calculate strengths from distances
        for (i, &distance) in distances.iter().enumerate() {
            let strength = self.calculate_strength_from_distance(distance);
            
            // Cache result if IDs are provided
            if let Some(id_pairs) = id_pairs {
                if let Some(&(id1, id2)) = id_pairs.get(i) {
                    let cache_key = if id1 < id2 { (id1, id2) } else { (id2, id1) };
                    
                    let mut cache = self.strength_cache.write();
                    if cache.len() < 50000 {
                        cache.insert(cache_key, strength);
                    }
                }
            }
            
            results.push(strength);
        }
        
        results
    }
    
    /// Check if connection should exist based on strength
    pub fn should_connect(&self, strength: f32) -> bool {
        strength >= self.config.min_strength_threshold
    }
    
    /// Get connection probability from strength
    pub fn connection_probability(&self, strength: f32) -> f32 {
        if strength <= 0.0 {
            0.0
        } else if strength >= 1.0 {
            1.0
        } else {
            // Sigmoid-like transformation for probability
            1.0 / (1.0 + E.powf(-10.0 * (strength - 0.5)))
        }
    }
    
    /// Calculate effective connectivity radius for given minimum strength
    pub fn effective_radius(&self, min_strength: f32) -> f32 {
        if min_strength <= 0.0 {
            return self.config.max_distance_um;
        }
        
        // Solve for distance where strength equals min_strength
        // This is an approximation for the Gaussian component
        let sigma = self.config.falloff_sigma_um;
        let ln_ratio = (min_strength / self.config.base_strength).ln();
        
        if ln_ratio >= 0.0 {
            0.0
        } else {
            (sigma * sigma * -2.0 * ln_ratio).sqrt().min(self.config.max_distance_um)
        }
    }
    
    /// Get configuration
    pub fn config(&self) -> &ConnectionStrengthConfig {
        &self.config
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> ConnectionCacheStatistics {
        let cache = self.strength_cache.read();
        let distance_stats = self.distance_calc.cache_stats();
        
        ConnectionCacheStatistics {
            strength_cache_size: cache.len(),
            distance_cache_stats: distance_stats,
            max_cached_strength: cache.values().fold(0.0f32, |acc, &x| acc.max(x)),
            min_cached_strength: cache.values().fold(1.0f32, |acc, &x| acc.min(x)),
        }
    }
    
    /// Clear all caches
    pub fn clear_caches(&self) {
        let mut cache = self.strength_cache.write();
        cache.clear();
        self.distance_calc.clear_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionCacheStatistics {
    pub strength_cache_size: usize,
    pub distance_cache_stats: crate::CacheStatistics,
    pub max_cached_strength: f32,
    pub min_cached_strength: f32,
}
```

### Step 3: Batch Neighbor Query Engine

```rust
// src/batch_neighbor_queries.rs
use crate::{
    Position3D, SpatialIndexingSystem, ConnectionStrengthCalculator, 
    ConnectionStrengthConfig, DistanceCalculator
};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Batch neighbor query configuration
#[derive(Debug, Clone)]
pub struct BatchQueryConfig {
    /// Batch size for optimal cache utilization
    pub optimal_batch_size: usize,
    
    /// Enable result caching across batch operations
    pub enable_batch_caching: bool,
    
    /// Maximum cached batch results
    pub max_cached_batches: usize,
    
    /// Prefetch radius multiplier for cache warming
    pub prefetch_radius_multiplier: f32,
    
    /// Enable parallel batch processing
    pub enable_parallel_processing: bool,
    
    /// Minimum batch size for parallel processing
    pub parallel_threshold: usize,
}

impl Default for BatchQueryConfig {
    fn default() -> Self {
        Self {
            optimal_batch_size: 64, // Cache line friendly
            enable_batch_caching: true,
            max_cached_batches: 100,
            prefetch_radius_multiplier: 1.2,
            enable_parallel_processing: true,
            parallel_threshold: 16,
        }
    }
}

/// Neighbor query result with strength information
#[derive(Debug, Clone)]
pub struct NeighborResult {
    pub column_index: u32,
    pub position: Position3D,
    pub distance: f32,
    pub connection_strength: f32,
    pub should_connect: bool,
}

/// Batch neighbor query result
#[derive(Debug, Clone)]
pub struct BatchNeighborResult {
    pub query_centers: Vec<u32>,
    pub neighbor_lists: Vec<Vec<NeighborResult>>,
    pub total_neighbors_found: usize,
    pub query_time_ms: f32,
    pub cache_hit_rate: f32,
}

/// High-performance batch neighbor query engine
pub struct BatchNeighborQueryEngine {
    /// Spatial indexing system
    spatial_index: SpatialIndexingSystem,
    
    /// Connection strength calculator
    strength_calc: ConnectionStrengthCalculator,
    
    /// Distance calculator
    distance_calc: DistanceCalculator,
    
    /// Query configuration
    config: BatchQueryConfig,
    
    /// Batch result cache
    batch_cache: RwLock<HashMap<BatchCacheKey, Vec<NeighborResult>>>,
    
    /// Query statistics
    query_stats: RwLock<BatchQueryStatistics>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BatchCacheKey {
    center_id: u32,
    radius_bits: u32,
    strength_threshold_bits: u32,
}

#[derive(Debug, Clone, Default)]
pub struct BatchQueryStatistics {
    pub total_batch_queries: u64,
    pub total_individual_queries: u64,
    pub total_neighbors_found: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_batch_size: f32,
    pub avg_query_time_ns: f32,
    pub speedup_factor: f32,
}

impl BatchNeighborQueryEngine {
    pub fn new(
        spatial_index: SpatialIndexingSystem,
        strength_config: ConnectionStrengthConfig,
        batch_config: BatchQueryConfig,
    ) -> Self {
        Self {
            spatial_index,
            strength_calc: ConnectionStrengthCalculator::new(strength_config),
            distance_calc: DistanceCalculator::new(),
            config: batch_config,
            batch_cache: RwLock::new(HashMap::new()),
            query_stats: RwLock::new(BatchQueryStatistics::default()),
        }
    }
    
    /// Find neighbors for a single column with connection strength
    pub fn find_neighbors(
        &self,
        center_column_id: u32,
        center_position: &Position3D,
        radius: f32,
        min_strength: f32,
    ) -> Vec<NeighborResult> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if self.config.enable_batch_caching {
            let cache_key = BatchCacheKey {
                center_id: center_column_id,
                radius_bits: radius.to_bits(),
                strength_threshold_bits: min_strength.to_bits(),
            };
            
            {
                let cache = self.batch_cache.read();
                if let Some(cached_result) = cache.get(&cache_key) {
                    self.record_cache_hit();
                    return cached_result.clone();
                }
            }
        }
        
        // Find spatial neighbors
        let spatial_neighbors = self.spatial_index.range_query(center_position, radius);
        let mut neighbor_results = Vec::with_capacity(spatial_neighbors.len());
        
        // Calculate connection strengths
        for &neighbor_id in &spatial_neighbors {
            if neighbor_id == center_column_id {
                continue; // Skip self
            }
            
            // Get neighbor position (this would be from grid or index)
            // For now, we'll use a placeholder - in real implementation,
            // this would come from the spatial index
            let neighbor_position = Position3D::new(0.0, 0.0, 0.0); // Placeholder
            
            let distance = self.distance_calc.distance(center_position, &neighbor_position);
            let strength = self.strength_calc.calculate_strength(
                center_position,
                &neighbor_position,
                Some(center_column_id),
                Some(neighbor_id),
            );
            
            if strength >= min_strength {
                neighbor_results.push(NeighborResult {
                    column_index: neighbor_id,
                    position: neighbor_position,
                    distance,
                    connection_strength: strength,
                    should_connect: self.strength_calc.should_connect(strength),
                });
            }
        }
        
        // Sort by distance (closest first)
        neighbor_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        let query_time = start_time.elapsed();
        self.update_individual_query_stats(neighbor_results.len(), query_time);
        
        // Cache result
        if self.config.enable_batch_caching {
            self.cache_neighbor_result(center_column_id, radius, min_strength, neighbor_results.clone());
        }
        
        neighbor_results
    }
    
    /// Batch neighbor finding for multiple columns
    pub fn batch_find_neighbors(
        &self,
        query_centers: &[(u32, Position3D)],
        radius: f32,
        min_strength: f32,
    ) -> BatchNeighborResult {
        let start_time = std::time::Instant::now();
        let mut neighbor_lists = Vec::with_capacity(query_centers.len());
        let mut total_neighbors = 0;
        let mut cache_hits = 0;
        
        if query_centers.len() >= self.config.parallel_threshold && self.config.enable_parallel_processing {
            // Parallel processing for large batches
            self.batch_find_neighbors_parallel(query_centers, radius, min_strength, &mut neighbor_lists, &mut total_neighbors, &mut cache_hits);
        } else {
            // Sequential processing for small batches
            self.batch_find_neighbors_sequential(query_centers, radius, min_strength, &mut neighbor_lists, &mut total_neighbors, &mut cache_hits);
        }
        
        let query_time = start_time.elapsed();
        let cache_hit_rate = if query_centers.len() > 0 {
            cache_hits as f32 / query_centers.len() as f32
        } else {
            0.0
        };
        
        self.update_batch_query_stats(query_centers.len(), total_neighbors, query_time);
        
        BatchNeighborResult {
            query_centers: query_centers.iter().map(|(id, _)| *id).collect(),
            neighbor_lists,
            total_neighbors_found: total_neighbors,
            query_time_ms: query_time.as_millis() as f32,
            cache_hit_rate,
        }
    }
    
    /// Sequential batch processing
    fn batch_find_neighbors_sequential(
        &self,
        query_centers: &[(u32, Position3D)],
        radius: f32,
        min_strength: f32,
        neighbor_lists: &mut Vec<Vec<NeighborResult>>,
        total_neighbors: &mut usize,
        cache_hits: &mut usize,
    ) {
        for &(center_id, center_pos) in query_centers {
            let neighbors = self.find_neighbors_with_cache_tracking(
                center_id, &center_pos, radius, min_strength
            );
            
            if neighbors.1 { // Cache hit
                *cache_hits += 1;
            }
            
            *total_neighbors += neighbors.0.len();
            neighbor_lists.push(neighbors.0);
        }
    }
    
    /// Parallel batch processing
    fn batch_find_neighbors_parallel(
        &self,
        query_centers: &[(u32, Position3D)],
        radius: f32,
        min_strength: f32,
        neighbor_lists: &mut Vec<Vec<NeighborResult>>,
        total_neighbors: &mut usize,
        cache_hits: &mut usize,
    ) {
        use rayon::prelude::*;
        
        let results: Vec<(Vec<NeighborResult>, bool)> = query_centers
            .par_iter()
            .map(|&(center_id, center_pos)| {
                self.find_neighbors_with_cache_tracking(center_id, &center_pos, radius, min_strength)
            })
            .collect();
        
        for (neighbors, was_cache_hit) in results {
            if was_cache_hit {
                *cache_hits += 1;
            }
            *total_neighbors += neighbors.len();
            neighbor_lists.push(neighbors);
        }
    }
    
    /// Find neighbors with cache tracking
    fn find_neighbors_with_cache_tracking(
        &self,
        center_id: u32,
        center_pos: &Position3D,
        radius: f32,
        min_strength: f32,
    ) -> (Vec<NeighborResult>, bool) {
        let cache_key = BatchCacheKey {
            center_id,
            radius_bits: radius.to_bits(),
            strength_threshold_bits: min_strength.to_bits(),
        };
        
        // Check cache
        if self.config.enable_batch_caching {
            let cache = self.batch_cache.read();
            if let Some(cached_result) = cache.get(&cache_key) {
                return (cached_result.clone(), true);
            }
        }
        
        // Calculate new result
        let neighbors = self.find_neighbors(center_id, center_pos, radius, min_strength);
        (neighbors, false)
    }
    
    /// Optimized k-nearest neighbors with connection strength
    pub fn k_nearest_neighbors_with_strength(
        &self,
        center_column_id: u32,
        center_position: &Position3D,
        k: usize,
        min_strength: f32,
    ) -> Vec<NeighborResult> {
        // Get k-nearest from spatial index
        let spatial_knn = self.spatial_index.k_nearest_neighbors(center_position, k * 2); // Get extra in case of filtering
        
        let mut neighbor_results = Vec::with_capacity(k);
        
        for (neighbor_id, distance) in spatial_knn {
            if neighbor_id == center_column_id {
                continue;
            }
            
            // Calculate connection strength
            let neighbor_position = Position3D::new(0.0, 0.0, 0.0); // Placeholder
            let strength = self.strength_calc.calculate_strength_from_distance(distance);
            
            if strength >= min_strength {
                neighbor_results.push(NeighborResult {
                    column_index: neighbor_id,
                    position: neighbor_position,
                    distance,
                    connection_strength: strength,
                    should_connect: self.strength_calc.should_connect(strength),
                });
                
                if neighbor_results.len() >= k {
                    break;
                }
            }
        }
        
        neighbor_results
    }
    
    /// Cache neighbor finding result
    fn cache_neighbor_result(
        &self,
        center_id: u32,
        radius: f32,
        min_strength: f32,
        result: Vec<NeighborResult>,
    ) {
        let cache_key = BatchCacheKey {
            center_id,
            radius_bits: radius.to_bits(),
            strength_threshold_bits: min_strength.to_bits(),
        };
        
        let mut cache = self.batch_cache.write();
        
        // Remove old entries if cache is full
        if cache.len() >= self.config.max_cached_batches {
            // Simple LRU: remove random entry
            if let Some(key_to_remove) = cache.keys().next().cloned() {
                cache.remove(&key_to_remove);
            }
        }
        
        cache.insert(cache_key, result);
    }
    
    /// Record cache hit
    fn record_cache_hit(&self) {
        let mut stats = self.query_stats.write();
        stats.cache_hits += 1;
    }
    
    /// Update individual query statistics
    fn update_individual_query_stats(&self, neighbors_found: usize, query_time: std::time::Duration) {
        let mut stats = self.query_stats.write();
        stats.total_individual_queries += 1;
        stats.total_neighbors_found += neighbors_found as u64;
        
        let total = stats.total_individual_queries as f32;
        stats.avg_query_time_ns = ((stats.avg_query_time_ns * (total - 1.0)) + query_time.as_nanos() as f32) / total;
    }
    
    /// Update batch query statistics
    fn update_batch_query_stats(&self, batch_size: usize, neighbors_found: usize, query_time: std::time::Duration) {
        let mut stats = self.query_stats.write();
        stats.total_batch_queries += 1;
        stats.total_neighbors_found += neighbors_found as u64;
        
        let total_batches = stats.total_batch_queries as f32;
        stats.avg_batch_size = ((stats.avg_batch_size * (total_batches - 1.0)) + batch_size as f32) / total_batches;
        
        // Calculate speedup factor (assuming individual queries would take longer)
        let estimated_individual_time = batch_size as f32 * stats.avg_query_time_ns;
        let actual_batch_time = query_time.as_nanos() as f32;
        
        if actual_batch_time > 0.0 {
            let current_speedup = estimated_individual_time / actual_batch_time;
            stats.speedup_factor = ((stats.speedup_factor * (total_batches - 1.0)) + current_speedup) / total_batches;
        }
    }
    
    /// Get query statistics
    pub fn query_statistics(&self) -> BatchQueryStatistics {
        self.query_stats.read().clone()
    }
    
    /// Clear all caches
    pub fn clear_caches(&self) {
        let mut cache = self.batch_cache.write();
        cache.clear();
        
        self.strength_calc.clear_caches();
        self.distance_calc.clear_cache();
    }
    
    /// Get comprehensive performance metrics
    pub fn performance_metrics(&self) -> NeighborFindingMetrics {
        let query_stats = self.query_statistics();
        let strength_cache_stats = self.strength_calc.cache_stats();
        let distance_cache_stats = self.distance_calc.cache_stats();
        
        NeighborFindingMetrics {
            query_stats,
            strength_cache_stats,
            distance_cache_stats,
            simd_enabled: self.distance_calc.is_simd_enabled(),
            batch_cache_size: self.batch_cache.read().len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeighborFindingMetrics {
    pub query_stats: BatchQueryStatistics,
    pub strength_cache_stats: crate::ConnectionCacheStatistics,
    pub distance_cache_stats: crate::CacheStatistics,
    pub simd_enabled: bool,
    pub batch_cache_size: usize,
}
```

## AI-Executable Test Suite

```rust
// tests/neighbor_finding_test.rs
use llmkg::{
    BatchNeighborQueryEngine, SpatialIndexingSystem, KDTreeConfig, SpatialPoint,
    Position3D, ConnectionStrengthConfig, BatchQueryConfig, DistanceCalculator
};
use std::time::Instant;

#[test]
fn test_distance_calculation_performance() {
    let calc = DistanceCalculator::new();
    
    let pos1 = Position3D::new(100.0, 200.0, 50.0);
    let pos2 = Position3D::new(150.0, 180.0, 75.0);
    
    // Warm up
    for _ in 0..1000 {
        calc.distance(&pos1, &pos2);
    }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..10000 {
        calc.distance(&pos1, &pos2);
    }
    let elapsed = start.elapsed();
    
    let ns_per_calc = elapsed.as_nanos() / 10000;
    println!("Distance calculation: {} ns", ns_per_calc);
    println!("SIMD enabled: {}", calc.is_simd_enabled());
    
    assert!(ns_per_calc < 100); // Should be very fast
    
    // Test accuracy
    let expected_distance = ((50.0_f32).powi(2) + (-20.0_f32).powi(2) + (25.0_f32).powi(2)).sqrt();
    let calculated_distance = calc.distance(&pos1, &pos2);
    let error = (calculated_distance - expected_distance).abs();
    
    assert!(error < 0.001, "Distance calculation error: {}", error);
}

#[test]
fn test_batch_distance_calculation() {
    let calc = DistanceCalculator::new();
    
    // Create test pairs
    let pairs: Vec<(Position3D, Position3D)> = (0..1000)
        .map(|i| {
            let pos1 = Position3D::new(i as f32, (i * 2) as f32, (i / 2) as f32);
            let pos2 = Position3D::new((i + 10) as f32, (i * 2 + 5) as f32, (i / 2 + 3) as f32);
            (pos1, pos2)
        })
        .collect();
    
    // Benchmark batch vs individual
    let start = Instant::now();
    let batch_results = calc.batch_distances(&pairs);
    let batch_time = start.elapsed();
    
    let start = Instant::now();
    let individual_results: Vec<f32> = pairs.iter()
        .map(|(p1, p2)| calc.distance(p1, p2))
        .collect();
    let individual_time = start.elapsed();
    
    println!("Batch time: {} μs", batch_time.as_micros());
    println!("Individual time: {} μs", individual_time.as_micros());
    
    // Results should be identical
    for (i, (&batch_dist, &individual_dist)) in batch_results.iter().zip(&individual_results).enumerate() {
        let error = (batch_dist - individual_dist).abs();
        assert!(error < 0.001, "Mismatch at {}: {} vs {}", i, batch_dist, individual_dist);
    }
    
    // Batch should be faster or at least not much slower
    let speedup = individual_time.as_nanos() as f32 / batch_time.as_nanos() as f32;
    println!("Batch speedup: {:.2}x", speedup);
}

#[test]
fn test_connection_strength_calculation() {
    let config = ConnectionStrengthConfig::default();
    let calc = crate::ConnectionStrengthCalculator::new(config.clone());
    
    let center = Position3D::new(0.0, 0.0, 0.0);
    
    // Test strength at various distances
    let test_distances = vec![0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0];
    
    for distance in test_distances {
        let pos = Position3D::new(distance, 0.0, 0.0);
        let strength = calc.calculate_strength(&center, &pos, None, None);
        
        println!("Distance: {} μm, Strength: {:.4}", distance, strength);
        
        // Strength should decrease with distance
        if distance <= config.max_distance_um {
            assert!(strength >= 0.0);
            assert!(strength <= 1.0);
        } else {
            assert_eq!(strength, 0.0);
        }
    }
    
    // Test biological curve properties
    let close_strength = calc.calculate_strength(&center, &Position3D::new(25.0, 0.0, 0.0), None, None);
    let far_strength = calc.calculate_strength(&center, &Position3D::new(200.0, 0.0, 0.0), None, None);
    
    assert!(close_strength > far_strength, "Closer connections should be stronger");
}

#[test]
fn test_neighbor_finding_accuracy() {
    let points = generate_grid_points(20, 20, 3);
    let spatial_index = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let neighbor_engine = BatchNeighborQueryEngine::new(
        spatial_index,
        ConnectionStrengthConfig::local_connections(),
        BatchQueryConfig::default(),
    );
    
    let center_pos = Position3D::new(100.0, 100.0, 50.0);
    let radius = 150.0;
    let min_strength = 0.1;
    
    let neighbors = neighbor_engine.find_neighbors(0, &center_pos, radius, min_strength);
    
    println!("Found {} neighbors within {} μm", neighbors.len(), radius);
    
    // Verify all neighbors are within radius
    for neighbor in &neighbors {
        assert!(neighbor.distance <= radius);
        assert!(neighbor.connection_strength >= min_strength);
        assert_eq!(neighbor.should_connect, neighbor.connection_strength >= min_strength);
    }
    
    // Neighbors should be sorted by distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].distance >= neighbors[i-1].distance);
    }
}

#[test]
fn test_batch_neighbor_query_performance() {
    let points = generate_test_points(5000);
    let spatial_index = SpatialIndexingSystem::new(points, KDTreeConfig::high_performance()).unwrap();
    
    let neighbor_engine = BatchNeighborQueryEngine::new(
        spatial_index,
        ConnectionStrengthConfig::fast_calculation(),
        BatchQueryConfig::default(),
    );
    
    // Create test query centers
    let query_centers: Vec<(u32, Position3D)> = (0..100)
        .map(|i| (i, Position3D::new(i as f32 * 10.0, i as f32 * 10.0, (i % 10) as f32 * 20.0)))
        .collect();
    
    let radius = 100.0;
    let min_strength = 0.05;
    
    // Benchmark individual vs batch queries
    let start = Instant::now();
    let mut individual_results = Vec::new();
    for &(id, pos) in &query_centers {
        individual_results.push(neighbor_engine.find_neighbors(id, &pos, radius, min_strength));
    }
    let individual_time = start.elapsed();
    
    // Clear caches for fair comparison
    neighbor_engine.clear_caches();
    
    let start = Instant::now();
    let batch_result = neighbor_engine.batch_find_neighbors(&query_centers, radius, min_strength);
    let batch_time = start.elapsed();
    
    println!("Individual queries: {} ms", individual_time.as_millis());
    println!("Batch query: {} ms", batch_time.as_millis());
    println!("Speedup: {:.2}x", individual_time.as_millis() as f32 / batch_time.as_millis() as f32);
    
    // Batch should be faster
    assert!(batch_time < individual_time);
    
    // Results should be similar (allowing for cache effects)
    assert_eq!(batch_result.neighbor_lists.len(), individual_results.len());
    assert_eq!(batch_result.query_centers.len(), query_centers.len());
}

#[test]
fn test_k_nearest_neighbors_with_strength() {
    let points = generate_grid_points(30, 30, 4);
    let spatial_index = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let neighbor_engine = BatchNeighborQueryEngine::new(
        spatial_index,
        ConnectionStrengthConfig::default(),
        BatchQueryConfig::default(),
    );
    
    let center_pos = Position3D::new(150.0, 150.0, 100.0);
    let k = 20;
    let min_strength = 0.01;
    
    let knn_results = neighbor_engine.k_nearest_neighbors_with_strength(0, &center_pos, k, min_strength);
    
    assert!(knn_results.len() <= k);
    
    // Should be sorted by distance
    for i in 1..knn_results.len() {
        assert!(knn_results[i].distance >= knn_results[i-1].distance);
    }
    
    // All should meet strength threshold
    for neighbor in &knn_results {
        assert!(neighbor.connection_strength >= min_strength);
    }
    
    println!("K-NN found {} neighbors (requested {})", knn_results.len(), k);
}

#[test]
fn test_caching_effectiveness() {
    let points = generate_test_points(1000);
    let spatial_index = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let neighbor_engine = BatchNeighborQueryEngine::new(
        spatial_index,
        ConnectionStrengthConfig::default(),
        BatchQueryConfig::default(),
    );
    
    let center_pos = Position3D::new(100.0, 100.0, 50.0);
    let radius = 100.0;
    let min_strength = 0.1;
    
    // First query (cache miss)
    let start = Instant::now();
    let result1 = neighbor_engine.find_neighbors(0, &center_pos, radius, min_strength);
    let first_time = start.elapsed();
    
    // Second identical query (cache hit)
    let start = Instant::now();
    let result2 = neighbor_engine.find_neighbors(0, &center_pos, radius, min_strength);
    let second_time = start.elapsed();
    
    // Results should be identical
    assert_eq!(result1.len(), result2.len());
    
    // Second query should be faster
    println!("First query: {} μs", first_time.as_micros());
    println!("Second query: {} μs", second_time.as_micros());
    
    let metrics = neighbor_engine.performance_metrics();
    println!("Cache stats: {:?}", metrics.distance_cache_stats);
    
    assert!(second_time <= first_time); // Should be same or faster due to caching
}

#[test]
fn test_performance_targets() {
    let points = generate_test_points(10000);
    let spatial_index = SpatialIndexingSystem::new(points, KDTreeConfig::high_performance()).unwrap();
    
    let neighbor_engine = BatchNeighborQueryEngine::new(
        spatial_index,
        ConnectionStrengthConfig::fast_calculation(),
        BatchQueryConfig::default(),
    );
    
    let center_pos = Position3D::new(250.0, 250.0, 150.0);
    let radius = 100.0;
    let min_strength = 0.05;
    
    // Warm up
    for _ in 0..10 {
        neighbor_engine.find_neighbors(0, &center_pos, radius, min_strength);
    }
    
    // Benchmark single query
    let start = Instant::now();
    let neighbors = neighbor_engine.find_neighbors(0, &center_pos, radius, min_strength);
    let query_time = start.elapsed();
    
    println!("Single neighbor query: {} ns", query_time.as_nanos());
    println!("Found {} neighbors", neighbors.len());
    
    // Performance target: < 1μs
    assert!(query_time.as_nanos() < 1_000_000, "Query took {} ns (target: < 1μs)", query_time.as_nanos());
    
    // Test batch performance
    let query_centers: Vec<(u32, Position3D)> = (0..50)
        .map(|i| (i, Position3D::new(i as f32 * 20.0, i as f32 * 20.0, (i % 5) as f32 * 50.0)))
        .collect();
    
    let start = Instant::now();
    let batch_result = neighbor_engine.batch_find_neighbors(&query_centers, radius, min_strength);
    let batch_time = start.elapsed();
    
    let avg_time_per_query = batch_time.as_nanos() / query_centers.len() as u128;
    println!("Batch query average: {} ns per query", avg_time_per_query);
    
    // Batch should be significantly faster per query
    assert!(avg_time_per_query < 500_000, "Batch query average {} ns (target: < 500ns)", avg_time_per_query);
    
    let metrics = neighbor_engine.performance_metrics();
    println!("Overall metrics: {:?}", metrics.query_stats);
}

// Helper functions
fn generate_test_points(count: usize) -> Vec<SpatialPoint> {
    let mut points = Vec::with_capacity(count);
    
    for i in 0..count {
        let x = (i % 100) as f32 * 5.0;
        let y = ((i / 100) % 100) as f32 * 5.0;
        let z = (i / 10000) as f32 * 100.0;
        
        points.push(crate::SpatialPoint::new(
            Position3D::new(x, y, z),
            i as u32,
        ));
    }
    
    points
}

fn generate_grid_points(width: u32, height: u32, depth: u32) -> Vec<SpatialPoint> {
    let mut points = Vec::new();
    let mut index = 0;
    
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let position = Position3D::new(
                    x as f32 * 10.0,
                    y as f32 * 10.0,
                    z as f32 * 50.0,
                );
                
                points.push(crate::SpatialPoint::new(position, index));
                index += 1;
            }
        }
    }
    
    points
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 7/7 neighbor finding tests passing
2. **Performance targets met**:
   - Single neighbor query < 1μs average case
   - Batch queries 10x faster than individual
   - Distance accuracy ±0.1%
   - Connection strength biological curves
3. **SIMD optimization**: Distance calculations use SIMD when available
4. **Caching effectiveness**: Significant speedup from caching mechanisms

## Verification Commands

```bash
# Run neighbor finding tests
cargo test neighbor_finding_test --release -- --nocapture

# Performance targets test
cargo test test_performance_targets --release -- --nocapture

# Batch processing test
cargo test test_batch_neighbor_query_performance --release -- --nocapture

# Accuracy verification
cargo test test_connection_strength_calculation --release -- --nocapture
```

## Files to Create

1. `src/distance_optimization.rs`
2. `src/connection_strength.rs`
3. `src/batch_neighbor_queries.rs`
4. `tests/neighbor_finding_test.rs`

## Expected Completion Time

2 hours for an AI assistant:
- 45 minutes: Distance calculation optimization with SIMD
- 45 minutes: Connection strength modeling and biological curves
- 30 minutes: Batch query engine and caching systems

## Next Task

Task 1.13: Parallel Allocation Engine (multi-threaded allocation with SIMD acceleration)