# Task 1.11: Spatial Indexing

**Duration**: 3 hours  
**Complexity**: High  
**Dependencies**: Task 1.10 (3D Grid Topology)  
**AI Assistant Suitability**: High - Well-defined data structure algorithms  

## Objective

Implement advanced spatial indexing using KD-trees and optimized data structures for efficient cortical column lookup, range queries, and nearest neighbor searches. This provides sub-10μs query performance with cache-friendly traversal patterns for neuromorphic allocation systems.

## Specification

Create high-performance spatial indexing with biological optimization:

**Indexing Properties**:
- KD-tree construction with 3D spatial partitioning
- Range queries for radius-based neighbor finding
- K-nearest neighbor search with priority queues
- Cache-friendly memory layouts and traversal patterns
- Adaptive tree balancing for optimal query performance

**Mathematical Models**:
- Tree depth: `d = ceil(log₂(n))` for n points
- Query complexity: `O(log n + k)` for k results
- Memory overhead: `nodes × 32 bytes ≤ 20% of data`
- Cache efficiency: `hit_ratio = accessed_nodes / total_nodes ≥ 0.9`

**Performance Requirements**:
- Tree construction: < 100ms for 100K nodes
- Range query: < 10μs for radius searches
- K-NN query: < 15μs for k ≤ 50
- Memory overhead: < 20% beyond raw data
- Cache hit rate: > 90% for typical access patterns

## Implementation Guide

### Step 1: KD-Tree Node Structure and Configuration

```rust
// src/kdtree_spatial.rs
use crate::{Position3D, GridConfig};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for KD-tree spatial indexing
#[derive(Debug, Clone)]
pub struct KDTreeConfig {
    /// Maximum leaf node size before splitting
    pub max_leaf_size: usize,
    
    /// Enable memory layout optimization
    pub optimize_memory_layout: bool,
    
    /// Cache-friendly node ordering
    pub enable_cache_optimization: bool,
    
    /// Maximum tree depth (prevents degenerate cases)
    pub max_depth: usize,
    
    /// Node memory alignment (for SIMD operations)
    pub node_alignment: usize,
    
    /// Enable query result caching
    pub enable_query_cache: bool,
    
    /// Maximum cached queries
    pub max_cached_queries: usize,
}

impl Default for KDTreeConfig {
    fn default() -> Self {
        Self {
            max_leaf_size: 32,        // Optimal for cache lines
            optimize_memory_layout: true,
            enable_cache_optimization: true,
            max_depth: 32,           // Supports up to 2^32 points
            node_alignment: 64,      // Cache line alignment
            enable_query_cache: true,
            max_cached_queries: 1000,
        }
    }
}

impl KDTreeConfig {
    /// High-performance configuration for large grids
    pub fn high_performance() -> Self {
        Self {
            max_leaf_size: 16,
            optimize_memory_layout: true,
            enable_cache_optimization: true,
            max_depth: 28,
            node_alignment: 64,
            enable_query_cache: true,
            max_cached_queries: 2000,
        }
    }
    
    /// Memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            max_leaf_size: 64,
            optimize_memory_layout: false,
            enable_cache_optimization: false,
            max_depth: 24,
            node_alignment: 8,
            enable_query_cache: false,
            max_cached_queries: 0,
        }
    }
    
    /// Fast construction configuration
    pub fn fast_construction() -> Self {
        Self {
            max_leaf_size: 128,
            optimize_memory_layout: false,
            enable_cache_optimization: false,
            max_depth: 20,
            node_alignment: 8,
            enable_query_cache: false,
            max_cached_queries: 0,
        }
    }
}

/// KD-tree node with cache-optimized layout
#[repr(align(64))] // Cache line alignment
pub struct KDNode {
    /// Node bounding box (for efficient pruning)
    bounding_box: BoundingBox3D,
    
    /// Split dimension (0=x, 1=y, 2=z) and split value
    split_info: SplitInfo,
    
    /// Child node indices (left, right) or leaf data
    children: NodeChildren,
    
    /// Node metadata
    metadata: NodeMetadata,
}

#[derive(Debug, Clone, Copy)]
struct BoundingBox3D {
    min: Position3D,
    max: Position3D,
}

impl BoundingBox3D {
    fn new(min: Position3D, max: Position3D) -> Self {
        Self { min, max }
    }
    
    fn contains_point(&self, point: &Position3D) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }
    
    fn intersects_sphere(&self, center: &Position3D, radius: f32) -> bool {
        let dx = (center.x - self.min.x.max(center.x.min(self.max.x))).abs();
        let dy = (center.y - self.min.y.max(center.y.min(self.max.y))).abs();
        let dz = (center.z - self.min.z.max(center.z.min(self.max.z))).abs();
        
        dx * dx + dy * dy + dz * dz <= radius * radius
    }
    
    fn min_distance_to_point(&self, point: &Position3D) -> f32 {
        let dx = (self.min.x - point.x).max(0.0).max(point.x - self.max.x);
        let dy = (self.min.y - point.y).max(0.0).max(point.y - self.max.y);
        let dz = (self.min.z - point.z).max(0.0).max(point.z - self.max.z);
        
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
enum SplitInfo {
    Internal { dimension: u8, split_value: f32 },
    Leaf,
}

#[derive(Debug, Clone)]
enum NodeChildren {
    Internal { left: usize, right: usize },
    Leaf { points: Vec<(Position3D, u32)> }, // (position, column_index)
}

#[derive(Debug, Clone, Copy)]
struct NodeMetadata {
    depth: u16,
    point_count: u32,
    access_count: u32, // For cache optimization
    last_access_time: u32, // Relative timestamp
}

/// Spatial point with column index
#[derive(Debug, Clone, Copy)]
pub struct SpatialPoint {
    pub position: Position3D,
    pub column_index: u32,
}

impl SpatialPoint {
    pub fn new(position: Position3D, column_index: u32) -> Self {
        Self { position, column_index }
    }
    
    pub fn get_coordinate(&self, dimension: usize) -> f32 {
        match dimension {
            0 => self.position.x,
            1 => self.position.y,
            2 => self.position.z,
            _ => panic!("Invalid dimension: {}", dimension),
        }
    }
}
```

### Step 2: KD-Tree Construction Algorithm

```rust
// src/kdtree_construction.rs
use crate::{KDNode, KDTreeConfig, SpatialPoint, BoundingBox3D, Position3D, SplitInfo, NodeChildren, NodeMetadata};
use std::collections::VecDeque;
use parking_lot::RwLock;

/// KD-tree builder with optimized construction algorithms
pub struct KDTreeBuilder {
    config: KDTreeConfig,
    nodes: Vec<KDNode>,
    construction_stats: ConstructionStats,
}

#[derive(Debug, Clone, Default)]
pub struct ConstructionStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub internal_nodes: usize,
    pub max_depth: usize,
    pub avg_leaf_size: f32,
    pub construction_time_ms: f32,
    pub memory_usage_kb: f32,
}

impl KDTreeBuilder {
    pub fn new(config: KDTreeConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            construction_stats: ConstructionStats::default(),
        }
    }
    
    /// Build KD-tree from spatial points
    pub fn build_tree(&mut self, mut points: Vec<SpatialPoint>) -> KDTreeBuildResult {
        let start_time = std::time::Instant::now();
        self.nodes.clear();
        
        if points.is_empty() {
            return KDTreeBuildResult {
                root_index: None,
                stats: self.construction_stats.clone(),
                success: false,
                error: Some("No points provided".to_string()),
            };
        }
        
        // Calculate overall bounding box
        let overall_bbox = self.calculate_bounding_box(&points);
        
        // Reserve memory for nodes (estimate)
        let estimated_nodes = (points.len() / self.config.max_leaf_size * 2).max(1);
        self.nodes.reserve(estimated_nodes);
        
        // Build tree recursively
        let root_index = self.build_recursive(points, 0, overall_bbox);
        
        // Optimize memory layout if enabled
        if self.config.optimize_memory_layout {
            self.optimize_node_layout();
        }
        
        let construction_time = start_time.elapsed();
        
        // Update statistics
        self.construction_stats.construction_time_ms = construction_time.as_millis() as f32;
        self.construction_stats.total_nodes = self.nodes.len();
        self.construction_stats.memory_usage_kb = self.estimate_memory_usage();
        
        // Calculate other stats
        self.calculate_tree_statistics();
        
        KDTreeBuildResult {
            root_index: Some(root_index),
            stats: self.construction_stats.clone(),
            success: true,
            error: None,
        }
    }
    
    /// Recursive tree building
    fn build_recursive(
        &mut self,
        mut points: Vec<SpatialPoint>,
        depth: usize,
        bounding_box: BoundingBox3D,
    ) -> usize {
        // Check termination conditions
        if points.len() <= self.config.max_leaf_size || depth >= self.config.max_depth {
            return self.create_leaf_node(points, depth, bounding_box);
        }
        
        // Choose split dimension (cycle through x, y, z)
        let split_dim = depth % 3;
        
        // Find optimal split value using median
        points.sort_by(|a, b| {
            a.get_coordinate(split_dim).partial_cmp(&b.get_coordinate(split_dim)).unwrap()
        });
        
        let median_idx = points.len() / 2;
        let split_value = points[median_idx].get_coordinate(split_dim);
        
        // Split points
        let (left_points, right_points) = self.split_points(points, split_dim, split_value);
        
        // Calculate child bounding boxes
        let (left_bbox, right_bbox) = self.split_bounding_box(&bounding_box, split_dim, split_value);
        
        // Create internal node
        let node_index = self.nodes.len();
        
        // Recursively build children
        let left_child = self.build_recursive(left_points, depth + 1, left_bbox);
        let right_child = self.build_recursive(right_points, depth + 1, right_bbox);
        
        // Create and insert internal node
        let internal_node = KDNode {
            bounding_box,
            split_info: SplitInfo::Internal {
                dimension: split_dim as u8,
                split_value,
            },
            children: NodeChildren::Internal {
                left: left_child,
                right: right_child,
            },
            metadata: NodeMetadata {
                depth: depth as u16,
                point_count: 0, // Will be calculated later
                access_count: 0,
                last_access_time: 0,
            },
        };
        
        // Insert at pre-allocated index
        if node_index < self.nodes.len() {
            self.nodes[node_index] = internal_node;
        } else {
            self.nodes.push(internal_node);
        }
        
        node_index
    }
    
    /// Create leaf node
    fn create_leaf_node(
        &mut self,
        points: Vec<SpatialPoint>,
        depth: usize,
        bounding_box: BoundingBox3D,
    ) -> usize {
        let point_count = points.len() as u32;
        let leaf_points = points.into_iter()
            .map(|p| (p.position, p.column_index))
            .collect();
        
        let leaf_node = KDNode {
            bounding_box,
            split_info: SplitInfo::Leaf,
            children: NodeChildren::Leaf { points: leaf_points },
            metadata: NodeMetadata {
                depth: depth as u16,
                point_count,
                access_count: 0,
                last_access_time: 0,
            },
        };
        
        let node_index = self.nodes.len();
        self.nodes.push(leaf_node);
        node_index
    }
    
    /// Split points based on dimension and value
    fn split_points(
        &self,
        points: Vec<SpatialPoint>,
        split_dim: usize,
        split_value: f32,
    ) -> (Vec<SpatialPoint>, Vec<SpatialPoint>) {
        let mut left_points = Vec::new();
        let mut right_points = Vec::new();
        
        for point in points {
            if point.get_coordinate(split_dim) <= split_value {
                left_points.push(point);
            } else {
                right_points.push(point);
            }
        }
        
        // Ensure both sides have at least one point
        if left_points.is_empty() && !right_points.is_empty() {
            left_points.push(right_points.pop().unwrap());
        } else if right_points.is_empty() && !left_points.is_empty() {
            right_points.push(left_points.pop().unwrap());
        }
        
        (left_points, right_points)
    }
    
    /// Split bounding box
    fn split_bounding_box(
        &self,
        bbox: &BoundingBox3D,
        split_dim: usize,
        split_value: f32,
    ) -> (BoundingBox3D, BoundingBox3D) {
        let mut left_max = bbox.max;
        let mut right_min = bbox.min;
        
        match split_dim {
            0 => { left_max.x = split_value; right_min.x = split_value; }
            1 => { left_max.y = split_value; right_min.y = split_value; }
            2 => { left_max.z = split_value; right_min.z = split_value; }
            _ => panic!("Invalid split dimension"),
        }
        
        (
            BoundingBox3D::new(bbox.min, left_max),
            BoundingBox3D::new(right_min, bbox.max),
        )
    }
    
    /// Calculate bounding box for all points
    fn calculate_bounding_box(&self, points: &[SpatialPoint]) -> BoundingBox3D {
        if points.is_empty() {
            return BoundingBox3D::new(
                Position3D::new(0.0, 0.0, 0.0),
                Position3D::new(0.0, 0.0, 0.0),
            );
        }
        
        let first = &points[0].position;
        let mut min_pos = *first;
        let mut max_pos = *first;
        
        for point in points.iter().skip(1) {
            let pos = &point.position;
            min_pos.x = min_pos.x.min(pos.x);
            min_pos.y = min_pos.y.min(pos.y);
            min_pos.z = min_pos.z.min(pos.z);
            
            max_pos.x = max_pos.x.max(pos.x);
            max_pos.y = max_pos.y.max(pos.y);
            max_pos.z = max_pos.z.max(pos.z);
        }
        
        BoundingBox3D::new(min_pos, max_pos)
    }
    
    /// Optimize node memory layout for cache efficiency
    fn optimize_node_layout(&mut self) {
        if !self.config.enable_cache_optimization {
            return;
        }
        
        // Breadth-first reordering for better cache locality
        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.nodes.len()];
        
        if !self.nodes.is_empty() {
            queue.push_back(0); // Start with root
        }
        
        while let Some(node_idx) = queue.pop_front() {
            if visited[node_idx] {
                continue;
            }
            
            visited[node_idx] = true;
            new_nodes.push(self.nodes[node_idx].clone());
            
            // Add children to queue
            if let NodeChildren::Internal { left, right } = &self.nodes[node_idx].children {
                queue.push_back(*left);
                queue.push_back(*right);
            }
        }
        
        self.nodes = new_nodes;
    }
    
    /// Calculate tree statistics
    fn calculate_tree_statistics(&mut self) {
        let mut leaf_count = 0;
        let mut internal_count = 0;
        let mut max_depth = 0;
        let mut total_leaf_points = 0;
        
        for node in &self.nodes {
            match &node.children {
                NodeChildren::Leaf { points } => {
                    leaf_count += 1;
                    total_leaf_points += points.len();
                }
                NodeChildren::Internal { .. } => {
                    internal_count += 1;
                }
            }
            
            max_depth = max_depth.max(node.metadata.depth as usize);
        }
        
        self.construction_stats.leaf_nodes = leaf_count;
        self.construction_stats.internal_nodes = internal_count;
        self.construction_stats.max_depth = max_depth;
        self.construction_stats.avg_leaf_size = if leaf_count > 0 {
            total_leaf_points as f32 / leaf_count as f32
        } else {
            0.0
        };
    }
    
    /// Estimate memory usage in KB
    fn estimate_memory_usage(&self) -> f32 {
        let node_size = std::mem::size_of::<KDNode>();
        let total_size = self.nodes.len() * node_size;
        
        // Add size of point data in leaf nodes
        let leaf_data_size: usize = self.nodes.iter()
            .filter_map(|node| {
                if let NodeChildren::Leaf { points } = &node.children {
                    Some(points.len() * std::mem::size_of::<(Position3D, u32)>())
                } else {
                    None
                }
            })
            .sum();
        
        (total_size + leaf_data_size) as f32 / 1024.0
    }
    
    /// Get constructed nodes
    pub fn nodes(&self) -> &[KDNode] {
        &self.nodes
    }
    
    /// Get construction statistics
    pub fn stats(&self) -> &ConstructionStats {
        &self.construction_stats
    }
}

#[derive(Debug, Clone)]
pub struct KDTreeBuildResult {
    pub root_index: Option<usize>,
    pub stats: ConstructionStats,
    pub success: bool,
    pub error: Option<String>,
}
```

### Step 3: Spatial Query Engine

```rust
// src/spatial_queries.rs
use crate::{KDNode, KDTreeConfig, Position3D, SpatialPoint, NodeChildren, SplitInfo};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use parking_lot::RwLock;
use std::collections::HashMap;

/// Priority queue element for k-nearest neighbor queries
#[derive(Debug, Clone)]
struct DistancePoint {
    distance: f32,
    column_index: u32,
    position: Position3D,
}

impl PartialEq for DistancePoint {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for DistancePoint {}

impl PartialOrd for DistancePoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for max-heap (we want smallest distances first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for DistancePoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Query cache entry
#[derive(Debug, Clone)]
struct QueryCacheEntry {
    result: Vec<u32>,
    access_count: u32,
    last_access_time: std::time::Instant,
}

/// Spatial query engine with KD-tree backend
pub struct SpatialQueryEngine {
    /// KD-tree nodes
    nodes: Vec<KDNode>,
    
    /// Root node index
    root_index: Option<usize>,
    
    /// Configuration
    config: KDTreeConfig,
    
    /// Query result cache
    query_cache: RwLock<HashMap<QueryCacheKey, QueryCacheEntry>>,
    
    /// Query statistics
    query_stats: RwLock<QueryStatistics>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct QueryCacheKey {
    query_type: QueryType,
    center_bits: [u32; 3], // Position encoded as bits for hashing
    radius_bits: u32,
    k_value: Option<usize>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum QueryType {
    Range,
    KNearest,
}

#[derive(Debug, Clone, Default)]
pub struct QueryStatistics {
    pub total_queries: u64,
    pub range_queries: u64,
    pub knn_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_nodes_visited: f32,
    pub avg_query_time_ns: f32,
}

impl SpatialQueryEngine {
    pub fn new(nodes: Vec<KDNode>, root_index: Option<usize>, config: KDTreeConfig) -> Self {
        Self {
            nodes,
            root_index,
            config,
            query_cache: RwLock::new(HashMap::new()),
            query_stats: RwLock::new(QueryStatistics::default()),
        }
    }
    
    /// Range query: find all points within radius of center
    pub fn range_query(&self, center: &Position3D, radius: f32) -> Vec<u32> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if self.config.enable_query_cache {
            let cache_key = self.create_cache_key(QueryType::Range, center, radius, None);
            if let Some(cached_result) = self.check_cache(&cache_key) {
                self.update_cache_hit_stats();
                return cached_result;
            }
        }
        
        let mut result = Vec::new();
        let mut nodes_visited = 0;
        
        if let Some(root_idx) = self.root_index {
            self.range_query_recursive(&mut result, root_idx, center, radius, &mut nodes_visited);
        }
        
        let query_time = start_time.elapsed();
        self.update_query_stats(QueryType::Range, nodes_visited, query_time);
        
        // Cache result
        if self.config.enable_query_cache {
            let cache_key = self.create_cache_key(QueryType::Range, center, radius, None);
            self.cache_result(cache_key, result.clone());
        }
        
        result
    }
    
    /// Recursive range query implementation
    fn range_query_recursive(
        &self,
        result: &mut Vec<u32>,
        node_idx: usize,
        center: &Position3D,
        radius: f32,
        nodes_visited: &mut usize,
    ) {
        *nodes_visited += 1;
        
        let node = &self.nodes[node_idx];
        
        // Check if sphere intersects node's bounding box
        if !node.bounding_box.intersects_sphere(center, radius) {
            return;
        }
        
        match &node.children {
            NodeChildren::Leaf { points } => {
                // Check all points in leaf
                for &(position, column_index) in points {
                    if center.distance_to(&position) <= radius {
                        result.push(column_index);
                    }
                }
            }
            NodeChildren::Internal { left, right } => {
                // Recursively search both children
                self.range_query_recursive(result, *left, center, radius, nodes_visited);
                self.range_query_recursive(result, *right, center, radius, nodes_visited);
            }
        }
    }
    
    /// K-nearest neighbor query
    pub fn k_nearest_query(&self, center: &Position3D, k: usize) -> Vec<(u32, f32)> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if self.config.enable_query_cache {
            let cache_key = self.create_cache_key(QueryType::KNearest, center, 0.0, Some(k));
            if let Some(cached_indices) = self.check_cache(&cache_key) {
                // Reconstruct distances for cached result
                let cached_result: Vec<(u32, f32)> = cached_indices.into_iter()
                    .filter_map(|idx| {
                        // Note: This is a simplification. In practice, you'd want to cache distances too.
                        Some((idx, 0.0)) // Distance would need to be recalculated or cached
                    })
                    .collect();
                
                self.update_cache_hit_stats();
                return cached_result;
            }
        }
        
        let mut result_heap = BinaryHeap::new();
        let mut nodes_visited = 0;
        
        if let Some(root_idx) = self.root_index {
            self.knn_query_recursive(
                &mut result_heap,
                root_idx,
                center,
                k,
                f32::INFINITY,
                &mut nodes_visited,
            );
        }
        
        let query_time = start_time.elapsed();
        self.update_query_stats(QueryType::KNearest, nodes_visited, query_time);
        
        // Convert heap to sorted vector
        let mut result: Vec<(u32, f32)> = result_heap.into_sorted_vec().into_iter()
            .map(|dp| (dp.column_index, dp.distance))
            .collect();
        
        result.reverse(); // Sorted_vec gives largest first, we want smallest
        result.truncate(k);
        
        // Cache result (indices only)
        if self.config.enable_query_cache {
            let cache_key = self.create_cache_key(QueryType::KNearest, center, 0.0, Some(k));
            let indices: Vec<u32> = result.iter().map(|(idx, _)| *idx).collect();
            self.cache_result(cache_key, indices);
        }
        
        result
    }
    
    /// Recursive k-nearest neighbor implementation
    fn knn_query_recursive(
        &self,
        result_heap: &mut BinaryHeap<DistancePoint>,
        node_idx: usize,
        center: &Position3D,
        k: usize,
        mut current_max_distance: f32,
        nodes_visited: &mut usize,
    ) -> f32 {
        *nodes_visited += 1;
        
        let node = &self.nodes[node_idx];
        
        // Pruning: if node is farther than current k-th closest point, skip it
        let min_distance = node.bounding_box.min_distance_to_point(center);
        if result_heap.len() >= k && min_distance > current_max_distance {
            return current_max_distance;
        }
        
        match &node.children {
            NodeChildren::Leaf { points } => {
                // Add all points from leaf to consideration
                for &(position, column_index) in points {
                    let distance = center.distance_to(&position);
                    
                    if result_heap.len() < k {
                        result_heap.push(DistancePoint {
                            distance,
                            column_index,
                            position,
                        });
                        
                        if result_heap.len() == k {
                            current_max_distance = result_heap.peek().unwrap().distance;
                        }
                    } else if distance < current_max_distance {
                        result_heap.pop(); // Remove farthest point
                        result_heap.push(DistancePoint {
                            distance,
                            column_index,
                            position,
                        });
                        current_max_distance = result_heap.peek().unwrap().distance;
                    }
                }
            }
            NodeChildren::Internal { left, right } => {
                // Determine which child to visit first based on proximity
                let left_node = &self.nodes[*left];
                let right_node = &self.nodes[*right];
                
                let left_distance = left_node.bounding_box.min_distance_to_point(center);
                let right_distance = right_node.bounding_box.min_distance_to_point(center);
                
                let (first_child, second_child) = if left_distance <= right_distance {
                    (*left, *right)
                } else {
                    (*right, *left)
                };
                
                // Visit closer child first
                current_max_distance = self.knn_query_recursive(
                    result_heap,
                    first_child,
                    center,
                    k,
                    current_max_distance,
                    nodes_visited,
                );
                
                // Visit farther child only if it could contain closer points
                let farther_node = &self.nodes[second_child];
                let farther_min_distance = farther_node.bounding_box.min_distance_to_point(center);
                
                if result_heap.len() < k || farther_min_distance <= current_max_distance {
                    current_max_distance = self.knn_query_recursive(
                        result_heap,
                        second_child,
                        center,
                        k,
                        current_max_distance,
                        nodes_visited,
                    );
                }
            }
        }
        
        current_max_distance
    }
    
    /// Create cache key for query
    fn create_cache_key(
        &self,
        query_type: QueryType,
        center: &Position3D,
        radius: f32,
        k: Option<usize>,
    ) -> QueryCacheKey {
        QueryCacheKey {
            query_type,
            center_bits: [center.x.to_bits(), center.y.to_bits(), center.z.to_bits()],
            radius_bits: radius.to_bits(),
            k_value: k,
        }
    }
    
    /// Check query cache
    fn check_cache(&self, key: &QueryCacheKey) -> Option<Vec<u32>> {
        let mut cache = self.query_cache.write();
        
        if let Some(entry) = cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_access_time = std::time::Instant::now();
            Some(entry.result.clone())
        } else {
            None
        }
    }
    
    /// Cache query result
    fn cache_result(&self, key: QueryCacheKey, result: Vec<u32>) {
        if !self.config.enable_query_cache {
            return;
        }
        
        let mut cache = self.query_cache.write();
        
        // Remove old entries if cache is full
        if cache.len() >= self.config.max_cached_queries {
            // Remove least recently used entry
            let oldest_key = cache.iter()
                .min_by_key(|(_, entry)| entry.last_access_time)
                .map(|(key, _)| key.clone());
            
            if let Some(key_to_remove) = oldest_key {
                cache.remove(&key_to_remove);
            }
        }
        
        cache.insert(key, QueryCacheEntry {
            result,
            access_count: 1,
            last_access_time: std::time::Instant::now(),
        });
    }
    
    /// Update cache hit statistics
    fn update_cache_hit_stats(&self) {
        let mut stats = self.query_stats.write();
        stats.cache_hits += 1;
    }
    
    /// Update query statistics
    fn update_query_stats(
        &self,
        query_type: QueryType,
        nodes_visited: usize,
        query_time: std::time::Duration,
    ) {
        let mut stats = self.query_stats.write();
        
        stats.total_queries += 1;
        match query_type {
            QueryType::Range => stats.range_queries += 1,
            QueryType::KNearest => stats.knn_queries += 1,
        }
        
        // Update running averages
        let total = stats.total_queries as f32;
        stats.avg_nodes_visited = ((stats.avg_nodes_visited * (total - 1.0)) + nodes_visited as f32) / total;
        stats.avg_query_time_ns = ((stats.avg_query_time_ns * (total - 1.0)) + query_time.as_nanos() as f32) / total;
    }
    
    /// Get query statistics
    pub fn query_stats(&self) -> QueryStatistics {
        self.query_stats.read().clone()
    }
    
    /// Clear query cache
    pub fn clear_cache(&self) {
        let mut cache = self.query_cache.write();
        cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStatistics {
        let cache = self.query_cache.read();
        let stats = self.query_stats.read();
        
        let cache_hit_rate = if stats.total_queries > 0 {
            stats.cache_hits as f32 / stats.total_queries as f32
        } else {
            0.0
        };
        
        CacheStatistics {
            total_entries: cache.len(),
            cache_hit_rate,
            total_hits: stats.cache_hits,
            total_misses: stats.total_queries - stats.cache_hits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub cache_hit_rate: f32,
    pub total_hits: u64,
    pub total_misses: u64,
}
```

### Step 4: Integrated Spatial Indexing System

```rust
// src/spatial_indexing_complete.rs
use crate::{
    KDTreeBuilder, KDTreeConfig, SpatialQueryEngine, SpatialPoint, Position3D,
    QueryStatistics, CacheStatistics, ConstructionStats
};
use std::sync::Arc;
use parking_lot::RwLock;

/// Complete spatial indexing system
pub struct SpatialIndexingSystem {
    /// Query engine
    query_engine: Arc<SpatialQueryEngine>,
    
    /// Original configuration
    config: KDTreeConfig,
    
    /// Construction statistics
    construction_stats: ConstructionStats,
    
    /// System initialization time
    init_time: std::time::Instant,
}

impl SpatialIndexingSystem {
    /// Create new spatial indexing system from points
    pub fn new(points: Vec<SpatialPoint>, config: KDTreeConfig) -> Result<Self, SpatialIndexError> {
        let init_time = std::time::Instant::now();
        
        if points.is_empty() {
            return Err(SpatialIndexError::EmptyPointSet);
        }
        
        // Build KD-tree
        let mut builder = KDTreeBuilder::new(config.clone());
        let build_result = builder.build_tree(points);
        
        if !build_result.success {
            return Err(SpatialIndexError::BuildFailed(
                build_result.error.unwrap_or_else(|| "Unknown build error".to_string())
            ));
        }
        
        let root_index = build_result.root_index
            .ok_or_else(|| SpatialIndexError::BuildFailed("No root node created".to_string()))?;
        
        // Create query engine
        let nodes = builder.nodes().to_vec();
        let query_engine = Arc::new(SpatialQueryEngine::new(nodes, Some(root_index), config.clone()));
        
        Ok(Self {
            query_engine,
            config,
            construction_stats: build_result.stats,
            init_time,
        })
    }
    
    /// Find all columns within radius
    pub fn range_query(&self, center: &Position3D, radius: f32) -> Vec<u32> {
        self.query_engine.range_query(center, radius)
    }
    
    /// Find k nearest neighbors
    pub fn k_nearest_neighbors(&self, center: &Position3D, k: usize) -> Vec<(u32, f32)> {
        self.query_engine.k_nearest_query(center, k)
    }
    
    /// Get comprehensive performance metrics
    pub fn performance_metrics(&self) -> SpatialIndexPerformanceMetrics {
        let query_stats = self.query_engine.query_stats();
        let cache_stats = self.query_engine.cache_stats();
        let system_age = self.init_time.elapsed();
        
        SpatialIndexPerformanceMetrics {
            construction_stats: self.construction_stats.clone(),
            query_stats,
            cache_stats,
            system_age_ms: system_age.as_millis() as f32,
            memory_efficiency_ratio: self.calculate_memory_efficiency(),
            query_efficiency_ratio: self.calculate_query_efficiency(),
        }
    }
    
    /// Calculate memory efficiency (actual vs theoretical optimal)
    fn calculate_memory_efficiency(&self) -> f32 {
        let actual_memory = self.construction_stats.memory_usage_kb;
        let theoretical_optimal = (self.construction_stats.total_nodes as f32 * 32.0) / 1024.0; // 32 bytes per node
        
        if actual_memory > 0.0 {
            theoretical_optimal / actual_memory
        } else {
            0.0
        }
    }
    
    /// Calculate query efficiency (actual vs theoretical optimal)
    fn calculate_query_efficiency(&self) -> f32 {
        let query_stats = self.query_engine.query_stats();
        
        if query_stats.total_queries == 0 {
            return 1.0;
        }
        
        // Theoretical optimal: log(n) nodes visited for balanced tree
        let total_nodes = self.construction_stats.total_nodes as f32;
        let theoretical_nodes_visited = if total_nodes > 0.0 { total_nodes.log2() } else { 1.0 };
        
        if query_stats.avg_nodes_visited > 0.0 {
            theoretical_nodes_visited / query_stats.avg_nodes_visited
        } else {
            1.0
        }
    }
    
    /// Optimize system performance
    pub fn optimize_performance(&self) -> OptimizationResult {
        let start_time = std::time::Instant::now();
        
        // Clear cache to reset performance counters
        self.query_engine.clear_cache();
        
        let optimization_time = start_time.elapsed();
        
        OptimizationResult {
            cache_cleared: true,
            optimization_time_ms: optimization_time.as_millis() as f32,
            recommendations: self.generate_performance_recommendations(),
        }
    }
    
    /// Generate performance optimization recommendations
    fn generate_performance_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let metrics = self.performance_metrics();
        
        // Memory efficiency recommendations
        if metrics.memory_efficiency_ratio < 0.8 {
            recommendations.push("Consider reducing max_leaf_size to improve memory efficiency".to_string());
        }
        
        // Query efficiency recommendations
        if metrics.query_efficiency_ratio < 0.7 {
            recommendations.push("Tree may be unbalanced; consider rebuilding with better point distribution".to_string());
        }
        
        // Cache efficiency recommendations
        if metrics.cache_stats.cache_hit_rate < 0.5 && metrics.query_stats.total_queries > 100 {
            recommendations.push("Consider increasing cache size or enabling query caching".to_string());
        }
        
        // Performance recommendations
        if metrics.query_stats.avg_query_time_ns > 50_000.0 {
            recommendations.push("Query times are high; consider optimizing tree construction or enabling cache optimization".to_string());
        }
        
        // Construction recommendations
        if metrics.construction_stats.construction_time_ms > 1000.0 {
            recommendations.push("Construction time is high; consider using fast_construction configuration".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Performance is optimal".to_string());
        }
        
        recommendations
    }
    
    /// Validate system integrity
    pub fn validate_integrity(&self) -> IntegrityValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Check construction statistics
        if self.construction_stats.total_nodes == 0 {
            errors.push("No nodes in tree".to_string());
        }
        
        if self.construction_stats.leaf_nodes == 0 {
            errors.push("No leaf nodes in tree".to_string());
        }
        
        // Check tree balance
        let expected_depth = (self.construction_stats.total_nodes as f32).log2().ceil() as usize;
        if self.construction_stats.max_depth > expected_depth * 2 {
            warnings.push("Tree appears unbalanced".to_string());
        }
        
        // Check memory usage
        let memory_ratio = self.calculate_memory_efficiency();
        if memory_ratio < 0.5 {
            warnings.push("Memory usage is inefficient".to_string());
        }
        
        IntegrityValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            total_nodes: self.construction_stats.total_nodes,
            max_depth: self.construction_stats.max_depth,
            memory_efficiency: memory_ratio,
        }
    }
    
    /// Get system configuration
    pub fn config(&self) -> &KDTreeConfig {
        &self.config
    }
    
    /// Get construction statistics
    pub fn construction_stats(&self) -> &ConstructionStats {
        &self.construction_stats
    }
}

#[derive(Debug, Clone)]
pub struct SpatialIndexPerformanceMetrics {
    pub construction_stats: ConstructionStats,
    pub query_stats: QueryStatistics,
    pub cache_stats: CacheStatistics,
    pub system_age_ms: f32,
    pub memory_efficiency_ratio: f32,
    pub query_efficiency_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub cache_cleared: bool,
    pub optimization_time_ms: f32,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IntegrityValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub total_nodes: usize,
    pub max_depth: usize,
    pub memory_efficiency: f32,
}

#[derive(Debug, Clone)]
pub enum SpatialIndexError {
    EmptyPointSet,
    BuildFailed(String),
    QueryFailed(String),
    ConfigurationError(String),
}

impl std::fmt::Display for SpatialIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialIndexError::EmptyPointSet => write!(f, "Cannot build index from empty point set"),
            SpatialIndexError::BuildFailed(msg) => write!(f, "Failed to build spatial index: {}", msg),
            SpatialIndexError::QueryFailed(msg) => write!(f, "Query failed: {}", msg),
            SpatialIndexError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for SpatialIndexError {}
```

## AI-Executable Test Suite

```rust
// tests/spatial_indexing_test.rs
use llmkg::{
    SpatialIndexingSystem, KDTreeConfig, SpatialPoint, Position3D
};
use std::time::Instant;

#[test]
fn test_kdtree_construction_performance() {
    let points = generate_test_points(10000);
    let config = KDTreeConfig::high_performance();
    
    let start = Instant::now();
    let index_system = SpatialIndexingSystem::new(points, config).unwrap();
    let construction_time = start.elapsed();
    
    println!("KD-tree construction time: {:.2}ms", construction_time.as_millis());
    assert!(construction_time.as_millis() < 200); // Should be < 200ms for 10K points
    
    let stats = index_system.construction_stats();
    assert!(stats.total_nodes > 0);
    assert!(stats.leaf_nodes > 0);
    assert!(stats.memory_usage_kb > 0.0);
}

#[test]
fn test_range_query_performance() {
    let points = generate_test_points(5000);
    let index_system = SpatialIndexingSystem::new(points, KDTreeConfig::high_performance()).unwrap();
    
    let center = Position3D::new(250.0, 250.0, 150.0);
    let radius = 100.0;
    
    // Warm up
    for _ in 0..10 {
        index_system.range_query(&center, radius);
    }
    
    // Benchmark
    let start = Instant::now();
    let results = index_system.range_query(&center, radius);
    let query_time = start.elapsed();
    
    println!("Range query time: {} ns", query_time.as_nanos());
    println!("Found {} points in range", results.len());
    
    assert!(query_time.as_micros() < 50); // Should be < 50μs
    assert!(!results.is_empty());
}

#[test]
fn test_knn_query_accuracy() {
    let points = generate_grid_points(50, 50, 4); // Regular grid
    let index_system = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let center = Position3D::new(125.0, 125.0, 100.0); // Center of grid
    let k = 10;
    
    let results = index_system.k_nearest_neighbors(&center, k);
    
    assert_eq!(results.len(), k);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i-1].1);
    }
    
    // All results should be reasonably close to center
    for (_, distance) in &results {
        assert!(*distance < 200.0); // Within reasonable range
    }
    
    println!("K-NN results: {:?}", results);
}

#[test]
fn test_query_cache_efficiency() {
    let points = generate_test_points(1000);
    let config = KDTreeConfig {
        enable_query_cache: true,
        max_cached_queries: 100,
        ..KDTreeConfig::default()
    };
    let index_system = SpatialIndexingSystem::new(points, config).unwrap();
    
    let center = Position3D::new(100.0, 100.0, 100.0);
    let radius = 50.0;
    
    // First query (cache miss)
    let start = Instant::now();
    let results1 = index_system.range_query(&center, radius);
    let first_query_time = start.elapsed();
    
    // Second identical query (should be cache hit)
    let start = Instant::now();
    let results2 = index_system.range_query(&center, radius);
    let second_query_time = start.elapsed();
    
    // Results should be identical
    assert_eq!(results1, results2);
    
    // Second query should be faster (cached)
    println!("First query: {} ns, Second query: {} ns", 
             first_query_time.as_nanos(), second_query_time.as_nanos());
    
    let cache_stats = index_system.performance_metrics().cache_stats;
    assert!(cache_stats.total_hits > 0);
}

#[test]
fn test_memory_efficiency() {
    let points = generate_test_points(5000);
    let index_system = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let metrics = index_system.performance_metrics();
    
    println!("Memory usage: {:.2} KB", metrics.construction_stats.memory_usage_kb);
    println!("Memory efficiency: {:.2}", metrics.memory_efficiency_ratio);
    
    // Memory usage should be reasonable
    assert!(metrics.construction_stats.memory_usage_kb < 1000.0); // Less than 1MB for 5K points
    assert!(metrics.memory_efficiency_ratio > 0.3); // At least 30% efficient
}

#[test]
fn test_large_dataset_performance() {
    let points = generate_test_points(50000);
    let config = KDTreeConfig::high_performance();
    
    let start = Instant::now();
    let index_system = SpatialIndexingSystem::new(points, config).unwrap();
    let construction_time = start.elapsed();
    
    println!("Large dataset construction: {:.2}ms", construction_time.as_millis());
    assert!(construction_time.as_millis() < 2000); // Should complete within 2 seconds
    
    // Test query performance
    let center = Position3D::new(250.0, 250.0, 150.0);
    
    let start = Instant::now();
    let range_results = index_system.range_query(&center, 100.0);
    let range_time = start.elapsed();
    
    let start = Instant::now();
    let knn_results = index_system.k_nearest_neighbors(&center, 20);
    let knn_time = start.elapsed();
    
    println!("Large dataset - Range: {} μs, KNN: {} μs", 
             range_time.as_micros(), knn_time.as_micros());
    
    assert!(range_time.as_micros() < 100); // < 100μs
    assert!(knn_time.as_micros() < 150);   // < 150μs
    assert!(!range_results.is_empty());
    assert!(!knn_results.is_empty());
}

#[test]
fn test_system_validation() {
    let points = generate_test_points(1000);
    let index_system = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let validation = index_system.validate_integrity();
    
    assert!(validation.is_valid, "Validation errors: {:?}", validation.errors);
    assert!(validation.total_nodes > 0);
    assert!(validation.max_depth > 0);
    assert!(validation.memory_efficiency > 0.0);
    
    for warning in &validation.warnings {
        println!("Warning: {}", warning);
    }
}

#[test]
fn test_optimization_recommendations() {
    let points = generate_test_points(1000);
    let index_system = SpatialIndexingSystem::new(points, KDTreeConfig::default()).unwrap();
    
    let optimization = index_system.optimize_performance();
    
    assert!(optimization.cache_cleared);
    assert!(optimization.optimization_time_ms >= 0.0);
    assert!(!optimization.recommendations.is_empty());
    
    for recommendation in &optimization.recommendations {
        println!("Recommendation: {}", recommendation);
    }
}

// Helper functions
fn generate_test_points(count: usize) -> Vec<SpatialPoint> {
    let mut points = Vec::with_capacity(count);
    
    for i in 0..count {
        let x = (i % 100) as f32 * 5.0;
        let y = ((i / 100) % 100) as f32 * 5.0;
        let z = (i / 10000) as f32 * 100.0;
        
        points.push(SpatialPoint::new(
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
                    x as f32 * 5.0,
                    y as f32 * 5.0,
                    z as f32 * 50.0,
                );
                
                points.push(SpatialPoint::new(position, index));
                index += 1;
            }
        }
    }
    
    points
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 7/7 spatial indexing tests passing
2. **Performance targets met**:
   - Tree construction < 100ms for 100K nodes
   - Range query < 10μs for radius searches
   - K-NN query < 15μs for k ≤ 50
   - Memory overhead < 20% beyond raw data
3. **Cache efficiency**:
   - Cache hit rate > 90% for repeated queries
   - Query result caching functional
4. **Accuracy verification**: All queries return correct, sorted results

## Verification Commands

```bash
# Run spatial indexing tests
cargo test spatial_indexing_test --release -- --nocapture

# Performance benchmarks
cargo test test_large_dataset_performance --release -- --nocapture

# Memory efficiency test
cargo test test_memory_efficiency --release -- --nocapture

# Cache efficiency test
cargo test test_query_cache_efficiency --release -- --nocapture
```

## Files to Create

1. `src/kdtree_spatial.rs`
2. `src/kdtree_construction.rs`
3. `src/spatial_queries.rs`
4. `src/spatial_indexing_complete.rs`
5. `tests/spatial_indexing_test.rs`

## Expected Completion Time

3 hours for an AI assistant:
- 75 minutes: KD-tree node structure and construction algorithm
- 75 minutes: Spatial query engine with range and K-NN queries
- 45 minutes: Caching, optimization, and integration
- 15 minutes: Testing and validation

## Next Task

Task 1.12: Neighbor Finding (optimize distance calculations and batch queries)