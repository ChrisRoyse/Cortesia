use crate::core::types::Relationship;
use crate::error::{GraphError, Result};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
pub struct CSRGraph {
    // CSR format arrays
    row_ptr: Vec<u64>,      // Row pointers (entity -> edge start index)
    col_idx: Vec<u32>,      // Column indices (target entities)
    rel_types: Vec<u8>,     // Relationship types
    weights: Vec<f32>,      // Edge weights
    
    // Metadata
    node_count: AtomicU32,
    edge_count: AtomicU32,
}

impl CSRGraph {
    pub fn new() -> Self {
        Self {
            row_ptr: vec![0],
            col_idx: Vec::new(),
            rel_types: Vec::new(),
            weights: Vec::new(),
            node_count: AtomicU32::new(0),
            edge_count: AtomicU32::new(0),
        }
    }
    
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            row_ptr: Vec::with_capacity(nodes + 1),
            col_idx: Vec::with_capacity(edges),
            rel_types: Vec::with_capacity(edges),
            weights: Vec::with_capacity(edges),
            node_count: AtomicU32::new(0),
            edge_count: AtomicU32::new(0),
        }
    }
    
    pub fn from_edges(edges: Vec<Relationship>, node_count: u32) -> Result<Self> {
        let mut adjacency_list: Vec<Vec<(u32, u8, f32)>> = vec![Vec::new(); node_count as usize];
        
        for edge in edges {
            let from_idx = edge.from.as_u32();
            let to_idx = edge.to.as_u32();
            if from_idx >= node_count || to_idx >= node_count {
                return Err(GraphError::EntityNotFound { id: from_idx.max(to_idx) });
            }
            adjacency_list[from_idx as usize].push((to_idx, edge.rel_type, edge.weight));
        }
        
        // Sort adjacency lists for better cache locality
        for list in &mut adjacency_list {
            list.sort_by_key(|&(to, _, _)| to);
        }
        
        // Build CSR format
        let mut row_ptr = Vec::with_capacity(node_count as usize + 1);
        let mut col_idx = Vec::new();
        let mut rel_types = Vec::new();
        let mut weights = Vec::new();
        
        row_ptr.push(0);
        
        for list in adjacency_list {
            for (to, rel_type, weight) in list {
                col_idx.push(to);
                rel_types.push(rel_type);
                weights.push(weight);
            }
            row_ptr.push(col_idx.len() as u64);
        }
        
        let edge_count = col_idx.len() as u32;
        Ok(Self {
            row_ptr,
            col_idx,
            rel_types,
            weights,
            node_count: AtomicU32::new(node_count),
            edge_count: AtomicU32::new(edge_count),
        })
    }
    
    #[inline]
    pub fn get_neighbors(&self, node_id: u32) -> &[u32] {
        if node_id >= self.node_count.load(Ordering::Relaxed) {
            return &[];
        }
        
        let start = self.row_ptr[node_id as usize] as usize;
        let end = self.row_ptr[node_id as usize + 1] as usize;
        &self.col_idx[start..end]
    }
    
    #[inline]
    pub fn get_edges(&self, node_id: u32) -> Vec<(u32, u8, f32)> {
        if node_id >= self.node_count.load(Ordering::Relaxed) {
            return Vec::new();
        }
        
        let start = self.row_ptr[node_id as usize] as usize;
        let end = self.row_ptr[node_id as usize + 1] as usize;
        
        (start..end).map(|i| {
            (self.col_idx[i], self.rel_types[i], self.weights[i])
        }).collect()
    }
    
    pub fn has_edge(&self, from: u32, to: u32) -> bool {
        let neighbors = self.get_neighbors(from);
        neighbors.binary_search(&to).is_ok()
    }
    
    pub fn degree(&self, node_id: u32) -> usize {
        if node_id >= self.node_count.load(Ordering::Relaxed) {
            return 0;
        }
        
        let start = self.row_ptr[node_id as usize];
        let end = self.row_ptr[node_id as usize + 1];
        (end - start) as usize
    }
    
    pub fn node_count(&self) -> u32 {
        self.node_count.load(Ordering::Relaxed)
    }
    
    pub fn edge_count(&self) -> u32 {
        self.edge_count.load(Ordering::Relaxed)
    }
    
    pub fn traverse_bfs(&self, start: u32, max_depth: u8) -> Vec<(u32, u8)> {
        let mut visited = vec![false; self.node_count.load(Ordering::Relaxed) as usize];
        let mut queue = std::collections::VecDeque::new();
        let mut result = Vec::new();
        
        queue.push_back((start, 0u8));
        visited[start as usize] = true;
        
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            
            result.push((node, depth));
            
            for &neighbor in self.get_neighbors(node) {
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        
        result
    }
    
    pub fn find_path(&self, start: u32, end: u32, max_depth: u8) -> Option<Vec<u32>> {
        if start == end {
            return Some(vec![start]);
        }
        
        let mut visited = vec![false; self.node_count.load(Ordering::Relaxed) as usize];
        let mut parent = vec![None; self.node_count.load(Ordering::Relaxed) as usize];
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back((start, 0u8));
        visited[start as usize] = true;
        
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            
            for &neighbor in self.get_neighbors(node) {
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    parent[neighbor as usize] = Some(node);
                    
                    if neighbor == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut current = end;
                        
                        while let Some(p) = parent[current as usize] {
                            path.push(p);
                            current = p;
                        }
                        
                        path.reverse();
                        return Some(path);
                    }
                    
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        
        None
    }
    
    pub fn memory_usage(&self) -> usize {
        self.row_ptr.capacity() * std::mem::size_of::<u64>() +
        self.col_idx.capacity() * std::mem::size_of::<u32>() +
        self.rel_types.capacity() * std::mem::size_of::<u8>() +
        self.weights.capacity() * std::mem::size_of::<f32>()
    }
    
    /// Get the capacity of the graph
    pub fn capacity(&self) -> usize {
        self.col_idx.capacity()
    }
    
    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: u32, to: u32, weight: f32) -> Result<()> {
        // Note: CSR format is typically immutable after construction
        // For dynamic updates, we would need to rebuild or use a different structure
        // This is a placeholder that returns an error
        Err(GraphError::UnsupportedOperation("CSRGraph does not support dynamic edge insertion. Use from_edges() to build.".to_string()))
    }
    
    /// Update an entity (not applicable for CSRGraph which stores edges)
    pub fn update_entity(&mut self, _id: u32, _data: Vec<u8>) -> Result<()> {
        Err(GraphError::UnsupportedOperation("CSRGraph does not store entity data".to_string()))
    }
    
    /// Remove an edge or entity
    pub fn remove(&mut self, _id: u32) -> Result<()> {
        Err(GraphError::UnsupportedOperation("CSRGraph does not support dynamic removal. Rebuild the graph.".to_string()))
    }
    
    /// Check if the graph contains an entity/node
    pub fn contains_entity(&self, id: u32) -> bool {
        id < self.node_count.load(Ordering::Relaxed)
    }
    
    /// Get encoded size of the graph
    pub fn encoded_size(&self) -> usize {
        // Calculate the size needed to serialize this graph
        std::mem::size_of::<u32>() * 2 + // node_count + edge_count
        self.row_ptr.len() * std::mem::size_of::<u64>() +
        self.col_idx.len() * std::mem::size_of::<u32>() +
        self.rel_types.len() * std::mem::size_of::<u8>() +
        self.weights.len() * std::mem::size_of::<f32>()
    }
    
    /// Get the weight of an edge
    pub fn get_edge_weight(&self, from: u32, to: u32) -> Option<f32> {
        if from >= self.node_count.load(Ordering::Relaxed) {
            return None;
        }
        
        let start = self.row_ptr[from as usize] as usize;
        let end = self.row_ptr[from as usize + 1] as usize;
        
        // Binary search for the target in sorted neighbors
        match self.col_idx[start..end].binary_search(&to) {
            Ok(idx) => Some(self.weights[start + idx]),
            Err(_) => None,
        }
    }
    
    /// Remove an edge (not supported in CSR format)
    pub fn remove_edge(&mut self, _from: u32, _to: u32) -> Result<()> {
        Err(GraphError::UnsupportedOperation("CSRGraph does not support dynamic edge removal. Rebuild the graph.".to_string()))
    }
    
    /// Update edge weight (not supported in CSR format)
    pub fn update_edge_weight(&mut self, _from: u32, _to: u32, _weight: f32) -> Result<()> {
        Err(GraphError::UnsupportedOperation("CSRGraph does not support dynamic weight updates. Rebuild the graph.".to_string()))
    }
}

#[cfg(target_feature = "avx2")]
pub mod simd {
    use std::arch::x86_64::*;
    
    #[inline]
    pub unsafe fn find_in_sorted_avx2(arr: &[u32], target: u32) -> Option<usize> {
        if arr.len() < 8 {
            return arr.iter().position(|&x| x == target);
        }
        
        let target_vec = _mm256_set1_epi32(target as i32);
        let chunks = arr.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi32(data, target_vec);
            let mask = _mm256_movemask_epi8(cmp);
            
            if mask != 0 {
                let bit_pos = mask.trailing_zeros() / 4;
                return Some(i * 8 + bit_pos as usize);
            }
        }
        
        remainder.iter().position(|&x| x == target).map(|p| arr.len() - remainder.len() + p)
    }
}