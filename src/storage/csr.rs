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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn create_test_relationship(from: u32, to: u32, rel_type: u8, weight: f32) -> Relationship {
        Relationship {
            from: crate::core::types::EntityKey::from_u32(from),
            to: crate::core::types::EntityKey::from_u32(to),
            rel_type,
            weight,
        }
    }

    #[test]
    fn test_from_edges_basic() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(0, 2, 1, 2.0),
            create_test_relationship(1, 2, 2, 3.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        
        // Node 0 should have 2 neighbors: [1, 2]
        let neighbors_0 = graph.get_neighbors(0);
        assert_eq!(neighbors_0.len(), 2);
        assert!(neighbors_0.contains(&1));
        assert!(neighbors_0.contains(&2));
        
        // Node 1 should have 1 neighbor: [2]
        let neighbors_1 = graph.get_neighbors(1);
        assert_eq!(neighbors_1.len(), 1);
        assert_eq!(neighbors_1[0], 2);
        
        // Node 2 should have no outgoing edges
        let neighbors_2 = graph.get_neighbors(2);
        assert_eq!(neighbors_2.len(), 0);
    }

    #[test]
    fn test_from_edges_sorting() {
        // Create edges in random order to test sorting
        let edges = vec![
            create_test_relationship(0, 5, 1, 1.0),
            create_test_relationship(0, 1, 1, 2.0),
            create_test_relationship(0, 3, 1, 3.0),
        ];

        let graph = CSRGraph::from_edges(edges, 6).unwrap();
        
        // Neighbors should be sorted
        let neighbors = graph.get_neighbors(0);
        assert_eq!(neighbors, &[1, 3, 5]);
    }

    #[test]
    fn test_from_edges_adjacency_list_building() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(0, 1, 2, 2.0), // Duplicate edge (different type)
            create_test_relationship(1, 0, 1, 3.0),
        ];

        let graph = CSRGraph::from_edges(edges, 2).unwrap();
        
        // Should handle duplicate edges correctly
        let neighbors_0 = graph.get_neighbors(0);
        assert_eq!(neighbors_0.len(), 2); // Two edges from 0 to 1
        assert_eq!(neighbors_0, &[1, 1]);
        
        let edges_0 = graph.get_edges(0);
        assert_eq!(edges_0.len(), 2);
        // Should preserve both relationship types and weights
        assert!(edges_0.contains(&(1, 1, 1.0)));
        assert!(edges_0.contains(&(1, 2, 2.0)));
    }

    #[test]
    fn test_from_edges_error_handling() {
        let edges = vec![
            create_test_relationship(0, 5, 1, 1.0), // Invalid 'to' node
        ];

        let result = CSRGraph::from_edges(edges, 3); // Only 3 nodes (0, 1, 2)
        assert!(result.is_err());
        
        let edges = vec![
            create_test_relationship(5, 0, 1, 1.0), // Invalid 'from' node
        ];
        
        let result = CSRGraph::from_edges(edges, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_format_structure() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(0, 2, 1, 2.0),
            create_test_relationship(2, 0, 2, 3.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        
        // Test CSR structure properties
        assert_eq!(graph.row_ptr.len(), 4); // nodes + 1
        assert_eq!(graph.col_idx.len(), 3); // number of edges
        assert_eq!(graph.rel_types.len(), 3);
        assert_eq!(graph.weights.len(), 3);
        
        // Row pointers should be monotonically increasing
        for i in 1..graph.row_ptr.len() {
            assert!(graph.row_ptr[i] >= graph.row_ptr[i-1]);
        }
    }

    #[test]
    fn test_has_edge() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 2.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 2));
        assert!(!graph.has_edge(0, 2));
        assert!(!graph.has_edge(1, 0));
    }

    #[test]
    fn test_degree() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(0, 2, 1, 2.0),
            create_test_relationship(0, 3, 1, 3.0),
        ];

        let graph = CSRGraph::from_edges(edges, 4).unwrap();
        
        assert_eq!(graph.degree(0), 3);
        assert_eq!(graph.degree(1), 0);
        assert_eq!(graph.degree(2), 0);
        assert_eq!(graph.degree(3), 0);
        assert_eq!(graph.degree(10), 0); // Invalid node
    }

    #[test]
    fn test_get_edge_weight() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.5),
            create_test_relationship(0, 2, 1, 2.5),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        
        assert_eq!(graph.get_edge_weight(0, 1), Some(1.5));
        assert_eq!(graph.get_edge_weight(0, 2), Some(2.5));
        assert_eq!(graph.get_edge_weight(0, 3), None);
        assert_eq!(graph.get_edge_weight(1, 0), None);
        assert_eq!(graph.get_edge_weight(10, 1), None); // Invalid node
    }

    #[test]
    fn test_traverse_bfs_basic() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(0, 2, 1, 1.0),
            create_test_relationship(1, 3, 1, 1.0),
            create_test_relationship(2, 3, 1, 1.0),
        ];

        let graph = CSRGraph::from_edges(edges, 4).unwrap();
        
        let result = graph.traverse_bfs(0, 3);
        
        // Should visit all reachable nodes
        let visited_nodes: HashSet<u32> = result.iter().map(|(node, _)| *node).collect();
        assert!(visited_nodes.contains(&0));
        assert!(visited_nodes.contains(&1));
        assert!(visited_nodes.contains(&2));
        assert!(visited_nodes.contains(&3));
        
        // Check depth constraints
        for (_, depth) in &result {
            assert!(*depth < 3);
        }
    }

    #[test]
    fn test_traverse_bfs_depth_limit() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 1.0),
            create_test_relationship(2, 3, 1, 1.0),
        ];

        let graph = CSRGraph::from_edges(edges, 4).unwrap();
        
        let result = graph.traverse_bfs(0, 2);
        
        // Should only reach nodes within depth 2
        let visited_nodes: HashSet<u32> = result.iter().map(|(node, _)| *node).collect();
        assert!(visited_nodes.contains(&0));
        assert!(visited_nodes.contains(&1));
        assert!(visited_nodes.contains(&2));
        assert!(!visited_nodes.contains(&3)); // Depth 3, should not be visited
    }

    #[test]
    fn test_find_path_basic() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 1.0),
            create_test_relationship(2, 3, 1, 1.0),
        ];

        let graph = CSRGraph::from_edges(edges, 4).unwrap();
        
        let path = graph.find_path(0, 3, 5).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
        
        // Path to self
        let self_path = graph.find_path(1, 1, 5).unwrap();
        assert_eq!(self_path, vec![1]);
        
        // No path
        let no_path = graph.find_path(3, 0, 5);
        assert!(no_path.is_none());
    }

    #[test]
    fn test_find_path_with_depth_limit() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 1.0),
            create_test_relationship(2, 3, 1, 1.0),
        ];

        let graph = CSRGraph::from_edges(edges, 4).unwrap();
        
        // Should find path within limit
        let path = graph.find_path(0, 2, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
        
        // Should not find path beyond limit
        let no_path = graph.find_path(0, 3, 2);
        assert!(no_path.is_none());
    }

    #[test]
    fn test_memory_usage() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 2.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        let memory_usage = graph.memory_usage();
        
        assert!(memory_usage > 0);
        
        // Should be approximately the sum of all arrays
        let expected_min = 
            graph.row_ptr.capacity() * std::mem::size_of::<u64>() +
            graph.col_idx.capacity() * std::mem::size_of::<u32>() +
            graph.rel_types.capacity() * std::mem::size_of::<u8>() +
            graph.weights.capacity() * std::mem::size_of::<f32>();
        
        assert!(memory_usage >= expected_min);
    }

    #[test]
    fn test_encoded_size() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
            create_test_relationship(1, 2, 1, 2.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        let encoded_size = graph.encoded_size();
        
        assert!(encoded_size > 0);
        
        // Should include all data arrays
        let expected_min = 
            std::mem::size_of::<u32>() * 2 + // node_count + edge_count
            graph.row_ptr.len() * std::mem::size_of::<u64>() +
            graph.col_idx.len() * std::mem::size_of::<u32>() +
            graph.rel_types.len() * std::mem::size_of::<u8>() +
            graph.weights.len() * std::mem::size_of::<f32>();
        
        assert_eq!(encoded_size, expected_min);
    }

    #[test]
    fn test_empty_graph() {
        let graph = CSRGraph::from_edges(vec![], 5).unwrap();
        
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 0);
        
        for i in 0..5 {
            assert_eq!(graph.get_neighbors(i).len(), 0);
            assert_eq!(graph.degree(i), 0);
        }
        
        let result = graph.traverse_bfs(0, 5);
        assert_eq!(result.len(), 1); // Just the starting node
        assert_eq!(result[0], (0, 0));
    }

    #[test]
    fn test_large_graph_properties() {
        // Create a larger graph to test scalability
        let mut edges = Vec::new();
        let node_count = 1000;
        
        // Create a chain: 0 -> 1 -> 2 -> ... -> 999
        for i in 0..node_count-1 {
            edges.push(create_test_relationship(i, i+1, 1, i as f32));
        }
        
        let graph = CSRGraph::from_edges(edges, node_count).unwrap();
        
        assert_eq!(graph.node_count(), node_count);
        assert_eq!(graph.edge_count(), node_count - 1);
        
        // Test path finding in large graph
        let path = graph.find_path(0, 999, 1000).unwrap();
        assert_eq!(path.len(), 1000);
        assert_eq!(path[0], 0);
        assert_eq!(path[999], 999);
        
        // Test that neighbors are sorted
        for i in 0..node_count-1 {
            let neighbors = graph.get_neighbors(i);
            if !neighbors.is_empty() {
                for j in 1..neighbors.len() {
                    assert!(neighbors[j] >= neighbors[j-1], "Neighbors should be sorted");
                }
            }
        }
    }

    #[test]
    fn test_unsupported_operations() {
        let mut graph = CSRGraph::new();
        
        // Test that unsupported operations return appropriate errors
        assert!(graph.add_edge(0, 1, 1.0).is_err());
        assert!(graph.update_entity(0, vec![]).is_err());
        assert!(graph.remove(0).is_err());
        assert!(graph.remove_edge(0, 1).is_err());
        assert!(graph.update_edge_weight(0, 1, 2.0).is_err());
    }

    #[test]
    fn test_contains_entity() {
        let edges = vec![
            create_test_relationship(0, 1, 1, 1.0),
        ];

        let graph = CSRGraph::from_edges(edges, 3).unwrap();
        
        assert!(graph.contains_entity(0));
        assert!(graph.contains_entity(1));
        assert!(graph.contains_entity(2));
        assert!(!graph.contains_entity(3));
        assert!(!graph.contains_entity(100));
    }

    #[cfg(test)]
    mod simd_tests {
        use super::*;
        
        #[test]
        #[cfg(target_feature = "avx2")]
        fn test_simd_find_in_sorted() {
            // Test SIMD search in sorted arrays
            let arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
            
            unsafe {
                assert_eq!(simd::find_in_sorted_avx2(&arr, 7), Some(3));
                assert_eq!(simd::find_in_sorted_avx2(&arr, 1), Some(0));
                assert_eq!(simd::find_in_sorted_avx2(&arr, 23), Some(11));
                assert_eq!(simd::find_in_sorted_avx2(&arr, 8), None);
                assert_eq!(simd::find_in_sorted_avx2(&arr, 0), None);
                assert_eq!(simd::find_in_sorted_avx2(&arr, 24), None);
            }
        }
        
        #[test]
        #[cfg(target_feature = "avx2")]
        fn test_simd_find_small_arrays() {
            // Test with arrays smaller than SIMD width
            let small_arr = [1, 3, 5];
            
            unsafe {
                assert_eq!(simd::find_in_sorted_avx2(&small_arr, 3), Some(1));
                assert_eq!(simd::find_in_sorted_avx2(&small_arr, 4), None);
            }
        }
        
        #[test]
        fn test_simd_fallback() {
            // Test that regular search works as fallback
            let arr = [1, 3, 5, 7];
            
            let result = arr.iter().position(|&x| x == 5);
            assert_eq!(result, Some(2));
        }
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        
        #[test]
        fn test_csr_invariants() {
            let edges = vec![
                create_test_relationship(0, 1, 1, 1.0),
                create_test_relationship(0, 2, 1, 2.0),
                create_test_relationship(1, 2, 1, 3.0),
                create_test_relationship(2, 0, 1, 4.0),
            ];

            let graph = CSRGraph::from_edges(edges, 3).unwrap();
            
            // CSR format invariants
            assert_eq!(graph.row_ptr.len(), graph.node_count() as usize + 1);
            assert_eq!(graph.col_idx.len(), graph.edge_count() as usize);
            assert_eq!(graph.rel_types.len(), graph.edge_count() as usize);
            assert_eq!(graph.weights.len(), graph.edge_count() as usize);
            
            // Row pointers should be monotonically non-decreasing
            for i in 1..graph.row_ptr.len() {
                assert!(graph.row_ptr[i] >= graph.row_ptr[i-1]);
            }
            
            // Last row pointer should equal number of edges
            assert_eq!(graph.row_ptr.last().unwrap(), &(graph.edge_count() as u64));
            
            // All column indices should be valid node IDs
            for &col in &graph.col_idx {
                assert!(col < graph.node_count());
            }
        }
        
        #[test]
        fn test_neighbor_sorting_property() {
            // Generate random edges and verify sorting
            let mut edges = Vec::new();
            for i in 0..10 {
                for j in 0..10 {
                    if i != j {
                        edges.push(create_test_relationship(i, j, 1, 1.0));
                    }
                }
            }
            
            let graph = CSRGraph::from_edges(edges, 10).unwrap();
            
            // Verify that all neighbor lists are sorted
            for i in 0..10 {
                let neighbors = graph.get_neighbors(i);
                for j in 1..neighbors.len() {
                    assert!(neighbors[j] >= neighbors[j-1], 
                        "Neighbors of node {} should be sorted", i);
                }
            }
        }
        
        #[test]
        fn test_bfs_properties() {
            let edges = vec![
                create_test_relationship(0, 1, 1, 1.0),
                create_test_relationship(0, 2, 1, 1.0),
                create_test_relationship(1, 3, 1, 1.0),
                create_test_relationship(2, 3, 1, 1.0),
                create_test_relationship(3, 4, 1, 1.0),
            ];

            let graph = CSRGraph::from_edges(edges, 5).unwrap();
            let result = graph.traverse_bfs(0, 5);
            
            // BFS properties
            // 1. Starting node should be at depth 0
            assert!(result.iter().any(|(node, depth)| *node == 0 && *depth == 0));
            
            // 2. For any node at depth d, all its neighbors should be at depth <= d+1
            for (node, depth) in &result {
                let neighbors = graph.get_neighbors(*node);
                for &neighbor in neighbors {
                    if let Some((_, neighbor_depth)) = result.iter().find(|(n, _)| *n == neighbor) {
                        assert!(*neighbor_depth <= depth + 1, 
                            "BFS depth property violated: node {} at depth {} has neighbor {} at depth {}", 
                            node, depth, neighbor, neighbor_depth);
                    }
                }
            }
            
            // 3. All nodes in result should be reachable from start
            // This is implicitly tested by the BFS algorithm itself
        }
    }
}