# MP031: Hamiltonian Path

## Task Description
Implement Hamiltonian path and cycle detection algorithms to find paths that visit every node exactly once, useful for neural pathway analysis and optimal signal routing in neuromorphic systems.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP030: Graph coloring (for constraint-based approaches)
- Understanding of NP-complete problems and backtracking

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/hamiltonian/path.rs`

2. Implement basic Hamiltonian path detection using backtracking:
   ```rust
   use std::collections::{HashMap, HashSet};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct HamiltonianResult<Id> {
       pub has_path: bool,
       pub has_cycle: bool,
       pub path: Option<Vec<Id>>,
       pub cycle: Option<Vec<Id>>,
       pub partial_paths: Vec<Vec<Id>>,
   }

   pub fn find_hamiltonian_path<G: Graph>(
       graph: &G,
       start_node: Option<G::Node::Id>,
   ) -> HamiltonianResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       if n == 0 {
           return HamiltonianResult {
               has_path: false,
               has_cycle: false,
               path: None,
               cycle: None,
               partial_paths: vec![],
           };
       }
       
       let mut partial_paths = Vec::new();
       
       // Try starting from each node (or specified start node)
       let start_nodes = if let Some(start) = start_node {
           vec![start]
       } else {
           nodes.clone()
       };
       
       for start in start_nodes {
           let mut path = vec![start.clone()];
           let mut visited = HashSet::new();
           visited.insert(start.clone());
           
           if let Some(complete_path) = backtrack_hamiltonian_path(
               graph, 
               &start, 
               &mut path, 
               &mut visited, 
               n,
               &mut partial_paths
           ) {
               // Check if it's also a cycle
               let has_cycle = if let Some(start_node) = graph.get_node(&complete_path[0]) {
                   start_node.neighbors().any(|neighbor| neighbor == complete_path[complete_path.len() - 1])
               } else {
                   false
               };
               
               let cycle = if has_cycle {
                   let mut cycle_path = complete_path.clone();
                   cycle_path.push(complete_path[0].clone());
                   Some(cycle_path)
               } else {
                   None
               };
               
               return HamiltonianResult {
                   has_path: true,
                   has_cycle,
                   path: Some(complete_path),
                   cycle,
                   partial_paths,
               };
           }
       }
       
       HamiltonianResult {
           has_path: false,
           has_cycle: false,
           path: None,
           cycle: None,
           partial_paths,
       }
   }

   fn backtrack_hamiltonian_path<G: Graph>(
       graph: &G,
       current_node: &G::Node::Id,
       path: &mut Vec<G::Node::Id>,
       visited: &mut HashSet<G::Node::Id>,
       target_length: usize,
       partial_paths: &mut Vec<Vec<G::Node::Id>>,
   ) -> Option<Vec<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       
       if path.len() == target_length {
           return Some(path.clone());
       }
       
       // Store partial path for analysis
       if path.len() > target_length / 2 {
           partial_paths.push(path.clone());
       }
       
       if let Some(node) = graph.get_node(current_node) {
           for neighbor in node.neighbors() {
               if !visited.contains(&neighbor) {
                   path.push(neighbor.clone());
                   visited.insert(neighbor.clone());
                   
                   if let Some(result) = backtrack_hamiltonian_path(
                       graph, 
                       &neighbor, 
                       path, 
                       visited, 
                       target_length,
                       partial_paths
                   ) {
                       return Some(result);
                   }
                   
                   // Backtrack
                   path.pop();
                   visited.remove(&neighbor);
               }
           }
       }
       
       None
   }
   ```

3. Implement Ore's theorem checker for Hamiltonian cycles:
   ```rust
   pub fn has_hamiltonian_cycle_ore<G: Graph>(graph: &G) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       if n < 3 {
           return false;
       }
       
       // Check Ore's condition: for every pair of non-adjacent vertices u,v:
       // deg(u) + deg(v) >= n
       for (i, node_u) in nodes.iter().enumerate() {
           if let Some(u_node) = graph.get_node(node_u) {
               let u_neighbors: HashSet<_> = u_node.neighbors().collect();
               let u_degree = u_neighbors.len();
               
               for node_v in nodes.iter().skip(i + 1) {
                   // Check if u and v are non-adjacent
                   if !u_neighbors.contains(node_v) {
                       if let Some(v_node) = graph.get_node(node_v) {
                           let v_degree = v_node.neighbors().count();
                           
                           if u_degree + v_degree < n {
                               return false; // Ore's condition violated
                           }
                       }
                   }
               }
           }
       }
       
       true // Ore's condition satisfied, Hamiltonian cycle exists
   }
   ```

4. Implement Dirac's theorem checker:
   ```rust
   pub fn has_hamiltonian_cycle_dirac<G: Graph>(graph: &G) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let n = graph.node_count();
       
       if n < 3 {
           return false;
       }
       
       // Check Dirac's condition: every vertex has degree >= n/2
       let min_degree = n / 2;
       
       for node in graph.nodes() {
           let degree = node.neighbors().count();
           if degree < min_degree {
               return false;
           }
       }
       
       true // Dirac's condition satisfied, Hamiltonian cycle exists
   }
   ```

5. Implement dynamic programming approach for small graphs:
   ```rust
   pub fn hamiltonian_path_dp<G: Graph>(
       graph: &G,
   ) -> HamiltonianResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Debug {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       if n > 20 {
           // DP approach is only feasible for small graphs
           return find_hamiltonian_path(graph, None);
       }
       
       // Create node index mapping
       let mut node_to_index = HashMap::new();
       for (i, node) in nodes.iter().enumerate() {
           node_to_index.insert(node.clone(), i);
       }
       
       // dp[mask][i] = true if there's a path visiting nodes in mask ending at i
       let mut dp = vec![vec![false; n]; 1 << n];
       let mut parent = vec![vec![None; n]; 1 << n];
       
       // Initialize: single nodes
       for i in 0..n {
           dp[1 << i][i] = true;
       }
       
       // Fill DP table
       for mask in 1..(1 << n) {
           for u in 0..n {
               if !dp[mask][u] || (mask & (1 << u)) == 0 {
                   continue;
               }
               
               if let Some(u_node) = graph.get_node(&nodes[u]) {
                   for neighbor in u_node.neighbors() {
                       if let Some(&v) = node_to_index.get(&neighbor) {
                           if (mask & (1 << v)) == 0 { // v not visited yet
                               let new_mask = mask | (1 << v);
                               if !dp[new_mask][v] {
                                   dp[new_mask][v] = true;
                                   parent[new_mask][v] = Some((mask, u));
                               }
                           }
                       }
                   }
               }
           }
       }
       
       // Check for Hamiltonian path
       let full_mask = (1 << n) - 1;
       for end_node in 0..n {
           if dp[full_mask][end_node] {
               // Reconstruct path
               let path = reconstruct_path(&parent, full_mask, end_node, &nodes);
               
               // Check for cycle
               let start_node = &path[0];
               let end_node = &path[path.len() - 1];
               let has_cycle = if let Some(end_node_ref) = graph.get_node(end_node) {
                   end_node_ref.neighbors().any(|neighbor| neighbor == *start_node)
               } else {
                   false
               };
               
               let cycle = if has_cycle {
                   let mut cycle_path = path.clone();
                   cycle_path.push(start_node.clone());
                   Some(cycle_path)
               } else {
                   None
               };
               
               return HamiltonianResult {
                   has_path: true,
                   has_cycle,
                   path: Some(path),
                   cycle,
                   partial_paths: vec![],
               };
           }
       }
       
       HamiltonianResult {
           has_path: false,
           has_cycle: false,
           path: None,
           cycle: None,
           partial_paths: vec![],
       }
   }

   fn reconstruct_path<Id: Clone>(
       parent: &[Vec<Option<(usize, usize)>>],
       mut mask: usize,
       mut node: usize,
       nodes: &[Id],
   ) -> Vec<Id> {
       let mut path = Vec::new();
       
       while let Some((prev_mask, prev_node)) = parent[mask][node] {
           path.push(nodes[node].clone());
           mask = prev_mask;
           node = prev_node;
       }
       path.push(nodes[node].clone());
       
       path.reverse();
       path
   }
   ```

6. Implement heuristic approaches for large graphs:
   ```rust
   pub fn hamiltonian_path_heuristic<G: Graph>(
       graph: &G,
       max_attempts: usize,
   ) -> HamiltonianResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       use rand::seq::SliceRandom;
       use rand::thread_rng;
       
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       let mut rng = thread_rng();
       let mut best_path = Vec::new();
       let mut partial_paths = Vec::new();
       
       for _ in 0..max_attempts {
           let mut shuffled_nodes = nodes.clone();
           shuffled_nodes.shuffle(&mut rng);
           
           // Try to construct a Hamiltonian path using greedy nearest neighbor
           let mut path = vec![shuffled_nodes[0].clone()];
           let mut visited = HashSet::new();
           visited.insert(shuffled_nodes[0].clone());
           
           let mut current = &shuffled_nodes[0];
           
           while path.len() < n {
               if let Some(node) = graph.get_node(current) {
                   // Find unvisited neighbors
                   let unvisited_neighbors: Vec<_> = node.neighbors()
                       .filter(|neighbor| !visited.contains(neighbor))
                       .collect();
                   
                   if unvisited_neighbors.is_empty() {
                       break; // Dead end
                   }
                   
                   // Choose neighbor with minimum degree (or random)
                   let next = unvisited_neighbors.into_iter()
                       .min_by_key(|neighbor| {
                           graph.get_node(neighbor)
                               .map(|n| n.neighbors().count())
                               .unwrap_or(0)
                       })
                       .unwrap();
                   
                   path.push(next.clone());
                   visited.insert(next.clone());
                   current = &next;
               } else {
                   break;
               }
           }
           
           if path.len() > best_path.len() {
               best_path = path.clone();
           }
           
           if path.len() > n / 2 {
               partial_paths.push(path);
           }
           
           if best_path.len() == n {
               // Found Hamiltonian path
               let has_cycle = if let Some(start_node) = graph.get_node(&best_path[0]) {
                   start_node.neighbors().any(|neighbor| neighbor == best_path[best_path.len() - 1])
               } else {
                   false
               };
               
               let cycle = if has_cycle {
                   let mut cycle_path = best_path.clone();
                   cycle_path.push(best_path[0].clone());
                   Some(cycle_path)
               } else {
                   None
               };
               
               return HamiltonianResult {
                   has_path: true,
                   has_cycle,
                   path: Some(best_path),
                   cycle,
                   partial_paths,
               };
           }
       }
       
       HamiltonianResult {
           has_path: false,
           has_cycle: false,
           path: None,
           cycle: None,
           partial_paths,
       }
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/hamiltonian/path.rs
pub trait HamiltonianPath: Graph {
    fn find_hamiltonian_path(&self, start: Option<Self::Node::Id>) -> HamiltonianResult<Self::Node::Id>;
    fn find_hamiltonian_cycle(&self) -> HamiltonianResult<Self::Node::Id>;
    fn has_hamiltonian_cycle_ore(&self) -> bool;
    fn has_hamiltonian_cycle_dirac(&self) -> bool;
    fn hamiltonian_path_dp(&self) -> HamiltonianResult<Self::Node::Id>;
    fn hamiltonian_path_heuristic(&self, attempts: usize) -> HamiltonianResult<Self::Node::Id>;
}

pub struct HamiltonianAnalysis<Id> {
    pub has_path: bool,
    pub has_cycle: bool,
    pub longest_path: Vec<Id>,
    pub theoretical_guarantees: bool,
    pub path_count_estimate: usize,
}
```

## Verification Steps
1. Test on graphs with known Hamiltonian properties (complete graphs, cycles)
2. Verify Ore's and Dirac's theorem implementations
3. Compare backtracking vs DP vs heuristic approaches
4. Test on neuromorphic network topologies
5. Benchmark performance and scalability limits

## Time Estimate
35 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP030: Graph coloring (for constraint satisfaction techniques)