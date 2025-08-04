# MP030: Graph Coloring

## Task Description
Implement graph coloring algorithms to assign colors to nodes such that no adjacent nodes share the same color, enabling resource allocation, scheduling, and conflict resolution in neuromorphic systems.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- Understanding of constraint satisfaction problems
- Basic knowledge of greedy algorithms and backtracking

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/coloring/mod.rs`

2. Implement greedy graph coloring:
   ```rust
   use std::collections::{HashMap, HashSet};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct ColoringResult<Id> {
       pub coloring: HashMap<Id, usize>,
       pub num_colors: usize,
       pub is_valid: bool,
       pub conflicts: Vec<(Id, Id)>,
   }

   pub fn greedy_coloring<G: Graph>(
       graph: &G,
       ordering: Option<Vec<G::Node::Id>>,
   ) -> ColoringResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut coloring = HashMap::new();
       let mut max_color = 0;
       
       // Determine node ordering
       let nodes: Vec<_> = if let Some(order) = ordering {
           order
       } else {
           // Default: order by degree (descending)
           let mut node_degrees: Vec<_> = graph.nodes()
               .map(|node| {
                   let degree = node.neighbors().count();
                   (node.id(), degree)
               })
               .collect();
           node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
           node_degrees.into_iter().map(|(id, _)| id).collect()
       };
       
       for node_id in nodes {
           if let Some(node) = graph.get_node(&node_id) {
               // Find colors used by neighbors
               let mut used_colors = HashSet::new();
               for neighbor_id in node.neighbors() {
                   if let Some(&color) = coloring.get(&neighbor_id) {
                       used_colors.insert(color);
                   }
               }
               
               // Find smallest available color
               let mut color = 0;
               while used_colors.contains(&color) {
                   color += 1;
               }
               
               coloring.insert(node_id, color);
               max_color = max_color.max(color);
           }
       }
       
       let num_colors = max_color + 1;
       let (is_valid, conflicts) = validate_coloring(graph, &coloring);
       
       ColoringResult {
           coloring,
           num_colors,
           is_valid,
           conflicts,
       }
   }
   ```

3. Implement Welsh-Powell algorithm:
   ```rust
   pub fn welsh_powell_coloring<G: Graph>(
       graph: &G,
   ) -> ColoringResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Sort nodes by degree (descending)
       let mut node_degrees: Vec<_> = graph.nodes()
           .map(|node| {
               let degree = node.neighbors().count();
               (node.id(), degree)
           })
           .collect();
       node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
       
       let mut coloring = HashMap::new();
       let mut current_color = 0;
       let mut uncolored_nodes: HashSet<_> = node_degrees.iter().map(|(id, _)| id.clone()).collect();
       
       while !uncolored_nodes.is_empty() {
           let mut colored_with_current = HashSet::new();
           
           // Find all nodes that can be colored with current color
           for (node_id, _) in &node_degrees {
               if !uncolored_nodes.contains(node_id) {
                   continue;
               }
               
               // Check if this node can use current color
               let mut can_use_color = true;
               if let Some(node) = graph.get_node(node_id) {
                   for neighbor_id in node.neighbors() {
                       if colored_with_current.contains(&neighbor_id) {
                           can_use_color = false;
                           break;
                       }
                   }
               }
               
               if can_use_color {
                   coloring.insert(node_id.clone(), current_color);
                   colored_with_current.insert(node_id.clone());
                   uncolored_nodes.remove(node_id);
               }
           }
           
           current_color += 1;
       }
       
       let num_colors = current_color;
       let (is_valid, conflicts) = validate_coloring(graph, &coloring);
       
       ColoringResult {
           coloring,
           num_colors,
           is_valid,
           conflicts,
       }
   }
   ```

4. Implement DSATUR algorithm:
   ```rust
   pub fn dsatur_coloring<G: Graph>(
       graph: &G,
   ) -> ColoringResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut coloring = HashMap::new();
       let mut uncolored_nodes: HashSet<_> = graph.nodes().map(|n| n.id()).collect();
       let mut saturation: HashMap<G::Node::Id, usize> = HashMap::new();
       
       // Initialize saturation degrees (number of different colors in neighborhood)
       for node in graph.nodes() {
           saturation.insert(node.id(), 0);
       }
       
       while !uncolored_nodes.is_empty() {
           // Find node with highest saturation degree, break ties by degree
           let selected_node = uncolored_nodes.iter()
               .max_by(|&a, &b| {
                   let sat_a = saturation.get(a).copied().unwrap_or(0);
                   let sat_b = saturation.get(b).copied().unwrap_or(0);
                   
                   sat_a.cmp(&sat_b).then_with(|| {
                       let deg_a = graph.get_node(a).map(|n| n.neighbors().count()).unwrap_or(0);
                       let deg_b = graph.get_node(b).map(|n| n.neighbors().count()).unwrap_or(0);
                       deg_a.cmp(&deg_b)
                   })
               })
               .cloned()
               .unwrap();
           
           // Color the selected node
           if let Some(node) = graph.get_node(&selected_node) {
               let mut used_colors = HashSet::new();
               for neighbor_id in node.neighbors() {
                   if let Some(&color) = coloring.get(&neighbor_id) {
                       used_colors.insert(color);
                   }
               }
               
               let mut color = 0;
               while used_colors.contains(&color) {
                   color += 1;
               }
               
               coloring.insert(selected_node.clone(), color);
               uncolored_nodes.remove(&selected_node);
               
               // Update saturation degrees of neighbors
               for neighbor_id in node.neighbors() {
                   if uncolored_nodes.contains(&neighbor_id) {
                       if let Some(neighbor_node) = graph.get_node(&neighbor_id) {
                           let mut neighbor_colors = HashSet::new();
                           for nn_id in neighbor_node.neighbors() {
                               if let Some(&c) = coloring.get(&nn_id) {
                                   neighbor_colors.insert(c);
                               }
                           }
                           saturation.insert(neighbor_id, neighbor_colors.len());
                       }
                   }
               }
           }
       }
       
       let num_colors = coloring.values().max().copied().unwrap_or(0) + 1;
       let (is_valid, conflicts) = validate_coloring(graph, &coloring);
       
       ColoringResult {
           coloring,
           num_colors,
           is_valid,
           conflicts,
       }
   }
   ```

5. Implement backtracking coloring for optimal solutions:
   ```rust
   pub fn backtrack_coloring<G: Graph>(
       graph: &G,
       max_colors: usize,
   ) -> Option<ColoringResult<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let mut coloring = HashMap::new();
       
       if backtrack_solve(graph, &nodes, 0, max_colors, &mut coloring) {
           let (is_valid, conflicts) = validate_coloring(graph, &coloring);
           Some(ColoringResult {
               coloring,
               num_colors: max_colors,
               is_valid,
               conflicts,
           })
       } else {
           None
       }
   }

   fn backtrack_solve<G: Graph>(
       graph: &G,
       nodes: &[G::Node::Id],
       node_index: usize,
       max_colors: usize,
       coloring: &mut HashMap<G::Node::Id, usize>,
   ) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if node_index >= nodes.len() {
           return true; // All nodes colored successfully
       }
       
       let current_node = &nodes[node_index];
       
       if let Some(node) = graph.get_node(current_node) {
           for color in 0..max_colors {
               // Check if color is safe
               let mut is_safe = true;
               for neighbor_id in node.neighbors() {
                   if coloring.get(&neighbor_id) == Some(&color) {
                       is_safe = false;
                       break;
                   }
               }
               
               if is_safe {
                   coloring.insert(current_node.clone(), color);
                   
                   if backtrack_solve(graph, nodes, node_index + 1, max_colors, coloring) {
                       return true;
                   }
                   
                   coloring.remove(current_node);
               }
           }
       }
       
       false
   }
   ```

6. Implement edge coloring:
   ```rust
   pub fn edge_coloring<G: Graph>(
       graph: &G,
   ) -> HashMap<(G::Node::Id, G::Node::Id), usize> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut edge_coloring = HashMap::new();
       let edges: Vec<_> = graph.edges()
           .map(|e| (e.source(), e.target()))
           .collect();
       
       for (source, target) in edges {
           // Find colors used by edges incident to source or target
           let mut used_colors = HashSet::new();
           
           for (other_edge, &color) in &edge_coloring {
               if other_edge.0 == source || other_edge.1 == source ||
                  other_edge.0 == target || other_edge.1 == target {
                   used_colors.insert(color);
               }
           }
           
           // Find smallest available color
           let mut color = 0;
           while used_colors.contains(&color) {
               color += 1;
           }
           
           edge_coloring.insert((source, target), color);
       }
       
       edge_coloring
   }

   fn validate_coloring<G: Graph>(
       graph: &G,
       coloring: &HashMap<G::Node::Id, usize>,
   ) -> (bool, Vec<(G::Node::Id, G::Node::Id)>) 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut conflicts = Vec::new();
       
       for edge in graph.edges() {
           let source = edge.source();
           let target = edge.target();
           
           if let (Some(&color1), Some(&color2)) = (coloring.get(&source), coloring.get(&target)) {
               if color1 == color2 {
                   conflicts.push((source, target));
               }
           }
       }
       
       (conflicts.is_empty(), conflicts)
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/coloring/mod.rs
pub trait GraphColoring: Graph {
    fn greedy_coloring(&self, ordering: Option<Vec<Self::Node::Id>>) -> ColoringResult<Self::Node::Id>;
    fn welsh_powell_coloring(&self) -> ColoringResult<Self::Node::Id>;
    fn dsatur_coloring(&self) -> ColoringResult<Self::Node::Id>;
    fn backtrack_coloring(&self, max_colors: usize) -> Option<ColoringResult<Self::Node::Id>>;
    fn edge_coloring(&self) -> HashMap<(Self::Node::Id, Self::Node::Id), usize>;
    fn chromatic_number_estimate(&self) -> usize;
}

pub struct ColoringAnalysis<Id> {
    pub chromatic_number: usize,
    pub coloring_quality: f32,
    pub color_distribution: HashMap<usize, usize>,
    pub largest_color_class: usize,
    pub is_optimal: bool,
}
```

## Verification Steps
1. Test coloring algorithms on graphs with known chromatic numbers
2. Verify no conflicts exist in generated colorings
3. Compare algorithm efficiency on different graph types
4. Test edge coloring on bipartite and complete graphs
5. Benchmark performance on neuromorphic network topologies

## Time Estimate
30 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms