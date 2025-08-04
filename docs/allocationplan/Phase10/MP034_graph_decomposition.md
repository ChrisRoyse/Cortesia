# MP034: Graph Decomposition

## Task Description
Implement graph decomposition algorithms to break down complex neural networks into simpler components, including tree decomposition, modular decomposition, and k-core decomposition for hierarchical analysis.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP015-MP018: Graph traversal and connectivity algorithms
- MP020: Community detection (related to modular structure)
- Understanding of tree structures and graph theory

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/decomposition/mod.rs`

2. Implement k-core decomposition:
   ```rust
   use std::collections::{HashMap, HashSet, VecDeque};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct KCoreDecomposition<Id> {
       pub core_numbers: HashMap<Id, usize>,
       pub k_cores: HashMap<usize, HashSet<Id>>,
       pub max_core: usize,
       pub degeneracy: usize,
   }

   pub fn k_core_decomposition<G: Graph>(
       graph: &G,
   ) -> KCoreDecomposition<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut core_numbers = HashMap::new();
       let mut degrees = HashMap::new();
       let mut neighbors: HashMap<G::Node::Id, HashSet<G::Node::Id>> = HashMap::new();
       
       // Initialize degrees and neighbor sets
       for node in graph.nodes() {
           let node_id = node.id();
           let node_neighbors: HashSet<_> = node.neighbors().collect();
           let degree = node_neighbors.len();
           
           degrees.insert(node_id.clone(), degree);
           neighbors.insert(node_id.clone(), node_neighbors);
           core_numbers.insert(node_id, 0);
       }
       
       let mut queue: VecDeque<_> = degrees.iter()
           .map(|(node, &degree)| (degree, node.clone()))
           .collect();
       queue.make_contiguous().sort_by_key(|(degree, _)| *degree);
       
       let mut processed = HashSet::new();
       let mut current_core = 0;
       
       while let Some((_, node)) = queue.pop_front() {
           if processed.contains(&node) {
               continue;
           }
           
           processed.insert(node.clone());
           let node_degree = degrees[&node];
           current_core = current_core.max(node_degree);
           core_numbers.insert(node.clone(), current_core);
           
           // Update neighbors' degrees
           if let Some(node_neighbors) = neighbors.get(&node) {
               for neighbor in node_neighbors {
                   if !processed.contains(neighbor) {
                       if let Some(neighbor_set) = neighbors.get_mut(neighbor) {
                           neighbor_set.remove(&node);
                           let new_degree = neighbor_set.len();
                           degrees.insert(neighbor.clone(), new_degree);
                           
                           // Re-insert neighbor with updated degree
                           queue.push_back((new_degree, neighbor.clone()));
                       }
                   }
               }
           }
           
           // Sort queue by degree
           queue.make_contiguous().sort_by_key(|(degree, _)| *degree);
       }
       
       // Group nodes by core number
       let mut k_cores: HashMap<usize, HashSet<G::Node::Id>> = HashMap::new();
       let mut max_core = 0;
       
       for (node, &core_num) in &core_numbers {
           k_cores.entry(core_num).or_insert_with(HashSet::new).insert(node.clone());
           max_core = max_core.max(core_num);
       }
       
       KCoreDecomposition {
           core_numbers,
           k_cores,
           max_core,
           degeneracy: max_core,
       }
   }
   ```

3. Implement tree decomposition:
   ```rust
   #[derive(Debug, Clone)]
   pub struct TreeNode<Id> {
       pub id: usize,
       pub bag: HashSet<Id>,
       pub children: Vec<usize>,
       pub parent: Option<usize>,
   }

   #[derive(Debug, Clone)]
   pub struct TreeDecomposition<Id> {
       pub tree_nodes: HashMap<usize, TreeNode<Id>>,
       pub tree_width: usize,
       pub root: usize,
       pub is_valid: bool,
   }

   pub fn tree_decomposition_greedy<G: Graph>(
       graph: &G,
   ) -> TreeDecomposition<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Greedy heuristic for tree decomposition
       let mut tree_nodes = HashMap::new();
       let mut current_id = 0;
       let mut remaining_nodes: HashSet<_> = graph.nodes().map(|n| n.id()).collect();
       let mut tree_width = 0;
       
       // Build adjacency list
       let mut adjacency: HashMap<G::Node::Id, HashSet<G::Node::Id>> = HashMap::new();
       for node in graph.nodes() {
           let neighbors: HashSet<_> = node.neighbors().collect();
           adjacency.insert(node.id(), neighbors);
       }
       
       let root_id = 0;
       
       // Greedy elimination ordering
       while !remaining_nodes.is_empty() {
           // Find node with minimum fill-in
           let (best_node, fill_in) = find_min_fill_node(&remaining_nodes, &adjacency);
           
           // Create tree node (bag)
           let mut bag = HashSet::new();
           bag.insert(best_node.clone());
           
           // Add neighbors to make clique
           if let Some(neighbors) = adjacency.get(&best_node) {
               for neighbor in neighbors {
                   if remaining_nodes.contains(neighbor) {
                       bag.insert(neighbor.clone());
                   }
               }
           }
           
           tree_width = tree_width.max(bag.len().saturating_sub(1));
           
           tree_nodes.insert(current_id, TreeNode {
               id: current_id,
               bag,
               children: vec![],
               parent: if current_id == 0 { None } else { Some(0) },
           });
           
           // Update adjacency (add fill-in edges)
           if let Some(neighbors) = adjacency.get(&best_node).cloned() {
               let neighbors_vec: Vec<_> = neighbors.iter()
                   .filter(|n| remaining_nodes.contains(*n))
                   .cloned()
                   .collect();
               
               for i in 0..neighbors_vec.len() {
                   for j in (i + 1)..neighbors_vec.len() {
                       let u = &neighbors_vec[i];
                       let v = &neighbors_vec[j];
                       
                       adjacency.get_mut(u).unwrap().insert(v.clone());
                       adjacency.get_mut(v).unwrap().insert(u.clone());
                   }
               }
           }
           
           // Remove eliminated node
           remaining_nodes.remove(&best_node);
           adjacency.remove(&best_node);
           for adj_list in adjacency.values_mut() {
               adj_list.remove(&best_node);
           }
           
           current_id += 1;
       }
       
       // Build parent-child relationships
       for i in 1..current_id {
           if let Some(parent) = tree_nodes.get_mut(&0) {
               parent.children.push(i);
           }
       }
       
       TreeDecomposition {
           tree_nodes,
           tree_width,
           root: root_id,
           is_valid: true, // Simplified validation
       }
   }

   fn find_min_fill_node<Id: Clone + Eq + std::hash::Hash>(
       remaining_nodes: &HashSet<Id>,
       adjacency: &HashMap<Id, HashSet<Id>>,
   ) -> (Id, usize) {
       let mut best_node = remaining_nodes.iter().next().unwrap().clone();
       let mut min_fill = usize::MAX;
       
       for node in remaining_nodes {
           let fill_in = calculate_fill_in(node, remaining_nodes, adjacency);
           if fill_in < min_fill {
               min_fill = fill_in;
               best_node = node.clone();
           }
       }
       
       (best_node, min_fill)
   }

   fn calculate_fill_in<Id: Clone + Eq + std::hash::Hash>(
       node: &Id,
       remaining_nodes: &HashSet<Id>,
       adjacency: &HashMap<Id, HashSet<Id>>,
   ) -> usize {
       if let Some(neighbors) = adjacency.get(node) {
           let remaining_neighbors: Vec<_> = neighbors.iter()
               .filter(|n| remaining_nodes.contains(*n))
               .collect();
           
           let mut fill_edges = 0;
           for i in 0..remaining_neighbors.len() {
               for j in (i + 1)..remaining_neighbors.len() {
                   let u = remaining_neighbors[i];
                   let v = remaining_neighbors[j];
                   
                   // Check if edge (u,v) exists
                   if let Some(u_neighbors) = adjacency.get(u) {
                       if !u_neighbors.contains(v) {
                           fill_edges += 1;
                       }
                   }
               }
           }
           fill_edges
       } else {
           0
       }
   }
   ```

4. Implement modular decomposition:
   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub enum ModuleType {
       Trivial,    // Single node
       Complete,   // Complete module (all pairs connected)
       Empty,      // Empty module (no pairs connected) 
       Prime,      // Prime module (neither complete nor empty)
   }

   #[derive(Debug, Clone)]
   pub struct Module<Id> {
       pub nodes: HashSet<Id>,
       pub module_type: ModuleType,
       pub children: Vec<Module<Id>>,
   }

   #[derive(Debug, Clone)]
   pub struct ModularDecomposition<Id> {
       pub root_module: Module<Id>,
       pub all_modules: Vec<Module<Id>>,
       pub modular_width: usize,
   }

   pub fn modular_decomposition<G: Graph>(
       graph: &G,
   ) -> ModularDecomposition<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: HashSet<_> = graph.nodes().map(|n| n.id()).collect();
       
       // Build adjacency representation
       let mut adjacency: HashMap<G::Node::Id, HashSet<G::Node::Id>> = HashMap::new();
       for node in graph.nodes() {
           let neighbors: HashSet<_> = node.neighbors().collect();
           adjacency.insert(node.id(), neighbors);
       }
       
       // Recursive modular decomposition
       let root_module = decompose_module(&nodes, &adjacency);
       let all_modules = collect_all_modules(&root_module);
       let modular_width = calculate_modular_width(&root_module);
       
       ModularDecomposition {
           root_module,
           all_modules,
           modular_width,
       }
   }

   fn decompose_module<Id: Clone + Eq + std::hash::Hash>(
       nodes: &HashSet<Id>,
       adjacency: &HashMap<Id, HashSet<Id>>,
   ) -> Module<Id> {
       if nodes.len() <= 1 {
           return Module {
               nodes: nodes.clone(),
               module_type: ModuleType::Trivial,
               children: vec![],
           };
       }
       
       // Check if module is complete or empty
       let module_type = classify_module(nodes, adjacency);
       
       match module_type {
           ModuleType::Complete | ModuleType::Empty | ModuleType::Trivial => {
               Module {
                   nodes: nodes.clone(),
                   module_type,
                   children: vec![],
               }
           }
           ModuleType::Prime => {
               // Find submodules (simplified heuristic)
               let submodules = find_submodules(nodes, adjacency);
               let children: Vec<_> = submodules.into_iter()
                   .map(|submodule| decompose_module(&submodule, adjacency))
                   .collect();
               
               Module {
                   nodes: nodes.clone(),
                   module_type: ModuleType::Prime,
                   children,
               }
           }
       }
   }

   fn classify_module<Id: Clone + Eq + std::hash::Hash>(
       nodes: &HashSet<Id>,
       adjacency: &HashMap<Id, HashSet<Id>>,
   ) -> ModuleType {
       if nodes.len() <= 1 {
           return ModuleType::Trivial;
       }
       
       let node_vec: Vec<_> = nodes.iter().collect();
       let mut all_connected = true;
       let mut none_connected = true;
       
       for i in 0..node_vec.len() {
           for j in (i + 1)..node_vec.len() {
               let u = node_vec[i];
               let v = node_vec[j];
               
               let connected = adjacency.get(u)
                   .map(|neighbors| neighbors.contains(v))
                   .unwrap_or(false);
               
               if connected {
                   none_connected = false;
               } else {
                   all_connected = false;
               }
           }
       }
       
       if all_connected {
           ModuleType::Complete
       } else if none_connected {
           ModuleType::Empty
       } else {
           ModuleType::Prime
       }
   }

   fn find_submodules<Id: Clone + Eq + std::hash::Hash>(
       nodes: &HashSet<Id>,
       adjacency: &HashMap<Id, HashSet<Id>>,
   ) -> Vec<HashSet<Id>> {
       // Simplified: return individual nodes as submodules
       nodes.iter().map(|node| {
           let mut submodule = HashSet::new();
           submodule.insert(node.clone());
           submodule
       }).collect()
   }

   fn collect_all_modules<Id: Clone>(module: &Module<Id>) -> Vec<Module<Id>> {
       let mut all_modules = vec![module.clone()];
       for child in &module.children {
           all_modules.extend(collect_all_modules(child));
       }
       all_modules
   }

   fn calculate_modular_width<Id>(module: &Module<Id>) -> usize {
       let mut max_width = module.nodes.len();
       for child in &module.children {
           max_width = max_width.max(calculate_modular_width(child));
       }
       max_width
   }
   ```

5. Implement graph separator decomposition:
   ```rust
   #[derive(Debug, Clone)]
   pub struct Separator<Id> {
       pub separator_nodes: HashSet<Id>,
       pub components: Vec<HashSet<Id>>,
       pub separator_size: usize,
       pub balance_ratio: f32,
   }

   pub fn find_vertex_separator<G: Graph>(
       graph: &G,
       target_balance: f32,
   ) -> Option<Separator<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       if n < 3 {
           return None;
       }
       
       // Try different separator candidates
       let mut best_separator = None;
       let mut best_score = f32::INFINITY;
       
       // Try single nodes as separators
       for node in &nodes {
           let separator = test_node_separator(graph, node);
           if let Some(sep) = separator {
               let score = evaluate_separator(&sep, n, target_balance);
               if score < best_score {
                   best_score = score;
                   best_separator = Some(sep);
               }
           }
       }
       
       // Try pairs of nodes
       for i in 0..nodes.len() {
           for j in (i + 1)..nodes.len() {
               let mut separator_set = HashSet::new();
               separator_set.insert(nodes[i].clone());
               separator_set.insert(nodes[j].clone());
               
               let separator = test_separator_set(graph, &separator_set);
               if let Some(sep) = separator {
                   let score = evaluate_separator(&sep, n, target_balance);
                   if score < best_score {
                       best_score = score;
                       best_separator = Some(sep);
                   }
               }
           }
       }
       
       best_separator
   }

   fn test_node_separator<G: Graph>(
       graph: &G,
       separator_node: &G::Node::Id,
   ) -> Option<Separator<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut separator_set = HashSet::new();
       separator_set.insert(separator_node.clone());
       test_separator_set(graph, &separator_set)
   }

   fn test_separator_set<G: Graph>(
       graph: &G,
       separator_set: &HashSet<G::Node::Id>,
   ) -> Option<Separator<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Find connected components after removing separator
       let remaining_nodes: HashSet<_> = graph.nodes()
           .map(|n| n.id())
           .filter(|id| !separator_set.contains(id))
           .collect();
       
       let components = find_connected_components_subset(graph, &remaining_nodes);
       
       if components.len() >= 2 {
           let balance_ratio = calculate_balance_ratio(&components);
           
           Some(Separator {
               separator_nodes: separator_set.clone(),
               components,
               separator_size: separator_set.len(),
               balance_ratio,
           })
       } else {
           None
       }
   }

   fn find_connected_components_subset<G: Graph>(
       graph: &G,
       nodes: &HashSet<G::Node::Id>,
   ) -> Vec<HashSet<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut visited = HashSet::new();
       let mut components = Vec::new();
       
       for node_id in nodes {
           if !visited.contains(node_id) {
               let mut component = HashSet::new();
               let mut stack = vec![node_id.clone()];
               
               while let Some(current) = stack.pop() {
                   if visited.contains(&current) || !nodes.contains(&current) {
                       continue;
                   }
                   
                   visited.insert(current.clone());
                   component.insert(current.clone());
                   
                   if let Some(node) = graph.get_node(&current) {
                       for neighbor in node.neighbors() {
                           if nodes.contains(&neighbor) && !visited.contains(&neighbor) {
                               stack.push(neighbor);
                           }
                       }
                   }
               }
               
               if !component.is_empty() {
                   components.push(component);
               }
           }
       }
       
       components
   }

   fn calculate_balance_ratio(components: &[HashSet<impl Clone>]) -> f32 {
       if components.is_empty() {
           return 0.0;
       }
       
       let sizes: Vec<_> = components.iter().map(|c| c.len()).collect();
       let max_size = *sizes.iter().max().unwrap();
       let min_size = *sizes.iter().min().unwrap();
       
       if max_size == 0 {
           1.0
       } else {
           min_size as f32 / max_size as f32
       }
   }

   fn evaluate_separator(separator: &Separator<impl Clone>, total_nodes: usize, target_balance: f32) -> f32 {
       let size_penalty = separator.separator_size as f32;
       let balance_penalty = (target_balance - separator.balance_ratio).abs() * 10.0;
       size_penalty + balance_penalty
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/decomposition/mod.rs
pub trait GraphDecomposition: Graph {
    fn k_core_decomposition(&self) -> KCoreDecomposition<Self::Node::Id>;
    fn tree_decomposition(&self) -> TreeDecomposition<Self::Node::Id>;
    fn modular_decomposition(&self) -> ModularDecomposition<Self::Node::Id>;
    fn find_separator(&self, balance: f32) -> Option<Separator<Self::Node::Id>>;
    fn degeneracy_ordering(&self) -> Vec<Self::Node::Id>;
}

pub struct DecompositionAnalysis<Id> {
    pub k_core_stats: KCoreDecomposition<Id>,
    pub tree_width: usize,
    pub modular_structure: ModularDecomposition<Id>,
    pub separators: Vec<Separator<Id>>,
    pub hierarchical_levels: usize,
}
```

## Verification Steps
1. Test k-core decomposition on graphs with known core structure
2. Verify tree decomposition width bounds
3. Test modular decomposition on structured graphs
4. Validate separator balance properties
5. Benchmark decomposition algorithms on neuromorphic networks

## Time Estimate
35 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP015-MP018: Graph traversal and connectivity algorithms
- MP020: Community detection (related concepts)