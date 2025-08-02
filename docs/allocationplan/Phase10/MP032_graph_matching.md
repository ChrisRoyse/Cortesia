# MP032: Graph Matching

## Task Description
Implement graph matching algorithms to find optimal pairings of nodes, including maximum matching, perfect matching, and weighted matching for neural resource allocation and connectivity optimization.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP019: Max flow algorithms (for matching via flows)
- Understanding of bipartite graphs and augmenting paths

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/matching/mod.rs`

2. Implement maximum matching using augmenting paths:
   ```rust
   use std::collections::{HashMap, HashSet, VecDeque};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct Matching<Id> {
       pub edges: Vec<(Id, Id)>,
       pub matched_nodes: HashSet<Id>,
       pub size: usize,
       pub is_perfect: bool,
       pub is_maximum: bool,
   }

   pub fn maximum_matching_bipartite<G: Graph>(
       graph: &G,
       left_nodes: &HashSet<G::Node::Id>,
       right_nodes: &HashSet<G::Node::Id>,
   ) -> Matching<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut matching_edges = Vec::new();
       let mut matched_left = HashSet::new();
       let mut matched_right = HashSet::new();
       
       // Find augmenting paths until no more exist
       loop {
           if let Some(augmenting_path) = find_augmenting_path_bfs(
               graph,
               left_nodes,
               right_nodes,
               &matched_left,
               &matched_right,
           ) {
               // Update matching along the augmenting path
               update_matching_with_path(
                   &augmenting_path,
                   &mut matching_edges,
                   &mut matched_left,
                   &mut matched_right,
               );
           } else {
               break; // No more augmenting paths
           }
       }
       
       let matched_nodes: HashSet<_> = matched_left.union(&matched_right).cloned().collect();
       let size = matching_edges.len();
       let is_perfect = matched_nodes.len() == left_nodes.len() + right_nodes.len();
       
       Matching {
           edges: matching_edges,
           matched_nodes,
           size,
           is_perfect,
           is_maximum: true, // Algorithm guarantees maximum matching
       }
   }

   fn find_augmenting_path_bfs<G: Graph>(
       graph: &G,
       left_nodes: &HashSet<G::Node::Id>,
       right_nodes: &HashSet<G::Node::Id>,
       matched_left: &HashSet<G::Node::Id>,
       matched_right: &HashSet<G::Node::Id>,
   ) -> Option<Vec<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut queue = VecDeque::new();
       let mut parent: HashMap<G::Node::Id, G::Node::Id> = HashMap::new();
       let mut visited = HashSet::new();
       
       // Start BFS from unmatched left nodes
       for node_id in left_nodes {
           if !matched_left.contains(node_id) {
               queue.push_back(node_id.clone());
               visited.insert(node_id.clone());
           }
       }
       
       while let Some(current) = queue.pop_front() {
           if let Some(node) = graph.get_node(&current) {
               for neighbor in node.neighbors() {
                   if visited.contains(&neighbor) {
                       continue;
                   }
                   
                   if left_nodes.contains(&current) && right_nodes.contains(&neighbor) {
                       // Left to right edge
                       if !matched_right.contains(&neighbor) {
                           // Found unmatched right node - augmenting path found
                           parent.insert(neighbor.clone(), current);
                           return Some(reconstruct_path(&parent, neighbor));
                       } else {
                           // Right node is matched, continue through its match
                           parent.insert(neighbor.clone(), current);
                           visited.insert(neighbor.clone());
                           queue.push_back(neighbor);
                       }
                   } else if right_nodes.contains(&current) && left_nodes.contains(&neighbor) {
                       // Right to left edge (must be a matching edge)
                       if is_matching_edge(&current, &neighbor, matched_left, matched_right) {
                           parent.insert(neighbor.clone(), current);
                           visited.insert(neighbor.clone());
                           queue.push_back(neighbor);
                       }
                   }
               }
           }
       }
       
       None // No augmenting path found
   }

   fn is_matching_edge<Id: Eq + std::hash::Hash>(
       left: &Id,
       right: &Id,
       matched_left: &HashSet<Id>,
       matched_right: &HashSet<Id>,
   ) -> bool {
       matched_left.contains(right) && matched_right.contains(left)
   }

   fn reconstruct_path<Id: Clone + Eq + std::hash::Hash>(
       parent: &HashMap<Id, Id>,
       end_node: Id,
   ) -> Vec<Id> {
       let mut path = Vec::new();
       let mut current = end_node;
       
       path.push(current.clone());
       while let Some(prev) = parent.get(&current) {
           path.push(prev.clone());
           current = prev.clone();
       }
       
       path.reverse();
       path
   }

   fn update_matching_with_path<Id: Clone + Eq + std::hash::Hash>(
       path: &[Id],
       matching_edges: &mut Vec<(Id, Id)>,
       matched_left: &mut HashSet<Id>,
       matched_right: &mut HashSet<Id>,
   ) {
       // Remove existing matching edges that are in the path
       let path_edges: HashSet<_> = path.windows(2)
           .map(|pair| (pair[0].clone(), pair[1].clone()))
           .collect();
       
       matching_edges.retain(|(u, v)| {
           !path_edges.contains(&(u.clone(), v.clone())) && 
           !path_edges.contains(&(v.clone(), u.clone()))
       });
       
       // Add new matching edges (every other edge in the path)
       for i in (0..path.len()-1).step_by(2) {
           let u = &path[i];
           let v = &path[i + 1];
           matching_edges.push((u.clone(), v.clone()));
           matched_left.insert(u.clone());
           matched_right.insert(v.clone());
       }
   }
   ```

3. Implement Hungarian algorithm for weighted bipartite matching:
   ```rust
   pub fn hungarian_algorithm<G: Graph>(
       graph: &G,
       left_nodes: &[G::Node::Id],
       right_nodes: &[G::Node::Id],
       weight_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
   ) -> (Matching<G::Node::Id>, f32) 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let n = left_nodes.len();
       let m = right_nodes.len();
       
       if n == 0 || m == 0 {
           return (Matching {
               edges: vec![],
               matched_nodes: HashSet::new(),
               size: 0,
               is_perfect: false,
               is_maximum: true,
           }, 0.0);
       }
       
       // Build cost matrix (negate weights for maximum weight matching)
       let mut cost_matrix = vec![vec![f32::INFINITY; m]; n];
       
       for (i, left_node) in left_nodes.iter().enumerate() {
           if let Some(node) = graph.get_node(left_node) {
               for neighbor in node.neighbors() {
                   if let Some(j) = right_nodes.iter().position(|r| r == &neighbor) {
                       cost_matrix[i][j] = -weight_fn(left_node, &neighbor);
                   }
               }
           }
       }
       
       // Run Hungarian algorithm
       let assignment = solve_hungarian(&cost_matrix);
       
       // Build matching from assignment
       let mut matching_edges = Vec::new();
       let mut matched_nodes = HashSet::new();
       let mut total_weight = 0.0;
       
       for (i, &j) in assignment.iter().enumerate() {
           if j < m && cost_matrix[i][j] < f32::INFINITY {
               let left_node = &left_nodes[i];
               let right_node = &right_nodes[j];
               matching_edges.push((left_node.clone(), right_node.clone()));
               matched_nodes.insert(left_node.clone());
               matched_nodes.insert(right_node.clone());
               total_weight += -cost_matrix[i][j]; // Convert back to original weight
           }
       }
       
       let matching = Matching {
           size: matching_edges.len(),
           is_perfect: matching_edges.len() == n.min(m),
           is_maximum: true,
           edges: matching_edges,
           matched_nodes,
       };
       
       (matching, total_weight)
   }

   fn solve_hungarian(cost_matrix: &[Vec<f32>]) -> Vec<usize> {
       let n = cost_matrix.len();
       let m = if n > 0 { cost_matrix[0].len() } else { 0 };
       
       if n == 0 || m == 0 {
           return vec![];
       }
       
       // Make square matrix by padding
       let size = n.max(m);
       let mut matrix = vec![vec![f32::INFINITY; size]; size];
       
       for i in 0..n {
           for j in 0..m {
               matrix[i][j] = cost_matrix[i][j];
           }
       }
       
       // Hungarian algorithm implementation (simplified)
       let mut u = vec![0.0; size];  // Labels for left vertices
       let mut v = vec![0.0; size];  // Labels for right vertices
       let mut assignment = vec![size; size]; // Assignment result
       
       // Initialize labels
       for i in 0..size {
           u[i] = matrix[i].iter().copied().fold(f32::INFINITY, f32::min);
       }
       
       // Find maximum matching
       for i in 0..size {
           loop {
               let mut visited_left = vec![false; size];
               let mut visited_right = vec![false; size];
               
               if find_augmenting_path_hungarian(
                   &matrix,
                   i,
                   &u,
                   &v,
                   &mut assignment,
                   &mut visited_left,
                   &mut visited_right,
               ) {
                   break;
               }
               
               // Update labels
               let delta = calculate_delta(&matrix, &u, &v, &visited_left, &visited_right);
               
               for j in 0..size {
                   if visited_left[j] {
                       u[j] += delta;
                   }
                   if visited_right[j] {
                       v[j] -= delta;
                   }
               }
           }
       }
       
       assignment[0..n].to_vec()
   }

   fn find_augmenting_path_hungarian(
       matrix: &[Vec<f32>],
       start: usize,
       u: &[f32],
       v: &[f32],
       assignment: &mut [usize],
       visited_left: &mut [bool],
       visited_right: &mut [bool],
   ) -> bool {
       if visited_left[start] {
           return false;
       }
       
       visited_left[start] = true;
       
       let size = matrix.len();
       for j in 0..size {
           if (u[start] + v[j] - matrix[start][j]).abs() < 1e-9 {
               if !visited_right[j] {
                   visited_right[j] = true;
                   
                   if assignment[j] == size || find_augmenting_path_hungarian(
                       matrix,
                       assignment[j],
                       u,
                       v,
                       assignment,
                       visited_left,
                       visited_right,
                   ) {
                       assignment[j] = start;
                       return true;
                   }
               }
           }
       }
       
       false
   }

   fn calculate_delta(
       matrix: &[Vec<f32>],
       u: &[f32],
       v: &[f32],
       visited_left: &[bool],
       visited_right: &[bool],
   ) -> f32 {
       let mut delta = f32::INFINITY;
       let size = matrix.len();
       
       for i in 0..size {
           if visited_left[i] {
               for j in 0..size {
                   if !visited_right[j] {
                       delta = delta.min(u[i] + v[j] - matrix[i][j]);
                   }
               }
           }
       }
       
       delta
   }
   ```

4. Implement general maximum matching using Edmonds' blossom algorithm:
   ```rust
   pub fn maximum_matching_general<G: Graph>(
       graph: &G,
   ) -> Matching<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Simplified implementation - for production use, implement full Edmonds' algorithm
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let mut matching_edges = Vec::new();
       let mut matched_nodes = HashSet::new();
       
       // Greedy approximation for general graphs
       for node_id in &nodes {
           if matched_nodes.contains(node_id) {
               continue;
           }
           
           if let Some(node) = graph.get_node(node_id) {
               for neighbor in node.neighbors() {
                   if !matched_nodes.contains(&neighbor) {
                       // Found unmatched edge
                       matching_edges.push((node_id.clone(), neighbor.clone()));
                       matched_nodes.insert(node_id.clone());
                       matched_nodes.insert(neighbor);
                       break;
                   }
               }
           }
       }
       
       let size = matching_edges.len();
       let is_perfect = matched_nodes.len() == nodes.len();
       
       Matching {
           edges: matching_edges,
           matched_nodes,
           size,
           is_perfect,
           is_maximum: false, // Greedy is not guaranteed to be maximum
       }
   }
   ```

5. Implement stable matching (Gale-Shapley algorithm):
   ```rust
   pub fn stable_matching<Id: Clone + Eq + std::hash::Hash>(
       men_preferences: &HashMap<Id, Vec<Id>>,
       women_preferences: &HashMap<Id, Vec<Id>>,
   ) -> HashMap<Id, Id> {
       let mut men_to_women = HashMap::new();
       let mut women_to_men = HashMap::new();
       let mut free_men: Vec<_> = men_preferences.keys().cloned().collect();
       let mut proposal_count: HashMap<Id, usize> = HashMap::new();
       
       for man in men_preferences.keys() {
           proposal_count.insert(man.clone(), 0);
       }
       
       while let Some(man) = free_men.pop() {
           let proposals_made = proposal_count.get(&man).copied().unwrap_or(0);
           
           if let Some(man_prefs) = men_preferences.get(&man) {
               if proposals_made < man_prefs.len() {
                   let woman = &man_prefs[proposals_made];
                   proposal_count.insert(man.clone(), proposals_made + 1);
                   
                   if let Some(current_partner) = women_to_men.get(woman) {
                       // Woman is already matched
                       if let Some(woman_prefs) = women_preferences.get(woman) {
                           let current_rank = woman_prefs.iter().position(|w| w == current_partner);
                           let new_rank = woman_prefs.iter().position(|w| w == &man);
                           
                           if let (Some(curr_rank), Some(new_rank)) = (current_rank, new_rank) {
                               if new_rank < curr_rank {
                                   // Woman prefers new man
                                   men_to_women.remove(current_partner);
                                   free_men.push(current_partner.clone());
                                   
                                   men_to_women.insert(man.clone(), woman.clone());
                                   women_to_men.insert(woman.clone(), man);
                               } else {
                                   // Woman prefers current partner
                                   free_men.push(man);
                               }
                           }
                       }
                   } else {
                       // Woman is free
                       men_to_women.insert(man.clone(), woman.clone());
                       women_to_men.insert(woman.clone(), man);
                   }
               }
           }
       }
       
       men_to_women
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/matching/mod.rs
pub trait GraphMatching: Graph {
    fn maximum_matching_bipartite(&self, left: &HashSet<Self::Node::Id>, right: &HashSet<Self::Node::Id>) -> Matching<Self::Node::Id>;
    fn maximum_matching_general(&self) -> Matching<Self::Node::Id>;
    fn weighted_matching(&self, weight_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32) -> (Matching<Self::Node::Id>, f32);
    fn perfect_matching(&self) -> Option<Matching<Self::Node::Id>>;
    fn min_cost_perfect_matching(&self, cost_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32) -> Option<(Matching<Self::Node::Id>, f32)>;
}

pub struct MatchingAnalysis<Id> {
    pub matching: Matching<Id>,
    pub matching_number: usize,
    pub vertex_cover_size: usize,
    pub independence_number: usize,
    pub is_bipartite: bool,
}
```

## Verification Steps
1. Test matching algorithms on bipartite graphs with known optimal matchings
2. Verify Hungarian algorithm optimality on weighted instances
3. Compare maximum matching sizes with theoretical bounds
4. Test stable matching on preference instances
5. Benchmark performance on neuromorphic network topologies

## Time Estimate
35 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP019: Max flow algorithms (for flow-based matching)