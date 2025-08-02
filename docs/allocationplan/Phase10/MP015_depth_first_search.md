# MP015: Depth-First Search Implementation

## Task Description
Implement depth-first search (DFS) with applications for cycle detection, topological sorting, and neural pathway exploration.

## Prerequisites
- MP001-MP014 completed
- Understanding of DFS algorithm
- Stack data structure knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/traversal.rs`

2. Implement basic DFS:
   ```rust
   pub fn dfs<G: Graph, F>(
       graph: &G,
       start: G::Node::Id,
       mut visit_fn: F,
   ) where
       F: FnMut(&G::Node::Id, &G::Node) -> ControlFlow<()>,
   {
       let mut visited = HashSet::new();
       let mut stack = vec![start];
       
       while let Some(node_id) = stack.pop() {
           if visited.insert(node_id.clone()) {
               if let Some(node) = graph.get_node(&node_id) {
                   if let ControlFlow::Break(_) = visit_fn(&node_id, &node) {
                       return;
                   }
                   
                   for neighbor in node.neighbors() {
                       if !visited.contains(&neighbor) {
                           stack.push(neighbor);
                       }
                   }
               }
           }
       }
   }
   ```

3. Implement iterative DFS with path tracking:
   ```rust
   pub struct DFSIterator<'a, G: Graph> {
       graph: &'a G,
       stack: Vec<(G::Node::Id, Vec<G::Node::Id>)>,
       visited: HashSet<G::Node::Id>,
   }
   
   impl<'a, G: Graph> Iterator for DFSIterator<'a, G> {
       type Item = (G::Node::Id, Vec<G::Node::Id>);
       
       fn next(&mut self) -> Option<Self::Item> {
           while let Some((node_id, path)) = self.stack.pop() {
               if self.visited.insert(node_id.clone()) {
                   let mut new_path = path.clone();
                   new_path.push(node_id.clone());
                   
                   if let Some(node) = self.graph.get_node(&node_id) {
                       for neighbor in node.neighbors() {
                           if !self.visited.contains(&neighbor) {
                               self.stack.push((neighbor, new_path.clone()));
                           }
                       }
                   }
                   
                   return Some((node_id, new_path));
               }
           }
           None
       }
   }
   ```

4. Implement DFS applications:
   ```rust
   pub fn has_cycle<G: Graph>(graph: &G) -> bool {
       let mut visited = HashSet::new();
       let mut rec_stack = HashSet::new();
       
       for node in graph.nodes() {
           if !visited.contains(&node.id()) {
               if dfs_cycle_detect(graph, node.id(), &mut visited, &mut rec_stack) {
                   return true;
               }
           }
       }
       false
   }
   
   pub fn topological_sort<G: Graph>(graph: &G) -> Result<Vec<G::Node::Id>, CycleError> {
       let mut visited = HashSet::new();
       let mut stack = Vec::new();
       
       for node in graph.nodes() {
           if !visited.contains(&node.id()) {
               dfs_topo_sort(graph, node.id(), &mut visited, &mut stack)?;
           }
       }
       
       stack.reverse();
       Ok(stack)
   }
   ```

5. Add neuromorphic-specific DFS variants:
   - Activity-guided search
   - Energy-aware traversal
   - Spike propagation paths

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/traversal.rs
pub trait DepthFirstSearch: Graph {
    fn dfs<F>(&self, start: Self::Node::Id, visit: F)
    where
        F: FnMut(&Self::Node::Id, &Self::Node) -> ControlFlow<()>;
    
    fn dfs_iter(&self, start: Self::Node::Id) -> DFSIterator<Self>;
    
    fn find_path_dfs(
        &self,
        start: Self::Node::Id,
        end: Self::Node::Id,
    ) -> Option<Vec<Self::Node::Id>>;
    
    fn connected_components(&self) -> Vec<Vec<Self::Node::Id>>;
}

pub struct DFSResult<Id> {
    pub visited_order: Vec<Id>,
    pub parent_map: HashMap<Id, Id>,
    pub discovery_time: HashMap<Id, usize>,
    pub finish_time: HashMap<Id, usize>,
}
```

## Verification Steps
1. Test DFS traversal order
2. Verify cycle detection accuracy
3. Test topological sort on DAGs
4. Benchmark against recursive implementation

## Time Estimate
25 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP011-MP014: For comparison with other algorithms