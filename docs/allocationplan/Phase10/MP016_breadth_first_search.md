# MP016: Breadth-First Search Implementation

## Task Description
Implement breadth-first search (BFS) for level-wise exploration, shortest unweighted paths, and neural layer analysis.

## Prerequisites
- MP015 completed (DFS implementation)
- Understanding of BFS algorithm
- Queue data structure knowledge

## Detailed Steps

1. Extend `src/neuromorphic/graph/algorithms/traversal.rs`

2. Implement basic BFS:
   ```rust
   use std::collections::VecDeque;
   
   pub fn bfs<G: Graph, F>(
       graph: &G,
       start: G::Node::Id,
       mut visit_fn: F,
   ) where
       F: FnMut(&G::Node::Id, &G::Node, usize) -> ControlFlow<()>,
   {
       let mut visited = HashSet::new();
       let mut queue = VecDeque::new();
       
       queue.push_back((start, 0));
       visited.insert(start);
       
       while let Some((node_id, level)) = queue.pop_front() {
           if let Some(node) = graph.get_node(&node_id) {
               if let ControlFlow::Break(_) = visit_fn(&node_id, &node, level) {
                   return;
               }
               
               for neighbor in node.neighbors() {
                   if visited.insert(neighbor.clone()) {
                       queue.push_back((neighbor, level + 1));
                   }
               }
           }
       }
   }
   ```

3. Implement BFS iterator:
   ```rust
   pub struct BFSIterator<'a, G: Graph> {
       graph: &'a G,
       queue: VecDeque<(G::Node::Id, usize)>,
       visited: HashSet<G::Node::Id>,
   }
   
   impl<'a, G: Graph> Iterator for BFSIterator<'a, G> {
       type Item = (G::Node::Id, usize);
       
       fn next(&mut self) -> Option<Self::Item> {
           while let Some((node_id, level)) = self.queue.pop_front() {
               if let Some(node) = self.graph.get_node(&node_id) {
                   for neighbor in node.neighbors() {
                       if self.visited.insert(neighbor.clone()) {
                           self.queue.push_back((neighbor, level + 1));
                       }
                   }
               }
               return Some((node_id, level));
           }
           None
       }
   }
   ```

4. Implement BFS applications:
   ```rust
   pub fn shortest_unweighted_path<G: Graph>(
       graph: &G,
       start: G::Node::Id,
       end: G::Node::Id,
   ) -> Option<Vec<G::Node::Id>> {
       let mut visited = HashSet::new();
       let mut queue = VecDeque::new();
       let mut parent = HashMap::new();
       
       queue.push_back(start.clone());
       visited.insert(start.clone());
       
       while let Some(current) = queue.pop_front() {
           if current == end {
               return Some(reconstruct_path(&parent, end));
           }
           
           if let Some(node) = graph.get_node(&current) {
               for neighbor in node.neighbors() {
                   if visited.insert(neighbor.clone()) {
                       parent.insert(neighbor.clone(), current.clone());
                       queue.push_back(neighbor);
                   }
               }
           }
       }
       None
   }
   
   pub fn graph_diameter<G: Graph>(graph: &G) -> Option<usize> {
       let mut max_distance = 0;
       
       for node in graph.nodes() {
           let distances = bfs_distances(graph, node.id());
           if let Some(&max) = distances.values().max() {
               max_distance = max_distance.max(max);
           }
       }
       
       Some(max_distance)
   }
   ```

5. Add neuromorphic-specific BFS:
   - Layer-wise activation propagation
   - Receptive field calculation
   - Signal spread analysis

## Expected Output
```rust
// Extended traversal.rs
pub trait BreadthFirstSearch: Graph {
    fn bfs<F>(&self, start: Self::Node::Id, visit: F)
    where
        F: FnMut(&Self::Node::Id, &Self::Node, usize) -> ControlFlow<()>;
    
    fn bfs_iter(&self, start: Self::Node::Id) -> BFSIterator<Self>;
    
    fn bfs_tree(&self, start: Self::Node::Id) -> BFSTree<Self::Node::Id>;
    
    fn level_order(&self, start: Self::Node::Id) -> Vec<Vec<Self::Node::Id>>;
}

pub struct BFSResult<Id> {
    pub distances: HashMap<Id, usize>,
    pub parents: HashMap<Id, Id>,
    pub levels: Vec<Vec<Id>>,
}
```

## Verification Steps
1. Test BFS traversal order
2. Verify shortest path correctness
3. Test level-order traversal
4. Compare performance with DFS

## Time Estimate
25 minutes

## Dependencies
- MP015: DFS implementation for comparison
- MP001-MP010: Graph infrastructure