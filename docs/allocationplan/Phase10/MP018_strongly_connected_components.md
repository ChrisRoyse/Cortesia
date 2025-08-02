# MP018: Strongly Connected Components

## Task Description
Implement Tarjan's and Kosaraju's algorithms for finding strongly connected components in directed neural graphs.

## Prerequisites
- MP015-MP016 completed (DFS/BFS)
- Understanding of SCC algorithms
- Stack operations knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/scc.rs`

2. Implement Tarjan's algorithm:
   ```rust
   pub struct TarjanSCC<'a, G: Graph> {
       graph: &'a G,
       index_counter: usize,
       stack: Vec<G::Node::Id>,
       indices: HashMap<G::Node::Id, usize>,
       lowlinks: HashMap<G::Node::Id, usize>,
       on_stack: HashSet<G::Node::Id>,
       sccs: Vec<Vec<G::Node::Id>>,
   }
   
   impl<'a, G: Graph> TarjanSCC<'a, G> {
       pub fn find_sccs(graph: &'a G) -> Vec<Vec<G::Node::Id>> {
           let mut tarjan = Self {
               graph,
               index_counter: 0,
               stack: Vec::new(),
               indices: HashMap::new(),
               lowlinks: HashMap::new(),
               on_stack: HashSet::new(),
               sccs: Vec::new(),
           };
           
           for node in graph.nodes() {
               if !tarjan.indices.contains_key(&node.id()) {
                   tarjan.strongconnect(node.id());
               }
           }
           
           tarjan.sccs
       }
       
       fn strongconnect(&mut self, v: G::Node::Id) {
           self.indices.insert(v.clone(), self.index_counter);
           self.lowlinks.insert(v.clone(), self.index_counter);
           self.index_counter += 1;
           self.stack.push(v.clone());
           self.on_stack.insert(v.clone());
           
           if let Some(node) = self.graph.get_node(&v) {
               for w in node.neighbors() {
                   if !self.indices.contains_key(&w) {
                       self.strongconnect(w.clone());
                       let w_lowlink = self.lowlinks[&w];
                       let v_lowlink = self.lowlinks[&v].min(w_lowlink);
                       self.lowlinks.insert(v.clone(), v_lowlink);
                   } else if self.on_stack.contains(&w) {
                       let w_index = self.indices[&w];
                       let v_lowlink = self.lowlinks[&v].min(w_index);
                       self.lowlinks.insert(v.clone(), v_lowlink);
                   }
               }
           }
           
           if self.lowlinks[&v] == self.indices[&v] {
               let mut scc = Vec::new();
               loop {
                   let w = self.stack.pop().unwrap();
                   self.on_stack.remove(&w);
                   scc.push(w.clone());
                   if w == v {
                       break;
                   }
               }
               self.sccs.push(scc);
           }
       }
   }
   ```

3. Implement Kosaraju's algorithm:
   ```rust
   pub fn kosaraju<G: Graph>(graph: &G) -> Vec<Vec<G::Node::Id>> {
       // First DFS to get finish times
       let mut visited = HashSet::new();
       let mut finish_stack = Vec::new();
       
       for node in graph.nodes() {
           if !visited.contains(&node.id()) {
               dfs_finish_time(graph, node.id(), &mut visited, &mut finish_stack);
           }
       }
       
       // Create transpose graph
       let transpose = graph.transpose();
       
       // Second DFS on transpose in reverse finish order
       let mut visited = HashSet::new();
       let mut sccs = Vec::new();
       
       while let Some(node_id) = finish_stack.pop() {
           if !visited.contains(&node_id) {
               let mut scc = Vec::new();
               dfs_collect(&transpose, node_id, &mut visited, &mut scc);
               sccs.push(scc);
           }
       }
       
       sccs
   }
   ```

4. Add SCC condensation:
   ```rust
   pub fn condensation_graph<G: Graph>(
       graph: &G,
       sccs: &[Vec<G::Node::Id>],
   ) -> CondensationGraph<G::Node::Id> {
       let mut scc_map = HashMap::new();
       for (idx, scc) in sccs.iter().enumerate() {
           for node in scc {
               scc_map.insert(node.clone(), idx);
           }
       }
       
       let mut condensed = CondensationGraph::new(sccs.len());
       
       for edge in graph.edges() {
           let source_scc = scc_map[&edge.source()];
           let target_scc = scc_map[&edge.target()];
           
           if source_scc != target_scc {
               condensed.add_edge(source_scc, target_scc);
           }
       }
       
       condensed
   }
   ```

5. Neuromorphic applications:
   - Feedback loop identification
   - Recurrent pathway analysis
   - Oscillation detection

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/scc.rs
pub trait StronglyConnectedComponents: Graph {
    fn tarjan_scc(&self) -> Vec<Vec<Self::Node::Id>>;
    fn kosaraju_scc(&self) -> Vec<Vec<Self::Node::Id>>;
    fn is_strongly_connected(&self) -> bool;
    fn condensation(&self) -> (CondensationGraph, Vec<Vec<Self::Node::Id>>);
}

pub struct SCCResult<Id> {
    pub components: Vec<Vec<Id>>,
    pub component_map: HashMap<Id, usize>,
    pub condensation: CondensationGraph,
}
```

## Verification Steps
1. Test on graphs with known SCCs
2. Verify both algorithms produce same results
3. Test condensation graph properties
4. Benchmark on large directed graphs

## Time Estimate
30 minutes

## Dependencies
- MP015: DFS implementation
- MP001-MP010: Graph infrastructure