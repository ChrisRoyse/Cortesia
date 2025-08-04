# MP017: Minimum Spanning Tree Algorithms

## Task Description
Implement Kruskal's and Prim's algorithms for finding minimum spanning trees, useful for optimizing neural network connectivity.

## Prerequisites
- MP001-MP016 completed
- Understanding of MST algorithms
- Union-Find data structure knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/mst.rs`

2. Implement Union-Find structure:
   ```rust
   pub struct UnionFind<Id: Clone + Eq + Hash> {
       parent: HashMap<Id, Id>,
       rank: HashMap<Id, usize>,
   }
   
   impl<Id: Clone + Eq + Hash> UnionFind<Id> {
       pub fn new() -> Self {
           Self {
               parent: HashMap::new(),
               rank: HashMap::new(),
           }
       }
       
       pub fn make_set(&mut self, x: Id) {
           self.parent.insert(x.clone(), x.clone());
           self.rank.insert(x, 0);
       }
       
       pub fn find(&mut self, x: &Id) -> Id {
           if self.parent[x] != *x {
               let root = self.find(&self.parent[x].clone());
               self.parent.insert(x.clone(), root.clone());
           }
           self.parent[x].clone()
       }
       
       pub fn union(&mut self, x: &Id, y: &Id) -> bool {
           let root_x = self.find(x);
           let root_y = self.find(y);
           
           if root_x == root_y {
               return false;
           }
           
           match self.rank[&root_x].cmp(&self.rank[&root_y]) {
               Ordering::Less => self.parent.insert(root_x, root_y),
               Ordering::Greater => self.parent.insert(root_y, root_x),
               Ordering::Equal => {
                   self.parent.insert(root_y, root_x.clone());
                   *self.rank.get_mut(&root_x).unwrap() += 1;
                   Some(root_x)
               }
           };
           true
       }
   }
   ```

3. Implement Kruskal's algorithm:
   ```rust
   pub fn kruskal<G: Graph>(graph: &G) -> MSTResult<G::Node::Id> {
       let mut edges: Vec<_> = graph.edges().collect();
       edges.sort_by(|a, b| a.weight().partial_cmp(&b.weight()).unwrap());
       
       let mut uf = UnionFind::new();
       for node in graph.nodes() {
           uf.make_set(node.id());
       }
       
       let mut mst_edges = Vec::new();
       let mut total_weight = 0.0;
       
       for edge in edges {
           if uf.union(&edge.source(), &edge.target()) {
               total_weight += edge.weight();
               mst_edges.push(MSTEdge {
                   source: edge.source(),
                   target: edge.target(),
                   weight: edge.weight(),
               });
               
               if mst_edges.len() == graph.node_count() - 1 {
                   break;
               }
           }
       }
       
       MSTResult {
           edges: mst_edges,
           total_weight,
       }
   }
   ```

4. Implement Prim's algorithm:
   ```rust
   pub fn prim<G: Graph>(graph: &G, start: G::Node::Id) -> MSTResult<G::Node::Id> {
       let mut in_mst = HashSet::new();
       let mut mst_edges = Vec::new();
       let mut total_weight = 0.0;
       let mut heap = BinaryHeap::new();
       
       in_mst.insert(start.clone());
       
       // Add all edges from start to heap
       if let Some(node) = graph.get_node(&start) {
           for neighbor in node.neighbors() {
               if let Some(weight) = graph.edge_weight(&start, &neighbor) {
                   heap.push(Reverse((OrderedFloat(weight), start.clone(), neighbor)));
               }
           }
       }
       
       while let Some(Reverse((weight, source, target))) = heap.pop() {
           if in_mst.contains(&target) {
               continue;
           }
           
           in_mst.insert(target.clone());
           total_weight += weight.0;
           mst_edges.push(MSTEdge {
               source,
               target: target.clone(),
               weight: weight.0,
           });
           
           // Add new edges from target
           if let Some(node) = graph.get_node(&target) {
               for neighbor in node.neighbors() {
                   if !in_mst.contains(&neighbor) {
                       if let Some(w) = graph.edge_weight(&target, &neighbor) {
                           heap.push(Reverse((OrderedFloat(w), target.clone(), neighbor)));
                       }
                   }
               }
           }
       }
       
       MSTResult {
           edges: mst_edges,
           total_weight,
       }
   }
   ```

5. Add neuromorphic applications:
   - Minimal wiring length
   - Energy-efficient connectivity
   - Redundancy elimination

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/mst.rs
pub struct MSTEdge<Id> {
    pub source: Id,
    pub target: Id,
    pub weight: f32,
}

pub struct MSTResult<Id> {
    pub edges: Vec<MSTEdge<Id>>,
    pub total_weight: f32,
}

pub trait MinimumSpanningTree: Graph {
    fn kruskal(&self) -> MSTResult<Self::Node::Id>;
    fn prim(&self, start: Self::Node::Id) -> MSTResult<Self::Node::Id>;
    fn boruvka(&self) -> MSTResult<Self::Node::Id>;
}
```

## Verification Steps
1. Test on known graphs with verified MSTs
2. Compare Kruskal and Prim results
3. Verify forest handling for disconnected graphs
4. Benchmark performance on dense vs sparse graphs

## Time Estimate
30 minutes

## Dependencies
- MP001-MP016: Graph infrastructure and traversal
- Priority queue and Union-Find structures