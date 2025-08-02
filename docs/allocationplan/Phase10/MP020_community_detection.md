# MP020: Community Detection Algorithms

## Task Description
Implement Louvain and Label Propagation algorithms for detecting communities in neural networks, identifying functional modules.

## Prerequisites
- MP001-MP019 completed
- Understanding of modularity
- Graph clustering concepts

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/community.rs`

2. Implement modularity calculation:
   ```rust
   pub fn modularity<G: Graph>(
       graph: &G,
       communities: &HashMap<G::Node::Id, usize>,
   ) -> f32 {
       let m = graph.edge_count() as f32;
       let mut q = 0.0;
       
       for edge in graph.edges() {
           let ci = communities[&edge.source()];
           let cj = communities[&edge.target()];
           
           if ci == cj {
               let ki = graph.get_node(&edge.source())
                   .map(|n| n.neighbors().count())
                   .unwrap_or(0) as f32;
               let kj = graph.get_node(&edge.target())
                   .map(|n| n.neighbors().count())
                   .unwrap_or(0) as f32;
               
               q += edge.weight() - (ki * kj) / (2.0 * m);
           }
       }
       
       q / (2.0 * m)
   }
   ```

3. Implement Louvain algorithm:
   ```rust
   pub fn louvain<G: Graph>(graph: &G) -> CommunityResult<G::Node::Id> {
       let mut communities: HashMap<G::Node::Id, usize> = HashMap::new();
       let mut node_to_comm: HashMap<G::Node::Id, usize> = HashMap::new();
       
       // Initialize each node in its own community
       for (idx, node) in graph.nodes().enumerate() {
           communities.insert(node.id(), idx);
           node_to_comm.insert(node.id(), idx);
       }
       
       let mut improvement = true;
       let mut current_modularity = modularity(graph, &communities);
       
       while improvement {
           improvement = false;
           
           // Phase 1: Local optimization
           for node in graph.nodes() {
               let node_id = node.id();
               let current_comm = communities[&node_id];
               
               // Remove node from its community
               let mut best_comm = current_comm;
               let mut best_gain = 0.0;
               
               // Try moving to neighbor communities
               let neighbor_comms = get_neighbor_communities(graph, &node_id, &communities);
               
               for &comm in &neighbor_comms {
                   if comm != current_comm {
                       // Calculate modularity gain
                       let gain = calculate_modularity_gain(
                           graph,
                           &node_id,
                           current_comm,
                           comm,
                           &communities,
                       );
                       
                       if gain > best_gain {
                           best_gain = gain;
                           best_comm = comm;
                       }
                   }
               }
               
               if best_comm != current_comm {
                   communities.insert(node_id, best_comm);
                   improvement = true;
               }
           }
           
           if !improvement {
               break;
           }
           
           // Phase 2: Build new graph
           let condensed = build_community_graph(graph, &communities);
           // Recursively apply to condensed graph
       }
       
       CommunityResult {
           communities: communities.into_iter()
               .fold(HashMap::new(), |mut acc, (node, comm)| {
                   acc.entry(comm).or_insert_with(Vec::new).push(node);
                   acc
               })
               .into_values()
               .collect(),
           modularity: current_modularity,
       }
   }
   ```

4. Implement Label Propagation:
   ```rust
   pub fn label_propagation<G: Graph>(
       graph: &G,
       max_iterations: usize,
   ) -> CommunityResult<G::Node::Id> {
       let mut labels: HashMap<G::Node::Id, usize> = HashMap::new();
       let mut rng = thread_rng();
       
       // Initialize each node with unique label
       for (idx, node) in graph.nodes().enumerate() {
           labels.insert(node.id(), idx);
       }
       
       for _ in 0..max_iterations {
           let mut changed = false;
           let mut nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
           nodes.shuffle(&mut rng);
           
           for node_id in nodes {
               if let Some(node) = graph.get_node(&node_id) {
                   // Count neighbor labels
                   let mut label_counts: HashMap<usize, f32> = HashMap::new();
                   
                   for neighbor in node.neighbors() {
                       if let Some(&label) = labels.get(&neighbor) {
                           let weight = graph.edge_weight(&node_id, &neighbor)
                               .unwrap_or(1.0);
                           *label_counts.entry(label).or_insert(0.0) += weight;
                       }
                   }
                   
                   // Select most frequent label
                   if let Some((&new_label, _)) = label_counts.iter()
                       .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                       if labels[&node_id] != new_label {
                           labels.insert(node_id, new_label);
                           changed = true;
                       }
                   }
               }
           }
           
           if !changed {
               break;
           }
       }
       
       // Convert labels to communities
       let communities = labels.into_iter()
           .fold(HashMap::new(), |mut acc, (node, label)| {
               acc.entry(label).or_insert_with(Vec::new).push(node);
               acc
           })
           .into_values()
           .collect();
       
       CommunityResult {
           communities,
           modularity: 0.0, // Calculate if needed
       }
   }
   ```

5. Add neuromorphic community metrics:
   - Functional modularity
   - Information flow between communities
   - Community synchronization

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/community.rs
pub struct CommunityResult<Id> {
    pub communities: Vec<Vec<Id>>,
    pub modularity: f32,
}

pub trait CommunityDetection: Graph {
    fn louvain(&self) -> CommunityResult<Self::Node::Id>;
    fn label_propagation(&self, max_iter: usize) -> CommunityResult<Self::Node::Id>;
    fn spectral_clustering(&self, k: usize) -> CommunityResult<Self::Node::Id>;
}

pub struct CommunityMetrics {
    pub modularity: f32,
    pub coverage: f32,
    pub conductance: f32,
}
```

## Verification Steps
1. Test on graphs with known community structure
2. Verify modularity improvement
3. Compare different algorithms
4. Test scalability on large graphs

## Time Estimate
30 minutes

## Dependencies
- MP001-MP019: Complete graph infrastructure
- Random number generation for label propagation