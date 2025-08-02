# MP026: Motif Detection Algorithm

## Task Description
Implement network motif detection to identify recurring patterns in neural connectivity that may represent functional units.

## Prerequisites
- MP001-MP025 completed
- Understanding of graph motifs
- Subgraph enumeration concepts

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/motifs.rs`

2. Define common motif patterns:
   ```rust
   #[derive(Debug, Clone, PartialEq)]
   pub enum MotifType {
       Triangle,
       Square,
       FeedForwardLoop,
       FeedbackLoop,
       BiDirectionalPair,
       Star3,
       Path3,
       Custom(Vec<(usize, usize)>), // Edge list representation
   }
   
   pub struct Motif<Id> {
       pub motif_type: MotifType,
       pub nodes: Vec<Id>,
       pub edges: Vec<(Id, Id)>,
       pub significance: f32,
   }
   ```

3. Implement motif detection:
   ```rust
   pub fn detect_motifs<G: Graph>(
       graph: &G,
       motif_types: &[MotifType],
       max_size: usize,
   ) -> Vec<Motif<G::Node::Id>> {
       let mut found_motifs = Vec::new();
       
       for motif_type in motif_types {
           let motifs = find_motif_instances(graph, motif_type, max_size);
           found_motifs.extend(motifs);
       }
       
       found_motifs
   }
   
   fn find_motif_instances<G: Graph>(
       graph: &G,
       motif_type: &MotifType,
       max_size: usize,
   ) -> Vec<Motif<G::Node::Id>> {
       match motif_type {
           MotifType::Triangle => find_triangles(graph),
           MotifType::FeedForwardLoop => find_feed_forward_loops(graph),
           MotifType::Star3 => find_star_motifs(graph, 3),
           _ => find_custom_motifs(graph, motif_type, max_size),
       }
   }
   ```

4. Implement specific motif finders:
   ```rust
   fn find_triangles<G: Graph>(graph: &G) -> Vec<Motif<G::Node::Id>> {
       let mut triangles = Vec::new();
       
       for node_a in graph.nodes() {
           for neighbor_b in node_a.neighbors() {
               for neighbor_c in node_a.neighbors() {
                   if neighbor_b != neighbor_c && graph.has_edge(&neighbor_b, &neighbor_c) {
                       let triangle = Motif {
                           motif_type: MotifType::Triangle,
                           nodes: vec![node_a.id(), neighbor_b, neighbor_c],
                           edges: vec![
                               (node_a.id(), neighbor_b.clone()),
                               (node_a.id(), neighbor_c.clone()),
                               (neighbor_b, neighbor_c),
                           ],
                           significance: 1.0,
                       };
                       triangles.push(triangle);
                   }
               }
           }
       }
       
       // Remove duplicates
       deduplicate_motifs(triangles)
   }
   
   fn find_feed_forward_loops<G: Graph>(graph: &G) -> Vec<Motif<G::Node::Id>> {
       let mut ffls = Vec::new();
       
       for node_a in graph.nodes() {
           for neighbor_b in node_a.neighbors() {
               for neighbor_c in node_a.neighbors() {
                   if neighbor_b != neighbor_c && graph.has_edge(&neighbor_b, &neighbor_c) {
                       // Check if it's a feed-forward (no back edge from C to A)
                       if !graph.has_edge(&neighbor_c, &node_a.id()) {
                           let ffl = Motif {
                               motif_type: MotifType::FeedForwardLoop,
                               nodes: vec![node_a.id(), neighbor_b.clone(), neighbor_c.clone()],
                               edges: vec![
                                   (node_a.id(), neighbor_b.clone()),
                                   (node_a.id(), neighbor_c.clone()),
                                   (neighbor_b, neighbor_c),
                               ],
                               significance: calculate_ffl_significance(graph, &node_a.id(), &neighbor_b, &neighbor_c),
                           };
                           ffls.push(ffl);
                       }
                   }
               }
           }
       }
       
       ffls
   }
   ```

5. Add neuromorphic-specific motifs:
   ```rust
   pub fn detect_neural_motifs<G: NeuromorphicGraph>(
       graph: &G,
   ) -> Vec<NeuralMotif<G::Node::Id>> {
       let mut neural_motifs = Vec::new();
       
       // Detect excitatory-inhibitory loops
       neural_motifs.extend(find_ei_loops(graph));
       
       // Detect convergent/divergent patterns
       neural_motifs.extend(find_convergent_patterns(graph));
       
       // Detect oscillatory circuits
       neural_motifs.extend(find_oscillatory_circuits(graph));
       
       neural_motifs
   }
   ```

## Expected Output
```rust
pub struct MotifResult<Id> {
    pub motifs: Vec<Motif<Id>>,
    pub counts: HashMap<MotifType, usize>,
    pub significance_scores: HashMap<MotifType, f32>,
}

pub trait MotifDetection: Graph {
    fn detect_motifs(&self, types: &[MotifType]) -> MotifResult<Self::Node::Id>;
    fn count_triangles(&self) -> usize;
    fn find_feed_forward_loops(&self) -> Vec<Motif<Self::Node::Id>>;
    fn motif_significance(&self, motif: &Motif<Self::Node::Id>) -> f32;
}
```

## Verification Steps
1. Test on graphs with known motif counts
2. Verify motif uniqueness (no duplicates)
3. Test significance scoring
4. Compare with reference motif detection tools

## Time Estimate
30 minutes

## Dependencies
- MP001-MP025: Graph infrastructure and basic algorithms
- MP025: Clustering coefficient for motif significance