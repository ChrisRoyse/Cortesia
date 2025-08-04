# MP021: PageRank Algorithm Implementation

## Task Description
Implement PageRank algorithm for identifying important neurons/nodes in the neural network based on connectivity patterns.

## Prerequisites
- MP001-MP020 completed
- Understanding of PageRank algorithm
- Linear algebra basics

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/centrality/pagerank.rs`

2. Implement basic PageRank:
   ```rust
   pub struct PageRankConfig {
       pub damping_factor: f32,
       pub max_iterations: usize,
       pub tolerance: f32,
       pub personalization: Option<HashMap<NodeId, f32>>,
   }
   
   impl Default for PageRankConfig {
       fn default() -> Self {
           Self {
               damping_factor: 0.85,
               max_iterations: 100,
               tolerance: 1e-6,
               personalization: None,
           }
       }
   }
   
   pub fn pagerank<G: Graph>(
       graph: &G,
       config: PageRankConfig,
   ) -> HashMap<G::Node::Id, f32> {
       let n = graph.node_count() as f32;
       let mut scores: HashMap<G::Node::Id, f32> = HashMap::new();
       let mut new_scores = HashMap::new();
       
       // Initialize scores
       for node in graph.nodes() {
           scores.insert(node.id(), 1.0 / n);
       }
       
       for iteration in 0..config.max_iterations {
           // Calculate new scores
           for node in graph.nodes() {
               let node_id = node.id();
               let mut rank = (1.0 - config.damping_factor) / n;
               
               // Add personalization if provided
               if let Some(ref personalization) = config.personalization {
                   if let Some(&pref) = personalization.get(&node_id) {
                       rank = (1.0 - config.damping_factor) * pref;
                   }
               }
               
               // Sum contributions from incoming links
               for source in graph.incoming_nodes(&node_id) {
                   let out_degree = graph.get_node(&source)
                       .map(|n| n.neighbors().count())
                       .unwrap_or(1) as f32;
                   
                   rank += config.damping_factor * scores[&source] / out_degree;
               }
               
               new_scores.insert(node_id, rank);
           }
           
           // Check convergence
           let mut converged = true;
           for (node_id, &new_score) in &new_scores {
               if (new_score - scores[node_id]).abs() > config.tolerance {
                   converged = false;
                   break;
               }
           }
           
           scores = new_scores.clone();
           
           if converged {
               break;
           }
       }
       
       // Normalize scores
       let sum: f32 = scores.values().sum();
       for score in scores.values_mut() {
           *score /= sum;
       }
       
       scores
   }
   ```

3. Implement weighted PageRank:
   ```rust
   pub fn weighted_pagerank<G: Graph>(
       graph: &G,
       config: PageRankConfig,
   ) -> HashMap<G::Node::Id, f32> {
       let mut scores = HashMap::new();
       let n = graph.node_count() as f32;
       
       // Initialize
       for node in graph.nodes() {
           scores.insert(node.id(), 1.0 / n);
       }
       
       for _ in 0..config.max_iterations {
           let mut new_scores = HashMap::new();
           
           for node in graph.nodes() {
               let node_id = node.id();
               let mut rank = (1.0 - config.damping_factor) / n;
               
               // Weighted contribution from incoming links
               for source in graph.incoming_nodes(&node_id) {
                   let edge_weight = graph.edge_weight(&source, &node_id).unwrap_or(1.0);
                   let total_out_weight = graph.get_node(&source)
                       .map(|n| {
                           n.neighbors()
                               .map(|neighbor| graph.edge_weight(&source, &neighbor).unwrap_or(1.0))
                               .sum()
                       })
                       .unwrap_or(1.0);
                   
                   rank += config.damping_factor * scores[&source] * edge_weight / total_out_weight;
               }
               
               new_scores.insert(node_id, rank);
           }
           
           scores = new_scores;
       }
       
       scores
   }
   ```

4. Add topic-sensitive PageRank:
   ```rust
   pub fn topic_sensitive_pagerank<G: Graph>(
       graph: &G,
       topic_nodes: &[G::Node::Id],
       config: PageRankConfig,
   ) -> HashMap<G::Node::Id, f32> {
       let mut personalization = HashMap::new();
       let topic_weight = 1.0 / topic_nodes.len() as f32;
       
       for node_id in topic_nodes {
           personalization.insert(node_id.clone(), topic_weight);
       }
       
       pagerank(graph, PageRankConfig {
           personalization: Some(personalization),
           ..config
       })
   }
   ```

5. Neuromorphic-specific variants:
   - Activity-weighted PageRank
   - Spike-rate PageRank
   - Temporal PageRank

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/centrality/pagerank.rs
pub struct PageRankResult<Id> {
    pub scores: HashMap<Id, f32>,
    pub iterations: usize,
    pub converged: bool,
}

pub trait PageRankAlgorithm: Graph {
    fn pagerank(&self, config: PageRankConfig) -> PageRankResult<Self::Node::Id>;
    fn weighted_pagerank(&self, config: PageRankConfig) -> PageRankResult<Self::Node::Id>;
    fn personalized_pagerank(
        &self,
        personalization: HashMap<Self::Node::Id, f32>,
        config: PageRankConfig,
    ) -> PageRankResult<Self::Node::Id>;
}
```

## Verification Steps
1. Test on graphs with known PageRank values
2. Verify convergence behavior
3. Test personalized PageRank
4. Compare with reference implementations

## Time Estimate
25 minutes

## Dependencies
- MP001-MP020: Graph infrastructure
- MP004: For centrality metric integration