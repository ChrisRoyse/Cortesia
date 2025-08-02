# MP035: Link Prediction

## Task Description
Implement link prediction algorithms to forecast potential connections in neural networks, enabling prediction of synaptic formation, neural pathway development, and network evolution patterns.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP021-MP024: Centrality measures
- MP025: Clustering coefficient
- MP029: Random walk algorithms
- Understanding of similarity measures and machine learning basics

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/prediction/link_prediction.rs`

2. Implement similarity-based link prediction:
   ```rust
   use std::collections::{HashMap, HashSet};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct LinkPrediction<Id> {
       pub candidate_links: Vec<(Id, Id)>,
       pub scores: HashMap<(Id, Id), f32>,
       pub method: PredictionMethod,
       pub threshold: f32,
   }

   #[derive(Debug, Clone)]
   pub enum PredictionMethod {
       CommonNeighbors,
       JaccardCoefficient,
       AdamicAdar,
       PreferentialAttachment,
       ResourceAllocation,
       Katz,
       PersonalizedPageRank,
   }

   pub fn common_neighbors_score<G: Graph>(
       graph: &G,
       node_a: &G::Node::Id,
       node_b: &G::Node::Id,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if let (Some(node_a_ref), Some(node_b_ref)) = (graph.get_node(node_a), graph.get_node(node_b)) {
           let neighbors_a: HashSet<_> = node_a_ref.neighbors().collect();
           let neighbors_b: HashSet<_> = node_b_ref.neighbors().collect();
           
           neighbors_a.intersection(&neighbors_b).count() as f32
       } else {
           0.0
       }
   }

   pub fn jaccard_coefficient<G: Graph>(
       graph: &G,
       node_a: &G::Node::Id,
       node_b: &G::Node::Id,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if let (Some(node_a_ref), Some(node_b_ref)) = (graph.get_node(node_a), graph.get_node(node_b)) {
           let neighbors_a: HashSet<_> = node_a_ref.neighbors().collect();
           let neighbors_b: HashSet<_> = node_b_ref.neighbors().collect();
           
           let intersection_size = neighbors_a.intersection(&neighbors_b).count();
           let union_size = neighbors_a.union(&neighbors_b).count();
           
           if union_size > 0 {
               intersection_size as f32 / union_size as f32
           } else {
               0.0
           }
       } else {
           0.0
       }
   }

   pub fn adamic_adar_score<G: Graph>(
       graph: &G,
       node_a: &G::Node::Id,
       node_b: &G::Node::Id,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if let (Some(node_a_ref), Some(node_b_ref)) = (graph.get_node(node_a), graph.get_node(node_b)) {
           let neighbors_a: HashSet<_> = node_a_ref.neighbors().collect();
           let neighbors_b: HashSet<_> = node_b_ref.neighbors().collect();
           
           let mut score = 0.0;
           for common_neighbor in neighbors_a.intersection(&neighbors_b) {
               if let Some(common_node) = graph.get_node(common_neighbor) {
                   let degree = common_node.neighbors().count() as f32;
                   if degree > 0.0 {
                       score += 1.0 / degree.ln();
                   }
               }
           }
           score
       } else {
           0.0
       }
   }

   pub fn resource_allocation_score<G: Graph>(
       graph: &G,
       node_a: &G::Node::Id,
       node_b: &G::Node::Id,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if let (Some(node_a_ref), Some(node_b_ref)) = (graph.get_node(node_a), graph.get_node(node_b)) {
           let neighbors_a: HashSet<_> = node_a_ref.neighbors().collect();
           let neighbors_b: HashSet<_> = node_b_ref.neighbors().collect();
           
           let mut score = 0.0;
           for common_neighbor in neighbors_a.intersection(&neighbors_b) {
               if let Some(common_node) = graph.get_node(common_neighbor) {
                   let degree = common_node.neighbors().count() as f32;
                   if degree > 0.0 {
                       score += 1.0 / degree;
                   }
               }
           }
           score
       } else {
           0.0
       }
   }

   pub fn preferential_attachment_score<G: Graph>(
       graph: &G,
       node_a: &G::Node::Id,
       node_b: &G::Node::Id,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       if let (Some(node_a_ref), Some(node_b_ref)) = (graph.get_node(node_a), graph.get_node(node_b)) {
           let degree_a = node_a_ref.neighbors().count() as f32;
           let degree_b = node_b_ref.neighbors().count() as f32;
           degree_a * degree_b
       } else {
           0.0
       }
   }
   ```

3. Implement Katz similarity:
   ```rust
   pub fn katz_similarity<G: Graph>(
       graph: &G,
       beta: f32,
       max_path_length: usize,
   ) -> HashMap<(G::Node::Id, G::Node::Id), f32> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       let mut node_indices = HashMap::new();
       
       for (i, node) in nodes.iter().enumerate() {
           node_indices.insert(node.clone(), i);
       }
       
       // Build adjacency matrix
       let mut adj_matrix = vec![vec![0.0f32; n]; n];
       for edge in graph.edges() {
           if let (Some(&i), Some(&j)) = (node_indices.get(&edge.source()), node_indices.get(&edge.target())) {
               adj_matrix[i][j] = 1.0;
               if !edge.is_directed() {
                   adj_matrix[j][i] = 1.0;
               }
           }
       }
       
       // Compute powers of adjacency matrix
       let mut katz_matrix = vec![vec![0.0f32; n]; n];
       let mut current_power = vec![vec![0.0f32; n]; n];
       
       // Initialize with identity matrix
       for i in 0..n {
           current_power[i][i] = 1.0;
       }
       
       let mut beta_power = 1.0;
       
       for path_length in 1..=max_path_length {
           // Multiply current_power by adjacency matrix
           let mut next_power = vec![vec![0.0f32; n]; n];
           for i in 0..n {
               for j in 0..n {
                   for k in 0..n {
                       next_power[i][j] += current_power[i][k] * adj_matrix[k][j];
                   }
               }
           }
           
           beta_power *= beta;
           
           // Add beta^path_length * A^path_length to Katz matrix
           for i in 0..n {
               for j in 0..n {
                   katz_matrix[i][j] += beta_power * next_power[i][j];
               }
           }
           
           current_power = next_power;
           
           if beta_power < 1e-10 {
               break; // Convergence
           }
       }
       
       // Convert back to node pairs
       let mut similarities = HashMap::new();
       for i in 0..n {
           for j in 0..n {
               if i != j {
                   similarities.insert((nodes[i].clone(), nodes[j].clone()), katz_matrix[i][j]);
               }
           }
       }
       
       similarities
   }
   ```

4. Implement personalized PageRank for link prediction:
   ```rust
   use crate::neuromorphic::graph::algorithms::walks::random_walk::simple_random_walk;

   pub fn personalized_pagerank_similarity<G: Graph>(
       graph: &G,
       source_node: &G::Node::Id,
       alpha: f32,
       max_iterations: usize,
       tolerance: f32,
   ) -> HashMap<G::Node::Id, f32> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       if n == 0 {
           return HashMap::new();
       }
       
       let mut pagerank: HashMap<G::Node::Id, f32> = HashMap::new();
       let initial_value = 1.0 / n as f32;
       
       for node in &nodes {
           pagerank.insert(node.clone(), initial_value);
       }
       
       for _ in 0..max_iterations {
           let mut new_pagerank = HashMap::new();
           
           // Initialize with teleport probability to source
           for node in &nodes {
               if node == source_node {
                   new_pagerank.insert(node.clone(), alpha);
               } else {
                   new_pagerank.insert(node.clone(), 0.0);
               }
           }
           
           // Add contributions from incoming links
           for node in graph.nodes() {
               let node_id = node.id();
               let current_pr = pagerank.get(&node_id).copied().unwrap_or(0.0);
               let out_degree = node.neighbors().count();
               
               if out_degree > 0 {
                   let contribution = (1.0 - alpha) * current_pr / out_degree as f32;
                   
                   for neighbor in node.neighbors() {
                       *new_pagerank.get_mut(&neighbor).unwrap() += contribution;
                   }
               }
           }
           
           // Check convergence
           let mut converged = true;
           for (node, &new_value) in &new_pagerank {
               let old_value = pagerank.get(node).copied().unwrap_or(0.0);
               if (new_value - old_value).abs() > tolerance {
                   converged = false;
                   break;
               }
           }
           
           pagerank = new_pagerank;
           
           if converged {
               break;
           }
       }
       
       pagerank
   }
   ```

5. Implement comprehensive link prediction framework:
   ```rust
   pub fn predict_links<G: Graph>(
       graph: &G,
       method: PredictionMethod,
       top_k: usize,
   ) -> LinkPrediction<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let mut scores = HashMap::new();
       let mut candidate_links = Vec::new();
       
       // Generate all possible node pairs (excluding existing edges)
       let existing_edges: HashSet<_> = graph.edges()
           .map(|e| (e.source(), e.target()))
           .collect();
       
       for i in 0..nodes.len() {
           for j in (i + 1)..nodes.len() {
               let node_a = &nodes[i];
               let node_b = &nodes[j];
               
               // Skip if edge already exists
               if existing_edges.contains(&(node_a.clone(), node_b.clone())) ||
                  existing_edges.contains(&(node_b.clone(), node_a.clone())) {
                   continue;
               }
               
               let score = match method {
                   PredictionMethod::CommonNeighbors => common_neighbors_score(graph, node_a, node_b),
                   PredictionMethod::JaccardCoefficient => jaccard_coefficient(graph, node_a, node_b),
                   PredictionMethod::AdamicAdar => adamic_adar_score(graph, node_a, node_b),
                   PredictionMethod::PreferentialAttachment => preferential_attachment_score(graph, node_a, node_b),
                   PredictionMethod::ResourceAllocation => resource_allocation_score(graph, node_a, node_b),
                   _ => 0.0, // Implement other methods as needed
               };
               
               if score > 0.0 {
                   scores.insert((node_a.clone(), node_b.clone()), score);
                   candidate_links.push((node_a.clone(), node_b.clone()));
               }
           }
       }
       
       // Sort by score and take top k
       candidate_links.sort_by(|a, b| {
           let score_a = scores.get(a).copied().unwrap_or(0.0);
           let score_b = scores.get(b).copied().unwrap_or(0.0);
           score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
       });
       
       candidate_links.truncate(top_k);
       
       let threshold = if !candidate_links.is_empty() {
           scores.get(&candidate_links[candidate_links.len().min(top_k) - 1])
               .copied().unwrap_or(0.0)
       } else {
           0.0
       };
       
       LinkPrediction {
           candidate_links,
           scores,
           method,
           threshold,
       }
   }

   pub fn evaluate_predictions<G: Graph>(
       predictions: &LinkPrediction<G::Node::Id>,
       true_future_edges: &[(G::Node::Id, G::Node::Id)],
   ) -> PredictionMetrics 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let predicted_set: HashSet<_> = predictions.candidate_links.iter().collect();
       let true_set: HashSet<_> = true_future_edges.iter().collect();
       
       let true_positives = predicted_set.intersection(&true_set).count();
       let false_positives = predicted_set.len() - true_positives;
       let false_negatives = true_set.len() - true_positives;
       
       let precision = if predicted_set.is_empty() {
           0.0
       } else {
           true_positives as f32 / predicted_set.len() as f32
       };
       
       let recall = if true_set.is_empty() {
           0.0
       } else {
           true_positives as f32 / true_set.len() as f32
       };
       
       let f1_score = if precision + recall > 0.0 {
           2.0 * precision * recall / (precision + recall)
       } else {
           0.0
       };
       
       PredictionMetrics {
           precision,
           recall,
           f1_score,
           true_positives,
           false_positives,
           false_negatives,
       }
   }

   #[derive(Debug, Clone)]
   pub struct PredictionMetrics {
       pub precision: f32,
       pub recall: f32,
       pub f1_score: f32,
       pub true_positives: usize,
       pub false_positives: usize,
       pub false_negatives: usize,
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/prediction/link_prediction.rs
pub trait LinkPrediction: Graph {
    fn predict_links(&self, method: PredictionMethod, top_k: usize) -> LinkPrediction<Self::Node::Id>;
    fn common_neighbors_score(&self, node_a: &Self::Node::Id, node_b: &Self::Node::Id) -> f32;
    fn jaccard_similarity(&self, node_a: &Self::Node::Id, node_b: &Self::Node::Id) -> f32;
    fn adamic_adar_score(&self, node_a: &Self::Node::Id, node_b: &Self::Node::Id) -> f32;
    fn katz_similarity(&self, beta: f32, max_length: usize) -> HashMap<(Self::Node::Id, Self::Node::Id), f32>;
    fn personalized_pagerank(&self, source: &Self::Node::Id, alpha: f32) -> HashMap<Self::Node::Id, f32>;
}

pub struct LinkPredictionResult<Id> {
    pub predictions: LinkPrediction<Id>,
    pub evaluation_metrics: PredictionMetrics,
    pub top_candidates: Vec<(Id, Id, f32)>,
    pub method_comparison: HashMap<PredictionMethod, f32>,
}
```

## Verification Steps
1. Test similarity measures on graphs with known link formation patterns
2. Compare different prediction methods on temporal networks
3. Evaluate prediction accuracy using cross-validation
4. Test on neuromorphic network evolution scenarios
5. Benchmark computational performance on large networks

## Time Estimate
30 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP021-MP024: Centrality measures for node importance
- MP025: Clustering coefficient for structural similarity
- MP029: Random walk algorithms for PageRank variants