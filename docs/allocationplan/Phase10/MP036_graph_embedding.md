# MP036: Graph Embedding

## Task Description
Implement graph embedding algorithms to map neural network nodes and structures into low-dimensional vector spaces, enabling machine learning applications, similarity analysis, and neural representation learning.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP029: Random walk algorithms (for Node2Vec, DeepWalk)
- MP028: Spectral graph analysis (for spectral embeddings)
- Understanding of dimensionality reduction and neural networks

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/embedding/mod.rs`

2. Implement Node2Vec embedding:
   ```rust
   use std::collections::{HashMap, HashSet};
   use rand::{Rng, thread_rng};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};
   use crate::neuromorphic::graph::algorithms::walks::random_walk::{biased_random_walk, BiasedWalkParams};

   #[derive(Debug, Clone)]
   pub struct GraphEmbedding<Id> {
       pub embeddings: HashMap<Id, Vec<f32>>,
       pub dimensions: usize,
       pub method: EmbeddingMethod,
       pub training_loss: f32,
   }

   #[derive(Debug, Clone)]
   pub enum EmbeddingMethod {
       Node2Vec { p: f32, q: f32 },
       DeepWalk,
       LINE,
       SpectralEmbedding,
       LaplacianEigenmaps,
   }

   pub fn node2vec_embedding<G: Graph>(
       graph: &G,
       dimensions: usize,
       walk_length: usize,
       num_walks: usize,
       window_size: usize,
       p: f32,  // Return parameter
       q: f32,  // In-out parameter
       learning_rate: f32,
       epochs: usize,
   ) -> GraphEmbedding<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       
       // Generate random walks
       let mut walks = Vec::new();
       let walk_params = BiasedWalkParams { p, q };
       
       for _ in 0..num_walks {
           for node in &nodes {
               let walk = biased_random_walk(graph, node.clone(), walk_length, walk_params.clone());
               walks.push(walk.path);
           }
       }
       
       // Initialize embeddings randomly
       let mut embeddings = HashMap::new();
       let mut rng = thread_rng();
       
       for node in &nodes {
           let embedding: Vec<f32> = (0..dimensions)
               .map(|_| rng.gen_range(-0.5..0.5))
               .collect();
           embeddings.insert(node.clone(), embedding);
       }
       
       // Train Skip-gram model
       let mut total_loss = 0.0;
       let vocab_size = nodes.len();
       
       for epoch in 0..epochs {
           let mut epoch_loss = 0.0;
           
           for walk in &walks {
               for (i, center_node) in walk.iter().enumerate() {
                   // Define context window
                   let start = if i >= window_size { i - window_size } else { 0 };
                   let end = (i + window_size + 1).min(walk.len());
                   
                   for j in start..end {
                       if i != j {
                           let context_node = &walk[j];
                           
                           // Skip-gram: predict context from center
                           let loss = skipgram_update(
                               &mut embeddings,
                               center_node,
                               context_node,
                               &nodes,
                               learning_rate,
                           );
                           epoch_loss += loss;
                       }
                   }
               }
           }
           
           total_loss = epoch_loss;
           
           // Decay learning rate
           let current_lr = learning_rate * (1.0 - epoch as f32 / epochs as f32);
           if current_lr < learning_rate * 0.01 {
               break;
           }
       }
       
       GraphEmbedding {
           embeddings,
           dimensions,
           method: EmbeddingMethod::Node2Vec { p, q },
           training_loss: total_loss,
       }
   }

   fn skipgram_update<Id: Clone + Eq + std::hash::Hash>(
       embeddings: &mut HashMap<Id, Vec<f32>>,
       center: &Id,
       context: &Id,
       vocab: &[Id],
       learning_rate: f32,
   ) -> f32 {
       let dim = embeddings.get(center).map(|v| v.len()).unwrap_or(0);
       
       if dim == 0 {
           return 0.0;
       }
       
       // Simplified skip-gram update
       let center_emb = embeddings.get(center).unwrap().clone();
       let context_emb = embeddings.get(context).unwrap().clone();
       
       // Compute dot product
       let mut dot_product = 0.0;
       for i in 0..dim {
           dot_product += center_emb[i] * context_emb[i];
       }
       
       // Sigmoid activation
       let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
       let error = 1.0 - sigmoid; // Target is 1 for positive samples
       
       // Update embeddings
       let center_emb_mut = embeddings.get_mut(center).unwrap();
       let context_emb_clone = context_emb.clone();
       
       for i in 0..dim {
           center_emb_mut[i] += learning_rate * error * context_emb_clone[i];
       }
       
       let context_emb_mut = embeddings.get_mut(context).unwrap();
       for i in 0..dim {
           context_emb_mut[i] += learning_rate * error * center_emb[i];
       }
       
       -sigmoid.ln() // Return loss
   }
   ```

3. Implement DeepWalk embedding:
   ```rust
   use crate::neuromorphic::graph::algorithms::walks::random_walk::simple_random_walk;

   pub fn deepwalk_embedding<G: Graph>(
       graph: &G,
       dimensions: usize,
       walk_length: usize,
       num_walks: usize,
       window_size: usize,
       learning_rate: f32,
       epochs: usize,
   ) -> GraphEmbedding<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       
       // Generate random walks
       let mut walks = Vec::new();
       
       for _ in 0..num_walks {
           for node in &nodes {
               let walk = simple_random_walk(graph, node.clone(), walk_length);
               walks.push(walk.path);
           }
       }
       
       // Initialize embeddings
       let mut embeddings = HashMap::new();
       let mut rng = thread_rng();
       
       for node in &nodes {
           let embedding: Vec<f32> = (0..dimensions)
               .map(|_| rng.gen_range(-0.1..0.1))
               .collect();
           embeddings.insert(node.clone(), embedding);
       }
       
       // Train word2vec-style model
       let mut total_loss = 0.0;
       
       for epoch in 0..epochs {
           let mut epoch_loss = 0.0;
           
           for walk in &walks {
               for (i, center_node) in walk.iter().enumerate() {
                   let start = if i >= window_size { i - window_size } else { 0 };
                   let end = (i + window_size + 1).min(walk.len());
                   
                   for j in start..end {
                       if i != j {
                           let context_node = &walk[j];
                           let loss = skipgram_update(
                               &mut embeddings,
                               center_node,
                               context_node,
                               &nodes,
                               learning_rate,
                           );
                           epoch_loss += loss;
                       }
                   }
               }
           }
           
           total_loss = epoch_loss;
       }
       
       GraphEmbedding {
           embeddings,
           dimensions,
           method: EmbeddingMethod::DeepWalk,
           training_loss: total_loss,
       }
   }
   ```

4. Implement spectral embedding:
   ```rust
   use crate::neuromorphic::graph::algorithms::spectral::analysis::{build_graph_matrices, compute_eigendecomposition};

   pub fn spectral_embedding<G: Graph>(
       graph: &G,
       dimensions: usize,
   ) -> GraphEmbedding<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let matrices = build_graph_matrices(graph);
       
       // Compute eigendecomposition of normalized Laplacian
       let eigen_decomp = compute_eigendecomposition(
           &matrices.normalized_laplacian,
           dimensions + 1, // Skip the first eigenvalue (should be 0)
           1000,
           1e-6,
       );
       
       let mut embeddings = HashMap::new();
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       
       // Use eigenvectors as embeddings (skip first eigenvector)
       for (node_id, &matrix_index) in &matrices.node_indices {
           let mut embedding = Vec::with_capacity(dimensions);
           
           for i in 1..=dimensions.min(eigen_decomp.eigenvectors.len() - 1) {
               if matrix_index < eigen_decomp.eigenvectors[i].len() {
                   embedding.push(eigen_decomp.eigenvectors[i][matrix_index]);
               } else {
                   embedding.push(0.0);
               }
           }
           
           // Find corresponding original node ID
           for original_node in &nodes {
               if original_node.to_string() == *node_id {
                   embeddings.insert(original_node.clone(), embedding);
                   break;
               }
           }
       }
       
       GraphEmbedding {
           embeddings,
           dimensions,
           method: EmbeddingMethod::SpectralEmbedding,
           training_loss: 0.0, // No training loss for spectral methods
       }
   }
   ```

5. Implement LINE (Large-scale Information Network Embedding):
   ```rust
   pub fn line_embedding<G: Graph>(
       graph: &G,
       dimensions: usize,
       learning_rate: f32,
       epochs: usize,
       negative_samples: usize,
   ) -> GraphEmbedding<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let edges: Vec<_> = graph.edges().collect();
       
       // Initialize embeddings
       let mut embeddings = HashMap::new();
       let mut context_embeddings = HashMap::new();
       let mut rng = thread_rng();
       
       for node in &nodes {
           let embedding: Vec<f32> = (0..dimensions)
               .map(|_| rng.gen_range(-0.1..0.1))
               .collect();
           let context_embedding: Vec<f32> = (0..dimensions)
               .map(|_| rng.gen_range(-0.1..0.1))
               .collect();
           
           embeddings.insert(node.clone(), embedding);
           context_embeddings.insert(node.clone(), context_embedding);
       }
       
       let mut total_loss = 0.0;
       
       for _ in 0..epochs {
           let mut epoch_loss = 0.0;
           
           // First-order proximity
           for edge in &edges {
               let source = edge.source();
               let target = edge.target();
               
               let loss = line_update_first_order(
                   &mut embeddings,
                   &mut context_embeddings,
                   &source,
                   &target,
                   &nodes,
                   negative_samples,
                   learning_rate,
               );
               epoch_loss += loss;
           }
           
           total_loss = epoch_loss;
       }
       
       GraphEmbedding {
           embeddings,
           dimensions,
           method: EmbeddingMethod::LINE,
           training_loss: total_loss,
       }
   }

   fn line_update_first_order<Id: Clone + Eq + std::hash::Hash>(
       embeddings: &mut HashMap<Id, Vec<f32>>,
       context_embeddings: &mut HashMap<Id, Vec<f32>>,
       source: &Id,
       target: &Id,
       vocab: &[Id],
       negative_samples: usize,
       learning_rate: f32,
   ) -> f32 {
       let dim = embeddings.get(source).map(|v| v.len()).unwrap_or(0);
       if dim == 0 { return 0.0; }
       
       let mut total_loss = 0.0;
       
       // Positive sample
       let source_emb = embeddings.get(source).unwrap().clone();
       let target_ctx = context_embeddings.get(target).unwrap().clone();
       
       let mut dot_product = 0.0;
       for i in 0..dim {
           dot_product += source_emb[i] * target_ctx[i];
       }
       
       let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
       let error = 1.0 - sigmoid;
       total_loss -= sigmoid.ln();
       
       // Update embeddings
       let source_emb_mut = embeddings.get_mut(source).unwrap();
       for i in 0..dim {
           source_emb_mut[i] += learning_rate * error * target_ctx[i];
       }
       
       let target_ctx_mut = context_embeddings.get_mut(target).unwrap();
       for i in 0..dim {
           target_ctx_mut[i] += learning_rate * error * source_emb[i];
       }
       
       // Negative samples
       let mut rng = thread_rng();
       for _ in 0..negative_samples {
           let negative_node = &vocab[rng.gen_range(0..vocab.len())];
           if negative_node == target { continue; }
           
           let negative_ctx = context_embeddings.get(negative_node).unwrap().clone();
           
           let mut neg_dot_product = 0.0;
           for i in 0..dim {
               neg_dot_product += source_emb[i] * negative_ctx[i];
           }
           
           let neg_sigmoid = 1.0 / (1.0 + (-(-neg_dot_product)).exp());
           let neg_error = -neg_sigmoid;
           total_loss -= (1.0 - neg_sigmoid).ln();
           
           let source_emb_mut = embeddings.get_mut(source).unwrap();
           for i in 0..dim {
               source_emb_mut[i] += learning_rate * neg_error * negative_ctx[i];
           }
           
           let negative_ctx_mut = context_embeddings.get_mut(negative_node).unwrap();
           for i in 0..dim {
               negative_ctx_mut[i] += learning_rate * neg_error * source_emb[i];
           }
       }
       
       total_loss
   }
   ```

6. Implement embedding evaluation and utilities:
   ```rust
   pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
       if vec1.len() != vec2.len() {
           return 0.0;
       }
       
       let mut dot_product = 0.0;
       let mut norm1 = 0.0;
       let mut norm2 = 0.0;
       
       for i in 0..vec1.len() {
           dot_product += vec1[i] * vec2[i];
           norm1 += vec1[i] * vec1[i];
           norm2 += vec2[i] * vec2[i];
       }
       
       if norm1 == 0.0 || norm2 == 0.0 {
           0.0
       } else {
           dot_product / (norm1.sqrt() * norm2.sqrt())
       }
   }

   pub fn find_nearest_neighbors<Id: Clone + Eq + std::hash::Hash>(
       embedding: &GraphEmbedding<Id>,
       node: &Id,
       k: usize,
   ) -> Vec<(Id, f32)> {
       if let Some(node_embedding) = embedding.embeddings.get(node) {
           let mut similarities = Vec::new();
           
           for (other_node, other_embedding) in &embedding.embeddings {
               if other_node != node {
                   let similarity = cosine_similarity(node_embedding, other_embedding);
                   similarities.push((other_node.clone(), similarity));
               }
           }
           
           similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
           similarities.truncate(k);
           similarities
       } else {
           Vec::new()
       }
   }

   pub fn evaluate_embedding_quality<G: Graph>(
       graph: &G,
       embedding: &GraphEmbedding<G::Node::Id>,
       test_edges: &[(G::Node::Id, G::Node::Id)],
   ) -> EmbeddingQualityMetrics 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut link_prediction_scores = Vec::new();
       
       for (node1, node2) in test_edges {
           if let (Some(emb1), Some(emb2)) = (embedding.embeddings.get(node1), embedding.embeddings.get(node2)) {
               let similarity = cosine_similarity(emb1, emb2);
               link_prediction_scores.push(similarity);
           }
       }
       
       let avg_similarity = if link_prediction_scores.is_empty() {
           0.0
       } else {
           link_prediction_scores.iter().sum::<f32>() / link_prediction_scores.len() as f32
       };
       
       EmbeddingQualityMetrics {
           avg_link_prediction_score: avg_similarity,
           embedding_variance: calculate_embedding_variance(embedding),
           dimension_utilization: calculate_dimension_utilization(embedding),
       }
   }

   fn calculate_embedding_variance<Id>(embedding: &GraphEmbedding<Id>) -> f32 {
       let mut total_variance = 0.0;
       let mut count = 0;
       
       for embedding_vec in embedding.embeddings.values() {
           let mean: f32 = embedding_vec.iter().sum::<f32>() / embedding_vec.len() as f32;
           let variance: f32 = embedding_vec.iter()
               .map(|x| (x - mean).powi(2))
               .sum::<f32>() / embedding_vec.len() as f32;
           total_variance += variance;
           count += 1;
       }
       
       if count > 0 { total_variance / count as f32 } else { 0.0 }
   }

   fn calculate_dimension_utilization<Id>(embedding: &GraphEmbedding<Id>) -> f32 {
       if embedding.dimensions == 0 { return 0.0; }
       
       let mut dimension_variances = vec![0.0; embedding.dimensions];
       let node_count = embedding.embeddings.len();
       
       if node_count == 0 { return 0.0; }
       
       // Calculate variance for each dimension
       for dim in 0..embedding.dimensions {
           let values: Vec<f32> = embedding.embeddings.values()
               .filter_map(|emb| emb.get(dim).copied())
               .collect();
           
           if !values.is_empty() {
               let mean = values.iter().sum::<f32>() / values.len() as f32;
               let variance = values.iter()
                   .map(|x| (x - mean).powi(2))
                   .sum::<f32>() / values.len() as f32;
               dimension_variances[dim] = variance;
           }
       }
       
       // Return average variance across dimensions
       dimension_variances.iter().sum::<f32>() / embedding.dimensions as f32
   }

   #[derive(Debug, Clone)]
   pub struct EmbeddingQualityMetrics {
       pub avg_link_prediction_score: f32,
       pub embedding_variance: f32,
       pub dimension_utilization: f32,
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/embedding/mod.rs
pub trait GraphEmbedding: Graph {
    fn node2vec(&self, dimensions: usize, walk_length: usize, num_walks: usize, 
               window_size: usize, p: f32, q: f32, lr: f32, epochs: usize) -> GraphEmbedding<Self::Node::Id>;
    fn deepwalk(&self, dimensions: usize, walk_length: usize, num_walks: usize, 
               window_size: usize, lr: f32, epochs: usize) -> GraphEmbedding<Self::Node::Id>;
    fn spectral_embedding(&self, dimensions: usize) -> GraphEmbedding<Self::Node::Id>;
    fn line_embedding(&self, dimensions: usize, lr: f32, epochs: usize, neg_samples: usize) -> GraphEmbedding<Self::Node::Id>;
    fn evaluate_embedding(&self, embedding: &GraphEmbedding<Self::Node::Id>, test_edges: &[(Self::Node::Id, Self::Node::Id)]) -> EmbeddingQualityMetrics;
}

pub struct EmbeddingResult<Id> {
    pub embedding: GraphEmbedding<Id>,
    pub quality_metrics: EmbeddingQualityMetrics,
    pub nearest_neighbors: HashMap<Id, Vec<(Id, f32)>>,
    pub training_time: Duration,
}
```

## Verification Steps
1. Test embeddings on graphs with known structural properties
2. Evaluate link prediction performance using embeddings
3. Compare different embedding methods on neuromorphic networks
4. Verify embedding quality metrics and nearest neighbor accuracy
5. Benchmark training time and memory usage

## Time Estimate
35 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP029: Random walk algorithms (for Node2Vec, DeepWalk)
- MP028: Spectral graph analysis (for spectral embeddings)