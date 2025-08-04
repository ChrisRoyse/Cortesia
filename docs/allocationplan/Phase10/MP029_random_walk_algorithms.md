# MP029: Random Walk Algorithms

## Task Description
Implement various random walk algorithms for neural network analysis, including standard random walks, biased walks, and specialized walks for exploring neuromorphic connectivity patterns and information propagation.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP021: PageRank (related random walk concepts)
- Understanding of probability theory and Markov chains

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/walks/random_walk.rs`

2. Implement basic random walk:
   ```rust
   use std::collections::HashMap;
   use rand::{Rng, thread_rng};
   use rand::seq::SliceRandom;
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct RandomWalk<Id> {
       pub path: Vec<Id>,
       pub step_count: usize,
       pub visit_counts: HashMap<Id, usize>,
       pub return_probability: f32,
   }

   pub fn simple_random_walk<G: Graph>(
       graph: &G,
       start_node: G::Node::Id,
       num_steps: usize,
   ) -> RandomWalk<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut rng = thread_rng();
       let mut path = Vec::with_capacity(num_steps + 1);
       let mut visit_counts = HashMap::new();
       let mut current_node = start_node.clone();
       
       path.push(current_node.clone());
       *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
       
       for _ in 0..num_steps {
           if let Some(node) = graph.get_node(&current_node) {
               let neighbors: Vec<_> = node.neighbors().collect();
               
               if neighbors.is_empty() {
                   break; // Dead end
               }
               
               // Choose random neighbor
               if let Some(&next_node) = neighbors.choose(&mut rng) {
                   current_node = next_node.clone();
                   path.push(current_node.clone());
                   *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
               }
           } else {
               break; // Invalid node
           }
       }
       
       // Calculate return probability
       let start_visits = visit_counts.get(&start_node).copied().unwrap_or(0);
       let return_probability = start_visits as f32 / path.len() as f32;
       
       RandomWalk {
           path,
           step_count: num_steps,
           visit_counts,
           return_probability,
       }
   }
   ```

3. Implement biased random walk (Node2Vec style):
   ```rust
   pub struct BiasedWalkParams {
       pub p: f32,  // Return parameter (probability of revisiting previous node)
       pub q: f32,  // In-out parameter (probability of exploring vs exploiting)
   }

   pub fn biased_random_walk<G: Graph>(
       graph: &G,
       start_node: G::Node::Id,
       num_steps: usize,
       params: BiasedWalkParams,
   ) -> RandomWalk<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut rng = thread_rng();
       let mut path = Vec::with_capacity(num_steps + 1);
       let mut visit_counts = HashMap::new();
       let mut current_node = start_node.clone();
       let mut previous_node: Option<G::Node::Id> = None;
       
       path.push(current_node.clone());
       *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
       
       for _ in 0..num_steps {
           if let Some(node) = graph.get_node(&current_node) {
               let neighbors: Vec<_> = node.neighbors().collect();
               
               if neighbors.is_empty() {
                   break;
               }
               
               // Calculate transition probabilities
               let mut transition_probs = Vec::new();
               let mut total_weight = 0.0;
               
               for neighbor in &neighbors {
                   let weight = if Some(neighbor) == previous_node.as_ref() {
                       // Return to previous node
                       1.0 / params.p
                   } else if let Some(prev) = &previous_node {
                       // Check if neighbor is connected to previous node
                       if let Some(prev_node) = graph.get_node(prev) {
                           if prev_node.neighbors().any(|n| n == *neighbor) {
                               1.0 // Stay local
                           } else {
                               1.0 / params.q // Explore farther
                           }
                       } else {
                           1.0
                       }
                   } else {
                       1.0 // First step
                   };
                   
                   transition_probs.push(weight);
                   total_weight += weight;
               }
               
               // Normalize probabilities
               for prob in &mut transition_probs {
                   *prob /= total_weight;
               }
               
               // Choose next node based on probabilities
               let rand_val: f32 = rng.gen();
               let mut cumulative_prob = 0.0;
               let mut selected_neighbor = neighbors[0].clone();
               
               for (i, &prob) in transition_probs.iter().enumerate() {
                   cumulative_prob += prob;
                   if rand_val <= cumulative_prob {
                       selected_neighbor = neighbors[i].clone();
                       break;
                   }
               }
               
               previous_node = Some(current_node.clone());
               current_node = selected_neighbor;
               path.push(current_node.clone());
               *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
           } else {
               break;
           }
       }
       
       let start_visits = visit_counts.get(&start_node).copied().unwrap_or(0);
       let return_probability = start_visits as f32 / path.len() as f32;
       
       RandomWalk {
           path,
           step_count: num_steps,
           visit_counts,
           return_probability,
       }
   }
   ```

4. Implement weighted random walk:
   ```rust
   pub fn weighted_random_walk<G: Graph>(
       graph: &G,
       start_node: G::Node::Id,
       num_steps: usize,
   ) -> RandomWalk<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut rng = thread_rng();
       let mut path = Vec::with_capacity(num_steps + 1);
       let mut visit_counts = HashMap::new();
       let mut current_node = start_node.clone();
       
       path.push(current_node.clone());
       *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
       
       for _ in 0..num_steps {
           if let Some(node) = graph.get_node(&current_node) {
               let neighbors: Vec<_> = node.neighbors().collect();
               
               if neighbors.is_empty() {
                   break;
               }
               
               // Calculate weights for each neighbor
               let mut weights = Vec::new();
               let mut total_weight = 0.0;
               
               for neighbor in &neighbors {
                   // Find edge weight
                   let weight = graph.edges()
                       .find(|e| e.source() == current_node && e.target() == *neighbor)
                       .map(|e| e.weight())
                       .unwrap_or(1.0);
                   
                   weights.push(weight);
                   total_weight += weight;
               }
               
               // Choose neighbor based on edge weights
               let rand_val: f32 = rng.gen_range(0.0..total_weight);
               let mut cumulative_weight = 0.0;
               let mut selected_neighbor = neighbors[0].clone();
               
               for (i, &weight) in weights.iter().enumerate() {
                   cumulative_weight += weight;
                   if rand_val <= cumulative_weight {
                       selected_neighbor = neighbors[i].clone();
                       break;
                   }
               }
               
               current_node = selected_neighbor;
               path.push(current_node.clone());
               *visit_counts.entry(current_node.clone()).or_insert(0) += 1;
           } else {
               break;
           }
       }
       
       let start_visits = visit_counts.get(&start_node).copied().unwrap_or(0);
       let return_probability = start_visits as f32 / path.len() as f32;
       
       RandomWalk {
           path,
           step_count: num_steps,
           visit_counts,
           return_probability,
       }
   }
   ```

5. Implement multiple random walks for sampling:
   ```rust
   pub struct WalkSampler<Id> {
       pub walks: Vec<RandomWalk<Id>>,
       pub aggregated_visits: HashMap<Id, usize>,
       pub node_frequencies: HashMap<Id, f32>,
       pub transition_matrix: HashMap<(Id, Id), f32>,
   }

   pub fn sample_multiple_walks<G: Graph>(
       graph: &G,
       num_walks: usize,
       walk_length: usize,
       start_distribution: Option<HashMap<G::Node::Id, f32>>,
   ) -> WalkSampler<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut rng = thread_rng();
       let mut walks = Vec::new();
       let mut aggregated_visits = HashMap::new();
       let mut transition_counts: HashMap<(G::Node::Id, G::Node::Id), usize> = HashMap::new();
       let mut node_transition_totals: HashMap<G::Node::Id, usize> = HashMap::new();
       
       // Determine starting nodes
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       
       for _ in 0..num_walks {
           let start_node = if let Some(ref dist) = start_distribution {
               // Sample from given distribution
               sample_from_distribution(dist, &mut rng)
           } else {
               // Uniform random start
               nodes.choose(&mut rng).unwrap().clone()
           };
           
           let walk = simple_random_walk(graph, start_node, walk_length);
           
           // Aggregate statistics
           for (node_id, &count) in &walk.visit_counts {
               *aggregated_visits.entry(node_id.clone()).or_insert(0) += count;
           }
           
           // Count transitions
           for window in walk.path.windows(2) {
               let from = window[0].clone();
               let to = window[1].clone();
               *transition_counts.entry((from.clone(), to)).or_insert(0) += 1;
               *node_transition_totals.entry(from).or_insert(0) += 1;
           }
           
           walks.push(walk);
       }
       
       // Calculate node frequencies
       let total_visits: usize = aggregated_visits.values().sum();
       let node_frequencies: HashMap<_, _> = aggregated_visits.iter()
           .map(|(node, &count)| (node.clone(), count as f32 / total_visits as f32))
           .collect();
       
       // Calculate transition probabilities
       let transition_matrix: HashMap<_, _> = transition_counts.iter()
           .map(|((from, to), &count)| {
               let total = node_transition_totals.get(from).copied().unwrap_or(1);
               ((from.clone(), to.clone()), count as f32 / total as f32)
           })
           .collect();
       
       WalkSampler {
           walks,
           aggregated_visits,
           node_frequencies,
           transition_matrix,
       }
   }

   fn sample_from_distribution<Id: Clone>(
       distribution: &HashMap<Id, f32>,
       rng: &mut impl Rng,
   ) -> Id {
       let total: f32 = distribution.values().sum();
       let rand_val: f32 = rng.gen_range(0.0..total);
       let mut cumulative = 0.0;
       
       for (node, &prob) in distribution {
           cumulative += prob;
           if rand_val <= cumulative {
               return node.clone();
           }
       }
       
       // Fallback to first key
       distribution.keys().next().unwrap().clone()
   }
   ```

6. Implement random walk analysis:
   ```rust
   pub struct WalkAnalysis<Id> {
       pub mixing_time: usize,
       pub stationary_distribution: HashMap<Id, f32>,
       pub hitting_times: HashMap<(Id, Id), f32>,
       pub cover_time: usize,
       pub commute_times: HashMap<(Id, Id), f32>,
   }

   pub fn analyze_random_walks<G: Graph>(
       graph: &G,
       walks: &[RandomWalk<G::Node::Id>],
   ) -> WalkAnalysis<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Estimate stationary distribution from long walks
       let mut total_visits = HashMap::new();
       let mut total_steps = 0;
       
       for walk in walks {
           for (node, &count) in &walk.visit_counts {
               *total_visits.entry(node.clone()).or_insert(0) += count;
               total_steps += count;
           }
       }
       
       let stationary_distribution: HashMap<_, _> = total_visits.iter()
           .map(|(node, &count)| (node.clone(), count as f32 / total_steps as f32))
           .collect();
       
       // Estimate mixing time (simplified)
       let mixing_time = estimate_mixing_time(walks);
       
       // Estimate hitting times
       let hitting_times = estimate_hitting_times(walks);
       
       // Estimate cover time (time to visit all nodes)
       let cover_time = estimate_cover_time(walks);
       
       // Estimate commute times
       let commute_times = estimate_commute_times(&hitting_times);
       
       WalkAnalysis {
           mixing_time,
           stationary_distribution,
           hitting_times,
           cover_time,
           commute_times,
       }
   }

   fn estimate_mixing_time<Id: Clone + Eq + std::hash::Hash>(
       walks: &[RandomWalk<Id>],
   ) -> usize {
       // Simplified: return average walk length where distribution stabilizes
       if walks.is_empty() {
           return 0;
       }
       
       walks.iter().map(|w| w.path.len()).sum::<usize>() / walks.len()
   }

   fn estimate_hitting_times<Id: Clone + Eq + std::hash::Hash>(
       walks: &[RandomWalk<Id>],
   ) -> HashMap<(Id, Id), f32> {
       let mut hitting_times = HashMap::new();
       
       for walk in walks {
           if walk.path.len() < 2 {
               continue;
           }
           
           let start = &walk.path[0];
           for (i, target) in walk.path.iter().enumerate().skip(1) {
               let key = (start.clone(), target.clone());
               if !hitting_times.contains_key(&key) {
                   hitting_times.insert(key, i as f32);
               }
           }
       }
       
       hitting_times
   }

   fn estimate_cover_time<Id: Clone + Eq + std::hash::Hash>(
       walks: &[RandomWalk<Id>],
   ) -> usize {
       if walks.is_empty() {
           return 0;
       }
       
       // Find maximum time to visit all unique nodes in any walk
       walks.iter()
           .map(|walk| {
               let mut seen = std::collections::HashSet::new();
               for (i, node) in walk.path.iter().enumerate() {
                   seen.insert(node);
                   if seen.len() == walk.visit_counts.len() {
                       return i;
                   }
               }
               walk.path.len()
           })
           .max()
           .unwrap_or(0)
   }

   fn estimate_commute_times<Id: Clone + Eq + std::hash::Hash>(
       hitting_times: &HashMap<(Id, Id), f32>,
   ) -> HashMap<(Id, Id), f32> {
       let mut commute_times = HashMap::new();
       
       for ((from, to), &time_forward) in hitting_times {
           if let Some(&time_backward) = hitting_times.get(&(to.clone(), from.clone())) {
               let commute_time = time_forward + time_backward;
               commute_times.insert((from.clone(), to.clone()), commute_time);
           }
       }
       
       commute_times
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/walks/random_walk.rs
pub trait RandomWalkAlgorithms: Graph {
    fn random_walk(&self, start: Self::Node::Id, steps: usize) -> RandomWalk<Self::Node::Id>;
    fn biased_walk(&self, start: Self::Node::Id, steps: usize, params: BiasedWalkParams) -> RandomWalk<Self::Node::Id>;
    fn weighted_walk(&self, start: Self::Node::Id, steps: usize) -> RandomWalk<Self::Node::Id>;
    fn sample_walks(&self, num_walks: usize, walk_length: usize) -> WalkSampler<Self::Node::Id>;
    fn analyze_walks(&self, walks: &[RandomWalk<Self::Node::Id>]) -> WalkAnalysis<Self::Node::Id>;
}

pub struct RandomWalkResult<Id> {
    pub walks: Vec<RandomWalk<Id>>,
    pub analysis: WalkAnalysis<Id>,
    pub sampler_stats: WalkSampler<Id>,
    pub computation_time: Duration,
}
```

## Verification Steps
1. Test random walks on simple graphs with known properties
2. Verify stationary distribution convergence on regular graphs
3. Compare biased vs unbiased walk exploration patterns
4. Test weighted walks on neuromorphic network topologies
5. Benchmark walk generation performance

## Time Estimate
30 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP015-MP016: Graph traversal algorithms
- MP021: PageRank (for comparison with stationary distributions)