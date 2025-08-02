pub mod cuda;

// Always re-export CudaGraphProcessor since both feature configurations define it
pub use cuda::CudaGraphProcessor;

use std::collections::{HashSet, VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// GPU acceleration interface for graph operations
pub trait GpuAccelerator {
    /// Perform parallel graph traversal on GPU
    fn parallel_traversal(&self, start_nodes: &[u32], max_depth: u32) -> Result<Vec<u32>, String>;
    
    /// Batch similarity computations on GPU
    fn batch_similarity(&self, embeddings: &[Vec<f32>], query: &[f32]) -> Result<Vec<f32>, String>;
    
    /// Parallel shortest path computation
    fn parallel_shortest_paths(&self, sources: &[u32], targets: &[u32]) -> Result<Vec<Option<Vec<u32>>>, String>;
}

/// CPU implementation with actual graph operations
pub struct CpuGraphProcessor {
    /// Adjacency list representation for graph traversal
    /// In a real implementation, this would be passed in or accessed from the graph
    pub graph: Option<Arc<HashMap<u32, Vec<u32>>>>,
}

impl Default for CpuGraphProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuGraphProcessor {
    /// Create a new CPU graph processor
    pub fn new() -> Self {
        CpuGraphProcessor { graph: None }
    }
    
    /// Set the graph for traversal operations
    pub fn with_graph(mut self, graph: Arc<HashMap<u32, Vec<u32>>>) -> Self {
        self.graph = Some(graph);
        self
    }
}

impl GpuAccelerator for CpuGraphProcessor {
    fn parallel_traversal(&self, start_nodes: &[u32], max_depth: u32) -> Result<Vec<u32>, String> {
        // If no graph is set, return just the start nodes
        let graph = match &self.graph {
            Some(g) => g,
            None => {
                // For testing/placeholder: create a simple graph
                let mut simple_graph: HashMap<u32, Vec<u32>> = HashMap::new();
                for &node in start_nodes {
                    simple_graph.insert(node, vec![]);
                }
                return Ok(start_nodes.to_vec());
            }
        };
        
        // Parallel BFS traversal using Rayon
        let visited_nodes = Arc::new(Mutex::new(HashSet::new()));
        
        // Process each start node in parallel
        start_nodes.par_iter().for_each(|&start_node| {
            let mut local_visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back((start_node, 0u32));
            local_visited.insert(start_node);
            
            while let Some((node, depth)) = queue.pop_front() {
                if depth >= max_depth {
                    continue;
                }
                
                if let Some(neighbors) = graph.get(&node) {
                    for &neighbor in neighbors {
                        if local_visited.insert(neighbor) {
                            queue.push_back((neighbor, depth + 1));
                        }
                    }
                }
            }
            
            // Merge local results into global visited set
            let mut global_visited = visited_nodes.lock().unwrap();
            global_visited.extend(local_visited);
        });
        
        // Convert to vector and return
        let result: Vec<u32> = visited_nodes.lock().unwrap().iter().cloned().collect();
        Ok(result)
    }
    
    fn batch_similarity(&self, embeddings: &[Vec<f32>], query: &[f32]) -> Result<Vec<f32>, String> {
        // Parallel cosine similarity computation using Rayon
        let similarities: Vec<f32> = embeddings
            .par_iter()
            .map(|embedding| cosine_similarity(embedding, query))
            .collect();
        
        Ok(similarities)
    }
    
    fn parallel_shortest_paths(&self, sources: &[u32], targets: &[u32]) -> Result<Vec<Option<Vec<u32>>>, String> {
        if sources.len() != targets.len() {
            return Err("Sources and targets must have the same length".to_string());
        }
        
        // If no graph is set, return None for all paths
        let graph = match &self.graph {
            Some(g) => g,
            None => return Ok(vec![None; sources.len()]),
        };
        
        // Compute shortest paths in parallel using Rayon
        let paths: Vec<Option<Vec<u32>>> = sources
            .par_iter()
            .zip(targets.par_iter())
            .map(|(&source, &target)| {
                dijkstra_shortest_path(graph, source, target)
            })
            .collect();
        
        Ok(paths)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Dijkstra's algorithm for finding shortest path between two nodes
fn dijkstra_shortest_path(graph: &HashMap<u32, Vec<u32>>, source: u32, target: u32) -> Option<Vec<u32>> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;
    
    // State for Dijkstra's algorithm
    #[derive(Clone, Eq, PartialEq)]
    struct State {
        cost: u32,
        node: u32,
    }
    
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse ordering for min-heap
            other.cost.cmp(&self.cost)
                .then_with(|| self.node.cmp(&other.node))
        }
    }
    
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    
    // Early exit if source or target don't exist in graph
    if !graph.contains_key(&source) || !graph.contains_key(&target) {
        return None;
    }
    
    // Distance tracking
    let mut distances: HashMap<u32, u32> = HashMap::new();
    let mut predecessors: HashMap<u32, u32> = HashMap::new();
    let mut heap = BinaryHeap::new();
    
    // Initialize source
    distances.insert(source, 0);
    heap.push(State { cost: 0, node: source });
    
    // Dijkstra's algorithm
    while let Some(State { cost, node }) = heap.pop() {
        // Found target
        if node == target {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = target;
            
            while current != source {
                path.push(current);
                match predecessors.get(&current) {
                    Some(&pred) => current = pred,
                    None => return None, // Path reconstruction failed
                }
            }
            path.push(source);
            path.reverse();
            
            return Some(path);
        }
        
        // Skip if we've already found a better path
        if cost > *distances.get(&node).unwrap_or(&u32::MAX) {
            continue;
        }
        
        // Explore neighbors
        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                let new_cost = cost + 1; // Assuming unit edge weights
                
                if new_cost < *distances.get(&neighbor).unwrap_or(&u32::MAX) {
                    distances.insert(neighbor, new_cost);
                    predecessors.insert(neighbor, node);
                    heap.push(State { cost: new_cost, node: neighbor });
                }
            }
        }
    }
    
    // No path found
    None
}