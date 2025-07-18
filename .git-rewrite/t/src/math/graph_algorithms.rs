use crate::error::Result;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;

/// Advanced graph algorithms for knowledge graph analysis
pub struct GraphAlgorithms {
    /// Cache for precomputed results
    algorithm_cache: HashMap<String, AlgorithmResult>,
}

impl GraphAlgorithms {
    pub fn new() -> Self {
        Self {
            algorithm_cache: HashMap::new(),
        }
    }

    /// Breadth-First Search (BFS) from a starting node
    pub fn bfs(&self, graph: &AdjacencyList, start: u32) -> Result<BfsResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut distances = HashMap::new();
        let mut parents = HashMap::new();
        let mut visit_order = Vec::new();

        queue.push_back(start);
        visited.insert(start);
        distances.insert(start, 0);
        parents.insert(start, None);

        while let Some(current) = queue.pop_front() {
            visit_order.push(current);

            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                        distances.insert(neighbor, distances[&current] + 1);
                        parents.insert(current, Some(current));
                    }
                }
            }
        }

        Ok(BfsResult {
            start_node: start,
            visited_nodes: visited,
            distances,
            parents,
            visit_order,
        })
    }

    /// Depth-First Search (DFS) from a starting node
    pub fn dfs(&self, graph: &AdjacencyList, start: u32) -> Result<DfsResult> {
        let mut visited = HashSet::new();
        let mut visit_order = Vec::new();
        let mut finish_order = Vec::new();
        let mut parents = HashMap::new();
        let mut discovery_time = HashMap::new();
        let mut finish_time = HashMap::new();
        let mut time = 0;

        self.dfs_visit(
            graph,
            start,
            &mut visited,
            &mut visit_order,
            &mut finish_order,
            &mut parents,
            &mut discovery_time,
            &mut finish_time,
            &mut time,
        );

        Ok(DfsResult {
            start_node: start,
            visited_nodes: visited,
            visit_order,
            finish_order,
            parents,
            discovery_time,
            finish_time,
        })
    }

    /// DFS recursive helper
    fn dfs_visit(
        &self,
        graph: &AdjacencyList,
        node: u32,
        visited: &mut HashSet<u32>,
        visit_order: &mut Vec<u32>,
        finish_order: &mut Vec<u32>,
        parents: &mut HashMap<u32, Option<u32>>,
        discovery_time: &mut HashMap<u32, u32>,
        finish_time: &mut HashMap<u32, u32>,
        time: &mut u32,
    ) {
        visited.insert(node);
        visit_order.push(node);
        *time += 1;
        discovery_time.insert(node, *time);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    parents.insert(neighbor, Some(node));
                    self.dfs_visit(
                        graph,
                        neighbor,
                        visited,
                        visit_order,
                        finish_order,
                        parents,
                        discovery_time,
                        finish_time,
                        time,
                    );
                }
            }
        }

        *time += 1;
        finish_time.insert(node, *time);
        finish_order.push(node);
    }

    /// Dijkstra's shortest path algorithm
    pub fn dijkstra(&self, graph: &WeightedAdjacencyList, start: u32) -> Result<DijkstraResult> {
        let mut distances = HashMap::new();
        let mut parents = HashMap::new();
        let mut visited = HashSet::new();
        let mut pq = BinaryHeap::new();

        // Initialize distances
        for &node in graph.keys() {
            distances.insert(node, f32::INFINITY);
        }
        distances.insert(start, 0.0);
        pq.push(Reverse((0.0 as u32, start))); // Using u32 for ordering

        while let Some(Reverse((_, current))) = pq.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Some(neighbors) = graph.get(&current) {
                for &(neighbor, weight) in neighbors {
                    let new_distance = distances[&current] + weight;
                    if new_distance < distances[&neighbor] {
                        distances.insert(neighbor, new_distance);
                        parents.insert(neighbor, Some(current));
                        pq.push(Reverse(((new_distance * 1000.0) as u32, neighbor)));
                    }
                }
            }
        }

        Ok(DijkstraResult {
            start_node: start,
            distances,
            parents,
        })
    }

    /// Reconstruct all shortest paths from Dijkstra's result
    fn reconstruct_all_paths(&self, start: u32, parents: &HashMap<u32, Option<u32>>) -> HashMap<u32, Vec<u32>> {
        let mut paths = HashMap::new();
        
        for (&target, _) in parents {
            if target != start {
                let path = self.reconstruct_path(start, target, parents);
                paths.insert(target, path);
            }
        }
        
        paths
    }

    /// Reconstruct path from start to target
    fn reconstruct_path(&self, start: u32, target: u32, parents: &HashMap<u32, Option<u32>>) -> Vec<u32> {
        let mut path = Vec::new();
        let mut current = target;
        
        while current != start {
            path.push(current);
            if let Some(Some(parent)) = parents.get(&current) {
                current = *parent;
            } else {
                break;
            }
        }
        
        path.push(start);
        path.reverse();
        path
    }

    /// A* pathfinding algorithm
    pub fn a_star(
        &self,
        graph: &WeightedAdjacencyList,
        start: u32,
        goal: u32,
        heuristic: &dyn Fn(u32, u32) -> f32,
    ) -> Result<AStarResult> {
        let mut open_set = BinaryHeap::new();
        let mut came_from = HashMap::new();
        let mut g_score = HashMap::new();
        let mut f_score = HashMap::new();

        for &node in graph.keys() {
            g_score.insert(node, f32::INFINITY);
            f_score.insert(node, f32::INFINITY);
        }

        g_score.insert(start, 0.0);
        f_score.insert(start, heuristic(start, goal));
        open_set.push(Reverse((f_score[&start] as u32, start)));

        while let Some(Reverse((_, current))) = open_set.pop() {
            if current == goal {
                let path = self.reconstruct_path_a_star(start, goal, &came_from);
                return Ok(AStarResult {
                    start_node: start,
                    goal_node: goal,
                    path: Some(path.clone()),
                    path_cost: g_score[&goal],
                    nodes_explored: came_from.len(),
                });
            }

            if let Some(neighbors) = graph.get(&current) {
                for &(neighbor, weight) in neighbors {
                    let tentative_g_score = g_score[&current] + weight;
                    
                    if tentative_g_score < g_score[&neighbor] {
                        came_from.insert(neighbor, current);
                        g_score.insert(neighbor, tentative_g_score);
                        let f = tentative_g_score + heuristic(neighbor, goal);
                        f_score.insert(neighbor, f);
                        open_set.push(Reverse((f as u32, neighbor)));
                    }
                }
            }
        }

        Ok(AStarResult {
            start_node: start,
            goal_node: goal,
            path: None,
            path_cost: f32::INFINITY,
            nodes_explored: came_from.len(),
        })
    }

    /// Reconstruct path for A* algorithm
    fn reconstruct_path_a_star(&self, start: u32, goal: u32, came_from: &HashMap<u32, u32>) -> Vec<u32> {
        let mut path = vec![goal];
        let mut current = goal;
        
        while current != start {
            if let Some(&parent) = came_from.get(&current) {
                path.push(parent);
                current = parent;
            } else {
                break;
            }
        }
        
        path.reverse();
        path
    }

    /// PageRank algorithm
    pub fn pagerank(&self, graph: &AdjacencyList, iterations: usize, damping: f32) -> Result<PageRankResult> {
        let nodes: Vec<u32> = graph.keys().cloned().collect();
        let n = nodes.len() as f32;
        let mut ranks = HashMap::new();
        let mut new_ranks = HashMap::new();

        // Initialize ranks
        for &node in &nodes {
            ranks.insert(node, 1.0 / n);
        }

        // Iterate
        for _ in 0..iterations {
            for &node in &nodes {
                let mut rank_sum = 0.0;
                
                // Find all nodes that link to this node
                for (&source, neighbors) in graph {
                    if neighbors.contains(&node) {
                        let out_degree = neighbors.len() as f32;
                        rank_sum += ranks[&source] / out_degree;
                    }
                }
                
                let new_rank = (1.0 - damping) / n + damping * rank_sum;
                new_ranks.insert(node, new_rank);
            }
            
            ranks = new_ranks.clone();
            new_ranks.clear();
        }

        // Sort by rank
        let mut ranked_nodes: Vec<(u32, f32)> = ranks.iter().map(|(&k, &v)| (k, v)).collect();
        ranked_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(PageRankResult {
            node_ranks: ranks,
            ranked_nodes,
            iterations_run: iterations,
            convergence_achieved: false, // Could implement convergence check
        })
    }

    /// Find strongly connected components using Tarjan's algorithm
    pub fn tarjan_scc(&self, graph: &AdjacencyList) -> Result<TarjanResult> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = HashMap::new();
        let mut lowlinks = HashMap::new();
        let mut on_stack = HashSet::new();
        let mut components = Vec::new();

        for &node in graph.keys() {
            if !indices.contains_key(&node) {
                self.tarjan_visit(
                    graph,
                    node,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut components,
                );
            }
        }

        let component_count = components.len();
        Ok(TarjanResult {
            strongly_connected_components: components,
            component_count,
        })
    }

    /// Tarjan's SCC helper function
    fn tarjan_visit(
        &self,
        graph: &AdjacencyList,
        node: u32,
        index: &mut usize,
        stack: &mut Vec<u32>,
        indices: &mut HashMap<u32, usize>,
        lowlinks: &mut HashMap<u32, usize>,
        on_stack: &mut HashSet<u32>,
        components: &mut Vec<Vec<u32>>,
    ) {
        indices.insert(node, *index);
        lowlinks.insert(node, *index);
        *index += 1;
        stack.push(node);
        on_stack.insert(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !indices.contains_key(&neighbor) {
                    self.tarjan_visit(
                        graph,
                        neighbor,
                        index,
                        stack,
                        indices,
                        lowlinks,
                        on_stack,
                        components,
                    );
                    lowlinks.insert(node, lowlinks[&node].min(lowlinks[&neighbor]));
                } else if on_stack.contains(&neighbor) {
                    lowlinks.insert(node, lowlinks[&node].min(indices[&neighbor]));
                }
            }
        }

        // If node is a root node, pop the stack and create an SCC
        if lowlinks[&node] == indices[&node] {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == node {
                    break;
                }
            }
            components.push(component);
        }
    }

    /// Betweenness centrality calculation
    pub fn betweenness_centrality(&self, graph: &AdjacencyList) -> Result<CentralityResult> {
        let nodes: Vec<u32> = graph.keys().cloned().collect();
        let mut centrality = HashMap::new();

        // Initialize centrality scores
        for &node in &nodes {
            centrality.insert(node, 0.0);
        }

        // For each node as source
        for &source in &nodes {
            let mut stack = Vec::new();
            let mut paths = HashMap::new();
            let mut dependencies = HashMap::new();
            let mut distances = HashMap::new();
            let mut predecessors: HashMap<u32, Vec<u32>> = HashMap::new();

            // Initialize
            for &node in &nodes {
                paths.insert(node, 0.0);
                dependencies.insert(node, 0.0);
                distances.insert(node, -1.0);
                predecessors.insert(node, Vec::new());
            }

            paths.insert(source, 1.0);
            distances.insert(source, 0.0);

            let mut queue = VecDeque::new();
            queue.push_back(source);

            // BFS
            while let Some(current) = queue.pop_front() {
                stack.push(current);
                
                if let Some(neighbors) = graph.get(&current) {
                    for &neighbor in neighbors {
                        // Neighbor found for the first time
                        if distances[&neighbor] < 0.0 {
                            queue.push_back(neighbor);
                            distances.insert(neighbor, distances[&current] + 1.0);
                        }
                        
                        // Shortest path to neighbor via current
                        if distances[&neighbor] == distances[&current] + 1.0 {
                            paths.insert(neighbor, paths[&neighbor] + paths[&current]);
                            predecessors.get_mut(&neighbor).unwrap().push(current);
                        }
                    }
                }
            }

            // Accumulation
            while let Some(w) = stack.pop() {
                for &predecessor in &predecessors[&w] {
                    let dependency_update = (paths[&predecessor] / paths[&w]) * (1.0 + dependencies[&w]);
                    dependencies.insert(predecessor, dependencies[&predecessor] + dependency_update);
                }
                if w != source {
                    centrality.insert(w, centrality[&w] + dependencies[&w]);
                }
            }
        }

        // Normalize (divide by 2 for undirected graphs)
        for &node in &nodes {
            centrality.insert(node, centrality[&node] / 2.0);
        }

        // Sort by centrality
        let mut ranked_nodes: Vec<(u32, f32)> = centrality.iter().map(|(&k, &v)| (k, v)).collect();
        ranked_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(CentralityResult {
            node_centrality: centrality,
            ranked_nodes,
            algorithm: "betweenness".to_string(),
        })
    }

    /// Closeness centrality calculation
    pub fn closeness_centrality(&self, graph: &AdjacencyList) -> Result<CentralityResult> {
        let nodes: Vec<u32> = graph.keys().cloned().collect();
        let mut centrality = HashMap::new();

        for &source in &nodes {
            let bfs_result = self.bfs(graph, source)?;
            let mut total_distance = 0.0;
            let mut reachable_nodes = 0;

            for &distance in bfs_result.distances.values() {
                if distance > 0 {
                    total_distance += distance as f32;
                    reachable_nodes += 1;
                }
            }

            let closeness = if total_distance > 0.0 {
                (reachable_nodes as f32) / total_distance
            } else {
                0.0
            };

            centrality.insert(source, closeness);
        }

        // Sort by centrality
        let mut ranked_nodes: Vec<(u32, f32)> = centrality.iter().map(|(&k, &v)| (k, v)).collect();
        ranked_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(CentralityResult {
            node_centrality: centrality,
            ranked_nodes,
            algorithm: "closeness".to_string(),
        })
    }

    /// Find all cycles in the graph using DFS
    pub fn find_cycles(&self, graph: &AdjacencyList) -> Result<CycleResult> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut cycles = Vec::new();
        let mut current_path = Vec::new();

        for &node in graph.keys() {
            if !visited.contains(&node) {
                self.find_cycles_dfs(
                    graph,
                    node,
                    &mut visited,
                    &mut rec_stack,
                    &mut cycles,
                    &mut current_path,
                );
            }
        }

        let cycle_count = cycles.len();
        let has_cycles = !cycles.is_empty();
        Ok(CycleResult {
            cycles,
            cycle_count,
            has_cycles,
        })
    }

    /// DFS helper for cycle detection
    fn find_cycles_dfs(
        &self,
        graph: &AdjacencyList,
        node: u32,
        visited: &mut HashSet<u32>,
        rec_stack: &mut HashSet<u32>,
        cycles: &mut Vec<Vec<u32>>,
        current_path: &mut Vec<u32>,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        current_path.push(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    self.find_cycles_dfs(graph, neighbor, visited, rec_stack, cycles, current_path);
                } else if rec_stack.contains(&neighbor) {
                    // Found a cycle
                    if let Some(cycle_start) = current_path.iter().position(|&x| x == neighbor) {
                        let cycle = current_path[cycle_start..].to_vec();
                        cycles.push(cycle);
                    }
                }
            }
        }

        current_path.pop();
        rec_stack.remove(&node);
    }

    /// Clear algorithm cache
    pub fn clear_cache(&mut self) {
        self.algorithm_cache.clear();
    }
}

// Type aliases for graph representations
pub type AdjacencyList = HashMap<u32, Vec<u32>>;
pub type WeightedAdjacencyList = HashMap<u32, Vec<(u32, f32)>>;

// Result structures for different algorithms

#[derive(Debug, Clone)]
pub struct BfsResult {
    pub start_node: u32,
    pub visited_nodes: HashSet<u32>,
    pub distances: HashMap<u32, usize>,
    pub parents: HashMap<u32, Option<u32>>,
    pub visit_order: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct DfsResult {
    pub start_node: u32,
    pub visited_nodes: HashSet<u32>,
    pub visit_order: Vec<u32>,
    pub finish_order: Vec<u32>,
    pub parents: HashMap<u32, Option<u32>>,
    pub discovery_time: HashMap<u32, u32>,
    pub finish_time: HashMap<u32, u32>,
}

#[derive(Debug, Clone)]
pub struct DijkstraResult {
    pub start_node: u32,
    pub distances: HashMap<u32, f32>,
    pub parents: HashMap<u32, Option<u32>>,
}

#[derive(Debug, Clone)]
pub struct AStarResult {
    pub start_node: u32,
    pub goal_node: u32,
    pub path: Option<Vec<u32>>,
    pub path_cost: f32,
    pub nodes_explored: usize,
}

#[derive(Debug, Clone)]
pub struct PageRankResult {
    pub node_ranks: HashMap<u32, f32>,
    pub ranked_nodes: Vec<(u32, f32)>,
    pub iterations_run: usize,
    pub convergence_achieved: bool,
}

#[derive(Debug, Clone)]
pub struct TarjanResult {
    pub strongly_connected_components: Vec<Vec<u32>>,
    pub component_count: usize,
}

#[derive(Debug, Clone)]
pub struct CentralityResult {
    pub node_centrality: HashMap<u32, f32>,
    pub ranked_nodes: Vec<(u32, f32)>,
    pub algorithm: String,
}

#[derive(Debug, Clone)]
pub struct CycleResult {
    pub cycles: Vec<Vec<u32>>,
    pub cycle_count: usize,
    pub has_cycles: bool,
}

#[derive(Debug, Clone)]
pub enum AlgorithmResult {
    Bfs(BfsResult),
    Dfs(DfsResult),
    Dijkstra(DijkstraResult),
    AStar(AStarResult),
    PageRank(PageRankResult),
    Tarjan(TarjanResult),
    Centrality(CentralityResult),
    Cycle(CycleResult),
}

