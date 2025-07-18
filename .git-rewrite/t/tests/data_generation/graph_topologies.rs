//! Graph Topology Generation
//! 
//! Provides deterministic generation of various graph topologies with known mathematical properties.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Properties of generated graphs with mathematical guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphProperties {
    pub entity_count: u64,
    pub edge_count: u64,
    pub average_degree: f64,
    pub clustering_coefficient: f64,
    pub diameter: u32,
    pub density: f64,
    pub connectivity: ConnectivityType,
    pub expected_path_length: f64,
}

/// Different types of graph connectivity patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityType {
    Connected,
    Forest,
    Complete,
    Random,
    SmallWorld,
    ScaleFree,
    Bipartite,
}

/// Lightweight graph representation for testing
#[derive(Debug, Clone)]
pub struct TestGraph {
    pub entities: Vec<TestEntity>,
    pub edges: Vec<TestEdge>,
    pub properties: GraphProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEntity {
    pub id: u32,
    pub name: String,
    pub entity_type: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEdge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    pub edge_type: String,
}

/// Graph topology generator with deterministic outcomes
pub struct GraphTopologyGenerator {
    rng: DeterministicRng,
    properties: GraphProperties,
}

impl GraphTopologyGenerator {
    /// Create a new generator with deterministic seed
    pub fn new(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("graph_topology_generator".to_string());
        
        Self {
            rng,
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::Connected,
                expected_path_length: 0.0,
            },
        }
    }

    /// Generate Erdős–Rényi random graph with predictable properties
    /// Mathematical guarantee: For p > ln(n)/n, graph is connected with high probability
    pub fn generate_erdos_renyi(&mut self, n: u64, p: f64) -> Result<TestGraph> {
        if n == 0 {
            return Err(anyhow!("Entity count must be positive"));
        }
        if p < 0.0 || p > 1.0 {
            return Err(anyhow!("Probability p must be between 0 and 1"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: n,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: p, // For Erdős–Rényi: C = p
                diameter: 0,
                density: p,
                connectivity: if p > (n as f64).ln() / (n as f64) {
                    ConnectivityType::Connected
                } else {
                    ConnectivityType::Random
                },
                expected_path_length: (n as f64).ln() / (n as f64 * p).ln(), // ln(n) / ln(np)
            },
        };

        // Add entities with deterministic IDs
        for i in 0..n {
            let entity = TestEntity {
                id: i as u32,
                name: format!("Entity_{}", i),
                entity_type: "Node".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("created_order".to_string(), i.to_string());
                    attrs.insert("expected_degree".to_string(), (p * (n - 1) as f64).to_string());
                    attrs
                },
            };
            graph.entities.push(entity);
        }

        // Add edges with probability p
        let mut edge_count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.rng.next_f64() < p {
                    let edge = TestEdge {
                        source: i as u32,
                        target: j as u32,
                        weight: 1.0,
                        edge_type: "undirected".to_string(),
                    };
                    graph.edges.push(edge);
                    edge_count += 1;
                }
            }
        }

        // Update calculated properties
        graph.properties.edge_count = edge_count;
        graph.properties.average_degree = (2.0 * edge_count as f64) / (n as f64);
        graph.properties.diameter = self.estimate_diameter(&graph);

        self.properties = graph.properties.clone();
        Ok(graph)
    }

    /// Generate Barabási–Albert preferential attachment model
    /// Mathematical guarantee: P(degree = k) ∝ k^(-3) (power law distribution)
    pub fn generate_barabasi_albert(&mut self, n: u64, m: u32) -> Result<TestGraph> {
        if n == 0 {
            return Err(anyhow!("Entity count must be positive"));
        }
        if m == 0 || m as u64 >= n {
            return Err(anyhow!("m must be positive and less than n"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: n,
                edge_count: 0,
                average_degree: 2.0 * m as f64,
                clustering_coefficient: 0.0, // Low clustering for BA model
                diameter: 0,
                density: (2.0 * m as f64) / (n as f64),
                connectivity: ConnectivityType::ScaleFree,
                expected_path_length: (n as f64).ln() / (m as f64).ln(), // Scales as ln(n)/ln(m)
            },
        };

        // Initialize with m+1 fully connected nodes
        for i in 0..=m as u64 {
            let entity = TestEntity {
                id: i as u32,
                name: format!("Entity_{}", i),
                entity_type: "Node".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("created_order".to_string(), i.to_string());
                    attrs.insert("initial_node".to_string(), "true".to_string());
                    attrs
                },
            };
            graph.entities.push(entity);
        }

        // Add initial complete graph edges
        for i in 0..=m as u64 {
            for j in (i + 1)..=m as u64 {
                let edge = TestEdge {
                    source: i as u32,
                    target: j as u32,
                    weight: 1.0,
                    edge_type: "initial".to_string(),
                };
                graph.edges.push(edge);
            }
        }

        // Keep track of degrees for preferential attachment
        let mut degrees: Vec<u32> = vec![m; (m + 1) as usize];
        let mut total_degree = (m + 1) * m;

        // Add remaining nodes with preferential attachment
        for i in (m + 1) as u64..n {
            let entity = TestEntity {
                id: i as u32,
                name: format!("Entity_{}", i),
                entity_type: "Node".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("created_order".to_string(), i.to_string());
                    attrs.insert("attachment_node".to_string(), "true".to_string());
                    attrs
                },
            };
            graph.entities.push(entity);
            degrees.push(0);

            // Choose m existing nodes to connect to based on degree
            let mut chosen_targets = Vec::new();
            for _ in 0..m {
                let mut target = None;
                let mut attempts = 0;
                
                while target.is_none() && attempts < 100 {
                    let random_degree_sum = self.rng.next_u32() % total_degree;
                    let mut cumulative_degree = 0;
                    
                    for (node_id, &node_degree) in degrees.iter().enumerate().take(i as usize) {
                        cumulative_degree += node_degree;
                        if random_degree_sum < cumulative_degree && !chosen_targets.contains(&(node_id as u32)) {
                            target = Some(node_id as u32);
                            break;
                        }
                    }
                    attempts += 1;
                }

                let target_id = target.unwrap_or((i - 1) as u32); // Fallback to previous node
                chosen_targets.push(target_id);

                let edge = TestEdge {
                    source: i as u32,
                    target: target_id,
                    weight: 1.0,
                    edge_type: "preferential".to_string(),
                };
                graph.edges.push(edge);

                // Update degrees
                degrees[i as usize] += 1;
                degrees[target_id as usize] += 1;
                total_degree += 2;
            }
        }

        // Calculate final properties
        graph.properties.edge_count = graph.edges.len() as u64;
        graph.properties.diameter = self.estimate_diameter(&graph);
        
        // BA model clustering coefficient: C ≈ (ln(n))²/n
        graph.properties.clustering_coefficient = ((n as f64).ln().powi(2)) / (n as f64);

        self.properties = graph.properties.clone();
        Ok(graph)
    }

    /// Generate Watts–Strogatz small-world network
    /// Mathematical properties: High clustering, low path length
    pub fn generate_watts_strogatz(&mut self, n: u64, k: u32, beta: f64) -> Result<TestGraph> {
        if n == 0 {
            return Err(anyhow!("Entity count must be positive"));
        }
        if k == 0 || k as u64 >= n || k % 2 != 0 {
            return Err(anyhow!("k must be positive, even, and less than n"));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(anyhow!("Beta must be between 0 and 1"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: n,
                edge_count: (n * k as u64) / 2,
                average_degree: k as f64,
                clustering_coefficient: (3.0 * (k as f64 - 2.0)) / (4.0 * (k as f64 - 1.0)) * (1.0 - beta), // C₀(1-β)
                diameter: 0,
                density: k as f64 / (n - 1) as f64,
                connectivity: ConnectivityType::SmallWorld,
                expected_path_length: if beta == 0.0 {
                    n as f64 / (2.0 * k as f64)
                } else {
                    (n as f64).ln() / (k as f64).ln() // Approximation for β > 0
                },
            },
        };

        // Add entities
        for i in 0..n {
            let entity = TestEntity {
                id: i as u32,
                name: format!("Entity_{}", i),
                entity_type: "Node".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("position".to_string(), i.to_string());
                    attrs.insert("ring_position".to_string(), format!("{:.3}", 2.0 * std::f64::consts::PI * i as f64 / n as f64));
                    attrs
                },
            };
            graph.entities.push(entity);
        }

        // Create initial ring lattice
        for i in 0..n {
            for j in 1..=(k/2) {
                let target = (i + j as u64) % n;
                let edge = TestEdge {
                    source: i as u32,
                    target: target as u32,
                    weight: 1.0,
                    edge_type: "lattice".to_string(),
                };
                graph.edges.push(edge);
            }
        }

        // Rewire edges with probability beta
        let mut rewired_edges = Vec::new();
        for edge in &graph.edges {
            if self.rng.next_f64() < beta {
                // Rewire this edge
                let mut new_target = self.rng.next_u32() % n as u32;
                let mut attempts = 0;
                
                // Ensure we don't create self-loops or duplicate edges
                while (new_target == edge.source || 
                       rewired_edges.iter().any(|e: &TestEdge| 
                           (e.source == edge.source && e.target == new_target) ||
                           (e.target == edge.source && e.source == new_target))) && 
                      attempts < 100 {
                    new_target = self.rng.next_u32() % n as u32;
                    attempts += 1;
                }

                let rewired_edge = TestEdge {
                    source: edge.source,
                    target: new_target,
                    weight: 1.0,
                    edge_type: "rewired".to_string(),
                };
                rewired_edges.push(rewired_edge);
            } else {
                rewired_edges.push(edge.clone());
            }
        }

        graph.edges = rewired_edges;
        graph.properties.diameter = self.estimate_diameter(&graph);

        self.properties = graph.properties.clone();
        Ok(graph)
    }

    /// Generate complete graph with all possible edges
    /// Mathematical guarantees: degree = n-1, edges = n(n-1)/2, diameter = 1
    pub fn generate_complete_graph(&mut self, n: u64) -> Result<TestGraph> {
        if n == 0 {
            return Err(anyhow!("Entity count must be positive"));
        }

        let edge_count = (n * (n - 1)) / 2;
        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: n,
                edge_count,
                average_degree: (n - 1) as f64,
                clustering_coefficient: 1.0, // Perfect clustering
                diameter: if n > 1 { 1 } else { 0 },
                density: 1.0, // Complete graph has maximum density
                connectivity: ConnectivityType::Complete,
                expected_path_length: 1.0, // All nodes directly connected
            },
        };

        // Add entities
        for i in 0..n {
            let entity = TestEntity {
                id: i as u32,
                name: format!("Entity_{}", i),
                entity_type: "Node".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("degree".to_string(), (n - 1).to_string());
                    attrs.insert("centrality".to_string(), "1.0".to_string());
                    attrs
                },
            };
            graph.entities.push(entity);
        }

        // Add all possible edges
        for i in 0..n {
            for j in (i + 1)..n {
                let edge = TestEdge {
                    source: i as u32,
                    target: j as u32,
                    weight: 1.0,
                    edge_type: "complete".to_string(),
                };
                graph.edges.push(edge);
            }
        }

        self.properties = graph.properties.clone();
        Ok(graph)
    }

    /// Generate tree structure with guaranteed properties
    /// Mathematical guarantees: edges = n-1, diameter ≈ 2*log_b(n), no cycles
    pub fn generate_tree(&mut self, n: u64, branching_factor: u32) -> Result<TestGraph> {
        if n == 0 {
            return Err(anyhow!("Entity count must be positive"));
        }
        if branching_factor == 0 {
            return Err(anyhow!("Branching factor must be positive"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: n,
                edge_count: if n > 1 { n - 1 } else { 0 },
                average_degree: if n > 1 { 2.0 * (n - 1) as f64 / n as f64 } else { 0.0 },
                clustering_coefficient: 0.0, // Trees have no triangles
                diameter: 0, // Will be calculated
                density: if n > 1 { 2.0 / (n - 1) as f64 } else { 0.0 },
                connectivity: ConnectivityType::Forest,
                expected_path_length: 0.0, // Will be calculated
            },
        };

        // Add root entity
        if n > 0 {
            let root_entity = TestEntity {
                id: 0,
                name: "Root_Entity_0".to_string(),
                entity_type: "Root".to_string(),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("level".to_string(), "0".to_string());
                    attrs.insert("parent".to_string(), "none".to_string());
                    attrs
                },
            };
            graph.entities.push(root_entity);
        }

        // Generate tree level by level
        let mut current_level = vec![0u32];
        let mut next_id = 1u32;
        let mut level = 1u32;

        while next_id < n as u32 && !current_level.is_empty() {
            let mut next_level = Vec::new();

            for &parent_id in &current_level {
                for child_idx in 0..branching_factor {
                    if next_id >= n as u32 {
                        break;
                    }

                    // Add child entity
                    let child_entity = TestEntity {
                        id: next_id,
                        name: format!("Entity_{}", next_id),
                        entity_type: "Node".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("level".to_string(), level.to_string());
                            attrs.insert("parent".to_string(), parent_id.to_string());
                            attrs.insert("child_index".to_string(), child_idx.to_string());
                            attrs
                        },
                    };
                    graph.entities.push(child_entity);

                    // Add edge from parent to child
                    let edge = TestEdge {
                        source: parent_id,
                        target: next_id,
                        weight: 1.0,
                        edge_type: "tree".to_string(),
                    };
                    graph.edges.push(edge);

                    next_level.push(next_id);
                    next_id += 1;
                }
            }

            current_level = next_level;
            level += 1;
        }

        // Calculate tree-specific properties
        let height = level - 1;
        graph.properties.diameter = if n > 1 { 2 * height } else { 0 };
        
        // Average path length in a balanced tree ≈ height/2 for most pairs
        graph.properties.expected_path_length = if n > 1 { height as f64 * 0.75 } else { 0.0 };

        self.properties = graph.properties.clone();
        Ok(graph)
    }

    /// Estimate graph diameter using BFS from multiple random starting points
    fn estimate_diameter(&mut self, graph: &TestGraph) -> u32 {
        if graph.entities.len() <= 1 {
            return 0;
        }

        // Build adjacency list for BFS
        let mut adj_list: HashMap<u32, Vec<u32>> = HashMap::new();
        for entity in &graph.entities {
            adj_list.insert(entity.id, Vec::new());
        }
        
        for edge in &graph.edges {
            adj_list.get_mut(&edge.source).unwrap().push(edge.target);
            adj_list.get_mut(&edge.target).unwrap().push(edge.source);
        }

        let mut max_distance = 0;
        let sample_count = (graph.entities.len() as f64).sqrt() as usize + 1;

        // Sample random starting points for diameter estimation
        for _ in 0..sample_count {
            let start_id = self.rng.next_u32() % graph.entities.len() as u32;
            let distances = self.bfs_distances(&adj_list, start_id);
            
            for distance in distances.values() {
                if *distance != u32::MAX {
                    max_distance = max_distance.max(*distance);
                }
            }
        }

        max_distance
    }

    /// BFS to compute distances from a starting node
    fn bfs_distances(&self, adj_list: &HashMap<u32, Vec<u32>>, start: u32) -> HashMap<u32, u32> {
        let mut distances = HashMap::new();
        let mut queue = std::collections::VecDeque::new();

        // Initialize distances
        for &node_id in adj_list.keys() {
            distances.insert(node_id, u32::MAX);
        }

        distances.insert(start, 0);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];
            
            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    if distances[&neighbor] == u32::MAX {
                        distances.insert(neighbor, current_distance + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances
    }

    /// Get the properties of the last generated graph
    pub fn get_properties(&self) -> &GraphProperties {
        &self.properties
    }

    /// Validate that generated graph has expected mathematical properties
    pub fn validate_graph_properties(&self, graph: &TestGraph, tolerance: f64) -> Result<()> {
        let props = &graph.properties;

        // Basic sanity checks
        if graph.entities.len() as u64 != props.entity_count {
            return Err(anyhow!("Entity count mismatch: expected {}, got {}", 
                props.entity_count, graph.entities.len()));
        }

        if graph.edges.len() as u64 != props.edge_count {
            return Err(anyhow!("Edge count mismatch: expected {}, got {}", 
                props.edge_count, graph.edges.len()));
        }

        // Validate average degree
        let actual_avg_degree = if props.entity_count > 0 {
            (2.0 * props.edge_count as f64) / (props.entity_count as f64)
        } else {
            0.0
        };

        if (actual_avg_degree - props.average_degree).abs() > tolerance {
            return Err(anyhow!("Average degree mismatch: expected {:.3}, got {:.3}", 
                props.average_degree, actual_avg_degree));
        }

        // Validate density
        let max_edges = if props.entity_count > 1 {
            (props.entity_count * (props.entity_count - 1)) / 2
        } else {
            0
        };

        let actual_density = if max_edges > 0 {
            props.edge_count as f64 / max_edges as f64
        } else {
            0.0
        };

        if (actual_density - props.density).abs() > tolerance {
            return Err(anyhow!("Density mismatch: expected {:.3}, got {:.3}", 
                props.density, actual_density));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erdos_renyi_generation() {
        let mut generator = GraphTopologyGenerator::new(42);
        let graph = generator.generate_erdos_renyi(100, 0.1).unwrap();
        
        assert_eq!(graph.entities.len(), 100);
        assert!(graph.edges.len() > 0);
        
        // Validate properties
        generator.validate_graph_properties(&graph, 0.1).unwrap();
    }

    #[test]
    fn test_barabasi_albert_generation() {
        let mut generator = GraphTopologyGenerator::new(42);
        let graph = generator.generate_barabasi_albert(50, 3).unwrap();
        
        assert_eq!(graph.entities.len(), 50);
        assert_eq!(graph.properties.average_degree, 6.0); // 2m
        
        generator.validate_graph_properties(&graph, 0.1).unwrap();
    }

    #[test]
    fn test_watts_strogatz_generation() {
        let mut generator = GraphTopologyGenerator::new(42);
        let graph = generator.generate_watts_strogatz(20, 4, 0.3).unwrap();
        
        assert_eq!(graph.entities.len(), 20);
        assert_eq!(graph.properties.average_degree, 4.0);
        
        generator.validate_graph_properties(&graph, 0.1).unwrap();
    }

    #[test]
    fn test_complete_graph_generation() {
        let mut generator = GraphTopologyGenerator::new(42);
        let graph = generator.generate_complete_graph(10).unwrap();
        
        assert_eq!(graph.entities.len(), 10);
        assert_eq!(graph.edges.len(), 45); // n(n-1)/2
        assert_eq!(graph.properties.diameter, 1);
        assert_eq!(graph.properties.clustering_coefficient, 1.0);
        
        generator.validate_graph_properties(&graph, 0.001).unwrap();
    }

    #[test]
    fn test_tree_generation() {
        let mut generator = GraphTopologyGenerator::new(42);
        let graph = generator.generate_tree(15, 2).unwrap();
        
        assert_eq!(graph.entities.len(), 15);
        assert_eq!(graph.edges.len(), 14); // n-1 edges
        assert_eq!(graph.properties.clustering_coefficient, 0.0);
        
        generator.validate_graph_properties(&graph, 0.1).unwrap();
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = GraphTopologyGenerator::new(12345);
        let mut gen2 = GraphTopologyGenerator::new(12345);
        
        let graph1 = gen1.generate_erdos_renyi(50, 0.2).unwrap();
        let graph2 = gen2.generate_erdos_renyi(50, 0.2).unwrap();
        
        // Same seed should produce identical graphs
        assert_eq!(graph1.entities.len(), graph2.entities.len());
        assert_eq!(graph1.edges.len(), graph2.edges.len());
        
        for (e1, e2) in graph1.edges.iter().zip(graph2.edges.iter()) {
            assert_eq!(e1.source, e2.source);
            assert_eq!(e1.target, e2.target);
        }
    }

    #[test]
    fn test_invalid_parameters() {
        let mut generator = GraphTopologyGenerator::new(42);
        
        // Test invalid probability
        assert!(generator.generate_erdos_renyi(10, 1.5).is_err());
        
        // Test invalid branching factor
        assert!(generator.generate_barabasi_albert(10, 0).is_err());
        
        // Test invalid rewiring probability
        assert!(generator.generate_watts_strogatz(10, 4, 1.5).is_err());
        
        // Test zero nodes
        assert!(generator.generate_complete_graph(0).is_err());
    }
}