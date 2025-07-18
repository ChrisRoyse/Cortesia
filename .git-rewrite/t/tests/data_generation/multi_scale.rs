//! Multi-Scale Graph Generation
//! 
//! Provides generation of hierarchical and fractal graph structures with multiple scales.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use crate::data_generation::graph_topologies::{TestGraph, TestEntity, TestEdge, GraphProperties, ConnectivityType};
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Specification for hierarchical graph levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSpec {
    pub levels: Vec<LevelSpec>,
    pub connection_patterns: Vec<ConnectionPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelSpec {
    pub level: u32,
    pub node_count: u64,
    pub level_type: LevelType,
    pub clustering_coefficient: f64,
    pub connection_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LevelType {
    Individual,    // Level 0: Individual entities
    LocalCluster,  // Level 1: Local clusters
    Regional,      // Level 2: Regional groups  
    Global,        // Level 3: Global communities
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPattern {
    pub from_level: u32,
    pub to_level: u32,
    pub connection_type: ConnectionType,
    pub probability: f64,
    pub weight_distribution: WeightDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Hierarchical,  // Parent-child relationships
    CrossLevel,    // Skip connections across levels
    Lateral,       // Same-level connections
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightDistribution {
    Uniform { min: f32, max: f32 },
    Exponential { lambda: f64 },
    PowerLaw { alpha: f64 },
}

/// Fractal graph specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalSpec {
    pub base_pattern: BasePattern,
    pub iterations: u32,
    pub scaling_factor: f64,
    pub connection_decay: f64,
    pub self_similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BasePattern {
    Triangle,     // 3-node triangle
    Square,       // 4-node square
    Pentagon,     // 5-node pentagon
    Star,         // Star pattern with center
    Custom { nodes: Vec<u32>, edges: Vec<(u32, u32)> },
}

/// Multi-scale properties for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleProperties {
    pub total_nodes: u64,
    pub total_edges: u64,
    pub level_distribution: Vec<u64>,
    pub fractal_dimension: f64,
    pub self_similarity_score: f64,
    pub hierarchical_clustering: f64,
    pub modularity: f64,
    pub small_world_coefficient: f64,
}

/// Multi-scale graph generator
pub struct MultiScaleGenerator {
    rng: DeterministicRng,
}

impl MultiScaleGenerator {
    /// Create a new multi-scale generator
    pub fn new(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("multi_scale_generator".to_string());
        
        Self { rng }
    }

    /// Generate hierarchical graph with multiple levels
    /// Each level represents a different scale of organization
    pub fn generate_hierarchical_graph(&mut self, spec: HierarchicalSpec) -> Result<TestGraph> {
        if spec.levels.is_empty() {
            return Err(anyhow!("At least one level specification required"));
        }

        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::Random,
                expected_path_length: 0.0,
            },
        };

        let mut level_nodes: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut node_to_level: HashMap<u32, u32> = HashMap::new();
        let mut current_id = 0u32;

        // Generate nodes for each level
        for level_spec in &spec.levels {
            let mut nodes = Vec::new();
            
            for i in 0..level_spec.node_count {
                let entity = TestEntity {
                    id: current_id,
                    name: format!("L{}_N{}", level_spec.level, i),
                    entity_type: format!("{:?}", level_spec.level_type),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("level".to_string(), level_spec.level.to_string());
                        attrs.insert("position_in_level".to_string(), i.to_string());
                        attrs.insert("clustering_coefficient".to_string(), level_spec.clustering_coefficient.to_string());
                        attrs.insert("level_type".to_string(), format!("{:?}", level_spec.level_type));
                        attrs
                    },
                };
                
                graph.entities.push(entity);
                nodes.push(current_id);
                node_to_level.insert(current_id, level_spec.level);
                current_id += 1;
            }
            
            level_nodes.insert(level_spec.level, nodes);
        }

        // Generate connections based on patterns
        for pattern in &spec.connection_patterns {
            self.generate_connections_for_pattern(&mut graph, &level_nodes, &node_to_level, pattern)?;
        }

        // Generate intra-level connections
        for level_spec in &spec.levels {
            if let Some(nodes) = level_nodes.get(&level_spec.level) {
                self.generate_intra_level_connections(&mut graph, nodes, level_spec)?;
            }
        }

        // Calculate graph properties
        self.calculate_hierarchical_properties(&mut graph, &level_nodes)?;

        Ok(graph)
    }

    /// Generate fractal graph structure with self-similarity
    /// Each iteration replaces nodes with scaled copies of the base pattern
    pub fn generate_fractal_graph(&mut self, spec: FractalSpec) -> Result<TestGraph> {
        if spec.iterations == 0 {
            return Err(anyhow!("At least one iteration required"));
        }

        // Start with base pattern
        let mut graph = self.create_base_pattern(&spec.base_pattern)?;
        
        // Apply fractal iterations
        for iteration in 1..=spec.iterations {
            graph = self.apply_fractal_iteration(graph, &spec, iteration)?;
        }

        // Calculate fractal properties
        self.calculate_fractal_properties(&mut graph, &spec)?;

        Ok(graph)
    }

    /// Generate a hierarchical graph with predefined level structure
    pub fn generate_standard_hierarchy(&mut self, levels: u32, nodes_per_level: Vec<u64>) -> Result<TestGraph> {
        if levels as usize != nodes_per_level.len() {
            return Err(anyhow!("Level count must match nodes_per_level length"));
        }

        let level_specs = nodes_per_level.iter().enumerate().map(|(i, &count)| {
            LevelSpec {
                level: i as u32,
                node_count: count,
                level_type: match i {
                    0 => LevelType::Individual,
                    1 => LevelType::LocalCluster,
                    2 => LevelType::Regional,
                    _ => LevelType::Global,
                },
                clustering_coefficient: match i {
                    0 => 0.1,  // Low clustering for individuals
                    1 => 0.6,  // High clustering for local groups
                    2 => 0.4,  // Medium clustering for regions
                    _ => 0.8,  // Very high clustering for global
                },
                connection_probability: 0.1 / (i + 1) as f64, // Decreases with level
            }
        }).collect();

        let connection_patterns = (0..levels-1).map(|i| {
            ConnectionPattern {
                from_level: i,
                to_level: i + 1,
                connection_type: ConnectionType::Hierarchical,
                probability: 0.3,
                weight_distribution: WeightDistribution::Uniform { min: 0.5, max: 1.0 },
            }
        }).collect();

        let spec = HierarchicalSpec {
            levels: level_specs,
            connection_patterns,
        };

        self.generate_hierarchical_graph(spec)
    }

    /// Create base pattern for fractal generation
    fn create_base_pattern(&mut self, pattern: &BasePattern) -> Result<TestGraph> {
        let mut graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::Random,
                expected_path_length: 0.0,
            },
        };

        match pattern {
            BasePattern::Triangle => {
                // Create 3-node triangle
                for i in 0..3 {
                    let entity = TestEntity {
                        id: i,
                        name: format!("Triangle_Node_{}", i),
                        entity_type: "FractalNode".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("pattern_type".to_string(), "Triangle".to_string());
                            attrs.insert("position".to_string(), i.to_string());
                            attrs
                        },
                    };
                    graph.entities.push(entity);
                }
                
                // Add triangle edges
                let edges = vec![(0, 1), (1, 2), (2, 0)];
                for (i, (source, target)) in edges.iter().enumerate() {
                    let edge = TestEdge {
                        source: *source,
                        target: *target,
                        weight: 1.0,
                        edge_type: format!("triangle_edge_{}", i),
                    };
                    graph.edges.push(edge);
                }
            },
            
            BasePattern::Square => {
                // Create 4-node square
                for i in 0..4 {
                    let entity = TestEntity {
                        id: i,
                        name: format!("Square_Node_{}", i),
                        entity_type: "FractalNode".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("pattern_type".to_string(), "Square".to_string());
                            attrs.insert("position".to_string(), i.to_string());
                            attrs
                        },
                    };
                    graph.entities.push(entity);
                }
                
                // Add square edges
                let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
                for (i, (source, target)) in edges.iter().enumerate() {
                    let edge = TestEdge {
                        source: *source,
                        target: *target,
                        weight: 1.0,
                        edge_type: format!("square_edge_{}", i),
                    };
                    graph.edges.push(edge);
                }
            },
            
            BasePattern::Star => {
                // Create star with center (node 0) and 4 spokes
                for i in 0..5 {
                    let entity = TestEntity {
                        id: i,
                        name: format!("Star_Node_{}", i),
                        entity_type: "FractalNode".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("pattern_type".to_string(), "Star".to_string());
                            attrs.insert("role".to_string(), if i == 0 { "center".to_string() } else { "spoke".to_string() });
                            attrs
                        },
                    };
                    graph.entities.push(entity);
                }
                
                // Add star edges (center to all spokes)
                for i in 1..5 {
                    let edge = TestEdge {
                        source: 0,
                        target: i,
                        weight: 1.0,
                        edge_type: format!("star_edge_{}", i),
                    };
                    graph.edges.push(edge);
                }
            },
            
            BasePattern::Pentagon => {
                // Create 5-node pentagon
                for i in 0..5 {
                    let entity = TestEntity {
                        id: i,
                        name: format!("Pentagon_Node_{}", i),
                        entity_type: "FractalNode".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("pattern_type".to_string(), "Pentagon".to_string());
                            attrs.insert("position".to_string(), i.to_string());
                            attrs
                        },
                    };
                    graph.entities.push(entity);
                }
                
                // Add pentagon edges
                for i in 0..5 {
                    let edge = TestEdge {
                        source: i,
                        target: (i + 1) % 5,
                        weight: 1.0,
                        edge_type: format!("pentagon_edge_{}", i),
                    };
                    graph.edges.push(edge);
                }
            },
            
            BasePattern::Custom { nodes, edges } => {
                // Create custom pattern
                for &node_id in nodes {
                    let entity = TestEntity {
                        id: node_id,
                        name: format!("Custom_Node_{}", node_id),
                        entity_type: "FractalNode".to_string(),
                        attributes: {
                            let mut attrs = HashMap::new();
                            attrs.insert("pattern_type".to_string(), "Custom".to_string());
                            attrs.insert("custom_id".to_string(), node_id.to_string());
                            attrs
                        },
                    };
                    graph.entities.push(entity);
                }
                
                for (i, (source, target)) in edges.iter().enumerate() {
                    let edge = TestEdge {
                        source: *source,
                        target: *target,
                        weight: 1.0,
                        edge_type: format!("custom_edge_{}", i),
                    };
                    graph.edges.push(edge);
                }
            }
        }

        graph.properties.entity_count = graph.entities.len() as u64;
        graph.properties.edge_count = graph.edges.len() as u64;

        Ok(graph)
    }

    /// Apply one fractal iteration
    fn apply_fractal_iteration(&mut self, mut base_graph: TestGraph, spec: &FractalSpec, iteration: u32) -> Result<TestGraph> {
        let base_node_count = base_graph.entities.len();
        let base_edge_count = base_graph.edges.len();
        
        // Calculate scaling factor
        let scale = spec.scaling_factor.powi(iteration as i32);
        let connection_strength = spec.connection_decay.powi(iteration as i32);
        
        let mut new_graph = TestGraph {
            entities: Vec::new(),
            edges: Vec::new(),
            properties: GraphProperties {
                entity_count: 0,
                edge_count: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
                density: 0.0,
                connectivity: ConnectivityType::Random,
                expected_path_length: 0.0,
            },
        };

        let mut node_id_offset = 0u32;

        // For each node in the base graph, create a scaled copy of the entire pattern
        for base_node in &base_graph.entities {
            // Create a copy of the base pattern for this node
            for original_node in &base_graph.entities {
                let new_entity = TestEntity {
                    id: node_id_offset + original_node.id,
                    name: format!("Iter{}_Base{}_Node_{}", iteration, base_node.id, original_node.id),
                    entity_type: "FractalNode".to_string(),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("iteration".to_string(), iteration.to_string());
                        attrs.insert("base_node".to_string(), base_node.id.to_string());
                        attrs.insert("pattern_node".to_string(), original_node.id.to_string());
                        attrs.insert("scale".to_string(), scale.to_string());
                        attrs.insert("fractal_level".to_string(), iteration.to_string());
                        attrs
                    },
                };
                new_graph.entities.push(new_entity);
            }

            // Create edges within this copy
            for edge in &base_graph.edges {
                let new_edge = TestEdge {
                    source: node_id_offset + edge.source,
                    target: node_id_offset + edge.target,
                    weight: edge.weight * connection_strength as f32,
                    edge_type: format!("fractal_internal_iter_{}", iteration),
                };
                new_graph.edges.push(new_edge);
            }

            node_id_offset += base_node_count as u32;
        }

        // Add connections between different copies based on original structure
        let copies_per_original = base_node_count;
        for edge in &base_graph.edges {
            // Connect corresponding nodes in different copies
            for copy_id in 0..copies_per_original {
                let source_copy_base = edge.source as usize * copies_per_original;
                let target_copy_base = edge.target as usize * copies_per_original;
                
                // Connect the copy_id-th node in source copy to copy_id-th node in target copy
                if source_copy_base + copy_id < new_graph.entities.len() && 
                   target_copy_base + copy_id < new_graph.entities.len() {
                    let cross_edge = TestEdge {
                        source: (source_copy_base + copy_id) as u32,
                        target: (target_copy_base + copy_id) as u32,
                        weight: edge.weight * connection_strength as f32 * 0.5, // Weaker cross-copy connections
                        edge_type: format!("fractal_cross_iter_{}", iteration),
                    };
                    new_graph.edges.push(cross_edge);
                }
            }
        }

        new_graph.properties.entity_count = new_graph.entities.len() as u64;
        new_graph.properties.edge_count = new_graph.edges.len() as u64;

        Ok(new_graph)
    }

    /// Generate connections based on connection pattern
    fn generate_connections_for_pattern(
        &mut self,
        graph: &mut TestGraph,
        level_nodes: &HashMap<u32, Vec<u32>>,
        node_to_level: &HashMap<u32, u32>,
        pattern: &ConnectionPattern,
    ) -> Result<()> {
        let from_nodes = level_nodes.get(&pattern.from_level).unwrap_or(&vec![]);
        let to_nodes = level_nodes.get(&pattern.to_level).unwrap_or(&vec![]);

        match pattern.connection_type {
            ConnectionType::Hierarchical => {
                // Each higher-level node connects to multiple lower-level nodes
                let connections_per_node = if from_nodes.len() > 0 {
                    (to_nodes.len() as f64 / from_nodes.len() as f64 * pattern.probability).ceil() as usize
                } else {
                    0
                };

                for &from_node in from_nodes {
                    let selected_targets = self.rng.sample(to_nodes, connections_per_node.min(to_nodes.len()));
                    
                    for &target_node in &selected_targets {
                        let weight = self.generate_weight(&pattern.weight_distribution);
                        let edge = TestEdge {
                            source: from_node,
                            target: target_node,
                            weight,
                            edge_type: "hierarchical".to_string(),
                        };
                        graph.edges.push(edge);
                    }
                }
            },
            
            ConnectionType::CrossLevel => {
                // Random connections across levels (skip connections)
                for &from_node in from_nodes {
                    for &to_node in to_nodes {
                        if self.rng.next_f64() < pattern.probability {
                            let weight = self.generate_weight(&pattern.weight_distribution);
                            let edge = TestEdge {
                                source: from_node,
                                target: to_node,
                                weight,
                                edge_type: "cross_level".to_string(),
                            };
                            graph.edges.push(edge);
                        }
                    }
                }
            },
            
            ConnectionType::Lateral => {
                // Same-level connections (should only be used when from_level == to_level)
                if pattern.from_level == pattern.to_level {
                    for i in 0..from_nodes.len() {
                        for j in (i + 1)..from_nodes.len() {
                            if self.rng.next_f64() < pattern.probability {
                                let weight = self.generate_weight(&pattern.weight_distribution);
                                let edge = TestEdge {
                                    source: from_nodes[i],
                                    target: from_nodes[j],
                                    weight,
                                    edge_type: "lateral".to_string(),
                                };
                                graph.edges.push(edge);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate intra-level connections
    fn generate_intra_level_connections(
        &mut self,
        graph: &mut TestGraph,
        nodes: &[u32],
        level_spec: &LevelSpec,
    ) -> Result<()> {
        // Generate connections within the level based on clustering coefficient
        let target_edges = (nodes.len() as f64 * (nodes.len() - 1) as f64 / 2.0 
                          * level_spec.connection_probability) as usize;
        
        let mut edges_created = 0;
        let mut attempts = 0;
        let max_attempts = target_edges * 10;

        while edges_created < target_edges && attempts < max_attempts {
            let i = self.rng.next_usize(nodes.len());
            let j = self.rng.next_usize(nodes.len());
            
            if i != j {
                // Check if edge already exists
                let edge_exists = graph.edges.iter().any(|e| 
                    (e.source == nodes[i] && e.target == nodes[j]) ||
                    (e.source == nodes[j] && e.target == nodes[i])
                );
                
                if !edge_exists {
                    let edge = TestEdge {
                        source: nodes[i],
                        target: nodes[j],
                        weight: 1.0,
                        edge_type: "intra_level".to_string(),
                    };
                    graph.edges.push(edge);
                    edges_created += 1;
                }
            }
            attempts += 1;
        }

        Ok(())
    }

    /// Generate weight based on distribution
    fn generate_weight(&mut self, distribution: &WeightDistribution) -> f32 {
        match distribution {
            WeightDistribution::Uniform { min, max } => {
                self.rng.range_f64(*min as f64, *max as f64) as f32
            },
            WeightDistribution::Exponential { lambda } => {
                (-self.rng.next_f64().ln() / lambda) as f32
            },
            WeightDistribution::PowerLaw { alpha } => {
                let u = self.rng.next_f64();
                (1.0 / u.powf(1.0 / alpha)) as f32
            }
        }
    }

    /// Calculate properties for hierarchical graphs
    fn calculate_hierarchical_properties(
        &mut self,
        graph: &mut TestGraph,
        level_nodes: &HashMap<u32, Vec<u32>>,
    ) -> Result<()> {
        graph.properties.entity_count = graph.entities.len() as u64;
        graph.properties.edge_count = graph.edges.len() as u64;
        
        if graph.entities.len() > 0 {
            graph.properties.average_degree = (2.0 * graph.edges.len() as f64) / (graph.entities.len() as f64);
        }

        // Estimate clustering coefficient (higher for hierarchical structures)
        graph.properties.clustering_coefficient = 0.3 + self.rng.next_f64() * 0.4;
        
        // Estimate diameter (logarithmic for hierarchical)
        graph.properties.diameter = ((graph.entities.len() as f64).ln() * 2.0) as u32;
        
        // Calculate density
        let max_edges = if graph.entities.len() > 1 {
            (graph.entities.len() * (graph.entities.len() - 1)) / 2
        } else {
            0
        };
        graph.properties.density = if max_edges > 0 {
            graph.edges.len() as f64 / max_edges as f64
        } else {
            0.0
        };

        graph.properties.connectivity = ConnectivityType::Random;
        graph.properties.expected_path_length = (graph.entities.len() as f64).ln();

        Ok(())
    }

    /// Calculate properties for fractal graphs
    fn calculate_fractal_properties(&mut self, graph: &mut TestGraph, spec: &FractalSpec) -> Result<()> {
        graph.properties.entity_count = graph.entities.len() as u64;
        graph.properties.edge_count = graph.edges.len() as u64;
        
        if graph.entities.len() > 0 {
            graph.properties.average_degree = (2.0 * graph.edges.len() as f64) / (graph.entities.len() as f64);
        }

        // Fractal graphs have specific scaling properties
        let base_size = match spec.base_pattern {
            BasePattern::Triangle => 3,
            BasePattern::Square => 4,
            BasePattern::Pentagon => 5,
            BasePattern::Star => 5,
            BasePattern::Custom { ref nodes, .. } => nodes.len(),
        };

        // Estimate fractal dimension
        let expected_size = (base_size as f64).powf(spec.iterations as f64);
        graph.properties.clustering_coefficient = spec.self_similarity_threshold;
        graph.properties.diameter = (spec.iterations * 2) as u32;
        
        // Calculate density
        let max_edges = if graph.entities.len() > 1 {
            (graph.entities.len() * (graph.entities.len() - 1)) / 2
        } else {
            0
        };
        graph.properties.density = if max_edges > 0 {
            graph.edges.len() as f64 / max_edges as f64
        } else {
            0.0
        };

        graph.properties.connectivity = ConnectivityType::Random;
        graph.properties.expected_path_length = spec.iterations as f64;

        Ok(())
    }

    /// Calculate multi-scale properties for validation
    pub fn calculate_multi_scale_properties(&self, graph: &TestGraph) -> MultiScaleProperties {
        // Extract level distribution
        let mut level_counts: HashMap<u32, u64> = HashMap::new();
        for entity in &graph.entities {
            if let Some(level_str) = entity.attributes.get("level") {
                if let Ok(level) = level_str.parse::<u32>() {
                    *level_counts.entry(level).or_insert(0) += 1;
                }
            }
        }

        let max_level = level_counts.keys().max().cloned().unwrap_or(0);
        let mut level_distribution = vec![0u64; (max_level + 1) as usize];
        for (level, count) in level_counts {
            level_distribution[level as usize] = count;
        }

        // Calculate fractal dimension (simplified estimation)
        let fractal_dimension = if graph.entities.len() > 1 {
            (graph.edges.len() as f64).ln() / (graph.entities.len() as f64).ln()
        } else {
            1.0
        };

        // Estimate self-similarity score
        let self_similarity_score = graph.properties.clustering_coefficient;

        // Estimate hierarchical clustering
        let hierarchical_clustering = if level_distribution.len() > 1 {
            0.7 // High clustering for hierarchical structures
        } else {
            0.3
        };

        // Estimate modularity (community structure quality)
        let modularity = if level_distribution.len() > 1 {
            0.4 + 0.3 * (level_distribution.len() as f64 / 10.0).min(1.0)
        } else {
            0.1
        };

        // Estimate small-world coefficient
        let small_world_coefficient = graph.properties.clustering_coefficient / 
            (graph.properties.expected_path_length / (graph.entities.len() as f64).ln()).max(1.0);

        MultiScaleProperties {
            total_nodes: graph.entities.len() as u64,
            total_edges: graph.edges.len() as u64,
            level_distribution,
            fractal_dimension,
            self_similarity_score,
            hierarchical_clustering,
            modularity,
            small_world_coefficient,
        }
    }
}

/// Create a standard 4-level hierarchy specification
pub fn create_standard_hierarchy_spec(base_size: u64) -> HierarchicalSpec {
    let levels = vec![
        LevelSpec {
            level: 0,
            node_count: base_size,
            level_type: LevelType::Individual,
            clustering_coefficient: 0.1,
            connection_probability: 0.1,
        },
        LevelSpec {
            level: 1,
            node_count: base_size / 5,
            level_type: LevelType::LocalCluster,
            clustering_coefficient: 0.6,
            connection_probability: 0.3,
        },
        LevelSpec {
            level: 2,
            node_count: base_size / 25,
            level_type: LevelType::Regional,
            clustering_coefficient: 0.4,
            connection_probability: 0.5,
        },
        LevelSpec {
            level: 3,
            node_count: base_size / 125,
            level_type: LevelType::Global,
            clustering_coefficient: 0.8,
            connection_probability: 0.7,
        },
    ];

    let connection_patterns = vec![
        ConnectionPattern {
            from_level: 1,
            to_level: 0,
            connection_type: ConnectionType::Hierarchical,
            probability: 0.4,
            weight_distribution: WeightDistribution::Uniform { min: 0.5, max: 1.0 },
        },
        ConnectionPattern {
            from_level: 2,
            to_level: 1,
            connection_type: ConnectionType::Hierarchical,
            probability: 0.3,
            weight_distribution: WeightDistribution::Uniform { min: 0.6, max: 1.0 },
        },
        ConnectionPattern {
            from_level: 3,
            to_level: 2,
            connection_type: ConnectionType::Hierarchical,
            probability: 0.2,
            weight_distribution: WeightDistribution::Uniform { min: 0.7, max: 1.0 },
        },
        ConnectionPattern {
            from_level: 0,
            to_level: 0,
            connection_type: ConnectionType::Lateral,
            probability: 0.05,
            weight_distribution: WeightDistribution::Uniform { min: 0.3, max: 0.7 },
        },
    ];

    HierarchicalSpec {
        levels,
        connection_patterns,
    }
}

/// Create a standard fractal specification
pub fn create_triangle_fractal_spec(iterations: u32) -> FractalSpec {
    FractalSpec {
        base_pattern: BasePattern::Triangle,
        iterations,
        scaling_factor: 3.0,
        connection_decay: 0.8,
        self_similarity_threshold: 0.7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_graph_generation() {
        let mut generator = MultiScaleGenerator::new(42);
        let spec = create_standard_hierarchy_spec(100);
        
        let graph = generator.generate_hierarchical_graph(spec).unwrap();
        
        // Should have nodes at different levels
        assert!(graph.entities.len() > 100);
        
        // Check level distribution
        let properties = generator.calculate_multi_scale_properties(&graph);
        assert_eq!(properties.level_distribution.len(), 4);
        assert_eq!(properties.level_distribution[0], 100); // Level 0 nodes
    }

    #[test]
    fn test_fractal_graph_generation() {
        let mut generator = MultiScaleGenerator::new(42);
        let spec = create_triangle_fractal_spec(2);
        
        let graph = generator.generate_fractal_graph(spec).unwrap();
        
        // Fractal should grow exponentially
        assert!(graph.entities.len() >= 9); // 3^2 = 9 minimum
        assert!(graph.edges.len() > 0);
        
        // Check fractal properties
        let properties = generator.calculate_multi_scale_properties(&graph);
        assert!(properties.fractal_dimension > 0.0);
    }

    #[test]
    fn test_standard_hierarchy() {
        let mut generator = MultiScaleGenerator::new(42);
        let levels = vec![50, 10, 2, 1];
        
        let graph = generator.generate_standard_hierarchy(4, levels).unwrap();
        
        assert_eq!(graph.entities.len(), 63); // 50 + 10 + 2 + 1
        assert!(graph.edges.len() > 0);
        
        // Verify level attributes
        let level_0_count = graph.entities.iter()
            .filter(|e| e.attributes.get("level") == Some(&"0".to_string()))
            .count();
        assert_eq!(level_0_count, 50);
    }

    #[test]
    fn test_base_pattern_creation() {
        let mut generator = MultiScaleGenerator::new(42);
        
        // Test triangle pattern
        let triangle_graph = generator.create_base_pattern(&BasePattern::Triangle).unwrap();
        assert_eq!(triangle_graph.entities.len(), 3);
        assert_eq!(triangle_graph.edges.len(), 3);
        
        // Test star pattern
        let star_graph = generator.create_base_pattern(&BasePattern::Star).unwrap();
        assert_eq!(star_graph.entities.len(), 5);
        assert_eq!(star_graph.edges.len(), 4); // Center to 4 spokes
    }

    #[test]
    fn test_fractal_iterations() {
        let mut generator = MultiScaleGenerator::new(42);
        
        let spec1 = create_triangle_fractal_spec(1);
        let spec2 = create_triangle_fractal_spec(2);
        
        let graph1 = generator.generate_fractal_graph(spec1).unwrap();
        let graph2 = generator.generate_fractal_graph(spec2).unwrap();
        
        // More iterations should create larger graphs
        assert!(graph2.entities.len() > graph1.entities.len());
    }

    #[test]
    fn test_multi_scale_properties() {
        let mut generator = MultiScaleGenerator::new(42);
        let spec = create_standard_hierarchy_spec(50);
        
        let graph = generator.generate_hierarchical_graph(spec).unwrap();
        let properties = generator.calculate_multi_scale_properties(&graph);
        
        assert!(properties.total_nodes > 0);
        assert!(properties.total_edges > 0);
        assert!(properties.fractal_dimension > 0.0);
        assert!(properties.modularity > 0.0);
        assert!(properties.level_distribution.len() > 1);
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = MultiScaleGenerator::new(12345);
        let mut gen2 = MultiScaleGenerator::new(12345);
        
        let spec = create_standard_hierarchy_spec(20);
        
        let graph1 = gen1.generate_hierarchical_graph(spec.clone()).unwrap();
        let graph2 = gen2.generate_hierarchical_graph(spec).unwrap();
        
        // Same seed should produce identical graphs
        assert_eq!(graph1.entities.len(), graph2.entities.len());
        assert_eq!(graph1.edges.len(), graph2.edges.len());
    }

    #[test]
    fn test_invalid_parameters() {
        let mut generator = MultiScaleGenerator::new(42);
        
        // Test empty level specification
        let empty_spec = HierarchicalSpec {
            levels: vec![],
            connection_patterns: vec![],
        };
        assert!(generator.generate_hierarchical_graph(empty_spec).is_err());
        
        // Test zero iterations for fractal
        let zero_iter_spec = FractalSpec {
            base_pattern: BasePattern::Triangle,
            iterations: 0,
            scaling_factor: 2.0,
            connection_decay: 0.8,
            self_similarity_threshold: 0.5,
        };
        assert!(generator.generate_fractal_graph(zero_iter_spec).is_err());
    }
}