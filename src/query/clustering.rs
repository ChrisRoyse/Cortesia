use crate::core::graph::KnowledgeGraph;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};

/// Hierarchical clustering implementation using Leiden algorithm
pub struct HierarchicalClusterer {
    leiden_algorithm: LeidenClustering,
    max_levels: usize,
    min_cluster_size: usize,
    resolution_levels: Vec<f64>,
}

impl HierarchicalClusterer {
    pub fn new(max_levels: usize, min_cluster_size: usize) -> Self {
        Self {
            leiden_algorithm: LeidenClustering::new(),
            max_levels,
            min_cluster_size,
            resolution_levels: vec![0.1, 0.5, 1.0, 2.0, 5.0], // Different granularities
        }
    }

    pub async fn cluster_graph(&self, graph: &KnowledgeGraph) -> Result<ClusterHierarchy> {
        let mut hierarchy = ClusterHierarchy::new();
        
        // Get all entities from the graph
        let entities = self.extract_entities(graph)?;
        let adjacency_matrix = self.build_adjacency_matrix(&entities, graph)?;
        
        // Perform clustering at different resolution levels
        for (level, resolution) in self.resolution_levels.iter().enumerate() {
            if level >= self.max_levels {
                break;
            }
            
            let communities = self.leiden_algorithm.cluster(
                &adjacency_matrix,
                *resolution,
                self.min_cluster_size,
            )?;
            
            let cluster_level = ClusterLevel {
                level,
                resolution: *resolution,
                communities,
            };
            
            hierarchy.levels.push(cluster_level);
        }
        
        // Build parent-child relationships between levels
        self.build_hierarchy_relationships(&mut hierarchy)?;
        
        Ok(hierarchy)
    }

    fn extract_entities(&self, _graph: &KnowledgeGraph) -> Result<Vec<u32>> {
        // Get all entity IDs from the graph
        // This would use the actual graph API to enumerate entities
        let mut entities = Vec::new();
        
        // For now, simulate extracting entity IDs
        // In actual implementation, this would iterate through the graph
        for i in 0..1000 { // Placeholder
            entities.push(i);
        }
        
        Ok(entities)
    }

    fn build_adjacency_matrix(&self, entities: &[u32], graph: &KnowledgeGraph) -> Result<AdjacencyMatrix> {
        let n = entities.len();
        let mut matrix = AdjacencyMatrix::new(n);
        
        for (i, &entity_id) in entities.iter().enumerate() {
            if let Ok(neighbors) = graph.get_neighbors(entity_id) {
                for &neighbor_id in &neighbors {
                    if let Some(j) = entities.iter().position(|&id| id == neighbor_id) {
                        matrix.set_edge(i, j, 1.0);
                    }
                }
            }
        }
        
        Ok(matrix)
    }

    fn build_hierarchy_relationships(&self, hierarchy: &mut ClusterHierarchy) -> Result<()> {
        let mut relationships = Vec::new();
        
        for level in 0..hierarchy.levels.len() - 1 {
            let current_level = &hierarchy.levels[level];
            let next_level = &hierarchy.levels[level + 1];
            
            // Build parent-child relationships based on entity overlap
            for (current_id, current_community) in &current_level.communities {
                for (next_id, next_community) in &next_level.communities {
                    let overlap = current_community.entities.intersection(&next_community.entities).count();
                    let overlap_ratio = overlap as f64 / current_community.entities.len() as f64;
                    
                    if overlap_ratio > 0.5 {
                        relationships.push((*current_id, *next_id));
                    }
                }
            }
        }
        
        // Add all relationships after collecting them
        for (current_id, next_id) in relationships {
            hierarchy.add_parent_child_relationship(current_id, next_id);
        }
        
        Ok(())
    }
}

/// Leiden clustering algorithm implementation
struct LeidenClustering {
    max_iterations: usize,
    min_improvement: f64,
}

impl LeidenClustering {
    fn new() -> Self {
        Self {
            max_iterations: 100,
            min_improvement: 0.001,
        }
    }

    fn cluster(&self, adjacency: &AdjacencyMatrix, resolution: f64, min_size: usize) -> Result<HashMap<u32, Community>> {
        let n = adjacency.size();
        let mut communities = HashMap::new();
        let mut node_to_community = vec![0u32; n];
        
        // Initialize each node as its own community
        for i in 0..n {
            let community = Community {
                id: i as u32,
                entities: vec![i as u32].into_iter().collect(),
                internal_edges: 0.0,
                external_edges: 0.0,
                total_degree: adjacency.degree(i),
            };
            communities.insert(i as u32, community);
            node_to_community[i] = i as u32;
        }
        
        let mut improved = true;
        let mut iteration = 0;
        
        while improved && iteration < self.max_iterations {
            improved = false;
            iteration += 1;
            
            // Local moving phase
            for node in 0..n {
                let current_community = node_to_community[node];
                let mut best_community = current_community;
                let mut best_gain = 0.0;
                
                // Get neighboring communities
                let neighbors = adjacency.get_neighbors(node);
                let mut neighbor_communities = HashSet::new();
                
                for neighbor in neighbors {
                    neighbor_communities.insert(node_to_community[neighbor]);
                }
                
                // Try moving to each neighboring community
                for &neighbor_community in &neighbor_communities {
                    if neighbor_community != current_community {
                        let gain = self.calculate_modularity_gain(
                            node,
                            current_community,
                            neighbor_community,
                            adjacency,
                            &communities,
                            resolution,
                        )?;
                        
                        if gain > best_gain {
                            best_gain = gain;
                            best_community = neighbor_community;
                        }
                    }
                }
                
                // Move node if beneficial
                if best_gain > self.min_improvement {
                    self.move_node(node, current_community, best_community, &mut communities, &mut node_to_community)?;
                    improved = true;
                }
            }
        }
        
        // Filter out communities that are too small
        communities.retain(|_, community| community.entities.len() >= min_size);
        
        Ok(communities)
    }

    fn calculate_modularity_gain(
        &self,
        node: usize,
        from_community: u32,
        to_community: u32,
        adjacency: &AdjacencyMatrix,
        communities: &HashMap<u32, Community>,
        resolution: f64,
    ) -> Result<f64> {
        let node_degree = adjacency.degree(node);
        let from_comm = communities.get(&from_community).ok_or_else(|| GraphError::InvalidInput("Community not found".to_string()))?;
        let to_comm = communities.get(&to_community).ok_or_else(|| GraphError::InvalidInput("Community not found".to_string()))?;
        
        let edges_to_from = adjacency.edges_between_node_and_community(node, &from_comm.entities);
        let edges_to_to = adjacency.edges_between_node_and_community(node, &to_comm.entities);
        
        let total_edges = adjacency.total_edges();
        
        // Modularity gain calculation
        let gain = (edges_to_to - edges_to_from) as f64 / total_edges as f64
            - resolution * node_degree as f64 * (to_comm.total_degree - from_comm.total_degree) as f64 / (2.0 * total_edges as f64).powi(2);
        
        Ok(gain)
    }

    fn move_node(
        &self,
        node: usize,
        from_community: u32,
        to_community: u32,
        communities: &mut HashMap<u32, Community>,
        node_to_community: &mut Vec<u32>,
    ) -> Result<()> {
        node_to_community[node] = to_community;
        
        // Update communities
        if let Some(from_comm) = communities.get_mut(&from_community) {
            from_comm.entities.remove(&(node as u32));
        }
        
        if let Some(to_comm) = communities.get_mut(&to_community) {
            to_comm.entities.insert(node as u32);
        }
        
        Ok(())
    }
}

/// Adjacency matrix representation for efficient clustering
struct AdjacencyMatrix {
    matrix: Vec<Vec<f64>>,
    size: usize,
}

impl AdjacencyMatrix {
    fn new(size: usize) -> Self {
        Self {
            matrix: vec![vec![0.0; size]; size],
            size,
        }
    }

    fn set_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.size && to < self.size {
            self.matrix[from][to] = weight;
            self.matrix[to][from] = weight; // Assume undirected graph
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn degree(&self, node: usize) -> f64 {
        self.matrix[node].iter().sum()
    }

    fn get_neighbors(&self, node: usize) -> Vec<usize> {
        self.matrix[node]
            .iter()
            .enumerate()
            .filter(|(_, &weight)| weight > 0.0)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn edges_between_node_and_community(&self, node: usize, community: &HashSet<u32>) -> f64 {
        community.iter()
            .map(|&comm_node| self.matrix[node][comm_node as usize])
            .sum()
    }

    fn total_edges(&self) -> f64 {
        self.matrix.iter()
            .enumerate()
            .map(|(i, row)| row.iter().skip(i + 1).sum::<f64>())
            .sum()
    }
}

/// Represents a community in the graph
#[derive(Debug, Clone)]
pub struct Community {
    pub id: u32,
    pub entities: HashSet<u32>,
    pub internal_edges: f64,
    pub external_edges: f64,
    pub total_degree: f64,
}

/// Represents a level in the cluster hierarchy
#[derive(Debug, Clone)]
pub struct ClusterLevel {
    pub level: usize,
    pub resolution: f64,
    pub communities: HashMap<u32, Community>,
}

/// Complete hierarchical clustering result
#[derive(Debug, Clone)]
pub struct ClusterHierarchy {
    pub levels: Vec<ClusterLevel>,
    pub parent_child_relationships: HashMap<u32, Vec<u32>>, // parent -> children
}

impl ClusterHierarchy {
    fn new() -> Self {
        Self {
            levels: Vec::new(),
            parent_child_relationships: HashMap::new(),
        }
    }

    fn add_parent_child_relationship(&mut self, parent: u32, child: u32) {
        self.parent_child_relationships
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(child);
    }

    /// Get communities at a specific level
    pub fn get_communities_at_level(&self, level: usize) -> Option<&HashMap<u32, Community>> {
        self.levels.get(level).map(|l| &l.communities)
    }

    /// Find the most specific community containing an entity
    pub fn find_entity_community(&self, entity_id: u32) -> Option<(usize, u32)> {
        for (level_idx, level) in self.levels.iter().enumerate() {
            for (community_id, community) in &level.communities {
                if community.entities.contains(&entity_id) {
                    return Some((level_idx, *community_id));
                }
            }
        }
        None
    }

    /// Get all entities in the same community as the given entity at a specific level
    pub fn get_community_members(&self, entity_id: u32, level: usize) -> Option<&HashSet<u32>> {
        let level_data = self.levels.get(level)?;
        
        for community in level_data.communities.values() {
            if community.entities.contains(&entity_id) {
                return Some(&community.entities);
            }
        }
        
        None
    }

    /// Get statistics about the clustering
    pub fn get_statistics(&self) -> ClusteringStatistics {
        let mut stats = ClusteringStatistics {
            num_levels: self.levels.len(),
            communities_per_level: Vec::new(),
            avg_community_size_per_level: Vec::new(),
            modularity_per_level: Vec::new(),
        };

        for level in &self.levels {
            stats.communities_per_level.push(level.communities.len());
            
            let total_entities: usize = level.communities.values()
                .map(|c| c.entities.len())
                .sum();
            
            let avg_size = if level.communities.is_empty() {
                0.0
            } else {
                total_entities as f64 / level.communities.len() as f64
            };
            
            stats.avg_community_size_per_level.push(avg_size);
            stats.modularity_per_level.push(0.0); // Would calculate actual modularity
        }

        stats
    }
}

/// Statistics about the clustering result
#[derive(Debug, Clone)]
pub struct ClusteringStatistics {
    pub num_levels: usize,
    pub communities_per_level: Vec<usize>,
    pub avg_community_size_per_level: Vec<f64>,
    pub modularity_per_level: Vec<f64>,
}