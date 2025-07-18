//! Vector Embedding Generation
//! 
//! Provides deterministic generation of vector embeddings with controlled similarity patterns.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Specification for generating clustered embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSpec {
    pub id: u32,
    pub size: u64,
    pub radius: f32,
    pub center: Option<Vec<f32>>,
    pub label: String,
}

/// Hierarchical structure for embedding generation
#[derive(Debug, Clone)]
pub struct HierarchicalStructure {
    pub nodes: Vec<HierarchicalNode>,
    pub edges: Vec<(u32, u32)>, // (parent, child) relationships
}

#[derive(Debug, Clone)]
pub struct HierarchicalNode {
    pub id: u32,
    pub level: u32,
    pub children: Vec<u32>,
    pub parent: Option<u32>,
    pub weight: f32,
}

/// Distance relationship specification
#[derive(Debug, Clone)]
pub struct DistanceConstraint {
    pub entity1: u32,
    pub entity2: u32,
    pub target_distance: f32,
    pub tolerance: f32,
}

/// Generated embedding test set with validation data
#[derive(Debug, Clone)]
pub struct EmbeddingTestSet {
    pub embeddings: HashMap<u32, Vec<f32>>,
    pub ground_truth_similarities: HashMap<(u32, u32), f32>,
    pub cluster_assignments: HashMap<u32, u32>,
    pub expected_nearest_neighbors: HashMap<u32, Vec<(u32, f32)>>,
    pub dimension: usize,
    pub properties: EmbeddingProperties,
}

/// Mathematical properties of generated embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProperties {
    pub dimension: usize,
    pub entity_count: u64,
    pub cluster_count: u32,
    pub average_norm: f64,
    pub variance: f64,
    pub min_distance: f32,
    pub max_distance: f32,
    pub average_distance: f32,
    pub silhouette_score: f64, // Quality of clustering
}

/// Deterministic embedding generator
pub struct EmbeddingGenerator {
    rng: DeterministicRng,
    dimension: usize,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    pub fn new(seed: u64, dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(anyhow!("Dimension must be positive"));
        }

        let mut rng = DeterministicRng::new(seed);
        rng.set_label("embedding_generator".to_string());

        Ok(Self { rng, dimension })
    }

    /// Generate embeddings with known cluster structure
    pub fn generate_clustered_embeddings(&mut self, cluster_specs: Vec<ClusterSpec>) -> Result<EmbeddingTestSet> {
        if cluster_specs.is_empty() {
            return Err(anyhow!("At least one cluster specification required"));
        }

        let mut embeddings = HashMap::new();
        let mut cluster_assignments = HashMap::new();
        let mut entity_id = 0u32;

        // Generate cluster centers if not provided
        let mut centers = Vec::new();
        for cluster_spec in &cluster_specs {
            let center = match &cluster_spec.center {
                Some(c) => {
                    if c.len() != self.dimension {
                        return Err(anyhow!("Cluster center dimension mismatch"));
                    }
                    c.clone()
                },
                None => self.generate_random_unit_vector(),
            };
            centers.push(center);
        }

        // Ensure cluster centers are well-separated
        self.separate_cluster_centers(&mut centers, 2.0)?;

        // Generate points for each cluster
        for (cluster_idx, cluster_spec) in cluster_specs.iter().enumerate() {
            let cluster_center = &centers[cluster_idx];
            
            for _ in 0..cluster_spec.size {
                let embedding = self.generate_point_in_sphere(
                    cluster_center,
                    cluster_spec.radius
                )?;
                
                embeddings.insert(entity_id, embedding);
                cluster_assignments.insert(entity_id, cluster_spec.id);
                entity_id += 1;
            }
        }

        // Compute ground truth similarities and nearest neighbors
        let ground_truth_similarities = self.compute_all_similarities(&embeddings);
        let expected_nearest_neighbors = self.compute_nearest_neighbors(&embeddings, 10);
        
        // Calculate embedding properties
        let properties = self.calculate_embedding_properties(&embeddings, &cluster_assignments)?;

        Ok(EmbeddingTestSet {
            embeddings,
            ground_truth_similarities,
            cluster_assignments,
            expected_nearest_neighbors,
            dimension: self.dimension,
            properties,
        })
    }

    /// Generate embeddings where specific pairs have exact distances
    pub fn generate_distance_controlled_embeddings(&mut self, 
        constraints: Vec<DistanceConstraint>) -> Result<HashMap<u32, Vec<f32>>> {
        
        if constraints.is_empty() {
            return Err(anyhow!("At least one distance constraint required"));
        }

        let mut embeddings = HashMap::new();
        let mut placed_entities = std::collections::HashSet::new();

        // Place first entity at origin
        if let Some(first_constraint) = constraints.first() {
            let origin = vec![0.0; self.dimension];
            embeddings.insert(first_constraint.entity1, origin);
            placed_entities.insert(first_constraint.entity1);
        }

        // Place entities to satisfy distance constraints
        for constraint in &constraints {
            if !placed_entities.contains(&constraint.entity1) && !embeddings.contains_key(&constraint.entity1) {
                embeddings.insert(constraint.entity1, self.generate_random_unit_vector());
                placed_entities.insert(constraint.entity1);
            }

            if !placed_entities.contains(&constraint.entity2) {
                let reference_embedding = embeddings[&constraint.entity1].clone();
                let target_embedding = self.generate_point_at_distance(
                    &reference_embedding, 
                    constraint.target_distance
                )?;
                embeddings.insert(constraint.entity2, target_embedding);
                placed_entities.insert(constraint.entity2);
            }
        }

        // Validate distance constraints
        for constraint in &constraints {
            let emb1 = &embeddings[&constraint.entity1];
            let emb2 = &embeddings[&constraint.entity2];
            let actual_distance = self.euclidean_distance(emb1, emb2);
            
            if (actual_distance - constraint.target_distance).abs() > constraint.tolerance {
                return Err(anyhow!("Failed to satisfy distance constraint: entities {}-{}, target: {:.3}, actual: {:.3}",
                    constraint.entity1, constraint.entity2, constraint.target_distance, actual_distance));
            }
        }

        Ok(embeddings)
    }

    /// Generate embeddings that reflect hierarchical relationships
    /// Parent embeddings are weighted averages of children
    /// Sibling embeddings are clustered together
    pub fn generate_hierarchical_embeddings(&mut self, 
        hierarchy: &HierarchicalStructure) -> Result<HashMap<u32, Vec<f32>>> {
        
        let mut embeddings = HashMap::new();
        
        // Sort nodes by level (bottom-up generation)
        let mut nodes_by_level: HashMap<u32, Vec<&HierarchicalNode>> = HashMap::new();
        let mut max_level = 0;
        
        for node in &hierarchy.nodes {
            nodes_by_level.entry(node.level).or_insert_with(Vec::new).push(node);
            max_level = max_level.max(node.level);
        }

        // Generate embeddings level by level, from leaves to root
        for level in (0..=max_level).rev() {
            if let Some(level_nodes) = nodes_by_level.get(&level) {
                for node in level_nodes {
                    let embedding = if node.children.is_empty() {
                        // Leaf node: generate random embedding
                        self.generate_random_unit_vector()
                    } else {
                        // Internal node: weighted average of children
                        self.compute_parent_embedding(node, &embeddings)?
                    };
                    
                    embeddings.insert(node.id, embedding);
                }
            }
        }

        // Add noise to make hierarchy approximately preserved
        self.add_hierarchical_noise(&mut embeddings, hierarchy, 0.1)?;

        Ok(embeddings)
    }

    /// Generate a random unit vector (normalized)
    pub fn generate_random_unit_vector(&mut self) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimension];
        let mut norm_squared = 0.0;

        // Generate vector with normal distribution
        for i in 0..self.dimension {
            let value = self.rng.normal(0.0, 1.0) as f32;
            vector[i] = value;
            norm_squared += value * value;
        }

        // Normalize to unit vector
        let norm = norm_squared.sqrt();
        if norm > 1e-10 {
            for i in 0..self.dimension {
                vector[i] /= norm;
            }
        }

        vector
    }

    /// Generate a point within a sphere of given radius around a center
    pub fn generate_point_in_sphere(&mut self, center: &[f32], radius: f32) -> Result<Vec<f32>> {
        if center.len() != self.dimension {
            return Err(anyhow!("Center dimension mismatch"));
        }

        // Generate random direction
        let direction = self.generate_random_unit_vector();
        
        // Generate random distance within sphere (uniform distribution in volume)
        let u = self.rng.next_f32();
        let distance = radius * u.powf(1.0 / self.dimension as f32);

        // Create point
        let mut point = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            point[i] = center[i] + distance * direction[i];
        }

        Ok(point)
    }

    /// Generate a point at exactly the specified distance from reference
    pub fn generate_point_at_distance(&mut self, reference: &[f32], distance: f32) -> Result<Vec<f32>> {
        if reference.len() != self.dimension {
            return Err(anyhow!("Reference dimension mismatch"));
        }

        let random_direction = self.generate_random_unit_vector();
        let mut result = vec![0.0; self.dimension];

        for i in 0..self.dimension {
            result[i] = reference[i] + distance * random_direction[i];
        }

        // Verify distance is correct (within floating point precision)
        let actual_distance = self.euclidean_distance(reference, &result);
        if (actual_distance - distance).abs() > 1e-6 {
            return Err(anyhow!("Failed to generate point at exact distance: target {:.6}, actual {:.6}", 
                distance, actual_distance));
        }

        Ok(result)
    }

    /// Compute Euclidean distance between two vectors
    pub fn euclidean_distance(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return f32::INFINITY;
        }

        let mut sum_squared = 0.0;
        for i in 0..vec1.len() {
            let diff = vec1[i] - vec2[i];
            sum_squared += diff * diff;
        }

        sum_squared.sqrt()
    }

    /// Compute cosine similarity between two vectors
    pub fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1_squared = 0.0;
        let mut norm2_squared = 0.0;

        for i in 0..vec1.len() {
            dot_product += vec1[i] * vec2[i];
            norm1_squared += vec1[i] * vec1[i];
            norm2_squared += vec2[i] * vec2[i];
        }

        let norm1 = norm1_squared.sqrt();
        let norm2 = norm2_squared.sqrt();

        if norm1 > 1e-10 && norm2 > 1e-10 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Separate cluster centers to ensure minimum distance
    fn separate_cluster_centers(&mut self, centers: &mut Vec<Vec<f32>>, min_distance: f32) -> Result<()> {
        let max_attempts = 1000;
        
        for i in 0..centers.len() {
            for j in (i + 1)..centers.len() {
                let mut attempts = 0;
                
                while self.euclidean_distance(&centers[i], &centers[j]) < min_distance && attempts < max_attempts {
                    // Move centers apart
                    let direction = self.generate_random_unit_vector();
                    for dim in 0..self.dimension {
                        centers[j][dim] += min_distance * 0.1 * direction[dim];
                    }
                    attempts += 1;
                }
                
                if attempts >= max_attempts {
                    return Err(anyhow!("Failed to separate cluster centers after {} attempts", max_attempts));
                }
            }
        }
        
        Ok(())
    }

    /// Compute all pairwise similarities
    fn compute_all_similarities(&self, embeddings: &HashMap<u32, Vec<f32>>) -> HashMap<(u32, u32), f32> {
        let mut similarities = HashMap::new();
        
        let entity_ids: Vec<u32> = embeddings.keys().cloned().collect();
        
        for i in 0..entity_ids.len() {
            for j in (i + 1)..entity_ids.len() {
                let id1 = entity_ids[i];
                let id2 = entity_ids[j];
                
                let emb1 = &embeddings[&id1];
                let emb2 = &embeddings[&id2];
                
                let similarity = self.cosine_similarity(emb1, emb2);
                similarities.insert((id1, id2), similarity);
                similarities.insert((id2, id1), similarity); // Symmetric
            }
        }
        
        similarities
    }

    /// Compute k nearest neighbors for each entity
    fn compute_nearest_neighbors(&self, embeddings: &HashMap<u32, Vec<f32>>, k: usize) -> HashMap<u32, Vec<(u32, f32)>> {
        let mut nearest_neighbors = HashMap::new();
        
        for (&query_id, query_embedding) in embeddings.iter() {
            let mut candidates = Vec::new();
            
            for (&candidate_id, candidate_embedding) in embeddings.iter() {
                if query_id != candidate_id {
                    let distance = self.euclidean_distance(query_embedding, candidate_embedding);
                    candidates.push((candidate_id, distance));
                }
            }
            
            // Sort by distance and take top k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            candidates.truncate(k);
            
            nearest_neighbors.insert(query_id, candidates);
        }
        
        nearest_neighbors
    }

    /// Calculate mathematical properties of embeddings
    fn calculate_embedding_properties(&self, 
        embeddings: &HashMap<u32, Vec<f32>>, 
        cluster_assignments: &HashMap<u32, u32>) -> Result<EmbeddingProperties> {
        
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to analyze"));
        }

        let entity_count = embeddings.len() as u64;
        let cluster_count = cluster_assignments.values().max().unwrap_or(&0) + 1;

        // Calculate norms and distances
        let mut norms = Vec::new();
        let mut distances = Vec::new();
        
        for embedding in embeddings.values() {
            let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt() as f64;
            norms.push(norm);
        }

        let entity_ids: Vec<u32> = embeddings.keys().cloned().collect();
        for i in 0..entity_ids.len() {
            for j in (i + 1)..entity_ids.len() {
                let emb1 = &embeddings[&entity_ids[i]];
                let emb2 = &embeddings[&entity_ids[j]];
                let distance = self.euclidean_distance(emb1, emb2);
                distances.push(distance);
            }
        }

        let average_norm = norms.iter().sum::<f64>() / norms.len() as f64;
        let variance = norms.iter().map(|&x| (x - average_norm).powi(2)).sum::<f64>() / norms.len() as f64;
        
        let min_distance = distances.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_distance = distances.iter().fold(0.0, |a, &b| a.max(b));
        let average_distance = distances.iter().sum::<f32>() / distances.len() as f32;

        // Calculate silhouette score (clustering quality)
        let silhouette_score = self.calculate_silhouette_score(embeddings, cluster_assignments);

        Ok(EmbeddingProperties {
            dimension: self.dimension,
            entity_count,
            cluster_count,
            average_norm,
            variance,
            min_distance,
            max_distance,
            average_distance,
            silhouette_score,
        })
    }

    /// Calculate silhouette score for clustering quality assessment
    fn calculate_silhouette_score(&self, 
        embeddings: &HashMap<u32, Vec<f32>>, 
        cluster_assignments: &HashMap<u32, u32>) -> f64 {
        
        let mut silhouette_scores = Vec::new();
        
        for (&entity_id, embedding) in embeddings.iter() {
            let cluster_id = cluster_assignments[&entity_id];
            
            // Calculate average distance to same cluster (a)
            let mut same_cluster_distances = Vec::new();
            for (&other_id, other_embedding) in embeddings.iter() {
                if other_id != entity_id && cluster_assignments[&other_id] == cluster_id {
                    let distance = self.euclidean_distance(embedding, other_embedding);
                    same_cluster_distances.push(distance as f64);
                }
            }
            
            let a = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };
            
            // Calculate minimum average distance to other clusters (b)
            let mut min_other_cluster_distance = f64::INFINITY;
            let unique_clusters: std::collections::HashSet<u32> = cluster_assignments.values().cloned().collect();
            
            for &other_cluster_id in &unique_clusters {
                if other_cluster_id != cluster_id {
                    let mut other_cluster_distances = Vec::new();
                    for (&other_id, other_embedding) in embeddings.iter() {
                        if cluster_assignments[&other_id] == other_cluster_id {
                            let distance = self.euclidean_distance(embedding, other_embedding);
                            other_cluster_distances.push(distance as f64);
                        }
                    }
                    
                    if !other_cluster_distances.is_empty() {
                        let avg_distance = other_cluster_distances.iter().sum::<f64>() / other_cluster_distances.len() as f64;
                        min_other_cluster_distance = min_other_cluster_distance.min(avg_distance);
                    }
                }
            }
            
            let b = min_other_cluster_distance;
            
            // Calculate silhouette score for this point
            let silhouette = if a.max(b) > 1e-10 {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            
            silhouette_scores.push(silhouette);
        }
        
        if silhouette_scores.is_empty() {
            0.0
        } else {
            silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
        }
    }

    /// Compute parent embedding as weighted average of children
    fn compute_parent_embedding(&self, 
        node: &HierarchicalNode, 
        embeddings: &HashMap<u32, Vec<f32>>) -> Result<Vec<f32>> {
        
        if node.children.is_empty() {
            return Err(anyhow!("Cannot compute parent embedding: no children"));
        }

        let mut parent_embedding = vec![0.0; self.dimension];
        let mut total_weight = 0.0;

        for &child_id in &node.children {
            if let Some(child_embedding) = embeddings.get(&child_id) {
                for i in 0..self.dimension {
                    parent_embedding[i] += child_embedding[i] * node.weight;
                }
                total_weight += node.weight;
            }
        }

        if total_weight > 1e-10 {
            for i in 0..self.dimension {
                parent_embedding[i] /= total_weight;
            }
        }

        Ok(parent_embedding)
    }

    /// Add hierarchical noise to preserve approximate relationships
    fn add_hierarchical_noise(&mut self, 
        embeddings: &mut HashMap<u32, Vec<f32>>, 
        _hierarchy: &HierarchicalStructure, 
        noise_level: f32) -> Result<()> {
        
        for embedding in embeddings.values_mut() {
            for i in 0..self.dimension {
                let noise = self.rng.normal(0.0, noise_level as f64) as f32;
                embedding[i] += noise;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustered_embedding_generation() {
        let mut generator = EmbeddingGenerator::new(42, 128).unwrap();
        
        let cluster_specs = vec![
            ClusterSpec {
                id: 0,
                size: 50,
                radius: 0.5,
                center: None,
                label: "Cluster 0".to_string(),
            },
            ClusterSpec {
                id: 1,
                size: 30,
                radius: 0.3,
                center: None,
                label: "Cluster 1".to_string(),
            },
        ];

        let test_set = generator.generate_clustered_embeddings(cluster_specs).unwrap();
        
        assert_eq!(test_set.embeddings.len(), 80);
        assert_eq!(test_set.dimension, 128);
        assert!(test_set.properties.silhouette_score > 0.0); // Should have good clustering
    }

    #[test]
    fn test_distance_controlled_embeddings() {
        let mut generator = EmbeddingGenerator::new(42, 64).unwrap();
        
        let constraints = vec![
            DistanceConstraint {
                entity1: 0,
                entity2: 1,
                target_distance: 1.0,
                tolerance: 0.001,
            },
            DistanceConstraint {
                entity1: 1,
                entity2: 2,
                target_distance: 2.0,
                tolerance: 0.001,
            },
        ];

        let embeddings = generator.generate_distance_controlled_embeddings(constraints).unwrap();
        
        assert_eq!(embeddings.len(), 3);
        
        // Verify distances
        let dist_01 = generator.euclidean_distance(&embeddings[&0], &embeddings[&1]);
        let dist_12 = generator.euclidean_distance(&embeddings[&1], &embeddings[&2]);
        
        assert!((dist_01 - 1.0).abs() < 0.001);
        assert!((dist_12 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_hierarchical_embeddings() {
        let mut generator = EmbeddingGenerator::new(42, 32).unwrap();
        
        let hierarchy = HierarchicalStructure {
            nodes: vec![
                HierarchicalNode { id: 0, level: 0, children: vec![1, 2], parent: None, weight: 1.0 },
                HierarchicalNode { id: 1, level: 1, children: vec![], parent: Some(0), weight: 1.0 },
                HierarchicalNode { id: 2, level: 1, children: vec![], parent: Some(0), weight: 1.0 },
            ],
            edges: vec![(0, 1), (0, 2)],
        };

        let embeddings = generator.generate_hierarchical_embeddings(&hierarchy).unwrap();
        
        assert_eq!(embeddings.len(), 3);
        
        // Verify hierarchical property (parent should be close to average of children)
        let parent_emb = &embeddings[&0];
        let child1_emb = &embeddings[&1];
        let child2_emb = &embeddings[&2];
        
        let mut expected_parent = vec![0.0; 32];
        for i in 0..32 {
            expected_parent[i] = (child1_emb[i] + child2_emb[i]) / 2.0;
        }
        
        let distance_to_expected = generator.euclidean_distance(parent_emb, &expected_parent);
        assert!(distance_to_expected < 1.0); // Should be reasonably close
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = EmbeddingGenerator::new(12345, 64).unwrap();
        let mut gen2 = EmbeddingGenerator::new(12345, 64).unwrap();
        
        let cluster_specs = vec![
            ClusterSpec {
                id: 0,
                size: 10,
                radius: 0.5,
                center: None,
                label: "Test Cluster".to_string(),
            },
        ];

        let test_set1 = gen1.generate_clustered_embeddings(cluster_specs.clone()).unwrap();
        let test_set2 = gen2.generate_clustered_embeddings(cluster_specs).unwrap();
        
        // Same seed should produce identical embeddings
        for (id, emb1) in &test_set1.embeddings {
            let emb2 = &test_set2.embeddings[id];
            for (v1, v2) in emb1.iter().zip(emb2.iter()) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_embedding_properties_calculation() {
        let mut generator = EmbeddingGenerator::new(42, 16).unwrap();
        
        let cluster_specs = vec![
            ClusterSpec {
                id: 0,
                size: 20,
                radius: 0.3,
                center: None,
                label: "Test Cluster".to_string(),
            },
        ];

        let test_set = generator.generate_clustered_embeddings(cluster_specs).unwrap();
        
        assert_eq!(test_set.properties.entity_count, 20);
        assert_eq!(test_set.properties.cluster_count, 1);
        assert!(test_set.properties.average_norm > 0.0);
        assert!(test_set.properties.min_distance >= 0.0);
        assert!(test_set.properties.max_distance > test_set.properties.min_distance);
    }

    #[test]
    fn test_invalid_parameters() {
        // Test invalid dimension
        assert!(EmbeddingGenerator::new(42, 0).is_err());
        
        let mut generator = EmbeddingGenerator::new(42, 64).unwrap();
        
        // Test empty cluster specs
        assert!(generator.generate_clustered_embeddings(vec![]).is_err());
        
        // Test empty distance constraints
        assert!(generator.generate_distance_controlled_embeddings(vec![]).is_err());
    }
}