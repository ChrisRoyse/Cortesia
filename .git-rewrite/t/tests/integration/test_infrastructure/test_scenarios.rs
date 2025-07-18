// Test Scenarios
// Predefined scenarios for integration testing

use std::collections::HashMap;
use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::query::RagParameters;

/// Predefined test scenarios
pub struct TestScenarios;

impl TestScenarios {
    /// Get default RAG parameters for testing
    pub fn default_rag_parameters() -> RagParameters {
        RagParameters {
            max_context_entities: 15,
            max_graph_depth: 2,
            similarity_threshold: 0.6,
            diversity_factor: 0.3,
            include_relationships: true,
            max_relationships_per_entity: 5,
            relationship_weight_threshold: 0.1,
            temporal_decay_factor: None,
            entity_type_filters: None,
            relationship_type_filters: None,
            scoring_weights: Default::default(),
        }
    }

    /// Get stress test RAG parameters
    pub fn stress_test_rag_parameters() -> RagParameters {
        RagParameters {
            max_context_entities: 100,
            max_graph_depth: 5,
            similarity_threshold: 0.3,
            diversity_factor: 0.5,
            include_relationships: true,
            max_relationships_per_entity: 20,
            relationship_weight_threshold: 0.0,
            temporal_decay_factor: None,
            entity_type_filters: None,
            relationship_type_filters: None,
            scoring_weights: Default::default(),
        }
    }

    /// Get minimal RAG parameters
    pub fn minimal_rag_parameters() -> RagParameters {
        RagParameters {
            max_context_entities: 5,
            max_graph_depth: 1,
            similarity_threshold: 0.8,
            diversity_factor: 0.0,
            include_relationships: false,
            max_relationships_per_entity: 0,
            relationship_weight_threshold: 1.0,
            temporal_decay_factor: None,
            entity_type_filters: None,
            relationship_type_filters: None,
            scoring_weights: Default::default(),
        }
    }

    /// Create a simple linear graph: A -> B -> C -> D
    pub fn create_linear_graph() -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        let keys = vec![
            EntityKey::from_hash("A"),
            EntityKey::from_hash("B"),
            EntityKey::from_hash("C"),
            EntityKey::from_hash("D"),
        ];

        for (i, key) in keys.iter().enumerate() {
            let entity = Entity::new(*key, format!("Entity {}", key))
                .with_attribute("index", i.to_string());
            entities.insert(*key, entity);
        }

        for i in 0..keys.len() - 1 {
            let rel = Relationship::new(
                format!("rel_{}", i),
                RelationshipType::Directed,
                1.0,
            );
            relationships.push((keys[i], keys[i + 1], rel));
        }

        (entities, relationships)
    }

    /// Create a star graph with center node connected to all others
    pub fn create_star_graph(size: usize) -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        let center = EntityKey::from_hash("center");
        entities.insert(center, Entity::new(center, "Center Node"));

        for i in 0..size {
            let key = EntityKey::from_hash(format!("node_{}", i));
            let entity = Entity::new(key, format!("Node {}", i));
            entities.insert(key, entity);

            let rel = Relationship::new(
                format!("rel_{}", i),
                RelationshipType::Undirected,
                1.0,
            );
            relationships.push((center, key, rel));
        }

        (entities, relationships)
    }

    /// Create a complete graph where every node connects to every other
    pub fn create_complete_graph(size: usize) -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        let mut keys = Vec::new();
        for i in 0..size {
            let key = EntityKey::from_hash(format!("node_{}", i));
            let entity = Entity::new(key, format!("Node {}", i));
            entities.insert(key, entity);
            keys.push(key);
        }

        for i in 0..size {
            for j in (i + 1)..size {
                let rel = Relationship::new(
                    format!("rel_{}_{}", i, j),
                    RelationshipType::Undirected,
                    1.0,
                );
                relationships.push((keys[i], keys[j], rel));
            }
        }

        (entities, relationships)
    }

    /// Create a cycle graph: A -> B -> C -> D -> A
    pub fn create_cycle_graph(size: usize) -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        let mut keys = Vec::new();
        for i in 0..size {
            let key = EntityKey::from_hash(format!("node_{}", i));
            let entity = Entity::new(key, format!("Node {}", i));
            entities.insert(key, entity);
            keys.push(key);
        }

        for i in 0..size {
            let next = (i + 1) % size;
            let rel = Relationship::new(
                format!("rel_{}", i),
                RelationshipType::Directed,
                1.0,
            );
            relationships.push((keys[i], keys[next], rel));
        }

        (entities, relationships)
    }

    /// Create a tree structure
    pub fn create_tree_graph(depth: usize, branching_factor: usize) -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();
        let mut node_counter = 0;

        fn create_node(
            parent_key: Option<EntityKey>,
            depth: usize,
            branching_factor: usize,
            node_counter: &mut usize,
            entities: &mut HashMap<EntityKey, Entity>,
            relationships: &mut Vec<(EntityKey, EntityKey, Relationship)>,
        ) -> EntityKey {
            let key = EntityKey::from_hash(format!("node_{}", *node_counter));
            let entity = Entity::new(key, format!("Node {}", *node_counter))
                .with_attribute("depth", depth.to_string());
            entities.insert(key, entity);
            *node_counter += 1;

            if let Some(parent) = parent_key {
                let rel = Relationship::new(
                    format!("rel_{}_{}", parent, key),
                    RelationshipType::Directed,
                    1.0,
                );
                relationships.push((parent, key, rel));
            }

            if depth > 0 {
                for _ in 0..branching_factor {
                    create_node(
                        Some(key),
                        depth - 1,
                        branching_factor,
                        node_counter,
                        entities,
                        relationships,
                    );
                }
            }

            key
        }

        create_node(None, depth, branching_factor, &mut node_counter, &mut entities, &mut relationships);

        (entities, relationships)
    }

    /// Create test embeddings for entities
    pub fn create_test_embeddings(
        entities: &HashMap<EntityKey, Entity>,
        dimension: usize,
    ) -> HashMap<EntityKey, Vec<f32>> {
        let mut embeddings = HashMap::new();

        for (i, &key) in entities.keys().enumerate() {
            let mut embedding = vec![0.0; dimension];
            
            // Create distinct embeddings
            embedding[i % dimension] = 1.0;
            embedding[(i + 1) % dimension] = 0.5;
            
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }

            embeddings.insert(key, embedding);
        }

        embeddings
    }

    /// Create clustered embeddings
    pub fn create_clustered_embeddings(
        entities: &HashMap<EntityKey, Entity>,
        dimension: usize,
        num_clusters: usize,
    ) -> HashMap<EntityKey, Vec<f32>> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let mut embeddings = HashMap::new();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Create cluster centers
        let mut cluster_centers = Vec::new();
        for i in 0..num_clusters {
            let mut center = vec![0.0; dimension];
            center[i * dimension / num_clusters] = 1.0;
            cluster_centers.push(center);
        }

        // Assign entities to clusters and create embeddings
        for (i, &key) in entities.keys().enumerate() {
            let cluster_idx = i % num_clusters;
            let center = &cluster_centers[cluster_idx];
            
            let mut embedding = center.clone();
            
            // Add noise
            for x in &mut embedding {
                *x += normal.sample(&mut rng) as f32;
            }
            
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }

            embeddings.insert(key, embedding);
        }

        embeddings
    }

    /// Get common entity types for testing
    pub fn common_entity_types() -> Vec<&'static str> {
        vec!["paper", "author", "venue", "topic", "institution", "project", "dataset", "model"]
    }

    /// Get common relationship types for testing
    pub fn common_relationship_types() -> Vec<&'static str> {
        vec!["authored", "cites", "published_in", "related_to", "part_of", "uses", "extends", "contradicts"]
    }

    /// Get test query strings
    pub fn test_queries() -> Vec<&'static str> {
        vec![
            "machine learning algorithms",
            "natural language processing",
            "computer vision models",
            "distributed systems",
            "graph neural networks",
            "quantum computing",
            "blockchain technology",
            "artificial intelligence ethics",
        ]
    }

    /// Create a heterogeneous graph with different entity and relationship types
    pub fn create_heterogeneous_graph() -> (HashMap<EntityKey, Entity>, Vec<(EntityKey, EntityKey, Relationship)>) {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        // Create papers
        for i in 0..5 {
            let key = EntityKey::from_hash(format!("paper_{}", i));
            let entity = Entity::new(key, format!("Paper {}", i))
                .with_attribute("type", "paper")
                .with_attribute("year", (2020 + i).to_string());
            entities.insert(key, entity);
        }

        // Create authors
        for i in 0..3 {
            let key = EntityKey::from_hash(format!("author_{}", i));
            let entity = Entity::new(key, format!("Author {}", i))
                .with_attribute("type", "author")
                .with_attribute("h_index", (10 + i * 5).to_string());
            entities.insert(key, entity);
        }

        // Create venues
        for i in 0..2 {
            let key = EntityKey::from_hash(format!("venue_{}", i));
            let entity = Entity::new(key, format!("Venue {}", i))
                .with_attribute("type", "venue")
                .with_attribute("impact_factor", (2.0 + i as f32).to_string());
            entities.insert(key, entity);
        }

        // Create authorship relationships
        relationships.push((
            EntityKey::from_hash("author_0"),
            EntityKey::from_hash("paper_0"),
            Relationship::new("authored_0", RelationshipType::Directed, 1.0),
        ));
        relationships.push((
            EntityKey::from_hash("author_1"),
            EntityKey::from_hash("paper_0"),
            Relationship::new("authored_1", RelationshipType::Directed, 0.8),
        ));
        relationships.push((
            EntityKey::from_hash("author_0"),
            EntityKey::from_hash("paper_1"),
            Relationship::new("authored_2", RelationshipType::Directed, 1.0),
        ));

        // Create publication relationships
        relationships.push((
            EntityKey::from_hash("paper_0"),
            EntityKey::from_hash("venue_0"),
            Relationship::new("published_0", RelationshipType::Directed, 1.0),
        ));
        relationships.push((
            EntityKey::from_hash("paper_1"),
            EntityKey::from_hash("venue_1"),
            Relationship::new("published_1", RelationshipType::Directed, 1.0),
        ));

        // Create citation relationships
        relationships.push((
            EntityKey::from_hash("paper_1"),
            EntityKey::from_hash("paper_0"),
            Relationship::new("cites_0", RelationshipType::Directed, 0.9),
        ));
        relationships.push((
            EntityKey::from_hash("paper_2"),
            EntityKey::from_hash("paper_0"),
            Relationship::new("cites_1", RelationshipType::Directed, 0.7),
        ));

        (entities, relationships)
    }
}

/// Test data validation utilities
pub struct TestDataValidator;

impl TestDataValidator {
    /// Validate graph connectivity
    pub fn validate_connectivity(
        entities: &HashMap<EntityKey, Entity>,
        relationships: &[(EntityKey, EntityKey, Relationship)],
    ) -> Result<(), String> {
        // Check all relationship endpoints exist
        for (source, target, _) in relationships {
            if !entities.contains_key(source) {
                return Err(format!("Source entity {:?} not found", source));
            }
            if !entities.contains_key(target) {
                return Err(format!("Target entity {:?} not found", target));
            }
        }

        Ok(())
    }

    /// Validate embedding dimensions
    pub fn validate_embeddings(
        embeddings: &HashMap<EntityKey, Vec<f32>>,
        expected_dim: usize,
    ) -> Result<(), String> {
        for (key, embedding) in embeddings {
            if embedding.len() != expected_dim {
                return Err(format!(
                    "Entity {:?} has embedding dimension {} but expected {}",
                    key,
                    embedding.len(),
                    expected_dim
                ));
            }

            // Check for NaN or infinite values
            for (i, &value) in embedding.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!(
                        "Entity {:?} has non-finite value at index {}: {}",
                        key, i, value
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate graph properties
    pub fn validate_graph_properties(
        entities: &HashMap<EntityKey, Entity>,
        relationships: &[(EntityKey, EntityKey, Relationship)],
    ) -> GraphProperties {
        let mut in_degree: HashMap<EntityKey, usize> = HashMap::new();
        let mut out_degree: HashMap<EntityKey, usize> = HashMap::new();

        for &key in entities.keys() {
            in_degree.insert(key, 0);
            out_degree.insert(key, 0);
        }

        for (source, target, rel) in relationships {
            *out_degree.get_mut(source).unwrap() += 1;
            *in_degree.get_mut(target).unwrap() += 1;

            if matches!(rel.relationship_type(), RelationshipType::Undirected) {
                *in_degree.get_mut(source).unwrap() += 1;
                *out_degree.get_mut(target).unwrap() += 1;
            }
        }

        let max_in_degree = in_degree.values().max().copied().unwrap_or(0);
        let max_out_degree = out_degree.values().max().copied().unwrap_or(0);
        let avg_degree = if entities.is_empty() {
            0.0
        } else {
            let total: usize = in_degree.values().sum();
            total as f64 / entities.len() as f64
        };

        GraphProperties {
            entity_count: entities.len(),
            relationship_count: relationships.len(),
            max_in_degree,
            max_out_degree,
            avg_degree,
            in_degree,
            out_degree,
        }
    }
}

/// Graph properties summary
#[derive(Debug, Clone)]
pub struct GraphProperties {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub avg_degree: f64,
    pub in_degree: HashMap<EntityKey, usize>,
    pub out_degree: HashMap<EntityKey, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_graph() {
        let (entities, relationships) = TestScenarios::create_linear_graph();
        assert_eq!(entities.len(), 4);
        assert_eq!(relationships.len(), 3);
        
        let result = TestDataValidator::validate_connectivity(&entities, &relationships);
        assert!(result.is_ok());
    }

    #[test]
    fn test_star_graph() {
        let (entities, relationships) = TestScenarios::create_star_graph(5);
        assert_eq!(entities.len(), 6); // center + 5 nodes
        assert_eq!(relationships.len(), 5);
        
        let props = TestDataValidator::validate_graph_properties(&entities, &relationships);
        assert_eq!(props.max_in_degree, 5); // center node
    }

    #[test]
    fn test_tree_graph() {
        let (entities, relationships) = TestScenarios::create_tree_graph(2, 2);
        assert_eq!(entities.len(), 7); // 1 + 2 + 4
        assert_eq!(relationships.len(), 6);
    }

    #[test]
    fn test_embeddings() {
        let (entities, _) = TestScenarios::create_linear_graph();
        let embeddings = TestScenarios::create_test_embeddings(&entities, 64);
        
        let result = TestDataValidator::validate_embeddings(&embeddings, 64);
        assert!(result.is_ok());
    }
}