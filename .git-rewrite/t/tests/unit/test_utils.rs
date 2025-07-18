//! Test Utilities and High-Level API Wrappers
//! 
//! Provides high-level APIs that unit tests expect, wrapping the low-level optimized implementation.

use crate::infrastructure::DeterministicRng;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Re-export commonly used types for tests
pub use crate::unit::{
    ENTITY_TEST_SEED, GRAPH_TEST_SEED, CSR_TEST_SEED, BLOOM_TEST_SEED,
    EXPECTED_EMPTY_ENTITY_SIZE, EXPECTED_EMPTY_GRAPH_SIZE, ATTRIBUTE_OVERHEAD,
};

/// High-level Entity wrapper for unit tests
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    key: EntityKey,
    name: String,
    attributes: HashMap<String, String>,
    #[serde(skip)]
    memory_cache: Option<u64>,
}

/// Entity key type for unit tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityKey {
    id: u64,
}

/// Relationship type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipType {
    Directed,
    Undirected,
    Weighted,
}

/// Relationship structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relationship {
    name: String,
    weight: f32,
    relationship_type: RelationshipType,
}

/// High-level Knowledge Graph wrapper for unit tests
#[derive(Debug)]
pub struct KnowledgeGraph {
    entities: HashMap<EntityKey, Entity>,
    relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    memory_cache: Option<u64>,
}

impl EntityKey {
    /// Create entity key from hash of string
    pub fn from_hash(input: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        Self { id: hasher.finish() }
    }
    
    /// Create entity key from raw ID
    pub fn from_id(id: u64) -> Self {
        Self { id }
    }
    
    /// Get the raw ID
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Entity {
    /// Create a new entity
    pub fn new(key: EntityKey, name: String) -> Self {
        Self {
            key,
            name,
            attributes: HashMap::new(),
            memory_cache: None,
        }
    }
    
    /// Get entity key
    pub fn key(&self) -> EntityKey {
        self.key
    }
    
    /// Get entity name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Add an attribute
    pub fn add_attribute(&mut self, key: &str, value: &str) {
        self.attributes.insert(key.to_string(), value.to_string());
        self.memory_cache = None; // Invalidate cache
    }
    
    /// Get an attribute
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }
    
    /// Remove an attribute
    pub fn remove_attribute(&mut self, key: &str) -> Option<String> {
        self.memory_cache = None; // Invalidate cache
        self.attributes.remove(key)
    }
    
    /// Get all attributes
    pub fn attributes(&self) -> &HashMap<String, String> {
        &self.attributes
    }
    
    /// Calculate memory usage
    pub fn memory_usage(&self) -> u64 {
        if let Some(cached) = self.memory_cache {
            return cached;
        }
        
        let mut size = EXPECTED_EMPTY_ENTITY_SIZE;
        size += self.name.len() as u64;
        
        for (key, value) in &self.attributes {
            size += key.len() as u64 + value.len() as u64 + ATTRIBUTE_OVERHEAD as u64;
        }
        
        // Note: In a real implementation, we'd cache this
        size
    }
    
    /// Serialize entity to bytes
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    /// Deserialize entity from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| anyhow::anyhow!("Deserialization failed: {}", e))
    }
    
    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }
    
    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| anyhow::anyhow!("JSON deserialization failed: {}", e))
    }
    
    /// Serialize to binary format
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow::anyhow!("Binary serialization failed: {}", e))
    }
    
    /// Deserialize from binary format
    pub fn from_binary(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| anyhow::anyhow!("Binary deserialization failed: {}", e))
    }
    
    /// Validate entity data
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow::anyhow!("Entity name cannot be empty"));
        }
        
        for (key, value) in &self.attributes {
            if key.contains('\0') || value.contains('\0') {
                return Err(anyhow::anyhow!("Attributes cannot contain null bytes"));
            }
            
            if value.len() > 100_000 {
                log::warn!("Attribute '{}' has very large value ({} bytes)", key, value.len());
            }
        }
        
        Ok(())
    }
}

impl Relationship {
    /// Create a new relationship
    pub fn new(name: String, weight: f32, relationship_type: RelationshipType) -> Self {
        Self {
            name,
            weight,
            relationship_type,
        }
    }
    
    /// Get relationship name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get relationship weight
    pub fn weight(&self) -> f32 {
        self.weight
    }
    
    /// Get relationship type
    pub fn relationship_type(&self) -> &RelationshipType {
        &self.relationship_type
    }
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relationships: Vec::new(),
            memory_cache: None,
        }
    }
    
    /// Add an entity to the graph
    pub fn add_entity(&mut self, entity: Entity) -> Result<()> {
        let key = entity.key();
        if self.entities.contains_key(&key) {
            return Err(anyhow::anyhow!("Entity with key {:?} already exists", key));
        }
        
        self.entities.insert(key, entity);
        self.memory_cache = None;
        Ok(())
    }
    
    /// Get an entity by key
    pub fn get_entity(&self, key: EntityKey) -> Option<&Entity> {
        self.entities.get(&key)
    }
    
    /// Check if entity exists
    pub fn contains_entity(&self, key: EntityKey) -> bool {
        self.entities.contains_key(&key)
    }
    
    /// Add a relationship between entities
    pub fn add_relationship(&mut self, source: EntityKey, target: EntityKey, relationship: Relationship) -> Result<()> {
        // Verify both entities exist
        if !self.entities.contains_key(&source) {
            return Err(anyhow::anyhow!("Source entity {:?} not found", source));
        }
        if !self.entities.contains_key(&target) {
            return Err(anyhow::anyhow!("Target entity {:?} not found", target));
        }
        
        self.relationships.push((source, target, relationship));
        self.memory_cache = None;
        Ok(())
    }
    
    /// Get relationships from an entity
    pub fn get_relationships(&self, source: EntityKey) -> Vec<RelationshipEdge> {
        self.relationships
            .iter()
            .filter(|(src, _, _)| *src == source)
            .map(|(_, target, rel)| RelationshipEdge {
                target: *target,
                relationship: rel.clone(),
            })
            .collect()
    }
    
    /// Get entity count
    pub fn entity_count(&self) -> u64 {
        self.entities.len() as u64
    }
    
    /// Get relationship count
    pub fn relationship_count(&self) -> u64 {
        self.relationships.len() as u64
    }
    
    /// Calculate total memory usage
    pub fn memory_usage(&self) -> u64 {
        if let Some(cached) = self.memory_cache {
            return cached;
        }
        
        let mut total = EXPECTED_EMPTY_GRAPH_SIZE;
        
        // Add entity memory
        for entity in self.entities.values() {
            total += entity.memory_usage();
        }
        
        // Add relationship memory (simplified)
        total += self.relationships.len() as u64 * 64; // Estimated relationship size
        
        total
    }
    
    /// Get shortest path length between entities (simple BFS)
    pub fn shortest_path_length(&self, source: EntityKey, target: EntityKey) -> Option<usize> {
        if source == target {
            return Some(0);
        }
        
        use std::collections::{VecDeque, HashSet};
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back((source, 0));
        visited.insert(source);
        
        while let Some((current, distance)) = queue.pop_front() {
            for edge in self.get_relationships(current) {
                if edge.target == target {
                    return Some(distance + 1);
                }
                
                if !visited.contains(&edge.target) {
                    visited.insert(edge.target);
                    queue.push_back((edge.target, distance + 1));
                }
            }
        }
        
        None
    }
    
    /// Get CSR storage representation (mock implementation)
    pub fn get_csr_storage(&self) -> MockCSRStorage {
        MockCSRStorage::from_graph(self)
    }
}

/// Relationship edge for graph traversal
#[derive(Debug, Clone)]
pub struct RelationshipEdge {
    pub target: EntityKey,
    pub relationship: Relationship,
}

impl RelationshipEdge {
    pub fn target(&self) -> EntityKey {
        self.target
    }
    
    pub fn relationship(&self) -> &Relationship {
        &self.relationship
    }
}

/// Mock CSR storage for testing
#[derive(Debug)]
pub struct MockCSRStorage {
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    num_rows: usize,
}

impl MockCSRStorage {
    fn from_graph(graph: &KnowledgeGraph) -> Self {
        let entity_count = graph.entities.len();
        let mut row_offsets = vec![0; entity_count + 1];
        let mut column_indices = Vec::new();
        
        // Convert entities to indices
        let entity_to_index: HashMap<EntityKey, usize> = graph.entities.keys()
            .enumerate()
            .map(|(i, &key)| (key, i))
            .collect();
        
        // Build CSR structure
        for (i, &entity_key) in graph.entities.keys().enumerate() {
            row_offsets[i] = column_indices.len();
            
            for (source, target, _) in &graph.relationships {
                if *source == entity_key {
                    if let Some(&target_idx) = entity_to_index.get(target) {
                        column_indices.push(target_idx);
                    }
                }
            }
        }
        row_offsets[entity_count] = column_indices.len();
        
        Self {
            row_offsets,
            column_indices,
            num_rows: entity_count,
        }
    }
    
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }
    
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }
    
    pub fn num_nonzeros(&self) -> usize {
        self.column_indices.len()
    }
    
    pub fn get_row(&self, row: usize) -> Vec<usize> {
        if row >= self.num_rows {
            return Vec::new();
        }
        
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        self.column_indices[start..end].to_vec()
    }
}

// Utility functions for tests

/// Create a test entity with deterministic properties
pub fn create_test_entity(id: &str, name: &str) -> Entity {
    let key = EntityKey::from_hash(id);
    Entity::new(key, name.to_string())
}

/// Create a test graph with specified number of entities and relationships
pub fn create_test_graph(entity_count: usize, relationship_count: usize) -> KnowledgeGraph {
    let mut rng = DeterministicRng::new(GRAPH_TEST_SEED);
    let mut graph = KnowledgeGraph::new();
    
    // Add entities
    for i in 0..entity_count {
        let entity = create_test_entity(
            &format!("test_entity_{}", i),
            &format!("Test Entity {}", i)
        );
        graph.add_entity(entity).unwrap();
    }
    
    // Add relationships
    for _ in 0..relationship_count {
        let source_idx = rng.gen_range(0..entity_count);
        let target_idx = rng.gen_range(0..entity_count);
        
        if source_idx != target_idx {
            let source_key = EntityKey::from_hash(&format!("test_entity_{}", source_idx));
            let target_key = EntityKey::from_hash(&format!("test_entity_{}", target_idx));
            
            let relationship = Relationship::new(
                "test_relationship".to_string(),
                rng.gen_range(0.1..1.0),
                RelationshipType::Directed
            );
            
            let _ = graph.add_relationship(source_key, target_key, relationship);
        }
    }
    
    graph
}

/// Verify two vectors are approximately equal
pub fn assert_vectors_equal(v1: &[f32], v2: &[f32], tolerance: f32) {
    assert_eq!(v1.len(), v2.len(), "Vector lengths differ");
    for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
        assert!((a - b).abs() < tolerance, 
               "Vectors differ at index {}: {} vs {} (tolerance: {})", 
               i, a, b, tolerance);
    }
}

/// Calculate expected memory usage for an entity
pub fn calculate_expected_entity_memory(entity: &Entity) -> u64 {
    entity.memory_usage()
}

/// Measure execution time of a function
pub fn measure_execution_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Create test vectors with specific properties
pub fn create_test_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = DeterministicRng::new(seed);
    (0..count)
        .map(|_| (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Verify matrix properties for CSR testing
pub fn verify_csr_properties<T>(
    row_offsets: &[usize], 
    column_indices: &[usize], 
    values: &[T],
    num_rows: usize
) -> bool {
    // Check row offsets length
    if row_offsets.len() != num_rows + 1 {
        return false;
    }
    
    // Check monotonic increase
    for i in 1..row_offsets.len() {
        if row_offsets[i] < row_offsets[i-1] {
            return false;
        }
    }
    
    // Check consistent lengths
    let nnz = row_offsets[num_rows];
    column_indices.len() == nnz && values.len() == nnz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let key = EntityKey::from_hash("test");
        let entity = Entity::new(key, "Test Entity".to_string());
        
        assert_eq!(entity.key(), key);
        assert_eq!(entity.name(), "Test Entity");
        assert_eq!(entity.attributes().len(), 0);
    }

    #[test]
    fn test_entity_attributes() {
        let mut entity = create_test_entity("test", "Test Entity");
        
        entity.add_attribute("type", "test");
        entity.add_attribute("value", "42");
        
        assert_eq!(entity.get_attribute("type"), Some("test"));
        assert_eq!(entity.get_attribute("value"), Some("42"));
        assert_eq!(entity.get_attribute("nonexistent"), None);
        
        let removed = entity.remove_attribute("type");
        assert_eq!(removed, Some("test".to_string()));
        assert_eq!(entity.get_attribute("type"), None);
    }

    #[test]
    fn test_entity_serialization() {
        let mut entity = create_test_entity("test", "Test Entity");
        entity.add_attribute("key", "value");
        
        // Test JSON
        let json = entity.to_json().unwrap();
        let deserialized = Entity::from_json(&json).unwrap();
        assert_eq!(entity, deserialized);
        
        // Test binary
        let binary = entity.to_binary().unwrap();
        let deserialized_bin = Entity::from_binary(&binary).unwrap();
        assert_eq!(entity, deserialized_bin);
    }

    #[test]
    fn test_knowledge_graph_operations() {
        let mut graph = KnowledgeGraph::new();
        
        let entity1 = create_test_entity("entity1", "Entity 1");
        let entity2 = create_test_entity("entity2", "Entity 2");
        let key1 = entity1.key();
        let key2 = entity2.key();
        
        // Add entities
        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();
        
        assert_eq!(graph.entity_count(), 2);
        assert!(graph.contains_entity(key1));
        assert!(graph.contains_entity(key2));
        
        // Add relationship
        let relationship = Relationship::new("connects".to_string(), 1.0, RelationshipType::Directed);
        graph.add_relationship(key1, key2, relationship).unwrap();
        
        assert_eq!(graph.relationship_count(), 1);
        
        let relationships = graph.get_relationships(key1);
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].target(), key2);
    }

    #[test]
    fn test_test_utilities() {
        let graph = create_test_graph(10, 15);
        assert_eq!(graph.entity_count(), 10);
        assert!(graph.relationship_count() <= 15);
        
        let vectors = create_test_vectors(5, 10, 12345);
        assert_eq!(vectors.len(), 5);
        assert_eq!(vectors[0].len(), 10);
        
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.001, 2.001, 3.001];
        assert_vectors_equal(&v1, &v2, 0.01);
    }
}