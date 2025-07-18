//! Simple test to verify the testing framework works

use llmkg_tests::*;

#[test]
fn test_simple_entity_creation() {
    let key = test_utils::EntityKey::from_hash("test");
    let entity = test_utils::Entity::new(key, "Test Entity".to_string());
    
    assert_eq!(entity.name(), "Test Entity");
    assert_eq!(entity.attributes().len(), 0);
}

#[test]
fn test_entity_attributes() {
    let mut entity = test_utils::create_test_entity("test", "Test Entity");
    
    entity.add_attribute("type", "test");
    entity.add_attribute("value", "42");
    
    assert_eq!(entity.get_attribute("type"), Some("test"));
    assert_eq!(entity.get_attribute("value"), Some("42"));
    assert_eq!(entity.get_attribute("nonexistent"), None);
}

#[test]
fn test_knowledge_graph_basic() {
    let mut graph = test_utils::KnowledgeGraph::new();
    
    let entity1 = test_utils::create_test_entity("entity1", "Entity 1");
    let entity2 = test_utils::create_test_entity("entity2", "Entity 2");
    let key1 = entity1.key();
    let key2 = entity2.key();
    
    // Add entities
    graph.add_entity(entity1).unwrap();
    graph.add_entity(entity2).unwrap();
    
    assert_eq!(graph.entity_count(), 2);
    assert!(graph.contains_entity(key1));
    assert!(graph.contains_entity(key2));
    
    // Add relationship
    let relationship = test_utils::Relationship::new(
        "connects".to_string(), 
        1.0, 
        test_utils::RelationshipType::Directed
    );
    graph.add_relationship(key1, key2, relationship).unwrap();
    
    assert_eq!(graph.relationship_count(), 1);
    
    let relationships = graph.get_relationships(key1);
    assert_eq!(relationships.len(), 1);
    assert_eq!(relationships[0].target(), key2);
}

#[test]
fn test_deterministic_generation() {
    let mut rng1 = DeterministicRng::new(12345);
    let mut rng2 = DeterministicRng::new(12345);
    
    for _ in 0..100 {
        assert_eq!(rng1.gen::<u64>(), rng2.gen::<u64>());
    }
}

#[tokio::test]
async fn test_data_generation() {
    let generator = data_generation::DataGenerator::new().unwrap();
    
    let params = data_generation::GenerationParams {
        size: data_generation::DataSize::Small,
        entity_count: 5,
        relationship_count: 8,
        embedding_dimension: 32,
        tags: vec!["test".to_string()],
    };

    let data = generator.generate(&params).await.unwrap();
    
    assert_eq!(data.entities.len(), 5);
    assert!(data.relationships.len() <= 8);
    assert_eq!(data.embeddings.len(), 5);
    assert_eq!(data.metadata.embedding_dimension, 32);
}

#[test]
fn test_vector_equality() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![1.001, 2.001, 3.001];
    
    test_utils::assert_vectors_equal(&v1, &v2, 0.01);
}

#[test]
fn test_test_graph_creation() {
    let graph = test_utils::create_test_graph(5, 8);
    assert_eq!(graph.entity_count(), 5);
    assert!(graph.relationship_count() <= 8);
}