use llmkg::core::types::{
    EntityKey, EntityData, EntityMeta, Relationship, AttributeValue, 
    RelationshipType, Weight, ContextEntity, QueryResult, GraphQuery, TraversalParams
};
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::graph::KnowledgeGraph;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::error::{GraphError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio;
use serde_json;

/// Helper to create test entity data
fn create_test_entity(type_id: u16, properties: &str, embedding_size: usize) -> EntityData {
    EntityData::new(type_id, properties.to_string(), vec![0.1; embedding_size])
}

#[tokio::test]
async fn test_type_compatibility_across_modules() {
    // Test that types work correctly across module boundaries
    let temp_dir = tempfile::tempdir().unwrap();
    
    // Create knowledge engine and graph
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(96, 10000).unwrap()
    ));
    
    let graph = Arc::new(tokio::sync::RwLock::new(
        KnowledgeGraph::new(96).unwrap()
    ));
    
    let embedding_store = Arc::new(tokio::sync::RwLock::new(
        EmbeddingStore::new(96, 8).unwrap()
    ));

    // Test 1: EntityData compatibility across modules
    let entity_data = create_test_entity(1, "Test entity", 96);
    
    // Add to knowledge engine
    let entity_key = {
        let mut engine_write = engine.write().await;
        engine_write.store_entity(
            "test_entity".to_string(),
            "test".to_string(),
            "test description".to_string(),
            HashMap::new()
        ).unwrap();
        EntityKey::from_raw_parts(1, 0)
    };
    
    // Add to graph
    {
        let mut graph_write = graph.write().await;
        graph_write.insert_entity(1, entity_data.clone()).unwrap();
    }
    
    // Add to embedding store
    {
        let mut embed_write = embedding_store.write().await;
        embed_write.store_embedding(&entity_data.embedding).unwrap();
    }

    // Test 2: Relationship compatibility
    let entity_key2 = {
        let mut engine_write = engine.write().await;
        engine_write.store_entity(
            "test_entity2".to_string(),
            "test".to_string(),
            "second test description".to_string(),
            HashMap::new()
        ).unwrap();
        EntityKey::from_raw_parts(2, 0)
    };
    
    let relationship = Relationship {
        from: entity_key,
        to: entity_key2,
        rel_type: 1,
        weight: 0.75,
    };
    
    // Add relationship through graph
    {
        let mut graph_write = graph.write().await;
        graph_write.add_relationship(entity_key, entity_key2, relationship.weight).unwrap();
    }

    // Test 3: Query compatibility
    let query_result = {
        let engine_read = engine.read().await;
        let results = engine_read.semantic_search("test embedding search", 5).unwrap();
        
        // Convert to QueryResult type
        QueryResult {
            entities: results.nodes.iter().take(2).map(|node| {
                ContextEntity {
                    id: EntityKey::from_raw_parts(node.id.parse::<u64>().unwrap_or(0), 0),
                    similarity: 0.8, // Placeholder similarity
                    neighbors: vec![],
                    properties: format!("Node ID: {}", node.id),
                }
            }).collect(),
            relationships: vec![relationship],
            confidence: 0.85,
            query_time_ms: results.query_time_ms,
        }
    };
    
    assert!(!query_result.entities.is_empty());
    assert_eq!(query_result.relationships.len(), 1);
    assert_eq!(query_result.confidence, 0.85);
}

#[tokio::test]
async fn test_serialization_deserialization_workflows() {
    // Test complete serialization/deserialization workflows
    
    // Test 1: AttributeValue serialization
    let mut test_object = HashMap::new();
    test_object.insert("key1".to_string(), AttributeValue::String("value1".to_string()));
    test_object.insert("key2".to_string(), AttributeValue::Number(42.0));
    test_object.insert("key3".to_string(), AttributeValue::Boolean(true));
    
    let complex_attribute = AttributeValue::Object(test_object);
    
    // Serialize
    let serialized = serde_json::to_string(&complex_attribute).unwrap();
    println!("Serialized AttributeValue: {}", serialized);
    
    // Deserialize
    let deserialized: AttributeValue = serde_json::from_str(&serialized).unwrap();
    
    // Verify
    if let AttributeValue::Object(obj) = deserialized {
        assert_eq!(obj.len(), 3);
        assert_eq!(obj.get("key1").unwrap().as_string(), Some("value1"));
        assert_eq!(obj.get("key2").unwrap().as_number(), Some(42.0));
        assert_eq!(obj.get("key3").unwrap().as_boolean(), Some(true));
    } else {
        panic!("Deserialization failed");
    }

    // Test 2: EntityData with embeddings
    let entity_data = EntityData::new(10, r#"{"name": "Test Entity", "description": "A test entity for serialization"}"#.to_string(), vec![0.1; 96]);
    
    let entity_json = serde_json::to_string(&entity_data).unwrap();
    let entity_restored: EntityData = serde_json::from_str(&entity_json).unwrap();
    
    assert_eq!(entity_data.type_id, entity_restored.type_id);
    assert_eq!(entity_data.properties, entity_restored.properties);
    assert_eq!(entity_data.embedding, entity_restored.embedding);

    // Test 3: Complex nested structures
    let nested_array = AttributeValue::Array(vec![
        AttributeValue::String("item1".to_string()),
        AttributeValue::Number(123.45),
        AttributeValue::Array(vec![
            AttributeValue::Boolean(true),
            AttributeValue::Null,
        ]),
        AttributeValue::Vector(vec![1.0, 2.0, 3.0]),
    ]);
    
    let nested_json = serde_json::to_string(&nested_array).unwrap();
    let nested_restored: AttributeValue = serde_json::from_str(&nested_json).unwrap();
    
    if let AttributeValue::Array(arr) = nested_restored {
        assert_eq!(arr.len(), 4);
        assert!(arr[0].as_string().is_some());
        assert!(arr[1].as_number().is_some());
        assert!(arr[2].as_array().is_some());
        assert!(arr[3].as_vector().is_some());
    } else {
        panic!("Nested array deserialization failed");
    }

    // Test 4: EntityMeta with Instant (special handling)
    let meta = EntityMeta {
        type_id: 5,
        embedding_offset: 1000,
        property_offset: 2000,
        degree: 15,
        last_accessed: Instant::now(),
    };
    
    // Note: Instant doesn't serialize by default, so we test with skip
    let meta_json = serde_json::to_string(&meta).unwrap();
    let meta_restored: EntityMeta = serde_json::from_str(&meta_json).unwrap();
    
    assert_eq!(meta.type_id, meta_restored.type_id);
    assert_eq!(meta.embedding_offset, meta_restored.embedding_offset);
    assert_eq!(meta.property_offset, meta_restored.property_offset);
    assert_eq!(meta.degree, meta_restored.degree);
}

#[tokio::test]
async fn test_cross_component_type_validation() {
    // Test type validation across different components
    
    // Test 1: Weight validation
    let valid_weights = vec![0.0, 0.5, 1.0];
    let invalid_weights = vec![-0.1, 1.1, f32::NAN, f32::INFINITY];
    
    for w in valid_weights {
        let weight = Weight::new(w);
        assert!(weight.is_ok(), "Weight {} should be valid", w);
        assert_eq!(weight.unwrap().value(), w);
    }
    
    for w in invalid_weights {
        let weight = Weight::new(w);
        assert!(weight.is_err(), "Weight {} should be invalid", w);
        
        if let Err(GraphError::InvalidWeight(val)) = weight {
            if w.is_nan() {
                assert!(val.is_nan());
            } else {
                assert_eq!(val, w);
            }
        } else {
            panic!("Expected InvalidWeight error");
        }
    }

    // Test 2: RelationshipType validation across components
    let rel_types = vec![
        (RelationshipType::Directed, false, false),
        (RelationshipType::Undirected, true, false),
        (RelationshipType::Weighted, false, true),
    ];
    
    for (rel_type, expected_bidirectional, expected_weighted) in rel_types {
        assert_eq!(rel_type.is_bidirectional(), expected_bidirectional);
        assert_eq!(rel_type.supports_weights(), expected_weighted);
        
        // Test Display implementation
        let display_str = format!("{}", rel_type);
        assert!(!display_str.is_empty());
    }

    // Test 3: GraphQuery validation
    let query = GraphQuery {
        query_text: "Find similar entities to 'machine learning'".to_string(),
        query_type: "similarity".to_string(),
        max_results: 10,
    };
    
    // Serialize and deserialize to ensure it works across boundaries
    let query_json = serde_json::to_string(&query).unwrap();
    let query_restored: GraphQuery = serde_json::from_str(&query_json).unwrap();
    
    assert_eq!(query.query_text, query_restored.query_text);
    assert_eq!(query.query_type, query_restored.query_type);
    assert_eq!(query.max_results, query_restored.max_results);

    // Test 4: TraversalParams validation
    let traversal = TraversalParams {
        max_depth: 5,
        max_paths: 100,
        include_bidirectional: true,
        edge_weight_threshold: Some(0.5),
    };
    
    let traversal_json = serde_json::to_string(&traversal).unwrap();
    let traversal_restored: TraversalParams = serde_json::from_str(&traversal_json).unwrap();
    
    assert_eq!(traversal.max_depth, traversal_restored.max_depth);
    assert_eq!(traversal.edge_weight_threshold, traversal_restored.edge_weight_threshold);
}

#[tokio::test]
async fn test_type_integration_with_knowledge_engine() {
    // Test complete integration of types with KnowledgeEngine
    let temp_dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(tokio::sync::RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));

    // Create entities with different attribute types
    let mut properties = HashMap::new();
    properties.insert("name".to_string(), AttributeValue::String("Test Entity".to_string()));
    properties.insert("value".to_string(), AttributeValue::Number(42.0));
    properties.insert("active".to_string(), AttributeValue::Boolean(true));
    properties.insert("tags".to_string(), AttributeValue::Array(vec![
        AttributeValue::String("tag1".to_string()),
        AttributeValue::String("tag2".to_string()),
    ]));
    properties.insert("metadata".to_string(), AttributeValue::Object({
        let mut meta = HashMap::new();
        meta.insert("created".to_string(), AttributeValue::String("2024-01-01".to_string()));
        meta.insert("version".to_string(), AttributeValue::Number(1.0));
        meta
    }));
    properties.insert("embedding".to_string(), AttributeValue::Vector(vec![0.1, 0.2, 0.3]));
    
    let entity_data = EntityData::new(1, serde_json::to_string(&properties).unwrap(), vec![0.1; 96]);

    // Add entity
    let entity_key = {
        let mut engine_write = engine.write().await;
        // Convert AttributeValue properties to String properties for store_entity
        let string_properties: HashMap<String, String> = properties.iter()
            .map(|(k, v)| (k.clone(), serde_json::to_string(v).unwrap()))
            .collect();
        engine_write.store_entity(
            "Test Entity".to_string(),
            "test".to_string(),
            "test description with attributes".to_string(),
            string_properties
        ).unwrap();
        EntityKey::from_raw_parts(1, 0)
    };

    // Verify entity was stored
    {
        let engine_read = engine.read().await;
        // For now, just verify the entity count increased
        assert!(engine_read.get_entity_count() > 0);
        
        // Note: The KnowledgeEngine doesn't have a direct get_entity method anymore
        // It uses query_triples or semantic_search for retrieval
    }

    // Test relationships with weights
    let entity_key2 = {
        let mut engine_write = engine.write().await;
        engine_write.store_entity(
            "Second Entity".to_string(),
            "test".to_string(),
            "second entity for relationship test".to_string(),
            HashMap::new()
        ).unwrap();
        EntityKey::from_raw_parts(2, 0)
    };

    // Note: KnowledgeEngine doesn't handle relationships directly
    // This would normally be done through a KnowledgeGraph instance
    // For this test, we'll just create the relationship object without storing it

    // Query with context
    let query_result = {
        let engine_read = engine.read().await;
        let similar = engine_read.semantic_search("similar entities test", 10).unwrap();
        
        QueryResult {
            entities: similar.nodes.iter().map(|node| {
                ContextEntity {
                    id: EntityKey::from_raw_parts(node.id.parse::<u64>().unwrap_or(0), 0),
                    similarity: 0.85, // Placeholder similarity
                    neighbors: vec![],
                    properties: format!("Node ID: {}", node.id),
                }
            }).collect(),
            relationships: vec![
                Relationship {
                    from: entity_key,
                    to: entity_key2,
                    rel_type: 1,
                    weight: 0.85,
                }
            ],
            confidence: 0.9,
            query_time_ms: similar.query_time_ms,
        }
    };

    assert!(!query_result.entities.is_empty());
    assert_eq!(query_result.relationships.len(), 1);
    assert_eq!(query_result.relationships[0].weight, 0.85);
}

#[tokio::test]
async fn test_type_boundary_conditions() {
    // Test edge cases and boundary conditions for types
    
    // Test 1: Empty collections
    let empty_array = AttributeValue::Array(vec![]);
    let empty_object = AttributeValue::Object(HashMap::new());
    let empty_vector = AttributeValue::Vector(vec![]);
    
    assert!(empty_array.as_array().unwrap().is_empty());
    assert!(empty_object.as_object().unwrap().is_empty());
    assert!(empty_vector.as_vector().unwrap().is_empty());

    // Test 2: Maximum values
    let max_entity_meta = EntityMeta {
        type_id: u16::MAX,
        embedding_offset: u32::MAX,
        property_offset: u32::MAX,
        degree: u16::MAX,
        last_accessed: Instant::now(),
    };
    
    assert_eq!(max_entity_meta.type_id, u16::MAX);
    assert_eq!(max_entity_meta.embedding_offset, u32::MAX);

    // Test 3: Special float values in vectors
    let special_vector = AttributeValue::Vector(vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN,
        f32::MAX,
        f32::EPSILON,
    ]);
    
    let vec_data = special_vector.as_vector().unwrap();
    assert_eq!(vec_data.len(), 7);
    assert!(vec_data.iter().all(|v| v.is_finite()));

    // Test 4: Large strings in AttributeValue
    let large_string = "x".repeat(10000);
    let large_attr = AttributeValue::String(large_string.clone());
    
    assert_eq!(large_attr.as_string().unwrap().len(), 10000);
    
    // Serialize and deserialize large string
    let json = serde_json::to_string(&large_attr).unwrap();
    let restored: AttributeValue = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.as_string().unwrap(), &large_string);

    // Test 5: EntityKey display
    let key = EntityKey::default();
    let display_str = format!("{}", key);
    assert!(!display_str.is_empty());
    
    // Test 6: Weight normalization edge cases
    let zero_weights = vec![Weight::new(0.0).unwrap(), Weight::new(0.0).unwrap()];
    let normalized = Weight::normalize(&zero_weights);
    assert_eq!(normalized.len(), 2);
    assert_eq!(normalized[0].value(), 0.0);
    assert_eq!(normalized[1].value(), 0.0);

    // Test single weight normalization
    let single_weight = vec![Weight::new(0.5).unwrap()];
    let normalized_single = Weight::normalize(&single_weight);
    assert_eq!(normalized_single.len(), 1);
    assert_eq!(normalized_single[0].value(), 1.0); // Should normalize to 1.0
}

#[tokio::test]
async fn test_type_performance_characteristics() {
    use std::time::Instant;
    
    // Test performance characteristics of different types
    
    // Test 1: AttributeValue construction performance
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = AttributeValue::String("test".to_string());
        let _ = AttributeValue::Number(42.0);
        let _ = AttributeValue::Boolean(true);
        let _ = AttributeValue::Null;
    }
    let attr_time = start.elapsed();
    println!("1000 AttributeValue constructions: {:?}", attr_time);
    assert!(attr_time.as_millis() < 100, "AttributeValue construction too slow");

    // Test 2: Large object serialization
    let mut large_object = HashMap::new();
    for i in 0..1000 {
        large_object.insert(
            format!("key_{}", i),
            AttributeValue::Number(i as f64)
        );
    }
    let large_attr = AttributeValue::Object(large_object);
    
    let start = Instant::now();
    let json = serde_json::to_string(&large_attr).unwrap();
    let serialize_time = start.elapsed();
    
    let start = Instant::now();
    let _: AttributeValue = serde_json::from_str(&json).unwrap();
    let deserialize_time = start.elapsed();
    
    println!("Large object (1000 keys) serialize: {:?}, deserialize: {:?}", 
             serialize_time, deserialize_time);
    
    // Test 3: EntityData with large embeddings
    let large_embedding = vec![0.1; 96];
    let entity = EntityData::new(1, "test".to_string(), large_embedding);
    
    let start = Instant::now();
    for _ in 0..100 {
        let _ = entity.clone();
    }
    let clone_time = start.elapsed();
    println!("100 EntityData clones (96D embedding): {:?}", clone_time);

    // Test 4: Weight operations performance
    let weights: Vec<Weight> = (0..100)
        .map(|i| Weight::new((i as f32) / 100.0).unwrap())
        .collect();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = Weight::normalize(&weights);
    }
    let normalize_time = start.elapsed();
    println!("1000 weight normalizations (100 weights): {:?}", normalize_time);
}

#[tokio::test]
async fn test_error_propagation_across_types() {
    // Test how errors propagate through type system
    
    // Test 1: Invalid weight in relationship
    let result = Weight::new(-0.5);
    assert!(result.is_err());
    
    match result {
        Err(GraphError::InvalidWeight(val)) => {
            assert_eq!(val, -0.5);
        },
        _ => panic!("Expected InvalidWeight error"),
    }

    // Test 2: Type conversion errors
    let string_attr = AttributeValue::String("not a number".to_string());
    assert!(string_attr.as_number().is_none());
    assert!(string_attr.as_boolean().is_none());
    assert!(string_attr.as_array().is_none());
    
    // Test 3: JSON parsing errors with EntityData
    let invalid_json = r#"{"invalid": json"#;
    let entity = EntityData::new(1, invalid_json.to_string(), vec![0.1]);
    
    // Attempting to parse properties should fail
    let parse_result: std::result::Result<HashMap<String, AttributeValue>, serde_json::Error> = 
        serde_json::from_str(&entity.properties);
    assert!(parse_result.is_err());

    // Test 4: Boundary validation
    let weights = vec![
        Weight::new(0.3).unwrap(),
        Weight::new(0.4).unwrap(),
        Weight::new(0.5).unwrap(), // Sum > 1.0
    ];
    
    // Normalization should handle this correctly
    let normalized = Weight::normalize(&weights);
    let sum: f32 = normalized.iter().map(|w| w.value()).sum();
    assert!((sum - 1.0).abs() < f32::EPSILON, "Normalized weights should sum to 1.0");
}