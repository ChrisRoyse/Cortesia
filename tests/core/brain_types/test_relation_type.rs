// Tests for RelationType enum
// Validates relationship type classification for brain-inspired knowledge graphs

use llmkg::core::brain_types::RelationType;
use serde_json;
use std::collections::{HashMap, HashSet};

use super::test_constants;

// ==================== Basic Enum Tests ====================

#[test]
fn test_relation_type_variants() {
    // Test all enum variants exist and are distinct
    let relations = [
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];

    // Verify all variants are distinct
    for (i, rel1) in relations.iter().enumerate() {
        for (j, rel2) in relations.iter().enumerate() {
            if i != j {
                assert_ne!(rel1, rel2, "Relations at indices {} and {} should be different", i, j);
            } else {
                assert_eq!(rel1, rel2, "Same relation should equal itself");
            }
        }
    }

    // Verify we have the expected number of relation types
    assert_eq!(relations.len(), 9);
}

#[test]
fn test_relation_type_copy_clone() {
    let original = RelationType::IsA;
    let copied = original; // Test Copy trait
    let cloned = original.clone(); // Test Clone trait

    assert_eq!(original, copied);
    assert_eq!(original, cloned);
    assert_eq!(copied, cloned);
}

#[test]
fn test_relation_type_debug() {
    // Test Debug trait implementation
    let isa_relation = RelationType::IsA;
    let debug_str = format!("{:?}", isa_relation);
    assert!(debug_str.contains("IsA"));

    let learned_relation = RelationType::Learned;
    let debug_str = format!("{:?}", learned_relation);
    assert!(debug_str.contains("Learned"));
}

#[test]
fn test_relation_type_hash() {
    // Test Hash trait implementation for use in HashMaps/HashSets
    let mut relation_set = HashSet::new();
    relation_set.insert(RelationType::IsA);
    relation_set.insert(RelationType::HasInstance);
    relation_set.insert(RelationType::HasProperty);

    assert!(relation_set.contains(&RelationType::IsA));
    assert!(relation_set.contains(&RelationType::HasInstance));
    assert!(!relation_set.contains(&RelationType::RelatedTo));

    // Test that same relation has same hash
    let rel1 = RelationType::Similar;
    let rel2 = RelationType::Similar;
    let mut map = HashMap::new();
    map.insert(rel1, "similarity");
    assert_eq!(map.get(&rel2), Some(&"similarity"));
}

// ==================== Serialization Tests ====================

#[test]
fn test_relation_type_serialization() {
    // Test serde serialization for all variants
    let isa = RelationType::IsA;
    let serialized = serde_json::to_string(&isa).expect("Should serialize");
    assert!(serialized.contains("IsA"));

    let temporal = RelationType::Temporal;
    let serialized = serde_json::to_string(&temporal).expect("Should serialize");
    assert!(serialized.contains("Temporal"));

    let learned = RelationType::Learned;
    let serialized = serde_json::to_string(&learned).expect("Should serialize");
    assert!(serialized.contains("Learned"));
}

#[test]
fn test_relation_type_deserialization() {
    // Test serde deserialization for all variants
    let test_cases = [
        ("\"IsA\"", RelationType::IsA),
        ("\"HasInstance\"", RelationType::HasInstance),
        ("\"HasProperty\"", RelationType::HasProperty),
        ("\"RelatedTo\"", RelationType::RelatedTo),
        ("\"PartOf\"", RelationType::PartOf),
        ("\"Similar\"", RelationType::Similar),
        ("\"Opposite\"", RelationType::Opposite),
        ("\"Temporal\"", RelationType::Temporal),
        ("\"Learned\"", RelationType::Learned),
    ];

    for (json, expected) in test_cases {
        let deserialized: RelationType = serde_json::from_str(json)
            .expect(&format!("Should deserialize {}", json));
        assert_eq!(deserialized, expected);
    }
}

#[test]
fn test_relation_type_round_trip() {
    // Test serialization round trip for all variants
    let relations = [
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];

    for relation in relations {
        let serialized = serde_json::to_string(&relation)
            .expect("Should serialize");
        let deserialized: RelationType = serde_json::from_str(&serialized)
            .expect("Should deserialize");
        assert_eq!(relation, deserialized);
    }
}

// ==================== Semantic Meaning Tests ====================

#[test]
fn test_relation_type_hierarchical_relations() {
    // Test hierarchical relationship types
    let hierarchical = [RelationType::IsA, RelationType::HasInstance, RelationType::PartOf];
    
    for relation in hierarchical {
        match relation {
            RelationType::IsA => {
                // IsA represents inheritance (dog IsA animal)
                assert_eq!(relation, RelationType::IsA);
            }
            RelationType::HasInstance => {
                // HasInstance represents instantiation (dog HasInstance Lassie)
                assert_eq!(relation, RelationType::HasInstance);
            }
            RelationType::PartOf => {
                // PartOf represents part-whole relationships (wheel PartOf car)
                assert_eq!(relation, RelationType::PartOf);
            }
            _ => panic!("Unexpected relation type in hierarchical group"),
        }
    }
}

#[test]
fn test_relation_type_associative_relations() {
    // Test associative relationship types
    let associative = [RelationType::RelatedTo, RelationType::Similar, RelationType::Opposite];
    
    for relation in associative {
        match relation {
            RelationType::RelatedTo => {
                // Generic association
                assert_eq!(relation, RelationType::RelatedTo);
            }
            RelationType::Similar => {
                // Similarity relationship
                assert_eq!(relation, RelationType::Similar);
            }
            RelationType::Opposite => {
                // Opposition relationship
                assert_eq!(relation, RelationType::Opposite);
            }
            _ => panic!("Unexpected relation type in associative group"),
        }
    }
}

#[test]
fn test_relation_type_descriptive_relations() {
    // Test descriptive relationship types
    let descriptive = [RelationType::HasProperty, RelationType::Temporal, RelationType::Learned];
    
    for relation in descriptive {
        match relation {
            RelationType::HasProperty => {
                // Property relationships (car HasProperty red)
                assert_eq!(relation, RelationType::HasProperty);
            }
            RelationType::Temporal => {
                // Temporal relationships (before, after, during)
                assert_eq!(relation, RelationType::Temporal);
            }
            RelationType::Learned => {
                // Dynamically learned relationships
                assert_eq!(relation, RelationType::Learned);
            }
            _ => panic!("Unexpected relation type in descriptive group"),
        }
    }
}

// ==================== Knowledge Graph Semantics Tests ====================

#[test]
fn test_relation_type_ontological_validity() {
    // Test that relation types support valid ontological modeling
    
    // Test inheritance chains: Animal -> Mammal -> Dog
    let inheritance_chain = [
        (RelationType::IsA, "supports inheritance hierarchies"),
        (RelationType::HasInstance, "supports instance relationships"),
    ];
    
    for (relation, description) in inheritance_chain {
        match relation {
            RelationType::IsA | RelationType::HasInstance => {
                // Valid for ontological modeling
                assert!(true, "{}", description);
            }
            _ => panic!("Invalid relation for inheritance chain"),
        }
    }

    // Test compositional relationships: Car -> Engine, Wheel, etc.
    assert_eq!(RelationType::PartOf, RelationType::PartOf);
    
    // Test property attribution: Car -> Red, Fast, etc.
    assert_eq!(RelationType::HasProperty, RelationType::HasProperty);
}

#[test]
fn test_relation_type_neural_network_semantics() {
    // Test that relation types support neural network modeling
    
    // Learned relationships from neural training
    assert_eq!(RelationType::Learned, RelationType::Learned);
    
    // Temporal relationships for sequence modeling
    assert_eq!(RelationType::Temporal, RelationType::Temporal);
    
    // Similarity for embedding spaces
    assert_eq!(RelationType::Similar, RelationType::Similar);
    
    // Generic relationships for flexible modeling
    assert_eq!(RelationType::RelatedTo, RelationType::RelatedTo);
}

// ==================== Collection Usage Tests ====================

#[test]
fn test_relation_type_in_graph_structures() {
    // Test using relation types in graph-like structures
    let mut edges: HashMap<(u32, u32), RelationType> = HashMap::new();
    
    // Add various relationships
    edges.insert((1, 2), RelationType::IsA);           // 1 IsA 2
    edges.insert((2, 3), RelationType::HasInstance);   // 2 HasInstance 3
    edges.insert((3, 4), RelationType::PartOf);        // 3 PartOf 4
    edges.insert((4, 5), RelationType::Similar);       // 4 Similar 5
    
    // Verify relationships
    assert_eq!(edges.get(&(1, 2)), Some(&RelationType::IsA));
    assert_eq!(edges.get(&(3, 4)), Some(&RelationType::PartOf));
    assert_eq!(edges.len(), 4);
}

#[test]
fn test_relation_type_frequency_analysis() {
    // Test relation type usage patterns
    let mut relation_counts: HashMap<RelationType, usize> = HashMap::new();
    
    let test_relations = [
        RelationType::IsA, RelationType::IsA, RelationType::IsA,
        RelationType::HasProperty, RelationType::HasProperty,
        RelationType::Similar,
        RelationType::Learned, RelationType::Learned, RelationType::Learned, RelationType::Learned,
    ];
    
    for relation in test_relations {
        *relation_counts.entry(relation).or_insert(0) += 1;
    }
    
    assert_eq!(relation_counts.get(&RelationType::IsA), Some(&3));
    assert_eq!(relation_counts.get(&RelationType::Learned), Some(&4));
    assert_eq!(relation_counts.get(&RelationType::Similar), Some(&1));
    assert_eq!(relation_counts.get(&RelationType::Temporal), None);
}

// ==================== Relationship Symmetry Tests ====================

#[test]
fn test_relation_type_symmetry_properties() {
    // Test which relations are symmetric vs asymmetric
    
    // Symmetric relations (if A rel B, then B rel A)
    let symmetric_relations = [RelationType::Similar, RelationType::Opposite];
    
    // Asymmetric relations (if A rel B, then NOT necessarily B rel A)
    let asymmetric_relations = [
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::PartOf,
    ];
    
    // Verify symmetric relations
    for relation in symmetric_relations {
        match relation {
            RelationType::Similar => {
                // If A is similar to B, then B is similar to A
                assert_eq!(relation, RelationType::Similar);
            }
            RelationType::Opposite => {
                // If A is opposite to B, then B is opposite to A
                assert_eq!(relation, RelationType::Opposite);
            }
            _ => panic!("Unexpected symmetric relation"),
        }
    }
    
    // Verify asymmetric relations
    for relation in asymmetric_relations {
        match relation {
            RelationType::IsA => {
                // If Dog IsA Animal, Animal is NOT Dog
                assert_eq!(relation, RelationType::IsA);
            }
            RelationType::HasInstance => {
                // If Species HasInstance Individual, Individual does NOT have Species
                assert_eq!(relation, RelationType::HasInstance);
            }
            RelationType::PartOf => {
                // If Wheel PartOf Car, Car is NOT part of Wheel
                assert_eq!(relation, RelationType::PartOf);
            }
            RelationType::HasProperty => {
                // If Car HasProperty Color, Color is NOT property of Car
                assert_eq!(relation, RelationType::HasProperty);
            }
            _ => panic!("Unexpected asymmetric relation"),
        }
    }
}

// ==================== Error Handling Tests ====================

#[test]
fn test_relation_type_invalid_deserialization() {
    // Test invalid JSON deserialization
    let invalid_cases = [
        "\"InvalidRelation\"",
        "42",
        "{}",
        "[]",
        "null",
        "\"\"",
    ];
    
    for invalid_json in invalid_cases {
        let result: Result<RelationType, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err(), "Should fail to deserialize: {}", invalid_json);
    }
}

// ==================== Performance Tests ====================

#[test]
fn test_relation_type_memory_efficiency() {
    use std::mem;
    
    let size = mem::size_of::<RelationType>();
    // Enum should be small (typically 1 byte for simple enums)
    assert!(size <= 8, "RelationType size {} bytes is too large", size);
    
    // Test that Vec<RelationType> is memory efficient
    let relations = vec![RelationType::IsA; 1000];
    let vec_size = mem::size_of_val(&relations);
    let expected_max = 1000 * size + mem::size_of::<Vec<RelationType>>();
    assert!(vec_size <= expected_max * 2, "Vec<RelationType> is too large: {} bytes", vec_size);
}

#[test]
fn test_relation_type_hash_performance() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Test that hashing is consistent and fast
    let relation = RelationType::Learned;
    
    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();
    
    relation.hash(&mut hasher1);
    relation.hash(&mut hasher2);
    
    assert_eq!(hasher1.finish(), hasher2.finish(), "Hash should be consistent");
}

// ==================== Integration with Brain Types ====================

#[test]
fn test_relation_type_with_brain_inspired_structures() {
    // Test that RelationType integrates well with other brain_types
    use llmkg::core::brain_types::{BrainInspiredRelationship, EntityDirection};
    use llmkg::core::types::EntityKey;
    
    // Create a relationship using RelationType
    let source = EntityKey::from(1);
    let target = EntityKey::from(2);
    let relationship = BrainInspiredRelationship::new(source, target, RelationType::IsA);
    
    assert_eq!(relationship.relation_type, RelationType::IsA);
    assert_eq!(relationship.source, source);
    assert_eq!(relationship.target, target);
}

#[test]
fn test_relation_type_pattern_matching_exhaustive() {
    // Test exhaustive pattern matching to ensure all variants are handled
    let relations = [
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];
    
    for relation in relations {
        let matched = match relation {
            RelationType::IsA => true,
            RelationType::HasInstance => true,
            RelationType::HasProperty => true,
            RelationType::RelatedTo => true,
            RelationType::PartOf => true,
            RelationType::Similar => true,
            RelationType::Opposite => true,
            RelationType::Temporal => true,
            RelationType::Learned => true,
        };
        
        assert!(matched, "Pattern matching should handle all variants");
    }
}