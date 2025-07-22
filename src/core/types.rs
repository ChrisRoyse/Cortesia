use serde::{Deserialize, Serialize};
use slotmap::new_key_type;
use std::collections::HashMap;

new_key_type! {
    pub struct EntityKey;
}

impl EntityKey {
    /// Create an EntityKey from raw parts (for test compatibility)
    pub fn from_raw_parts(id: u64, _version: u32) -> Self {
        // This is a hack for test compatibility. In real usage, EntityKeys
        // should only be created through SlotMap::insert
        unsafe {
            std::mem::transmute::<u64, EntityKey>(id)
        }
    }
}

impl std::fmt::Display for EntityKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub type TypeId = u16;
pub type EmbeddingSize = u16;
pub type EdgeWeight = f32;

/// Attribute value types for entity and relationship properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<AttributeValue>),
    Object(HashMap<String, AttributeValue>),
    Vector(Vec<f32>),
    Null,
}

impl AttributeValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_number(&self) -> Option<f64> {
        match self {
            AttributeValue::Number(n) => Some(*n),
            _ => None,
        }
    }
    
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            AttributeValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    pub fn as_array(&self) -> Option<&Vec<AttributeValue>> {
        match self {
            AttributeValue::Array(a) => Some(a),
            _ => None,
        }
    }
    
    pub fn as_object(&self) -> Option<&HashMap<String, AttributeValue>> {
        match self {
            AttributeValue::Object(o) => Some(o),
            _ => None,
        }
    }
    
    pub fn as_vector(&self) -> Option<&Vec<f32>> {
        match self {
            AttributeValue::Vector(v) => Some(v),
            _ => None,
        }
    }
    
    pub fn is_null(&self) -> bool {
        matches!(self, AttributeValue::Null)
    }
}

/// Types of relationships in the knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    Directed,
    Undirected,
    Weighted,
}

impl RelationshipType {
    pub fn is_bidirectional(&self) -> bool {
        matches!(self, RelationshipType::Undirected)
    }
    
    pub fn supports_weights(&self) -> bool {
        matches!(self, RelationshipType::Weighted)
    }
}

impl std::fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationshipType::Directed => write!(f, "directed"),
            RelationshipType::Undirected => write!(f, "undirected"),
            RelationshipType::Weighted => write!(f, "weighted"),
        }
    }
}

/// Weight value with validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Weight(f32);

impl Weight {
    pub fn new(value: f32) -> crate::error::Result<Self> {
        if value < 0.0 || value > 1.0 || value.is_nan() || value.is_infinite() {
            return Err(crate::error::GraphError::InvalidWeight(value));
        }
        Ok(Weight(value))
    }
    
    pub fn value(&self) -> f32 {
        self.0
    }
    
    pub fn normalize(weights: &[Weight]) -> Vec<Weight> {
        let sum: f32 = weights.iter().map(|w| w.0).sum();
        if sum > 0.0 {
            weights.iter().map(|w| Weight(w.0 / sum)).collect()
        } else {
            weights.to_vec()
        }
    }
}

impl std::ops::Add for Weight {
    type Output = Weight;
    
    fn add(self, other: Weight) -> Weight {
        Weight((self.0 + other.0).min(1.0))
    }
}

impl std::ops::Mul for Weight {
    type Output = Weight;
    
    fn mul(self, other: Weight) -> Weight {
        Weight(self.0 * other.0)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct EntityMeta {
    pub type_id: u16,
    pub embedding_offset: u32,
    pub property_offset: u32,
    pub degree: u16,
    #[serde(skip, default = "std::time::Instant::now")]
    pub last_accessed: std::time::Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityData {
    pub type_id: u16,
    pub properties: String,
    pub embedding: Vec<f32>,
}

impl EntityData {
    /// Create a new EntityData instance
    pub fn new(type_id: u16, properties: String, embedding: Vec<f32>) -> Self {
        Self {
            type_id,
            properties,
            embedding,
        }
    }
}

/// Entity type that combines EntityKey, EntityData, and additional metadata
#[derive(Clone, Debug)]
pub struct Entity {
    pub id: u32,
    pub key: EntityKey,
    pub data: EntityData,
    pub activation: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Relationship {
    pub from: EntityKey,
    pub to: EntityKey,
    pub rel_type: u8,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextEntity {
    pub id: EntityKey,
    pub similarity: f32,
    pub neighbors: Vec<EntityKey>,
    pub properties: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub entities: Vec<ContextEntity>,
    pub relationships: Vec<Relationship>,
    pub confidence: f32,
    pub query_time_ms: u64,
}

// Ultra-fast optimized representations for maximum performance
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct CompactEntity {
    pub id: u32,
    pub type_id: u16,
    pub degree: u16,
    pub embedding_offset: u32,
    pub property_offset: u32,
}

// For zero-copy neighbor access
#[derive(Debug, Clone)]
pub struct NeighborSlice {
    pub data: *const u32,
    pub len: u16,
}

// SIMD-friendly relationship representation for batched operations
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct SIMDRelationship {
    pub from: [u32; 8],
    pub to: [u32; 8],
    pub rel_type: [u8; 8],
    pub weight: [f32; 8],
    pub count: u8, // How many relationships are valid in this SIMD block
}

// Zero-copy query parameters for maximum performance
#[derive(Debug)]
pub struct QueryParams {
    pub embedding: *const f32,
    pub embedding_dim: u16,
    pub max_entities: u16,
    pub max_depth: u8,
    pub similarity_threshold: f32,
}

impl CompactEntity {
    pub fn from_meta_and_id(id: u32, meta: &EntityMeta) -> Self {
        Self {
            id,
            type_id: meta.type_id,
            degree: meta.degree,
            embedding_offset: meta.embedding_offset,
            property_offset: meta.property_offset,
        }
    }
}

// Unsafe but ultra-fast neighbor access
impl NeighborSlice {
    pub unsafe fn new(ptr: *const u32, len: u16) -> Self {
        Self { data: ptr, len }
    }
    
    pub fn as_slice(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.data, self.len as usize) }
    }
    
    pub fn len(&self) -> usize {
        self.len as usize
    }
}

impl QueryParams {
    pub unsafe fn new(embedding: &[f32], max_entities: usize, max_depth: u8) -> Self {
        Self {
            embedding: embedding.as_ptr(),
            embedding_dim: embedding.len() as u16,
            max_entities: max_entities as u16,
            max_depth,
            similarity_threshold: 0.0,
        }
    }
    
    pub unsafe fn embedding_slice(&self) -> &[f32] {
        std::slice::from_raw_parts(self.embedding, self.embedding_dim as usize)
    }
}

// Graph query types for GraphQueryEngine
#[derive(Debug, Clone)]
pub struct GraphQuery {
    pub query_text: String,
    pub query_type: String,
    pub max_results: usize,
}

#[derive(Debug, Clone)]
pub struct TraversalParams {
    pub max_depth: usize,
    pub max_paths: usize,
    pub include_bidirectional: bool,
    pub edge_weight_threshold: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_entity_key_display() {
        // EntityKey is a slotmap key type, so we can't construct it directly
        // but we can test the Display implementation through debug formatting
        let key = EntityKey::default();
        let display_str = format!("{}", key);
        let debug_str = format!("{:?}", key);
        assert_eq!(display_str, debug_str);
    }

    mod attribute_value_tests {
        use super::*;

        #[test]
        fn test_as_string_methods() {
            let string_val = AttributeValue::String("test".to_string());
            let number_val = AttributeValue::Number(42.0);
            let null_val = AttributeValue::Null;

            assert_eq!(string_val.as_string(), Some("test"));
            assert_eq!(number_val.as_string(), None);
            assert_eq!(null_val.as_string(), None);
        }

        #[test]
        fn test_as_string_with_empty_string() {
            let empty_string = AttributeValue::String("".to_string());
            assert_eq!(empty_string.as_string(), Some(""));
        }

        #[test]
        fn test_as_number_methods() {
            let number_val = AttributeValue::Number(42.5);
            let string_val = AttributeValue::String("test".to_string());
            let null_val = AttributeValue::Null;

            assert_eq!(number_val.as_number(), Some(42.5));
            assert_eq!(string_val.as_number(), None);
            assert_eq!(null_val.as_number(), None);
        }

        #[test]
        fn test_as_number_with_special_values() {
            let nan_val = AttributeValue::Number(f64::NAN);
            let inf_val = AttributeValue::Number(f64::INFINITY);
            let neg_inf_val = AttributeValue::Number(f64::NEG_INFINITY);

            assert!(nan_val.as_number().unwrap().is_nan());
            assert_eq!(inf_val.as_number(), Some(f64::INFINITY));
            assert_eq!(neg_inf_val.as_number(), Some(f64::NEG_INFINITY));
        }

        #[test]
        fn test_as_boolean_methods() {
            let bool_true = AttributeValue::Boolean(true);
            let bool_false = AttributeValue::Boolean(false);
            let string_val = AttributeValue::String("true".to_string());
            let null_val = AttributeValue::Null;

            assert_eq!(bool_true.as_boolean(), Some(true));
            assert_eq!(bool_false.as_boolean(), Some(false));
            assert_eq!(string_val.as_boolean(), None);
            assert_eq!(null_val.as_boolean(), None);
        }

        #[test]
        fn test_as_array_methods() {
            let array_val = AttributeValue::Array(vec![
                AttributeValue::String("test".to_string()),
                AttributeValue::Number(42.0),
            ]);
            let empty_array = AttributeValue::Array(vec![]);
            let string_val = AttributeValue::String("test".to_string());
            let null_val = AttributeValue::Null;

            assert_eq!(array_val.as_array().unwrap().len(), 2);
            assert_eq!(empty_array.as_array().unwrap().len(), 0);
            assert_eq!(string_val.as_array(), None);
            assert_eq!(null_val.as_array(), None);
        }

        #[test]
        fn test_as_object_methods() {
            let mut obj = HashMap::new();
            obj.insert("key1".to_string(), AttributeValue::String("value1".to_string()));
            obj.insert("key2".to_string(), AttributeValue::Number(42.0));
            
            let object_val = AttributeValue::Object(obj);
            let empty_object = AttributeValue::Object(HashMap::new());
            let string_val = AttributeValue::String("test".to_string());
            let null_val = AttributeValue::Null;

            assert_eq!(object_val.as_object().unwrap().len(), 2);
            assert_eq!(empty_object.as_object().unwrap().len(), 0);
            assert_eq!(string_val.as_object(), None);
            assert_eq!(null_val.as_object(), None);
        }

        #[test]
        fn test_as_vector_methods() {
            let vector_val = AttributeValue::Vector(vec![1.0, 2.0, 3.0]);
            let empty_vector = AttributeValue::Vector(vec![]);
            let string_val = AttributeValue::String("test".to_string());
            let null_val = AttributeValue::Null;

            assert_eq!(vector_val.as_vector().unwrap().len(), 3);
            assert_eq!(vector_val.as_vector().unwrap()[0], 1.0);
            assert_eq!(empty_vector.as_vector().unwrap().len(), 0);
            assert_eq!(string_val.as_vector(), None);
            assert_eq!(null_val.as_vector(), None);
        }

        #[test]
        fn test_is_null_method() {
            let null_val = AttributeValue::Null;
            let string_val = AttributeValue::String("test".to_string());
            let number_val = AttributeValue::Number(42.0);
            let bool_val = AttributeValue::Boolean(false);

            assert!(null_val.is_null());
            assert!(!string_val.is_null());
            assert!(!number_val.is_null());
            assert!(!bool_val.is_null());
        }

        #[test]
        fn test_attribute_value_clone_and_equality() {
            let original = AttributeValue::String("test".to_string());
            let cloned = original.clone();
            
            assert_eq!(original, cloned);
            assert_eq!(original.as_string(), cloned.as_string());
        }

        #[test]
        fn test_nested_attribute_values() {
            let nested_array = AttributeValue::Array(vec![
                AttributeValue::Array(vec![
                    AttributeValue::Number(1.0),
                    AttributeValue::Number(2.0),
                ]),
                AttributeValue::String("nested".to_string()),
            ]);

            let array = nested_array.as_array().unwrap();
            assert_eq!(array.len(), 2);
            assert!(array[0].as_array().is_some());
            assert_eq!(array[1].as_string(), Some("nested"));
        }
    }

    mod relationship_type_tests {
        use super::*;

        #[test]
        fn test_is_bidirectional() {
            assert!(RelationshipType::Undirected.is_bidirectional());
            assert!(!RelationshipType::Directed.is_bidirectional());
            assert!(!RelationshipType::Weighted.is_bidirectional());
        }

        #[test]
        fn test_supports_weights() {
            assert!(RelationshipType::Weighted.supports_weights());
            assert!(!RelationshipType::Directed.supports_weights());
            assert!(!RelationshipType::Undirected.supports_weights());
        }

        #[test]
        fn test_relationship_type_display() {
            assert_eq!(format!("{}", RelationshipType::Directed), "directed");
            assert_eq!(format!("{}", RelationshipType::Undirected), "undirected");
            assert_eq!(format!("{}", RelationshipType::Weighted), "weighted");
        }

        #[test]
        fn test_relationship_type_clone_and_copy() {
            let rel_type = RelationshipType::Directed;
            let copied = rel_type;
            let cloned = rel_type.clone();
            
            assert_eq!(rel_type, copied);
            assert_eq!(rel_type, cloned);
        }

        #[test]
        fn test_relationship_type_equality() {
            assert_eq!(RelationshipType::Directed, RelationshipType::Directed);
            assert_ne!(RelationshipType::Directed, RelationshipType::Undirected);
            assert_ne!(RelationshipType::Undirected, RelationshipType::Weighted);
        }
    }

    mod weight_tests {
        use super::*;

        #[test]
        fn test_valid_weight_creation() {
            assert!(Weight::new(0.0).is_ok());
            assert!(Weight::new(0.5).is_ok());
            assert!(Weight::new(1.0).is_ok());
            
            let weight = Weight::new(0.7).unwrap();
            assert_eq!(weight.value(), 0.7);
        }

        #[test]
        fn test_invalid_weight_negative() {
            assert!(Weight::new(-0.1).is_err());
            assert!(Weight::new(-1.0).is_err());
            
            match Weight::new(-0.5) {
                Err(crate::error::GraphError::InvalidWeight(val)) => assert_eq!(val, -0.5),
                _ => panic!("Expected InvalidWeight error"),
            }
        }

        #[test]
        fn test_invalid_weight_greater_than_one() {
            assert!(Weight::new(1.1).is_err());
            assert!(Weight::new(2.0).is_err());
            
            match Weight::new(1.5) {
                Err(crate::error::GraphError::InvalidWeight(val)) => assert_eq!(val, 1.5),
                _ => panic!("Expected InvalidWeight error"),
            }
        }

        #[test]
        fn test_invalid_weight_nan() {
            assert!(Weight::new(f32::NAN).is_err());
            
            match Weight::new(f32::NAN) {
                Err(crate::error::GraphError::InvalidWeight(val)) => assert!(val.is_nan()),
                _ => panic!("Expected InvalidWeight error"),
            }
        }

        #[test]
        fn test_invalid_weight_infinite() {
            assert!(Weight::new(f32::INFINITY).is_err());
            assert!(Weight::new(f32::NEG_INFINITY).is_err());
            
            match Weight::new(f32::INFINITY) {
                Err(crate::error::GraphError::InvalidWeight(val)) => assert!(val.is_infinite()),
                _ => panic!("Expected InvalidWeight error"),
            }
        }

        #[test]
        fn test_weight_addition() {
            let w1 = Weight::new(0.3).unwrap();
            let w2 = Weight::new(0.4).unwrap();
            let result = w1 + w2;
            assert!((result.value() - 0.7).abs() < f32::EPSILON);
        }

        #[test]
        fn test_weight_addition_clamping() {
            let w1 = Weight::new(0.8).unwrap();
            let w2 = Weight::new(0.7).unwrap();
            let result = w1 + w2;
            assert_eq!(result.value(), 1.0); // Should clamp to 1.0
        }

        #[test]
        fn test_weight_multiplication() {
            let w1 = Weight::new(0.5).unwrap();
            let w2 = Weight::new(0.6).unwrap();
            let result = w1 * w2;
            assert!((result.value() - 0.3).abs() < f32::EPSILON);
        }

        #[test]
        fn test_weight_multiplication_edge_cases() {
            let w1 = Weight::new(0.0).unwrap();
            let w2 = Weight::new(0.5).unwrap();
            let result = w1 * w2;
            assert_eq!(result.value(), 0.0);

            let w3 = Weight::new(1.0).unwrap();
            let w4 = Weight::new(0.5).unwrap();
            let result2 = w3 * w4;
            assert_eq!(result2.value(), 0.5);
        }

        #[test]
        fn test_weight_normalize_empty() {
            let weights: Vec<Weight> = vec![];
            let normalized = Weight::normalize(&weights);
            assert!(normalized.is_empty());
        }

        #[test]
        fn test_weight_normalize_single() {
            let weights = vec![Weight::new(0.5).unwrap()];
            let normalized = Weight::normalize(&weights);
            assert_eq!(normalized.len(), 1);
            assert_eq!(normalized[0].value(), 1.0);
        }

        #[test]
        fn test_weight_normalize_multiple() {
            let weights = vec![
                Weight::new(0.2).unwrap(),
                Weight::new(0.3).unwrap(),
                Weight::new(0.5).unwrap(),
            ];
            let normalized = Weight::normalize(&weights);
            
            assert_eq!(normalized.len(), 3);
            assert_eq!(normalized[0].value(), 0.2);
            assert_eq!(normalized[1].value(), 0.3);
            assert_eq!(normalized[2].value(), 0.5);
        }

        #[test]
        fn test_weight_normalize_zero_sum() {
            let weights = vec![Weight::new(0.0).unwrap(), Weight::new(0.0).unwrap()];
            let normalized = Weight::normalize(&weights);
            
            assert_eq!(normalized.len(), 2);
            assert_eq!(normalized[0].value(), 0.0);
            assert_eq!(normalized[1].value(), 0.0);
        }

        #[test]
        fn test_weight_normalize_non_unit_sum() {
            let weights = vec![
                Weight::new(0.1).unwrap(),
                Weight::new(0.2).unwrap(),
            ];
            let normalized = Weight::normalize(&weights);
            
            assert_eq!(normalized.len(), 2);
            assert!((normalized[0].value() - (0.1 / 0.3)).abs() < f32::EPSILON);
            assert!((normalized[1].value() - (0.2 / 0.3)).abs() < f32::EPSILON);
        }

        #[test]
        fn test_weight_clone_and_copy() {
            let weight = Weight::new(0.5).unwrap();
            let copied = weight;
            let cloned = weight.clone();
            
            assert_eq!(weight.value(), copied.value());
            assert_eq!(weight.value(), cloned.value());
        }

        #[test]
        fn test_weight_equality() {
            let w1 = Weight::new(0.5).unwrap();
            let w2 = Weight::new(0.5).unwrap();
            let w3 = Weight::new(0.7).unwrap();
            
            assert_eq!(w1, w2);
            assert_ne!(w1, w3);
        }
    }

    mod entity_meta_tests {
        use super::*;

        #[test]
        fn test_entity_meta_creation() {
            let meta = EntityMeta {
                type_id: 1,
                embedding_offset: 100,
                property_offset: 200,
                degree: 5,
                last_accessed: std::time::Instant::now(),
            };

            assert_eq!(meta.type_id, 1);
            assert_eq!(meta.embedding_offset, 100);
            assert_eq!(meta.property_offset, 200);
            assert_eq!(meta.degree, 5);
        }

        #[test]
        fn test_entity_meta_clone() {
            let meta = EntityMeta {
                type_id: 1,
                embedding_offset: 100,
                property_offset: 200,
                degree: 5,
                last_accessed: std::time::Instant::now(),
            };

            let cloned = meta.clone();
            assert_eq!(meta.type_id, cloned.type_id);
            assert_eq!(meta.embedding_offset, cloned.embedding_offset);
            assert_eq!(meta.property_offset, cloned.property_offset);
            assert_eq!(meta.degree, cloned.degree);
        }

        #[test]
        fn test_entity_meta_copy() {
            let meta = EntityMeta {
                type_id: 1,
                embedding_offset: 100,
                property_offset: 200,
                degree: 5,
                last_accessed: std::time::Instant::now(),
            };

            let copied = meta;
            assert_eq!(meta.type_id, copied.type_id);
            assert_eq!(meta.embedding_offset, copied.embedding_offset);
        }

        #[test]
        fn test_entity_meta_max_values() {
            let meta = EntityMeta {
                type_id: u16::MAX,
                embedding_offset: u32::MAX,
                property_offset: u32::MAX,
                degree: u16::MAX,
                last_accessed: std::time::Instant::now(),
            };

            assert_eq!(meta.type_id, u16::MAX);
            assert_eq!(meta.embedding_offset, u32::MAX);
            assert_eq!(meta.property_offset, u32::MAX);
            assert_eq!(meta.degree, u16::MAX);
        }
    }

    mod entity_data_tests {
        use super::*;

        #[test]
        fn test_entity_data_creation() {
            let data = EntityData {
                type_id: 1,
                properties: "test properties".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
            };

            assert_eq!(data.type_id, 1);
            assert_eq!(data.properties, "test properties");
            assert_eq!(data.embedding.len(), 3);
            assert_eq!(data.embedding[0], 0.1);
        }

        #[test]
        fn test_entity_data_empty_properties() {
            let data = EntityData {
                type_id: 1,
                properties: String::new(),
                embedding: vec![],
            };

            assert_eq!(data.type_id, 1);
            assert!(data.properties.is_empty());
            assert!(data.embedding.is_empty());
        }

        #[test]
        fn test_entity_data_clone() {
            let data = EntityData {
                type_id: 1,
                properties: "test".to_string(),
                embedding: vec![0.1, 0.2],
            };

            let cloned = data.clone();
            assert_eq!(data.type_id, cloned.type_id);
            assert_eq!(data.properties, cloned.properties);
            assert_eq!(data.embedding, cloned.embedding);
        }

        #[test]
        fn test_entity_data_large_embedding() {
            let large_embedding: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
            let data = EntityData {
                type_id: 1,
                properties: "large embedding".to_string(),
                embedding: large_embedding.clone(),
            };

            assert_eq!(data.embedding.len(), 1000);
            // 999 * 0.001 = 0.999, accounting for floating point precision
            assert!((data.embedding[999] - 0.999).abs() < f32::EPSILON);
        }
    }

    mod compact_entity_tests {
        use super::*;

        #[test]
        fn test_compact_entity_from_meta_and_id() {
            let meta = EntityMeta {
                type_id: 42,
                embedding_offset: 1000,
                property_offset: 2000,
                degree: 10,
                last_accessed: std::time::Instant::now(),
            };

            let compact = CompactEntity::from_meta_and_id(123, &meta);

            // Copy packed fields to local variables to avoid unaligned reference errors
            let id = compact.id;
            let type_id = compact.type_id;
            let embedding_offset = compact.embedding_offset;
            let property_offset = compact.property_offset;
            let degree = compact.degree;

            assert_eq!(id, 123);
            assert_eq!(type_id, 42);
            assert_eq!(embedding_offset, 1000);
            assert_eq!(property_offset, 2000);
            assert_eq!(degree, 10);
        }

        #[test]
        fn test_compact_entity_memory_layout() {
            let compact = CompactEntity {
                id: 1,
                type_id: 2,
                degree: 3,
                embedding_offset: 4,
                property_offset: 5,
            };

            // Test that the struct is indeed compact and copyable
            let copied = compact;
            
            // Copy packed fields to local variables to avoid unaligned reference errors
            let id1 = compact.id;
            let id2 = copied.id;
            let type_id1 = compact.type_id;
            let type_id2 = copied.type_id;

            assert_eq!(id1, id2);
            assert_eq!(type_id1, type_id2);
        }

        #[test]
        fn test_compact_entity_max_values() {
            let compact = CompactEntity {
                id: u32::MAX,
                type_id: u16::MAX,
                degree: u16::MAX,
                embedding_offset: u32::MAX,
                property_offset: u32::MAX,
            };

            // Copy packed fields to local variables to avoid unaligned reference errors
            let id = compact.id;
            let type_id = compact.type_id;
            let degree = compact.degree;

            assert_eq!(id, u32::MAX);
            assert_eq!(type_id, u16::MAX);
            assert_eq!(degree, u16::MAX);
        }
    }

    mod neighbor_slice_tests {
        use super::*;

        #[test]
        fn test_neighbor_slice_creation_and_access() {
            let data = vec![1u32, 2u32, 3u32, 4u32];
            let slice = unsafe { NeighborSlice::new(data.as_ptr(), 4) };

            assert_eq!(slice.len(), 4);
            
            let slice_data = slice.as_slice();
            assert_eq!(slice_data.len(), 4);
            assert_eq!(slice_data[0], 1);
            assert_eq!(slice_data[3], 4);
        }

        #[test]
        fn test_neighbor_slice_empty() {
            let slice = unsafe { NeighborSlice::new(std::ptr::null(), 0) };
            assert_eq!(slice.len(), 0);
        }

        #[test]
        fn test_neighbor_slice_len_conversion() {
            let data = vec![1u32, 2u32];
            let slice = unsafe { NeighborSlice::new(data.as_ptr(), 2) };

            assert_eq!(slice.len(), 2);
            assert_eq!(slice.len as usize, 2);
        }

        #[test]
        fn test_neighbor_slice_max_len() {
            let data = vec![1u32; u16::MAX as usize];
            let slice = unsafe { NeighborSlice::new(data.as_ptr(), u16::MAX) };

            assert_eq!(slice.len(), u16::MAX as usize);
        }
    }

    mod query_params_tests {
        use super::*;

        #[test]
        fn test_query_params_creation() {
            let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
            let params = unsafe { QueryParams::new(&embedding, 100, 5) };

            assert_eq!(params.embedding_dim, 4);
            assert_eq!(params.max_entities, 100);
            assert_eq!(params.max_depth, 5);
            assert_eq!(params.similarity_threshold, 0.0);
        }

        #[test]
        fn test_query_params_embedding_slice() {
            let embedding = vec![0.1f32, 0.2, 0.3];
            let params = unsafe { QueryParams::new(&embedding, 50, 3) };

            let slice = unsafe { params.embedding_slice() };
            assert_eq!(slice.len(), 3);
            assert_eq!(slice[0], 0.1);
            assert_eq!(slice[2], 0.3);
        }

        #[test]
        fn test_query_params_empty_embedding() {
            let embedding: Vec<f32> = vec![];
            let params = unsafe { QueryParams::new(&embedding, 10, 1) };

            assert_eq!(params.embedding_dim, 0);
            assert_eq!(params.max_entities, 10);
            assert_eq!(params.max_depth, 1);
        }

        #[test]
        fn test_query_params_large_values() {
            let embedding = vec![1.0f32; 1000];
            let params = unsafe { QueryParams::new(&embedding, usize::MAX, u8::MAX) };

            assert_eq!(params.embedding_dim, 1000);
            assert_eq!(params.max_entities, u16::MAX); // Capped at u16::MAX
            assert_eq!(params.max_depth, u8::MAX);
        }

        #[test]
        fn test_query_params_max_entities_overflow() {
            let embedding = vec![0.5f32];
            let large_max_entities = (u16::MAX as usize) + 1;
            let params = unsafe { QueryParams::new(&embedding, large_max_entities, 1) };

            // Should wrap around due to casting to u16
            assert_eq!(params.max_entities, 0);
        }
    }

    mod graph_query_tests {
        use super::*;

        #[test]
        fn test_graph_query_creation() {
            let query = GraphQuery {
                query_text: "find similar entities".to_string(),
                query_type: "similarity".to_string(),
                max_results: 50,
            };

            assert_eq!(query.query_text, "find similar entities");
            assert_eq!(query.query_type, "similarity");
            assert_eq!(query.max_results, 50);
        }

        #[test]
        fn test_graph_query_clone() {
            let query = GraphQuery {
                query_text: "test query".to_string(),
                query_type: "test".to_string(),
                max_results: 10,
            };

            let cloned = query.clone();
            assert_eq!(query.query_text, cloned.query_text);
            assert_eq!(query.query_type, cloned.query_type);
            assert_eq!(query.max_results, cloned.max_results);
        }

        #[test]
        fn test_graph_query_empty_strings() {
            let query = GraphQuery {
                query_text: String::new(),
                query_type: String::new(),
                max_results: 0,
            };

            assert!(query.query_text.is_empty());
            assert!(query.query_type.is_empty());
            assert_eq!(query.max_results, 0);
        }
    }

    mod traversal_params_tests {
        use super::*;

        #[test]
        fn test_traversal_params_creation() {
            let params = TraversalParams {
                max_depth: 5,
                max_paths: 100,
                include_bidirectional: true,
                edge_weight_threshold: Some(0.5),
            };

            assert_eq!(params.max_depth, 5);
            assert_eq!(params.max_paths, 100);
            assert!(params.include_bidirectional);
            assert_eq!(params.edge_weight_threshold, Some(0.5));
        }

        #[test]
        fn test_traversal_params_no_threshold() {
            let params = TraversalParams {
                max_depth: 3,
                max_paths: 50,
                include_bidirectional: false,
                edge_weight_threshold: None,
            };

            assert_eq!(params.max_depth, 3);
            assert_eq!(params.max_paths, 50);
            assert!(!params.include_bidirectional);
            assert_eq!(params.edge_weight_threshold, None);
        }

        #[test]
        fn test_traversal_params_clone() {
            let params = TraversalParams {
                max_depth: 2,
                max_paths: 25,
                include_bidirectional: true,
                edge_weight_threshold: Some(0.8),
            };

            let cloned = params.clone();
            assert_eq!(params.max_depth, cloned.max_depth);
            assert_eq!(params.max_paths, cloned.max_paths);
            assert_eq!(params.include_bidirectional, cloned.include_bidirectional);
            assert_eq!(params.edge_weight_threshold, cloned.edge_weight_threshold);
        }

        #[test]
        fn test_traversal_params_extreme_values() {
            let params = TraversalParams {
                max_depth: 0,
                max_paths: usize::MAX,
                include_bidirectional: false,
                edge_weight_threshold: Some(0.0),
            };

            assert_eq!(params.max_depth, 0);
            assert_eq!(params.max_paths, usize::MAX);
            assert_eq!(params.edge_weight_threshold, Some(0.0));
        }
    }

    mod relationship_tests {
        use super::*;

        #[test]
        fn test_relationship_creation() {
            let from_key = EntityKey::default();
            let to_key = EntityKey::default();
            
            let relationship = Relationship {
                from: from_key,
                to: to_key,
                rel_type: 1,
                weight: 0.75,
            };

            assert_eq!(relationship.from, from_key);
            assert_eq!(relationship.to, to_key);
            assert_eq!(relationship.rel_type, 1);
            assert_eq!(relationship.weight, 0.75);
        }

        #[test]
        fn test_relationship_copy() {
            let relationship = Relationship {
                from: EntityKey::default(),
                to: EntityKey::default(),
                rel_type: 2,
                weight: 0.5,
            };

            let copied = relationship;
            assert_eq!(relationship.rel_type, copied.rel_type);
            assert_eq!(relationship.weight, copied.weight);
        }

        #[test]
        fn test_relationship_extreme_values() {
            let relationship = Relationship {
                from: EntityKey::default(),
                to: EntityKey::default(),
                rel_type: u8::MAX,
                weight: f32::MAX,
            };

            assert_eq!(relationship.rel_type, u8::MAX);
            assert_eq!(relationship.weight, f32::MAX);
        }
    }

    mod context_entity_tests {
        use super::*;

        #[test]
        fn test_context_entity_creation() {
            let entity = ContextEntity {
                id: EntityKey::default(),
                similarity: 0.85,
                neighbors: vec![EntityKey::default(), EntityKey::default()],
                properties: "test properties".to_string(),
            };

            assert_eq!(entity.similarity, 0.85);
            assert_eq!(entity.neighbors.len(), 2);
            assert_eq!(entity.properties, "test properties");
        }

        #[test]
        fn test_context_entity_empty_neighbors() {
            let entity = ContextEntity {
                id: EntityKey::default(),
                similarity: 1.0,
                neighbors: vec![],
                properties: String::new(),
            };

            assert_eq!(entity.similarity, 1.0);
            assert!(entity.neighbors.is_empty());
            assert!(entity.properties.is_empty());
        }

        #[test]
        fn test_context_entity_clone() {
            let entity = ContextEntity {
                id: EntityKey::default(),
                similarity: 0.75,
                neighbors: vec![EntityKey::default()],
                properties: "clone test".to_string(),
            };

            let cloned = entity.clone();
            assert_eq!(entity.similarity, cloned.similarity);
            assert_eq!(entity.neighbors.len(), cloned.neighbors.len());
            assert_eq!(entity.properties, cloned.properties);
        }
    }

    mod query_result_tests {
        use super::*;

        #[test]
        fn test_query_result_creation() {
            let result = QueryResult {
                entities: vec![],
                relationships: vec![],
                confidence: 0.95,
                query_time_ms: 150,
            };

            assert!(result.entities.is_empty());
            assert!(result.relationships.is_empty());
            assert_eq!(result.confidence, 0.95);
            assert_eq!(result.query_time_ms, 150);
        }

        #[test]
        fn test_query_result_with_data() {
            let entity = ContextEntity {
                id: EntityKey::default(),
                similarity: 0.8,
                neighbors: vec![],
                properties: "test".to_string(),
            };

            let relationship = Relationship {
                from: EntityKey::default(),
                to: EntityKey::default(),
                rel_type: 1,
                weight: 0.5,
            };

            let result = QueryResult {
                entities: vec![entity],
                relationships: vec![relationship],
                confidence: 0.85,
                query_time_ms: 200,
            };

            assert_eq!(result.entities.len(), 1);
            assert_eq!(result.relationships.len(), 1);
            assert_eq!(result.confidence, 0.85);
            assert_eq!(result.query_time_ms, 200);
        }

        #[test]
        fn test_query_result_clone() {
            let result = QueryResult {
                entities: vec![],
                relationships: vec![],
                confidence: 0.9,
                query_time_ms: 100,
            };

            let cloned = result.clone();
            assert_eq!(result.confidence, cloned.confidence);
            assert_eq!(result.query_time_ms, cloned.query_time_ms);
        }
    }

    mod simd_relationship_tests {
        use super::*;

        #[test]
        fn test_simd_relationship_creation() {
            let simd_rel = SIMDRelationship {
                from: [1, 2, 3, 4, 5, 6, 7, 8],
                to: [10, 20, 30, 40, 50, 60, 70, 80],
                rel_type: [1, 1, 2, 2, 3, 3, 4, 4],
                weight: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                count: 8,
            };

            assert_eq!(simd_rel.from[0], 1);
            assert_eq!(simd_rel.to[7], 80);
            assert_eq!(simd_rel.rel_type[2], 2);
            assert_eq!(simd_rel.weight[4], 0.5);
            assert_eq!(simd_rel.count, 8);
        }

        #[test]
        fn test_simd_relationship_partial_fill() {
            let simd_rel = SIMDRelationship {
                from: [1, 2, 3, 0, 0, 0, 0, 0],
                to: [10, 20, 30, 0, 0, 0, 0, 0],
                rel_type: [1, 1, 2, 0, 0, 0, 0, 0],
                weight: [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                count: 3,
            };

            assert_eq!(simd_rel.count, 3);
            assert_eq!(simd_rel.from[2], 3);
            assert_eq!(simd_rel.from[3], 0); // Unused slot
        }

        #[test]
        fn test_simd_relationship_copy() {
            let simd_rel = SIMDRelationship {
                from: [1; 8],
                to: [2; 8],
                rel_type: [3; 8],
                weight: [0.5; 8],
                count: 8,
            };

            let copied = simd_rel;
            assert_eq!(simd_rel.count, copied.count);
            assert_eq!(simd_rel.from[0], copied.from[0]);
        }
    }
}