use serde::{Deserialize, Serialize};
use slotmap::new_key_type;
use std::collections::HashMap;

new_key_type! {
    pub struct EntityKey;
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