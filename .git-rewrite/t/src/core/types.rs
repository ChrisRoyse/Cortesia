use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

new_key_type! {
    pub struct EntityKey;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct EntityMeta {
    pub type_id: u16,
    pub embedding_offset: u32,
    pub property_offset: u32,
    pub degree: u16,
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
    pub from: u32,
    pub to: u32,
    pub rel_type: u8,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextEntity {
    pub id: u32, // Changed from EntityKey to u32 for serialization
    pub similarity: f32,
    pub neighbors: Vec<u32>,
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