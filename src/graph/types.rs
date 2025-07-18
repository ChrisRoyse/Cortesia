use serde::{Deserialize, Serialize};

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    Entity(String),
    Concept,
    Property,
    Relationship,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    IsA,
    HasProperty,
    RelatedTo,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyKey {
    Confidence,
    Weight,
    Timestamp,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    String(String),
    Int(i64),
    Float(f32),
    Bool(bool),
    List(Vec<PropertyValue>),
}