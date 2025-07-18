pub mod types;
pub mod operations;

use std::collections::HashMap;
use crate::error::Result;

pub use types::{NodeId, EdgeId, NodeType, EdgeType, PropertyKey, PropertyValue};

/// A simple graph structure for cognitive pattern testing
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<EdgeId, Edge>,
    node_counter: NodeId,
    edge_counter: EdgeId,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub node_type: NodeType,
    pub properties: Vec<Property>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: EdgeType,
    pub properties: Vec<Property>,
}

#[derive(Debug, Clone)]
pub struct Property {
    pub key: PropertyKey,
    pub value: PropertyValue,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            node_counter: 0,
            edge_counter: 0,
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.node_counter;
        self.node_counter += 1;
        self.nodes.insert(id, Node { id, ..node });
        id
    }

    pub fn add_edge(&mut self, edge: Edge) -> EdgeId {
        let id = self.edge_counter;
        self.edge_counter += 1;
        self.edges.insert(id, Edge { id, ..edge });
        id
    }

    pub fn get_node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(&id)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Node {} not found", id)))
    }

    pub fn get_edge(&self, id: EdgeId) -> Result<&Edge> {
        self.edges.get(&id)
            .ok_or_else(|| crate::error::GraphError::InvalidInput(format!("Edge {} not found", id)))
    }

    pub fn get_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn get_edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    pub fn get_edges_from(&self, node_id: NodeId) -> impl Iterator<Item = &Edge> {
        self.edges.values().filter(move |e| e.source == node_id)
    }

    pub fn get_edges_to(&self, node_id: NodeId) -> impl Iterator<Item = &Edge> {
        self.edges.values().filter(move |e| e.target == node_id)
    }
}

impl Node {
    pub fn new(
        name: String,
        node_type: NodeType,
        property_key: PropertyKey,
        property_value: PropertyValue,
    ) -> Self {
        Self {
            id: 0, // Will be set by Graph
            name,
            node_type,
            properties: vec![Property {
                key: property_key,
                value: property_value,
            }],
        }
    }
}

impl Edge {
    pub fn new(
        source: NodeId,
        target: NodeId,
        edge_type: EdgeType,
        property_key: PropertyKey,
        property_value: PropertyValue,
    ) -> Self {
        Self {
            id: 0, // Will be set by Graph
            source,
            target,
            edge_type,
            properties: vec![Property {
                key: property_key,
                value: property_value,
            }],
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}