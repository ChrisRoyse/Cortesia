# MP001: Graph Traits Foundation

## Task Description
Define and implement the core graph trait system that will serve as the foundation for all advanced graph algorithms in the neuromorphic system.

## Prerequisites
- Rust development environment
- Understanding of trait-based design patterns
- Basic graph theory knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/traits.rs` file
2. Define `GraphNode` trait with:
   - `id()` method returning unique identifier
   - `neighbors()` method returning iterator over connected nodes
   - `weight()` method for node weight/importance

3. Define `GraphEdge` trait with:
   - `source()` and `target()` methods
   - `weight()` method for edge weight
   - `is_directed()` method

4. Define `Graph` trait with:
   - Associated types for Node and Edge
   - `nodes()` iterator method
   - `edges()` iterator method
   - `add_node()` and `add_edge()` methods
   - `remove_node()` and `remove_edge()` methods

5. Create marker traits:
   - `DirectedGraph` for directed graphs
   - `UndirectedGraph` for undirected graphs
   - `WeightedGraph` for weighted graphs

## Expected Output
```rust
// src/neuromorphic/graph/traits.rs
pub trait GraphNode {
    type Id: Eq + Hash + Clone;
    fn id(&self) -> Self::Id;
    fn neighbors(&self) -> Box<dyn Iterator<Item = Self::Id> + '_>;
    fn weight(&self) -> f32;
}

pub trait GraphEdge {
    type NodeId: Eq + Hash + Clone;
    fn source(&self) -> Self::NodeId;
    fn target(&self) -> Self::NodeId;
    fn weight(&self) -> f32;
    fn is_directed(&self) -> bool;
}

pub trait Graph {
    type Node: GraphNode;
    type Edge: GraphEdge;
    // ... methods
}
```

## Verification Steps
1. Compile the traits module
2. Create a simple test graph implementation
3. Verify trait bounds compile correctly
4. Check that all methods have appropriate lifetimes

## Time Estimate
25 minutes

## Dependencies
None - this is the foundation task