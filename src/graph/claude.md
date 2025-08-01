# Directory Overview: Graph Module

## 1. High-Level Summary

The `src/graph` directory provides a lightweight, flexible graph data structure specifically designed for cognitive pattern testing within the LLMKG system. This module implements a basic but complete graph abstraction with nodes, edges, properties, and essential operations. It serves as a foundational component for representing knowledge graphs, relationships, and cognitive patterns in a structured format.

The graph implementation focuses on simplicity and extensibility, supporting typed nodes and edges with custom properties, making it suitable for various AI and knowledge representation tasks.

## 2. Tech Stack

* **Languages:** Rust
* **External Dependencies:** 
  - `serde` (Serialize, Deserialize) - For JSON/binary serialization
  - `thiserror` (via `crate::error`) - For structured error handling
* **Internal Dependencies:**
  - `crate::error` - Project-wide error handling system
* **Standard Library:** `std::collections::HashMap` - For efficient node/edge storage

## 3. Directory Structure

```
src/graph/
├── mod.rs         # Main module with core Graph, Node, Edge structs and implementations
├── operations.rs  # Additional graph operations and traits (GraphOps)
└── types.rs       # Type definitions, enums, and aliases
```

## 4. File Breakdown

### `mod.rs`

**Purpose:** Core module containing the main graph data structures and basic operations.

**Key Structs:**

* **`Graph`**
  - **Description:** Main graph container with HashMap-based storage for nodes and edges.
  - **Fields:**
    - `nodes: HashMap<NodeId, Node>` - Stores all nodes indexed by ID
    - `edges: HashMap<EdgeId, Edge>` - Stores all edges indexed by ID  
    - `node_counter: NodeId` - Auto-incrementing counter for node IDs
    - `edge_counter: EdgeId` - Auto-incrementing counter for edge IDs
  - **Methods:**
    - `new() -> Self` - Creates empty graph
    - `add_node(&mut self, node: Node) -> NodeId` - Adds node and returns generated ID
    - `add_edge(&mut self, edge: Edge) -> EdgeId` - Adds edge and returns generated ID
    - `get_node(&self, id: NodeId) -> Result<&Node>` - Retrieves node by ID
    - `get_edge(&self, id: EdgeId) -> Result<&Edge>` - Retrieves edge by ID
    - `get_nodes(&self) -> impl Iterator<Item = &Node>` - Returns iterator over all nodes
    - `get_edges(&self) -> impl Iterator<Item = &Edge>` - Returns iterator over all edges
    - `get_edges_from(&self, node_id: NodeId) -> impl Iterator<Item = &Edge>` - Gets outgoing edges
    - `get_edges_to(&self, node_id: NodeId) -> impl Iterator<Item = &Edge>` - Gets incoming edges

* **`Node`**
  - **Description:** Represents a graph node with metadata and properties.
  - **Fields:**
    - `id: NodeId` - Unique identifier (set by Graph)
    - `name: String` - Human-readable node name
    - `node_type: NodeType` - Categorizes the node type
    - `properties: Vec<Property>` - List of key-value properties
  - **Methods:**
    - `new() -> Self` - Creates new node with single property

* **`Edge`**  
  - **Description:** Represents a directed edge connecting two nodes.
  - **Fields:**
    - `id: EdgeId` - Unique identifier (set by Graph)
    - `source: NodeId` - Source node ID
    - `target: NodeId` - Target node ID
    - `edge_type: EdgeType` - Categorizes the relationship type
    - `properties: Vec<Property>` - List of key-value properties
  - **Methods:**
    - `new() -> Self` - Creates new edge with single property

* **`Property`**
  - **Description:** Key-value pair for storing metadata on nodes and edges.
  - **Fields:**
    - `key: PropertyKey` - Property identifier
    - `value: PropertyValue` - Property value (supports multiple types)

### `operations.rs`

**Purpose:** Extended operations and trait definitions for graph functionality.

**Traits:**

* **`GraphOps`**
  - **Description:** Trait providing additional graph operations beyond basic CRUD.
  - **Methods:**
    - `node_count(&self) -> usize` - Returns total number of nodes
    - `edge_count(&self) -> usize` - Returns total number of edges
    - `has_node(&self, id: NodeId) -> bool` - Checks if node exists
    - `has_edge(&self, id: EdgeId) -> bool` - Checks if edge exists
    - `neighbors(&self, node_id: NodeId) -> Vec<NodeId>` - Gets neighbor node IDs

**Implementations:**
- `GraphOps for Graph` - Implements all trait methods for the main Graph struct

### `types.rs`

**Purpose:** Centralized type definitions, aliases, and enums for the graph system.

**Type Aliases:**
- `NodeId = usize` - Node identifier type
- `EdgeId = usize` - Edge identifier type

**Enums:**

* **`NodeType`** (Serializable)
  - **Values:**
    - `Entity(String)` - Named entity node
    - `Concept` - Abstract concept node
    - `Property` - Property/attribute node
    - `Relationship` - Relationship descriptor node

* **`EdgeType`** (Serializable)
  - **Values:**
    - `IsA` - Inheritance/classification relationship
    - `HasProperty` - Property ownership relationship
    - `RelatedTo` - General relationship
    - `Custom(String)` - User-defined relationship type

* **`PropertyKey`** (Serializable)
  - **Values:**
    - `Confidence` - Confidence score metadata
    - `Weight` - Relationship weight
    - `Timestamp` - Temporal metadata
    - `Custom(String)` - User-defined property key

* **`PropertyValue`** (Serializable)
  - **Values:**
    - `String(String)` - Text value
    - `Int(i64)` - Integer value
    - `Float(f32)` - Floating point value
    - `Bool(bool)` - Boolean value
    - `List(Vec<PropertyValue>)` - Nested list of values

## 5. Key Variables and Logic

**ID Management:**
- Auto-incrementing counters (`node_counter`, `edge_counter`) ensure unique IDs
- IDs start at 0 and increment for each new node/edge
- Node/Edge constructors set ID to 0 (placeholder), Graph reassigns during insertion

**Error Handling:**
- Uses project-wide `crate::error::Result<T>` and `GraphError` types
- `InvalidInput` errors for missing nodes/edges with descriptive messages
- Follows Rust error handling patterns with `Result` returns

**Iterator Pattern:**
- Extensive use of iterators for efficient traversal
- `get_edges_from`/`get_edges_to` use filter closures for directional edge queries
- All iteration methods return lazy iterators for memory efficiency

## 6. Dependencies

**Internal:**
- `crate::error` - Centralized error handling system with `GraphError` enum and `Result<T>` type

**External:**
- `serde` - Serialization/deserialization for all public types
- `std::collections::HashMap` - Core storage mechanism for nodes and edges

## 7. Usage Patterns

**Basic Graph Creation:**
```rust
use crate::graph::{Graph, Node, Edge, NodeType, EdgeType, PropertyKey, PropertyValue};

let mut graph = Graph::new();
let node1 = Node::new("Person".to_string(), NodeType::Entity("Person".to_string()), 
                     PropertyKey::Confidence, PropertyValue::Float(0.9));
let node_id = graph.add_node(node1);
```

**Traversal Operations:**
```rust
// Get all neighbors of a node
let neighbors = graph.neighbors(node_id);

// Get outgoing edges
for edge in graph.get_edges_from(node_id) {
    println!("Edge to: {}", edge.target);
}
```

## 8. Architecture Notes

**Design Principles:**
- **Simplicity:** Focused on essential graph operations without complex algorithms
- **Extensibility:** Enum-based typing allows custom node/edge/property types
- **Type Safety:** Strong typing with custom enums prevents invalid relationships
- **Memory Efficiency:** HashMap storage for O(1) lookups, lazy iterators for traversal

**Integration Points:**
- Designed specifically for "cognitive pattern testing" based on module comment
- Serializable types enable persistence and network communication
- Error integration with larger LLMKG system for consistent error handling
- Property system supports flexible metadata for AI/ML applications

**Limitations:**
- No built-in graph algorithms (shortest path, clustering, etc.)
- Single-threaded design (no concurrent access protection)
- In-memory only (no built-in persistence layer)
- Basic property system (no schema validation or indexing)