# MP005: Graph Visualization Setup

## Task Description
Set up graph visualization infrastructure to enable debugging and analysis of neuromorphic network structures.

## Prerequisites
- MP001-MP004 completed
- Basic understanding of graph layouts
- Familiarity with visualization formats

## Detailed Steps

1. Create `src/neuromorphic/graph/visualization.rs`

2. Define visualization trait:
   ```rust
   pub trait GraphVisualizer {
       fn to_dot(&self) -> String;
       fn to_gexf(&self) -> String;
       fn to_json_vis(&self) -> String;
   }
   ```

3. Implement DOT format export:
   - Generate GraphViz compatible output
   - Include node labels and weights
   - Color nodes by activation level
   - Style edges by synaptic strength

4. Implement GEXF format export:
   - Create Gephi-compatible XML
   - Include dynamic attributes
   - Support time-series data

5. Implement JSON visualization format:
   - D3.js compatible structure
   - Include layout hints
   - Add interactive metadata

6. Create layout algorithms:
   - Force-directed layout
   - Hierarchical layout
   - Circular layout
   - 3D spatial layout (matching cortical columns)

## Expected Output
```rust
// src/neuromorphic/graph/visualization.rs
impl GraphVisualizer for NeuromorphicGraph {
    fn to_dot(&self) -> String {
        let mut dot = String::from("digraph NeuromorphicGraph {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=circle];\n");
        
        // Add nodes with activation-based colors
        for (id, node) in &self.nodes {
            let color = activation_to_color(node.activation);
            dot.push_str(&format!(
                "  {} [label=\"{}\", fillcolor=\"{}\", style=filled];\n",
                id, id, color
            ));
        }
        
        // Add edges with weight-based thickness
        for edge in &self.edges {
            let width = edge.weight.abs() * 3.0;
            dot.push_str(&format!(
                "  {} -> {} [penwidth={:.1}];\n",
                edge.source_id, edge.target_id, width
            ));
        }
        
        dot.push_str("}\n");
        dot
    }
}
```

## Verification Steps
1. Export a small graph to DOT and visualize with GraphViz
2. Test GEXF export with Gephi import
3. Verify JSON format with a D3.js example
4. Check layout algorithm performance

## Time Estimate
25 minutes

## Dependencies
- MP001-MP002: Graph implementation
- MP004: Metrics for node sizing/coloring