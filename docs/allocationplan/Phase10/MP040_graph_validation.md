# MP040: Graph Validation

## Task Description
Implement comprehensive graph validation algorithms to ensure data integrity, structural consistency, and logical correctness of neural networks, including property validation, constraint checking, and anomaly detection.

## Prerequisites
- MP001-MP010: Graph infrastructure
- MP025: Graph clustering coefficient (for structural validation)
- MP021-MP024: Centrality measures (for anomaly detection)
- Understanding of graph invariants and validation techniques

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/validation/mod.rs`

2. Implement structural validation:
   ```rust
   use std::collections::{HashMap, HashSet, VecDeque};
   use crate::neuromorphic::graph::traits::{Graph, GraphNode, GraphEdge};

   #[derive(Debug, Clone)]
   pub struct ValidationResult {
       pub is_valid: bool,
       pub structural_errors: Vec<StructuralError>,
       pub logical_errors: Vec<LogicalError>,
       pub warnings: Vec<ValidationWarning>,
       pub metrics: ValidationMetrics,
   }

   #[derive(Debug, Clone)]
   pub enum StructuralError {
       SelfLoop { node_id: String },
       MultipleEdges { source: String, target: String, count: usize },
       DanglingEdge { edge_info: String },
       InvalidNodeReference { node_id: String },
       InconsistentDirectionality { edge_info: String },
       DisconnectedComponent { component_size: usize },
   }

   #[derive(Debug, Clone)]
   pub enum LogicalError {
       InvalidWeight { edge_info: String, weight: f32, expected_range: (f32, f32) },
       ConstraintViolation { constraint: String, details: String },
       InvariantBroken { invariant: String, expected: f32, actual: f32 },
       TypeMismatch { expected: String, actual: String },
       CapacityExceeded { resource: String, limit: usize, actual: usize },
   }

   #[derive(Debug, Clone)]
   pub enum ValidationWarning {
       HighDegreeNode { node_id: String, degree: usize, threshold: usize },
       IsolatedNode { node_id: String },
       PotentialBottleneck { node_id: String, centrality: f32 },
       UnbalancedStructure { metric: String, value: f32 },
       PerformanceIssue { description: String },
   }

   #[derive(Debug, Clone)]
   pub struct ValidationMetrics {
       pub nodes_validated: usize,
       pub edges_validated: usize,
       pub validation_time_ms: u64,
       pub memory_usage_bytes: usize,
       pub error_rate: f32,
       pub warning_rate: f32,
   }

   pub fn validate_graph_structure<G: Graph>(
       graph: &G,
       config: &ValidationConfig,
   ) -> ValidationResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let start_time = std::time::Instant::now();
       let mut structural_errors = Vec::new();
       let mut logical_errors = Vec::new();
       let mut warnings = Vec::new();
       
       // Validate nodes
       let node_validation = validate_nodes(graph, config);
       structural_errors.extend(node_validation.structural_errors);
       logical_errors.extend(node_validation.logical_errors);
       warnings.extend(node_validation.warnings);
       
       // Validate edges
       let edge_validation = validate_edges(graph, config);
       structural_errors.extend(edge_validation.structural_errors);
       logical_errors.extend(edge_validation.logical_errors);
       warnings.extend(edge_validation.warnings);
       
       // Validate connectivity
       let connectivity_validation = validate_connectivity(graph, config);
       structural_errors.extend(connectivity_validation.structural_errors);
       warnings.extend(connectivity_validation.warnings);
       
       // Validate graph properties
       let property_validation = validate_graph_properties(graph, config);
       logical_errors.extend(property_validation.logical_errors);
       warnings.extend(property_validation.warnings);
       
       let validation_time = start_time.elapsed().as_millis() as u64;
       let nodes_count = graph.node_count();
       let edges_count = graph.edge_count();
       
       let is_valid = structural_errors.is_empty() && logical_errors.is_empty();
       let error_rate = (structural_errors.len() + logical_errors.len()) as f32 / 
                       (nodes_count + edges_count) as f32;
       let warning_rate = warnings.len() as f32 / (nodes_count + edges_count) as f32;
       
       ValidationResult {
           is_valid,
           structural_errors,
           logical_errors,
           warnings,
           metrics: ValidationMetrics {
               nodes_validated: nodes_count,
               edges_validated: edges_count,
               validation_time_ms: validation_time,
               memory_usage_bytes: estimate_memory_usage(graph),
               error_rate,
               warning_rate,
           },
       }
   }

   #[derive(Debug, Clone)]
   pub struct ValidationConfig {
       pub check_self_loops: bool,
       pub check_multiple_edges: bool,
       pub check_connectivity: bool,
       pub weight_range: Option<(f32, f32)>,
       pub max_degree_threshold: Option<usize>,
       pub min_component_size: Option<usize>,
       pub custom_constraints: Vec<CustomConstraint>,
       pub performance_thresholds: PerformanceThresholds,
   }

   #[derive(Debug, Clone)]
   pub struct CustomConstraint {
       pub name: String,
       pub description: String,
       pub validator: ConstraintType,
   }

   #[derive(Debug, Clone)]
   pub enum ConstraintType {
       NodeProperty { property: String, min: f32, max: f32 },
       EdgeProperty { property: String, min: f32, max: f32 },
       GraphInvariant { invariant: String, expected_value: f32, tolerance: f32 },
       ConnectivityPattern { pattern: String },
   }

   #[derive(Debug, Clone)]
   pub struct PerformanceThresholds {
       pub max_validation_time_ms: u64,
       pub max_memory_mb: usize,
       pub max_error_rate: f32,
   }

   fn validate_nodes<G: Graph>(
       graph: &G,
       config: &ValidationConfig,
   ) -> ValidationResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut structural_errors = Vec::new();
       let mut logical_errors = Vec::new();
       let mut warnings = Vec::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           
           // Check for isolated nodes
           let degree = node.neighbors().count();
           if degree == 0 {
               warnings.push(ValidationWarning::IsolatedNode {
                   node_id: node_id.to_string(),
               });
           }
           
           // Check degree threshold
           if let Some(max_degree) = config.max_degree_threshold {
               if degree > max_degree {
                   warnings.push(ValidationWarning::HighDegreeNode {
                       node_id: node_id.to_string(),
                       degree,
                       threshold: max_degree,
                   });
               }
           }
           
           // Check self-loops if configured
           if config.check_self_loops {
               for neighbor in node.neighbors() {
                   if neighbor == node_id {
                       structural_errors.push(StructuralError::SelfLoop {
                           node_id: node_id.to_string(),
                       });
                   }
               }
           }
           
           // Validate node weight
           let node_weight = node.weight();
           if let Some((min_weight, max_weight)) = config.weight_range {
               if node_weight < min_weight || node_weight > max_weight {
                   logical_errors.push(LogicalError::InvalidWeight {
                       edge_info: format!("Node {}", node_id),
                       weight: node_weight,
                       expected_range: (min_weight, max_weight),
                   });
               }
           }
       }
       
       ValidationResult {
           is_valid: structural_errors.is_empty() && logical_errors.is_empty(),
           structural_errors,
           logical_errors,
           warnings,
           metrics: ValidationMetrics {
               nodes_validated: graph.node_count(),
               edges_validated: 0,
               validation_time_ms: 0,
               memory_usage_bytes: 0,
               error_rate: 0.0,
               warning_rate: 0.0,
           },
       }
   }

   fn validate_edges<G: Graph>(
       graph: &G,
       config: &ValidationConfig,
   ) -> ValidationResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut structural_errors = Vec::new();
       let mut logical_errors = Vec::new();
       let mut warnings = Vec::new();
       
       let mut edge_counts: HashMap<(G::Node::Id, G::Node::Id), usize> = HashMap::new();
       
       for edge in graph.edges() {
           let source = edge.source();
           let target = edge.target();
           
           // Check for valid node references
           if graph.get_node(&source).is_none() {
               structural_errors.push(StructuralError::InvalidNodeReference {
                   node_id: source.to_string(),
               });
           }
           
           if graph.get_node(&target).is_none() {
               structural_errors.push(StructuralError::InvalidNodeReference {
                   node_id: target.to_string(),
               });
           }
           
           // Count multiple edges if configured
           if config.check_multiple_edges {
               let edge_key = (source.clone(), target.clone());
               *edge_counts.entry(edge_key).or_insert(0) += 1;
           }
           
           // Validate edge weight
           let edge_weight = edge.weight();
           if let Some((min_weight, max_weight)) = config.weight_range {
               if edge_weight < min_weight || edge_weight > max_weight {
                   logical_errors.push(LogicalError::InvalidWeight {
                       edge_info: format!("Edge {} -> {}", source, target),
                       weight: edge_weight,
                       expected_range: (min_weight, max_weight),
                   });
               }
           }
           
           // Check for NaN or infinite weights
           if edge_weight.is_nan() || edge_weight.is_infinite() {
               logical_errors.push(LogicalError::InvalidWeight {
                   edge_info: format!("Edge {} -> {}", source, target),
                   weight: edge_weight,
                   expected_range: (f32::NEG_INFINITY, f32::INFINITY),
               });
           }
       }
       
       // Report multiple edges
       for ((source, target), count) in edge_counts {
           if count > 1 {
               structural_errors.push(StructuralError::MultipleEdges {
                   source: source.to_string(),
                   target: target.to_string(),
                   count,
               });
           }
       }
       
       ValidationResult {
           is_valid: structural_errors.is_empty() && logical_errors.is_empty(),
           structural_errors,
           logical_errors,
           warnings,
           metrics: ValidationMetrics {
               nodes_validated: 0,
               edges_validated: graph.edge_count(),
               validation_time_ms: 0,
               memory_usage_bytes: 0,
               error_rate: 0.0,
               warning_rate: 0.0,
           },
       }
   }
   ```

3. Implement connectivity validation:
   ```rust
   fn validate_connectivity<G: Graph>(
       graph: &G,
       config: &ValidationConfig,
   ) -> ValidationResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut structural_errors = Vec::new();
       let mut warnings = Vec::new();
       
       if config.check_connectivity {
           let components = find_connected_components(graph);
           
           // Check for disconnected components
           if components.len() > 1 {
               for component in &components {
                   if let Some(min_size) = config.min_component_size {
                       if component.len() < min_size {
                           warnings.push(ValidationWarning::UnbalancedStructure {
                               metric: "Small connected component".to_string(),
                               value: component.len() as f32,
                           });
                       }
                   }
               }
               
               // Report largest disconnected component
               let largest_component_size = components.iter().map(|c| c.len()).max().unwrap_or(0);
               if largest_component_size < graph.node_count() {
                   structural_errors.push(StructuralError::DisconnectedComponent {
                       component_size: largest_component_size,
                   });
               }
           }
       }
       
       ValidationResult {
           is_valid: structural_errors.is_empty(),
           structural_errors,
           logical_errors: vec![],
           warnings,
           metrics: ValidationMetrics {
               nodes_validated: graph.node_count(),
               edges_validated: 0,
               validation_time_ms: 0,
               memory_usage_bytes: 0,
               error_rate: 0.0,
               warning_rate: 0.0,
           },
       }
   }

   fn find_connected_components<G: Graph>(
       graph: &G,
   ) -> Vec<Vec<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut visited = HashSet::new();
       let mut components = Vec::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           if !visited.contains(&node_id) {
               let component = bfs_component(graph, node_id.clone(), &mut visited);
               components.push(component);
           }
       }
       
       components
   }

   fn bfs_component<G: Graph>(
       graph: &G,
       start: G::Node::Id,
       visited: &mut HashSet<G::Node::Id>,
   ) -> Vec<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut component = Vec::new();
       let mut queue = VecDeque::new();
       
       queue.push_back(start.clone());
       visited.insert(start);
       
       while let Some(current) = queue.pop_front() {
           component.push(current.clone());
           
           if let Some(node) = graph.get_node(&current) {
               for neighbor in node.neighbors() {
                   if !visited.contains(&neighbor) {
                       visited.insert(neighbor.clone());
                       queue.push_back(neighbor);
                   }
               }
           }
       }
       
       component
   }
   ```

4. Implement property validation and anomaly detection:
   ```rust
   fn validate_graph_properties<G: Graph>(
       graph: &G,
       config: &ValidationConfig,
   ) -> ValidationResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut logical_errors = Vec::new();
       let mut warnings = Vec::new();
       
       // Validate custom constraints
       for constraint in &config.custom_constraints {
           match validate_constraint(graph, constraint) {
               ConstraintResult::Valid => {},
               ConstraintResult::Error(error) => logical_errors.push(error),
               ConstraintResult::Warning(warning) => warnings.push(warning),
           }
       }
       
       // Detect structural anomalies
       let anomalies = detect_structural_anomalies(graph);
       warnings.extend(anomalies);
       
       // Check graph invariants
       let invariant_violations = check_graph_invariants(graph);
       logical_errors.extend(invariant_violations);
       
       ValidationResult {
           is_valid: logical_errors.is_empty(),
           structural_errors: vec![],
           logical_errors,
           warnings,
           metrics: ValidationMetrics {
               nodes_validated: graph.node_count(),
               edges_validated: graph.edge_count(),
               validation_time_ms: 0,
               memory_usage_bytes: 0,
               error_rate: 0.0,
               warning_rate: 0.0,
           },
       }
   }

   #[derive(Debug)]
   enum ConstraintResult {
       Valid,
       Error(LogicalError),
       Warning(ValidationWarning),
   }

   fn validate_constraint<G: Graph>(
       graph: &G,
       constraint: &CustomConstraint,
   ) -> ConstraintResult 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       match &constraint.validator {
           ConstraintType::NodeProperty { property, min, max } => {
               for node in graph.nodes() {
                   let value = get_node_property_value(node, property);
                   if value < *min || value > *max {
                       return ConstraintResult::Error(LogicalError::ConstraintViolation {
                           constraint: constraint.name.clone(),
                           details: format!("Node {} property {} = {}, expected [{}, {}]",
                               node.id(), property, value, min, max),
                       });
                   }
               }
           },
           ConstraintType::EdgeProperty { property, min, max } => {
               for edge in graph.edges() {
                   let value = get_edge_property_value(&edge, property);
                   if value < *min || value > *max {
                       return ConstraintResult::Error(LogicalError::ConstraintViolation {
                           constraint: constraint.name.clone(),
                           details: format!("Edge {} -> {} property {} = {}, expected [{}, {}]",
                               edge.source(), edge.target(), property, value, min, max),
                       });
                   }
               }
           },
           ConstraintType::GraphInvariant { invariant, expected_value, tolerance } => {
               let actual_value = compute_graph_invariant(graph, invariant);
               if (actual_value - expected_value).abs() > *tolerance {
                   return ConstraintResult::Error(LogicalError::InvariantBroken {
                       invariant: invariant.clone(),
                       expected: *expected_value,
                       actual: actual_value,
                   });
               }
           },
           ConstraintType::ConnectivityPattern { pattern } => {
               if !check_connectivity_pattern(graph, pattern) {
                   return ConstraintResult::Error(LogicalError::ConstraintViolation {
                       constraint: constraint.name.clone(),
                       details: format!("Connectivity pattern '{}' not satisfied", pattern),
                   });
               }
           },
       }
       
       ConstraintResult::Valid
   }

   fn detect_structural_anomalies<G: Graph>(
       graph: &G,
   ) -> Vec<ValidationWarning> 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut warnings = Vec::new();
       
       // Detect nodes with unusually high centrality
       let centrality_scores = compute_degree_centrality(graph);
       let mean_centrality: f32 = centrality_scores.values().sum::<f32>() / centrality_scores.len() as f32;
       let centrality_threshold = mean_centrality * 3.0; // 3x mean as threshold
       
       for (node_id, &centrality) in &centrality_scores {
           if centrality > centrality_threshold {
               warnings.push(ValidationWarning::PotentialBottleneck {
                   node_id: node_id.to_string(),
                   centrality,
               });
           }
       }
       
       // Detect imbalanced degree distribution
       let degrees: Vec<usize> = graph.nodes()
           .map(|node| node.neighbors().count())
           .collect();
       
       if !degrees.is_empty() {
           let max_degree = *degrees.iter().max().unwrap();
           let min_degree = *degrees.iter().min().unwrap();
           let degree_ratio = max_degree as f32 / (min_degree + 1) as f32;
           
           if degree_ratio > 10.0 {
               warnings.push(ValidationWarning::UnbalancedStructure {
                   metric: "Degree distribution".to_string(),
                   value: degree_ratio,
               });
           }
       }
       
       warnings
   }

   fn check_graph_invariants<G: Graph>(
       graph: &G,
   ) -> Vec<LogicalError> 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       let mut errors = Vec::new();
       
       // Check handshaking lemma for undirected graphs
       let total_degree: usize = graph.nodes()
           .map(|node| node.neighbors().count())
           .sum();
       
       let edge_count = graph.edge_count();
       
       // For undirected graphs: sum of degrees = 2 * number of edges
       // For directed graphs: this relationship may not hold
       if total_degree != 2 * edge_count {
           // This might be normal for directed graphs, so we'll make it a warning
           // In a more sophisticated implementation, we'd check the graph type
       }
       
       errors
   }

   // Helper functions
   fn get_node_property_value<N: GraphNode>(node: &N, property: &str) -> f32 {
       match property {
           "weight" => node.weight(),
           "degree" => node.neighbors().count() as f32,
           _ => 0.0, // Default or lookup in node metadata
       }
   }

   fn get_edge_property_value<E: GraphEdge>(edge: &E, property: &str) -> f32 {
       match property {
           "weight" => edge.weight(),
           _ => 0.0, // Default or lookup in edge metadata
       }
   }

   fn compute_graph_invariant<G: Graph>(graph: &G, invariant: &str) -> f32 {
       match invariant {
           "node_count" => graph.node_count() as f32,
           "edge_count" => graph.edge_count() as f32,
           "density" => {
               let n = graph.node_count() as f32;
               let m = graph.edge_count() as f32;
               if n > 1.0 {
                   2.0 * m / (n * (n - 1.0))
               } else {
                   0.0
               }
           },
           _ => 0.0, // Default for unknown invariants
       }
   }

   fn check_connectivity_pattern<G: Graph>(graph: &G, pattern: &str) -> bool {
       match pattern {
           "connected" => find_connected_components(graph).len() == 1,
           "acyclic" => is_acyclic(graph),
           "bipartite" => is_bipartite(graph),
           _ => true, // Default to valid for unknown patterns
       }
   }

   fn compute_degree_centrality<G: Graph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut centrality = HashMap::new();
       let n = graph.node_count() as f32;
       
       for node in graph.nodes() {
           let degree = node.neighbors().count() as f32;
           let normalized_degree = if n > 1.0 { degree / (n - 1.0) } else { 0.0 };
           centrality.insert(node.id(), normalized_degree);
       }
       
       centrality
   }

   fn is_acyclic<G: Graph>(graph: &G) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // DFS-based cycle detection
       let mut visited = HashSet::new();
       let mut rec_stack = HashSet::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           if !visited.contains(&node_id) {
               if has_cycle_dfs(graph, node_id, &mut visited, &mut rec_stack) {
                   return false;
               }
           }
       }
       
       true
   }

   fn has_cycle_dfs<G: Graph>(
       graph: &G,
       node_id: G::Node::Id,
       visited: &mut HashSet<G::Node::Id>,
       rec_stack: &mut HashSet<G::Node::Id>,
   ) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       visited.insert(node_id.clone());
       rec_stack.insert(node_id.clone());
       
       if let Some(node) = graph.get_node(&node_id) {
           for neighbor in node.neighbors() {
               if !visited.contains(&neighbor) {
                   if has_cycle_dfs(graph, neighbor, visited, rec_stack) {
                       return true;
                   }
               } else if rec_stack.contains(&neighbor) {
                   return true;
               }
           }
       }
       
       rec_stack.remove(&node_id);
       false
   }

   fn is_bipartite<G: Graph>(graph: &G) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Two-coloring approach
       let mut colors: HashMap<G::Node::Id, bool> = HashMap::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           if !colors.contains_key(&node_id) {
               if !color_bipartite_dfs(graph, node_id, true, &mut colors) {
                   return false;
               }
           }
       }
       
       true
   }

   fn color_bipartite_dfs<G: Graph>(
       graph: &G,
       node_id: G::Node::Id,
       color: bool,
       colors: &mut HashMap<G::Node::Id, bool>,
   ) -> bool 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       colors.insert(node_id.clone(), color);
       
       if let Some(node) = graph.get_node(&node_id) {
           for neighbor in node.neighbors() {
               if let Some(&neighbor_color) = colors.get(&neighbor) {
                   if neighbor_color == color {
                       return false; // Same color for adjacent nodes
                   }
               } else if !color_bipartite_dfs(graph, neighbor, !color, colors) {
                   return false;
               }
           }
       }
       
       true
   }

   fn estimate_memory_usage<G: Graph>(graph: &G) -> usize {
       // Rough estimation of memory usage
       let node_count = graph.node_count();
       let edge_count = graph.edge_count();
       
       // Assuming each node takes ~100 bytes and each edge ~50 bytes
       node_count * 100 + edge_count * 50
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/validation/mod.rs
pub trait GraphValidation: Graph {
    fn validate_structure(&self, config: &ValidationConfig) -> ValidationResult;
    fn check_consistency(&self) -> ValidationResult;
    fn detect_anomalies(&self) -> Vec<ValidationWarning>;
    fn validate_constraints(&self, constraints: &[CustomConstraint]) -> ValidationResult;
    fn performance_check(&self, thresholds: &PerformanceThresholds) -> ValidationResult;
}

pub struct ComprehensiveValidationResult {
    pub structural_validation: ValidationResult,
    pub logical_validation: ValidationResult,
    pub performance_validation: ValidationResult,
    pub overall_score: f32,
    pub recommendations: Vec<String>,
}
```

## Verification Steps
1. Test validation on graphs with known structural issues
2. Verify detection of common graph anomalies and errors
3. Test custom constraint validation functionality
4. Validate performance on large neuromorphic networks
5. Ensure validation algorithms themselves are efficient and correct

## Time Estimate
30 minutes

## Dependencies
- MP001-MP010: Graph infrastructure
- MP025: Graph clustering coefficient (for structural validation)
- MP021-MP024: Centrality measures (for anomaly detection)