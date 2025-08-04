# MP069: Graph Property Verification

## Task Description
Implement comprehensive graph property verification framework to validate structural invariants, topological properties, and graph-theoretic constraints across all implemented algorithms.

## Prerequisites
- MP001-MP060 completed
- MP061-MP068 test frameworks implemented
- Understanding of graph theory and structural properties

## Detailed Steps

1. Create `tests/graph_property_verification/structural_invariants/mod.rs`

2. Implement basic graph property validators:
   ```rust
   use std::collections::{HashMap, HashSet, VecDeque};
   
   pub struct GraphPropertyValidator;
   
   impl GraphPropertyValidator {
       pub fn validate_basic_graph_invariants<G: Graph>(graph: &G) -> Result<(), PropertyError> {
           // Invariant 1: Node ID uniqueness
           let mut seen_nodes = HashSet::new();
           for node in graph.nodes() {
               if !seen_nodes.insert(node.id()) {
                   return Err(PropertyError::DuplicateNodeId(node.id()));
               }
           }
           
           // Invariant 2: Edge consistency (endpoints exist)
           for edge in graph.edges() {
               if !graph.contains_node(edge.source()) {
                   return Err(PropertyError::OrphanedEdge {
                       edge_source: edge.source(),
                       edge_target: edge.target(),
                       missing_node: edge.source(),
                   });
               }
               if !graph.contains_node(edge.target()) {
                   return Err(PropertyError::OrphanedEdge {
                       edge_source: edge.source(),
                       edge_target: edge.target(),
                       missing_node: edge.target(),
                   });
               }
           }
           
           // Invariant 3: Edge weight validity
           for edge in graph.edges() {
               let weight = edge.weight();
               if weight.is_nan() || weight.is_infinite() {
                   return Err(PropertyError::InvalidEdgeWeight {
                       source: edge.source(),
                       target: edge.target(),
                       weight,
                   });
               }
           }
           
           // Invariant 4: Self-loop detection (if not allowed)
           for edge in graph.edges() {
               if edge.source() == edge.target() && !graph.allows_self_loops() {
                   return Err(PropertyError::UnexpectedSelfLoop {
                       node: edge.source(),
                   });
               }
           }
           
           // Invariant 5: Multi-edge detection (if not allowed)
           let mut edge_set = HashSet::new();
           for edge in graph.edges() {
               let edge_key = (edge.source(), edge.target());
               if !edge_set.insert(edge_key) && !graph.allows_multiple_edges() {
                   return Err(PropertyError::UnexpectedMultipleEdges {
                       source: edge.source(),
                       target: edge.target(),
                   });
               }
           }
           
           Ok(())
       }
       
       pub fn validate_connectivity_properties<G: Graph>(graph: &G) -> Result<ConnectivityReport, PropertyError> {
           let mut report = ConnectivityReport::new();
           
           // Calculate connected components
           let components = Self::find_connected_components(graph);
           report.connected_components = components.len();
           report.is_connected = components.len() == 1;
           
           // Calculate strongly connected components (for directed graphs)
           if graph.is_directed() {
               let scc = Self::find_strongly_connected_components(graph);
               report.strongly_connected_components = scc.len();
               report.is_strongly_connected = scc.len() == 1;
           }
           
           // Check for bridges and articulation points
           report.bridges = Self::find_bridges(graph);
           report.articulation_points = Self::find_articulation_points(graph);
           
           // Calculate diameter and radius
           if report.is_connected {
               let distances = Self::calculate_all_pairs_shortest_paths(graph);
               report.diameter = Self::calculate_diameter(&distances);
               report.radius = Self::calculate_radius(&distances);
               report.average_path_length = Self::calculate_average_path_length(&distances);
           }
           
           Ok(report)
       }
       
       pub fn validate_degree_properties<G: Graph>(graph: &G) -> Result<DegreeReport, PropertyError> {
           let mut report = DegreeReport::new();
           let mut degree_sequence = Vec::new();
           let mut in_degree_sequence = Vec::new();
           let mut out_degree_sequence = Vec::new();
           
           for node in graph.nodes() {
               let node_id = node.id();
               
               // Calculate degrees
               let degree = graph.degree(node_id);
               let in_degree = graph.in_degree(node_id);
               let out_degree = graph.out_degree(node_id);
               
               degree_sequence.push(degree);
               in_degree_sequence.push(in_degree);
               out_degree_sequence.push(out_degree);
               
               // Update statistics
               report.max_degree = report.max_degree.max(degree);
               report.min_degree = report.min_degree.min(degree);
           }
           
           // Verify handshaking lemma: sum of degrees = 2 * number of edges
           let degree_sum: usize = degree_sequence.iter().sum();
           let expected_degree_sum = 2 * graph.edge_count();
           
           if degree_sum != expected_degree_sum {
               return Err(PropertyError::HandshakingLemmaViolation {
                   computed_degree_sum: degree_sum,
                   expected_degree_sum,
                   edge_count: graph.edge_count(),
               });
           }
           
           // For directed graphs, verify in-degree sum = out-degree sum = edge count
           if graph.is_directed() {
               let in_degree_sum: usize = in_degree_sequence.iter().sum();
               let out_degree_sum: usize = out_degree_sequence.iter().sum();
               
               if in_degree_sum != out_degree_sum || in_degree_sum != graph.edge_count() {
                   return Err(PropertyError::DirectedDegreeViolation {
                       in_degree_sum,
                       out_degree_sum,
                       edge_count: graph.edge_count(),
                   });
               }
           }
           
           // Calculate degree distribution statistics
           report.degree_distribution = Self::calculate_degree_distribution(&degree_sequence);
           report.average_degree = degree_sum as f64 / graph.node_count() as f64;
           
           Ok(report)
       }
   }
   ```

3. Create topological property verification:
   ```rust
   pub struct TopologicalPropertyValidator;
   
   impl TopologicalPropertyValidator {
       pub fn validate_planarity<G: Graph>(graph: &G) -> Result<PlanarityReport, PropertyError> {
           let mut report = PlanarityReport::new();
           
           // Quick checks for non-planarity
           let n = graph.node_count();
           let m = graph.edge_count();
           
           // Necessary condition: m <= 3n - 6 for n >= 3
           if n >= 3 && m > 3 * n - 6 {
               report.is_planar = false;
               report.reason = Some("Too many edges for planar graph".to_string());
               return Ok(report);
           }
           
           // Check for K5 or K3,3 subgraphs (Kuratowski's theorem)
           if Self::contains_k5_minor(graph) {
               report.is_planar = false;
               report.reason = Some("Contains K5 minor".to_string());
               return Ok(report);
           }
           
           if Self::contains_k33_minor(graph) {
               report.is_planar = false;
               report.reason = Some("Contains K3,3 minor".to_string());
               return Ok(report);
           }
           
           // For small graphs, use brute force planarity testing
           if n <= 20 {
               report.is_planar = Self::brute_force_planarity_test(graph);
               if !report.is_planar {
                   report.reason = Some("Failed planarity embedding test".to_string());
               }
           } else {
               // Use linear-time planarity algorithm for larger graphs
               report.is_planar = Self::hopcroft_tarjan_planarity_test(graph);
               if !report.is_planar {
                   report.reason = Some("Failed Hopcroft-Tarjan test".to_string());
               }
           }
           
           Ok(report)
       }
       
       pub fn validate_bipartiteness<G: Graph>(graph: &G) -> Result<BipartiteReport, PropertyError> {
           let mut report = BipartiteReport::new();
           let mut coloring = HashMap::new();
           let mut queue = VecDeque::new();
           
           // Try to 2-color the graph
           for start_node in graph.nodes() {
               if coloring.contains_key(&start_node.id()) {
                   continue;
               }
               
               // Start BFS from this component
               queue.push_back(start_node.id());
               coloring.insert(start_node.id(), 0);
               
               while let Some(current) = queue.pop_front() {
                   let current_color = coloring[&current];
                   let next_color = 1 - current_color;
                   
                   for neighbor in graph.neighbors(current) {
                       if let Some(&neighbor_color) = coloring.get(&neighbor) {
                           if neighbor_color == current_color {
                               report.is_bipartite = false;
                               report.odd_cycle = Some(Self::find_odd_cycle_from(graph, current, neighbor));
                               return Ok(report);
                           }
                       } else {
                           coloring.insert(neighbor, next_color);
                           queue.push_back(neighbor);
                       }
                   }
               }
           }
           
           // Successfully 2-colored
           report.is_bipartite = true;
           
           // Extract bipartition
           let mut partition_a = Vec::new();
           let mut partition_b = Vec::new();
           
           for (&node, &color) in &coloring {
               if color == 0 {
                   partition_a.push(node);
               } else {
                   partition_b.push(node);
               }
           }
           
           report.partition_a = partition_a;
           report.partition_b = partition_b;
           
           Ok(report)
       }
       
       pub fn validate_tree_properties<G: Graph>(graph: &G) -> Result<TreeReport, PropertyError> {
           let mut report = TreeReport::new();
           
           let n = graph.node_count();
           let m = graph.edge_count();
           
           // Tree property 1: Connected
           let connectivity = GraphPropertyValidator::validate_connectivity_properties(graph)?;
           report.is_connected = connectivity.is_connected;
           
           if !report.is_connected {
               report.is_tree = false;
               report.is_forest = Self::validate_forest_properties(graph);
               return Ok(report);
           }
           
           // Tree property 2: n-1 edges for n nodes
           report.has_correct_edge_count = (m == n - 1);
           
           // Tree property 3: Acyclic
           report.is_acyclic = Self::is_acyclic(graph);
           
           // A connected graph is a tree iff it has n-1 edges and is acyclic
           report.is_tree = report.is_connected && report.has_correct_edge_count && report.is_acyclic;
           
           if report.is_tree {
               // Calculate tree-specific properties
               report.leaves = Self::find_leaves(graph);
               report.diameter = Self::calculate_tree_diameter(graph);
               report.center_nodes = Self::find_tree_center(graph);
           }
           
           Ok(report)
       }
       
       pub fn validate_dag_properties<G: Graph>(graph: &G) -> Result<DagReport, PropertyError> {
           let mut report = DagReport::new();
           
           if !graph.is_directed() {
               return Err(PropertyError::UndirectedGraphForDAG);
           }
           
           // Check for cycles using DFS
           let cycle_detection = Self::detect_cycles_dfs(graph);
           report.is_acyclic = cycle_detection.is_none();
           report.cycle_found = cycle_detection;
           
           if report.is_acyclic {
               // Calculate DAG-specific properties
               report.topological_ordering = Some(Self::topological_sort(graph)?);
               report.longest_path = Self::calculate_longest_path_in_dag(graph);
               report.sources = Self::find_source_nodes(graph);
               report.sinks = Self::find_sink_nodes(graph);
               
               // Calculate levels (distance from sources)
               report.node_levels = Self::calculate_node_levels(graph);
               report.height = report.node_levels.values().max().copied().unwrap_or(0);
           }
           
           Ok(report)
       }
   }
   ```

4. Implement algorithm-specific property validation:
   ```rust
   pub struct AlgorithmPropertyValidator;
   
   impl AlgorithmPropertyValidator {
       pub fn validate_shortest_path_properties<G: Graph>(
           graph: &G,
           source: NodeId,
           distances: &HashMap<NodeId, f64>
       ) -> Result<(), PropertyError> {
           // Property 1: Source distance is 0
           if let Some(&source_dist) = distances.get(&source) {
               if (source_dist - 0.0).abs() > 1e-10 {
                   return Err(PropertyError::InvalidSourceDistance {
                       source,
                       distance: source_dist,
                   });
               }
           }
           
           // Property 2: Triangle inequality
           for (&u, &dist_u) in distances {
               for (&v, &dist_v) in distances {
                   if let Some(edge_weight) = graph.edge_weight(u, v) {
                       if dist_v > dist_u + edge_weight + 1e-10 {
                           return Err(PropertyError::TriangleInequalityViolation {
                               source,
                               intermediate: u,
                               target: v,
                               computed_distance: dist_v,
                               triangle_bound: dist_u + edge_weight,
                           });
                       }
                   }
               }
           }
           
           // Property 3: Unreachable nodes should not have distances
           let reachable_nodes = Self::find_reachable_nodes(graph, source);
           for &node in distances.keys() {
               if !reachable_nodes.contains(&node) {
                   return Err(PropertyError::UnreachableNodeWithDistance {
                       source,
                       unreachable_node: node,
                   });
               }
           }
           
           Ok(())
       }
       
       pub fn validate_spanning_tree_properties<G: Graph>(
           original_graph: &G,
           spanning_tree: &G
       ) -> Result<(), PropertyError> {
           // Property 1: Same node set
           let original_nodes: HashSet<_> = original_graph.nodes().map(|n| n.id()).collect();
           let tree_nodes: HashSet<_> = spanning_tree.nodes().map(|n| n.id()).collect();
           
           if original_nodes != tree_nodes {
               return Err(PropertyError::SpanningTreeNodeMismatch {
                   original_count: original_nodes.len(),
                   tree_count: tree_nodes.len(),
               });
           }
           
           // Property 2: Tree has n-1 edges
           let n = original_graph.node_count();
           if spanning_tree.edge_count() != n - 1 {
               return Err(PropertyError::SpanningTreeEdgeCount {
                   expected: n - 1,
                   actual: spanning_tree.edge_count(),
               });
           }
           
           // Property 3: Tree is connected
           let connectivity = GraphPropertyValidator::validate_connectivity_properties(spanning_tree)?;
           if !connectivity.is_connected {
               return Err(PropertyError::SpanningTreeNotConnected);
           }
           
           // Property 4: Tree is acyclic
           if !TopologicalPropertyValidator::is_acyclic(spanning_tree) {
               return Err(PropertyError::SpanningTreeHasCycle);
           }
           
           // Property 5: All edges exist in original graph
           for edge in spanning_tree.edges() {
               if !original_graph.has_edge(edge.source(), edge.target()) {
                   return Err(PropertyError::SpanningTreeInvalidEdge {
                       source: edge.source(),
                       target: edge.target(),
                   });
               }
           }
           
           Ok(())
       }
       
       pub fn validate_flow_properties<G: Graph>(
           graph: &G,
           flow: &HashMap<(NodeId, NodeId), f64>,
           source: NodeId,
           sink: NodeId
       ) -> Result<FlowValidationReport, PropertyError> {
           let mut report = FlowValidationReport::new();
           
           // Property 1: Capacity constraints
           for (&(u, v), &flow_value) in flow {
               if let Some(capacity) = graph.edge_weight(u, v) {
                   if flow_value > capacity + 1e-10 {
                       return Err(PropertyError::FlowCapacityViolation {
                           edge: (u, v),
                           flow: flow_value,
                           capacity,
                       });
                   }
                   if flow_value < -1e-10 {
                       return Err(PropertyError::NegativeFlow {
                           edge: (u, v),
                           flow: flow_value,
                       });
                   }
               }
           }
           
           // Property 2: Flow conservation
           for node in graph.nodes() {
               let node_id = node.id();
               
               if node_id == source || node_id == sink {
                   continue; // Skip source and sink
               }
               
               let mut inflow = 0.0;
               let mut outflow = 0.0;
               
               for (&(u, v), &flow_value) in flow {
                   if v == node_id {
                       inflow += flow_value;
                   }
                   if u == node_id {
                       outflow += flow_value;
                   }
               }
               
               if (inflow - outflow).abs() > 1e-10 {
                   return Err(PropertyError::FlowConservationViolation {
                       node: node_id,
                       inflow,
                       outflow,
                   });
               }
           }
           
           // Calculate total flow value
           let mut source_outflow = 0.0;
           let mut sink_inflow = 0.0;
           
           for (&(u, v), &flow_value) in flow {
               if u == source {
                   source_outflow += flow_value;
               }
               if v == sink {
                   sink_inflow += flow_value;
               }
           }
           
           report.flow_value = source_outflow;
           
           // Property 3: Source outflow equals sink inflow
           if (source_outflow - sink_inflow).abs() > 1e-10 {
               return Err(PropertyError::FlowValueMismatch {
                   source_outflow,
                   sink_inflow,
               });
           }
           
           Ok(report)
       }
   }
   ```

5. Create comprehensive property test suite:
   ```rust
   pub struct ComprehensivePropertyTester;
   
   impl ComprehensivePropertyTester {
       pub fn run_full_property_validation<G: Graph>(graph: &G) -> PropertyTestReport {
           let mut report = PropertyTestReport::new();
           
           // Basic invariants
           match GraphPropertyValidator::validate_basic_graph_invariants(graph) {
               Ok(_) => report.basic_invariants = PropertyTestResult::Passed,
               Err(e) => report.basic_invariants = PropertyTestResult::Failed(e.to_string()),
           }
           
           // Connectivity properties
           match GraphPropertyValidator::validate_connectivity_properties(graph) {
               Ok(connectivity_report) => {
                   report.connectivity = PropertyTestResult::Passed;
                   report.connectivity_details = Some(connectivity_report);
               }
               Err(e) => report.connectivity = PropertyTestResult::Failed(e.to_string()),
           }
           
           // Degree properties
           match GraphPropertyValidator::validate_degree_properties(graph) {
               Ok(degree_report) => {
                   report.degree_properties = PropertyTestResult::Passed;
                   report.degree_details = Some(degree_report);
               }
               Err(e) => report.degree_properties = PropertyTestResult::Failed(e.to_string()),
           }
           
           // Topological properties
           match TopologicalPropertyValidator::validate_planarity(graph) {
               Ok(planarity_report) => {
                   report.planarity = PropertyTestResult::Passed;
                   report.planarity_details = Some(planarity_report);
               }
               Err(e) => report.planarity = PropertyTestResult::Failed(e.to_string()),
           }
           
           match TopologicalPropertyValidator::validate_bipartiteness(graph) {
               Ok(bipartite_report) => {
                   report.bipartiteness = PropertyTestResult::Passed;
                   report.bipartite_details = Some(bipartite_report);
               }
               Err(e) => report.bipartiteness = PropertyTestResult::Failed(e.to_string()),
           }
           
           // Tree properties (if applicable)
           match TopologicalPropertyValidator::validate_tree_properties(graph) {
               Ok(tree_report) => {
                   report.tree_properties = PropertyTestResult::Passed;
                   report.tree_details = Some(tree_report);
               }
               Err(e) => report.tree_properties = PropertyTestResult::Failed(e.to_string()),
           }
           
           // DAG properties (if directed)
           if graph.is_directed() {
               match TopologicalPropertyValidator::validate_dag_properties(graph) {
                   Ok(dag_report) => {
                       report.dag_properties = PropertyTestResult::Passed;
                       report.dag_details = Some(dag_report);
                   }
                   Err(e) => report.dag_properties = PropertyTestResult::Failed(e.to_string()),
               }
           }
           
           report
       }
       
       pub fn validate_algorithm_output_properties(
           graph: &TestGraph,
           algorithm_results: &AlgorithmResults
       ) -> AlgorithmPropertyReport {
           let mut report = AlgorithmPropertyReport::new();
           
           // Validate shortest path results
           if let Some(shortest_paths) = &algorithm_results.shortest_paths {
               for (source, distances) in shortest_paths {
                   match AlgorithmPropertyValidator::validate_shortest_path_properties(
                       graph, *source, distances
                   ) {
                       Ok(_) => report.shortest_path_valid = true,
                       Err(e) => {
                           report.shortest_path_valid = false;
                           report.shortest_path_errors.push(e);
                       }
                   }
               }
           }
           
           // Validate spanning tree results
           if let Some(spanning_tree) = &algorithm_results.spanning_tree {
               match AlgorithmPropertyValidator::validate_spanning_tree_properties(
                   graph, spanning_tree
               ) {
                   Ok(_) => report.spanning_tree_valid = true,
                   Err(e) => {
                       report.spanning_tree_valid = false;
                       report.spanning_tree_error = Some(e);
                   }
               }
           }
           
           // Validate flow results
           if let Some((flow, source, sink)) = &algorithm_results.max_flow {
               match AlgorithmPropertyValidator::validate_flow_properties(
                   graph, flow, *source, *sink
               ) {
                   Ok(flow_report) => {
                       report.flow_valid = true;
                       report.flow_details = Some(flow_report);
                   }
                   Err(e) => {
                       report.flow_valid = false;
                       report.flow_error = Some(e);
                   }
               }
           }
           
           report
       }
   }
   ```

## Expected Output
```rust
#[cfg(test)]
mod graph_property_tests {
    use super::*;
    
    #[test]
    fn test_basic_graph_invariants() {
        let graphs = create_test_graph_suite();
        
        for (name, graph) in graphs {
            let result = GraphPropertyValidator::validate_basic_graph_invariants(&graph);
            assert!(result.is_ok(), "Basic invariants failed for graph {}: {:?}", name, result.err());
        }
    }
    
    #[test]
    fn test_connectivity_properties() {
        let graphs = create_connectivity_test_graphs();
        
        for (name, graph) in graphs {
            let result = GraphPropertyValidator::validate_connectivity_properties(&graph);
            assert!(result.is_ok(), "Connectivity validation failed for graph {}: {:?}", name, result.err());
        }
    }
    
    #[test]
    fn test_topological_properties() {
        let planar_graph = create_planar_test_graph();
        let planarity_result = TopologicalPropertyValidator::validate_planarity(&planar_graph);
        assert!(planarity_result.is_ok());
        
        let bipartite_graph = create_bipartite_test_graph();
        let bipartite_result = TopologicalPropertyValidator::validate_bipartiteness(&bipartite_graph);
        assert!(bipartite_result.is_ok());
    }
    
    #[test]
    fn test_algorithm_output_properties() {
        let graph = create_comprehensive_test_graph();
        let algorithm_results = run_all_algorithms(&graph);
        
        let report = ComprehensivePropertyTester::validate_algorithm_output_properties(
            &graph, &algorithm_results
        );
        
        assert!(report.all_valid(), "Algorithm output property validation failed: {:?}", report);
    }
    
    #[test]
    fn test_comprehensive_property_validation() {
        let test_graphs = create_diverse_graph_collection();
        
        for (name, graph) in test_graphs {
            let report = ComprehensivePropertyTester::run_full_property_validation(&graph);
            assert!(
                report.all_passed(),
                "Comprehensive property validation failed for graph {}: {:?}",
                name, report
            );
        }
    }
}
```

## Verification Steps
1. Execute graph property validation on diverse graph types
2. Verify structural invariants across all test graphs
3. Test topological property detection accuracy
4. Validate algorithm output property consistency
5. Check edge cases and boundary conditions
6. Ensure comprehensive coverage of graph theory properties

## Time Estimate
35 minutes

## Dependencies
- MP001-MP060: All algorithm implementations
- MP061-MP068: Test framework infrastructure
- Graph theory verification utilities
- Topological analysis tools
- Mathematical property validation libraries