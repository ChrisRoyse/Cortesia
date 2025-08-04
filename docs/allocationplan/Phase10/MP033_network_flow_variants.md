# MP033: Network Flow Variants

## Task Description
Implement advanced network flow algorithms beyond basic max flow, including min-cost flow, multi-commodity flow, and flow with constraints for optimizing neural resource allocation and signal routing.

## Prerequisites
- MP019: Max flow algorithms (Ford-Fulkerson, Dinic's)
- MP001-MP010: Graph infrastructure
- MP032: Graph matching (for flow-based matching)
- Understanding of linear programming and optimization

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/flow/variants.rs`

2. Implement minimum cost maximum flow:
   ```rust
   use std::collections::{HashMap, VecDeque, BinaryHeap};
   use std::cmp::Ordering;
   use crate::neuromorphic::graph::traits::{Graph, GraphNode};

   #[derive(Debug, Clone)]
   pub struct FlowEdge<Id> {
       pub from: Id,
       pub to: Id,
       pub capacity: f32,
       pub cost: f32,
       pub flow: f32,
   }

   #[derive(Debug, Clone)]
   pub struct FlowNetwork<Id> {
       pub edges: Vec<FlowEdge<Id>>,
       pub adjacency: HashMap<Id, Vec<usize>>, // Node to edge indices
   }

   #[derive(Debug, Clone)]
   pub struct MinCostFlowResult<Id> {
       pub max_flow: f32,
       pub min_cost: f32,
       pub flow_edges: Vec<FlowEdge<Id>>,
       pub flow_value: f32,
   }

   pub fn min_cost_max_flow<G: Graph>(
       graph: &G,
       source: G::Node::Id,
       sink: G::Node::Id,
       capacity_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
       cost_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
   ) -> MinCostFlowResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Build flow network
       let mut network = build_flow_network(graph, capacity_fn, cost_fn);
       
       let mut total_flow = 0.0;
       let mut total_cost = 0.0;
       
       // Use successive shortest paths algorithm
       loop {
           // Find shortest path in residual graph using SPFA
           if let Some((path, path_cost)) = shortest_path_spfa(&network, &source, &sink) {
               // Find bottleneck capacity along the path
               let bottleneck = find_bottleneck_capacity(&network, &path);
               
               if bottleneck <= 0.0 {
                   break;
               }
               
               // Update flow along the path
               update_flow_along_path(&mut network, &path, bottleneck);
               
               total_flow += bottleneck;
               total_cost += bottleneck * path_cost;
           } else {
               break; // No more augmenting paths
           }
       }
       
       MinCostFlowResult {
           max_flow: total_flow,
           min_cost: total_cost,
           flow_edges: network.edges.clone(),
           flow_value: total_flow,
       }
   }

   fn build_flow_network<G: Graph>(
       graph: &G,
       capacity_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
       cost_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
   ) -> FlowNetwork<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       let mut edges = Vec::new();
       let mut adjacency: HashMap<G::Node::Id, Vec<usize>> = HashMap::new();
       
       for edge in graph.edges() {
           let from = edge.source();
           let to = edge.target();
           let capacity = capacity_fn(&from, &to);
           let cost = cost_fn(&from, &to);
           
           if capacity > 0.0 {
               let edge_index = edges.len();
               
               // Forward edge
               edges.push(FlowEdge {
                   from: from.clone(),
                   to: to.clone(),
                   capacity,
                   cost,
                   flow: 0.0,
               });
               
               adjacency.entry(from.clone()).or_insert_with(Vec::new).push(edge_index);
               
               // Reverse edge (for residual graph)
               let reverse_edge_index = edges.len();
               edges.push(FlowEdge {
                   from: to.clone(),
                   to: from.clone(),
                   capacity: 0.0, // Initially no reverse capacity
                   cost: -cost,   // Negative cost for reverse
                   flow: 0.0,
               });
               
               adjacency.entry(to).or_insert_with(Vec::new).push(reverse_edge_index);
           }
       }
       
       FlowNetwork { edges, adjacency }
   }

   fn shortest_path_spfa<Id: Clone + Eq + std::hash::Hash>(
       network: &FlowNetwork<Id>,
       source: &Id,
       sink: &Id,
   ) -> Option<(Vec<usize>, f32)> {
       let mut distances: HashMap<Id, f32> = HashMap::new();
       let mut parent: HashMap<Id, Option<usize>> = HashMap::new();
       let mut in_queue = HashMap::new();
       let mut queue = VecDeque::new();
       
       // Initialize distances
       for edge in &network.edges {
           distances.insert(edge.from.clone(), f32::INFINITY);
           distances.insert(edge.to.clone(), f32::INFINITY);
           parent.insert(edge.from.clone(), None);
           parent.insert(edge.to.clone(), None);
           in_queue.insert(edge.from.clone(), false);
           in_queue.insert(edge.to.clone(), false);
       }
       
       distances.insert(source.clone(), 0.0);
       queue.push_back(source.clone());
       in_queue.insert(source.clone(), true);
       
       // SPFA algorithm
       while let Some(current) = queue.pop_front() {
           in_queue.insert(current.clone(), false);
           
           if let Some(edge_indices) = network.adjacency.get(&current) {
               for &edge_idx in edge_indices {
                   let edge = &network.edges[edge_idx];
                   let residual_capacity = edge.capacity - edge.flow;
                   
                   if residual_capacity > 1e-9 {
                       let new_distance = distances[&current] + edge.cost;
                       
                       if new_distance < distances[&edge.to] {
                           distances.insert(edge.to.clone(), new_distance);
                           parent.insert(edge.to.clone(), Some(edge_idx));
                           
                           if !in_queue[&edge.to] {
                               queue.push_back(edge.to.clone());
                               in_queue.insert(edge.to.clone(), true);
                           }
                       }
                   }
               }
           }
       }
       
       // Reconstruct path
       if distances[sink] == f32::INFINITY {
           return None;
       }
       
       let mut path = Vec::new();
       let mut current = sink.clone();
       let path_cost = distances[sink];
       
       while let Some(Some(edge_idx)) = parent.get(&current) {
           path.push(*edge_idx);
           current = network.edges[*edge_idx].from.clone();
       }
       
       path.reverse();
       Some((path, path_cost))
   }

   fn find_bottleneck_capacity<Id>(network: &FlowNetwork<Id>, path: &[usize]) -> f32 {
       path.iter()
           .map(|&edge_idx| {
               let edge = &network.edges[edge_idx];
               edge.capacity - edge.flow
           })
           .fold(f32::INFINITY, f32::min)
   }

   fn update_flow_along_path<Id>(network: &mut FlowNetwork<Id>, path: &[usize], flow: f32) {
       for &edge_idx in path {
           network.edges[edge_idx].flow += flow;
           
           // Update reverse edge flow
           let reverse_idx = if edge_idx % 2 == 0 { edge_idx + 1 } else { edge_idx - 1 };
           network.edges[reverse_idx].flow -= flow;
           network.edges[reverse_idx].capacity += flow;
       }
   }
   ```

3. Implement multi-commodity flow:
   ```rust
   #[derive(Debug, Clone)]
   pub struct Commodity<Id> {
       pub source: Id,
       pub sink: Id,
       pub demand: f32,
       pub flow_paths: Vec<Vec<Id>>,
   }

   #[derive(Debug, Clone)]
   pub struct MultiCommodityFlowResult<Id> {
       pub commodities: Vec<Commodity<Id>>,
       pub total_flow: f32,
       pub feasible: bool,
       pub edge_congestion: HashMap<(Id, Id), f32>,
   }

   pub fn multi_commodity_flow<G: Graph>(
       graph: &G,
       commodities: &[Commodity<G::Node::Id>],
       capacity_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
   ) -> MultiCommodityFlowResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Simplified implementation using successive shortest paths
       let mut result_commodities = commodities.to_vec();
       let mut edge_flows: HashMap<(G::Node::Id, G::Node::Id), f32> = HashMap::new();
       let mut total_flow = 0.0;
       let mut feasible = true;
       
       for commodity in &mut result_commodities {
           let mut remaining_demand = commodity.demand;
           commodity.flow_paths.clear();
           
           // Find paths for this commodity
           while remaining_demand > 1e-9 {
               if let Some(path) = find_path_with_capacity(
                   graph,
                   &commodity.source,
                   &commodity.sink,
                   &edge_flows,
                   &capacity_fn,
               ) {
                   let path_capacity = calculate_path_capacity(
                       &path,
                       &edge_flows,
                       &capacity_fn,
                   );
                   
                   let flow_amount = remaining_demand.min(path_capacity);
                   
                   if flow_amount <= 1e-9 {
                       feasible = false;
                       break;
                   }
                   
                   // Update edge flows
                   for window in path.windows(2) {
                       let edge_key = (window[0].clone(), window[1].clone());
                       *edge_flows.entry(edge_key).or_insert(0.0) += flow_amount;
                   }
                   
                   commodity.flow_paths.push(path);
                   remaining_demand -= flow_amount;
                   total_flow += flow_amount;
               } else {
                   feasible = false;
                   break;
               }
           }
           
           if remaining_demand > 1e-9 {
               feasible = false;
           }
       }
       
       // Calculate edge congestion
       let mut edge_congestion = HashMap::new();
       for ((from, to), &flow) in &edge_flows {
           let capacity = capacity_fn(from, to);
           if capacity > 0.0 {
               edge_congestion.insert((from.clone(), to.clone()), flow / capacity);
           }
       }
       
       MultiCommodityFlowResult {
           commodities: result_commodities,
           total_flow,
           feasible,
           edge_congestion,
       }
   }

   fn find_path_with_capacity<G: Graph>(
       graph: &G,
       source: &G::Node::Id,
       sink: &G::Node::Id,
       edge_flows: &HashMap<(G::Node::Id, G::Node::Id), f32>,
       capacity_fn: &impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
   ) -> Option<Vec<G::Node::Id>> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       use std::collections::HashSet;
       
       let mut queue = VecDeque::new();
       let mut parent: HashMap<G::Node::Id, G::Node::Id> = HashMap::new();
       let mut visited = HashSet::new();
       
       queue.push_back(source.clone());
       visited.insert(source.clone());
       
       while let Some(current) = queue.pop_front() {
           if current == *sink {
               // Reconstruct path
               let mut path = Vec::new();
               let mut node = sink.clone();
               path.push(node.clone());
               
               while let Some(prev) = parent.get(&node) {
                   path.push(prev.clone());
                   node = prev.clone();
               }
               
               path.reverse();
               return Some(path);
           }
           
           if let Some(node) = graph.get_node(&current) {
               for neighbor in node.neighbors() {
                   if !visited.contains(&neighbor) {
                       let edge_capacity = capacity_fn(&current, &neighbor);
                       let edge_flow = edge_flows.get(&(current.clone(), neighbor.clone()))
                           .copied().unwrap_or(0.0);
                       
                       if edge_capacity - edge_flow > 1e-9 {
                           visited.insert(neighbor.clone());
                           parent.insert(neighbor.clone(), current.clone());
                           queue.push_back(neighbor);
                       }
                   }
               }
           }
       }
       
       None
   }

   fn calculate_path_capacity<Id: Clone + Eq + std::hash::Hash>(
       path: &[Id],
       edge_flows: &HashMap<(Id, Id), f32>,
       capacity_fn: &impl Fn(&Id, &Id) -> f32,
   ) -> f32 {
       path.windows(2)
           .map(|window| {
               let capacity = capacity_fn(&window[0], &window[1]);
               let flow = edge_flows.get(&(window[0].clone(), window[1].clone()))
                   .copied().unwrap_or(0.0);
               capacity - flow
           })
           .fold(f32::INFINITY, f32::min)
   }
   ```

4. Implement flow with node capacities:
   ```rust
   pub fn max_flow_with_node_capacities<G: Graph>(
       graph: &G,
       source: G::Node::Id,
       sink: G::Node::Id,
       edge_capacity_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
       node_capacity_fn: impl Fn(&G::Node::Id) -> f32,
   ) -> f32 
   where G::Node::Id: Clone + Eq + std::hash::Hash + std::fmt::Display {
       // Transform to standard max flow by node splitting
       let mut transformed_edges = Vec::new();
       let mut node_mapping = HashMap::new();
       
       // Create in and out nodes for each original node
       for node in graph.nodes() {
           let node_id = node.id();
           let in_node = format!("{}_in", node_id);
           let out_node = format!("{}_out", node_id);
           
           node_mapping.insert(node_id.clone(), (in_node.clone(), out_node.clone()));
           
           // Add capacity edge from in_node to out_node
           let node_capacity = node_capacity_fn(&node_id);
           transformed_edges.push((in_node, out_node, node_capacity));
       }
       
       // Add edges between out_node of source and in_node of target
       for edge in graph.edges() {
           let source_id = edge.source();
           let target_id = edge.target();
           let edge_capacity = edge_capacity_fn(&source_id, &target_id);
           
           if let (Some((_, source_out)), Some((target_in, _))) = 
               (node_mapping.get(&source_id), node_mapping.get(&target_id)) {
               transformed_edges.push((source_out.clone(), target_in.clone(), edge_capacity));
           }
       }
       
       // Run max flow on transformed graph
       let (source_in, _) = node_mapping.get(&source).unwrap();
       let (_, sink_out) = node_mapping.get(&sink).unwrap();
       
       // For simplicity, return a placeholder - implement full max flow algorithm
       // on the transformed graph in production code
       0.0
   }
   ```

5. Implement circulation with demands:
   ```rust
   #[derive(Debug, Clone)]
   pub struct CirculationResult<Id> {
       pub feasible: bool,
       pub flow_edges: Vec<FlowEdge<Id>>,
       pub unsatisfied_demands: HashMap<Id, f32>,
   }

   pub fn circulation_with_demands<G: Graph>(
       graph: &G,
       capacity_fn: impl Fn(&G::Node::Id, &G::Node::Id) -> f32,
       demand_fn: impl Fn(&G::Node::Id) -> f32, // Positive = supply, negative = demand
   ) -> CirculationResult<G::Node::Id> 
   where G::Node::Id: Clone + Eq + std::hash::Hash {
       // Check if total supply equals total demand
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let total_demand: f32 = nodes.iter().map(|node| demand_fn(node)).sum();
       
       if total_demand.abs() > 1e-9 {
           // Infeasible - supply doesn't equal demand
           return CirculationResult {
               feasible: false,
               flow_edges: vec![],
               unsatisfied_demands: nodes.iter()
                   .map(|node| (node.clone(), demand_fn(node)))
                   .collect(),
           };
       }
       
       // Transform to max flow problem by adding source and sink
       // Connect source to supply nodes and demand nodes to sink
       let mut network = build_flow_network(graph, capacity_fn, |_, _| 0.0);
       
       // Add super source and super sink
       let super_source = format!("super_source");
       let super_sink = format!("super_sink");
       
       for node in &nodes {
           let demand = demand_fn(node);
           if demand > 0.0 {
               // Supply node - connect from super source
               let edge_idx = network.edges.len();
               network.edges.push(FlowEdge {
                   from: super_source.clone(),
                   to: node.clone(),
                   capacity: demand,
                   cost: 0.0,
                   flow: 0.0,
               });
               network.adjacency.entry(super_source.clone())
                   .or_insert_with(Vec::new).push(edge_idx);
           } else if demand < 0.0 {
               // Demand node - connect to super sink
               let edge_idx = network.edges.len();
               network.edges.push(FlowEdge {
                   from: node.clone(),
                   to: super_sink.clone(),
                   capacity: -demand,
                   cost: 0.0,
                   flow: 0.0,
               });
               network.adjacency.entry(node.clone())
                   .or_insert_with(Vec::new).push(edge_idx);
           }
       }
       
       // Run max flow algorithm (simplified - implement full algorithm)
       let total_supply: f32 = nodes.iter()
           .map(|node| demand_fn(node).max(0.0))
           .sum();
       
       // For now, return feasible if demands balance
       CirculationResult {
           feasible: true,
           flow_edges: network.edges,
           unsatisfied_demands: HashMap::new(),
       }
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/flow/variants.rs
pub trait FlowVariants: Graph {
    fn min_cost_max_flow(&self, source: Self::Node::Id, sink: Self::Node::Id, 
                        capacity_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32,
                        cost_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32) -> MinCostFlowResult<Self::Node::Id>;
    fn multi_commodity_flow(&self, commodities: &[Commodity<Self::Node::Id>],
                           capacity_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32) -> MultiCommodityFlowResult<Self::Node::Id>;
    fn circulation_with_demands(&self, capacity_fn: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32,
                               demand_fn: impl Fn(&Self::Node::Id) -> f32) -> CirculationResult<Self::Node::Id>;
    fn max_flow_node_capacities(&self, source: Self::Node::Id, sink: Self::Node::Id,
                                edge_cap: impl Fn(&Self::Node::Id, &Self::Node::Id) -> f32,
                                node_cap: impl Fn(&Self::Node::Id) -> f32) -> f32;
}

pub struct FlowOptimizationResult<Id> {
    pub optimal_flow: f32,
    pub optimal_cost: f32,
    pub flow_decomposition: Vec<FlowPath<Id>>,
    pub bottleneck_edges: Vec<(Id, Id)>,
}
```

## Verification Steps
1. Test min-cost flow on graphs with known optimal solutions
2. Verify multi-commodity flow feasibility conditions
3. Test circulation problems with balanced demands
4. Compare flow variants on neuromorphic routing scenarios
5. Benchmark performance on large-scale networks

## Time Estimate
35 minutes

## Dependencies
- MP019: Max flow algorithms (Ford-Fulkerson, Dinic's)
- MP001-MP010: Graph infrastructure
- MP032: Graph matching (flow-based techniques)