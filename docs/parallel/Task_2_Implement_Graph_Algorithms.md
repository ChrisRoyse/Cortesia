# Task 2: Implement Cognitive Patterns as Graph Algorithms

## Overview
Each cognitive pattern should be implemented as a pure graph algorithm that operates on the knowledge graph structure. No external LLM calls, just intelligent graph traversal and analysis.

## Core Graph Operations Needed

### Base Graph Operations
```rust
// These should be available in BrainEnhancedGraph
trait GraphOperations {
    // Navigation
    async fn get_neighbors(&self, entity: EntityKey) -> Result<Vec<EntityKey>>;
    async fn get_relationships(&self, from: EntityKey, to: Option<EntityKey>) -> Result<Vec<Relationship>>;
    async fn find_paths(&self, from: EntityKey, to: EntityKey, max_depth: usize) -> Result<Vec<Path>>;
    
    // Analysis  
    async fn get_subgraph(&self, center: EntityKey, radius: usize) -> Result<Subgraph>;
    async fn compute_centrality(&self, entity: EntityKey) -> Result<f32>;
    async fn find_clusters(&self) -> Result<Vec<Cluster>>;
    
    // Pattern Matching
    async fn find_pattern(&self, pattern: GraphPattern) -> Result<Vec<Match>>;
    async fn structural_similarity(&self, subgraph_a: &Subgraph, subgraph_b: &Subgraph) -> Result<f32>;
}
```

## Cognitive Pattern Implementations

### 1. Convergent Thinking - Finding Common Patterns
**File**: `src/cognitive/convergent.rs`

```rust
pub struct ConvergentThinking {
    graph: Arc<BrainEnhancedGraph>,
    config: ConvergentConfig,
}

impl ConvergentThinking {
    pub async fn find_convergence(&self, concepts: Vec<EntityKey>) -> Result<ConvergentResult> {
        // Step 1: Find common ancestors
        let common_ancestors = self.find_common_ancestors(&concepts).await?;
        
        // Step 2: Identify shared properties
        let shared_properties = self.extract_shared_properties(&concepts).await?;
        
        // Step 3: Analyze relationship patterns
        let relationship_patterns = self.analyze_relationship_patterns(&concepts).await?;
        
        // Step 4: Calculate convergence point
        let convergence = self.calculate_convergence_point(
            &common_ancestors,
            &shared_properties,
            &relationship_patterns
        ).await?;
        
        Ok(ConvergentResult {
            convergence_point: convergence.entity,
            confidence: convergence.score,
            supporting_patterns: convergence.patterns,
            synthesis: self.synthesize_finding(convergence),
        })
    }
    
    async fn find_common_ancestors(&self, concepts: &[EntityKey]) -> Result<Vec<EntityKey>> {
        let mut ancestor_counts = HashMap::new();
        
        for concept in concepts {
            let ancestors = self.graph.find_ancestors(*concept, 5).await?;
            for ancestor in ancestors {
                *ancestor_counts.entry(ancestor).or_insert(0) += 1;
            }
        }
        
        // Return ancestors that appear for most concepts
        let threshold = concepts.len() * 3 / 4; // 75% threshold
        Ok(ancestor_counts.into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(ancestor, _)| ancestor)
            .collect())
    }
}
```

### 2. Divergent Thinking - Exploring Possibilities
**File**: `src/cognitive/divergent.rs`

```rust
impl DivergentThinking {
    pub async fn generate_ideas(&self, seed: EntityKey) -> Result<DivergentResult> {
        let mut ideas = Vec::new();
        let mut explored = HashSet::new();
        
        // Strategy 1: Direct associations
        let direct_ideas = self.explore_direct_associations(seed).await?;
        ideas.extend(direct_ideas);
        
        // Strategy 2: Analogical connections
        let analogies = self.find_analogical_connections(seed).await?;
        ideas.extend(analogies);
        
        // Strategy 3: Combinatorial exploration
        let combinations = self.explore_combinations(seed).await?;
        ideas.extend(combinations);
        
        // Strategy 4: Inverse relationships
        let inversions = self.explore_inversions(seed).await?;
        ideas.extend(inversions);
        
        Ok(DivergentResult {
            ideas: self.rank_ideas(ideas),
            exploration_breadth: explored.len(),
            novelty_score: self.calculate_novelty(&ideas),
        })
    }
    
    async fn explore_direct_associations(&self, seed: EntityKey) -> Result<Vec<Idea>> {
        let neighbors = self.graph.get_neighbors(seed).await?;
        let mut ideas = Vec::new();
        
        for neighbor in neighbors {
            let relationship = self.graph.get_relationship(seed, neighbor).await?;
            let idea = Idea {
                concept: neighbor,
                connection_type: relationship.relation_type,
                novelty: self.calculate_edge_novelty(&relationship),
                description: format!("{:?} {} {:?}", seed, relationship.relation_type, neighbor),
            };
            ideas.push(idea);
        }
        
        Ok(ideas)
    }
}
```

### 3. Lateral Thinking - Finding Creative Connections
**File**: `src/cognitive/lateral.rs`

```rust
impl LateralThinking {
    pub async fn find_lateral_connections(&self, concept: EntityKey) -> Result<LateralResult> {
        // Find structurally similar subgraphs in different domains
        let source_subgraph = self.graph.get_subgraph(concept, 2).await?;
        let source_domain = self.identify_domain(&source_subgraph).await?;
        
        let mut lateral_connections = Vec::new();
        
        // Search for similar patterns in other domains
        let other_domains = self.graph.get_domains().await?
            .into_iter()
            .filter(|d| d != &source_domain)
            .collect::<Vec<_>>();
            
        for domain in other_domains {
            let similar_patterns = self.find_similar_patterns_in_domain(
                &source_subgraph,
                &domain
            ).await?;
            
            for pattern in similar_patterns {
                let connection = LateralConnection {
                    source: concept,
                    target: pattern.center_entity,
                    similarity_score: pattern.similarity,
                    bridge_concepts: self.find_bridge_concepts(&source_subgraph, &pattern).await?,
                    insight: self.generate_insight(&source_subgraph, &pattern),
                };
                lateral_connections.push(connection);
            }
        }
        
        Ok(LateralResult {
            connections: lateral_connections,
            cross_domain_insights: self.synthesize_insights(&lateral_connections),
        })
    }
}
```

### 4. Systems Thinking - Understanding Whole Systems
**File**: `src/cognitive/systems.rs`

```rust
impl SystemsThinking {
    pub async fn analyze_system(&self, entry_point: EntityKey) -> Result<SystemsResult> {
        // Identify system boundaries
        let system_boundary = self.identify_system_boundary(entry_point).await?;
        
        // Find feedback loops
        let feedback_loops = self.find_feedback_loops(&system_boundary).await?;
        
        // Identify key leverage points
        let leverage_points = self.find_leverage_points(&system_boundary).await?;
        
        // Analyze emergent properties
        let emergent_properties = self.analyze_emergent_properties(&system_boundary).await?;
        
        // Detect system dynamics
        let dynamics = self.analyze_system_dynamics(&system_boundary, &feedback_loops).await?;
        
        Ok(SystemsResult {
            system_map: system_boundary,
            feedback_loops,
            leverage_points,
            emergent_properties,
            dynamics,
            health_score: self.calculate_system_health(&system_boundary, &feedback_loops),
        })
    }
    
    async fn find_feedback_loops(&self, boundary: &SystemBoundary) -> Result<Vec<FeedbackLoop>> {
        let mut loops = Vec::new();
        
        for entity in &boundary.entities {
            // Use DFS to find cycles that include this entity
            let cycles = self.find_cycles_from(*entity, boundary).await?;
            
            for cycle in cycles {
                if self.is_feedback_loop(&cycle).await? {
                    let loop_type = self.classify_feedback_loop(&cycle).await?;
                    loops.push(FeedbackLoop {
                        entities: cycle,
                        loop_type,
                        strength: self.calculate_loop_strength(&cycle).await?,
                    });
                }
            }
        }
        
        Ok(loops)
    }
}
```

### 5. Critical Thinking - Evaluating Arguments
**File**: `src/cognitive/critical.rs`

```rust
impl CriticalThinking {
    pub async fn evaluate_claim(&self, claim: EntityKey) -> Result<CriticalResult> {
        // Find supporting evidence
        let supporting_evidence = self.find_supporting_evidence(claim).await?;
        
        // Find contradicting evidence
        let contradicting_evidence = self.find_contradicting_evidence(claim).await?;
        
        // Check logical consistency
        let logical_analysis = self.analyze_logical_consistency(claim).await?;
        
        // Identify assumptions
        let assumptions = self.identify_assumptions(claim).await?;
        
        // Evaluate source credibility
        let source_credibility = self.evaluate_sources(&supporting_evidence).await?;
        
        Ok(CriticalResult {
            claim,
            validity_score: self.calculate_validity(
                &supporting_evidence,
                &contradicting_evidence,
                &logical_analysis
            ),
            supporting_evidence,
            contradicting_evidence,
            assumptions,
            logical_issues: logical_analysis.issues,
            recommendation: self.generate_recommendation(&logical_analysis),
        })
    }
    
    async fn analyze_logical_consistency(&self, claim: EntityKey) -> Result<LogicalAnalysis> {
        let claim_subgraph = self.graph.get_subgraph(claim, 3).await?;
        let mut issues = Vec::new();
        
        // Check for circular reasoning
        let circular_paths = self.find_circular_dependencies(&claim_subgraph).await?;
        if !circular_paths.is_empty() {
            issues.push(LogicalIssue::CircularReasoning(circular_paths));
        }
        
        // Check for contradictions
        let contradictions = self.find_contradictions(&claim_subgraph).await?;
        if !contradictions.is_empty() {
            issues.push(LogicalIssue::Contradiction(contradictions));
        }
        
        // Check for missing links
        let gaps = self.find_reasoning_gaps(&claim_subgraph).await?;
        if !gaps.is_empty() {
            issues.push(LogicalIssue::MissingEvidence(gaps));
        }
        
        Ok(LogicalAnalysis {
            is_consistent: issues.is_empty(),
            issues,
            confidence: self.calculate_logical_confidence(&claim_subgraph),
        })
    }
}
```

### 6. Abstract Thinking - Pattern Extraction
**File**: `src/cognitive/abstract_pattern.rs`

```rust
impl AbstractThinking {
    pub async fn extract_abstractions(&self, instances: Vec<EntityKey>) -> Result<AbstractResult> {
        // Find common structure
        let common_structure = self.extract_common_structure(&instances).await?;
        
        // Build abstraction hierarchy
        let hierarchy = self.build_abstraction_hierarchy(&common_structure).await?;
        
        // Identify abstract patterns
        let patterns = self.identify_abstract_patterns(&instances).await?;
        
        // Generate abstract concepts
        let abstract_concepts = self.generate_abstract_concepts(&patterns).await?;
        
        Ok(AbstractResult {
            abstractions: abstract_concepts,
            hierarchy,
            patterns,
            generalization_power: self.calculate_generalization_power(&abstract_concepts),
        })
    }
    
    async fn extract_common_structure(&self, instances: &[EntityKey]) -> Result<CommonStructure> {
        let mut subgraphs = Vec::new();
        
        for instance in instances {
            let subgraph = self.graph.get_subgraph(*instance, 2).await?;
            subgraphs.push(subgraph);
        }
        
        // Find common nodes and edges across all subgraphs
        let common_nodes = self.find_intersection_nodes(&subgraphs)?;
        let common_edges = self.find_common_edge_patterns(&subgraphs)?;
        let common_properties = self.extract_common_properties(&subgraphs)?;
        
        Ok(CommonStructure {
            nodes: common_nodes,
            edge_patterns: common_edges,
            properties: common_properties,
            structural_similarity: self.calculate_structural_similarity(&subgraphs),
        })
    }
}
```

### 7. Adaptive Thinking - Strategy Selection
**File**: `src/cognitive/adaptive.rs`

```rust
impl AdaptiveThinking {
    pub async fn select_strategy(&self, context: QueryContext) -> Result<AdaptiveResult> {
        // Analyze query characteristics
        let query_features = self.extract_query_features(&context).await?;
        
        // Review historical performance
        let performance_history = self.get_performance_history(&query_features).await?;
        
        // Identify applicable strategies
        let candidate_strategies = self.identify_candidate_strategies(&query_features).await?;
        
        // Evaluate each strategy
        let mut strategy_scores = Vec::new();
        for strategy in candidate_strategies {
            let score = self.evaluate_strategy(&strategy, &query_features, &performance_history).await?;
            strategy_scores.push((strategy, score));
        }
        
        // Select best strategy
        strategy_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected_strategy = strategy_scores[0].0.clone();
        
        // Adapt parameters
        let adapted_parameters = self.adapt_parameters(&selected_strategy, &context).await?;
        
        Ok(AdaptiveResult {
            selected_strategy,
            adapted_parameters,
            confidence: strategy_scores[0].1,
            alternatives: strategy_scores.into_iter().skip(1).take(2).collect(),
        })
    }
}
```

## Common Graph Patterns

### Pattern: Finding Paths Between Concepts
```rust
async fn find_reasoning_path(
    graph: &BrainEnhancedGraph,
    from: EntityKey,
    to: EntityKey,
) -> Result<Vec<Path>> {
    // BFS with relationship type constraints
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut paths = Vec::new();
    
    queue.push_back(vec![from]);
    
    while let Some(path) = queue.pop_front() {
        let current = *path.last().unwrap();
        
        if current == to {
            paths.push(Path { nodes: path });
            continue;
        }
        
        if path.len() >= MAX_DEPTH {
            continue;
        }
        
        let neighbors = graph.get_neighbors(current).await?;
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                let mut new_path = path.clone();
                new_path.push(neighbor);
                queue.push_back(new_path);
                visited.insert(neighbor);
            }
        }
    }
    
    Ok(paths)
}
```

### Pattern: Subgraph Similarity
```rust
async fn calculate_subgraph_similarity(
    graph: &BrainEnhancedGraph,
    subgraph_a: &Subgraph,
    subgraph_b: &Subgraph,
) -> Result<f32> {
    // Compare structural properties
    let node_similarity = jaccard_similarity(&subgraph_a.nodes, &subgraph_b.nodes);
    
    // Compare edge patterns
    let edge_similarity = compare_edge_patterns(subgraph_a, subgraph_b)?;
    
    // Compare properties
    let property_similarity = compare_properties(subgraph_a, subgraph_b)?;
    
    // Weighted combination
    Ok(0.4 * node_similarity + 0.4 * edge_similarity + 0.2 * property_similarity)
}
```

## Testing Strategy

Each cognitive pattern should have tests that:
1. Create a specific graph structure
2. Run the pattern algorithm
3. Verify expected graph-based results

Example:
```rust
#[test]
async fn test_convergent_finds_common_ancestor() {
    let graph = BrainEnhancedGraph::new().await.unwrap();
    
    // Create hierarchy: A -> B -> D, A -> C -> D
    let a = graph.insert_entity(create_entity("A")).await.unwrap();
    let b = graph.insert_entity(create_entity("B")).await.unwrap();
    let c = graph.insert_entity(create_entity("C")).await.unwrap();
    let d = graph.insert_entity(create_entity("D")).await.unwrap();
    
    graph.insert_relationship(create_relation(a, b)).await.unwrap();
    graph.insert_relationship(create_relation(a, c)).await.unwrap();
    graph.insert_relationship(create_relation(b, d)).await.unwrap();
    graph.insert_relationship(create_relation(c, d)).await.unwrap();
    
    let convergent = ConvergentThinking::new(Arc::new(graph));
    let result = convergent.find_convergence(vec![b, c]).await.unwrap();
    
    assert_eq!(result.convergence_point, a);
    assert!(result.confidence > 0.8);
}
```

## Success Criteria

- All cognitive patterns work without external LLM calls
- Each pattern demonstrates its intended behavior through graph operations
- Tests pass using only graph structures
- Performance is acceptable (< 100ms for typical operations)
- Results are deterministic and reproducible