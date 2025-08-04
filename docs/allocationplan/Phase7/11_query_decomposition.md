# Micro Task 11: Query Decomposition

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 10_context_analysis.md  
**Skills Required**: Query parsing, logical decomposition

## Objective

Implement intelligent query decomposition that breaks complex queries into simpler, independent sub-queries that can be processed in parallel for more efficient activation spreading.

## Context

Complex queries often contain multiple intents, relationships, or constraints that benefit from parallel processing. Query decomposition identifies these components and creates a processing strategy that maximizes parallelism while maintaining logical dependencies.

## Specifications

### Decomposition Strategy

1. **Decomposition Types**
   - Conjunctive decomposition (AND operations)
   - Disjunctive decomposition (OR operations)
   - Sequential decomposition (dependency chains)
   - Parallel decomposition (independent components)

2. **Sub-query Classification**
   - Primary queries (main intent)
   - Supporting queries (context/validation)
   - Filtering queries (constraint application)
   - Enrichment queries (additional information)

3. **Processing Orchestration**
   - Dependency graph creation
   - Parallel execution planning
   - Result merging strategies
   - Conflict resolution

## Implementation Guide

### Step 1: Core Decomposition Types
```rust
// File: src/query/decomposition.rs

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDecomposition {
    pub original_query: String,
    pub sub_queries: Vec<SubQuery>,
    pub execution_plan: ExecutionPlan,
    pub merging_strategy: MergingStrategy,
    pub decomposition_confidence: f32,
    pub estimated_performance_gain: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuery {
    pub id: String,
    pub query_text: String,
    pub intent_type: QueryIntent,
    pub priority: QueryPriority,
    pub execution_requirements: ExecutionRequirements,
    pub expected_result_type: ResultType,
    pub dependencies: Vec<String>,
    pub contributes_to: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPriority {
    Critical,    // Must succeed for valid results
    Important,   // Significantly improves results
    Useful,      // Adds value but not essential
    Optional,    // Nice-to-have enhancement
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequirements {
    pub max_execution_time: Option<std::time::Duration>,
    pub required_confidence: f32,
    pub can_fail_gracefully: bool,
    pub resource_intensity: ResourceIntensity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceIntensity {
    Low,         // Simple pattern matching
    Medium,      // Standard activation spreading
    High,        // Complex reasoning required
    VeryHigh,    // Extensive computation needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultType {
    EntityList,
    RelationshipGraph,
    ConceptExplanation,
    ComparisonMatrix,
    CausalChain,
    ValidationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub stages: Vec<ExecutionStage>,
    pub parallelizable_groups: Vec<Vec<String>>, // Sub-query IDs that can run in parallel
    pub sequential_dependencies: Vec<(String, String)>, // (prerequisite, dependent)
    pub estimated_total_time: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStage {
    pub stage_id: usize,
    pub sub_query_ids: Vec<String>,
    pub stage_type: StageType,
    pub can_skip_on_failure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageType {
    Initialization,  // Setup and validation
    Primary,         // Main query processing
    Enhancement,     // Additional context gathering
    Validation,      // Result verification
    Finalization,    // Result merging and cleanup
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergingStrategy {
    Union,           // Combine all results
    Intersection,    // Only overlapping results
    Weighted,        // Weight by sub-query importance
    Hierarchical,    // Primary results with supporting context
    Custom,          // Domain-specific merging logic
}
```

### Step 2: Query Decomposer Implementation
```rust
pub struct QueryDecomposer {
    pattern_analyzer: DecompositionPatternAnalyzer,
    dependency_resolver: DependencyResolver,
    execution_planner: ExecutionPlanner,
    config: DecompositionConfig,
}

#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    pub max_sub_queries: usize,
    pub min_decomposition_benefit: f32,
    pub prefer_parallel_execution: bool,
    pub allow_partial_failures: bool,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 8,
            min_decomposition_benefit: 0.2, // 20% performance improvement threshold
            prefer_parallel_execution: true,
            allow_partial_failures: true,
        }
    }
}

impl QueryDecomposer {
    pub fn new() -> Self {
        Self {
            pattern_analyzer: DecompositionPatternAnalyzer::new(),
            dependency_resolver: DependencyResolver::new(),
            execution_planner: ExecutionPlanner::new(),
            config: DecompositionConfig::default(),
        }
    }
    
    pub fn decompose_query(
        &self,
        parsed_query: &ParsedQuery,
        context: &QueryContext,
    ) -> Result<Option<QueryDecomposition>> {
        // Analyze if decomposition would be beneficial
        if !self.should_decompose(parsed_query, context)? {
            return Ok(None);
        }
        
        // Identify decomposition patterns
        let decomposition_patterns = self.pattern_analyzer
            .identify_patterns(&parsed_query.original_query, &parsed_query.intent_type)?;
        
        // Generate sub-queries based on patterns
        let sub_queries = self.generate_sub_queries(parsed_query, &decomposition_patterns, context)?;
        
        // Resolve dependencies between sub-queries
        let sub_queries_with_deps = self.dependency_resolver
            .resolve_dependencies(sub_queries)?;
        
        // Create execution plan
        let execution_plan = self.execution_planner
            .create_execution_plan(&sub_queries_with_deps)?;
        
        // Determine merging strategy
        let merging_strategy = self.determine_merging_strategy(&parsed_query.intent_type, &sub_queries_with_deps)?;
        
        // Calculate confidence and performance estimates
        let decomposition_confidence = self.calculate_decomposition_confidence(&sub_queries_with_deps)?;
        let performance_gain = self.estimate_performance_gain(&execution_plan)?;
        
        Ok(Some(QueryDecomposition {
            original_query: parsed_query.original_query.clone(),
            sub_queries: sub_queries_with_deps,
            execution_plan,
            merging_strategy,
            decomposition_confidence,
            estimated_performance_gain: performance_gain,
        }))
    }
    
    fn should_decompose(&self, parsed_query: &ParsedQuery, context: &QueryContext) -> Result<bool> {
        // Don't decompose simple queries
        if matches!(parsed_query.complexity, ComplexityLevel::Simple) {
            return Ok(false);
        }
        
        // Check for decomposition indicators
        let indicators = vec![
            parsed_query.original_query.contains(" and "),
            parsed_query.original_query.contains(" or "),
            parsed_query.original_query.contains("compare"),
            parsed_query.original_query.contains("relationship"),
            parsed_query.entities.len() > 3,
            context.complexity.decomposition_needed,
        ];
        
        let indicator_count = indicators.iter().filter(|&&x| x).count();
        Ok(indicator_count >= 2)
    }
}
```

### Step 3: Pattern Analysis
```rust
pub struct DecompositionPatternAnalyzer {
    patterns: Vec<DecompositionPattern>,
}

#[derive(Debug, Clone)]
pub struct DecompositionPattern {
    pub pattern_name: String,
    pub trigger_keywords: Vec<String>,
    pub intent_types: Vec<QueryIntent>,
    pub decomposition_type: DecompositionType,
    pub sub_query_templates: Vec<SubQueryTemplate>,
}

#[derive(Debug, Clone)]
pub enum DecompositionType {
    ConjunctiveAnd,      // Query contains multiple AND conditions
    DisjunctiveOr,       // Query contains OR alternatives  
    Sequential,          // Dependencies require ordered execution
    Comparison,          // Compare multiple entities
    Hierarchical,        // Navigate hierarchies
    CausalChain,         // Follow cause-effect relationships
}

#[derive(Debug, Clone)]
pub struct SubQueryTemplate {
    pub template: String,
    pub priority: QueryPriority,
    pub variable_slots: Vec<String>,
    pub result_type: ResultType,
}

impl DecompositionPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Self::create_decomposition_patterns(),
        }
    }
    
    fn create_decomposition_patterns() -> Vec<DecompositionPattern> {
        vec![
            DecompositionPattern {
                pattern_name: "comparison_decomposition".into(),
                trigger_keywords: vec!["compare".into(), "difference".into(), "versus".into(), "vs".into()],
                intent_types: vec![QueryIntent::Comparison { entities: vec![], aspect: "".into(), comparison_type: ComparisonType::Differences }],
                decomposition_type: DecompositionType::Comparison,
                sub_query_templates: vec![
                    SubQueryTemplate {
                        template: "What are the properties of {entity1}?".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity1".into()],
                        result_type: ResultType::EntityList,
                    },
                    SubQueryTemplate {
                        template: "What are the properties of {entity2}?".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity2".into()],
                        result_type: ResultType::EntityList,
                    },
                    SubQueryTemplate {
                        template: "How are {entity1} and {entity2} similar?".into(),
                        priority: QueryPriority::Important,
                        variable_slots: vec!["entity1".into(), "entity2".into()],
                        result_type: ResultType::ComparisonMatrix,
                    },
                ],
            },
            DecompositionPattern {
                pattern_name: "relationship_exploration".into(),
                trigger_keywords: vec!["related".into(), "connection".into(), "relationship".into()],
                intent_types: vec![QueryIntent::Relationship { entity1: "".into(), entity2: "".into(), relation_type: RelationType::Association, direction: RelationDirection::Bidirectional }],
                decomposition_type: DecompositionType::Sequential,
                sub_query_templates: vec![
                    SubQueryTemplate {
                        template: "What is {entity1}?".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity1".into()],
                        result_type: ResultType::ConceptExplanation,
                    },
                    SubQueryTemplate {
                        template: "What is {entity2}?".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity2".into()],
                        result_type: ResultType::ConceptExplanation,
                    },
                    SubQueryTemplate {
                        template: "Find direct relationships between {entity1} and {entity2}".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity1".into(), "entity2".into()],
                        result_type: ResultType::RelationshipGraph,
                    },
                    SubQueryTemplate {
                        template: "Find indirect relationships between {entity1} and {entity2}".into(),
                        priority: QueryPriority::Useful,
                        variable_slots: vec!["entity1".into(), "entity2".into()],
                        result_type: ResultType::RelationshipGraph,
                    },
                ],
            },
            DecompositionPattern {
                pattern_name: "conjunctive_filter".into(),
                trigger_keywords: vec!["and".into(), "also".into(), "both".into(), "all".into()],
                intent_types: vec![QueryIntent::Filter { entity_type: "".into(), property: "".into(), value: "".into(), operator: FilterOperator::Equals }],
                decomposition_type: DecompositionType::ConjunctiveAnd,
                sub_query_templates: vec![
                    SubQueryTemplate {
                        template: "Find {entity_type} with {property1}".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity_type".into(), "property1".into()],
                        result_type: ResultType::EntityList,
                    },
                    SubQueryTemplate {
                        template: "Find {entity_type} with {property2}".into(),
                        priority: QueryPriority::Critical,
                        variable_slots: vec!["entity_type".into(), "property2".into()],
                        result_type: ResultType::EntityList,
                    },
                ],
            },
        ]
    }
    
    pub fn identify_patterns(&self, query: &str, intent: &QueryIntent) -> Result<Vec<DecompositionPattern>> {
        let query_lower = query.to_lowercase();
        let mut matching_patterns = Vec::new();
        
        for pattern in &self.patterns {
            // Check if keywords match
            let keyword_matches = pattern.trigger_keywords.iter()
                .filter(|keyword| query_lower.contains(keyword.as_str()))
                .count();
            
            // Check if intent type matches
            let intent_matches = pattern.intent_types.iter()
                .any(|pattern_intent| std::mem::discriminant(pattern_intent) == std::mem::discriminant(intent));
            
            if keyword_matches > 0 || intent_matches {
                matching_patterns.push(pattern.clone());
            }
        }
        
        // Sort by relevance (more keyword matches = higher relevance)
        matching_patterns.sort_by(|a, b| {
            let a_matches = a.trigger_keywords.iter()
                .filter(|keyword| query_lower.contains(keyword.as_str()))
                .count();
            let b_matches = b.trigger_keywords.iter()
                .filter(|keyword| query_lower.contains(keyword.as_str()))
                .count();
            b_matches.cmp(&a_matches)
        });
        
        Ok(matching_patterns)
    }
}
```

### Step 4: Sub-query Generation
```rust
impl QueryDecomposer {
    fn generate_sub_queries(
        &self,
        parsed_query: &ParsedQuery,
        patterns: &[DecompositionPattern],
        context: &QueryContext,
    ) -> Result<Vec<SubQuery>> {
        let mut sub_queries = Vec::new();
        
        for pattern in patterns.iter().take(1) { // Use the most relevant pattern
            let variables = self.extract_variables_from_query(parsed_query)?;
            
            for template in &pattern.sub_query_templates {
                if let Some(sub_query) = self.instantiate_template(template, &variables, context)? {
                    sub_queries.push(sub_query);
                }
            }
        }
        
        // Add validation sub-queries if needed
        if context.confidence.verification_level != VerificationLevel::None {
            sub_queries.extend(self.generate_validation_queries(parsed_query)?);
        }
        
        // Limit total number of sub-queries
        sub_queries.truncate(self.config.max_sub_queries);
        
        Ok(sub_queries)
    }
    
    fn extract_variables_from_query(&self, parsed_query: &ParsedQuery) -> Result<HashMap<String, String>> {
        let mut variables = HashMap::new();
        
        // Extract entities as variables
        for (i, entity) in parsed_query.entities.iter().enumerate() {
            variables.insert(format!("entity{}", i + 1), entity.name.clone());
            
            // Also map by entity type
            match entity.entity_type {
                Some(ref entity_type) => {
                    variables.insert("entity_type".into(), entity_type.clone());
                }
                None => {}
            }
        }
        
        // Extract intent-specific variables
        match &parsed_query.intent_type {
            QueryIntent::Filter { entity_type, property, .. } => {
                variables.insert("entity_type".into(), entity_type.clone());
                variables.insert("property1".into(), property.clone());
            }
            QueryIntent::Comparison { entities, aspect, .. } => {
                for (i, entity) in entities.iter().enumerate() {
                    variables.insert(format!("entity{}", i + 1), entity.clone());
                }
                variables.insert("aspect".into(), aspect.clone());
            }
            QueryIntent::Relationship { entity1, entity2, .. } => {
                variables.insert("entity1".into(), entity1.clone());
                variables.insert("entity2".into(), entity2.clone());
            }
            _ => {}
        }
        
        Ok(variables)
    }
    
    fn instantiate_template(
        &self,
        template: &SubQueryTemplate,
        variables: &HashMap<String, String>,
        context: &QueryContext,
    ) -> Result<Option<SubQuery>> {
        let mut query_text = template.template.clone();
        
        // Replace variable placeholders
        for (var_name, var_value) in variables {
            let placeholder = format!("{{{}}}", var_name);
            if query_text.contains(&placeholder) {
                query_text = query_text.replace(&placeholder, var_value);
            }
        }
        
        // Check if all variables were resolved
        if query_text.contains('{') {
            return Ok(None); // Template couldn't be fully instantiated
        }
        
        // Generate unique ID
        let id = format!("sub_query_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        
        // Determine execution requirements based on context
        let execution_requirements = ExecutionRequirements {
            max_execution_time: Some(std::time::Duration::from_millis(
                match template.priority {
                    QueryPriority::Critical => 5000,
                    QueryPriority::Important => 3000,
                    QueryPriority::Useful => 2000,
                    QueryPriority::Optional => 1000,
                }
            )),
            required_confidence: context.confidence.required_confidence * 0.8, // Slightly lower for sub-queries
            can_fail_gracefully: !matches!(template.priority, QueryPriority::Critical),
            resource_intensity: self.estimate_resource_intensity(&query_text),
        };
        
        // Parse the sub-query to get its intent
        // This is a simplified version - in practice, you'd use the full intent parser
        let intent_type = self.infer_intent_from_template(&query_text)?;
        
        Ok(Some(SubQuery {
            id,
            query_text,
            intent_type,
            priority: template.priority.clone(),
            execution_requirements,
            expected_result_type: template.result_type.clone(),
            dependencies: Vec::new(), // Will be filled by dependency resolver
            contributes_to: Vec::new(),
        }))
    }
    
    fn estimate_resource_intensity(&self, query: &str) -> ResourceIntensity {
        let complexity_indicators = vec![
            query.contains("complex"),
            query.contains("relationship"),
            query.contains("mechanism"),
            query.contains("analysis"),
            query.split_whitespace().count() > 10,
        ];
        
        match complexity_indicators.iter().filter(|&&x| x).count() {
            0..=1 => ResourceIntensity::Low,
            2 => ResourceIntensity::Medium,
            3 => ResourceIntensity::High,
            _ => ResourceIntensity::VeryHigh,
        }
    }
    
    fn generate_validation_queries(&self, parsed_query: &ParsedQuery) -> Result<Vec<SubQuery>> {
        let mut validation_queries = Vec::new();
        
        // Generate confidence validation queries
        for entity in &parsed_query.entities {
            let id = format!("validate_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
            let query_text = format!("Verify that {} is a valid entity", entity.name);
            
            validation_queries.push(SubQuery {
                id,
                query_text,
                intent_type: QueryIntent::Definition {
                    entity: entity.name.clone(),
                    detail_level: DetailLevel::Basic,
                },
                priority: QueryPriority::Useful,
                execution_requirements: ExecutionRequirements {
                    max_execution_time: Some(std::time::Duration::from_millis(1000)),
                    required_confidence: 0.7,
                    can_fail_gracefully: true,
                    resource_intensity: ResourceIntensity::Low,
                },
                expected_result_type: ResultType::ValidationResult,
                dependencies: Vec::new(),
                contributes_to: Vec::new(),
            });
        }
        
        Ok(validation_queries)
    }
}
```

### Step 5: Execution Planning
```rust
pub struct ExecutionPlanner {
    optimization_strategy: OptimizationStrategy,
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    MinimizeLatency,     // Fastest overall completion
    MaximizeParallelism, // Use all available resources
    BalanceResources,    // Optimize resource utilization
    MinimizeFailures,    // Most robust execution
}

impl ExecutionPlanner {
    pub fn new() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::BalanceResources,
        }
    }
    
    pub fn create_execution_plan(&self, sub_queries: &[SubQuery]) -> Result<ExecutionPlan> {
        // Create dependency graph
        let dependency_graph = self.build_dependency_graph(sub_queries)?;
        
        // Identify parallelizable groups using topological sorting
        let stages = self.create_execution_stages(&dependency_graph, sub_queries)?;
        
        // Calculate parallelizable groups within each stage
        let parallelizable_groups = self.identify_parallel_groups(&stages, &dependency_graph)?;
        
        // Estimate total execution time
        let estimated_time = self.estimate_execution_time(&stages, sub_queries)?;
        
        Ok(ExecutionPlan {
            stages,
            parallelizable_groups,
            sequential_dependencies: dependency_graph.edges(),
            estimated_total_time: estimated_time,
        })
    }
    
    fn build_dependency_graph(&self, sub_queries: &[SubQuery]) -> Result<DependencyGraph> {
        let mut graph = DependencyGraph::new();
        
        // Add all nodes
        for sub_query in sub_queries {
            graph.add_node(sub_query.id.clone());
        }
        
        // Add dependency edges
        for sub_query in sub_queries {
            for dependency in &sub_query.dependencies {
                graph.add_edge(dependency.clone(), sub_query.id.clone());
            }
        }
        
        Ok(graph)
    }
    
    fn create_execution_stages(
        &self,
        dependency_graph: &DependencyGraph,
        sub_queries: &[SubQuery],
    ) -> Result<Vec<ExecutionStage>> {
        let topological_order = dependency_graph.topological_sort()?;
        let mut stages = Vec::new();
        let mut current_stage = Vec::new();
        let mut processed = HashSet::new();
        
        for node in topological_order {
            // Check if all dependencies are processed
            let dependencies_ready = sub_queries.iter()
                .find(|sq| sq.id == node)
                .map(|sq| sq.dependencies.iter().all(|dep| processed.contains(dep)))
                .unwrap_or(false);
            
            if dependencies_ready || processed.is_empty() {
                current_stage.push(node.clone());
                processed.insert(node);
            } else {
                // Start new stage
                if !current_stage.is_empty() {
                    stages.push(ExecutionStage {
                        stage_id: stages.len(),
                        sub_query_ids: current_stage,
                        stage_type: self.determine_stage_type(stages.len()),
                        can_skip_on_failure: false,
                    });
                    current_stage = Vec::new();
                }
                current_stage.push(node.clone());
                processed.insert(node);
            }
        }
        
        // Add final stage if needed
        if !current_stage.is_empty() {
            stages.push(ExecutionStage {
                stage_id: stages.len(),
                sub_query_ids: current_stage,
                stage_type: self.determine_stage_type(stages.len()),
                can_skip_on_failure: false,
            });
        }
        
        Ok(stages)
    }
    
    fn determine_stage_type(&self, stage_index: usize) -> StageType {
        match stage_index {
            0 => StageType::Initialization,
            1 => StageType::Primary,
            i if i < 4 => StageType::Enhancement,
            _ => StageType::Finalization,
        }
    }
}

struct DependencyGraph {
    nodes: HashSet<String>,
    edges: Vec<(String, String)>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: Vec::new(),
        }
    }
    
    fn add_node(&mut self, node: String) {
        self.nodes.insert(node);
    }
    
    fn add_edge(&mut self, from: String, to: String) {
        self.edges.push((from, to));
    }
    
    fn edges(&self) -> Vec<(String, String)> {
        self.edges.clone()
    }
    
    fn topological_sort(&self) -> Result<Vec<String>> {
        // Simplified topological sort implementation
        let mut result = Vec::new();
        let mut remaining_nodes = self.nodes.clone();
        
        while !remaining_nodes.is_empty() {
            // Find nodes with no incoming edges
            let nodes_with_no_deps: Vec<String> = remaining_nodes.iter()
                .filter(|node| {
                    !self.edges.iter().any(|(_, to)| to == *node && remaining_nodes.contains(to))
                })
                .cloned()
                .collect();
            
            if nodes_with_no_deps.is_empty() {
                return Err(Error::CircularDependency);
            }
            
            // Add these nodes to result and remove them
            for node in nodes_with_no_deps {
                result.push(node.clone());
                remaining_nodes.remove(&node);
            }
        }
        
        Ok(result)
    }
}
```

## File Locations

- `src/query/decomposition.rs` - Main implementation
- `src/query/pattern_analyzer.rs` - Pattern analysis
- `src/query/dependency_resolver.rs` - Dependency resolution
- `src/query/execution_planner.rs` - Execution planning
- `tests/query/decomposition_tests.rs` - Test implementation

## Success Criteria

- [ ] Complex queries decomposed correctly
- [ ] Sub-queries are independent and meaningful
- [ ] Dependency resolution works properly
- [ ] Execution plans optimize for parallelism
- [ ] Performance gain estimation accurate
- [ ] Merging strategies preserve intent
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_comparison_query_decomposition() {
    let decomposer = QueryDecomposer::new();
    
    let parsed_query = ParsedQuery {
        original_query: "Compare lions and tigers".to_string(),
        intent_type: QueryIntent::Comparison {
            entities: vec!["lions".into(), "tigers".into()],
            aspect: "general".into(),
            comparison_type: ComparisonType::Differences,
        },
        entities: vec![
            ExtractedEntity {
                name: "lions".into(),
                entity_type: Some("animal".into()),
                aliases: vec![],
                confidence: 0.9,
                span: TextSpan { start: 8, end: 13 },
            },
            ExtractedEntity {
                name: "tigers".into(),
                entity_type: Some("animal".into()),
                aliases: vec![],
                confidence: 0.9,
                span: TextSpan { start: 18, end: 24 },
            },
        ],
        context: QueryContext::default(),
        confidence: 0.8,
        sub_queries: vec![],
        complexity: ComplexityLevel::Compound,
    };
    
    let context = QueryContext::default();
    let decomposition = decomposer.decompose_query(&parsed_query, &context).unwrap().unwrap();
    
    // Should have sub-queries for each entity plus comparison
    assert!(decomposition.sub_queries.len() >= 2);
    assert!(decomposition.sub_queries.iter().any(|sq| sq.query_text.contains("lions")));
    assert!(decomposition.sub_queries.iter().any(|sq| sq.query_text.contains("tigers")));
}

#[test]
fn test_relationship_query_decomposition() {
    let decomposer = QueryDecomposer::new();
    
    let parsed_query = ParsedQuery {
        original_query: "How are dogs related to wolves?".to_string(),
        intent_type: QueryIntent::Relationship {
            entity1: "dogs".into(),
            entity2: "wolves".into(),
            relation_type: RelationType::Similarity,
            direction: RelationDirection::Bidirectional,
        },
        entities: vec![
            ExtractedEntity {
                name: "dogs".into(),
                entity_type: Some("animal".into()),
                aliases: vec![],
                confidence: 0.9,
                span: TextSpan { start: 8, end: 12 },
            },
            ExtractedEntity {
                name: "wolves".into(),
                entity_type: Some("animal".into()),
                aliases: vec![],
                confidence: 0.9,
                span: TextSpan { start: 24, end: 30 },
            },
        ],
        context: QueryContext::default(),
        confidence: 0.8,
        sub_queries: vec![],
        complexity: ComplexityLevel::Compound,
    };
    
    let context = QueryContext::default();
    let decomposition = decomposer.decompose_query(&parsed_query, &context).unwrap().unwrap();
    
    // Should have definition queries for each entity plus relationship query
    assert!(decomposition.sub_queries.len() >= 3);
    
    // Check for proper dependency ordering
    let execution_plan = &decomposition.execution_plan;
    assert!(!execution_plan.stages.is_empty());
}

#[test]
fn test_execution_plan_creation() {
    let planner = ExecutionPlanner::new();
    
    let sub_queries = vec![
        SubQuery {
            id: "sq1".into(),
            query_text: "What is a dog?".into(),
            intent_type: QueryIntent::Definition {
                entity: "dog".into(),
                detail_level: DetailLevel::Basic,
            },
            priority: QueryPriority::Critical,
            execution_requirements: ExecutionRequirements {
                max_execution_time: Some(std::time::Duration::from_millis(1000)),
                required_confidence: 0.8,
                can_fail_gracefully: false,
                resource_intensity: ResourceIntensity::Low,
            },
            expected_result_type: ResultType::ConceptExplanation,
            dependencies: vec![],
            contributes_to: vec!["sq3".into()],
        },
        SubQuery {
            id: "sq2".into(),
            query_text: "What is a wolf?".into(),
            intent_type: QueryIntent::Definition {
                entity: "wolf".into(),
                detail_level: DetailLevel::Basic,
            },
            priority: QueryPriority::Critical,
            execution_requirements: ExecutionRequirements {
                max_execution_time: Some(std::time::Duration::from_millis(1000)),
                required_confidence: 0.8,
                can_fail_gracefully: false,
                resource_intensity: ResourceIntensity::Low,
            },
            expected_result_type: ResultType::ConceptExplanation,
            dependencies: vec![],
            contributes_to: vec!["sq3".into()],
        },
        SubQuery {
            id: "sq3".into(),
            query_text: "Find relationships between dogs and wolves".into(),
            intent_type: QueryIntent::Relationship {
                entity1: "dogs".into(),
                entity2: "wolves".into(),
                relation_type: RelationType::Similarity,
                direction: RelationDirection::Bidirectional,
            },
            priority: QueryPriority::Critical,
            execution_requirements: ExecutionRequirements {
                max_execution_time: Some(std::time::Duration::from_millis(2000)),
                required_confidence: 0.8,
                can_fail_gracefully: false,
                resource_intensity: ResourceIntensity::Medium,
            },
            expected_result_type: ResultType::RelationshipGraph,
            dependencies: vec!["sq1".into(), "sq2".into()],
            contributes_to: vec![],
        },
    ];
    
    let plan = planner.create_execution_plan(&sub_queries).unwrap();
    
    // Should have at least 2 stages (definitions can run in parallel, then relationship)
    assert!(plan.stages.len() >= 2);
    
    // First stage should have the definition queries
    assert!(plan.stages[0].sub_query_ids.contains(&"sq1".into()));
    assert!(plan.stages[0].sub_query_ids.contains(&"sq2".into()));
    
    // Last stage should have the relationship query
    assert!(plan.stages.last().unwrap().sub_query_ids.contains(&"sq3".into()));
}

#[test]
fn test_parallel_execution_identification() {
    let decomposer = QueryDecomposer::new();
    
    // Test query that should result in parallel sub-queries
    let parsed_query = ParsedQuery {
        original_query: "Find animals that are large and live in water".to_string(),
        intent_type: QueryIntent::Filter {
            entity_type: "animals".into(),
            property: "size_and_habitat".into(),
            value: "large_water".into(),
            operator: FilterOperator::HasProperty,
        },
        entities: vec![
            ExtractedEntity {
                name: "animals".into(),
                entity_type: Some("organism".into()),
                aliases: vec![],
                confidence: 0.9,
                span: TextSpan { start: 5, end: 12 },
            },
        ],
        context: QueryContext::default(),
        confidence: 0.8,
        sub_queries: vec![],
        complexity: ComplexityLevel::Compound,
    };
    
    let context = QueryContext::default();
    if let Some(decomposition) = decomposer.decompose_query(&parsed_query, &context).unwrap() {
        // Should identify parallelizable groups
        assert!(!decomposition.execution_plan.parallelizable_groups.is_empty());
    }
}

#[test]
fn test_performance_estimation() {
    let decomposer = QueryDecomposer::new();
    
    let simple_query = ParsedQuery {
        original_query: "What is a cat?".to_string(),
        intent_type: QueryIntent::Definition {
            entity: "cat".into(),
            detail_level: DetailLevel::Basic,
        },
        entities: vec![],
        context: QueryContext::default(),
        confidence: 0.8,
        sub_queries: vec![],
        complexity: ComplexityLevel::Simple,
    };
    
    let context = QueryContext::default();
    let result = decomposer.decompose_query(&simple_query, &context).unwrap();
    
    // Simple queries shouldn't be decomposed
    assert!(result.is_none());
    
    let complex_query = ParsedQuery {
        original_query: "Compare the evolutionary relationships between primates and their cognitive abilities".to_string(),
        intent_type: QueryIntent::Comparison {
            entities: vec!["primates".into(), "cognitive abilities".into()],
            aspect: "evolution".into(),
            comparison_type: ComparisonType::Similarities,
        },
        entities: vec![],
        context: QueryContext::default(),
        confidence: 0.8,
        sub_queries: vec![],
        complexity: ComplexityLevel::Complex,
    };
    
    if let Some(decomposition) = decomposer.decompose_query(&complex_query, &context).unwrap() {
        // Should show performance benefit for complex queries
        assert!(decomposition.estimated_performance_gain > 0.1);
    }
}
```

## Quality Gates

- [ ] Decomposition preserves original query intent
- [ ] Sub-queries are grammatically correct
- [ ] Dependency resolution prevents deadlocks
- [ ] Execution plans are optimal for given strategy
- [ ] Performance estimation correlates with actual gains

## Next Task

Upon completion, proceed to **12_intent_tests.md**