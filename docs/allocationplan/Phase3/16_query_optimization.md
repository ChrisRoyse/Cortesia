# Task 16: Query Optimization for Inheritance Resolution
**Estimated Time**: 15-20 minutes
**Dependencies**: 15_inheritance_validation.md
**Stage**: Performance Optimization

## Objective
Optimize inheritance resolution queries through advanced Cypher query optimization, index utilization, and query caching strategies to achieve sub-10ms inheritance chain resolution times.

## Specific Requirements

### 1. Query Pattern Optimization
- Optimize inheritance chain traversal queries
- Implement query result caching with intelligent invalidation
- Add query execution plan analysis and optimization
- Build query performance monitoring and alerting

### 2. Index Strategy Optimization
- Create specialized indices for inheritance operations
- Implement composite indices for multi-property queries
- Add query hint optimization for complex traversals
- Build index usage monitoring and recommendation system

### 3. Execution Plan Optimization
- Analyze and optimize Cypher execution plans
- Implement query rewriting for better performance
- Add parallel query execution where possible
- Build query complexity analysis and limits

## Implementation Steps

### 1. Create Query Optimization Framework
```rust
// src/inheritance/optimization/query_optimizer.rs
use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct QueryOptimizer {
    query_cache: Arc<RwLock<HashMap<String, CachedQuery>>>,
    execution_plan_cache: Arc<RwLock<HashMap<String, ExecutionPlan>>>,
    performance_monitor: Arc<QueryPerformanceMonitor>,
    optimization_config: QueryOptimizationConfig,
}

#[derive(Debug, Clone)]
pub struct CachedQuery {
    pub query_text: String,
    pub optimized_query: String,
    pub execution_plan: ExecutionPlan,
    pub cached_at: DateTime<Utc>,
    pub access_count: u32,
    pub avg_execution_time: Duration,
    pub cache_hits: u32,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub estimated_cost: f64,
    pub estimated_rows: i64,
    pub plan_steps: Vec<PlanStep>,
    pub index_usage: Vec<IndexUsage>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub step_type: StepType,
    pub estimated_cost: f64,
    pub estimated_rows: i64,
    pub index_used: Option<String>,
    pub optimization_applied: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum StepType {
    NodeByIdSeek,
    NodeByLabelScan,
    Expand,
    Filter,
    Project,
    Sort,
    Limit,
    VariableLengthExpand,
}

impl QueryOptimizer {
    pub async fn new(config: QueryOptimizationConfig) -> Self {
        Self {
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_plan_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: Arc::new(QueryPerformanceMonitor::new()),
            optimization_config: config,
        }
    }
    
    pub async fn optimize_inheritance_chain_query(
        &self,
        concept_id: &str,
        max_depth: Option<i32>,
    ) -> Result<OptimizedQuery, QueryOptimizationError> {
        let query_key = format!("inheritance_chain_{}_{:?}", concept_id, max_depth);
        
        // Check cache first
        if let Some(cached) = self.get_cached_query(&query_key).await {
            return Ok(cached.into());
        }
        
        // Build optimized query
        let base_query = self.build_inheritance_chain_query(concept_id, max_depth);
        let optimized_query = self.apply_query_optimizations(&base_query).await?;
        
        // Analyze execution plan
        let execution_plan = self.analyze_execution_plan(&optimized_query).await?;
        
        // Cache the optimized query
        self.cache_optimized_query(query_key, optimized_query.clone(), execution_plan).await?;
        
        Ok(optimized_query)
    }
    
    fn build_inheritance_chain_query(&self, concept_id: &str, max_depth: Option<i32>) -> String {
        let depth_limit = max_depth.unwrap_or(self.optimization_config.default_max_depth);
        
        format!(r#"
            // Optimized inheritance chain query with early termination
            MATCH path = (c:Concept {{id: $concept_id}})
                        -[:INHERITS_FROM*0..{}]->(ancestor:Concept)
            WHERE ALL(r IN relationships(path) WHERE r.is_active = true)
            WITH path, ancestor, length(path) as depth
            ORDER BY depth
            WITH collect({{
                concept_id: ancestor.id,
                concept_name: ancestor.name,
                depth: depth,
                properties: ancestor.properties
            }}) as inheritance_chain
            RETURN inheritance_chain
        "#, depth_limit)
    }
    
    async fn apply_query_optimizations(&self, query: &str) -> Result<OptimizedQuery, OptimizationError> {
        let mut optimized = query.to_string();
        
        // Apply index hints for better performance
        optimized = self.add_index_hints(&optimized).await?;
        
        // Apply query rewriting rules
        optimized = self.apply_rewriting_rules(&optimized).await?;
        
        // Add performance hints
        optimized = self.add_performance_hints(&optimized).await?;
        
        Ok(OptimizedQuery {
            original_query: query.to_string(),
            optimized_query: optimized,
            optimizations_applied: self.get_applied_optimizations().await,
            estimated_improvement: self.estimate_performance_improvement(query, &optimized).await?,
        })
    }
    
    async fn add_index_hints(&self, query: &str) -> Result<String, OptimizationError> {
        // Add index hints for concept lookups
        let mut optimized = query.replace(
            "MATCH (c:Concept {id: $concept_id})",
            "MATCH (c:Concept) USING INDEX c:Concept(id) WHERE c.id = $concept_id"
        );
        
        // Add hints for relationship traversals
        optimized = optimized.replace(
            "-[:INHERITS_FROM*",
            "-[:INHERITS_FROM* // Use relationship index for better performance"
        );
        
        Ok(optimized)
    }
    
    async fn analyze_execution_plan(&self, query: &str) -> Result<ExecutionPlan, AnalysisError> {
        // This would integrate with Neo4j's EXPLAIN/PROFILE functionality
        let plan_query = format!("EXPLAIN {}", query);
        
        // Execute plan analysis (simplified for this example)
        let plan_result = self.execute_plan_analysis(&plan_query).await?;
        
        Ok(ExecutionPlan {
            plan_id: uuid::Uuid::new_v4().to_string(),
            estimated_cost: plan_result.total_cost,
            estimated_rows: plan_result.estimated_rows,
            plan_steps: plan_result.steps,
            index_usage: plan_result.index_usage,
            optimization_opportunities: self.identify_optimization_opportunities(&plan_result).await?,
        })
    }
}
```

### 2. Implement Performance-Optimized Query Builders
```rust
// src/inheritance/optimization/optimized_queries.rs
pub struct OptimizedQueryBuilder {
    optimizer: Arc<QueryOptimizer>,
    query_templates: HashMap<QueryType, String>,
    parameter_cache: Arc<RwLock<HashMap<String, QueryParameters>>>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QueryType {
    InheritanceChain,
    PropertyResolution,
    ConceptAncestors,
    ConceptDescendants,
    InheritanceValidation,
}

impl OptimizedQueryBuilder {
    pub fn new(optimizer: Arc<QueryOptimizer>) -> Self {
        let mut query_templates = HashMap::new();
        
        // Pre-optimized query templates
        query_templates.insert(
            QueryType::InheritanceChain,
            r#"
                // High-performance inheritance chain query with batching
                CALL {
                    MATCH (c:Concept) 
                    USING INDEX c:Concept(id) 
                    WHERE c.id = $concept_id
                    
                    CALL apoc.path.expandConfig(c, {
                        relationshipFilter: "INHERITS_FROM>",
                        labelFilter: "+Concept",
                        minLevel: 0,
                        maxLevel: $max_depth,
                        limit: 1000
                    }) YIELD path
                    
                    WITH nodes(path) as chain_nodes
                    UNWIND range(0, size(chain_nodes)-1) as idx
                    WITH chain_nodes[idx] as node, idx as depth
                    
                    RETURN {
                        concept_id: node.id,
                        name: node.name,
                        depth: depth,
                        properties: node.properties
                    } as concept_info
                    ORDER BY depth
                }
                RETURN collect(concept_info) as inheritance_chain
            "#.to_string()
        );
        
        query_templates.insert(
            QueryType::PropertyResolution,
            r#"
                // Optimized property resolution with inheritance
                MATCH (c:Concept) 
                USING INDEX c:Concept(id) 
                WHERE c.id = $concept_id
                
                OPTIONAL MATCH path = (c)-[:INHERITS_FROM*0..10]->(ancestor:Concept)
                WHERE ALL(r IN relationships(path) WHERE r.is_active = true)
                
                WITH c, ancestor, length(path) as inheritance_depth
                ORDER BY inheritance_depth
                
                WITH c, collect(ancestor) as ancestors
                
                CALL {
                    WITH ancestors
                    UNWIND ancestors as anc
                    MATCH (anc)-[:HAS_PROPERTY]->(prop:Property)
                    WHERE prop.is_active = true
                    RETURN prop.name as prop_name, 
                           prop.value as prop_value,
                           prop.type as prop_type,
                           anc.id as source_concept
                    ORDER BY length(()-[:INHERITS_FROM*]->(anc))
                }
                
                WITH prop_name, 
                     collect({value: prop_value, type: prop_type, source: source_concept})[0] as resolved_prop
                RETURN collect({name: prop_name, value: resolved_prop.value, 
                               type: resolved_prop.type, source: resolved_prop.source}) as properties
            "#.to_string()
        );
        
        Self {
            optimizer,
            query_templates,
            parameter_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn build_inheritance_chain_query(
        &self,
        concept_id: &str,
        max_depth: Option<i32>,
    ) -> Result<PreparedQuery, QueryBuildError> {
        let query_template = self.query_templates
            .get(&QueryType::InheritanceChain)
            .ok_or(QueryBuildError::TemplateNotFound)?;
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
            "max_depth".to_string() => max_depth.unwrap_or(10).into(),
        ];
        
        // Apply final optimizations
        let optimized_query = self.optimizer
            .apply_runtime_optimizations(query_template, &parameters)
            .await?;
        
        Ok(PreparedQuery {
            query: optimized_query,
            parameters,
            estimated_cost: self.estimate_query_cost(&optimized_query, &parameters).await?,
            cache_ttl: Duration::from_secs(300), // 5 minutes
        })
    }
    
    pub async fn build_batch_inheritance_query(
        &self,
        concept_ids: &[String],
        max_depth: Option<i32>,
    ) -> Result<PreparedQuery, QueryBuildError> {
        // Build optimized batch query for multiple concepts
        let batch_query = format!(r#"
            // Batch inheritance chain query for optimal performance
            UNWIND $concept_ids as concept_id
            
            CALL {{
                WITH concept_id
                MATCH (c:Concept) 
                USING INDEX c:Concept(id) 
                WHERE c.id = concept_id
                
                CALL apoc.path.expandConfig(c, {{
                    relationshipFilter: "INHERITS_FROM>",
                    labelFilter: "+Concept", 
                    minLevel: 0,
                    maxLevel: $max_depth,
                    limit: 100
                }}) YIELD path
                
                WITH concept_id, nodes(path) as chain_nodes
                UNWIND range(0, size(chain_nodes)-1) as idx
                WITH concept_id, chain_nodes[idx] as node, idx as depth
                
                RETURN concept_id, {{
                    concept_id: node.id,
                    name: node.name, 
                    depth: depth,
                    properties: node.properties
                }} as concept_info
                ORDER BY depth
            }}
            
            WITH concept_id, collect(concept_info) as inheritance_chain
            RETURN {{concept_id: concept_id, chain: inheritance_chain}} as result
        "#);
        
        let parameters = hashmap![
            "concept_ids".to_string() => concept_ids.iter().cloned().collect::<Vec<_>>().into(),
            "max_depth".to_string() => max_depth.unwrap_or(10).into(),
        ];
        
        Ok(PreparedQuery {
            query: batch_query,
            parameters,
            estimated_cost: self.estimate_batch_cost(concept_ids.len()).await?,
            cache_ttl: Duration::from_secs(600), // 10 minutes for batch queries
        })
    }
}
```

### 3. Implement Query Performance Monitoring
```rust
// src/inheritance/optimization/query_monitor.rs
pub struct QueryPerformanceMonitor {
    execution_metrics: Arc<RwLock<HashMap<String, QueryMetrics>>>,
    slow_query_threshold: Duration,
    performance_alerts: Arc<RwLock<Vec<PerformanceAlert>>>,
}

#[derive(Debug, Clone)]
pub struct QueryMetrics {
    pub query_signature: String,
    pub total_executions: u64,
    pub total_execution_time: Duration,
    pub avg_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub p95_execution_time: Duration,
    pub cache_hit_rate: f64,
    pub index_hit_rate: f64,
    pub rows_processed: u64,
}

impl QueryPerformanceMonitor {
    pub async fn record_query_execution(
        &self,
        query_signature: String,
        execution_time: Duration,
        rows_processed: u64,
        cache_hit: bool,
        index_used: bool,
    ) {
        let mut metrics = self.execution_metrics.write().await;
        
        let entry = metrics.entry(query_signature.clone()).or_insert_with(|| {
            QueryMetrics {
                query_signature: query_signature.clone(),
                total_executions: 0,
                total_execution_time: Duration::from_millis(0),
                avg_execution_time: Duration::from_millis(0),
                min_execution_time: Duration::from_secs(999),
                max_execution_time: Duration::from_millis(0),
                p95_execution_time: Duration::from_millis(0),
                cache_hit_rate: 0.0,
                index_hit_rate: 0.0,
                rows_processed: 0,
            }
        });
        
        // Update metrics
        entry.total_executions += 1;
        entry.total_execution_time += execution_time;
        entry.avg_execution_time = entry.total_execution_time / entry.total_executions as u32;
        entry.min_execution_time = entry.min_execution_time.min(execution_time);
        entry.max_execution_time = entry.max_execution_time.max(execution_time);
        entry.rows_processed += rows_processed;
        
        // Update hit rates
        entry.cache_hit_rate = self.calculate_hit_rate(entry.total_executions, cache_hit, entry.cache_hit_rate);
        entry.index_hit_rate = self.calculate_hit_rate(entry.total_executions, index_used, entry.index_hit_rate);
        
        // Check for performance alerts
        if execution_time > self.slow_query_threshold {
            self.create_slow_query_alert(&query_signature, execution_time).await;
        }
    }
    
    pub async fn get_performance_report(&self) -> QueryPerformanceReport {
        let metrics = self.execution_metrics.read().await;
        let alerts = self.performance_alerts.read().await;
        
        let mut sorted_metrics: Vec<_> = metrics.values().cloned().collect();
        sorted_metrics.sort_by(|a, b| b.avg_execution_time.cmp(&a.avg_execution_time));
        
        QueryPerformanceReport {
            total_queries: metrics.len(),
            slowest_queries: sorted_metrics.into_iter().take(10).collect(),
            total_alerts: alerts.len(),
            recent_alerts: alerts.iter().rev().take(20).cloned().collect(),
            overall_cache_hit_rate: self.calculate_overall_cache_hit_rate(&metrics).await,
            overall_performance_score: self.calculate_performance_score(&metrics).await,
        }
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Inheritance chain queries optimized with specialized indices
- [ ] Query result caching with intelligent invalidation strategies
- [ ] Execution plan analysis and optimization recommendations
- [ ] Batch query optimization for multiple concept lookups
- [ ] Query performance monitoring with alerting

### Performance Requirements
- [ ] Inheritance chain resolution < 10ms for chains up to 10 levels
- [ ] Property resolution queries < 5ms with caching
- [ ] Batch queries handle 100+ concepts within 50ms
- [ ] Query cache hit ratio > 90% for repeated queries
- [ ] Index utilization rate > 95% for optimized queries

### Testing Requirements
- [ ] Unit tests for query optimization algorithms
- [ ] Performance benchmarks for optimized vs unoptimized queries
- [ ] Load tests with concurrent query execution
- [ ] Cache effectiveness tests under various patterns

## Validation Steps

1. **Test query optimization effectiveness**:
   ```rust
   let optimizer = QueryOptimizer::new(config).await;
   let optimized = optimizer.optimize_inheritance_chain_query("concept_123", Some(10)).await?;
   assert!(optimized.estimated_improvement > 0.3); // 30% improvement
   ```

2. **Benchmark query performance**:
   ```rust
   let start = Instant::now();
   let chain = optimized_query_builder.build_inheritance_chain_query("concept", None).await?;
   let duration = start.elapsed();
   assert!(duration < Duration::from_millis(10));
   ```

3. **Run optimization tests**:
   ```bash
   cargo test query_optimization_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/optimization/query_optimizer.rs` - Core query optimization engine
- `src/inheritance/optimization/optimized_queries.rs` - Pre-optimized query builders
- `src/inheritance/optimization/query_monitor.rs` - Performance monitoring
- `src/inheritance/optimization/mod.rs` - Module exports
- `tests/inheritance/optimization_tests.rs` - Query optimization test suite

## Success Metrics
- Query performance improvement: >50% for inheritance operations
- Cache hit ratio: >90% for repeated queries
- Index utilization: >95% for all optimized queries
- Slow query reduction: >80% fewer queries above threshold

## Next Task
Upon completion, proceed to **17_semantic_search.md** to implement semantic similarity search capabilities.