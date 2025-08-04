# Micro Task 36: Query Processor

**Priority**: CRITICAL  
**Estimated Time**: 60 minutes  
**Dependencies**: Tasks 01-35 completed  
**Skills Required**: System integration, pipeline design

## Objective

Implement the main query processing pipeline that integrates all components into a unified, high-performance query system.

## Context

The QueryProcessor is the central orchestrator that coordinates spreading activation, intent recognition, attention mechanisms, pathway management, and explanation generation to deliver complete query results.

## Specifications

### Core Pipeline Requirements

1. **Unified Interface**
   - Single entry point for all query types
   - Consistent result format
   - Error handling and recovery
   - Performance monitoring integration

2. **Component Integration**
   - Intent parser → activation seeds
   - Spreader → attention mechanism
   - Pathways → explanations
   - TMS → belief integration

3. **Performance Targets**
   - < 50ms for complex queries
   - > 100 concurrent queries/second
   - < 50MB memory usage
   - > 90% intent recognition accuracy

## Implementation Guide

### Step 1: Main Processor Structure
```rust
// File: src/query/processor.rs

use std::sync::Arc;
use tokio::sync::RwLock;

pub struct QueryProcessor {
    // Core components
    intent_parser: Arc<QueryIntentParser>,
    spreader: Arc<ActivationSpreader>,
    attention: Arc<AttentionMechanism>,
    pathway_tracer: Arc<PathwayTracer>,
    explainer: Arc<QueryExplainer>,
    
    // Integration components
    belief_engine: Option<Arc<BeliefAwareQueryEngine>>,
    cache_manager: Arc<QueryCacheManager>,
    performance_monitor: Arc<PerformanceMonitor>,
    
    // Configuration
    config: QueryProcessorConfig,
    
    // State
    active_queries: Arc<RwLock<HashMap<QueryId, ActiveQuery>>>,
}

#[derive(Debug, Clone)]
pub struct QueryProcessorConfig {
    pub max_concurrent_queries: usize,
    pub default_timeout: Duration,
    pub enable_caching: bool,
    pub enable_belief_integration: bool,
    pub performance_logging: bool,
}

impl Default for QueryProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_queries: 1000,
            default_timeout: Duration::from_secs(30),
            enable_caching: true,
            enable_belief_integration: true,
            performance_logging: true,
        }
    }
}
```

### Step 2: Query Processing Pipeline
```rust
impl QueryProcessor {
    pub async fn process_query(
        &self,
        query: &str,
        context: &QueryContext,
    ) -> Result<QueryResult> {
        let query_id = QueryId::generate();
        let start_time = Instant::now();
        
        // Register active query
        self.register_active_query(query_id, query).await?;
        
        let result = self.process_query_internal(query_id, query, context).await;
        
        // Cleanup and metrics
        self.unregister_active_query(query_id).await;
        self.record_query_metrics(query_id, &result, start_time.elapsed());
        
        result
    }
    
    async fn process_query_internal(
        &self,
        query_id: QueryId,
        query: &str,
        context: &QueryContext,
    ) -> Result<QueryResult> {
        // Step 1: Check cache
        if self.config.enable_caching {
            if let Some(cached) = self.cache_manager.get(query, context).await? {
                return Ok(cached);
            }
        }
        
        // Step 2: Parse intent
        let parsed_query = self.intent_parser.parse_intent(query).await?;
        self.update_query_progress(query_id, QueryPhase::IntentParsed).await;
        
        // Step 3: Handle compound queries
        if !parsed_query.sub_queries.is_empty() {
            return self.process_compound_query(query_id, parsed_query, context).await;
        }
        
        // Step 4: Create activation seeds
        let seeds = self.create_activation_seeds(&parsed_query, context).await?;
        self.update_query_progress(query_id, QueryPhase::SeedsCreated).await;
        
        // Step 5: Spread activation with attention
        let activation_result = self.spread_with_attention(
            &seeds,
            &parsed_query,
            context,
        ).await?;
        self.update_query_progress(query_id, QueryPhase::ActivationComplete).await;
        
        // Step 6: Extract and rank results
        let raw_results = self.extract_results(&activation_result, &parsed_query).await?;
        let ranked_results = self.rank_results(raw_results, &parsed_query).await?;
        
        // Step 7: Generate explanations
        let explanations = if context.require_explanations {
            self.explainer.explain_results(&ranked_results, &activation_result).await?
        } else {
            Vec::new()
        };
        self.update_query_progress(query_id, QueryPhase::ExplanationsGenerated).await;
        
        // Step 8: Belief integration (if enabled)
        let final_results = if self.config.enable_belief_integration {
            self.integrate_beliefs(ranked_results, &parsed_query).await?
        } else {
            ranked_results
        };
        
        // Step 9: Build final result
        let result = QueryResult {
            query_id,
            original_query: query.to_string(),
            intent: parsed_query,
            results: final_results,
            explanations,
            activation_trace: activation_result.history,
            processing_time: start_time.elapsed(),
            metadata: self.build_result_metadata(&activation_result),
        };
        
        // Step 10: Cache result
        if self.config.enable_caching {
            self.cache_manager.store(query, context, &result).await?;
        }
        
        Ok(result)
    }
}
```

### Step 3: Activation with Attention Integration
```rust
impl QueryProcessor {
    async fn spread_with_attention(
        &self,
        seeds: &[ActivationSeed],
        parsed_query: &ParsedQuery,
        context: &QueryContext,
    ) -> Result<ActivationResult> {
        // Create initial activation state
        let mut initial_state = ActivationState::new();
        for seed in seeds {
            initial_state.set_activation(seed.node_id, seed.strength);
        }
        
        // Apply context bias
        self.apply_context_bias(&mut initial_state, context).await?;
        
        // Set up attention focus
        let attention_focus = self.create_attention_focus(parsed_query, context)?;
        self.attention.set_focus(attention_focus).await?;
        
        // Spread activation with attention guidance
        let spreader_with_attention = self.spreader.with_attention(self.attention.clone());
        let activation_result = spreader_with_attention
            .spread_activation(&initial_state, &self.graph)
            .await?;
        
        Ok(activation_result)
    }
    
    async fn apply_context_bias(
        &self,
        state: &mut ActivationState,
        context: &QueryContext,
    ) -> Result<()> {
        // Domain bias
        if let Some(domain) = &context.domain {
            let domain_nodes = self.graph.find_nodes_by_domain(domain).await?;
            for node in domain_nodes {
                let current = state.get_activation(node);
                state.set_activation(node, current + 0.1); // Small boost
            }
        }
        
        // Temporal bias
        if let Some(time_range) = &context.temporal_range {
            let temporal_nodes = self.graph.find_nodes_in_time_range(time_range).await?;
            for node in temporal_nodes {
                let current = state.get_activation(node);
                state.set_activation(node, current + 0.05);
            }
        }
        
        Ok(())
    }
}
```

### Step 4: Compound Query Handling
```rust
impl QueryProcessor {
    async fn process_compound_query(
        &self,
        query_id: QueryId,
        parsed_query: ParsedQuery,
        context: &QueryContext,
    ) -> Result<QueryResult> {
        let mut sub_results = Vec::new();
        
        // Process sub-queries in parallel
        let sub_query_futures: Vec<_> = parsed_query.sub_queries
            .iter()
            .enumerate()
            .map(|(i, sub_query)| {
                let processor = self.clone();
                let sub_context = context.clone();
                async move {
                    let sub_query_text = &sub_query.original_query;
                    processor.process_query_internal(
                        QueryId::sub_query(query_id, i),
                        sub_query_text,
                        &sub_context,
                    ).await
                }
            })
            .collect();
        
        let sub_results_vec = futures::try_join_all(sub_query_futures).await?;
        
        // Merge sub-query results
        let merged_result = self.merge_sub_query_results(
            parsed_query,
            sub_results_vec,
            context,
        ).await?;
        
        Ok(merged_result)
    }
    
    async fn merge_sub_query_results(
        &self,
        main_query: ParsedQuery,
        sub_results: Vec<QueryResult>,
        context: &QueryContext,
    ) -> Result<QueryResult> {
        // Combine results based on query intent
        let merger = SubQueryMerger::new(main_query.intent_type.clone());
        
        let merged_results = merger.merge_results(&sub_results)?;
        let merged_explanations = merger.merge_explanations(&sub_results)?;
        
        Ok(QueryResult {
            query_id: QueryId::generate(),
            original_query: main_query.original_query,
            intent: main_query,
            results: merged_results,
            explanations: merged_explanations,
            activation_trace: Vec::new(), // Combined trace would be too large
            processing_time: sub_results.iter()
                .map(|r| r.processing_time)
                .max()
                .unwrap_or_default(),
            metadata: ResultMetadata::compound(sub_results.len()),
        })
    }
}
```

### Step 5: Performance Monitoring Integration
```rust
impl QueryProcessor {
    fn record_query_metrics(&self, query_id: QueryId, result: &Result<QueryResult>, duration: Duration) {
        if !self.config.performance_logging {
            return;
        }
        
        let metrics = QueryMetrics {
            query_id,
            duration,
            success: result.is_ok(),
            result_count: result.as_ref().map(|r| r.results.len()).unwrap_or(0),
            activation_iterations: result.as_ref()
                .map(|r| r.activation_trace.len())
                .unwrap_or(0),
            memory_usage: self.estimate_memory_usage(result),
        };
        
        self.performance_monitor.record_query_metrics(metrics);
    }
    
    async fn check_resource_limits(&self) -> Result<()> {
        let active_count = self.active_queries.read().await.len();
        
        if active_count >= self.config.max_concurrent_queries {
            return Err(Error::TooManyConcurrentQueries);
        }
        
        let memory_usage = self.estimate_current_memory_usage().await;
        if memory_usage > self.config.max_memory_usage {
            return Err(Error::MemoryLimitExceeded);
        }
        
        Ok(())
    }
}
```

## File Locations

- `src/query/processor.rs` - Main implementation
- `src/query/pipeline.rs` - Pipeline stages
- `src/query/merger.rs` - Sub-query merging
- `src/query/monitoring.rs` - Performance monitoring
- `tests/query/processor_tests.rs` - Test implementation

## Success Criteria

- [ ] Full pipeline processes queries correctly
- [ ] Performance targets met (< 50ms)
- [ ] Concurrent query support functional
- [ ] Error handling robust
- [ ] Memory usage within limits
- [ ] All integration points working
- [ ] All tests pass

## Test Requirements

```rust
#[tokio::test]
async fn test_full_query_pipeline() {
    let processor = QueryProcessor::new().await;
    
    let query = "Find carnivorous mammals similar to wolves";
    let context = QueryContext::default();
    
    let start = Instant::now();
    let result = processor.process_query(query, &context).await.unwrap();
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(50));
    assert!(!result.results.is_empty());
    assert!(!result.explanations.is_empty());
    assert!(result.intent.confidence > 0.8);
}

#[tokio::test]
async fn test_concurrent_queries() {
    let processor = Arc::new(QueryProcessor::new().await);
    
    let queries = vec![
        "What animals live in water?",
        "How are birds related to reptiles?",
        "Find large predators",
    ];
    
    let handles: Vec<_> = queries.into_iter().map(|query| {
        let p = processor.clone();
        tokio::spawn(async move {
            p.process_query(query, &QueryContext::default()).await
        })
    }).collect();
    
    let results = futures::try_join_all(handles).await.unwrap();
    assert!(results.iter().all(|r| r.is_ok()));
}

#[tokio::test]
async fn test_compound_query_processing() {
    let processor = QueryProcessor::new().await;
    
    let query = "Compare the digestive systems of carnivorous and herbivorous mammals";
    let result = processor.process_query(query, &QueryContext::default()).await.unwrap();
    
    // Should have processed as compound query
    assert!(result.intent.sub_queries.len() >= 2);
    assert!(!result.results.is_empty());
}
```

## Quality Gates

- [ ] No memory leaks under sustained load
- [ ] Graceful degradation under resource pressure
- [ ] Consistent results for identical queries
- [ ] Proper error recovery and cleanup
- [ ] Performance metrics accurate

## Next Task

Upon completion, proceed to **37_parallel_optimization.md**