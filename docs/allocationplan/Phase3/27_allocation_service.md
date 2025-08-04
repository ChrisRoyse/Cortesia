# Task 27: Memory Allocation Service Implementation

**Estimated Time**: 15-20 minutes  
**Dependencies**: 26_knowledge_graph_service.md  
**Stage**: Service Layer  

## Objective
Implement a specialized memory allocation service that provides fine-grained control over memory placement, neural-guided allocation strategies, and real-time performance monitoring for knowledge graph operations.

## Specific Requirements

### 1. Neural-Guided Allocation Engine
- Integration with Phase 2 TTFS encoding for placement decisions
- Cortical column awareness for memory locality optimization
- Spike pattern analysis for allocation priority determination
- Dynamic allocation strategy adaptation based on access patterns

### 2. Memory Pool Management
- Pre-allocated memory pools for different concept types
- Dynamic pool resizing based on usage patterns
- Memory fragmentation prevention and defragmentation
- Resource quota enforcement and monitoring

### 3. Performance-Optimized Allocation
- Sub-10ms allocation response times
- Batch allocation support for bulk operations
- Allocation conflict resolution and retry mechanisms
- Real-time allocation metrics and monitoring

## Implementation Steps

### 1. Create Core Allocation Service
```rust
// src/services/allocation_service.rs
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};

pub struct MemoryAllocationService {
    // Core allocation components
    allocation_engine: Arc<AllocationEngine>,
    ttfs_integration: Arc<TTFSIntegrationService>,
    cortical_processor: Arc<MultiColumnProcessor>,
    
    // Memory management
    memory_pools: Arc<RwLock<HashMap<ConceptType, MemoryPool>>>,
    allocation_tracker: Arc<AllocationTracker>,
    fragmentation_monitor: Arc<FragmentationMonitor>,
    
    // Performance optimization
    allocation_cache: Arc<RwLock<LRUCache<String, AllocationDecision>>>,
    batch_processor: Arc<BatchAllocationProcessor>,
    metrics_collector: Arc<AllocationMetricsCollector>,
    
    // Resource management
    allocation_semaphore: Arc<Semaphore>,
    resource_monitor: Arc<ResourceMonitor>,
    
    config: AllocationServiceConfig,
}

impl MemoryAllocationService {
    pub async fn new(
        allocation_engine: Arc<AllocationEngine>,
        ttfs_integration: Arc<TTFSIntegrationService>,
        cortical_processor: Arc<MultiColumnProcessor>,
        config: AllocationServiceConfig,
    ) -> Result<Self, AllocationServiceError> {
        // Initialize memory pools
        let memory_pools = Arc::new(RwLock::new(
            Self::initialize_memory_pools(&config).await?
        ));
        
        // Initialize tracking and monitoring
        let allocation_tracker = Arc::new(AllocationTracker::new());
        let fragmentation_monitor = Arc::new(FragmentationMonitor::new());
        let metrics_collector = Arc::new(AllocationMetricsCollector::new());
        
        // Initialize optimization components
        let allocation_cache = Arc::new(RwLock::new(
            LRUCache::new(config.allocation_cache_size)
        ));
        let batch_processor = Arc::new(BatchAllocationProcessor::new(config.batch_size));
        
        // Initialize resource management
        let allocation_semaphore = Arc::new(Semaphore::new(config.max_concurrent_allocations));
        let resource_monitor = Arc::new(ResourceMonitor::new());
        
        Ok(Self {
            allocation_engine,
            ttfs_integration,
            cortical_processor,
            memory_pools,
            allocation_tracker,
            fragmentation_monitor,
            allocation_cache,
            batch_processor,
            metrics_collector,
            allocation_semaphore,
            resource_monitor,
            config,
        })
    }
    
    pub async fn allocate_memory(
        &self,
        request: AllocationRequest,
    ) -> Result<AllocationResult, AllocationError> {
        let allocation_start = Instant::now();
        
        // Acquire allocation permit
        let _permit = self.allocation_semaphore.acquire().await?;
        
        // Check allocation cache
        let cache_key = self.generate_allocation_cache_key(&request);
        if let Some(cached_decision) = self.allocation_cache.read().await.get(&cache_key) {
            if cached_decision.is_valid() {
                return self.execute_cached_allocation(&request, cached_decision).await;
            }
        }
        
        // Execute fresh allocation
        let result = self.execute_memory_allocation(request, allocation_start).await;
        
        // Update metrics
        let allocation_time = allocation_start.elapsed();
        self.metrics_collector.record_allocation(allocation_time, &result).await;
        
        result
    }
    
    async fn execute_memory_allocation(
        &self,
        request: AllocationRequest,
        allocation_start: Instant,
    ) -> Result<AllocationResult, AllocationError> {
        // Generate TTFS encoding for neural guidance
        let ttfs_encoding = self.ttfs_integration
            .encode_content(&request.content)
            .await?;
        
        // Analyze with cortical columns
        let cortical_analysis = self.cortical_processor
            .analyze_content_for_allocation(&request.content, &ttfs_encoding)
            .await?;
        
        // Determine optimal allocation strategy
        let allocation_strategy = self.determine_allocation_strategy(
            &request,
            &ttfs_encoding,
            &cortical_analysis,
        ).await?;
        
        // Select memory pool
        let selected_pool = self.select_memory_pool(
            &request.concept_type,
            &allocation_strategy,
        ).await?;
        
        // Execute allocation with neural guidance
        let placement_decision = self.allocation_engine
            .determine_optimal_placement(
                &request.content,
                &TTFSSpikePattern {
                    first_spike_time: ttfs_encoding,
                    content_hash: self.generate_content_hash(&request.content),
                    creation_time: allocation_start,
                },
                &cortical_analysis,
            )
            .await?;
        
        // Allocate memory slot
        let memory_slot = selected_pool
            .allocate_slot(&placement_decision, &request)
            .await?;
        
        // Track allocation
        self.allocation_tracker
            .track_allocation(&memory_slot, &placement_decision)
            .await;
        
        // Cache allocation decision
        let allocation_decision = AllocationDecision {
            memory_slot: memory_slot.clone(),
            placement_decision: placement_decision.clone(),
            strategy: allocation_strategy.clone(),
            timestamp: Utc::now(),
            ttl: Duration::from_secs(3600), // 1 hour cache
        };
        
        let cache_key = self.generate_allocation_cache_key(&request);
        self.allocation_cache.write().await.put(cache_key, allocation_decision);
        
        Ok(AllocationResult {
            memory_slot,
            allocation_path: placement_decision.hierarchy_path,
            neural_pathway_id: placement_decision.neural_pathway_reference,
            ttfs_encoding,
            cortical_column_id: cortical_analysis.assigned_column_id,
            allocation_time_ms: allocation_start.elapsed().as_millis() as u64,
            confidence_score: placement_decision.confidence_score,
            memory_pool_id: selected_pool.pool_id(),
            allocation_strategy: allocation_strategy.strategy_type,
        })
    }
    
    pub async fn batch_allocate_memory(
        &self,
        requests: Vec<AllocationRequest>,
    ) -> Result<Vec<AllocationResult>, AllocationError> {
        // Process batch with parallel allocation
        self.batch_processor
            .process_allocation_batch(requests, self)
            .await
    }
    
    pub async fn deallocate_memory(
        &self,
        deallocation_request: DeallocationRequest,
    ) -> Result<DeallocationResult, AllocationError> {
        let deallocation_start = Instant::now();
        
        // Find memory slot
        let memory_slot = self.allocation_tracker
            .find_allocation(&deallocation_request.concept_id)
            .await?;
        
        // Deallocate from appropriate pool
        let pool = self.memory_pools.read().await
            .get(&memory_slot.concept_type)
            .ok_or(AllocationError::PoolNotFound)?
            .clone();
        
        pool.deallocate_slot(&memory_slot).await?;
        
        // Update tracking
        self.allocation_tracker
            .track_deallocation(&deallocation_request.concept_id)
            .await;
        
        // Update metrics
        let deallocation_time = deallocation_start.elapsed();
        self.metrics_collector.record_deallocation(deallocation_time).await;
        
        Ok(DeallocationResult {
            concept_id: deallocation_request.concept_id,
            freed_memory_size: memory_slot.size_bytes,
            deallocation_time_ms: deallocation_time.as_millis() as u64,
            pool_id: memory_slot.pool_id,
        })
    }
    
    async fn determine_allocation_strategy(
        &self,
        request: &AllocationRequest,
        ttfs_encoding: &TTFSEncoding,
        cortical_analysis: &CorticalAnalysis,
    ) -> Result<AllocationStrategy, AllocationError> {
        // Analyze request characteristics
        let content_complexity = self.analyze_content_complexity(&request.content);
        let access_pattern_prediction = self.predict_access_pattern(&request);
        let resource_requirements = self.estimate_resource_requirements(&request);
        
        // Select strategy based on analysis
        let strategy_type = if content_complexity.is_high() {
            AllocationStrategyType::HighComplexity
        } else if access_pattern_prediction.is_frequent() {
            AllocationStrategyType::HighPerformance
        } else if resource_requirements.is_large() {
            AllocationStrategyType::ResourceOptimized
        } else {
            AllocationStrategyType::Balanced
        };
        
        Ok(AllocationStrategy {
            strategy_type,
            neural_guidance_weight: cortical_analysis.confidence_score,
            locality_preference: cortical_analysis.locality_preference,
            performance_priority: access_pattern_prediction.priority_score,
            resource_constraints: resource_requirements,
        })
    }
    
    async fn select_memory_pool(
        &self,
        concept_type: &ConceptType,
        strategy: &AllocationStrategy,
    ) -> Result<Arc<MemoryPool>, AllocationError> {
        let pools = self.memory_pools.read().await;
        
        let pool = pools.get(concept_type)
            .ok_or(AllocationError::PoolNotFound)?;
        
        // Check pool capacity and health
        if !pool.has_capacity_for_strategy(strategy).await {
            // Attempt pool expansion or find alternative
            return self.find_alternative_pool(concept_type, strategy).await;
        }
        
        Ok(pool.clone())
    }
    
    pub async fn get_allocation_metrics(&self) -> AllocationMetrics {
        AllocationMetrics {
            total_allocations: self.allocation_tracker.get_total_allocations().await,
            active_allocations: self.allocation_tracker.get_active_allocations().await,
            average_allocation_time: self.metrics_collector.get_average_allocation_time().await,
            memory_utilization: self.calculate_memory_utilization().await,
            fragmentation_level: self.fragmentation_monitor.get_fragmentation_level().await,
            cache_hit_rate: self.calculate_cache_hit_rate().await,
            pool_statistics: self.get_pool_statistics().await,
        }
    }
    
    async fn initialize_memory_pools(
        config: &AllocationServiceConfig,
    ) -> Result<HashMap<ConceptType, MemoryPool>, AllocationServiceError> {
        let mut pools = HashMap::new();
        
        for pool_config in &config.pool_configurations {
            let pool = MemoryPool::new(pool_config.clone()).await?;
            pools.insert(pool_config.concept_type.clone(), pool);
        }
        
        Ok(pools)
    }
}

#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub concept_id: String,
    pub concept_type: ConceptType,
    pub content: String,
    pub semantic_embedding: Option<Vec<f32>>,
    pub priority: AllocationPriority,
    pub resource_requirements: ResourceRequirements,
    pub locality_hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub memory_slot: MemorySlot,
    pub allocation_path: Vec<String>,
    pub neural_pathway_id: String,
    pub ttfs_encoding: TTFSEncoding,
    pub cortical_column_id: String,
    pub allocation_time_ms: u64,
    pub confidence_score: f64,
    pub memory_pool_id: String,
    pub allocation_strategy: AllocationStrategyType,
}
```

### 2. Implement Memory Pool Management
```rust
// src/services/allocation_service/memory_pool.rs
#[derive(Debug, Clone)]
pub struct MemoryPool {
    pool_id: String,
    concept_type: ConceptType,
    total_capacity: usize,
    allocated_slots: Arc<RwLock<HashMap<String, MemorySlot>>>,
    free_slots: Arc<RwLock<Vec<MemorySlot>>>,
    fragmentation_tracker: Arc<FragmentationTracker>,
    config: PoolConfig,
}

impl MemoryPool {
    pub async fn allocate_slot(
        &self,
        placement_decision: &PlacementDecision,
        request: &AllocationRequest,
    ) -> Result<MemorySlot, AllocationError> {
        let mut free_slots = self.free_slots.write().await;
        
        // Find optimal slot based on placement decision
        let slot_index = self.find_optimal_slot_index(
            &free_slots,
            placement_decision,
            request,
        )?;
        
        let mut slot = free_slots.remove(slot_index);
        slot.concept_id = Some(request.concept_id.clone());
        slot.allocation_timestamp = Some(Utc::now());
        
        // Track allocation
        self.allocated_slots.write().await.insert(
            request.concept_id.clone(),
            slot.clone(),
        );
        
        Ok(slot)
    }
    
    pub async fn has_capacity_for_strategy(
        &self,
        strategy: &AllocationStrategy,
    ) -> bool {
        let free_slots = self.free_slots.read().await;
        let required_size = strategy.resource_constraints.memory_size;
        
        free_slots.iter().any(|slot| slot.size_bytes >= required_size)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Neural-guided allocation decisions work correctly
- [ ] Memory pools manage resources efficiently
- [ ] Batch allocation processes multiple requests
- [ ] Deallocation releases resources properly
- [ ] Cache improves allocation performance

### Performance Requirements
- [ ] Single allocation operations < 10ms
- [ ] Batch allocations process 100+ requests/second
- [ ] Memory utilization > 85% efficiency
- [ ] Cache hit rate > 70% for repeated patterns

### Testing Requirements
- [ ] Unit tests for allocation algorithms
- [ ] Performance tests for allocation speed
- [ ] Memory leak tests for deallocation
- [ ] Concurrent allocation stress tests

## Validation Steps

1. **Test allocation service creation**:
   ```rust
   let service = MemoryAllocationService::new(engine, ttfs, processor, config).await?;
   let metrics = service.get_allocation_metrics().await;
   ```

2. **Test memory allocation operations**:
   ```rust
   let result = service.allocate_memory(allocation_request).await?;
   assert!(result.allocation_time_ms < 10);
   ```

3. **Run allocation service tests**:
   ```bash
   cargo test allocation_service_tests
   ```

## Files to Create/Modify
- `src/services/allocation_service.rs` - Main allocation service
- `src/services/allocation_service/memory_pool.rs` - Memory pool management
- `src/services/allocation_service/metrics.rs` - Allocation metrics
- `tests/services/allocation_service_tests.rs` - Test suite

## Success Metrics
- Allocation latency: < 10ms (99th percentile)
- Memory efficiency: > 85% utilization
- Cache effectiveness: > 70% hit rate
- Concurrent allocations: 1000+ operations/second

## Next Task
Upon completion, proceed to **28_retrieval_service.md** to implement the memory retrieval service.