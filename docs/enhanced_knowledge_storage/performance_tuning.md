# Performance Tuning Guide

Comprehensive guide for optimizing the Enhanced Knowledge Storage System for various performance requirements and resource constraints.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Configuration Optimization](#configuration-optimization)
- [Model Selection Strategy](#model-selection-strategy)
- [Memory Management](#memory-management)
- [Processing Optimization](#processing-optimization)
- [Storage Optimization](#storage-optimization)
- [Monitoring and Profiling](#monitoring-and-profiling)
- [Deployment Scenarios](#deployment-scenarios)
- [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Performance Overview

### Key Performance Metrics

The Enhanced Knowledge Storage System tracks several critical performance metrics:

- **Processing Latency**: Time to process documents (2-10 seconds typical)
- **Memory Usage**: RAM consumed by models and processing (200MB-8GB range)
- **Throughput**: Documents processed per minute (varies by complexity)
- **Quality Score**: Processing accuracy (0.0-1.0, target >0.7)
- **Cache Hit Rate**: Model loading efficiency (target >80%)

### Performance Characteristics by Configuration

| Configuration | Processing Time | Memory Usage | Quality Score | Use Case |
|---------------|----------------|--------------|---------------|----------|
| Minimal | 1-3 seconds | 200MB-500MB | 0.6-0.7 | Development, testing |
| Balanced | 3-6 seconds | 1GB-2GB | 0.7-0.8 | Production, general use |
| High Quality | 5-15 seconds | 2GB-8GB | 0.8-0.95 | Research, critical applications |
| Memory Constrained | 2-8 seconds | 200MB-1GB | 0.5-0.7 | Edge devices, containers |

## Configuration Optimization

### Memory-Optimized Configuration

```rust
use llmkg::enhanced_knowledge_storage::*;
use std::time::Duration;

/// Optimized for environments with limited memory (1GB or less)
pub fn create_memory_optimized_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 800_000_000,     // 800MB limit
        max_concurrent_models: 1,          // Only one model at a time
        idle_timeout: Duration::from_secs(30), // Aggressive eviction
        min_memory_threshold: 100_000_000, // 100MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Use smallest available models
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        
        // Small chunks to minimize memory peaks
        max_chunk_size: 512,
        min_chunk_size: 64,
        chunk_overlap_size: 16,
        
        // Relaxed thresholds for efficiency
        min_entity_confidence: 0.4,
        min_relationship_confidence: 0.3,
        
        // Reduce processing overhead
        preserve_context: false,
        enable_quality_validation: false,
    };
    
    (model_config, processing_config)
}
```

### Speed-Optimized Configuration

```rust
/// Optimized for maximum processing speed
pub fn create_speed_optimized_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 4_000_000_000,   // 4GB for model caching
        max_concurrent_models: 3,          // Keep multiple models loaded
        idle_timeout: Duration::from_secs(600), // Keep models in memory longer
        min_memory_threshold: 500_000_000, // 500MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Use smaller, faster models
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m".to_string(), 
        semantic_analysis_model: "smollm2_135m".to_string(),
        
        // Optimize chunk sizes for processing speed
        max_chunk_size: 1024,
        min_chunk_size: 128,
        chunk_overlap_size: 32,
        
        // Lower quality thresholds for speed
        min_entity_confidence: 0.5,
        min_relationship_confidence: 0.4,
        
        // Disable expensive operations
        preserve_context: false,
        enable_quality_validation: false,
    };
    
    (model_config, processing_config)
}
```

### Quality-Optimized Configuration

```rust
/// Optimized for maximum processing quality
pub fn create_quality_optimized_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 8_000_000_000,   // 8GB for large models
        max_concurrent_models: 5,          // Multiple specialized models
        idle_timeout: Duration::from_secs(1800), // Keep models loaded longer
        min_memory_threshold: 1_000_000_000, // 1GB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Use largest, most capable models
        entity_extraction_model: "smollm_1_7b".to_string(),
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        semantic_analysis_model: "smollm2_360m".to_string(),
        
        // Larger chunks for better context
        max_chunk_size: 4096,
        min_chunk_size: 256,
        chunk_overlap_size: 128,
        
        // High quality thresholds
        min_entity_confidence: 0.8,
        min_relationship_confidence: 0.7,
        
        // Enable all quality features
        preserve_context: true,
        enable_quality_validation: true,
    };
    
    (model_config, processing_config)
}
```

### Balanced Configuration

```rust
/// Balanced configuration for production use
pub fn create_balanced_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 2_000_000_000,   // 2GB balanced limit
        max_concurrent_models: 3,          // Reasonable concurrency
        idle_timeout: Duration::from_secs(300), // 5-minute timeout
        min_memory_threshold: 200_000_000, // 200MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        // Balanced model selection
        entity_extraction_model: "smollm2_360m".to_string(),
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        
        // Balanced chunk sizes
        max_chunk_size: 2048,
        min_chunk_size: 128,
        chunk_overlap_size: 64,
        
        // Moderate quality thresholds
        min_entity_confidence: 0.6,
        min_relationship_confidence: 0.5,
        
        // Enable context preservation
        preserve_context: true,
        enable_quality_validation: true,
    };
    
    (model_config, processing_config)
}
```

## Model Selection Strategy

### Performance vs Quality Trade-offs

| Model | Parameters | Memory | Speed | Quality | Best For |
|-------|------------|--------|-------|---------|----------|
| smollm2_135m | 135M | ~300MB | Fast | Good | Development, speed-critical |
| smollm2_360m | 360M | ~800MB | Medium | Better | Production, balanced |
| smollm_1_7b | 1.7B | ~3.5GB | Slow | Best | Research, high-quality |
| smollm_135m_instruct | 135M | ~300MB | Fast | Good | Instruction following |
| smollm_360m_instruct | 360M | ~800MB | Medium | Better | Complex instructions |

### Dynamic Model Selection

```rust
pub struct AdaptiveModelSelector {
    available_memory: u64,
    processing_requirements: QualityRequirements,
    performance_constraints: PerformanceConstraints,
}

impl AdaptiveModelSelector {
    pub fn select_optimal_models(&self) -> ModelConfiguration {
        let mut config = ModelConfiguration::default();
        
        // Select entity extraction model based on memory and quality requirements
        config.entity_extraction_model = if self.available_memory > 3_000_000_000 && 
                                            self.processing_requirements.quality_target > 0.8 {
            "smollm_1_7b".to_string()
        } else if self.available_memory > 1_000_000_000 {
            "smollm2_360m".to_string()
        } else {
            "smollm2_135m".to_string()
        };
        
        // Select relationship extraction model
        config.relationship_extraction_model = if self.available_memory > 1_000_000_000 &&
                                                  self.processing_requirements.relationship_accuracy > 0.7 {
            "smollm_360m_instruct".to_string()
        } else {
            "smollm_135m_instruct".to_string()
        };
        
        // Select semantic analysis model
        config.semantic_analysis_model = if self.performance_constraints.max_latency < Duration::from_secs(5) {
            "smollm2_135m".to_string()
        } else {
            "smollm2_360m".to_string()
        };
        
        config
    }
}

#[derive(Debug, Clone)]
pub struct QualityRequirements {
    pub quality_target: f32,
    pub entity_accuracy: f32,
    pub relationship_accuracy: f32,
    pub context_preservation: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceConstraints {
    pub max_latency: Duration,
    pub max_memory_usage: u64,
    pub throughput_target: f32, // documents per minute
}
```

## Memory Management

### Memory Monitoring and Optimization

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct AdvancedMemoryManager {
    current_usage: Arc<RwLock<u64>>,
    peak_usage: Arc<RwLock<u64>>,
    model_memory_tracker: ModelMemoryTracker,
    gc_coordinator: GarbageCollectionCoordinator,
}

impl AdvancedMemoryManager {
    pub async fn optimize_memory_usage(&self) -> MemoryOptimizationResult {
        let current = *self.current_usage.read().await;
        let available = self.get_available_memory();
        
        if current > 0.8 * available {
            // Critical memory situation - aggressive cleanup
            self.perform_aggressive_cleanup().await
        } else if current > 0.6 * available {
            // High memory usage - moderate cleanup
            self.perform_moderate_cleanup().await
        } else {
            // Normal operation - no cleanup needed
            MemoryOptimizationResult::NoActionNeeded
        }
    }
    
    async fn perform_aggressive_cleanup(&self) -> MemoryOptimizationResult {
        let mut freed_memory = 0u64;
        
        // Evict all idle models
        freed_memory += self.model_memory_tracker.evict_all_idle_models().await;
        
        // Force garbage collection
        self.gc_coordinator.force_collection().await;
        
        // Clear processing caches
        freed_memory += self.clear_processing_caches().await;
        
        MemoryOptimizationResult::Optimized { 
            memory_freed: freed_memory,
            optimization_level: OptimizationLevel::Aggressive 
        }
    }
    
    async fn perform_moderate_cleanup(&self) -> MemoryOptimizationResult {
        let mut freed_memory = 0u64;
        
        // Evict least recently used models
        freed_memory += self.model_memory_tracker.evict_lru_models(2).await;
        
        // Clear old cache entries
        freed_memory += self.clear_old_cache_entries().await;
        
        MemoryOptimizationResult::Optimized { 
            memory_freed: freed_memory,
            optimization_level: OptimizationLevel::Moderate 
        }
    }
}

#[derive(Debug)]
pub enum MemoryOptimizationResult {
    NoActionNeeded,
    Optimized { memory_freed: u64, optimization_level: OptimizationLevel },
}

#[derive(Debug)]
pub enum OptimizationLevel {
    Moderate,
    Aggressive,
}
```

### Memory-Aware Processing

```rust
pub struct MemoryAwareProcessor {
    processor: IntelligentKnowledgeProcessor,
    memory_manager: Arc<AdvancedMemoryManager>,
    processing_queue: VecDeque<ProcessingTask>,
}

impl MemoryAwareProcessor {
    pub async fn process_with_memory_management(
        &mut self,
        content: &str,
        title: &str
    ) -> Result<KnowledgeProcessingResult, EnhancedStorageError> {
        // Check memory before processing
        let memory_status = self.memory_manager.check_memory_availability().await;
        
        match memory_status {
            MemoryStatus::Available => {
                // Proceed with normal processing
                self.processor.process_knowledge(content, title).await
            },
            MemoryStatus::Limited => {
                // Use memory-constrained configuration
                self.process_with_limited_memory(content, title).await
            },
            MemoryStatus::Critical => {
                // Queue for later processing or use minimal processing
                self.queue_or_process_minimal(content, title).await
            }
        }
    }
    
    async fn process_with_limited_memory(
        &self,
        content: &str,
        title: &str
    ) -> Result<KnowledgeProcessingResult, EnhancedStorageError> {
        // Temporarily reduce chunk sizes and model complexity
        let original_config = self.processor.get_config().clone();
        let limited_config = KnowledgeProcessingConfig {
            max_chunk_size: original_config.max_chunk_size / 2,
            min_chunk_size: original_config.min_chunk_size / 2,
            entity_extraction_model: "smollm2_135m".to_string(),
            ..original_config
        };
        
        // Process with limited configuration
        let temp_processor = IntelligentKnowledgeProcessor::new(
            self.processor.model_manager.clone(),
            limited_config
        );
        
        temp_processor.process_knowledge(content, title).await
    }
}

#[derive(Debug, PartialEq)]
pub enum MemoryStatus {
    Available,   // < 60% usage
    Limited,     // 60-80% usage
    Critical,    // > 80% usage
}
```

## Processing Optimization

### Parallel Processing Strategies

```rust
use tokio::task::JoinSet;
use std::sync::Arc;

pub struct ParallelProcessingEngine {
    processors: Vec<Arc<IntelligentKnowledgeProcessor>>,
    load_balancer: LoadBalancer,
    task_scheduler: TaskScheduler,
}

impl ParallelProcessingEngine {
    pub async fn process_documents_parallel(
        &self,
        documents: Vec<(String, String)>, // (content, title) pairs
        concurrency_limit: usize
    ) -> Vec<Result<KnowledgeProcessingResult, EnhancedStorageError>> {
        let mut join_set = JoinSet::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency_limit));
        
        for (content, title) in documents {
            let processor = self.load_balancer.select_least_loaded_processor();
            let semaphore_clone = semaphore.clone();
            
            join_set.spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                processor.process_knowledge(&content, &title).await
            });
        }
        
        let mut results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            results.push(result.unwrap());
        }
        
        results
    }
    
    pub async fn process_large_document_streaming(
        &self,
        content: &str,
        title: &str,
        chunk_size: usize
    ) -> Result<KnowledgeProcessingResult, EnhancedStorageError> {
        // Split document into processing chunks
        let content_chunks = self.split_document_intelligently(content, chunk_size);
        
        // Process chunks in parallel
        let chunk_results = self.process_chunks_parallel(content_chunks, title).await?;
        
        // Merge results intelligently
        self.merge_processing_results(chunk_results, title).await
    }
    
    fn split_document_intelligently(&self, content: &str, target_chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        
        // Split by paragraphs first
        for paragraph in content.split("\n\n") {
            if current_chunk.len() + paragraph.len() > target_chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }
            
            if paragraph.len() > target_chunk_size {
                // Split large paragraphs by sentences
                for sentence in paragraph.split(". ") {
                    if current_chunk.len() + sentence.len() > target_chunk_size && !current_chunk.is_empty() {
                        chunks.push(current_chunk.clone());
                        current_chunk.clear();
                    }
                    current_chunk.push_str(sentence);
                    if !sentence.ends_with('.') {
                        current_chunk.push_str(". ");
                    }
                }
            } else {
                current_chunk.push_str(paragraph);
                current_chunk.push_str("\n\n");
            }
        }
        
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        
        chunks
    }
}
```

### Caching Strategies

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tokio::sync::RwLock;

pub struct MultiLevelCache {
    model_cache: Arc<RwLock<HashMap<String, CachedModel>>>,
    processing_cache: Arc<RwLock<HashMap<u64, CachedProcessingResult>>>,
    entity_cache: Arc<RwLock<HashMap<u64, Vec<ContextualEntity>>>>,
    chunk_cache: Arc<RwLock<HashMap<u64, Vec<SemanticChunk>>>>,
}

impl MultiLevelCache {
    pub async fn get_or_process(
        &self,
        content: &str,
        title: &str,
        processor: &IntelligentKnowledgeProcessor
    ) -> Result<KnowledgeProcessingResult, EnhancedStorageError> {
        let content_hash = self.calculate_content_hash(content, title);
        
        // Check processing cache first
        if let Some(cached_result) = self.get_cached_processing_result(content_hash).await {
            return Ok(cached_result);
        }
        
        // Check partial caches (entities, chunks)
        let cached_entities = self.get_cached_entities(content_hash).await;
        let cached_chunks = self.get_cached_chunks(content_hash).await;
        
        if cached_entities.is_some() && cached_chunks.is_some() {
            // Reconstruct result from partial caches
            return self.reconstruct_from_partial_cache(
                cached_entities.unwrap(),
                cached_chunks.unwrap(),
                content,
                title
            ).await;
        }
        
        // No cache hit - process normally and cache results
        let result = processor.process_knowledge(content, title).await?;
        
        // Cache the full result and components
        self.cache_processing_result(content_hash, &result).await;
        self.cache_entities(content_hash, &result.global_entities).await;
        self.cache_chunks(content_hash, &result.chunks).await;
        
        Ok(result)
    }
    
    fn calculate_content_hash(&self, content: &str, title: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        title.hash(&mut hasher);
        hasher.finish()
    }
    
    pub async fn evict_oldest_entries(&self, target_evictions: usize) {
        // Implement LRU eviction logic
        let mut processing_cache = self.processing_cache.write().await;
        let mut entity_cache = self.entity_cache.write().await;
        let mut chunk_cache = self.chunk_cache.write().await;
        
        // Sort by access time and remove oldest entries
        // Implementation would track access times and perform LRU eviction
    }
}

#[derive(Debug, Clone)]
pub struct CachedProcessingResult {
    pub result: KnowledgeProcessingResult,
    pub cached_at: std::time::Instant,
    pub access_count: u64,
    pub last_accessed: std::time::Instant,
}
```

## Storage Optimization

### Hierarchical Storage Optimization

```rust
pub struct OptimizedHierarchicalStorage {
    storage_engine: HierarchicalStorageEngine,
    index_optimizer: IndexOptimizer,
    compression_manager: CompressionManager,
}

impl OptimizedHierarchicalStorage {
    pub async fn store_with_optimization(
        &self,
        result: &KnowledgeProcessingResult
    ) -> Result<StorageResult, StorageError> {
        // Analyze content for optimal storage strategy
        let storage_strategy = self.analyze_optimal_storage_strategy(result);
        
        match storage_strategy {
            StorageStrategy::HighCompression => {
                self.store_with_high_compression(result).await
            },
            StorageStrategy::FastAccess => {
                self.store_for_fast_access(result).await
            },
            StorageStrategy::Balanced => {
                self.store_balanced(result).await
            }
        }
    }
    
    fn analyze_optimal_storage_strategy(&self, result: &KnowledgeProcessingResult) -> StorageStrategy {
        let entity_density = result.global_entities.len() as f32 / result.chunks.len() as f32;
        let avg_chunk_size = result.chunks.iter()
            .map(|c| c.content.len())
            .sum::<usize>() as f32 / result.chunks.len() as f32;
        
        if entity_density > 5.0 && avg_chunk_size > 2000.0 {
            StorageStrategy::HighCompression
        } else if entity_density < 2.0 && avg_chunk_size < 1000.0 {
            StorageStrategy::FastAccess
        } else {
            StorageStrategy::Balanced
        }
    }
    
    async fn store_with_high_compression(
        &self,
        result: &KnowledgeProcessingResult
    ) -> Result<StorageResult, StorageError> {
        // Use aggressive compression for entity-dense documents
        let compressed_entities = self.compression_manager
            .compress_entities(&result.global_entities, CompressionLevel::High).await?;
        
        let compressed_chunks = self.compression_manager
            .compress_chunks(&result.chunks, CompressionLevel::High).await?;
        
        self.storage_engine.store_compressed(
            &result.document_id,
            compressed_entities,
            compressed_chunks
        ).await
    }
}

#[derive(Debug, Clone)]
pub enum StorageStrategy {
    HighCompression,  // Optimize for storage space
    FastAccess,       // Optimize for retrieval speed
    Balanced,         // Balance compression and access speed
}
```

### Index Optimization

```rust
pub struct AdaptiveIndexManager {
    semantic_index: SemanticIndex,
    entity_index: EntityIndex,
    relationship_index: RelationshipIndex,
    access_pattern_analyzer: AccessPatternAnalyzer,
}

impl AdaptiveIndexManager {
    pub async fn optimize_indices_based_on_usage(&self) {
        let usage_patterns = self.access_pattern_analyzer.analyze_recent_patterns().await;
        
        match usage_patterns.primary_access_pattern {
            AccessPattern::EntityHeavy => {
                self.optimize_entity_index().await;
            },
            AccessPattern::SemanticSearch => {
                self.optimize_semantic_index().await;
            },
            AccessPattern::RelationshipTraversal => {
                self.optimize_relationship_index().await;
            },
            AccessPattern::Mixed => {
                self.balance_all_indices().await;
            }
        }
    }
    
    async fn optimize_entity_index(&self) {
        // Rebuild entity index with optimized structure
        self.entity_index.rebuild_with_config(EntityIndexConfig {
            clustering_threshold: 0.8,
            max_cluster_size: 1000,
            enable_fuzzy_matching: true,
            cache_frequent_entities: true,
        }).await;
    }
    
    async fn optimize_semantic_index(&self) {
        // Optimize semantic index for similarity search
        self.semantic_index.rebuild_with_config(SemanticIndexConfig {
            vector_dimensions: 384,
            index_type: IndexType::HNSW,
            ef_construction: 200,
            m_connections: 16,
            enable_quantization: true,
        }).await;
    }
}

#[derive(Debug, Clone)]
pub struct UsagePatterns {
    pub primary_access_pattern: AccessPattern,
    pub query_frequency: HashMap<String, u64>,
    pub entity_access_frequency: HashMap<String, u64>,
    pub relationship_traversal_frequency: u64,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    EntityHeavy,           // Frequent entity lookups
    SemanticSearch,        // Frequent similarity searches
    RelationshipTraversal, // Frequent relationship queries
    Mixed,                 // No clear dominant pattern
}
```

## Monitoring and Profiling

### Performance Monitoring System

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    latency_histogram: LatencyHistogram,
    throughput_counter: Arc<AtomicU64>,
    memory_sampler: MemorySampler,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_documents_processed: u64,
    pub total_processing_time: Duration,
    pub average_processing_time: Duration,
    pub peak_memory_usage: u64,
    pub current_memory_usage: u64,
    pub cache_hit_rate: f32,
    pub model_load_times: HashMap<String, Duration>,
    pub error_rate: f32,
    pub quality_distribution: QualityDistribution,
}

impl PerformanceMonitor {
    pub async fn record_processing_event(
        &self,
        processing_time: Duration,
        memory_used: u64,
        quality_score: f32,
        cache_hit: bool
    ) {
        // Update atomic counters
        self.throughput_counter.fetch_add(1, Ordering::Relaxed);
        
        // Update detailed metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_documents_processed += 1;
        metrics.total_processing_time += processing_time;
        metrics.average_processing_time = 
            metrics.total_processing_time / metrics.total_documents_processed as u32;
        
        if memory_used > metrics.peak_memory_usage {
            metrics.peak_memory_usage = memory_used;
        }
        
        // Update cache hit rate
        let total_requests = metrics.total_documents_processed as f32;
        let cache_hits = if cache_hit { 1.0 } else { 0.0 };
        metrics.cache_hit_rate = (metrics.cache_hit_rate * (total_requests - 1.0) + cache_hits) / total_requests;
        
        // Update quality distribution
        metrics.quality_distribution.add_sample(quality_score);
        
        // Record latency histogram
        self.latency_histogram.record(processing_time).await;
    }
    
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        let latency_percentiles = self.latency_histogram.get_percentiles().await;
        let current_throughput = self.calculate_current_throughput().await;
        
        PerformanceReport {
            overview: PerformanceOverview {
                total_processed: metrics.total_documents_processed,
                average_latency: metrics.average_processing_time,
                current_throughput,
                cache_hit_rate: metrics.cache_hit_rate,
                error_rate: metrics.error_rate,
            },
            latency_analysis: latency_percentiles,
            memory_analysis: MemoryAnalysis {
                current_usage: metrics.current_memory_usage,
                peak_usage: metrics.peak_memory_usage,
                efficiency_score: self.calculate_memory_efficiency(&metrics),
            },
            quality_analysis: metrics.quality_distribution.clone(),
            recommendations: self.generate_recommendations(&metrics).await,
        }
    }
    
    async fn generate_recommendations(&self, metrics: &PerformanceMetrics) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Latency recommendations
        if metrics.average_processing_time > Duration::from_secs(10) {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Latency,
                priority: Priority::High,
                description: "Consider using smaller models or reducing chunk sizes".to_string(),
                expected_improvement: "30-50% latency reduction".to_string(),
            });
        }
        
        // Memory recommendations
        if metrics.peak_memory_usage > 6_000_000_000 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Memory,
                priority: Priority::Medium,
                description: "Consider reducing max_concurrent_models or model sizes".to_string(),
                expected_improvement: "20-40% memory reduction".to_string(),
            });
        }
        
        // Cache recommendations
        if metrics.cache_hit_rate < 0.6 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Caching,
                priority: Priority::Medium,
                description: "Increase cache size or idle timeout for better hit rates".to_string(),
                expected_improvement: "15-25% performance improvement".to_string(),
            });
        }
        
        // Quality recommendations
        if metrics.quality_distribution.average < 0.7 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Quality,
                priority: Priority::High,
                description: "Consider using larger models or adjusting confidence thresholds".to_string(),
                expected_improvement: "10-20% quality improvement".to_string(),
            });
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub overview: PerformanceOverview,
    pub latency_analysis: LatencyPercentiles,
    pub memory_analysis: MemoryAnalysis,
    pub quality_analysis: QualityDistribution,
    pub recommendations: Vec<OptimizationRecommendation>,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub description: String,
    pub expected_improvement: String,
}

#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    Latency,
    Memory,
    Caching,
    Quality,
    Throughput,
}

#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}
```

### Profiling Tools

```rust
pub struct ProcessingProfiler {
    stage_timers: HashMap<String, StageTimer>,
    memory_tracker: MemoryTracker,
    model_performance_tracker: ModelPerformanceTracker,
}

impl ProcessingProfiler {
    pub async fn profile_processing(
        &mut self,
        processor: &IntelligentKnowledgeProcessor,
        content: &str,
        title: &str
    ) -> ProfilingResult {
        let mut profile = ProcessingProfile::new();
        
        // Profile global context analysis
        let context_start = std::time::Instant::now();
        let global_context = processor.context_analyzer
            .analyze_global_context(content, title).await?;
        profile.context_analysis_time = context_start.elapsed();
        
        // Profile semantic chunking
        let chunking_start = std::time::Instant::now();
        let chunks = processor.semantic_chunker
            .create_semantic_chunks(content).await?;
        profile.semantic_chunking_time = chunking_start.elapsed();
        
        // Profile entity extraction
        let entity_start = std::time::Instant::now();
        let mut all_entities = Vec::new();
        for chunk in &chunks {
            let entities = processor.entity_extractor
                .extract_entities_with_context(&chunk.content).await?;
            all_entities.extend(entities);
        }
        profile.entity_extraction_time = entity_start.elapsed();
        profile.entities_extracted = all_entities.len();
        
        // Profile relationship mapping
        let relationship_start = std::time::Instant::now();
        let mut all_relationships = Vec::new();
        for (chunk, entities) in chunks.iter().zip(&chunk_entities) {
            let relationships = processor.relationship_mapper
                .extract_complex_relationships(&chunk.content, entities).await?;
            all_relationships.extend(relationships);
        }
        profile.relationship_mapping_time = relationship_start.elapsed();
        profile.relationships_extracted = all_relationships.len();
        
        // Memory usage analysis
        profile.peak_memory_usage = self.memory_tracker.get_peak_usage();
        profile.memory_efficiency = self.calculate_memory_efficiency(&profile);
        
        // Model performance analysis
        profile.model_performance = self.model_performance_tracker.get_performance_summary();
        
        ProfilingResult {
            profile,
            recommendations: self.analyze_bottlenecks(&profile),
        }
    }
    
    fn analyze_bottlenecks(&self, profile: &ProcessingProfile) -> Vec<BottleneckRecommendation> {
        let mut recommendations = Vec::new();
        let total_time = profile.total_processing_time();
        
        // Identify slowest stages
        if profile.entity_extraction_time > total_time * 0.5 {
            recommendations.push(BottleneckRecommendation {
                stage: "Entity Extraction".to_string(),
                issue: "Consumes >50% of processing time".to_string(),
                suggestion: "Consider using smaller entity extraction model".to_string(),
                potential_speedup: "2-3x faster processing".to_string(),
            });
        }
        
        if profile.semantic_chunking_time > total_time * 0.3 {
            recommendations.push(BottleneckRecommendation {
                stage: "Semantic Chunking".to_string(),
                issue: "Consumes >30% of processing time".to_string(),
                suggestion: "Reduce chunk overlap or use simpler chunking strategy".to_string(),
                potential_speedup: "20-40% faster processing".to_string(),
            });
        }
        
        // Memory efficiency analysis
        if profile.memory_efficiency < 0.5 {
            recommendations.push(BottleneckRecommendation {
                stage: "Memory Usage".to_string(),
                issue: "Low memory efficiency detected".to_string(),
                suggestion: "Enable model unloading or reduce concurrent models".to_string(),
                potential_speedup: "Better resource utilization".to_string(),
            });
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingProfile {
    pub context_analysis_time: Duration,
    pub semantic_chunking_time: Duration,
    pub entity_extraction_time: Duration,
    pub relationship_mapping_time: Duration,
    pub entities_extracted: usize,
    pub relationships_extracted: usize,
    pub peak_memory_usage: u64,
    pub memory_efficiency: f32,
    pub model_performance: ModelPerformanceSummary,
}

impl ProcessingProfile {
    pub fn total_processing_time(&self) -> Duration {
        self.context_analysis_time + 
        self.semantic_chunking_time + 
        self.entity_extraction_time + 
        self.relationship_mapping_time
    }
}
```

## Deployment Scenarios

### Container Deployment (Memory Constrained)

```rust
/// Configuration optimized for containerized deployment with memory limits
pub fn create_container_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 1_500_000_000,   // 1.5GB limit for containers
        max_concurrent_models: 2,          // Limited concurrency
        idle_timeout: Duration::from_secs(60), // Quick eviction
        min_memory_threshold: 200_000_000, // 200MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        max_chunk_size: 1024,
        min_chunk_size: 128,
        chunk_overlap_size: 32,
        min_entity_confidence: 0.5,
        min_relationship_confidence: 0.4,
        preserve_context: true,
        enable_quality_validation: false,
    };
    
    (model_config, processing_config)
}
```

### Edge Device Deployment

```rust
/// Ultra-lightweight configuration for edge devices
pub fn create_edge_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 500_000_000,     // 500MB for edge devices
        max_concurrent_models: 1,          // Single model only
        idle_timeout: Duration::from_secs(30), // Aggressive eviction
        min_memory_threshold: 50_000_000,  // 50MB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        entity_extraction_model: "smollm2_135m".to_string(),
        relationship_extraction_model: "smollm_135m".to_string(),
        semantic_analysis_model: "smollm2_135m".to_string(),
        max_chunk_size: 512,
        min_chunk_size: 64,
        chunk_overlap_size: 16,
        min_entity_confidence: 0.4,
        min_relationship_confidence: 0.3,
        preserve_context: false,
        enable_quality_validation: false,
    };
    
    (model_config, processing_config)
}
```

### High-Performance Server Deployment

```rust
/// Configuration for high-performance server deployment
pub fn create_server_config() -> (ModelResourceConfig, KnowledgeProcessingConfig) {
    let model_config = ModelResourceConfig {
        max_memory_usage: 16_000_000_000,  // 16GB for servers
        max_concurrent_models: 8,          // Multiple specialized models
        idle_timeout: Duration::from_secs(3600), // Keep models loaded
        min_memory_threshold: 2_000_000_000, // 2GB minimum
    };
    
    let processing_config = KnowledgeProcessingConfig {
        entity_extraction_model: "smollm_1_7b".to_string(),
        relationship_extraction_model: "smollm_360m_instruct".to_string(),
        semantic_analysis_model: "smollm2_360m".to_string(),
        max_chunk_size: 4096,
        min_chunk_size: 256,
        chunk_overlap_size: 128,
        min_entity_confidence: 0.8,
        min_relationship_confidence: 0.7,
        preserve_context: true,
        enable_quality_validation: true,
    };
    
    (model_config, processing_config)
}
```

## Troubleshooting Performance Issues

### Common Performance Problems and Solutions

#### Problem: High Memory Usage
```rust
pub async fn diagnose_memory_usage(processor: &IntelligentKnowledgeProcessor) {
    let model_manager = &processor.model_manager;
    let resource_usage = model_manager.get_current_resource_usage().await;
    
    println!("Memory Diagnosis:");
    println!("  Current usage: {:.2} GB", resource_usage.memory_usage as f64 / 1_000_000_000.0);
    println!("  Peak usage: {:.2} GB", resource_usage.peak_memory_usage as f64 / 1_000_000_000.0);
    println!("  Loaded models: {}", resource_usage.loaded_models.len());
    
    for model in &resource_usage.loaded_models {
        println!("    {}: {:.2} MB", model.name, model.memory_footprint as f64 / 1_000_000.0);
    }
    
    if resource_usage.memory_usage > 0.8 * resource_usage.max_memory_usage {
        println!("⚠️  High memory usage detected!");
        println!("Recommendations:");
        println!("  - Reduce max_concurrent_models");
        println!("  - Use smaller models (smollm2_135m instead of smollm_1_7b)");
        println!("  - Decrease idle_timeout for more aggressive eviction");
        println!("  - Process documents in smaller batches");
    }
}
```

#### Problem: Slow Processing Times
```rust
pub async fn diagnose_processing_speed(
    processor: &IntelligentKnowledgeProcessor,
    test_content: &str
) -> ProcessingDiagnosis {
    let start_time = std::time::Instant::now();
    
    // Profile each stage
    let context_start = std::time::Instant::now();
    let global_context = processor.context_analyzer.analyze_global_context(test_content, "Test").await.unwrap();
    let context_time = context_start.elapsed();
    
    let chunking_start = std::time::Instant::now();
    let chunks = processor.semantic_chunker.create_semantic_chunks(test_content).await.unwrap();
    let chunking_time = chunking_start.elapsed();
    
    let entity_start = std::time::Instant::now();
    let entities = processor.entity_extractor.extract_entities_with_context(test_content).await.unwrap();
    let entity_time = entity_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    let diagnosis = ProcessingDiagnosis {
        total_time,
        context_analysis_time: context_time,
        semantic_chunking_time: chunking_time,
        entity_extraction_time: entity_time,
        bottleneck: identify_bottleneck(context_time, chunking_time, entity_time),
    };
    
    print_speed_diagnosis(&diagnosis);
    diagnosis
}

fn identify_bottleneck(context_time: Duration, chunking_time: Duration, entity_time: Duration) -> ProcessingStage {
    if entity_time > context_time && entity_time > chunking_time {
        ProcessingStage::EntityExtraction
    } else if chunking_time > context_time && chunking_time > entity_time {
        ProcessingStage::SemanticChunking
    } else {
        ProcessingStage::ContextAnalysis
    }
}

fn print_speed_diagnosis(diagnosis: &ProcessingDiagnosis) {
    println!("Processing Speed Diagnosis:");
    println!("  Total time: {:?}", diagnosis.total_time);
    println!("  Context analysis: {:?} ({:.1}%)", 
        diagnosis.context_analysis_time,
        diagnosis.context_analysis_time.as_millis() as f64 / diagnosis.total_time.as_millis() as f64 * 100.0
    );
    println!("  Semantic chunking: {:?} ({:.1}%)", 
        diagnosis.semantic_chunking_time,
        diagnosis.semantic_chunking_time.as_millis() as f64 / diagnosis.total_time.as_millis() as f64 * 100.0
    );
    println!("  Entity extraction: {:?} ({:.1}%)", 
        diagnosis.entity_extraction_time,
        diagnosis.entity_extraction_time.as_millis() as f64 / diagnosis.total_time.as_millis() as f64 * 100.0
    );
    println!("  Bottleneck: {:?}", diagnosis.bottleneck);
    
    match diagnosis.bottleneck {
        ProcessingStage::EntityExtraction => {
            println!("Recommendations:");
            println!("  - Use smaller entity extraction model");
            println!("  - Reduce chunk sizes to process less content per extraction");
            println!("  - Lower entity confidence threshold");
        },
        ProcessingStage::SemanticChunking => {
            println!("Recommendations:");
            println!("  - Reduce chunk overlap");
            println!("  - Use simpler chunking strategy");
            println!("  - Increase minimum chunk size");
        },
        ProcessingStage::ContextAnalysis => {
            println!("Recommendations:");
            println!("  - Use smaller semantic analysis model");
            println!("  - Disable global context preservation if not needed");
        }
    }
}
```

This performance tuning guide provides comprehensive strategies for optimizing the Enhanced Knowledge Storage System across different deployment scenarios and performance requirements.