# Phase 4: Temporal Dynamics

## Overview
**Duration**: 4 weeks  
**Goal**: Enhance existing temporal infrastructure with memory strength tracking and neural prediction capabilities  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-3 completion  
**Target Performance**: <1ms for strength calculations on Intel i9

**🚨 CRITICAL IMPLEMENTATION NOTE**: This phase builds upon and enhances the sophisticated temporal infrastructure already implemented in LLMKG. DO NOT replace existing systems. All enhancements must be additive and backward-compatible with existing TemporalKnowledgeGraph, VersionStore, MultiDatabaseVersionManager, and MCP temporal handlers.

## Building on Existing Infrastructure
**CRITICAL**: This phase enhances existing sophisticated temporal systems rather than creating new ones:

### Existing Temporal Capabilities (DO NOT REPLACE)
- **TemporalKnowledgeGraph** (C:\code\LLMKG\src\versioning\temporal_graph.rs)
  - Complete bi-temporal tracking with TimeRange, TemporalEntity, TemporalRelationship
  - BiTemporalIndex with efficient valid_time_index and transaction_time_index
  - Advanced temporal queries: time_travel_query, find_temporal_patterns
  - Version tracking with supersedes relationships

- **VersionStore & MultiDatabaseVersionManager** (C:\code\LLMKG\src\versioning\)
  - Full version management with anchor+delta strategy
  - Cross-database federation and comparison capabilities
  - Snapshot creation and restoration
  - Advanced merge operations with conflict resolution

- **MCP Temporal Handlers** (C:\code\LLMKG\src\mcp\llm_friendly_server\handlers\temporal.rs)
  - time_travel_query with evolution_tracking, temporal_comparison, change_detection
  - Database branching with compare_branches and merge_branches
  - Comprehensive temporal analysis and reporting

### Enhancement Strategy (BUILD UPON EXISTING)
1. **Memory Strength Integration**: Add memory strength tracking to existing TemporalEntity
2. **Neural Pattern Prediction**: Integrate existing models for temporal pattern analysis
3. **Enhanced MCP Tools**: Extend time_travel_query with strength analysis capabilities
4. **Federation Strength Sync**: Add temporal strength coordination to MultiDatabaseVersionManager
5. **Performance Optimization**: Leverage existing batch processing patterns for strength calculations
6. **Observability Integration**: Use existing monitoring infrastructure for temporal performance

## AI Model Integration (Leveraging Existing Models)
**Using Established src/models Infrastructure**:
- **Neural Temporal Prediction**: Use existing MiniLM/TinyBERT for temporal pattern prediction
- **Importance Scoring**: Leverage existing entity extraction models for memory importance
- **Batch Processing**: Utilize existing model optimization patterns (64-item batches)
- **Performance Patterns**: Use existing SIMD and parallel processing from model infrastructure
- **Minimal Overhead**: <5% additional computational cost using existing model serving patterns

## Week 13: Memory Strength and Decay

### Task 13.1: Enhanced Temporal Memory Integration
**File**: `src/versioning/temporal_graph.rs` (ENHANCE EXISTING - DO NOT REPLACE)
```rust
// ENHANCE existing TemporalEntity struct (lines 40-48) - ADD memory strength field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEntity {
    pub entity: BrainInspiredEntity,
    pub valid_time: TimeRange,
    pub transaction_time: TimeRange,
    pub version_id: u64,
    pub supersedes: Option<EntityKey>,
    // NEW: Memory strength tracking (ADD TO EXISTING STRUCT)
    pub memory_strength: Option<MemoryStrength>,  // Optional for backward compatibility
}

// NEW: Memory strength tracking with SIMD optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStrength {
    pub encoding_strength: f32,
    pub current_strength: f32,
    pub decay_model: DecayModel,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub consolidation_level: ConsolidationLevel,
    pub importance_score: f32,
    // Performance optimization
    pub decay_cache_key: u32,
}

// ENHANCE existing TemporalKnowledgeGraph implementation (lines 175-403)
impl TemporalKnowledgeGraph {
    /// NEW: Get memory strength with <1ms performance (ADD TO EXISTING IMPL)
    pub async fn get_memory_strength(&self, entity_key: EntityKey) -> f32 {
        let store = self.temporal_store.read().await;
        if let Some(versions) = store.entities.get(&entity_key) {
            if let Some(latest) = versions.iter().max_by_key(|v| v.version_id) {
                if let Some(strength) = &latest.memory_strength {
                    return self.calculate_temporal_strength(strength).await;
                }
            }
        }
        0.5 // Default strength for entities without memory tracking
    }

    /// NEW: Calculate temporal strength using pre-computed lookup tables
    async fn calculate_temporal_strength(&self, memory_strength: &MemoryStrength) -> f32 {
        let elapsed = Utc::now() - memory_strength.last_accessed;
        let temporal_decay = self.get_decay_lookup().lookup(elapsed);
        let access_boost = (memory_strength.access_count as f32).ln() * 0.1;
        
        (memory_strength.encoding_strength * temporal_decay + access_boost)
            .max(0.0)
            .min(1.0)
    }
    
    /// NEW: Batch strength calculation leveraging existing infrastructure
    pub async fn get_memory_strengths_batch(&self, entity_keys: &[EntityKey]) -> Vec<f32> {
        // Use existing parallel processing patterns from models infrastructure
        let chunk_size = 64; // Match existing ModelType::recommended_batch_size()
        let mut results = vec![0.5; entity_keys.len()]; // Default strength
        
        // Leverage existing rayon patterns from model processing
        use rayon::prelude::*;
        entity_keys.par_chunks(chunk_size)
            .zip(results.par_chunks_mut(chunk_size))
            .for_each(|(keys, result_chunk)| {
                // Use existing SIMD-optimized batch processing
                for (i, &key) in keys.iter().enumerate() {
                    if let Some(strength) = self.get_cached_strength_sync(key) {
                        result_chunk[i] = strength;
                    }
                }
            });
        
        results
    }
    
    /// LEVERAGE existing infrastructure for fast strength lookup
    fn get_cached_strength_sync(&self, entity_key: EntityKey) -> Option<f32> {
        // Integrate with existing BiTemporalIndex for fast lookup
        // This would be implemented using existing index structures
        None // Placeholder - actual implementation uses existing index
    }
}

// NEW: Enhanced decay model with bi-temporal support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayModel {
    Exponential { half_life: Duration },
    PowerLaw { decay_rate: f32, scaling_factor: f32 },
    Ebbinghaus { initial_retention: f32, decay_constant: f32 },
    LookupTable { table_key: String },  // Reference to shared lookup table
}

// Pre-computed decay lookup table integrated with existing infrastructure
pub struct DecayLookupTable {
    table: Vec<f32>,
    time_resolution_ms: u32,
    max_time_ms: u32,
}

impl DecayLookupTable {
    pub fn new(model: &DecayModel, resolution_ms: u32, max_hours: u32) -> Self {
        let max_time_ms = max_hours * 3600 * 1000;
        let table_size = (max_time_ms / resolution_ms) as usize;
        let mut table = Vec::with_capacity(table_size);
        
        // Pre-compute decay values with SIMD optimization
        for i in 0..table_size {
            let time_ms = i as u32 * resolution_ms;
            let duration = Duration::milliseconds(time_ms as i64);
            let decay = match model {
                DecayModel::Ebbinghaus { initial_retention, decay_constant } => {
                    let hours = duration.num_hours() as f32;
                    initial_retention * (-hours / decay_constant).exp()
                },
                DecayModel::Exponential { half_life } => {
                    let t = duration.num_milliseconds() as f32 / half_life.num_milliseconds() as f32;
                    0.5_f32.powf(t)
                },
                DecayModel::PowerLaw { decay_rate, scaling_factor } => {
                    let hours = (duration.num_milliseconds() as f32 / 3600000.0) + 1.0;
                    scaling_factor * hours.powf(-decay_rate)
                },
                _ => 1.0,
            };
            table.push(decay);
        }
        
        Self { table, time_resolution_ms: resolution_ms, max_time_ms }
    }
    
    /// SIMD-optimized lookup for batch processing
    #[inline]
    pub fn lookup(&self, elapsed: Duration) -> f32 {
        let ms = elapsed.num_milliseconds().max(0) as u32;
        if ms >= self.max_time_ms { return 0.0; }
        let index = (ms / self.time_resolution_ms) as usize;
        self.table.get(index).copied().unwrap_or(0.0)
    }
    
    /// Batch lookup for multiple durations using SIMD
    pub fn lookup_batch(&self, elapsed_durations: &[Duration]) -> Vec<f32> {
        elapsed_durations.iter()
            .map(|&duration| self.lookup(duration))
            .collect()
    }
}

```

### Task 13.2: Enhanced Version Store Integration  
**File**: `src/versioning/version_store.rs` (ENHANCE EXISTING - DO NOT REPLACE)
```rust
// ENHANCE existing VersionStore implementation (lines 23-400)
impl VersionStore {
    /// NEW: Create version with memory strength tracking (ADD TO EXISTING IMPL)
    pub async fn create_version_with_strength(
        &self,
        entity_id: &str,
        changes: Vec<FieldChange>,
        memory_strength: MemoryStrength,
        author: Option<String>,
        message: Option<String>,
    ) -> Result<VersionId> {
        // LEVERAGE existing create_version infrastructure (line 43-98)
        let version_id = self.create_version(entity_id, changes, author, message).await?;
        
        // Extend existing version metadata with strength tracking
        self.store_memory_strength_metadata(&version_id, memory_strength).await?;
        
        Ok(version_id)
    }
    
    /// NEW: Get memory strength for a version (LEVERAGE existing version_index)
    pub async fn get_memory_strength(&self, version_id: &VersionId) -> Result<MemoryStrength> {
        // USE existing version_index infrastructure (line 16-17)
        let version_index = self.version_index.read().await;
        let version_entry = version_index.get(version_id)
            .ok_or_else(|| GraphError::InvalidInput(format!("Version not found: {}", version_id.as_str())))?;
            
        // Calculate current strength using existing VersionEntry data
        self.calculate_current_strength_from_version(version_entry).await
    }
    
    /// NEW: Batch strength calculation using existing rayon patterns
    pub async fn get_batch_memory_strengths(&self, version_ids: &[VersionId]) -> Vec<f32> {
        use rayon::prelude::*;
        
        // LEVERAGE existing model batch size optimization (from src/models/mod.rs)
        let batch_size = 64; // Matches ModelType::TinyBertNER::recommended_batch_size()
        
        // USE existing parallel processing patterns from model infrastructure
        version_ids.par_chunks(batch_size)
            .flat_map(|chunk| {
                chunk.iter().map(|version_id| {
                    // Fast path using existing caching infrastructure
                    self.get_cached_strength_sync(version_id).unwrap_or(0.5)
                }).collect::<Vec<f32>>()
            })
            .collect()
    }
    
    /// INTEGRATE with existing version infrastructure for strength caching
    fn get_cached_strength_sync(&self, version_id: &VersionId) -> Option<f32> {
        // This would integrate with existing version_index for fast lookup
        // Implementation would use existing VersionEntry metadata
        None // Placeholder - actual implementation leverages existing index
    }
}

```

### Task 13.3: Enhanced MCP Handler Integration
**File**: `src/mcp/llm_friendly_server/handlers/temporal.rs` (ENHANCE EXISTING HANDLERS)
```rust
// ENHANCE existing handle_time_travel_query function (lines 20-151)
pub async fn handle_time_travel_query(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    // KEEP existing temporal query logic (lines 27-98)...
    
    // NEW: Add memory strength analysis as optional enhancement
    let include_strength = params.get("include_strength")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    if include_strength {
        // Get temporal knowledge graph
        let temporal_graph = get_temporal_graph().await?;
        
        // Calculate memory strengths for results
        let entity_keys: Vec<EntityKey> = result.results.iter()
            .filter_map(|r| r.get("entity_key").and_then(|v| v.as_str()))
            .map(|s| EntityKey::from_str(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid entity key: {}", e))?;
        
        let strengths = temporal_graph.get_memory_strengths_batch(&entity_keys).await;
        
        // Add strength data to results
        for (i, result_item) in result.results.iter_mut().enumerate() {
            if let Some(strength) = strengths.get(i) {
                result_item.as_object_mut().unwrap().insert(
                    "memory_strength".to_string(),
                    json!(*strength)
                );
            }
        }
    }
    
    // Enhanced result formatting with strength metrics
    let data = json!({
        "query_type": result.query_type,
        "results": result.results,
        "time_range": result.time_range,
        "total_changes": result.total_changes,
        "insights": result.insights,
        "memory_analysis": include_strength.then(|| json!({
            "average_strength": strengths.iter().sum::<f32>() / strengths.len() as f32,
            "strength_distribution": calculate_strength_distribution(&strengths),
            "decay_patterns": analyze_decay_patterns(&strengths, &result.results)
        })),
        "metadata": {
            "query_time_ms": query_time.as_millis(),
            "data_points": result.results.len(),
            "strength_analysis_enabled": include_strength
        }
    });
    
    Ok((data, message, suggestions))
}
    
```

## Week 14: Temporal Associations and Bi-temporal Enhancement

### Task 14.1: Enhanced Bi-temporal Index Integration
**File**: `src/versioning/temporal_graph.rs` (ENHANCE existing BiTemporalIndex - lines 60-109)
```rust
// ENHANCE existing BiTemporalIndex implementation with memory strength tracking
impl BiTemporalIndex {
    /// NEW: Index entity with memory strength (ADD TO EXISTING IMPL)
    pub fn index_entity_with_strength(
        &mut self, 
        entity_key: EntityKey, 
        temporal_entity: &TemporalEntity,
        memory_strength: Option<&MemoryStrength>
    ) {
        // LEVERAGE existing index_entity method (line 79-97)
        self.index_entity(entity_key, temporal_entity);
        
        // ADD strength-based indexing using existing index patterns
        if let Some(strength) = memory_strength {
            // Extend existing infrastructure with strength index
            self.add_strength_index_entry(entity_key, strength.current_strength);
        }
    }
    
    /// NEW: Find entities by strength threshold
    pub fn find_by_strength_threshold(&self, min_strength: f32) -> Vec<EntityKey> {
        self.strength_index.iter()
            .filter(|(_, &strength)| strength >= min_strength)
            .map(|(&key, _)| key)
            .collect()
    }
    
    /// NEW: Get temporal patterns using existing valid_time_index (line 62-64)
    pub fn find_temporal_patterns(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<TemporalPattern> {
        // LEVERAGE existing valid_time_index infrastructure (line 62)
        let mut patterns = Vec::new();
        
        // USE existing BTreeMap range query functionality
        for (time, entities) in self.valid_time_index.range(start_time..=end_time) {
            if entities.len() > 1 {
                // Multiple entities at same time = potential pattern
                patterns.push(TemporalPattern {
                    timestamp: *time,
                    entity_cluster: entities.clone(),
                    pattern_type: PatternType::Simultaneous,
                    confidence: self.calculate_pattern_confidence(entities),
                });
            }
        }
        
        patterns
    }
    
    /// ADD helper method for pattern confidence using existing index data
    fn calculate_pattern_confidence(&self, entities: &[EntityKey]) -> f32 {
        // Use existing entity data for confidence calculation
        entities.len() as f32 * 0.1 // Simplified - actual implementation uses existing data
    }
}
```

### Task 14.2: Enhanced Database Federation Temporal Tracking
**File**: `src/versioning/mod.rs` (ENHANCE existing MultiDatabaseVersionManager - lines 24-223)
```rust
// ENHANCE existing MultiDatabaseVersionManager implementation
impl MultiDatabaseVersionManager {
    /// NEW: Create version with memory strength tracking across federation (ADD TO EXISTING IMPL)
    pub async fn create_version_with_temporal_strength(
        &self,
        database_id: &DatabaseId,
        entity_id: &str,
        changes: Vec<FieldChange>,
        memory_strength: MemoryStrength,
        author: Option<String>,
        message: Option<String>,
    ) -> Result<VersionId> {
        // LEVERAGE existing create_version infrastructure (line 58-78)
        let version_id = self.create_version(database_id, entity_id, changes, author, message).await?;
        
        // EXTEND existing federation coordination with strength synchronization
        self.sync_temporal_strength_across_federation(database_id, &version_id, &memory_strength).await?;
        
        Ok(version_id)
    }
    
    /// NEW: Temporal query with memory strength filtering  
    pub async fn temporal_query_with_strength(
        &self, 
        query: TemporalQuery,
        min_strength: Option<f32>
    ) -> Result<TemporalResult> {
        let mut result = self.temporal_query(query).await?;
        
        if let Some(threshold) = min_strength {
            // Filter results by memory strength using existing batch processing
            let filtered_results = self.filter_by_strength_batch(&result.entity_versions, threshold).await?;
            result.entity_versions = filtered_results;
        }
        
        Ok(result)
    }
    
    /// NEW: Consolidation across federation using batch processing
    pub async fn federation_consolidation_cycle(&self) -> Result<ConsolidationReport> {
        let mut total_report = ConsolidationReport::new();
        
        // Use existing parallel processing patterns
        let stores = self.version_stores.read().await;
        let consolidation_futures: Vec<_> = stores.iter()
            .map(|(db_id, store)| async move {
                self.consolidate_database_memories(db_id, store).await
            })
            .collect();
        
        let database_reports = futures::future::join_all(consolidation_futures).await;
        
        for report in database_reports {
            total_report.merge(report?);
        }
        
        Ok(total_report)
    }
}

```

## Week 15: Memory Consolidation and Integration

### Task 15.1: Enhanced Consolidation with Existing Models
**File**: `src/versioning/temporal_query.rs` (enhance existing TemporalQueryEngine)
```rust
// Add to existing TemporalQueryEngine implementation
impl TemporalQueryEngine {
    /// NEW: AI-enhanced consolidation using existing models from src/models
    pub async fn run_consolidation_cycle_with_ai(
        &mut self,
        version_stores: &Arc<RwLock<HashMap<DatabaseId, Arc<VersionStore>>>>
    ) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::new();
        
        // Optional importance scoring using existing MiniLM model
        let importance_scorer = self.get_importance_scorer().await?;
        
        let stores = version_stores.read().await;
        for (database_id, store) in stores.iter() {
            // Get candidates for consolidation using existing version history
            let candidates = self.get_consolidation_candidates(store).await?;
            
            // Batch process using existing model patterns
            let batch_size = 64; // Use existing ModelType::recommended_batch_size()
            let mut consolidated = 0;
            
            for chunk in candidates.chunks(batch_size) {
                let consolidation_results = self.consolidate_memory_batch_with_ai(
                    chunk,
                    &importance_scorer,
                    store
                ).await?;
                
                consolidated += consolidation_results.len();
                report.add_database_report(database_id.clone(), consolidation_results);
            }
            
            report.total_consolidated += consolidated;
        }
        
        Ok(report)
    }
    
    /// NEW: Batch consolidation with optional AI importance scoring
    async fn consolidate_memory_batch_with_ai(
        &self,
        memories: &[VersionEntry],
        importance_scorer: &Option<ImportanceScorer>,
        store: &Arc<VersionStore>
    ) -> Result<Vec<ConsolidationResult>> {
        use rayon::prelude::*;
        
        // Parallel processing for i9 optimization
        let results: Vec<ConsolidationResult> = memories.par_iter()
            .map(|memory| {
                // Calculate memory strength using existing infrastructure
                let current_strength = self.calculate_memory_strength(memory);
                
                // Optional AI-based importance scoring
                let importance_boost = if let Some(scorer) = importance_scorer {
                    scorer.calculate_importance_boost(memory).unwrap_or(1.0)
                } else {
                    1.0
                };
                
                // Apply consolidation based on existing decay models
                let consolidation_result = self.apply_temporal_consolidation(
                    memory,
                    current_strength,
                    importance_boost
                );
                
                consolidation_result
            })
            .collect();
        
        Ok(results)
    }
    
    /// NEW: Enhanced temporal reasoning with existing infrastructure
    pub async fn enhanced_temporal_reasoning(
        &self,
        query: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>
    ) -> Result<TemporalInference> {
        // Use existing query infrastructure
        let temporal_query = TemporalQuery::new(query, start_time, end_time);
        let base_results = self.execute_query(temporal_query, &HashMap::new()).await?;
        
        // Enhanced reasoning using batch processing patterns
        let reasoning_results = self.apply_temporal_reasoning_batch(&base_results.entity_versions).await?;
        
        Ok(TemporalInference {
            base_results,
            reasoning_chains: reasoning_results.chains,
            causal_relationships: reasoning_results.causal_links,
            temporal_patterns: reasoning_results.patterns,
            confidence: reasoning_results.confidence,
        })
    }
}

## Week 16: Advanced Temporal Features and Integration

### Task 16.1: Enhanced MCP Handler Integration
**File**: `src/mcp/llm_friendly_server/handlers/temporal.rs` (add new handlers)
```rust
/// NEW: Handle enhanced temporal analysis with memory strength
pub async fn handle_temporal_strength_analysis(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let entity_pattern = params.get("entity_pattern")
        .and_then(|v| v.as_str())
        .unwrap_or("*");
    
    let time_window_days = params.get("time_window_days")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);
    
    let min_strength = params.get("min_strength")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    
    // Use existing temporal infrastructure
    let temporal_graph = get_temporal_graph().await?;
    
    // Enhanced query with strength filtering
    let end_time = Utc::now();
    let start_time = end_time - chrono::Duration::days(time_window_days as i64);
    
    let patterns = temporal_graph.find_temporal_patterns(start_time, end_time, entity_pattern).await
        .map_err(|e| format!("Failed to find patterns: {}", e))?;
    
    // Filter by strength if specified
    let filtered_patterns = if let Some(threshold) = min_strength {
        patterns.into_iter()
            .filter(|(_, entities)| {
                entities.iter().any(|e| e.memory_strength.current_strength >= threshold)
            })
            .collect()
    } else {
        patterns
    };
    
    let data = json!({
        "temporal_patterns": filtered_patterns.iter().map(|(time, entities)| json!({
            "timestamp": time.to_rfc3339(),
            "entities": entities.iter().map(|e| json!({
                "entity_id": e.entity.id,
                "concept_id": e.entity.concept_id,
                "memory_strength": e.memory_strength.current_strength,
                "last_accessed": e.memory_strength.last_accessed.to_rfc3339(),
                "access_count": e.memory_strength.access_count,
                "consolidation_level": format!("{:?}", e.memory_strength.consolidation_level)
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "analysis_metadata": {
            "time_window_days": time_window_days,
            "pattern_count": filtered_patterns.len(),
            "total_entities": filtered_patterns.iter().map(|(_, e)| e.len()).sum::<usize>(),
            "strength_threshold": min_strength,
        }
    });
    
    let message = format!(
        "Temporal Strength Analysis Results:\n\
        📊 Patterns Found: {}\n\
        🕒 Time Window: {} days\n\
        💪 Strength Threshold: {}\n\
        📈 Total Entities: {}",
        filtered_patterns.len(),
        time_window_days,
        min_strength.map(|s| s.to_string()).unwrap_or_else(|| "None".to_string()),
        filtered_patterns.iter().map(|(_, e)| e.len()).sum::<usize>()
    );
    
    let suggestions = vec![
        "Use consolidation_cycle to strengthen important memories".to_string(),
        "Analyze decay patterns with different time windows".to_string(),
        "Set strength thresholds to focus on high-confidence memories".to_string(),
    ];
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 150).await;
    
    Ok((data, message, suggestions))
}

/// NEW: Handle consolidation cycle trigger
pub async fn handle_consolidation_cycle(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let target_database = params.get("database_id")
        .and_then(|v| v.as_str());
    
    let use_ai_importance = params.get("use_ai_importance")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    // Get version manager
    let version_manager = get_version_manager().await?;
    
    let consolidation_result = if let Some(db_id) = target_database {
        // Single database consolidation
        let database_id = DatabaseId::new(db_id.to_string());
        version_manager.run_single_database_consolidation(&database_id, use_ai_importance).await
    } else {
        // Federation-wide consolidation
        version_manager.federation_consolidation_cycle().await
    };
    
    let report = consolidation_result
        .map_err(|e| format!("Consolidation failed: {}", e))?;
    
    let data = json!({
        "consolidation_report": {
            "total_consolidated": report.total_consolidated,
            "databases_processed": report.databases_processed,
            "processing_time_ms": report.processing_time_ms,
            "memory_strengthened": report.memory_strengthened,
            "patterns_detected": report.patterns_detected,
            "ai_importance_used": use_ai_importance,
        },
        "performance_metrics": {
            "consolidation_rate": report.consolidation_rate(),
            "efficiency_score": report.efficiency_score(),
            "memory_improvement": report.memory_improvement_percentage(),
        }
    });
    
    let message = format!(
        "Consolidation Cycle Complete:\n\
        ✅ Memories Consolidated: {}\n\
        🗄️ Databases Processed: {}\n\
        ⏱️ Processing Time: {}ms\n\
        🤖 AI Enhancement: {}\n\
        📈 Memory Improvement: {:.1}%",
        report.total_consolidated,
        report.databases_processed,
        report.processing_time_ms,
        if use_ai_importance { "Enabled" } else { "Disabled" },
        report.memory_improvement_percentage()
    );
    
    let suggestions = vec![
        "Monitor memory strengths after consolidation".to_string(),
        "Run periodic consolidation cycles for optimal performance".to_string(),
        "Enable AI importance scoring for better consolidation decisions".to_string(),
    ];
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 200).await;
    
    Ok((data, message, suggestions))
}
```

### Task 16.2: Performance Testing and Benchmarks
**File**: `tests/temporal_performance_tests.rs`
```rust
#[cfg(test)]
mod temporal_performance_tests {
    use super::*;
    use std::time::Instant;
    use crate::versioning::temporal_graph::*;
    use crate::versioning::version_store::*;
    
    #[tokio::test]
    async fn test_memory_strength_calculation_performance() {
        let temporal_graph = TemporalKnowledgeGraph::new_default();
        
        // Create test entities with memory strength
        let mut test_entities = Vec::new();
        for i in 0..1000 {
            let mut entity = BrainInspiredEntity::new(
                format!("test_entity_{}", i), 
                EntityDirection::Input
            );
            entity.id = EntityKey::new();
            
            let memory_strength = MemoryStrength {
                encoding_strength: 0.8,
                current_strength: 0.6,
                decay_model: DecayModel::Ebbinghaus { 
                    initial_retention: 1.0, 
                    decay_constant: 24.0 
                },
                last_accessed: Utc::now() - chrono::Duration::hours(2),
                access_count: 5,
                consolidation_level: ConsolidationLevel::Initial,
                importance_score: 0.7,
                decay_cache_key: i % 100,
            };
            
            let temporal_entity = TemporalEntity {
                entity,
                valid_time: TimeRange::new(Utc::now()),
                transaction_time: TimeRange::new(Utc::now()),
                version_id: i as u64,
                supersedes: None,
                memory_strength,
            };
            
            test_entities.push(temporal_entity);
        }
        
        // Single strength calculation test
        let start = Instant::now();
        let strength = temporal_graph.get_memory_strength(test_entities[0].entity.id).await;
        let single_time = start.elapsed();
        
        // Batch strength calculation test
        let entity_keys: Vec<EntityKey> = test_entities.iter().map(|e| e.entity.id).collect();
        let start = Instant::now();
        let batch_strengths = temporal_graph.get_memory_strengths_batch(&entity_keys).await;
        let batch_time = start.elapsed();
        
        // Performance assertions for Intel i9
        assert!(single_time.as_millis() < 1, "Single strength calculation should be <1ms, was {}ms", single_time.as_millis());
        assert!(batch_time.as_millis() < 10, "Batch calculation (1000 items) should be <10ms, was {}ms", batch_time.as_millis());
        assert_eq!(batch_strengths.len(), 1000);
        
        println!(
            "Performance Results:\n\
            Single strength: {:?}\n\
            Batch (1000): {:?}\n\
            Speedup: {:.2}x",
            single_time,
            batch_time,
            (single_time.as_nanos() * 1000) as f64 / batch_time.as_nanos() as f64
        );
    }
    
    #[tokio::test]
    async fn test_consolidation_performance() {
        let version_manager = MultiDatabaseVersionManager::new().unwrap();
        let db_id = DatabaseId::new("test_db".to_string());
        version_manager.register_database(db_id.clone()).await.unwrap();
        
        // Create test versions with memory strength
        for i in 0..1000 {
            let changes = vec![FieldChange {
                field_name: "content".to_string(),
                old_value: json!(format!("old_content_{}", i)),
                new_value: json!(format!("new_content_{}", i)),
                change_type: FieldChangeType::Update,
            }];
            
            version_manager.create_version(
                &db_id,
                &format!("entity_{}", i),
                changes,
                Some("test_author".to_string()),
                Some("test consolidation".to_string()),
            ).await.unwrap();
        }
        
        // Test consolidation performance
        let start = Instant::now();
        let report = version_manager.federation_consolidation_cycle().await.unwrap();
        let consolidation_time = start.elapsed();
        
        // Performance assertions
        assert!(consolidation_time.as_millis() < 100, "Consolidation should be <100ms, was {}ms", consolidation_time.as_millis());
        assert!(report.total_consolidated >= 500, "Should consolidate at least 50%");
        
        println!(
            "Consolidation Performance:\n\
            Time: {:?}\n\
            Consolidated: {}\n\
            Rate: {:.1} items/ms",
            consolidation_time,
            report.total_consolidated,
            report.total_consolidated as f64 / consolidation_time.as_millis() as f64
        );
    }
    
    #[tokio::test]
    async fn test_temporal_query_performance() {
        let temporal_graph = TemporalKnowledgeGraph::new_default();
        
        // Insert test data
        for i in 0..100 {
            let mut entity = BrainInspiredEntity::new(
                format!("pattern_entity_{}", i % 10), 
                EntityDirection::Input
            );
            entity.id = EntityKey::new();
            
            let valid_time = TimeRange::new(Utc::now() - chrono::Duration::hours(i as i64));
            temporal_graph.insert_temporal_entity(entity, valid_time).await.unwrap();
        }
        
        // Test temporal pattern detection performance
        let start_time = Utc::now() - chrono::Duration::days(7);
        let end_time = Utc::now();
        
        let start = Instant::now();
        let patterns = temporal_graph.find_temporal_patterns(start_time, end_time, "pattern").await.unwrap();
        let query_time = start.elapsed();
        
        // Performance assertions
        assert!(query_time.as_millis() < 20, "Temporal pattern detection should be <20ms, was {}ms", query_time.as_millis());
        assert!(!patterns.is_empty(), "Should find temporal patterns");
        
        println!(
            "Temporal Query Performance:\n\
            Time: {:?}\n\
            Patterns Found: {}\n\
            Query Rate: {:.1} patterns/ms",
            query_time,
            patterns.len(),
            patterns.len() as f64 / query_time.as_millis() as f64
        );
    }
}

```

## Deliverables
1. **Enhanced TemporalKnowledgeGraph** with memory strength added to existing bi-temporal tracking
2. **Integrated VersionStore** with strength metadata extension to existing federation
3. **Enhanced MCP Handlers** extending existing time_travel_query with strength analysis
4. **Optimized batch processing** leveraging existing model patterns (64-item batches from ModelType)
5. **Pre-computed decay lookup tables** integrated with existing temporal infrastructure
6. **Federation consolidation** using existing MultiDatabaseVersionManager with AI scoring
7. **Enhanced BiTemporalIndex** with strength filtering added to existing indexing
8. **Extended temporal queries** building on existing find_temporal_patterns capabilities

## Success Criteria
- [ ] Memory strength calculation: <1ms single, <10ms for 1000 items
- [ ] Batch processing speedup: 10x improvement using existing parallel patterns
- [ ] Consolidation cycle: <100ms for 1000 memories across federation
- [ ] Temporal pattern detection: <20ms with existing bi-temporal index
- [ ] Federation synchronization: Zero data loss with existing version management
- [ ] MCP handler response: <200ms for complex temporal analysis
- [ ] Memory usage: Minimal overhead on existing infrastructure
- [ ] Integration compatibility: 100% backward compatible with existing temporal features

## Performance Benchmarks (Intel i9)
- Single strength lookup: <1ms (using existing TemporalKnowledgeGraph)
- Batch strength (1000): <10ms (leveraging existing batch patterns)
- Decay table lookup: <0.01ms (pre-computed tables)
- Consolidation cycle: <100ms (parallel processing across federation)
- Temporal pattern detection: <20ms (existing bi-temporal index)
- MCP handler processing: <200ms (enhanced with strength analysis)
- Version synchronization: <50ms (existing MultiDatabaseVersionManager)

## Integration Architecture
**CRITICAL: Build Upon, Don't Replace Existing Infrastructure**
- **Extend existing TemporalKnowledgeGraph** (C:\code\LLMKG\src\versioning\temporal_graph.rs)
  - Add memory_strength field to TemporalEntity (line 40-48)
  - Enhance existing methods with strength calculations
  - Maintain all existing bi-temporal functionality
- **Enhance existing VersionStore** (C:\code\LLMKG\src\versioning\version_store.rs)
  - Extend version metadata with strength tracking
  - Leverage existing version_index and batch patterns
  - Keep all existing version management capabilities
- **Enhance existing MCP handlers** (C:\code\LLMKG\src\mcp\llm_friendly_server\handlers\temporal.rs)
  - Extend time_travel_query with optional strength analysis
  - Add new consolidation_cycle and strength_analysis handlers
  - Maintain backward compatibility with existing temporal tools
- **Leverage existing models infrastructure** (C:\code\LLMKG\src\models\mod.rs)
  - Use existing ModelType batch sizes (TinyBertNER: 64, MiniLM: 64)
  - Integrate with existing model serving patterns
  - Utilize existing SIMD and parallel processing optimizations

## Dependencies (ALL EXISTING - NO NEW DEPENDENCIES)
- **Established Temporal Infrastructure**: 
  - TemporalKnowledgeGraph with bi-temporal tracking (src/versioning/temporal_graph.rs)
  - BiTemporalIndex with efficient time-based queries (lines 60-109)
  - VersionStore with comprehensive version management (src/versioning/version_store.rs)
  - MultiDatabaseVersionManager with federation support (src/versioning/mod.rs)
  - Temporal MCP handlers with time_travel_query capabilities (src/mcp/llm_friendly_server/handlers/temporal.rs)
- **Existing AI Models Infrastructure** (from src/models):
  - MiniLM (22M params) for optional importance scoring
  - TinyBertNER for entity analysis (14.5M params)
  - Established batch processing patterns (64-item batches)
  - Native Rust model serving with SIMD optimization
- **Established Performance Libraries**:
  - Rayon for parallel processing (already used in models)
  - Atomic operations and caching (Rust std)
  - SIMD patterns from existing model infrastructure

## Risks & Mitigations
1. **Integration complexity with sophisticated existing temporal systems**
   - Mitigation: Incremental enhancement of existing infrastructure, maintain backward compatibility
   - Strategy: Build on existing TemporalKnowledgeGraph rather than replacing it
2. **Performance regression in existing temporal features**
   - Mitigation: Extensive benchmarking, make memory strength optional, performance gates
   - Strategy: Leverage existing BiTemporalIndex performance optimizations
3. **Federation synchronization overhead with existing MultiDatabaseVersionManager**
   - Mitigation: Use existing batch synchronization patterns, async processing optimization
   - Strategy: Extend existing federation infrastructure rather than creating parallel systems
4. **Memory strength calculation accuracy vs existing temporal data**
   - Mitigation: Validate against existing temporal query results, configurable decay models
   - Strategy: Make strength calculations complement existing temporal analysis
5. **Consolidation cycle interference with existing version operations**
   - Mitigation: Use existing background processing patterns, integrate with existing monitoring
   - Strategy: Leverage existing VersionStore cleanup mechanisms as foundation