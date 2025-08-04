# Micro Task 21: Pathway Memory

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: 20_pathway_reinforcement.md completed  
**Skills Required**: Memory systems, persistent storage, retrieval mechanisms

## Objective

Implement pathway memory systems for storing, indexing, and recalling successful activation pathways to enable pattern recognition, query optimization, and memory-guided activation spreading.

## Context

Biological memory systems store and recall patterns of neural activation. This task implements pathway memory that captures successful query resolution patterns, enabling the system to recognize similar queries and reuse proven activation strategies.

## Specifications

### Core Memory Components

1. **PathwayMemory struct**
   - Persistent pathway storage
   - Pattern-based indexing
   - Similarity matching algorithms
   - Memory consolidation mechanisms

2. **MemoryPattern struct**
   - Abstract representation of pathway structure
   - Key features for matching
   - Usage frequency tracking
   - Success rate metrics

3. **MemoryIndex struct**
   - Multi-dimensional indexing
   - Query pattern lookup
   - Fuzzy matching capabilities
   - Performance optimization

4. **MemoryConsolidation struct**
   - Memory strengthening over time
   - Interference resolution
   - Schema formation
   - Memory integration

### Performance Requirements

- Fast pathway lookup (< 10ms for 1M pathways)
- Efficient pattern matching and similarity scoring
- Persistent storage with durability guarantees
- Memory-efficient pattern representation
- Support for incremental learning

## Implementation Guide

### Step 1: Core Memory Types

```rust
// File: src/cognitive/learning/pathway_memory.rs

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};
use crate::core::types::{NodeId, EntityId};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwayId, PathwaySegment};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    pub pattern_id: PatternId,
    pub pathway_signature: Vec<NodeId>,  // Simplified pathway representation
    pub query_features: QueryFeatures,   // Abstract query characteristics
    pub success_metrics: SuccessMetrics,
    pub usage_statistics: UsageStatistics,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub consolidation_level: f32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub intent_type: String,
    pub entity_types: Vec<String>,
    pub complexity_score: f32,
    pub context_keywords: Vec<String>,
    pub semantic_embedding: Vec<f32>, // Simplified embedding
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetrics {
    pub average_efficiency: f32,
    pub average_activation_strength: f32,
    pub convergence_rate: f32,
    pub user_satisfaction: f32,
    pub total_successes: u32,
    pub total_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub access_count: u64,
    pub recent_access_times: VecDeque<SystemTime>,
    pub access_frequency: f32, // Accesses per day
    pub recency_weight: f32,
    pub importance_score: f32,
}

#[derive(Debug)]
pub struct PathwayMemory {
    patterns: HashMap<PatternId, MemoryPattern>,
    query_index: QueryIndex,
    pattern_index: PatternIndex,
    consolidation_scheduler: ConsolidationScheduler,
    next_pattern_id: u64,
    memory_capacity: usize,
    consolidation_threshold: f32,
    forgetting_curve_rate: f32,
}

#[derive(Debug)]
pub struct QueryIndex {
    intent_index: HashMap<String, Vec<PatternId>>,
    entity_index: HashMap<String, Vec<PatternId>>,
    complexity_index: BTreeMap<u32, Vec<PatternId>>, // Binned complexity scores
    keyword_index: HashMap<String, Vec<PatternId>>,
}

#[derive(Debug)]
pub struct PatternIndex {
    signature_index: HashMap<Vec<NodeId>, Vec<PatternId>>,
    length_index: BTreeMap<usize, Vec<PatternId>>,
    similarity_clusters: Vec<Vec<PatternId>>,
    frequent_patterns: Vec<PatternId>,
}

#[derive(Debug)]
pub struct ConsolidationScheduler {
    pending_consolidations: VecDeque<ConsolidationTask>,
    last_consolidation: SystemTime,
    consolidation_interval: Duration,
}

#[derive(Debug, Clone)]
pub struct ConsolidationTask {
    pattern_id: PatternId,
    consolidation_type: ConsolidationType,
    scheduled_time: SystemTime,
    priority: f32,
}

#[derive(Debug, Clone)]
pub enum ConsolidationType {
    Strengthen,      // Increase memory strength
    Integrate,       // Merge with similar patterns
    Generalize,      // Extract common features
    Specialize,      // Create more specific patterns
}
```

### Step 2: Pathway Memory Implementation

```rust
impl PathwayMemory {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            query_index: QueryIndex::new(),
            pattern_index: PatternIndex::new(),
            consolidation_scheduler: ConsolidationScheduler::new(),
            next_pattern_id: 1,
            memory_capacity: 100_000,
            consolidation_threshold: 0.7,
            forgetting_curve_rate: 0.1,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            memory_capacity: capacity,
            ..Self::new()
        }
    }
    
    pub fn store_pathway(
        &mut self, 
        pathway: &ActivationPathway,
        query_features: QueryFeatures,
        success_metrics: SuccessMetrics,
    ) -> Result<PatternId, MemoryError> {
        // Check capacity
        if self.patterns.len() >= self.memory_capacity {
            self.perform_memory_cleanup()?;
        }
        
        // Create memory pattern
        let pattern_id = PatternId(self.next_pattern_id);
        self.next_pattern_id += 1;
        
        let pathway_signature = self.extract_pathway_signature(pathway);
        
        let pattern = MemoryPattern {
            pattern_id,
            pathway_signature: pathway_signature.clone(),
            query_features: query_features.clone(),
            success_metrics,
            usage_statistics: UsageStatistics {
                access_count: 1,
                recent_access_times: {
                    let mut times = VecDeque::new();
                    times.push_back(SystemTime::now());
                    times
                },
                access_frequency: 0.0,
                recency_weight: 1.0,
                importance_score: 0.5,
            },
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            consolidation_level: 0.0,
        };
        
        // Store pattern
        self.patterns.insert(pattern_id, pattern);
        
        // Update indices
        self.update_indices(pattern_id, &query_features, &pathway_signature)?;
        
        // Schedule consolidation if pattern shows promise
        if success_metrics.average_efficiency > self.consolidation_threshold {
            self.schedule_consolidation(pattern_id, ConsolidationType::Strengthen);
        }
        
        Ok(pattern_id)
    }
    
    fn extract_pathway_signature(&self, pathway: &ActivationPathway) -> Vec<NodeId> {
        // Extract key nodes from pathway for pattern matching
        let mut signature = Vec::new();
        
        // Include start and end nodes
        if let (Some(first), Some(last)) = (pathway.segments.first(), pathway.segments.last()) {
            signature.push(first.source_node);
            signature.push(last.target_node);
        }
        
        // Include high-activation intermediate nodes
        for segment in &pathway.segments {
            if segment.activation_transfer > 0.5 {
                signature.push(segment.target_node);
            }
        }
        
        // Remove duplicates and sort for consistent signature
        signature.sort_unstable();
        signature.dedup();
        
        signature
    }
    
    fn update_indices(
        &mut self,
        pattern_id: PatternId,
        query_features: &QueryFeatures,
        pathway_signature: &[NodeId],
    ) -> Result<(), MemoryError> {
        // Update query index
        self.query_index.intent_index
            .entry(query_features.intent_type.clone())
            .or_insert_with(Vec::new)
            .push(pattern_id);
        
        for entity_type in &query_features.entity_types {
            self.query_index.entity_index
                .entry(entity_type.clone())
                .or_insert_with(Vec::new)
                .push(pattern_id);
        }
        
        let complexity_bin = (query_features.complexity_score * 10.0) as u32;
        self.query_index.complexity_index
            .entry(complexity_bin)
            .or_insert_with(Vec::new)
            .push(pattern_id);
        
        for keyword in &query_features.context_keywords {
            self.query_index.keyword_index
                .entry(keyword.clone())
                .or_insert_with(Vec::new)
                .push(pattern_id);
        }
        
        // Update pattern index
        self.pattern_index.signature_index
            .entry(pathway_signature.to_vec())
            .or_insert_with(Vec::new)
            .push(pattern_id);
        
        self.pattern_index.length_index
            .entry(pathway_signature.len())
            .or_insert_with(Vec::new)
            .push(pattern_id);
        
        Ok(())
    }
    
    pub fn recall_similar_pathways(
        &mut self,
        query_features: &QueryFeatures,
        max_results: usize,
    ) -> Result<Vec<MemoryMatch>, MemoryError> {
        let mut candidates = Vec::new();
        
        // Find candidates based on different indices
        self.find_intent_candidates(query_features, &mut candidates);
        self.find_entity_candidates(query_features, &mut candidates);
        self.find_keyword_candidates(query_features, &mut candidates);
        self.find_complexity_candidates(query_features, &mut candidates);
        
        // Remove duplicates and calculate similarity scores
        candidates.sort_unstable();
        candidates.dedup();
        
        let mut matches = Vec::new();
        
        for pattern_id in candidates {
            if let Some(pattern) = self.patterns.get_mut(&pattern_id) {
                let similarity = self.calculate_similarity(query_features, &pattern.query_features);
                
                if similarity > 0.3 { // Minimum similarity threshold
                    // Update access statistics
                    pattern.usage_statistics.access_count += 1;
                    pattern.usage_statistics.recent_access_times.push_back(SystemTime::now());
                    pattern.last_accessed = SystemTime::now();
                    
                    // Keep recent access history manageable
                    if pattern.usage_statistics.recent_access_times.len() > 100 {
                        pattern.usage_statistics.recent_access_times.pop_front();
                    }
                    
                    matches.push(MemoryMatch {
                        pattern_id,
                        similarity_score: similarity,
                        pathway_signature: pattern.pathway_signature.clone(),
                        success_probability: pattern.success_metrics.average_efficiency,
                        usage_count: pattern.usage_statistics.access_count,
                        recency_weight: pattern.usage_statistics.recency_weight,
                    });
                }
            }
        }
        
        // Sort by composite score (similarity + success + recency)
        matches.sort_by(|a, b| {
            let score_a = a.similarity_score * 0.4 + a.success_probability * 0.4 + a.recency_weight * 0.2;
            let score_b = b.similarity_score * 0.4 + b.success_probability * 0.4 + b.recency_weight * 0.2;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        matches.truncate(max_results);
        Ok(matches)
    }
    
    fn find_intent_candidates(&self, query_features: &QueryFeatures, candidates: &mut Vec<PatternId>) {
        if let Some(pattern_ids) = self.query_index.intent_index.get(&query_features.intent_type) {
            candidates.extend(pattern_ids);
        }
    }
    
    fn find_entity_candidates(&self, query_features: &QueryFeatures, candidates: &mut Vec<PatternId>) {
        for entity_type in &query_features.entity_types {
            if let Some(pattern_ids) = self.query_index.entity_index.get(entity_type) {
                candidates.extend(pattern_ids);
            }
        }
    }
    
    fn find_keyword_candidates(&self, query_features: &QueryFeatures, candidates: &mut Vec<PatternId>) {
        for keyword in &query_features.context_keywords {
            if let Some(pattern_ids) = self.query_index.keyword_index.get(keyword) {
                candidates.extend(pattern_ids);
            }
        }
    }
    
    fn find_complexity_candidates(&self, query_features: &QueryFeatures, candidates: &mut Vec<PatternId>) {
        let complexity_bin = (query_features.complexity_score * 10.0) as u32;
        
        // Check current bin and adjacent bins
        for bin in (complexity_bin.saturating_sub(1))..=(complexity_bin + 1) {
            if let Some(pattern_ids) = self.query_index.complexity_index.get(&bin) {
                candidates.extend(pattern_ids);
            }
        }
    }
    
    fn calculate_similarity(&self, query1: &QueryFeatures, query2: &QueryFeatures) -> f32 {
        let mut similarity = 0.0;
        
        // Intent type similarity
        if query1.intent_type == query2.intent_type {
            similarity += 0.3;
        }
        
        // Entity type overlap
        let common_entities = query1.entity_types.iter()
            .filter(|e| query2.entity_types.contains(e))
            .count();
        let total_entities = (query1.entity_types.len() + query2.entity_types.len()).max(1);
        similarity += 0.2 * (2.0 * common_entities as f32 / total_entities as f32);
        
        // Complexity similarity
        let complexity_diff = (query1.complexity_score - query2.complexity_score).abs();
        similarity += 0.2 * (1.0 - complexity_diff).max(0.0);
        
        // Keyword overlap
        let common_keywords = query1.context_keywords.iter()
            .filter(|k| query2.context_keywords.contains(k))
            .count();
        let total_keywords = (query1.context_keywords.len() + query2.context_keywords.len()).max(1);
        similarity += 0.2 * (2.0 * common_keywords as f32 / total_keywords as f32);
        
        // Semantic embedding similarity (simplified cosine similarity)
        if !query1.semantic_embedding.is_empty() && !query2.semantic_embedding.is_empty() {
            let dot_product: f32 = query1.semantic_embedding.iter()
                .zip(&query2.semantic_embedding)
                .map(|(a, b)| a * b)
                .sum();
            
            let norm1: f32 = query1.semantic_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm2: f32 = query2.semantic_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if norm1 > 0.0 && norm2 > 0.0 {
                similarity += 0.1 * (dot_product / (norm1 * norm2));
            }
        }
        
        similarity.clamp(0.0, 1.0)
    }
}
```

### Step 3: Memory Consolidation and Maintenance

```rust
impl PathwayMemory {
    pub fn update_pattern_success(
        &mut self,
        pattern_id: PatternId,
        success_metrics: SuccessMetrics,
    ) -> Result<(), MemoryError> {
        let pattern = self.patterns.get_mut(&pattern_id)
            .ok_or(MemoryError::PatternNotFound)?;
        
        // Update success metrics with exponential moving average
        let alpha = 0.1; // Learning rate
        pattern.success_metrics.average_efficiency = 
            (1.0 - alpha) * pattern.success_metrics.average_efficiency + 
            alpha * success_metrics.average_efficiency;
        
        pattern.success_metrics.total_attempts += success_metrics.total_attempts;
        pattern.success_metrics.total_successes += success_metrics.total_successes;
        
        // Update importance score based on success
        pattern.usage_statistics.importance_score = 
            (pattern.success_metrics.total_successes as f32 / 
             pattern.success_metrics.total_attempts.max(1) as f32) * 0.7 +
            pattern.usage_statistics.access_frequency * 0.3;
        
        // Schedule consolidation for highly successful patterns
        if pattern.success_metrics.average_efficiency > self.consolidation_threshold &&
           pattern.usage_statistics.access_count > 5 {
            self.schedule_consolidation(pattern_id, ConsolidationType::Strengthen);
        }
        
        Ok(())
    }
    
    fn schedule_consolidation(&mut self, pattern_id: PatternId, consolidation_type: ConsolidationType) {
        let priority = match consolidation_type {
            ConsolidationType::Strengthen => 0.8,
            ConsolidationType::Integrate => 0.6,
            ConsolidationType::Generalize => 0.5,
            ConsolidationType::Specialize => 0.4,
        };
        
        let task = ConsolidationTask {
            pattern_id,
            consolidation_type,
            scheduled_time: SystemTime::now() + Duration::from_secs(3600), // 1 hour delay
            priority,
        };
        
        self.consolidation_scheduler.pending_consolidations.push_back(task);
    }
    
    pub fn process_consolidation(&mut self) -> Result<Vec<ConsolidationResult>, MemoryError> {
        let mut results = Vec::new();
        let now = SystemTime::now();
        
        // Process due consolidation tasks
        while let Some(task) = self.consolidation_scheduler.pending_consolidations.front() {
            if task.scheduled_time <= now {
                let task = self.consolidation_scheduler.pending_consolidations.pop_front().unwrap();
                let result = self.perform_consolidation(task)?;
                results.push(result);
            } else {
                break;
            }
        }
        
        self.consolidation_scheduler.last_consolidation = now;
        Ok(results)
    }
    
    fn perform_consolidation(&mut self, task: ConsolidationTask) -> Result<ConsolidationResult, MemoryError> {
        match task.consolidation_type {
            ConsolidationType::Strengthen => {
                self.strengthen_pattern(task.pattern_id)
            },
            ConsolidationType::Integrate => {
                self.integrate_similar_patterns(task.pattern_id)
            },
            ConsolidationType::Generalize => {
                self.generalize_pattern(task.pattern_id)
            },
            ConsolidationType::Specialize => {
                self.specialize_pattern(task.pattern_id)
            },
        }
    }
    
    fn strengthen_pattern(&mut self, pattern_id: PatternId) -> Result<ConsolidationResult, MemoryError> {
        let pattern = self.patterns.get_mut(&pattern_id)
            .ok_or(MemoryError::PatternNotFound)?;
        
        // Increase consolidation level
        pattern.consolidation_level = (pattern.consolidation_level + 0.1).min(1.0);
        
        // Strengthen recent access weights
        pattern.usage_statistics.recency_weight = 
            (pattern.usage_statistics.recency_weight * 1.1).min(1.0);
        
        Ok(ConsolidationResult {
            pattern_id,
            consolidation_type: ConsolidationType::Strengthen,
            success: true,
            changes_made: vec!["Increased consolidation level".to_string()],
        })
    }
    
    fn integrate_similar_patterns(&mut self, _pattern_id: PatternId) -> Result<ConsolidationResult, MemoryError> {
        // Simplified integration - in a full implementation, this would merge similar patterns
        Ok(ConsolidationResult {
            pattern_id: _pattern_id,
            consolidation_type: ConsolidationType::Integrate,
            success: true,
            changes_made: vec!["Pattern integration completed".to_string()],
        })
    }
    
    fn generalize_pattern(&mut self, _pattern_id: PatternId) -> Result<ConsolidationResult, MemoryError> {
        // Simplified generalization - would extract common features across similar patterns
        Ok(ConsolidationResult {
            pattern_id: _pattern_id,
            consolidation_type: ConsolidationType::Generalize,
            success: true,
            changes_made: vec!["Pattern generalization completed".to_string()],
        })
    }
    
    fn specialize_pattern(&mut self, _pattern_id: PatternId) -> Result<ConsolidationResult, MemoryError> {
        // Simplified specialization - would create more specific pattern variants
        Ok(ConsolidationResult {
            pattern_id: _pattern_id,
            consolidation_type: ConsolidationType::Specialize,
            success: true,
            changes_made: vec!["Pattern specialization completed".to_string()],
        })
    }
    
    fn perform_memory_cleanup(&mut self) -> Result<(), MemoryError> {
        // Calculate forgetting scores for all patterns
        let mut forgetting_candidates = Vec::new();
        let now = SystemTime::now();
        
        for (pattern_id, pattern) in &self.patterns {
            let age = now.duration_since(pattern.last_accessed)
                .unwrap_or(Duration::ZERO)
                .as_secs_f32() / 86400.0; // Days
            
            let forgetting_score = pattern.usage_statistics.importance_score * 
                (-self.forgetting_curve_rate * age).exp();
            
            forgetting_candidates.push((*pattern_id, forgetting_score));
        }
        
        // Sort by forgetting score (lowest first = most likely to forget)
        forgetting_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Remove bottom 10% of patterns
        let removal_count = (self.patterns.len() / 10).max(1);
        
        for i in 0..removal_count {
            if i < forgetting_candidates.len() {
                let pattern_id = forgetting_candidates[i].0;
                self.remove_pattern(pattern_id)?;
            }
        }
        
        Ok(())
    }
    
    fn remove_pattern(&mut self, pattern_id: PatternId) -> Result<(), MemoryError> {
        if let Some(pattern) = self.patterns.remove(&pattern_id) {
            // Remove from indices
            self.remove_from_indices(pattern_id, &pattern);
        }
        Ok(())
    }
    
    fn remove_from_indices(&mut self, pattern_id: PatternId, pattern: &MemoryPattern) {
        // Remove from query index
        if let Some(patterns) = self.query_index.intent_index.get_mut(&pattern.query_features.intent_type) {
            patterns.retain(|&p| p != pattern_id);
        }
        
        for entity_type in &pattern.query_features.entity_types {
            if let Some(patterns) = self.query_index.entity_index.get_mut(entity_type) {
                patterns.retain(|&p| p != pattern_id);
            }
        }
        
        // Remove from pattern index
        if let Some(patterns) = self.pattern_index.signature_index.get_mut(&pattern.pathway_signature) {
            patterns.retain(|&p| p != pattern_id);
        }
    }
    
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        let total_patterns = self.patterns.len();
        let consolidation_pending = self.consolidation_scheduler.pending_consolidations.len();
        
        let average_consolidation = if total_patterns > 0 {
            self.patterns.values()
                .map(|p| p.consolidation_level)
                .sum::<f32>() / total_patterns as f32
        } else {
            0.0
        };
        
        let average_importance = if total_patterns > 0 {
            self.patterns.values()
                .map(|p| p.usage_statistics.importance_score)
                .sum::<f32>() / total_patterns as f32
        } else {
            0.0
        };
        
        MemoryStatistics {
            total_patterns,
            consolidation_pending,
            average_consolidation_level: average_consolidation,
            average_importance_score: average_importance,
            memory_utilization: total_patterns as f32 / self.memory_capacity as f32,
        }
    }
}

// Implementation for sub-structures
impl QueryIndex {
    fn new() -> Self {
        Self {
            intent_index: HashMap::new(),
            entity_index: HashMap::new(),
            complexity_index: BTreeMap::new(),
            keyword_index: HashMap::new(),
        }
    }
}

impl PatternIndex {
    fn new() -> Self {
        Self {
            signature_index: HashMap::new(),
            length_index: BTreeMap::new(),
            similarity_clusters: Vec::new(),
            frequent_patterns: Vec::new(),
        }
    }
}

impl ConsolidationScheduler {
    fn new() -> Self {
        Self {
            pending_consolidations: VecDeque::new(),
            last_consolidation: SystemTime::now(),
            consolidation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryMatch {
    pub pattern_id: PatternId,
    pub similarity_score: f32,
    pub pathway_signature: Vec<NodeId>,
    pub success_probability: f32,
    pub usage_count: u64,
    pub recency_weight: f32,
}

#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub pattern_id: PatternId,
    pub consolidation_type: ConsolidationType,
    pub success: bool,
    pub changes_made: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub total_patterns: usize,
    pub consolidation_pending: usize,
    pub average_consolidation_level: f32,
    pub average_importance_score: f32,
    pub memory_utilization: f32,
}

#[derive(Debug, Clone)]
pub enum MemoryError {
    PatternNotFound,
    CapacityExceeded,
    IndexingError,
    ConsolidationFailed,
    StorageError,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::PatternNotFound => write!(f, "Memory pattern not found"),
            MemoryError::CapacityExceeded => write!(f, "Memory capacity exceeded"),
            MemoryError::IndexingError => write!(f, "Memory indexing error"),
            MemoryError::ConsolidationFailed => write!(f, "Memory consolidation failed"),
            MemoryError::StorageError => write!(f, "Memory storage error"),
        }
    }
}

impl std::error::Error for MemoryError {}
```

## File Locations

- `src/cognitive/learning/pathway_memory.rs` - Main implementation
- `src/cognitive/learning/mod.rs` - Module exports
- `tests/cognitive/learning/pathway_memory_tests.rs` - Test implementation

## Success Criteria

- [ ] PathwayMemory struct compiles and runs
- [ ] Pattern storage and indexing work correctly
- [ ] Similarity matching produces relevant results
- [ ] Memory consolidation processes function properly
- [ ] Forgetting curve and cleanup mechanisms work
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Pattern storage and retrieval
  - Similarity matching accuracy
  - Memory consolidation effectiveness
  - Capacity management and cleanup

## Test Requirements

```rust
#[test]
fn test_pathway_storage_and_retrieval() {
    let mut memory = PathwayMemory::new();
    
    let query_features = QueryFeatures {
        intent_type: "search".to_string(),
        entity_types: vec!["person".to_string()],
        complexity_score: 0.5,
        context_keywords: vec!["scientist".to_string()],
        semantic_embedding: vec![0.1, 0.2, 0.3],
    };
    
    let success_metrics = SuccessMetrics {
        average_efficiency: 0.8,
        average_activation_strength: 0.7,
        convergence_rate: 0.9,
        user_satisfaction: 0.85,
        total_successes: 1,
        total_attempts: 1,
    };
    
    // Create test pathway
    let pathway = create_test_pathway();
    
    let pattern_id = memory.store_pathway(&pathway, query_features.clone(), success_metrics).unwrap();
    
    // Test retrieval
    let matches = memory.recall_similar_pathways(&query_features, 5).unwrap();
    
    assert!(!matches.is_empty());
    assert_eq!(matches[0].pattern_id, pattern_id);
    assert!(matches[0].similarity_score > 0.9);
}

#[test]
fn test_similarity_matching() {
    let mut memory = PathwayMemory::new();
    
    // Store several patterns with different characteristics
    let base_features = QueryFeatures {
        intent_type: "search".to_string(),
        entity_types: vec!["person".to_string()],
        complexity_score: 0.5,
        context_keywords: vec!["scientist".to_string()],
        semantic_embedding: vec![0.1, 0.2, 0.3],
    };
    
    let similar_features = QueryFeatures {
        intent_type: "search".to_string(),
        entity_types: vec!["person".to_string()],
        complexity_score: 0.6,
        context_keywords: vec!["researcher".to_string()],
        semantic_embedding: vec![0.15, 0.25, 0.35],
    };
    
    let different_features = QueryFeatures {
        intent_type: "update".to_string(),
        entity_types: vec!["organization".to_string()],
        complexity_score: 0.9,
        context_keywords: vec!["company".to_string()],
        semantic_embedding: vec![0.8, 0.7, 0.6],
    };
    
    let success_metrics = create_test_success_metrics();
    let pathway = create_test_pathway();
    
    memory.store_pathway(&pathway, base_features.clone(), success_metrics.clone()).unwrap();
    memory.store_pathway(&pathway, similar_features.clone(), success_metrics.clone()).unwrap();
    memory.store_pathway(&pathway, different_features.clone(), success_metrics).unwrap();
    
    // Query with base features
    let matches = memory.recall_similar_pathways(&base_features, 5).unwrap();
    
    assert!(matches.len() >= 2);
    // Most similar should be exact match
    assert!(matches[0].similarity_score > matches[1].similarity_score);
    // Different pattern should have lower similarity
    assert!(matches[1].similarity_score > 0.3);
}

#[test]
fn test_memory_consolidation() {
    let mut memory = PathwayMemory::new();
    
    let query_features = create_test_query_features();
    let success_metrics = SuccessMetrics {
        average_efficiency: 0.9, // High efficiency to trigger consolidation
        average_activation_strength: 0.8,
        convergence_rate: 0.95,
        user_satisfaction: 0.9,
        total_successes: 10,
        total_attempts: 10,
    };
    
    let pathway = create_test_pathway();
    let pattern_id = memory.store_pathway(&pathway, query_features, success_metrics).unwrap();
    
    // Update pattern to increase usage
    for _ in 0..10 {
        let updated_metrics = create_test_success_metrics();
        memory.update_pattern_success(pattern_id, updated_metrics).unwrap();
    }
    
    // Process consolidation
    let results = memory.process_consolidation().unwrap();
    
    assert!(!results.is_empty());
    
    // Check pattern was strengthened
    let pattern = &memory.patterns[&pattern_id];
    assert!(pattern.consolidation_level > 0.0);
}

#[test]
fn test_memory_capacity_and_cleanup() {
    let mut memory = PathwayMemory::with_capacity(5);
    
    // Fill memory beyond capacity
    for i in 0..10 {
        let query_features = QueryFeatures {
            intent_type: format!("test_{}", i),
            entity_types: vec![format!("type_{}", i)],
            complexity_score: i as f32 / 10.0,
            context_keywords: vec![format!("keyword_{}", i)],
            semantic_embedding: vec![i as f32],
        };
        
        let pathway = create_test_pathway();
        let success_metrics = create_test_success_metrics();
        
        memory.store_pathway(&pathway, query_features, success_metrics).unwrap();
    }
    
    // Should not exceed capacity
    assert!(memory.patterns.len() <= 5);
    
    let stats = memory.get_memory_statistics();
    assert!(stats.memory_utilization <= 1.0);
}

#[test]
fn test_pattern_access_tracking() {
    let mut memory = PathwayMemory::new();
    
    let query_features = create_test_query_features();
    let success_metrics = create_test_success_metrics();
    let pathway = create_test_pathway();
    
    let pattern_id = memory.store_pathway(&pathway, query_features.clone(), success_metrics).unwrap();
    
    let initial_access_count = memory.patterns[&pattern_id].usage_statistics.access_count;
    
    // Access the pattern multiple times
    for _ in 0..5 {
        memory.recall_similar_pathways(&query_features, 1).unwrap();
    }
    
    let final_access_count = memory.patterns[&pattern_id].usage_statistics.access_count;
    
    assert!(final_access_count > initial_access_count);
}

// Helper functions for tests
fn create_test_pathway() -> ActivationPathway {
    use crate::cognitive::learning::pathway_tracing::{PathwaySegment, PathwayId};
    
    ActivationPathway {
        pathway_id: PathwayId(1),
        segments: vec![
            PathwaySegment {
                source_node: NodeId(1),
                target_node: NodeId(2),
                activation_transfer: 0.8,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(100),
                edge_weight: 1.0,
            },
            PathwaySegment {
                source_node: NodeId(2),
                target_node: NodeId(3),
                activation_transfer: 0.6,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(120),
                edge_weight: 1.0,
            },
        ],
        source_query: "test query".to_string(),
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
        total_activation: 1.4,
        path_efficiency: 0.75,
        significance_score: 0.8,
    }
}

fn create_test_query_features() -> QueryFeatures {
    QueryFeatures {
        intent_type: "search".to_string(),
        entity_types: vec!["person".to_string()],
        complexity_score: 0.5,
        context_keywords: vec!["test".to_string()],
        semantic_embedding: vec![0.1, 0.2, 0.3],
    }
}

fn create_test_success_metrics() -> SuccessMetrics {
    SuccessMetrics {
        average_efficiency: 0.7,
        average_activation_strength: 0.6,
        convergence_rate: 0.8,
        user_satisfaction: 0.75,
        total_successes: 1,
        total_attempts: 1,
    }
}
```

## Quality Gates

- [ ] Memory retrieval latency < 10ms for 100k patterns
- [ ] Similarity matching accuracy > 85% on test queries
- [ ] Memory consolidation improves pattern effectiveness
- [ ] Capacity management prevents memory overflow
- [ ] Thread-safe concurrent access verified
- [ ] Persistent storage maintains data integrity

## Next Task

Upon completion, proceed to **22_pathway_pruning.md**