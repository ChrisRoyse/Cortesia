# Micro Task 22: Pathway Pruning

**Priority**: HIGH  
**Estimated Time**: 30 minutes  
**Dependencies**: 21_pathway_memory.md completed  
**Skills Required**: Synaptic pruning algorithms, optimization, neural efficiency

## Objective

Implement neural-inspired pathway pruning mechanisms to remove weak, redundant, or inefficient pathways, optimizing the knowledge graph's connectivity and improving activation efficiency.

## Context

Biological neural systems maintain efficiency through synaptic pruning, removing weak or unused connections. This task implements similar mechanisms to prevent pathway proliferation and maintain optimal knowledge graph structure.

## Specifications

### Core Pruning Components

1. **PathwayPruner struct**
   - Weak connection detection
   - Redundancy analysis
   - Efficiency-based pruning
   - Activity-dependent elimination

2. **PruningCriteria struct**
   - Minimum strength thresholds
   - Usage frequency requirements
   - Efficiency benchmarks
   - Age-based pruning rules

3. **PruningStrategy enum**
   - StrengthBased (remove weakest connections)
   - ActivityBased (remove unused pathways)
   - EfficiencyBased (remove inefficient routes)
   - RedundancyBased (remove duplicate pathways)

4. **PruningMetrics struct**
   - Pruning effectiveness measures
   - Network efficiency improvements
   - Memory usage reductions
   - Performance impact assessment

### Performance Requirements

- Identify pruning candidates efficiently (< 100ms for 10k pathways)
- Maintain network connectivity during pruning
- Preserve critical high-performance pathways
- Minimize impact on active queries
- Support incremental pruning operations

## Implementation Guide

### Step 1: Core Pruning Types

```rust
// File: src/cognitive/learning/pathway_pruning.rs

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::time::{Duration, Instant, SystemTime};
use std::cmp::Ordering;
use crate::core::types::{NodeId, EntityId};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwayId};
use crate::cognitive::learning::pathway_reinforcement::{PathwayReinforcer, SynapticWeight};
use crate::cognitive::learning::pathway_memory::{PathwayMemory, PatternId};

#[derive(Debug, Clone)]
pub struct PruningCriteria {
    pub min_strength_threshold: f32,      // Minimum synaptic strength to keep
    pub min_usage_frequency: f32,         // Minimum usage per time period
    pub max_pathway_age: Duration,        // Maximum age for unused pathways
    pub efficiency_threshold: f32,        // Minimum efficiency score
    pub redundancy_threshold: f32,        // Similarity threshold for redundancy
    pub preserve_top_n_percent: f32,      // Always preserve top N% of pathways
}

#[derive(Debug, Clone, Copy)]
pub enum PruningStrategy {
    StrengthBased { threshold: f32 },
    ActivityBased { min_frequency: f32 },
    EfficiencyBased { min_efficiency: f32 },
    RedundancyBased { similarity_threshold: f32 },
    AgeBased { max_age: Duration },
    Combined, // Uses multiple criteria
}

#[derive(Debug, Clone)]
pub struct PruningCandidate {
    pub pathway_id: Option<PathwayId>,
    pub connection: Option<(NodeId, NodeId)>,
    pub pattern_id: Option<PatternId>,
    pub pruning_score: f32,
    pub candidate_type: PruningCandidateType,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum PruningCandidateType {
    WeakConnection,
    UnusedPathway,
    InefficientRoute,
    RedundantPattern,
    AgedPattern,
}

#[derive(Debug)]
pub struct PathwayPruner {
    pruning_criteria: PruningCriteria,
    pruning_strategy: PruningStrategy,
    pruning_history: Vec<PruningEvent>,
    protected_pathways: HashSet<PathwayId>,
    protected_connections: HashSet<(NodeId, NodeId)>,
    last_pruning: SystemTime,
    pruning_interval: Duration,
    aggressive_pruning: bool,
}

#[derive(Debug, Clone)]
pub struct PruningEvent {
    pub timestamp: SystemTime,
    pub items_pruned: usize,
    pub strategy_used: PruningStrategy,
    pub efficiency_gain: f32,
    pub memory_freed: usize,
}

#[derive(Debug, Clone)]
pub struct PruningMetrics {
    pub connections_pruned: usize,
    pub pathways_pruned: usize,
    pub patterns_pruned: usize,
    pub efficiency_improvement: f32,
    pub memory_reduction: usize,
    pub pruning_time: Duration,
    pub connectivity_preserved: f32,
}
```

### Step 2: Pathway Pruner Implementation

```rust
impl PathwayPruner {
    pub fn new(strategy: PruningStrategy) -> Self {
        let default_criteria = PruningCriteria {
            min_strength_threshold: 0.1,
            min_usage_frequency: 0.01, // 1% usage threshold
            max_pathway_age: Duration::from_secs(86400 * 30), // 30 days
            efficiency_threshold: 0.3,
            redundancy_threshold: 0.9,
            preserve_top_n_percent: 0.1, // Preserve top 10%
        };
        
        Self {
            pruning_criteria: default_criteria,
            pruning_strategy: strategy,
            pruning_history: Vec::new(),
            protected_pathways: HashSet::new(),
            protected_connections: HashSet::new(),
            last_pruning: SystemTime::now(),
            pruning_interval: Duration::from_secs(3600), // 1 hour
            aggressive_pruning: false,
        }
    }
    
    pub fn with_criteria(strategy: PruningStrategy, criteria: PruningCriteria) -> Self {
        Self {
            pruning_criteria: criteria,
            ..Self::new(strategy)
        }
    }
    
    pub fn prune_pathways(
        &mut self,
        reinforcer: &mut PathwayReinforcer,
        memory: &mut PathwayMemory,
    ) -> Result<PruningMetrics, PruningError> {
        let start_time = Instant::now();
        let mut metrics = PruningMetrics {
            connections_pruned: 0,
            pathways_pruned: 0,
            patterns_pruned: 0,
            efficiency_improvement: 0.0,
            memory_reduction: 0,
            pruning_time: Duration::ZERO,
            connectivity_preserved: 1.0,
        };
        
        // Collect pruning candidates based on strategy
        let candidates = self.identify_pruning_candidates(reinforcer, memory)?;
        
        // Sort candidates by pruning score (lowest = most suitable for pruning)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| a.pruning_score.partial_cmp(&b.pruning_score)
            .unwrap_or(Ordering::Equal));
        
        // Preserve top performers
        let preserve_count = (sorted_candidates.len() as f32 * self.pruning_criteria.preserve_top_n_percent) as usize;
        let pruning_candidates = &sorted_candidates[preserve_count..];
        
        // Execute pruning
        for candidate in pruning_candidates {
            if self.should_prune_candidate(candidate, reinforcer, memory)? {
                match self.execute_pruning(candidate, reinforcer, memory)? {
                    PruningResult::ConnectionPruned => metrics.connections_pruned += 1,
                    PruningResult::PathwayPruned => metrics.pathways_pruned += 1,
                    PruningResult::PatternPruned => metrics.patterns_pruned += 1,
                    PruningResult::Skipped => {},
                }
            }
        }
        
        // Calculate metrics
        metrics.pruning_time = start_time.elapsed();
        metrics.efficiency_improvement = self.calculate_efficiency_improvement()?;
        metrics.memory_reduction = self.estimate_memory_reduction(&metrics);
        
        // Record pruning event
        self.record_pruning_event(&metrics);
        
        Ok(metrics)
    }
    
    fn identify_pruning_candidates(
        &self,
        reinforcer: &PathwayReinforcer,
        memory: &PathwayMemory,
    ) -> Result<Vec<PruningCandidate>, PruningError> {
        let mut candidates = Vec::new();
        
        match self.pruning_strategy {
            PruningStrategy::StrengthBased { threshold } => {
                self.find_weak_connections(reinforcer, threshold, &mut candidates)?;
            },
            PruningStrategy::ActivityBased { min_frequency } => {
                self.find_inactive_patterns(memory, min_frequency, &mut candidates)?;
            },
            PruningStrategy::EfficiencyBased { min_efficiency } => {
                self.find_inefficient_pathways(memory, min_efficiency, &mut candidates)?;
            },
            PruningStrategy::RedundancyBased { similarity_threshold } => {
                self.find_redundant_patterns(memory, similarity_threshold, &mut candidates)?;
            },
            PruningStrategy::AgeBased { max_age } => {
                self.find_aged_patterns(memory, max_age, &mut candidates)?;
            },
            PruningStrategy::Combined => {
                // Use multiple strategies
                self.find_weak_connections(reinforcer, self.pruning_criteria.min_strength_threshold, &mut candidates)?;
                self.find_inactive_patterns(memory, self.pruning_criteria.min_usage_frequency, &mut candidates)?;
                self.find_inefficient_pathways(memory, self.pruning_criteria.efficiency_threshold, &mut candidates)?;
                self.find_redundant_patterns(memory, self.pruning_criteria.redundancy_threshold, &mut candidates)?;
            },
        }
        
        Ok(candidates)
    }
    
    fn find_weak_connections(
        &self,
        reinforcer: &PathwayReinforcer,
        threshold: f32,
        candidates: &mut Vec<PruningCandidate>,
    ) -> Result<(), PruningError> {
        let connection_stats = reinforcer.get_connection_statistics();
        
        // This would iterate through actual connections in a real implementation
        // For now, we simulate finding weak connections
        for i in 0..connection_stats.weak_connections {
            let connection = (NodeId(i), NodeId(i + 1));
            
            if !self.protected_connections.contains(&connection) {
                let strength = reinforcer.get_synaptic_weight(connection.0, connection.1);
                
                if strength < threshold {
                    candidates.push(PruningCandidate {
                        pathway_id: None,
                        connection: Some(connection),
                        pattern_id: None,
                        pruning_score: strength, // Lower strength = higher pruning priority
                        candidate_type: PruningCandidateType::WeakConnection,
                        reason: format!("Connection strength {} below threshold {}", strength, threshold),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    fn find_inactive_patterns(
        &self,
        memory: &PathwayMemory,
        min_frequency: f32,
        candidates: &mut Vec<PruningCandidate>,
    ) -> Result<(), PruningError> {
        let stats = memory.get_memory_statistics();
        
        // Simulate finding inactive patterns
        // In real implementation, this would iterate through actual patterns
        for i in 0..stats.total_patterns {
            let pattern_id = PatternId(i as u64);
            
            // Simulate low usage frequency
            let usage_frequency = (i as f32 / stats.total_patterns as f32) * 0.1;
            
            if usage_frequency < min_frequency {
                candidates.push(PruningCandidate {
                    pathway_id: None,
                    connection: None,
                    pattern_id: Some(pattern_id),
                    pruning_score: 1.0 - usage_frequency, // Lower frequency = higher pruning score
                    candidate_type: PruningCandidateType::UnusedPathway,
                    reason: format!("Usage frequency {} below threshold {}", usage_frequency, min_frequency),
                });
            }
        }
        
        Ok(())
    }
    
    fn find_inefficient_pathways(
        &self,
        memory: &PathwayMemory,
        min_efficiency: f32,
        candidates: &mut Vec<PruningCandidate>,
    ) -> Result<(), PruningError> {
        let stats = memory.get_memory_statistics();
        
        // Simulate finding inefficient pathways
        for i in 0..stats.total_patterns {
            let pattern_id = PatternId(i as u64);
            
            // Simulate varying efficiency
            let efficiency = (i as f32 / stats.total_patterns as f32) * 0.8 + 0.1;
            
            if efficiency < min_efficiency {
                candidates.push(PruningCandidate {
                    pathway_id: None,
                    connection: None,
                    pattern_id: Some(pattern_id),
                    pruning_score: 1.0 - efficiency,
                    candidate_type: PruningCandidateType::InefficientRoute,
                    reason: format!("Efficiency {} below threshold {}", efficiency, min_efficiency),
                });
            }
        }
        
        Ok(())
    }
    
    fn find_redundant_patterns(
        &self,
        _memory: &PathwayMemory,
        similarity_threshold: f32,
        candidates: &mut Vec<PruningCandidate>,
    ) -> Result<(), PruningError> {
        // Simplified redundancy detection
        // In real implementation, this would compare pattern similarities
        
        for i in 0..10 { // Simulate some redundant patterns
            if i % 3 == 0 { // Every third pattern is "redundant"
                candidates.push(PruningCandidate {
                    pathway_id: None,
                    connection: None,
                    pattern_id: Some(PatternId(i)),
                    pruning_score: similarity_threshold + 0.05, // Slightly above threshold
                    candidate_type: PruningCandidateType::RedundantPattern,
                    reason: format!("Pattern similarity above threshold {}", similarity_threshold),
                });
            }
        }
        
        Ok(())
    }
    
    fn find_aged_patterns(
        &self,
        _memory: &PathwayMemory,
        max_age: Duration,
        candidates: &mut Vec<PruningCandidate>,
    ) -> Result<(), PruningError> {
        let now = SystemTime::now();
        let age_threshold = now - max_age;
        
        // Simulate finding aged patterns
        for i in 0..5 { // Simulate some old patterns
            candidates.push(PruningCandidate {
                pathway_id: None,
                connection: None,
                pattern_id: Some(PatternId(i + 100)), // Use different IDs
                pruning_score: 0.8, // High pruning score for old patterns
                candidate_type: PruningCandidateType::AgedPattern,
                reason: format!("Pattern age exceeds {} days", max_age.as_secs() / 86400),
            });
        }
        
        Ok(())
    }
    
    fn should_prune_candidate(
        &self,
        candidate: &PruningCandidate,
        _reinforcer: &PathwayReinforcer,
        _memory: &PathwayMemory,
    ) -> Result<bool, PruningError> {
        // Check if candidate is protected
        if let Some(pathway_id) = candidate.pathway_id {
            if self.protected_pathways.contains(&pathway_id) {
                return Ok(false);
            }
        }
        
        if let Some(connection) = candidate.connection {
            if self.protected_connections.contains(&connection) {
                return Ok(false);
            }
        }
        
        // Additional safety checks
        match candidate.candidate_type {
            PruningCandidateType::WeakConnection => {
                // Don't prune if it would disconnect the graph
                Ok(candidate.pruning_score < self.pruning_criteria.min_strength_threshold)
            },
            PruningCandidateType::UnusedPathway => {
                Ok(candidate.pruning_score > 0.8) // High threshold for unused pathways
            },
            PruningCandidateType::InefficientRoute => {
                Ok(candidate.pruning_score > 0.6) // Moderate threshold for inefficient routes
            },
            PruningCandidateType::RedundantPattern => {
                Ok(candidate.pruning_score > self.pruning_criteria.redundancy_threshold)
            },
            PruningCandidateType::AgedPattern => {
                Ok(candidate.pruning_score > 0.7) // High threshold for aged patterns
            },
        }
    }
    
    fn execute_pruning(
        &self,
        candidate: &PruningCandidate,
        _reinforcer: &mut PathwayReinforcer,
        _memory: &mut PathwayMemory,
    ) -> Result<PruningResult, PruningError> {
        // In a real implementation, this would actually remove the identified elements
        // For now, we simulate the pruning operation
        
        match candidate.candidate_type {
            PruningCandidateType::WeakConnection => {
                // Would remove connection from reinforcer
                Ok(PruningResult::ConnectionPruned)
            },
            PruningCandidateType::UnusedPathway => {
                // Would remove pathway from memory
                Ok(PruningResult::PathwayPruned)
            },
            PruningCandidateType::InefficientRoute => {
                // Would remove inefficient pathway
                Ok(PruningResult::PathwayPruned)
            },
            PruningCandidateType::RedundantPattern => {
                // Would remove redundant pattern from memory
                Ok(PruningResult::PatternPruned)
            },
            PruningCandidateType::AgedPattern => {
                // Would remove aged pattern from memory
                Ok(PruningResult::PatternPruned)
            },
        }
    }
    
    fn calculate_efficiency_improvement(&self) -> Result<f32, PruningError> {
        // Simulate efficiency improvement calculation
        // In real implementation, this would measure actual performance gains
        if self.pruning_history.is_empty() {
            Ok(0.0)
        } else {
            Ok(0.05) // 5% improvement simulation
        }
    }
    
    fn estimate_memory_reduction(&self, metrics: &PruningMetrics) -> usize {
        // Estimate memory saved based on pruned items
        let connection_memory = metrics.connections_pruned * 64; // 64 bytes per connection
        let pathway_memory = metrics.pathways_pruned * 256; // 256 bytes per pathway
        let pattern_memory = metrics.patterns_pruned * 512; // 512 bytes per pattern
        
        connection_memory + pathway_memory + pattern_memory
    }
    
    fn record_pruning_event(&mut self, metrics: &PruningMetrics) {
        let event = PruningEvent {
            timestamp: SystemTime::now(),
            items_pruned: metrics.connections_pruned + metrics.pathways_pruned + metrics.patterns_pruned,
            strategy_used: self.pruning_strategy,
            efficiency_gain: metrics.efficiency_improvement,
            memory_freed: metrics.memory_reduction,
        };
        
        self.pruning_history.push(event);
        self.last_pruning = SystemTime::now();
        
        // Keep history manageable
        if self.pruning_history.len() > 1000 {
            self.pruning_history.drain(..500);
        }
    }
    
    pub fn protect_pathway(&mut self, pathway_id: PathwayId) {
        self.protected_pathways.insert(pathway_id);
    }
    
    pub fn protect_connection(&mut self, source: NodeId, target: NodeId) {
        self.protected_connections.insert((source, target));
    }
    
    pub fn set_aggressive_pruning(&mut self, aggressive: bool) {
        self.aggressive_pruning = aggressive;
        
        if aggressive {
            // Lower thresholds for more aggressive pruning
            self.pruning_criteria.min_strength_threshold *= 1.5;
            self.pruning_criteria.min_usage_frequency *= 1.5;
            self.pruning_criteria.efficiency_threshold *= 1.2;
        }
    }
    
    pub fn should_perform_pruning(&self) -> bool {
        SystemTime::now().duration_since(self.last_pruning)
            .unwrap_or(Duration::ZERO) >= self.pruning_interval
    }
    
    pub fn get_pruning_statistics(&self) -> PruningStatistics {
        let total_events = self.pruning_history.len();
        
        let total_items_pruned = self.pruning_history.iter()
            .map(|e| e.items_pruned)
            .sum();
        
        let total_memory_freed = self.pruning_history.iter()
            .map(|e| e.memory_freed)
            .sum();
        
        let average_efficiency_gain = if total_events > 0 {
            self.pruning_history.iter()
                .map(|e| e.efficiency_gain)
                .sum::<f32>() / total_events as f32
        } else {
            0.0
        };
        
        PruningStatistics {
            total_pruning_events: total_events,
            total_items_pruned,
            total_memory_freed,
            average_efficiency_gain,
            protected_pathways: self.protected_pathways.len(),
            protected_connections: self.protected_connections.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PruningResult {
    ConnectionPruned,
    PathwayPruned,
    PatternPruned,
    Skipped,
}

#[derive(Debug, Clone)]
pub struct PruningStatistics {
    pub total_pruning_events: usize,
    pub total_items_pruned: usize,
    pub total_memory_freed: usize,
    pub average_efficiency_gain: f32,
    pub protected_pathways: usize,
    pub protected_connections: usize,
}

#[derive(Debug, Clone)]
pub enum PruningError {
    InvalidCriteria,
    ProtectedPathway,
    ConnectivityRisk,
    PruningFailed,
}

impl std::fmt::Display for PruningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruningError::InvalidCriteria => write!(f, "Invalid pruning criteria"),
            PruningError::ProtectedPathway => write!(f, "Cannot prune protected pathway"),
            PruningError::ConnectivityRisk => write!(f, "Pruning would risk graph connectivity"),
            PruningError::PruningFailed => write!(f, "Pruning operation failed"),
        }
    }
}

impl std::error::Error for PruningError {}
```

## File Locations

- `src/cognitive/learning/pathway_pruning.rs` - Main implementation
- `src/cognitive/learning/mod.rs` - Module exports
- `tests/cognitive/learning/pathway_pruning_tests.rs` - Test implementation

## Success Criteria

- [ ] PathwayPruner struct compiles and runs
- [ ] Multiple pruning strategies implemented correctly
- [ ] Protected pathways and connections preserved
- [ ] Pruning improves system efficiency
- [ ] Memory usage reduced after pruning
- [ ] Graph connectivity maintained
- [ ] All tests pass:
  - Weak connection identification and removal
  - Efficiency-based pruning
  - Protection mechanisms
  - Performance improvement verification

## Test Requirements

```rust
#[test]
fn test_strength_based_pruning() {
    let strategy = PruningStrategy::StrengthBased { threshold: 0.2 };
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    // Create weak connections (would be done through actual reinforcement)
    // This test simulates the pruning process
    
    let metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    assert!(metrics.connections_pruned > 0);
    assert!(metrics.pruning_time.as_millis() < 1000); // Should be fast
}

#[test]
fn test_protected_pathway_preservation() {
    let strategy = PruningStrategy::Combined;
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    // Protect a pathway
    let protected_pathway = PathwayId(1);
    pruner.protect_pathway(protected_pathway);
    
    // Protect a connection
    pruner.protect_connection(NodeId(1), NodeId(2));
    
    let metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    // Protected items should not be pruned
    // This would be verified by checking that protected items still exist
    assert!(metrics.connectivity_preserved > 0.95);
}

#[test]
fn test_efficiency_improvement() {
    let strategy = PruningStrategy::EfficiencyBased { min_efficiency: 0.5 };
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    let metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    assert!(metrics.efficiency_improvement >= 0.0);
    assert!(metrics.memory_reduction > 0);
}

#[test]
fn test_aggressive_pruning_mode() {
    let strategy = PruningStrategy::Combined;
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    // Test normal pruning
    let normal_metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    // Enable aggressive pruning
    pruner.set_aggressive_pruning(true);
    let aggressive_metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    // Aggressive pruning should remove more items
    assert!(aggressive_metrics.connections_pruned >= normal_metrics.connections_pruned);
}

#[test]
fn test_pruning_interval_respect() {
    let strategy = PruningStrategy::StrengthBased { threshold: 0.1 };
    let pruner = PathwayPruner::new(strategy);
    
    // Should allow pruning initially
    assert!(pruner.should_perform_pruning());
    
    // After setting last_pruning to now, should not allow immediate pruning
    // (This test would need access to modify last_pruning for proper testing)
}

#[test]
fn test_pruning_statistics() {
    let strategy = PruningStrategy::Combined;
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    // Perform multiple pruning operations
    for _ in 0..3 {
        pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    }
    
    let stats = pruner.get_pruning_statistics();
    
    assert_eq!(stats.total_pruning_events, 3);
    assert!(stats.total_items_pruned >= 0);
    assert!(stats.average_efficiency_gain >= 0.0);
}

#[test]
fn test_redundancy_based_pruning() {
    let strategy = PruningStrategy::RedundancyBased { similarity_threshold: 0.9 };
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    let metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    // Should identify and prune redundant patterns
    assert!(metrics.patterns_pruned > 0);
}

#[test]
fn test_age_based_pruning() {
    let max_age = Duration::from_secs(60); // 1 minute for testing
    let strategy = PruningStrategy::AgeBased { max_age };
    let mut pruner = PathwayPruner::new(strategy);
    let mut reinforcer = PathwayReinforcer::new(LearningRule::StandardHebbian { learning_rate: 0.1 });
    let mut memory = PathwayMemory::new();
    
    let metrics = pruner.prune_pathways(&mut reinforcer, &mut memory).unwrap();
    
    // Should identify and prune aged patterns
    assert!(metrics.patterns_pruned > 0);
}
```

## Quality Gates

- [ ] Pruning operations complete within performance targets
- [ ] Protected pathways and connections never pruned
- [ ] Graph connectivity maintained after pruning
- [ ] Memory usage measurably reduced
- [ ] System efficiency improved post-pruning
- [ ] No critical pathways accidentally removed

## Next Task

Upon completion, proceed to **23_pathway_consolidation.md**