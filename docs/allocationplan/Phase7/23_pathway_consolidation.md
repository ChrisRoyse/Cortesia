# Micro Task 23: Pathway Consolidation

**Priority**: HIGH  
**Estimated Time**: 35 minutes  
**Dependencies**: 22_pathway_pruning.md completed  
**Skills Required**: Pattern merging, schema formation, memory consolidation

## Objective

Implement pathway consolidation mechanisms that merge similar pathways into generalized patterns, create hierarchical pathway schemas, and optimize pathway representations for improved efficiency and recall.

## Context

Biological memory consolidation transforms specific experiences into generalized knowledge structures. This task implements similar mechanisms to merge successful pathways into reusable patterns, creating a hierarchical knowledge structure that improves query efficiency.

## Specifications

### Core Consolidation Components

1. **PathwayConsolidator struct**
   - Pattern similarity detection
   - Pathway merging algorithms
   - Hierarchical schema creation
   - Consolidated pattern optimization

2. **ConsolidationStrategy enum**
   - SimilarityMerge (merge highly similar pathways)
   - HierarchicalGroup (create parent-child relationships)
   - FeatureExtraction (extract common sub-patterns)
   - TemplateCreation (create reusable templates)

3. **ConsolidatedPattern struct**
   - Generalized pathway representation
   - Variable components and fixed structure
   - Success probability distribution
   - Usage context constraints

4. **ConsolidationMetrics struct**
   - Consolidation effectiveness measures
   - Memory efficiency improvements
   - Pattern generalization quality
   - Recall performance impact

### Performance Requirements

- Identify consolidation opportunities efficiently
- Maintain pattern specificity while generalizing
- Preserve critical pathway variations
- Support incremental consolidation
- Minimize impact on active queries

## Implementation Guide

### Step 1: Core Consolidation Types

```rust
// File: src/cognitive/learning/pathway_consolidation.rs

use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};
use crate::core::types::{NodeId, EntityId};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwayId, PathwaySegment};
use crate::cognitive::learning::pathway_memory::{PathwayMemory, PatternId, MemoryPattern};

#[derive(Debug, Clone, Copy)]
pub enum ConsolidationStrategy {
    SimilarityMerge { threshold: f32 },
    HierarchicalGroup { min_group_size: usize },
    FeatureExtraction { min_frequency: f32 },
    TemplateCreation { generalization_level: f32 },
    AdaptiveConsolidation, // Uses multiple strategies based on data
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedPattern {
    pub pattern_id: ConsolidatedPatternId,
    pub template_structure: PathwayTemplate,
    pub variable_components: Vec<VariableComponent>,
    pub fixed_components: Vec<FixedComponent>,
    pub consolidated_from: Vec<PatternId>,
    pub usage_contexts: Vec<UsageContext>,
    pub success_distribution: SuccessDistribution,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
    pub consolidation_level: usize, // Depth of consolidation hierarchy
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidatedPatternId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayTemplate {
    pub template_nodes: Vec<TemplateNode>,
    pub template_edges: Vec<TemplateEdge>,
    pub activation_pattern: Vec<f32>,
    pub timing_constraints: Vec<TimingConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateNode {
    pub node_type: NodeType,
    pub position_index: usize,
    pub is_variable: bool,
    pub constraints: Vec<NodeConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Fixed(NodeId),
    Variable { entity_type: String, constraints: Vec<String> },
    Wildcard { min_activation: f32, max_activation: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEdge {
    pub source_index: usize,
    pub target_index: usize,
    pub weight_range: (f32, f32),
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableComponent {
    pub component_id: String,
    pub possible_values: Vec<NodeId>,
    pub selection_probability: HashMap<NodeId, f32>,
    pub context_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedComponent {
    pub component_id: String,
    pub fixed_nodes: Vec<NodeId>,
    pub required_activation: f32,
    pub timing_requirements: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageContext {
    pub context_type: String,
    pub context_features: HashMap<String, String>,
    pub success_rate: f32,
    pub usage_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessDistribution {
    pub mean_efficiency: f32,
    pub efficiency_variance: f32,
    pub success_rate: f32,
    pub confidence_interval: (f32, f32),
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraint {
    pub source_index: usize,
    pub target_index: usize,
    pub min_delay: Duration,
    pub max_delay: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConstraint {
    pub constraint_type: String,
    pub constraint_value: String,
    pub required: bool,
}

#[derive(Debug)]
pub struct PathwayConsolidator {
    consolidation_strategy: ConsolidationStrategy,
    consolidated_patterns: HashMap<ConsolidatedPatternId, ConsolidatedPattern>,
    consolidation_candidates: Vec<ConsolidationCandidate>,
    next_pattern_id: u64,
    similarity_threshold: f32,
    min_consolidation_size: usize,
    consolidation_history: Vec<ConsolidationEvent>,
}

#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub candidate_patterns: Vec<PatternId>,
    pub similarity_score: f32,
    pub consolidation_benefit: f32,
    pub strategy: ConsolidationStrategy,
    pub estimated_savings: ConsolidationSavings,
}

#[derive(Debug, Clone)]
pub struct ConsolidationSavings {
    pub memory_reduction: usize,
    pub access_time_improvement: f32,
    pub generalization_value: f32,
}

#[derive(Debug, Clone)]
pub struct ConsolidationEvent {
    pub timestamp: SystemTime,
    pub strategy_used: ConsolidationStrategy,
    pub patterns_consolidated: usize,
    pub memory_saved: usize,
    pub efficiency_improvement: f32,
}
```

### Step 2: Pathway Consolidator Implementation

```rust
impl PathwayConsolidator {
    pub fn new(strategy: ConsolidationStrategy) -> Self {
        Self {
            consolidation_strategy: strategy,
            consolidated_patterns: HashMap::new(),
            consolidation_candidates: Vec::new(),
            next_pattern_id: 1,
            similarity_threshold: 0.8,
            min_consolidation_size: 3,
            consolidation_history: Vec::new(),
        }
    }
    
    pub fn consolidate_pathways(
        &mut self,
        memory: &mut PathwayMemory,
    ) -> Result<ConsolidationMetrics, ConsolidationError> {
        let start_time = Instant::now();
        
        // Identify consolidation opportunities
        self.identify_consolidation_candidates(memory)?;
        
        // Sort candidates by benefit
        self.consolidation_candidates.sort_by(|a, b| 
            b.consolidation_benefit.partial_cmp(&a.consolidation_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        );
        
        let mut metrics = ConsolidationMetrics {
            patterns_consolidated: 0,
            consolidated_patterns_created: 0,
            memory_efficiency_gain: 0.0,
            access_time_improvement: 0.0,
            consolidation_time: Duration::ZERO,
            generalization_quality: 0.0,
        };
        
        // Process consolidation candidates
        for candidate in &self.consolidation_candidates {
            if self.should_consolidate_candidate(candidate)? {
                let consolidation_result = self.execute_consolidation(candidate, memory)?;
                self.update_metrics(&mut metrics, &consolidation_result);
            }
        }
        
        metrics.consolidation_time = start_time.elapsed();
        
        // Record consolidation event
        self.record_consolidation_event(&metrics);
        
        // Clear processed candidates
        self.consolidation_candidates.clear();
        
        Ok(metrics)
    }
    
    fn identify_consolidation_candidates(
        &mut self,
        memory: &PathwayMemory,
    ) -> Result<(), ConsolidationError> {
        let stats = memory.get_memory_statistics();
        
        match self.consolidation_strategy {
            ConsolidationStrategy::SimilarityMerge { threshold } => {
                self.find_similar_patterns(memory, threshold)?;
            },
            ConsolidationStrategy::HierarchicalGroup { min_group_size } => {
                self.find_hierarchical_groups(memory, min_group_size)?;
            },
            ConsolidationStrategy::FeatureExtraction { min_frequency } => {
                self.find_common_features(memory, min_frequency)?;
            },
            ConsolidationStrategy::TemplateCreation { generalization_level } => {
                self.find_template_opportunities(memory, generalization_level)?;
            },
            ConsolidationStrategy::AdaptiveConsolidation => {
                self.adaptive_consolidation_search(memory)?;
            },
        }
        
        Ok(())
    }
    
    fn find_similar_patterns(
        &mut self,
        _memory: &PathwayMemory,
        threshold: f32,
    ) -> Result<(), ConsolidationError> {
        // Simulate finding similar patterns
        // In real implementation, this would analyze actual patterns in memory
        
        for group_id in 0..5 { // Simulate 5 groups of similar patterns
            let pattern_ids: Vec<PatternId> = (0..4)
                .map(|i| PatternId(group_id * 10 + i))
                .collect();
            
            let candidate = ConsolidationCandidate {
                candidate_patterns: pattern_ids,
                similarity_score: threshold + 0.05, // Just above threshold
                consolidation_benefit: 0.7,
                strategy: ConsolidationStrategy::SimilarityMerge { threshold },
                estimated_savings: ConsolidationSavings {
                    memory_reduction: 1024,
                    access_time_improvement: 0.15,
                    generalization_value: 0.6,
                },
            };
            
            self.consolidation_candidates.push(candidate);
        }
        
        Ok(())
    }
    
    fn find_hierarchical_groups(
        &mut self,
        _memory: &PathwayMemory,
        min_group_size: usize,
    ) -> Result<(), ConsolidationError> {
        // Simulate finding hierarchical grouping opportunities
        
        for hierarchy_level in 0..3 {
            let pattern_ids: Vec<PatternId> = (0..min_group_size)
                .map(|i| PatternId(hierarchy_level * 100 + i as u64))
                .collect();
            
            let candidate = ConsolidationCandidate {
                candidate_patterns: pattern_ids,
                similarity_score: 0.6,
                consolidation_benefit: 0.8,
                strategy: ConsolidationStrategy::HierarchicalGroup { min_group_size },
                estimated_savings: ConsolidationSavings {
                    memory_reduction: 2048,
                    access_time_improvement: 0.25,
                    generalization_value: 0.8,
                },
            };
            
            self.consolidation_candidates.push(candidate);
        }
        
        Ok(())
    }
    
    fn find_common_features(
        &mut self,
        _memory: &PathwayMemory,
        min_frequency: f32,
    ) -> Result<(), ConsolidationError> {
        // Simulate finding common feature patterns
        
        for feature_group in 0..3 {
            let pattern_ids: Vec<PatternId> = (0..6)
                .map(|i| PatternId(feature_group * 200 + i))
                .collect();
            
            let candidate = ConsolidationCandidate {
                candidate_patterns: pattern_ids,
                similarity_score: min_frequency + 0.1,
                consolidation_benefit: 0.6,
                strategy: ConsolidationStrategy::FeatureExtraction { min_frequency },
                estimated_savings: ConsolidationSavings {
                    memory_reduction: 1536,
                    access_time_improvement: 0.2,
                    generalization_value: 0.7,
                },
            };
            
            self.consolidation_candidates.push(candidate);
        }
        
        Ok(())
    }
    
    fn find_template_opportunities(
        &mut self,
        _memory: &PathwayMemory,
        generalization_level: f32,
    ) -> Result<(), ConsolidationError> {
        // Simulate finding template creation opportunities
        
        for template_type in 0..4 {
            let pattern_ids: Vec<PatternId> = (0..5)
                .map(|i| PatternId(template_type * 300 + i))
                .collect();
            
            let candidate = ConsolidationCandidate {
                candidate_patterns: pattern_ids,
                similarity_score: 0.7,
                consolidation_benefit: generalization_level,
                strategy: ConsolidationStrategy::TemplateCreation { generalization_level },
                estimated_savings: ConsolidationSavings {
                    memory_reduction: 3072,
                    access_time_improvement: 0.3,
                    generalization_value: generalization_level,
                },
            };
            
            self.consolidation_candidates.push(candidate);
        }
        
        Ok(())
    }
    
    fn adaptive_consolidation_search(
        &mut self,
        memory: &PathwayMemory,
    ) -> Result<(), ConsolidationError> {
        // Use multiple strategies adaptively
        self.find_similar_patterns(memory, self.similarity_threshold)?;
        self.find_hierarchical_groups(memory, self.min_consolidation_size)?;
        self.find_common_features(memory, 0.3)?;
        self.find_template_opportunities(memory, 0.7)?;
        
        Ok(())
    }
    
    fn should_consolidate_candidate(
        &self,
        candidate: &ConsolidationCandidate,
    ) -> Result<bool, ConsolidationError> {
        // Check if consolidation would be beneficial
        let min_benefit_threshold = 0.5;
        let min_similarity_threshold = 0.6;
        let min_group_size = 2;
        
        Ok(candidate.consolidation_benefit >= min_benefit_threshold &&
           candidate.similarity_score >= min_similarity_threshold &&
           candidate.candidate_patterns.len() >= min_group_size)
    }
    
    fn execute_consolidation(
        &mut self,
        candidate: &ConsolidationCandidate,
        _memory: &mut PathwayMemory,
    ) -> Result<ConsolidationResult, ConsolidationError> {
        let consolidated_pattern_id = ConsolidatedPatternId(self.next_pattern_id);
        self.next_pattern_id += 1;
        
        // Create consolidated pattern based on strategy
        let consolidated_pattern = match candidate.strategy {
            ConsolidationStrategy::SimilarityMerge { .. } => {
                self.create_merged_pattern(consolidated_pattern_id, &candidate.candidate_patterns)?
            },
            ConsolidationStrategy::HierarchicalGroup { .. } => {
                self.create_hierarchical_pattern(consolidated_pattern_id, &candidate.candidate_patterns)?
            },
            ConsolidationStrategy::FeatureExtraction { .. } => {
                self.create_feature_pattern(consolidated_pattern_id, &candidate.candidate_patterns)?
            },
            ConsolidationStrategy::TemplateCreation { .. } => {
                self.create_template_pattern(consolidated_pattern_id, &candidate.candidate_patterns)?
            },
            ConsolidationStrategy::AdaptiveConsolidation => {
                self.create_adaptive_pattern(consolidated_pattern_id, &candidate.candidate_patterns)?
            },
        };
        
        // Store consolidated pattern
        self.consolidated_patterns.insert(consolidated_pattern_id, consolidated_pattern);
        
        Ok(ConsolidationResult {
            consolidated_pattern_id,
            source_patterns: candidate.candidate_patterns.clone(),
            memory_saved: candidate.estimated_savings.memory_reduction,
            efficiency_improvement: candidate.estimated_savings.access_time_improvement,
            generalization_achieved: candidate.estimated_savings.generalization_value,
        })
    }
    
    fn create_merged_pattern(
        &self,
        pattern_id: ConsolidatedPatternId,
        source_patterns: &[PatternId],
    ) -> Result<ConsolidatedPattern, ConsolidationError> {
        // Create a pattern that merges common elements from source patterns
        let template_structure = PathwayTemplate {
            template_nodes: vec![
                TemplateNode {
                    node_type: NodeType::Variable { 
                        entity_type: "merged_entity".to_string(),
                        constraints: vec!["similar_activation".to_string()],
                    },
                    position_index: 0,
                    is_variable: true,
                    constraints: vec![],
                },
            ],
            template_edges: vec![],
            activation_pattern: vec![0.7, 0.5, 0.3],
            timing_constraints: vec![],
        };
        
        Ok(ConsolidatedPattern {
            pattern_id,
            template_structure,
            variable_components: vec![],
            fixed_components: vec![],
            consolidated_from: source_patterns.to_vec(),
            usage_contexts: vec![],
            success_distribution: SuccessDistribution {
                mean_efficiency: 0.75,
                efficiency_variance: 0.1,
                success_rate: 0.85,
                confidence_interval: (0.7, 0.8),
                sample_size: source_patterns.len(),
            },
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            consolidation_level: 1,
        })
    }
    
    fn create_hierarchical_pattern(
        &self,
        pattern_id: ConsolidatedPatternId,
        source_patterns: &[PatternId],
    ) -> Result<ConsolidatedPattern, ConsolidationError> {
        // Create a hierarchical pattern with parent-child relationships
        let template_structure = PathwayTemplate {
            template_nodes: vec![
                TemplateNode {
                    node_type: NodeType::Variable { 
                        entity_type: "parent_concept".to_string(),
                        constraints: vec!["hierarchical_relation".to_string()],
                    },
                    position_index: 0,
                    is_variable: true,
                    constraints: vec![],
                },
                TemplateNode {
                    node_type: NodeType::Variable { 
                        entity_type: "child_concept".to_string(),
                        constraints: vec!["related_to_parent".to_string()],
                    },
                    position_index: 1,
                    is_variable: true,
                    constraints: vec![],
                },
            ],
            template_edges: vec![
                TemplateEdge {
                    source_index: 0,
                    target_index: 1,
                    weight_range: (0.5, 1.0),
                    required: true,
                },
            ],
            activation_pattern: vec![0.8, 0.6],
            timing_constraints: vec![],
        };
        
        Ok(ConsolidatedPattern {
            pattern_id,
            template_structure,
            variable_components: vec![],
            fixed_components: vec![],
            consolidated_from: source_patterns.to_vec(),
            usage_contexts: vec![],
            success_distribution: SuccessDistribution {
                mean_efficiency: 0.8,
                efficiency_variance: 0.08,
                success_rate: 0.9,
                confidence_interval: (0.75, 0.85),
                sample_size: source_patterns.len(),
            },
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            consolidation_level: 2,
        })
    }
    
    fn create_feature_pattern(
        &self,
        pattern_id: ConsolidatedPatternId,
        source_patterns: &[PatternId],
    ) -> Result<ConsolidatedPattern, ConsolidationError> {
        // Extract common features into a reusable pattern
        let template_structure = PathwayTemplate {
            template_nodes: vec![
                TemplateNode {
                    node_type: NodeType::Wildcard { 
                        min_activation: 0.3,
                        max_activation: 1.0,
                    },
                    position_index: 0,
                    is_variable: true,
                    constraints: vec![],
                },
            ],
            template_edges: vec![],
            activation_pattern: vec![0.6],
            timing_constraints: vec![],
        };
        
        Ok(ConsolidatedPattern {
            pattern_id,
            template_structure,
            variable_components: vec![
                VariableComponent {
                    component_id: "common_feature".to_string(),
                    possible_values: vec![NodeId(1), NodeId(2), NodeId(3)],
                    selection_probability: {
                        let mut prob_map = HashMap::new();
                        prob_map.insert(NodeId(1), 0.4);
                        prob_map.insert(NodeId(2), 0.4);
                        prob_map.insert(NodeId(3), 0.2);
                        prob_map
                    },
                    context_dependencies: vec!["feature_context".to_string()],
                },
            ],
            fixed_components: vec![],
            consolidated_from: source_patterns.to_vec(),
            usage_contexts: vec![],
            success_distribution: SuccessDistribution {
                mean_efficiency: 0.7,
                efficiency_variance: 0.12,
                success_rate: 0.8,
                confidence_interval: (0.65, 0.75),
                sample_size: source_patterns.len(),
            },
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            consolidation_level: 1,
        })
    }
    
    fn create_template_pattern(
        &self,
        pattern_id: ConsolidatedPatternId,
        source_patterns: &[PatternId],
    ) -> Result<ConsolidatedPattern, ConsolidationError> {
        // Create a highly generalized template
        let template_structure = PathwayTemplate {
            template_nodes: vec![
                TemplateNode {
                    node_type: NodeType::Variable { 
                        entity_type: "template_entity".to_string(),
                        constraints: vec!["template_constraint".to_string()],
                    },
                    position_index: 0,
                    is_variable: true,
                    constraints: vec![
                        NodeConstraint {
                            constraint_type: "activation_range".to_string(),
                            constraint_value: "0.5-1.0".to_string(),
                            required: true,
                        },
                    ],
                },
            ],
            template_edges: vec![],
            activation_pattern: vec![0.75],
            timing_constraints: vec![],
        };
        
        Ok(ConsolidatedPattern {
            pattern_id,
            template_structure,
            variable_components: vec![],
            fixed_components: vec![
                FixedComponent {
                    component_id: "template_anchor".to_string(),
                    fixed_nodes: vec![NodeId(100)], // Template anchor node
                    required_activation: 0.5,
                    timing_requirements: Some(Duration::from_millis(100)),
                },
            ],
            consolidated_from: source_patterns.to_vec(),
            usage_contexts: vec![
                UsageContext {
                    context_type: "template_usage".to_string(),
                    context_features: {
                        let mut features = HashMap::new();
                        features.insert("generalization_level".to_string(), "high".to_string());
                        features
                    },
                    success_rate: 0.85,
                    usage_count: 0,
                },
            ],
            success_distribution: SuccessDistribution {
                mean_efficiency: 0.8,
                efficiency_variance: 0.05,
                success_rate: 0.85,
                confidence_interval: (0.78, 0.82),
                sample_size: source_patterns.len(),
            },
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            consolidation_level: 3,
        })
    }
    
    fn create_adaptive_pattern(
        &self,
        pattern_id: ConsolidatedPatternId,
        source_patterns: &[PatternId],
    ) -> Result<ConsolidatedPattern, ConsolidationError> {
        // Choose best strategy based on source patterns
        // For simplicity, use template creation for adaptive
        self.create_template_pattern(pattern_id, source_patterns)
    }
    
    fn update_metrics(
        &self,
        metrics: &mut ConsolidationMetrics,
        result: &ConsolidationResult,
    ) {
        metrics.patterns_consolidated += result.source_patterns.len();
        metrics.consolidated_patterns_created += 1;
        metrics.memory_efficiency_gain += result.memory_saved as f32;
        metrics.access_time_improvement += result.efficiency_improvement;
        metrics.generalization_quality += result.generalization_achieved;
    }
    
    fn record_consolidation_event(&mut self, metrics: &ConsolidationMetrics) {
        let event = ConsolidationEvent {
            timestamp: SystemTime::now(),
            strategy_used: self.consolidation_strategy,
            patterns_consolidated: metrics.patterns_consolidated,
            memory_saved: metrics.memory_efficiency_gain as usize,
            efficiency_improvement: metrics.access_time_improvement,
        };
        
        self.consolidation_history.push(event);
        
        // Keep history manageable
        if self.consolidation_history.len() > 1000 {
            self.consolidation_history.drain(..500);
        }
    }
    
    pub fn get_consolidated_pattern(&self, pattern_id: ConsolidatedPatternId) -> Option<&ConsolidatedPattern> {
        self.consolidated_patterns.get(&pattern_id)
    }
    
    pub fn get_consolidation_statistics(&self) -> ConsolidationStatistics {
        let total_events = self.consolidation_history.len();
        let total_patterns_consolidated = self.consolidation_history.iter()
            .map(|e| e.patterns_consolidated)
            .sum();
        let total_memory_saved = self.consolidation_history.iter()
            .map(|e| e.memory_saved)
            .sum();
        let average_efficiency_improvement = if total_events > 0 {
            self.consolidation_history.iter()
                .map(|e| e.efficiency_improvement)
                .sum::<f32>() / total_events as f32
        } else {
            0.0
        };
        
        ConsolidationStatistics {
            total_consolidation_events: total_events,
            total_patterns_consolidated,
            total_consolidated_patterns: self.consolidated_patterns.len(),
            total_memory_saved,
            average_efficiency_improvement,
            consolidation_levels: self.get_consolidation_level_distribution(),
        }
    }
    
    fn get_consolidation_level_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        
        for pattern in self.consolidated_patterns.values() {
            *distribution.entry(pattern.consolidation_level).or_insert(0) += 1;
        }
        
        distribution
    }
}

#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub consolidated_pattern_id: ConsolidatedPatternId,
    pub source_patterns: Vec<PatternId>,
    pub memory_saved: usize,
    pub efficiency_improvement: f32,
    pub generalization_achieved: f32,
}

#[derive(Debug, Clone)]
pub struct ConsolidationMetrics {
    pub patterns_consolidated: usize,
    pub consolidated_patterns_created: usize,
    pub memory_efficiency_gain: f32,
    pub access_time_improvement: f32,
    pub consolidation_time: Duration,
    pub generalization_quality: f32,
}

#[derive(Debug, Clone)]
pub struct ConsolidationStatistics {
    pub total_consolidation_events: usize,
    pub total_patterns_consolidated: usize,
    pub total_consolidated_patterns: usize,
    pub total_memory_saved: usize,
    pub average_efficiency_improvement: f32,
    pub consolidation_levels: HashMap<usize, usize>,
}

#[derive(Debug, Clone)]
pub enum ConsolidationError {
    InsufficientPatterns,
    ConsolidationFailed,
    InvalidStrategy,
    MemoryError,
}

impl std::fmt::Display for ConsolidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsolidationError::InsufficientPatterns => write!(f, "Insufficient patterns for consolidation"),
            ConsolidationError::ConsolidationFailed => write!(f, "Consolidation operation failed"),
            ConsolidationError::InvalidStrategy => write!(f, "Invalid consolidation strategy"),
            ConsolidationError::MemoryError => write!(f, "Memory error during consolidation"),
        }
    }
}

impl std::error::Error for ConsolidationError {}
```

## File Locations

- `src/cognitive/learning/pathway_consolidation.rs` - Main implementation
- `src/cognitive/learning/mod.rs` - Module exports
- `tests/cognitive/learning/pathway_consolidation_tests.rs` - Test implementation

## Success Criteria

- [ ] PathwayConsolidator struct compiles and runs
- [ ] Multiple consolidation strategies implemented
- [ ] Hierarchical pattern creation functional
- [ ] Memory efficiency measurably improved
- [ ] Pattern generalization maintains effectiveness
- [ ] Template creation enables reuse
- [ ] All tests pass:
  - Similarity-based consolidation
  - Hierarchical pattern formation
  - Template creation and reuse
  - Memory efficiency verification

## Test Requirements

```rust
#[test]
fn test_similarity_based_consolidation() {
    let strategy = ConsolidationStrategy::SimilarityMerge { threshold: 0.8 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    
    assert!(metrics.patterns_consolidated > 0);
    assert!(metrics.consolidated_patterns_created > 0);
    assert!(metrics.memory_efficiency_gain > 0.0);
}

#[test]
fn test_hierarchical_grouping() {
    let strategy = ConsolidationStrategy::HierarchicalGroup { min_group_size: 3 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    
    assert!(metrics.consolidated_patterns_created > 0);
    assert!(metrics.generalization_quality > 0.0);
    
    let stats = consolidator.get_consolidation_statistics();
    assert!(stats.consolidation_levels.contains_key(&2)); // Hierarchical level
}

#[test]
fn test_template_creation() {
    let strategy = ConsolidationStrategy::TemplateCreation { generalization_level: 0.7 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    
    assert!(metrics.consolidated_patterns_created > 0);
    assert!(metrics.generalization_quality >= 0.7);
    
    // Check that templates were created
    let stats = consolidator.get_consolidation_statistics();
    assert!(stats.total_consolidated_patterns > 0);
}

#[test]
fn test_adaptive_consolidation() {
    let strategy = ConsolidationStrategy::AdaptiveConsolidation;
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    
    // Adaptive should use multiple strategies
    assert!(metrics.patterns_consolidated > 0);
    assert!(metrics.consolidation_time.as_millis() > 0);
}

#[test]
fn test_consolidation_metrics() {
    let strategy = ConsolidationStrategy::SimilarityMerge { threshold: 0.7 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    // Perform multiple consolidations
    for _ in 0..3 {
        consolidator.consolidate_pathways(&mut memory).unwrap();
    }
    
    let stats = consolidator.get_consolidation_statistics();
    
    assert_eq!(stats.total_consolidation_events, 3);
    assert!(stats.total_patterns_consolidated > 0);
    assert!(stats.average_efficiency_improvement >= 0.0);
}

#[test]
fn test_consolidated_pattern_retrieval() {
    let strategy = ConsolidationStrategy::TemplateCreation { generalization_level: 0.8 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    consolidator.consolidate_pathways(&mut memory).unwrap();
    
    // Check that consolidated patterns can be retrieved
    for (pattern_id, _) in &consolidator.consolidated_patterns {
        let retrieved_pattern = consolidator.get_consolidated_pattern(*pattern_id);
        assert!(retrieved_pattern.is_some());
        assert!(retrieved_pattern.unwrap().consolidation_level > 0);
    }
}

#[test]
fn test_feature_extraction() {
    let strategy = ConsolidationStrategy::FeatureExtraction { min_frequency: 0.3 };
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    
    assert!(metrics.patterns_consolidated > 0);
    
    // Check that variable components were created
    for pattern in consolidator.consolidated_patterns.values() {
        if !pattern.variable_components.is_empty() {
            assert!(!pattern.variable_components[0].possible_values.is_empty());
        }
    }
}

#[test]
fn test_consolidation_efficiency() {
    let strategy = ConsolidationStrategy::AdaptiveConsolidation;
    let mut consolidator = PathwayConsolidator::new(strategy);
    let mut memory = PathwayMemory::new();
    
    let start_time = Instant::now();
    let metrics = consolidator.consolidate_pathways(&mut memory).unwrap();
    let elapsed = start_time.elapsed();
    
    // Should complete within reasonable time
    assert!(elapsed.as_millis() < 1000);
    assert!(metrics.access_time_improvement >= 0.0);
}
```

## Quality Gates

- [ ] Consolidation completes within performance targets
- [ ] Memory efficiency measurably improved
- [ ] Pattern generalization maintains accuracy
- [ ] Template reuse reduces redundancy
- [ ] Hierarchical structures properly formed
- [ ] No loss of critical pathway information

## Next Task

Upon completion, proceed to **24_pathway_tests.md**