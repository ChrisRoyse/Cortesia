//! Memory hierarchy management for unified memory system

use super::types::*;
use std::collections::HashMap;
use std::time::Duration;

/// Memory hierarchy structure and management
#[derive(Debug, Clone)]
pub struct MemoryHierarchy {
    pub levels: Vec<MemoryLevel>,
    pub transition_thresholds: TransitionThresholds,
    pub consolidation_rules: Vec<ConsolidationRule>,
}

impl MemoryHierarchy {
    /// Create new memory hierarchy
    pub fn new() -> Self {
        Self {
            levels: Self::create_default_levels(),
            transition_thresholds: TransitionThresholds::default(),
            consolidation_rules: Self::create_default_consolidation_rules(),
        }
    }
    
    /// Create default memory levels
    fn create_default_levels() -> Vec<MemoryLevel> {
        vec![
            MemoryLevel {
                level_id: "sensory_buffer".to_string(),
                memory_type: MemoryType::SensoryBuffer,
                capacity: MemoryCapacity {
                    max_items: 10,
                    max_size_bytes: 1024,
                    utilization_threshold: 0.8,
                    overflow_strategy: OverflowStrategy::LeastRecentlyUsed,
                },
                retention_period: Duration::from_millis(500),
                access_speed: AccessSpeed::Immediate,
                stability: 0.1,
            },
            MemoryLevel {
                level_id: "working_memory".to_string(),
                memory_type: MemoryType::WorkingMemory,
                capacity: MemoryCapacity {
                    max_items: 7,
                    max_size_bytes: 8192,
                    utilization_threshold: 0.85,
                    overflow_strategy: OverflowStrategy::ImportanceBased,
                },
                retention_period: Duration::from_secs(30),
                access_speed: AccessSpeed::Fast,
                stability: 0.3,
            },
            MemoryLevel {
                level_id: "short_term_memory".to_string(),
                memory_type: MemoryType::ShortTermMemory,
                capacity: MemoryCapacity {
                    max_items: 100,
                    max_size_bytes: 65536,
                    utilization_threshold: 0.75,
                    overflow_strategy: OverflowStrategy::ForgettingCurve,
                },
                retention_period: Duration::from_secs(600),
                access_speed: AccessSpeed::Medium,
                stability: 0.5,
            },
            MemoryLevel {
                level_id: "long_term_memory".to_string(),
                memory_type: MemoryType::LongTermMemory,
                capacity: MemoryCapacity {
                    max_items: 1000000,
                    max_size_bytes: 1073741824, // 1GB
                    utilization_threshold: 0.9,
                    overflow_strategy: OverflowStrategy::ImportanceBased,
                },
                retention_period: Duration::from_secs(86400 * 365), // 1 year
                access_speed: AccessSpeed::Slow,
                stability: 0.8,
            },
            MemoryLevel {
                level_id: "semantic_memory".to_string(),
                memory_type: MemoryType::SemanticMemory,
                capacity: MemoryCapacity {
                    max_items: 500000,
                    max_size_bytes: 536870912, // 512MB
                    utilization_threshold: 0.85,
                    overflow_strategy: OverflowStrategy::ImportanceBased,
                },
                retention_period: Duration::from_secs(86400 * 365 * 10), // 10 years
                access_speed: AccessSpeed::Medium,
                stability: 0.9,
            },
            MemoryLevel {
                level_id: "episodic_memory".to_string(),
                memory_type: MemoryType::EpisodicMemory,
                capacity: MemoryCapacity {
                    max_items: 100000,
                    max_size_bytes: 268435456, // 256MB
                    utilization_threshold: 0.8,
                    overflow_strategy: OverflowStrategy::ForgettingCurve,
                },
                retention_period: Duration::from_secs(86400 * 365 * 5), // 5 years
                access_speed: AccessSpeed::Slow,
                stability: 0.7,
            },
            MemoryLevel {
                level_id: "procedural_memory".to_string(),
                memory_type: MemoryType::ProceduralMemory,
                capacity: MemoryCapacity {
                    max_items: 10000,
                    max_size_bytes: 134217728, // 128MB
                    utilization_threshold: 0.9,
                    overflow_strategy: OverflowStrategy::ImportanceBased,
                },
                retention_period: Duration::from_secs(86400 * 365 * 20), // 20 years
                access_speed: AccessSpeed::Fast,
                stability: 0.95,
            },
        ]
    }
    
    /// Create default consolidation rules
    fn create_default_consolidation_rules() -> Vec<ConsolidationRule> {
        vec![
            ConsolidationRule {
                rule_id: "working_to_short_term".to_string(),
                source_memory: MemoryType::WorkingMemory,
                target_memory: MemoryType::ShortTermMemory,
                conditions: vec![
                    ConsolidationCondition::AccessCountThreshold(3),
                    ConsolidationCondition::ImportanceThreshold(0.6),
                ],
                consolidation_strength: 0.7,
            },
            ConsolidationRule {
                rule_id: "short_term_to_long_term".to_string(),
                source_memory: MemoryType::ShortTermMemory,
                target_memory: MemoryType::LongTermMemory,
                conditions: vec![
                    ConsolidationCondition::AccessCountThreshold(5),
                    ConsolidationCondition::TimeThreshold(Duration::from_secs(600)),
                    ConsolidationCondition::ImportanceThreshold(0.7),
                ],
                consolidation_strength: 0.8,
            },
            ConsolidationRule {
                rule_id: "episodic_to_semantic".to_string(),
                source_memory: MemoryType::EpisodicMemory,
                target_memory: MemoryType::SemanticMemory,
                conditions: vec![
                    ConsolidationCondition::AccessCountThreshold(10),
                    ConsolidationCondition::TimeThreshold(Duration::from_secs(86400)),
                    ConsolidationCondition::ContextualRelevance(0.8),
                ],
                consolidation_strength: 0.9,
            },
        ]
    }
    
    /// Get memory level by type
    pub fn get_level(&self, memory_type: &MemoryType) -> Option<&MemoryLevel> {
        self.levels.iter().find(|level| level.memory_type == *memory_type)
    }
    
    /// Get memory level by ID
    pub fn get_level_by_id(&self, level_id: &str) -> Option<&MemoryLevel> {
        self.levels.iter().find(|level| level.level_id == level_id)
    }
    
    /// Check if consolidation should occur
    pub fn should_consolidate(&self, source_memory: &MemoryType, item_stats: &ItemStats) -> Option<ConsolidationRule> {
        for rule in &self.consolidation_rules {
            if rule.source_memory == *source_memory
                && self.evaluate_consolidation_conditions(&rule.conditions, item_stats) {
                    return Some(rule.clone());
                }
        }
        None
    }
    
    /// Evaluate consolidation conditions
    fn evaluate_consolidation_conditions(&self, conditions: &[ConsolidationCondition], item_stats: &ItemStats) -> bool {
        for condition in conditions {
            match condition {
                ConsolidationCondition::AccessCountThreshold(threshold) => {
                    if item_stats.access_count < *threshold {
                        return false;
                    }
                }
                ConsolidationCondition::ImportanceThreshold(threshold) => {
                    if item_stats.importance_score < *threshold {
                        return false;
                    }
                }
                ConsolidationCondition::TimeThreshold(threshold) => {
                    if item_stats.age < *threshold {
                        return false;
                    }
                }
                ConsolidationCondition::RehearsalRequired => {
                    if !item_stats.rehearsed {
                        return false;
                    }
                }
                ConsolidationCondition::ContextualRelevance(threshold) => {
                    if item_stats.contextual_relevance < *threshold {
                        return false;
                    }
                }
            }
        }
        true
    }
    
    /// Get consolidation path from source to target
    pub fn get_consolidation_path(&self, source: &MemoryType, target: &MemoryType) -> Vec<MemoryType> {
        // Define consolidation hierarchy
        let hierarchy_order = [
            MemoryType::SensoryBuffer,
            MemoryType::WorkingMemory,
            MemoryType::ShortTermMemory,
            MemoryType::LongTermMemory,
            MemoryType::SemanticMemory,
            MemoryType::EpisodicMemory,
            MemoryType::ProceduralMemory,
        ];
        
        let source_idx = hierarchy_order.iter().position(|t| t == source);
        let target_idx = hierarchy_order.iter().position(|t| t == target);
        
        if let (Some(start), Some(end)) = (source_idx, target_idx) {
            if start < end {
                hierarchy_order[start..=end].to_vec()
            } else {
                vec![source.clone(), target.clone()]
            }
        } else {
            vec![source.clone(), target.clone()]
        }
    }
    
    /// Calculate consolidation priority
    pub fn calculate_consolidation_priority(&self, rule: &ConsolidationRule, item_stats: &ItemStats) -> f32 {
        let mut priority = rule.consolidation_strength;
        
        // Boost priority based on access count
        priority *= 1.0 + (item_stats.access_count as f32 * 0.1);
        
        // Boost priority based on importance
        priority *= 1.0 + item_stats.importance_score;
        
        // Boost priority based on age (older items more likely to consolidate)
        let age_factor = item_stats.age.as_secs() as f32 / 86400.0; // Days
        priority *= 1.0 + (age_factor * 0.05);
        
        priority.min(1.0)
    }
    
    /// Get memory level statistics
    pub fn get_level_statistics(&self) -> HashMap<String, LevelStatistics> {
        let mut stats = HashMap::new();
        
        for level in &self.levels {
            stats.insert(level.level_id.clone(), LevelStatistics {
                memory_type: level.memory_type.clone(),
                capacity_utilization: 0.6, // Would be calculated from actual usage
                average_access_time: level.access_speed.clone(),
                stability_score: level.stability,
                consolidation_rate: 0.1, // Would be calculated from actual consolidation events
            });
        }
        
        stats
    }
    
    /// Optimize memory hierarchy
    pub fn optimize_hierarchy(&mut self, performance_data: &PerformanceAnalysis) {
        // Adjust capacity based on utilization
        for level in &mut self.levels {
            if let Some(&utilization) = performance_data.memory_utilization.get(&level.memory_type) {
                if utilization > 0.9 {
                    // Increase capacity if highly utilized
                    level.capacity.max_items = (level.capacity.max_items as f32 * 1.2) as usize;
                    level.capacity.max_size_bytes = (level.capacity.max_size_bytes as f32 * 1.2) as usize;
                } else if utilization < 0.3 {
                    // Decrease capacity if underutilized
                    level.capacity.max_items = (level.capacity.max_items as f32 * 0.9) as usize;
                    level.capacity.max_size_bytes = (level.capacity.max_size_bytes as f32 * 0.9) as usize;
                }
            }
        }
        
        // Adjust consolidation rules based on performance
        for rule in &mut self.consolidation_rules {
            // Make consolidation more aggressive if memory is constrained
            if performance_data.bottlenecks.iter().any(|b| b.bottleneck_type == BottleneckType::HighMemoryUsage) {
                rule.consolidation_strength *= 1.1;
            }
        }
    }
}

/// Statistics for an individual memory item
#[derive(Debug, Clone)]
pub struct ItemStats {
    pub access_count: u32,
    pub importance_score: f32,
    pub age: Duration,
    pub rehearsed: bool,
    pub contextual_relevance: f32,
}

/// Statistics for a memory level
#[derive(Debug, Clone)]
pub struct LevelStatistics {
    pub memory_type: MemoryType,
    pub capacity_utilization: f32,
    pub average_access_time: AccessSpeed,
    pub stability_score: f32,
    pub consolidation_rate: f32,
}

impl Default for TransitionThresholds {
    fn default() -> Self {
        Self {
            working_to_short_term: TransitionCriteria {
                access_count_threshold: 2,
                importance_threshold: 0.5,
                time_threshold: Duration::from_secs(10),
                rehearsal_requirement: false,
            },
            short_term_to_long_term: TransitionCriteria {
                access_count_threshold: 5,
                importance_threshold: 0.7,
                time_threshold: Duration::from_secs(300),
                rehearsal_requirement: true,
            },
            episodic_to_semantic: TransitionCriteria {
                access_count_threshold: 10,
                importance_threshold: 0.8,
                time_threshold: Duration::from_secs(3600),
                rehearsal_requirement: false,
            },
            consolidation_threshold: 0.6,
        }
    }
}

impl Default for MemoryHierarchy {
    fn default() -> Self {
        Self::new()
    }
}