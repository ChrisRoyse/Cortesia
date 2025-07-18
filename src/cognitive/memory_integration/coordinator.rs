//! Memory coordination and strategy management

use super::types::*;
use super::hierarchy::MemoryHierarchy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Memory coordinator for unified memory system
#[derive(Debug, Clone)]
pub struct MemoryCoordinator {
    pub retrieval_strategies: Vec<RetrievalStrategy>,
    pub consolidation_policies: Vec<ConsolidationPolicy>,
    pub memory_hierarchy: MemoryHierarchy,
    pub cross_memory_links: Arc<RwLock<HashMap<String, Vec<CrossMemoryLink>>>>,
}

impl MemoryCoordinator {
    /// Create new memory coordinator
    pub fn new() -> Self {
        Self {
            retrieval_strategies: Self::create_default_strategies(),
            consolidation_policies: Self::create_default_policies(),
            memory_hierarchy: MemoryHierarchy::new(),
            cross_memory_links: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create default retrieval strategies
    fn create_default_strategies() -> Vec<RetrievalStrategy> {
        vec![
            RetrievalStrategy {
                strategy_id: "parallel_comprehensive".to_string(),
                strategy_type: RetrievalType::ParallelSearch,
                memory_priority: vec![
                    MemoryType::WorkingMemory,
                    MemoryType::SemanticMemory,
                    MemoryType::LongTermMemory,
                ],
                fusion_method: FusionMethod::WeightedAverage,
                confidence_weighting: ConfidenceWeighting::default(),
            },
            RetrievalStrategy {
                strategy_id: "hierarchical_efficient".to_string(),
                strategy_type: RetrievalType::HierarchicalSearch,
                memory_priority: vec![
                    MemoryType::WorkingMemory,
                    MemoryType::ShortTermMemory,
                    MemoryType::LongTermMemory,
                ],
                fusion_method: FusionMethod::MaximumConfidence,
                confidence_weighting: ConfidenceWeighting::default(),
            },
            RetrievalStrategy {
                strategy_id: "contextual_adaptive".to_string(),
                strategy_type: RetrievalType::ContextualSearch,
                memory_priority: vec![
                    MemoryType::EpisodicMemory,
                    MemoryType::SemanticMemory,
                    MemoryType::WorkingMemory,
                ],
                fusion_method: FusionMethod::ContextualFusion,
                confidence_weighting: ConfidenceWeighting {
                    working_memory_weight: 0.3,
                    sdr_storage_weight: 0.4,
                    graph_storage_weight: 0.3,
                    recency_factor: 0.3,
                    importance_factor: 0.4,
                },
            },
            RetrievalStrategy {
                strategy_id: "fast_lookup".to_string(),
                strategy_type: RetrievalType::HierarchicalSearch,
                memory_priority: vec![
                    MemoryType::WorkingMemory,
                    MemoryType::ProceduralMemory,
                ],
                fusion_method: FusionMethod::MaximumConfidence,
                confidence_weighting: ConfidenceWeighting {
                    working_memory_weight: 0.6,
                    sdr_storage_weight: 0.2,
                    graph_storage_weight: 0.2,
                    recency_factor: 0.1,
                    importance_factor: 0.2,
                },
            },
        ]
    }
    
    /// Create default consolidation policies
    fn create_default_policies() -> Vec<ConsolidationPolicy> {
        vec![
            ConsolidationPolicy {
                policy_id: "time_based_consolidation".to_string(),
                trigger_conditions: vec![
                    ConsolidationTrigger::TimeBasedTrigger(std::time::Duration::from_secs(300)),
                ],
                consolidation_rules: vec![
                    ConsolidationRule {
                        rule_id: "working_to_short_term_time".to_string(),
                        source_memory: MemoryType::WorkingMemory,
                        target_memory: MemoryType::ShortTermMemory,
                        conditions: vec![
                            ConsolidationCondition::TimeThreshold(std::time::Duration::from_secs(30)),
                            ConsolidationCondition::AccessCountThreshold(2),
                        ],
                        consolidation_strength: 0.6,
                    },
                ],
                priority: 0.8,
            },
            ConsolidationPolicy {
                policy_id: "importance_based_consolidation".to_string(),
                trigger_conditions: vec![
                    ConsolidationTrigger::ImportanceBasedTrigger(0.8),
                ],
                consolidation_rules: vec![
                    ConsolidationRule {
                        rule_id: "important_to_long_term".to_string(),
                        source_memory: MemoryType::ShortTermMemory,
                        target_memory: MemoryType::LongTermMemory,
                        conditions: vec![
                            ConsolidationCondition::ImportanceThreshold(0.8),
                            ConsolidationCondition::AccessCountThreshold(3),
                        ],
                        consolidation_strength: 0.9,
                    },
                ],
                priority: 0.9,
            },
            ConsolidationPolicy {
                policy_id: "capacity_based_consolidation".to_string(),
                trigger_conditions: vec![
                    ConsolidationTrigger::CapacityBasedTrigger(0.85),
                ],
                consolidation_rules: vec![
                    ConsolidationRule {
                        rule_id: "capacity_pressure_consolidation".to_string(),
                        source_memory: MemoryType::WorkingMemory,
                        target_memory: MemoryType::ShortTermMemory,
                        conditions: vec![
                            ConsolidationCondition::AccessCountThreshold(1),
                        ],
                        consolidation_strength: 0.7,
                    },
                ],
                priority: 0.95,
            },
        ]
    }
    
    /// Get retrieval strategy by ID
    pub fn get_strategy(&self, strategy_id: &str) -> Option<&RetrievalStrategy> {
        self.retrieval_strategies.iter().find(|s| s.strategy_id == strategy_id)
    }
    
    /// Get best retrieval strategy for context
    pub fn get_best_strategy(&self, context: &str) -> &RetrievalStrategy {
        // Simple heuristic - in practice would be more sophisticated
        if context.contains("fast") || context.contains("quick") {
            self.get_strategy("fast_lookup").unwrap_or(&self.retrieval_strategies[0])
        } else if context.contains("comprehensive") || context.contains("thorough") {
            self.get_strategy("parallel_comprehensive").unwrap_or(&self.retrieval_strategies[0])
        } else if context.contains("contextual") || context.contains("episodic") {
            self.get_strategy("contextual_adaptive").unwrap_or(&self.retrieval_strategies[0])
        } else {
            self.get_strategy("hierarchical_efficient").unwrap_or(&self.retrieval_strategies[0])
        }
    }
    
    /// Add custom retrieval strategy
    pub fn add_strategy(&mut self, strategy: RetrievalStrategy) {
        self.retrieval_strategies.push(strategy);
    }
    
    /// Get consolidation policy by ID
    pub fn get_policy(&self, policy_id: &str) -> Option<&ConsolidationPolicy> {
        self.consolidation_policies.iter().find(|p| p.policy_id == policy_id)
    }
    
    /// Get active consolidation policies
    pub fn get_active_policies(&self) -> Vec<&ConsolidationPolicy> {
        // Sort by priority and return active policies
        let mut policies: Vec<&ConsolidationPolicy> = self.consolidation_policies.iter().collect();
        policies.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        policies
    }
    
    /// Add consolidation policy
    pub fn add_policy(&mut self, policy: ConsolidationPolicy) {
        self.consolidation_policies.push(policy);
    }
    
    /// Create cross-memory link
    pub async fn create_cross_memory_link(&self, link: CrossMemoryLink) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut links = self.cross_memory_links.write().await;
        links.entry(link.link_id.clone()).or_insert_with(Vec::new).push(link);
        Ok(())
    }
    
    /// Get cross-memory links for item
    pub async fn get_cross_memory_links(&self, item_id: &str) -> Vec<CrossMemoryLink> {
        let links = self.cross_memory_links.read().await;
        links.get(item_id).cloned().unwrap_or_default()
    }
    
    /// Update cross-memory link strength
    pub async fn update_link_strength(&self, link_id: &str, new_strength: f32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut links = self.cross_memory_links.write().await;
        
        for link_vec in links.values_mut() {
            for link in link_vec.iter_mut() {
                if link.link_id == link_id {
                    link.link_strength = new_strength;
                    return Ok(());
                }
            }
        }
        
        Err("Link not found".into())
    }
    
    /// Prune weak cross-memory links
    pub async fn prune_weak_links(&self, threshold: f32) -> usize {
        let mut links = self.cross_memory_links.write().await;
        let mut pruned_count = 0;
        
        for link_vec in links.values_mut() {
            let original_len = link_vec.len();
            link_vec.retain(|link| link.link_strength >= threshold);
            pruned_count += original_len - link_vec.len();
        }
        
        // Remove empty entries
        links.retain(|_, link_vec| !link_vec.is_empty());
        
        pruned_count
    }
    
    /// Get memory hierarchy statistics
    pub fn get_hierarchy_stats(&self) -> HashMap<String, super::hierarchy::LevelStatistics> {
        self.memory_hierarchy.get_level_statistics()
    }
    
    /// Optimize coordination strategies
    pub fn optimize_strategies(&mut self, performance_data: &PerformanceAnalysis) {
        // Adjust strategy weights based on performance
        for strategy in &mut self.retrieval_strategies {
            if performance_data.average_retrieval_time > std::time::Duration::from_millis(100) {
                // Favor faster strategies
                if strategy.strategy_id == "fast_lookup" {
                    strategy.confidence_weighting.working_memory_weight += 0.1;
                }
            }
            
            if performance_data.success_rate < 0.8 {
                // Favor more comprehensive strategies
                if strategy.strategy_id == "parallel_comprehensive" {
                    strategy.confidence_weighting.importance_factor += 0.1;
                }
            }
        }
        
        // Adjust consolidation policies based on memory pressure
        for policy in &mut self.consolidation_policies {
            if performance_data.bottlenecks.iter().any(|b| b.bottleneck_type == BottleneckType::HighMemoryUsage) {
                // Make consolidation more aggressive
                policy.priority *= 1.1;
                for rule in &mut policy.consolidation_rules {
                    rule.consolidation_strength *= 1.1;
                }
            }
        }
    }
    
    /// Generate coordinator report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("Memory Coordinator Report\n");
        report.push_str("========================\n\n");
        
        report.push_str(&format!("Retrieval Strategies: {}\n", self.retrieval_strategies.len()));
        for strategy in &self.retrieval_strategies {
            report.push_str(&format!("  - {}: {:?}\n", strategy.strategy_id, strategy.strategy_type));
        }
        
        report.push_str(&format!("\nConsolidation Policies: {}\n", self.consolidation_policies.len()));
        for policy in &self.consolidation_policies {
            report.push_str(&format!("  - {}: priority {:.2}\n", policy.policy_id, policy.priority));
        }
        
        report.push_str(&format!("\nMemory Hierarchy Levels: {}\n", self.memory_hierarchy.levels.len()));
        for level in &self.memory_hierarchy.levels {
            report.push_str(&format!("  - {}: {:?}\n", level.level_id, level.memory_type));
        }
        
        report.push_str(&format!("\nConsolidation Rules: {}\n", self.memory_hierarchy.consolidation_rules.len()));
        for rule in &self.memory_hierarchy.consolidation_rules {
            report.push_str(&format!("  - {}: {:?} -> {:?}\n", 
                rule.rule_id, rule.source_memory, rule.target_memory));
        }
        
        report
    }
    
    /// Validate coordinator configuration
    pub fn validate_configuration(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        // Check for duplicate strategy IDs
        let mut strategy_ids = std::collections::HashSet::new();
        for strategy in &self.retrieval_strategies {
            if !strategy_ids.insert(&strategy.strategy_id) {
                issues.push(format!("Duplicate strategy ID: {}", strategy.strategy_id));
            }
        }
        
        // Check for duplicate policy IDs
        let mut policy_ids = std::collections::HashSet::new();
        for policy in &self.consolidation_policies {
            if !policy_ids.insert(&policy.policy_id) {
                issues.push(format!("Duplicate policy ID: {}", policy.policy_id));
            }
        }
        
        // Check for empty memory priority lists
        for strategy in &self.retrieval_strategies {
            if strategy.memory_priority.is_empty() {
                issues.push(format!("Empty memory priority list in strategy: {}", strategy.strategy_id));
            }
        }
        
        // Check consolidation rule consistency
        for rule in &self.memory_hierarchy.consolidation_rules {
            if rule.source_memory == rule.target_memory {
                issues.push(format!("Self-referential consolidation rule: {}", rule.rule_id));
            }
        }
        
        issues
    }
}

impl Default for MemoryCoordinator {
    fn default() -> Self {
        Self::new()
    }
}