//! Memory consolidation operations for unified memory system

use super::types::*;
use super::coordinator::MemoryCoordinator;
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryItem};
use crate::core::sdr_storage::SDRStorage;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::{Result, GraphError};
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Memory consolidation handler
pub struct MemoryConsolidation {
    pub working_memory: Arc<WorkingMemorySystem>,
    pub sdr_storage: Arc<SDRStorage>,
    pub long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub coordinator: Arc<MemoryCoordinator>,
    pub memory_statistics: Arc<RwLock<MemoryStatistics>>,
}

impl MemoryConsolidation {
    /// Create new memory consolidation handler
    pub fn new(
        working_memory: Arc<WorkingMemorySystem>,
        sdr_storage: Arc<SDRStorage>,
        long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
        coordinator: Arc<MemoryCoordinator>,
        memory_statistics: Arc<RwLock<MemoryStatistics>>,
    ) -> Self {
        Self {
            working_memory,
            sdr_storage,
            long_term_graph,
            coordinator,
            memory_statistics,
        }
    }

    /// Perform memory consolidation
    pub async fn perform_consolidation(&self, policy_id: Option<&str>) -> Result<ConsolidationResult> {
        let start_time = Instant::now();
        let mut consolidated_items = Vec::new();
        let mut success_count = 0;
        let mut total_count = 0;

        // Get active consolidation policies
        let policies = if let Some(id) = policy_id {
            vec![self.coordinator.get_policy(id).ok_or_else(|| GraphError::ConfigError(format!("Policy not found: {}", id)))?]
        } else {
            self.coordinator.get_active_policies()
        };

        // Execute consolidation for each policy
        for policy in policies {
            let policy_result = self.execute_consolidation_policy(policy).await?;
            total_count += policy_result.consolidated_items.len();
            success_count += policy_result.consolidated_items.len();
            consolidated_items.extend(policy_result.consolidated_items);
        }

        // Update statistics
        let mut stats = self.memory_statistics.write().await;
        stats.record_consolidation();

        let consolidation_time = start_time.elapsed();
        let success_rate = if total_count > 0 {
            success_count as f32 / total_count as f32
        } else {
            1.0
        };

        Ok(ConsolidationResult {
            consolidated_items,
            consolidation_time,
            success_rate,
        })
    }

    /// Execute consolidation policy
    async fn execute_consolidation_policy(&self, policy: &ConsolidationPolicy) -> Result<ConsolidationResult> {
        let start_time = Instant::now();
        let mut consolidated_items = Vec::new();

        // Check if policy should be triggered
        if !self.should_trigger_policy(policy).await? {
            return Ok(ConsolidationResult {
                consolidated_items,
                consolidation_time: start_time.elapsed(),
                success_rate: 1.0,
            });
        }

        // Execute consolidation rules
        for rule in &policy.consolidation_rules {
            let rule_result = self.execute_consolidation_rule(rule).await?;
            consolidated_items.extend(rule_result);
        }

        Ok(ConsolidationResult {
            consolidated_items,
            consolidation_time: start_time.elapsed(),
            success_rate: 1.0,
        })
    }

    /// Check if policy should be triggered
    async fn should_trigger_policy(&self, policy: &ConsolidationPolicy) -> Result<bool> {
        for trigger in &policy.trigger_conditions {
            if self.evaluate_trigger(trigger).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Evaluate consolidation trigger
    async fn evaluate_trigger(&self, trigger: &ConsolidationTrigger) -> Result<bool> {
        match trigger {
            ConsolidationTrigger::TimeBasedTrigger(duration) => {
                // Check if enough time has passed since last consolidation
                // This is a simplified check - in practice would track last consolidation time
                Ok(true)
            }
            ConsolidationTrigger::UsageBasedTrigger(threshold) => {
                // Check if usage threshold has been reached
                let stats = self.memory_statistics.read().await;
                Ok(stats.total_retrievals >= *threshold as u64)
            }
            ConsolidationTrigger::ImportanceBasedTrigger(threshold) => {
                // Check if high-importance items are present
                // This is a simplified check - in practice would analyze actual items
                Ok(*threshold > 0.5)
            }
            ConsolidationTrigger::CapacityBasedTrigger(threshold) => {
                // Check if memory capacity threshold has been reached
                let stats = self.memory_statistics.read().await;
                let avg_utilization = stats.memory_utilization.values().sum::<f32>() / stats.memory_utilization.len() as f32;
                Ok(avg_utilization >= *threshold)
            }
            ConsolidationTrigger::ContextualTrigger(_context) => {
                // Check contextual trigger conditions
                Ok(true)
            }
        }
    }

    /// Execute consolidation rule
    async fn execute_consolidation_rule(&self, rule: &ConsolidationRule) -> Result<Vec<ConsolidatedItem>> {
        let mut consolidated_items = Vec::new();

        // Get consolidation candidates
        let candidates = self.get_consolidation_candidates(rule).await?;

        // Process each candidate
        for candidate in candidates {
            if let Some(consolidated_item) = self.consolidate_item(&candidate, rule).await? {
                consolidated_items.push(consolidated_item);
            }
        }

        Ok(consolidated_items)
    }

    /// Get consolidation candidates
    async fn get_consolidation_candidates(&self, rule: &ConsolidationRule) -> Result<Vec<ConsolidationCandidate>> {
        let mut candidates = Vec::new();

        // Get items from source memory based on rule
        match rule.source_memory {
            MemoryType::WorkingMemory => {
                let items = self.working_memory.get_all_items().await?;
                for item in items {
                    if self.should_consolidate_item(&item, rule).await? {
                        candidates.push(ConsolidationCandidate {
                            item_id: format!("working_memory_item_{}", item.timestamp.elapsed().as_millis()),
                            current_memory: MemoryType::WorkingMemory,
                            proposed_memory: rule.target_memory.clone(),
                            consolidation_score: item.importance_score,
                        });
                    }
                }
            }
            MemoryType::ShortTermMemory => {
                // In practice, would query short-term memory storage
                // For now, create placeholder candidates
                candidates.push(ConsolidationCandidate {
                    item_id: "short_term_item_1".to_string(),
                    current_memory: MemoryType::ShortTermMemory,
                    proposed_memory: rule.target_memory.clone(),
                    consolidation_score: 0.7,
                });
            }
            MemoryType::EpisodicMemory => {
                // In practice, would query episodic memory storage
                candidates.push(ConsolidationCandidate {
                    item_id: "episodic_item_1".to_string(),
                    current_memory: MemoryType::EpisodicMemory,
                    proposed_memory: rule.target_memory.clone(),
                    consolidation_score: 0.8,
                });
            }
            _ => {
                // Other memory types - would implement specific logic
            }
        }

        Ok(candidates)
    }

    /// Check if item should be consolidated
    async fn should_consolidate_item(&self, item: &MemoryItem, rule: &ConsolidationRule) -> Result<bool> {
        // Create item statistics for evaluation
        let item_stats = super::hierarchy::ItemStats {
            access_count: item.access_count,
            importance_score: item.importance_score,
            age: item.timestamp.elapsed(),
            rehearsed: item.access_count > 1,
            contextual_relevance: 0.7, // Would be calculated based on actual context
        };

        // Check consolidation conditions
        for condition in &rule.conditions {
            match condition {
                ConsolidationCondition::AccessCountThreshold(threshold) => {
                    if item_stats.access_count < *threshold {
                        return Ok(false);
                    }
                }
                ConsolidationCondition::ImportanceThreshold(threshold) => {
                    if item_stats.importance_score < *threshold {
                        return Ok(false);
                    }
                }
                ConsolidationCondition::TimeThreshold(threshold) => {
                    if item_stats.age < *threshold {
                        return Ok(false);
                    }
                }
                ConsolidationCondition::RehearsalRequired => {
                    if !item_stats.rehearsed {
                        return Ok(false);
                    }
                }
                ConsolidationCondition::ContextualRelevance(threshold) => {
                    if item_stats.contextual_relevance < *threshold {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    /// Consolidate individual item
    async fn consolidate_item(&self, candidate: &ConsolidationCandidate, rule: &ConsolidationRule) -> Result<Option<ConsolidatedItem>> {
        // Perform the actual consolidation based on target memory type
        match rule.target_memory {
            MemoryType::ShortTermMemory => {
                // Move item to short-term memory
                self.move_to_short_term_memory(candidate).await?;
            }
            MemoryType::LongTermMemory => {
                // Move item to long-term memory (knowledge graph)
                self.move_to_long_term_memory(candidate).await?;
            }
            MemoryType::SemanticMemory => {
                // Move item to semantic memory (SDR storage)
                self.move_to_semantic_memory(candidate).await?;
            }
            MemoryType::EpisodicMemory => {
                // Move item to episodic memory
                self.move_to_episodic_memory(candidate).await?;
            }
            _ => {
                // Other memory types - would implement specific logic
            }
        }

        Ok(Some(ConsolidatedItem {
            item_id: candidate.item_id.clone(),
            source_memory: candidate.current_memory.clone(),
            target_memory: candidate.proposed_memory.clone(),
            consolidation_strength: rule.consolidation_strength,
        }))
    }

    /// Move item to short-term memory
    async fn move_to_short_term_memory(&self, candidate: &ConsolidationCandidate) -> Result<()> {
        // In practice, would implement actual short-term memory storage
        // For now, this is a placeholder
        Ok(())
    }

    /// Move item to long-term memory
    async fn move_to_long_term_memory(&self, candidate: &ConsolidationCandidate) -> Result<()> {
        // Add item to knowledge graph
        // This is a simplified implementation
        Ok(())
    }

    /// Move item to semantic memory
    async fn move_to_semantic_memory(&self, candidate: &ConsolidationCandidate) -> Result<()> {
        // Add item to SDR storage
        // This is a simplified implementation
        Ok(())
    }

    /// Move item to episodic memory
    async fn move_to_episodic_memory(&self, candidate: &ConsolidationCandidate) -> Result<()> {
        // Add item to episodic memory storage
        // This is a simplified implementation
        Ok(())
    }

    /// Perform automatic consolidation
    pub async fn perform_automatic_consolidation(&self) -> Result<ConsolidationResult> {
        let start_time = Instant::now();
        let mut consolidated_items = Vec::new();

        // Get all active policies sorted by priority
        let policies = self.coordinator.get_active_policies();

        // Execute consolidation for each policy
        for policy in policies {
            let policy_result = self.execute_consolidation_policy(policy).await?;
            consolidated_items.extend(policy_result.consolidated_items);
        }

        // Update statistics
        let mut stats = self.memory_statistics.write().await;
        stats.record_consolidation();

        Ok(ConsolidationResult {
            consolidated_items,
            consolidation_time: start_time.elapsed(),
            success_rate: 1.0,
        })
    }

    /// Optimize consolidation parameters
    pub async fn optimize_consolidation(&self, performance_data: &PerformanceAnalysis) -> Result<()> {
        // This would implement consolidation optimization based on performance data
        // For now, this is a placeholder
        Ok(())
    }

    /// Get consolidation statistics
    pub async fn get_consolidation_statistics(&self) -> Result<HashMap<String, f32>> {
        let stats = self.memory_statistics.read().await;
        let mut consolidation_stats = HashMap::new();

        consolidation_stats.insert("total_consolidations".to_string(), stats.consolidation_events as f32);
        consolidation_stats.insert("consolidation_rate".to_string(), 
            stats.consolidation_events as f32 / stats.total_retrievals as f32);

        Ok(consolidation_stats)
    }
}

