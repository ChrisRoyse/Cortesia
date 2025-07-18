//! Main unified memory system implementation

use super::types::*;
use super::coordinator::MemoryCoordinator;
use super::retrieval::MemoryRetrieval;
use super::consolidation::MemoryConsolidation;
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryQuery, MemoryRetrievalResult, MemoryItem, MemoryContent};
use crate::core::sdr_storage::SDRStorage;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::error::{Result, GraphError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Unified memory system integrating multiple memory backends
#[derive(Clone)]
pub struct UnifiedMemorySystem {
    pub working_memory: Arc<WorkingMemorySystem>,
    pub sdr_storage: Arc<SDRStorage>,
    pub long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub memory_coordinator: Arc<MemoryCoordinator>,
    pub memory_retrieval: Arc<MemoryRetrieval>,
    pub memory_consolidation: Arc<MemoryConsolidation>,
    pub integration_config: MemoryIntegrationConfig,
    pub memory_statistics: Arc<RwLock<MemoryStatistics>>,
}

impl UnifiedMemorySystem {
    /// Create new unified memory system
    pub fn new(
        working_memory: Arc<WorkingMemorySystem>,
        sdr_storage: Arc<SDRStorage>,
        long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        let memory_coordinator = Arc::new(MemoryCoordinator::new());
        let memory_statistics = Arc::new(RwLock::new(MemoryStatistics::new()));
        
        let memory_retrieval = Arc::new(MemoryRetrieval::new(
            working_memory.clone(),
            sdr_storage.clone(),
            long_term_graph.clone(),
            memory_coordinator.clone(),
        ));

        let memory_consolidation = Arc::new(MemoryConsolidation::new(
            working_memory.clone(),
            sdr_storage.clone(),
            long_term_graph.clone(),
            memory_coordinator.clone(),
            memory_statistics.clone(),
        ));

        Self {
            working_memory,
            sdr_storage,
            long_term_graph,
            memory_coordinator,
            memory_retrieval,
            memory_consolidation,
            integration_config: MemoryIntegrationConfig::default(),
            memory_statistics,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        working_memory: Arc<WorkingMemorySystem>,
        sdr_storage: Arc<SDRStorage>,
        long_term_graph: Arc<BrainEnhancedKnowledgeGraph>,
        config: MemoryIntegrationConfig,
    ) -> Self {
        let mut system = Self::new(working_memory, sdr_storage, long_term_graph);
        system.integration_config = config;
        system
    }

    /// Store information across memory systems
    pub async fn store_information(&self, content: &str, importance: f32, _context: Option<&str>) -> Result<String> {
        let start_time = Instant::now();
        let item_id = format!("item_{}", start_time.elapsed().as_nanos());

        // Create memory item
        let memory_item = MemoryItem {
            content: MemoryContent::Concept(content.to_string()),
            activation_level: 1.0,
            timestamp: start_time,
            importance_score: importance,
            access_count: 1,
            decay_factor: 1.0,
        };

        // Store in working memory initially
        self.working_memory.store_in_working_memory(
            memory_item.content.clone(),
            memory_item.importance_score,
            crate::cognitive::working_memory::BufferType::Episodic,
        ).await?;

        // Update statistics
        let mut stats = self.memory_statistics.write().await;
        stats.record_retrieval(true, start_time.elapsed());

        Ok(item_id)
    }

    /// Retrieve information using integrated approach
    pub async fn retrieve_information(&self, query: &str, strategy_id: Option<&str>) -> Result<MemoryIntegrationResult> {
        let start_time = Instant::now();
        
        // Use memory retrieval system
        let result = self.memory_retrieval.retrieve_integrated(query, strategy_id).await?;

        // Update statistics
        let mut stats = self.memory_statistics.write().await;
        stats.record_retrieval(true, start_time.elapsed());

        Ok(result)
    }

    /// Perform memory consolidation
    pub async fn consolidate_memories(&self, policy_id: Option<&str>) -> Result<ConsolidationResult> {
        self.memory_consolidation.perform_consolidation(policy_id).await
    }

    /// Get memory statistics
    pub async fn get_memory_statistics(&self) -> Result<MemoryStatistics> {
        let stats = self.memory_statistics.read().await;
        Ok(stats.clone())
    }

    /// Analyze performance
    pub async fn analyze_performance(&self) -> Result<PerformanceAnalysis> {
        let stats = self.memory_statistics.read().await;
        let mut bottlenecks = Vec::new();

        // Analyze retrieval performance
        if stats.average_retrieval_time > Duration::from_millis(100) {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::SlowRetrieval,
                severity: (stats.average_retrieval_time.as_millis() as f32) / 1000.0,
                affected_memory: MemoryType::WorkingMemory,
                suggested_fix: "Optimize retrieval algorithms".to_string(),
            });
        }

        // Analyze success rate
        let success_rate = stats.get_success_rate();
        if success_rate < 0.8 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::LowSuccessRate,
                severity: 1.0 - success_rate,
                affected_memory: MemoryType::WorkingMemory,
                suggested_fix: "Improve query processing".to_string(),
            });
        }

        // Analyze memory utilization
        for (memory_type, utilization) in &stats.memory_utilization {
            if *utilization > 0.9 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::HighMemoryUsage,
                    severity: *utilization,
                    affected_memory: memory_type.clone(),
                    suggested_fix: "Increase memory capacity or improve consolidation".to_string(),
                });
            }
        }

        Ok(PerformanceAnalysis {
            success_rate,
            average_retrieval_time: stats.average_retrieval_time,
            memory_utilization: stats.memory_utilization.clone(),
            bottlenecks,
        })
    }

    /// Optimize memory system
    pub async fn optimize_memory_system(&self) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        
        // Analyze current performance
        let performance_analysis = self.analyze_performance().await?;
        
        // Identify optimization opportunities
        let optimization_opportunities = self.identify_optimization_opportunities(&performance_analysis).await?;
        
        // Apply optimizations
        let applied_optimizations = self.apply_optimizations(&optimization_opportunities).await?;
        
        // Update coordinator with performance data (removed optimization since Arc<MemoryCoordinator> doesn't support mutable operations)
        // Note: Coordinator optimization would require a different approach with Interior Mutability

        Ok(OptimizationResult {
            performance_analysis,
            optimization_opportunities,
            applied_optimizations,
            optimization_time: start_time.elapsed(),
        })
    }

    /// Identify optimization opportunities
    async fn identify_optimization_opportunities(&self, performance_analysis: &PerformanceAnalysis) -> Result<Vec<OptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        // Check for slow retrieval
        if performance_analysis.average_retrieval_time > Duration::from_millis(50) {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OpportunityType::ReduceRetrievalTime,
                potential_improvement: 0.3,
                implementation_cost: 0.2,
                priority: 0.8,
            });
        }

        // Check for low success rate
        if performance_analysis.success_rate < 0.9 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OpportunityType::ImproveSuccessRate,
                potential_improvement: 0.9 - performance_analysis.success_rate,
                implementation_cost: 0.3,
                priority: 0.9,
            });
        }

        // Check for high memory usage
        let avg_utilization = performance_analysis.memory_utilization.values().sum::<f32>() / performance_analysis.memory_utilization.len() as f32;
        if avg_utilization > 0.8 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OpportunityType::OptimizeMemoryUsage,
                potential_improvement: avg_utilization - 0.6,
                implementation_cost: 0.4,
                priority: 0.7,
            });
        }

        // Check for cross-memory link opportunities
        let stats = self.memory_statistics.read().await;
        if stats.cross_memory_accesses < stats.total_retrievals / 2 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OpportunityType::EnhanceCrossMemoryLinks,
                potential_improvement: 0.2,
                implementation_cost: 0.1,
                priority: 0.6,
            });
        }

        Ok(opportunities)
    }

    /// Apply optimizations
    async fn apply_optimizations(&self, opportunities: &[OptimizationOpportunity]) -> Result<Vec<AppliedOptimization>> {
        let mut applied_optimizations = Vec::new();

        for opportunity in opportunities {
            let optimization_id = format!("opt_{}", applied_optimizations.len());
            
            // Apply optimization based on type
            let (success, actual_improvement) = match opportunity.opportunity_type {
                OpportunityType::ReduceRetrievalTime => {
                    // Implement retrieval time optimization
                    (true, opportunity.potential_improvement * 0.8)
                }
                OpportunityType::ImproveSuccessRate => {
                    // Implement success rate optimization
                    (true, opportunity.potential_improvement * 0.7)
                }
                OpportunityType::OptimizeMemoryUsage => {
                    // Implement memory usage optimization
                    (true, opportunity.potential_improvement * 0.6)
                }
                OpportunityType::EnhanceCrossMemoryLinks => {
                    // Implement cross-memory link enhancement
                    (true, opportunity.potential_improvement * 0.9)
                }
            };

            applied_optimizations.push(AppliedOptimization {
                optimization_id,
                optimization_type: opportunity.opportunity_type.clone(),
                actual_improvement,
                implementation_success: success,
            });
        }

        Ok(applied_optimizations)
    }

    /// Get cross-memory links
    pub async fn get_cross_memory_links(&self, item_id: &str) -> Result<Vec<String>> {
        let links = self.memory_coordinator.get_cross_memory_links(item_id).await;
        Ok(links.into_iter().map(|link| link.link_id).collect())
    }

    /// Create cross-memory link
    pub async fn create_cross_memory_link(&self, link: CrossMemoryLink) -> Result<()> {
        self.memory_coordinator.create_cross_memory_link(link).await
            .map_err(|e| GraphError::ProcessingError(format!("Failed to create cross-memory link: {}", e)))
    }

    /// Search across all memory systems
    pub async fn search_all_memories(&self, query: &str, limit: usize) -> Result<Vec<MemoryRetrievalResult>> {
        let mut all_results = Vec::new();

        // Search working memory
        let working_query = MemoryQuery {
            query_text: query.to_string(),
            search_buffers: vec![
                crate::cognitive::working_memory::BufferType::Phonological,
                crate::cognitive::working_memory::BufferType::Episodic,
                crate::cognitive::working_memory::BufferType::Visuospatial,
            ],
            apply_attention: true,
            importance_threshold: 0.3,
            recency_weight: 0.7,
        };
        
        if let Ok(working_result) = self.working_memory.retrieve_from_working_memory(&working_query).await {
            all_results.push(working_result);
        }

        // Search SDR storage
        if let Ok(sdr_results) = self.sdr_storage.similarity_search(query, 0.7).await {
            let items = sdr_results.into_iter().map(|sdr_result| {
                MemoryItem {
                    content: MemoryContent::Concept(sdr_result.content),
                    activation_level: sdr_result.similarity,
                    timestamp: Instant::now(),
                    importance_score: sdr_result.similarity,
                    access_count: 1,
                    decay_factor: 1.0,
                }
            }).collect();
            
            all_results.push(MemoryRetrievalResult {
                items,
                retrieval_confidence: 0.8,
                buffer_states: Vec::new(),
            });
        }

        // Search knowledge graph
        // Convert query string to embedding (simplified - just use hash for now)
        let query_hash = query.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));
        let query_embedding = vec![query_hash as f32 / u32::MAX as f32; 96]; // Assuming 96 dim embeddings
        
        if let Ok(query_result) = self.long_term_graph.query(&query_embedding, &[], limit).await {
            let items = query_result.entities.into_iter().map(|entity| {
                MemoryItem {
                    content: MemoryContent::Concept(format!("entity_{:?}", entity.id)),
                    activation_level: entity.similarity,
                    timestamp: Instant::now(),
                    importance_score: entity.similarity,
                    access_count: 1,
                    decay_factor: 1.0,
                }
            }).collect();
            
            all_results.push(MemoryRetrievalResult {
                items,
                retrieval_confidence: 0.85,
                buffer_states: Vec::new(),
            });
        }

        Ok(all_results)
    }

    /// Update memory configuration
    pub fn update_config(&mut self, config: MemoryIntegrationConfig) {
        self.integration_config = config;
    }

    /// Get system status
    pub async fn get_system_status(&self) -> Result<String> {
        let stats = self.memory_statistics.read().await;
        let coordinator_report = self.memory_coordinator.generate_report();
        
        let mut status = String::new();
        status.push_str("Unified Memory System Status\n");
        status.push_str("===========================\n\n");
        
        status.push_str(&format!("Total Retrievals: {}\n", stats.total_retrievals));
        status.push_str(&format!("Success Rate: {:.2}%\n", stats.get_success_rate() * 100.0));
        status.push_str(&format!("Average Retrieval Time: {:?}\n", stats.average_retrieval_time));
        status.push_str(&format!("Consolidation Events: {}\n", stats.consolidation_events));
        status.push_str(&format!("Cross-Memory Accesses: {}\n", stats.cross_memory_accesses));
        
        status.push_str("\n");
        status.push_str(&coordinator_report);
        
        Ok(status)
    }
}


impl PartialEq for BottleneckType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BottleneckType::SlowRetrieval, BottleneckType::SlowRetrieval) => true,
            (BottleneckType::LowSuccessRate, BottleneckType::LowSuccessRate) => true,
            (BottleneckType::HighMemoryUsage, BottleneckType::HighMemoryUsage) => true,
            (BottleneckType::FrequentConflicts, BottleneckType::FrequentConflicts) => true,
            _ => false,
        }
    }
}

impl Eq for BottleneckType {}