//! Type definitions for memory integration system

use std::collections::HashMap;
use std::time::Duration;

/// Memory integration configuration
#[derive(Debug, Clone)]
pub struct MemoryIntegrationConfig {
    pub enable_parallel_retrieval: bool,
    pub default_strategy: String,
    pub consolidation_frequency: Duration,
    pub optimization_frequency: Duration,
    pub cross_memory_linking: bool,
    pub memory_hierarchy_depth: usize,
}

/// Memory types in the hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryType {
    SensoryBuffer,
    WorkingMemory,
    ShortTermMemory,
    LongTermMemory,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
}

/// Memory capacity configuration
#[derive(Debug, Clone)]
pub struct MemoryCapacity {
    pub max_items: usize,
    pub max_size_bytes: usize,
    pub utilization_threshold: f32,
    pub overflow_strategy: OverflowStrategy,
}

/// Overflow handling strategies
#[derive(Debug, Clone)]
pub enum OverflowStrategy {
    LeastRecentlyUsed,
    LeastFrequentlyUsed,
    ImportanceBased,
    ForgettingCurve,
    RandomEviction,
}

/// Memory access speed categories
#[derive(Debug, Clone)]
pub enum AccessSpeed {
    Immediate,      // < 1ms
    Fast,          // 1-10ms
    Medium,        // 10-100ms
    Slow,          // 100ms-1s
    VerySlow,      // > 1s
}

/// Memory level in the hierarchy
#[derive(Debug, Clone)]
pub struct MemoryLevel {
    pub level_id: String,
    pub memory_type: MemoryType,
    pub capacity: MemoryCapacity,
    pub retention_period: Duration,
    pub access_speed: AccessSpeed,
    pub stability: f32,
}

/// Transition criteria between memory levels
#[derive(Debug, Clone)]
pub struct TransitionCriteria {
    pub access_count_threshold: u32,
    pub importance_threshold: f32,
    pub time_threshold: Duration,
    pub rehearsal_requirement: bool,
}

/// Thresholds for memory transitions
#[derive(Debug, Clone)]
pub struct TransitionThresholds {
    pub working_to_short_term: TransitionCriteria,
    pub short_term_to_long_term: TransitionCriteria,
    pub episodic_to_semantic: TransitionCriteria,
    pub consolidation_threshold: f32,
}

/// Cross-memory link for connecting information across memory systems
#[derive(Debug, Clone)]
pub struct CrossMemoryLink {
    pub link_id: String,
    pub source_memory: MemoryType,
    pub target_memory: MemoryType,
    pub link_strength: f32,
    pub link_type: LinkType,
    pub bidirectional: bool,
}

/// Types of cross-memory links
#[derive(Debug, Clone)]
pub enum LinkType {
    Associative,
    Causal,
    Temporal,
    Contextual,
    Semantic,
}

/// Retrieval strategy types
#[derive(Debug, Clone)]
pub enum RetrievalType {
    ParallelSearch,
    HierarchicalSearch,
    AdaptiveSearch,
    ContextualSearch,
}

/// Methods for fusing results from multiple memory systems
#[derive(Debug, Clone)]
pub enum FusionMethod {
    WeightedAverage,
    MaximumConfidence,
    MajorityVoting,
    RankFusion,
    ContextualFusion,
}

/// Confidence weighting configuration
#[derive(Debug, Clone)]
pub struct ConfidenceWeighting {
    pub working_memory_weight: f32,
    pub sdr_storage_weight: f32,
    pub graph_storage_weight: f32,
    pub recency_factor: f32,
    pub importance_factor: f32,
}

/// Retrieval strategy configuration
#[derive(Debug, Clone)]
pub struct RetrievalStrategy {
    pub strategy_id: String,
    pub strategy_type: RetrievalType,
    pub memory_priority: Vec<MemoryType>,
    pub fusion_method: FusionMethod,
    pub confidence_weighting: ConfidenceWeighting,
}

/// Consolidation rule for memory transitions
#[derive(Debug, Clone)]
pub struct ConsolidationRule {
    pub rule_id: String,
    pub source_memory: MemoryType,
    pub target_memory: MemoryType,
    pub conditions: Vec<ConsolidationCondition>,
    pub consolidation_strength: f32,
}

/// Conditions for memory consolidation
#[derive(Debug, Clone)]
pub enum ConsolidationCondition {
    AccessCountThreshold(u32),
    ImportanceThreshold(f32),
    TimeThreshold(Duration),
    RehearsalRequired,
    ContextualRelevance(f32),
}

/// Consolidation trigger events
#[derive(Debug, Clone)]
pub enum ConsolidationTrigger {
    TimeBasedTrigger(Duration),
    UsageBasedTrigger(u32),
    ImportanceBasedTrigger(f32),
    CapacityBasedTrigger(f32),
    ContextualTrigger(String),
}

/// Consolidation policy
#[derive(Debug, Clone)]
pub struct ConsolidationPolicy {
    pub policy_id: String,
    pub trigger_conditions: Vec<ConsolidationTrigger>,
    pub consolidation_rules: Vec<ConsolidationRule>,
    pub priority: f32,
}

/// Memory statistics tracking
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub total_retrievals: u64,
    pub successful_retrievals: u64,
    pub failed_retrievals: u64,
    pub average_retrieval_time: Duration,
    pub memory_utilization: HashMap<MemoryType, f32>,
    pub consolidation_events: u64,
    pub cross_memory_accesses: u64,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f32,
    pub affected_memory: MemoryType,
    pub suggested_fix: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckType {
    SlowRetrieval,
    LowSuccessRate,
    HighMemoryUsage,
    FrequentConflicts,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub success_rate: f32,
    pub average_retrieval_time: Duration,
    pub memory_utilization: HashMap<MemoryType, f32>,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OpportunityType,
    pub potential_improvement: f32,
    pub implementation_cost: f32,
    pub priority: f32,
}

/// Types of optimization opportunities
#[derive(Debug, Clone)]
pub enum OpportunityType {
    ImproveSuccessRate,
    ReduceRetrievalTime,
    OptimizeMemoryUsage,
    EnhanceCrossMemoryLinks,
}

/// Applied optimization record
#[derive(Debug, Clone)]
pub struct AppliedOptimization {
    pub optimization_id: String,
    pub optimization_type: OpportunityType,
    pub actual_improvement: f32,
    pub implementation_success: bool,
}

/// Consolidated memory item
#[derive(Debug, Clone)]
pub struct ConsolidatedItem {
    pub item_id: String,
    pub source_memory: MemoryType,
    pub target_memory: MemoryType,
    pub consolidation_strength: f32,
}

/// Consolidation operation result
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    pub consolidated_items: Vec<ConsolidatedItem>,
    pub consolidation_time: Duration,
    pub success_rate: f32,
}

/// Optimization execution result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub performance_analysis: PerformanceAnalysis,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub applied_optimizations: Vec<AppliedOptimization>,
    pub optimization_time: Duration,
}

/// Memory integration query result
#[derive(Debug, Clone)]
pub struct MemoryIntegrationResult {
    pub primary_results: Vec<crate::cognitive::working_memory::MemoryRetrievalResult>,
    pub secondary_results: Vec<crate::cognitive::working_memory::MemoryRetrievalResult>,
    pub fusion_confidence: f32,
    pub retrieval_strategy_used: String,
    pub cross_memory_links_activated: Vec<String>,
}

/// Memory consolidation candidate
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub item_id: String,
    pub current_memory: MemoryType,
    pub proposed_memory: MemoryType,
    pub consolidation_score: f32,
}

impl Default for MemoryIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_parallel_retrieval: true,
            default_strategy: "parallel_comprehensive".to_string(),
            consolidation_frequency: Duration::from_secs(300),
            optimization_frequency: Duration::from_secs(3600),
            cross_memory_linking: true,
            memory_hierarchy_depth: 7,
        }
    }
}

impl Default for ConfidenceWeighting {
    fn default() -> Self {
        Self {
            working_memory_weight: 0.4,
            sdr_storage_weight: 0.3,
            graph_storage_weight: 0.3,
            recency_factor: 0.2,
            importance_factor: 0.3,
        }
    }
}

impl MemoryStatistics {
    /// Create new memory statistics
    pub fn new() -> Self {
        Self {
            total_retrievals: 0,
            successful_retrievals: 0,
            failed_retrievals: 0,
            average_retrieval_time: Duration::from_millis(50),
            memory_utilization: HashMap::new(),
            consolidation_events: 0,
            cross_memory_accesses: 0,
        }
    }
    
    /// Calculate success rate
    pub fn get_success_rate(&self) -> f32 {
        if self.total_retrievals == 0 {
            0.0
        } else {
            self.successful_retrievals as f32 / self.total_retrievals as f32
        }
    }
    
    /// Record retrieval attempt
    pub fn record_retrieval(&mut self, success: bool, duration: Duration) {
        self.total_retrievals += 1;
        
        if success {
            self.successful_retrievals += 1;
        } else {
            self.failed_retrievals += 1;
        }
        
        // Update average retrieval time
        let total_time = self.average_retrieval_time.as_millis() as f64 * (self.total_retrievals - 1) as f64
            + duration.as_millis() as f64;
        self.average_retrieval_time = Duration::from_millis((total_time / self.total_retrievals as f64) as u64);
    }
    
    /// Record consolidation event
    pub fn record_consolidation(&mut self) {
        self.consolidation_events += 1;
    }
    
    /// Record cross-memory access
    pub fn record_cross_memory_access(&mut self) {
        self.cross_memory_accesses += 1;
    }
    
    /// Update memory utilization
    pub fn update_memory_utilization(&mut self, memory_type: MemoryType, utilization: f32) {
        self.memory_utilization.insert(memory_type, utilization);
    }
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified retrieval result from memory integration system
#[derive(Debug, Clone)]
pub struct UnifiedRetrievalResult {
    pub items: Vec<RetrievedItem>,
    pub source_memories: Vec<MemoryType>,
    pub total_confidence: f32,
    pub retrieval_time: Duration,
    pub metadata: HashMap<String, String>,
}

/// Individual retrieved item
#[derive(Debug, Clone)]
pub struct RetrievedItem {
    pub content: String,
    pub confidence: f32,
    pub source: MemoryType,
    pub relevance_score: f32,
}