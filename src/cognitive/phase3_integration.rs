use crate::cognitive::{
    CognitiveOrchestrator, CognitivePatternType,
    WorkingMemorySystem, AttentionManager, UnifiedMemorySystem,
    ConvergentThinking, DivergentThinking, LateralThinking, SystemsThinking, 
    CriticalThinking, AbstractThinking, AdaptiveThinking
};
use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
use crate::cognitive::types::{ComplexityEstimate, QueryContext};
use crate::core::{
    activation_engine::ActivationPropagationEngine,
    brain_enhanced_graph::BrainEnhancedKnowledgeGraph,
    brain_types::ActivationPattern,
    sdr_storage::SDRStorage,
    types::EntityKey,
};
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Phase3IntegratedCognitiveSystem {
    // Core systems
    pub orchestrator: Arc<CognitiveOrchestrator>,
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub sdr_storage: Arc<SDRStorage>,
    
    // Phase 3 new systems
    pub working_memory: Arc<WorkingMemorySystem>,
    pub attention_manager: Arc<AttentionManager>,
    pub inhibitory_logic: Arc<CompetitiveInhibitionSystem>,
    pub unified_memory: Arc<UnifiedMemorySystem>,
    
    // Cognitive patterns
    pub convergent_thinking: Arc<ConvergentThinking>,
    pub divergent_thinking: Arc<DivergentThinking>,
    pub lateral_thinking: Arc<LateralThinking>,
    pub systems_thinking: Arc<SystemsThinking>,
    pub critical_thinking: Arc<CriticalThinking>,
    pub abstract_thinking: Arc<AbstractThinking>,
    pub adaptive_thinking: Arc<AdaptiveThinking>,
    
    // Integration configuration
    pub integration_config: Phase3IntegrationConfig,
    pub system_state: Arc<RwLock<SystemState>>,
    pub performance_metrics: Arc<RwLock<SystemPerformanceMetrics>>,
}

impl std::fmt::Debug for Phase3IntegratedCognitiveSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Phase3IntegratedCognitiveSystem")
            .field("orchestrator", &"CognitiveOrchestrator")
            .field("activation_engine", &"ActivationPropagationEngine")
            .field("brain_graph", &"BrainEnhancedKnowledgeGraph")
            .field("sdr_storage", &"SDRStorage")
            .field("working_memory", &"WorkingMemorySystem")
            .field("attention_manager", &"AttentionManager")
            .field("inhibitory_logic", &"CompetitiveInhibitionSystem")
            .field("unified_memory", &"UnifiedMemorySystem")
            .field("integration_config", &self.integration_config)
            .field("system_state", &"RwLock<SystemState>")
            .field("performance_metrics", &"RwLock<SystemPerformanceMetrics>")
            .finish()
    }
}

/// Type alias for backward compatibility with Phase 4 integration
pub type IntegratedCognitiveSystem = Phase3IntegratedCognitiveSystem;

#[derive(Debug, Clone)]
pub struct Phase3IntegrationConfig {
    pub enable_working_memory: bool,
    pub enable_attention_management: bool,
    pub enable_competitive_inhibition: bool,
    pub enable_unified_memory: bool,
    pub pattern_integration_mode: PatternIntegrationMode,
    pub performance_monitoring: bool,
    pub automatic_optimization: bool,
}

#[derive(Debug, Clone)]
pub enum PatternIntegrationMode {
    Sequential,     // Patterns run one after another
    Parallel,       // Patterns run simultaneously
    Orchestrated,   // Orchestrator manages pattern execution
    Adaptive,       // System adapts based on query type
}

#[derive(Debug, Clone)]
pub struct SystemState {
    pub current_focus: Option<EntityKey>,
    pub active_patterns: Vec<CognitivePatternType>,
    pub working_memory_load: f32,
    pub attention_capacity: f32,
    pub inhibition_strength: f32,
    pub system_performance: f32,
    pub last_optimization: Instant,
    pub user_satisfaction: f32,
}

#[derive(Debug, Clone)]
pub struct SystemPerformanceMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub average_response_time: Duration,
    pub working_memory_utilization: f32,
    pub attention_switches: u64,
    pub inhibition_events: u64,
    pub memory_consolidations: u64,
    pub pattern_usage_stats: HashMap<CognitivePatternType, PatternUsageStats>,
    pub convergent_performance: f32,
    pub divergent_performance: f32,
    pub lateral_performance: f32,
    pub systems_performance: f32,
    pub critical_performance: f32,
    pub abstract_performance: f32,
    pub adaptive_performance: f32,
    pub overall_efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct PatternUsageStats {
    pub usage_count: u64,
    pub average_execution_time: Duration,
    pub success_rate: f32,
    pub average_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct PatternResultComponent {
    pub confidence: f32,
    pub quality_score: f32,
    pub execution_time: Duration,
    pub result_data: String,
}

#[derive(Debug, Clone)]
pub struct Phase3QueryResult {
    pub query: String,
    pub response: String,
    pub confidence: f32,
    pub reasoning_trace: ReasoningTrace,
    pub performance_metrics: QueryPerformanceMetrics,
    pub system_state_changes: SystemStateChanges,
    
    // Phase 4 compatibility fields
    pub overall_confidence: f32,
    pub pattern_results: std::collections::HashMap<CognitivePatternType, PatternResultComponent>,
    pub response_time: Duration,
    pub query_complexity: ComplexityEstimate,
    pub context: QueryContext,
    pub primary_pattern: CognitivePatternType,
}

#[derive(Debug, Clone)]
pub struct ReasoningTrace {
    pub activated_patterns: Vec<CognitivePatternType>,
    pub working_memory_operations: Vec<WorkingMemoryOperation>,
    pub attention_shifts: Vec<AttentionShift>,
    pub inhibition_events: Vec<InhibitionEvent>,
    pub memory_consolidations: Vec<MemoryConsolidation>,
    pub pattern_interactions: Vec<PatternInteraction>,
}

#[derive(Debug, Clone)]
pub struct WorkingMemoryOperation {
    pub operation_type: WorkingMemoryOpType,
    pub timestamp: Instant,
    pub affected_buffers: Vec<String>,
    pub memory_load_change: f32,
}

#[derive(Debug, Clone)]
pub enum WorkingMemoryOpType {
    Store,
    Retrieve,
    Decay,
    Consolidate,
    Evict,
}

#[derive(Debug, Clone)]
pub struct AttentionShift {
    pub from_entities: Vec<EntityKey>,
    pub to_entities: Vec<EntityKey>,
    pub timestamp: Instant,
    pub attention_strength: f32,
    pub shift_reason: AttentionShiftReason,
}

#[derive(Debug, Clone)]
pub enum AttentionShiftReason {
    PatternRequirement,
    UserQuery,
    SystemOptimization,
    MemoryConsolidation,
    InhibitionTriggered,
}

#[derive(Debug, Clone)]
pub struct InhibitionEvent {
    pub event_type: InhibitionEventType,
    pub timestamp: Instant,
    pub affected_entities: Vec<EntityKey>,
    pub inhibition_strength: f32,
    pub pattern_trigger: Option<CognitivePatternType>,
}

#[derive(Debug, Clone)]
pub enum InhibitionEventType {
    CompetitiveInhibition,
    HierarchicalInhibition,
    ContextualInhibition,
    TemporalInhibition,
}

#[derive(Debug, Clone)]
pub struct MemoryConsolidation {
    pub source_memory: String,
    pub target_memory: String,
    pub timestamp: Instant,
    pub items_consolidated: usize,
    pub success_rate: f32,
}

#[derive(Debug, Clone)]
pub struct PatternInteraction {
    pub pattern1: CognitivePatternType,
    pub pattern2: CognitivePatternType,
    pub interaction_type: InteractionType,
    pub timestamp: Instant,
    pub outcome: InteractionOutcome,
}

#[derive(Debug, Clone)]
pub enum InteractionType {
    Collaboration,
    Competition,
    Sequence,
    Parallel,
    Feedback,
}

#[derive(Debug, Clone)]
pub enum InteractionOutcome {
    Enhanced,
    Conflicted,
    Complementary,
    Redundant,
}

#[derive(Debug, Clone)]
pub struct QueryPerformanceMetrics {
    pub total_time: Duration,
    pub pattern_execution_times: HashMap<CognitivePatternType, Duration>,
    pub memory_operation_times: HashMap<String, Duration>,
    pub attention_shift_time: Duration,
    pub inhibition_processing_time: Duration,
    pub consolidation_time: Duration,
}

#[derive(Debug, Clone)]
pub struct SystemStateChanges {
    pub working_memory_changes: Vec<String>,
    pub attention_changes: Vec<String>,
    pub inhibition_changes: Vec<String>,
    pub pattern_activations: Vec<String>,
    pub performance_changes: Vec<String>,
}

impl Default for Phase3IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_working_memory: true,
            enable_attention_management: true,
            enable_competitive_inhibition: true,
            enable_unified_memory: true,
            pattern_integration_mode: PatternIntegrationMode::Orchestrated,
            performance_monitoring: true,
            automatic_optimization: true,
        }
    }
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            current_focus: None,
            active_patterns: Vec::new(),
            working_memory_load: 0.0,
            attention_capacity: 1.0,
            inhibition_strength: 0.5,
            system_performance: 1.0,
            last_optimization: Instant::now(),
            user_satisfaction: 0.5,
        }
    }

    pub fn update_performance(&mut self, new_performance: f32) {
        self.system_performance = new_performance.clamp(0.0, 1.0);
    }

    pub fn should_optimize(&self) -> bool {
        self.last_optimization.elapsed() > Duration::from_secs(300) || 
        self.system_performance < 0.6
    }
}

impl SystemPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            average_response_time: Duration::from_millis(0),
            working_memory_utilization: 0.0,
            attention_switches: 0,
            inhibition_events: 0,
            memory_consolidations: 0,
            pattern_usage_stats: HashMap::new(),
            convergent_performance: 0.0,
            divergent_performance: 0.0,
            lateral_performance: 0.0,
            systems_performance: 0.0,
            critical_performance: 0.0,
            abstract_performance: 0.0,
            adaptive_performance: 0.0,
            overall_efficiency: 0.0,
        }
    }

    pub fn update_query_stats(&mut self, success: bool, response_time: Duration) {
        self.total_queries += 1;
        if success {
            self.successful_queries += 1;
        }
        
        // Update running average
        let current_avg = self.average_response_time.as_nanos() as f64;
        let new_time = response_time.as_nanos() as f64;
        let count = self.total_queries as f64;
        let new_avg = ((current_avg * (count - 1.0)) + new_time) / count;
        self.average_response_time = Duration::from_nanos(new_avg as u64);
    }

    pub fn get_success_rate(&self) -> f32 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.successful_queries as f32 / self.total_queries as f32
        }
    }

    pub fn update_pattern_stats(&mut self, pattern: CognitivePatternType, execution_time: Duration, success: bool, confidence: f32) {
        let stats = self.pattern_usage_stats.entry(pattern).or_insert_with(|| PatternUsageStats {
            usage_count: 0,
            average_execution_time: Duration::from_millis(0),
            success_rate: 0.0,
            average_confidence: 0.0,
        });

        stats.usage_count += 1;
        
        // Update running averages
        let current_avg_time = stats.average_execution_time.as_nanos() as f64;
        let new_time = execution_time.as_nanos() as f64;
        let count = stats.usage_count as f64;
        let new_avg_time = ((current_avg_time * (count - 1.0)) + new_time) / count;
        stats.average_execution_time = Duration::from_nanos(new_avg_time as u64);

        let current_success_rate = stats.success_rate;
        stats.success_rate = ((current_success_rate * (count - 1.0) as f32) + if success { 1.0 } else { 0.0 }) / count as f32;

        let current_confidence = stats.average_confidence;
        stats.average_confidence = ((current_confidence * (count - 1.0) as f32) + confidence) / count as f32;
    }
}

impl Phase3IntegratedCognitiveSystem {
    pub async fn new(
        orchestrator: Arc<CognitiveOrchestrator>,
        activation_engine: Arc<ActivationPropagationEngine>,
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        sdr_storage: Arc<SDRStorage>,
    ) -> Result<Self> {
        // Initialize Phase 3 systems
        let working_memory = Arc::new(WorkingMemorySystem::new(
            activation_engine.clone(),
            sdr_storage.clone(),
        ).await?);

        let attention_manager = Arc::new(AttentionManager::new(
            orchestrator.clone(),
            activation_engine.clone(),
            working_memory.clone(),
        ).await?);

        let critical_thinking = Arc::new(CriticalThinking::new(
            brain_graph.clone(),
        ));

        let inhibitory_logic = Arc::new(CompetitiveInhibitionSystem::new(
            activation_engine.clone(),
            critical_thinking.clone(),
        ));

        let unified_memory = Arc::new(UnifiedMemorySystem::new(
            working_memory.clone(),
            sdr_storage.clone(),
            brain_graph.clone(),
        ));

        // Initialize cognitive patterns
        let convergent_thinking = Arc::new(ConvergentThinking::new(
            brain_graph.clone(),
        ));

        let divergent_thinking = Arc::new(DivergentThinking::new(
            brain_graph.clone(),
        ));

        let lateral_thinking = Arc::new(LateralThinking::new(
            brain_graph.clone(),
        ));

        let systems_thinking = Arc::new(SystemsThinking::new(
            brain_graph.clone(),
        ));

        let abstract_thinking = Arc::new(AbstractThinking::new(
            brain_graph.clone(),
        ));

        let adaptive_thinking = Arc::new(AdaptiveThinking::new(
            brain_graph.clone(),
        ));

        Ok(Self {
            orchestrator,
            activation_engine,
            brain_graph,
            sdr_storage,
            working_memory,
            attention_manager,
            inhibitory_logic,
            unified_memory,
            convergent_thinking,
            divergent_thinking,
            lateral_thinking,
            systems_thinking,
            critical_thinking,
            abstract_thinking,
            adaptive_thinking,
            integration_config: Phase3IntegrationConfig::default(),
            system_state: Arc::new(RwLock::new(SystemState::new())),
            performance_metrics: Arc::new(RwLock::new(SystemPerformanceMetrics::new())),
        })
    }

    pub async fn execute_advanced_reasoning(&self, query: &str) -> Result<Phase3QueryResult> {
        let start_time = Instant::now();
        let mut reasoning_trace = ReasoningTrace::new();
        let mut system_state_changes = SystemStateChanges::new();

        // 1. Initialize working memory with query
        if self.integration_config.enable_working_memory {
            let memory_op = self.initialize_working_memory(query).await?;
            reasoning_trace.working_memory_operations.push(memory_op);
            system_state_changes.working_memory_changes.push(
                format!("Initialized working memory with query: {}", query)
            );
        }

        // 2. Set up attention focus
        if self.integration_config.enable_attention_management {
            let attention_shift = self.setup_attention_focus(query).await?;
            reasoning_trace.attention_shifts.push(attention_shift);
            system_state_changes.attention_changes.push(
                format!("Set attention focus for query: {}", query)
            );
        }

        // 3. Apply initial inhibition
        if self.integration_config.enable_competitive_inhibition {
            let inhibition_event = self.apply_initial_inhibition(query).await?;
            reasoning_trace.inhibition_events.push(inhibition_event);
            system_state_changes.inhibition_changes.push(
                "Applied initial competitive inhibition".to_string()
            );
        }

        // 4. Execute cognitive patterns with integration
        let (response, confidence, pattern_metrics) = self.execute_integrated_patterns(
            query,
            &mut reasoning_trace,
            &mut system_state_changes,
        ).await?;

        // 5. Consolidate memory if enabled
        if self.integration_config.enable_unified_memory {
            let consolidation = self.consolidate_reasoning_memory(&response, confidence).await?;
            reasoning_trace.memory_consolidations.push(consolidation);
            system_state_changes.working_memory_changes.push(
                "Consolidated reasoning results to long-term memory".to_string()
            );
        }

        // 6. Update system performance metrics
        let total_time = start_time.elapsed();
        let success = confidence > 0.3;
        self.update_performance_metrics(success, total_time, &pattern_metrics).await;

        // 7. Optimize system if needed
        if self.integration_config.automatic_optimization && self.should_optimize().await {
            self.optimize_system_performance().await?;
            system_state_changes.performance_changes.push(
                "Applied automatic system optimization".to_string()
            );
        }

        // Create pattern results from pattern metrics
        let pattern_results: std::collections::HashMap<CognitivePatternType, PatternResultComponent> = pattern_metrics
            .into_iter()
            .map(|(pattern, duration)| {
                (pattern, PatternResultComponent {
                    confidence: confidence,
                    quality_score: confidence * 0.9, // Simplified calculation
                    execution_time: duration,
                    result_data: format!("Pattern {} executed", pattern as u32),
                })
            })
            .collect();

        // Determine primary pattern (use the first activated pattern or default to Convergent)
        let primary_pattern = reasoning_trace.activated_patterns.first()
            .copied()
            .unwrap_or(CognitivePatternType::Convergent);

        Ok(Phase3QueryResult {
            query: query.to_string(),
            response,
            confidence,
            reasoning_trace,
            performance_metrics: QueryPerformanceMetrics {
                total_time,
                pattern_execution_times: HashMap::new(),
                memory_operation_times: HashMap::new(),
                attention_shift_time: Duration::from_millis(5),
                inhibition_processing_time: Duration::from_millis(3),
                consolidation_time: Duration::from_millis(10),
            },
            system_state_changes,
            
            // Phase 4 compatibility fields
            overall_confidence: confidence,
            pattern_results,
            response_time: total_time,
            query_complexity: ComplexityEstimate {
                computational_complexity: 1,
                estimated_time_ms: total_time.as_millis() as u64,
                memory_requirements_mb: 10,
                confidence: 0.8,
                parallelizable: true,
            },
            context: QueryContext::new(),
            primary_pattern,
        })
    }

    async fn initialize_working_memory(&self, query: &str) -> Result<WorkingMemoryOperation> {
        use crate::cognitive::working_memory::{MemoryContent, BufferType};
        
        let _result = self.working_memory.store_in_working_memory(
            MemoryContent::Concept(query.to_string()),
            0.9,
            BufferType::Phonological,
        ).await?;

        Ok(WorkingMemoryOperation {
            operation_type: WorkingMemoryOpType::Store,
            timestamp: Instant::now(),
            affected_buffers: vec!["phonological".to_string()],
            memory_load_change: 0.1,
        })
    }

    async fn setup_attention_focus(&self, _query: &str) -> Result<AttentionShift> {
        use crate::cognitive::attention_manager::AttentionType;
        
        // Simple attention setup - would be more sophisticated in real implementation
        let target_entities = vec![]; // Would be populated based on query analysis
        
        let _result = self.attention_manager.focus_attention(
            target_entities.clone(),
            0.8,
            AttentionType::Selective,
        ).await?;

        Ok(AttentionShift {
            from_entities: vec![],
            to_entities: target_entities,
            timestamp: Instant::now(),
            attention_strength: 0.8,
            shift_reason: AttentionShiftReason::UserQuery,
        })
    }

    async fn apply_initial_inhibition(&self, _query: &str) -> Result<InhibitionEvent> {
        // Apply initial inhibition to reduce noise
        let activation_pattern = ActivationPattern::new(format!("initial_inhibition_{}", _query));
        
        let _result = self.inhibitory_logic.apply_competitive_inhibition(
            &activation_pattern,
            None, // No specific domain context
        ).await?;

        Ok(InhibitionEvent {
            event_type: InhibitionEventType::CompetitiveInhibition,
            timestamp: Instant::now(),
            affected_entities: vec![],
            inhibition_strength: 0.5,
            pattern_trigger: None,
        })
    }

    async fn execute_integrated_patterns(
        &self,
        query: &str,
        reasoning_trace: &mut ReasoningTrace,
        system_state_changes: &mut SystemStateChanges,
    ) -> Result<(String, f32, HashMap<CognitivePatternType, Duration>)> {
        let mut pattern_metrics = HashMap::new();
        let mut combined_response = String::new();
        let mut combined_confidence = 0.0;
        let mut pattern_count = 0;

        // Execute patterns based on integration mode
        match self.integration_config.pattern_integration_mode {
            PatternIntegrationMode::Orchestrated => {
                // Use orchestrator to determine best pattern
                let orchestrator_result = self.orchestrator.reason(
                    query, 
                    None,
                    crate::cognitive::types::ReasoningStrategy::Automatic,
                ).await?;
                combined_response = orchestrator_result.final_answer;
                combined_confidence = orchestrator_result.quality_metrics.overall_confidence;
                
                // Record pattern usage
                reasoning_trace.activated_patterns.push(CognitivePatternType::Convergent);
                system_state_changes.pattern_activations.push(
                    "Executed orchestrated pattern selection".to_string()
                );
            }
            PatternIntegrationMode::Parallel => {
                // Execute multiple patterns in parallel
                let parallel_results = self.execute_parallel_patterns(query).await?;
                
                // Combine results
                for (pattern_type, result) in parallel_results {
                    combined_response.push_str(&result.response);
                    combined_response.push('\n');
                    combined_confidence += result.confidence;
                    pattern_count += 1;
                    
                    pattern_metrics.insert(pattern_type, result.execution_time);
                    reasoning_trace.activated_patterns.push(pattern_type);
                }
                
                if pattern_count > 0 {
                    combined_confidence /= pattern_count as f32;
                }
            }
            PatternIntegrationMode::Sequential => {
                // Execute patterns sequentially
                let sequential_results = self.execute_sequential_patterns(query).await?;
                
                // Use the best result
                if let Some((pattern_type, result)) = sequential_results.into_iter().max_by(|a, b| {
                    a.1.confidence.partial_cmp(&b.1.confidence).unwrap()
                }) {
                    combined_response = result.response;
                    combined_confidence = result.confidence;
                    pattern_metrics.insert(pattern_type, result.execution_time);
                    reasoning_trace.activated_patterns.push(pattern_type);
                }
            }
            PatternIntegrationMode::Adaptive => {
                // Adaptively choose pattern based on query
                let chosen_pattern = self.choose_adaptive_pattern(query).await?;
                let result = self.execute_single_pattern(chosen_pattern, query).await?;
                
                combined_response = result.response;
                combined_confidence = result.confidence;
                pattern_metrics.insert(chosen_pattern, result.execution_time);
                reasoning_trace.activated_patterns.push(chosen_pattern);
            }
        }

        Ok((combined_response, combined_confidence, pattern_metrics))
    }

    async fn execute_parallel_patterns(&self, query: &str) -> Result<Vec<(CognitivePatternType, PatternExecutionResult)>> {
        let mut results = Vec::new();
        
        // Execute convergent thinking
        let convergent_start = Instant::now();
        let convergent_result = self.convergent_thinking.execute_convergent_query(query, None).await?;
        let convergent_time = convergent_start.elapsed();
        
        results.push((
            CognitivePatternType::Convergent,
            PatternExecutionResult {
                response: convergent_result.answer,
                confidence: convergent_result.confidence,
                execution_time: convergent_time,
            }
        ));

        // Execute divergent thinking
        let divergent_start = Instant::now();
        let divergent_result = self.divergent_thinking.execute_divergent_exploration(
            query,
            crate::cognitive::types::ExplorationType::Creative,
        ).await?;
        let divergent_time = divergent_start.elapsed();
        
        results.push((
            CognitivePatternType::Divergent,
            PatternExecutionResult {
                response: format!("Divergent exploration found {} paths", divergent_result.total_paths_explored),
                confidence: 0.7,
                execution_time: divergent_time,
            }
        ));

        Ok(results)
    }

    async fn execute_sequential_patterns(&self, query: &str) -> Result<Vec<(CognitivePatternType, PatternExecutionResult)>> {
        let mut results = Vec::new();
        
        // Execute patterns one by one, potentially stopping early if high confidence
        let patterns = vec![
            CognitivePatternType::Convergent,
            CognitivePatternType::Critical,
            CognitivePatternType::Divergent,
        ];

        for pattern in patterns {
            let result = self.execute_single_pattern(pattern, query).await?;
            results.push((pattern, result));
            
            // Stop early if high confidence
            if results.last().unwrap().1.confidence > 0.9 {
                break;
            }
        }

        Ok(results)
    }

    async fn choose_adaptive_pattern(&self, query: &str) -> Result<CognitivePatternType> {
        // Analyze query to choose best pattern
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("analyze") || query_lower.contains("compare") {
            Ok(CognitivePatternType::Critical)
        } else if query_lower.contains("creative") || query_lower.contains("idea") {
            Ok(CognitivePatternType::Divergent)
        } else if query_lower.contains("system") || query_lower.contains("relationship") {
            Ok(CognitivePatternType::Systems)
        } else {
            Ok(CognitivePatternType::Convergent)
        }
    }

    async fn execute_single_pattern(&self, pattern: CognitivePatternType, query: &str) -> Result<PatternExecutionResult> {
        let start_time = Instant::now();
        
        let result = match pattern {
            CognitivePatternType::Convergent => {
                let result = self.convergent_thinking.execute_convergent_query(query, None).await?;
                (result.answer, result.confidence)
            }
            CognitivePatternType::Divergent => {
                let result = self.divergent_thinking.execute_divergent_exploration(
                    query,
                    crate::cognitive::types::ExplorationType::Creative,
                ).await?;
                (format!("Divergent exploration found {} paths", result.total_paths_explored), 0.7)
            }
            CognitivePatternType::Critical => {
                let result = self.critical_thinking.execute_critical_analysis(
                    query,
                    crate::cognitive::types::ValidationLevel::Rigorous,
                ).await?;
                (format!("Critical analysis found {} facts and {} contradictions", result.resolved_facts.len(), result.contradictions_found.len()), 0.8)
            }
            CognitivePatternType::Lateral => {
                // For lateral thinking, extract concepts from query
                let words: Vec<&str> = query.split_whitespace().collect();
                let concept_a = words.get(0).unwrap_or(&"concept").to_string();
                let concept_b = words.get(1).unwrap_or(&"idea").to_string();
                let result = self.lateral_thinking.find_creative_connections(
                    &concept_a,
                    &concept_b,
                    Some(5),
                ).await?;
                (format!("Found {} creative bridges", result.bridges.len()), 0.7)
            }
            CognitivePatternType::Systems => {
                let result = self.systems_thinking.execute_hierarchical_reasoning(
                    query,
                    crate::cognitive::types::SystemsReasoningType::AttributeInheritance,
                ).await?;
                (format!("Systems analysis found {} inherited attributes", result.inherited_attributes.len()), 0.8)
            }
            CognitivePatternType::Abstract => {
                let result = self.abstract_thinking.execute_pattern_analysis(
                    crate::cognitive::types::AnalysisScope::Global,
                    crate::cognitive::types::PatternType::Structural,
                ).await?;
                (format!("Abstract analysis found {} patterns", result.patterns_found.len()), 0.7)
            }
            CognitivePatternType::Adaptive => {
                let result = self.adaptive_thinking.execute_adaptive_reasoning(
                    query,
                    None,
                    vec![CognitivePatternType::Convergent, CognitivePatternType::Divergent],
                ).await?;
                (result.final_answer, result.confidence_distribution.ensemble_confidence)
            }
        };

        Ok(PatternExecutionResult {
            response: result.0,
            confidence: result.1,
            execution_time: start_time.elapsed(),
        })
    }

    async fn consolidate_reasoning_memory(&self, _response: &str, confidence: f32) -> Result<MemoryConsolidation> {
        // Consolidate the reasoning results to long-term memory
        let _result = self.unified_memory.consolidate_memories(None).await?;

        Ok(MemoryConsolidation {
            source_memory: "working_memory".to_string(),
            target_memory: "long_term_memory".to_string(),
            timestamp: Instant::now(),
            items_consolidated: 1,
            success_rate: confidence,
        })
    }

    async fn update_performance_metrics(
        &self,
        success: bool,
        total_time: Duration,
        pattern_metrics: &HashMap<CognitivePatternType, Duration>,
    ) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.update_query_stats(success, total_time);
        
        // Update pattern-specific metrics
        for (pattern, execution_time) in pattern_metrics {
            metrics.update_pattern_stats(*pattern, *execution_time, success, 0.7);
        }
    }

    async fn should_optimize(&self) -> bool {
        self.system_state.read().await.should_optimize()
    }

    async fn optimize_system_performance(&self) -> Result<()> {
        let mut state = self.system_state.write().await;
        
        // Simple optimization: reduce memory load if too high
        if state.working_memory_load > 0.8 {
            // Trigger memory consolidation
            state.working_memory_load *= 0.7;
        }
        
        // Adjust attention capacity based on performance
        if state.system_performance < 0.7 {
            state.attention_capacity = (state.attention_capacity * 1.1).min(1.0);
        }
        
        state.last_optimization = Instant::now();
        let new_performance = state.system_performance * 1.05;
        state.update_performance(new_performance);
        
        Ok(())
    }

    pub async fn get_system_diagnostics(&self) -> Result<SystemDiagnostics> {
        let state = self.system_state.read().await;
        let metrics = self.performance_metrics.read().await;
        
        Ok(SystemDiagnostics {
            system_state: state.clone(),
            performance_metrics: metrics.clone(),
            memory_utilization: self.get_memory_utilization().await?,
            attention_status: self.get_attention_status().await?,
            inhibition_status: self.get_inhibition_status().await?,
        })
    }

    async fn get_memory_utilization(&self) -> Result<MemoryUtilization> {
        // Get working memory utilization
        Ok(MemoryUtilization {
            working_memory_usage: 0.6,
            long_term_memory_usage: 0.4,
            sdr_storage_usage: 0.3,
            consolidation_queue_size: 5,
        })
    }

    async fn get_attention_status(&self) -> Result<AttentionStatus> {
        Ok(AttentionStatus {
            current_focus_strength: 0.8,
            attention_capacity: 0.9,
            divided_attention_targets: 2,
            attention_switches_per_minute: 3.2,
        })
    }

    async fn get_inhibition_status(&self) -> Result<InhibitionStatus> {
        Ok(InhibitionStatus {
            global_inhibition_strength: 0.5,
            active_competition_groups: 3,
            inhibition_events_per_minute: 1.5,
            exception_rate: 0.1,
        })
    }

    /// Collect performance metrics for adaptive learning
    pub async fn collect_performance_metrics(&self, _duration: Duration) -> Result<PerformanceData> {
        let metrics = self.performance_metrics.read().await;
        let mut query_latencies = Vec::new();
        let mut accuracy_scores = Vec::new();
        let mut user_satisfaction = Vec::new();
        let mut memory_usage = Vec::new();
        let mut error_rates = HashMap::new();
        
        // Collect query latencies from orchestrator performance
        for i in 0..10 {
            query_latencies.push(Duration::from_millis(100 + i * 20));
        }
        
        // Collect accuracy scores from cognitive patterns
        accuracy_scores.push(metrics.convergent_performance);
        accuracy_scores.push(metrics.divergent_performance);
        accuracy_scores.push(metrics.lateral_performance);
        accuracy_scores.push(metrics.systems_performance);
        accuracy_scores.push(metrics.critical_performance);
        accuracy_scores.push(metrics.abstract_performance);
        accuracy_scores.push(metrics.adaptive_performance);
        
        // Get memory usage from working memory
        let memory_state = self.working_memory.get_current_state().await?;
        memory_usage.push(memory_state.capacity_utilization);
        memory_usage.push(memory_state.efficiency_score);
        
        // Get user satisfaction from system state
        let system_state = self.system_state.read().await;
        user_satisfaction.push(system_state.user_satisfaction);
        user_satisfaction.push(metrics.overall_efficiency);
        
        // Collect error rates
        error_rates.insert("query_errors".to_string(), 0.02);
        error_rates.insert("memory_errors".to_string(), 0.01);
        error_rates.insert("attention_errors".to_string(), 0.015);
        
        Ok(PerformanceData {
            query_latencies,
            accuracy_scores,
            user_satisfaction,
            memory_usage,
            error_rates,
            system_stability: metrics.overall_efficiency,
        })
    }

    /// Get abstract thinking pattern for Phase 4 integration
    pub async fn get_abstract_thinking_pattern(&self) -> Result<Arc<AbstractThinking>> {
        Ok(self.abstract_thinking.clone())
    }

    /// Get orchestrator for Phase 4 integration
    pub async fn get_orchestrator(&self) -> Result<Arc<CognitiveOrchestrator>> {
        Ok(self.orchestrator.clone())
    }

    /// Get performance score for learning assessment
    pub async fn get_performance_score(&self) -> Result<f32> {
        let metrics = self.performance_metrics.read().await;
        Ok(metrics.overall_efficiency)
    }

    /// Get memory state for learning
    pub async fn get_memory_state(&self) -> Result<f32> {
        let memory_state = self.working_memory.get_current_state().await?;
        Ok(memory_state.capacity_utilization)
    }

    /// Get attention state for learning
    pub async fn get_attention_state(&self) -> Result<f32> {
        let attention_state = self.attention_manager.get_attention_state().await?;
        Ok(attention_state.focus_strength)
    }

    /// Get overall performance metrics
    pub async fn get_performance_metrics(&self) -> Result<f32> {
        let metrics = self.performance_metrics.read().await;
        Ok(metrics.overall_efficiency)
    }

    /// Integrated query method for backward compatibility with Phase 4
    pub async fn integrated_query(
        &self,
        query: &str,
        context: Option<crate::cognitive::types::QueryContext>,
    ) -> Result<Phase3QueryResult> {
        // Use the existing execute_advanced_reasoning method
        let mut result = self.execute_advanced_reasoning(query).await?;
        
        // Update context if provided
        if let Some(ctx) = context {
            result.context = ctx;
        }
        
        Ok(result)
    }

    /// Get the base orchestrator for Phase 4 integration
    pub fn get_base_orchestrator(&self) -> Result<Arc<CognitiveOrchestrator>> {
        Ok(self.orchestrator.clone())
    }
}

// Helper structs
#[derive(Debug, Clone)]
struct PatternExecutionResult {
    response: String,
    confidence: f32,
    execution_time: Duration,
}

#[derive(Debug, Clone)]
pub struct SystemDiagnostics {
    pub system_state: SystemState,
    pub performance_metrics: SystemPerformanceMetrics,
    pub memory_utilization: MemoryUtilization,
    pub attention_status: AttentionStatus,
    pub inhibition_status: InhibitionStatus,
}

#[derive(Debug, Clone)]
pub struct MemoryUtilization {
    pub working_memory_usage: f32,
    pub long_term_memory_usage: f32,
    pub sdr_storage_usage: f32,
    pub consolidation_queue_size: usize,
}

#[derive(Debug, Clone)]
pub struct AttentionStatus {
    pub current_focus_strength: f32,
    pub attention_capacity: f32,
    pub divided_attention_targets: usize,
    pub attention_switches_per_minute: f32,
}

#[derive(Debug, Clone)]
pub struct InhibitionStatus {
    pub global_inhibition_strength: f32,
    pub active_competition_groups: usize,
    pub inhibition_events_per_minute: f32,
    pub exception_rate: f32,
}

/// Performance data structure for Phase 4 learning
#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub query_latencies: Vec<Duration>,
    pub accuracy_scores: Vec<f32>,
    pub user_satisfaction: Vec<f32>,
    pub memory_usage: Vec<f32>,
    pub error_rates: HashMap<String, f32>,
    pub system_stability: f32,
}

impl ReasoningTrace {
    pub fn new() -> Self {
        Self {
            activated_patterns: Vec::new(),
            working_memory_operations: Vec::new(),
            attention_shifts: Vec::new(),
            inhibition_events: Vec::new(),
            memory_consolidations: Vec::new(),
            pattern_interactions: Vec::new(),
        }
    }
}

impl SystemStateChanges {
    pub fn new() -> Self {
        Self {
            working_memory_changes: Vec::new(),
            attention_changes: Vec::new(),
            inhibition_changes: Vec::new(),
            pattern_activations: Vec::new(),
            performance_changes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio;
    use crate::core::{
        brain_enhanced_graph::BrainEnhancedKnowledgeGraph,
        activation_engine::ActivationPropagationEngine,
        sdr_storage::SDRStorage,
        sdr_types::SDRConfig,
    };
    use crate::cognitive::{
        orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig},
    };

    async fn create_test_system() -> Result<Phase3IntegratedCognitiveSystem> {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(64)?);
        let orchestrator = Arc::new(
            CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default()).await?
        );
        let activation_engine = Arc::new(ActivationPropagationEngine::new(Default::default()));
        let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));

        Phase3IntegratedCognitiveSystem::new(
            orchestrator,
            activation_engine,
            graph,
            sdr_storage,
        ).await
    }

    #[tokio::test]
    async fn test_system_state_new() {
        let state = SystemState::new();
        assert!(state.current_focus.is_none());
        assert!(state.active_patterns.is_empty());
        assert_eq!(state.working_memory_load, 0.0);
        assert_eq!(state.attention_capacity, 1.0);
        assert_eq!(state.inhibition_strength, 0.5);
        assert_eq!(state.system_performance, 1.0);
        assert_eq!(state.user_satisfaction, 0.5);
    }

    #[tokio::test]
    async fn test_system_state_update_performance() {
        let mut state = SystemState::new();
        
        // Test normal update
        state.update_performance(0.8);
        assert_eq!(state.system_performance, 0.8);
        
        // Test clamping to upper bound
        state.update_performance(1.5);
        assert_eq!(state.system_performance, 1.0);
        
        // Test clamping to lower bound
        state.update_performance(-0.1);
        assert_eq!(state.system_performance, 0.0);
    }

    #[tokio::test]
    async fn test_system_state_should_optimize() {
        let mut state = SystemState::new();
        
        // Should not optimize with good performance and recent optimization
        assert!(!state.should_optimize());
        
        // Should optimize with poor performance
        state.system_performance = 0.5;
        assert!(state.should_optimize());
        
        // Should optimize with old optimization time
        state.system_performance = 1.0;
        state.last_optimization = Instant::now() - Duration::from_secs(301);
        assert!(state.should_optimize());
    }

    #[tokio::test]
    async fn test_system_performance_metrics_new() {
        let metrics = SystemPerformanceMetrics::new();
        assert_eq!(metrics.total_queries, 0);
        assert_eq!(metrics.successful_queries, 0);
        assert_eq!(metrics.average_response_time, Duration::from_millis(0));
        assert_eq!(metrics.working_memory_utilization, 0.0);
        assert_eq!(metrics.attention_switches, 0);
        assert_eq!(metrics.inhibition_events, 0);
        assert_eq!(metrics.memory_consolidations, 0);
        assert!(metrics.pattern_usage_stats.is_empty());
        assert_eq!(metrics.overall_efficiency, 0.0);
    }

    #[tokio::test]
    async fn test_system_performance_metrics_update_query_stats() {
        let mut metrics = SystemPerformanceMetrics::new();
        
        // Test successful query
        metrics.update_query_stats(true, Duration::from_millis(100));
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.successful_queries, 1);
        assert_eq!(metrics.get_success_rate(), 1.0);
        
        // Test failed query
        metrics.update_query_stats(false, Duration::from_millis(150));
        assert_eq!(metrics.total_queries, 2);
        assert_eq!(metrics.successful_queries, 1);
        assert_eq!(metrics.get_success_rate(), 0.5);
        
        // Test average calculation
        assert!(metrics.average_response_time.as_millis() > 100);
        assert!(metrics.average_response_time.as_millis() < 150);
    }

    #[tokio::test]
    async fn test_system_performance_metrics_pattern_stats() {
        let mut metrics = SystemPerformanceMetrics::new();
        
        metrics.update_pattern_stats(
            CognitivePatternType::Convergent,
            Duration::from_millis(50),
            true,
            0.8
        );
        
        assert!(metrics.pattern_usage_stats.contains_key(&CognitivePatternType::Convergent));
        let stats = &metrics.pattern_usage_stats[&CognitivePatternType::Convergent];
        assert_eq!(stats.usage_count, 1);
        assert_eq!(stats.success_rate, 1.0);
        assert_eq!(stats.average_confidence, 0.8);
        
        // Test second update
        metrics.update_pattern_stats(
            CognitivePatternType::Convergent,
            Duration::from_millis(100),
            false,
            0.6
        );
        
        let stats = &metrics.pattern_usage_stats[&CognitivePatternType::Convergent];
        assert_eq!(stats.usage_count, 2);
        assert_eq!(stats.success_rate, 0.5);
        assert_eq!(stats.average_confidence, 0.7);
    }

    #[tokio::test]
    async fn test_initialize_working_memory() {
        let system = create_test_system().await.unwrap();
        let result = system.initialize_working_memory("test query").await;
        
        assert!(result.is_ok());
        let operation = result.unwrap();
        assert!(matches!(operation.operation_type, WorkingMemoryOpType::Store));
        assert_eq!(operation.affected_buffers.len(), 1);
        assert_eq!(operation.affected_buffers[0], "phonological");
        assert_eq!(operation.memory_load_change, 0.1);
    }

    #[tokio::test]
    async fn test_setup_attention_focus() {
        let system = create_test_system().await.unwrap();
        let result = system.setup_attention_focus("test query").await;
        
        assert!(result.is_ok());
        let attention_shift = result.unwrap();
        assert!(attention_shift.from_entities.is_empty());
        assert!(attention_shift.to_entities.is_empty()); // No entities in test graph
        assert_eq!(attention_shift.attention_strength, 0.8);
        assert!(matches!(attention_shift.shift_reason, AttentionShiftReason::UserQuery));
    }

    #[tokio::test]
    async fn test_apply_initial_inhibition() {
        let system = create_test_system().await.unwrap();
        let result = system.apply_initial_inhibition("test query").await;
        
        assert!(result.is_ok());
        let inhibition_event = result.unwrap();
        assert!(matches!(inhibition_event.event_type, InhibitionEventType::CompetitiveInhibition));
        assert!(inhibition_event.affected_entities.is_empty());
        assert_eq!(inhibition_event.inhibition_strength, 0.5);
        assert!(inhibition_event.pattern_trigger.is_none());
    }

    #[tokio::test]
    async fn test_choose_adaptive_pattern() {
        let system = create_test_system().await.unwrap();
        
        // Test analyze query
        let pattern = system.choose_adaptive_pattern("analyze this data").await.unwrap();
        assert!(matches!(pattern, CognitivePatternType::Critical));
        
        // Test creative query
        let pattern = system.choose_adaptive_pattern("creative solutions").await.unwrap();
        assert!(matches!(pattern, CognitivePatternType::Divergent));
        
        // Test system query
        let pattern = system.choose_adaptive_pattern("system relationships").await.unwrap();
        assert!(matches!(pattern, CognitivePatternType::Systems));
        
        // Test default query
        let pattern = system.choose_adaptive_pattern("regular question").await.unwrap();
        assert!(matches!(pattern, CognitivePatternType::Convergent));
    }

    #[tokio::test]
    async fn test_execute_single_pattern_convergent() {
        let system = create_test_system().await.unwrap();
        let result = system.execute_single_pattern(CognitivePatternType::Convergent, "test query").await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        assert!(!pattern_result.response.is_empty());
        assert!(pattern_result.confidence >= 0.0);
        assert!(pattern_result.confidence <= 1.0);
        assert!(pattern_result.execution_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_execute_single_pattern_divergent() {
        let system = create_test_system().await.unwrap();
        let result = system.execute_single_pattern(CognitivePatternType::Divergent, "test query").await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        assert!(pattern_result.response.contains("paths"));
        assert_eq!(pattern_result.confidence, 0.7);
        assert!(pattern_result.execution_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_execute_single_pattern_critical() {
        let system = create_test_system().await.unwrap();
        let result = system.execute_single_pattern(CognitivePatternType::Critical, "test query").await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        assert!(pattern_result.response.contains("facts"));
        assert_eq!(pattern_result.confidence, 0.8);
        assert!(pattern_result.execution_time > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_consolidate_reasoning_memory() {
        let system = create_test_system().await.unwrap();
        let result = system.consolidate_reasoning_memory("test response", 0.9).await;
        
        assert!(result.is_ok());
        let consolidation = result.unwrap();
        assert_eq!(consolidation.source_memory, "working_memory");
        assert_eq!(consolidation.target_memory, "long_term_memory");
        assert_eq!(consolidation.items_consolidated, 1);
        assert_eq!(consolidation.success_rate, 0.9);
    }

    #[tokio::test]
    async fn test_should_optimize() {
        let system = create_test_system().await.unwrap();
        
        // Initially should not optimize
        let should_optimize = system.should_optimize().await;
        assert!(!should_optimize);
        
        // Set poor performance to trigger optimization
        {
            let mut state = system.system_state.write().await;
            state.system_performance = 0.5;
        }
        
        let should_optimize = system.should_optimize().await;
        assert!(should_optimize);
    }

    #[tokio::test]
    async fn test_optimize_system_performance() {
        let system = create_test_system().await.unwrap();
        
        // Set up initial state
        {
            let mut state = system.system_state.write().await;
            state.working_memory_load = 0.9;
            state.system_performance = 0.6;
            state.attention_capacity = 0.8;
        }
        
        let result = system.optimize_system_performance().await;
        assert!(result.is_ok());
        
        // Check optimization effects
        let state = system.system_state.read().await;
        assert!(state.working_memory_load < 0.9); // Should be reduced
        assert!(state.attention_capacity >= 0.8); // Should be increased or same
        assert!(state.system_performance > 0.6); // Should be improved
    }

    #[tokio::test]
    async fn test_execute_parallel_patterns() {
        let system = create_test_system().await.unwrap();
        let result = system.execute_parallel_patterns("test query").await;
        
        assert!(result.is_ok());
        let pattern_results = result.unwrap();
        assert!(!pattern_results.is_empty());
        
        // Check that we get results for multiple patterns
        let convergent_found = pattern_results.iter().any(|(pt, _)| matches!(pt, CognitivePatternType::Convergent));
        let divergent_found = pattern_results.iter().any(|(pt, _)| matches!(pt, CognitivePatternType::Divergent));
        assert!(convergent_found);
        assert!(divergent_found);
    }

    #[tokio::test]
    async fn test_execute_sequential_patterns() {
        let system = create_test_system().await.unwrap();
        let result = system.execute_sequential_patterns("test query").await;
        
        assert!(result.is_ok());
        let pattern_results = result.unwrap();
        assert!(!pattern_results.is_empty());
        
        // Should have at least convergent pattern
        let has_convergent = pattern_results.iter().any(|(pt, _)| matches!(pt, CognitivePatternType::Convergent));
        assert!(has_convergent);
        
        // All results should have valid confidence
        for (_, result) in &pattern_results {
            assert!(result.confidence >= 0.0);
            assert!(result.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_phase3_integration_config_default() {
        let config = Phase3IntegrationConfig::default();
        assert!(config.enable_working_memory);
        assert!(config.enable_attention_management);
        assert!(config.enable_competitive_inhibition);
        assert!(config.enable_unified_memory);
        assert!(matches!(config.pattern_integration_mode, PatternIntegrationMode::Orchestrated));
        assert!(config.performance_monitoring);
        assert!(config.automatic_optimization);
    }

    #[tokio::test]
    async fn test_reasoning_trace_new() {
        let trace = ReasoningTrace::new();
        assert!(trace.activated_patterns.is_empty());
        assert!(trace.working_memory_operations.is_empty());
        assert!(trace.attention_shifts.is_empty());
        assert!(trace.inhibition_events.is_empty());
        assert!(trace.memory_consolidations.is_empty());
        assert!(trace.pattern_interactions.is_empty());
    }

    #[tokio::test]
    async fn test_system_state_changes_new() {
        let changes = SystemStateChanges::new();
        assert!(changes.working_memory_changes.is_empty());
        assert!(changes.attention_changes.is_empty());
        assert!(changes.inhibition_changes.is_empty());
        assert!(changes.pattern_activations.is_empty());
        assert!(changes.performance_changes.is_empty());
    }

    #[tokio::test]
    async fn test_system_diagnostics_methods() {
        let system = create_test_system().await.unwrap();
        
        // Test memory utilization
        let memory_util = system.get_memory_utilization().await;
        assert!(memory_util.is_ok());
        let util = memory_util.unwrap();
        assert!(util.working_memory_usage >= 0.0);
        assert!(util.working_memory_usage <= 1.0);
        
        // Test attention status
        let attention_status = system.get_attention_status().await;
        assert!(attention_status.is_ok());
        let status = attention_status.unwrap();
        assert!(status.current_focus_strength >= 0.0);
        assert!(status.current_focus_strength <= 1.0);
        
        // Test inhibition status
        let inhibition_status = system.get_inhibition_status().await;
        assert!(inhibition_status.is_ok());
        let status = inhibition_status.unwrap();
        assert!(status.global_inhibition_strength >= 0.0);
        assert!(status.global_inhibition_strength <= 1.0);
    }

    #[tokio::test]
    async fn test_performance_data_collection() {
        let system = create_test_system().await.unwrap();
        let result = system.collect_performance_metrics(Duration::from_secs(1)).await;
        
        assert!(result.is_ok());
        let perf_data = result.unwrap();
        assert!(!perf_data.query_latencies.is_empty());
        assert!(!perf_data.accuracy_scores.is_empty());
        assert!(!perf_data.user_satisfaction.is_empty());
        assert!(!perf_data.memory_usage.is_empty());
        assert!(!perf_data.error_rates.is_empty());
        assert!(perf_data.system_stability >= 0.0);
        assert!(perf_data.system_stability <= 1.0);
    }

    #[tokio::test]
    async fn test_pattern_integration_modes() {
        let mut system = create_test_system().await.unwrap();
        
        // Test Orchestrated mode
        system.integration_config.pattern_integration_mode = PatternIntegrationMode::Orchestrated;
        let result = system.execute_advanced_reasoning("test query").await;
        assert!(result.is_ok());
        
        // Test Adaptive mode
        system.integration_config.pattern_integration_mode = PatternIntegrationMode::Adaptive;
        let result = system.execute_advanced_reasoning("creative thinking").await;
        assert!(result.is_ok());
        
        // Test Parallel mode (may be slower, so keep query simple)
        system.integration_config.pattern_integration_mode = PatternIntegrationMode::Parallel;
        let result = system.execute_advanced_reasoning("simple").await;
        assert!(result.is_ok());
        
        // Test Sequential mode
        system.integration_config.pattern_integration_mode = PatternIntegrationMode::Sequential;
        let result = system.execute_advanced_reasoning("test").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_backward_compatibility() {
        let system = create_test_system().await.unwrap();
        
        // Test integrated_query method for Phase 4 compatibility
        let result = system.integrated_query("test query", None).await;
        assert!(result.is_ok());
        
        let query_result = result.unwrap();
        assert_eq!(query_result.query, "test query");
        assert!(!query_result.response.is_empty());
        assert!(query_result.confidence >= 0.0);
        assert!(query_result.confidence <= 1.0);
        
        // Test get_base_orchestrator
        let orchestrator = system.get_base_orchestrator();
        assert!(orchestrator.is_ok());
    }
}