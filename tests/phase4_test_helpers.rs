use anyhow::Result;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use uuid::Uuid;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedGraph,
    brain_types::{EntityKey, BrainInspiredEntity, ActivationPattern, RelationshipType},
    sdr_storage::{SDRStorage, SDRParameters, SDRPattern},
    activation_engine::ActivationPropagationEngine,
};

use llmkg::cognitive::{
    working_memory::WorkingMemorySystem,
    attention_manager::{AttentionManager, AttentionConfig, AttentionType},
    inhibitory_logic::{CompetitiveInhibitionSystem, InhibitionConfig, CompetitionGroup},
    abstract_pattern::{AbstractThinking, AbstractionConfig},
    orchestrator::{CognitiveOrchestrator, OrchestratorConfig},
    phase3_integration::IntegratedCognitiveSystem,
    types::*,
};

use llmkg::learning::{
    hebbian::HebbianLearningEngine,
    homeostasis::{SynapticHomeostasis, HomeostasisConfig},
    optimization_agent::GraphOptimizationAgent,
    adaptive_learning::AdaptiveLearningSystem,
    phase4_integration::{Phase4LearningSystem, Phase4Config},
    types::*,
};

/// Create a test brain graph with basic structure
pub async fn create_test_brain_graph() -> Result<BrainEnhancedGraph> {
    let mut graph = BrainEnhancedGraph::new().await?;
    
    // Add some basic entities for testing
    let concepts = vec![
        ("AI", "Artificial Intelligence concept"),
        ("ML", "Machine Learning subconcept"),
        ("DL", "Deep Learning subconcept"),
        ("NN", "Neural Network implementation"),
        ("Knowledge", "Abstract knowledge concept"),
        ("Learning", "Learning process"),
        ("Memory", "Memory system"),
    ];
    
    let mut entity_map = HashMap::new();
    
    for (name, description) in concepts {
        let entity = BrainInspiredEntity {
            key: EntityKey::new(),
            name: name.to_string(),
            entity_type: "concept".to_string(),
            attributes: HashMap::from([
                ("description".to_string(), description.to_string()),
                ("category".to_string(), "test_entity".to_string()),
            ]),
            semantic_embedding: vec![0.1; 768], // Simplified embedding
            activation_pattern: ActivationPattern {
                current_activation: 0.5,
                activation_history: vec![0.5],
                decay_rate: 0.1,
                last_activated: SystemTime::now(),
            },
            temporal_aspects: Default::default(),
            ingestion_time: SystemTime::now(),
        };
        
        entity_map.insert(name, entity.key);
        graph.insert_entity(entity).await?;
    }
    
    // Add relationships
    let relationships = vec![
        ("AI", "ML", RelationshipType::Contains, 0.9),
        ("ML", "DL", RelationshipType::Contains, 0.85),
        ("DL", "NN", RelationshipType::ImplementedBy, 0.8),
        ("AI", "Knowledge", RelationshipType::RelatedTo, 0.7),
        ("Knowledge", "Learning", RelationshipType::EnabledBy, 0.75),
        ("Learning", "Memory", RelationshipType::RequiresResource, 0.8),
        ("Memory", "Knowledge", RelationshipType::Stores, 0.85),
    ];
    
    for (source, target, rel_type, weight) in relationships {
        if let (Some(&source_key), Some(&target_key)) = (entity_map.get(source), entity_map.get(target)) {
            graph.insert_relationship(llmkg::core::brain_types::BrainInspiredRelationship {
                source: source_key,
                target: target_key,
                relation_type: rel_type,
                weight,
                is_inhibitory: false,
                temporal_decay: 0.01,
                last_strengthened: SystemTime::now(),
                activation_count: 1,
                creation_time: SystemTime::now(),
                ingestion_time: SystemTime::now(),
            }).await?;
        }
    }
    
    Ok(graph)
}

/// Create a test activation engine
pub async fn create_test_activation_engine() -> Result<ActivationPropagationEngine> {
    let config = llmkg::core::activation_engine::ActivationConfig {
        propagation_threshold: 0.3,
        max_propagation_depth: 5,
        decay_factor: 0.1,
        activation_floor: 0.0,
        activation_ceiling: 1.0,
        temporal_window: Duration::from_secs(300),
    };
    
    ActivationPropagationEngine::with_config(config).await
}

/// Create a test inhibition system
pub async fn create_test_inhibition_system() -> Result<CompetitiveInhibitionSystem> {
    let config = InhibitionConfig {
        default_competition_strength: 0.5,
        mutual_inhibition_factor: 0.3,
        lateral_inhibition_range: 2,
        winner_take_all_threshold: 0.8,
        inhibition_decay_rate: 0.05,
        min_activation_threshold: 0.1,
    };
    
    let mut system = CompetitiveInhibitionSystem::with_config(config).await?;
    
    // Add some competition groups
    system.create_competition_group(
        "concepts",
        vec![],  // Will be populated dynamically
        0.6,
        llmkg::cognitive::inhibitory_logic::CompetitionType::WinnerTakeAll,
    ).await?;
    
    Ok(system)
}

/// Create a test attention manager
pub async fn create_test_attention_manager() -> Result<AttentionManager> {
    let config = AttentionConfig {
        max_focus_items: 7,
        focus_decay_rate: 0.1,
        attention_shift_threshold: 0.7,
        sustained_attention_bonus: 0.2,
        divided_attention_penalty: 0.3,
        attention_restoration_rate: 0.05,
    };
    
    AttentionManager::with_config(config).await
}

/// Create a test working memory system
pub async fn create_test_working_memory() -> Result<WorkingMemorySystem> {
    let activation_engine = Arc::new(ActivationPropagationEngine::new(
        llmkg::core::activation_engine::ActivationConfig::default()
    ));
    let sdr_storage = Arc::new(SDRStorage::new(SDRParameters::default())?);
    
    WorkingMemorySystem::new(activation_engine, sdr_storage).await
}

/// Create a test SDR storage
pub async fn create_test_sdr_storage() -> Result<SDRStorage> {
    let params = SDRParameters {
        dimensions: 2048,
        sparsity: 0.02,
        similarity_threshold: 0.8,
        learning_rate: 0.1,
        boost_strength: 2.0,
        duty_cycle_period: 1000,
    };
    
    SDRStorage::with_parameters(params).await
}

/// Create a test abstract thinking component
pub async fn create_test_abstract_thinking() -> Result<AbstractThinking> {
    let config = AbstractionConfig {
        abstraction_levels: 5,
        pattern_threshold: 0.6,
        generalization_rate: 0.3,
        specialization_penalty: 0.2,
        analogy_strength_threshold: 0.7,
        max_abstraction_depth: 3,
    };
    
    let brain_graph = Arc::new(create_test_brain_graph().await?);
    AbstractThinking::with_config(brain_graph, config).await
}

/// Create a test cognitive orchestrator
pub async fn create_test_orchestrator() -> Result<CognitiveOrchestrator> {
    let config = OrchestratorConfig {
        pattern_timeout: Duration::from_secs(5),
        ensemble_threshold: 0.6,
        conflict_resolution_strategy: llmkg::cognitive::orchestrator::ConflictResolution::WeightedVoting,
        max_parallel_patterns: 3,
        pattern_selection_strategy: llmkg::cognitive::orchestrator::SelectionStrategy::ContextAware,
        performance_tracking_window: Duration::from_secs(3600),
    };
    
    CognitiveOrchestrator::with_config(config).await
}

/// Create a test Hebbian engine with dependencies
pub async fn create_test_hebbian_engine() -> Result<HebbianLearningEngine> {
    let brain_graph = Arc::new(create_test_brain_graph().await?);
    let activation_engine = Arc::new(create_test_activation_engine().await?);
    let inhibition_system = Arc::new(create_test_inhibition_system().await?);
    
    HebbianLearningEngine::new(brain_graph, activation_engine, inhibition_system).await
}

/// Create a test optimization agent
pub async fn create_test_optimization_agent() -> Result<GraphOptimizationAgent> {
    let brain_graph = Arc::new(create_test_brain_graph().await?);
    let sdr_storage = Arc::new(create_test_sdr_storage().await?);
    let abstract_thinking = Arc::new(create_test_abstract_thinking().await?);
    let orchestrator = Arc::new(create_test_orchestrator().await?);
    let hebbian_engine = Arc::new(Mutex::new(create_test_hebbian_engine().await?));
    
    GraphOptimizationAgent::new(
        brain_graph,
        sdr_storage,
        abstract_thinking,
        orchestrator,
        hebbian_engine,
    ).await
}

/// Create a test integrated cognitive system (Phase 3)
pub async fn create_test_integrated_cognitive_system() -> Result<IntegratedCognitiveSystem> {
    let brain_graph = Arc::new(create_test_brain_graph().await?);
    let sdr_storage = Arc::new(create_test_sdr_storage().await?);
    let activation_engine = Arc::new(create_test_activation_engine().await?);
    let working_memory = Arc::new(create_test_working_memory().await?);
    let attention_manager = Arc::new(create_test_attention_manager().await?);
    let inhibition_system = Arc::new(create_test_inhibition_system().await?);
    
    IntegratedCognitiveSystem::new(
        brain_graph,
        sdr_storage,
        activation_engine,
        working_memory,
        attention_manager,
        inhibition_system,
    ).await
}

/// Create a test adaptive learning system
pub async fn create_test_adaptive_learning_system() -> Result<AdaptiveLearningSystem> {
    let integrated_cognitive_system = Arc::new(create_test_integrated_cognitive_system().await?);
    let working_memory = Arc::new(create_test_working_memory().await?);
    let attention_manager = Arc::new(create_test_attention_manager().await?);
    let orchestrator = Arc::new(create_test_orchestrator().await?);
    let hebbian_engine = Arc::new(Mutex::new(create_test_hebbian_engine().await?));
    let optimization_agent = Arc::new(Mutex::new(create_test_optimization_agent().await?));
    
    AdaptiveLearningSystem::new(
        integrated_cognitive_system,
        working_memory,
        attention_manager,
        orchestrator,
        hebbian_engine,
        optimization_agent,
    ).await
}

/// Create a test Phase 4 learning system
pub async fn create_test_phase4_learning_system() -> Result<Phase4LearningSystem> {
    let integrated_cognitive_system = Arc::new(create_test_integrated_cognitive_system().await?);
    let brain_graph = Arc::new(create_test_brain_graph().await?);
    let sdr_storage = Arc::new(create_test_sdr_storage().await?);
    let activation_engine = Arc::new(create_test_activation_engine().await?);
    let attention_manager = Arc::new(create_test_attention_manager().await?);
    let working_memory = Arc::new(create_test_working_memory().await?);
    let inhibition_system = Arc::new(create_test_inhibition_system().await?);
    let orchestrator = Arc::new(create_test_orchestrator().await?);
    
    Phase4LearningSystem::new(
        integrated_cognitive_system,
        brain_graph,
        sdr_storage,
        activation_engine,
        attention_manager,
        working_memory,
        inhibition_system,
        orchestrator,
    ).await
}

/// Create a test Phase 4 cognitive system
pub async fn create_test_phase4_cognitive_system() -> Result<llmkg::cognitive::phase4_integration::Phase4CognitiveSystem> {
    let phase3_system = Arc::new(create_test_integrated_cognitive_system().await?);
    let phase4_learning = Arc::new(create_test_phase4_learning_system().await?);
    
    llmkg::cognitive::phase4_integration::Phase4CognitiveSystem::new(
        phase3_system,
        phase4_learning,
    ).await
}

/// Generate synthetic activation events for testing
pub fn generate_test_activation_events(num_events: usize) -> Vec<ActivationEvent> {
    let mut rng = StdRng::seed_from_u64(42);
    let base_time = std::time::Instant::now();
    let mut events = Vec::new();
    
    for i in 0..num_events {
        let event = ActivationEvent {
            entity_key: EntityKey::new(),
            activation_strength: rng.gen_range(0.3..0.9),
            timestamp: base_time + Duration::from_millis(i as u64 * 100),
            context: ActivationContext {
                query_id: format!("test_query_{}", i / 10),
                cognitive_pattern: match rng.gen_range(0..5) {
                    0 => CognitivePatternType::Convergent,
                    1 => CognitivePatternType::Divergent,
                    2 => CognitivePatternType::Transform,
                    3 => CognitivePatternType::EmergencePattern,
                    _ => CognitivePatternType::AbstractThinking,
                },
                user_session: Some(format!("test_session_{}", i / 50)),
                outcome_quality: Some(rng.gen_range(0.5..0.95)),
            },
        };
        events.push(event);
    }
    
    events
}

/// Generate a test learning context
pub fn generate_test_learning_context() -> LearningContext {
    LearningContext {
        performance_pressure: 0.6,
        user_satisfaction_level: 0.75,
        learning_urgency: 0.5,
        session_id: "test_session".to_string(),
        learning_goals: vec![
            LearningGoal {
                goal_type: LearningGoalType::PerformanceImprovement,
                target_improvement: 0.1,
                deadline: Some(SystemTime::now() + Duration::from_secs(3600)),
            },
            LearningGoal {
                goal_type: LearningGoalType::MemoryEfficiency,
                target_improvement: 0.15,
                deadline: None,
            },
        ],
    }
}

/// Create test user feedback
pub fn generate_test_user_feedback(num_samples: usize) -> Vec<UserFeedback> {
    let mut rng = StdRng::seed_from_u64(123);
    let mut feedback = Vec::new();
    
    for i in 0..num_samples {
        feedback.push(UserFeedback {
            feedback_id: Uuid::new_v4(),
            session_id: format!("session_{}", i / 10),
            query_id: format!("query_{}", i),
            satisfaction_score: rng.gen_range(0.5..0.95),
            response_quality: rng.gen_range(0.6..0.9),
            response_speed: rng.gen_range(0.4..0.85),
            accuracy_rating: rng.gen_range(0.65..0.95),
            feedback_text: if rng.gen_bool(0.3) {
                Some("Test feedback text".to_string())
            } else {
                None
            },
            timestamp: SystemTime::now() - Duration::from_secs(i as u64 * 60),
        });
    }
    
    feedback
}

/// Performance measurement utilities
pub struct PerformanceMeasurer {
    start_time: std::time::Instant,
    checkpoints: Vec<(String, std::time::Instant)>,
}

impl PerformanceMeasurer {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            checkpoints: Vec::new(),
        }
    }
    
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), std::time::Instant::now()));
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Total time: {:?}\n", self.start_time.elapsed()));
        
        let mut last_time = self.start_time;
        for (name, time) in &self.checkpoints {
            let delta = time.duration_since(last_time);
            report.push_str(&format!("  {}: {:?}\n", name, delta));
            last_time = *time;
        }
        
        report
    }
}

/// Memory usage tracker
pub fn measure_memory_usage() -> f32 {
    // Simplified memory measurement
    // In a real implementation, this would use system APIs
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // Allocate a small test block to measure overhead
    let layout = Layout::from_size_align(1024, 8).unwrap();
    unsafe {
        let ptr = System.alloc(layout);
        System.dealloc(ptr, layout);
    }
    
    // Return a simulated memory usage percentage
    0.45 // 45% usage
}

/// Test data validation utilities
pub fn validate_learning_update(update: &LearningUpdate) -> Result<()> {
    // Validate learning efficiency is reasonable
    if update.learning_efficiency < 0.0 || update.learning_efficiency > 1.0 {
        return Err(anyhow::anyhow!("Learning efficiency out of bounds: {}", update.learning_efficiency));
    }
    
    // Validate weight changes
    for change in &update.strengthened_connections {
        if change.new_weight < change.old_weight {
            return Err(anyhow::anyhow!("Strengthened connection has lower weight"));
        }
    }
    
    for change in &update.weakened_connections {
        if change.new_weight > change.old_weight {
            return Err(anyhow::anyhow!("Weakened connection has higher weight"));
        }
    }
    
    Ok(())
}

/// Create a populated Phase 4 system for integration testing
pub async fn create_populated_phase4_system() -> Result<llmkg::cognitive::phase4_integration::Phase4CognitiveSystem> {
    let system = create_test_phase4_cognitive_system().await?;
    
    // Populate with initial knowledge
    let mut rng = StdRng::seed_from_u64(999);
    
    // Add diverse entities
    for i in 0..100 {
        let entity = BrainInspiredEntity {
            key: EntityKey::new(),
            name: format!("Entity_{}", i),
            entity_type: ["concept", "instance", "property", "process"][i % 4].to_string(),
            attributes: HashMap::from([
                ("index".to_string(), i.to_string()),
                ("category".to_string(), format!("category_{}", i % 10)),
            ]),
            semantic_embedding: (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            activation_pattern: ActivationPattern {
                current_activation: rng.gen_range(0.1..0.9),
                activation_history: vec![0.5],
                decay_rate: 0.1,
                last_activated: SystemTime::now(),
            },
            temporal_aspects: Default::default(),
            ingestion_time: SystemTime::now(),
        };
        
        system.phase3_system.brain_graph.insert_entity(entity).await?;
    }
    
    // Run initial learning to establish baseline
    let initial_events = generate_test_activation_events(500);
    let learning_context = generate_test_learning_context();
    
    system.phase4_learning.hebbian_engine.lock().unwrap()
        .apply_hebbian_learning(initial_events, learning_context).await?;
    
    Ok(system)
}