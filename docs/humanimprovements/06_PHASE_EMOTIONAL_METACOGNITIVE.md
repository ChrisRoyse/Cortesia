# Phase 6: Emotional & Metacognitive Systems

## Overview
**Duration**: 4 weeks  
**Goal**: Add emotional influence on memory and metacognitive monitoring capabilities  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-5 completion  

## Week 21: Emotional Memory System

### Task 21.1: Emotion-Memory Integration
**File**: `src/emotional/emotional_memory.rs` (new file)
```rust
pub struct EmotionalMemory {
    emotion_tagger: EmotionTagger,
    mood_congruence: MoodCongruenceSystem,
    emotional_consolidation: EmotionalConsolidation,
    affect_regulation: AffectRegulation,
}

pub struct EmotionalTag {
    valence: f32,          // -1.0 (negative) to 1.0 (positive)
    arousal: f32,          // 0.0 (calm) to 1.0 (excited)
    dominance: f32,        // 0.0 (submissive) to 1.0 (dominant)
    
    // Basic emotions
    emotions: BasicEmotions,
    
    // Complex emotions
    complex_emotions: Vec<ComplexEmotion>,
    
    // Intensity and duration
    intensity: f32,
    duration: Duration,
    peak_moment: Option<Instant>,
}

pub struct BasicEmotions {
    joy: f32,
    sadness: f32,
    anger: f32,
    fear: f32,
    surprise: f32,
    disgust: f32,
    trust: f32,
    anticipation: f32,
}

pub struct ComplexEmotion {
    name: String,  // "nostalgia", "schadenfreude", etc.
    composition: HashMap<String, f32>,  // Basic emotion mixture
    cultural_specificity: f32,
}

impl EmotionalMemory {
    pub fn tag_memory_with_emotion(&mut self, 
        memory: &mut Memory,
        context: &EmotionalContext
    ) {
        let emotional_tag = self.emotion_tagger.analyze_emotion(context);
        
        // Emotion influences encoding strength
        memory.encoding_strength *= self.calculate_emotional_boost(&emotional_tag);
        
        // Tag memory
        memory.emotional_tag = Some(emotional_tag);
        
        // Mark for priority consolidation if highly emotional
        if emotional_tag.intensity > EMOTIONAL_INTENSITY_THRESHOLD {
            memory.consolidation_priority = ConsolidationPriority::High;
        }
    }
    
    pub fn mood_congruent_retrieval(&self,
        current_mood: &EmotionalState,
        memories: &[Memory]
    ) -> Vec<Memory> {
        // Memories matching current mood are more accessible
        let mut scored_memories = memories.iter()
            .map(|m| {
                let congruence = self.mood_congruence.calculate_congruence(
                    current_mood,
                    &m.emotional_tag
                );
                (m.clone(), congruence)
            })
            .collect::<Vec<_>>();
            
        scored_memories.sort_by_key(|(_, score)| OrderedFloat(-score));
        
        scored_memories.into_iter()
            .map(|(m, _)| m)
            .collect()
    }
    
    fn calculate_emotional_boost(&self, tag: &EmotionalTag) -> f32 {
        // High arousal enhances memory
        let arousal_boost = tag.arousal * 0.5;
        
        // Extreme valence (very positive or negative) enhances memory
        let valence_boost = tag.valence.abs() * 0.3;
        
        // Surprise enhances memory
        let surprise_boost = tag.emotions.surprise * 0.4;
        
        1.0 + arousal_boost + valence_boost + surprise_boost
    }
}
```

### Task 21.2: Affect Regulation and Memory
**File**: `src/emotional/affect_regulation.rs` (new file)
```rust
pub struct AffectRegulation {
    regulation_strategies: Vec<RegulationStrategy>,
    emotional_goals: EmotionalGoals,
    regulation_history: RegulationHistory,
}

pub enum RegulationStrategy {
    Suppression {
        target_emotion: String,
        effectiveness: f32,
        cognitive_cost: f32,
    },
    Reappraisal {
        reframe_function: Box<dyn Fn(&Memory) -> Memory>,
        success_rate: f32,
    },
    Distraction {
        alternative_focus: Vec<AttentionTarget>,
        duration: Duration,
    },
    SocialSharing {
        increases_intensity: bool,
        social_bonds_strengthened: f32,
    },
    Rumination {
        amplification_factor: f32,
        negative_only: bool,
    },
}

impl AffectRegulation {
    pub fn regulate_emotional_memory(&mut self,
        memory: &mut Memory,
        strategy: &RegulationStrategy
    ) -> RegulationOutcome {
        match strategy {
            RegulationStrategy::Suppression { target_emotion, effectiveness, cognitive_cost } => {
                // Suppression weakens emotional intensity but costs cognitive resources
                if let Some(tag) = &mut memory.emotional_tag {
                    tag.intensity *= 1.0 - effectiveness;
                    
                    // But suppression can backfire
                    if rand::random::<f32>() < REBOUND_PROBABILITY {
                        tag.intensity *= REBOUND_AMPLIFICATION;
                    }
                }
                
                RegulationOutcome::Suppressed {
                    success: true,
                    cognitive_depletion: *cognitive_cost,
                    rebound_risk: REBOUND_PROBABILITY,
                }
            },
            
            RegulationStrategy::Reappraisal { reframe_function, success_rate } => {
                if rand::random::<f32>() < *success_rate {
                    // Reappraisal changes the memory itself
                    *memory = reframe_function(memory);
                    
                    RegulationOutcome::Reappraised {
                        new_valence: memory.emotional_tag.as_ref().map(|t| t.valence),
                        memory_modified: true,
                    }
                } else {
                    RegulationOutcome::Failed
                }
            },
            
            _ => RegulationOutcome::InProgress,
        }
    }
    
    pub fn select_regulation_strategy(&self,
        current_state: &EmotionalState,
        goal_state: &EmotionalState
    ) -> RegulationStrategy {
        // Choose strategy based on:
        // - Current emotional intensity
        // - Distance to goal state
        // - Available cognitive resources
        // - Past strategy effectiveness
        
        self.regulation_strategies.iter()
            .max_by_key(|s| {
                let effectiveness = self.predict_effectiveness(s, current_state, goal_state);
                let cost = self.calculate_cost(s, current_state);
                OrderedFloat(effectiveness / cost)
            })
            .cloned()
            .unwrap_or(RegulationStrategy::Distraction {
                alternative_focus: vec![],
                duration: Duration::from_secs(300),
            })
    }
}
```

### Task 21.3: Emotional Learning
**File**: `src/emotional/emotional_learning.rs` (new file)
```rust
pub struct EmotionalLearning {
    fear_conditioning: FearConditioning,
    reward_learning: RewardLearning,
    emotional_schemas: HashMap<SchemaId, EmotionalSchema>,
}

pub struct FearConditioning {
    conditioned_stimuli: HashMap<Stimulus, FearResponse>,
    extinction_trials: HashMap<Stimulus, u32>,
    context_dependency: HashMap<(Stimulus, Context), f32>,
}

impl FearConditioning {
    pub fn condition_fear(&mut self,
        neutral_stimulus: &Stimulus,
        aversive_outcome: &Outcome,
        context: &Context
    ) {
        let fear_response = FearResponse {
            intensity: aversive_outcome.severity,
            physiological_arousal: self.calculate_arousal(aversive_outcome),
            behavioral_response: self.determine_response(aversive_outcome),
            formed_at: Instant::now(),
        };
        
        self.conditioned_stimuli.insert(neutral_stimulus.clone(), fear_response);
        
        // Context-dependent fear
        let context_key = (neutral_stimulus.clone(), context.clone());
        self.context_dependency.insert(context_key, 1.0);
    }
    
    pub fn extinction_learning(&mut self,
        stimulus: &Stimulus,
        safe_exposures: u32
    ) {
        if let Some(fear_response) = self.conditioned_stimuli.get_mut(stimulus) {
            // Extinction doesn't erase fear, it inhibits it
            let extinction_strength = (safe_exposures as f32).ln() / 10.0;
            fear_response.inhibition = extinction_strength.min(0.9);
            
            self.extinction_trials.insert(stimulus.clone(), safe_exposures);
        }
    }
    
    pub fn spontaneous_recovery(&mut self, stimulus: &Stimulus) -> Option<FearResponse> {
        // Fear can return after time
        if let Some(extinction_trials) = self.extinction_trials.get(stimulus) {
            let time_since_extinction = self.get_time_since_extinction(stimulus);
            let recovery_probability = time_since_extinction.as_secs_f32() / 86400.0; // Daily increase
            
            if rand::random::<f32>() < recovery_probability {
                self.conditioned_stimuli.get(stimulus).cloned()
            } else {
                None
            }
        } else {
            self.conditioned_stimuli.get(stimulus).cloned()
        }
    }
}

pub struct EmotionalSchema {
    trigger_pattern: TriggerPattern,
    emotional_response: EmotionalResponse,
    coping_strategies: Vec<CopingStrategy>,
    formation_history: Vec<SchemaFormationEvent>,
    modification_resistance: f32,
}
```

## Week 22: Metacognitive Monitoring

### Task 22.1: Metamemory System
**File**: `src/metacognitive/metamemory.rs` (new file)
```rust
pub struct Metamemory {
    monitoring: MetamemoryMonitoring,
    control: MetamemoryControl,
    knowledge: MetamemoryKnowledge,
}

pub struct MetamemoryMonitoring {
    // Judgments about memory
    feeling_of_knowing: FeelingOfKnowing,
    tip_of_tongue: TipOfTongueDetector,
    confidence_judgments: ConfidenceJudgments,
    source_monitoring: SourceMonitoring,
}

pub struct FeelingOfKnowing {
    cue_familiarity_threshold: f32,
    partial_activation_detector: PartialActivationDetector,
}

impl FeelingOfKnowing {
    pub fn assess_feeling_of_knowing(&self,
        cue: &RetrievalCue,
        memory_state: &MemoryState
    ) -> FOKJudgment {
        // Check cue familiarity
        let cue_familiarity = self.calculate_cue_familiarity(cue, memory_state);
        
        // Detect partial activation
        let partial_activation = self.partial_activation_detector.detect(cue, memory_state);
        
        // Competition from similar memories
        let competition = self.assess_competition(cue, memory_state);
        
        FOKJudgment {
            strength: (cue_familiarity * 0.4 + partial_activation * 0.4 - competition * 0.2).max(0.0),
            confidence: self.calculate_confidence(cue_familiarity, partial_activation),
            predicted_retrieval_success: self.predict_success(cue_familiarity, partial_activation, competition),
        }
    }
}

pub struct ConfidenceJudgments {
    calibration_history: CalibrationHistory,
    overconfidence_bias: f32,
    dunning_kruger_effect: bool,
}

impl ConfidenceJudgments {
    pub fn judge_memory_accuracy(&self,
        memory: &Memory,
        retrieval_context: &RetrievalContext
    ) -> ConfidenceJudgment {
        let objective_strength = memory.current_strength;
        let retrieval_fluency = retrieval_context.retrieval_time.as_secs_f32().recip();
        let vividness = memory.vividness;
        
        // People use fluency as a cue for accuracy
        let fluency_based_confidence = retrieval_fluency * 0.5;
        
        // Vividness increases confidence (but not necessarily accuracy)
        let vividness_based_confidence = vividness * 0.3;
        
        // Actual memory strength contribution
        let strength_based_confidence = objective_strength * 0.2;
        
        let raw_confidence = fluency_based_confidence + vividness_based_confidence + strength_based_confidence;
        
        // Apply individual biases
        let biased_confidence = raw_confidence * (1.0 + self.overconfidence_bias);
        
        ConfidenceJudgment {
            confidence: biased_confidence.min(1.0),
            actual_accuracy: objective_strength,
            calibration_error: (biased_confidence - objective_strength).abs(),
            basis: ConfidenceBasis {
                fluency_contribution: fluency_based_confidence,
                vividness_contribution: vividness_based_confidence,
                strength_contribution: strength_based_confidence,
            },
        }
    }
    
    pub fn update_calibration(&mut self,
        judgment: &ConfidenceJudgment,
        actual_accuracy: f32
    ) {
        self.calibration_history.add_judgment(judgment.confidence, actual_accuracy);
        
        // Adjust bias based on calibration
        let calibration_error = judgment.confidence - actual_accuracy;
        self.overconfidence_bias *= 0.9; // Slow learning
        self.overconfidence_bias += calibration_error * 0.1;
    }
}
```

### Task 22.2: Metacognitive Control
**File**: `src/metacognitive/metacognitive_control.rs` (new file)
```rust
pub struct MetacognitiveControl {
    strategy_selector: StrategySelector,
    effort_allocator: EffortAllocator,
    monitoring_scheduler: MonitoringScheduler,
}

pub struct StrategySelector {
    available_strategies: Vec<MemoryStrategy>,
    strategy_effectiveness: HashMap<(StrategyId, ContextType), f32>,
    adaptation_rate: f32,
}

impl StrategySelector {
    pub fn select_encoding_strategy(&self,
        material: &Material,
        goals: &LearningGoals,
        resources: &CognitiveResources
    ) -> EncodingStrategy {
        let strategies = vec![
            EncodingStrategy::Rehearsal { 
                type_: RehearsalType::Maintenance 
            },
            EncodingStrategy::Elaboration { 
                depth: ElaborationDepth::Semantic 
            },
            EncodingStrategy::Organization { 
                method: OrganizationMethod::Hierarchical 
            },
            EncodingStrategy::Imagery { 
                vividness: ImageryVividness::High 
            },
            EncodingStrategy::SelfReference { 
                personal_relevance: self.assess_relevance(material) 
            },
        ];
        
        strategies.into_iter()
            .max_by_key(|s| {
                let effectiveness = self.predict_effectiveness(s, material, goals);
                let cost = self.calculate_cognitive_cost(s, resources);
                OrderedFloat(effectiveness / cost)
            })
            .unwrap()
    }
    
    pub fn select_retrieval_strategy(&self,
        cue: &RetrievalCue,
        fok: &FOKJudgment
    ) -> RetrievalStrategy {
        if fok.strength > HIGH_FOK_THRESHOLD {
            // Strong feeling of knowing - try direct retrieval
            RetrievalStrategy::DirectRetrieval
        } else if fok.strength > MEDIUM_FOK_THRESHOLD {
            // Moderate FOK - use generation strategies
            RetrievalStrategy::Generate { 
                constraint: self.infer_constraints(cue) 
            }
        } else {
            // Low FOK - need recognition or external cues
            RetrievalStrategy::Recognition
        }
    }
}

pub struct EffortAllocator {
    total_resources: CognitiveResources,
    task_priorities: BinaryHeap<PrioritizedTask>,
    effort_value_function: Box<dyn Fn(f32, f32) -> f32>,
}

impl EffortAllocator {
    pub fn allocate_study_time(&mut self,
        items: &[StudyItem]
    ) -> HashMap<ItemId, Duration> {
        let mut allocations = HashMap::new();
        let total_time = self.total_resources.time_available;
        
        // Calculate value of studying each item
        let item_values: Vec<(ItemId, f32)> = items.iter()
            .map(|item| {
                let current_strength = item.current_memory_strength;
                let goal_strength = item.target_strength;
                let learning_rate = item.estimated_learning_rate;
                
                // Diminishing returns for well-learned items
                let marginal_value = (self.effort_value_function)(
                    current_strength,
                    goal_strength
                );
                
                (item.id, marginal_value * item.importance)
            })
            .collect();
        
        // Allocate time proportionally to value
        let total_value: f32 = item_values.iter().map(|(_, v)| v).sum();
        
        for (item_id, value) in item_values {
            let time_fraction = value / total_value;
            let allocated_time = Duration::from_secs_f32(
                total_time.as_secs_f32() * time_fraction
            );
            allocations.insert(item_id, allocated_time);
        }
        
        allocations
    }
}
```

### Task 22.3: Self-Monitoring and Awareness
**File**: `src/metacognitive/self_awareness.rs` (new file)
```rust
pub struct SelfAwareness {
    memory_self_knowledge: MemorySelfKnowledge,
    performance_monitor: PerformanceMonitor,
    limitation_awareness: LimitationAwareness,
}

pub struct MemorySelfKnowledge {
    // Knowledge about own memory abilities
    strengths: Vec<MemoryStrength>,
    weaknesses: Vec<MemoryWeakness>,
    typical_errors: Vec<ErrorPattern>,
    effective_strategies: Vec<(Strategy, Context, Effectiveness)>,
}

pub struct PerformanceMonitor {
    recent_performance: CircularBuffer<PerformanceMetric>,
    baseline_performance: PerformanceBaseline,
    anomaly_detector: AnomalyDetector,
}

impl PerformanceMonitor {
    pub fn monitor_memory_performance(&mut self,
        task: &MemoryTask,
        outcome: &TaskOutcome
    ) -> PerformanceAssessment {
        let metric = PerformanceMetric {
            task_type: task.task_type(),
            accuracy: outcome.accuracy,
            response_time: outcome.response_time,
            confidence_calibration: outcome.confidence_calibration,
            timestamp: Instant::now(),
        };
        
        self.recent_performance.push(metric.clone());
        
        // Compare to baseline
        let deviation = self.baseline_performance.calculate_deviation(&metric);
        
        // Detect anomalies
        let is_anomaly = self.anomaly_detector.is_anomaly(&metric, &self.recent_performance);
        
        PerformanceAssessment {
            current_performance: metric,
            trend: self.calculate_trend(),
            deviation_from_baseline: deviation,
            is_anomaly,
            suggested_interventions: self.suggest_interventions(deviation, is_anomaly),
        }
    }
}

pub struct LimitationAwareness {
    // Understanding of cognitive limitations
    working_memory_capacity: CapacityEstimate,
    forgetting_rate: ForgettingRateEstimate,
    interference_susceptibility: InterferenceMeasure,
    bias_awareness: BiasAwareness,
}

impl LimitationAwareness {
    pub fn assess_task_feasibility(&self,
        task: &CognitiveTask
    ) -> FeasibilityAssessment {
        let required_capacity = task.estimate_wm_load();
        let required_time = task.estimate_duration();
        let interference_risk = task.assess_interference_risk();
        
        FeasibilityAssessment {
            within_capacity: required_capacity <= self.working_memory_capacity.estimated_capacity,
            sufficient_time: required_time <= task.time_available,
            interference_manageable: interference_risk < self.interference_susceptibility.threshold,
            recommended_adaptations: self.suggest_adaptations(task),
            confidence: self.assessment_confidence(task),
        }
    }
}
```

## Week 23: Importance and Prioritization

### Task 23.1: Importance Scoring System
**File**: `src/emotional/importance_scoring.rs` (new file)
```rust
pub struct ImportanceScoring {
    personal_relevance: PersonalRelevanceScorer,
    goal_relevance: GoalRelevanceScorer,
    survival_relevance: SurvivalRelevanceScorer,
    social_relevance: SocialRelevanceScorer,
    novelty_detector: NoveltyDetector,
}

impl ImportanceScoring {
    pub fn calculate_importance(&self,
        memory: &Memory,
        context: &Context,
        self_model: &SelfModel
    ) -> ImportanceScore {
        let personal = self.personal_relevance.score(memory, self_model);
        let goal = self.goal_relevance.score(memory, &context.active_goals);
        let survival = self.survival_relevance.score(memory);
        let social = self.social_relevance.score(memory, &context.social_context);
        let novelty = self.novelty_detector.assess_novelty(memory);
        
        // Weighted combination
        let weights = ImportanceWeights {
            personal: 0.3,
            goal: 0.25,
            survival: 0.2,
            social: 0.15,
            novelty: 0.1,
        };
        
        let total_score = personal * weights.personal +
                         goal * weights.goal +
                         survival * weights.survival +
                         social * weights.social +
                         novelty * weights.novelty;
                         
        ImportanceScore {
            total: total_score,
            components: ImportanceComponents {
                personal, goal, survival, social, novelty
            },
            timestamp: Instant::now(),
            context_dependent: true,
        }
    }
    
    pub fn update_importance_over_time(&mut self,
        memory: &mut Memory,
        time_elapsed: Duration
    ) {
        // Some aspects decay, others crystallize
        let decay_factor = (-time_elapsed.as_secs_f32() / IMPORTANCE_HALF_LIFE).exp();
        
        // Goal relevance decays as goals change
        memory.importance.components.goal *= decay_factor;
        
        // But personal relevance can increase with reflection
        if memory.rehearsal_count > REFLECTION_THRESHOLD {
            memory.importance.components.personal *= 1.1;
        }
        
        // Recalculate total
        memory.importance.total = memory.importance.components.weighted_sum();
    }
}
```

### Task 23.2: Priority-Based Processing
**File**: `src/emotional/priority_processing.rs` (new file)
```rust
pub struct PriorityProcessor {
    priority_queue: BinaryHeap<PrioritizedMemory>,
    processing_capacity: ProcessingCapacity,
    urgency_calculator: UrgencyCalculator,
}

impl PriorityProcessor {
    pub fn prioritize_for_consolidation(&mut self,
        memories: Vec<Memory>
    ) -> Vec<Memory> {
        // Clear and rebuild priority queue
        self.priority_queue.clear();
        
        for memory in memories {
            let priority = self.calculate_consolidation_priority(&memory);
            self.priority_queue.push(PrioritizedMemory { memory, priority });
        }
        
        // Take top memories up to capacity
        let mut selected = Vec::new();
        let mut used_capacity = 0.0;
        
        while let Some(prioritized) = self.priority_queue.pop() {
            let required_capacity = self.estimate_processing_capacity(&prioritized.memory);
            if used_capacity + required_capacity <= self.processing_capacity.total {
                selected.push(prioritized.memory);
                used_capacity += required_capacity;
            } else {
                break;
            }
        }
        
        selected
    }
    
    fn calculate_consolidation_priority(&self, memory: &Memory) -> f32 {
        let importance = memory.importance.total;
        let urgency = self.urgency_calculator.calculate(memory);
        let emotional_priority = memory.emotional_tag
            .as_ref()
            .map(|e| e.intensity * e.valence.abs())
            .unwrap_or(0.0);
        let recency = memory.recency_weight();
        
        // Combine factors
        importance * 0.4 + urgency * 0.3 + emotional_priority * 0.2 + recency * 0.1
    }
}
```

## Week 24: Integration and Testing

### Task 24.1: Emotional-Cognitive Integration
**File**: `src/integration/emotion_cognition_bridge.rs` (new file)
```rust
pub struct EmotionCognitionBridge {
    hot_cognition: HotCognition,
    cold_cognition: ColdCognition,
    integration_mode: IntegrationMode,
}

impl EmotionCognitionBridge {
    pub fn process_with_emotion(&mut self,
        cognitive_task: &CognitiveTask,
        emotional_state: &EmotionalState
    ) -> ProcessingResult {
        match self.integration_mode {
            IntegrationMode::EmotionFirst => {
                // Emotion biases initial processing
                let biased_task = self.hot_cognition.apply_emotional_bias(
                    cognitive_task,
                    emotional_state
                );
                self.cold_cognition.process(biased_task)
            },
            IntegrationMode::Parallel => {
                // Process in parallel, then integrate
                let hot_result = self.hot_cognition.process(cognitive_task, emotional_state);
                let cold_result = self.cold_cognition.process(cognitive_task);
                self.integrate_results(hot_result, cold_result)
            },
            IntegrationMode::Regulated => {
                // Use emotion regulation before processing
                let regulated_state = self.regulate_emotion_for_task(
                    emotional_state,
                    cognitive_task
                );
                self.process_with_regulated_emotion(cognitive_task, regulated_state)
            },
        }
    }
}
```

### Task 24.2: Comprehensive Testing
**File**: `tests/emotional_metacognitive_tests.rs`
```rust
#[test]
fn test_emotional_enhancement() {
    let mut em_system = EmotionalMemory::new();
    
    // Create neutral memory
    let mut neutral_memory = create_test_memory("Grocery shopping");
    neutral_memory.encoding_strength = 0.5;
    
    // Create emotional memory
    let mut emotional_memory = create_test_memory("Won the lottery");
    let emotional_context = EmotionalContext {
        state: EmotionalState::joy(0.9),
        arousal: 0.9,
    };
    
    em_system.tag_memory_with_emotion(&mut emotional_memory, &emotional_context);
    
    // Emotional memory should be stronger
    assert!(emotional_memory.encoding_strength > neutral_memory.encoding_strength);
}

#[test]
fn test_mood_congruent_retrieval() {
    let mut em_system = EmotionalMemory::new();
    let sad_mood = EmotionalState::sadness(0.7);
    
    let memories = vec![
        create_emotional_memory("Happy birthday", 0.8, 0.1),  // Happy
        create_emotional_memory("Funeral", -0.8, 0.7),       // Sad
        create_emotional_memory("Normal day", 0.0, 0.3),     // Neutral
    ];
    
    let retrieved = em_system.mood_congruent_retrieval(&sad_mood, &memories);
    
    // Sad memory should be first
    assert_eq!(retrieved[0].content, "Funeral");
}

#[test]
fn test_metamemory_accuracy() {
    let mut metamemory = Metamemory::new();
    
    // Test feeling of knowing
    let strong_cue = RetrievalCue::strong("Einstein relativity");
    let weak_cue = RetrievalCue::weak("Random xyz");
    
    let strong_fok = metamemory.monitoring.assess_fok(&strong_cue);
    let weak_fok = metamemory.monitoring.assess_fok(&weak_cue);
    
    assert!(strong_fok.strength > weak_fok.strength);
}

#[test]
fn test_importance_scoring() {
    let mut scorer = ImportanceScoring::new();
    
    let survival_memory = create_memory_with_type("Snake encounter", MemoryType::Survival);
    let trivial_memory = create_memory_with_type("Saw a cloud", MemoryType::Trivial);
    
    let survival_importance = scorer.calculate_importance(&survival_memory);
    let trivial_importance = scorer.calculate_importance(&trivial_memory);
    
    assert!(survival_importance.total > trivial_importance.total * 2.0);
}
```

### Task 24.3: Performance Optimization
```rust
// Optimizations:
1. Emotion caching for repeated assessments
2. Batch emotional tagging
3. Parallel metacognitive monitoring
4. Importance score indexing
5. Priority queue optimization
```

### Task 24.4: API Documentation
**File**: `docs/emotional_metacognitive_api.md`
```markdown
# Emotional & Metacognitive API

## Emotional Memory

### Tag Memory with Emotion
```rust
POST /memory/emotional/tag
{
    "memory_id": "uuid",
    "emotional_context": {
        "valence": 0.8,
        "arousal": 0.6,
        "emotions": {
            "joy": 0.8,
            "surprise": 0.3
        }
    }
}
```

### Mood Congruent Retrieval
```rust
POST /memory/retrieve/mood_congruent
{
    "current_mood": {
        "valence": -0.5,
        "arousal": 0.3
    },
    "limit": 10
}
```

## Metacognitive Monitoring

### Assess Feeling of Knowing
```rust
POST /metacognitive/fok
{
    "retrieval_cue": "Einstein theory",
    "context": {}
}
```

### Get Memory Confidence
```rust
GET /metacognitive/confidence/{memory_id}
```
```

## Deliverables
1. **Emotional memory system** with tagging and mood congruence
2. **Affect regulation** strategies for memory modification
3. **Metamemory monitoring** with FOK and confidence judgments
4. **Metacognitive control** for strategy selection
5. **Importance scoring** system with multiple factors
6. **Priority-based processing** for limited resources

## Success Criteria
- [ ] Emotional memories show enhanced strength (1.5x+)
- [ ] Mood congruent retrieval shows 30%+ effect
- [ ] FOK predictions correlate with actual retrieval (r > 0.6)
- [ ] Confidence calibration improves with feedback
- [ ] Important memories prioritized for consolidation
- [ ] System handles 1000+ emotional tags efficiently

## Dependencies
- Emotion recognition models
- Cognitive resource models
- Priority queue implementations

## Risks & Mitigations
1. **Emotion categorization ambiguity**
   - Mitigation: Use dimensional model (valence/arousal)
2. **Metacognitive overhead**
   - Mitigation: Selective monitoring, caching
3. **Priority computation cost**
   - Mitigation: Incremental updates, approximations