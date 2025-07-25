# Phase 6: Emotional & Metacognitive Systems

## Overview
**Duration**: 4 weeks  
**Goal**: Add emotional influence on memory and metacognitive monitoring capabilities  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-5 completion  
**Target Performance**: <5ms for emotion tagging, <3ms for metacognitive assessment on Intel i9

## Integration with Existing Systems
**Building on existing infrastructure**:
- **Model Loading**: Leverages `src/models/` directory with Candle framework integration
- **Storage Patterns**: Uses existing `storage/PersistentMMapStorage` for emotional metadata
- **Monitoring Integration**: Extends `monitoring/PerformanceMonitor` for emotional/metacognitive metrics
- **Validation Enhancement**: Builds on existing `validation/HumanValidationInterface` system
- **MCP Tool Integration**: Adds new handlers to existing MCP server architecture in `src/mcp/`
- **Batch Processing**: Uses established `streaming/BatchProcessor` patterns from existing systems
- **Cognitive Integration**: Extends existing `cognitive/CognitiveOrchestrator` for metacognitive awareness

## Multi-Database Architecture - Phase 6
**New in Phase 6**: Implement analytics database for emotional and metacognitive metrics
- **Analytics Database**: OLAP cubes for emotional pattern analysis
- **Real-time Aggregation**: Live emotional state tracking and trends
- **Cross-Database Analytics**: Correlation analysis across all databases
- **Metacognitive Dashboards**: Real-time confidence and learning metrics

## AI Model Integration (Rust/Candle)
**Selected Models** (stored in `src/models/pretrained/`):
- **RoBERTa-emotion** (125M params) - For categorical emotion detection
- **DistilBERT-VA** (66M params) - For valence-arousal dimensional analysis  
- **ConfidenceNet** (20M params) - For metacognitive confidence calibration
- **FOKPredictor** (15M params) - For feeling-of-knowing assessment
- **StrategySelector** (10M params) - For optimal strategy selection
- **ImportanceScorer** (5M params) - For AI-enhanced importance assessment
- All models use Candle framework with INT8 quantization for speed

**Available Models from src/models** (can be repurposed):
- **DistilBERT-NER** (66M params) - Repurposed for emotional content analysis
- **T5-Small** (60M params) - For emotional content generation and reframing
- **all-MiniLM-L6-v2** (22M params) - For semantic similarity in emotional contexts
- **TinyBERT-NER** (14.5M params) - Lightweight emotional pattern detection
- **Dependency Parser** (40M params) - For analyzing emotional expression structure
- **Intent Classifier** (30M params) - For understanding emotional intent
- **Relation Classifier** (25M params) - For emotional relationship extraction

## Week 21: Emotional Memory System

### Task 21.1: Emotion-Memory Integration
**File**: `src/emotional/emotional_memory.rs` (new file)
```rust
use candle_core::{Device, Tensor};
use crate::models::{load_model, ModelType};
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use crate::core::types::EntityKey;
use crate::validation::ValidationResult;
use std::sync::Arc;
use dashmap::DashMap;

pub struct EmotionalMemory {
    // AI models using existing Candle infrastructure
    emotion_recognition: Arc<dyn crate::models::ModelInterface>,  // RoBERTa-emotion 125M
    valence_arousal_model: Arc<dyn crate::models::ModelInterface>, // DistilBERT-VA 66M
    
    // Core components
    emotion_tagger: AIEmotionTagger,
    mood_congruence: MoodCongruenceSystem,
    emotional_consolidation: EmotionalConsolidation,
    affect_regulation: AffectRegulation,
    
    // Integration with existing systems
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    device: Device,
    
    // Performance optimizations (using existing patterns)
    emotion_cache: DashMap<EntityKey, EmotionalTag>,
    batch_processor: crate::streaming::BatchProcessor<EmotionalTask>,
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
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Result<Self, crate::error::GraphError> {
        let device = Device::Cpu; // Use existing device selection logic
        
        // Load models using existing infrastructure
        let emotion_recognition = crate::models::load_model(
            "src/models/pretrained/roberta_emotion_int8.safetensors",
            ModelType::EmotionClassifier,
            &device,
        )?;
        
        let valence_arousal_model = crate::models::load_model(
            "src/models/pretrained/distilbert_va_int8.safetensors", 
            ModelType::ValenceArousal,
            &device,
        )?;
        
        Ok(Self {
            emotion_recognition,
            valence_arousal_model,
            emotion_tagger: AIEmotionTagger::new(),
            mood_congruence: MoodCongruenceSystem::new(),
            emotional_consolidation: EmotionalConsolidation::new(),
            affect_regulation: AffectRegulation::new(),
            storage,
            monitor,
            device,
            emotion_cache: DashMap::with_capacity(10_000),
            batch_processor: crate::streaming::BatchProcessor::new(32),
        })
    }
    
    pub async fn tag_memory_with_emotion(&mut self, 
        entity_key: &EntityKey,
        content: &str,
        context: &EmotionalContext
    ) -> Result<EmotionalTag, crate::error::GraphError> {
        // Monitor performance
        let _timer = self.monitor.start_timer("emotion_tagging");
        
        // Check cache first
        if let Some(cached_tag) = self.emotion_cache.get(entity_key) {
            return Ok(cached_tag.clone());
        }
        
        // Parallel AI analysis using existing async patterns
        let (
            emotion_categories,
            valence_arousal,
            contextual_emotions
        ) = tokio::join!(
            self.detect_emotions_ai(content),
            self.analyze_valence_arousal_ai(content),
            self.analyze_contextual_emotions(context)
        );
        
        // Combine AI predictions
        let emotional_tag = self.create_comprehensive_tag(
            emotion_categories,
            valence_arousal,
            contextual_emotions
        );
        
        // Store emotional metadata using existing storage patterns
        self.storage.store_metadata(
            entity_key,
            "emotional_tag",
            &serde_json::to_vec(&emotional_tag)?
        )?;
        
        // Cache the tag
        self.emotion_cache.insert(entity_key.clone(), emotional_tag.clone());
        
        // Update monitoring metrics
        self.monitor.record_metric("emotional_tags_created", 1.0);
        
        Ok(emotional_tag)
    }
    
    async fn detect_emotions_ai(&self, content: &str) -> BasicEmotions {
        // Use existing model interface for emotion classification
        match self.emotion_recognition.classify_emotions(content).await {
            Ok(emotions) => emotions,
            Err(_) => BasicEmotions::neutral(),
        }
    }
    
    async fn analyze_valence_arousal_ai(&self, content: &str) -> (f32, f32) {
        // Use existing model interface for valence-arousal prediction
        match self.valence_arousal_model.predict_va(content).await {
            Ok((valence, arousal)) => (valence, arousal),
            Err(_) => (0.0, 0.5),  // Neutral valence, medium arousal
        }
    }
    
    pub async fn mood_congruent_retrieval(&self,
        current_mood: &EmotionalState,
        entity_keys: &[EntityKey]
    ) -> Vec<(EntityKey, f32)> {
        // Use existing batch processing patterns
        let mut tasks = Vec::new();
        
        for chunk in entity_keys.chunks(64) {
            let chunk_future = self.batch_processor.process_batch(
                chunk.to_vec(),
                |key| async move {
                    // Load emotional tag from storage
                    let tag = self.load_emotional_tag(key).await?;
                    let congruence = if let Some(tag) = tag {
                        self.calculate_congruence_ai(current_mood, &tag).await
                    } else {
                        0.0
                    };
                    Ok((key.clone(), congruence))
                }
            );
            tasks.push(chunk_future);
        }
        
        // Await all batches
        let mut all_scores = Vec::new();
        for batch_result in futures::future::join_all(tasks).await {
            if let Ok(scores) = batch_result {
                all_scores.extend(scores);
            }
        }
        
        // Sort by congruence score using existing ordering utilities
        all_scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        all_scores
    }
    
    async fn calculate_congruence_ai(
        &self,
        current_mood: &EmotionalState,
        memory_emotion: &EmotionalTag
    ) -> f32 {
        // Use SIMD for fast vector similarity (from existing patterns)
        let mood_vec = current_mood.to_vector();
        let memory_vec = memory_emotion.to_vector();
        
        // Cosine similarity with SIMD
        let similarity = crate::math::simd_cosine_similarity(&mood_vec, &memory_vec);
        
        // Boost for matching valence
        let valence_match = 1.0 - (current_mood.valence - memory_emotion.valence).abs();
        
        similarity * 0.7 + valence_match * 0.3
    }
    
    async fn calculate_emotional_boost_ai(&self, tag: &EmotionalTag) -> f32 {
        // AI-enhanced boost calculation
        let emotion_vector = tag.to_feature_vector();
        
        // Use lightweight MLP for boost prediction
        let boost = match self.emotion_tagger.predict_memory_boost(&emotion_vector).await {
            Ok(b) => b,
            Err(_) => {
                // Fallback to rule-based calculation
                let arousal_boost = tag.arousal * 0.5;
                let valence_boost = tag.valence.abs() * 0.3;
                let surprise_boost = tag.emotions.surprise * 0.4;
                1.0 + arousal_boost + valence_boost + surprise_boost
            }
        };
        
        // Ensure reasonable bounds
        boost.max(1.0).min(3.0)
    }
}
```

### Task 21.2: Affect Regulation and Memory
**File**: `src/emotional/affect_regulation.rs` (new file)
```rust
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use std::sync::Arc;
use std::time::Duration;

pub struct AffectRegulation {
    regulation_strategies: Vec<RegulationStrategy>,
    emotional_goals: EmotionalGoals,
    regulation_history: RegulationHistory,
    // Integration with existing systems
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
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
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Self {
        Self {
            regulation_strategies: vec![
                RegulationStrategy::Suppression {
                    target_emotion: "negative".to_string(),
                    effectiveness: 0.6,
                    cognitive_cost: 0.8,
                },
                RegulationStrategy::Reappraisal {
                    reframe_function: Box::new(|memory| {
                        // Default reframing logic
                        memory.clone()
                    }),
                    success_rate: 0.7,
                },
            ],
            emotional_goals: EmotionalGoals::default(),
            regulation_history: RegulationHistory::new(storage.clone()),
            storage,
            monitor,
        }
    }
    
    pub fn regulate_emotional_memory(&mut self,
        memory: &mut Memory,
        strategy: &RegulationStrategy
    ) -> RegulationOutcome {
        let _timer = self.monitor.start_timer("affect_regulation");
        
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
        // Use existing cognitive patterns for strategy selection
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

### Task 21.3: Emotional Learning Integration
**File**: `src/emotional/emotional_learning.rs` (new file)
```rust
use crate::learning::{HebbianLearningEngine, LearningContext};
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use std::sync::Arc;
use std::collections::HashMap;

pub struct EmotionalLearning {
    fear_conditioning: FearConditioning,
    reward_learning: RewardLearning,
    emotional_schemas: HashMap<SchemaId, EmotionalSchema>,
    // Integration with existing learning systems
    hebbian_engine: Arc<HebbianLearningEngine>,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
}

impl EmotionalLearning {
    pub fn new(
        hebbian_engine: Arc<HebbianLearningEngine>,
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Self {
        Self {
            fear_conditioning: FearConditioning::new(storage.clone()),
            reward_learning: RewardLearning::new(storage.clone()),
            emotional_schemas: HashMap::new(),
            hebbian_engine,
            storage,
            monitor,
        }
    }
    
    pub async fn condition_emotional_response(
        &mut self,
        stimulus: &Stimulus,
        emotional_outcome: &EmotionalOutcome,
        context: &LearningContext
    ) -> Result<(), crate::error::GraphError> {
        let _timer = self.monitor.start_timer("emotional_conditioning");
        
        // Use existing Hebbian learning for emotional associations
        self.hebbian_engine.strengthen_association(
            &stimulus.to_entity_key(),
            &emotional_outcome.to_entity_key(),
            emotional_outcome.intensity,
            context
        ).await?;
        
        // Store emotional learning in persistent storage
        self.storage.store_emotional_association(
            stimulus,
            emotional_outcome,
            context.timestamp
        )?;
        
        self.monitor.record_metric("emotional_associations_learned", 1.0);
        
        Ok(())
    }
}

pub struct FearConditioning {
    conditioned_stimuli: HashMap<Stimulus, FearResponse>,
    extinction_trials: HashMap<Stimulus, u32>,
    context_dependency: HashMap<(Stimulus, Context), f32>,
    storage: Arc<PersistentMMapStorage>,
}

impl FearConditioning {
    pub fn new(storage: Arc<PersistentMMapStorage>) -> Self {
        Self {
            conditioned_stimuli: HashMap::new(),
            extinction_trials: HashMap::new(),
            context_dependency: HashMap::new(),
            storage,
        }
    }
    
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
        
        // Persist to storage
        let _ = self.storage.store_fear_conditioning(neutral_stimulus, &fear_response, context);
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
use candle_core::{Device, Tensor};
use crate::models::{load_model, ModelType};
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use crate::validation::{ValidationResult, ValidationTask};
use crate::cognitive::{CognitiveOrchestrator, WorkingMemorySystem};
use std::sync::Arc;
use lru::LruCache;

pub struct Metamemory {
    // Core components building on existing cognitive system
    monitoring: MetamemoryMonitoring,
    control: MetamemoryControl,
    knowledge: MetamemoryKnowledge,
    
    // AI models using existing infrastructure
    confidence_model: Arc<dyn crate::models::ModelInterface>,  // 20M params
    fok_predictor: Arc<dyn crate::models::ModelInterface>,     // 15M params
    strategy_selector: Arc<dyn crate::models::ModelInterface>, // 10M params
    
    // Integration with existing systems
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    cognitive_orchestrator: Arc<CognitiveOrchestrator>,
    working_memory: Arc<WorkingMemorySystem>,
    device: Device,
    
    // Performance optimization using existing patterns
    judgment_cache: LruCache<JudgmentKey, ConfidenceJudgment>,
    batch_processor: crate::streaming::BatchProcessor<MetacognitiveTask>,
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
    fok_model: Arc<dyn crate::models::ModelInterface>,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    device: Device,
    cache: DashMap<JudgmentKey, FOKJudgment>,
}

impl FeelingOfKnowing {
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Result<Self, crate::error::GraphError> {
        let device = Device::Cpu;
        
        let fok_model = crate::models::load_model(
            "src/models/pretrained/fok_predictor_int8.safetensors",
            ModelType::FOKPredictor,
            &device,
        )?;
        
        Ok(Self {
            cue_familiarity_threshold: 0.3,
            partial_activation_detector: PartialActivationDetector::new(),
            fok_model,
            storage,
            monitor,
            device,
            cache: DashMap::with_capacity(1000),
        })
    }
    
    pub async fn assess_feeling_of_knowing(&self,
        cue: &RetrievalCue,
        memory_state: &MemoryState
    ) -> FOKJudgment {
        let _timer = self.monitor.start_timer("fok_assessment");
        
        // Check cache
        let cache_key = (cue.hash(), memory_state.hash());
        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }
        
        // Parallel assessment for performance
        let (
            cue_familiarity,
            partial_activation,
            competition,
            ai_prediction
        ) = tokio::join!(
            self.calculate_cue_familiarity_async(cue, memory_state),
            self.partial_activation_detector.detect_async(cue, memory_state),
            self.assess_competition_async(cue, memory_state),
            self.predict_fok_ai(cue, memory_state)
        );
        
        // Combine rule-based and AI predictions
        let rule_based_strength = (cue_familiarity * 0.4 + partial_activation * 0.4 - competition * 0.2).max(0.0);
        let combined_strength = rule_based_strength * 0.6 + ai_prediction.strength * 0.4;
        
        let judgment = FOKJudgment {
            strength: combined_strength,
            confidence: self.calculate_confidence(cue_familiarity, partial_activation, ai_prediction.confidence),
            predicted_retrieval_success: ai_prediction.retrieval_probability,
            components: FOKComponents {
                cue_familiarity,
                partial_activation,
                competition,
                ai_prediction: ai_prediction.strength,
            },
        };
        
        // Cache result
        self.cache.insert(cache_key, judgment.clone());
        
        self.monitor.record_metric("fok_assessments_completed", 1.0);
        
        judgment
    }
    
    async fn predict_fok_ai(&self, cue: &RetrievalCue, memory_state: &MemoryState) -> FOKPrediction {
        // Prepare features for AI model
        let features = self.extract_fok_features(cue, memory_state);
        
        match self.fok_model.predict(&features).await {
            Ok(pred) => pred,
            Err(_) => FOKPrediction::default(),
        }
    }
}

pub struct ConfidenceJudgments {
    calibration_history: CalibrationHistory,
    overconfidence_bias: f32,
    dunning_kruger_effect: bool,
    confidence_model: Arc<dyn crate::models::ModelInterface>,
    feature_extractor: ConfidenceFeatureExtractor,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    device: Device,
}

impl ConfidenceJudgments {
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Result<Self, crate::error::GraphError> {
        let device = Device::Cpu;
        
        let confidence_model = crate::models::load_model(
            "src/models/pretrained/confidence_net_int8.safetensors",
            ModelType::ConfidenceCalibration,
            &device,
        )?;
        
        Ok(Self {
            calibration_history: CalibrationHistory::new(storage.clone()),
            overconfidence_bias: 0.1,
            dunning_kruger_effect: false,
            confidence_model,
            feature_extractor: ConfidenceFeatureExtractor::new(),
            storage,
            monitor,
            device,
        })
    }
    
    pub async fn judge_memory_accuracy(&self,
        memory: &Memory,
        retrieval_context: &RetrievalContext
    ) -> ConfidenceJudgment {
        let _timer = self.monitor.start_timer("confidence_judgment");
        
        // Extract features for AI model
        let features = self.feature_extractor.extract(
            memory,
            retrieval_context
        );
        
        // Parallel computation
        let (
            ai_confidence,
            rule_based_confidence
        ) = tokio::join!(
            self.predict_confidence_ai(&features),
            self.calculate_rule_based_confidence(memory, retrieval_context)
        );
        
        // Combine AI and rule-based predictions
        let combined_confidence = ai_confidence * 0.7 + rule_based_confidence * 0.3;
        
        // Apply individual calibration
        let calibrated_confidence = self.apply_calibration(combined_confidence);
        
        let judgment = ConfidenceJudgment {
            confidence: calibrated_confidence.min(1.0),
            actual_accuracy: memory.current_strength,
            calibration_error: (calibrated_confidence - memory.current_strength).abs(),
            basis: ConfidenceBasis {
                fluency_contribution: retrieval_context.retrieval_time.as_secs_f32().recip() * 0.5,
                vividness_contribution: memory.vividness * 0.3,
                strength_contribution: memory.current_strength * 0.2,
                ai_contribution: ai_confidence,
            },
            calibration_quality: self.calibration_history.get_calibration_score(),
        };
        
        self.monitor.record_metric("confidence_judgments_made", 1.0);
        
        judgment
    }
    
    async fn predict_confidence_ai(&self, features: &ConfidenceFeatures) -> f32 {
        match self.confidence_model.predict(features).await {
            Ok(conf) => conf,
            Err(_) => 0.5,  // Default to medium confidence
        }
    }
    
    async fn calculate_rule_based_confidence(
        &self,
        memory: &Memory,
        retrieval_context: &RetrievalContext
    ) -> f32 {
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
        raw_confidence * (1.0 + self.overconfidence_bias)
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
use crate::cognitive::{CognitiveOrchestrator, WorkingMemorySystem};
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use std::sync::Arc;
use std::collections::HashMap;

pub struct MetacognitiveControl {
    strategy_selector: StrategySelector,
    effort_allocator: EffortAllocator,
    monitoring_scheduler: MonitoringScheduler,
    // Integration with existing cognitive systems
    cognitive_orchestrator: Arc<CognitiveOrchestrator>,
    working_memory: Arc<WorkingMemorySystem>,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
}

pub struct StrategySelector {
    available_strategies: Vec<MemoryStrategy>,
    strategy_effectiveness: HashMap<(StrategyId, ContextType), f32>,
    adaptation_rate: f32,
    strategy_model: Arc<dyn crate::models::ModelInterface>,
    feature_encoder: StrategyFeatureEncoder,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    device: Device,
}

impl StrategySelector {
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Result<Self, crate::error::GraphError> {
        let device = Device::Cpu;
        
        let strategy_model = crate::models::load_model(
            "src/models/pretrained/strategy_selector_int8.safetensors",
            ModelType::StrategySelector,
            &device,
        )?;
        
        // Load historical effectiveness data from storage
        let strategy_effectiveness = storage.load_metadata("strategy_effectiveness")
            .unwrap_or_default();
        
        Ok(Self {
            available_strategies: vec![
                MemoryStrategy::Rehearsal,
                MemoryStrategy::Elaboration,
                MemoryStrategy::Organization,
                MemoryStrategy::Imagery,
                MemoryStrategy::SelfReference,
                MemoryStrategy::DualCoding,
                MemoryStrategy::Spacing,
            ],
            strategy_effectiveness,
            adaptation_rate: 0.1,
            strategy_model,
            feature_encoder: StrategyFeatureEncoder::new(),
            storage,
            monitor,
            device,
        })
    }
    
    pub async fn select_encoding_strategy(&self,
        material: &Material,
        goals: &LearningGoals,
        resources: &CognitiveResources
    ) -> EncodingStrategy {
        let _timer = self.monitor.start_timer("strategy_selection");
        
        // Encode features for AI model
        let features = self.feature_encoder.encode_context(
            material,
            goals,
            resources
        );
        
        // Get AI strategy recommendation
        let ai_recommendation = self.strategy_model.recommend(&features).await
            .unwrap_or(StrategyRecommendation::default());
        
        // Evaluate all strategies in parallel using existing patterns
        let strategy_futures: Vec<_> = self.available_strategies.iter()
            .map(|strategy| async move {
                let effectiveness = self.predict_effectiveness_hybrid(
                    strategy,
                    material,
                    goals,
                    &ai_recommendation
                ).await;
                let cost = self.calculate_cognitive_cost(strategy, resources);
                (strategy.clone(), effectiveness / cost.max(0.1))
            })
            .collect();
        
        let strategy_scores = futures::future::join_all(strategy_futures).await;
        
        // Select best strategy
        let selected_strategy = strategy_scores.into_iter()
            .max_by_key(|(_, score)| OrderedFloat(*score))
            .map(|(strategy, _)| self.instantiate_strategy(strategy, material))
            .unwrap_or(EncodingStrategy::Rehearsal { 
                type_: RehearsalType::Maintenance 
            });
        
        self.monitor.record_metric("strategies_selected", 1.0);
        
        selected_strategy
    }
    
    async fn predict_effectiveness_hybrid(
        &self,
        strategy: &MemoryStrategy,
        material: &Material,
        goals: &LearningGoals,
        ai_rec: &StrategyRecommendation
    ) -> f32 {
        // Combine historical effectiveness with AI prediction
        let context_type = self.categorize_context(material, goals);
        let historical = self.strategy_effectiveness
            .get(&(strategy.id(), context_type))
            .copied()
            .unwrap_or(0.5);
        
        let ai_score = ai_rec.get_score_for(strategy);
        
        // Weighted combination favoring AI when available
        historical * 0.4 + ai_score * 0.6
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
    // Integration with existing resource management
    working_memory: Arc<WorkingMemorySystem>,
    monitor: Arc<PerformanceMonitor>,
}

impl EffortAllocator {
    pub fn new(
        working_memory: Arc<WorkingMemorySystem>,
        monitor: Arc<PerformanceMonitor>
    ) -> Self {
        Self {
            total_resources: CognitiveResources::default(),
            task_priorities: BinaryHeap::new(),
            effort_value_function: Box::new(|current, goal| {
                // Diminishing returns function
                let gap = goal - current;
                gap * (1.0 - current).powf(0.5)
            }),
            working_memory,
            monitor,
        }
    }
    
    pub fn allocate_study_time(&mut self,
        items: &[StudyItem]
    ) -> HashMap<ItemId, Duration> {
        let _timer = self.monitor.start_timer("effort_allocation");
        
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
        
        self.monitor.record_metric("study_time_allocated", total_time.as_secs_f32());
        
        allocations
    }
}
```

### Task 22.3: Self-Monitoring and Awareness
**File**: `src/metacognitive/self_awareness.rs` (new file)
```rust
use crate::cognitive::{CognitiveOrchestrator, WorkingMemorySystem};
use crate::monitoring::PerformanceMonitor;
use crate::storage::PersistentMMapStorage;
use std::sync::Arc;
use std::collections::VecDeque;

pub struct SelfAwareness {
    memory_self_knowledge: MemorySelfKnowledge,
    performance_monitor: PerformanceMonitor,
    limitation_awareness: LimitationAwareness,
    // Integration with existing systems
    cognitive_orchestrator: Arc<CognitiveOrchestrator>,
    working_memory: Arc<WorkingMemorySystem>,
    storage: Arc<PersistentMMapStorage>,
    system_monitor: Arc<PerformanceMonitor>,
}

pub struct MemorySelfKnowledge {
    // Knowledge about own memory abilities
    strengths: Vec<MemoryStrength>,
    weaknesses: Vec<MemoryWeakness>,
    typical_errors: Vec<ErrorPattern>,
    effective_strategies: Vec<(Strategy, Context, Effectiveness)>,
    // Integration with existing validation system
    validation_history: ValidationHistory,
}

pub struct PerformanceMonitor {
    recent_performance: VecDeque<PerformanceMetric>,
    baseline_performance: PerformanceBaseline,
    anomaly_detector: AnomalyDetector,
    // Integration with existing monitoring
    system_monitor: Arc<crate::monitoring::PerformanceMonitor>,
}

impl PerformanceMonitor {
    pub fn monitor_memory_performance(&mut self,
        task: &MemoryTask,
        outcome: &TaskOutcome
    ) -> PerformanceAssessment {
        let _timer = self.system_monitor.start_timer("memory_performance_monitoring");
        
        let metric = PerformanceMetric {
            task_type: task.task_type(),
            accuracy: outcome.accuracy,
            response_time: outcome.response_time,
            confidence_calibration: outcome.confidence_calibration,
            timestamp: Instant::now(),
        };
        
        self.recent_performance.push_back(metric.clone());
        
        // Keep only recent metrics (sliding window)
        if self.recent_performance.len() > 1000 {
            self.recent_performance.pop_front();
        }
        
        // Compare to baseline
        let deviation = self.baseline_performance.calculate_deviation(&metric);
        
        // Detect anomalies
        let is_anomaly = self.anomaly_detector.is_anomaly(&metric, &self.recent_performance);
        
        let assessment = PerformanceAssessment {
            current_performance: metric,
            trend: self.calculate_trend(),
            deviation_from_baseline: deviation,
            is_anomaly,
            suggested_interventions: self.suggest_interventions(deviation, is_anomaly),
        };
        
        // Record metrics in system monitor
        self.system_monitor.record_metric("memory_accuracy", outcome.accuracy);
        self.system_monitor.record_metric("memory_response_time", outcome.response_time.as_secs_f32());
        self.system_monitor.record_metric("confidence_calibration", outcome.confidence_calibration);
        
        assessment
    }
    
    fn calculate_trend(&self) -> PerformanceTrend {
        if self.recent_performance.len() < 10 {
            return PerformanceTrend::Insufficient;
        }
        
        let recent: Vec<_> = self.recent_performance.iter().rev().take(10).collect();
        let older: Vec<_> = self.recent_performance.iter().rev().skip(10).take(10).collect();
        
        let recent_avg = recent.iter().map(|m| m.accuracy).sum::<f32>() / recent.len() as f32;
        let older_avg = older.iter().map(|m| m.accuracy).sum::<f32>() / older.len().max(1) as f32;
        
        if recent_avg > older_avg + 0.05 {
            PerformanceTrend::Improving
        } else if recent_avg < older_avg - 0.05 {
            PerformanceTrend::Declining
        } else {
            PerformanceTrend::Stable
        }
    }
}

pub struct LimitationAwareness {
    // Understanding of cognitive limitations
    working_memory_capacity: CapacityEstimate,
    forgetting_rate: ForgettingRateEstimate,
    interference_susceptibility: InterferenceMeasure,
    bias_awareness: BiasAwareness,
    // Integration with existing working memory system
    working_memory: Arc<WorkingMemorySystem>,
}

impl LimitationAwareness {
    pub fn new(working_memory: Arc<WorkingMemorySystem>) -> Self {
        Self {
            working_memory_capacity: CapacityEstimate::from_working_memory(&working_memory),
            forgetting_rate: ForgettingRateEstimate::default(),
            interference_susceptibility: InterferenceMeasure::default(),
            bias_awareness: BiasAwareness::default(),
            working_memory,
        }
    }
    
    pub fn assess_task_feasibility(&self,
        task: &CognitiveTask
    ) -> FeasibilityAssessment {
        let required_capacity = task.estimate_wm_load();
        let required_time = task.estimate_duration();
        let interference_risk = task.assess_interference_risk();
        
        // Check with actual working memory system
        let current_capacity = self.working_memory.get_available_capacity();
        let capacity_sufficient = required_capacity <= current_capacity;
        
        FeasibilityAssessment {
            within_capacity: capacity_sufficient,
            sufficient_time: required_time <= task.time_available,
            interference_manageable: interference_risk < self.interference_susceptibility.threshold,
            recommended_adaptations: self.suggest_adaptations(task),
            confidence: self.assessment_confidence(task),
            current_wm_load: current_capacity,
            required_wm_load: required_capacity,
        }
    }
    
    fn suggest_adaptations(&self, task: &CognitiveTask) -> Vec<TaskAdaptation> {
        let mut adaptations = Vec::new();
        
        if task.estimate_wm_load() > self.working_memory_capacity.estimated_capacity {
            adaptations.push(TaskAdaptation::ChunkInformation);
            adaptations.push(TaskAdaptation::UseExternalMemory);
        }
        
        if task.assess_interference_risk() > self.interference_susceptibility.threshold {
            adaptations.push(TaskAdaptation::ReduceInterference);
            adaptations.push(TaskAdaptation::SequentialProcessing);
        }
        
        adaptations
    }
}
```

## Week 23: Importance and Prioritization

### Task 23.1: Importance Scoring System
**File**: `src/emotional/importance_scoring.rs` (new file)
```rust
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use crate::core::types::EntityKey;
use std::sync::Arc;
use lru::LruCache;
use std::collections::HashMap;

pub struct ImportanceScoring {
    personal_relevance: PersonalRelevanceScorer,
    goal_relevance: GoalRelevanceScorer,
    survival_relevance: SurvivalRelevanceScorer,
    social_relevance: SocialRelevanceScorer,
    novelty_detector: NoveltyDetector,
    // AI components using existing infrastructure
    importance_model: Arc<dyn crate::models::ModelInterface>,  // 5M params
    feature_combiner: FeatureCombiner,
    // Integration with existing systems
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
    device: Device,
    // Performance optimization using existing patterns
    importance_cache: LruCache<EntityKey, ImportanceScore>,
    batch_scorer: crate::streaming::BatchProcessor<ImportanceTask>,
}

impl ImportanceScoring {
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Result<Self, crate::error::GraphError> {
        let device = Device::Cpu;
        
        let importance_model = crate::models::load_model(
            "src/models/pretrained/importance_scorer_int8.safetensors",
            ModelType::ImportanceScorer,
            &device,
        )?;
        
        Ok(Self {
            personal_relevance: PersonalRelevanceScorer::new(),
            goal_relevance: GoalRelevanceScorer::new(),
            survival_relevance: SurvivalRelevanceScorer::new(),
            social_relevance: SocialRelevanceScorer::new(),
            novelty_detector: NoveltyDetector::new(),
            importance_model,
            feature_combiner: FeatureCombiner::new(),
            storage,
            monitor,
            device,
            importance_cache: LruCache::new(5000),
            batch_scorer: crate::streaming::BatchProcessor::new(64),
        })
    }
    
    pub async fn calculate_importance(&self,
        entity_key: &EntityKey,
        context: &Context,
        self_model: &SelfModel
    ) -> Result<ImportanceScore, crate::error::GraphError> {
        let _timer = self.monitor.start_timer("importance_calculation");
        
        // Check cache
        if let Some(cached) = self.importance_cache.get(entity_key) {
            return Ok(cached.clone());
        }
        
        // Load entity data from storage
        let entity_data = self.storage.get_entity(entity_key).await?;
        
        // Parallel scoring using existing async patterns
        let (
            personal,
            goal,
            survival,
            social,
            novelty,
            ai_importance
        ) = tokio::join!(
            self.personal_relevance.score_async(&entity_data, self_model),
            self.goal_relevance.score_async(&entity_data, &context.active_goals),
            self.survival_relevance.score_async(&entity_data),
            self.social_relevance.score_async(&entity_data, &context.social_context),
            self.novelty_detector.assess_novelty_async(&entity_data),
            self.predict_importance_ai(&entity_data, context, self_model)
        );
        
        // Combine features for final score
        let features = ImportanceFeatures {
            personal,
            goal,
            survival,
            social,
            novelty,
            entity_features: self.extract_entity_features(&entity_data),
            context_features: self.extract_context_features(context),
        };
        
        // AI-enhanced weighted combination
        let weights = self.importance_model.predict_weights(&features).await
            .unwrap_or(ImportanceWeights::default());
        
        let total_score = personal * weights.personal +
                         goal * weights.goal +
                         survival * weights.survival +
                         social * weights.social +
                         novelty * weights.novelty +
                         ai_importance * weights.ai_factor;
                         
        let score = ImportanceScore {
            total: total_score.min(1.0),
            components: ImportanceComponents {
                personal, goal, survival, social, novelty
            },
            ai_score: ai_importance,
            timestamp: std::time::Instant::now(),
            context_dependent: true,
            confidence: weights.confidence,
            entity_key: entity_key.clone(),
        };
        
        // Store in cache and persistent storage
        self.importance_cache.put(entity_key.clone(), score.clone());
        self.storage.store_importance_score(entity_key, &score)?;
        
        self.monitor.record_metric("importance_scores_calculated", 1.0);
        
        Ok(score)
    }
    
    pub async fn batch_calculate_importance(
        &self,
        entity_keys: &[EntityKey],
        context: &Context,
        self_model: &SelfModel
    ) -> Vec<ImportanceScore> {
        // Use existing batch processing infrastructure
        let batch_results = self.batch_scorer.process_batch(
            entity_keys.to_vec(),
            |key| async move {
                self.calculate_importance(&key, context, self_model).await
            }
        ).await;
        
        match batch_results {
            Ok(scores) => scores,
            Err(e) => {
                self.monitor.record_error("batch_importance_scoring", &e.to_string());
                vec![]
            }
        }
    }
    
    async fn predict_importance_ai(
        &self,
        entity_data: &EntityData,
        context: &Context,
        self_model: &SelfModel
    ) -> f32 {
        let features = self.feature_combiner.combine(
            entity_data,
            context,
            self_model
        );
        
        match self.importance_model.predict_importance(&features).await {
            Ok(score) => score,
            Err(_) => 0.5,  // Default medium importance
        }
    }
    
    pub fn update_importance_over_time(&mut self,
        entity_key: &EntityKey,
        time_elapsed: Duration
    ) -> Result<(), crate::error::GraphError> {
        // Load current importance score
        if let Some(mut importance) = self.storage.load_importance_score(entity_key)? {
            // Some aspects decay, others crystallize
            let decay_factor = (-time_elapsed.as_secs_f32() / IMPORTANCE_HALF_LIFE).exp();
            
            // Goal relevance decays as goals change
            importance.components.goal *= decay_factor;
            
            // But personal relevance can increase with reflection
            if let Some(rehearsal_count) = self.storage.get_rehearsal_count(entity_key)? {
                if rehearsal_count > REFLECTION_THRESHOLD {
                    importance.components.personal *= 1.1;
                }
            }
            
            // Recalculate total
            importance.total = importance.components.weighted_sum();
            
            // Store updated importance
            self.storage.store_importance_score(entity_key, &importance)?;
            
            // Update cache
            self.importance_cache.put(entity_key.clone(), importance);
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub total: f32,
    pub components: ImportanceComponents,
    pub ai_score: f32,
    pub timestamp: std::time::Instant,
    pub context_dependent: bool,
    pub confidence: f32,
    pub entity_key: EntityKey,
}

#[derive(Debug, Clone)]
pub struct ImportanceComponents {
    pub personal: f32,
    pub goal: f32,
    pub survival: f32,
    pub social: f32,
    pub novelty: f32,
}

impl ImportanceComponents {
    pub fn weighted_sum(&self) -> f32 {
        // Default weights - can be learned/adapted
        self.personal * 0.3 +
        self.goal * 0.25 +
        self.survival * 0.2 +
        self.social * 0.15 +
        self.novelty * 0.1
    }
}
```

### Task 23.2: Priority-Based Processing
**File**: `src/emotional/priority_processing.rs` (new file)
```rust
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use crate::core::types::EntityKey;
use std::sync::Arc;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

pub struct PriorityProcessor {
    priority_queue: BinaryHeap<PrioritizedEntity>,
    processing_capacity: ProcessingCapacity,
    urgency_calculator: UrgencyCalculator,
    // Integration with existing systems
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
}

#[derive(Clone, Debug)]
pub struct PrioritizedEntity {
    pub entity_key: EntityKey,
    pub priority: f32,
    pub urgency: f32,
    pub importance: f32,
    pub emotional_weight: f32,
}

impl PartialEq for PrioritizedEntity {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedEntity {}

impl PartialOrd for PrioritizedEntity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl Ord for PrioritizedEntity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority).unwrap_or(Ordering::Equal)
    }
}

impl PriorityProcessor {
    pub fn new(
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Self {
        Self {
            priority_queue: BinaryHeap::new(),
            processing_capacity: ProcessingCapacity::default(),
            urgency_calculator: UrgencyCalculator::new(),
            storage,
            monitor,
        }
    }
    
    pub async fn prioritize_for_consolidation(&mut self,
        entity_keys: Vec<EntityKey>
    ) -> Result<Vec<EntityKey>, crate::error::GraphError> {
        let _timer = self.monitor.start_timer("consolidation_prioritization");
        
        // Clear and rebuild priority queue
        self.priority_queue.clear();
        
        // Calculate priorities for all entities in parallel
        let priority_futures: Vec<_> = entity_keys.into_iter()
            .map(|key| async {
                let priority = self.calculate_consolidation_priority(&key).await?;
                Ok(PrioritizedEntity {
                    entity_key: key.clone(),
                    priority,
                    urgency: 0.0, // Will be filled in
                    importance: 0.0, // Will be filled in
                    emotional_weight: 0.0, // Will be filled in
                })
            })
            .collect();
        
        let priorities: Result<Vec<_>, crate::error::GraphError> = 
            futures::future::try_join_all(priority_futures).await;
        
        for prioritized in priorities? {
            self.priority_queue.push(prioritized);
        }
        
        // Take top entities up to capacity
        let mut selected = Vec::new();
        let mut used_capacity = 0.0;
        
        while let Some(prioritized) = self.priority_queue.pop() {
            let required_capacity = self.estimate_processing_capacity(&prioritized.entity_key).await?;
            if used_capacity + required_capacity <= self.processing_capacity.total {
                selected.push(prioritized.entity_key);
                used_capacity += required_capacity;
            } else {
                break;
            }
        }
        
        self.monitor.record_metric("entities_prioritized_for_consolidation", selected.len() as f32);
        self.monitor.record_metric("consolidation_capacity_used", used_capacity);
        
        Ok(selected)
    }
    
    async fn calculate_consolidation_priority(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Load importance score from storage
        let importance = self.storage.load_importance_score(entity_key)?
            .map(|score| score.total)
            .unwrap_or(0.5);
        
        // Calculate urgency
        let urgency = self.urgency_calculator.calculate(entity_key).await?;
        
        // Load emotional metadata
        let emotional_priority = self.storage.load_metadata(entity_key, "emotional_tag")?
            .and_then(|data| serde_json::from_slice::<EmotionalTag>(&data).ok())
            .map(|tag| tag.intensity * tag.valence.abs())
            .unwrap_or(0.0);
        
        // Calculate recency weight
        let recency = self.calculate_recency_weight(entity_key).await?;
        
        // Combine factors using learned weights
        let priority = importance * 0.4 + urgency * 0.3 + emotional_priority * 0.2 + recency * 0.1;
        
        Ok(priority)
    }
    
    async fn estimate_processing_capacity(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Estimate based on entity complexity and relationship count
        let entity_data = self.storage.get_entity(entity_key).await?;
        let relationship_count = self.storage.get_relationship_count(entity_key).await?;
        
        // More complex entities require more processing capacity
        let base_capacity = 1.0;
        let complexity_factor = entity_data.attributes.len() as f32 * 0.1;
        let relationship_factor = relationship_count as f32 * 0.05;
        
        Ok(base_capacity + complexity_factor + relationship_factor)
    }
    
    async fn calculate_recency_weight(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Get last access time from storage
        let last_access = self.storage.get_last_access_time(entity_key)?
            .unwrap_or_else(|| std::time::SystemTime::now());
        
        let elapsed = std::time::SystemTime::now()
            .duration_since(last_access)
            .unwrap_or_default();
        
        // Exponential decay based on time
        let recency_weight = (-elapsed.as_secs_f32() / 86400.0).exp(); // Daily half-life
        
        Ok(recency_weight)
    }
}

pub struct UrgencyCalculator {
    deadline_weights: HashMap<DeadlineType, f32>,
    decay_rates: HashMap<UrgencyType, f32>,
}

impl UrgencyCalculator {
    pub fn new() -> Self {
        Self {
            deadline_weights: [
                (DeadlineType::Immediate, 1.0),
                (DeadlineType::Daily, 0.8),
                (DeadlineType::Weekly, 0.6),
                (DeadlineType::Monthly, 0.4),
                (DeadlineType::None, 0.1),
            ].iter().cloned().collect(),
            decay_rates: [
                (UrgencyType::MemoryDecay, 0.1),
                (UrgencyType::GoalRelevance, 0.05),
                (UrgencyType::ExternalDeadline, 0.2),
            ].iter().cloned().collect(),
        }
    }
    
    pub async fn calculate(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Check for deadlines associated with this entity
        // This would typically involve checking calendar, task lists, etc.
        let deadline_urgency = self.calculate_deadline_urgency(entity_key).await?;
        
        // Check memory decay urgency
        let decay_urgency = self.calculate_decay_urgency(entity_key).await?;
        
        // Check goal relevance urgency
        let goal_urgency = self.calculate_goal_urgency(entity_key).await?;
        
        // Combine urgency factors
        let total_urgency = deadline_urgency.max(decay_urgency).max(goal_urgency);
        
        Ok(total_urgency)
    }
    
    async fn calculate_deadline_urgency(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // This would check for any deadlines associated with the entity
        // For now, return a placeholder
        Ok(0.0)
    }
    
    async fn calculate_decay_urgency(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Calculate urgency based on memory decay
        // Entities that are forgetting rapidly need urgent attention
        Ok(0.0)
    }
    
    async fn calculate_goal_urgency(&self, entity_key: &EntityKey) -> Result<f32, crate::error::GraphError> {
        // Calculate urgency based on relevance to current goals
        Ok(0.0)
    }
}

#[derive(Default)]
pub struct ProcessingCapacity {
    pub total: f32,
    pub available: f32,
    pub consolidation_capacity: f32,
    pub emotional_processing_capacity: f32,
    pub metacognitive_capacity: f32,
}
```

## Week 24: Integration and Testing

### Task 24.1: MCP Tool Integration
**File**: `src/mcp/emotional_metacognitive_handlers.rs` (new file)
```rust
use crate::mcp::llm_friendly_server::types::{LLMMCPTool, LLMExample};
use crate::emotional::EmotionalMemory;
use crate::metacognitive::Metamemory;
use serde_json::json;
use std::sync::Arc;

/// Add emotional and metacognitive capabilities to MCP server
pub fn register_emotional_metacognitive_tools() -> Vec<LLMMCPTool> {
    vec![
        // Emotional tagging tool
        LLMMCPTool {
            name: "tag_emotional_memory".to_string(),
            description: "Tag memory content with emotional metadata using AI models".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to analyze for emotional significance"
                    },
                    "context": {
                        "type": "object",
                        "description": "Emotional context information",
                        "properties": {
                            "current_mood": {"type": "string"},
                            "situation": {"type": "string"}
                        }
                    }
                },
                "required": ["content"]
            }),
            examples: vec![
                LLMExample {
                    description: "Tag a memory about winning an award".to_string(),
                    input: json!({
                        "content": "I won the Nobel Prize for my research",
                        "context": {
                            "current_mood": "euphoric",
                            "situation": "award_ceremony"
                        }
                    }),
                    output: json!({
                        "emotional_tag": {
                            "valence": 0.95,
                            "arousal": 0.85,
                            "emotions": {
                                "joy": 0.9,
                                "pride": 0.8,
                                "surprise": 0.6
                            },
                            "intensity": 0.9
                        }
                    })
                }
            ]
        },
        
        // Metacognitive assessment tool
        LLMMCPTool {
            name: "assess_confidence".to_string(),
            description: "Assess confidence in memory accuracy using metacognitive models".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query or retrieval cue to assess"
                    },
                    "retrieved_content": {
                        "type": "string",
                        "description": "Content that was retrieved"
                    }
                },
                "required": ["query", "retrieved_content"]
            }),
            examples: vec![
                LLMExample {
                    description: "Assess confidence in historical fact recall".to_string(),
                    input: json!({
                        "query": "When did World War II end?",
                        "retrieved_content": "World War II ended in 1945"
                    }),
                    output: json!({
                        "confidence": 0.92,
                        "fok_strength": 0.88,
                        "basis": {
                            "fluency": 0.9,
                            "vividness": 0.7,
                            "consistency": 0.95
                        }
                    })
                }
            ]
        },
        
        // Enhanced validation tool with emotional/metacognitive features
        LLMMCPTool {
            name: "validate_knowledge_enhanced".to_string(),
            description: "Enhanced knowledge validation including emotional and metacognitive assessment".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity to validate (optional)",
                        "maxLength": 128
                    },
                    "include_emotional": {
                        "type": "boolean",
                        "description": "Include emotional consistency checks",
                        "default": true
                    },
                    "include_metacognitive": {
                        "type": "boolean",
                        "description": "Include metacognitive confidence assessment", 
                        "default": true
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence threshold",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }),
            examples: vec![
                LLMExample {
                    description: "Validate knowledge with emotional and metacognitive checks".to_string(),
                    input: json!({
                        "entity": "Einstein",
                        "include_emotional": true,
                        "include_metacognitive": true,
                        "confidence_threshold": 0.8
                    }),
                    output: json!({
                        "validation_result": "passed",
                        "confidence_scores": {
                            "semantic": 0.95,
                            "emotional": 0.88,
                            "metacognitive": 0.82
                        },
                        "issues_found": [],
                        "emotional_consistency": true,
                        "metacognitive_coherence": true
                    })
                }
            ]
        }
    ]
}

pub struct EmotionCognitionBridge {
    emotional_memory: Arc<EmotionalMemory>,
    metamemory: Arc<Metamemory>,
    integration_mode: IntegrationMode,
}

#[derive(Debug, Clone)]
pub enum IntegrationMode {
    EmotionFirst,
    Parallel,
    Regulated,
}

impl EmotionCognitionBridge {
    pub fn new(
        emotional_memory: Arc<EmotionalMemory>,
        metamemory: Arc<Metamemory>
    ) -> Self {
        Self {
            emotional_memory,
            metamemory,
            integration_mode: IntegrationMode::Parallel,
        }
    }
    
    pub async fn process_with_emotion(&mut self,
        cognitive_task: &CognitiveTask,
        emotional_state: &EmotionalState
    ) -> ProcessingResult {
        match self.integration_mode {
            IntegrationMode::EmotionFirst => {
                // Emotion biases initial processing
                let biased_task = self.apply_emotional_bias(
                    cognitive_task,
                    emotional_state
                ).await;
                self.process_cognitively(&biased_task).await
            },
            IntegrationMode::Parallel => {
                // Process in parallel, then integrate
                let (emotional_result, cognitive_result) = tokio::join!(
                    self.process_emotionally(cognitive_task, emotional_state),
                    self.process_cognitively(cognitive_task)
                );
                self.integrate_results(emotional_result, cognitive_result).await
            },
            IntegrationMode::Regulated => {
                // Use emotion regulation before processing
                let regulated_state = self.regulate_emotion_for_task(
                    emotional_state,
                    cognitive_task
                ).await;
                self.process_with_regulated_emotion(cognitive_task, &regulated_state).await
            },
        }
    }
}
```

### Task 24.2: Enhanced Validation Integration
**File**: `src/validation/emotional_metacognitive_validation.rs` (new file)
```rust
use crate::validation::{ValidationResult, HumanValidationInterface};
use crate::emotional::EmotionalMemory;
use crate::metacognitive::Metamemory;
use crate::core::types::EntityKey;
use crate::storage::PersistentMMapStorage;
use crate::monitoring::PerformanceMonitor;
use std::sync::Arc;

/// Enhanced validation system that includes emotional and metacognitive checks
pub struct EnhancedValidationSystem {
    base_validator: Arc<HumanValidationInterface>,
    emotional_memory: Arc<EmotionalMemory>,
    metamemory: Arc<Metamemory>,
    storage: Arc<PersistentMMapStorage>,
    monitor: Arc<PerformanceMonitor>,
}

impl EnhancedValidationSystem {
    pub fn new(
        base_validator: Arc<HumanValidationInterface>,
        emotional_memory: Arc<EmotionalMemory>,
        metamemory: Arc<Metamemory>,
        storage: Arc<PersistentMMapStorage>,
        monitor: Arc<PerformanceMonitor>
    ) -> Self {
        Self {
            base_validator,
            emotional_memory,
            metamemory,
            storage,
            monitor,
        }
    }
    
    /// Enhanced validate_knowledge tool with emotional/metacognitive features
    pub async fn validate_knowledge_enhanced(
        &self,
        entity_key: &EntityKey,
        include_emotional: bool,
        include_metacognitive: bool,
        confidence_threshold: f32
    ) -> Result<EnhancedValidationResult, crate::error::GraphError> {
        let _timer = self.monitor.start_timer("enhanced_validation");
        
        // Start with base validation
        let base_result = self.base_validator.validate_entity(entity_key).await?;
        
        let mut result = EnhancedValidationResult {
            base_result,
            emotional_consistency: None,
            metacognitive_confidence: None,
            overall_score: 0.0,
        };
        
        // Add emotional consistency check
        if include_emotional {
            result.emotional_consistency = Some(
                self.check_emotional_consistency(entity_key).await?
            );
        }
        
        // Add metacognitive confidence assessment
        if include_metacognitive {
            result.metacognitive_confidence = Some(
                self.assess_metacognitive_confidence(entity_key, confidence_threshold).await?
            );
        }
        
        // Calculate overall score
        result.overall_score = self.calculate_combined_score(&result);
        
        self.monitor.record_metric("enhanced_validations_completed", 1.0);
        
        Ok(result)
    }
    
    async fn check_emotional_consistency(
        &self,
        entity_key: &EntityKey
    ) -> Result<EmotionalConsistencyResult, crate::error::GraphError> {
        // Load emotional tags for entity and related entities
        let emotional_context = self.load_emotional_context(entity_key).await?;
        
        // Check for emotional conflicts or inconsistencies
        let is_consistent = self.analyze_emotional_coherence(&emotional_context);
        let consistency_score = self.calculate_emotional_consistency_score(&emotional_context);
        let conflicting_emotions = self.identify_emotional_conflicts(&emotional_context);
        
        Ok(EmotionalConsistencyResult {
            is_consistent,
            consistency_score,
            conflicting_emotions,
        })
    }
    
    async fn assess_metacognitive_confidence(
        &self,
        entity_key: &EntityKey,
        threshold: f32
    ) -> Result<MetacognitiveConfidenceResult, crate::error::GraphError> {
        // Assess confidence in knowledge about this entity
        let confidence_judgment = self.metamemory.assess_entity_confidence(entity_key).await?;
        
        Ok(MetacognitiveConfidenceResult {
            confidence_score: confidence_judgment.confidence,
            calibration_quality: confidence_judgment.calibration_quality,
            uncertainty_factors: confidence_judgment.uncertainty_sources,
            meets_threshold: confidence_judgment.confidence >= threshold,
        })
    }
    
    async fn load_emotional_context(
        &self,
        entity_key: &EntityKey
    ) -> Result<EmotionalContext, crate::error::GraphError> {
        // Load emotional tags for the entity and its related entities
        let mut context = EmotionalContext::new();
        
        // Load direct emotional tag
        if let Some(tag_data) = self.storage.load_metadata(entity_key, "emotional_tag")? {
            if let Ok(tag) = serde_json::from_slice::<EmotionalTag>(&tag_data) {
                context.primary_emotion = Some(tag);
            }
        }
        
        // Load related entities' emotional contexts
        let related_entities = self.storage.get_related_entities(entity_key, 1).await?;
        for related_key in related_entities {
            if let Some(tag_data) = self.storage.load_metadata(&related_key, "emotional_tag")? {
                if let Ok(tag) = serde_json::from_slice::<EmotionalTag>(&tag_data) {
                    context.related_emotions.push((related_key, tag));
                }
            }
        }
        
        Ok(context)
    }
    
    fn analyze_emotional_coherence(&self, context: &EmotionalContext) -> bool {
        // Check if emotions are coherent and not contradictory
        if let Some(primary) = &context.primary_emotion {
            for (_, related_emotion) in &context.related_emotions {
                // Check for major valence conflicts
                let valence_diff = (primary.valence - related_emotion.valence).abs();
                if valence_diff > EMOTIONAL_CONFLICT_THRESHOLD {
                    return false;
                }
            }
        }
        true
    }
    
    fn calculate_emotional_consistency_score(&self, context: &EmotionalContext) -> f32 {
        if context.related_emotions.is_empty() {
            return 1.0; // No conflicts possible
        }
        
        let Some(primary) = &context.primary_emotion else {
            return 0.5; // No primary emotion to compare against
        };
        
        let mut total_consistency = 0.0;
        for (_, related_emotion) in &context.related_emotions {
            let valence_consistency = 1.0 - (primary.valence - related_emotion.valence).abs();
            let arousal_consistency = 1.0 - (primary.arousal - related_emotion.arousal).abs();
            let emotion_consistency = (valence_consistency + arousal_consistency) / 2.0;
            total_consistency += emotion_consistency;
        }
        
        total_consistency / context.related_emotions.len() as f32
    }
    
    fn identify_emotional_conflicts(&self, context: &EmotionalContext) -> Vec<String> {
        let mut conflicts = Vec::new();
        
        if let Some(primary) = &context.primary_emotion {
            for (related_key, related_emotion) in &context.related_emotions {
                let valence_diff = (primary.valence - related_emotion.valence).abs();
                if valence_diff > EMOTIONAL_CONFLICT_THRESHOLD {
                    conflicts.push(format!(
                        "Valence conflict with {}: primary={:.2}, related={:.2}",
                        related_key, primary.valence, related_emotion.valence
                    ));
                }
            }
        }
        
        conflicts
    }
    
    fn calculate_combined_score(&self, result: &EnhancedValidationResult) -> f32 {
        let mut score = result.base_result.score * 0.5; // Base validation worth 50%
        
        if let Some(emotional) = &result.emotional_consistency {
            score += emotional.consistency_score * 0.25; // Emotional consistency worth 25%
        }
        
        if let Some(metacognitive) = &result.metacognitive_confidence {
            score += metacognitive.confidence_score * 0.25; // Metacognitive confidence worth 25%
        }
        
        score.min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedValidationResult {
    pub base_result: ValidationResult,
    pub emotional_consistency: Option<EmotionalConsistencyResult>,
    pub metacognitive_confidence: Option<MetacognitiveConfidenceResult>,
    pub overall_score: f32,
}

#[derive(Debug, Clone)]
pub struct EmotionalConsistencyResult {
    pub is_consistent: bool,
    pub consistency_score: f32,
    pub conflicting_emotions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MetacognitiveConfidenceResult {
    pub confidence_score: f32,
    pub calibration_quality: f32,
    pub uncertainty_factors: Vec<String>,
    pub meets_threshold: bool,
}

#[derive(Debug)]
pub struct EmotionalContext {
    pub primary_emotion: Option<EmotionalTag>,
    pub related_emotions: Vec<(EntityKey, EmotionalTag)>,
}

impl EmotionalContext {
    pub fn new() -> Self {
        Self {
            primary_emotion: None,
            related_emotions: Vec::new(),
        }
    }
}

const EMOTIONAL_CONFLICT_THRESHOLD: f32 = 0.7;
```

### Task 24.3: Comprehensive Testing
**File**: `tests/emotional_metacognitive_tests.rs`
```rust
use llmkg::emotional::{EmotionalMemory, EmotionalTag, BasicEmotions};
use llmkg::metacognitive::{Metamemory, FOKJudgment, ConfidenceJudgment};
use llmkg::validation::emotional_metacognitive_validation::EnhancedValidationSystem;
use llmkg::storage::PersistentMMapStorage;
use llmkg::monitoring::PerformanceMonitor;
use llmkg::core::types::EntityKey;
use std::sync::Arc;

#[tokio::test]
async fn test_emotional_enhancement() {
    let storage = Arc::new(PersistentMMapStorage::new("test_emotional").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    let mut em_system = EmotionalMemory::new(storage.clone(), monitor.clone()).unwrap();
    
    let entity_key = EntityKey::new("test_memory");
    let content = "I won the lottery today!";
    let context = EmotionalContext {
        current_mood: "euphoric".to_string(),
        situation: "lottery_win".to_string(),
    };
    
    let emotional_tag = em_system.tag_memory_with_emotion(&entity_key, content, &context).await.unwrap();
    
    // Emotional memory should have high positive valence and arousal
    assert!(emotional_tag.valence > 0.7);
    assert!(emotional_tag.arousal > 0.6);
    assert!(emotional_tag.emotions.joy > 0.8);
}

#[tokio::test]
async fn test_mood_congruent_retrieval() {
    let storage = Arc::new(PersistentMMapStorage::new("test_mood_retrieval").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    let em_system = EmotionalMemory::new(storage.clone(), monitor.clone()).unwrap();
    
    let sad_mood = EmotionalState::new(-0.7, 0.6); // Sad but aroused
    
    let entity_keys = vec![
        EntityKey::new("happy_memory"),
        EntityKey::new("sad_memory"),
        EntityKey::new("neutral_memory"),
    ];
    
    // Pre-populate with emotional tags
    let happy_tag = EmotionalTag::new(0.8, 0.6, BasicEmotions::happy());
    let sad_tag = EmotionalTag::new(-0.8, 0.7, BasicEmotions::sad());
    let neutral_tag = EmotionalTag::new(0.0, 0.3, BasicEmotions::neutral());
    
    storage.store_metadata(&entity_keys[0], "emotional_tag", &serde_json::to_vec(&happy_tag).unwrap()).unwrap();
    storage.store_metadata(&entity_keys[1], "emotional_tag", &serde_json::to_vec(&sad_tag).unwrap()).unwrap();
    storage.store_metadata(&entity_keys[2], "emotional_tag", &serde_json::to_vec(&neutral_tag).unwrap()).unwrap();
    
    let retrieved = em_system.mood_congruent_retrieval(&sad_mood, &entity_keys).await;
    
    // Sad memory should be first (highest congruence)
    assert_eq!(retrieved[0].0, entity_keys[1]);
    assert!(retrieved[0].1 > retrieved[1].1); // Sad memory has higher congruence score
}

#[tokio::test]
async fn test_metamemory_accuracy() {
    let storage = Arc::new(PersistentMMapStorage::new("test_metamemory").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    let metamemory = Metamemory::new(storage.clone(), monitor.clone()).unwrap();
    
    // Test feeling of knowing
    let strong_cue = RetrievalCue::new("Einstein relativity", 0.9);
    let weak_cue = RetrievalCue::new("Random xyz", 0.1);
    
    let memory_state = MemoryState::default();
    
    let strong_fok = metamemory.monitoring.feeling_of_knowing.assess_feeling_of_knowing(&strong_cue, &memory_state).await;
    let weak_fok = metamemory.monitoring.feeling_of_knowing.assess_feeling_of_knowing(&weak_cue, &memory_state).await;
    
    assert!(strong_fok.strength > weak_fok.strength);
    assert!(strong_fok.confidence > weak_fok.confidence);
}

#[tokio::test]
async fn test_importance_scoring() {
    let storage = Arc::new(PersistentMMapStorage::new("test_importance").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    let scorer = ImportanceScoring::new(storage.clone(), monitor.clone()).unwrap();
    
    let survival_entity = EntityKey::new("snake_encounter");
    let trivial_entity = EntityKey::new("saw_cloud");
    
    let context = Context::default();
    let self_model = SelfModel::default();
    
    let survival_importance = scorer.calculate_importance(&survival_entity, &context, &self_model).await.unwrap();
    let trivial_importance = scorer.calculate_importance(&trivial_entity, &context, &self_model).await.unwrap();
    
    assert!(survival_importance.total > trivial_importance.total * 2.0);
    assert!(survival_importance.components.survival > 0.8);
    assert!(trivial_importance.components.survival < 0.2);
}

#[tokio::test]
async fn test_enhanced_validation() {
    let storage = Arc::new(PersistentMMapStorage::new("test_enhanced_validation").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    
    let base_validator = Arc::new(HumanValidationInterface::new(storage.clone()));
    let emotional_memory = Arc::new(EmotionalMemory::new(storage.clone(), monitor.clone()).unwrap());
    let metamemory = Arc::new(Metamemory::new(storage.clone(), monitor.clone()).unwrap());
    
    let validator = EnhancedValidationSystem::new(
        base_validator,
        emotional_memory,
        metamemory,
        storage.clone(),
        monitor.clone()
    );
    
    let entity_key = EntityKey::new("Einstein");
    
    let result = validator.validate_knowledge_enhanced(
        &entity_key,
        true,  // include_emotional
        true,  // include_metacognitive
        0.8    // confidence_threshold
    ).await.unwrap();
    
    assert!(result.overall_score > 0.0);
    assert!(result.emotional_consistency.is_some());
    assert!(result.metacognitive_confidence.is_some());
}

#[tokio::test]
async fn test_batch_processing() {
    let storage = Arc::new(PersistentMMapStorage::new("test_batch").unwrap());
    let monitor = Arc::new(PerformanceMonitor::new());
    let mut em_system = EmotionalMemory::new(storage.clone(), monitor.clone()).unwrap();
    
    let entity_keys: Vec<EntityKey> = (0..100)
        .map(|i| EntityKey::new(&format!("entity_{}", i)))
        .collect();
    
    let contexts: Vec<EmotionalContext> = entity_keys.iter()
        .map(|_| EmotionalContext::default())
        .collect();
    
    let start_time = std::time::Instant::now();
    
    // Process in batches
    for (chunk_keys, chunk_contexts) in entity_keys.chunks(32).zip(contexts.chunks(32)) {
        let futures: Vec<_> = chunk_keys.iter().zip(chunk_contexts.iter())
            .map(|(key, context)| {
                em_system.tag_memory_with_emotion(key, "test content", context)
            })
            .collect();
        
        let results = futures::future::join_all(futures).await;
        
        // Verify all succeeded
        for result in results {
            assert!(result.is_ok());
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("Batch processed 100 entities in {:?}", elapsed);
    
    // Should process at reasonable speed (target: <5ms per emotion tagging)
    assert!(elapsed.as_millis() < 1000); // Allow 1 second for 100 entities
}

// Helper structs for testing
struct EmotionalContext {
    current_mood: String,
    situation: String,
}

impl Default for EmotionalContext {
    fn default() -> Self {
        Self {
            current_mood: "neutral".to_string(),
            situation: "testing".to_string(),
        }
    }
}

struct EmotionalState {
    valence: f32,
    arousal: f32,
}

impl EmotionalState {
    pub fn new(valence: f32, arousal: f32) -> Self {
        Self { valence, arousal }
    }
    
    pub fn to_vector(&self) -> Vec<f32> {
        vec![self.valence, self.arousal]
    }
}

struct RetrievalCue {
    content: String,
    strength: f32,
}

impl RetrievalCue {
    pub fn new(content: &str, strength: f32) -> Self {
        Self {
            content: content.to_string(),
            strength,
        }
    }
    
    pub fn hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.content.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Default)]
struct MemoryState {
    activation: f32,
}

impl MemoryState {
    pub fn hash(&self) -> u64 {
        (self.activation * 1000.0) as u64
    }
}

#[derive(Default)]
struct Context {
    active_goals: Vec<String>,
    social_context: String,
}

#[derive(Default)]
struct SelfModel {
    personality: Vec<f32>,
}
```

### Task 24.4: Performance Monitoring Integration
**File**: `src/monitoring/emotional_metacognitive_metrics.rs` (new file)
```rust
use crate::monitoring::{PerformanceMonitor, MetricType, Counter, Histogram, Gauge};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance monitoring specifically for emotional and metacognitive systems
pub struct EmotionalMetacognitiveMonitor {
    base_monitor: Arc<PerformanceMonitor>,
    // Emotional metrics
    emotion_tagging_latency: Histogram,
    emotion_cache_hits: Counter,
    emotion_cache_misses: Counter,
    mood_congruence_calculations: Counter,
    emotional_conflicts_detected: Counter,
    // Metacognitive metrics
    confidence_assessments: Counter,
    fok_predictions: Counter,
    strategy_selections: Counter,
    calibration_updates: Counter,
    metacognitive_accuracy: Histogram,
    // Integration metrics
    enhanced_validations: Counter,
    batch_processing_throughput: Histogram,
    combined_scores: Histogram,
}

impl EmotionalMetacognitiveMonitor {
    pub fn new(base_monitor: Arc<PerformanceMonitor>) -> Self {
        Self {
            base_monitor,
            emotion_tagging_latency: Histogram::new("emotion_tagging_latency_ms"),
            emotion_cache_hits: Counter::new("emotion_cache_hits"),
            emotion_cache_misses: Counter::new("emotion_cache_misses"),
            mood_congruence_calculations: Counter::new("mood_congruence_calculations"),
            emotional_conflicts_detected: Counter::new("emotional_conflicts_detected"),
            confidence_assessments: Counter::new("confidence_assessments"),
            fok_predictions: Counter::new("fok_predictions"),
            strategy_selections: Counter::new("strategy_selections"),
            calibration_updates: Counter::new("calibration_updates"),
            metacognitive_accuracy: Histogram::new("metacognitive_accuracy"),
            enhanced_validations: Counter::new("enhanced_validations"),
            batch_processing_throughput: Histogram::new("batch_processing_throughput"),
            combined_scores: Histogram::new("combined_validation_scores"),
        }
    }
    
    /// Start timing an emotional operation
    pub fn start_emotion_timer(&self, operation: &str) -> EmotionalTimer {
        EmotionalTimer {
            operation: operation.to_string(),
            start_time: Instant::now(),
            monitor: self,
        }
    }
    
    /// Record emotional cache hit
    pub fn record_emotion_cache_hit(&self) {
        self.emotion_cache_hits.increment();
        self.base_monitor.record_metric("emotion_cache_hit_rate", 1.0);
    }
    
    /// Record emotional cache miss
    pub fn record_emotion_cache_miss(&self) {
        self.emotion_cache_misses.increment();
        self.base_monitor.record_metric("emotion_cache_hit_rate", 0.0);
    }
    
    /// Record mood congruence calculation
    pub fn record_mood_congruence_calculation(&self, score: f32, latency: Duration) {
        self.mood_congruence_calculations.increment();
        self.base_monitor.record_metric("mood_congruence_score", score);
        self.base_monitor.record_metric("mood_congruence_latency_ms", latency.as_secs_f32() * 1000.0);
    }
    
    /// Record emotional conflict detection
    pub fn record_emotional_conflict(&self, conflict_severity: f32) {
        self.emotional_conflicts_detected.increment();
        self.base_monitor.record_metric("emotional_conflict_severity", conflict_severity);
    }
    
    /// Record confidence assessment
    pub fn record_confidence_assessment(&self, confidence: f32, accuracy: f32, latency: Duration) {
        self.confidence_assessments.increment();
        self.base_monitor.record_metric("confidence_score", confidence);
        self.base_monitor.record_metric("confidence_accuracy", accuracy);
        self.base_monitor.record_metric("confidence_assessment_latency_ms", latency.as_secs_f32() * 1000.0);
        
        // Record calibration error
        let calibration_error = (confidence - accuracy).abs();
        self.base_monitor.record_metric("confidence_calibration_error", calibration_error);
    }
    
    /// Record feeling-of-knowing prediction
    pub fn record_fok_prediction(&self, strength: f32, actual_retrieval_success: bool, latency: Duration) {
        self.fok_predictions.increment();
        self.base_monitor.record_metric("fok_strength", strength);
        self.base_monitor.record_metric("fok_accuracy", if actual_retrieval_success { 1.0 } else { 0.0 });
        self.base_monitor.record_metric("fok_prediction_latency_ms", latency.as_secs_f32() * 1000.0);
    }
    
    /// Record strategy selection
    pub fn record_strategy_selection(&self, strategy: &str, effectiveness: f32, latency: Duration) {
        self.strategy_selections.increment();
        self.base_monitor.record_metric("strategy_effectiveness", effectiveness);
        self.base_monitor.record_metric("strategy_selection_latency_ms", latency.as_secs_f32() * 1000.0);
    }
    
    /// Record enhanced validation
    pub fn record_enhanced_validation(&self, 
        base_score: f32,
        emotional_score: Option<f32>,
        metacognitive_score: Option<f32>,
        combined_score: f32,
        latency: Duration
    ) {
        self.enhanced_validations.increment();
        self.combined_scores.observe(combined_score);
        
        self.base_monitor.record_metric("validation_base_score", base_score);
        self.base_monitor.record_metric("validation_combined_score", combined_score);
        self.base_monitor.record_metric("enhanced_validation_latency_ms", latency.as_secs_f32() * 1000.0);
        
        if let Some(emotional) = emotional_score {
            self.base_monitor.record_metric("validation_emotional_score", emotional);
        }
        
        if let Some(metacognitive) = metacognitive_score {
            self.base_monitor.record_metric("validation_metacognitive_score", metacognitive);
        }
    }
    
    /// Record batch processing performance
    pub fn record_batch_processing(&self, 
        batch_size: usize,
        processing_time: Duration,
        success_count: usize
    ) {
        let throughput = batch_size as f32 / processing_time.as_secs_f32();
        self.batch_processing_throughput.observe(throughput);
        
        self.base_monitor.record_metric("batch_size", batch_size as f32);
        self.base_monitor.record_metric("batch_throughput", throughput);
        self.base_monitor.record_metric("batch_success_rate", success_count as f32 / batch_size as f32);
        self.base_monitor.record_metric("batch_processing_time_ms", processing_time.as_secs_f32() * 1000.0);
    }
    
    /// Get performance summary
    pub fn get_performance_summary(&self) -> EmotionalMetacognitivePerformanceSummary {
        EmotionalMetacognitivePerformanceSummary {
            total_emotion_tags: self.emotion_cache_hits.value() + self.emotion_cache_misses.value(),
            emotion_cache_hit_rate: {
                let total = self.emotion_cache_hits.value() + self.emotion_cache_misses.value();
                if total > 0 {
                    self.emotion_cache_hits.value() as f32 / total as f32
                } else {
                    0.0
                }
            },
            avg_emotion_tagging_latency: self.emotion_tagging_latency.mean(),
            total_confidence_assessments: self.confidence_assessments.value(),
            avg_confidence_accuracy: self.metacognitive_accuracy.mean(),
            total_fok_predictions: self.fok_predictions.value(),
            total_strategy_selections: self.strategy_selections.value(),
            total_enhanced_validations: self.enhanced_validations.value(),
            avg_combined_score: self.combined_scores.mean(),
            avg_batch_throughput: self.batch_processing_throughput.mean(),
        }
    }
}

pub struct EmotionalTimer<'a> {
    operation: String,
    start_time: Instant,
    monitor: &'a EmotionalMetacognitiveMonitor,
}

impl<'a> Drop for EmotionalTimer<'a> {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        
        match self.operation.as_str() {
            "emotion_tagging" => {
                self.monitor.emotion_tagging_latency.observe(elapsed.as_secs_f32() * 1000.0);
            },
            _ => {
                self.monitor.base_monitor.record_metric(
                    &format!("{}_latency_ms", self.operation),
                    elapsed.as_secs_f32() * 1000.0
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmotionalMetacognitivePerformanceSummary {
    pub total_emotion_tags: u64,
    pub emotion_cache_hit_rate: f32,
    pub avg_emotion_tagging_latency: f32,
    pub total_confidence_assessments: u64,
    pub avg_confidence_accuracy: f32,
    pub total_fok_predictions: u64,
    pub total_strategy_selections: u64,
    pub total_enhanced_validations: u64,
    pub avg_combined_score: f32,
    pub avg_batch_throughput: f32,
}

impl EmotionalMetacognitivePerformanceSummary {
    /// Check if performance meets targets
    pub fn meets_performance_targets(&self) -> bool {
        // Target: <5ms for emotion tagging
        let emotion_tagging_ok = self.avg_emotion_tagging_latency < 5.0;
        
        // Target: >90% cache hit rate for emotions
        let cache_hit_rate_ok = self.emotion_cache_hit_rate > 0.9;
        
        // Target: >80% confidence accuracy
        let confidence_accuracy_ok = self.avg_confidence_accuracy > 0.8;
        
        // Target: >100 entities/second batch throughput
        let batch_throughput_ok = self.avg_batch_throughput > 100.0;
        
        emotion_tagging_ok && cache_hit_rate_ok && confidence_accuracy_ok && batch_throughput_ok
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "Emotional & Metacognitive Performance Report\n\
            ============================================\n\
            Emotion Tagging:\n\
            - Total tags created: {}\n\
            - Cache hit rate: {:.1}%\n\
            - Avg latency: {:.2}ms (target: <5ms)\n\
            \n\
            Metacognitive Assessment:\n\
            - Total confidence assessments: {}\n\
            - Avg accuracy: {:.1}%\n\
            - Total FOK predictions: {}\n\
            - Total strategy selections: {}\n\
            \n\
            Enhanced Validation:\n\
            - Total enhanced validations: {}\n\
            - Avg combined score: {:.2}\n\
            \n\
            Batch Processing:\n\
            - Avg throughput: {:.1} entities/second (target: >100)\n\
            \n\
            Performance Status: {}\n",
            self.total_emotion_tags,
            self.emotion_cache_hit_rate * 100.0,
            self.avg_emotion_tagging_latency,
            self.total_confidence_assessments,
            self.avg_confidence_accuracy * 100.0,
            self.total_fok_predictions,
            self.total_strategy_selections,
            self.total_enhanced_validations,
            self.avg_combined_score,
            self.avg_batch_throughput,
            if self.meets_performance_targets() { "MEETS TARGETS" } else { "NEEDS IMPROVEMENT" }
        )
    }
}
```

## Deliverables
1. **AI-powered emotional memory** with integrated RoBERTa-emotion and DistilBERT-VA models using existing Candle infrastructure
2. **Neural metamemory monitoring** with confidence calibration and FOK prediction models
3. **Intelligent strategy selection** with AI-enhanced effectiveness prediction
4. **AI importance scoring** with lightweight neural network scoring
5. **Parallel processing pipeline** optimized for performance using existing batch processing patterns
6. **Comprehensive caching system** for repeated assessments using existing storage patterns
7. **Enhanced MCP tool integration** adding emotional and metacognitive capabilities to existing server
8. **Integrated validation system** extending existing validation with emotional/metacognitive checks
9. **Performance monitoring** using existing monitoring infrastructure with specialized metrics

## Success Criteria
- [ ] Emotion tagging: <5ms with AI models using existing infrastructure
- [ ] Mood congruent retrieval: <10ms for 1000 entities using batch processing
- [ ] FOK prediction accuracy: >85% with neural model
- [ ] Confidence calibration: <3ms per judgment
- [ ] Strategy selection: <4ms with AI recommendation
- [ ] Importance scoring: <2ms with caching using existing patterns
- [ ] Batch processing: 1000+ entities/second using existing BatchProcessor
- [ ] Enhanced validation: Seamless integration with existing validation system
- [ ] MCP tool integration: New tools available in existing MCP server
- [ ] Total system integration: All components work with existing storage, monitoring, and cognitive systems

## Dependencies
- **Existing Infrastructure**: 
  - `src/models/` - Candle framework and model loading
  - `src/storage/PersistentMMapStorage` - For emotional metadata storage
  - `src/monitoring/PerformanceMonitor` - For metrics and performance tracking
  - `src/validation/HumanValidationInterface` - For enhanced validation
  - `src/mcp/` - For tool integration
  - `src/streaming/BatchProcessor` - For batch processing
  - `src/cognitive/CognitiveOrchestrator` - For metacognitive integration
- **New AI Models** (to be added to `src/models/pretrained/`):
  - RoBERTa-emotion (125M params)
  - DistilBERT-VA (66M params)
  - ConfidenceNet (20M params)
  - FOKPredictor (15M params)
  - StrategySelector (10M params)
  - ImportanceScorer (5M params)
- **Candle framework** with INT8 quantization support
- **Tokio** for async parallel processing (already available)
- **DashMap and LruCache** for caching (already available)

## Risks & Mitigations
1. **Model inference latency**
   - Mitigation: INT8 quantization, parallel processing, aggressive caching using existing patterns
2. **Integration complexity with existing systems**
   - Mitigation: Build on existing interfaces and patterns, maintain backward compatibility
3. **Emotion categorization ambiguity**
   - Mitigation: Ensemble approach with multiple models, fallback to rule-based systems
4. **Metacognitive overhead**
   - Mitigation: Async monitoring, batching, selective AI usage based on importance
5. **Memory usage with multiple models**
   - Mitigation: Use existing model sharing patterns, lazy loading, pruning unused models
6. **Storage integration challenges**
   - Mitigation: Use existing metadata storage patterns, maintain consistent interfaces