# Phase 4: Temporal Dynamics

## Overview
**Duration**: 4 weeks  
**Goal**: Implement time-based memory effects including decay, consolidation, and temporal associations  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-3 completion  

## Week 13: Memory Strength and Decay

### Task 13.1: Temporal Memory Model
**File**: `src/temporal/memory_strength.rs` (new file)
```rust
pub struct TemporalMemory {
    memories: HashMap<MemoryId, Memory>,
    decay_model: DecayModel,
    consolidation_schedule: ConsolidationSchedule,
}

pub struct Memory {
    id: MemoryId,
    content: MemoryContent,
    encoding_strength: f32,
    current_strength: f32,
    creation_time: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u32,
    consolidation_level: ConsolidationLevel,
    emotional_weight: f32,
    importance_score: f32,
}

pub enum DecayModel {
    Exponential { half_life: Duration },
    PowerLaw { decay_rate: f32, scaling_factor: f32 },
    Ebbinghaus { initial_retention: f32, decay_constant: f32 },
    Custom { decay_fn: Box<dyn Fn(Duration, u32) -> f32> },
}

impl TemporalMemory {
    pub fn get_memory_strength(&self, memory_id: MemoryId) -> f32 {
        let memory = &self.memories[&memory_id];
        let base_strength = self.calculate_base_strength(memory);
        let temporal_decay = self.calculate_temporal_decay(memory);
        let retrieval_boost = self.calculate_retrieval_strength(memory);
        
        (base_strength * temporal_decay + retrieval_boost)
            .max(0.0)
            .min(1.0)
    }
    
    fn calculate_temporal_decay(&self, memory: &Memory) -> f32 {
        let time_elapsed = Utc::now().signed_duration_since(memory.last_accessed);
        
        match &self.decay_model {
            DecayModel::Ebbinghaus { initial_retention, decay_constant } => {
                let hours = time_elapsed.num_hours() as f32;
                initial_retention * (-hours / decay_constant).exp()
            },
            DecayModel::PowerLaw { decay_rate, scaling_factor } => {
                let days = time_elapsed.num_days() as f32 + 1.0;
                scaling_factor * days.powf(-decay_rate)
            },
            _ => 1.0,
        }
    }
    
    fn calculate_retrieval_strength(&self, memory: &Memory) -> f32 {
        // Spacing effect: distributed practice strengthens memory
        let access_intervals = self.get_access_intervals(memory);
        let spacing_bonus = self.calculate_spacing_bonus(access_intervals);
        
        // Testing effect: retrieval strengthens memory
        let retrieval_bonus = (memory.access_count as f32).ln() * 0.1;
        
        spacing_bonus + retrieval_bonus
    }
}
```

### Task 13.2: Forgetting Curves Implementation
**File**: `src/temporal/forgetting_curves.rs` (new file)
```rust
pub struct ForgettingCurve {
    curve_type: CurveType,
    parameters: CurveParameters,
    modifiers: Vec<CurveModifier>,
}

pub enum CurveType {
    Ebbinghaus,
    PowerLaw,
    Logarithmic,
    Adaptive,  // Learns from user's actual forgetting patterns
}

pub struct CurveParameters {
    initial_strength: f32,
    decay_rate: f32,
    minimum_strength: f32,
    consolidation_threshold: f32,
}

pub enum CurveModifier {
    EmotionalBoost { factor: f32 },
    ImportanceWeight { weight: f32 },
    InterferenceEffect { strength: f32 },
    SleepConsolidation { quality: f32 },
}

impl ForgettingCurve {
    pub fn calculate_retention(&self, 
        time_elapsed: Duration, 
        rehearsal_count: u32,
        context: &MemoryContext
    ) -> f32 {
        let base_retention = match self.curve_type {
            CurveType::Ebbinghaus => self.ebbinghaus_retention(time_elapsed),
            CurveType::PowerLaw => self.power_law_retention(time_elapsed),
            CurveType::Logarithmic => self.logarithmic_retention(time_elapsed),
            CurveType::Adaptive => self.adaptive_retention(time_elapsed, context),
        };
        
        // Apply modifiers
        let modified_retention = self.apply_modifiers(base_retention, context);
        
        // Apply rehearsal effects
        let rehearsal_factor = 1.0 + (rehearsal_count as f32 * 0.2).min(2.0);
        
        (modified_retention * rehearsal_factor).min(1.0)
    }
    
    pub fn predict_optimal_review_time(&self, 
        current_strength: f32, 
        target_strength: f32
    ) -> Duration {
        // Calculate when memory will decay to target strength
        // Used for spaced repetition scheduling
        self.inverse_retention_function(current_strength, target_strength)
    }
}
```

### Task 13.3: Memory Interference Model
**File**: `src/temporal/interference.rs` (new file)
```rust
pub struct InterferenceModel {
    proactive_interference: ProactiveInterference,
    retroactive_interference: RetroactiveInterference,
    similarity_threshold: f32,
}

pub struct ProactiveInterference {
    // Old memories interfere with new learning
    pub fn calculate_interference(&self, 
        new_memory: &Memory,
        existing_memories: &[Memory]
    ) -> f32 {
        let mut interference = 0.0;
        
        for existing in existing_memories {
            if existing.creation_time < new_memory.creation_time {
                let similarity = calculate_similarity(new_memory, existing);
                let recency_weight = self.recency_weight(existing);
                interference += similarity * recency_weight * existing.current_strength;
            }
        }
        
        interference.min(1.0)
    }
}

pub struct RetroactiveInterference {
    // New memories interfere with old memories
    pub fn apply_interference(&mut self,
        target_memory: &mut Memory,
        interfering_memories: &[Memory]
    ) {
        for interferer in interfering_memories {
            if interferer.creation_time > target_memory.creation_time {
                let similarity = calculate_similarity(target_memory, interferer);
                let interference_strength = similarity * interferer.encoding_strength;
                target_memory.current_strength *= 1.0 - (interference_strength * 0.3);
            }
        }
    }
}
```

## Week 14: Temporal Associations

### Task 14.1: Temporal Context Model
**File**: `src/temporal/temporal_context.rs` (new file)
```rust
pub struct TemporalContext {
    time_bins: BTreeMap<DateTime<Utc>, Vec<MemoryId>>,
    temporal_links: HashMap<MemoryId, Vec<TemporalLink>>,
    context_drift_rate: f32,
}

pub struct TemporalLink {
    from_memory: MemoryId,
    to_memory: MemoryId,
    temporal_distance: Duration,
    link_strength: f32,
    link_type: TemporalLinkType,
}

pub enum TemporalLinkType {
    Sequential,      // A happened before B
    Simultaneous,    // A and B happened together
    Periodic,        // A and B have periodic relationship
    Causal,          // A caused B
}

impl TemporalContext {
    pub fn encode_temporal_context(&mut self, 
        memory: &Memory,
        window: Duration
    ) -> TemporalSignature {
        // Find memories in temporal window
        let nearby_memories = self.find_memories_in_window(
            memory.creation_time, 
            window
        );
        
        // Create temporal links
        for nearby in nearby_memories {
            let link = TemporalLink {
                from_memory: memory.id,
                to_memory: nearby.id,
                temporal_distance: self.calculate_temporal_distance(memory, nearby),
                link_strength: self.calculate_link_strength(memory, nearby),
                link_type: self.infer_link_type(memory, nearby),
            };
            
            self.temporal_links.entry(memory.id)
                .or_insert_with(Vec::new)
                .push(link);
        }
        
        TemporalSignature {
            timestamp: memory.creation_time,
            context_vector: self.generate_context_vector(memory),
            linked_memories: self.temporal_links[&memory.id].clone(),
        }
    }
    
    pub fn retrieve_by_temporal_context(&self, 
        query_time: DateTime<Utc>,
        context_similarity_threshold: f32
    ) -> Vec<MemoryId> {
        // Use temporal context to aid retrieval
        let query_context = self.reconstruct_context(query_time);
        
        self.memories.iter()
            .filter(|(_, memory)| {
                let similarity = self.context_similarity(
                    &query_context, 
                    &memory.temporal_signature
                );
                similarity > context_similarity_threshold
            })
            .map(|(id, _)| *id)
            .collect()
    }
}
```

### Task 14.2: Chronological Organization
**File**: `src/temporal/chronological_organization.rs` (new file)
```rust
pub struct ChronologicalOrganizer {
    timeline: Timeline,
    episodes: Vec<Episode>,
    temporal_clusters: Vec<TemporalCluster>,
}

pub struct Timeline {
    events: BTreeMap<DateTime<Utc>, Vec<Event>>,
    granularity: TimeGranularity,
    compression_strategy: CompressionStrategy,
}

pub struct Episode {
    id: EpisodeId,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    key_memories: Vec<MemoryId>,
    theme: Option<String>,
    emotional_tone: EmotionalTone,
    significance_score: f32,
}

impl ChronologicalOrganizer {
    pub fn organize_into_episodes(&mut self, memories: &[Memory]) -> Vec<Episode> {
        // Segment continuous experiences into episodes
        let segments = self.segment_by_temporal_gaps(memories);
        
        segments.into_iter().map(|segment| {
            Episode {
                id: EpisodeId::new(),
                start_time: segment.first().unwrap().creation_time,
                end_time: segment.last().unwrap().creation_time,
                key_memories: self.extract_key_memories(&segment),
                theme: self.infer_episode_theme(&segment),
                emotional_tone: self.analyze_emotional_tone(&segment),
                significance_score: self.calculate_significance(&segment),
            }
        }).collect()
    }
    
    pub fn temporal_compression(&mut self, timeline: &Timeline) -> CompressedTimeline {
        // Compress timeline while preserving important events
        match timeline.compression_strategy {
            CompressionStrategy::Logarithmic => {
                // Recent events have higher resolution
                self.logarithmic_compression(timeline)
            },
            CompressionStrategy::ImportanceBased => {
                // Keep important events regardless of age
                self.importance_based_compression(timeline)
            },
            CompressionStrategy::Adaptive => {
                // Adjust compression based on memory density
                self.adaptive_compression(timeline)
            },
        }
    }
}
```

### Task 14.3: Time-based Retrieval Cues
**File**: `src/temporal/temporal_retrieval.rs` (new file)
```rust
pub struct TemporalRetrieval {
    time_cues: HashMap<TimeCueType, Vec<MemoryId>>,
    periodicity_detector: PeriodicityDetector,
}

pub enum TimeCueType {
    TimeOfDay(NaiveTime),
    DayOfWeek(Weekday),
    Season(Season),
    Holiday(String),
    Anniversary(NaiveDate),
    RelativeTime(RelativeTimeDescription),
}

impl TemporalRetrieval {
    pub fn retrieve_by_time_cue(&self, cue: TimeCue) -> Vec<RetrievalResult> {
        let mut results = Vec::new();
        
        match cue {
            TimeCue::SpecificTime(datetime) => {
                // Find memories from that specific time
                results.extend(self.retrieve_exact_time(datetime));
            },
            TimeCue::TimeOfDay(time) => {
                // Find memories from similar times of day
                results.extend(self.retrieve_by_time_of_day(time));
            },
            TimeCue::Periodic(period) => {
                // Find memories with matching periodicity
                results.extend(self.retrieve_periodic_memories(period));
            },
            TimeCue::Relative(description) => {
                // "Last week", "Two months ago", etc.
                results.extend(self.retrieve_relative_time(description));
            },
        }
        
        self.rank_by_temporal_relevance(results)
    }
    
    pub fn detect_temporal_patterns(&mut self, memories: &[Memory]) -> Vec<TemporalPattern> {
        self.periodicity_detector.detect_patterns(memories)
    }
}
```

## Week 15: Memory Consolidation

### Task 15.1: Sleep-like Consolidation
**File**: `src/temporal/consolidation.rs` (new file)
```rust
pub struct SleepConsolidation {
    consolidation_phases: Vec<ConsolidationPhase>,
    replay_probability: f32,
    consolidation_strength: f32,
}

pub enum ConsolidationPhase {
    QuietWake { duration: Duration },
    NREM { stage: u8, duration: Duration },
    REM { duration: Duration },
}

impl SleepConsolidation {
    pub fn run_consolidation_cycle(&mut self, 
        memory_store: &mut TemporalMemory
    ) -> ConsolidationReport {
        let mut report = ConsolidationReport::new();
        
        for phase in &self.consolidation_phases {
            match phase {
                ConsolidationPhase::NREM { stage, duration } => {
                    // Consolidate declarative memories
                    let consolidated = self.consolidate_declarative_memories(
                        memory_store, 
                        *stage, 
                        *duration
                    );
                    report.nrem_consolidated += consolidated.len();
                },
                ConsolidationPhase::REM { duration } => {
                    // Consolidate procedural and emotional memories
                    let consolidated = self.consolidate_procedural_memories(
                        memory_store, 
                        *duration
                    );
                    report.rem_consolidated += consolidated.len();
                },
                _ => {},
            }
        }
        
        // Memory replay
        self.replay_important_memories(memory_store);
        
        // Synaptic homeostasis - weaken unimportant connections
        self.synaptic_downscaling(memory_store);
        
        report
    }
    
    fn replay_important_memories(&self, memory_store: &mut TemporalMemory) {
        let important_memories = memory_store.get_high_importance_memories();
        
        for memory in important_memories {
            if rand::random::<f32>() < self.replay_probability {
                // Strengthen memory through replay
                memory.current_strength *= 1.0 + self.consolidation_strength;
                memory.consolidation_level = memory.consolidation_level.upgrade();
            }
        }
    }
}
```

### Task 15.2: Systems Consolidation
**File**: `src/temporal/systems_consolidation.rs` (new file)
```rust
pub struct SystemsConsolidation {
    transfer_threshold: f32,
    semantic_extractor: SemanticExtractor,
}

impl SystemsConsolidation {
    pub fn transfer_to_semantic(&mut self, 
        episodic_memory: &Memory,
        semantic_store: &mut SemanticMemory
    ) -> Option<SemanticKnowledge> {
        if episodic_memory.consolidation_level < ConsolidationLevel::WellConsolidated {
            return None;
        }
        
        // Extract semantic content from episodic memory
        let semantic_features = self.semantic_extractor.extract(episodic_memory);
        
        // Remove episodic details, keep semantic knowledge
        let semantic_knowledge = SemanticKnowledge {
            concepts: semantic_features.concepts,
            relationships: semantic_features.relationships,
            general_rules: self.infer_general_rules(&semantic_features),
            source_episodes: vec![episodic_memory.id],
        };
        
        semantic_store.integrate_knowledge(semantic_knowledge.clone());
        
        Some(semantic_knowledge)
    }
    
    pub fn gradient_consolidation(&mut self,
        recent_memories: &[Memory],
        remote_memories: &mut [Memory]
    ) {
        // Gradually transfer dependency from hippocampus to cortex
        for memory in recent_memories {
            let age = Utc::now().signed_duration_since(memory.creation_time);
            let transfer_rate = self.calculate_transfer_rate(age);
            
            if transfer_rate > self.transfer_threshold {
                self.strengthen_cortical_representation(memory);
                self.weaken_hippocampal_dependence(memory);
            }
        }
    }
}
```

## Week 16: Advanced Temporal Features

### Task 16.1: Temporal Reasoning Engine
**File**: `src/temporal/temporal_reasoning.rs` (new file)
```rust
pub struct TemporalReasoner {
    timeline_analyzer: TimelineAnalyzer,
    causal_inference: CausalInference,
}

impl TemporalReasoner {
    pub fn reason_about_time(&self, query: TemporalQuery) -> TemporalInference {
        match query {
            TemporalQuery::WhenDidXHappen(event) => {
                self.locate_event_in_time(event)
            },
            TemporalQuery::WhatHappenedBefore(event) => {
                self.find_preceding_events(event)
            },
            TemporalQuery::Duration(start, end) => {
                self.calculate_duration_context(start, end)
            },
            TemporalQuery::CausalChain(event) => {
                self.trace_causal_chain(event)
            },
        }
    }
    
    pub fn mental_time_travel(&self, 
        target_time: DateTime<Utc>,
        perspective: Perspective
    ) -> MentalTimeState {
        // Reconstruct mental state at target time
        let context = self.reconstruct_temporal_context(target_time);
        let active_memories = self.get_active_memories_at_time(target_time);
        let emotional_state = self.reconstruct_emotional_state(target_time);
        
        MentalTimeState {
            target_time,
            context,
            active_memories,
            emotional_state,
            perspective,
        }
    }
}
```

### Task 16.2: Temporal API Integration
**File**: `src/mcp/llm_friendly_server/handlers/temporal.rs`
```rust
pub async fn handle_time_travel_query(params: Value) -> Result<Value> {
    let query_type = params["query_type"].as_str().unwrap();
    let entity = params.get("entity").and_then(|v| v.as_str());
    let timestamp = params.get("timestamp").and_then(|v| v.as_str());
    
    let result = match query_type {
        "point_in_time" => {
            let time = DateTime::parse_from_rfc3339(timestamp.unwrap())?;
            temporal_memory.get_state_at_time(time, entity)
        },
        "evolution_tracking" => {
            temporal_memory.track_evolution(entity.unwrap(), time_range)
        },
        "memory_strength" => {
            temporal_memory.get_strength_over_time(entity.unwrap())
        },
        _ => return Err("Unknown query type"),
    };
    
    Ok(json!({
        "query_type": query_type,
        "results": result,
        "temporal_metadata": {
            "current_time": Utc::now(),
            "memory_age_distribution": temporal_memory.get_age_distribution(),
        }
    }))
}
```

### Task 16.3: Performance and Testing
**File**: `tests/temporal_dynamics_tests.rs`
```rust
#[test]
fn test_forgetting_curve() {
    let mut memory = create_test_memory();
    let curve = ForgettingCurve::ebbinghaus();
    
    // Test immediate recall
    assert_eq!(curve.calculate_retention(Duration::seconds(0), 0), 1.0);
    
    // Test after 1 hour
    let one_hour = curve.calculate_retention(Duration::hours(1), 0);
    assert!(one_hour < 0.6 && one_hour > 0.4);
    
    // Test spacing effect
    let with_rehearsal = curve.calculate_retention(Duration::hours(24), 3);
    let without_rehearsal = curve.calculate_retention(Duration::hours(24), 0);
    assert!(with_rehearsal > without_rehearsal);
}

#[test]
fn test_temporal_associations() {
    // Test that memories close in time are associated
}

#[test]
fn test_consolidation() {
    // Test memory consolidation effects
}
```

## Deliverables
1. **Memory strength model** with multiple decay functions
2. **Forgetting curves** implementation with modifiers
3. **Temporal context** encoding and retrieval
4. **Sleep-like consolidation** system
5. **Temporal reasoning** capabilities
6. **Time-based retrieval** mechanisms

## Success Criteria
- [ ] Memory decay follows psychological research curves
- [ ] Spacing effect demonstrable in tests
- [ ] Consolidation strengthens important memories
- [ ] Temporal context aids retrieval by 30%+
- [ ] Time-based queries return relevant results
- [ ] Performance remains under 100ms

## Dependencies
- DateTime handling library
- Statistical modeling tools
- Time series analysis library

## Risks & Mitigations
1. **Computational cost of continuous decay**
   - Mitigation: Lazy evaluation, batch updates
2. **Memory bloat from temporal metadata**
   - Mitigation: Compression, periodic cleanup
3. **Complexity of temporal reasoning**
   - Mitigation: Start simple, iterate