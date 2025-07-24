# Phase 5: Episodic & Working Memory

## Overview
**Duration**: 4 weeks  
**Goal**: Implement episodic memory with rich context and working memory with cognitive constraints  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-4 completion  

## Week 17: Episodic Memory System

### Task 17.1: Rich Episodic Encoding
**File**: `src/episodic/episodic_memory.rs` (new file)
```rust
pub struct EpisodicMemory {
    episodes: HashMap<EpisodeId, Episode>,
    episode_index: EpisodeIndex,
    autobiographical_organizer: AutobiographicalOrganizer,
}

pub struct Episode {
    id: EpisodeId,
    // What happened
    event: Event,
    
    // Where it happened
    spatial_context: SpatialContext,
    
    // When it happened
    temporal_context: TemporalContext,
    
    // Who was involved
    social_context: SocialContext,
    
    // Internal state
    cognitive_context: CognitiveContext,
    emotional_context: EmotionalContext,
    
    // Sensory details
    sensory_details: SensorySnapshot,
    
    // Narrative structure
    narrative: Option<Narrative>,
    
    // Metadata
    encoding_strength: f32,
    vividness: f32,
    personal_significance: f32,
    rehearsal_count: u32,
}

pub struct SpatialContext {
    location: Location,
    spatial_layout: Option<SpatialMap>,
    navigation_path: Option<Vec<Position>>,
    landmarks: Vec<Landmark>,
}

pub struct SocialContext {
    people_present: Vec<PersonId>,
    social_roles: HashMap<PersonId, SocialRole>,
    interactions: Vec<SocialInteraction>,
    conversation_snippets: Vec<ConversationMemory>,
}

pub struct CognitiveContext {
    active_goals: Vec<Goal>,
    thoughts: Vec<Thought>,
    attention_focus: AttentionState,
    mental_models: Vec<MentalModel>,
}

impl EpisodicMemory {
    pub fn encode_episode(&mut self, 
        event: Event,
        context: FullContext,
        encoding_params: EncodingParameters
    ) -> EpisodeId {
        let episode = Episode {
            id: EpisodeId::new(),
            event: event.clone(),
            spatial_context: self.extract_spatial_context(&context),
            temporal_context: self.extract_temporal_context(&context),
            social_context: self.extract_social_context(&context),
            cognitive_context: self.extract_cognitive_context(&context),
            emotional_context: self.extract_emotional_context(&context),
            sensory_details: self.capture_sensory_snapshot(&context),
            narrative: None,
            encoding_strength: encoding_params.calculate_strength(&event, &context),
            vividness: self.calculate_vividness(&context),
            personal_significance: self.evaluate_significance(&event, &context),
            rehearsal_count: 0,
        };
        
        // Index episode for efficient retrieval
        self.episode_index.index_episode(&episode);
        
        // Organize autobiographically
        self.autobiographical_organizer.integrate_episode(&episode);
        
        let id = episode.id;
        self.episodes.insert(id, episode);
        id
    }
    
    pub fn retrieve_episode(&self, cues: RetrievalCues) -> Option<Episode> {
        // Use multiple cues for retrieval
        let candidates = self.episode_index.search(cues);
        
        // Pattern completion for partial cues
        let completed = self.pattern_complete_episodes(candidates, &cues);
        
        // Return most activated episode
        completed.into_iter()
            .max_by_key(|e| OrderedFloat(e.activation_strength))
            .map(|e| e.episode)
    }
}
```

### Task 17.2: Autobiographical Organization
**File**: `src/episodic/autobiographical_memory.rs` (new file)
```rust
pub struct AutobiographicalMemory {
    life_periods: Vec<LifePeriod>,
    general_events: HashMap<GeneralEventId, GeneralEvent>,
    specific_episodes: HashMap<EpisodeId, Episode>,
    self_schemas: Vec<SelfSchema>,
}

pub struct LifePeriod {
    id: LifePeriodId,
    name: String,  // "College Years", "First Job", etc.
    start_date: NaiveDate,
    end_date: Option<NaiveDate>,
    themes: Vec<LifeTheme>,
    key_events: Vec<GeneralEventId>,
    self_concept: SelfConcept,
    typical_activities: Vec<Activity>,
    important_people: Vec<PersonId>,
    defining_episodes: Vec<EpisodeId>,
}

pub struct GeneralEvent {
    id: GeneralEventId,
    description: String,  // "Sunday dinners at grandma's"
    frequency: EventFrequency,
    typical_sequence: Vec<EventStep>,
    variations: Vec<EventVariation>,
    associated_episodes: Vec<EpisodeId>,
    emotional_tone: EmotionalSignature,
}

pub enum EventFrequency {
    Daily,
    Weekly,
    Monthly,
    Occasional,
    Unique,
}

impl AutobiographicalMemory {
    pub fn organize_hierarchically(&mut self) {
        // Level 1: Life periods (years-decades)
        self.identify_life_periods();
        
        // Level 2: General events (days-months)
        self.extract_general_events();
        
        // Level 3: Specific episodes (seconds-hours)
        self.link_specific_episodes();
        
        // Create cross-level associations
        self.build_hierarchical_links();
    }
    
    pub fn retrieve_by_life_period(&self, 
        period_cue: &str
    ) -> AutobiographicalRetrieval {
        let matching_period = self.find_matching_period(period_cue);
        
        AutobiographicalRetrieval {
            life_period: matching_period.clone(),
            representative_events: self.get_representative_events(&matching_period),
            vivid_episodes: self.get_vivid_episodes(&matching_period),
            self_at_time: self.reconstruct_self_concept(&matching_period),
        }
    }
    
    pub fn generate_life_narrative(&self, 
        start: Option<NaiveDate>,
        end: Option<NaiveDate>
    ) -> LifeNarrative {
        let periods = self.get_periods_in_range(start, end);
        
        LifeNarrative {
            chapters: periods.iter().map(|p| self.create_chapter(p)).collect(),
            themes: self.extract_overarching_themes(&periods),
            character_development: self.trace_self_evolution(&periods),
            key_turning_points: self.identify_turning_points(&periods),
        }
    }
}
```

### Task 17.3: Flashbulb Memories
**File**: `src/episodic/flashbulb_memory.rs` (new file)
```rust
pub struct FlashbulbMemory {
    episode: Episode,
    // Special properties of flashbulb memories
    consequentiality: f32,  // Personal importance
    surprise_level: f32,
    emotional_intensity: f32,
    
    // Canonical categories (Brown & Kulik)
    informant: Option<String>,  // Who told you
    own_affect: EmotionalState,  // How you felt
    aftermath: Vec<Consequence>,  // What happened next
    others_affect: HashMap<PersonId, EmotionalState>,
    ongoing_activity: Activity,  // What you were doing
    
    // Confidence vs accuracy
    confidence_rating: f32,
    verified_accuracy: Option<f32>,
}

impl FlashbulbMemory {
    pub fn encode_flashbulb(&mut self,
        triggering_event: &Event,
        context: &FullContext
    ) -> Option<FlashbulbMemory> {
        let surprise = self.calculate_surprise(triggering_event, context);
        let consequentiality = self.evaluate_consequentiality(triggering_event);
        let emotion = self.measure_emotional_intensity(context);
        
        // Flashbulb memories form only under specific conditions
        if surprise > SURPRISE_THRESHOLD && 
           consequentiality > CONSEQUENTIALITY_THRESHOLD &&
           emotion > EMOTION_THRESHOLD {
            
            Some(FlashbulbMemory {
                episode: self.create_detailed_episode(triggering_event, context),
                consequentiality,
                surprise_level: surprise,
                emotional_intensity: emotion,
                informant: self.extract_informant(context),
                own_affect: context.emotional_context.current_state.clone(),
                aftermath: self.predict_consequences(triggering_event),
                others_affect: self.observe_others_reactions(context),
                ongoing_activity: context.current_activity.clone(),
                confidence_rating: 0.95,  // Initially very high
                verified_accuracy: None,
            })
        } else {
            None
        }
    }
    
    pub fn rehearse_flashbulb(&mut self, flashbulb: &mut FlashbulbMemory) {
        // Flashbulb memories change with rehearsal
        flashbulb.confidence_rating = (flashbulb.confidence_rating * 1.05).min(1.0);
        
        // But accuracy may decrease
        if let Some(accuracy) = flashbulb.verified_accuracy {
            // Introduce small distortions
            flashbulb.episode = self.introduce_rehearsal_distortions(
                &flashbulb.episode,
                flashbulb.rehearsal_count
            );
        }
    }
}
```

## Week 18: Working Memory Implementation

### Task 18.1: Limited Capacity Working Memory
**File**: `src/working_memory/working_memory.rs` (new file)
```rust
pub struct WorkingMemory {
    // Classic 7±2 capacity limit
    capacity: usize,
    
    // Current contents
    items: VecDeque<WorkingMemoryItem>,
    
    // Subsystems
    phonological_loop: PhonologicalLoop,
    visuospatial_sketchpad: VisuospatialSketchpad,
    episodic_buffer: EpisodicBuffer,
    central_executive: CentralExecutive,
    
    // Maintenance processes
    rehearsal_process: RehearsalProcess,
    decay_rate: f32,
}

pub struct WorkingMemoryItem {
    id: ItemId,
    content: ItemContent,
    activation_level: f32,
    entry_time: Instant,
    last_rehearsal: Instant,
    modality: Modality,
    chunk_size: usize,  // How many "slots" it occupies
}

pub enum ItemContent {
    Verbal(String),
    Visual(VisualRepresentation),
    Spatial(SpatialRepresentation),
    Semantic(SemanticContent),
    Episodic(EpisodeReference),
    Multimodal(Vec<ItemContent>),
}

impl WorkingMemory {
    pub fn add_item(&mut self, item: ItemContent) -> Result<ItemId, WorkingMemoryError> {
        let chunk_size = self.calculate_chunk_size(&item);
        
        // Check capacity
        let current_load = self.items.iter().map(|i| i.chunk_size).sum::<usize>();
        if current_load + chunk_size > self.capacity {
            // Try to make room
            self.evict_lowest_activation(chunk_size)?;
        }
        
        let wm_item = WorkingMemoryItem {
            id: ItemId::new(),
            content: item,
            activation_level: 1.0,
            entry_time: Instant::now(),
            last_rehearsal: Instant::now(),
            modality: self.determine_modality(&item),
            chunk_size,
        };
        
        let id = wm_item.id;
        self.items.push_back(wm_item);
        
        Ok(id)
    }
    
    pub fn update(&mut self, dt: Duration) {
        // Decay all items
        for item in &mut self.items {
            let time_since_rehearsal = item.last_rehearsal.elapsed();
            let decay = (-time_since_rehearsal.as_secs_f32() * self.decay_rate).exp();
            item.activation_level *= decay;
        }
        
        // Remove items below threshold
        self.items.retain(|item| item.activation_level > ACTIVATION_THRESHOLD);
        
        // Automatic rehearsal for high-priority items
        self.rehearsal_process.rehearse(&mut self.items);
    }
    
    pub fn retrieve(&self, cue: &RetrievalCue) -> Option<&WorkingMemoryItem> {
        self.items.iter()
            .filter(|item| self.matches_cue(item, cue))
            .max_by_key(|item| OrderedFloat(item.activation_level))
    }
}
```

### Task 18.2: Working Memory Subsystems
**File**: `src/working_memory/subsystems.rs` (new file)
```rust
pub struct PhonologicalLoop {
    store_capacity: Duration,  // ~2 seconds
    articulatory_rehearsal: ArticulatoryProcess,
    word_length_effect: bool,
    phonological_similarity_effect: bool,
}

impl PhonologicalLoop {
    pub fn store_verbal(&mut self, content: &str) -> LoopStorage {
        let words = self.segment_into_words(content);
        
        // Word length effect: longer words take more capacity
        let total_duration = words.iter()
            .map(|w| self.estimate_articulation_time(w))
            .sum::<Duration>();
            
        if total_duration <= self.store_capacity {
            LoopStorage::Complete(content.to_string())
        } else {
            // Store only what fits
            let stored = self.fit_to_capacity(words);
            LoopStorage::Partial(stored)
        }
    }
    
    pub fn rehearse(&mut self, content: &str) {
        // Subvocal rehearsal refreshes the trace
        self.articulatory_rehearsal.rehearse(content);
    }
}

pub struct VisuospatialSketchpad {
    visual_cache: VisualCache,
    spatial_system: SpatialSystem,
    capacity: usize,  // Number of objects
}

impl VisuospatialSketchpad {
    pub fn store_visual(&mut self, image: VisualRepresentation) -> SketchpadStorage {
        let features = self.extract_visual_features(&image);
        
        if features.len() <= self.capacity {
            self.visual_cache.store(features);
            SketchpadStorage::Complete
        } else {
            // Store most salient features
            let salient = self.select_salient_features(features, self.capacity);
            self.visual_cache.store(salient);
            SketchpadStorage::Degraded
        }
    }
    
    pub fn mental_rotation(&mut self, object: &SpatialObject, angle: f32) {
        self.spatial_system.rotate(object, angle);
    }
}

pub struct CentralExecutive {
    attention_controller: AttentionController,
    task_switcher: TaskSwitcher,
    inhibition_system: InhibitionSystem,
}

impl CentralExecutive {
    pub fn focus_attention(&mut self, target: AttentionTarget) {
        self.attention_controller.focus_on(target);
        self.inhibition_system.suppress_irrelevant();
    }
    
    pub fn switch_task(&mut self, from: TaskId, to: TaskId) -> SwitchCost {
        self.task_switcher.switch(from, to)
    }
    
    pub fn update_goals(&mut self, new_goal: Goal) {
        self.attention_controller.update_goal_relevance(new_goal);
    }
}
```

### Task 18.3: Chunking and Capacity
**File**: `src/working_memory/chunking.rs` (new file)
```rust
pub struct ChunkingSystem {
    chunk_patterns: HashMap<PatternId, ChunkPattern>,
    learned_chunks: HashMap<ChunkId, LearnedChunk>,
    chunking_strategies: Vec<ChunkingStrategy>,
}

pub struct ChunkPattern {
    elements: Vec<Element>,
    frequency: u32,
    meaningfulness: f32,
    retrieval_strength: f32,
}

pub struct LearnedChunk {
    id: ChunkId,
    components: Vec<ItemId>,
    unified_representation: UnifiedRep,
    formation_time: Instant,
    usage_count: u32,
}

impl ChunkingSystem {
    pub fn chunk_items(&mut self, items: &[WorkingMemoryItem]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        
        // Try different chunking strategies
        for strategy in &self.chunking_strategies {
            let strategy_chunks = match strategy {
                ChunkingStrategy::Semantic => self.semantic_chunking(items),
                ChunkingStrategy::Temporal => self.temporal_chunking(items),
                ChunkingStrategy::Spatial => self.spatial_chunking(items),
                ChunkingStrategy::Phonological => self.phonological_chunking(items),
                ChunkingStrategy::Learned => self.apply_learned_chunks(items),
            };
            
            chunks.extend(strategy_chunks);
        }
        
        // Select best chunking
        self.select_optimal_chunking(chunks)
    }
    
    pub fn learn_new_chunk(&mut self, items: &[ItemId], context: &Context) {
        if self.is_meaningful_group(items, context) {
            let chunk = LearnedChunk {
                id: ChunkId::new(),
                components: items.to_vec(),
                unified_representation: self.create_unified_rep(items),
                formation_time: Instant::now(),
                usage_count: 1,
            };
            
            self.learned_chunks.insert(chunk.id, chunk);
        }
    }
    
    pub fn calculate_effective_capacity(&self, chunks: &[Chunk]) -> usize {
        // Miller's magic number applies to chunks, not items
        let chunk_count = chunks.len();
        let avg_chunk_size = chunks.iter()
            .map(|c| c.size())
            .sum::<usize>() / chunk_count.max(1);
            
        // Effective capacity increases with chunking
        BASE_CAPACITY + (avg_chunk_size - 1) * CHUNKING_BONUS
    }
}
```

## Week 19: Memory Integration

### Task 19.1: Episodic-Semantic Interface
**File**: `src/integration/episodic_semantic_bridge.rs` (new file)
```rust
pub struct EpisodicSemanticBridge {
    extraction_rules: Vec<ExtractionRule>,
    generalization_engine: GeneralizationEngine,
    schema_builder: SchemaBuilder,
}

impl EpisodicSemanticBridge {
    pub fn extract_semantic_from_episodes(&mut self,
        episodes: &[Episode]
    ) -> SemanticKnowledge {
        // Find repeated patterns across episodes
        let patterns = self.find_recurring_patterns(episodes);
        
        // Abstract away episodic details
        let abstractions = patterns.into_iter()
            .map(|p| self.generalization_engine.abstract_pattern(p))
            .collect();
            
        // Build schemas from abstractions
        let schemas = self.schema_builder.build_schemas(abstractions);
        
        SemanticKnowledge {
            concepts: self.extract_concepts(episodes),
            relationships: self.extract_relationships(episodes),
            schemas,
            rules: self.infer_rules(episodes),
        }
    }
    
    pub fn instantiate_semantic_in_episodic(&self,
        semantic: &SemanticKnowledge,
        context: &Context
    ) -> Episode {
        // Use semantic knowledge to fill in episodic details
        let template = self.select_relevant_schema(semantic, context);
        
        Episode {
            event: self.instantiate_event(template, context),
            spatial_context: self.default_spatial_context(template),
            temporal_context: self.current_temporal_context(),
            social_context: self.infer_social_context(template, context),
            cognitive_context: context.cognitive_context.clone(),
            emotional_context: self.predict_emotional_context(template),
            // ... other fields
        }
    }
}
```

### Task 19.2: Working Memory Gateway
**File**: `src/integration/working_memory_gateway.rs` (new file)
```rust
pub struct WorkingMemoryGateway {
    encoding_threshold: f32,
    consolidation_delay: Duration,
    pending_memories: VecDeque<PendingMemory>,
}

impl WorkingMemoryGateway {
    pub fn process_working_memory(&mut self,
        wm: &WorkingMemory,
        ltm: &mut LongTermMemory
    ) {
        // Check for items ready for encoding
        for item in wm.items.iter() {
            if self.should_encode_to_ltm(item) {
                self.pending_memories.push_back(PendingMemory {
                    content: item.content.clone(),
                    context: self.capture_encoding_context(wm),
                    scheduled_time: Instant::now() + self.consolidation_delay,
                });
            }
        }
        
        // Process pending memories
        let now = Instant::now();
        while let Some(pending) = self.pending_memories.front() {
            if pending.scheduled_time <= now {
                let pending = self.pending_memories.pop_front().unwrap();
                self.encode_to_ltm(pending, ltm);
            } else {
                break;
            }
        }
    }
    
    fn should_encode_to_ltm(&self, item: &WorkingMemoryItem) -> bool {
        // Encode if: rehearsed enough, emotionally significant, 
        // goal-relevant, or surprising
        item.activation_level > self.encoding_threshold ||
        item.rehearsal_count > REHEARSAL_THRESHOLD ||
        item.emotional_significance > EMOTION_THRESHOLD ||
        item.goal_relevance > GOAL_THRESHOLD ||
        item.surprise_value > SURPRISE_THRESHOLD
    }
}
```

## Week 20: Testing and Optimization

### Task 20.1: Episodic Memory Tests
**File**: `tests/episodic_memory_tests.rs`
```rust
#[test]
fn test_episodic_encoding() {
    let mut em = EpisodicMemory::new();
    
    let event = Event {
        description: "Meeting old friend at coffee shop".to_string(),
        participants: vec!["self", "John"],
        actions: vec!["greeting", "conversation", "reminiscing"],
    };
    
    let context = create_rich_context();
    let episode_id = em.encode_episode(event, context, EncodingParameters::default());
    
    // Test retrieval with partial cues
    let cues = RetrievalCues {
        location: Some("coffee shop"),
        people: Some(vec!["John"]),
        ..Default::default()
    };
    
    let retrieved = em.retrieve_episode(cues);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, episode_id);
}

#[test]
fn test_working_memory_capacity() {
    let mut wm = WorkingMemory::new(7);
    
    // Test capacity limits
    for i in 0..7 {
        assert!(wm.add_item(ItemContent::Verbal(i.to_string())).is_ok());
    }
    
    // 8th item should trigger eviction
    assert!(wm.add_item(ItemContent::Verbal("8".to_string())).is_ok());
    assert_eq!(wm.items.len(), 7);
}

#[test]
fn test_chunking() {
    let mut chunking = ChunkingSystem::new();
    
    // Phone number should be chunked
    let items = vec!["5", "5", "5", "1", "2", "3", "4"];
    let chunks = chunking.chunk_items(&items);
    
    // Should produce 3 chunks: 555-123-4 or similar
    assert!(chunks.len() < items.len());
}
```

### Task 20.2: Performance Benchmarks
**File**: `benches/episodic_benchmarks.rs`
```rust
fn benchmark_episodic_retrieval(c: &mut Criterion) {
    let em = create_large_episodic_memory(10000);
    
    c.bench_function("retrieve episode with spatial cue", |b| {
        b.iter(|| {
            em.retrieve_episode(RetrievalCues {
                location: Some("home"),
                ..Default::default()
            })
        })
    });
}

fn benchmark_working_memory_update(c: &mut Criterion) {
    let mut wm = WorkingMemory::new(7);
    fill_working_memory(&mut wm);
    
    c.bench_function("working memory decay update", |b| {
        b.iter(|| wm.update(Duration::from_millis(100)))
    });
}
```

### Task 20.3: Integration Tests
**File**: `tests/memory_integration_tests.rs`
```rust
#[test]
fn test_working_to_episodic_transfer() {
    let mut system = MemorySystem::new();
    
    // Add items to working memory
    system.working_memory.add_item(ItemContent::Verbal("Important meeting tomorrow"));
    
    // Rehearse to strengthen
    for _ in 0..5 {
        system.working_memory.rehearse_all();
    }
    
    // Process gateway
    system.process_memory_gateway();
    
    // Should be in episodic memory
    let episodes = system.episodic_memory.search("meeting");
    assert!(!episodes.is_empty());
}
```

## Deliverables
1. **Rich episodic memory** with full context encoding
2. **Autobiographical organization** with hierarchical structure
3. **Working memory** with 7±2 capacity and subsystems
4. **Chunking system** for capacity optimization
5. **Memory integration** between WM and LTM
6. **Comprehensive test suite** for all components

## Success Criteria
- [ ] Episodic memories contain all context dimensions
- [ ] Autobiographical retrieval follows psychological findings
- [ ] Working memory shows capacity limitations
- [ ] Chunking increases effective capacity
- [ ] Gateway successfully transfers important items to LTM
- [ ] Performance meets targets (< 50ms for WM, < 200ms for EM)

## Dependencies
- Spatial representation library
- Rich context modeling
- Memory benchmark data

## Risks & Mitigations
1. **Complex episodic structure overhead**
   - Mitigation: Lazy loading, selective detail encoding
2. **Working memory real-time constraints**
   - Mitigation: Efficient data structures, parallel updates
3. **Integration complexity**
   - Mitigation: Clear interfaces, modular design