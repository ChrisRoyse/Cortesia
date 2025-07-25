# Phase 5: Episodic & Working Memory Enhancement

## Overview
**Duration**: 4 weeks  
**Goal**: Extend existing working memory and add episodic memory capabilities with AI enhancement  
**Priority**: MEDIUM  
**Dependencies**: Phases 1-4 completion  
**Target Performance**: <10ms for episode encoding/retrieval on Intel i9

## CRITICAL FINDINGS ADDRESSED
**Existing Infrastructure Analysis**: This plan now properly extends existing systems:
- **WorkingMemorySystem**: Already implemented with sophisticated buffer management, decay, attention integration
- **AttentionManager**: Full attention orchestration with cognitive pattern coordination  
- **ModelLoader**: Comprehensive ONNX runtime with 8+ models available
- **MCP Integration**: 28+ existing MCP tools with established patterns
- **Storage Infrastructure**: Advanced knowledge engine, SDR storage, batch processing
- **Cognitive Systems**: CognitiveOrchestrator, inhibitory logic, pattern detection

## Enhanced Integration Strategy
**Extend NOT Replace**: Build upon existing working memory and attention systems
- **Working Memory Enhancement**: Extend existing `WorkingMemorySystem` with episodic bridges
- **Model Reuse**: Leverage existing models (DistilBertNER, MiniLM, T5Small, TinyBertNER)
- **MCP Extension**: Add episodic handlers using existing MCP patterns from `handlers/mod.rs`
- **Storage Extension**: Use existing `KnowledgeEngine`, `SDRStorage`, batch processing infrastructure
- **Attention Integration**: Extend existing `AttentionManager` for episodic attention weighting
- **Cognitive Coordination**: Integrate with existing `CognitiveOrchestrator` and pattern systems

## Multi-Database Architecture - Phase 5
**New in Phase 5**: Extend event sourcing for episodic memory and optimize working memory storage
- **Event Sourcing Extensions**: Complex event hierarchies for episodes
- **Working Memory Cache**: High-speed temporal cache for active memories  
- **Episode Clustering**: Intelligent clustering of related episodes
- **Cross-Database Linking**: Link episodes to semantic and temporal databases

## AI Model Integration 
**Available Models in src/models**:
- **DistilBERT-NER** (66M params) - For episode understanding and context encoding
- **all-MiniLM-L6-v2** (22M params) - For similarity and retrieval
- **T5-Small** (60M params) - For narrative generation
- **Model Infrastructure**: Use existing `ModelLoader`, ONNX runtime, and batch processing

## Week 17: Episodic Memory Extension

### Task 17.1: Working Memory Episodic Bridge
**File**: `src/cognitive/episodic_bridge.rs` (new file)  
**Purpose**: Bridge existing WorkingMemorySystem to episodic operations
```rust
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, MemoryItem, BufferType, MemoryQuery};
use crate::cognitive::attention_manager::{AttentionManager, AttentionType};
use crate::models::model_loader::ModelLoader;
use crate::models::onnx_runtime::OnnxModel;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;
use std::time::{Duration, Instant};

/// Episodic bridge for existing working memory system
/// Extends functionality without breaking existing API
pub struct WorkingMemoryEpisodicBridge {
    // Use existing systems - no duplication
    working_memory: Arc<WorkingMemorySystem>,
    attention_manager: Arc<AttentionManager>,
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    model_loader: Arc<ModelLoader>,
    
    // Episodic-specific components
    episode_storage: Arc<RwLock<EpisodeStorage>>,
    episode_index: Arc<RwLock<EpisodeIndex>>,
    encoding_cache: Arc<RwLock<EncodingCache>>,
    
    // AI model IDs (loaded through existing ModelLoader)
    context_encoder_id: String,     // DistilBertNER for context understanding
    similarity_model_id: String,    // MiniLM for episode similarity
    narrative_generator_id: String, // T5Small for narrative generation
    chunking_model_id: String,      // TinyBertNER for intelligent chunking
}

/// Episode representation that integrates with working memory
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: EpisodeId,
    pub event: Event,
    pub context: EpisodeContext,
    pub working_memory_snapshot: WorkingMemorySnapshot,
    pub attention_pattern: AttentionPattern,
    pub narrative: Option<String>,
    pub encoding_strength: f32,
    pub importance_score: f32,
    pub episode_embedding: Vec<f32>,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]  
pub struct Event {
    pub description: String,
    pub participants: Vec<String>,
    pub actions: Vec<String>,
    pub confidence: f32,
    pub extracted_entities: Vec<EntityKey>,
}

#[derive(Debug, Clone)]
pub struct EpisodeContext {
    pub spatial: SpatialContext,
    pub temporal: TemporalContext, 
    pub social: SocialContext,
    pub cognitive: CognitiveContext,
}

#[derive(Debug, Clone)]
pub struct WorkingMemorySnapshot {
    pub active_items: Vec<MemoryItem>,
    pub buffer_states: Vec<crate::cognitive::working_memory::BufferState>,
    pub central_executive_state: f32,
    pub snapshot_time: Instant,
}

#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub focused_entities: Vec<EntityKey>,
    pub attention_weights: std::collections::HashMap<EntityKey, f32>,
    pub attention_type: AttentionType,
    pub focus_strength: f32,
}

impl WorkingMemoryEpisodicBridge {
    pub async fn new(
        working_memory: Arc<WorkingMemorySystem>,
        attention_manager: Arc<AttentionManager>,
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        model_loader: Arc<ModelLoader>,
    ) -> Result<Self> {
        // Load existing models for episodic enhancement
        let context_encoder = model_loader.load_model(crate::models::ModelType::DistilBertNER).await?;
        let similarity_model = model_loader.load_model(crate::models::ModelType::MiniLM).await?;
        let narrative_generator = model_loader.load_model(crate::models::ModelType::T5Small).await?;
        let chunking_model = model_loader.load_model(crate::models::ModelType::TinyBertNER).await?;
        
        Ok(Self {
            working_memory,
            attention_manager,
            knowledge_engine,
            model_loader,
            episode_storage: Arc::new(RwLock::new(EpisodeStorage::new())),
            episode_index: Arc::new(RwLock::new(EpisodeIndex::new())),
            encoding_cache: Arc::new(RwLock::new(EncodingCache::new(10_000))),
            // Store model references for reuse
            context_encoder_id: "context_encoder".to_string(),
            similarity_model_id: "similarity_model".to_string(), 
            narrative_generator_id: "narrative_generator".to_string(),
            chunking_model_id: "chunking_model".to_string(),
        })
    }
    
    /// Encode episode using existing working memory state
    pub async fn encode_episode_from_working_memory(
        &self,
        event_description: String,
        use_current_context: bool,
    ) -> Result<EpisodeId> {
        // 1. Capture current working memory state using existing API
        let wm_state = self.working_memory.get_current_state().await?;
        let wm_items = self.working_memory.get_all_items().await?;
        
        // 2. Get current attention state using existing AttentionManager
        let attention_state = self.attention_manager.get_attention_state().await?;
        
        // 3. Create working memory snapshot
        let wm_snapshot = WorkingMemorySnapshot {
            active_items: wm_items,
            buffer_states: wm_state.buffer_states,
            central_executive_state: wm_state.efficiency_score,
            snapshot_time: Instant::now(),
        };
        
        // 4. Extract event details using existing model infrastructure
        let event = self.extract_event_details(&event_description).await?;
        
        // 5. Build episode context from working memory and attention
        let context = if use_current_context {
            self.build_context_from_current_state(&wm_snapshot, &attention_state).await?
        } else {
            EpisodeContext::default()
        };
        
        // 6. Create attention pattern from current state
        let attention_pattern = AttentionPattern {
            focused_entities: attention_state.current_targets,
            attention_weights: std::collections::HashMap::new(), // Simplified for now
            attention_type: attention_state.attention_type,
            focus_strength: attention_state.focus_strength,
        };
        
        // 7. Generate episode embedding using existing model
        let episode_text = self.format_episode_for_encoding(&event, &context, &wm_snapshot);
        let episode_embedding = self.generate_episode_embedding(&episode_text).await?;
        
        // 8. Calculate importance using working memory context
        let importance_score = self.calculate_episode_importance(&event, &wm_snapshot, &attention_pattern).await?;
        
        // 9. Generate narrative if important enough
        let narrative = if importance_score > 0.7 {
            Some(self.generate_episode_narrative(&event, &context).await?)
        } else {
            None
        };
        
        // 10. Create and store episode
        let episode_id = EpisodeId::new();
        let episode = Episode {
            id: episode_id,
            event,
            context,
            working_memory_snapshot: wm_snapshot,
            attention_pattern,
            narrative,
            encoding_strength: importance_score,
            importance_score,
            episode_embedding,
            timestamp: Instant::now(),
        };
        
        // 11. Store using existing storage patterns
        self.store_episode(episode).await?;
        
        Ok(episode_id)
    }
    
    /// Retrieve episodes that match working memory query
    pub async fn retrieve_episodes_for_working_memory(
        &self,
        query: &MemoryQuery,
        max_results: usize,
    ) -> Result<Vec<Episode>> {
        // 1. Generate query embedding using existing model
        let query_embedding = self.generate_query_embedding(&query.query_text).await?;
        
        // 2. Search episode index for similar episodes
        let episode_candidates = self.episode_index.read().await
            .find_similar_episodes(&query_embedding, max_results * 2)
            .await?;
        
        // 3. Score episodes based on working memory relevance
        let mut scored_episodes = Vec::new();
        for episode_id in episode_candidates {
            if let Some(episode) = self.episode_storage.read().await.get(&episode_id) {
                let score = self.calculate_working_memory_relevance(episode, query).await?;
                if score > query.importance_threshold {
                    scored_episodes.push((episode.clone(), score));
                }
            }
        }
        
        // 4. Sort by relevance and return top results
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored_episodes.into_iter()
            .take(max_results)
            .map(|(episode, _)| episode)
            .collect())
    }
    
    /// Update working memory with episodic context
    pub async fn enhance_working_memory_with_episodes(
        &self,
        query: &str,
        buffer_type: BufferType,
        importance_boost: f32,
    ) -> Result<Vec<EpisodeId>> {
        // 1. Find relevant episodes
        let memory_query = MemoryQuery {
            query_text: query.to_string(),
            search_buffers: vec![buffer_type.clone()],
            apply_attention: true,
            importance_threshold: 0.3,
            recency_weight: 0.7,
        };
        
        let relevant_episodes = self.retrieve_episodes_for_working_memory(&memory_query, 5).await?;
        
        // 2. Extract important concepts from episodes
        let mut enhanced_items = Vec::new();
        for episode in &relevant_episodes {
            // Extract key concepts from episode narrative or event
            let concepts = self.extract_episode_concepts(episode).await?;
            
            for concept in concepts {
                let memory_content = MemoryContent::Concept(concept);
                
                // Store in working memory with episodic boost using existing API
                let _ = self.working_memory.store_in_working_memory_with_attention(
                    memory_content,
                    episode.importance_score + importance_boost,
                    buffer_type.clone(),
                    importance_boost,
                ).await?;
            }
        }
        
        Ok(relevant_episodes.into_iter().map(|e| e.id).collect())
    }
    
    // Helper methods using existing infrastructure
    async fn extract_event_details(&self, description: &str) -> Result<Event> {
        // Use existing DistilBertNER model for entity extraction
        let model = self.model_loader.load_model(crate::models::ModelType::DistilBertNER).await?;
        
        // Extract entities using existing model infrastructure
        let entities = model.extract_entities(description).await
            .unwrap_or_else(|_| vec![]);
        
        // Simple rule-based extraction for actions and participants
        let participants = self.extract_participants(description);
        let actions = self.extract_actions(description);
        
        Ok(Event {
            description: description.to_string(),
            participants,
            actions,
            confidence: 0.8, // Default confidence
            extracted_entities: entities,
        })
    }
    
    async fn build_context_from_current_state(
        &self,
        wm_snapshot: &WorkingMemorySnapshot,
        attention_state: &crate::cognitive::attention_manager::AttentionStateInfo,
    ) -> Result<EpisodeContext> {
        // Build rich context from current working memory and attention state
        let spatial = SpatialContext::from_working_memory(&wm_snapshot.active_items);
        let temporal = TemporalContext::now();
        let social = SocialContext::from_working_memory(&wm_snapshot.active_items);
        let cognitive = CognitiveContext {
            cognitive_load: attention_state.cognitive_load,
            attention_capacity: attention_state.attention_capacity,
            focus_strength: attention_state.focus_strength,
        };
        
        Ok(EpisodeContext {
            spatial,
            temporal,
            social,
            cognitive,
        })
    }
    
    async fn generate_episode_embedding(&self, episode_text: &str) -> Result<Vec<f32>> {
        // Use existing MiniLM model for embedding generation
        let model = self.model_loader.load_model(crate::models::ModelType::MiniLM).await?;
        model.encode_text(episode_text).await
            .map_err(|e| crate::error::Error::ModelError(e.to_string()))
    }
    
    async fn calculate_episode_importance(
        &self,
        event: &Event,
        wm_snapshot: &WorkingMemorySnapshot,
        attention_pattern: &AttentionPattern,
    ) -> Result<f32> {
        // Calculate importance based on:
        // 1. Event confidence
        // 2. Working memory load (higher load = more important context)
        // 3. Attention focus strength
        // 4. Number of active entities
        
        let event_importance = event.confidence;
        let wm_importance = (wm_snapshot.active_items.len() as f32 / 15.0).min(1.0);
        let attention_importance = attention_pattern.focus_strength;
        let entity_importance = (event.extracted_entities.len() as f32 / 10.0).min(1.0);
        
        let combined_importance = (
            event_importance * 0.3 +
            wm_importance * 0.2 +
            attention_importance * 0.3 +
            entity_importance * 0.2
        ).min(1.0);
        
        Ok(combined_importance)
    }
    
    async fn store_episode(&self, episode: Episode) -> Result<()> {
        // Store in episode storage
        self.episode_storage.write().await.insert(episode.id, episode.clone());
        
        // Index for retrieval
        self.episode_index.write().await.index_episode(&episode).await?;
        
        // Store relevant triples in knowledge engine for integration
        self.store_episode_triples(&episode).await?;
        
        Ok(())
    }
}

// Supporting types and implementations
pub struct EpisodeStorage {
    episodes: DashMap<EpisodeId, Episode>,
}

impl EpisodeStorage {
    pub fn new() -> Self {
        Self {
            episodes: DashMap::new(),
        }
    }
    
    pub fn insert(&mut self, id: EpisodeId, episode: Episode) {
        self.episodes.insert(id, episode);
    }
    
    pub fn get(&self, id: &EpisodeId) -> Option<Episode> {
        self.episodes.get(id).map(|e| e.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpisodeId(uuid::Uuid);

impl EpisodeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

// Context types that use working memory data
#[derive(Debug, Clone, Default)]
pub struct SpatialContext {
    pub location: Option<String>,
    pub landmarks: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub timestamp: Instant,
    pub duration: Option<Duration>,
}

impl TemporalContext {
    pub fn now() -> Self {
        Self {
            timestamp: Instant::now(),
            duration: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SocialContext {
    pub people_present: Vec<String>,
    pub social_roles: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CognitiveContext {
    pub cognitive_load: f32,
    pub attention_capacity: f32, 
    pub focus_strength: f32,
}

impl Default for CognitiveContext {
    fn default() -> Self {
        Self {
            cognitive_load: 0.0,
            attention_capacity: 1.0,
            focus_strength: 0.0,
        }
    }
}
```

pub struct EpisodicMemory {
    episodes: DashMap<EpisodeId, Episode>,
    episode_index: EpisodeIndex,
    autobiographical_organizer: AutobiographicalOrganizer,
    
    // Integration with existing systems
    model_loader: Arc<ModelLoader>,
    working_memory: Arc<WorkingMemorySystem>,
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    
    // AI models loaded through existing infrastructure
    context_encoder_id: String,     // DistilBERT-NER model ID
    similarity_model_id: String,    // all-MiniLM-L6-v2 model ID  
    narrative_generator_id: String, // T5-Small model ID
    
    // Performance optimization
    batch_processor: EpisodeBatchProcessor,
    encoding_cache: Arc<RwLock<EncodingCache>>,
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
    rehearsal_count: AtomicU32,
    
    // AI-enhanced fields
    episode_embedding: Vec<f32>,      // Full episode encoding
    context_embedding: Vec<f32>,      // Context-specific encoding
    emotion_vector: Vec<f32>,         // Emotional signature
    scene_embedding: Option<Vec<f32>>, // Spatial scene encoding
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
    pub async fn new(
        model_loader: Arc<ModelLoader>,
        working_memory: Arc<WorkingMemorySystem>,
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    ) -> Result<Self> {
        // Load models using existing infrastructure
        let context_encoder_id = model_loader.load_model(
            ModelType::DistilBertNER,
            ModelConfig::default(),
        ).await?;
        
        let similarity_model_id = model_loader.load_model(
            ModelType::MiniLM,
            ModelConfig::default(),
        ).await?;
        
        let narrative_generator_id = model_loader.load_model(
            ModelType::T5Small,
            ModelConfig::default(),
        ).await?;
        
        Ok(Self {
            episodes: DashMap::with_capacity(100_000),
            episode_index: EpisodeIndex::new()?,
            autobiographical_organizer: AutobiographicalOrganizer::new(),
            model_loader,
            working_memory,
            knowledge_engine,
            context_encoder_id,
            similarity_model_id,
            narrative_generator_id,
            batch_processor: EpisodeBatchProcessor::new(),
            encoding_cache: Arc::new(RwLock::new(EncodingCache::new(10_000))),
        })
    }

    pub async fn encode_episode(
        &self, 
        event: Event,
        context: FullContext,
        encoding_params: EncodingParameters
    ) -> Result<EpisodeId> {
        let id = EpisodeId::new();
        
        // Check encoding cache first for performance
        let cache_key = self.calculate_cache_key(&event, &context);
        if let Some(cached_episode) = self.encoding_cache.read().await.get(&cache_key) {
            return Ok(cached_episode.id);
        }
        
        // Use existing model infrastructure for batch processing
        let batch_inputs = vec![
            self.format_episode_text(&event, &context),
            context.to_string(),
            context.emotional_context.to_string(),
        ];
        
        // Parallel AI processing using existing model loader
        let (episode_emb, context_emb, emotion_vec) = tokio::join!(
            self.model_loader.batch_encode(&self.context_encoder_id, &batch_inputs[0..1]),
            self.model_loader.batch_encode(&self.similarity_model_id, &batch_inputs[1..2]),
            self.extract_emotion_features(&context.emotional_context)
        );
        
        let (episode_emb, context_emb, emotion_vec) = (
            episode_emb.map_err(|e| crate::error::Error::ModelError(e.to_string()))?,
            context_emb.map_err(|e| crate::error::Error::ModelError(e.to_string()))?,
            emotion_vec?
        );
        
        // Extract contexts using simpler, faster methods
        let (spatial_ctx, temporal_ctx, social_ctx, cognitive_ctx, emotional_ctx) = tokio::join!(
            self.extract_spatial_context(&context),
            self.extract_temporal_context(&context),
            self.extract_social_context(&context),
            self.extract_cognitive_context(&context),
            self.extract_emotional_context(&context)
        );
        
        let episode = Episode {
            id,
            event: event.clone(),
            spatial_context: spatial_ctx,
            temporal_context: temporal_ctx,
            social_context: social_ctx,
            cognitive_context: cognitive_ctx,
            emotional_context: emotional_ctx,
            sensory_details: self.capture_sensory_snapshot(&context),
            narrative: None,
            encoding_strength: encoding_params.calculate_strength(&event, &context),
            vividness: self.calculate_vividness_ai(&context, &emotion_vec),
            personal_significance: self.evaluate_significance_ai(&event, &context, &episode_emb),
            rehearsal_count: AtomicU32::new(0),
            // AI fields
            episode_embedding: episode_emb,
            context_embedding: context_emb,
            emotion_vector: emotion_vec,
            scene_embedding: scene_emb,
        };
        
        // Index episode with embeddings for fast retrieval
        self.episode_index.index_with_embeddings(&episode).await;
        
        // Organize autobiographically
        self.autobiographical_organizer.integrate_episode(&episode);
        
        self.episodes.insert(id, episode);
        id
    }
    
    pub async fn retrieve_episode(&self, cues: RetrievalCues) -> Option<Episode> {
        // Encode retrieval cues
        let cue_embedding = self.encode_retrieval_cues(&cues).await;
        
        // Fast similarity search using SIMD
        let candidates = self.find_similar_episodes(&cue_embedding, 20).await;
        
        // Neural pattern completion for partial cues
        let completed = self.neural_pattern_complete(candidates, &cues, &cue_embedding).await;
        
        // Score and rank with AI
        let scored = self.score_episodes_neural(completed, &cues).await;
        
        // Return highest scoring episode
        scored.into_iter()
            .max_by_key(|(_, score)| OrderedFloat(*score))
            .and_then(|(id, _)| self.episodes.get(&id).map(|e| e.clone()))
    }
    
    async fn encode_full_episode(&self, event: &Event, context: &FullContext) -> Vec<f32> {
        // Combine event and context into single representation
        let combined_text = format!(
            "{} at {} with {}", 
            event.description,
            context.spatial_info.location_name,
            context.social_info.people_present.join(", ")
        );
        
        self.context_encoder.encode(&combined_text).await.unwrap_or_else(|_| {
            vec![0.0; 768]  // DistilBERT dimension
        })
    }
    
    async fn find_similar_episodes(&self, embedding: &[f32], k: usize) -> Vec<EpisodeId> {
        let mut similarities = Vec::new();
        
        // Parallel similarity computation
        for entry in self.episodes.iter() {
            let sim = self.similarity_engine.compute_similarity(
                embedding,
                &entry.episode_embedding
            );
            similarities.push((entry.key().clone(), sim));
        }
        
        // Sort by similarity
        similarities.sort_by_key(|(_, sim)| OrderedFloat(-sim));
        similarities.truncate(k);
        
        similarities.into_iter().map(|(id, _)| id).collect()
    }
    
    fn calculate_vividness_ai(&self, context: &FullContext, emotion_vec: &[f32]) -> f32 {
        // Use emotion intensity and sensory richness
        let emotion_intensity = emotion_vec.iter().map(|x| x.abs()).sum::<f32>() / emotion_vec.len() as f32;
        let sensory_richness = context.sensory_info.richness_score();
        
        (emotion_intensity * 0.6 + sensory_richness * 0.4).min(1.0)
    }
}

// Lightweight context encoder (40M params)
pub struct ContextEncoder {
    // DistilBERT variant optimized for episode encoding
}

// Scene embedder CNN (10M params)
pub struct SceneEmbedder {
    // Small CNN for spatial scene understanding
}

// Emotion tagger (15M params)
pub struct EmotionTagger {
    // Transformer for emotional context tagging
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

## Week 18: Working Memory Enhancement

### Task 18.1: Enhance Existing WorkingMemorySystem with AI
**File**: `src/cognitive/working_memory_episodic_bridge.rs` (new file)  
**Integration**: Extend existing `WorkingMemorySystem` rather than replace
```rust
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, MemoryItem, BufferType};
use crate::cognitive::attention_manager::AttentionManager;
use crate::models::{ModelLoader, ModelType};
use crate::core::types::EntityKey;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Episodic bridge for existing working memory system
pub struct WorkingMemoryEpisodicBridge {
    // Use existing working memory system
    working_memory: Arc<WorkingMemorySystem>,
    attention_manager: Arc<AttentionManager>,
    
    // AI enhancement models (loaded through existing infrastructure)
    model_loader: Arc<ModelLoader>,
    chunking_model_id: String,      // 5M params for intelligent chunking
    priority_scorer_id: String,     // 3M params for importance scoring  
    memory_encoder_id: String,      // 8M params for content encoding
    
    // Performance optimization
    encoding_cache: Arc<RwLock<WorkingMemoryCache>>,
    batch_processor: WorkingMemoryBatchProcessor,
}

// Use existing MemoryItem from WorkingMemorySystem, extend with AI features
#[derive(Debug, Clone)]
pub struct EnhancedMemoryItem {
    // Base item from existing system
    pub base_item: MemoryItem,
    
    // AI-enhanced fields for episodic integration
    pub content_embedding: Option<Vec<f32>>,     // Neural representation (cached)
    pub semantic_links: Vec<EntityKey>,          // Related items in WM  
    pub attention_weight: f32,                   // Current attention allocation
    pub chunk_metadata: ChunkMetadata,           // Chunking information
    pub episode_links: Vec<EpisodeId>,           // Connected episodes
}

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_size: usize,        // How many "slots" it occupies
    pub chunk_strategy: ChunkingStrategy,
    pub chunk_confidence: f32,
}

#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    Semantic,    // Group by meaning
    Temporal,    // Group by time sequence  
    Spatial,     // Group by spatial relationship
    Phonological, // Group by sound similarity
    Neural,      // AI-determined chunking
}

impl WorkingMemoryEpisodicBridge {
    pub async fn new(
        working_memory: Arc<WorkingMemorySystem>,
        attention_manager: Arc<AttentionManager>,
        model_loader: Arc<ModelLoader>,
    ) -> Result<Self> {
        // Load AI enhancement models using existing infrastructure
        let chunking_model_id = model_loader.load_model(
            ModelType::TinyBertNER,  // Use lightweight model for chunking
            ModelConfig::default(),
        ).await?;
        
        let priority_scorer_id = model_loader.load_model(
            ModelType::IntentClassifier,  // Use intent classifier for priority
            ModelConfig::default(),
        ).await?;
        
        let memory_encoder_id = model_loader.load_model(
            ModelType::MiniLM,  // Use MiniLM for memory encoding
            ModelConfig::default(),
        ).await?;
        
        Ok(Self {
            working_memory,
            attention_manager,
            model_loader,
            chunking_model_id,
            priority_scorer_id,
            memory_encoder_id,
            encoding_cache: Arc::new(RwLock::new(WorkingMemoryCache::new(5_000))),
            batch_processor: WorkingMemoryBatchProcessor::new(),
        })
    }
    
    pub async fn add_item(&mut self, item: ItemContent) -> Result<ItemId, WorkingMemoryError> {
        // AI-enhanced chunking
        let chunk_size = self.calculate_chunk_size_ai(&item).await;
        
        // Check capacity
        let current_load = self.items.read().await.iter().map(|i| i.chunk_size).sum::<usize>();
        if current_load + chunk_size > self.capacity {
            // AI-guided eviction
            self.evict_lowest_priority_ai(chunk_size).await?;
        }
        
        // Encode content with neural model
        let content_embedding = self.memory_encoder.encode(&item).await?;
        
        // Calculate importance with AI
        let importance_score = self.priority_scorer.score(&item, &content_embedding).await?;
        
        // Find semantic links to existing items
        let semantic_links = self.find_semantic_links(&content_embedding).await;
        
        // Calculate attention weight
        let attention_weight = self.attention_controller.allocate_attention(&item, importance_score);
        
        let wm_item = WorkingMemoryItem {
            id: ItemId::new(),
            content: item,
            activation_level: 1.0,
            entry_time: Instant::now(),
            last_rehearsal: Instant::now(),
            modality: self.determine_modality(&item),
            chunk_size,
            content_embedding,
            importance_score,
            semantic_links,
            attention_weight,
        };
        
        let id = wm_item.id;
        self.items.write().await.push_back(wm_item);
        
        // Update central executive with new item
        self.central_executive.register_item(id, importance_score).await;
        
        Ok(id)
    }
    
    pub async fn update(&mut self, dt: Duration) {
        let mut items = self.items.write().await;
        
        // Parallel decay calculation for i9 optimization
        let decay_futures: Vec<_> = items.iter_mut()
            .map(|item| async {
                let time_since_rehearsal = item.last_rehearsal.elapsed();
                let base_decay = (-time_since_rehearsal.as_secs_f32() * self.decay_rate).exp();
                
                // AI-modulated decay based on importance
                let importance_factor = 1.0 + (item.importance_score * 0.5);
                let attention_boost = item.attention_weight * 0.3;
                
                item.activation_level *= base_decay * importance_factor + attention_boost;
            })
            .collect();
        
        futures::future::join_all(decay_futures).await;
        
        // AI-guided selective rehearsal
        let rehearsal_candidates = self.select_rehearsal_candidates_ai(&items).await;
        for idx in rehearsal_candidates {
            if let Some(item) = items.get_mut(idx) {
                self.rehearsal_process.rehearse_item(item).await;
                item.last_rehearsal = Instant::now();
                item.activation_level = (item.activation_level * 1.2).min(1.0);
            }
        }
        
        // Remove items below threshold
        items.retain(|item| item.activation_level > ACTIVATION_THRESHOLD);
        
        // Update attention allocation
        self.attention_controller.update_allocation(&items, dt).await;
    }
    
    pub async fn retrieve(&self, cue: &RetrievalCue) -> Option<WorkingMemoryItem> {
        let items = self.items.read().await;
        
        // Encode retrieval cue
        let cue_embedding = self.memory_encoder.encode_cue(cue).await.ok()?;
        
        // Parallel similarity computation using SIMD
        let mut candidates: Vec<_> = items.iter()
            .enumerate()
            .map(|(idx, item)| {
                let semantic_sim = simd_cosine_similarity(&cue_embedding, &item.content_embedding);
                let cue_match = self.matches_cue(item, cue) as i32 as f32;
                let activation = item.activation_level;
                let attention = item.attention_weight;
                
                // Combined score with AI weighting
                let score = semantic_sim * 0.4 + cue_match * 0.3 + activation * 0.2 + attention * 0.1;
                (idx, score)
            })
            .collect();
        
        // Sort by score
        candidates.sort_by_key(|(_, score)| OrderedFloat(-score));
        
        // Return best match
        candidates.first()
            .and_then(|(idx, _)| items.get(*idx))
            .cloned()
    }
    
    async fn calculate_chunk_size_ai(&self, item: &ItemContent) -> usize {
        // Use chunking model to determine optimal chunk size
        match self.chunking_model.predict_chunk_size(item).await {
            Ok(size) => size.max(1).min(4),  // Constrain to reasonable range
            Err(_) => self.calculate_chunk_size_fallback(item),
        }
    }
    
    async fn evict_lowest_priority_ai(&mut self, needed_slots: usize) -> Result<(), WorkingMemoryError> {
        let mut items = self.items.write().await;
        
        // Score all items for eviction priority
        let mut eviction_scores: Vec<_> = items.iter()
            .enumerate()
            .map(|(idx, item)| {
                // Lower score = more likely to evict
                let priority = item.importance_score * 0.4 + 
                              item.activation_level * 0.3 + 
                              item.attention_weight * 0.3;
                (idx, priority)
            })
            .collect();
        
        eviction_scores.sort_by_key(|(_, score)| OrderedFloat(*score));
        
        // Evict items until we have enough space
        let mut freed_slots = 0;
        let mut to_remove = Vec::new();
        
        for (idx, _) in eviction_scores {
            if freed_slots >= needed_slots {
                break;
            }
            freed_slots += items[idx].chunk_size;
            to_remove.push(idx);
        }
        
        // Remove in reverse order to maintain indices
        for idx in to_remove.into_iter().rev() {
            items.remove(idx);
        }
        
        Ok(())
    }
    
    async fn select_rehearsal_candidates_ai(&self, items: &VecDeque<WorkingMemoryItem>) -> Vec<usize> {
        // Use AI to select which items need rehearsal
        let mut candidates = Vec::new();
        
        for (idx, item) in items.iter().enumerate() {
            let needs_rehearsal = item.activation_level < 0.5 && 
                                 item.importance_score > 0.6 &&
                                 item.last_rehearsal.elapsed() > Duration::from_millis(500);
            
            if needs_rehearsal {
                candidates.push(idx);
            }
        }
        
        // Limit rehearsal to prevent overload
        candidates.truncate(3);
        candidates
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
    // AI components
    chunk_encoder: ONNXModel<ChunkEncoder>,      // 5M params for chunk analysis
    pattern_recognizer: ONNXModel<PatternNet>,   // 3M params for pattern detection
    chunk_optimizer: ChunkOptimizer,             // Optimizes chunk boundaries
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
    pub fn new() -> Result<Self> {
        Ok(Self {
            chunk_patterns: HashMap::new(),
            learned_chunks: HashMap::new(),
            chunking_strategies: vec![
                ChunkingStrategy::Semantic,
                ChunkingStrategy::Temporal,
                ChunkingStrategy::Spatial,
                ChunkingStrategy::Phonological,
                ChunkingStrategy::Learned,
                ChunkingStrategy::Neural,  // AI-based chunking
            ],
            chunk_encoder: ONNXModel::load("models/chunk_encoder_int8.onnx")?,
            pattern_recognizer: ONNXModel::load("models/pattern_net_int8.onnx")?,
            chunk_optimizer: ChunkOptimizer::new(),
        })
    }
    
    pub async fn chunk_items(&mut self, items: &[WorkingMemoryItem]) -> Vec<Chunk> {
        // Parallel chunking strategies for i9 optimization
        let strategy_futures: Vec<_> = self.chunking_strategies.iter()
            .map(|strategy| async move {
                match strategy {
                    ChunkingStrategy::Semantic => self.semantic_chunking(items).await,
                    ChunkingStrategy::Temporal => self.temporal_chunking(items),
                    ChunkingStrategy::Spatial => self.spatial_chunking(items),
                    ChunkingStrategy::Phonological => self.phonological_chunking(items),
                    ChunkingStrategy::Learned => self.apply_learned_chunks(items),
                    ChunkingStrategy::Neural => self.neural_chunking(items).await,
                }
            })
            .collect();
        
        let all_chunks = futures::future::join_all(strategy_futures).await;
        
        // Merge all strategy results
        let mut chunks = Vec::new();
        for strategy_chunks in all_chunks {
            chunks.extend(strategy_chunks);
        }
        
        // Use AI to select optimal chunking
        self.select_optimal_chunking_ai(chunks).await
    }
    
    async fn neural_chunking(&self, items: &[WorkingMemoryItem]) -> Vec<Chunk> {
        if items.is_empty() {
            return Vec::new();
        }
        
        // Encode all items
        let encodings: Vec<_> = items.iter()
            .map(|item| &item.content_embedding)
            .collect();
        
        // Use pattern recognizer to find natural boundaries
        let boundaries = self.pattern_recognizer.find_chunk_boundaries(&encodings).await
            .unwrap_or_else(|_| vec![]);
        
        // Create chunks based on boundaries
        let mut chunks = Vec::new();
        let mut start = 0;
        
        for boundary in boundaries {
            if boundary > start && boundary <= items.len() {
                let chunk_items = items[start..boundary].to_vec();
                let chunk = self.create_neural_chunk(chunk_items).await;
                chunks.push(chunk);
                start = boundary;
            }
        }
        
        // Handle remaining items
        if start < items.len() {
            let chunk_items = items[start..].to_vec();
            let chunk = self.create_neural_chunk(chunk_items).await;
            chunks.push(chunk);
        }
        
        chunks
    }
    
    async fn select_optimal_chunking_ai(&self, candidate_chunks: Vec<Chunk>) -> Vec<Chunk> {
        if candidate_chunks.is_empty() {
            return Vec::new();
        }
        
        // Score each chunking configuration
        let mut scored_configs = Vec::new();
        
        // Group chunks by strategy
        let mut strategy_groups: HashMap<ChunkingStrategy, Vec<Chunk>> = HashMap::new();
        for chunk in candidate_chunks {
            strategy_groups.entry(chunk.strategy).or_insert_with(Vec::new).push(chunk);
        }
        
        // Score each strategy's chunking
        for (strategy, chunks) in strategy_groups {
            let score = self.chunk_optimizer.score_chunking(&chunks).await;
            scored_configs.push((strategy, chunks, score));
        }
        
        // Select best configuration
        scored_configs.sort_by_key(|(_, _, score)| OrderedFloat(-score));
        
        scored_configs.into_iter()
            .next()
            .map(|(_, chunks, _)| chunks)
            .unwrap_or_default()
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

// Lightweight neural models for working memory
pub struct ChunkingNet {
    // 5M parameter model for intelligent chunking
}

pub struct PriorityNet {
    // 3M parameter model for importance scoring
}

pub struct MemoryEncoder {
    // 8M parameter model for content encoding
}

## Week 19: Memory Integration & MCP Handlers

### Task 19.1: MCP Tool Integration for Episodic Memory
**File**: `src/mcp/llm_friendly_server/handlers/episodic_memory.rs` (new file)  
**Integration**: Add MCP handlers using existing patterns from `handlers/mod.rs`
```rust
use crate::cognitive::episodic_bridge::{WorkingMemoryEpisodicBridge, Episode, EpisodeId};
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType, MemoryQuery};
use crate::cognitive::attention_manager::AttentionManager;
use crate::mcp::llm_friendly_server::error_handling::{LlmkgError, HandlerResult};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::mcp::llm_friendly_server::validation::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle store_episode request using existing validation patterns
pub async fn handle_store_episode(
    episodic_bridge: &Arc<WorkingMemoryEpisodicBridge>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Use existing validation functions from validation.rs
    let event_description = validate_string_field("event", params.get("event"), true, Some(10000), Some(1))?;
    let use_current_context = validate_boolean_field("use_current_context", params.get("use_current_context"), false, Some(true))?;
    let importance_boost = validate_f32_field("importance_boost", params.get("importance_boost"), false, Some(0.0), Some(1.0), Some(0.0))?;
    
    // Optional context parameters
    let location = validate_optional_string_field("location", params.get("location"), Some(200), Some(1))?;
    let people = validate_optional_string_array_field("people", params.get("people"))?;
    
    // Store episode using existing bridge
    let episode_id = episodic_bridge
        .encode_episode_from_working_memory(event_description, use_current_context)
        .await
        .map_err(|e| LlmkgError::ProcessingError(format!("Failed to encode episode: {}", e)))?;
    
    // Update usage stats using existing pattern
    update_usage_stats(usage_stats, StatsOperation::EpisodeStored).await;
    
    let response = json!({
        "episode_id": episode_id.to_string(),
        "status": "success", 
        "message": "Episode stored with working memory context",
        "context_captured": use_current_context,
        "importance_boost_applied": importance_boost
    });
    
    Ok((response, "Episode stored".to_string(), vec![]))
}

/// Handle query_episodes request with working memory integration
pub async fn handle_query_episodes(
    episodic_bridge: &Arc<WorkingMemoryEpisodicBridge>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Use existing validation patterns
    let query_text = validate_string_field("query", params.get("query"), true, Some(1000), Some(1))?;
    let max_results = validate_u32_field("max_results", params.get("max_results"), false, Some(1), Some(50), Some(10))?;
    let buffer_types = validate_optional_string_array_field("buffer_types", params.get("buffer_types"))?;
    let importance_threshold = validate_f32_field("importance_threshold", params.get("importance_threshold"), false, Some(0.0), Some(1.0), Some(0.3))?;
    
    // Convert buffer types
    let search_buffers = if let Some(types) = buffer_types {
        types.into_iter().map(|t| match t.as_str() {
            "phonological" => BufferType::Phonological,
            "visuospatial" => BufferType::Visuospatial,
            "episodic" => BufferType::Episodic,
            _ => BufferType::Episodic,
        }).collect()
    } else {
        vec![BufferType::Episodic, BufferType::Phonological, BufferType::Visuospatial]
    };
    
    // Create memory query using existing structure
    let memory_query = MemoryQuery {
        query_text,
        search_buffers,
        apply_attention: true,
        importance_threshold,
        recency_weight: 0.7,
    };
    
    // Retrieve episodes using bridge
    let episodes = episodic_bridge
        .retrieve_episodes_for_working_memory(&memory_query, max_results as usize)
        .await
        .map_err(|e| LlmkgError::ProcessingError(format!("Query failed: {}", e)))?;
    
    // Format response
    let episode_data: Vec<Value> = episodes.into_iter()
        .map(|ep| json!({
            "id": ep.id.to_string(),
            "event": ep.event.description,
            "participants": ep.event.participants,
            "importance": ep.importance_score,
            "attention_focus": ep.attention_pattern.focus_strength,
            "working_memory_items": ep.working_memory_snapshot.active_items.len(),
            "timestamp": ep.timestamp.elapsed().as_secs(),
            "narrative": ep.narrative,
        }))
        .collect();
    
    // Update usage stats
    update_usage_stats(usage_stats, StatsOperation::EpisodeQueried).await;
    
    let response = json!({
        "episodes": episode_data,
        "count": episode_data.len(),
        "query": memory_query.query_text,
        "status": "success"
    });
    
    Ok((response, format!("Found {} episodes", episode_data.len()), vec![]))
}

/// Handle enhance_working_memory_with_episodes request
pub async fn handle_enhance_working_memory_with_episodes(
    episodic_bridge: &Arc<WorkingMemoryEpisodicBridge>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Use existing validation patterns
    let query = validate_string_field("query", params.get("query"), true, Some(1000), Some(1))?;
    let buffer_type_str = validate_string_field("buffer_type", params.get("buffer_type"), false, Some(20), Some(1))?;
    let importance_boost = validate_f32_field("importance_boost", params.get("importance_boost"), false, Some(0.0), Some(1.0), Some(0.2))?;
    
    // Convert buffer type
    let buffer_type = match buffer_type_str.as_str() {
        "phonological" => BufferType::Phonological,
        "visuospatial" => BufferType::Visuospatial,
        "episodic" => BufferType::Episodic,
        _ => BufferType::Episodic,
    };
    
    // Enhance working memory with episodic context
    let episode_ids = episodic_bridge
        .enhance_working_memory_with_episodes(&query, buffer_type, importance_boost)
        .await
        .map_err(|e| LlmkgError::ProcessingError(format!("Enhancement failed: {}", e)))?;
    
    // Update usage stats
    update_usage_stats(usage_stats, StatsOperation::WorkingMemoryEnhanced).await;
    
    let response = json!({
        "enhanced_episodes": episode_ids.len(),
        "episode_ids": episode_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "query": query,
        "buffer_type": buffer_type_str,
        "importance_boost": importance_boost,
        "status": "success",
        "message": format!("Working memory enhanced with {} episodes", episode_ids.len())
    });
    
    Ok((response, format!("Enhanced working memory with {} episodes", episode_ids.len()), vec![]))
}

/// Handle working_memory_query request
pub async fn handle_working_memory_query(
    working_memory: &Arc<WorkingMemorySystem>,
    attention_manager: &Arc<AttentionManager>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let query_text = params.get("query")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'query' parameter")?;
    
    let buffer_types = params.get("buffers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|b| b.as_str())
                .map(|s| match s {
                    "phonological" => BufferType::Phonological,
                    "visuospatial" => BufferType::Visuospatial,
                    "episodic" => BufferType::Episodic,
                    _ => BufferType::Episodic,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![BufferType::Episodic, BufferType::Phonological, BufferType::Visuospatial]);
    
    // Create memory query using existing structure
    let memory_query = crate::cognitive::working_memory::MemoryQuery {
        query_text: query_text.to_string(),
        search_buffers: buffer_types,
        apply_attention: true,
        importance_threshold: 0.1,
        recency_weight: 0.5,
    };
    
    // Retrieve from working memory
    let result = working_memory.retrieve_from_working_memory(&memory_query)
        .await
        .map_err(|e| format!("Working memory query failed: {}", e))?;
    
    let items_data: Vec<Value> = result.items.into_iter()
        .map(|item| json!({
            "content": match &item.content {
                MemoryContent::Concept(c) => c.clone(),
                MemoryContent::Entity(e) => e.concept_id.clone(),
                _ => "Unknown content".to_string(),
            },
            "activation": item.activation_level,
            "importance": item.importance_score,
            "timestamp": item.timestamp.elapsed().as_secs(),
        }))
        .collect();
    
    let response = json!({
        "items": items_data,
        "confidence": result.retrieval_confidence,
        "buffer_states": result.buffer_states.iter().map(|bs| json!({
            "type": format!("{:?}", bs.buffer_type),
            "load": bs.current_load,
            "utilization": bs.capacity_utilization,
        })).collect::<Vec<_>>(),
        "status": "success"
    });
    
    Ok((response, format!("Retrieved {} items", items_data.len()), vec![]))
}

/// Handle working_memory_store request
pub async fn handle_working_memory_store(
    working_memory: &Arc<WorkingMemorySystem>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let content = params.get("content")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'content' parameter")?;
    
    let importance = params.get("importance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    
    let buffer_type = params.get("buffer")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "phonological" => BufferType::Phonological,
            "visuospatial" => BufferType::Visuospatial,
            "episodic" => BufferType::Episodic,
            _ => BufferType::Episodic,
        })
        .unwrap_or(BufferType::Episodic);
    
    // Create memory content
    let memory_content = MemoryContent::Concept(content.to_string());
    
    // Store in working memory using existing system
    let result = working_memory.store_in_working_memory(
        memory_content,
        importance,
        buffer_type,
    ).await.map_err(|e| format!("Storage failed: {}", e))?;
    
    // Update usage stats
    update_usage_stats(usage_stats, StatsOperation::WorkingMemoryStored).await;
    
    let response = json!({
        "success": result.success,
        "evicted_items": result.evicted_items.len(),
        "buffer_utilization": result.buffer_state.capacity_utilization,
        "status": "success",
        "message": "Content stored in working memory"
    });
    
    Ok((response, "Stored in working memory".to_string(), vec![]))
}
```

### Task 19.2: Episodic-Semantic Interface
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

## Week 20: Integration & Performance Optimization

### Task 20.1: Storage Integration with Existing Systems
**File**: `src/cognitive/episodic_storage_integration.rs` (new file)
**Goal**: Integrate episodic memory with existing KnowledgeEngine and storage patterns
```rust
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::triple::Triple;
use crate::cognitive::episodic_bridge::{WorkingMemoryEpisodicBridge, Episode, EpisodeId};
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::sdr_storage::SDRStorage;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Integrates episodic memory with existing storage systems
/// Uses existing KnowledgeEngine, SDRStorage, and batch processing
pub struct EpisodicStorageIntegration {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    episodic_bridge: Arc<WorkingMemoryEpisodicBridge>,
    sdr_storage: Arc<SDRStorage>,
    integration_config: IntegrationConfig,
}

#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub batch_size: usize,
    pub enable_triple_generation: bool,
    pub enable_sdr_storage: bool,
    pub enable_consolidation: bool,
    pub performance_mode: PerformanceMode,
}

#[derive(Debug, Clone)]
pub enum PerformanceMode {
    HighThroughput,  // Optimize for many episodes
    LowLatency,      // Optimize for quick access
    Balanced,        // Default balanced mode
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            enable_triple_generation: true,
            enable_sdr_storage: true,
            enable_consolidation: true,
            performance_mode: PerformanceMode::Balanced,
        }
    }
}

impl EpisodicMemoryMigration {
    pub async fn new(
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        episodic_memory: Arc<RwLock<EpisodicMemory>>,
        working_memory: Arc<WorkingMemorySystem>,
    ) -> Result<Self> {
        Ok(Self {
            knowledge_engine,
            episodic_memory,
            working_memory,
            migration_config: MigrationConfig::default(),
        })
    }
    
    /// Migrate existing knowledge to episodic format
    pub async fn migrate_knowledge_to_episodes(&self) -> Result<MigrationResult> {
        let mut result = MigrationResult::new();
        
        // Get existing triples from knowledge engine
        let triples = self.knowledge_engine.read().await
            .get_all_triples()
            .map_err(|e| crate::error::Error::MigrationError(e.to_string()))?;
        
        // Process triples in batches
        for batch in triples.chunks(self.migration_config.batch_size) {
            let batch_result = self.migrate_triple_batch(batch).await?;
            result.merge(batch_result);
        }
        
        Ok(result)
    }
    
    async fn migrate_triple_batch(&self, triples: &[Triple]) -> Result<MigrationResult> {
        let mut result = MigrationResult::new();
        
        for triple in triples {
            // Convert triple to episode format if it represents an event
            if self.is_event_triple(triple) {
                let episode = self.convert_triple_to_episode(triple).await?;
                
                // Store in episodic memory
                let episode_id = self.episodic_memory.write().await
                    .store_episode(episode)
                    .await?;
                
                // Create link back to original triple if configured
                if self.migration_config.create_episode_links {
                    self.create_episode_triple_link(&episode_id, triple).await?;
                }
                
                result.episodes_created += 1;
            } else {
                // Keep as semantic knowledge
                result.triples_preserved += 1;
            }
        }
        
        Ok(result)
    }
    
    fn is_event_triple(&self, triple: &Triple) -> bool {
        // Identify event-like triples based on predicate
        let event_predicates = [
            "happened_at", "occurred_on", "took_place", "experienced",
            "met_with", "went_to", "participated_in", "witnessed"
        ];
        
        event_predicates.iter()
            .any(|pred| triple.predicate.contains(pred))
    }
    
    async fn convert_triple_to_episode(&self, triple: &Triple) -> Result<Episode> {
        // Extract event information from triple
        let event = Event {
            description: format!("{} {} {}", triple.subject, triple.predicate, triple.object),
            participants: self.extract_participants(&triple.subject, &triple.object),
            confidence: triple.confidence,
        };
        
        // Create basic context from available information
        let context = FullContext::from_triple(triple);
        
        // Use existing encoding infrastructure
        let episode_id = EpisodeId::new();
        let episode = Episode {
            id: episode_id,
            event,
            spatial_context: self.infer_spatial_context(triple).await,
            temporal_context: self.infer_temporal_context(triple).await,
            social_context: self.infer_social_context(triple).await,
            cognitive_context: CognitiveContext::default(),
            emotional_context: EmotionalContext::default(),
            sensory_details: SensorySnapshot::default(),
            narrative: None,
            encoding_strength: triple.confidence,
            vividness: 0.5, // Default vividness for migrated data
            personal_significance: self.assess_significance(triple).await,
            rehearsal_count: std::sync::atomic::AtomicU32::new(0),
            episode_embedding: vec![], // Will be computed on first access
            context_embedding: vec![],
            emotion_vector: vec![],
            scene_embedding: None,
        };
        
        Ok(episode)
    }
    
    /// Migrate working memory items to enhanced format
    pub async fn migrate_working_memory(&self) -> Result<MigrationResult> {
        if !self.migration_config.migrate_working_memory {
            return Ok(MigrationResult::new());
        }
        
        let mut result = MigrationResult::new();
        
        // Get current working memory items
        let items = self.working_memory.get_all_items().await?;
        
        // Enhance each item with episodic connections
        for item in items {
            let enhanced_item = self.enhance_memory_item(item).await?;
            
            // Update in working memory (this would require extending WorkingMemorySystem)
            // For now, just count as migrated
            result.working_memory_items_enhanced += 1;
        }
        
        Ok(result)
    }
    
    async fn enhance_memory_item(&self, item: crate::cognitive::working_memory::MemoryItem) -> Result<EnhancedMemoryItem> {
        // Find related episodes for this memory item
        let related_episodes = self.find_related_episodes(&item).await?;
        
        Ok(EnhancedMemoryItem {
            base_item: item,
            content_embedding: None, // Compute lazily
            semantic_links: vec![],   // Extract from knowledge engine
            attention_weight: 0.5,   // Default attention weight
            chunk_metadata: ChunkMetadata {
                chunk_size: 1,
                chunk_strategy: ChunkingStrategy::Semantic,
                chunk_confidence: 0.8,
            },
            episode_links: related_episodes,
        })
    }
    
    /// Verify migration integrity
    pub async fn verify_migration(&self) -> Result<MigrationVerification> {
        let mut verification = MigrationVerification::new();
        
        // Check data consistency
        verification.data_integrity = self.check_data_integrity().await?;
        
        // Check performance impact
        verification.performance_impact = self.measure_performance_impact().await?;
        
        // Check feature compatibility
        verification.feature_compatibility = self.check_feature_compatibility().await?;
        
        Ok(verification)
    }
}

#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub episodes_created: usize,
    pub triples_preserved: usize,
    pub working_memory_items_enhanced: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl MigrationResult {
    pub fn new() -> Self {
        Self {
            episodes_created: 0,
            triples_preserved: 0,
            working_memory_items_enhanced: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn merge(&mut self, other: MigrationResult) {
        self.episodes_created += other.episodes_created;
        self.triples_preserved += other.triples_preserved;
        self.working_memory_items_enhanced += other.working_memory_items_enhanced;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

#[derive(Debug, Clone)]
pub struct MigrationVerification {
    pub data_integrity: bool,
    pub performance_impact: f32,
    pub feature_compatibility: bool,
    pub recommendations: Vec<String>,
}

impl MigrationVerification {
    pub fn new() -> Self {
        Self {
            data_integrity: false,
            performance_impact: 0.0,
            feature_compatibility: false,
            recommendations: Vec::new(),
        }
    }
}
```

### Task 20.2: Testing and Optimization
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
1. **Integrated episodic memory system** using existing `ModelLoader` and AI infrastructure
2. **Enhanced working memory bridge** extending existing `WorkingMemorySystem` and `AttentionManager`
3. **MCP tool integration** with new episodic memory and working memory handlers
4. **Migration system** for existing knowledge and storage patterns
5. **Batch processing enhancement** using existing models with performance optimization
6. **Comprehensive test suite** with integration and performance benchmarks

## Success Criteria
- [ ] **Working Memory Bridge**: Seamless integration with existing `WorkingMemorySystem` without API breaking changes
- [ ] **Attention Integration**: Full coordination with existing `AttentionManager` for episodic attention weighting
- [ ] **Model Reuse**: Effective utilization of existing models (DistilBERT-NER, MiniLM, T5-Small, TinyBERT-NER)
- [ ] **MCP Handler Integration**: New episodic handlers follow existing patterns from `handlers/mod.rs`
- [ ] **Storage Extension**: Episodes stored using existing `KnowledgeEngine` and `SDRStorage` patterns
- [ ] **Performance Targets**:
  - Episode encoding: <10ms using existing model infrastructure
  - Working memory enhancement: <5ms using existing buffer operations
  - Episodic retrieval: <8ms using existing similarity search
  - MCP handler response: <5ms using existing validation patterns
- [ ] **Cognitive Coordination**: Integration with existing `CognitiveOrchestrator` and pattern systems
- [ ] **Batch Processing**: Utilization of existing batch processing infrastructure for encoding operations

## Dependencies
- **CRITICAL - Existing Systems to Extend (Not Replace)**:
  - `src/cognitive/working_memory.rs` - Sophisticated buffer management system
  - `src/cognitive/attention_manager.rs` - Full attention orchestration with pattern coordination
  - `src/models/model_loader.rs` - Comprehensive ONNX runtime with 8+ models
  - `src/mcp/llm_friendly_server/handlers/mod.rs` - 28+ MCP tools with established patterns
  - `src/core/knowledge_engine.rs` - Advanced knowledge storage and retrieval
  - `src/core/sdr_storage.rs` - SDR storage with batch processing capabilities
  - `src/cognitive/orchestrator.rs` - Cognitive pattern coordination and orchestration
- **Available Model Infrastructure**:
  - ModelLoader with comprehensive ONNX runtime integration
  - DistilBERT-NER (66M params) - Already available for context encoding
  - all-MiniLM-L6-v2 (22M params) - Already available for similarity and retrieval
  - T5-Small (60M params) - Already available for narrative generation  
  - TinyBERT-NER (14.5M params) - Already available for chunking and pattern detection
  - Intent Classifier (30M params) - Already available for priority scoring
- **Existing Infrastructure to Leverage**:
  - MCP validation patterns from `validation.rs`
  - Error handling from `error_handling.rs`
  - Usage stats from `types.rs`
  - Temporal tracking systems
  - Batch processing and SIMD operations
  - DashMap and Tokio async patterns

## Integration Strategy
1. **Phase 1**: Create episodic bridge that extends existing WorkingMemorySystem
2. **Phase 2**: Add MCP handlers that follow existing patterns and validation
3. **Phase 3**: Integrate with existing AttentionManager for coordinated focus
4. **Phase 4**: Extend storage using existing KnowledgeEngine and SDRStorage patterns
5. **Phase 5**: Performance optimization using existing batch processing infrastructure

## Risks & Mitigations
1. **Breaking Existing Systems**
   - **Risk**: High - Modifying core working memory could impact 28+ existing systems
   - **Mitigation**: Bridge pattern - extend without modifying existing APIs, comprehensive integration testing
2. **Model Infrastructure Compatibility**
   - **Risk**: Medium - Existing models may not be optimally suited for episodic tasks
   - **Mitigation**: Multi-model ensemble approach, leverage existing fallback patterns, extensive testing
3. **Performance Impact on Existing Operations**
   - **Risk**: Medium - Additional episodic processing could slow existing workflows
   - **Mitigation**: Lazy loading, optional episodic features, use existing performance monitoring
4. **Storage Integration Complexity**
   - **Risk**: Medium - Episodes need to integrate with existing knowledge storage patterns
   - **Mitigation**: Follow existing storage patterns, use established KnowledgeEngine APIs
5. **MCP Handler Pattern Compliance**
   - **Risk**: Low - New handlers must follow established patterns
   - **Mitigation**: Use existing validation, error handling, and response patterns from handlers/mod.rs