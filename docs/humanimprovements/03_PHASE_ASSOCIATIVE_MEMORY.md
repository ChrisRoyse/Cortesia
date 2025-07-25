# Phase 3: Associative Memory Networks

## Overview
**Duration**: 4 weeks  
**Goal**: Implement human-like associative memory fully integrated with existing cognitive architecture  
**Priority**: HIGH  
**Dependencies**: Phase 1 & 2 completion  
**Target Performance**: <2ms for activation spreading on Intel i9 with full cognitive integration

## Critical Integration Requirements
**IMPORTANT**: This phase INTEGRATES with existing systems - DO NOT rebuild existing functionality

### Existing Cognitive Systems (DO NOT REBUILD):
- ✅ **WorkingMemorySystem** (3 buffers, decay, consolidation) - `src/cognitive/working_memory.rs`
- ✅ **AttentionManager** (selective, divided, sustained attention) - `src/cognitive/attention_manager.rs`
- ✅ **CompetitiveInhibitionSystem** (lateral inhibition, winner-takes-all) - `src/cognitive/inhibitory/mod.rs`
- ✅ **HebbianLearningEngine** (STDP, synaptic plasticity, correlation tracking) - `src/learning/hebbian.rs`
- ✅ **NeuralBridgeFinder** (creative pathfinding, bridge discovery) - `src/cognitive/neural_bridge_finder.rs`
- ✅ **FederationManager** (cross-shard coordination) - `src/federation/mod.rs`
- ✅ **CognitiveOrchestrator** (pattern coordination) - `src/cognitive/orchestrator.rs`

### Available AI Models (DO NOT REBUILD):
- ✅ **all-MiniLM-L6-v2** (22M params) - Semantic similarity via `src/models/rust_embeddings.rs`
- ✅ **DistilBERT-NER** (66M params) - Entity recognition via `src/models/rust_bert_models.rs`
- ✅ **T5-Small** (60M params) - Text generation via `src/models/rust_t5_models.rs`
- ✅ **Native SIMD operations** - Optimized similarity calculations

### Federation Infrastructure (DO NOT REBUILD):
- ✅ **DatabaseRegistry** - Multi-database coordination
- ✅ **QueryRouter** - Federated query planning and execution
- ✅ **ResultMerger** - Cross-shard result aggregation
- ✅ **FederationCoordinator** - Transaction management

## Week 9: Associative Memory Integration Layer

### Task 9.1: Cognitive Systems Integration Hub
**File**: `src/associative/cognitive_integration_hub.rs` (new file)
```rust
use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType};
use crate::cognitive::attention_manager::{AttentionManager, AttentionType, AttentionResult};
use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
use crate::learning::hebbian::HebbianLearningEngine;
use crate::cognitive::neural_bridge_finder::NeuralBridgeFinder;
use crate::federation::FederationManager;
use crate::models::{ModelType, rust_embeddings::RustEmbeddingModel};
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Instant, Duration};
use dashmap::DashMap;

/// Central hub for coordinating all cognitive systems in associative memory operations
/// This DOES NOT replace existing systems - it coordinates them
pub struct CognitiveIntegrationHub {
    // Existing cognitive systems (DO NOT recreate these)
    orchestrator: Arc<CognitiveOrchestrator>,
    working_memory: Arc<WorkingMemorySystem>,
    attention_manager: Arc<AttentionManager>,
    inhibition_system: Arc<CompetitiveInhibitionSystem>,
    hebbian_engine: Arc<HebbianLearningEngine>,
    bridge_finder: Arc<NeuralBridgeFinder>,
    federation_manager: Arc<FederationManager>,
    
    // Existing AI models (DO NOT recreate these)
    embedding_model: Arc<RustEmbeddingModel>,
    brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    
    // NEW: Associative-specific coordination state
    activation_cache: Arc<DashMap<EntityKey, AssociativeActivation>>,
    spreading_config: SpreadingActivationConfig,
    pattern_completion_cache: Arc<DashMap<String, CompletedPattern>>,
}

#[derive(Debug, Clone)]
pub struct AssociativeActivation {
    pub entity_key: EntityKey,
    pub activation_strength: f32,
    pub attention_weight: f32,
    pub inhibition_applied: f32,
    pub hebbian_boost: f32,
    pub federation_sources: Vec<String>,
    pub timestamp: Instant,
    pub working_memory_stored: bool,
}

#[derive(Debug, Clone)]
pub struct SpreadingActivationConfig {
    pub max_spread_depth: usize,
    pub activation_threshold: f32,
    pub semantic_weight: f32,
    pub attention_amplification: f32,
    pub inhibition_strength: f32,
    pub temporal_decay_rate: f32,
    pub cross_shard_enabled: bool,
    pub working_memory_integration: bool,
}

#[derive(Debug, Clone)]
pub struct CompletedPattern {
    pub partial_cues: Vec<String>,
    pub completed_concepts: Vec<String>,
    pub completion_confidence: f32,
    pub neural_bridge_paths: Vec<String>,
    pub working_memory_sources: Vec<String>,
    pub timestamp: Instant,
}

pub struct ActivationNode {
    id: NodeId,
    concept: String,
    base_activation: f32,
    current_activation: AtomicF32,
    decay_rate: f32,
    last_activated: AtomicInstant,
    // AI-enhanced fields
    embedding: Vec<f32>,              // Pre-computed embedding
    semantic_neighbors: Vec<NodeId>,  // Pre-computed nearest neighbors
    context_weight: f32,              // Attention-based weight
}

pub struct ActivationParams {
    spread_rate: f32,           // How much activation spreads
    decay_constant: f32,        // How fast activation decays
    threshold: f32,             // Minimum activation to spread
    max_iterations: usize,      // Prevent infinite loops
    spread_function: SpreadType,
    // AI parameters
    semantic_spread_weight: f32,  // Weight for semantic similarity
    attention_threshold: f32,     // Minimum attention weight
    parallel_threads: usize,      // For i9 optimization
}

impl CognitiveIntegrationHub {
    pub async fn new(
        orchestrator: Arc<CognitiveOrchestrator>,
        working_memory: Arc<WorkingMemorySystem>,
        attention_manager: Arc<AttentionManager>,
        inhibition_system: Arc<CompetitiveInhibitionSystem>,
        hebbian_engine: Arc<HebbianLearningEngine>,
        bridge_finder: Arc<NeuralBridgeFinder>,
        federation_manager: Arc<FederationManager>,
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Result<Self> {
        // Load existing embedding model (DON'T recreate)
        let embedding_model = Arc::new(RustEmbeddingModel::load_model(
            ModelType::MiniLM,
            "./src/models/pretrained/all_minilm_l6_v2"
        )?);
        
        Ok(Self {
            orchestrator,
            working_memory,
            attention_manager,
            inhibition_system,
            hebbian_engine,
            bridge_finder,
            federation_manager,
            embedding_model,
            brain_graph,
            activation_cache: Arc::new(DashMap::with_capacity(100_000)),
            spreading_config: SpreadingActivationConfig {
                max_spread_depth: 5,
                activation_threshold: 0.1,
                semantic_weight: 0.3,
                attention_amplification: 2.0,
                inhibition_strength: 0.7,
                temporal_decay_rate: 0.05,
                cross_shard_enabled: true,
                working_memory_integration: true,
            },
            pattern_completion_cache: Arc::new(DashMap::with_capacity(10_000)),
        })
    }

    /// Coordinate associative activation across ALL existing cognitive systems
    pub async fn coordinate_associative_activation(
        &self, 
        concept: &str, 
        initial_strength: f32
    ) -> Result<AssociativeActivationResult> {
        let start_time = Instant::now();
        
        // 1. COORDINATE with AttentionManager (existing system)
        let attention_targets = self.find_concept_entities(concept).await?;
        let attention_result = self.attention_manager.focus_attention(
            attention_targets.clone(),
            initial_strength,
            AttentionType::Selective,
        ).await?;
        let attention_weight = attention_result.attention_strength;
        
        // 2. COORDINATE with WorkingMemorySystem (existing system)
        let working_memory_content = MemoryContent::Concept(concept.to_string());
        let memory_storage_result = self.working_memory.store_in_working_memory_with_attention(
            working_memory_content,
            initial_strength,
            BufferType::Episodic,
            attention_weight,
        ).await?;
        
        // 3. COORDINATE with HebbianLearningEngine (existing system)
        let activation_events = attention_targets.iter().map(|&entity_key| {
            crate::learning::types::ActivationEvent {
                entity_key,
                activation_strength: initial_strength * attention_weight,
                timestamp: start_time,
                context: crate::learning::types::ActivationContext {
                    query_id: format!("associative_{}", uuid::Uuid::new_v4()),
                    cognitive_pattern: crate::cognitive::types::CognitivePatternType::Divergent,
                    user_session: None,
                    outcome_quality: None,
                },
            }
        }).collect();
        
        let learning_update = self.hebbian_engine.apply_hebbian_learning(
            activation_events,
            crate::learning::types::LearningContext::AssociativeMemory {
                time_window: Duration::from_millis(1000),
                co_activation_threshold: 0.5,
            },
        ).await?;
        
        // 4. COORDINATE with NeuralBridgeFinder (existing system)
        let bridge_paths = self.bridge_finder.find_creative_bridges(
            concept,
            "*"  // Find bridges to any concept
        ).await?;
        
        // 5. COORDINATE with CompetitiveInhibitionSystem (existing system)
        let activation_pattern = self.create_activation_pattern_from_targets(&attention_targets, attention_weight)?;
        let inhibition_result = self.inhibition_system.apply_competitive_inhibition(
            &activation_pattern,
            Some(concept.to_string()),
        ).await?;
        
        // 6. COORDINATE with FederationManager for cross-shard activation (existing system)
        let federated_results = if self.spreading_config.cross_shard_enabled {
            self.federation_manager.execute_similarity_search(
                concept,
                0.3,  // similarity threshold
                50    // max results
            ).await?
        } else {
            Vec::new()
        };
        
        // 7. Apply semantic spreading using existing embedding model
        let semantic_activations = self.compute_semantic_activations(
            concept,
            &attention_targets,
            initial_strength * attention_weight
        ).await?;
        
        // 8. Consolidate all activation sources into final result
        let mut final_activations = Vec::new();
        
        // Add attention-focused entities
        for &entity_key in &attention_targets {
            final_activations.push(AssociativeActivation {
                entity_key,
                activation_strength: initial_strength * attention_weight,
                attention_weight,
                inhibition_applied: inhibition_result.inhibition_strength_applied,
                hebbian_boost: learning_update.learning_efficiency,
                federation_sources: federated_results.iter().map(|r| r.database_id.clone()).collect(),
                timestamp: start_time,
                working_memory_stored: memory_storage_result.success,
            });
        }
        
        // Add semantic activations
        final_activations.extend(semantic_activations);
        
        // Add bridge-discovered concepts
        for bridge_path in &bridge_paths {
            for intermediate_concept in &bridge_path.intermediate_concepts {
                if let Ok(entity_keys) = self.find_concept_entities(intermediate_concept).await {
                    for entity_key in entity_keys {
                        final_activations.push(AssociativeActivation {
                            entity_key,
                            activation_strength: bridge_path.connection_strength * 0.5,
                            attention_weight: 0.3,
                            inhibition_applied: 0.0,
                            hebbian_boost: bridge_path.creativity_score,
                            federation_sources: vec![],
                            timestamp: start_time,
                            working_memory_stored: false,
                        });
                    }
                }
            }
        }
        
        // Cache results
        for activation in &final_activations {
            self.activation_cache.insert(activation.entity_key, activation.clone());
        }
        
        // Sort by combined activation strength
        final_activations.sort_by(|a, b| {
            let score_a = a.activation_strength * (1.0 + a.attention_weight + a.hebbian_boost);
            let score_b = b.activation_strength * (1.0 + b.attention_weight + b.hebbian_boost);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        final_activations.truncate(100); // Limit results
        
        Ok(AssociativeActivationResult {
            activations: final_activations,
            attention_result,
            inhibition_result,
            learning_update,
            bridge_paths,
            federated_results,
            processing_time_ms: start_time.elapsed().as_millis() as f32,
        })
    }
    
    async fn spread_activation_worker(
        &self,
        state: Arc<ActivationState>,
        thread_id: usize
    ) -> Vec<(String, f32, usize)> {
        let mut local_results = Vec::new();
        
        while let Some((node_id, activation, depth)) = state.get_next_node() {
            if depth >= self.activation_params.max_iterations {
                continue;
            }
            
            // Get node
            let node = match self.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };
            
            // Record activation
            local_results.push((
                node.concept.clone(),
                activation,
                depth
            ));
            
            // Spread to edges
            if let Some(edges) = self.edges.get(&node_id) {
                for edge in edges.iter() {
                    let spread = self.calculate_ai_spread(activation, edge, &node).await;
                    if spread > self.activation_params.threshold {
                        state.add_to_queue(edge.target, spread, depth + 1);
                    }
                }
            }
        }
        
        local_results
    }
    
    async fn calculate_ai_spread(
        &self,
        source_activation: f32,
        edge: &Edge,
        source_node: &ActivationNode
    ) -> f32 {
        // Base spread calculation
        let base_spread = match self.activation_params.spread_function {
            SpreadType::Linear => source_activation * edge.weight * self.activation_params.spread_rate,
            SpreadType::Logarithmic => (source_activation * edge.weight).ln() * self.activation_params.spread_rate,
            SpreadType::Sigmoid => sigmoid(source_activation * edge.weight) * self.activation_params.spread_rate,
        };
        
        // Semantic similarity boost
        if let Some(target_node) = self.nodes.get(&edge.target) {
            let semantic_sim = self.similarity_engine.compute_similarity(
                &source_node.embedding,
                &target_node.embedding
            );
            
            // Attention-based modulation
            let attention_weight = if source_node.context_weight > self.activation_params.attention_threshold {
                source_node.context_weight * target_node.context_weight
            } else {
                1.0
            };
            
            base_spread * (1.0 + semantic_sim * self.activation_params.semantic_spread_weight) * attention_weight
        } else {
            base_spread
        }
    }
    
    async fn get_semantic_neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f32)> {
        if let Some(node) = self.nodes.get(&node_id) {
            // Use pre-computed neighbors if available
            if !node.semantic_neighbors.is_empty() {
                return node.semantic_neighbors.iter()
                    .map(|&id| (id, 1.0))
                    .collect();
            }
            
            // Otherwise compute on the fly using SIMD
            self.similarity_engine.find_nearest_neighbors(
                &node.embedding,
                20  // Top 20 neighbors
            )
        } else {
            Vec::new()
        }
    }
}

// SIMD-optimized similarity for i9
pub struct SIMDSimilarity {
    embedding_dim: usize,
}

impl SIMDSimilarity {
    pub fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            
            let mut sum = _mm256_setzero_ps();
            for i in (0..a.len()).step_by(8) {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }
            
            let dist_sq = hsum_ps_avx(sum);
            1.0 / (1.0 + dist_sq.sqrt())  // Convert distance to similarity
        }
    }
}
```

### Task 9.2: Semantic Similarity Integration (Uses Existing Models)
**File**: `src/associative/semantic_integration.rs` (new file)
```rust
use crate::models::{ModelType, rust_embeddings::RustEmbeddingModel};
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;

/// Integrates existing semantic models with associative memory (NO NEW MODELS)
pub struct SemanticIntegration {
    embedding_model: Arc<RustEmbeddingModel>,  // Uses existing model
    similarity_cache: Arc<dashmap::DashMap<(EntityKey, EntityKey), f32>>,
    embedding_cache: Arc<dashmap::DashMap<String, Vec<f32>>>,
}

impl SemanticIntegration {
    pub fn new(embedding_model: Arc<RustEmbeddingModel>) -> Self {
        Self {
            embedding_model,
            similarity_cache: Arc::new(dashmap::DashMap::with_capacity(100_000)),
            embedding_cache: Arc::new(dashmap::DashMap::with_capacity(50_000)),
        }
    }
    
    /// Compute semantic similarity using existing all-MiniLM-L6-v2 model
    pub async fn compute_similarity(&self, concept_a: &str, concept_b: &str) -> Result<f32> {
        // Check cache first
        let cache_key = (concept_a.to_string(), concept_b.to_string());
        if let Some(cached) = self.get_cached_similarity(&cache_key) {
            return Ok(cached);
        }
        
        // Get embeddings using existing model
        let embedding_a = self.get_or_compute_embedding(concept_a).await?;
        let embedding_b = self.get_or_compute_embedding(concept_b).await?;
        
        // Compute cosine similarity using SIMD optimization
        let similarity = self.simd_cosine_similarity(&embedding_a, &embedding_b);
        
        // Cache result
        self.cache_similarity(cache_key, similarity);
        
        Ok(similarity)
    }
    
    async fn get_or_compute_embedding(&self, concept: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.embedding_cache.get(concept) {
            return Ok(cached.clone());
        }
        
        // Use existing embedding model (DON'T create new one)
        let embedding = self.embedding_model.encode_text(concept).await?;
        self.embedding_cache.insert(concept.to_string(), embedding.clone());
        
        Ok(embedding)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn simd_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            
            let mut dot_product = _mm256_setzero_ps();
            let mut norm_a = _mm256_setzero_ps();
            let mut norm_b = _mm256_setzero_ps();
            
            for i in (0..a.len().min(b.len())).step_by(8) {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                
                dot_product = _mm256_fmadd_ps(va, vb, dot_product);
                norm_a = _mm256_fmadd_ps(va, va, norm_a);
                norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
            }
            
            let dot = hsum_ps_avx(dot_product);
            let norm_a_val = hsum_ps_avx(norm_a).sqrt();
            let norm_b_val = hsum_ps_avx(norm_b).sqrt();
            
            if norm_a_val == 0.0 || norm_b_val == 0.0 {
                0.0
            } else {
                dot / (norm_a_val * norm_b_val)
            }
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn simd_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        // Fallback implementation for non-x86_64
        let dot: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
        let norm_a: f32 = a.iter().map(|ai| ai * ai).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|bi| bi * bi).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
```

### Task 9.3: Working Memory Integration (Uses Existing System)
**File**: `src/associative/working_memory_integration.rs` (new file)
```rust
use crate::cognitive::working_memory::{
    WorkingMemorySystem, MemoryContent, BufferType, MemoryQuery, MemoryItem
};
use crate::cognitive::attention_manager::{AttentionManager, AttentionType};
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::time::Duration;

/// Integrates associative memory with existing WorkingMemorySystem
/// DOES NOT replace working memory - coordinates with it
pub struct WorkingMemoryIntegration {
    working_memory: Arc<WorkingMemorySystem>,  // Uses existing system
    attention_manager: Arc<AttentionManager>,   // Uses existing system
    associative_config: AssociativeMemoryConfig,
}

#[derive(Debug, Clone)]
pub struct AssociativeMemoryConfig {
    pub episodic_buffer_priority: f32,
    pub phonological_buffer_priority: f32,
    pub visuospatial_buffer_priority: f32,
    pub attention_boost_threshold: f32,
    pub memory_consolidation_delay: Duration,
    pub association_strength_threshold: f32,
}

impl WorkingMemoryIntegration {
    pub fn new(
        working_memory: Arc<WorkingMemorySystem>,
        attention_manager: Arc<AttentionManager>,
    ) -> Self {
        Self {
            working_memory,
            attention_manager,
            associative_config: AssociativeMemoryConfig {
                episodic_buffer_priority: 0.8,
                phonological_buffer_priority: 0.6,
                visuospatial_buffer_priority: 0.4,
                attention_boost_threshold: 0.7,
                memory_consolidation_delay: Duration::from_millis(2000),
                association_strength_threshold: 0.5,
            },
        }
    }
    
    /// Store associative activations in appropriate working memory buffers
    pub async fn store_associative_activations(
        &self,
        activations: &[crate::associative::cognitive_integration_hub::AssociativeActivation],
    ) -> Result<WorkingMemoryStorageResult> {
        let mut storage_results = Vec::new();
        
        for activation in activations {
            // Determine appropriate buffer based on activation characteristics
            let buffer_type = self.determine_optimal_buffer(activation);
            
            // Create memory content
            let memory_content = MemoryContent::Concept(
                format!("entity_{:?}", activation.entity_key)
            );
            
            // Apply attention boost if activation is strong enough
            let storage_result = if activation.attention_weight > self.associative_config.attention_boost_threshold {
                self.working_memory.store_in_working_memory_with_attention(
                    memory_content,
                    activation.activation_strength,
                    buffer_type,
                    activation.attention_weight,
                ).await?
            } else {
                self.working_memory.store_in_working_memory(
                    memory_content,
                    activation.activation_strength,
                    buffer_type,
                ).await?
            };
            
            storage_results.push(storage_result);
        }
        
        Ok(WorkingMemoryStorageResult {
            total_stored: storage_results.len(),
            episodic_stored: storage_results.iter().filter(|r| matches!(r.buffer_state.buffer_type, BufferType::Episodic)).count(),
            phonological_stored: storage_results.iter().filter(|r| matches!(r.buffer_state.buffer_type, BufferType::Phonological)).count(),
            visuospatial_stored: storage_results.iter().filter(|r| matches!(r.buffer_state.buffer_type, BufferType::Visuospatial)).count(),
            total_evicted: storage_results.iter().map(|r| r.evicted_items.len()).sum(),
            average_capacity_utilization: storage_results.iter().map(|r| r.buffer_state.capacity_utilization).sum::<f32>() / storage_results.len() as f32,
        })
    }
    
    /// Retrieve relevant memories for associative priming
    pub async fn retrieve_associative_memories(
        &self,
        priming_concepts: &[String],
        max_results: usize,
    ) -> Result<Vec<MemoryItem>> {
        let mut all_relevant_memories = Vec::new();
        
        for concept in priming_concepts {
            let query = MemoryQuery {
                query_text: concept.clone(),
                search_buffers: vec![BufferType::Episodic, BufferType::Phonological],
                apply_attention: true,
                importance_threshold: self.associative_config.association_strength_threshold,
                recency_weight: 0.3,
            };
            
            let retrieval_result = self.working_memory.retrieve_from_working_memory(&query).await?;
            all_relevant_memories.extend(retrieval_result.items);
        }
        
        // Sort by relevance and limit results
        all_relevant_memories.sort_by(|a, b| {
            b.importance_score.partial_cmp(&a.importance_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_relevant_memories.truncate(max_results);
        
        Ok(all_relevant_memories)
    }
    
    fn determine_optimal_buffer(
        &self, 
        activation: &crate::associative::cognitive_integration_hub::AssociativeActivation
    ) -> BufferType {
        // Use episodic buffer for high-importance associative memories
        if activation.activation_strength > 0.7 && activation.attention_weight > 0.5 {
            BufferType::Episodic
        }
        // Use phonological buffer for concept-based activations
        else if activation.hebbian_boost > 0.3 {
            BufferType::Phonological
        }
        // Use visuospatial buffer for spatial/relational concepts
        else {
            BufferType::Visuospatial
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkingMemoryStorageResult {
    pub total_stored: usize,
    pub episodic_stored: usize,
    pub phonological_stored: usize,
    pub visuospatial_stored: usize,
    pub total_evicted: usize,
    pub average_capacity_utilization: f32,
}
```

## Week 10: Pattern Completion via Cognitive Systems Integration

### Task 10.1: Neural Bridge Pattern Completion (Uses Existing Systems)
**File**: `src/associative/pattern_completion.rs` (new file)
```rust
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryItem, MemoryQuery, BufferType};
use crate::cognitive::attention_manager::{AttentionManager, AttentionType};
use crate::cognitive::neural_bridge_finder::{NeuralBridgeFinder, BridgePath};
use crate::learning::hebbian::HebbianLearningEngine;
use crate::models::{ModelType, rust_embeddings::RustEmbeddingModel};
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use dashmap::DashMap;

/// Pattern completion using existing cognitive systems coordination
/// DOES NOT create new AI models - uses existing ones
pub struct CognitivePatternCompletion {
    // Existing systems (DO NOT recreate)
    working_memory: Arc<WorkingMemorySystem>,
    attention_manager: Arc<AttentionManager>,
    bridge_finder: Arc<NeuralBridgeFinder>,
    hebbian_engine: Arc<HebbianLearningEngine>,
    embedding_model: Arc<RustEmbeddingModel>,
    
    // NEW: Pattern completion coordination
    completion_cache: Arc<DashMap<String, CompletedPattern>>,
    completion_config: PatternCompletionConfig,
}

#[derive(Debug, Clone)]
pub struct PatternCompletionConfig {
    pub completion_threshold: f32,
    pub max_completion_depth: usize,
    pub semantic_similarity_weight: f32,
    pub working_memory_weight: f32,
    pub bridge_path_weight: f32,
    pub hebbian_association_weight: f32,
    pub attention_amplification: f32,
}

#[derive(Debug, Clone)]
pub struct CompletedPattern {
    pub partial_cues: Vec<String>,
    pub completed_concepts: Vec<CompletedConcept>,
    pub completion_confidence: f32,
    pub completion_method: CompletionMethod,
    pub processing_time_ms: f32,
    pub source_systems: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompletedConcept {
    pub concept: String,
    pub confidence: f32,
    pub source: CompletionSource,
    pub entity_key: Option<EntityKey>,
}

#[derive(Debug, Clone)]
pub enum CompletionMethod {
    WorkingMemoryRetrieval,
    NeuralBridgeDiscovery,
    HebbianAssociation,
    SemanticSimilarity,
    HybridCompletion,
}

#[derive(Debug, Clone)]
pub enum CompletionSource {
    EpisodicMemory,
    PhonologicalMemory,
    VisuospatialMemory,
    BridgePath(String),
    HebbianConnection,
    SemanticEmbedding,
}

pub struct MemoryPattern {
    id: PatternId,
    components: Vec<(NodeId, f32)>,  // concept and typical activation
    frequency: AtomicU32,
    last_accessed: AtomicInstant,
    // AI fields
    pattern_embedding: Vec<f32>,     // Encoded pattern representation
    confidence_score: f32,            // Model confidence in pattern
    semantic_signature: Vec<f32>,     // Semantic fingerprint
}

impl CognitivePatternCompletion {
    pub async fn new(
        working_memory: Arc<WorkingMemorySystem>,
        attention_manager: Arc<AttentionManager>,
        bridge_finder: Arc<NeuralBridgeFinder>,
        hebbian_engine: Arc<HebbianLearningEngine>,
    ) -> Result<Self> {
        // Load existing embedding model (DON'T create new one)
        let embedding_model = Arc::new(RustEmbeddingModel::load_model(
            ModelType::MiniLM,
            "./src/models/pretrained/all_minilm_l6_v2"
        )?);
        
        Ok(Self {
            working_memory,
            attention_manager,
            bridge_finder,
            hebbian_engine,
            embedding_model,
            completion_cache: Arc::new(DashMap::with_capacity(10_000)),
            completion_config: PatternCompletionConfig {
                completion_threshold: 0.6,
                max_completion_depth: 4,
                semantic_similarity_weight: 0.3,
                working_memory_weight: 0.4,
                bridge_path_weight: 0.2,
                hebbian_association_weight: 0.1,
                attention_amplification: 1.5,
            },
        })
    }

    /// Complete partial pattern using coordination of ALL existing cognitive systems
    pub async fn complete_pattern_with_cognitive_coordination(
        &self, 
        partial_cues: &[String]
    ) -> Result<CompletedPattern> {
        let start_time = Instant::now();
        let partial_key = partial_cues.join("|");
        
        // Check cache first
        if let Some(cached) = self.completion_cache.get(&partial_key) {
            return Ok(cached.clone());
        }
        
        // 1. COORDINATE with AttentionManager for focus
        let attention_targets = self.extract_entity_keys_from_cues(partial_cues).await?;
        let attention_result = self.attention_manager.focus_attention(
            attention_targets,
            0.8, // High focus for pattern completion
            AttentionType::Selective,
        ).await?;
        
        // 2. COORDINATE with WorkingMemorySystem for similar patterns
        let mut completed_concepts = Vec::new();
        let mut source_systems = Vec::new();
        
        for cue in partial_cues {
            let memory_query = MemoryQuery {
                query_text: cue.clone(),
                search_buffers: vec![BufferType::Episodic, BufferType::Phonological, BufferType::Visuospatial],
                apply_attention: true,
                importance_threshold: 0.3,
                recency_weight: 0.4,
            };
            
            let memory_retrieval = self.working_memory.retrieve_from_working_memory(&memory_query).await?;
            for memory_item in memory_retrieval.items {
                if let crate::cognitive::working_memory::MemoryContent::Concept(concept) = memory_item.content {
                    if !partial_cues.contains(&concept) {
                        completed_concepts.push(CompletedConcept {
                            concept: concept.clone(),
                            confidence: memory_item.importance_score * self.completion_config.working_memory_weight,
                            source: match memory_query.search_buffers[0] {
                                BufferType::Episodic => CompletionSource::EpisodicMemory,
                                BufferType::Phonological => CompletionSource::PhonologicalMemory,
                                BufferType::Visuospatial => CompletionSource::VisuospatialMemory,
                            },
                            entity_key: None,
                        });
                    }
                }
            }
        }
        source_systems.push("WorkingMemorySystem".to_string());
        
        // 3. COORDINATE with NeuralBridgeFinder for creative completions
        if partial_cues.len() >= 2 {
            for i in 0..partial_cues.len() {
                for j in (i + 1)..partial_cues.len() {
                    let bridge_paths = self.bridge_finder.find_creative_bridges(
                        &partial_cues[i],
                        &partial_cues[j]
                    ).await?;
                    
                    for bridge_path in bridge_paths {
                        for intermediate in &bridge_path.intermediate_concepts {
                            if !partial_cues.contains(intermediate) && 
                               !completed_concepts.iter().any(|c| &c.concept == intermediate) {
                                completed_concepts.push(CompletedConcept {
                                    concept: intermediate.clone(),
                                    confidence: bridge_path.connection_strength * self.completion_config.bridge_path_weight,
                                    source: CompletionSource::BridgePath(bridge_path.bridge_id.clone()),
                                    entity_key: None,
                                });
                            }
                        }
                    }
                }
            }
            source_systems.push("NeuralBridgeFinder".to_string());
        }
        
        // 4. COORDINATE with embedding model for semantic similarity
        for cue in partial_cues {
            let cue_embedding = self.embedding_model.encode_text(cue).await?;
            
            // Find semantically similar concepts in existing knowledge base
            let similar_concepts = self.find_semantically_similar_concepts(
                &cue_embedding,
                10,  // max results
                0.4  // similarity threshold
            ).await?;
            
            for (concept, similarity) in similar_concepts {
                if !partial_cues.contains(&concept) && 
                   !completed_concepts.iter().any(|c| c.concept == concept) {
                    completed_concepts.push(CompletedConcept {
                        concept,
                        confidence: similarity * self.completion_config.semantic_similarity_weight,
                        source: CompletionSource::SemanticEmbedding,
                        entity_key: None,
                    });
                }
            }
        }
        source_systems.push("RustEmbeddingModel".to_string());
        
        // Apply attention amplification
        for concept in &mut completed_concepts {
            concept.confidence *= attention_result.attention_strength * self.completion_config.attention_amplification;
        }
        
        // Sort by confidence and limit results
        completed_concepts.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        completed_concepts.truncate(20);
        
        let overall_confidence = if completed_concepts.is_empty() {
            0.0
        } else {
            completed_concepts.iter().map(|c| c.confidence).sum::<f32>() / completed_concepts.len() as f32
        };
        
        let completion_method = if source_systems.len() > 2 {
            CompletionMethod::HybridCompletion
        } else if source_systems.contains(&"WorkingMemorySystem".to_string()) {
            CompletionMethod::WorkingMemoryRetrieval
        } else if source_systems.contains(&"NeuralBridgeFinder".to_string()) {
            CompletionMethod::NeuralBridgeDiscovery
        } else {
            CompletionMethod::SemanticSimilarity
        };
        
        let completed_pattern = CompletedPattern {
            partial_cues: partial_cues.to_vec(),
            completed_concepts,
            completion_confidence: overall_confidence,
            completion_method,
            processing_time_ms: start_time.elapsed().as_millis() as f32,
            source_systems,
        };
        
        // Cache result
        self.completion_cache.insert(partial_key, completed_pattern.clone());
        
        Ok(completed_pattern)
    }
    
    async fn encode_partial_pattern(&self, partial: &[(String, f32)]) -> Vec<f32> {
        // Convert to tensor format
        let input_tensor = self.pattern_to_tensor(partial);
        
        // Encode with pattern encoder
        self.pattern_encoder.encode(&input_tensor).await.unwrap_or_else(|_| {
            // Fallback to simple encoding
            vec![0.0; 128]
        })
    }
    
    async fn score_candidates_neural(
        &self,
        partial: &[(String, f32)],
        candidates: Vec<(PatternId, f32)>,
        partial_encoding: &[f32]
    ) -> Vec<(PatternId, f32)> {
        let mut scored = Vec::new();
        
        // Batch process candidates for efficiency
        let batch_size = 32;
        for chunk in candidates.chunks(batch_size) {
            let patterns: Vec<_> = chunk.iter()
                .filter_map(|(id, _)| self.patterns.get(id))
                .collect();
            
            if patterns.is_empty() {
                continue;
            }
            
            // Prepare batch input
            let batch_input = self.prepare_batch_scoring_input(
                partial,
                &patterns,
                partial_encoding
            );
            
            // Score with neural model
            let scores = self.pattern_encoder.score_batch(&batch_input).await.unwrap_or_else(|_| {
                vec![0.0; patterns.len()]
            });
            
            // Combine with similarity scores
            for ((pattern_id, sim_score), neural_score) in chunk.iter().zip(scores.iter()) {
                let combined_score = 0.7 * neural_score + 0.3 * sim_score;
                scored.push((*pattern_id, combined_score));
            }
        }
        
        scored.sort_by_key(|(_, score)| OrderedFloat(-score));
        scored
    }
    
    async fn generate_ai_completion(
        &self,
        partial: &[(String, f32)],
        pattern_id: &PatternId
    ) -> Option<CompletePattern> {
        let pattern = self.patterns.get(pattern_id)?;
        
        // Use completion model to fill in missing components
        let completion_input = self.prepare_completion_input(partial, &pattern);
        let completed_components = self.completion_model.complete(&completion_input).await.ok()?;
        
        Some(CompletePattern {
            pattern_id: *pattern_id,
            original_components: pattern.components.clone(),
            completed_components,
            confidence: self.calculate_completion_confidence(partial, &completed_components),
            method: CompletionMethod::Neural,
        })
    }
    
    pub async fn learn_pattern(&self, activated_concepts: &[(NodeId, f32)]) {
        // Check if pattern is significant enough
        if activated_concepts.len() < 3 || self.is_trivial_pattern(activated_concepts) {
            return;
        }
        
        // Encode pattern
        let pattern_embedding = self.encode_full_pattern(activated_concepts).await;
        
        // Check for novelty using similarity search
        let similar = self.similarity_index.search(&pattern_embedding, 5);
        let is_novel = similar.is_empty() || similar[0].1 < 0.9;
        
        if is_novel {
            let pattern_id = PatternId::new();
            let semantic_sig = self.compute_semantic_signature(activated_concepts).await;
            
            let pattern = MemoryPattern {
                id: pattern_id,
                components: activated_concepts.to_vec(),
                frequency: AtomicU32::new(1),
                last_accessed: AtomicInstant::now(),
                pattern_embedding: pattern_embedding.clone(),
                confidence_score: self.assess_pattern_quality(activated_concepts),
                semantic_signature: semantic_sig,
            };
            
            // Add to index
            self.similarity_index.add(pattern_id, &pattern_embedding);
            
            // Store pattern
            self.patterns.insert(pattern_id, pattern);
        } else if let Some((existing_id, _)) = similar.first() {
            // Increment frequency of existing pattern
            if let Some(mut existing) = self.patterns.get_mut(existing_id) {
                existing.frequency.fetch_add(1, Ordering::Relaxed);
                existing.last_accessed.store(Instant::now(), Ordering::Relaxed);
            }
        }
    }
}

// Lightweight pattern encoder model
pub struct PatternEncoder {
    // 10M parameter transformer optimized for patterns
}

// Completion network (small autoencoder)
pub struct CompletionNet {
    // 5M parameter model for pattern completion
}
```

### Task 10.2: Attention-Based Priming System (Uses Existing AttentionManager)
**File**: `src/associative/attention_priming_system.rs` (new file)
```rust
use crate::cognitive::attention_manager::{AttentionManager, AttentionType, AttentionTarget, AttentionTargetType};
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType};
use crate::core::types::EntityKey;
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Instant, Duration};
use dashmap::DashMap;

/// Priming system that coordinates with existing AttentionManager
/// DOES NOT replace attention - enhances it with priming effects
pub struct AttentionPrimingSystem {
    attention_manager: Arc<AttentionManager>,  // Uses existing system
    working_memory: Arc<WorkingMemorySystem>,  // Uses existing system
    active_primes: Arc<DashMap<EntityKey, PrimeState>>,
    priming_config: PrimingConfig,
}

#[derive(Debug, Clone)]
pub struct PrimeState {
    pub strength: f32,
    pub source: PrimeSource,
    pub timestamp: Instant,
    pub attention_amplification: f32,
    pub working_memory_trace: bool,
}

#[derive(Debug, Clone)]
pub enum PrimeSource {
    DirectActivation,      // Explicitly primed
    AssociativeSpread,     // Primed through associations
    WorkingMemoryTrace,    // Primed from working memory
    AttentionFocus,        // Primed by attention focus
    HebbianResonance,      // Primed by learned associations
}

#[derive(Debug, Clone)]
pub struct PrimingConfig {
    pub prime_decay_rate: f32,
    pub spread_factor: f32,
    pub attention_amplification_factor: f32,
    pub working_memory_boost: f32,
    pub max_prime_age: Duration,
    pub minimum_prime_strength: f32,
}

impl AttentionPrimingSystem {
    pub fn new(
        attention_manager: Arc<AttentionManager>,
        working_memory: Arc<WorkingMemorySystem>,
    ) -> Self {
        Self {
            attention_manager,
            working_memory,
            active_primes: Arc::new(DashMap::with_capacity(10_000)),
            priming_config: PrimingConfig {
                prime_decay_rate: 0.1,
                spread_factor: 0.7,
                attention_amplification_factor: 1.5,
                working_memory_boost: 1.3,
                max_prime_age: Duration::from_secs(300), // 5 minutes
                minimum_prime_strength: 0.05,
            },
        }
    }
    
    /// Apply priming by coordinating with AttentionManager
    pub async fn apply_prime_with_attention(
        &self,
        entity_key: EntityKey,
        strength: f32,
        source: PrimeSource,
    ) -> Result<PrimingResult> {
        let prime_state = PrimeState {
            strength,
            source: source.clone(),
            timestamp: Instant::now(),
            attention_amplification: 1.0,
            working_memory_trace: false,
        };
        
        // 1. Store prime state
        self.active_primes.insert(entity_key, prime_state.clone());
        
        // 2. COORDINATE with AttentionManager to focus on primed concept
        let attention_target = AttentionTarget {
            entity_key,
            attention_weight: strength * self.priming_config.attention_amplification_factor,
            priority: strength,
            duration: Duration::from_millis((strength * 10000.0) as u64),
            target_type: AttentionTargetType::Entity,
        };
        
        let attention_result = self.attention_manager.manage_divided_attention(
            vec![attention_target]
        ).await?;
        
        // 3. COORDINATE with WorkingMemorySystem to boost memory traces
        let memory_content = MemoryContent::Concept(format!("entity_{:?}", entity_key));
        let memory_storage = self.working_memory.store_in_working_memory_with_attention(
            memory_content,
            strength,
            BufferType::Episodic,
            strength * self.priming_config.working_memory_boost,
        ).await?;
        
        // 4. Spread priming to related concepts (using attention focus)
        let spread_results = self.spread_prime_through_attention(
            entity_key,
            strength * self.priming_config.spread_factor
        ).await?;
        
        Ok(PrimingResult {
            primed_entity: entity_key,
            prime_strength: strength,
            attention_result,
            memory_storage_success: memory_storage.success,
            spread_count: spread_results.len(),
            priming_source: source,
        })
    }
    
    /// Get current priming boost for an entity (decays over time)
    pub fn get_priming_boost(&self, entity_key: EntityKey) -> f32 {
        if let Some(prime) = self.active_primes.get(&entity_key) {
            let age = prime.timestamp.elapsed().as_secs_f32();
            let decay_factor = (-age * self.priming_config.prime_decay_rate).exp();
            let base_boost = prime.strength * decay_factor;
            
            // Apply attention amplification if still active
            base_boost * prime.attention_amplification
        } else {
            0.0
        }
    }
    
    /// Spread priming through attention network
    async fn spread_prime_through_attention(
        &self,
        source_entity: EntityKey,
        spread_strength: f32,
    ) -> Result<Vec<EntityKey>> {
        let mut spread_entities = Vec::new();
        
        if spread_strength > self.priming_config.minimum_prime_strength {
            // Use attention system to find related entities that should be primed
            let attention_state = self.attention_manager.get_attention_state().await?;
            
            // Prime entities that are currently in attention or recently attended
            for &target_entity in &attention_state.current_targets {
                if target_entity != source_entity {
                    let spread_prime = PrimeState {
                        strength: spread_strength,
                        source: PrimeSource::AssociativeSpread,
                        timestamp: Instant::now(),
                        attention_amplification: 1.2,
                        working_memory_trace: true,
                    };
                    
                    self.active_primes.insert(target_entity, spread_prime);
                    spread_entities.push(target_entity);
                }
            }
        }
        
        Ok(spread_entities)
    }
    
    /// Clean up expired primes
    pub async fn cleanup_expired_primes(&self) -> usize {
        let mut removed_count = 0;
        let current_time = Instant::now();
        
        self.active_primes.retain(|_key, prime| {
            let is_expired = current_time.duration_since(prime.timestamp) > self.priming_config.max_prime_age
                || self.get_current_strength(prime, current_time) < self.priming_config.minimum_prime_strength;
            
            if is_expired {
                removed_count += 1;
            }
            
            !is_expired
        });
        
        removed_count
    }
    
    fn get_current_strength(&self, prime: &PrimeState, current_time: Instant) -> f32 {
        let age = current_time.duration_since(prime.timestamp).as_secs_f32();
        prime.strength * (-age * self.priming_config.prime_decay_rate).exp()
    }
}

#[derive(Debug, Clone)]
pub struct PrimingResult {
    pub primed_entity: EntityKey,
    pub prime_strength: f32,
    pub attention_result: crate::cognitive::attention_manager::AttentionResult,
    pub memory_storage_success: bool,
    pub spread_count: usize,
    pub priming_source: PrimeSource,
}
```

### Task 10.3: Multi-System Tip-of-the-Tongue Retrieval
**File**: `src/associative/multi_system_tot_retrieval.rs` (new file)
```rust
use crate::cognitive::working_memory::{WorkingMemorySystem, MemoryQuery, BufferType};
use crate::cognitive::neural_bridge_finder::NeuralBridgeFinder;
use crate::models::{ModelType, rust_embeddings::RustEmbeddingModel};
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::federation::FederationManager;
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;

/// Tip-of-the-tongue retrieval using coordination of multiple cognitive systems
pub struct MultiSystemTotRetrieval {
    // Existing systems (DO NOT recreate)
    working_memory: Arc<WorkingMemorySystem>,
    bridge_finder: Arc<NeuralBridgeFinder>,
    embedding_model: Arc<RustEmbeddingModel>,
    brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    federation_manager: Arc<FederationManager>,
    
    // NEW: TOT-specific coordination
    retrieval_strategies: Vec<RetrievalStrategy>,
}

#[derive(Debug, Clone)]
pub struct PartialMemoryHints {
    pub sounds_like: Option<String>,
    pub starts_with: Option<String>,
    pub syllable_pattern: Option<String>,
    pub semantic_features: Vec<String>,
    pub associated_concepts: Vec<String>,
    pub context_clues: Vec<String>,
    pub emotional_valence: Option<f32>,
    pub approximate_length: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RetrievalCandidate {
    pub concept: String,
    pub confidence: f32,
    pub retrieval_method: RetrievalMethod,
    pub supporting_evidence: Vec<String>,
    pub phonetic_similarity: Option<f32>,
    pub semantic_similarity: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum RetrievalMethod {
    WorkingMemoryTrace,
    SemanticSimilarity,
    PhoneticMatching,
    BridgePathDiscovery,
    AssociativeChaining,
    FederatedSearch,
    HybridRetrieval,
}

#[derive(Debug, Clone)]
pub enum RetrievalStrategy {
    PhoneticFirst,
    SemanticFirst,
    AssociativeFirst,
    BreadthFirst,
    ConfidenceThreshold,
}

impl MultiSystemTotRetrieval {
    pub async fn new(
        working_memory: Arc<WorkingMemorySystem>,
        bridge_finder: Arc<NeuralBridgeFinder>,
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        federation_manager: Arc<FederationManager>,
    ) -> Result<Self> {
        // Load existing embedding model
        let embedding_model = Arc::new(RustEmbeddingModel::load_model(
            ModelType::MiniLM,
            "./src/models/pretrained/all_minilm_l6_v2"
        )?);
        
        Ok(Self {
            working_memory,
            bridge_finder,
            embedding_model,
            brain_graph,
            federation_manager,
            retrieval_strategies: vec![
                RetrievalStrategy::SemanticFirst,
                RetrievalStrategy::AssociativeFirst,
                RetrievalStrategy::PhoneticFirst,
                RetrievalStrategy::BreadthFirst,
            ],
        })
    }
    
    /// Multi-system tip-of-the-tongue retrieval using ALL cognitive systems
    pub async fn retrieve_from_partial_hints(
        &self,
        hints: &PartialMemoryHints,
    ) -> Result<Vec<RetrievalCandidate>> {
        let mut all_candidates = Vec::new();
        
        // 1. COORDINATE with WorkingMemorySystem for traces of similar concepts
        for context_clue in &hints.context_clues {
            let memory_query = MemoryQuery {
                query_text: context_clue.clone(),
                search_buffers: vec![BufferType::Episodic, BufferType::Phonological],
                apply_attention: true,
                importance_threshold: 0.2,
                recency_weight: 0.6,
            };
            
            let memory_results = self.working_memory.retrieve_from_working_memory(&memory_query).await?;
            for memory_item in memory_results.items {
                if let crate::cognitive::working_memory::MemoryContent::Concept(concept) = memory_item.content {
                    // Check if concept matches any of the partial hints
                    let candidate_score = self.evaluate_candidate_against_hints(&concept, hints);
                    if candidate_score > 0.3 {
                        all_candidates.push(RetrievalCandidate {
                            concept: concept.clone(),
                            confidence: candidate_score,
                            retrieval_method: RetrievalMethod::WorkingMemoryTrace,
                            supporting_evidence: vec![format!("Found in {} buffer", 
                                match memory_query.search_buffers[0] {
                                    BufferType::Episodic => "episodic",
                                    BufferType::Phonological => "phonological",
                                    BufferType::Visuospatial => "visuospatial",
                                }
                            )],
                            phonetic_similarity: None,
                            semantic_similarity: Some(memory_item.importance_score),
                        });
                    }
                }
            }
        }
        
        // 2. COORDINATE with NeuralBridgeFinder for associative retrieval
        for associated_concept in &hints.associated_concepts {
            // Find bridge paths that might lead to the target concept
            let bridge_paths = self.bridge_finder.find_creative_bridges(
                associated_concept,
                "*" // Find bridges to any concept
            ).await?;
            
            for bridge_path in bridge_paths {
                for intermediate_concept in &bridge_path.intermediate_concepts {
                    let candidate_score = self.evaluate_candidate_against_hints(intermediate_concept, hints);
                    if candidate_score > 0.4 {
                        all_candidates.push(RetrievalCandidate {
                            concept: intermediate_concept.clone(),
                            confidence: candidate_score * bridge_path.connection_strength,
                            retrieval_method: RetrievalMethod::BridgePathDiscovery,
                            supporting_evidence: vec![
                                format!("Bridge path: {}", bridge_path.explanation),
                                format!("Creativity score: {:.2}", bridge_path.creativity_score)
                            ],
                            phonetic_similarity: None,
                            semantic_similarity: Some(bridge_path.connection_strength),
                        });
                    }
                }
            }
        }
        
        // 3. COORDINATE with embedding model for semantic similarity
        if !hints.semantic_features.is_empty() {
            let combined_features = hints.semantic_features.join(" ");
            let feature_embedding = self.embedding_model.encode_text(&combined_features).await?;
            
            // Find semantically similar concepts in knowledge base
            let similar_concepts = self.find_semantically_similar_concepts_from_graph(
                &feature_embedding,
                20,  // max results
                0.3  // similarity threshold
            ).await?;
            
            for (concept, similarity) in similar_concepts {
                let candidate_score = self.evaluate_candidate_against_hints(&concept, hints);
                if candidate_score > 0.3 {
                    all_candidates.push(RetrievalCandidate {
                        concept: concept.clone(),
                        confidence: candidate_score * similarity,
                        retrieval_method: RetrievalMethod::SemanticSimilarity,
                        supporting_evidence: vec![
                            format!("Semantic similarity: {:.2}", similarity),
                            format!("Matched features: {}", combined_features)
                        ],
                        phonetic_similarity: None,
                        semantic_similarity: Some(similarity),
                    });
                }
            }
        }
        
        // 4. COORDINATE with FederationManager for cross-database search
        if !hints.associated_concepts.is_empty() {
            let search_query = hints.associated_concepts.join(" OR ");
            let federated_results = self.federation_manager.execute_similarity_search(
                &search_query,
                0.3,  // threshold
                30    // max results
            ).await?;
            
            for fed_result in federated_results {
                if let Some(concept) = fed_result.entity_data.get("concept") {
                    let candidate_score = self.evaluate_candidate_against_hints(concept, hints);
                    if candidate_score > 0.3 {
                        all_candidates.push(RetrievalCandidate {
                            concept: concept.clone(),
                            confidence: candidate_score * fed_result.similarity_score,
                            retrieval_method: RetrievalMethod::FederatedSearch,
                            supporting_evidence: vec![
                                format!("Found in database: {}", fed_result.database_id),
                                format!("Federated similarity: {:.2}", fed_result.similarity_score)
                            ],
                            phonetic_similarity: None,
                            semantic_similarity: Some(fed_result.similarity_score),
                        });
                    }
                }
            }
        }
        
        // 5. Apply phonetic matching if hints provided
        if let Some(sounds_like) = &hints.sounds_like {
            let phonetic_candidates = self.find_phonetically_similar_concepts(sounds_like).await?;
            for (concept, phonetic_score) in phonetic_candidates {
                let candidate_score = self.evaluate_candidate_against_hints(&concept, hints);
                if candidate_score > 0.4 {
                    all_candidates.push(RetrievalCandidate {
                        concept: concept.clone(),
                        confidence: candidate_score * phonetic_score,
                        retrieval_method: RetrievalMethod::PhoneticMatching,
                        supporting_evidence: vec![
                            format!("Phonetic similarity to: {}", sounds_like),
                            format!("Phonetic score: {:.2}", phonetic_score)
                        ],
                        phonetic_similarity: Some(phonetic_score),
                        semantic_similarity: None,
                    });
                }
            }
        }
        
        // Deduplicate and sort by confidence
        let mut unique_candidates = HashMap::new();
        for candidate in all_candidates {
            unique_candidates.entry(candidate.concept.clone())
                .and_modify(|existing: &mut RetrievalCandidate| {
                    if candidate.confidence > existing.confidence {
                        *existing = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }
        
        let mut final_candidates: Vec<_> = unique_candidates.into_values().collect();
        final_candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        final_candidates.truncate(10);
        
        Ok(final_candidates)
    }
    
    /// Evaluate how well a candidate concept matches the provided hints
    fn evaluate_candidate_against_hints(&self, concept: &str, hints: &PartialMemoryHints) -> f32 {
        let mut score = 0.0;
        let mut total_weight = 0.0;
        
        // Check prefix match
        if let Some(starts_with) = &hints.starts_with {
            total_weight += 0.3;
            if concept.to_lowercase().starts_with(&starts_with.to_lowercase()) {
                score += 0.3;
            }
        }
        
        // Check approximate length
        if let Some(expected_length) = hints.approximate_length {
            total_weight += 0.2;
            let length_diff = (concept.len() as f32 - expected_length as f32).abs();
            let length_score = (1.0 - length_diff / expected_length.max(1) as f32).max(0.0);
            score += 0.2 * length_score;
        }
        
        // Check syllable pattern (simple approximation)
        if let Some(pattern) = &hints.syllable_pattern {
            total_weight += 0.2;
            let estimated_syllables = concept.chars().filter(|c| "aeiouAEIOU".contains(*c)).count();
            let pattern_syllables = pattern.chars().filter(|c| "aeiouAEIOU".contains(*c)).count();
            if estimated_syllables == pattern_syllables {
                score += 0.2;
            }
        }
        
        // Default weight for semantic features presence
        total_weight += 0.3;
        for feature in &hints.semantic_features {
            if concept.to_lowercase().contains(&feature.to_lowercase()) {
                score += 0.3 / hints.semantic_features.len() as f32;
            }
        }
        
        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }
```

## Week 11: Associative Learning

### Task 11.1: Associative Learning Integration
**File**: `src/associative/learning_integration.rs` (new file)
```rust
use crate::learning::hebbian::HebbianLearningEngine;
use crate::cognitive::inhibitory::CompetitiveInhibitionSystem;
use crate::learning::types::{ActivationEvent, LearningContext, LearningUpdate};

pub struct AssociativeLearningIntegrator {
    hebbian_engine: Arc<HebbianLearningEngine>,
    inhibition_system: Arc<CompetitiveInhibitionSystem>,
    learning_config: AssociativeLearningConfig,
}

impl AssociativeLearningIntegrator {
    pub fn new(
        hebbian_engine: Arc<HebbianLearningEngine>,
        inhibition_system: Arc<CompetitiveInhibitionSystem>,
    ) -> Self {
        Self {
            hebbian_engine,
            inhibition_system,
            learning_config: AssociativeLearningConfig::default(),
        }
    }
    
    pub async fn learn_association(&mut self, 
        network: &mut ActivationNetwork,
        concepts: &[(String, f32)],
        time_window: Duration
    ) -> Result<LearningUpdate> {
        // Create activation events from concepts
        let activation_events: Vec<ActivationEvent> = concepts.iter()
            .map(|(concept, strength)| ActivationEvent {
                entity_key: network.concept_to_key(concept),
                activation_strength: *strength,
                timestamp: Instant::now(),
            }).collect();
        
        // Apply Hebbian learning using existing engine
        let learning_result = self.hebbian_engine.apply_hebbian_learning(
            activation_events,
            LearningContext::AssociativeMemory {
                time_window,
                co_activation_threshold: self.learning_config.co_activation_threshold,
            }
        ).await?;
        
        // Update competitive inhibition based on learned associations
        let inhibition_updates = self.inhibition_system.update_from_learning_results(
            &learning_result
        ).await?;
        
        Ok(LearningUpdate {
            strengthened_connections: learning_result.strengthened_connections,
            weakened_connections: learning_result.weakened_connections,
            new_connections: learning_result.new_connections,
            pruned_connections: learning_result.pruned_connections,
            learning_efficiency: learning_result.learning_efficiency,
            inhibition_updates,
        })
    }
    
    pub async fn consolidate_associations(&mut self, network: &mut ActivationNetwork) -> Result<ConsolidationResult> {
        // Use existing Hebbian engine consolidation
        let consolidation_result = self.hebbian_engine.consolidate_frequent_patterns().await?;
        
        // Apply competitive dynamics to consolidation
        let competitive_consolidation = self.inhibition_system.apply_competitive_consolidation(
            &consolidation_result
        ).await?;
        
        Ok(ConsolidationResult {
            consolidated_patterns: consolidation_result.consolidated_patterns,
            strengthened_paths: consolidation_result.strengthened_paths,
            pruned_weak_connections: consolidation_result.pruned_weak_connections,
            competitive_adjustments: competitive_consolidation,
        })
    }
}
```

### Task 11.2: Inhibitory Integration for Associative Memory
**File**: `src/associative/inhibition_integration.rs` (new file)
```rust
use crate::cognitive::inhibitory::{CompetitiveInhibitionSystem, CompetitionGroup, CompetitionType};
use crate::core::brain_types::ActivationPattern;

pub struct AssociativeInhibitionManager {
    inhibition_system: Arc<CompetitiveInhibitionSystem>,
    associative_groups: Arc<RwLock<Vec<AssociativeCompetitionGroup>>>,
}

pub struct AssociativeCompetitionGroup {
    base_group: CompetitionGroup,
    associative_strength: f32,
    temporal_window: Duration,
    memory_based_inhibition: bool,
}

impl AssociativeInhibitionManager {
    pub fn new(inhibition_system: Arc<CompetitiveInhibitionSystem>) -> Self {
        Self {
            inhibition_system,
            associative_groups: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn apply_associative_inhibition(
        &self, 
        activation_pattern: &mut ActivationPattern,
        context: AssociativeContext
    ) -> Result<InhibitionResult> {
        // Apply existing competitive inhibition
        let base_inhibition = self.inhibition_system.apply_inhibition(activation_pattern).await?;
        
        // Apply associative-specific inhibition
        let associative_inhibition = self.apply_associative_specific_inhibition(
            activation_pattern,
            &context
        ).await?;
        
        // Apply temporal inhibition based on recent activations
        let temporal_inhibition = self.apply_temporal_inhibition(
            activation_pattern,
            context.recent_activations
        ).await?;
        
        Ok(InhibitionResult {
            base_inhibition,
            associative_inhibition,
            temporal_inhibition,
            final_pattern: activation_pattern.clone(),
        })
    }
    
    async fn apply_associative_specific_inhibition(
        &self,
        pattern: &mut ActivationPattern,
        context: &AssociativeContext
    ) -> Result<Vec<InhibitionEvent>> {
        let mut inhibition_events = Vec::new();
        let groups = self.associative_groups.read().await;
        
        for group in groups.iter() {
            // Apply memory-based inhibition
            if group.memory_based_inhibition {
                let memory_inhibition = self.apply_memory_based_inhibition(
                    pattern,
                    group,
                    &context.working_memory_state
                ).await?;
                inhibition_events.extend(memory_inhibition);
            }
            
            // Apply strength-weighted competition
            let strength_inhibition = self.apply_associative_strength_inhibition(
                pattern,
                group
            ).await?;
            inhibition_events.extend(strength_inhibition);
        }
        
        Ok(inhibition_events)
    }
    
    pub async fn create_associative_competition_group(
        &self,
        competing_concepts: Vec<EntityKey>,
        competition_type: CompetitionType,
        associative_strength: f32
    ) -> Result<()> {
        let base_group = CompetitionGroup {
            group_id: format!("assoc_{}", uuid::Uuid::new_v4()),
            competing_entities: competing_concepts,
            competition_type,
            winner_takes_all: associative_strength > 0.8,
            inhibition_strength: associative_strength,
            priority: 1.0,
        };
        
        let associative_group = AssociativeCompetitionGroup {
            base_group,
            associative_strength,
            temporal_window: Duration::from_millis(500),
            memory_based_inhibition: true,
        };
        
        self.associative_groups.write().await.push(associative_group);
        
        // Register with main inhibition system
        self.inhibition_system.add_competition_group(associative_group.base_group.clone()).await?;
        
        Ok(())
    }
}
```

## Week 12: Integration and Advanced Features

### Task 12.1: Memory Consolidation Engine
**File**: `src/associative/consolidation_engine.rs` (new file)
```rust
pub struct ConsolidationEngine {
    replay_buffer: VecDeque<ActivationTrace>,
    consolidation_cycles: usize,
}

impl ConsolidationEngine {
    pub fn run_consolidation_cycle(&mut self, network: &mut ActivationNetwork) {
        // Replay important activation patterns
        let important_traces = self.select_important_traces();
        
        for trace in important_traces {
            // Reactivate pattern with reduced strength
            network.replay_activation_trace(&trace, REPLAY_STRENGTH);
            
            // Strengthen connections in pattern
            self.strengthen_trace_connections(&trace, network);
        }
        
        // Prune weak connections
        self.prune_weak_connections(network);
    }
}
```

### Task 12.2: MCP Handlers for Associative Memory
**File**: `src/mcp/llm_friendly_server/handlers/associative.rs` (new file)
```rust
use crate::associative::activation_network::ActivationNetwork;
use crate::associative::pattern_completion::PatternCompletion;
use crate::associative::learning_integration::AssociativeLearningIntegrator;
use crate::federation::FederationManager;
use serde_json::{json, Value};

/// MCP handler for spreading activation across associative memory network
pub async fn handle_activate_concept(params: Value) -> Result<Value> {
    let concept = params["concept"].as_str().unwrap();
    let strength = params["strength"].as_f64().unwrap_or(1.0) as f32;
    let include_path = params["include_activation_path"].as_bool().unwrap_or(false);
    let cross_shard = params["cross_shard"].as_bool().unwrap_or(true);
    
    // Get activation network from global state
    let activation_network = get_activation_network().await?;
    
    // Perform activation spreading
    let result = activation_network.activate(concept, strength).await?;
    
    // If cross-shard requested, federate across database shards
    let federated_results = if cross_shard {
        let federation_manager = get_federation_manager().await?;
        federation_manager.execute_cross_shard_activation(concept, strength).await?
    } else {
        Vec::new()
    };
    
    Ok(json!({
        "concept": concept,
        "activated_concepts": result.activated_concepts,
        "total_activated": result.total_spread,
        "max_depth": result.max_depth_reached,
        "processing_time_ms": result.processing_time_ms,
        "attention_weight": result.attention_weight,
        "inhibition_applied": result.inhibition_applied,
        "learning_updates": result.learning_updates,
        "cross_shard_results": federated_results,
        "activation_path": if include_path { Some(result.activation_path) } else { None }
    }))
}

/// MCP handler for finding associative connections between concepts
pub async fn handle_find_associations(params: Value) -> Result<Value> {
    let concepts = params["concepts"].as_array().unwrap();
    let max_distance = params["max_distance"].as_u64().unwrap_or(3) as usize;
    let include_bridge_paths = params["include_bridge_paths"].as_bool().unwrap_or(false);
    let use_attention_weighting = params["use_attention_weighting"].as_bool().unwrap_or(true);
    
    let activation_network = get_activation_network().await?;
    let bridge_finder = get_bridge_finder().await?;
    
    // Find direct associations
    let direct_associations = activation_network.find_concept_associations(
        concepts, 
        max_distance,
        use_attention_weighting
    ).await?;
    
    // Find creative bridge paths if requested
    let bridge_paths = if include_bridge_paths && concepts.len() >= 2 {
        let concept_pairs = create_concept_pairs(concepts);
        let mut all_bridges = Vec::new();
        
        for (concept_a, concept_b) in concept_pairs {
            let bridges = bridge_finder.find_creative_bridges(
                concept_a.as_str().unwrap(),
                concept_b.as_str().unwrap()
            ).await?;
            all_bridges.extend(bridges);
        }
        all_bridges
    } else {
        Vec::new()
    };
    
    Ok(json!({
        "query_concepts": concepts,
        "direct_associations": direct_associations,
        "bridge_paths": bridge_paths,
        "connection_strength": calculate_association_strength(&direct_associations),
        "attention_weighted": use_attention_weighting
    }))
}

/// MCP handler for pattern completion in associative memory
pub async fn handle_complete_pattern(params: Value) -> Result<Value> {
    let partial_pattern = params["partial_pattern"].as_array().unwrap();
    let completion_confidence = params["completion_confidence"].as_f64().unwrap_or(0.7) as f32;
    let max_completions = params["max_completions"].as_u64().unwrap_or(5) as usize;
    
    let pattern_completion = get_pattern_completion().await?;
    
    // Convert JSON to internal format
    let partial: Vec<(String, f32)> = partial_pattern.iter()
        .filter_map(|item| {
            if let (Some(concept), Some(strength)) = (
                item["concept"].as_str(),
                item["strength"].as_f64()
            ) {
                Some((concept.to_string(), strength as f32))
            } else {
                None
            }
        })
        .collect();
    
    // Attempt pattern completion
    let completions = pattern_completion.complete_multiple_patterns(
        &partial,
        max_completions,
        completion_confidence
    ).await?;
    
    Ok(json!({
        "partial_pattern": partial_pattern,
        "completions": completions,
        "completion_method": "neural_associative",
        "confidence_threshold": completion_confidence
    }))
}

/// MCP handler for learning new associations
pub async fn handle_learn_association(params: Value) -> Result<Value> {
    let concepts = params["concepts"].as_array().unwrap();
    let strengths = params["strengths"].as_array();
    let learning_context = params["context"].as_str().unwrap_or("general");
    let time_window_ms = params["time_window_ms"].as_u64().unwrap_or(1000);
    
    let learning_integrator = get_learning_integrator().await?;
    let activation_network = get_activation_network().await?;
    
    // Prepare concept-strength pairs
    let concept_strengths: Vec<(String, f32)> = if let Some(strengths) = strengths {
        concepts.iter().zip(strengths.iter())
            .filter_map(|(c, s)| {
                Some((c.as_str()?.to_string(), s.as_f64()? as f32))
            })
            .collect()
    } else {
        concepts.iter()
            .filter_map(|c| Some((c.as_str()?.to_string(), 1.0)))
            .collect()
    };
    
    // Apply associative learning
    let learning_result = learning_integrator.learn_association(
        &mut *activation_network.write().await,
        &concept_strengths,
        Duration::from_millis(time_window_ms)
    ).await?;
    
    Ok(json!({
        "learned_concepts": concepts,
        "learning_context": learning_context,
        "strengthened_connections": learning_result.strengthened_connections.len(),
        "new_connections": learning_result.new_connections.len(),
        "learning_efficiency": learning_result.learning_efficiency,
        "time_window_ms": time_window_ms
    }))
}

/// MCP handler for cross-shard activation spreading
pub async fn handle_cross_shard_activation(params: Value) -> Result<Value> {
    let concept = params["concept"].as_str().unwrap();
    let strength = params["strength"].as_f64().unwrap_or(1.0) as f32;
    let target_shards = params["target_shards"].as_array();
    let merge_strategy = params["merge_strategy"].as_str().unwrap_or("weighted_average");
    
    let federation_manager = get_federation_manager().await?;
    
    // Execute federated activation
    let federated_query = create_federated_activation_query(
        concept, 
        strength, 
        target_shards, 
        merge_strategy
    );
    
    let federation_result = federation_manager.execute_federated_query(federated_query).await?;
    
    Ok(json!({
        "concept": concept,
        "cross_shard_results": federation_result.shard_results,
        "merged_activations": federation_result.merged_results,
        "total_shards_queried": federation_result.shards_queried,
        "merge_strategy": merge_strategy,
        "federation_time_ms": federation_result.processing_time_ms
    }))
}
```

### Task 12.3: Performance Optimization
```rust
// Optimizations:
1. Parallel activation spreading
2. Sparse matrix representation for edges
3. Activation caching for common queries
4. Batch updates for learning
5. GPU acceleration for large networks
```

### Task 12.4: Testing Suite
**File**: `tests/associative_memory_tests.rs`
```rust
#[test]
fn test_spreading_activation() {
    let mut network = create_test_network();
    network.add_edge("dog", "cat", 0.8);
    network.add_edge("cat", "mouse", 0.7);
    
    let result = network.activate("dog", 1.0);
    
    assert!(result.contains_concept("cat"));
    assert!(result.contains_concept("mouse"));
    assert!(result.get_activation("cat") > result.get_activation("mouse"));
}

#[test]
fn test_context_modulation() {
    // Test that context affects activation spreading
}

#[test]
fn test_pattern_completion() {
    // Test partial pattern completion
}
```

## Deliverables
1. **AI-enhanced spreading activation** with semantic similarity and cognitive system integration
2. **Neural pattern completion** with autoencoder model and working memory integration
3. **SIMD-optimized similarity** calculations with existing performance monitoring
4. **Parallel activation spreading** for i9 optimization with federation support
5. **AttentionManager integration** for context weighting and focus control
6. **CompetitiveInhibitionSystem integration** for winner-takes-all dynamics
7. **HebbianLearningEngine integration** for connection strengthening
8. **NeuralBridgeFinder integration** for creative associations
9. **FederationManager integration** for cross-shard activation
10. **MCP handlers** for associative memory operations
11. **Pre-computed embeddings** integration from existing models

## Success Criteria
- [ ] Activation spreading: <2ms for 1000 nodes on i9 with cognitive integration
- [ ] Pattern completion accuracy > 90% with neural model and working memory
- [ ] Semantic similarity boost: 30% improvement in relevance using bridge finder
- [ ] Parallel spreading: 8x speedup with i9 cores and federation support
- [ ] Memory pattern learning: <5ms per pattern with Hebbian integration
- [ ] SIMD similarity: <0.1ms per comparison with existing optimizations
- [ ] Total memory < 2GB for 1M nodes across cognitive systems
- [ ] Attention weighting: 95% accuracy in focus-based activation
- [ ] Competitive inhibition: Winner-takes-all in <1ms
- [ ] Cross-shard activation: <10ms for distributed spreading
- [ ] MCP handler response time: <50ms for associative operations

## Performance Benchmarks (Intel i9)
- Single activation spread: 0.5-2ms
- Pattern encoding: 3ms
- Pattern completion: 5-8ms
- Similarity computation (SIMD): 0.05ms
- Parallel spreading (8 threads): 8x speedup
- Hebbian learning update: 0.1ms
- Priming effect application: 1ms

## Dependencies
**Existing Cognitive Systems**:
- CognitiveOrchestrator - Pattern selection and coordination
- WorkingMemorySystem - Memory buffers and decay mechanisms
- AttentionManager - Focus control and attention weighting
- CompetitiveInhibitionSystem - Winner-takes-all and lateral inhibition
- HebbianLearningEngine - Connection strengthening and coactivation tracking
- NeuralBridgeFinder - Creative pathfinding between concepts
- GraphStructurePredictor - Missing link prediction
- FederationManager - Cross-shard coordination and activation

**AI/ML Dependencies**:
- Candle framework (Rust native ML)
- Models from src/models (all ported to Rust):
  - all-MiniLM-L6-v2 (22M params) - For semantic similarity
  - Pre-computed embeddings from Phase 2
- SIMD intrinsics for x86_64

**Infrastructure**:
- DashMap for concurrent access
- Rust-native graph operations
- MCP server framework for tool handlers
- Federation infrastructure for multi-database operations

## Risks & Mitigations
1. **Parallel activation conflicts**
   - Mitigation: Lock-free data structures, atomic operations
2. **Pattern encoder latency**
   - Mitigation: Batch processing, caching
3. **Memory fragmentation**
   - Mitigation: Pre-allocation, memory pools