use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::cognitive::types::CognitivePatternType;
use crate::cognitive::working_memory::WorkingMemorySystem;
use crate::core::activation_engine::ActivationPropagationEngine;
use crate::core::types::EntityKey;
use crate::error::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::SystemTime;
use ahash::AHashMap;

#[derive(Clone)]
pub struct AttentionManager {
    pub orchestrator: Arc<CognitiveOrchestrator>,
    pub activation_engine: Arc<ActivationPropagationEngine>,
    pub working_memory: Arc<WorkingMemorySystem>,
    pub attention_state: Arc<RwLock<AttentionState>>,
    pub focus_history: Arc<RwLock<VecDeque<AttentionFocus>>>,
    pub attention_config: AttentionConfig,
}

impl std::fmt::Debug for AttentionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionManager")
            .field("orchestrator", &"CognitiveOrchestrator")
            .field("activation_engine", &"ActivationPropagationEngine")
            .field("working_memory", &"WorkingMemorySystem")
            .field("attention_state", &"Arc<RwLock<AttentionState>>")
            .field("focus_history", &"Arc<RwLock<VecDeque<AttentionFocus>>>")
            .field("attention_config", &self.attention_config)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct AttentionState {
    pub current_focus: AttentionFocus,
    pub attention_capacity: f32,
    pub divided_attention_targets: Vec<AttentionTarget>,
    pub inhibition_strength: f32,
    pub cognitive_load: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionFocus {
    pub target_entities: Vec<EntityKey>,
    pub attention_weights: AHashMap<EntityKey, f32>,
    pub focus_strength: f32,
    pub timestamp: SystemTime,
    pub focus_type: AttentionType,
    pub focus_id: String,
}

#[derive(Debug, Clone)]
pub struct AttentionTarget {
    pub entity_key: EntityKey,
    pub attention_weight: f32,
    pub priority: f32,
    pub duration: std::time::Duration,
    pub target_type: AttentionTargetType,
}

#[derive(Debug, Clone)]
pub enum AttentionType {
    Selective,    // Focus on specific entities
    Divided,      // Split attention across multiple targets
    Sustained,    // Maintain focus over time
    Executive,    // Control and coordinate other attention
    Alternating,  // Switch between targets
}

#[derive(Debug, Clone)]
pub enum AttentionTargetType {
    Entity,
    Concept,
    Relationship,
    Pattern,
    Memory,
}

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub max_attention_targets: usize,
    pub attention_decay_rate: f32,
    pub focus_switch_threshold: f32,
    pub divided_attention_penalty: f32,
    pub executive_control_strength: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionResult {
    pub focused_entities: Vec<EntityKey>,
    pub attention_strength: f32,
    pub working_memory_updates: Vec<String>,
    pub cognitive_load_change: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionShiftResult {
    pub shift_duration: std::time::Duration,
    pub attention_continuity: f32,
    pub working_memory_impact: f32,
    pub shift_success: bool,
}

#[derive(Debug, Clone)]
pub struct AttentionFadeResult {
    pub duration: std::time::Duration,
    pub memory_impact: f32,
    pub final_attention_levels: AHashMap<EntityKey, f32>,
}

#[derive(Debug, Clone)]
pub struct CoordinatedAttentionResult {
    pub attention_config: AttentionConfiguration,
    pub attention_result: AttentionResult,
    pub pattern_compatibility: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionConfiguration {
    pub target_entities: Vec<EntityKey>,
    pub focus_strength: f32,
    pub attention_type: AttentionType,
    pub expected_duration: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct AttentionRequirements {
    pub required_focus_strength: f32,
    pub preferred_attention_type: AttentionType,
    pub target_entity_types: Vec<AttentionTargetType>,
    pub sustained_attention_needed: bool,
}

#[derive(Debug, Clone)]
pub struct ActivationModulation {
    pub focused_entities: Vec<EntityKey>,
    pub memory_updates: Vec<String>,
    pub inhibition_changes: AHashMap<EntityKey, f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            max_attention_targets: 7,
            attention_decay_rate: 0.1,
            focus_switch_threshold: 0.3,
            divided_attention_penalty: 0.2,
            executive_control_strength: 0.8,
        }
    }
}

impl AttentionState {
    pub fn new() -> Self {
        Self {
            current_focus: AttentionFocus::empty(),
            attention_capacity: 1.0,
            divided_attention_targets: Vec::new(),
            inhibition_strength: 0.5,
            cognitive_load: 0.0,
        }
    }

    pub fn update_cognitive_load(&mut self, new_load: f32) {
        self.cognitive_load = new_load.clamp(0.0, 1.0);
        
        // Adjust attention capacity based on cognitive load
        self.attention_capacity = (1.0 - self.cognitive_load * 0.5).max(0.2);
    }

    pub fn can_add_target(&self, max_targets: usize) -> bool {
        self.divided_attention_targets.len() < max_targets
    }
}

impl AttentionFocus {
    pub fn empty() -> Self {
        Self {
            target_entities: Vec::new(),
            attention_weights: AHashMap::new(),
            focus_strength: 0.0,
            timestamp: SystemTime::now(),
            focus_type: AttentionType::Selective,
            focus_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    pub fn new(
        targets: Vec<EntityKey>,
        weights: AHashMap<EntityKey, f32>,
        strength: f32,
        focus_type: AttentionType,
    ) -> Self {
        Self {
            target_entities: targets,
            attention_weights: weights,
            focus_strength: strength,
            timestamp: SystemTime::now(),
            focus_type,
            focus_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    pub fn is_focused_on(&self, entity: &EntityKey) -> bool {
        self.attention_weights.get(entity).map_or(false, |&weight| weight > 0.1)
    }

    pub fn get_attention_weight(&self, entity: &EntityKey) -> f32 {
        self.attention_weights.get(entity).copied().unwrap_or(0.0)
    }
}

impl AttentionManager {
    pub async fn new(
        orchestrator: Arc<CognitiveOrchestrator>,
        activation_engine: Arc<ActivationPropagationEngine>,
        working_memory: Arc<WorkingMemorySystem>,
    ) -> Result<Self> {
        Ok(Self {
            orchestrator,
            activation_engine,
            working_memory,
            attention_state: Arc::new(RwLock::new(AttentionState::new())),
            focus_history: Arc::new(RwLock::new(VecDeque::new())),
            attention_config: AttentionConfig::default(),
        })
    }

    pub async fn focus_attention(
        &self,
        targets: Vec<EntityKey>,
        focus_strength: f32,
        attention_type: AttentionType,
    ) -> Result<AttentionResult> {
        let mut attention_state = self.attention_state.write().await;
        
        // 1. Check attention capacity
        if targets.len() > self.attention_config.max_attention_targets {
            return Err(crate::error::GraphError::InvalidInput(
                "Too many attention targets".to_string(),
            ));
        }
        
        // 2. Calculate attention weights for targets
        let attention_weights = self.calculate_attention_weights(&targets, focus_strength, &attention_type).await?;
        
        // 3. Apply attention to activation engine
        let activation_modulation = self.apply_attention_to_activation(
            &attention_weights,
            &attention_state.current_focus,
        ).await?;
        
        // 4. Update working memory with focused concepts
        self.update_working_memory_with_focus(&attention_weights).await?;
        
        // 5. Calculate cognitive load change
        let cognitive_load_change = self.calculate_cognitive_load_change(&targets, focus_strength);
        let new_cognitive_load = attention_state.cognitive_load + cognitive_load_change;
        attention_state.update_cognitive_load(new_cognitive_load);
        
        // 6. Store attention focus in history
        let attention_focus = AttentionFocus::new(
            targets.clone(),
            attention_weights.clone(),
            focus_strength,
            attention_type,
        );
        
        attention_state.current_focus = attention_focus.clone();
        
        let mut focus_history = self.focus_history.write().await;
        focus_history.push_back(attention_focus);
        
        // Keep only recent focus history
        if focus_history.len() > 100 {
            focus_history.pop_front();
        }
        
        Ok(AttentionResult {
            focused_entities: targets,
            attention_strength: focus_strength,
            working_memory_updates: activation_modulation.memory_updates,
            cognitive_load_change,
        })
    }

    pub async fn shift_attention(
        &self,
        from_targets: Vec<EntityKey>,
        to_targets: Vec<EntityKey>,
        shift_speed: f32,
    ) -> Result<AttentionShiftResult> {
        let start_time = std::time::Instant::now();
        
        // 1. Gradually reduce attention on old targets
        let fade_out_result = self.fade_attention(
            from_targets,
            shift_speed,
        ).await?;
        
        // 2. Gradually increase attention on new targets
        let fade_in_result = self.ramp_attention(
            to_targets,
            shift_speed,
        ).await?;
        
        // 3. Update working memory during transition
        self.update_memory_during_shift(
            &fade_out_result,
            &fade_in_result,
        ).await?;
        
        let shift_duration = start_time.elapsed();
        
        Ok(AttentionShiftResult {
            shift_duration,
            attention_continuity: self.calculate_attention_continuity(&fade_out_result, &fade_in_result),
            working_memory_impact: fade_in_result.memory_impact,
            shift_success: true,
        })
    }

    pub async fn coordinate_with_cognitive_patterns(
        &self,
        pattern_type: CognitivePatternType,
        query: &str,
    ) -> Result<CoordinatedAttentionResult> {
        // 1. Analyze query to determine attention requirements
        let attention_requirements = self.analyze_attention_needs(pattern_type, query).await?;
        
        // 2. Set up attention configuration for the pattern
        let attention_config = self.configure_attention_for_pattern(
            pattern_type,
            attention_requirements,
        ).await?;
        
        // 3. Apply attention configuration
        let attention_result = self.focus_attention(
            attention_config.target_entities.clone(),
            attention_config.focus_strength,
            attention_config.attention_type.clone(),
        ).await?;
        
        // 4. Assess pattern compatibility
        let pattern_compatibility = self.assess_pattern_compatibility(pattern_type);
        
        Ok(CoordinatedAttentionResult {
            attention_config,
            attention_result,
            pattern_compatibility,
        })
    }

    pub async fn manage_divided_attention(
        &self,
        targets: Vec<AttentionTarget>,
    ) -> Result<AttentionResult> {
        let mut attention_state = self.attention_state.write().await;
        
        // Check if we can handle divided attention
        if targets.len() > self.attention_config.max_attention_targets {
            return Err(crate::error::GraphError::InvalidInput(
                "Too many divided attention targets".to_string(),
            ));
        }
        
        // Calculate attention weights with divided attention penalty
        let mut attention_weights = AHashMap::new();
        let total_weight = targets.len() as f32;
        
        for target in &targets {
            let weight = (target.attention_weight / total_weight) * 
                         (1.0 - self.attention_config.divided_attention_penalty);
            attention_weights.insert(target.entity_key, weight);
        }
        
        // Apply divided attention
        let activation_modulation = self.apply_attention_to_activation(
            &attention_weights,
            &attention_state.current_focus,
        ).await?;
        
        // Update attention state
        attention_state.divided_attention_targets = targets.clone();
        attention_state.current_focus = AttentionFocus::new(
            targets.iter().map(|t| t.entity_key).collect(),
            attention_weights,
            0.8, // Reduced strength for divided attention
            AttentionType::Divided,
        );
        
        Ok(AttentionResult {
            focused_entities: targets.iter().map(|t| t.entity_key).collect(),
            attention_strength: 0.8,
            working_memory_updates: activation_modulation.memory_updates,
            cognitive_load_change: 0.3, // Divided attention increases cognitive load
        })
    }

    pub async fn executive_control(
        &self,
        control_command: ExecutiveCommand,
    ) -> Result<AttentionResult> {
        let mut attention_state = self.attention_state.write().await;
        
        match control_command {
            ExecutiveCommand::SwitchFocus { from, to, urgency } => {
                self.shift_attention(vec![from], vec![to], urgency).await?;
            }
            ExecutiveCommand::InhibitDistraction { distractors } => {
                self.apply_inhibition_to_distractors(distractors).await?;
            }
            ExecutiveCommand::BoostAttention { target, boost_factor } => {
                self.boost_attention_to_target(target, boost_factor).await?;
            }
            ExecutiveCommand::ClearFocus => {
                attention_state.current_focus = AttentionFocus::empty();
                attention_state.divided_attention_targets.clear();
            }
        }
        
        Ok(AttentionResult {
            focused_entities: attention_state.current_focus.target_entities.clone(),
            attention_strength: attention_state.current_focus.focus_strength,
            working_memory_updates: vec!["Executive control applied".to_string()],
            cognitive_load_change: 0.0,
        })
    }

    // Helper methods
    async fn calculate_attention_weights(
        &self,
        targets: &[EntityKey],
        focus_strength: f32,
        attention_type: &AttentionType,
    ) -> Result<AHashMap<EntityKey, f32>> {
        let mut weights = AHashMap::new();
        
        match attention_type {
            AttentionType::Selective => {
                // Focus all attention on primary target
                if !targets.is_empty() {
                    weights.insert(targets[0], focus_strength);
                }
            }
            AttentionType::Divided => {
                // Distribute attention across targets
                let weight_per_target = focus_strength / targets.len() as f32;
                for &target in targets {
                    weights.insert(target, weight_per_target);
                }
            }
            AttentionType::Sustained => {
                // Maintain consistent attention
                for &target in targets {
                    weights.insert(target, focus_strength * 0.8);
                }
            }
            AttentionType::Executive => {
                // Strategic attention allocation
                for (i, &target) in targets.iter().enumerate() {
                    let weight = focus_strength * (1.0 - i as f32 * 0.2);
                    weights.insert(target, weight.max(0.1));
                }
            }
            AttentionType::Alternating => {
                // Focus on first target, others get minimal attention
                for (i, &target) in targets.iter().enumerate() {
                    let weight = if i == 0 { focus_strength } else { 0.1 };
                    weights.insert(target, weight);
                }
            }
        }
        
        Ok(weights)
    }

    async fn apply_attention_to_activation(
        &self,
        attention_weights: &AHashMap<EntityKey, f32>,
        _current_focus: &AttentionFocus,
    ) -> Result<ActivationModulation> {
        let mut focused_entities = Vec::new();
        let mut memory_updates = Vec::new();
        let mut inhibition_changes = AHashMap::new();
        
        // Apply attention modulation to activation engine
        for (&entity, &weight) in attention_weights {
            // Boost activation for focused entities
            // This would integrate with the activation engine
            focused_entities.push(entity);
            
            // Apply inhibition to competing entities
            let inhibition_strength = weight * 0.5;
            inhibition_changes.insert(entity, inhibition_strength);
            
            memory_updates.push(format!("Focused on entity {:?} with weight {:.2}", entity, weight));
        }
        
        Ok(ActivationModulation {
            focused_entities,
            memory_updates,
            inhibition_changes,
        })
    }

    async fn update_working_memory_with_focus(
        &self,
        attention_weights: &AHashMap<EntityKey, f32>,
    ) -> Result<()> {
        // Update working memory with attention focus
        for (&_entity, &weight) in attention_weights {
            if weight > 0.3 {
                // Store focused entity in working memory
                // This would require integration with working memory system
            }
        }
        
        Ok(())
    }

    async fn fade_attention(
        &self,
        targets: Vec<EntityKey>,
        fade_speed: f32,
    ) -> Result<AttentionFadeResult> {
        let start_time = std::time::Instant::now();
        let mut final_attention_levels = AHashMap::new();
        
        // Gradually reduce attention on targets
        for target in targets {
            let current_weight = self.attention_state.read().await
                .current_focus.get_attention_weight(&target);
            
            let final_weight = current_weight * (1.0 - fade_speed);
            final_attention_levels.insert(target, final_weight);
        }
        
        Ok(AttentionFadeResult {
            duration: start_time.elapsed(),
            memory_impact: 0.2,
            final_attention_levels,
        })
    }

    async fn ramp_attention(
        &self,
        targets: Vec<EntityKey>,
        ramp_speed: f32,
    ) -> Result<AttentionFadeResult> {
        let start_time = std::time::Instant::now();
        let mut final_attention_levels = AHashMap::new();
        
        // Gradually increase attention on targets
        for target in targets {
            let target_weight = ramp_speed;
            final_attention_levels.insert(target, target_weight);
        }
        
        Ok(AttentionFadeResult {
            duration: start_time.elapsed(),
            memory_impact: 0.3,
            final_attention_levels,
        })
    }

    async fn update_memory_during_shift(
        &self,
        _fade_out: &AttentionFadeResult,
        _fade_in: &AttentionFadeResult,
    ) -> Result<()> {
        // Update working memory during attention shift
        // This would coordinate with the working memory system
        Ok(())
    }

    fn calculate_attention_continuity(
        &self,
        fade_out: &AttentionFadeResult,
        fade_in: &AttentionFadeResult,
    ) -> f32 {
        // Calculate how smooth the attention transition was
        let transition_smoothness = 1.0 - (fade_out.duration.as_secs_f32() + fade_in.duration.as_secs_f32()) / 2.0;
        transition_smoothness.clamp(0.0, 1.0)
    }

    fn calculate_cognitive_load_change(&self, targets: &[EntityKey], focus_strength: f32) -> f32 {
        // Calculate how much the cognitive load changes with this attention focus
        let target_complexity = targets.len() as f32 * 0.1;
        let focus_complexity = focus_strength * 0.2;
        target_complexity + focus_complexity
    }

    async fn analyze_attention_needs(
        &self,
        pattern_type: CognitivePatternType,
        _query: &str,
    ) -> Result<AttentionRequirements> {
        let requirements = match pattern_type {
            CognitivePatternType::Convergent => AttentionRequirements {
                required_focus_strength: 0.8,
                preferred_attention_type: AttentionType::Selective,
                target_entity_types: vec![AttentionTargetType::Entity, AttentionTargetType::Concept],
                sustained_attention_needed: true,
            },
            CognitivePatternType::Divergent => AttentionRequirements {
                required_focus_strength: 0.5,
                preferred_attention_type: AttentionType::Divided,
                target_entity_types: vec![AttentionTargetType::Pattern, AttentionTargetType::Relationship],
                sustained_attention_needed: false,
            },
            CognitivePatternType::Critical => AttentionRequirements {
                required_focus_strength: 0.9,
                preferred_attention_type: AttentionType::Executive,
                target_entity_types: vec![AttentionTargetType::Entity, AttentionTargetType::Relationship],
                sustained_attention_needed: true,
            },
            _ => AttentionRequirements {
                required_focus_strength: 0.6,
                preferred_attention_type: AttentionType::Selective,
                target_entity_types: vec![AttentionTargetType::Entity],
                sustained_attention_needed: false,
            },
        };
        
        Ok(requirements)
    }

    async fn configure_attention_for_pattern(
        &self,
        _pattern_type: CognitivePatternType,
        requirements: AttentionRequirements,
    ) -> Result<AttentionConfiguration> {
        // This would analyze the current context and configure attention appropriately
        let target_entities = vec![]; // Would be populated based on current context
        
        Ok(AttentionConfiguration {
            target_entities,
            focus_strength: requirements.required_focus_strength,
            attention_type: requirements.preferred_attention_type,
            expected_duration: std::time::Duration::from_secs(30),
        })
    }

    fn assess_pattern_compatibility(&self, pattern_type: CognitivePatternType) -> f32 {
        // Assess how well the current attention state supports the pattern
        match pattern_type {
            CognitivePatternType::Convergent => 0.9,
            CognitivePatternType::Divergent => 0.7,
            CognitivePatternType::Critical => 0.8,
            _ => 0.6,
        }
    }

    async fn apply_inhibition_to_distractors(&self, _distractors: Vec<EntityKey>) -> Result<()> {
        // Apply inhibition to prevent distraction
        Ok(())
    }

    async fn boost_attention_to_target(&self, _target: EntityKey, _boost_factor: f32) -> Result<()> {
        // Boost attention to a specific target
        Ok(())
    }

    pub async fn focus_attention_with_memory_coordination(
        &self,
        targets: Vec<EntityKey>,
        focus_strength: f32,
        attention_type: AttentionType,
    ) -> Result<AttentionResult> {
        // 1. Get relevant items from working memory
        let relevant_memory_items = self.working_memory
            .get_attention_relevant_items(&targets, None)
            .await?;
        
        // 2. Adjust focus strength based on memory load
        let memory_load = self.calculate_memory_load(&relevant_memory_items);
        let adjusted_focus_strength = if memory_load > 0.8 {
            focus_strength * 0.8 // Reduce focus strength when memory is heavily loaded
        } else {
            focus_strength
        };
        
        // 3. Apply standard attention focus
        let attention_result = self.focus_attention(targets.clone(), adjusted_focus_strength, attention_type.clone()).await?;
        
        // 4. Update working memory with attention-boosted storage
        for target in &targets {
            // Store attention target in working memory with boost
            let _ = self.working_memory.store_in_working_memory_with_attention(
                crate::cognitive::working_memory::MemoryContent::Concept(target.to_string()),
                adjusted_focus_strength,
                crate::cognitive::working_memory::BufferType::Episodic,
                adjusted_focus_strength,
            ).await;
        }
        
        Ok(attention_result)
    }

    pub async fn shift_attention_with_memory_preservation(
        &self,
        from_targets: Vec<EntityKey>,
        to_targets: Vec<EntityKey>,
        shift_speed: f32,
    ) -> Result<AttentionShiftResult> {
        // 1. Preserve important information from current focus in working memory
        let current_focus_items = self.working_memory
            .get_attention_relevant_items(&from_targets, None)
            .await?;
        
        // 2. Store high-importance items with attention boost before shifting
        for item in current_focus_items {
            if item.importance_score > 0.7 {
                let _ = self.working_memory.store_in_working_memory_with_attention(
                    item.content.clone(),
                    item.importance_score,
                    crate::cognitive::working_memory::BufferType::Episodic,
                    0.3, // Moderate boost to preserve during shift
                ).await;
            }
        }
        
        // 3. Perform the attention shift
        let shift_result = self.shift_attention(from_targets, to_targets, shift_speed).await?;
        
        Ok(shift_result)
    }

    pub async fn executive_attention_with_memory_management(
        &self,
        command: ExecutiveCommand,
    ) -> Result<AttentionResult> {
        match command {
            ExecutiveCommand::SwitchFocus { from, to, urgency } => {
                // Use memory-aware attention shifting
                let _shift_result = self.shift_attention_with_memory_preservation(
                    vec![from],
                    vec![to],
                    urgency,
                ).await?;
                
                Ok(AttentionResult {
                    focused_entities: vec![to],
                    attention_strength: urgency,
                    working_memory_updates: vec![],
                    cognitive_load_change: 0.1,
                })
            },
            ExecutiveCommand::InhibitDistraction { distractors } => {
                // Apply inhibition and update working memory
                self.apply_inhibition_to_distractors(distractors).await?;
                
                Ok(AttentionResult {
                    focused_entities: vec![],
                    attention_strength: 0.0,
                    working_memory_updates: vec![],
                    cognitive_load_change: -0.1,
                })
            },
            ExecutiveCommand::BoostAttention { target, boost_factor } => {
                // Boost attention and store in working memory
                self.boost_attention_to_target(target, boost_factor).await?;
                
                let _ = self.working_memory.store_in_working_memory_with_attention(
                    crate::cognitive::working_memory::MemoryContent::Concept(target.to_string()),
                    boost_factor,
                    crate::cognitive::working_memory::BufferType::Episodic,
                    boost_factor,
                ).await;
                
                Ok(AttentionResult {
                    focused_entities: vec![target],
                    attention_strength: boost_factor,
                    working_memory_updates: vec![],
                    cognitive_load_change: boost_factor * 0.2,
                })
            },
            ExecutiveCommand::ClearFocus => {
                // Clear focus and update working memory
                let empty_result = self.focus_attention(vec![], 0.0, AttentionType::Selective).await?;
                Ok(empty_result)
            },
        }
    }

    fn calculate_memory_load(&self, memory_items: &[crate::cognitive::working_memory::MemoryItem]) -> f32 {
        if memory_items.is_empty() {
            return 0.0;
        }
        
        // Calculate memory load based on number of items and their importance
        let item_count_factor = (memory_items.len() as f32) / 10.0; // Normalize to 0-1 range
        let importance_factor = memory_items.iter()
            .map(|item| item.importance_score)
            .sum::<f32>() / memory_items.len() as f32;
        
        (item_count_factor * 0.6 + importance_factor * 0.4).min(1.0)
    }

    pub async fn get_attention_memory_state(&self) -> Result<AttentionMemoryState> {
        let attention_state = self.attention_state.read().await;
        let current_targets = attention_state.current_focus.target_entities.clone();
        
        let relevant_memory_items = self.working_memory
            .get_attention_relevant_items(&current_targets, None)
            .await?;
        
        let memory_load = self.calculate_memory_load(&relevant_memory_items);
        
        Ok(AttentionMemoryState {
            current_attention_targets: current_targets,
            memory_load,
            relevant_memory_items: relevant_memory_items.len(),
            attention_memory_coordination: if memory_load > 0.8 { 
                "High memory load affecting attention" 
            } else { 
                "Good attention-memory coordination" 
            }.to_string(),
        })
    }

    pub async fn get_attention_state(&self) -> Result<AttentionStateInfo> {
        let attention_state = self.attention_state.read().await;
        
        Ok(AttentionStateInfo {
            focus_strength: attention_state.current_focus.focus_strength,
            attention_capacity: attention_state.attention_capacity,
            cognitive_load: attention_state.cognitive_load,
            current_targets: attention_state.current_focus.target_entities.clone(),
            attention_type: attention_state.current_focus.focus_type.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct AttentionMemoryState {
    pub current_attention_targets: Vec<EntityKey>,
    pub memory_load: f32,
    pub relevant_memory_items: usize,
    pub attention_memory_coordination: String,
}

#[derive(Debug, Clone)]
pub struct AttentionStateInfo {
    pub focus_strength: f32,
    pub attention_capacity: f32,
    pub cognitive_load: f32,
    pub current_targets: Vec<EntityKey>,
    pub attention_type: AttentionType,
}

#[derive(Debug, Clone)]
pub enum ExecutiveCommand {
    SwitchFocus { from: EntityKey, to: EntityKey, urgency: f32 },
    InhibitDistraction { distractors: Vec<EntityKey> },
    BoostAttention { target: EntityKey, boost_factor: f32 },
    ClearFocus,
}