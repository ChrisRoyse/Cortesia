use llmkg::core::{
    brain_types::{BrainInspiredEntity, BrainInspiredRelationship, LogicGateType},
    types::EntityKey,
};
use llmkg::cognitive::{
    working_memory::{MemoryContent, MemoryItem},
    CognitivePatternType,
};
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct SyntheticDataGenerator {
    pub rng: StdRng,
    pub complexity_levels: Vec<ComplexityLevel>,
    pub domain_knowledge: Vec<KnowledgeDomain>,
    pub reasoning_patterns: Vec<ReasoningPattern>,
}

#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Simple,      // Single-step reasoning
    Moderate,    // Multi-step reasoning
    Complex,     // Cross-domain reasoning
    Extreme,     // Multi-domain, multi-step with conflicts
}

#[derive(Debug, Clone)]
pub enum KnowledgeDomain {
    Philosophy,
    Science,
    Technology,
    Ethics,
    Mathematics,
    Psychology,
    Economics,
    Politics,
    Art,
    Literature,
}

#[derive(Debug, Clone)]
pub struct ReasoningPattern {
    pub pattern_type: CognitivePatternType,
    pub required_domains: Vec<KnowledgeDomain>,
    pub complexity_range: (ComplexityLevel, ComplexityLevel),
    pub expected_steps: usize,
}

#[derive(Debug, Clone)]
pub struct SyntheticReasoningScenario {
    pub scenario_id: String,
    pub title: String,
    pub description: String,
    pub query: String,
    pub expected_patterns: Vec<CognitivePatternType>,
    pub complexity_level: ComplexityLevel,
    pub domains: Vec<KnowledgeDomain>,
    pub synthetic_entities: Vec<BrainInspiredEntity>,
    pub synthetic_relationships: Vec<BrainInspiredRelationship>,
    pub memory_items: Vec<MemoryItem>,
    pub attention_targets: Vec<EntityKey>,
    pub inhibition_conflicts: Vec<InhibitionConflict>,
    pub expected_reasoning_steps: Vec<ReasoningStep>,
    pub success_criteria: SuccessCriteria,
}

#[derive(Debug, Clone)]
pub struct InhibitionConflict {
    pub conflict_type: ConflictType,
    pub competing_entities: Vec<EntityKey>,
    pub conflict_strength: f32,
    pub resolution_hint: String,
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    SemanticConflict,
    TemporalConflict,
    CausalConflict,
    EthicalConflict,
    LogicalConflict,
}

#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub step_type: ReasoningStepType,
    pub description: String,
    pub required_pattern: CognitivePatternType,
    pub memory_operations: Vec<MemoryOperation>,
    pub attention_changes: Vec<AttentionChange>,
    pub expected_confidence: f32,
}

#[derive(Debug, Clone)]
pub enum ReasoningStepType {
    Analyze,
    Synthesize,
    Evaluate,
    Generate,
    Compare,
    Conclude,
}

#[derive(Debug, Clone)]
pub struct MemoryOperation {
    pub operation_type: MemoryOperationType,
    pub content: String,
    pub importance: f32,
    pub buffer_type: String,
}

#[derive(Debug, Clone)]
pub enum MemoryOperationType {
    Store,
    Retrieve,
    Consolidate,
    Forget,
}

#[derive(Debug, Clone)]
pub struct AttentionChange {
    pub change_type: AttentionChangeType,
    pub target_entities: Vec<EntityKey>,
    pub attention_strength: f32,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum AttentionChangeType {
    Focus,
    Shift,
    Divide,
    Sustain,
}

#[derive(Debug, Clone)]
pub struct SuccessCriteria {
    pub minimum_confidence: f32,
    pub required_patterns: Vec<CognitivePatternType>,
    pub maximum_time_ms: u64,
    pub memory_efficiency_threshold: f32,
    pub attention_stability_threshold: f32,
    pub inhibition_success_rate: f32,
}

impl SyntheticDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            complexity_levels: vec![
                ComplexityLevel::Simple,
                ComplexityLevel::Moderate,
                ComplexityLevel::Complex,
                ComplexityLevel::Extreme,
            ],
            domain_knowledge: vec![
                KnowledgeDomain::Philosophy,
                KnowledgeDomain::Science,
                KnowledgeDomain::Technology,
                KnowledgeDomain::Ethics,
                KnowledgeDomain::Mathematics,
                KnowledgeDomain::Psychology,
                KnowledgeDomain::Economics,
                KnowledgeDomain::Politics,
                KnowledgeDomain::Art,
                KnowledgeDomain::Literature,
            ],
            reasoning_patterns: Self::create_reasoning_patterns(),
        }
    }

    fn create_reasoning_patterns() -> Vec<ReasoningPattern> {
        vec![
            ReasoningPattern {
                pattern_type: CognitivePatternType::Convergent,
                required_domains: vec![KnowledgeDomain::Mathematics, KnowledgeDomain::Science],
                complexity_range: (ComplexityLevel::Simple, ComplexityLevel::Complex),
                expected_steps: 3,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Divergent,
                required_domains: vec![KnowledgeDomain::Art, KnowledgeDomain::Technology],
                complexity_range: (ComplexityLevel::Moderate, ComplexityLevel::Extreme),
                expected_steps: 5,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Critical,
                required_domains: vec![KnowledgeDomain::Philosophy, KnowledgeDomain::Ethics],
                complexity_range: (ComplexityLevel::Complex, ComplexityLevel::Extreme),
                expected_steps: 6,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Systems,
                required_domains: vec![KnowledgeDomain::Economics, KnowledgeDomain::Politics],
                complexity_range: (ComplexityLevel::Complex, ComplexityLevel::Extreme),
                expected_steps: 7,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Lateral,
                required_domains: vec![KnowledgeDomain::Art, KnowledgeDomain::Psychology],
                complexity_range: (ComplexityLevel::Moderate, ComplexityLevel::Complex),
                expected_steps: 4,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Abstract,
                required_domains: vec![KnowledgeDomain::Mathematics, KnowledgeDomain::Philosophy],
                complexity_range: (ComplexityLevel::Complex, ComplexityLevel::Extreme),
                expected_steps: 8,
            },
            ReasoningPattern {
                pattern_type: CognitivePatternType::Adaptive,
                required_domains: vec![KnowledgeDomain::Technology, KnowledgeDomain::Psychology],
                complexity_range: (ComplexityLevel::Moderate, ComplexityLevel::Extreme),
                expected_steps: 6,
            },
        ]
    }

    pub fn generate_scenario(&mut self, complexity: ComplexityLevel) -> SyntheticReasoningScenario {
        let scenario_id = format!("synthetic_scenario_{}", uuid::Uuid::new_v4());
        
        let (title, description, query, domains) = self.generate_scenario_content(&complexity);
        let expected_patterns = self.select_patterns_for_complexity(&complexity);
        
        let synthetic_entities = self.generate_synthetic_entities(&domains, &complexity);
        let synthetic_relationships = self.generate_synthetic_relationships(&synthetic_entities);
        let memory_items = self.generate_memory_items(&domains, &complexity);
        let attention_targets = self.generate_attention_targets(&synthetic_entities);
        let inhibition_conflicts = self.generate_inhibition_conflicts(&synthetic_entities, &complexity);
        let expected_reasoning_steps = self.generate_reasoning_steps(&expected_patterns, &complexity);
        let success_criteria = self.generate_success_criteria(&complexity);

        SyntheticReasoningScenario {
            scenario_id,
            title,
            description,
            query,
            expected_patterns,
            complexity_level: complexity,
            domains,
            synthetic_entities,
            synthetic_relationships,
            memory_items,
            attention_targets,
            inhibition_conflicts,
            expected_reasoning_steps,
            success_criteria,
        }
    }

    fn generate_scenario_content(&mut self, complexity: &ComplexityLevel) -> (String, String, String, Vec<KnowledgeDomain>) {
        match complexity {
            ComplexityLevel::Simple => {
                let domain = self.domain_knowledge.choose(&mut self.rng).unwrap().clone();
                match domain {
                    KnowledgeDomain::Mathematics => (
                        "Simple Mathematical Problem".to_string(),
                        "Solve a basic algebraic equation".to_string(),
                        "Solve for x: 2x + 5 = 13".to_string(),
                        vec![KnowledgeDomain::Mathematics],
                    ),
                    KnowledgeDomain::Science => (
                        "Basic Scientific Concept".to_string(),
                        "Explain a fundamental scientific principle".to_string(),
                        "Explain why ice floats on water".to_string(),
                        vec![KnowledgeDomain::Science],
                    ),
                    _ => (
                        "Simple Reasoning Task".to_string(),
                        "Basic logical reasoning".to_string(),
                        "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?".to_string(),
                        vec![domain],
                    ),
                }
            },
            ComplexityLevel::Moderate => {
                let domains = self.domain_knowledge.choose_multiple(&mut self.rng, 2).cloned().collect::<Vec<_>>();
                (
                    "Interdisciplinary Analysis".to_string(),
                    "Analyze a problem requiring knowledge from multiple domains".to_string(),
                    "How do psychological biases affect economic decision-making in financial markets?".to_string(),
                    domains,
                )
            },
            ComplexityLevel::Complex => {
                let domains = self.domain_knowledge.choose_multiple(&mut self.rng, 3).cloned().collect::<Vec<_>>();
                (
                    "Multi-Domain Synthesis".to_string(),
                    "Synthesize insights from multiple complex domains".to_string(),
                    "Design an ethical framework for AI governance that considers technological capabilities, economic impacts, and philosophical principles of autonomy and justice.".to_string(),
                    domains,
                )
            },
            ComplexityLevel::Extreme => {
                let domains = self.domain_knowledge.choose_multiple(&mut self.rng, 5).cloned().collect::<Vec<_>>();
                (
                    "Existential Challenge Resolution".to_string(),
                    "Address a fundamental challenge requiring deep reasoning across multiple domains with conflicting perspectives".to_string(),
                    "Develop a comprehensive solution to prevent AI alignment failures that addresses technical challenges, economic incentives, political governance, ethical considerations, and existential risks while managing competing interests of different stakeholders.".to_string(),
                    domains,
                )
            },
        }
    }

    fn select_patterns_for_complexity(&mut self, complexity: &ComplexityLevel) -> Vec<CognitivePatternType> {
        match complexity {
            ComplexityLevel::Simple => vec![CognitivePatternType::Convergent],
            ComplexityLevel::Moderate => vec![
                CognitivePatternType::Convergent,
                CognitivePatternType::Critical,
            ],
            ComplexityLevel::Complex => vec![
                CognitivePatternType::Convergent,
                CognitivePatternType::Divergent,
                CognitivePatternType::Critical,
                CognitivePatternType::Systems,
            ],
            ComplexityLevel::Extreme => vec![
                CognitivePatternType::Convergent,
                CognitivePatternType::Divergent,
                CognitivePatternType::Critical,
                CognitivePatternType::Systems,
                CognitivePatternType::Lateral,
                CognitivePatternType::Abstract,
                CognitivePatternType::Adaptive,
            ],
        }
    }

    fn generate_synthetic_entities(&mut self, domains: &[KnowledgeDomain], complexity: &ComplexityLevel) -> Vec<BrainInspiredEntity> {
        let entity_count = match complexity {
            ComplexityLevel::Simple => 5,
            ComplexityLevel::Moderate => 15,
            ComplexityLevel::Complex => 30,
            ComplexityLevel::Extreme => 50,
        };

        let mut entities = Vec::new();
        
        for i in 0..entity_count {
            let domain = domains.choose(&mut self.rng).unwrap();
            let (name, description) = self.generate_entity_content(domain, i);
            
            entities.push(BrainInspiredEntity {
                name,
                description,
                entity_type: format!("{:?}Entity", domain),
                activation_threshold: self.rng.gen_range(0.3..0.8),
                current_activation: 0.0,
                importance: self.rng.gen_range(0.1..1.0),
                creation_timestamp: std::time::SystemTime::now(),
                last_accessed: std::time::SystemTime::now(),
                access_count: 0,
                related_entities: Vec::new(),
                semantic_embedding: vec![self.rng.gen_range(-1.0..1.0); 128],
                logic_gate: Some(LogicGateType::AND),
                inhibitory_connections: Vec::new(),
                temporal_decay: 0.95,
                context_sensitivity: self.rng.gen_range(0.2..0.9),
            });
        }

        entities
    }

    fn generate_entity_content(&mut self, domain: &KnowledgeDomain, index: usize) -> (String, String) {
        match domain {
            KnowledgeDomain::Philosophy => {
                let concepts = ["consciousness", "free will", "existence", "knowledge", "reality", "ethics", "meaning"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("philosophical_{}_concept_{}", concept, index),
                    format!("A philosophical concept related to {}", concept),
                )
            },
            KnowledgeDomain::Science => {
                let concepts = ["quantum mechanics", "relativity", "evolution", "thermodynamics", "genetics", "neuroscience"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("scientific_{}_principle_{}", concept.replace(" ", "_"), index),
                    format!("A scientific principle from {}", concept),
                )
            },
            KnowledgeDomain::Technology => {
                let concepts = ["artificial intelligence", "quantum computing", "biotechnology", "nanotechnology", "blockchain"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("tech_{}_system_{}", concept.replace(" ", "_"), index),
                    format!("A technological system involving {}", concept),
                )
            },
            KnowledgeDomain::Ethics => {
                let concepts = ["utilitarianism", "deontology", "virtue ethics", "consequentialism", "care ethics"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("ethical_{}_framework_{}", concept.replace(" ", "_"), index),
                    format!("An ethical framework based on {}", concept),
                )
            },
            KnowledgeDomain::Mathematics => {
                let concepts = ["algebra", "calculus", "topology", "number theory", "geometry", "statistics"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("mathematical_{}_concept_{}", concept, index),
                    format!("A mathematical concept from {}", concept),
                )
            },
            KnowledgeDomain::Psychology => {
                let concepts = ["cognition", "behavior", "personality", "memory", "perception", "emotion"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("psychological_{}_process_{}", concept, index),
                    format!("A psychological process related to {}", concept),
                )
            },
            KnowledgeDomain::Economics => {
                let concepts = ["supply and demand", "market efficiency", "game theory", "behavioral economics", "macroeconomics"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("economic_{}_principle_{}", concept.replace(" ", "_"), index),
                    format!("An economic principle from {}", concept),
                )
            },
            KnowledgeDomain::Politics => {
                let concepts = ["democracy", "governance", "policy", "international relations", "political theory"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("political_{}_system_{}", concept.replace(" ", "_"), index),
                    format!("A political system or concept related to {}", concept),
                )
            },
            KnowledgeDomain::Art => {
                let concepts = ["aesthetics", "creativity", "expression", "interpretation", "cultural meaning"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("artistic_{}_element_{}", concept.replace(" ", "_"), index),
                    format!("An artistic element related to {}", concept),
                )
            },
            KnowledgeDomain::Literature => {
                let concepts = ["narrative", "symbolism", "character", "theme", "style", "interpretation"];
                let concept = concepts.choose(&mut self.rng).unwrap();
                (
                    format!("literary_{}_device_{}", concept, index),
                    format!("A literary device or concept related to {}", concept),
                )
            },
        }
    }

    fn generate_synthetic_relationships(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        let relationship_count = (entities.len() as f32 * 1.5) as usize;

        for i in 0..relationship_count {
            let entity1 = entities.choose(&mut self.rng).unwrap();
            let entity2 = entities.choose(&mut self.rng).unwrap();
            
            if entity1.name != entity2.name {
                let relationship_types = [
                    "influences", "depends_on", "conflicts_with", "supports", "derives_from",
                    "applies_to", "emerges_from", "constrains", "enables", "synthesizes_with"
                ];
                
                let relationship_type = relationship_types.choose(&mut self.rng).unwrap();
                
                relationships.push(BrainInspiredRelationship {
                    name: format!("relationship_{}", i),
                    description: format!("{} {} {}", entity1.name, relationship_type, entity2.name),
                    source_entity: entity1.name.clone(),
                    target_entity: entity2.name.clone(),
                    relationship_type: relationship_type.to_string(),
                    strength: self.rng.gen_range(0.1..1.0),
                    confidence: self.rng.gen_range(0.3..0.9),
                    temporal_weight: self.rng.gen_range(0.5..1.0),
                    context_conditions: Vec::new(),
                    bidirectional: self.rng.gen_bool(0.3),
                    creation_timestamp: std::time::SystemTime::now(),
                    last_accessed: std::time::SystemTime::now(),
                    access_count: 0,
                });
            }
        }

        relationships
    }

    fn generate_memory_items(&mut self, domains: &[KnowledgeDomain], complexity: &ComplexityLevel) -> Vec<MemoryItem> {
        let item_count = match complexity {
            ComplexityLevel::Simple => 3,
            ComplexityLevel::Moderate => 8,
            ComplexityLevel::Complex => 15,
            ComplexityLevel::Extreme => 25,
        };

        let mut items = Vec::new();
        
        for i in 0..item_count {
            let domain = domains.choose(&mut self.rng).unwrap();
            let content = self.generate_memory_content(domain, i);
            
            items.push(MemoryItem {
                content,
                activation_level: self.rng.gen_range(0.2..0.9),
                timestamp: Instant::now(),
                importance_score: self.rng.gen_range(0.1..1.0),
                access_count: self.rng.gen_range(0..10),
                decay_factor: self.rng.gen_range(0.8..1.0),
            });
        }

        items
    }

    fn generate_memory_content(&mut self, domain: &KnowledgeDomain, index: usize) -> MemoryContent {
        match self.rng.gen_range(0..3) {
            0 => MemoryContent::Concept(format!("memory_concept_{}_{}", domain.to_string(), index)),
            1 => MemoryContent::Relationship(
                format!("entity_a_{}", index),
                format!("entity_b_{}", index),
                self.rng.gen_range(0.1..1.0),
            ),
            _ => MemoryContent::Composite(vec![
                MemoryContent::Concept(format!("composite_concept_1_{}", index)),
                MemoryContent::Concept(format!("composite_concept_2_{}", index)),
            ]),
        }
    }

    fn generate_attention_targets(&mut self, entities: &[BrainInspiredEntity]) -> Vec<EntityKey> {
        let target_count = (entities.len() / 3).max(1);
        
        entities.choose_multiple(&mut self.rng, target_count)
            .enumerate()
            .map(|(i, _)| EntityKey::new(i))
            .collect()
    }

    fn generate_inhibition_conflicts(&mut self, entities: &[BrainInspiredEntity], complexity: &ComplexityLevel) -> Vec<InhibitionConflict> {
        let conflict_count = match complexity {
            ComplexityLevel::Simple => 1,
            ComplexityLevel::Moderate => 2,
            ComplexityLevel::Complex => 4,
            ComplexityLevel::Extreme => 7,
        };

        let mut conflicts = Vec::new();
        
        for i in 0..conflict_count {
            let conflict_types = [
                ConflictType::SemanticConflict,
                ConflictType::TemporalConflict,
                ConflictType::CausalConflict,
                ConflictType::EthicalConflict,
                ConflictType::LogicalConflict,
            ];
            
            let conflict_type = conflict_types.choose(&mut self.rng).unwrap().clone();
            let competing_entities = entities.choose_multiple(&mut self.rng, 3)
                .enumerate()
                .map(|(idx, _)| EntityKey::new(idx))
                .collect();
            
            conflicts.push(InhibitionConflict {
                conflict_type: conflict_type.clone(),
                competing_entities,
                conflict_strength: self.rng.gen_range(0.3..0.9),
                resolution_hint: self.generate_resolution_hint(&conflict_type),
            });
        }

        conflicts
    }

    fn generate_resolution_hint(&self, conflict_type: &ConflictType) -> String {
        match conflict_type {
            ConflictType::SemanticConflict => "Consider the specificity hierarchy and context".to_string(),
            ConflictType::TemporalConflict => "Prioritize recency while maintaining causal consistency".to_string(),
            ConflictType::CausalConflict => "Evaluate evidence strength and logical consistency".to_string(),
            ConflictType::EthicalConflict => "Apply ethical frameworks and consider stakeholder impacts".to_string(),
            ConflictType::LogicalConflict => "Use formal logical rules and contradiction resolution".to_string(),
        }
    }

    fn generate_reasoning_steps(&mut self, patterns: &[CognitivePatternType], complexity: &ComplexityLevel) -> Vec<ReasoningStep> {
        let step_count = match complexity {
            ComplexityLevel::Simple => 2,
            ComplexityLevel::Moderate => 4,
            ComplexityLevel::Complex => 6,
            ComplexityLevel::Extreme => 10,
        };

        let mut steps = Vec::new();
        
        for i in 0..step_count {
            let pattern = patterns.choose(&mut self.rng).unwrap();
            let step_types = [
                ReasoningStepType::Analyze,
                ReasoningStepType::Synthesize,
                ReasoningStepType::Evaluate,
                ReasoningStepType::Generate,
                ReasoningStepType::Compare,
                ReasoningStepType::Conclude,
            ];
            
            let step_type = step_types.choose(&mut self.rng).unwrap().clone();
            
            steps.push(ReasoningStep {
                step_number: i + 1,
                step_type: step_type.clone(),
                description: self.generate_step_description(&step_type, pattern),
                required_pattern: *pattern,
                memory_operations: self.generate_memory_operations(i),
                attention_changes: self.generate_attention_changes(i),
                expected_confidence: self.rng.gen_range(0.4..0.9),
            });
        }

        steps
    }

    fn generate_step_description(&self, step_type: &ReasoningStepType, pattern: &CognitivePatternType) -> String {
        match (step_type, pattern) {
            (ReasoningStepType::Analyze, CognitivePatternType::Critical) => 
                "Critically analyze the assumptions and evidence".to_string(),
            (ReasoningStepType::Synthesize, CognitivePatternType::Divergent) => 
                "Generate multiple creative solutions".to_string(),
            (ReasoningStepType::Evaluate, CognitivePatternType::Convergent) => 
                "Evaluate options and converge on best solution".to_string(),
            (ReasoningStepType::Generate, CognitivePatternType::Lateral) => 
                "Generate novel connections and insights".to_string(),
            (ReasoningStepType::Compare, CognitivePatternType::Systems) => 
                "Compare systemic relationships and feedback loops".to_string(),
            (ReasoningStepType::Conclude, CognitivePatternType::Abstract) => 
                "Draw abstract conclusions and generalizations".to_string(),
            _ => format!("Apply {:?} thinking to {:?} the problem", pattern, step_type),
        }
    }

    fn generate_memory_operations(&mut self, step: usize) -> Vec<MemoryOperation> {
        let op_count = self.rng.gen_range(1..4);
        let mut operations = Vec::new();
        
        for i in 0..op_count {
            let op_types = [
                MemoryOperationType::Store,
                MemoryOperationType::Retrieve,
                MemoryOperationType::Consolidate,
                MemoryOperationType::Forget,
            ];
            
            let op_type = op_types.choose(&mut self.rng).unwrap().clone();
            
            operations.push(MemoryOperation {
                operation_type: op_type.clone(),
                content: format!("memory_content_step_{}_op_{}", step, i),
                importance: self.rng.gen_range(0.2..0.9),
                buffer_type: ["phonological", "visuospatial", "episodic"]
                    .choose(&mut self.rng).unwrap().to_string(),
            });
        }

        operations
    }

    fn generate_attention_changes(&mut self, step: usize) -> Vec<AttentionChange> {
        let change_count = self.rng.gen_range(1..3);
        let mut changes = Vec::new();
        
        for i in 0..change_count {
            let change_types = [
                AttentionChangeType::Focus,
                AttentionChangeType::Shift,
                AttentionChangeType::Divide,
                AttentionChangeType::Sustain,
            ];
            
            let change_type = change_types.choose(&mut self.rng).unwrap().clone();
            
            changes.push(AttentionChange {
                change_type: change_type.clone(),
                target_entities: vec![EntityKey::new(step * 10 + i)],
                attention_strength: self.rng.gen_range(0.3..0.9),
                reason: format!("attention_change_step_{}_reason_{}", step, i),
            });
        }

        changes
    }

    fn generate_success_criteria(&mut self, complexity: &ComplexityLevel) -> SuccessCriteria {
        match complexity {
            ComplexityLevel::Simple => SuccessCriteria {
                minimum_confidence: 0.8,
                required_patterns: vec![CognitivePatternType::Convergent],
                maximum_time_ms: 1000,
                memory_efficiency_threshold: 0.9,
                attention_stability_threshold: 0.8,
                inhibition_success_rate: 0.9,
            },
            ComplexityLevel::Moderate => SuccessCriteria {
                minimum_confidence: 0.6,
                required_patterns: vec![CognitivePatternType::Convergent, CognitivePatternType::Critical],
                maximum_time_ms: 5000,
                memory_efficiency_threshold: 0.7,
                attention_stability_threshold: 0.6,
                inhibition_success_rate: 0.8,
            },
            ComplexityLevel::Complex => SuccessCriteria {
                minimum_confidence: 0.5,
                required_patterns: vec![
                    CognitivePatternType::Convergent,
                    CognitivePatternType::Divergent,
                    CognitivePatternType::Critical,
                    CognitivePatternType::Systems,
                ],
                maximum_time_ms: 15000,
                memory_efficiency_threshold: 0.6,
                attention_stability_threshold: 0.5,
                inhibition_success_rate: 0.7,
            },
            ComplexityLevel::Extreme => SuccessCriteria {
                minimum_confidence: 0.4,
                required_patterns: vec![
                    CognitivePatternType::Convergent,
                    CognitivePatternType::Divergent,
                    CognitivePatternType::Critical,
                    CognitivePatternType::Systems,
                    CognitivePatternType::Lateral,
                    CognitivePatternType::Abstract,
                ],
                maximum_time_ms: 30000,
                memory_efficiency_threshold: 0.5,
                attention_stability_threshold: 0.4,
                inhibition_success_rate: 0.6,
            },
        }
    }

    pub fn generate_test_suite(&mut self, scenarios_per_complexity: usize) -> Vec<SyntheticReasoningScenario> {
        let mut test_suite = Vec::new();
        
        for complexity in &self.complexity_levels.clone() {
            for _ in 0..scenarios_per_complexity {
                test_suite.push(self.generate_scenario(complexity.clone()));
            }
        }

        test_suite
    }

    pub fn generate_stress_test_scenarios(&mut self, count: usize) -> Vec<SyntheticReasoningScenario> {
        let mut stress_scenarios = Vec::new();
        
        for _ in 0..count {
            stress_scenarios.push(self.generate_scenario(ComplexityLevel::Extreme));
        }

        stress_scenarios
    }

    pub fn generate_benchmark_scenarios(&mut self) -> Vec<SyntheticReasoningScenario> {
        vec![
            self.generate_scenario(ComplexityLevel::Simple),
            self.generate_scenario(ComplexityLevel::Moderate),
            self.generate_scenario(ComplexityLevel::Complex),
            self.generate_scenario(ComplexityLevel::Extreme),
        ]
    }
}

impl std::fmt::Display for KnowledgeDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// Test functions for the synthetic data generator
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let mut generator = SyntheticDataGenerator::new(42);
        
        // Test simple scenario generation
        let simple_scenario = generator.generate_scenario(ComplexityLevel::Simple);
        assert_eq!(simple_scenario.complexity_level, ComplexityLevel::Simple);
        assert!(simple_scenario.synthetic_entities.len() == 5);
        assert!(simple_scenario.expected_patterns.len() == 1);
        
        // Test extreme scenario generation
        let extreme_scenario = generator.generate_scenario(ComplexityLevel::Extreme);
        assert_eq!(extreme_scenario.complexity_level, ComplexityLevel::Extreme);
        assert!(extreme_scenario.synthetic_entities.len() == 50);
        assert!(extreme_scenario.expected_patterns.len() >= 6);
        assert!(extreme_scenario.expected_reasoning_steps.len() == 10);
        
        // Test test suite generation
        let test_suite = generator.generate_test_suite(2);
        assert_eq!(test_suite.len(), 8); // 4 complexity levels * 2 scenarios each
        
        println!("✓ Synthetic data generation test passed");
    }

    #[test]
    fn test_scenario_content_quality() {
        let mut generator = SyntheticDataGenerator::new(123);
        
        let scenario = generator.generate_scenario(ComplexityLevel::Complex);
        
        // Check that scenario has meaningful content
        assert!(!scenario.title.is_empty());
        assert!(!scenario.description.is_empty());
        assert!(!scenario.query.is_empty());
        assert!(scenario.domains.len() >= 3);
        
        // Check that entities have proper structure
        for entity in &scenario.synthetic_entities {
            assert!(!entity.name.is_empty());
            assert!(!entity.description.is_empty());
            assert!(entity.activation_threshold > 0.0);
            assert!(entity.importance > 0.0);
        }
        
        // Check that relationships are meaningful
        for relationship in &scenario.synthetic_relationships {
            assert!(!relationship.name.is_empty());
            assert!(!relationship.source_entity.is_empty());
            assert!(!relationship.target_entity.is_empty());
            assert!(relationship.strength > 0.0);
        }
        
        println!("✓ Scenario content quality test passed");
    }

    #[test]
    fn test_reasoning_step_generation() {
        let mut generator = SyntheticDataGenerator::new(456);
        
        let scenario = generator.generate_scenario(ComplexityLevel::Moderate);
        
        // Check reasoning steps are logically structured
        assert_eq!(scenario.expected_reasoning_steps.len(), 4);
        
        for (i, step) in scenario.expected_reasoning_steps.iter().enumerate() {
            assert_eq!(step.step_number, i + 1);
            assert!(!step.description.is_empty());
            assert!(step.memory_operations.len() > 0);
            assert!(step.expected_confidence > 0.0);
        }
        
        println!("✓ Reasoning step generation test passed");
    }

    #[test]
    fn test_success_criteria_scaling() {
        let mut generator = SyntheticDataGenerator::new(789);
        
        let simple = generator.generate_scenario(ComplexityLevel::Simple);
        let extreme = generator.generate_scenario(ComplexityLevel::Extreme);
        
        // Success criteria should scale with complexity
        assert!(simple.success_criteria.minimum_confidence > extreme.success_criteria.minimum_confidence);
        assert!(simple.success_criteria.maximum_time_ms < extreme.success_criteria.maximum_time_ms);
        assert!(simple.success_criteria.required_patterns.len() < extreme.success_criteria.required_patterns.len());
        
        println!("✓ Success criteria scaling test passed");
    }
}