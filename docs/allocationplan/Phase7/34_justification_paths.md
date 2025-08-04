# Micro Task 34: Justification Paths for Justification Tracing

**Priority**: CRITICAL  
**Estimated Time**: 35 minutes  
**Dependencies**: Phase 6 TMS, Tasks 31-33 (complete belief and context system)  
**Skills Required**: Rust graph algorithms, logical reasoning, path analysis

## Objective

Implement comprehensive justification path tracing that maps activation paths to logical inference chains, enabling users to understand why specific nodes were activated and providing transparent reasoning explanations with confidence assessments.

## Context

Building on the multi-context belief system, this component provides the crucial "explainability" layer by tracing how activation spread through justification networks. It transforms technical activation paths into understandable reasoning chains that users can follow and evaluate.

## Specifications

### Required Components

1. **JustificationPathBuilder**
   - Constructs comprehensive justification paths from activation traces
   - Maps activation flows to logical inference steps
   - Handles complex multi-path justifications
   - Supports path filtering and ranking

2. **ReasoningChainAnalyzer**
   - Analyzes justification chains for logical validity
   - Identifies weak links and reasoning gaps
   - Calculates path strength and reliability metrics
   - Detects circular reasoning and invalid inferences

3. **PathExplanationGenerator**
   - Converts technical paths into human-readable explanations
   - Generates natural language reasoning descriptions
   - Creates visual path representations
   - Supports different explanation complexity levels

4. **JustificationValidator**
   - Validates justification paths against logical rules
   - Checks path consistency with TMS constraints
   - Verifies temporal validity of reasoning chains
   - Identifies potential logical fallacies

### Performance Requirements

- Path construction: <2ms for chains up to 10 steps
- Path analysis: <5ms for complex multi-path justifications
- Explanation generation: <10ms for detailed explanations
- Validation checking: <1ms per justification link

## Implementation Guide

### Step 1: Justification Path Builder

```rust
// File: src/core/justification/path_builder.rs

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use crate::core::types::{NodeId, Timestamp};
use crate::core::activation::belief_aware_state::BeliefAwareActivationState;
use crate::tms::{BeliefId, JustificationId, JustificationLink, InferenceRule};

#[derive(Debug, Clone)]
pub struct JustificationPathBuilder {
    // Path construction parameters
    pub max_path_depth: usize,
    pub min_confidence_threshold: f32,
    pub include_alternative_paths: bool,
    
    // Path ranking criteria
    pub path_ranking_criteria: PathRankingCriteria,
    
    // Construction state
    pub visited_nodes: HashSet<NodeId>,
    pub path_cache: HashMap<(NodeId, NodeId), Vec<JustificationPath>>,
}

#[derive(Debug, Clone)]
pub struct JustificationPath {
    // Path structure
    pub path_id: JustificationPathId,
    pub source_node: NodeId,
    pub target_node: NodeId,
    pub reasoning_steps: Vec<ReasoningStep>,
    
    // Path metrics
    pub total_strength: f32,
    pub min_confidence: f32,
    pub path_length: usize,
    pub complexity_score: f32,
    
    // Path metadata
    pub construction_method: PathConstructionMethod,
    pub temporal_span: Option<(Timestamp, Timestamp)>,
    pub context_dependencies: Vec<ContextId>,
    
    // Quality indicators
    pub has_circular_reasoning: bool,
    pub has_weak_links: bool,
    pub reliability_score: f32,
    pub explanation_quality: ExplanationQuality,
}

#[derive(Debug, Clone)]
pub struct ReasoningStep {
    // Step identification
    pub step_id: ReasoningStepId,
    pub step_number: usize,
    
    // Logical structure
    pub premise_nodes: Vec<NodeId>,
    pub conclusion_node: NodeId,
    pub inference_rule: InferenceRule,
    
    // Justification details
    pub justification_link: JustificationLink,
    pub confidence_level: f32,
    pub belief_basis: Vec<BeliefId>,
    
    // Step metadata
    pub step_type: ReasoningStepType,
    pub temporal_context: Option<Timestamp>,
    pub context_requirements: Vec<ContextRequirement>,
    
    // Quality measures
    pub logical_validity: LogicalValidity,
    pub evidence_strength: f32,
    pub alternative_inferences: Vec<AlternativeInference>,
}

#[derive(Debug, Clone)]
pub enum ReasoningStepType {
    Deductive,     // Logical deduction from premises
    Inductive,     // Generalization from examples
    Abductive,     // Best explanation inference
    Analogical,    // Reasoning by analogy
    Default,       // Non-monotonic default reasoning
    Temporal,      // Time-based inference
    Causal,        // Cause-effect reasoning
}

#[derive(Debug, Clone)]
pub enum LogicalValidity {
    Valid,         // Logically sound
    Probable,      // Likely but not certain
    Uncertain,     // Validity unclear
    Invalid,       // Logically unsound
    Fallacious,    // Contains logical fallacy
}

impl JustificationPathBuilder {
    pub fn new() -> Self {
        Self {
            max_path_depth: 15,
            min_confidence_threshold: 0.1,
            include_alternative_paths: true,
            path_ranking_criteria: PathRankingCriteria::default(),
            visited_nodes: HashSet::new(),
            path_cache: HashMap::new(),
        }
    }
    
    pub async fn build_justification_paths(
        &mut self,
        activation_state: &BeliefAwareActivationState,
        target_nodes: Vec<NodeId>,
        tms: &TruthMaintenanceSystem,
    ) -> Result<JustificationPathCollection, PathBuildingError> {
        let mut path_collection = JustificationPathCollection::new();
        
        // Build paths for each target node
        for target_node in target_nodes {
            let node_paths = self.build_paths_to_node(
                target_node,
                activation_state,
                tms,
            ).await?;
            
            path_collection.add_node_paths(target_node, node_paths);
        }
        
        // Analyze path relationships
        self.analyze_path_relationships(&mut path_collection).await?;
        
        // Rank all paths
        self.rank_paths(&mut path_collection).await?;
        
        Ok(path_collection)
    }
    
    async fn build_paths_to_node(
        &mut self,
        target_node: NodeId,
        activation_state: &BeliefAwareActivationState,
        tms: &TruthMaintenanceSystem,
    ) -> Result<Vec<JustificationPath>, PathBuildingError> {
        let mut paths = Vec::new();
        self.visited_nodes.clear();
        
        // Use breadth-first search to find all justification paths
        let mut path_queue = VecDeque::new();
        
        // Get target node's justifications
        if let Some(belief_activation) = activation_state.get_belief_activation(target_node) {
            for justification_chain in &activation_state.active_justifications[&target_node] {
                // Create initial path
                let initial_path = JustificationPath {
                    path_id: JustificationPathId::new(),
                    source_node: target_node, // Will be updated as we trace back
                    target_node,
                    reasoning_steps: Vec::new(),
                    total_strength: justification_chain.total_strength,
                    min_confidence: justification_chain.confidence_path.iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .copied()
                        .unwrap_or(0.0),
                    path_length: 0,
                    complexity_score: 0.0,
                    construction_method: PathConstructionMethod::BreadthFirst,
                    temporal_span: None,
                    context_dependencies: Vec::new(),
                    has_circular_reasoning: false,
                    has_weak_links: false,
                    reliability_score: 0.0,
                    explanation_quality: ExplanationQuality::Unknown,
                };
                
                path_queue.push_back((initial_path, justification_chain.belief_path.clone()));
            }
        }
        
        // Process queue to build complete paths
        while let Some((mut current_path, remaining_beliefs)) = path_queue.pop_front() {
            if current_path.path_length >= self.max_path_depth {
                paths.push(current_path);
                continue;
            }
            
            if remaining_beliefs.is_empty() {
                // Complete path found
                self.finalize_path(&mut current_path, activation_state, tms).await?;
                paths.push(current_path);
                continue;
            }
            
            // Process next belief in chain
            let current_belief = remaining_beliefs[0];
            let remaining = remaining_beliefs[1..].to_vec();
            
            // Get justifications for current belief
            let belief_justifications = tms.get_justifications_for_belief(
                current_belief,
                &activation_state.belief_context,
            ).await?;
            
            if belief_justifications.is_empty() {
                // Base belief - complete the path
                self.add_base_belief_step(&mut current_path, current_belief, tms).await?;
                self.finalize_path(&mut current_path, activation_state, tms).await?;
                paths.push(current_path);
            } else {
                // Continue building path
                for justification in belief_justifications {
                    let mut extended_path = current_path.clone();
                    
                    // Add reasoning step
                    let reasoning_step = self.create_reasoning_step(
                        current_belief,
                        &justification,
                        extended_path.reasoning_steps.len(),
                        tms,
                    ).await?;
                    
                    extended_path.reasoning_steps.push(reasoning_step);
                    extended_path.path_length += 1;
                    
                    // Check for circular reasoning
                    if self.check_circular_reasoning(&extended_path) {
                        extended_path.has_circular_reasoning = true;
                        self.finalize_path(&mut extended_path, activation_state, tms).await?;
                        paths.push(extended_path);
                    } else {
                        // Continue with remaining beliefs
                        path_queue.push_back((extended_path, remaining.clone()));
                    }
                }
            }
        }
        
        // Filter and rank paths
        paths.retain(|path| path.min_confidence >= self.min_confidence_threshold);
        paths.sort_by(|a, b| {
            b.reliability_score.partial_cmp(&a.reliability_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(paths)
    }
    
    async fn create_reasoning_step(
        &self,
        belief_id: BeliefId,
        justification: &JustificationLink,
        step_number: usize,
        tms: &TruthMaintenanceSystem,
    ) -> Result<ReasoningStep, PathBuildingError> {
        // Get belief details
        let belief = tms.get_belief(belief_id).await?
            .ok_or(PathBuildingError::BeliefNotFound(belief_id))?;
        
        // Determine reasoning step type
        let step_type = self.determine_reasoning_type(&justification).await?;
        
        // Analyze logical validity
        let logical_validity = self.analyze_step_validity(
            &justification,
            &step_type,
            tms,
        ).await?;
        
        // Find alternative inferences
        let alternative_inferences = self.find_alternative_inferences(
            belief_id,
            &justification,
            tms,
        ).await?;
        
        Ok(ReasoningStep {
            step_id: ReasoningStepId::new(),
            step_number,
            premise_nodes: justification.antecedent_nodes.clone(),
            conclusion_node: belief.node_id,
            inference_rule: justification.inference_rule.clone(),
            justification_link: justification.clone(),
            confidence_level: justification.confidence,
            belief_basis: vec![belief_id],
            step_type,
            temporal_context: belief.temporal_context,
            context_requirements: self.extract_context_requirements(&justification).await?,
            logical_validity,
            evidence_strength: justification.evidence_strength,
            alternative_inferences,
        })
    }
    
    async fn finalize_path(
        &self,
        path: &mut JustificationPath,
        activation_state: &BeliefAwareActivationState,
        tms: &TruthMaintenanceSystem,
    ) -> Result<(), PathBuildingError> {
        // Calculate final metrics
        path.complexity_score = self.calculate_complexity_score(path).await?;
        path.reliability_score = self.calculate_reliability_score(path).await?;
        
        // Check for weak links
        path.has_weak_links = path.reasoning_steps.iter()
            .any(|step| step.confidence_level < 0.3 || step.evidence_strength < 0.4);
        
        // Determine explanation quality
        path.explanation_quality = self.assess_explanation_quality(path).await?;
        
        // Set temporal span
        if !path.reasoning_steps.is_empty() {
            let timestamps: Vec<_> = path.reasoning_steps.iter()
                .filter_map(|step| step.temporal_context)
                .collect();
            
            if !timestamps.is_empty() {
                let min_time = timestamps.iter().min().copied().unwrap();
                let max_time = timestamps.iter().max().copied().unwrap();
                path.temporal_span = Some((min_time, max_time));
            }
        }
        
        // Extract context dependencies
        path.context_dependencies = path.reasoning_steps.iter()
            .flat_map(|step| step.context_requirements.iter())
            .filter_map(|req| match req {
                ContextRequirement::SpecificContext(ctx_id) => Some(*ctx_id),
                _ => None,
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        
        Ok(())
    }
    
    fn check_circular_reasoning(&self, path: &JustificationPath) -> bool {
        let mut seen_nodes = HashSet::new();
        
        for step in &path.reasoning_steps {
            if seen_nodes.contains(&step.conclusion_node) {
                return true;
            }
            seen_nodes.insert(step.conclusion_node);
            
            for &premise in &step.premise_nodes {
                if seen_nodes.contains(&premise) {
                    return true;
                }
            }
        }
        
        false
    }
    
    async fn calculate_reliability_score(&self, path: &JustificationPath) -> Result<f32, PathBuildingError> {
        if path.reasoning_steps.is_empty() {
            return Ok(1.0);
        }
        
        // Base score from confidence levels
        let confidence_product: f32 = path.reasoning_steps.iter()
            .map(|step| step.confidence_level)
            .product();
        
        // Penalty for path length
        let length_penalty = 0.95_f32.powi(path.path_length as i32);
        
        // Penalty for weak links
        let weak_link_penalty = if path.has_weak_links { 0.8 } else { 1.0 };
        
        // Penalty for circular reasoning
        let circular_penalty = if path.has_circular_reasoning { 0.6 } else { 1.0 };
        
        // Bonus for valid logical steps
        let validity_bonus = path.reasoning_steps.iter()
            .map(|step| match step.logical_validity {
                LogicalValidity::Valid => 1.0,
                LogicalValidity::Probable => 0.9,
                LogicalValidity::Uncertain => 0.7,
                LogicalValidity::Invalid => 0.5,
                LogicalValidity::Fallacious => 0.3,
            })
            .sum::<f32>() / path.reasoning_steps.len() as f32;
        
        Ok(confidence_product * length_penalty * weak_link_penalty * circular_penalty * validity_bonus)
    }
    
    async fn assess_explanation_quality(&self, path: &JustificationPath) -> Result<ExplanationQuality, PathBuildingError> {
        let reliability = path.reliability_score;
        let complexity = path.complexity_score;
        let has_issues = path.has_circular_reasoning || path.has_weak_links;
        
        if reliability > 0.8 && complexity < 0.5 && !has_issues {
            Ok(ExplanationQuality::Excellent)
        } else if reliability > 0.6 && complexity < 0.7 {
            Ok(ExplanationQuality::Good)
        } else if reliability > 0.4 {
            Ok(ExplanationQuality::Fair)
        } else {
            Ok(ExplanationQuality::Poor)
        }
    }
}

#[derive(Debug, Clone)]
pub struct JustificationPathCollection {
    pub paths_by_node: HashMap<NodeId, Vec<JustificationPath>>,
    pub all_paths: Vec<JustificationPath>,
    pub path_relationships: Vec<PathRelationship>,
    pub collection_metrics: CollectionMetrics,
}

#[derive(Debug, Clone)]
pub struct PathRelationship {
    pub relationship_type: PathRelationshipType,
    pub path_a: JustificationPathId,
    pub path_b: JustificationPathId,
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub enum PathRelationshipType {
    SharedPremises,    // Paths share common premises
    Contradictory,     // Paths lead to contradictory conclusions
    Alternative,       // Paths are alternative routes to same conclusion
    Dependent,         // One path depends on the other
    Independent,       // Paths are logically independent
}

#[derive(Debug, Clone)]
pub enum ExplanationQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum PathConstructionMethod {
    BreadthFirst,
    DepthFirst,
    BestFirst,
    Heuristic,
}
```

### Step 2: Reasoning Chain Analyzer

```rust
// File: src/core/justification/reasoning_analyzer.rs

use std::collections::{HashMap, HashSet};
use crate::core::justification::path_builder::{JustificationPath, ReasoningStep, LogicalValidity};

pub struct ReasoningChainAnalyzer {
    // Analysis configuration
    pub logical_rules: Vec<LogicalRule>,
    pub fallacy_detectors: Vec<FallacyDetector>,
    pub validity_checkers: Vec<ValidityChecker>,
    
    // Analysis cache
    pub analysis_cache: HashMap<JustificationPathId, ReasoningAnalysis>,
    
    // Pattern recognition
    pub common_patterns: Vec<ReasoningPattern>,
    pub known_fallacies: Vec<KnownFallacy>,
}

#[derive(Debug, Clone)]
pub struct ReasoningAnalysis {
    // Overall assessment
    pub overall_validity: LogicalValidity,
    pub logical_strength: f32,
    pub reasoning_quality: ReasoningQuality,
    
    // Detailed analysis
    pub step_analyses: Vec<StepAnalysis>,
    pub detected_fallacies: Vec<DetectedFallacy>,
    pub reasoning_gaps: Vec<ReasoningGap>,
    pub strength_profile: StrengthProfile,
    
    // Recommendations
    pub improvement_suggestions: Vec<ImprovementSuggestion>,
    pub alternative_reasoning: Vec<AlternativeReasoning>,
    
    // Metrics
    pub analysis_confidence: f32,
    pub completeness_score: f32,
}

#[derive(Debug, Clone)]
pub struct StepAnalysis {
    pub step_id: ReasoningStepId,
    pub validity_assessment: ValidityAssessment,
    pub logical_form: LogicalForm,
    pub evidence_adequacy: EvidenceAdequacy,
    pub inference_quality: InferenceQuality,
    pub context_appropriateness: f32,
}

#[derive(Debug, Clone)]
pub enum ReasoningQuality {
    Rigorous,     // Highly logical and well-supported
    Sound,        // Generally valid reasoning
    Adequate,     // Acceptable but could be stronger
    Questionable, // Some logical issues
    Flawed,       // Significant logical problems
}

impl ReasoningChainAnalyzer {
    pub fn new() -> Self {
        Self {
            logical_rules: LogicalRule::load_standard_rules(),
            fallacy_detectors: FallacyDetector::load_standard_detectors(),
            validity_checkers: ValidityChecker::load_standard_checkers(),
            analysis_cache: HashMap::new(),
            common_patterns: ReasoningPattern::load_common_patterns(),
            known_fallacies: KnownFallacy::load_known_fallacies(),
        }
    }
    
    pub async fn analyze_reasoning_chain(
        &mut self,
        path: &JustificationPath,
    ) -> Result<ReasoningAnalysis, ReasoningAnalysisError> {
        // Check cache first
        if let Some(cached) = self.analysis_cache.get(&path.path_id) {
            return Ok(cached.clone());
        }
        
        // Analyze each reasoning step
        let mut step_analyses = Vec::new();
        for step in &path.reasoning_steps {
            let step_analysis = self.analyze_reasoning_step(step).await?;
            step_analyses.push(step_analysis);
        }
        
        // Detect logical fallacies
        let detected_fallacies = self.detect_fallacies(path).await?;
        
        // Identify reasoning gaps
        let reasoning_gaps = self.identify_reasoning_gaps(path).await?;
        
        // Calculate strength profile
        let strength_profile = self.calculate_strength_profile(path, &step_analyses).await?;
        
        // Assess overall validity
        let overall_validity = self.assess_overall_validity(&step_analyses, &detected_fallacies).await?;
        
        // Calculate logical strength
        let logical_strength = self.calculate_logical_strength(&step_analyses, &detected_fallacies).await?;
        
        // Determine reasoning quality
        let reasoning_quality = self.determine_reasoning_quality(
            overall_validity.clone(),
            logical_strength,
            &detected_fallacies,
        ).await?;
        
        // Generate improvement suggestions
        let improvement_suggestions = self.generate_improvement_suggestions(
            path,
            &step_analyses,
            &detected_fallacies,
            &reasoning_gaps,
        ).await?;
        
        // Find alternative reasoning approaches
        let alternative_reasoning = self.find_alternative_reasoning(path).await?;
        
        let analysis = ReasoningAnalysis {
            overall_validity,
            logical_strength,
            reasoning_quality,
            step_analyses,
            detected_fallacies,
            reasoning_gaps,
            strength_profile,
            improvement_suggestions,
            alternative_reasoning,
            analysis_confidence: self.calculate_analysis_confidence(path).await?,
            completeness_score: self.calculate_completeness_score(path).await?,
        };
        
        // Cache result
        self.analysis_cache.insert(path.path_id, analysis.clone());
        
        Ok(analysis)
    }
    
    async fn analyze_reasoning_step(
        &self,
        step: &ReasoningStep,
    ) -> Result<StepAnalysis, ReasoningAnalysisError> {
        // Assess validity using multiple checkers
        let validity_assessment = self.assess_step_validity(step).await?;
        
        // Analyze logical form
        let logical_form = self.analyze_logical_form(step).await?;
        
        // Evaluate evidence adequacy
        let evidence_adequacy = self.evaluate_evidence_adequacy(step).await?;
        
        // Assess inference quality
        let inference_quality = self.assess_inference_quality(step).await?;
        
        // Check context appropriateness
        let context_appropriateness = self.check_context_appropriateness(step).await?;
        
        Ok(StepAnalysis {
            step_id: step.step_id,
            validity_assessment,
            logical_form,
            evidence_adequacy,
            inference_quality,
            context_appropriateness,
        })
    }
    
    async fn detect_fallacies(
        &self,
        path: &JustificationPath,
    ) -> Result<Vec<DetectedFallacy>, ReasoningAnalysisError> {
        let mut detected_fallacies = Vec::new();
        
        // Check each fallacy detector
        for detector in &self.fallacy_detectors {
            let fallacies = detector.detect_in_path(path).await?;
            detected_fallacies.extend(fallacies);
        }
        
        // Check for common fallacy patterns
        detected_fallacies.extend(self.detect_pattern_fallacies(path).await?);
        
        // Remove duplicates and rank by confidence
        detected_fallacies.sort_by(|a, b| {
            b.detection_confidence.partial_cmp(&a.detection_confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        detected_fallacies.dedup_by(|a, b| a.fallacy_type == b.fallacy_type);
        
        Ok(detected_fallacies)
    }
    
    async fn identify_reasoning_gaps(
        &self,
        path: &JustificationPath,
    ) -> Result<Vec<ReasoningGap>, ReasoningAnalysisError> {
        let mut gaps = Vec::new();
        
        // Check for missing premises
        for i in 0..path.reasoning_steps.len() {
            let step = &path.reasoning_steps[i];
            let missing_premises = self.find_missing_premises(step).await?;
            
            if !missing_premises.is_empty() {
                gaps.push(ReasoningGap {
                    gap_type: ReasoningGapType::MissingPremises,
                    location: GapLocation::Step(i),
                    description: format!("Missing premises: {:?}", missing_premises),
                    severity: self.assess_gap_severity(&missing_premises).await?,
                    suggested_fixes: self.suggest_premise_fixes(&missing_premises).await?,
                });
            }
        }
        
        // Check for logical leaps
        for i in 1..path.reasoning_steps.len() {
            let prev_step = &path.reasoning_steps[i - 1];
            let curr_step = &path.reasoning_steps[i];
            
            if self.is_logical_leap(prev_step, curr_step).await? {
                gaps.push(ReasoningGap {
                    gap_type: ReasoningGapType::LogicalLeap,
                    location: GapLocation::Between(i - 1, i),
                    description: "Logical leap detected between steps".to_string(),
                    severity: GapSeverity::Medium,
                    suggested_fixes: self.suggest_leap_fixes(prev_step, curr_step).await?,
                });
            }
        }
        
        // Check for insufficient evidence
        for (i, step) in path.reasoning_steps.iter().enumerate() {
            if step.evidence_strength < 0.3 {
                gaps.push(ReasoningGap {
                    gap_type: ReasoningGapType::InsufficientEvidence,
                    location: GapLocation::Step(i),
                    description: format!("Evidence strength too low: {}", step.evidence_strength),
                    severity: GapSeverity::High,
                    suggested_fixes: vec![
                        "Gather additional supporting evidence".to_string(),
                        "Consider alternative reasoning approaches".to_string(),
                    ],
                });
            }
        }
        
        Ok(gaps)
    }
    
    async fn calculate_strength_profile(
        &self,
        path: &JustificationPath,
        step_analyses: &[StepAnalysis],
    ) -> Result<StrengthProfile, ReasoningAnalysisError> {
        let step_strengths: Vec<f32> = step_analyses.iter()
            .map(|analysis| self.calculate_step_strength(analysis))
            .collect();
        
        let overall_strength = if step_strengths.is_empty() {
            0.0
        } else {
            step_strengths.iter().sum::<f32>() / step_strengths.len() as f32
        };
        
        let strength_variance = if step_strengths.len() > 1 {
            let mean = overall_strength;
            step_strengths.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f32>() / step_strengths.len() as f32
        } else {
            0.0
        };
        
        let weakest_step = step_strengths.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);
        
        let strongest_step = step_strengths.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);
        
        Ok(StrengthProfile {
            overall_strength,
            strength_variance,
            step_strengths,
            weakest_step,
            strongest_step,
            consistency_score: 1.0 - strength_variance,
        })
    }
    
    fn calculate_step_strength(&self, analysis: &StepAnalysis) -> f32 {
        let validity_score = match analysis.validity_assessment.overall_validity {
            LogicalValidity::Valid => 1.0,
            LogicalValidity::Probable => 0.8,
            LogicalValidity::Uncertain => 0.5,
            LogicalValidity::Invalid => 0.2,
            LogicalValidity::Fallacious => 0.0,
        };
        
        let evidence_score = match analysis.evidence_adequacy {
            EvidenceAdequacy::Strong => 1.0,
            EvidenceAdequacy::Adequate => 0.8,
            EvidenceAdequacy::Weak => 0.4,
            EvidenceAdequacy::Insufficient => 0.1,
        };
        
        let inference_score = match analysis.inference_quality {
            InferenceQuality::Excellent => 1.0,
            InferenceQuality::Good => 0.8,
            InferenceQuality::Fair => 0.6,
            InferenceQuality::Poor => 0.3,
        };
        
        (validity_score + evidence_score + inference_score + analysis.context_appropriateness) / 4.0
    }
}

#[derive(Debug, Clone)]
pub struct DetectedFallacy {
    pub fallacy_type: FallacyType,
    pub location: FallacyLocation,
    pub description: String,
    pub detection_confidence: f32,
    pub severity: FallacySeverity,
    pub examples: Vec<String>,
    pub correction_suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FallacyType {
    AdHominem,
    StrawMan,
    FalseEquivalence,
    SlipperySlope,
    CircularReasoning,
    AppealToAuthority,
    AppealToEmotion,
    FalseDichotomy,
    HastyGeneralization,
    PostHocErgoProterHoc,
}

#[derive(Debug, Clone)]
pub struct ReasoningGap {
    pub gap_type: ReasoningGapType,
    pub location: GapLocation,
    pub description: String,
    pub severity: GapSeverity,
    pub suggested_fixes: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ReasoningGapType {
    MissingPremises,
    LogicalLeap,
    InsufficientEvidence,
    UnstatedAssumptions,
    ContextualMisalignment,
}

#[derive(Debug, Clone)]
pub struct StrengthProfile {
    pub overall_strength: f32,
    pub strength_variance: f32,
    pub step_strengths: Vec<f32>,
    pub weakest_step: Option<usize>,
    pub strongest_step: Option<usize>,
    pub consistency_score: f32,
}
```

### Step 3: Path Explanation Generator

```rust
// File: src/core/justification/explanation_generator.rs

use std::collections::HashMap;
use crate::core::justification::path_builder::{JustificationPath, ReasoningStep};
use crate::core::justification::reasoning_analyzer::ReasoningAnalysis;

pub struct PathExplanationGenerator {
    // Generation configuration
    pub explanation_style: ExplanationStyle,
    pub complexity_level: ComplexityLevel,
    pub include_technical_details: bool,
    pub include_confidence_levels: bool,
    
    // Templates and patterns
    pub explanation_templates: HashMap<ReasoningStepType, ExplanationTemplate>,
    pub natural_language_patterns: Vec<LanguagePattern>,
    
    // Customization
    pub domain_vocabulary: DomainVocabulary,
    pub user_expertise_level: ExpertiseLevel,
}

#[derive(Debug, Clone)]
pub enum ExplanationStyle {
    Narrative,      // Story-like explanation
    Logical,        // Step-by-step logical progression
    Conversational, // Question-answer style
    Technical,      // Detailed technical explanation
    Visual,         // Diagram-based explanation
}

#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Simple,         // Basic explanation
    Intermediate,   // Moderate detail
    Advanced,       // Full technical detail
    Expert,         // All nuances included
}

#[derive(Debug, Clone)]
pub struct PathExplanation {
    // Main explanation content
    pub title: String,
    pub summary: String,
    pub detailed_explanation: String,
    
    // Structured components
    pub reasoning_steps_explained: Vec<StepExplanation>,
    pub key_insights: Vec<String>,
    pub assumptions_made: Vec<String>,
    
    // Quality indicators
    pub explanation_confidence: f32,
    pub completeness_level: ComplexityLevel,
    pub clarity_score: f32,
    
    // Supporting materials
    pub visual_diagram: Option<VisualDiagram>,
    pub alternative_explanations: Vec<AlternativeExplanation>,
    pub related_concepts: Vec<String>,
    
    // Interactive elements
    pub expandable_sections: Vec<ExpandableSection>,
    pub drill_down_options: Vec<DrillDownOption>,
}

#[derive(Debug, Clone)]
pub struct StepExplanation {
    pub step_number: usize,
    pub short_description: String,
    pub detailed_description: String,
    pub reasoning_type: String,
    pub confidence_level: String,
    pub key_concepts: Vec<String>,
    pub potential_issues: Vec<String>,
}

impl PathExplanationGenerator {
    pub fn new() -> Self {
        Self {
            explanation_style: ExplanationStyle::Logical,
            complexity_level: ComplexityLevel::Intermediate,
            include_technical_details: false,
            include_confidence_levels: true,
            explanation_templates: ExplanationTemplate::load_default_templates(),
            natural_language_patterns: LanguagePattern::load_patterns(),
            domain_vocabulary: DomainVocabulary::default(),
            user_expertise_level: ExpertiseLevel::Intermediate,
        }
    }
    
    pub async fn generate_explanation(
        &self,
        path: &JustificationPath,
        analysis: Option<&ReasoningAnalysis>,
    ) -> Result<PathExplanation, ExplanationError> {
        // Generate title and summary
        let title = self.generate_title(path).await?;
        let summary = self.generate_summary(path, analysis).await?;
        
        // Generate step-by-step explanations
        let step_explanations = self.generate_step_explanations(path).await?;
        
        // Generate detailed explanation
        let detailed_explanation = self.generate_detailed_explanation(
            path,
            &step_explanations,
            analysis,
        ).await?;
        
        // Extract key insights
        let key_insights = self.extract_key_insights(path, analysis).await?;
        
        // Identify assumptions
        let assumptions_made = self.identify_assumptions(path).await?;
        
        // Generate visual diagram if requested
        let visual_diagram = if matches!(self.explanation_style, ExplanationStyle::Visual) {
            Some(self.generate_visual_diagram(path).await?)
        } else {
            None
        };
        
        // Create alternative explanations
        let alternative_explanations = self.generate_alternative_explanations(path).await?;
        
        // Calculate explanation metrics
        let explanation_confidence = self.calculate_explanation_confidence(path, analysis).await?;
        let clarity_score = self.calculate_clarity_score(&detailed_explanation).await?;
        
        Ok(PathExplanation {
            title,
            summary,
            detailed_explanation,
            reasoning_steps_explained: step_explanations,
            key_insights,
            assumptions_made,
            explanation_confidence,
            completeness_level: self.complexity_level.clone(),
            clarity_score,
            visual_diagram,
            alternative_explanations,
            related_concepts: self.find_related_concepts(path).await?,
            expandable_sections: self.create_expandable_sections(path).await?,
            drill_down_options: self.create_drill_down_options(path).await?,
        })
    }
    
    async fn generate_step_explanations(
        &self,
        path: &JustificationPath,
    ) -> Result<Vec<StepExplanation>, ExplanationError> {
        let mut step_explanations = Vec::new();
        
        for (i, step) in path.reasoning_steps.iter().enumerate() {
            let step_explanation = self.generate_single_step_explanation(step, i + 1).await?;
            step_explanations.push(step_explanation);
        }
        
        Ok(step_explanations)
    }
    
    async fn generate_single_step_explanation(
        &self,
        step: &ReasoningStep,
        step_number: usize,
    ) -> Result<StepExplanation, ExplanationError> {
        // Get appropriate template for this reasoning type
        let template = self.explanation_templates.get(&step.step_type)
            .unwrap_or(&ExplanationTemplate::default());
        
        // Generate descriptions
        let short_description = self.generate_short_description(step, template).await?;
        let detailed_description = self.generate_detailed_description(step, template).await?;
        
        // Convert reasoning type to human-readable string
        let reasoning_type = self.format_reasoning_type(&step.step_type);
        
        // Format confidence level
        let confidence_level = self.format_confidence_level(step.confidence_level);
        
        // Extract key concepts
        let key_concepts = self.extract_step_concepts(step).await?;
        
        // Identify potential issues
        let potential_issues = self.identify_step_issues(step).await?;
        
        Ok(StepExplanation {
            step_number,
            short_description,
            detailed_description,
            reasoning_type,
            confidence_level,
            key_concepts,
            potential_issues,
        })
    }
    
    async fn generate_detailed_explanation(
        &self,
        path: &JustificationPath,
        step_explanations: &[StepExplanation],
        analysis: Option<&ReasoningAnalysis>,
    ) -> Result<String, ExplanationError> {
        let mut explanation = String::new();
        
        // Introduction
        explanation.push_str(&format!(
            "This reasoning path consists of {} steps that lead to the conclusion. ",
            step_explanations.len()
        ));
        
        if let Some(analysis) = analysis {
            explanation.push_str(&format!(
                "The overall reasoning quality is {:?} with a logical strength of {:.2}. ",
                analysis.reasoning_quality,
                analysis.logical_strength
            ));
        }
        
        explanation.push_str("\n\n");
        
        // Step-by-step explanation
        for (i, step_explanation) in step_explanations.iter().enumerate() {
            explanation.push_str(&format!(
                "**Step {}**: {}\n\n",
                i + 1,
                step_explanation.detailed_description
            ));
            
            if self.include_confidence_levels {
                explanation.push_str(&format!(
                    "*Confidence: {}*\n\n",
                    step_explanation.confidence_level
                ));
            }
            
            if !step_explanation.potential_issues.is_empty() {
                explanation.push_str("⚠️ **Potential Issues**: ");
                explanation.push_str(&step_explanation.potential_issues.join(", "));
                explanation.push_str("\n\n");
            }
        }
        
        // Analysis summary if available
        if let Some(analysis) = analysis {
            if !analysis.detected_fallacies.is_empty() {
                explanation.push_str("**Detected Issues**:\n");
                for fallacy in &analysis.detected_fallacies {
                    explanation.push_str(&format!(
                        "- {}: {}\n",
                        format!("{:?}", fallacy.fallacy_type),
                        fallacy.description
                    ));
                }
                explanation.push_str("\n");
            }
            
            if !analysis.improvement_suggestions.is_empty() {
                explanation.push_str("**Suggestions for Improvement**:\n");
                for suggestion in &analysis.improvement_suggestions {
                    explanation.push_str(&format!(
                        "- {}\n",
                        suggestion.description
                    ));
                }
            }
        }
        
        Ok(explanation)
    }
    
    fn format_confidence_level(&self, confidence: f32) -> String {
        match confidence {
            c if c >= 0.9 => "Very High".to_string(),
            c if c >= 0.7 => "High".to_string(),
            c if c >= 0.5 => "Medium".to_string(),
            c if c >= 0.3 => "Low".to_string(),
            _ => "Very Low".to_string(),
        }
    }
    
    fn format_reasoning_type(&self, reasoning_type: &ReasoningStepType) -> String {
        match reasoning_type {
            ReasoningStepType::Deductive => "Logical Deduction".to_string(),
            ReasoningStepType::Inductive => "Inductive Reasoning".to_string(),
            ReasoningStepType::Abductive => "Best Explanation".to_string(),
            ReasoningStepType::Analogical => "Reasoning by Analogy".to_string(),
            ReasoningStepType::Default => "Default Reasoning".to_string(),
            ReasoningStepType::Temporal => "Time-based Inference".to_string(),
            ReasoningStepType::Causal => "Cause-Effect Reasoning".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisualDiagram {
    pub diagram_type: DiagramType,
    pub nodes: Vec<DiagramNode>,
    pub edges: Vec<DiagramEdge>,
    pub layout_info: LayoutInfo,
}

#[derive(Debug, Clone)]
pub enum DiagramType {
    FlowChart,
    Tree,
    Network,
    Timeline,
}

#[derive(Debug, Clone)]
pub struct AlternativeExplanation {
    pub title: String,
    pub description: String,
    pub explanation_style: ExplanationStyle,
    pub suitability_score: f32,
}
```

### Step 4: Justification Validator

```rust
// File: src/core/justification/validator.rs

use std::collections::{HashMap, HashSet};
use crate::core::justification::path_builder::{JustificationPath, ReasoningStep};
use crate::tms::TruthMaintenanceSystem;

pub struct JustificationValidator {
    // Validation rules
    pub logical_rules: Vec<LogicalValidationRule>,
    pub temporal_rules: Vec<TemporalValidationRule>,
    pub contextual_rules: Vec<ContextualValidationRule>,
    
    // Validation cache
    pub validation_cache: HashMap<JustificationPathId, ValidationResult>,
    
    // Configuration
    pub strict_mode: bool,
    pub check_temporal_consistency: bool,
    pub validate_context_requirements: bool,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    // Overall validation
    pub is_valid: bool,
    pub validation_confidence: f32,
    pub overall_score: f32,
    
    // Detailed results
    pub step_validations: Vec<StepValidationResult>,
    pub logical_issues: Vec<LogicalIssue>,
    pub temporal_issues: Vec<TemporalIssue>,
    pub contextual_issues: Vec<ContextualIssue>,
    
    // Recommendations
    pub validation_warnings: Vec<ValidationWarning>,
    pub correction_suggestions: Vec<CorrectionSuggestion>,
    
    // Metadata
    pub validation_timestamp: Timestamp,
    pub validation_method: ValidationMethod,
}

impl JustificationValidator {
    pub async fn validate_justification_path(
        &mut self,
        path: &JustificationPath,
        tms: &TruthMaintenanceSystem,
    ) -> Result<ValidationResult, ValidationError> {
        // Check cache
        if let Some(cached) = self.validation_cache.get(&path.path_id) {
            return Ok(cached.clone());
        }
        
        // Validate each step
        let mut step_validations = Vec::new();
        for step in &path.reasoning_steps {
            let step_validation = self.validate_reasoning_step(step, tms).await?;
            step_validations.push(step_validation);
        }
        
        // Check overall logical consistency
        let logical_issues = self.check_logical_consistency(path).await?;
        
        // Validate temporal aspects
        let temporal_issues = if self.check_temporal_consistency {
            self.check_temporal_validity(path, tms).await?
        } else {
            Vec::new()
        };
        
        // Validate contextual requirements
        let contextual_issues = if self.validate_context_requirements {
            self.check_contextual_validity(path, tms).await?
        } else {
            Vec::new()
        };
        
        // Calculate overall validation score
        let overall_score = self.calculate_validation_score(
            &step_validations,
            &logical_issues,
            &temporal_issues,
            &contextual_issues,
        ).await?;
        
        // Determine if path is valid
        let is_valid = overall_score >= 0.7 && 
            logical_issues.iter().all(|issue| issue.severity != IssueSeverity::Critical) &&
            temporal_issues.iter().all(|issue| issue.severity != IssueSeverity::Critical);
        
        // Generate warnings and suggestions
        let validation_warnings = self.generate_validation_warnings(
            &logical_issues,
            &temporal_issues,
            &contextual_issues,
        ).await?;
        
        let correction_suggestions = self.generate_correction_suggestions(
            path,
            &logical_issues,
            &temporal_issues,
            &contextual_issues,
        ).await?;
        
        let result = ValidationResult {
            is_valid,
            validation_confidence: self.calculate_validation_confidence(&step_validations).await?,
            overall_score,
            step_validations,
            logical_issues,
            temporal_issues,
            contextual_issues,
            validation_warnings,
            correction_suggestions,
            validation_timestamp: Timestamp::now(),
            validation_method: if self.strict_mode {
                ValidationMethod::Strict
            } else {
                ValidationMethod::Standard
            },
        };
        
        // Cache result
        self.validation_cache.insert(path.path_id, result.clone());
        
        Ok(result)
    }
    
    async fn validate_reasoning_step(
        &self,
        step: &ReasoningStep,
        tms: &TruthMaintenanceSystem,
    ) -> Result<StepValidationResult, ValidationError> {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        
        // Check inference rule validity
        if !self.is_valid_inference_rule(&step.inference_rule) {
            issues.push(StepIssue {
                issue_type: StepIssueType::InvalidInferenceRule,
                description: format!("Invalid inference rule: {:?}", step.inference_rule),
                severity: IssueSeverity::High,
            });
        }
        
        // Check premise-conclusion relationship
        if !self.check_premise_conclusion_validity(step).await? {
            issues.push(StepIssue {
                issue_type: StepIssueType::InvalidInference,
                description: "Conclusion does not follow from premises".to_string(),
                severity: IssueSeverity::Critical,
            });
        }
        
        // Check confidence level reasonableness
        if step.confidence_level < 0.1 && step.logical_validity == LogicalValidity::Valid {
            warnings.push(StepWarning {
                warning_type: StepWarningType::ConfidenceMismatch,
                description: "Very low confidence for apparently valid step".to_string(),
            });
        }
        
        // Check evidence adequacy
        if step.evidence_strength < 0.2 {
            issues.push(StepIssue {
                issue_type: StepIssueType::InsufficientEvidence,
                description: "Evidence strength too low for reliable inference".to_string(),
                severity: IssueSeverity::Medium,
            });
        }
        
        // Check belief basis in TMS
        for &belief_id in &step.belief_basis {
            let belief_status = tms.get_belief_status_in_context(
                belief_id,
                &step.context_requirements,
            ).await?;
            
            if belief_status == BeliefStatus::Out {
                issues.push(StepIssue {
                    issue_type: StepIssueType::RetractedBelief,
                    description: format!("Step relies on retracted belief: {:?}", belief_id),
                    severity: IssueSeverity::Critical,
                });
            }
        }
        
        // Calculate step validation score
        let step_score = if issues.iter().any(|i| i.severity == IssueSeverity::Critical) {
            0.0
        } else {
            1.0 - (issues.len() as f32 * 0.2) - (warnings.len() as f32 * 0.1)
        };
        
        Ok(StepValidationResult {
            step_id: step.step_id,
            is_valid: step_score >= 0.6,
            validation_score: step_score.max(0.0),
            issues,
            warnings,
        })
    }
}

#[derive(Debug, Clone)]
pub struct StepValidationResult {
    pub step_id: ReasoningStepId,
    pub is_valid: bool,
    pub validation_score: f32,
    pub issues: Vec<StepIssue>,
    pub warnings: Vec<StepWarning>,
}

#[derive(Debug, Clone)]
pub struct LogicalIssue {
    pub issue_type: LogicalIssueType,
    pub description: String,
    pub severity: IssueSeverity,
    pub affected_steps: Vec<usize>,
    pub correction_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub enum LogicalIssueType {
    CircularReasoning,
    InvalidInference,
    MissingPremise,
    InconsistentPremises,
    LogicalFallacy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub enum ValidationMethod {
    Standard,
    Strict,
    Permissive,
    Custom,
}
```

## File Locations

- `src/core/justification/path_builder.rs` - Justification path construction
- `src/core/justification/reasoning_analyzer.rs` - Reasoning chain analysis
- `src/core/justification/explanation_generator.rs` - Human-readable explanations
- `src/core/justification/validator.rs` - Path validation and verification
- `tests/justification/justification_tests.rs` - Comprehensive test suite

## Success Criteria

- [ ] JustificationPathBuilder constructs accurate reasoning paths
- [ ] ReasoningChainAnalyzer detects logical fallacies and gaps
- [ ] PathExplanationGenerator creates clear, understandable explanations
- [ ] JustificationValidator ensures logical consistency
- [ ] Path construction completes in <2ms for 10-step chains
- [ ] Fallacy detection accuracy >90% for known patterns
- [ ] Explanation quality rated as "Good" or better by users
- [ ] All tests pass:
  - Path construction and ranking
  - Logical fallacy detection
  - Reasoning gap identification
  - Explanation generation quality
  - Validation accuracy
  - Performance benchmarks

## Test Requirements

```rust
#[test]
async fn test_justification_path_construction() {
    let tms = setup_test_tms().await;
    let activation_state = setup_test_activation_state().await;
    let mut builder = JustificationPathBuilder::new();
    
    let target_nodes = vec![NodeId(5), NodeId(7)];
    
    let path_collection = builder.build_justification_paths(
        &activation_state,
        target_nodes,
        &tms,
    ).await.unwrap();
    
    assert!(!path_collection.all_paths.is_empty());
    
    // Verify paths have valid structure
    for path in &path_collection.all_paths {
        assert!(path.reasoning_steps.len() <= builder.max_path_depth);
        assert!(path.reliability_score >= 0.0 && path.reliability_score <= 1.0);
        assert!(!path.reasoning_steps.is_empty() || path.path_length == 0);
    }
}

#[test]
async fn test_fallacy_detection() {
    let mut analyzer = ReasoningChainAnalyzer::new();
    
    // Create a path with known circular reasoning
    let circular_path = create_circular_reasoning_path();
    
    let analysis = analyzer.analyze_reasoning_chain(&circular_path)
        .await.unwrap();
    
    // Should detect circular reasoning
    assert!(analysis.detected_fallacies.iter()
        .any(|f| f.fallacy_type == FallacyType::CircularReasoning));
    
    assert!(analysis.reasoning_quality == ReasoningQuality::Flawed ||
           analysis.reasoning_quality == ReasoningQuality::Questionable);
}

#[test]
async fn test_explanation_generation() {
    let generator = PathExplanationGenerator::new();
    let test_path = create_test_justification_path();
    
    let explanation = generator.generate_explanation(&test_path, None)
        .await.unwrap();
    
    assert!(!explanation.title.is_empty());
    assert!(!explanation.summary.is_empty());
    assert!(!explanation.detailed_explanation.is_empty());
    assert_eq!(explanation.reasoning_steps_explained.len(), test_path.reasoning_steps.len());
    
    // Check explanation quality
    assert!(explanation.clarity_score > 0.5);
    assert!(explanation.explanation_confidence > 0.0);
}

#[test]
async fn test_path_validation() {
    let tms = setup_test_tms().await;
    let mut validator = JustificationValidator::new();
    
    // Test valid path
    let valid_path = create_valid_justification_path();
    let result = validator.validate_justification_path(&valid_path, &tms)
        .await.unwrap();
    
    assert!(result.is_valid);
    assert!(result.overall_score > 0.7);
    assert!(result.logical_issues.iter()
        .all(|issue| issue.severity != IssueSeverity::Critical));
    
    // Test invalid path
    let invalid_path = create_invalid_justification_path();
    let result = validator.validate_justification_path(&invalid_path, &tms)
        .await.unwrap();
    
    assert!(!result.is_valid);
    assert!(result.overall_score < 0.7);
}

#[test]
async fn test_reasoning_gap_identification() {
    let analyzer = ReasoningChainAnalyzer::new();
    
    // Create path with missing premises
    let gapped_path = create_path_with_missing_premises();
    
    let analysis = analyzer.analyze_reasoning_chain(&gapped_path)
        .await.unwrap();
    
    // Should identify missing premises
    assert!(analysis.reasoning_gaps.iter()
        .any(|gap| gap.gap_type == ReasoningGapType::MissingPremises));
    
    // Should provide suggestions
    assert!(!analysis.improvement_suggestions.is_empty());
}
```

## Quality Gates

- [ ] Path construction latency <2ms for 10-step chains
- [ ] Logical fallacy detection precision >85%, recall >90%
- [ ] Explanation clarity score >0.7 for generated explanations
- [ ] Validation accuracy >95% for known valid/invalid patterns
- [ ] Memory usage scales linearly with path complexity
- [ ] No false positives for well-formed logical arguments

## Next Task

Upon completion, proceed to **35_belief_tests.md**