# Micro Task 29: Explanation Quality Assessment

**Priority**: CRITICAL  
**Estimated Time**: 30 minutes  
**Dependencies**: 28_evidence_collection.md completed  
**Skills Required**: Quality metrics, human-AI interaction, evaluation frameworks

## Objective

Implement comprehensive explanation quality assessment system that evaluates explanations across multiple dimensions to ensure they meet user needs and provide meaningful insights into the AI's reasoning process.

## Context

Quality assessment is essential for ensuring explanations are useful, accurate, and trustworthy. This task creates metrics and evaluation frameworks to assess explanation effectiveness and guide continuous improvement of the explanation generation system.

## Specifications

### Core Quality Components

1. **QualityAssessor struct**
   - Multi-dimensional quality evaluation
   - Real-time quality scoring
   - Comparative quality analysis
   - Quality trend tracking

2. **QualityMetrics struct**
   - Comprehensive quality dimensions
   - Weighted scoring system
   - Contextual quality assessment
   - User preference integration

3. **QualityValidator struct**
   - Quality threshold enforcement
   - Quality improvement suggestions
   - Outlier detection
   - Quality calibration

4. **QualityReporter struct**
   - Quality analytics and reporting
   - Quality trend analysis
   - Performance insights
   - Recommendation generation

### Performance Requirements

- Quality assessment < 10ms per explanation
- Real-time quality scoring
- Memory efficient quality tracking
- Scalable quality analytics
- Consistent quality standards

## Implementation Guide

### Step 1: Core Quality Types

```rust
// File: src/cognitive/explanation/quality_assessment.rs

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::cognitive::explanation::templates::ExplanationContext;
use crate::cognitive::explanation::reasoning_extraction::{ReasoningChain, ReasoningAnalysis};
use crate::cognitive::explanation::evidence_collection::{Evidence, EvidenceCollection};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub clarity: f32,
    pub completeness: f32,
    pub accuracy: f32,
    pub relevance: f32,
    pub coherence: f32,
    pub confidence_calibration: f32,
    pub evidence_support: f32,
    pub user_comprehension: f32,
    pub actionability: f32,
    pub trustworthiness: f32,
    pub overall_quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub assessment_id: AssessmentId,
    pub explanation_text: String,
    pub explanation_context: String,
    pub metrics: QualityMetrics,
    pub detailed_scores: DetailedScores,
    pub quality_issues: Vec<QualityIssue>,
    pub improvement_suggestions: Vec<ImprovementSuggestion>,
    pub assessment_metadata: AssessmentMetadata,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssessmentId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedScores {
    pub linguistic_quality: LinguisticQuality,
    pub content_quality: ContentQuality,
    pub structural_quality: StructuralQuality,
    pub contextual_quality: ContextualQuality,
    pub technical_quality: TechnicalQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticQuality {
    pub readability: f32,
    pub vocabulary_appropriateness: f32,
    pub grammar_correctness: f32,
    pub style_consistency: f32,
    pub conciseness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentQuality {
    pub factual_accuracy: f32,
    pub logical_consistency: f32,
    pub information_completeness: f32,
    pub bias_minimization: f32,
    pub uncertainty_handling: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralQuality {
    pub organization: f32,
    pub flow: f32,
    pub transitions: f32,
    pub emphasis: f32,
    pub formatting: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualQuality {
    pub audience_appropriateness: f32,
    pub domain_relevance: f32,
    pub cultural_sensitivity: f32,
    pub temporal_relevance: f32,
    pub purpose_alignment: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalQuality {
    pub reasoning_trace_fidelity: f32,
    pub system_transparency: f32,
    pub limitation_acknowledgment: f32,
    pub verification_support: f32,
    pub reproducibility: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_id: String,
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub location: Option<TextLocation>,
    pub impact_assessment: ImpactAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    Clarity,
    Accuracy,
    Completeness,
    Relevance,
    Coherence,
    Evidence,
    Confidence,
    Bias,
    Comprehension,
    Technical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLocation {
    pub start_char: usize,
    pub end_char: usize,
    pub sentence_index: Option<usize>,
    pub paragraph_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub user_experience_impact: f32,
    pub trust_impact: f32,
    pub understanding_impact: f32,
    pub decision_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementSuggestion {
    pub suggestion_id: String,
    pub suggestion_type: SuggestionType,
    pub priority: SuggestionPriority,
    pub description: String,
    pub expected_improvement: f32,
    pub implementation_difficulty: ImplementationDifficulty,
    pub specific_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    ContentRevision,
    StructuralReorganization,
    EvidenceEnhancement,
    ClarityImprovement,
    ConfidenceCalibration,
    BiasReduction,
    TechnicalCorrection,
    AudienceAdaptation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Easy,
    Moderate,
    Difficult,
    VeryDifficult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentMetadata {
    pub assessor_version: String,
    pub assessment_method: AssessmentMethod,
    pub baseline_comparisons: Vec<BaselineComparison>,
    pub assessment_duration: Duration,
    pub confidence_in_assessment: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentMethod {
    Automated,
    HybridHumanAI,
    ExpertEvaluation,
    UserFeedback,
    ComparativeAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_type: BaselineType,
    pub baseline_score: f32,
    pub comparison_result: f32,
    pub statistical_significance: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineType {
    TemplateBaseline,
    PreviousVersion,
    HumanExpert,
    CompetitorSystem,
    RandomBaseline,
}

#[derive(Debug)]
pub struct QualityAssessor {
    assessment_config: AssessmentConfig,
    quality_calculators: HashMap<String, QualityCalculator>,
    assessment_history: VecDeque<QualityAssessment>,
    baseline_repository: BaselineRepository,
    next_assessment_id: u64,
}

#[derive(Debug, Clone)]
pub struct AssessmentConfig {
    pub quality_weights: HashMap<String, f32>,
    pub quality_thresholds: HashMap<String, f32>,
    pub enable_detailed_analysis: bool,
    pub enable_comparative_assessment: bool,
    pub max_assessment_history: usize,
    pub quality_calibration_enabled: bool,
}

#[derive(Debug)]
pub struct QualityCalculator {
    pub calculator_name: String,
    pub calculation_method: CalculationMethod,
    pub normalization_strategy: NormalizationStrategy,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Debug)]
pub enum CalculationMethod {
    RuleBased,
    StatisticalAnalysis,
    MachineLearning,
    HeuristicCombination,
    ExternalAPI,
}

#[derive(Debug)]
pub enum NormalizationStrategy {
    MinMax,
    ZScore,
    Sigmoid,
    Linear,
    Percentile,
}

#[derive(Debug)]
pub struct BaselineRepository {
    baselines: HashMap<String, BaselineData>,
    baseline_metadata: HashMap<String, BaselineMetadata>,
}

#[derive(Debug, Clone)]
pub struct BaselineData {
    pub baseline_id: String,
    pub baseline_type: BaselineType,
    pub quality_metrics: QualityMetrics,
    pub sample_count: usize,
    pub last_updated: Instant,
}

#[derive(Debug, Clone)]
pub struct BaselineMetadata {
    pub creation_date: Instant,
    pub data_source: String,
    pub validation_status: ValidationStatus,
    pub confidence_level: f32,
    pub update_frequency: UpdateFrequency,
}

#[derive(Debug, Clone)]
pub enum ValidationStatus {
    Validated,
    Pending,
    InvalidData,
    Outdated,
}

#[derive(Debug, Clone)]
pub enum UpdateFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
    Manual,
}
```

### Step 2: Quality Assessor Implementation

```rust
impl QualityAssessor {
    pub fn new() -> Self {
        let config = AssessmentConfig {
            quality_weights: HashMap::from([
                ("clarity".to_string(), 0.15),
                ("completeness".to_string(), 0.12),
                ("accuracy".to_string(), 0.18),
                ("relevance".to_string(), 0.12),
                ("coherence".to_string(), 0.10),
                ("confidence_calibration".to_string(), 0.08),
                ("evidence_support".to_string(), 0.10),
                ("user_comprehension".to_string(), 0.08),
                ("actionability".to_string(), 0.05),
                ("trustworthiness".to_string(), 0.02),
            ]),
            quality_thresholds: HashMap::from([
                ("minimum_acceptable".to_string(), 0.6),
                ("good_quality".to_string(), 0.75),
                ("excellent_quality".to_string(), 0.9),
            ]),
            enable_detailed_analysis: true,
            enable_comparative_assessment: true,
            max_assessment_history: 1000,
            quality_calibration_enabled: true,
        };
        
        Self {
            assessment_config: config,
            quality_calculators: Self::create_default_calculators(),
            assessment_history: VecDeque::new(),
            baseline_repository: BaselineRepository::new(),
            next_assessment_id: 1,
        }
    }
    
    pub fn assess_explanation_quality(
        &mut self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
        context: &ExplanationContext,
    ) -> Result<QualityAssessment, QualityAssessmentError> {
        let assessment_id = AssessmentId(self.next_assessment_id);
        self.next_assessment_id += 1;
        
        let start_time = Instant::now();
        
        // Calculate core quality metrics
        let metrics = self.calculate_quality_metrics(
            explanation_text,
            reasoning_chain,
            evidence_collection,
            context,
        )?;
        
        // Calculate detailed scores
        let detailed_scores = self.calculate_detailed_scores(
            explanation_text,
            reasoning_chain,
            evidence_collection,
            context,
        )?;
        
        // Identify quality issues
        let quality_issues = self.identify_quality_issues(
            explanation_text,
            &metrics,
            &detailed_scores,
            reasoning_chain,
        )?;
        
        // Generate improvement suggestions
        let improvement_suggestions = self.generate_improvement_suggestions(
            &quality_issues,
            &metrics,
            explanation_text,
        )?;
        
        // Create assessment metadata
        let assessment_metadata = AssessmentMetadata {
            assessor_version: "1.0".to_string(),
            assessment_method: AssessmentMethod::Automated,
            baseline_comparisons: self.perform_baseline_comparisons(&metrics)?,
            assessment_duration: start_time.elapsed(),
            confidence_in_assessment: self.calculate_assessment_confidence(&metrics),
        };
        
        let assessment = QualityAssessment {
            assessment_id,
            explanation_text: explanation_text.to_string(),
            explanation_context: context.query.clone(),
            metrics,
            detailed_scores,
            quality_issues,
            improvement_suggestions,
            assessment_metadata,
            timestamp: Instant::now(),
        };
        
        // Store assessment in history
        self.store_assessment_in_history(assessment.clone());
        
        Ok(assessment)
    }
    
    fn calculate_quality_metrics(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
        context: &ExplanationContext,
    ) -> Result<QualityMetrics, QualityAssessmentError> {
        let clarity = self.calculate_clarity(explanation_text, context)?;
        let completeness = self.calculate_completeness(explanation_text, reasoning_chain, context)?;
        let accuracy = self.calculate_accuracy(explanation_text, reasoning_chain, evidence_collection)?;
        let relevance = self.calculate_relevance(explanation_text, context)?;
        let coherence = self.calculate_coherence(explanation_text, reasoning_chain)?;
        let confidence_calibration = self.calculate_confidence_calibration(explanation_text, reasoning_chain)?;
        let evidence_support = self.calculate_evidence_support(explanation_text, evidence_collection)?;
        let user_comprehension = self.calculate_user_comprehension(explanation_text, context)?;
        let actionability = self.calculate_actionability(explanation_text, context)?;
        let trustworthiness = self.calculate_trustworthiness(explanation_text, reasoning_chain, evidence_collection)?;
        
        // Calculate weighted overall quality
        let overall_quality = self.calculate_weighted_overall_quality(&[
            ("clarity", clarity),
            ("completeness", completeness),
            ("accuracy", accuracy),
            ("relevance", relevance),
            ("coherence", coherence),
            ("confidence_calibration", confidence_calibration),
            ("evidence_support", evidence_support),
            ("user_comprehension", user_comprehension),
            ("actionability", actionability),
            ("trustworthiness", trustworthiness),
        ]);
        
        Ok(QualityMetrics {
            clarity,
            completeness,
            accuracy,
            relevance,
            coherence,
            confidence_calibration,
            evidence_support,
            user_comprehension,
            actionability,
            trustworthiness,
            overall_quality,
        })
    }
    
    fn calculate_clarity(&self, explanation_text: &str, context: &ExplanationContext) -> Result<f32, QualityAssessmentError> {
        let mut clarity_score = 0.0;
        
        // Sentence length analysis
        let sentences = self.split_into_sentences(explanation_text);
        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            sentences.iter().map(|s| s.split_whitespace().count()).sum::<usize>() as f32 / sentences.len() as f32
        };
        
        // Optimal sentence length is 15-20 words
        let sentence_length_score = if avg_sentence_length >= 10.0 && avg_sentence_length <= 25.0 {
            1.0 - (avg_sentence_length - 17.5).abs() / 12.5
        } else if avg_sentence_length < 5.0 {
            0.3
        } else {
            0.1
        };
        clarity_score += sentence_length_score * 0.3;
        
        // Vocabulary complexity
        let complex_words = self.count_complex_words(explanation_text);
        let total_words = explanation_text.split_whitespace().count();
        let complexity_ratio = if total_words > 0 {
            complex_words as f32 / total_words as f32
        } else {
            0.0
        };
        
        // Lower complexity ratio = higher clarity (for general audience)
        let vocabulary_score = match context.query_type.as_str() {
            "technical" => (complexity_ratio * 2.0).min(1.0), // Technical content can be more complex
            _ => (1.0 - complexity_ratio).max(0.0),
        };
        clarity_score += vocabulary_score * 0.3;
        
        // Structural clarity (presence of organizing elements)
        let structure_score = self.assess_structural_clarity(explanation_text);
        clarity_score += structure_score * 0.4;
        
        Ok(clarity_score.min(1.0))
    }
    
    fn calculate_completeness(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        context: &ExplanationContext,
    ) -> Result<f32, QualityAssessmentError> {
        let mut completeness_score = 0.0;
        
        // Check if explanation covers key reasoning steps
        let step_coverage = if reasoning_chain.steps.is_empty() {
            0.5 // Neutral if no reasoning steps
        } else {
            let covered_steps = reasoning_chain.steps.iter()
                .filter(|step| {
                    explanation_text.contains(&step.premise) || 
                    explanation_text.contains(&step.conclusion) ||
                    self.semantic_similarity(&step.conclusion, explanation_text) > 0.3
                })
                .count();
            
            covered_steps as f32 / reasoning_chain.steps.len() as f32
        };
        completeness_score += step_coverage * 0.4;
        
        // Check if explanation addresses the query
        let query_coverage = if explanation_text.to_lowercase().contains(&context.query.to_lowercase()) ||
                                self.semantic_similarity(&context.query, explanation_text) > 0.5 {
            1.0
        } else {
            0.3
        };
        completeness_score += query_coverage * 0.3;
        
        // Check for confidence information
        let confidence_coverage = if explanation_text.contains("confidence") || 
                                     explanation_text.contains("certain") ||
                                     explanation_text.contains("uncertain") ||
                                     explanation_text.contains("likely") {
            1.0
        } else {
            0.6
        };
        completeness_score += confidence_coverage * 0.3;
        
        Ok(completeness_score.min(1.0))
    }
    
    fn calculate_accuracy(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
    ) -> Result<f32, QualityAssessmentError> {
        let mut accuracy_score = 0.0;
        
        // Reasoning chain accuracy (based on chain confidence)
        let reasoning_accuracy = reasoning_chain.confidence_score;
        accuracy_score += reasoning_accuracy * 0.4;
        
        // Evidence support accuracy
        let evidence_accuracy = if evidence_collection.evidence_items.is_empty() {
            0.5 // Neutral if no evidence
        } else {
            evidence_collection.quality_summary.average_confidence
        };
        accuracy_score += evidence_accuracy * 0.4;
        
        // Internal consistency check
        let consistency_score = self.check_internal_consistency(explanation_text);
        accuracy_score += consistency_score * 0.2;
        
        Ok(accuracy_score.min(1.0))
    }
    
    fn calculate_relevance(&self, explanation_text: &str, context: &ExplanationContext) -> Result<f32, QualityAssessmentError> {
        let mut relevance_score = 0.0;
        
        // Query relevance
        let query_relevance = self.semantic_similarity(&context.query, explanation_text);
        relevance_score += query_relevance * 0.5;
        
        // Context relevance (metadata alignment)
        let context_relevance = if !context.metadata.is_empty() {
            let relevant_metadata = context.metadata.iter()
                .filter(|(key, value)| {
                    explanation_text.to_lowercase().contains(&key.to_lowercase()) ||
                    explanation_text.to_lowercase().contains(&value.to_lowercase())
                })
                .count();
            
            relevant_metadata as f32 / context.metadata.len() as f32
        } else {
            0.8 // Neutral if no metadata
        };
        relevance_score += context_relevance * 0.3;
        
        // Entity relevance
        let entity_relevance = if context.entities.is_empty() {
            0.8 // Neutral if no entities
        } else {
            // Simple heuristic: assume explanation is relevant if it mentions reasoning concepts
            if explanation_text.contains("reasoning") || 
               explanation_text.contains("analysis") ||
               explanation_text.contains("conclusion") {
                0.9
            } else {
                0.6
            }
        };
        relevance_score += entity_relevance * 0.2;
        
        Ok(relevance_score.min(1.0))
    }
    
    fn calculate_coherence(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> Result<f32, QualityAssessmentError> {
        let mut coherence_score = 0.0;
        
        // Logical flow assessment
        let logical_flow = self.assess_logical_flow(explanation_text);
        coherence_score += logical_flow * 0.4;
        
        // Transition quality
        let transition_quality = self.assess_transition_quality(explanation_text);
        coherence_score += transition_quality * 0.3;
        
        // Reasoning chain coherence
        let chain_coherence = reasoning_chain.coherence_score;
        coherence_score += chain_coherence * 0.3;
        
        Ok(coherence_score.min(1.0))
    }
    
    fn calculate_confidence_calibration(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
    ) -> Result<f32, QualityAssessmentError> {
        let mut calibration_score = 0.0;
        
        // Check if confidence is mentioned when uncertainty exists
        let chain_confidence = reasoning_chain.confidence_score;
        let mentions_uncertainty = explanation_text.contains("uncertain") || 
                                   explanation_text.contains("might") ||
                                   explanation_text.contains("possibly") ||
                                   explanation_text.contains("likely");
        
        if chain_confidence < 0.7 && mentions_uncertainty {
            calibration_score += 1.0; // Good calibration
        } else if chain_confidence > 0.8 && !mentions_uncertainty {
            calibration_score += 0.9; // Confident and doesn't mention uncertainty
        } else if chain_confidence < 0.5 && !mentions_uncertainty {
            calibration_score += 0.3; // Poor calibration - should mention uncertainty
        } else {
            calibration_score += 0.7; // Neutral calibration
        }
        
        Ok(calibration_score.min(1.0))
    }
    
    fn calculate_evidence_support(
        &self,
        explanation_text: &str,
        evidence_collection: &EvidenceCollection,
    ) -> Result<f32, QualityAssessmentError> {
        if evidence_collection.evidence_items.is_empty() {
            return Ok(0.5); // Neutral if no evidence
        }
        
        let mut support_score = 0.0;
        
        // Evidence quality
        let evidence_quality = evidence_collection.quality_summary.average_quality;
        support_score += evidence_quality * 0.4;
        
        // Evidence mention in explanation
        let evidence_mentioned = evidence_collection.evidence_items.iter()
            .filter(|evidence| {
                explanation_text.contains(&evidence.content.primary_text) ||
                self.semantic_similarity(&evidence.content.primary_text, explanation_text) > 0.3
            })
            .count();
        
        let mention_ratio = evidence_mentioned as f32 / evidence_collection.evidence_items.len() as f32;
        support_score += mention_ratio * 0.4;
        
        // Evidence verification status
        let verified_ratio = evidence_collection.quality_summary.verified_count as f32 / 
                            evidence_collection.evidence_items.len() as f32;
        support_score += verified_ratio * 0.2;
        
        Ok(support_score.min(1.0))
    }
    
    fn calculate_user_comprehension(&self, explanation_text: &str, context: &ExplanationContext) -> Result<f32, QualityAssessmentError> {
        let mut comprehension_score = 0.0;
        
        // Length appropriateness
        let word_count = explanation_text.split_whitespace().count();
        let length_score = match word_count {
            0..=20 => 0.3,
            21..=100 => 1.0,
            101..=300 => 0.9,
            301..=500 => 0.7,
            _ => 0.4,
        };
        comprehension_score += length_score * 0.3;
        
        // Readability (simplified)
        let readability_score = self.calculate_readability(explanation_text);
        comprehension_score += readability_score * 0.4;
        
        // Examples and analogies
        let examples_score = if explanation_text.contains("example") || 
                               explanation_text.contains("like") ||
                               explanation_text.contains("similar to") {
            1.0
        } else {
            0.6
        };
        comprehension_score += examples_score * 0.3;
        
        Ok(comprehension_score.min(1.0))
    }
    
    fn calculate_actionability(&self, explanation_text: &str, context: &ExplanationContext) -> Result<f32, QualityAssessmentError> {
        let mut actionability_score = 0.0;
        
        // Check for actionable language
        let actionable_words = ["should", "can", "try", "consider", "recommend", "suggest"];
        let contains_actionable = actionable_words.iter()
            .any(|&word| explanation_text.to_lowercase().contains(word));
        
        if contains_actionable {
            actionability_score += 0.8;
        } else {
            actionability_score += 0.4; // Explanations don't always need to be actionable
        }
        
        Ok(actionability_score.min(1.0))
    }
    
    fn calculate_trustworthiness(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
    ) -> Result<f32, QualityAssessmentError> {
        let mut trust_score = 0.0;
        
        // Transparency about limitations
        let acknowledges_limitations = explanation_text.contains("limitation") ||
                                       explanation_text.contains("may not") ||
                                       explanation_text.contains("uncertain");
        
        trust_score += if acknowledges_limitations { 0.9 } else { 0.7 };
        
        Ok(trust_score.min(1.0))
    }
    
    fn calculate_weighted_overall_quality(&self, component_scores: &[(&str, f32)]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (component, score) in component_scores {
            if let Some(&weight) = self.assessment_config.quality_weights.get(*component) {
                weighted_sum += score * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
    
    // Helper methods for quality calculations
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(&['.', '!', '?'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
    
    fn count_complex_words(&self, text: &str) -> usize {
        text.split_whitespace()
            .filter(|word| word.len() > 7) // Simple heuristic for complex words
            .count()
    }
    
    fn assess_structural_clarity(&self, text: &str) -> f32 {
        let mut clarity_score = 0.0;
        
        // Check for organizing elements
        if text.contains("first") || text.contains("1.") || text.contains("initially") {
            clarity_score += 0.3;
        }
        if text.contains("second") || text.contains("2.") || text.contains("then") {
            clarity_score += 0.3;
        }
        if text.contains("finally") || text.contains("conclusion") || text.contains("therefore") {
            clarity_score += 0.4;
        }
        
        clarity_score.min(1.0)
    }
    
    fn semantic_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple word overlap similarity
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }
    
    fn check_internal_consistency(&self, text: &str) -> f32 {
        // Simple consistency check - look for contradictory statements
        let contradictory_patterns = [
            ("not", "is"),
            ("never", "always"),
            ("impossible", "possible"),
        ];
        
        for (neg, pos) in contradictory_patterns {
            if text.contains(neg) && text.contains(pos) {
                // Found potential contradiction - reduce consistency score
                return 0.3;
            }
        }
        
        0.9 // Good consistency if no obvious contradictions
    }
    
    fn assess_logical_flow(&self, text: &str) -> f32 {
        // Check for logical connectors
        let connectors = ["because", "therefore", "thus", "consequently", "as a result", "since"];
        let connector_count = connectors.iter()
            .map(|&connector| text.matches(connector).count())
            .sum::<usize>();
        
        let sentences = self.split_into_sentences(text);
        if sentences.len() <= 1 {
            return 0.5;
        }
        
        let connector_ratio = connector_count as f32 / (sentences.len() - 1) as f32;
        connector_ratio.min(1.0)
    }
    
    fn assess_transition_quality(&self, text: &str) -> f32 {
        // Check for transition words
        let transitions = ["however", "moreover", "furthermore", "additionally", "meanwhile", "subsequently"];
        let has_transitions = transitions.iter()
            .any(|&transition| text.contains(transition));
        
        if has_transitions { 0.8 } else { 0.5 }
    }
    
    fn calculate_readability(&self, text: &str) -> f32 {
        // Simplified readability calculation
        let words = text.split_whitespace().count();
        let sentences = self.split_into_sentences(text).len();
        
        if sentences == 0 {
            return 0.0;
        }
        
        let avg_words_per_sentence = words as f32 / sentences as f32;
        
        // Optimal range: 10-20 words per sentence
        if avg_words_per_sentence >= 10.0 && avg_words_per_sentence <= 20.0 {
            1.0 - (avg_words_per_sentence - 15.0).abs() / 10.0
        } else {
            0.5
        }
    }
    
    fn identify_quality_issues(
        &self,
        explanation_text: &str,
        metrics: &QualityMetrics,
        detailed_scores: &DetailedScores,
        reasoning_chain: &ReasoningChain,
    ) -> Result<Vec<QualityIssue>, QualityAssessmentError> {
        let mut issues = Vec::new();
        
        // Check each quality dimension against thresholds
        if metrics.clarity < 0.6 {
            issues.push(QualityIssue {
                issue_id: "clarity_low".to_string(),
                issue_type: IssueType::Clarity,
                severity: if metrics.clarity < 0.4 { IssueSeverity::High } else { IssueSeverity::Medium },
                description: "Explanation lacks clarity and may be difficult to understand".to_string(),
                location: None,
                impact_assessment: ImpactAssessment {
                    user_experience_impact: 0.8,
                    trust_impact: 0.6,
                    understanding_impact: 0.9,
                    decision_impact: 0.7,
                },
            });
        }
        
        if metrics.completeness < 0.5 {
            issues.push(QualityIssue {
                issue_id: "completeness_low".to_string(),
                issue_type: IssueType::Completeness,
                severity: IssueSeverity::Medium,
                description: "Explanation may be missing important information".to_string(),
                location: None,
                impact_assessment: ImpactAssessment {
                    user_experience_impact: 0.7,
                    trust_impact: 0.5,
                    understanding_impact: 0.8,
                    decision_impact: 0.6,
                },
            });
        }
        
        if metrics.accuracy < 0.7 {
            issues.push(QualityIssue {
                issue_id: "accuracy_low".to_string(),
                issue_type: IssueType::Accuracy,
                severity: IssueSeverity::High,
                description: "Explanation may contain inaccurate information".to_string(),
                location: None,
                impact_assessment: ImpactAssessment {
                    user_experience_impact: 0.9,
                    trust_impact: 0.95,
                    understanding_impact: 0.8,
                    decision_impact: 0.9,
                },
            });
        }
        
        // Check for confidence calibration issues
        if metrics.confidence_calibration < 0.5 {
            issues.push(QualityIssue {
                issue_id: "poor_confidence_calibration".to_string(),
                issue_type: IssueType::Confidence,
                severity: IssueSeverity::Medium,
                description: "Explanation confidence does not match reasoning quality".to_string(),
                location: None,
                impact_assessment: ImpactAssessment {
                    user_experience_impact: 0.6,
                    trust_impact: 0.8,
                    understanding_impact: 0.5,
                    decision_impact: 0.7,
                },
            });
        }
        
        Ok(issues)
    }
    
    fn generate_improvement_suggestions(
        &self,
        issues: &[QualityIssue],
        metrics: &QualityMetrics,
        explanation_text: &str,
    ) -> Result<Vec<ImprovementSuggestion>, QualityAssessmentError> {
        let mut suggestions = Vec::new();
        
        for issue in issues {
            match issue.issue_type {
                IssueType::Clarity => {
                    suggestions.push(ImprovementSuggestion {
                        suggestion_id: "improve_clarity".to_string(),
                        suggestion_type: SuggestionType::ClarityImprovement,
                        priority: SuggestionPriority::High,
                        description: "Simplify language and improve sentence structure".to_string(),
                        expected_improvement: 0.3,
                        implementation_difficulty: ImplementationDifficulty::Moderate,
                        specific_actions: vec![
                            "Break down complex sentences".to_string(),
                            "Use simpler vocabulary".to_string(),
                            "Add transitional phrases".to_string(),
                        ],
                    });
                },
                IssueType::Completeness => {
                    suggestions.push(ImprovementSuggestion {
                        suggestion_id: "improve_completeness".to_string(),
                        suggestion_type: SuggestionType::ContentRevision,
                        priority: SuggestionPriority::Medium,
                        description: "Add missing information and reasoning steps".to_string(),
                        expected_improvement: 0.25,
                        implementation_difficulty: ImplementationDifficulty::Moderate,
                        specific_actions: vec![
                            "Include all reasoning steps".to_string(),
                            "Address the original query directly".to_string(),
                            "Add confidence information".to_string(),
                        ],
                    });
                },
                IssueType::Accuracy => {
                    suggestions.push(ImprovementSuggestion {
                        suggestion_id: "improve_accuracy".to_string(),
                        suggestion_type: SuggestionType::TechnicalCorrection,
                        priority: SuggestionPriority::Critical,
                        description: "Verify and correct factual information".to_string(),
                        expected_improvement: 0.4,
                        implementation_difficulty: ImplementationDifficulty::Difficult,
                        specific_actions: vec![
                            "Cross-reference with reliable sources".to_string(),
                            "Improve reasoning chain quality".to_string(),
                            "Add uncertainty indicators where appropriate".to_string(),
                        ],
                    });
                },
                _ => {
                    // General improvement suggestion
                    suggestions.push(ImprovementSuggestion {
                        suggestion_id: format!("improve_{:?}", issue.issue_type).to_lowercase(),
                        suggestion_type: SuggestionType::ContentRevision,
                        priority: SuggestionPriority::Medium,
                        description: format!("Address {} issues", format!("{:?}", issue.issue_type).to_lowercase()),
                        expected_improvement: 0.2,
                        implementation_difficulty: ImplementationDifficulty::Moderate,
                        specific_actions: vec!["Review and revise content".to_string()],
                    });
                }
            }
        }
        
        Ok(suggestions)
    }
    
    fn perform_baseline_comparisons(&self, metrics: &QualityMetrics) -> Result<Vec<BaselineComparison>, QualityAssessmentError> {
        let mut comparisons = Vec::new();
        
        // Compare with template baseline
        if let Some(template_baseline) = self.baseline_repository.get_baseline("template_baseline") {
            let comparison_result = metrics.overall_quality - template_baseline.quality_metrics.overall_quality;
            comparisons.push(BaselineComparison {
                baseline_type: BaselineType::TemplateBaseline,
                baseline_score: template_baseline.quality_metrics.overall_quality,
                comparison_result,
                statistical_significance: None,
            });
        }
        
        Ok(comparisons)
    }
    
    fn calculate_assessment_confidence(&self, metrics: &QualityMetrics) -> f32 {
        // Confidence in assessment based on metric consistency
        let metric_values = [
            metrics.clarity,
            metrics.completeness,
            metrics.accuracy,
            metrics.relevance,
            metrics.coherence,
        ];
        
        let mean = metric_values.iter().sum::<f32>() / metric_values.len() as f32;
        let variance = metric_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / metric_values.len() as f32;
        
        // Lower variance = higher confidence in assessment
        (1.0 - variance).max(0.5)
    }
    
    fn store_assessment_in_history(&mut self, assessment: QualityAssessment) {
        self.assessment_history.push_back(assessment);
        
        // Maintain history size limit
        if self.assessment_history.len() > self.assessment_config.max_assessment_history {
            self.assessment_history.pop_front();
        }
    }
    
    fn create_default_calculators() -> HashMap<String, QualityCalculator> {
        let mut calculators = HashMap::new();
        
        calculators.insert("clarity".to_string(), QualityCalculator {
            calculator_name: "clarity".to_string(),
            calculation_method: CalculationMethod::HeuristicCombination,
            normalization_strategy: NormalizationStrategy::MinMax,
            weight: 0.15,
            enabled: true,
        });
        
        calculators.insert("accuracy".to_string(), QualityCalculator {
            calculator_name: "accuracy".to_string(),
            calculation_method: CalculationMethod::StatisticalAnalysis,
            normalization_strategy: NormalizationStrategy::MinMax,
            weight: 0.18,
            enabled: true,
        });
        
        calculators
    }
    
    pub fn get_quality_statistics(&self) -> QualityStatistics {
        if self.assessment_history.is_empty() {
            return QualityStatistics::default();
        }
        
        let recent_assessments: Vec<_> = self.assessment_history.iter()
            .filter(|assessment| assessment.timestamp.elapsed() < Duration::from_hours(24))
            .collect();
        
        let avg_quality = if recent_assessments.is_empty() {
            0.0
        } else {
            recent_assessments.iter()
                .map(|assessment| assessment.metrics.overall_quality)
                .sum::<f32>() / recent_assessments.len() as f32
        };
        
        QualityStatistics {
            total_assessments: self.assessment_history.len(),
            recent_assessments: recent_assessments.len(),
            average_quality: avg_quality,
            quality_trend: self.calculate_quality_trend(),
        }
    }
    
    fn calculate_quality_trend(&self) -> QualityTrend {
        if self.assessment_history.len() < 2 {
            return QualityTrend::Stable;
        }
        
        let recent_window = self.assessment_history.len().min(10);
        let recent_avg = self.assessment_history.iter()
            .rev()
            .take(recent_window)
            .map(|a| a.metrics.overall_quality)
            .sum::<f32>() / recent_window as f32;
        
        let older_window = self.assessment_history.len().min(20) - recent_window;
        if older_window == 0 {
            return QualityTrend::Stable;
        }
        
        let older_avg = self.assessment_history.iter()
            .rev()
            .skip(recent_window)
            .take(older_window)
            .map(|a| a.metrics.overall_quality)
            .sum::<f32>() / older_window as f32;
        
        let difference = recent_avg - older_avg;
        
        if difference > 0.05 {
            QualityTrend::Improving
        } else if difference < -0.05 {
            QualityTrend::Declining
        } else {
            QualityTrend::Stable
        }
    }
}

impl BaselineRepository {
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            baseline_metadata: HashMap::new(),
        }
    }
    
    pub fn get_baseline(&self, baseline_id: &str) -> Option<&BaselineData> {
        self.baselines.get(baseline_id)
    }
}

#[derive(Debug, Clone, Default)]
pub struct QualityStatistics {
    pub total_assessments: usize,
    pub recent_assessments: usize,
    pub average_quality: f32,
    pub quality_trend: QualityTrend,
}

#[derive(Debug, Clone, Default)]
pub enum QualityTrend {
    Improving,
    Declining,
    #[default]
    Stable,
}

#[derive(Debug, Clone)]
pub enum QualityAssessmentError {
    CalculationError(String),
    InsufficientData,
    ConfigurationError(String),
    BaselineNotFound(String),
}

impl std::fmt::Display for QualityAssessmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityAssessmentError::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
            QualityAssessmentError::InsufficientData => write!(f, "Insufficient data for quality assessment"),
            QualityAssessmentError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            QualityAssessmentError::BaselineNotFound(id) => write!(f, "Baseline not found: {}", id),
        }
    }
}

impl std::error::Error for QualityAssessmentError {}
```

### Step 3: Quality Integration and Detailed Scores

```rust
impl QualityAssessor {
    fn calculate_detailed_scores(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
        context: &ExplanationContext,
    ) -> Result<DetailedScores, QualityAssessmentError> {
        let linguistic_quality = self.calculate_linguistic_quality(explanation_text, context)?;
        let content_quality = self.calculate_content_quality(explanation_text, reasoning_chain, evidence_collection)?;
        let structural_quality = self.calculate_structural_quality(explanation_text)?;
        let contextual_quality = self.calculate_contextual_quality(explanation_text, context)?;
        let technical_quality = self.calculate_technical_quality(explanation_text, reasoning_chain)?;
        
        Ok(DetailedScores {
            linguistic_quality,
            content_quality,
            structural_quality,
            contextual_quality,
            technical_quality,
        })
    }
    
    fn calculate_linguistic_quality(&self, explanation_text: &str, context: &ExplanationContext) -> Result<LinguisticQuality, QualityAssessmentError> {
        let readability = self.calculate_readability(explanation_text);
        let vocabulary_appropriateness = self.assess_vocabulary_appropriateness(explanation_text, context);
        let grammar_correctness = self.assess_grammar_correctness(explanation_text);
        let style_consistency = self.assess_style_consistency(explanation_text);
        let conciseness = self.assess_conciseness(explanation_text);
        
        Ok(LinguisticQuality {
            readability,
            vocabulary_appropriateness,
            grammar_correctness,
            style_consistency,
            conciseness,
        })
    }
    
    fn calculate_content_quality(
        &self,
        explanation_text: &str,
        reasoning_chain: &ReasoningChain,
        evidence_collection: &EvidenceCollection,
    ) -> Result<ContentQuality, QualityAssessmentError> {
        let factual_accuracy = self.assess_factual_accuracy(explanation_text, evidence_collection);
        let logical_consistency = reasoning_chain.coherence_score;
        let information_completeness = self.assess_information_completeness(explanation_text, reasoning_chain);
        let bias_minimization = self.assess_bias_minimization(explanation_text);
        let uncertainty_handling = self.assess_uncertainty_handling(explanation_text, reasoning_chain);
        
        Ok(ContentQuality {
            factual_accuracy,
            logical_consistency,
            information_completeness,
            bias_minimization,
            uncertainty_handling,
        })
    }
    
    fn calculate_structural_quality(&self, explanation_text: &str) -> Result<StructuralQuality, QualityAssessmentError> {
        let organization = self.assess_organization(explanation_text);
        let flow = self.assess_logical_flow(explanation_text);
        let transitions = self.assess_transition_quality(explanation_text);
        let emphasis = self.assess_emphasis(explanation_text);
        let formatting = self.assess_formatting(explanation_text);
        
        Ok(StructuralQuality {
            organization,
            flow,
            transitions,
            emphasis,
            formatting,
        })
    }
    
    fn calculate_contextual_quality(&self, explanation_text: &str, context: &ExplanationContext) -> Result<ContextualQuality, QualityAssessmentError> {
        let audience_appropriateness = self.assess_audience_appropriateness(explanation_text, context);
        let domain_relevance = self.calculate_relevance(explanation_text, context)?;
        let cultural_sensitivity = self.assess_cultural_sensitivity(explanation_text);
        let temporal_relevance = self.assess_temporal_relevance(explanation_text);
        let purpose_alignment = self.assess_purpose_alignment(explanation_text, context);
        
        Ok(ContextualQuality {
            audience_appropriateness,
            domain_relevance,
            cultural_sensitivity,
            temporal_relevance,
            purpose_alignment,
        })
    }
    
    fn calculate_technical_quality(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> Result<TechnicalQuality, QualityAssessmentError> {
        let reasoning_trace_fidelity = self.assess_reasoning_trace_fidelity(explanation_text, reasoning_chain);
        let system_transparency = self.assess_system_transparency(explanation_text);
        let limitation_acknowledgment = self.assess_limitation_acknowledgment(explanation_text);
        let verification_support = self.assess_verification_support(explanation_text);
        let reproducibility = self.assess_reproducibility(explanation_text, reasoning_chain);
        
        Ok(TechnicalQuality {
            reasoning_trace_fidelity,
            system_transparency,
            limitation_acknowledgment,
            verification_support,
            reproducibility,
        })
    }
    
    // Additional assessment methods for detailed scores
    fn assess_vocabulary_appropriateness(&self, explanation_text: &str, context: &ExplanationContext) -> f32 {
        // Assess if vocabulary matches audience level
        let complex_word_ratio = self.count_complex_words(explanation_text) as f32 / 
                                explanation_text.split_whitespace().count() as f32;
        
        match context.query_type.as_str() {
            "technical" => complex_word_ratio.min(1.0),
            "general" => (1.0 - complex_word_ratio * 2.0).max(0.0),
            _ => (1.0 - complex_word_ratio).max(0.0),
        }
    }
    
    fn assess_grammar_correctness(&self, explanation_text: &str) -> f32 {
        // Simple grammar checking heuristics
        let sentences = self.split_into_sentences(explanation_text);
        let mut correct_sentences = 0;
        
        for sentence in &sentences {
            // Basic checks: starts with capital, ends with punctuation
            if sentence.chars().next().map_or(false, |c| c.is_uppercase()) &&
               sentence.ends_with(&['.', '!', '?'][..]) {
                correct_sentences += 1;
            }
        }
        
        if sentences.is_empty() {
            0.5
        } else {
            correct_sentences as f32 / sentences.len() as f32
        }
    }
    
    fn assess_style_consistency(&self, explanation_text: &str) -> f32 {
        // Check for consistent style throughout
        let sentences = self.split_into_sentences(explanation_text);
        if sentences.len() < 2 {
            return 1.0;
        }
        
        // Simple heuristic: consistent sentence length variation
        let avg_length = sentences.iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f32 / sentences.len() as f32;
        
        let variance = sentences.iter()
            .map(|s| {
                let len = s.split_whitespace().count() as f32;
                (len - avg_length).powi(2)
            })
            .sum::<f32>() / sentences.len() as f32;
        
        // Lower variance = more consistent style
        (1.0 - (variance / (avg_length * avg_length)).sqrt()).max(0.0)
    }
    
    fn assess_conciseness(&self, explanation_text: &str) -> f32 {
        let word_count = explanation_text.split_whitespace().count();
        
        // Optimal range: 50-200 words for most explanations
        match word_count {
            0..=20 => 0.3,
            21..=50 => 0.7,
            51..=200 => 1.0,
            201..=400 => 0.8,
            _ => 0.5,
        }
    }
    
    fn assess_factual_accuracy(&self, explanation_text: &str, evidence_collection: &EvidenceCollection) -> f32 {
        if evidence_collection.evidence_items.is_empty() {
            return 0.5; // Neutral if no evidence to check against
        }
        
        evidence_collection.quality_summary.average_confidence
    }
    
    fn assess_information_completeness(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> f32 {
        // Already calculated in calculate_completeness
        self.calculate_completeness(explanation_text, reasoning_chain, &ExplanationContext {
            query: String::new(),
            query_type: String::new(),
            activation_data: HashMap::new(),
            pathways: vec![],
            entities: vec![],
            evidence: vec![],
            confidence: 0.0,
            processing_time: 0.0,
            metadata: HashMap::new(),
        }).unwrap_or(0.5)
    }
    
    fn assess_bias_minimization(&self, explanation_text: &str) -> f32 {
        // Simple bias detection heuristics
        let biased_words = ["always", "never", "all", "none", "everyone", "no one"];
        let bias_count = biased_words.iter()
            .map(|&word| explanation_text.to_lowercase().matches(word).count())
            .sum::<usize>();
        
        let word_count = explanation_text.split_whitespace().count();
        if word_count == 0 {
            return 0.5;
        }
        
        let bias_ratio = bias_count as f32 / word_count as f32;
        (1.0 - bias_ratio * 10.0).max(0.0)
    }
    
    fn assess_uncertainty_handling(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> f32 {
        // Already calculated in calculate_confidence_calibration
        self.calculate_confidence_calibration(explanation_text, reasoning_chain).unwrap_or(0.5)
    }
    
    fn assess_organization(&self, explanation_text: &str) -> f32 {
        // Check for logical organization patterns
        let has_intro = explanation_text.to_lowercase().contains("first") || 
                       explanation_text.to_lowercase().contains("initially") ||
                       explanation_text.to_lowercase().contains("to begin");
        
        let has_body = explanation_text.to_lowercase().contains("then") ||
                      explanation_text.to_lowercase().contains("next") ||
                      explanation_text.to_lowercase().contains("second");
        
        let has_conclusion = explanation_text.to_lowercase().contains("finally") ||
                            explanation_text.to_lowercase().contains("therefore") ||
                            explanation_text.to_lowercase().contains("in conclusion");
        
        let organization_score = [has_intro, has_body, has_conclusion].iter()
            .map(|&present| if present { 0.33 } else { 0.0 })
            .sum::<f32>();
        
        organization_score.max(0.4) // Minimum score for basic organization
    }
    
    fn assess_emphasis(&self, explanation_text: &str) -> f32 {
        // Check for emphasis techniques
        let has_formatting = explanation_text.contains("**") || explanation_text.contains("*");
        let has_emphasis_words = explanation_text.to_lowercase().contains("important") ||
                                explanation_text.to_lowercase().contains("key") ||
                                explanation_text.to_lowercase().contains("significant");
        
        if has_formatting || has_emphasis_words { 0.8 } else { 0.5 }
    }
    
    fn assess_formatting(&self, explanation_text: &str) -> f32 {
        // Basic formatting assessment
        let has_paragraphs = explanation_text.contains("\n\n");
        let has_lists = explanation_text.contains("1.") || explanation_text.contains("•");
        let proper_punctuation = explanation_text.chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count() > 0;
        
        let formatting_features = [has_paragraphs, has_lists, proper_punctuation].iter()
            .map(|&present| if present { 0.33 } else { 0.0 })
            .sum::<f32>();
        
        formatting_features.max(0.3)
    }
    
    fn assess_audience_appropriateness(&self, explanation_text: &str, context: &ExplanationContext) -> f32 {
        // Use vocabulary appropriateness as proxy
        self.assess_vocabulary_appropriateness(explanation_text, context)
    }
    
    fn assess_cultural_sensitivity(&self, explanation_text: &str) -> f32 {
        // Simple check for potentially insensitive content
        let sensitive_terms = ["all people", "everyone knows", "obviously", "clearly"];
        let contains_assumptions = sensitive_terms.iter()
            .any(|&term| explanation_text.to_lowercase().contains(term));
        
        if contains_assumptions { 0.6 } else { 0.9 }
    }
    
    fn assess_temporal_relevance(&self, explanation_text: &str) -> f32 {
        // Check for temporal indicators
        let temporal_words = ["currently", "now", "today", "recently", "latest"];
        let has_temporal_context = temporal_words.iter()
            .any(|&word| explanation_text.to_lowercase().contains(word));
        
        if has_temporal_context { 0.9 } else { 0.7 }
    }
    
    fn assess_purpose_alignment(&self, explanation_text: &str, context: &ExplanationContext) -> f32 {
        // Check if explanation serves its intended purpose
        match context.query_type.as_str() {
            "factual" => {
                if explanation_text.contains("fact") || explanation_text.contains("information") {
                    0.9
                } else {
                    0.6
                }
            },
            "reasoning" => {
                if explanation_text.contains("because") || explanation_text.contains("reasoning") {
                    0.9
                } else {
                    0.6
                }
            },
            _ => 0.7,
        }
    }
    
    fn assess_reasoning_trace_fidelity(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> f32 {
        // Check how well explanation reflects actual reasoning process
        let step_mentions = reasoning_chain.steps.iter()
            .filter(|step| {
                explanation_text.contains(&step.premise) || 
                explanation_text.contains(&step.conclusion)
            })
            .count();
        
        if reasoning_chain.steps.is_empty() {
            0.5
        } else {
            step_mentions as f32 / reasoning_chain.steps.len() as f32
        }
    }
    
    fn assess_system_transparency(&self, explanation_text: &str) -> f32 {
        // Check if explanation is transparent about AI nature
        let transparency_indicators = ["AI", "system", "algorithm", "computer", "analysis"];
        let mentions_ai = transparency_indicators.iter()
            .any(|&indicator| explanation_text.to_lowercase().contains(indicator));
        
        if mentions_ai { 0.8 } else { 0.5 }
    }
    
    fn assess_limitation_acknowledgment(&self, explanation_text: &str) -> f32 {
        // Check if limitations are acknowledged
        let limitation_words = ["limitation", "may not", "uncertain", "approximate", "estimate"];
        let acknowledges_limits = limitation_words.iter()
            .any(|&word| explanation_text.to_lowercase().contains(word));
        
        if acknowledges_limits { 0.9 } else { 0.6 }
    }
    
    fn assess_verification_support(&self, explanation_text: &str) -> f32 {
        // Check if explanation supports verification
        let verification_words = ["evidence", "source", "reference", "verify", "check"];
        let supports_verification = verification_words.iter()
            .any(|&word| explanation_text.to_lowercase().contains(word));
        
        if supports_verification { 0.8 } else { 0.5 }
    }
    
    fn assess_reproducibility(&self, explanation_text: &str, reasoning_chain: &ReasoningChain) -> f32 {
        // Check if reasoning process is reproducible
        let has_clear_steps = explanation_text.contains("step") || 
                             explanation_text.contains("first") ||
                             explanation_text.contains("then");
        
        let has_systematic_approach = reasoning_chain.steps.len() > 1;
        
        if has_clear_steps && has_systematic_approach { 0.9 } else { 0.6 }
    }
}
```

## File Locations

- `src/cognitive/explanation/quality_assessment.rs` - Main implementation
- `src/cognitive/explanation/mod.rs` - Module exports and integration
- `tests/cognitive/explanation/quality_assessment_tests.rs` - Test implementation

## Success Criteria

- [ ] Quality assessment system evaluates all dimensions accurately
- [ ] Quality metrics provide meaningful insights
- [ ] Quality issues identification helps improve explanations
- [ ] Improvement suggestions are actionable and relevant
- [ ] Assessment performance meets real-time requirements
- [ ] Quality trends tracking provides useful insights
- [ ] All tests pass:
  - Individual quality metric calculations
  - Overall quality scoring accuracy
  - Issue identification effectiveness
  - Improvement suggestion relevance
  - Performance benchmarks

## Test Requirements

```rust
#[test]
fn test_quality_metrics_calculation() {
    let mut assessor = QualityAssessor::new();
    
    let reasoning_chain = create_test_reasoning_chain();
    let evidence_collection = create_test_evidence_collection();
    let context = create_test_explanation_context();
    
    let explanation_text = "This AI system analyzed your query by first identifying key concepts, then finding relevant connections, and finally synthesizing the information to provide a confident answer.";
    
    let assessment = assessor.assess_explanation_quality(
        explanation_text,
        &reasoning_chain,
        &evidence_collection,
        &context,
    ).unwrap();
    
    // Check that all metrics are calculated
    assert!(assessment.metrics.clarity > 0.0);
    assert!(assessment.metrics.completeness > 0.0);
    assert!(assessment.metrics.accuracy > 0.0);
    assert!(assessment.metrics.overall_quality > 0.0);
    
    // Check that overall quality is within valid range
    assert!(assessment.metrics.overall_quality <= 1.0);
    assert!(assessment.metrics.overall_quality >= 0.0);
}

#[test]
fn test_clarity_assessment() {
    let assessor = QualityAssessor::new();
    let context = create_test_explanation_context();
    
    // High clarity text
    let clear_text = "The system works by processing your question. First, it identifies key words. Then, it searches for relevant information. Finally, it provides an answer.";
    let clarity_score = assessor.calculate_clarity(clear_text, &context).unwrap();
    assert!(clarity_score > 0.7);
    
    // Low clarity text
    let unclear_text = "The sophisticated algorithmic implementation utilizes comprehensive computational methodologies to facilitate optimal informational retrieval and synthesis processes.";
    let clarity_score = assessor.calculate_clarity(unclear_text, &context).unwrap();
    assert!(clarity_score < 0.6);
}

#[test]
fn test_completeness_assessment() {
    let assessor = QualityAssessor::new();
    let reasoning_chain = create_test_reasoning_chain();
    let context = create_test_explanation_context();
    
    // Complete explanation
    let complete_text = "To answer your question about AI, I first identified the concept of artificial intelligence, then analyzed its key characteristics, and concluded that AI involves computational intelligence with high confidence.";
    let completeness_score = assessor.calculate_completeness(complete_text, &reasoning_chain, &context).unwrap();
    assert!(completeness_score > 0.7);
    
    // Incomplete explanation
    let incomplete_text = "AI is computational intelligence.";
    let completeness_score = assessor.calculate_completeness(incomplete_text, &reasoning_chain, &context).unwrap();
    assert!(completeness_score < 0.6);
}

#[test]
fn test_quality_issues_identification() {
    let mut assessor = QualityAssessor::new();
    
    let reasoning_chain = create_test_reasoning_chain();
    let evidence_collection = create_test_evidence_collection();
    let context = create_test_explanation_context();
    
    // Low quality explanation that should trigger issues
    let poor_explanation = "Yes.";
    
    let assessment = assessor.assess_explanation_quality(
        poor_explanation,
        &reasoning_chain,
        &evidence_collection,
        &context,
    ).unwrap();
    
    // Should identify multiple quality issues
    assert!(!assessment.quality_issues.is_empty());
    
    // Should have improvement suggestions
    assert!(!assessment.improvement_suggestions.is_empty());
    
    // Overall quality should be low
    assert!(assessment.metrics.overall_quality < 0.5);
}

#[test]
fn test_improvement_suggestions() {
    let assessor = QualityAssessor::new();
    
    let metrics = QualityMetrics {
        clarity: 0.3,
        completeness: 0.4,
        accuracy: 0.8,
        relevance: 0.7,
        coherence: 0.6,
        confidence_calibration: 0.5,
        evidence_support: 0.7,
        user_comprehension: 0.4,
        actionability: 0.6,
        trustworthiness: 0.7,
        overall_quality: 0.55,
    };
    
    let issues = vec![
        QualityIssue {
            issue_id: "clarity_low".to_string(),
            issue_type: IssueType::Clarity,
            severity: IssueSeverity::High,
            description: "Low clarity".to_string(),
            location: None,
            impact_assessment: ImpactAssessment {
                user_experience_impact: 0.8,
                trust_impact: 0.6,
                understanding_impact: 0.9,
                decision_impact: 0.7,
            },
        },
    ];
    
    let suggestions = assessor.generate_improvement_suggestions(
        &issues,
        &metrics,
        "unclear explanation text",
    ).unwrap();
    
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| matches!(s.suggestion_type, SuggestionType::ClarityImprovement)));
}

#[test]
fn test_quality_assessment_performance() {
    let mut assessor = QualityAssessor::new();
    
    let reasoning_chain = create_test_reasoning_chain();
    let evidence_collection = create_test_evidence_collection();
    let context = create_test_explanation_context();
    
    let explanation_text = "Test explanation for performance measurement.";
    
    let start_time = Instant::now();
    let _assessment = assessor.assess_explanation_quality(
        explanation_text,
        &reasoning_chain,
        &evidence_collection,
        &context,
    ).unwrap();
    let elapsed = start_time.elapsed();
    
    // Should complete within performance target
    assert!(elapsed < Duration::from_millis(50));
}

#[test]
fn test_baseline_comparison() {
    let assessor = QualityAssessor::new();
    
    let metrics = QualityMetrics {
        clarity: 0.8,
        completeness: 0.7,
        accuracy: 0.9,
        relevance: 0.8,
        coherence: 0.8,
        confidence_calibration: 0.7,
        evidence_support: 0.8,
        user_comprehension: 0.8,
        actionability: 0.6,
        trustworthiness: 0.8,
        overall_quality: 0.78,
    };
    
    let comparisons = assessor.perform_baseline_comparisons(&metrics).unwrap();
    
    // Should have at least one baseline comparison
    assert!(!comparisons.is_empty());
}

fn create_test_reasoning_chain() -> ReasoningChain {
    // Implementation similar to previous tests
    ReasoningChain {
        chain_id: ChainId(1),
        steps: vec![
            ReasoningStep {
                step_id: StepId(1),
                step_type: StepType::EntityRecognition,
                premise: "Identify AI concept".to_string(),
                conclusion: "Found artificial intelligence".to_string(),
                evidence: vec![],
                confidence: 0.9,
                activation_nodes: vec![NodeId(1)],
                logical_operation: LogicalOperation::DirectReference,
                timestamp: Instant::now(),
            },
        ],
        connections: vec![],
        source_pathways: vec![],
        confidence_score: 0.85,
        completeness_score: 0.8,
        coherence_score: 0.9,
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
    }
}

fn create_test_evidence_collection() -> EvidenceCollection {
    EvidenceCollection {
        collection_id: CollectionId(1),
        query: "test query".to_string(),
        evidence_items: vec![],
        relationships: vec![],
        collection_strategy: CollectionStrategy::Quality,
        quality_summary: QualitySummary {
            average_quality: 0.8,
            average_relevance: 0.7,
            average_confidence: 0.8,
            total_evidence_count: 1,
            high_quality_count: 1,
            verified_count: 1,
        },
        collection_time: Duration::from_millis(10),
        timestamp: Instant::now(),
    }
}

fn create_test_explanation_context() -> ExplanationContext {
    ExplanationContext {
        query: "What is artificial intelligence?".to_string(),
        query_type: "factual".to_string(),
        activation_data: HashMap::new(),
        pathways: vec![],
        entities: vec![],
        evidence: vec![],
        confidence: 0.8,
        processing_time: 0.0,
        metadata: HashMap::new(),
    }
}
```

## Quality Gates

- [ ] Quality assessment < 10ms per explanation
- [ ] Memory usage < 15MB for quality tracking history
- [ ] Quality metric accuracy correlation with human ratings > 75%
- [ ] Issue identification precision > 80%
- [ ] Improvement suggestion relevance > 70%
- [ ] Thread-safe concurrent quality assessment

## Next Task

Upon completion, proceed to **30_explanation_tests.md**