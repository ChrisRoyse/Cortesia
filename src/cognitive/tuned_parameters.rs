use std::collections::HashMap as AHashMap;

/// Tuned parameters for cognitive patterns optimized for synthetic test data
/// These parameters have been calibrated to work effectively with the comprehensive
/// test knowledge base that includes proper hierarchies, contradictions, and patterns.

#[derive(Debug, Clone)]
pub struct TunedCognitiveParameters {
    pub convergent: ConvergentParameters,
    pub divergent: DivergentParameters,
    pub lateral: LateralParameters,
    pub systems: SystemsParameters,
    pub critical: CriticalParameters,
    pub abstract_thinking: AbstractParameters,
    pub adaptive: AdaptiveParameters,
}

impl TunedCognitiveParameters {
    pub fn new_optimized() -> Self {
        Self {
            convergent: ConvergentParameters::optimized(),
            divergent: DivergentParameters::optimized(),
            lateral: LateralParameters::optimized(),
            systems: SystemsParameters::optimized(),
            critical: CriticalParameters::optimized(),
            abstract_thinking: AbstractParameters::optimized(),
            adaptive: AdaptiveParameters::optimized(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvergentParameters {
    pub activation_threshold: f32,
    pub max_depth: usize,
    pub beam_width: usize,
    pub concept_relevance_threshold: f32,
    pub confidence_weight: f32,
}

impl ConvergentParameters {
    pub fn optimized() -> Self {
        Self {
            activation_threshold: 0.3,      // Lower threshold for better matching
            max_depth: 6,                   // Slightly deeper search
            beam_width: 5,                  // Wider beam for better coverage
            concept_relevance_threshold: 0.05, // Very low threshold for better recall
            confidence_weight: 0.8,         // High confidence in direct matches
        }
    }
}

#[derive(Debug, Clone)]
pub struct DivergentParameters {
    pub exploration_breadth: usize,
    pub creativity_threshold: f32,
    pub max_exploration_depth: usize,
    pub novelty_weight: f32,
    pub min_exploration_results: usize,
}

impl DivergentParameters {
    pub fn optimized() -> Self {
        Self {
            exploration_breadth: 25,        // Increased breadth for more exploration
            creativity_threshold: 0.2,     // Lower threshold for more creative connections
            max_exploration_depth: 5,      // Deeper exploration
            novelty_weight: 0.3,           // Balanced novelty vs relevance
            min_exploration_results: 3,    // Ensure minimum results
        }
    }
}

#[derive(Debug, Clone)]
pub struct LateralParameters {
    pub novelty_threshold: f32,
    pub max_bridge_length: usize,
    pub creativity_weight: f32,
    pub plausibility_weight: f32,
    pub min_bridge_confidence: f32,
}

impl LateralParameters {
    pub fn optimized() -> Self {
        Self {
            novelty_threshold: 0.4,        // Moderate novelty requirement
            max_bridge_length: 4,          // Allow longer bridges
            creativity_weight: 0.6,        // Emphasize creativity
            plausibility_weight: 0.4,      // But maintain plausibility
            min_bridge_confidence: 0.3,    // Lower confidence for exploration
        }
    }
}

#[derive(Debug, Clone)]
pub struct SystemsParameters {
    pub max_inheritance_depth: usize,
    pub attribute_confidence_threshold: f32,
    pub hierarchy_weight: f32,
    pub exception_tolerance: f32,
    pub inheritance_decay: f32,
}

impl SystemsParameters {
    pub fn optimized() -> Self {
        Self {
            max_inheritance_depth: 8,      // Deeper inheritance traversal
            attribute_confidence_threshold: 0.1, // Lower threshold for attributes
            hierarchy_weight: 0.9,         // High weight for hierarchical relationships
            exception_tolerance: 0.8,      // Allow exceptions
            inheritance_decay: 0.95,       // Slow decay for inheritance
        }
    }
}

#[derive(Debug, Clone)]
pub struct CriticalParameters {
    pub contradiction_threshold: f32,
    pub confidence_difference_threshold: f32,
    pub resolution_strategy_preference: ResolutionPreference,
    pub uncertainty_weight: f32,
    pub validation_strictness: f32,
}

#[derive(Debug, Clone)]
pub enum ResolutionPreference {
    PreferSpecific,     // Prefer specific over general facts
    PreferHighConfidence, // Prefer high confidence facts
    PreferRecent,       // Prefer more recently added facts
    Balanced,           // Balance all factors
}

impl CriticalParameters {
    pub fn optimized() -> Self {
        Self {
            contradiction_threshold: 0.6,  // Moderate threshold for contradictions
            confidence_difference_threshold: 0.2, // Detect significant differences
            resolution_strategy_preference: ResolutionPreference::PreferSpecific,
            uncertainty_weight: 0.3,       // Moderate uncertainty consideration
            validation_strictness: 0.7,    // High but not excessive strictness
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbstractParameters {
    pub pattern_frequency_threshold: f32,
    pub abstraction_confidence_threshold: f32,
    pub complexity_weight: f32,
    pub similarity_threshold: f32,
    pub min_pattern_instances: usize,
}

impl AbstractParameters {
    pub fn optimized() -> Self {
        Self {
            pattern_frequency_threshold: 0.3, // Moderate frequency requirement
            abstraction_confidence_threshold: 0.5, // Reasonable confidence
            complexity_weight: 0.4,          // Balance complexity vs simplicity
            similarity_threshold: 0.6,       // Good similarity requirement
            min_pattern_instances: 2,        // Minimum instances for pattern
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveParameters {
    pub strategy_selection_confidence: f32,
    pub ensemble_threshold: f32,
    pub pattern_weights: AHashMap<String, f32>,
    pub learning_rate: f32,
    pub exploration_probability: f32,
}

impl AdaptiveParameters {
    pub fn optimized() -> Self {
        let mut pattern_weights = AHashMap::new();
        pattern_weights.insert("Convergent".to_string(), 1.0);
        pattern_weights.insert("Divergent".to_string(), 0.8);
        pattern_weights.insert("Lateral".to_string(), 0.6);
        pattern_weights.insert("Systems".to_string(), 0.9);
        pattern_weights.insert("Critical".to_string(), 1.1);
        pattern_weights.insert("Abstract".to_string(), 0.7);
        
        Self {
            strategy_selection_confidence: 0.6, // Moderate confidence for selection
            ensemble_threshold: 0.7,           // Use ensemble for complex queries
            pattern_weights,
            learning_rate: 0.1,               // Moderate learning rate
            exploration_probability: 0.2,     // Some exploration of new strategies
        }
    }
}

/// Query complexity analysis for adaptive pattern selection
#[derive(Debug, Clone)]
pub struct QueryComplexityAnalyzer {
    pub keyword_weights: AHashMap<String, f32>,
    pub pattern_indicators: AHashMap<String, Vec<String>>,
}

impl QueryComplexityAnalyzer {
    pub fn new_optimized() -> Self {
        let mut keyword_weights = AHashMap::new();
        // Factual query indicators
        keyword_weights.insert("what".to_string(), 0.3);
        keyword_weights.insert("who".to_string(), 0.3);
        keyword_weights.insert("where".to_string(), 0.3);
        keyword_weights.insert("when".to_string(), 0.3);
        
        // Exploration indicators
        keyword_weights.insert("types".to_string(), 0.8);
        keyword_weights.insert("examples".to_string(), 0.8);
        keyword_weights.insert("possibilities".to_string(), 0.9);
        keyword_weights.insert("brainstorm".to_string(), 0.9);
        
        // Creative indicators
        keyword_weights.insert("creative".to_string(), 0.9);
        keyword_weights.insert("innovative".to_string(), 0.8);
        keyword_weights.insert("connect".to_string(), 0.7);
        keyword_weights.insert("relate".to_string(), 0.7);
        
        // Systems indicators
        keyword_weights.insert("inherit".to_string(), 0.9);
        keyword_weights.insert("properties".to_string(), 0.8);
        keyword_weights.insert("hierarchy".to_string(), 0.9);
        keyword_weights.insert("system".to_string(), 0.8);
        
        // Critical thinking indicators
        keyword_weights.insert("contradiction".to_string(), 0.9);
        keyword_weights.insert("however".to_string(), 0.7);
        keyword_weights.insert("but".to_string(), 0.6);
        keyword_weights.insert("exception".to_string(), 0.8);
        
        // Abstract indicators
        keyword_weights.insert("pattern".to_string(), 0.9);
        keyword_weights.insert("structure".to_string(), 0.7);
        keyword_weights.insert("similarity".to_string(), 0.8);
        keyword_weights.insert("abstraction".to_string(), 0.9);
        
        let mut pattern_indicators = AHashMap::new();
        pattern_indicators.insert("Convergent".to_string(), vec![
            "what is".to_string(), "define".to_string(), "specific".to_string()
        ]);
        pattern_indicators.insert("Divergent".to_string(), vec![
            "types of".to_string(), "examples of".to_string(), "brainstorm".to_string()
        ]);
        pattern_indicators.insert("Lateral".to_string(), vec![
            "how is".to_string(), "connected to".to_string(), "relate to".to_string()
        ]);
        pattern_indicators.insert("Systems".to_string(), vec![
            "inherit".to_string(), "properties".to_string(), "hierarchy".to_string()
        ]);
        pattern_indicators.insert("Critical".to_string(), vec![
            "contradiction".to_string(), "exception".to_string(), "but".to_string()
        ]);
        pattern_indicators.insert("Abstract".to_string(), vec![
            "pattern".to_string(), "structure".to_string(), "similarity".to_string()
        ]);
        
        Self {
            keyword_weights,
            pattern_indicators,
        }
    }
    
    pub fn analyze_query(&self, query: &str) -> QueryAnalysis {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();
        
        let mut complexity_score = 0.0;
        let mut pattern_scores = AHashMap::new();
        
        // Initialize pattern scores
        for pattern in ["Convergent", "Divergent", "Lateral", "Systems", "Critical", "Abstract"] {
            pattern_scores.insert(pattern.to_string(), 0.0);
        }
        
        // Analyze keywords
        for word in &words {
            if let Some(weight) = self.keyword_weights.get(*word) {
                complexity_score += weight;
            }
        }
        
        // Check pattern indicators
        for (pattern, indicators) in &self.pattern_indicators {
            for indicator in indicators {
                if query_lower.contains(indicator) {
                    let current_score = pattern_scores.get(pattern).unwrap_or(&0.0);
                    pattern_scores.insert(pattern.clone(), current_score + 1.0);
                }
            }
        }
        
        // Normalize scores
        complexity_score = (complexity_score / words.len() as f32).min(1.0);
        
        // Find best pattern
        let best_pattern = pattern_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "Convergent".to_string());
        
        let requires_ensemble = complexity_score > 0.7 || pattern_scores.values().filter(|&v| *v > 0.5).count() > 1;
        let ambiguity_level = self.calculate_ambiguity(&pattern_scores);
        
        QueryAnalysis {
            complexity_score,
            pattern_scores,
            recommended_pattern: best_pattern,
            requires_ensemble,
            ambiguity_level,
        }
    }
    
    fn calculate_ambiguity(&self, pattern_scores: &AHashMap<String, f32>) -> f32 {
        let scores: Vec<f32> = pattern_scores.values().cloned().collect();
        let max_score = scores.iter().fold(0.0f32, |a, &b| a.max(b));
        let second_max = scores.iter().filter(|&&x| x < max_score).fold(0.0f32, |a, &b| a.max(b));
        
        if max_score > 0.0 {
            second_max / max_score
        } else {
            1.0 // High ambiguity if no clear indicators
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub complexity_score: f32,
    pub pattern_scores: AHashMap<String, f32>,
    pub recommended_pattern: String,
    pub requires_ensemble: bool,
    pub ambiguity_level: f32,
}

/// Confidence calibration for different data types
#[derive(Debug, Clone)]
pub struct ConfidenceCalibration {
    pub base_confidence: f32,
    pub entity_type_modifiers: AHashMap<String, f32>,
    pub relationship_type_modifiers: AHashMap<String, f32>,
    pub depth_decay_factor: f32,
}

impl ConfidenceCalibration {
    pub fn new_optimized() -> Self {
        let mut entity_type_modifiers = AHashMap::new();
        entity_type_modifiers.insert("Input".to_string(), 1.0);
        entity_type_modifiers.insert("Gate".to_string(), 0.9);
        entity_type_modifiers.insert("Output".to_string(), 0.8);
        
        let mut relationship_type_modifiers = AHashMap::new();
        relationship_type_modifiers.insert("IsA".to_string(), 0.95);
        relationship_type_modifiers.insert("HasProperty".to_string(), 0.9);
        relationship_type_modifiers.insert("RelatedTo".to_string(), 0.7);
        relationship_type_modifiers.insert("Similar".to_string(), 0.6);
        relationship_type_modifiers.insert("Temporal".to_string(), 0.8);
        
        Self {
            base_confidence: 0.7,
            entity_type_modifiers,
            relationship_type_modifiers,
            depth_decay_factor: 0.95,
        }
    }
    
    pub fn calibrate_confidence(
        &self,
        raw_confidence: f32,
        entity_type: &str,
        relationship_type: Option<&str>,
        depth: usize,
    ) -> f32 {
        let mut calibrated = raw_confidence * self.base_confidence;
        
        // Apply entity type modifier
        if let Some(modifier) = self.entity_type_modifiers.get(entity_type) {
            calibrated *= modifier;
        }
        
        // Apply relationship type modifier
        if let Some(rel_type) = relationship_type {
            if let Some(modifier) = self.relationship_type_modifiers.get(rel_type) {
                calibrated *= modifier;
            }
        }
        
        // Apply depth decay
        calibrated *= self.depth_decay_factor.powi(depth as i32);
        
        calibrated.min(1.0).max(0.0)
    }
}