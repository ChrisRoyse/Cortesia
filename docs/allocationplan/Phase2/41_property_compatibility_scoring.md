# Task 41: Property Compatibility Scoring

## Metadata
- **Micro-Phase**: 2.41
- **Duration**: 15-20 minutes
- **Dependencies**: Task 36 (property_inheritance_engine), Task 39 (scoring_framework_design)
- **Output**: `src/allocation_scoring/property_compatibility_scoring.rs`

## Description
Implement intelligent property compatibility scoring that analyzes property inheritance patterns, detects compatibility conflicts, and scores allocation candidates based on property alignment with inheritance hierarchies. This system handles property overrides, exceptions, and type compatibility with <0.3ms per evaluation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_scoring::{AllocationContext, ScoringStrategy};
    use crate::hierarchy_detection::ExtractedConcept;
    use std::collections::HashMap;

    #[test]
    fn test_property_compatibility_strategy_creation() {
        let strategy = PropertyCompatibilityStrategy::new();
        assert_eq!(strategy.name(), "property_compatibility");
        assert!(strategy.supports_parallel());
        assert!(strategy.is_enabled());
        assert_eq!(strategy.get_compatibility_threshold(), 0.7);
    }
    
    #[test]
    fn test_basic_property_matching() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        // Create concept with properties
        let golden_retriever = create_concept_with_properties("golden retriever", 0.85, hashmap!{
            "color" => "golden",
            "size" => "large",
            "temperament" => "friendly",
            "energy_level" => "high"
        });
        
        // Create context with expected properties
        let dog_context = create_context_with_properties("dog", &["mammal", "animal"], hashmap!{
            "has_fur" => "true",
            "is_domestic" => "true",
            "locomotion" => "quadruped"
        });
        
        let score = strategy.score(&golden_retriever, &dog_context).unwrap();
        
        // Should have reasonable compatibility despite different properties
        assert!(score >= 0.5);
        assert!(score <= 1.0);
    }
    
    #[test]
    fn test_exact_property_match() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        let concept = create_concept_with_properties("test_concept", 0.9, hashmap!{
            "color" => "red",
            "shape" => "round",
            "material" => "metal"
        });
        
        let context = create_context_with_properties("test_parent", &[], hashmap!{
            "color" => "red",
            "shape" => "round", 
            "material" => "metal"
        });
        
        let score = strategy.score(&concept, &context).unwrap();
        
        // Perfect match should score very high
        assert!(score > 0.95);
    }
    
    #[test]
    fn test_property_conflict_detection() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        let concept = create_concept_with_properties("penguin", 0.85, hashmap!{
            "can_fly" => "false",
            "habitat" => "antarctica",
            "locomotion" => "swimming"
        });
        
        let bird_context = create_context_with_properties("bird", &["animal"], hashmap!{
            "can_fly" => "true",
            "has_feathers" => "true",
            "locomotion" => "flying"
        });
        
        let score = strategy.score(&concept, &bird_context).unwrap();
        let conflicts = strategy.detect_property_conflicts(&concept, &bird_context).unwrap();
        
        // Should detect conflicts and lower score
        assert!(score < 0.7); // Lower due to conflicts
        assert!(!conflicts.is_empty());
        assert!(conflicts.iter().any(|c| c.property_name == "can_fly"));
        assert!(conflicts.iter().any(|c| c.property_name == "locomotion"));
    }
    
    #[test]
    fn test_inheritance_aware_scoring() {
        let mut strategy = PropertyCompatibilityStrategy::new();
        
        // Enable inheritance awareness
        strategy.set_inheritance_aware(true);
        
        let dog = create_concept_with_properties("dog", 0.9, hashmap!{
            "legs" => "4",
            "is_domestic" => "true",
            "makes_sound" => "bark"
        });
        
        // Context with inherited properties from mammal -> animal
        let mammal_context = create_context_with_inherited_properties("mammal", &["animal"], 
            hashmap!{
                // Direct properties
                "warm_blooded" => "true",
                "has_fur" => "true"
            },
            hashmap!{
                // Inherited from animal
                "is_alive" => "true",
                "can_move" => "true",
                "needs_food" => "true"
            }
        );
        
        let score = strategy.score(&dog, &mammal_context).unwrap();
        
        // Should score high due to good compatibility with inherited properties
        assert!(score > 0.8);
        
        // Check inheritance analysis
        let analysis = strategy.analyze_inheritance_compatibility(&dog, &mammal_context).unwrap();
        assert!(analysis.compatible_inherited_properties > 0);
    }
    
    #[test]
    fn test_property_type_compatibility() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        // Test numeric property compatibility
        let concept1 = create_concept_with_properties("concept1", 0.8, hashmap!{
            "weight" => "50",
            "height" => "1.5",
            "age" => "5"
        });
        
        let context1 = create_context_with_properties("parent1", &[], hashmap!{
            "weight" => "40-60", // Range compatibility
            "height" => "1.0-2.0", // Range compatibility
            "age" => "young" // Type conversion needed
        });
        
        let score = strategy.score(&concept1, &context1).unwrap();
        assert!(score > 0.6); // Should handle type conversions reasonably
        
        // Test boolean property compatibility
        let concept2 = create_concept_with_properties("concept2", 0.8, hashmap!{
            "is_active" => "yes",
            "has_feature" => "1",
            "enabled" => "true"
        });
        
        let context2 = create_context_with_properties("parent2", &[], hashmap!{
            "is_active" => "true",
            "has_feature" => "true", 
            "enabled" => "false" // Conflict
        });
        
        let score2 = strategy.score(&concept2, &context2).unwrap();
        assert!(score2 < score); // Should be lower due to boolean conflict
    }
    
    #[test]
    fn test_property_weight_configuration() {
        let mut strategy = PropertyCompatibilityStrategy::new();
        
        // Configure property weights
        let weights = PropertyWeights {
            exact_match_weight: 0.5,
            partial_match_weight: 0.3,
            type_compatibility_weight: 0.1,
            inheritance_compatibility_weight: 0.1,
        };
        
        strategy.set_property_weights(weights.clone());
        assert_eq!(strategy.get_property_weights().exact_match_weight, 0.5);
        
        let concept = create_concept_with_properties("test", 0.8, hashmap!{
            "color" => "red",
            "size" => "large"
        });
        
        let context = create_context_with_properties("parent", &[], hashmap!{
            "color" => "red", // Exact match
            "size" => "big"   // Partial match
        });
        
        let score = strategy.score(&concept, &context).unwrap();
        assert!(score > 0.7); // Should benefit from exact match emphasis
    }
    
    #[test]
    fn test_property_exception_handling() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        // Platypus: mammal that lays eggs (exception to live birth)
        let platypus = create_concept_with_properties("platypus", 0.85, hashmap!{
            "reproduction" => "lays_eggs",
            "habitat" => "aquatic",
            "has_bill" => "true"
        });
        
        let mammal_context = create_context_with_properties("mammal", &["animal"], hashmap!{
            "reproduction" => "live_birth",
            "warm_blooded" => "true",
            "has_fur" => "true"
        });
        
        let score = strategy.score(&platypus, &mammal_context).unwrap();
        let exceptions = strategy.identify_property_exceptions(&platypus, &mammal_context).unwrap();
        
        // Should identify exception but still allow reasonable compatibility
        assert!(score > 0.5); // Not too low despite exception
        assert!(!exceptions.is_empty());
        assert!(exceptions.iter().any(|e| e.property_name == "reproduction"));
    }
    
    #[test]
    fn test_batch_property_scoring() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        let concepts = vec![
            create_concept_with_properties("dog", 0.9, hashmap!{"legs" => "4", "domestic" => "true"}),
            create_concept_with_properties("cat", 0.85, hashmap!{"legs" => "4", "independent" => "true"}),
            create_concept_with_properties("bird", 0.8, hashmap!{"wings" => "2", "can_fly" => "true"}),
            create_concept_with_properties("fish", 0.82, hashmap!{"fins" => "multiple", "aquatic" => "true"}),
        ];
        
        let animal_context = create_context_with_properties("animal", &[], hashmap!{
            "is_alive" => "true",
            "can_move" => "true"
        });
        
        let start = std::time::Instant::now();
        let scores = strategy.batch_score(&concepts, &animal_context).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete in under 5ms
        assert!(elapsed < std::time::Duration::from_millis(5));
        assert_eq!(scores.len(), 4);
        
        // All should have reasonable scores
        for score in &scores {
            assert!(*score >= 0.3 && *score <= 1.0);
        }
    }
    
    #[test]
    fn test_property_similarity_calculation() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        // Test string similarity
        assert!(strategy.calculate_property_similarity("red", "red") > 0.99);
        assert!(strategy.calculate_property_similarity("large", "big") > 0.6);
        assert!(strategy.calculate_property_similarity("small", "tiny") > 0.6);
        assert!(strategy.calculate_property_similarity("red", "blue") < 0.3);
        
        // Test numeric similarity
        assert!(strategy.calculate_property_similarity("10", "12") > 0.8);
        assert!(strategy.calculate_property_similarity("10", "20") < 0.6);
        
        // Test boolean similarity
        assert!(strategy.calculate_property_similarity("true", "yes") > 0.8);
        assert!(strategy.calculate_property_similarity("false", "no") > 0.8);
        assert!(strategy.calculate_property_similarity("true", "false") < 0.1);
    }
    
    #[test]
    fn test_missing_property_handling() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        let sparse_concept = create_concept_with_properties("minimal", 0.7, hashmap!{
            "name" => "test"
        });
        
        let rich_context = create_context_with_properties("detailed", &[], hashmap!{
            "color" => "red",
            "size" => "large", 
            "weight" => "heavy",
            "material" => "metal",
            "shape" => "round"
        });
        
        let score = strategy.score(&sparse_concept, &rich_context).unwrap();
        let analysis = strategy.analyze_missing_properties(&sparse_concept, &rich_context).unwrap();
        
        // Should handle missing properties gracefully
        assert!(score >= 0.3); // Not penalized too heavily for missing properties
        assert!(!analysis.missing_properties.is_empty());
        assert_eq!(analysis.missing_properties.len(), 5);
    }
    
    #[test]
    fn test_property_scoring_performance() {
        let strategy = PropertyCompatibilityStrategy::new();
        
        // Create concept with many properties
        let mut properties = HashMap::new();
        for i in 0..100 {
            properties.insert(format!("prop_{}", i), format!("value_{}", i));
        }
        
        let complex_concept = create_concept_with_properties("complex", 0.8, properties.clone());
        let complex_context = create_context_with_properties("parent", &[], properties);
        
        let start = std::time::Instant::now();
        let score = strategy.score(&complex_concept, &complex_context).unwrap();
        let elapsed = start.elapsed();
        
        // Should handle 100 properties in under 0.3ms
        assert!(elapsed < std::time::Duration::from_micros(300));
        assert!(score > 0.95); // Perfect match should score very high
    }
    
    fn create_concept_with_properties(name: &str, confidence: f32, properties: HashMap<&str, &str>) -> ExtractedConcept {
        use crate::hierarchy_detection::{ExtractedConcept, ConceptType, TextSpan};
        
        let string_properties: HashMap<String, String> = properties.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        
        ExtractedConcept {
            name: name.to_string(),
            concept_type: ConceptType::Entity,
            properties: string_properties,
            source_span: TextSpan {
                start: 0,
                end: name.len(),
                text: name.to_string(),
            },
            confidence,
            suggested_parent: None,
            semantic_features: vec![0.5; 100],
            extracted_at: 0,
        }
    }
    
    fn create_context_with_properties(target: &str, ancestors: &[&str], properties: HashMap<&str, &str>) -> AllocationContext {
        use crate::allocation_scoring::AllocationContext;
        
        let mut context_properties: HashMap<String, String> = properties.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties: context_properties,
            allocation_timestamp: 0,
        }
    }
    
    fn create_context_with_inherited_properties(target: &str, ancestors: &[&str], 
                                              direct_props: HashMap<&str, &str>, 
                                              inherited_props: HashMap<&str, &str>) -> AllocationContext {
        let mut all_properties = HashMap::new();
        
        // Add direct properties
        for (k, v) in direct_props {
            all_properties.insert(k.to_string(), v.to_string());
        }
        
        // Add inherited properties with prefix
        for (k, v) in inherited_props {
            all_properties.insert(format!("inherited_{}", k), v.to_string());
        }
        
        AllocationContext {
            target_concept: target.to_string(),
            ancestor_concepts: ancestors.iter().map(|s| s.to_string()).collect(),
            context_properties: all_properties,
            allocation_timestamp: 0,
        }
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rayon::prelude::*;
use crate::allocation_scoring::{AllocationContext, ScoringStrategy, ScoringError};
use crate::hierarchy_detection::ExtractedConcept;

/// Configuration for property compatibility scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyCompatibilityConfig {
    /// Threshold for considering properties compatible
    pub compatibility_threshold: f32,
    
    /// Whether to consider inheritance in scoring
    pub inheritance_aware: bool,
    
    /// Weight for exact property matches
    pub exact_match_weight: f32,
    
    /// Weight for partial property matches
    pub partial_match_weight: f32,
    
    /// Weight for type compatibility
    pub type_compatibility_weight: f32,
    
    /// Penalty for property conflicts
    pub conflict_penalty: f32,
    
    /// Whether to use fuzzy string matching
    pub use_fuzzy_matching: bool,
    
    /// Threshold for fuzzy string similarity
    pub fuzzy_threshold: f32,
}

impl Default for PropertyCompatibilityConfig {
    fn default() -> Self {
        Self {
            compatibility_threshold: 0.7,
            inheritance_aware: true,
            exact_match_weight: 0.5,
            partial_match_weight: 0.3,
            type_compatibility_weight: 0.1,
            conflict_penalty: 0.3,
            use_fuzzy_matching: true,
            fuzzy_threshold: 0.6,
        }
    }
}

/// Weights for different types of property compatibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyWeights {
    /// Weight for exact property matches
    pub exact_match_weight: f32,
    
    /// Weight for partial/fuzzy matches
    pub partial_match_weight: f32,
    
    /// Weight for type compatibility
    pub type_compatibility_weight: f32,
    
    /// Weight for inheritance compatibility
    pub inheritance_compatibility_weight: f32,
}

impl Default for PropertyWeights {
    fn default() -> Self {
        Self {
            exact_match_weight: 0.4,
            partial_match_weight: 0.3,
            type_compatibility_weight: 0.2,
            inheritance_compatibility_weight: 0.1,
        }
    }
}

impl PropertyWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.exact_match_weight + self.partial_match_weight + 
                   self.type_compatibility_weight + self.inheritance_compatibility_weight;
        
        if total > 0.0 {
            self.exact_match_weight /= total;
            self.partial_match_weight /= total;
            self.type_compatibility_weight /= total;
            self.inheritance_compatibility_weight /= total;
        }
    }
}

/// Property conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConflict {
    /// Name of the conflicting property
    pub property_name: String,
    
    /// Value in the concept
    pub concept_value: String,
    
    /// Expected value from context
    pub expected_value: String,
    
    /// Severity of the conflict (0.0-1.0)
    pub conflict_severity: f32,
    
    /// Type of conflict
    pub conflict_type: ConflictType,
}

/// Types of property conflicts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictType {
    /// Values are completely incompatible
    ValueMismatch,
    
    /// Types are incompatible (e.g., string vs number)
    TypeMismatch,
    
    /// Boolean contradiction
    BooleanContradiction,
    
    /// Numeric range violation
    RangeViolation,
    
    /// Inheritance contradiction
    InheritanceViolation,
}

/// Property exception information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyException {
    /// Name of the property with exception
    pub property_name: String,
    
    /// Value that overrides inherited value
    pub override_value: String,
    
    /// Inherited value being overridden
    pub inherited_value: String,
    
    /// Reason for the exception
    pub exception_reason: String,
    
    /// Confidence in exception detection
    pub confidence: f32,
}

/// Analysis of inheritance compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceCompatibilityAnalysis {
    /// Number of compatible inherited properties
    pub compatible_inherited_properties: usize,
    
    /// Number of conflicting inherited properties
    pub conflicting_inherited_properties: usize,
    
    /// Properties that are exceptions to inheritance
    pub property_exceptions: Vec<PropertyException>,
    
    /// Overall inheritance compatibility score
    pub inheritance_score: f32,
}

/// Analysis of missing properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPropertyAnalysis {
    /// Properties present in context but missing from concept
    pub missing_properties: Vec<String>,
    
    /// Properties that could be inferred or defaulted
    pub inferable_properties: Vec<String>,
    
    /// Impact of missing properties on compatibility
    pub missing_property_impact: f32,
}

/// Property type classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyType {
    String,
    Numeric,
    Boolean,
    Range,
    List,
    Unknown,
}

/// Advanced property compatibility scoring strategy
pub struct PropertyCompatibilityStrategy {
    /// Strategy name
    name: String,
    
    /// Configuration
    config: PropertyCompatibilityConfig,
    
    /// Property weights
    weights: PropertyWeights,
    
    /// Whether strategy is enabled
    enabled: bool,
    
    /// String similarity calculator
    string_similarity: StringSimilarityCalculator,
    
    /// Type analyzer
    type_analyzer: PropertyTypeAnalyzer,
}

impl PropertyCompatibilityStrategy {
    /// Create a new property compatibility strategy
    pub fn new() -> Self {
        Self {
            name: "property_compatibility".to_string(),
            config: PropertyCompatibilityConfig::default(),
            weights: PropertyWeights::default(),
            enabled: true,
            string_similarity: StringSimilarityCalculator::new(),
            type_analyzer: PropertyTypeAnalyzer::new(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: PropertyCompatibilityConfig) -> Self {
        let mut strategy = Self::new();
        strategy.config = config;
        strategy
    }
    
    /// Calculate comprehensive property compatibility score
    fn calculate_property_compatibility(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        let concept_props = &concept.properties;
        let context_props = &context.context_properties;
        
        if concept_props.is_empty() && context_props.is_empty() {
            return Ok(1.0); // No properties to compare
        }
        
        if concept_props.is_empty() || context_props.is_empty() {
            return Ok(0.5); // Neutral score when one side has no properties
        }
        
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        // Calculate exact matches
        let exact_matches = self.calculate_exact_matches(concept_props, context_props);
        let exact_score = exact_matches.0 / exact_matches.1.max(1) as f32;
        total_score += exact_score * self.weights.exact_match_weight;
        total_weight += self.weights.exact_match_weight;
        
        // Calculate partial matches
        let partial_matches = self.calculate_partial_matches(concept_props, context_props)?;
        total_score += partial_matches * self.weights.partial_match_weight;
        total_weight += self.weights.partial_match_weight;
        
        // Calculate type compatibility
        let type_compatibility = self.calculate_type_compatibility(concept_props, context_props)?;
        total_score += type_compatibility * self.weights.type_compatibility_weight;
        total_weight += self.weights.type_compatibility_weight;
        
        // Calculate inheritance compatibility if enabled
        if self.config.inheritance_aware {
            let inheritance_compatibility = self.calculate_inheritance_compatibility(concept, context)?;
            total_score += inheritance_compatibility * self.weights.inheritance_compatibility_weight;
            total_weight += self.weights.inheritance_compatibility_weight;
        }
        
        // Apply conflict penalties
        let conflicts = self.detect_property_conflicts(concept, context)?;
        let conflict_penalty = conflicts.len() as f32 * self.config.conflict_penalty;
        
        let final_score = if total_weight > 0.0 {
            (total_score / total_weight - conflict_penalty).max(0.0).min(1.0)
        } else {
            0.5
        };
        
        Ok(final_score)
    }
    
    /// Calculate exact property matches
    fn calculate_exact_matches(&self, concept_props: &HashMap<String, String>, context_props: &HashMap<String, String>) -> (usize, usize) {
        let mut matches = 0;
        let mut total_comparable = 0;
        
        for (key, concept_value) in concept_props {
            if let Some(context_value) = context_props.get(key) {
                total_comparable += 1;
                if concept_value == context_value {
                    matches += 1;
                }
            }
        }
        
        (matches, total_comparable)
    }
    
    /// Calculate partial/fuzzy matches
    fn calculate_partial_matches(&self, concept_props: &HashMap<String, String>, context_props: &HashMap<String, String>) -> Result<f32, ScoringError> {
        let mut total_similarity = 0.0;
        let mut comparable_pairs = 0;
        
        for (key, concept_value) in concept_props {
            if let Some(context_value) = context_props.get(key) {
                comparable_pairs += 1;
                let similarity = self.calculate_property_similarity(concept_value, context_value);
                total_similarity += similarity;
            }
        }
        
        Ok(if comparable_pairs > 0 {
            total_similarity / comparable_pairs as f32
        } else {
            0.0
        })
    }
    
    /// Calculate property value similarity
    pub fn calculate_property_similarity(&self, value1: &str, value2: &str) -> f32 {
        if value1 == value2 {
            return 1.0;
        }
        
        // Try different similarity approaches based on value types
        let type1 = self.type_analyzer.classify_property_type(value1);
        let type2 = self.type_analyzer.classify_property_type(value2);
        
        match (type1, type2) {
            (PropertyType::Numeric, PropertyType::Numeric) => {
                self.calculate_numeric_similarity(value1, value2)
            }
            (PropertyType::Boolean, PropertyType::Boolean) => {
                self.calculate_boolean_similarity(value1, value2)
            }
            (PropertyType::Range, PropertyType::Numeric) | (PropertyType::Numeric, PropertyType::Range) => {
                self.calculate_range_compatibility(value1, value2)
            }
            _ => {
                // Use string similarity for all other cases
                if self.config.use_fuzzy_matching {
                    self.string_similarity.calculate_similarity(value1, value2)
                } else {
                    0.0
                }
            }
        }
    }
    
    /// Calculate numeric value similarity
    fn calculate_numeric_similarity(&self, value1: &str, value2: &str) -> f32 {
        let num1 = value1.parse::<f64>().ok();
        let num2 = value2.parse::<f64>().ok();
        
        match (num1, num2) {
            (Some(n1), Some(n2)) => {
                let diff = (n1 - n2).abs();
                let max_val = n1.abs().max(n2.abs());
                if max_val > 0.0 {
                    (1.0 - (diff / max_val)).max(0.0) as f32
                } else {
                    1.0
                }
            }
            _ => 0.0,
        }
    }
    
    /// Calculate boolean value similarity
    fn calculate_boolean_similarity(&self, value1: &str, value2: &str) -> f32 {
        let bool1 = self.parse_boolean(value1);
        let bool2 = self.parse_boolean(value2);
        
        match (bool1, bool2) {
            (Some(b1), Some(b2)) => {
                if b1 == b2 { 1.0 } else { 0.0 }
            }
            _ => 0.0,
        }
    }
    
    /// Parse boolean values from strings
    fn parse_boolean(&self, value: &str) -> Option<bool> {
        let lower = value.to_lowercase();
        match lower.as_str() {
            "true" | "yes" | "1" | "on" | "enabled" => Some(true),
            "false" | "no" | "0" | "off" | "disabled" => Some(false),
            _ => None,
        }
    }
    
    /// Calculate range compatibility
    fn calculate_range_compatibility(&self, value1: &str, value2: &str) -> f32 {
        // Simplified range compatibility - could be more sophisticated
        if value1.contains("-") || value2.contains("-") {
            // One is a range, try to check if the other falls within
            0.5 // Placeholder implementation
        } else {
            self.calculate_numeric_similarity(value1, value2)
        }
    }
    
    /// Calculate type compatibility between property sets
    fn calculate_type_compatibility(&self, concept_props: &HashMap<String, String>, context_props: &HashMap<String, String>) -> Result<f32, ScoringError> {
        let mut compatible_types = 0;
        let mut total_compared = 0;
        
        for (key, concept_value) in concept_props {
            if let Some(context_value) = context_props.get(key) {
                total_compared += 1;
                let concept_type = self.type_analyzer.classify_property_type(concept_value);
                let context_type = self.type_analyzer.classify_property_type(context_value);
                
                if self.are_types_compatible(&concept_type, &context_type) {
                    compatible_types += 1;
                }
            }
        }
        
        Ok(if total_compared > 0 {
            compatible_types as f32 / total_compared as f32
        } else {
            1.0
        })
    }
    
    /// Check if two property types are compatible
    fn are_types_compatible(&self, type1: &PropertyType, type2: &PropertyType) -> bool {
        match (type1, type2) {
            (PropertyType::Numeric, PropertyType::Numeric) => true,
            (PropertyType::Boolean, PropertyType::Boolean) => true,
            (PropertyType::String, PropertyType::String) => true,
            (PropertyType::Range, PropertyType::Numeric) | (PropertyType::Numeric, PropertyType::Range) => true,
            (PropertyType::List, PropertyType::List) => true,
            (PropertyType::Unknown, _) | (_, PropertyType::Unknown) => true, // Assume compatible if unknown
            _ => false,
        }
    }
    
    /// Calculate inheritance compatibility
    fn calculate_inheritance_compatibility(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        // Look for inherited properties (prefixed with "inherited_")
        let inherited_props: HashMap<String, String> = context.context_properties.iter()
            .filter_map(|(k, v)| {
                if k.starts_with("inherited_") {
                    Some((k.strip_prefix("inherited_").unwrap().to_string(), v.clone()))
                } else {
                    None
                }
            })
            .collect();
        
        if inherited_props.is_empty() {
            return Ok(1.0); // No inheritance to check
        }
        
        let mut compatible_count = 0;
        let mut total_inherited = 0;
        
        for (inherited_key, inherited_value) in &inherited_props {
            total_inherited += 1;
            
            if let Some(concept_value) = concept.properties.get(inherited_key) {
                // Property exists in concept - check compatibility
                let similarity = self.calculate_property_similarity(concept_value, inherited_value);
                if similarity > self.config.fuzzy_threshold {
                    compatible_count += 1;
                }
            } else {
                // Property not in concept - assume inherited
                compatible_count += 1;
            }
        }
        
        Ok(if total_inherited > 0 {
            compatible_count as f32 / total_inherited as f32
        } else {
            1.0
        })
    }
    
    /// Detect property conflicts between concept and context
    pub fn detect_property_conflicts(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<Vec<PropertyConflict>, ScoringError> {
        let mut conflicts = Vec::new();
        
        for (key, concept_value) in &concept.properties {
            if let Some(context_value) = context.context_properties.get(key) {
                let similarity = self.calculate_property_similarity(concept_value, context_value);
                
                if similarity < 0.3 { // Threshold for conflict
                    let conflict_type = self.classify_conflict_type(concept_value, context_value);
                    let conflict_severity = 1.0 - similarity;
                    
                    conflicts.push(PropertyConflict {
                        property_name: key.clone(),
                        concept_value: concept_value.clone(),
                        expected_value: context_value.clone(),
                        conflict_severity,
                        conflict_type,
                    });
                }
            }
        }
        
        Ok(conflicts)
    }
    
    /// Classify the type of conflict between two property values
    fn classify_conflict_type(&self, value1: &str, value2: &str) -> ConflictType {
        let type1 = self.type_analyzer.classify_property_type(value1);
        let type2 = self.type_analyzer.classify_property_type(value2);
        
        if type1 != type2 {
            return ConflictType::TypeMismatch;
        }
        
        match type1 {
            PropertyType::Boolean => {
                let bool1 = self.parse_boolean(value1);
                let bool2 = self.parse_boolean(value2);
                if let (Some(b1), Some(b2)) = (bool1, bool2) {
                    if b1 != b2 {
                        ConflictType::BooleanContradiction
                    } else {
                        ConflictType::ValueMismatch
                    }
                } else {
                    ConflictType::ValueMismatch
                }
            }
            PropertyType::Numeric => {
                // Could check for range violations here
                ConflictType::ValueMismatch
            }
            _ => ConflictType::ValueMismatch,
        }
    }
    
    /// Identify property exceptions
    pub fn identify_property_exceptions(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<Vec<PropertyException>, ScoringError> {
        let mut exceptions = Vec::new();
        
        // Look for inherited properties that are overridden
        for (key, concept_value) in &concept.properties {
            let inherited_key = format!("inherited_{}", key);
            if let Some(inherited_value) = context.context_properties.get(&inherited_key) {
                if concept_value != inherited_value {
                    // This is a potential exception
                    let similarity = self.calculate_property_similarity(concept_value, inherited_value);
                    
                    if similarity < 0.5 { // Threshold for considering it an exception
                        exceptions.push(PropertyException {
                            property_name: key.clone(),
                            override_value: concept_value.clone(),
                            inherited_value: inherited_value.clone(),
                            exception_reason: "Property value differs from inherited value".to_string(),
                            confidence: 1.0 - similarity,
                        });
                    }
                }
            }
        }
        
        Ok(exceptions)
    }
    
    /// Analyze inheritance compatibility in detail
    pub fn analyze_inheritance_compatibility(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<InheritanceCompatibilityAnalysis, ScoringError> {
        let inherited_props: Vec<_> = context.context_properties.iter()
            .filter(|(k, _)| k.starts_with("inherited_"))
            .collect();
        
        let mut compatible_count = 0;
        let mut conflicting_count = 0;
        
        for (inherited_key, inherited_value) in &inherited_props {
            let actual_key = inherited_key.strip_prefix("inherited_").unwrap();
            
            if let Some(concept_value) = concept.properties.get(actual_key) {
                let similarity = self.calculate_property_similarity(concept_value, inherited_value);
                
                if similarity > 0.7 {
                    compatible_count += 1;
                } else {
                    conflicting_count += 1;
                }
            } else {
                // Missing property is considered compatible (inherited)
                compatible_count += 1;
            }
        }
        
        let property_exceptions = self.identify_property_exceptions(concept, context)?;
        
        let inheritance_score = if inherited_props.len() > 0 {
            compatible_count as f32 / inherited_props.len() as f32
        } else {
            1.0
        };
        
        Ok(InheritanceCompatibilityAnalysis {
            compatible_inherited_properties: compatible_count,
            conflicting_inherited_properties: conflicting_count,
            property_exceptions,
            inheritance_score,
        })
    }
    
    /// Analyze missing properties
    pub fn analyze_missing_properties(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<MissingPropertyAnalysis, ScoringError> {
        let mut missing_properties = Vec::new();
        let mut inferable_properties = Vec::new();
        
        for key in context.context_properties.keys() {
            if !key.starts_with("inherited_") && !concept.properties.contains_key(key) {
                missing_properties.push(key.clone());
                
                // Simple heuristic for inferable properties
                if key.contains("color") || key.contains("size") || key.contains("type") {
                    inferable_properties.push(key.clone());
                }
            }
        }
        
        let missing_property_impact = if context.context_properties.len() > 0 {
            missing_properties.len() as f32 / context.context_properties.len() as f32
        } else {
            0.0
        };
        
        Ok(MissingPropertyAnalysis {
            missing_properties,
            inferable_properties,
            missing_property_impact,
        })
    }
    
    /// Batch score multiple concepts
    pub fn batch_score(&self, concepts: &[ExtractedConcept], context: &AllocationContext) -> Result<Vec<f32>, ScoringError> {
        let scores: Result<Vec<_>, _> = concepts.par_iter()
            .map(|concept| self.calculate_property_compatibility(concept, context))
            .collect();
        
        scores
    }
    
    /// Get compatibility threshold
    pub fn get_compatibility_threshold(&self) -> f32 {
        self.config.compatibility_threshold
    }
    
    /// Set inheritance awareness
    pub fn set_inheritance_aware(&mut self, aware: bool) {
        self.config.inheritance_aware = aware;
    }
    
    /// Set property weights
    pub fn set_property_weights(&mut self, mut weights: PropertyWeights) {
        weights.normalize();
        self.weights = weights;
    }
    
    /// Get property weights
    pub fn get_property_weights(&self) -> &PropertyWeights {
        &self.weights
    }
    
    /// Check if strategy is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl ScoringStrategy for PropertyCompatibilityStrategy {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn score(&self, concept: &ExtractedConcept, context: &AllocationContext) -> Result<f32, ScoringError> {
        if !self.enabled {
            return Ok(0.0);
        }
        
        self.calculate_property_compatibility(concept, context)
    }
    
    fn supports_parallel(&self) -> bool {
        true
    }
    
    fn weight_preference(&self) -> f32 {
        0.25 // Property compatibility is moderately important
    }
}

/// String similarity calculator using various algorithms
pub struct StringSimilarityCalculator {
    // Configuration for similarity algorithms
}

impl StringSimilarityCalculator {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Calculate similarity between two strings using multiple methods
    pub fn calculate_similarity(&self, s1: &str, s2: &str) -> f32 {
        if s1 == s2 {
            return 1.0;
        }
        
        // Use a combination of similarity measures
        let levenshtein_sim = self.levenshtein_similarity(s1, s2);
        let jaro_sim = self.jaro_similarity(s1, s2);
        let token_sim = self.token_similarity(s1, s2);
        
        // Weighted combination
        (levenshtein_sim * 0.4 + jaro_sim * 0.3 + token_sim * 0.3).max(0.0).min(1.0)
    }
    
    /// Levenshtein distance based similarity
    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f32 {
        let distance = self.levenshtein_distance(s1, s2);
        let max_len = s1.len().max(s2.len());
        
        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f32 / max_len as f32)
        }
    }
    
    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let v1: Vec<char> = s1.chars().collect();
        let v2: Vec<char> = s2.chars().collect();
        let len1 = v1.len();
        let len2 = v2.len();
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if v1[i - 1] == v2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }
        
        matrix[len1][len2]
    }
    
    /// Simplified Jaro similarity
    fn jaro_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Simplified implementation for brevity
        if s1 == s2 {
            1.0
        } else if s1.len() == 0 || s2.len() == 0 {
            0.0
        } else {
            let common_chars = self.count_common_chars(s1, s2);
            if common_chars == 0 {
                0.0
            } else {
                let avg_len = (s1.len() + s2.len()) as f32 / 2.0;
                common_chars as f32 / avg_len
            }
        }
    }
    
    /// Count common characters (simplified)
    fn count_common_chars(&self, s1: &str, s2: &str) -> usize {
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();
        chars1.intersection(&chars2).count()
    }
    
    /// Token-based similarity
    fn token_similarity(&self, s1: &str, s2: &str) -> f32 {
        let tokens1: std::collections::HashSet<&str> = s1.split_whitespace().collect();
        let tokens2: std::collections::HashSet<&str> = s2.split_whitespace().collect();
        
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }
        
        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Property type analyzer
pub struct PropertyTypeAnalyzer {
    // Patterns and rules for type classification
}

impl PropertyTypeAnalyzer {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Classify the type of a property value
    pub fn classify_property_type(&self, value: &str) -> PropertyType {
        let trimmed = value.trim().to_lowercase();
        
        // Check for boolean values
        if matches!(trimmed.as_str(), "true" | "false" | "yes" | "no" | "1" | "0" | "on" | "off" | "enabled" | "disabled") {
            return PropertyType::Boolean;
        }
        
        // Check for numeric values
        if trimmed.parse::<f64>().is_ok() {
            return PropertyType::Numeric;
        }
        
        // Check for ranges (contains dash between numbers)
        if trimmed.contains("-") && self.looks_like_range(&trimmed) {
            return PropertyType::Range;
        }
        
        // Check for lists (contains commas or semicolons)
        if trimmed.contains(",") || trimmed.contains(";") {
            return PropertyType::List;
        }
        
        // Default to string
        PropertyType::String
    }
    
    /// Check if a string looks like a numeric range
    fn looks_like_range(&self, value: &str) -> bool {
        let parts: Vec<&str> = value.split("-").collect();
        if parts.len() == 2 {
            parts[0].trim().parse::<f64>().is_ok() && parts[1].trim().parse::<f64>().is_ok()
        } else {
            false
        }
    }
}

impl Default for PropertyCompatibilityStrategy {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create PropertyCompatibilityStrategy with configurable weights and fuzzy matching
2. Implement comprehensive property comparison including type analysis
3. Add inheritance-aware compatibility scoring with exception detection
4. Implement conflict detection and property exception identification
5. Add batch processing capabilities for efficient parallel evaluation
6. Ensure performance meets <0.3ms per evaluation requirement

## Success Criteria
- [ ] PropertyCompatibilityStrategy compiles without errors
- [ ] Property similarity calculation handles different data types correctly
- [ ] Inheritance compatibility analysis detects exceptions accurately
- [ ] Conflict detection identifies incompatible property values
- [ ] Batch scoring achieves <0.3ms average per evaluation
- [ ] All tests pass with comprehensive coverage