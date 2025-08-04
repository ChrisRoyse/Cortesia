# Task 08: Ambiguity Detection

## Metadata
- **Micro-Phase**: 2.8
- **Duration**: 15-20 minutes
- **Dependencies**: Task 05 (validated_fact_structure)
- **Output**: `src/quality_integration/ambiguity_detector.rs`

## Description
Create the AmbiguityDetector that identifies and categorizes different types of ambiguities in validated facts. This component helps ensure facts are sufficiently clear and unambiguous before allocation to the neuromorphic system.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{ValidatedFact, FactContent, ConfidenceComponents};

    #[test]
    fn test_ambiguity_detector_creation() {
        let detector = AmbiguityDetector::new();
        assert_eq!(detector.detection_rules.len(), 5);
        assert!(detector.is_enabled);
    }
    
    #[test]
    fn test_entity_ambiguity_detection() {
        let detector = AmbiguityDetector::new();
        
        // Test clear entity reference
        let clear_content = FactContent::new("The African elephant has large ears");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let clear_fact = ValidatedFact::new(clear_content, confidence);
        
        let result = detector.detect_entity_ambiguities(&clear_fact);
        assert!(result.ambiguities.is_empty());
        assert_eq!(result.severity, AmbiguitySeverity::None);
        
        // Test ambiguous pronouns
        let ambiguous_content = FactContent::new("It has large ears and it trumpets loudly");
        let ambiguous_fact = ValidatedFact::new(ambiguous_content, confidence);
        
        let ambiguous_result = detector.detect_entity_ambiguities(&ambiguous_fact);
        assert!(!ambiguous_result.ambiguities.is_empty());
        assert!(ambiguous_result.severity != AmbiguitySeverity::None);
    }
    
    #[test]
    fn test_temporal_ambiguity_detection() {
        let detector = AmbiguityDetector::new();
        
        // Test clear temporal reference
        let clear_content = FactContent::new("In 1969, humans first landed on the moon");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let clear_fact = ValidatedFact::new(clear_content, confidence);
        
        let result = detector.detect_temporal_ambiguities(&clear_fact);
        assert!(result.ambiguities.is_empty());
        
        // Test ambiguous temporal reference
        let ambiguous_content = FactContent::new("Recently, scientists discovered something");
        let ambiguous_fact = ValidatedFact::new(ambiguous_content, confidence);
        
        let ambiguous_result = detector.detect_temporal_ambiguities(&ambiguous_fact);
        assert!(!ambiguous_result.ambiguities.is_empty());
        assert!(ambiguous_result.severity != AmbiguitySeverity::None);
    }
    
    #[test]
    fn test_semantic_ambiguity_detection() {
        let detector = AmbiguityDetector::new();
        
        // Test clear semantic meaning
        let clear_content = FactContent::new("Water boils at 100 degrees Celsius");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let clear_fact = ValidatedFact::new(clear_content, confidence);
        
        let result = detector.detect_semantic_ambiguities(&clear_fact);
        assert!(result.ambiguities.len() <= 1); // Might detect minor issues
        
        // Test semantically ambiguous
        let ambiguous_content = FactContent::new("The bank is near the bank by the river");
        let ambiguous_fact = ValidatedFact::new(ambiguous_content, confidence);
        
        let ambiguous_result = detector.detect_semantic_ambiguities(&ambiguous_fact);
        assert!(!ambiguous_result.ambiguities.is_empty());
    }
    
    #[test]
    fn test_comprehensive_ambiguity_detection() {
        let detector = AmbiguityDetector::new();
        
        // Test fact with multiple ambiguity types
        let content = FactContent::new("It was recently found that they often do that thing there");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let fact = ValidatedFact::new(content, confidence);
        
        let result = detector.detect_all_ambiguities(&fact);
        assert!(!result.detected_ambiguities.is_empty());
        assert!(result.total_ambiguity_count > 0);
        assert!(result.severity == AmbiguitySeverity::High || result.severity == AmbiguitySeverity::Critical);
        
        // Test clear fact
        let clear_content = FactContent::new("The African elephant weighs up to 6 tons");
        let clear_fact = ValidatedFact::new(clear_content, confidence);
        
        let clear_result = detector.detect_all_ambiguities(&clear_fact);
        assert!(clear_result.total_ambiguity_count <= 1); // Should be minimal
        assert!(clear_result.severity == AmbiguitySeverity::None || clear_result.severity == AmbiguitySeverity::Low);
    }
    
    #[test]
    fn test_ambiguity_severity_classification() {
        let detector = AmbiguityDetector::new();
        
        let content = FactContent::new("Test content");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let fact = ValidatedFact::new(content, confidence);
        
        // Test severity calculation
        assert_eq!(detector.calculate_severity(0), AmbiguitySeverity::None);
        assert_eq!(detector.calculate_severity(1), AmbiguitySeverity::Low);
        assert_eq!(detector.calculate_severity(3), AmbiguitySeverity::Medium);
        assert_eq!(detector.calculate_severity(5), AmbiguitySeverity::High);
        assert_eq!(detector.calculate_severity(8), AmbiguitySeverity::Critical);
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::quality_integration::ValidatedFact;
use regex::Regex;

/// Types of ambiguities that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmbiguityType {
    /// Unclear entity references (pronouns, etc.)
    EntityReference,
    /// Vague temporal references
    TemporalVagueness,
    /// Semantic ambiguity (multiple meanings)
    SemanticAmbiguity,
    /// Incomplete information
    IncompleteInformation,
    /// Contradictory statements
    Contradiction,
}

/// Severity levels for ambiguities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbiguitySeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Individual ambiguity detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityInstance {
    /// Type of ambiguity detected
    pub ambiguity_type: AmbiguityType,
    
    /// Specific text that triggered the detection
    pub problematic_text: String,
    
    /// Position in the text (start, end)
    pub text_position: (usize, usize),
    
    /// Detailed description of the issue
    pub description: String,
    
    /// Confidence that this is actually an ambiguity (0.0-1.0)
    pub detection_confidence: f32,
    
    /// Suggested resolution if available
    pub suggested_resolution: Option<String>,
}

/// Result of ambiguity detection for a specific type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityDetectionResult {
    /// Type of ambiguity checked
    pub ambiguity_type: AmbiguityType,
    
    /// List of detected ambiguities
    pub ambiguities: Vec<AmbiguityInstance>,
    
    /// Overall severity for this type
    pub severity: AmbiguitySeverity,
    
    /// Timestamp of detection
    pub detected_at: u64,
}

/// Comprehensive ambiguity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAmbiguityResult {
    /// All detected ambiguities by type
    pub detected_ambiguities: HashMap<AmbiguityType, AmbiguityDetectionResult>,
    
    /// Total number of ambiguities found
    pub total_ambiguity_count: usize,
    
    /// Overall severity across all types
    pub severity: AmbiguitySeverity,
    
    /// Whether the fact should be rejected due to ambiguities
    pub should_reject: bool,
    
    /// Detailed analysis summary
    pub analysis_summary: String,
    
    /// Timestamp of comprehensive analysis
    pub analyzed_at: u64,
}

/// Detection rule for specific ambiguity patterns
#[derive(Debug, Clone)]
pub struct DetectionRule {
    /// Pattern to match (regex)
    pub pattern: Regex,
    
    /// Type of ambiguity this rule detects
    pub ambiguity_type: AmbiguityType,
    
    /// Description of what this rule detects
    pub description: String,
    
    /// Base confidence for detections from this rule
    pub base_confidence: f32,
}

/// Main ambiguity detection engine
#[derive(Debug, Clone)]
pub struct AmbiguityDetector {
    /// Whether detection is enabled
    pub is_enabled: bool,
    
    /// Collection of detection rules
    pub detection_rules: Vec<DetectionRule>,
    
    /// Severity thresholds
    pub severity_thresholds: HashMap<AmbiguitySeverity, usize>,
    
    /// Maximum allowed ambiguities before rejection
    pub max_allowed_ambiguities: usize,
}

impl AmbiguityDetector {
    /// Create a new ambiguity detector with default rules
    pub fn new() -> Self {
        let mut detector = Self {
            is_enabled: true,
            detection_rules: Vec::new(),
            severity_thresholds: HashMap::new(),
            max_allowed_ambiguities: 3,
        };
        
        detector.initialize_default_rules();
        detector.initialize_severity_thresholds();
        detector
    }
    
    /// Initialize default detection rules
    fn initialize_default_rules(&mut self) {
        // Pronoun ambiguity detection
        self.detection_rules.push(DetectionRule {
            pattern: Regex::new(r"\b(it|they|this|that|those|these)\b").unwrap(),
            ambiguity_type: AmbiguityType::EntityReference,
            description: "Ambiguous pronoun reference".to_string(),
            base_confidence: 0.7,
        });
        
        // Temporal vagueness detection
        self.detection_rules.push(DetectionRule {
            pattern: Regex::new(r"\b(recently|lately|soon|eventually|sometimes|often|usually)\b").unwrap(),
            ambiguity_type: AmbiguityType::TemporalVagueness,
            description: "Vague temporal reference".to_string(),
            base_confidence: 0.8,
        });
        
        // Incomplete information patterns
        self.detection_rules.push(DetectionRule {
            pattern: Regex::new(r"\b(something|someone|somewhere|somehow|some|various|several)\b").unwrap(),
            ambiguity_type: AmbiguityType::IncompleteInformation,
            description: "Incomplete or vague information".to_string(),
            base_confidence: 0.6,
        });
        
        // Semantic ambiguity (common ambiguous words)
        self.detection_rules.push(DetectionRule {
            pattern: Regex::new(r"\b(bank|bark|fair|light|right|left|point|case|run|set)\b").unwrap(),
            ambiguity_type: AmbiguityType::SemanticAmbiguity,
            description: "Potentially ambiguous word meaning".to_string(),
            base_confidence: 0.5,
        });
        
        // Contradiction indicators
        self.detection_rules.push(DetectionRule {
            pattern: Regex::new(r"\b(but|however|although|despite|nevertheless|contradicts|opposite)\b").unwrap(),
            ambiguity_type: AmbiguityType::Contradiction,
            description: "Potential contradiction or conflicting information".to_string(),
            base_confidence: 0.4,
        });
    }
    
    /// Initialize severity thresholds
    fn initialize_severity_thresholds(&mut self) {
        self.severity_thresholds.insert(AmbiguitySeverity::None, 0);
        self.severity_thresholds.insert(AmbiguitySeverity::Low, 1);
        self.severity_thresholds.insert(AmbiguitySeverity::Medium, 2);
        self.severity_thresholds.insert(AmbiguitySeverity::High, 4);
        self.severity_thresholds.insert(AmbiguitySeverity::Critical, 7);
    }
    
    /// Detect entity reference ambiguities
    pub fn detect_entity_ambiguities(&self, fact: &ValidatedFact) -> AmbiguityDetectionResult {
        self.detect_ambiguities_by_type(fact, AmbiguityType::EntityReference)
    }
    
    /// Detect temporal ambiguities
    pub fn detect_temporal_ambiguities(&self, fact: &ValidatedFact) -> AmbiguityDetectionResult {
        self.detect_ambiguities_by_type(fact, AmbiguityType::TemporalVagueness)
    }
    
    /// Detect semantic ambiguities
    pub fn detect_semantic_ambiguities(&self, fact: &ValidatedFact) -> AmbiguityDetectionResult {
        self.detect_ambiguities_by_type(fact, AmbiguityType::SemanticAmbiguity)
    }
    
    /// Detect ambiguities of a specific type
    fn detect_ambiguities_by_type(&self, fact: &ValidatedFact, ambiguity_type: AmbiguityType) -> AmbiguityDetectionResult {
        let mut ambiguities = Vec::new();
        let text = &fact.content.text;
        
        for rule in &self.detection_rules {
            if rule.ambiguity_type == ambiguity_type {
                for mat in rule.pattern.find_iter(text) {
                    let instance = AmbiguityInstance {
                        ambiguity_type: ambiguity_type.clone(),
                        problematic_text: mat.as_str().to_string(),
                        text_position: (mat.start(), mat.end()),
                        description: rule.description.clone(),
                        detection_confidence: rule.base_confidence,
                        suggested_resolution: self.suggest_resolution(&ambiguity_type, mat.as_str()),
                    };
                    ambiguities.push(instance);
                }
            }
        }
        
        let severity = self.calculate_severity(ambiguities.len());
        
        AmbiguityDetectionResult {
            ambiguity_type,
            ambiguities,
            severity,
            detected_at: current_timestamp(),
        }
    }
    
    /// Detect all types of ambiguities
    pub fn detect_all_ambiguities(&self, fact: &ValidatedFact) -> ComprehensiveAmbiguityResult {
        if !self.is_enabled {
            return ComprehensiveAmbiguityResult {
                detected_ambiguities: HashMap::new(),
                total_ambiguity_count: 0,
                severity: AmbiguitySeverity::None,
                should_reject: false,
                analysis_summary: "Ambiguity detection disabled".to_string(),
                analyzed_at: current_timestamp(),
            };
        }
        
        let mut detected_ambiguities = HashMap::new();
        let mut total_count = 0;
        
        // Check each ambiguity type
        for ambiguity_type in [
            AmbiguityType::EntityReference,
            AmbiguityType::TemporalVagueness,
            AmbiguityType::SemanticAmbiguity,
            AmbiguityType::IncompleteInformation,
            AmbiguityType::Contradiction,
        ] {
            let result = self.detect_ambiguities_by_type(fact, ambiguity_type.clone());
            total_count += result.ambiguities.len();
            detected_ambiguities.insert(ambiguity_type, result);
        }
        
        let overall_severity = self.calculate_severity(total_count);
        let should_reject = total_count > self.max_allowed_ambiguities;
        
        let analysis_summary = format!(
            "Detected {} ambiguities across {} types. Severity: {:?}",
            total_count,
            detected_ambiguities.len(),
            overall_severity
        );
        
        ComprehensiveAmbiguityResult {
            detected_ambiguities,
            total_ambiguity_count: total_count,
            severity: overall_severity,
            should_reject,
            analysis_summary,
            analyzed_at: current_timestamp(),
        }
    }
    
    /// Calculate severity based on ambiguity count
    pub fn calculate_severity(&self, count: usize) -> AmbiguitySeverity {
        if count == 0 {
            AmbiguitySeverity::None
        } else if count == 1 {
            AmbiguitySeverity::Low
        } else if count <= 2 {
            AmbiguitySeverity::Medium
        } else if count <= 4 {
            AmbiguitySeverity::High
        } else {
            AmbiguitySeverity::Critical
        }
    }
    
    /// Suggest resolution for detected ambiguity
    fn suggest_resolution(&self, ambiguity_type: &AmbiguityType, text: &str) -> Option<String> {
        match ambiguity_type {
            AmbiguityType::EntityReference => {
                Some("Replace pronoun with specific entity name".to_string())
            },
            AmbiguityType::TemporalVagueness => {
                Some("Specify exact date, time, or time period".to_string())
            },
            AmbiguityType::SemanticAmbiguity => {
                Some("Provide additional context to clarify meaning".to_string())
            },
            AmbiguityType::IncompleteInformation => {
                Some("Provide more specific details".to_string())
            },
            AmbiguityType::Contradiction => {
                Some("Resolve conflicting information".to_string())
            },
        }
    }
    
    /// Update maximum allowed ambiguities
    pub fn set_max_allowed_ambiguities(&mut self, max: usize) {
        self.max_allowed_ambiguities = max;
    }
    
    /// Enable or disable detection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }
    
    /// Add custom detection rule
    pub fn add_custom_rule(&mut self, rule: DetectionRule) {
        self.detection_rules.push(rule);
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for AmbiguityDetector {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create AmbiguityDetector with configurable detection rules
2. Implement detection methods for different ambiguity types
3. Add severity classification and threshold management
4. Implement comprehensive detection that combines all types
5. Ensure detection rules are accurate and configurable

## Success Criteria
- [ ] AmbiguityDetector struct compiles without errors
- [ ] Individual ambiguity type detection works correctly
- [ ] Severity classification accurately reflects ambiguity levels
- [ ] Comprehensive detection combines results appropriately
- [ ] All tests pass with realistic ambiguity scenarios