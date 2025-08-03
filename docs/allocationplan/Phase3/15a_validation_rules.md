# Task 15a: Create Validation Rules Types

**Time**: 5 minutes
**Dependencies**: 14l_error_mod_file.md
**Stage**: Inheritance System

## Objective
Create data structures for inheritance validation rules and constraints.

## Implementation
Create `src/inheritance/validation/rules.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    pub structural_rules: StructuralRules,
    pub semantic_rules: SemanticRules,
    pub performance_rules: PerformanceRules,
    pub custom_rules: Vec<CustomRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralRules {
    pub max_inheritance_depth: u32,
    pub allow_multiple_inheritance: bool,
    pub allow_circular_references: bool,
    pub max_children_per_concept: Option<u32>,
    pub require_unique_property_names: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRules {
    pub require_concept_existence: bool,
    pub validate_property_types: bool,
    pub enforce_inheritance_compatibility: bool,
    pub validate_relationship_semantics: bool,
    pub check_naming_conventions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRules {
    pub max_resolution_time_ms: u64,
    pub max_cache_memory_mb: usize,
    pub warn_on_deep_chains: bool,
    pub max_concurrent_operations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub rule_type: CustomRuleType,
    pub parameters: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CustomRuleType {
    ConceptNaming,
    PropertyConstraint,
    InheritancePattern,
    PerformanceThreshold,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            structural_rules: StructuralRules {
                max_inheritance_depth: 20,
                allow_multiple_inheritance: true,
                allow_circular_references: false,
                max_children_per_concept: Some(100),
                require_unique_property_names: true,
            },
            semantic_rules: SemanticRules {
                require_concept_existence: true,
                validate_property_types: true,
                enforce_inheritance_compatibility: true,
                validate_relationship_semantics: true,
                check_naming_conventions: true,
            },
            performance_rules: PerformanceRules {
                max_resolution_time_ms: 1000,
                max_cache_memory_mb: 100,
                warn_on_deep_chains: true,
                max_concurrent_operations: 50,
            },
            custom_rules: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub rule_id: String,
    pub severity: ValidationSeverity,
    pub message: String,
    pub concept_id: Option<String>,
    pub property_name: Option<String>,
    pub suggested_fix: Option<String>,
}

impl ValidationResult {
    pub fn new(rule_id: &str, severity: ValidationSeverity, message: &str) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            severity,
            message: message.to_string(),
            concept_id: None,
            property_name: None,
            suggested_fix: None,
        }
    }

    pub fn with_concept(mut self, concept_id: &str) -> Self {
        self.concept_id = Some(concept_id.to_string());
        self
    }

    pub fn with_property(mut self, property_name: &str) -> Self {
        self.property_name = Some(property_name.to_string());
        self
    }

    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggested_fix = Some(suggestion.to_string());
        self
    }
}
```

## Success Criteria
- Validation rules are comprehensively defined
- Default rules provide good baseline
- Severity levels are appropriate

## Next Task
15b_structural_validator.md