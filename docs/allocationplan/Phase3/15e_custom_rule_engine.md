# Task 15e: Implement Custom Rule Engine

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 15d_performance_validator.md
**Stage**: Inheritance System

## Objective
Create engine for executing custom validation rules.

## Implementation
Create `src/inheritance/validation/custom_rule_engine.rs`:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use crate::inheritance::validation::rules::*;

#[async_trait]
pub trait CustomRuleExecutor: Send + Sync {
    async fn execute(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>>;
    fn rule_type(&self) -> CustomRuleType;
    fn rule_id(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct ValidationContext {
    pub concept_id: Option<String>,
    pub property_name: Option<String>,
    pub operation: String,
    pub parameters: HashMap<String, String>,
}

pub struct CustomRuleEngine {
    executors: HashMap<String, Arc<dyn CustomRuleExecutor>>,
    rules: Vec<CustomRule>,
}

impl CustomRuleEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            executors: HashMap::new(),
            rules: Vec::new(),
        };
        
        // Register built-in custom rule executors
        engine.register_builtin_executors();
        engine
    }

    fn register_builtin_executors(&mut self) {
        self.register_executor(Arc::new(ConceptNamingRuleExecutor));
        self.register_executor(Arc::new(PropertyConstraintRuleExecutor));
        self.register_executor(Arc::new(InheritancePatternRuleExecutor));
        self.register_executor(Arc::new(PerformanceThresholdRuleExecutor));
    }

    pub fn register_executor(&mut self, executor: Arc<dyn CustomRuleExecutor>) {
        self.executors.insert(executor.rule_id().to_string(), executor);
    }

    pub fn add_rule(&mut self, rule: CustomRule) {
        self.rules.push(rule);
    }

    pub fn remove_rule(&mut self, rule_id: &str) {
        self.rules.retain(|r| r.id != rule_id);
    }

    pub async fn execute_rules(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            
            if let Some(executor) = self.executors.get(&rule.id) {
                let mut rule_context = context.clone();
                rule_context.parameters.extend(rule.parameters.clone());
                
                match executor.execute(&rule_context).await {
                    Ok(mut rule_results) => results.append(&mut rule_results),
                    Err(e) => {
                        results.push(ValidationResult::new(
                            &rule.id,
                            ValidationSeverity::Error,
                            &format!("Failed to execute custom rule '{}': {}", rule.name, e)
                        ));
                    }
                }
            } else {
                results.push(ValidationResult::new(
                    &rule.id,
                    ValidationSeverity::Error,
                    &format!("No executor found for custom rule '{}'", rule.name)
                ));
            }
        }
        
        Ok(results)
    }

    pub fn get_enabled_rules(&self) -> Vec<&CustomRule> {
        self.rules.iter().filter(|r| r.enabled).collect()
    }

    pub fn get_rules_by_type(&self, rule_type: &CustomRuleType) -> Vec<&CustomRule> {
        self.rules.iter().filter(|r| &r.rule_type == rule_type).collect()
    }
}

// Built-in custom rule executors

pub struct ConceptNamingRuleExecutor;

#[async_trait]
impl CustomRuleExecutor for ConceptNamingRuleExecutor {
    async fn execute(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if let Some(concept_id) = &context.concept_id {
            let pattern = context.parameters.get("pattern").unwrap_or(&"^[a-z][a-z0-9_]*$".to_string());
            
            if !self.matches_pattern(concept_id, pattern) {
                results.push(
                    ValidationResult::new(
                        "concept_naming_rule",
                        ValidationSeverity::Warning,
                        &format!("Concept name '{}' does not match required pattern '{}'", concept_id, pattern)
                    )
                    .with_concept(concept_id)
                );
            }
        }
        
        Ok(results)
    }

    fn rule_type(&self) -> CustomRuleType {
        CustomRuleType::ConceptNaming
    }

    fn rule_id(&self) -> &str {
        "concept_naming_rule"
    }
}

impl ConceptNamingRuleExecutor {
    fn matches_pattern(&self, text: &str, pattern: &str) -> bool {
        // Simple pattern matching - in a real implementation, use regex
        match pattern {
            "^[a-z][a-z0-9_]*$" => {
                !text.is_empty() && 
                text.chars().next().unwrap().is_lowercase() &&
                text.chars().all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
            }
            _ => true, // Default to pass for unknown patterns
        }
    }
}

pub struct PropertyConstraintRuleExecutor;

#[async_trait]
impl CustomRuleExecutor for PropertyConstraintRuleExecutor {
    async fn execute(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if let Some(property_name) = &context.property_name {
            let min_length = context.parameters.get("min_length")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            
            if property_name.len() < min_length {
                results.push(
                    ValidationResult::new(
                        "property_constraint_rule",
                        ValidationSeverity::Warning,
                        &format!("Property name '{}' is shorter than minimum length {}", property_name, min_length)
                    )
                    .with_property(property_name)
                );
            }
        }
        
        Ok(results)
    }

    fn rule_type(&self) -> CustomRuleType {
        CustomRuleType::PropertyConstraint
    }

    fn rule_id(&self) -> &str {
        "property_constraint_rule"
    }
}

pub struct InheritancePatternRuleExecutor;

#[async_trait]
impl CustomRuleExecutor for InheritancePatternRuleExecutor {
    async fn execute(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let forbidden_pattern = context.parameters.get("forbidden_pattern");
        
        if let Some(pattern) = forbidden_pattern {
            // Check if inheritance follows forbidden pattern
            if context.operation.contains(pattern) {
                results.push(
                    ValidationResult::new(
                        "inheritance_pattern_rule",
                        ValidationSeverity::Warning,
                        &format!("Operation follows forbidden pattern '{}'", pattern)
                    )
                );
            }
        }
        
        Ok(results)
    }

    fn rule_type(&self) -> CustomRuleType {
        CustomRuleType::InheritancePattern
    }

    fn rule_id(&self) -> &str {
        "inheritance_pattern_rule"
    }
}

pub struct PerformanceThresholdRuleExecutor;

#[async_trait]
impl CustomRuleExecutor for PerformanceThresholdRuleExecutor {
    async fn execute(&self, context: &ValidationContext) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let max_time_ms = context.parameters.get("max_time_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1000);
        
        // This would measure actual performance in a real implementation
        let simulated_time_ms = 150;
        
        if simulated_time_ms > max_time_ms {
            results.push(
                ValidationResult::new(
                    "performance_threshold_rule",
                    ValidationSeverity::Warning,
                    &format!("Operation took {}ms, exceeding threshold of {}ms", simulated_time_ms, max_time_ms)
                )
            );
        }
        
        Ok(results)
    }

    fn rule_type(&self) -> CustomRuleType {
        CustomRuleType::PerformanceThreshold
    }

    fn rule_id(&self) -> &str {
        "performance_threshold_rule"
    }
}
```

## Success Criteria
- Custom rule engine executes rules correctly
- Built-in executors handle common patterns
- Rule registration and management works

## Next Task
15f_validation_coordinator.md