# Task 15c: Implement Semantic Validator

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 15b_structural_validator.md
**Stage**: Inheritance System

## Objective
Create validator for semantic correctness of inheritance relationships.

## Implementation
Create `src/inheritance/validation/semantic_validator.rs`:

```rust
use std::sync::Arc;
use std::collections::HashSet;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::validation::rules::*;
use crate::inheritance::property_types::*;

pub struct SemanticValidator {
    connection_manager: Arc<Neo4jConnectionManager>,
    rules: SemanticRules,
}

impl SemanticValidator {
    pub fn new(connection_manager: Arc<Neo4jConnectionManager>, rules: SemanticRules) -> Self {
        Self {
            connection_manager,
            rules,
        }
    }

    pub async fn validate_concept_existence(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.require_concept_existence {
            let exists = self.concept_exists(concept_id).await?;
            
            if !exists {
                results.push(
                    ValidationResult::new(
                        "concept_not_found",
                        ValidationSeverity::Critical,
                        &format!("Referenced concept '{}' does not exist", concept_id)
                    )
                    .with_concept(concept_id)
                    .with_suggestion("Create the missing concept or remove the reference")
                );
            }
        }
        
        Ok(results)
    }

    pub async fn validate_property_types(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.validate_property_types {
            let properties = self.get_concept_properties(concept_id).await?;
            
            for property in properties {
                if let Err(validation_error) = self.validate_property_value(&property) {
                    results.push(
                        ValidationResult::new(
                            "invalid_property_type",
                            ValidationSeverity::Error,
                            &validation_error
                        )
                        .with_concept(concept_id)
                        .with_property(&property.name)
                        .with_suggestion("Correct the property value type")
                    );
                }
            }
        }
        
        Ok(results)
    }

    pub async fn validate_inheritance_compatibility(&self, child_id: &str, parent_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.enforce_inheritance_compatibility {
            // Check property conflicts
            let conflicts = self.find_property_conflicts(child_id, parent_id).await?;
            
            for conflict in conflicts {
                results.push(
                    ValidationResult::new(
                        "property_inheritance_conflict",
                        ValidationSeverity::Warning,
                        &format!("Property '{}' has conflicting definitions between child and parent", conflict)
                    )
                    .with_concept(child_id)
                    .with_property(&conflict)
                    .with_suggestion("Consider using property exceptions to resolve conflicts")
                );
            }
        }
        
        Ok(results)
    }

    pub async fn validate_naming_conventions(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.check_naming_conventions {
            // Validate concept name format
            if !self.is_valid_concept_name(concept_id) {
                results.push(
                    ValidationResult::new(
                        "invalid_concept_name",
                        ValidationSeverity::Warning,
                        &format!("Concept name '{}' does not follow naming conventions", concept_id)
                    )
                    .with_concept(concept_id)
                    .with_suggestion("Use lowercase letters, numbers, and underscores only")
                );
            }
            
            // Validate property names
            let properties = self.get_concept_properties(concept_id).await?;
            for property in properties {
                if !self.is_valid_property_name(&property.name) {
                    results.push(
                        ValidationResult::new(
                            "invalid_property_name",
                            ValidationSeverity::Warning,
                            &format!("Property name '{}' does not follow naming conventions", property.name)
                        )
                        .with_concept(concept_id)
                        .with_property(&property.name)
                        .with_suggestion("Use camelCase for property names")
                    );
                }
            }
        }
        
        Ok(results)
    }

    pub async fn validate_relationship_semantics(&self, child_id: &str, parent_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.validate_relationship_semantics {
            // Check if the inheritance makes semantic sense
            if self.are_semantically_incompatible(child_id, parent_id).await? {
                results.push(
                    ValidationResult::new(
                        "semantically_invalid_inheritance",
                        ValidationSeverity::Warning,
                        &format!("Inheritance from '{}' to '{}' may not be semantically valid", parent_id, child_id)
                    )
                    .with_concept(child_id)
                    .with_suggestion("Review if this inheritance relationship makes logical sense")
                );
            }
        }
        
        Ok(results)
    }

    async fn concept_exists(&self, concept_id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $concept_id})
            RETURN count(c) > 0 as exists
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            Ok(record.get("exists").unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    async fn get_concept_properties(&self, concept_id: &str) -> Result<Vec<PropertyNode>, Box<dyn std::error::Error>> {
        // This would typically fetch properties from the database
        // For now, return empty vector
        Ok(Vec::new())
    }

    fn validate_property_value(&self, property: &PropertyNode) -> Result<(), String> {
        match &property.value {
            PropertyValue::Text(text) => {
                if text.is_empty() {
                    Err("Text property cannot be empty".to_string())
                } else {
                    Ok(())
                }
            }
            PropertyValue::Number(num) => {
                if num.is_finite() {
                    Ok(())
                } else {
                    Err("Number property must be finite".to_string())
                }
            }
            PropertyValue::Boolean(_) => Ok(()),
            PropertyValue::List(list) => {
                if list.len() > 1000 {
                    Err("List property is too large".to_string())
                } else {
                    Ok(())
                }
            }
        }
    }

    async fn find_property_conflicts(&self, child_id: &str, parent_id: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // This would check for property name conflicts
        // For now, return empty vector
        Ok(Vec::new())
    }

    fn is_valid_concept_name(&self, name: &str) -> bool {
        // Simple validation: lowercase, numbers, underscores
        name.chars().all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
    }

    fn is_valid_property_name(&self, name: &str) -> bool {
        // Simple validation: camelCase
        !name.is_empty() && name.chars().next().unwrap().is_lowercase()
    }

    async fn are_semantically_incompatible(&self, _child_id: &str, _parent_id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        // This would check semantic compatibility using concept types, categories, etc.
        // For now, always return false (compatible)
        Ok(false)
    }
}
```

## Success Criteria
- Validates concept existence properly
- Checks property type correctness
- Enforces naming conventions

## Next Task
15d_performance_validator.md