# Task 15b: Implement Structural Validator

**Time**: 7 minutes
**Dependencies**: 15a_validation_rules.md
**Stage**: Inheritance System

## Objective
Create validator for structural integrity of inheritance hierarchies.

## Implementation
Create `src/inheritance/validation/structural_validator.rs`:

```rust
use std::sync::Arc;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::validation::rules::*;
use crate::inheritance::hierarchy_types::*;

pub struct StructuralValidator {
    connection_manager: Arc<Neo4jConnectionManager>,
    rules: StructuralRules,
}

impl StructuralValidator {
    pub fn new(connection_manager: Arc<Neo4jConnectionManager>, rules: StructuralRules) -> Self {
        Self {
            connection_manager,
            rules,
        }
    }

    pub async fn validate_inheritance_depth(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let depth = self.get_inheritance_depth(concept_id).await?;
        
        if depth > self.rules.max_inheritance_depth {
            results.push(
                ValidationResult::new(
                    "max_depth_exceeded",
                    ValidationSeverity::Error,
                    &format!("Inheritance depth {} exceeds maximum allowed depth {}", depth, self.rules.max_inheritance_depth)
                )
                .with_concept(concept_id)
                .with_suggestion(&format!("Reduce inheritance chain depth to {} or less", self.rules.max_inheritance_depth))
            );
        } else if depth > (self.rules.max_inheritance_depth * 3 / 4) {
            results.push(
                ValidationResult::new(
                    "depth_warning",
                    ValidationSeverity::Warning,
                    &format!("Inheritance depth {} is approaching maximum limit", depth)
                )
                .with_concept(concept_id)
            );
        }
        
        Ok(results)
    }

    pub async fn validate_circular_references(&self) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if !self.rules.allow_circular_references {
            let cycles = self.detect_cycles().await?;
            
            for cycle in cycles {
                results.push(
                    ValidationResult::new(
                        "circular_reference",
                        ValidationSeverity::Critical,
                        &format!("Circular inheritance detected involving concepts: {}", cycle.join(" -> "))
                    )
                    .with_suggestion("Remove one inheritance relationship to break the cycle")
                );
            }
        }
        
        Ok(results)
    }

    pub async fn validate_multiple_inheritance(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let parent_count = self.get_direct_parent_count(concept_id).await?;
        
        if parent_count > 1 && !self.rules.allow_multiple_inheritance {
            results.push(
                ValidationResult::new(
                    "multiple_inheritance_not_allowed",
                    ValidationSeverity::Error,
                    &format!("Concept has {} parents but multiple inheritance is not allowed", parent_count)
                )
                .with_concept(concept_id)
                .with_suggestion("Remove all but one parent relationship")
            );
        }
        
        Ok(results)
    }

    pub async fn validate_children_count(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if let Some(max_children) = self.rules.max_children_per_concept {
            let children_count = self.get_direct_children_count(concept_id).await?;
            
            if children_count > max_children {
                results.push(
                    ValidationResult::new(
                        "too_many_children",
                        ValidationSeverity::Warning,
                        &format!("Concept has {} children, exceeding recommended maximum of {}", children_count, max_children)
                    )
                    .with_concept(concept_id)
                    .with_suggestion("Consider restructuring the hierarchy to reduce fan-out")
                );
            }
        }
        
        Ok(results)
    }

    async fn get_inheritance_depth(&self, concept_id: &str) -> Result<u32, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (child:Concept {id: $concept_id})-[:INHERITS_FROM*]->(ancestor:Concept)
            RETURN max(length(path)) as max_depth
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            Ok(record.get("max_depth").unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    async fn detect_cycles(&self) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (c:Concept)-[:INHERITS_FROM*]->(c)
            RETURN [node in nodes(path) | node.id] as cycle_path
        "#;
        
        let result = session.run(query, None).await?;
        
        let mut cycles = Vec::new();
        for record in result {
            let cycle_path: Vec<String> = record.get("cycle_path")?;
            cycles.push(cycle_path);
        }
        
        Ok(cycles)
    }

    async fn get_direct_parent_count(&self, concept_id: &str) -> Result<u32, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (child:Concept {id: $concept_id})-[:INHERITS_FROM]->(parent:Concept)
            RETURN count(parent) as parent_count
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            Ok(record.get("parent_count").unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    async fn get_direct_children_count(&self, concept_id: &str) -> Result<u32, Box<dyn std::error::Error>> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (parent:Concept {id: $concept_id})<-[:INHERITS_FROM]-(child:Concept)
            RETURN count(child) as children_count
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        let result = session.run(query, Some(parameters)).await?;
        
        if let Some(record) = result.next().await? {
            Ok(record.get("children_count").unwrap_or(0))
        } else {
            Ok(0)
        }
    }
}
```

## Success Criteria
- Validates inheritance depth correctly
- Detects circular references
- Checks multiple inheritance constraints

## Next Task
15c_semantic_validator.md