# Task 03: Fact Content Structure

## Metadata
- **Micro-Phase**: 2.3
- **Duration**: 15 minutes
- **Dependencies**: None
- **Output**: `src/quality_integration/fact_content.rs`

## Description
Create the FactContent structure that holds the raw content and entities from Phase 0A parsing. This represents the actual information to be allocated.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fact_content_creation() {
        let fact = FactContent::new("Elephants have trunks");
        assert_eq!(fact.text, "Elephants have trunks");
        assert!(fact.entities.is_empty());
        assert!(fact.metadata.created_at > 0);
    }
    
    #[test]
    fn test_fact_with_entities() {
        let mut fact = FactContent::new("The African elephant is large");
        fact.add_entity("African elephant".to_string());
        fact.add_entity("large".to_string());
        assert_eq!(fact.entities.len(), 2);
        assert!(fact.entities.contains(&"African elephant".to_string()));
    }
    
    #[test]
    fn test_fact_length_validation() {
        let fact = FactContent::new("Short fact");
        assert!(fact.is_valid_length());
        
        let long_text = "a".repeat(10000);
        let long_fact = FactContent::new(&long_text);
        assert!(!long_fact.is_valid_length());
    }
    
    #[test]
    fn test_fact_hash() {
        let fact1 = FactContent::new("Test fact");
        let fact2 = FactContent::new("Test fact");
        assert_eq!(fact1.content_hash(), fact2.content_hash());
        
        let fact3 = FactContent::new("Different fact");
        assert_ne!(fact1.content_hash(), fact3.content_hash());
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Metadata about fact creation and processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactMetadata {
    pub created_at: u64,
    pub source: Option<String>,
    pub confidence_source: Option<String>,
}

/// Raw fact content from Phase 0A parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactContent {
    /// The raw text of the fact
    pub text: String,
    
    /// Entities identified in the fact
    pub entities: HashSet<String>,
    
    /// Metadata about the fact
    pub metadata: FactMetadata,
}

impl FactContent {
    /// Create a new fact with content
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            entities: HashSet::new(),
            metadata: FactMetadata {
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                source: None,
                confidence_source: None,
            },
        }
    }
    
    /// Add an entity to this fact
    pub fn add_entity(&mut self, entity: String) {
        self.entities.insert(entity);
    }
    
    /// Check if fact length is within valid bounds
    pub fn is_valid_length(&self) -> bool {
        !self.text.is_empty() && self.text.len() < 5000
    }
    
    /// Get a hash of the content for deduplication
    pub fn content_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.text.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Check if this fact contains an entity
    pub fn contains_entity(&self, entity: &str) -> bool {
        self.entities.contains(entity)
    }
    
    /// Get the number of entities
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl PartialEq for FactContent {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.entities == other.entities
    }
}
```

## Verification Steps
1. Create FactContent structure with text and entities
2. Implement metadata tracking
3. Add entity management methods
4. Implement content hashing for deduplication
5. Ensure all tests pass

## Success Criteria
- [ ] FactContent struct compiles
- [ ] Entity storage works correctly
- [ ] Content hashing functional
- [ ] Length validation works
- [ ] All tests pass