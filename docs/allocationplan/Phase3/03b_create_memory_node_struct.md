# Task 03b: Create Memory Node Struct

**Estimated Time**: 8 minutes  
**Dependencies**: 03a_create_concept_node_struct.md  
**Next Task**: 03c_create_property_node_struct.md  

## Objective
Create the MemoryNode data structure for temporal memory storage.

## Single Action
Add MemoryNode struct and related enums to node_types.rs.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub context: Option<String>,
    pub memory_type: MemoryType,
    pub strength: f32,
    pub decay_rate: f32,
    pub consolidation_level: ConsolidationLevel,
    pub neural_pattern: Option<String>,
    pub retrieval_count: i32,
    pub created_at: DateTime<Utc>,
    pub last_strengthened: DateTime<Utc>,
    pub associated_emotions: Vec<String>,
    pub sensory_modalities: Vec<SensoryModality>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    Working,
    Sensory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsolidationLevel {
    Fresh,
    Consolidating,
    Consolidated,
    Archived,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Tactile,
    Olfactory,
    Gustatory,
    Proprioceptive,
}

impl MemoryNode {
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            content,
            context: None,
            memory_type,
            strength: 1.0,
            decay_rate: 0.1,
            consolidation_level: ConsolidationLevel::Fresh,
            neural_pattern: None,
            retrieval_count: 0,
            created_at: now,
            last_strengthened: now,
            associated_emotions: Vec::new(),
            sensory_modalities: Vec::new(),
        }
    }
    
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
    
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }
    
    pub fn strengthen(&mut self) {
        self.strength = (self.strength + 0.1).min(1.0);
        self.retrieval_count += 1;
        self.last_strengthened = Utc::now();
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    
    #[test]
    fn test_memory_node_creation() {
        let memory = MemoryNode::new(
            "Test memory content".to_string(),
            MemoryType::Episodic
        );
        
        assert_eq!(memory.content, "Test memory content");
        assert_eq!(memory.memory_type, MemoryType::Episodic);
        assert_eq!(memory.strength, 1.0);
        assert_eq!(memory.consolidation_level, ConsolidationLevel::Fresh);
        assert!(!memory.id.is_empty());
    }
    
    #[test]
    fn test_memory_strengthening() {
        let mut memory = MemoryNode::new(
            "Test".to_string(),
            MemoryType::Semantic
        ).with_strength(0.5);
        
        let initial_strength = memory.strength;
        memory.strengthen();
        
        assert!(memory.strength > initial_strength);
        assert_eq!(memory.retrieval_count, 1);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run memory tests
cargo test memory_tests
```

## Acceptance Criteria
- [ ] MemoryNode struct compiles without errors
- [ ] All memory type enums defined
- [ ] Constructor and builder methods work
- [ ] Strengthening mechanism functions
- [ ] Tests pass

## Duration
6-8 minutes for memory node implementation and testing.