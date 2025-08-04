# AI Prompt: Micro Phase 1.2 - Property Value System

You are tasked with implementing a flexible property value system that supports different data types for the inheritance hierarchy. Your goal is to create `src/properties/value.rs` with a comprehensive PropertyValue enum and conversion system.

## Your Task
Implement the `PropertyValue` enum and related types to handle different kinds of property values in the inheritance system, including type conversions, comparisons, and similarity calculations.

## Specific Requirements
1. Create `src/properties/value.rs` with PropertyValue enum supporting at least 5 data types
2. Implement value conversion traits (From/Into) for common Rust types
3. Add serialization support for storage and network transfer
4. Implement comparison operations including equality and similarity
5. Ensure memory-efficient representation (< 64 bytes per value)
6. Add type introspection methods

## Expected Code Structure
You must implement these exact signatures:

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
    Map(HashMap<String, PropertyValue>),
}

impl PropertyValue {
    pub fn type_name(&self) -> &'static str {
        // Return string representation of the type
    }
    
    pub fn as_string(&self) -> Option<&str> {
        // Extract string value if this is a String variant
    }
    
    pub fn as_integer(&self) -> Option<i64> {
        // Extract integer value if this is an Integer variant
    }
    
    pub fn as_float(&self) -> Option<f64> {
        // Extract float value if this is a Float variant
    }
    
    pub fn as_boolean(&self) -> Option<bool> {
        // Extract boolean value if this is a Boolean variant
    }
    
    pub fn as_list(&self) -> Option<&Vec<PropertyValue>> {
        // Extract list value if this is a List variant
    }
    
    pub fn as_map(&self) -> Option<&HashMap<String, PropertyValue>> {
        // Extract map value if this is a Map variant
    }
    
    pub fn similarity(&self, other: &PropertyValue) -> f32 {
        // Calculate similarity score between 0.0 and 1.0
        // Same type and value = 1.0
        // Different types = 0.0
        // Similar values of same type = between 0.0 and 1.0
    }
    
    pub fn size_estimate(&self) -> usize {
        // Estimate memory footprint in bytes
    }
}

// Required trait implementations
impl From<&str> for PropertyValue {
    fn from(value: &str) -> Self {
        PropertyValue::String(value.to_string())
    }
}

impl From<String> for PropertyValue {
    fn from(value: String) -> Self {
        PropertyValue::String(value)
    }
}

impl From<i64> for PropertyValue {
    fn from(value: i64) -> Self {
        PropertyValue::Integer(value)
    }
}

impl From<f64> for PropertyValue {
    fn from(value: f64) -> Self {
        PropertyValue::Float(value)
    }
}

impl From<bool> for PropertyValue {
    fn from(value: bool) -> Self {
        PropertyValue::Boolean(value)
    }
}

impl From<Vec<PropertyValue>> for PropertyValue {
    fn from(value: Vec<PropertyValue>) -> Self {
        PropertyValue::List(value)
    }
}

impl From<HashMap<String, PropertyValue>> for PropertyValue {
    fn from(value: HashMap<String, PropertyValue>) -> Self {
        PropertyValue::Map(value)
    }
}

impl ToString for PropertyValue {
    fn to_string(&self) -> String {
        // Convert any PropertyValue to its string representation
    }
}

impl std::fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Implement display formatting
    }
}
```

## Success Criteria (You must verify these)
- [ ] PropertyValue supports exactly 6 data types (String, Integer, Float, Boolean, List, Map)
- [ ] Conversion from common Rust types works correctly
- [ ] Equality comparison is accurate for all types
- [ ] Serialization round-trip preserves all values exactly
- [ ] Memory footprint estimate is reasonable (< 64 bytes for simple values)
- [ ] Similarity calculation provides meaningful scores
- [ ] Type introspection methods return correct values
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_value_conversions() {
        assert_eq!(PropertyValue::from("test"), PropertyValue::String("test".to_string()));
        assert_eq!(PropertyValue::from(42i64), PropertyValue::Integer(42));
        assert_eq!(PropertyValue::from(3.14f64), PropertyValue::Float(3.14));
        assert_eq!(PropertyValue::from(true), PropertyValue::Boolean(true));
        
        let value = PropertyValue::String("hello".to_string());
        assert_eq!(value.to_string(), "hello");
    }

    #[test]
    fn test_type_introspection() {
        assert_eq!(PropertyValue::String("test".to_string()).type_name(), "String");
        assert_eq!(PropertyValue::Integer(42).type_name(), "Integer");
        assert_eq!(PropertyValue::Boolean(true).type_name(), "Boolean");
        
        let string_val = PropertyValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));
        assert_eq!(string_val.as_integer(), None);
    }

    #[test]
    fn test_similarity_calculation() {
        let val1 = PropertyValue::String("apple".to_string());
        let val2 = PropertyValue::String("apply".to_string());
        let val3 = PropertyValue::String("apple".to_string());
        let val4 = PropertyValue::Integer(42);
        
        assert_eq!(val1.similarity(&val3), 1.0); // Identical values
        assert!(val1.similarity(&val2) > 0.7); // Similar strings
        assert!(val1.similarity(&val4) < 0.1); // Different types
    }

    #[test]
    fn test_complex_types() {
        let list_val = PropertyValue::from(vec![
            PropertyValue::from(1i64),
            PropertyValue::from("test"),
            PropertyValue::from(true),
        ]);
        
        let mut map = HashMap::new();
        map.insert("key1".to_string(), PropertyValue::from("value1"));
        map.insert("key2".to_string(), PropertyValue::from(42i64));
        let map_val = PropertyValue::from(map);
        
        assert_eq!(list_val.type_name(), "List");
        assert_eq!(map_val.type_name(), "Map");
        
        if let Some(list) = list_val.as_list() {
            assert_eq!(list.len(), 3);
        }
    }

    #[test]
    fn test_serialization() {
        let original = PropertyValue::Map({
            let mut map = HashMap::new();
            map.insert("string".to_string(), PropertyValue::from("test"));
            map.insert("number".to_string(), PropertyValue::from(42i64));
            map.insert("bool".to_string(), PropertyValue::from(true));
            map
        });
        
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: PropertyValue = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_memory_efficiency() {
        let small_string = PropertyValue::from("short");
        let small_int = PropertyValue::from(42i64);
        let small_bool = PropertyValue::from(true);
        
        assert!(small_string.size_estimate() < 64);
        assert!(small_int.size_estimate() < 64);
        assert!(small_bool.size_estimate() < 64);
    }

    #[test]
    fn test_display_formatting() {
        let values = vec![
            PropertyValue::from("test string"),
            PropertyValue::from(42i64),
            PropertyValue::from(3.14f64),
            PropertyValue::from(true),
        ];
        
        for value in values {
            let display_str = format!("{}", value);
            assert!(!display_str.is_empty());
        }
    }
}
```

## File to Create
Create exactly this file: `src/properties/value.rs`

## Dependencies Required
You may need to add dependencies to Cargo.toml:
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## String Similarity Implementation Note
For the similarity calculation between strings, you can implement a simple character-based similarity:
- Identical strings: 1.0
- Same length with different characters: proportion of matching characters
- Different lengths: use Levenshtein distance or similar algorithm
- Different types: 0.0

## When Complete
Respond with "MICRO PHASE 1.2 COMPLETE" and a brief summary of what you implemented, including:
- Which similarity algorithm you used for strings
- Memory efficiency strategies employed
- Any design decisions made for complex types
- Confirmation that all tests pass