# Micro Phase 1.2: Property Value System

**Estimated Time**: 25 minutes
**Dependencies**: Micro 1.1 (Basic Node Structure)
**Objective**: Create a flexible property value system that supports different data types

## Task Description

Implement the `PropertyValue` enum and related types to handle different kinds of property values in the inheritance system.

## Deliverables

Create `src/properties/value.rs` with:

1. **PropertyValue enum**: Supports strings, numbers, booleans, lists
2. **Value conversion traits**: From/Into implementations
3. **Serialization support**: For storage and network transfer
4. **Comparison operations**: Equality and similarity

## Success Criteria

- [ ] PropertyValue supports at least 5 data types
- [ ] Conversion from common Rust types works
- [ ] Equality comparison is accurate
- [ ] Serialization round-trip preserves values
- [ ] Memory footprint is reasonable (< 64 bytes per value)

## Implementation Requirements

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<PropertyValue>),
    Map(HashMap<String, PropertyValue>),
}

impl PropertyValue {
    pub fn type_name(&self) -> &'static str;
    pub fn as_string(&self) -> Option<&str>;
    pub fn as_integer(&self) -> Option<i64>;
    pub fn as_boolean(&self) -> Option<bool>;
    pub fn similarity(&self, other: &PropertyValue) -> f32;
}

// Required trait implementations
impl From<&str> for PropertyValue;
impl From<String> for PropertyValue;
impl From<i64> for PropertyValue;
impl From<bool> for PropertyValue;
impl ToString for PropertyValue;
```

## Test Requirements

Must pass all conversion tests:
```rust
#[test]
fn test_property_value_conversions() {
    assert_eq!(PropertyValue::from("test"), PropertyValue::String("test".to_string()));
    assert_eq!(PropertyValue::from(42i64), PropertyValue::Integer(42));
    assert_eq!(PropertyValue::from(true), PropertyValue::Boolean(true));
    
    let value = PropertyValue::String("hello".to_string());
    assert_eq!(value.to_string(), "hello");
}

#[test]
fn test_similarity_calculation() {
    let val1 = PropertyValue::String("apple".to_string());
    let val2 = PropertyValue::String("apply".to_string());
    let val3 = PropertyValue::Integer(42);
    
    assert!(val1.similarity(&val2) > 0.7); // Similar strings
    assert!(val1.similarity(&val3) < 0.1); // Different types
}
```

## File Location
`src/properties/value.rs`

## Next Micro Phase
After completion, proceed to Micro 1.3: Hierarchy Tree Structure