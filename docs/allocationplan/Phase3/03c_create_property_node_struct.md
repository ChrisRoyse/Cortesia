# Task 03c: Create Property Node Struct

**Estimated Time**: 10 minutes  
**Dependencies**: 03b_create_memory_node_struct.md  
**Next Task**: 03d_create_exception_node_struct.md  

## Objective
Create the PropertyNode data structure for inheritance system properties.

## Single Action
Add PropertyNode struct and property value types to node_types.rs.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyNode {
    pub id: String,
    pub name: String,
    pub value: PropertyValue,
    pub data_type: DataType,
    pub property_type: String,
    pub is_inheritable: bool,
    pub inheritance_priority: i32,
    pub property_source: PropertySource,
    pub visibility: PropertyVisibility,
    pub validation_rules: Vec<ValidationRule>,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub version: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<PropertyValue>),
    Object(std::collections::HashMap<String, PropertyValue>),
    Null,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Text,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    DateTime,
    Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertySource {
    Direct,
    Inherited,
    Computed,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyVisibility {
    Public,
    Protected,
    Private,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationRule {
    pub rule_type: String,
    pub parameters: std::collections::HashMap<String, String>,
    pub error_message: String,
}

impl PropertyNode {
    pub fn new(name: String, value: PropertyValue, data_type: DataType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            value,
            data_type,
            property_type: "general".to_string(),
            is_inheritable: true,
            inheritance_priority: 0,
            property_source: PropertySource::Direct,
            visibility: PropertyVisibility::Public,
            validation_rules: Vec::new(),
            created_at: now,
            modified_at: now,
            version: 1,
        }
    }
    
    pub fn with_inheritance_priority(mut self, priority: i32) -> Self {
        self.inheritance_priority = priority;
        self
    }
    
    pub fn set_inheritable(mut self, inheritable: bool) -> Self {
        self.is_inheritable = inheritable;
        self
    }
    
    pub fn update_value(&mut self, new_value: PropertyValue) {
        self.value = new_value;
        self.modified_at = Utc::now();
        self.version += 1;
    }
    
    pub fn is_valid(&self) -> bool {
        // Basic validation - can be extended
        !self.name.is_empty() && self.inheritance_priority >= 0
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    
    #[test]
    fn test_property_node_creation() {
        let property = PropertyNode::new(
            "test_property".to_string(),
            PropertyValue::String("test_value".to_string()),
            DataType::Text,
        );
        
        assert_eq!(property.name, "test_property");
        assert_eq!(property.value, PropertyValue::String("test_value".to_string()));
        assert_eq!(property.data_type, DataType::Text);
        assert!(property.is_inheritable);
        assert!(property.is_valid());
    }
    
    #[test]
    fn test_property_value_update() {
        let mut property = PropertyNode::new(
            "counter".to_string(),
            PropertyValue::Number(1.0),
            DataType::Integer,
        );
        
        let original_version = property.version;
        property.update_value(PropertyValue::Number(2.0));
        
        assert_eq!(property.value, PropertyValue::Number(2.0));
        assert_eq!(property.version, original_version + 1);
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run property tests
cargo test property_tests
```

## Acceptance Criteria
- [ ] PropertyNode struct compiles without errors
- [ ] PropertyValue enum supports all data types
- [ ] Builder pattern methods work
- [ ] Value update mechanism functions
- [ ] Tests pass

## Duration
8-10 minutes for property node implementation and testing.