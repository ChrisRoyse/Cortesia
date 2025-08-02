# Task 03d: Create Exception Node Struct

**Estimated Time**: 7 minutes  
**Dependencies**: 03c_create_property_node_struct.md  
**Next Task**: 03e_create_version_node_struct.md  

## Objective
Create the ExceptionNode data structure for inheritance override handling.

## Single Action
Add ExceptionNode struct and exception types to node_types.rs.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExceptionNode {
    pub id: String,
    pub exception_type: ExceptionType,
    pub target_property: String,
    pub override_value: PropertyValue,
    pub condition: Option<String>,
    pub confidence: f32,
    pub priority: i32,
    pub created_at: DateTime<Utc>,
    pub applied_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_by: String,
    pub justification: String,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExceptionType {
    Override,       // Completely override inherited value
    Modify,         // Modify inherited value with operation
    Block,          // Block inheritance of this property
    Conditional,    // Apply only under certain conditions
    Temporary,      // Temporary override with expiration
}

impl ExceptionNode {
    pub fn new(
        exception_type: ExceptionType,
        target_property: String,
        override_value: PropertyValue,
        created_by: String,
        justification: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            exception_type,
            target_property,
            override_value,
            condition: None,
            confidence: 1.0,
            priority: 0,
            created_at: Utc::now(),
            applied_at: None,
            expires_at: None,
            created_by,
            justification,
            is_active: true,
        }
    }
    
    pub fn with_condition(mut self, condition: String) -> Self {
        self.condition = Some(condition);
        self
    }
    
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self.exception_type = ExceptionType::Temporary;
        self
    }
    
    pub fn apply_exception(&mut self) {
        self.applied_at = Some(Utc::now());
    }
    
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }
    
    pub fn is_applicable(&self) -> bool {
        self.is_active && !self.is_expired()
    }
}

#[cfg(test)]
mod exception_tests {
    use super::*;
    
    #[test]
    fn test_exception_node_creation() {
        let exception = ExceptionNode::new(
            ExceptionType::Override,
            "color".to_string(),
            PropertyValue::String("red".to_string()),
            "system".to_string(),
            "Special case override".to_string(),
        );
        
        assert_eq!(exception.exception_type, ExceptionType::Override);
        assert_eq!(exception.target_property, "color");
        assert!(exception.is_applicable());
        assert!(!exception.is_expired());
    }
    
    #[test]
    fn test_exception_with_expiration() {
        use chrono::Duration;
        
        let future_time = Utc::now() + Duration::hours(1);
        let exception = ExceptionNode::new(
            ExceptionType::Override,
            "temp_prop".to_string(),
            PropertyValue::Boolean(true),
            "user".to_string(),
            "Temporary override".to_string(),
        ).with_expiration(future_time);
        
        assert_eq!(exception.exception_type, ExceptionType::Temporary);
        assert!(exception.expires_at.is_some());
        assert!(!exception.is_expired());
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run exception tests
cargo test exception_tests
```

## Acceptance Criteria
- [ ] ExceptionNode struct compiles without errors
- [ ] All exception types defined
- [ ] Expiration logic works correctly
- [ ] Application tracking functions
- [ ] Tests pass

## Duration
5-7 minutes for exception node implementation and testing.