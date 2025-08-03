# Task 12h: Create Property Exception Types

**Time**: 4 minutes
**Dependencies**: 12g_merge_properties.md
**Stage**: Inheritance System

## Objective
Create data structures for property exceptions.

## Implementation
Add to `src/inheritance/property_types.rs`:

```rust
#[derive(Debug, Clone)]
pub struct ExceptionNode {
    pub id: String,
    pub property_name: String,
    pub original_value: PropertyValue,
    pub exception_value: PropertyValue,
    pub exception_reason: String,
    pub confidence: f32,
    pub precedence: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug)]
pub struct ExceptionNodeBuilder {
    property_name: Option<String>,
    original_value: Option<PropertyValue>,
    exception_value: Option<PropertyValue>,
    exception_reason: Option<String>,
    confidence: Option<f32>,
}

impl ExceptionNodeBuilder {
    pub fn new() -> Self {
        Self {
            property_name: None,
            original_value: None,
            exception_value: None,
            exception_reason: None,
            confidence: None,
        }
    }

    pub fn property_name(mut self, name: &str) -> Self {
        self.property_name = Some(name.to_string());
        self
    }

    pub fn original_value(mut self, value: PropertyValue) -> Self {
        self.original_value = Some(value);
        self
    }

    pub fn exception_value(mut self, value: PropertyValue) -> Self {
        self.exception_value = Some(value);
        self
    }

    pub fn exception_reason(mut self, reason: &str) -> Self {
        self.exception_reason = Some(reason.to_string());
        self
    }

    pub fn confidence(mut self, conf: f32) -> Self {
        self.confidence = Some(conf);
        self
    }

    pub fn build(self) -> Result<ExceptionNode, Box<dyn std::error::Error>> {
        Ok(ExceptionNode {
            id: uuid::Uuid::new_v4().to_string(),
            property_name: self.property_name.ok_or("Property name required")?,
            original_value: self.original_value.ok_or("Original value required")?,
            exception_value: self.exception_value.ok_or("Exception value required")?,
            exception_reason: self.exception_reason.ok_or("Exception reason required")?,
            confidence: self.confidence.unwrap_or(1.0),
            precedence: 0,
            created_at: chrono::Utc::now(),
        })
    }
}
```

## Success Criteria
- Exception types compile without errors
- Builder pattern works correctly

## Next Task
12i_property_exception_handler.md