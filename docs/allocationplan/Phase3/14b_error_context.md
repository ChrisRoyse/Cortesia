# Task 14b: Create Error Context and Recovery

**Time**: 6 minutes
**Dependencies**: 14a_exception_handling_types.md
**Stage**: Inheritance System

## Objective
Add error context and recovery mechanisms.

## Implementation
Add to `src/inheritance/error_types.rs`:

```rust
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub concept_id: Option<String>,
    pub property_name: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trace_id: String,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            concept_id: None,
            property_name: None,
            timestamp: chrono::Utc::now(),
            trace_id: uuid::Uuid::new_v4().to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_concept(mut self, concept_id: &str) -> Self {
        self.concept_id = Some(concept_id.to_string());
        self
    }

    pub fn with_property(mut self, property_name: &str) -> Self {
        self.property_name = Some(property_name.to_string());
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

#[derive(Debug)]
pub struct InheritanceErrorWithContext {
    pub error: InheritanceError,
    pub context: ErrorContext,
    pub recovery_suggestion: Option<String>,
}

impl InheritanceErrorWithContext {
    pub fn new(error: InheritanceError, context: ErrorContext) -> Self {
        let recovery_suggestion = Self::generate_recovery_suggestion(&error);
        Self {
            error,
            context,
            recovery_suggestion,
        }
    }

    fn generate_recovery_suggestion(error: &InheritanceError) -> Option<String> {
        match error {
            InheritanceError::CycleDetected { .. } => 
                Some("Remove one of the inheritance relationships to break the cycle".to_string()),
            InheritanceError::MaxDepthExceeded(_) => 
                Some("Reduce inheritance chain depth or increase the maximum depth limit".to_string()),
            InheritanceError::ConceptNotFound(_) => 
                Some("Verify the concept ID exists or create the missing concept".to_string()),
            InheritanceError::CacheError(_) => 
                Some("Clear cache and retry operation".to_string()),
            _ => None,
        }
    }
}

impl fmt::Display for InheritanceErrorWithContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (Operation: {}, Trace: {})", 
               self.error, self.context.operation, self.context.trace_id)?;
        
        if let Some(suggestion) = &self.recovery_suggestion {
            write!(f, "\nSuggestion: {}", suggestion)?;
        }
        
        Ok(())
    }
}

impl Error for InheritanceErrorWithContext {}
```

## Success Criteria
- Error context captures operation details
- Recovery suggestions are helpful

## Next Task
14c_error_handler_trait.md