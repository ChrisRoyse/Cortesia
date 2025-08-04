# Task 12l: Integrate Exceptions with Inheritance

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 12k_get_property_exceptions.md
**Stage**: Inheritance System

## Objective
Integrate exception handling into the property inheritance resolution.

## Implementation
Modify `resolve_inherited_properties` in `property_inheritance_engine.rs` to check exceptions:

```rust
// Add this to the property inheritance engine
impl PropertyInheritanceEngine {
    async fn check_property_exceptions(
        &self,
        concept_id: &str,
        property_name: &str,
        property_value: &PropertyValue,
    ) -> Result<Option<PropertyValue>, Box<dyn std::error::Error>> {
        // Create exception handler
        let exception_handler = crate::inheritance::property_exceptions::PropertyExceptionHandler::new(
            self.connection_manager.clone()
        );
        
        // Get exceptions for this property
        let exceptions = exception_handler.get_property_exceptions(concept_id, property_name).await?;
        
        // Return the highest priority exception value if any exists
        if let Some(exception) = exceptions.first() {
            Ok(Some(exception.exception_value.clone()))
        } else {
            Ok(None)
        }
    }
}
```

Then update the inheritance resolution to use exceptions:

```rust
// In resolve_inherited_properties method, replace the property creation with:
let final_value = match self.check_property_exceptions(&inheritance_chain.child_concept_id, &property.name, &property.value).await? {
    Some(exception_value) => exception_value,
    None => property.value.clone(),
};

inherited_properties.push(InheritedProperty {
    property: PropertyNode {
        name: property.name.clone(),
        value: final_value,
        is_inheritable: property.is_inheritable,
        inheritance_priority: property.inheritance_priority,
    },
    source_concept_id: link.parent_concept_id.clone(),
    inheritance_depth: link.depth_from_child as i32,
    inheritance_strength: link.weight,
    has_exception: exceptions.first().is_some(),
    exception_reason: exceptions.first().map(|e| e.exception_reason.clone()),
});
```

## Success Criteria
- Exceptions are properly applied during inheritance resolution
- Exception values override inherited values

## Next Task
13a_cache_types.md