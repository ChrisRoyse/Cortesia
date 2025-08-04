# Task 05: Test Tantivy Basic Functionality on Windows

**Time: 10 minutes**

## Context
Environment setup complete. Test Tantivy indexing and search with special characters common in code files.

## Objective
Validate Tantivy works on Windows with special characters like `[workspace]`, `Result<T, E>`, `#[derive]`.

## Implementation
Add TantivyValidator to validation.rs with these key methods:

```rust
pub struct TantivyValidator;

impl TantivyValidator {
    pub fn validate_tantivy_windows() -> Result<()> {
        let schema = Self::create_code_schema()?;
        Self::test_special_characters(&schema)?;
        Ok(())
    }
    
    fn create_code_schema() -> Result<Schema> {
        let mut builder = Schema::builder();
        builder.add_text_field("body", TEXT | STORED);
        Ok(builder.build())
    }
    
    fn test_special_characters(schema: &Schema) -> Result<()> {
        let index = Index::create_in_ram(schema.clone());
        // Test docs: "[workspace]", "Result<T, E>", "#[derive]"
        // Verify searches return expected results
        Ok(())
    }
}
```

## Success Criteria
- [ ] TantivyValidator struct added to validation.rs
- [ ] Special characters `[workspace]`, `Result<T, E>`, `#[derive]` are searchable
- [ ] Test passes: `cargo test test_tantivy_validation`
- [ ] Code compiles without errors

## Next Task
Task 06: Test LanceDB basic functionality on Windows