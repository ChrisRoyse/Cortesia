# Task 011: Generate Special Characters Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 010. Special characters are critical for accurate search results in code files and need comprehensive testing.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_special_characters_tests()` method that creates test files containing various special characters commonly found in code and configuration files.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate test files with Rust-specific special characters
3. Include configuration file special characters
4. Add edge cases like nested brackets and operators
5. Create files with known expected matches

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_special_characters_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Rust workspace configuration
        let workspace_content = r#"
[workspace]
members = ["backend", "frontend", "shared"]
resolver = "2"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }

# This is a comment with special chars: @#$%^&*()
"#;
        let mut workspace_file = self.create_test_file("workspace_test.toml", workspace_content, TestFileType::SpecialCharacters)?;
        workspace_file.expected_matches = vec![
            "[workspace]".to_string(),
            "[workspace.dependencies]".to_string(),
            "serde = {".to_string(),
        ];
        files.push(workspace_file);
        
        // Rust type signatures and generics
        let rust_types_content = r#"
pub fn process<T, E>() -> Result<T, E> where T: Clone {
    Ok(default_value())
}

pub struct Data<'a> {
    reference: &'a str,
    optional: Option<Box<dyn Send + Sync>>,
}

impl<T> Display for Data<T> where T: Debug {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// Test various brackets and operators
let array = [1, 2, 3];
let tuple = (x, y, z);
let closure = |x: i32| -> i32 { x * 2 };
"#;
        let mut rust_types_file = self.create_test_file("rust_types.rs", rust_types_content, TestFileType::SpecialCharacters)?;
        rust_types_file.expected_matches = vec![
            "Result<T, E>".to_string(),
            "&'a str".to_string(),
            "Box<dyn Send + Sync>".to_string(),
            "Formatter<'_>".to_string(),
        ];
        files.push(rust_types_file);
        
        // Derive macros and attributes
        let derive_content = r#"
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_serialization() {
        // Test content
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    Ok(())
}
"#;
        let mut derive_file = self.create_test_file("derive_macros.rs", derive_content, TestFileType::SpecialCharacters)?;
        derive_file.expected_matches = vec![
            "#[derive(Debug".to_string(),
            "#[serde(".to_string(),
            "#[cfg(test)]".to_string(),
            "#[tokio::main]".to_string(),
        ];
        files.push(derive_file);
        
        // Complex operators and symbols
        let operators_content = r#"
fn complex_operations() {
    let result = x && y || z;
    let pointer = &mut variable;
    let range = 0..=100;
    let match_expr = match value {
        Some(x) if x > 0 => x,
        None => 0,
        _ => -1,
    };
    
    // Bitwise operations
    let bits = a & b | c ^ d << 2 >> 1;
    
    // Path operations
    let path = std::path::Path::new("./test");
    let url = "https://example.com/api?param=value&other=123";
}
"#;
        let mut operators_file = self.create_test_file("operators.rs", operators_content, TestFileType::SpecialCharacters)?;
        operators_file.expected_matches = vec![
            "&mut".to_string(),
            "0..=100".to_string(),
            "std::path::Path".to_string(),
            "param=value&other".to_string(),
        ];
        files.push(operators_file);
        
        // JSON and configuration syntax
        let json_content = r#"
{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "@types/node": "^18.0.0",
        "typescript": "~4.8.0"
    },
    "scripts": {
        "build": "tsc && node dist/index.js",
        "test": "jest --coverage"
    },
    "config": {
        "api_url": "https://api.example.com/v1",
        "timeout_ms": 5000
    }
}
"#;
        let mut json_file = self.create_test_file("package.json", json_content, TestFileType::SpecialCharacters)?;
        json_file.expected_matches = vec![
            "\"@types/node\"".to_string(),
            "\"~4.8.0\"".to_string(),
            "\"https://api.example.com/v1\"".to_string(),
        ];
        files.push(json_file);
        
        Ok(files)
    }
}
```

## Success Criteria
- Method generates 5+ test files with diverse special characters
- Each file includes expected_matches for validation
- Files cover Rust syntax, configuration files, and operators
- Content is realistic and represents actual use cases
- Files are created successfully in the test directory

## Time Limit
10 minutes maximum