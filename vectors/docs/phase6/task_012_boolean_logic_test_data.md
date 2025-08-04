# Task 012: Generate Boolean Logic Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-011. Boolean logic (AND, OR, NOT) requires precise testing to ensure query parsing and execution work correctly.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_boolean_logic_tests()` method that creates test files specifically designed to validate AND, OR, NOT operations with various combinations and precedence rules.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files for testing AND operations
3. Generate files for testing OR operations  
4. Generate files for testing NOT operations
5. Include precedence and complex boolean expressions

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_boolean_logic_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // File for AND operations testing
        let and_content = r#"
pub struct DataProcessor {
    pub active: bool,
    pub ready: bool,
}

impl DataProcessor {
    pub fn process_data(&self) -> Result<(), Error> {
        if self.active && self.ready {
            println!("Processing data...");
            Ok(())
        } else {
            Err(Error::NotReady)
        }
    }
    
    pub fn validate_input(&self, input: &str) -> bool {
        !input.is_empty() && input.len() > 5 && input.contains("valid")
    }
}
"#;
        let mut and_file = self.create_test_file("boolean_and.rs", and_content, TestFileType::BooleanLogic)?;
        and_file.expected_matches = vec![
            "pub".to_string(),
            "struct".to_string(),
            "active".to_string(),
            "ready".to_string(),
        ];
        files.push(and_file);
        
        // File for OR operations testing
        let or_content = r#"
pub enum Status {
    Active,
    Inactive,
    Pending,
}

pub fn check_status(status: &Status) -> String {
    match status {
        Status::Active | Status::Pending => "Available".to_string(),
        Status::Inactive => "Unavailable".to_string(),
    }
}

pub fn is_valid_state(x: i32, y: i32) -> bool {
    x > 0 || y > 0 || (x == 0 && y == 0)
}

pub fn handle_error(error: &Error) {
    match error {
        Error::Network | Error::Timeout => retry_operation(),
        Error::InvalidInput | Error::ParseError => log_error(error),
        _ => panic!("Unexpected error"),
    }
}
"#;
        let mut or_file = self.create_test_file("boolean_or.rs", or_content, TestFileType::BooleanLogic)?;
        or_file.expected_matches = vec![
            "Status::Active".to_string(),
            "Status::Pending".to_string(),
            "Error::Network".to_string(),
            "Error::Timeout".to_string(),
        ];
        files.push(or_file);
        
        // File for NOT operations testing
        let not_content = r#"
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

impl ValidationResult {
    pub fn is_invalid(&self) -> bool {
        !self.is_valid
    }
    
    pub fn has_no_errors(&self) -> bool {
        !self.errors.is_empty()  // Note: This is intentionally wrong for testing
    }
    
    pub fn should_not_process(&self) -> bool {
        !self.is_valid && !self.errors.is_empty()
    }
}

pub fn filter_items(items: &[Item]) -> Vec<Item> {
    items.iter()
        .filter(|item| !item.is_deleted && !item.is_archived)
        .cloned()
        .collect()
}
"#;
        let mut not_file = self.create_test_file("boolean_not.rs", not_content, TestFileType::BooleanLogic)?;
        not_file.expected_matches = vec![
            "!self.is_valid".to_string(),
            "!self.errors.is_empty()".to_string(),
            "!item.is_deleted".to_string(),
            "!item.is_archived".to_string(),
        ];
        files.push(not_file);
        
        // Complex boolean expressions
        let complex_content = r#"
pub fn complex_validation(user: &User, request: &Request) -> bool {
    // Complex boolean logic with precedence
    (user.is_admin || user.has_permission("write")) 
        && !user.is_suspended 
        && (request.is_valid() || request.has_override())
        && !(request.is_expired() && !request.can_extend())
}

pub fn evaluate_conditions(a: bool, b: bool, c: bool, d: bool) -> bool {
    // Test precedence: AND has higher precedence than OR
    a || b && c || d && !a && b
}

pub fn access_control(user: &User, resource: &Resource) -> AccessResult {
    match (user.role, resource.security_level) {
        (Role::Admin, _) => AccessResult::Granted,
        (Role::User, SecurityLevel::Public) => AccessResult::Granted,
        (Role::User, SecurityLevel::Internal) if user.department == resource.owner => AccessResult::Granted,
        (Role::Guest, SecurityLevel::Public) if !resource.requires_auth => AccessResult::Granted,
        _ => AccessResult::Denied,
    }
}
"#;
        let mut complex_file = self.create_test_file("complex_boolean.rs", complex_content, TestFileType::BooleanLogic)?;
        complex_file.expected_matches = vec![
            "user.is_admin".to_string(),
            "user.has_permission".to_string(),
            "!user.is_suspended".to_string(),
            "Role::Admin".to_string(),
        ];
        files.push(complex_file);
        
        Ok(files)
    }
}
```

## Success Criteria
- Method generates 4+ test files covering AND, OR, NOT operations
- Files include realistic Rust code with boolean logic
- Expected matches are specific to boolean operations
- Complex precedence cases are included
- Files test edge cases like double negation

## Time Limit
10 minutes maximum