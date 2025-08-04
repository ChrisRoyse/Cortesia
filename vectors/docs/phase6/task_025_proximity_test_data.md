# Task 025: Generate Proximity Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-012. Proximity search allows finding terms within a specific distance of each other, crucial for code analysis and contextual search.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_proximity_tests()` method that creates test files for validating proximity search operators (NEAR/WITHIN) with various word distances and cross-boundary scenarios.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files testing proximity at different word distances (2, 5, 10, 20 words)
3. Include cross-line and cross-paragraph proximity scenarios
4. Add negative proximity tests (words too far apart)
5. Create code-specific proximity tests (function definitions, variable usage)
6. Include boundary condition tests (exact distance matches)
7. Generate performance stress tests with complex proximity patterns

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_proximity_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Near distance test - words within 3 positions
        let near_content = r#"
pub fn process_data() -> Result<ProcessedData, ProcessError> {
    let input = get_input_data();
    let processed = transform_data(input);
    validate_processed_data(&processed)
}

fn calculate_metrics(data: &ProcessedData) -> MetricsResult {
    let start_time = Instant::now();
    let result = compute_statistics(data);
    let end_time = Instant::now();
    
    MetricsResult {
        duration: end_time - start_time,
        processed_items: result.count,
        error_rate: result.errors / result.total
    }
}
"#;
        let mut near_file = self.create_test_file("proximity_near.rs", near_content, TestFileType::Proximity)?;
        near_file.expected_matches = vec![
            "process_data() -> Result<ProcessedData".to_string(),  // Should match "process NEAR/3 data"
            "get_input_data()".to_string(),    // Should match "input NEAR/3 data"
            "transform_data(input)".to_string(), // Should match "transform NEAR/3 data"
            "start_time = Instant".to_string(),  // Should match "start NEAR/3 time"
        ];
        files.push(near_file);
        
        // Within distance test - broader proximity
        let within_content = r#"
/// Configuration management system
/// Handles loading and validation of configuration files
/// Supports multiple formats including TOML, JSON, and YAML
pub struct ConfigurationManager {
    config_path: PathBuf,
    cache: HashMap<String, ConfigValue>,
    validation_rules: Vec<ValidationRule>,
}

impl ConfigurationManager {
    /// Creates a new configuration manager
    /// with the specified configuration file path
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        
        if !config_path.exists() {
            return Err(ConfigError::FileNotFound(config_path));
        }
        
        let manager = ConfigurationManager {
            config_path,
            cache: HashMap::new(),
            validation_rules: Vec::new(),
        };
        
        Ok(manager)
    }
}
"#;
        let mut within_file = self.create_test_file("proximity_within.rs", within_content, TestFileType::Proximity)?;
        within_file.expected_matches = vec![
            "Configuration management system".to_string(),  // Should match "Configuration WITHIN/10 system"
            "configuration file path".to_string(),    // Should match "configuration WITHIN/5 file"
            "config_path.exists()".to_string(),       // Should match "config WITHIN/3 path"
            "validation_rules: Vec".to_string(),      // Should match "validation WITHIN/5 rules"
        ];
        files.push(within_file);
        
        // Cross-boundary proximity test
        let cross_boundary_content = r#"
fn database_connection() -> Result<Connection> {
    let database_url = env::var("DATABASE_URL")
        .context("DATABASE_URL environment variable not set")?;
    
    let connection_pool = Pool::builder()
        .max_size(10)
        .build(database_url)
        .context("Failed to create database connection pool")?;
    
    // Test cross-line proximity
    let migration_status = check_migration_status(&connection_pool)
        .await
        .context("Migration status check failed")?;
    
    if migration_status.pending_migrations > 0 {
        println!("Warning: {} pending migrations detected", 
                migration_status.pending_migrations);
    }
    
    Ok(connection_pool.get()?)
}
"#;
        let mut cross_boundary_file = self.create_test_file("proximity_cross_boundary.rs", cross_boundary_content, TestFileType::Proximity)?;
        cross_boundary_file.expected_matches = vec![
            "database_connection() -> Result".to_string(),  // Should match "database NEAR/5 connection"
            "DATABASE_URL environment variable".to_string(), // Cross-line proximity
            "migration_status.pending_migrations".to_string(), // Should work across lines
            "connection_pool.get()".to_string(),            // Should match "connection NEAR/3 pool"
        ];
        files.push(cross_boundary_file);
        
        // Negative proximity test - words too far apart
        let negative_proximity_content = r#"
pub struct UserAuthentication {
    username: String,
    password_hash: String,
    email_address: String,
    created_at: DateTime<Utc>,
    last_login: Option<DateTime<Utc>>,
    failed_attempts: u32,
    account_locked: bool,
    two_factor_enabled: bool,
    recovery_codes: Vec<String>,
    session_tokens: HashMap<String, SessionInfo>,
}

impl UserAuthentication {
    pub fn verify_credentials(&self, provided_password: &str) -> AuthResult {
        if self.account_locked {
            return AuthResult::AccountLocked;
        }
        
        let password_valid = bcrypt::verify(provided_password, &self.password_hash)
            .map_err(|_| AuthError::HashingError)?;
            
        if password_valid {
            AuthResult::Success
        } else {
            AuthResult::InvalidCredentials
        }
    }
}
"#;
        let mut negative_file = self.create_test_file("proximity_negative.rs", negative_proximity_content, TestFileType::Proximity)?;
        negative_file.expected_matches = vec![
            // These should NOT match with NEAR/3 but might with WITHIN/20
            "UserAuthentication".to_string(),     // Start of struct
            "verify_credentials".to_string(),     // Method name - far from "password"
            "account_locked".to_string(),         // Field name
            "AuthResult::Success".to_string(),    // End result
        ];
        files.push(negative_file);
        
        // Code structure proximity test
        let code_structure_content = r#"
#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse<T> {
    pub status: ResponseStatus,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub enum ResponseStatus {
    Success,
    Error,
    Partial,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        ApiResponse {
            status: ResponseStatus::Success,
            data: Some(data),
            error: None,
            metadata: ResponseMetadata::default(),
        }
    }
    
    pub fn error(error: ApiError) -> Self {
        ApiResponse {
            status: ResponseStatus::Error,
            data: None,
            error: Some(error),
            metadata: ResponseMetadata::default(),
        }
    }
}
"#;
        let mut code_structure_file = self.create_test_file("proximity_code_structure.rs", code_structure_content, TestFileType::Proximity)?;
        code_structure_file.expected_matches = vec![
            "ApiResponse<T>".to_string(),           // Should match "Api NEAR/2 Response"
            "ResponseStatus::Success".to_string(),   // Should match "Response NEAR/3 Status"
            "Some(data)".to_string(),               // Should match "Some NEAR/2 data"
            "ResponseMetadata::default()".to_string(), // Should match "Response NEAR/3 Metadata"
        ];
        files.push(code_structure_file);
        
        Ok(files)
    }
}
```

## Success Criteria
- Method generates 5+ test files with diverse proximity scenarios
- Each file includes expected_matches for validation testing
- Files test different proximity distances (NEAR/3, WITHIN/10, etc.)
- Includes both positive and negative proximity test cases
- Cross-boundary proximity tests work across lines and paragraphs
- Code-specific proximity patterns are thoroughly tested
- Performance considerations for complex proximity queries

## Time Limit
10 minutes maximum