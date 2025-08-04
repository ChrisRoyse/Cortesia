# Task 026: Generate Wildcard Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-012. Wildcard patterns (*,?) are essential for flexible code search and pattern matching in development environments.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_wildcard_tests()` method that creates test files for validating wildcard pattern matching with various complexity levels and edge cases.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files testing single wildcards (*, ?) and combinations
3. Include complex wildcard patterns with multiple wildcards
4. Add character range and set matching patterns
5. Create code-specific wildcard tests (function names, variable patterns)
6. Include edge cases (empty matches, overlapping patterns)
7. Generate performance stress tests with expensive wildcard operations

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_wildcard_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Basic wildcard patterns
        let basic_wildcard_content = r#"
// Function naming patterns
pub fn create_user_account() -> Result<Account> {}
pub fn create_admin_account() -> Result<Account> {}
pub fn create_guest_account() -> Result<Account> {}
pub fn delete_user_account(id: u64) -> Result<()> {}
pub fn delete_admin_account(id: u64) -> Result<()> {}
pub fn update_user_profile(data: ProfileData) -> Result<()> {}
pub fn update_admin_settings(settings: AdminSettings) -> Result<()> {}

// Variable naming patterns  
let user_id = 12345;
let admin_id = 67890;
let guest_id = 11111;
let session_id = "abc123";
let request_id = "req_456";
let transaction_id = "txn_789";

// Type patterns
struct UserConfig { name: String }
struct AdminConfig { permissions: Vec<String> }
struct GuestConfig { access_level: u8 }
enum UserType { Standard, Premium, Trial }
enum AdminType { Super, Moderator, ReadOnly }
"#;
        let mut basic_file = self.create_test_file("wildcard_basic.rs", basic_wildcard_content, TestFileType::Wildcard)?;
        basic_file.expected_matches = vec![
            "create_*_account".to_string(),        // Should match all create_X_account functions
            "*_id".to_string(),                    // Should match all variables ending in _id
            "*Config".to_string(),                 // Should match UserConfig, AdminConfig, GuestConfig
            "*Type".to_string(),                   // Should match UserType, AdminType
            "?ser*".to_string(),                   // Should match "user" in various contexts
        ];
        files.push(basic_file);
        
        // Complex wildcard combinations
        let complex_wildcard_content = r#"
impl DatabaseConnection {
    pub async fn find_users_by_email(&self, pattern: &str) -> Result<Vec<User>> {}
    pub async fn find_users_by_name(&self, pattern: &str) -> Result<Vec<User>> {}
    pub async fn find_users_by_role(&self, role: UserRole) -> Result<Vec<User>> {}
    pub async fn find_posts_by_title(&self, title: &str) -> Result<Vec<Post>> {}
    pub async fn find_posts_by_author(&self, author_id: u64) -> Result<Vec<Post>> {}
    pub async fn find_posts_by_category(&self, category: &str) -> Result<Vec<Post>> {}
    pub async fn create_user_session(&self, user_id: u64) -> Result<Session> {}
    pub async fn create_admin_session(&self, admin_id: u64) -> Result<Session> {}
    pub async fn validate_user_token(&self, token: &str) -> Result<bool> {}
    pub async fn validate_admin_token(&self, token: &str) -> Result<bool> {}
}

// File path patterns
const USER_DATA_PATH: &str = "/data/users/profiles/";
const ADMIN_DATA_PATH: &str = "/data/admin/settings/";
const LOG_FILE_PATH: &str = "/logs/application/debug.log";
const CONFIG_FILE_PATH: &str = "/config/app/production.toml";
const BACKUP_FILE_PATH: &str = "/backups/daily/2023-12-01.sql";
"#;
        let mut complex_file = self.create_test_file("wildcard_complex.rs", complex_wildcard_content, TestFileType::Wildcard)?;
        complex_file.expected_matches = vec![
            "find_*_by_*".to_string(),             // Should match find_users_by_email, find_posts_by_title, etc.
            "*_*_session".to_string(),             // Should match create_user_session, create_admin_session
            "validate_*_token".to_string(),        // Should match both validate functions
            "/*/app/*".to_string(),                // Should match config file path
            "*_DATA_PATH".to_string(),             // Should match USER_DATA_PATH, ADMIN_DATA_PATH
        ];
        files.push(complex_file);
        
        // Character range and set patterns
        let character_range_content = r#"
// Version patterns
const VERSION_1_0_0: &str = "1.0.0";
const VERSION_1_2_3: &str = "1.2.3";
const VERSION_2_0_0: &str = "2.0.0";
const VERSION_2_1_5: &str = "2.1.5";
const VERSION_3_0_0_BETA: &str = "3.0.0-beta";

// HTTP status codes
fn handle_200_ok() -> Response {}
fn handle_201_created() -> Response {}
fn handle_400_bad_request() -> Response {}
fn handle_401_unauthorized() -> Response {}
fn handle_404_not_found() -> Response {}
fn handle_500_internal_error() -> Response {}

// Database table names
CREATE TABLE user_profiles_2023 (id INTEGER);
CREATE TABLE user_profiles_2024 (id INTEGER);
CREATE TABLE admin_logs_2023 (id INTEGER);
CREATE TABLE admin_logs_2024 (id INTEGER);
CREATE TABLE temp_data_jan (id INTEGER);
CREATE TABLE temp_data_feb (id INTEGER);
CREATE TABLE temp_data_mar (id INTEGER);

// API endpoint patterns
/api/v1/users/{id}
/api/v2/users/{id}/profile
/api/v1/posts/{id}/comments
/api/v2/admin/{id}/permissions
"#;
        let mut char_range_file = self.create_test_file("wildcard_char_ranges.rs", character_range_content, TestFileType::Wildcard)?;
        char_range_file.expected_matches = vec![
            "VERSION_[123]_*".to_string(),         // Should match versions 1.x.x, 2.x.x, 3.x.x
            "handle_[2-5]??_*".to_string(),        // Should match HTTP status handlers
            "*_202[34]".to_string(),               // Should match 2023 and 2024 tables
            "/api/v[12]/".to_string(),             // Should match API v1 and v2 endpoints
            "temp_data_[jfm]*".to_string(),        // Should match jan, feb, mar tables
        ];
        files.push(char_range_file);
        
        // Edge cases and empty matches
        let edge_cases_content = r#"
// Empty and minimal patterns
pub fn a() {}
pub fn b() {}
pub fn c() {}
pub fn aa() {}
pub fn bb() {}
pub fn cc() {}
pub fn ab() {}
pub fn ac() {}
pub fn bc() {}

// Single character variables
let x = 1;
let y = 2;
let z = 3;
let a = "test";
let b = "data";
let c = "info";

// Overlapping patterns
fn test_function_one() {}
fn test_function_two() {}
fn test_method_one() {}
fn test_method_two() {}
fn validate_function_input() {}
fn validate_method_input() {}

// Potential empty matches
let empty_string = "";
let null_value = None;
let default_config = Default::default();
let _ = placeholder_value;

// Special characters in names
fn test_with_underscores_() {}
fn test__double__underscores() {}
fn test___triple___underscores() {}
"#;
        let mut edge_cases_file = self.create_test_file("wildcard_edge_cases.rs", edge_cases_content, TestFileType::Wildcard)?;
        edge_cases_file.expected_matches = vec![
            "?".to_string(),                       // Should match single character functions
            "??".to_string(),                      // Should match two character functions
            "*function*".to_string(),              // Should match functions containing "function"
            "test_*_*".to_string(),               // Should match complex test patterns
            "*___*".to_string(),                   // Should match triple underscore pattern
        ];
        files.push(edge_cases_file);
        
        // Performance stress test with expensive wildcards
        let performance_stress_content = r#"
// Large number of similar patterns for stress testing
mod performance_test {
    // Generate many similar function names
    pub fn process_data_chunk_001() -> Result<()> { Ok(()) }
    pub fn process_data_chunk_002() -> Result<()> { Ok(()) }
    pub fn process_data_chunk_003() -> Result<()> { Ok(()) }
    pub fn process_data_chunk_004() -> Result<()> { Ok(()) }
    pub fn process_data_chunk_005() -> Result<()> { Ok(()) }
    pub fn process_user_data_001() -> Result<()> { Ok(()) }
    pub fn process_user_data_002() -> Result<()> { Ok(()) }
    pub fn process_user_data_003() -> Result<()> { Ok(()) }
    pub fn process_admin_data_001() -> Result<()> { Ok(()) }
    pub fn process_admin_data_002() -> Result<()> { Ok(()) }
    
    // Many variable patterns
    let config_database_host_primary = "localhost";
    let config_database_host_secondary = "backup";
    let config_database_port_primary = 5432;
    let config_database_port_secondary = 5433;
    let config_redis_host_primary = "redis1";
    let config_redis_host_secondary = "redis2";
    let config_redis_port_primary = 6379;
    let config_redis_port_secondary = 6380;
    
    // Nested wildcard patterns that could be expensive
    fn deeply_nested_function_with_many_parameters_and_complex_name() {}
    fn another_deeply_nested_function_with_similar_pattern_structure() {}
    fn yet_another_complex_function_name_following_same_pattern() {}
}
"#;
        let mut performance_file = self.create_test_file("wildcard_performance.rs", performance_stress_content, TestFileType::Wildcard)?;
        performance_file.expected_matches = vec![
            "process_*_chunk_*".to_string(),       // Should match all chunk processing functions
            "config_*_*_primary".to_string(),      // Should match all primary config variables
            "*_host_*".to_string(),                // Should match all host variables
            "*deeply*nested*function*".to_string(),// Complex wildcard with multiple parts
            "process_*_data_00?".to_string(),      // Combination of * and ? wildcards
        ];
        files.push(performance_file);
        
        Ok(files)
    }
}
```

## Success Criteria
- Method generates 5+ test files with diverse wildcard patterns
- Each file includes expected_matches for validation testing
- Files test single wildcards (*,?) and complex combinations
- Character ranges and sets are properly tested
- Edge cases including empty matches are covered
- Performance stress tests with expensive wildcard operations
- Code-specific patterns (functions, variables, types) are tested

## Time Limit
10 minutes maximum