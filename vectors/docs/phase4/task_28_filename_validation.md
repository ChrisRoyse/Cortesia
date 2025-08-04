# Task 28: Complete Windows Filename Validation System

## Context
You are implementing Phase 4 of a vector indexing system. Extended path support has been implemented. Now you need to create a comprehensive filename validation system that goes beyond basic reserved character checking to handle all Windows filename edge cases and provide detailed validation feedback.

## Current State
- `src/windows.rs` has extended path support
- Basic filename validation exists with reserved characters and names
- Extended path handling is working
- Need comprehensive validation with detailed error reporting

## Task Objective
Implement a complete Windows filename validation system with detailed error reporting, sanitization capabilities, and integration with the file indexing system.

## Implementation Requirements

### 1. Add detailed validation error types
Add this error enum to `src/windows.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum FilenameValidationError {
    Empty,
    TooLong { length: usize, max_length: usize },
    ContainsReservedChar { char: char, position: usize },
    ContainsControlChar { char_code: u32, position: usize },
    EndsWithSpaceOrPeriod { char: char },
    ReservedName { name: String, case_insensitive: bool },
    ContainsNullByte { position: usize },
    InvalidUnicode { position: usize },
    StartsWithSpace,
    OnlyWhitespace,
}

impl std::fmt::Display for FilenameValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilenameValidationError::Empty => write!(f, "Filename cannot be empty"),
            FilenameValidationError::TooLong { length, max_length } => {
                write!(f, "Filename too long: {} characters (max {})", length, max_length)
            }
            FilenameValidationError::ContainsReservedChar { char, position } => {
                write!(f, "Contains reserved character '{}' at position {}", char, position)
            }
            FilenameValidationError::ContainsControlChar { char_code, position } => {
                write!(f, "Contains control character (code {}) at position {}", char_code, position)
            }
            FilenameValidationError::EndsWithSpaceOrPeriod { char } => {
                write!(f, "Filename cannot end with '{}'", char)
            }
            FilenameValidationError::ReservedName { name, case_insensitive } => {
                if *case_insensitive {
                    write!(f, "Reserved filename '{}' (case-insensitive)", name)
                } else {
                    write!(f, "Reserved filename '{}'", name)
                }
            }
            FilenameValidationError::ContainsNullByte { position } => {
                write!(f, "Contains null byte at position {}", position)
            }
            FilenameValidationError::InvalidUnicode { position } => {
                write!(f, "Invalid Unicode sequence at position {}", position)
            }
            FilenameValidationError::StartsWithSpace => {
                write!(f, "Filename cannot start with a space")
            }
            FilenameValidationError::OnlyWhitespace => {
                write!(f, "Filename cannot consist only of whitespace")
            }
        }
    }
}

impl std::error::Error for FilenameValidationError {}
```

### 2. Add comprehensive validation result
Add this validation result structure:
```rust
#[derive(Debug, Clone)]
pub struct FilenameValidationResult {
    pub is_valid: bool,
    pub errors: Vec<FilenameValidationError>,
    pub warnings: Vec<String>,
    pub sanitized_name: Option<String>,
}

impl FilenameValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            sanitized_name: None,
        }
    }
    
    pub fn add_error(&mut self, error: FilenameValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}
```

### 3. Add comprehensive filename validation
Replace the basic validation with this comprehensive version:
```rust
impl WindowsPathHandler {
    pub fn validate_filename_detailed(&self, filename: &str) -> FilenameValidationResult {
        let mut result = FilenameValidationResult::new();
        
        // Check for empty filename
        if filename.is_empty() {
            result.add_error(FilenameValidationError::Empty);
            return result;
        }
        
        // Check for only whitespace
        if filename.trim().is_empty() {
            result.add_error(FilenameValidationError::OnlyWhitespace);
            return result;
        }
        
        // Check if starts with space
        if filename.starts_with(' ') {
            result.add_error(FilenameValidationError::StartsWithSpace);
        }
        
        // Check length
        const MAX_FILENAME_LENGTH: usize = 255;
        if filename.len() > MAX_FILENAME_LENGTH {
            result.add_error(FilenameValidationError::TooLong {
                length: filename.len(),
                max_length: MAX_FILENAME_LENGTH,
            });
        }
        
        // Check each character
        for (pos, ch) in filename.char_indices() {
            // Check for null bytes
            if ch == '\0' {
                result.add_error(FilenameValidationError::ContainsNullByte { position: pos });
                continue;
            }
            
            // Check for reserved characters
            if RESERVED_CHARS.contains(&ch) {
                result.add_error(FilenameValidationError::ContainsReservedChar {
                    char: ch,
                    position: pos,
                });
                continue;
            }
            
            // Check for control characters (0-31)
            let char_code = ch as u32;
            if char_code <= 31 {
                result.add_error(FilenameValidationError::ContainsControlChar {
                    char_code,
                    position: pos,
                });
                continue;
            }
            
            // Check for problematic Unicode ranges
            if self.is_problematic_unicode(ch) {
                result.add_warning(format!(
                    "Character '{}' at position {} may cause compatibility issues",
                    ch, pos
                ));
            }
        }
        
        // Check for trailing spaces or periods
        if let Some(last_char) = filename.chars().last() {
            if last_char == ' ' || last_char == '.' {
                result.add_error(FilenameValidationError::EndsWithSpaceOrPeriod {
                    char: last_char,
                });
            }
        }
        
        // Check for reserved names
        let upper_filename = filename.to_uppercase();
        let base_name = upper_filename.split('.').next().unwrap_or("");
        
        if RESERVED_NAMES.contains(&base_name) {
            result.add_error(FilenameValidationError::ReservedName {
                name: filename.to_string(),
                case_insensitive: true,
            });
        }
        
        // Add warnings for potentially problematic names
        self.add_filename_warnings(filename, &mut result);
        
        // Generate sanitized version if there are errors
        if !result.is_valid {
            result.sanitized_name = Some(self.sanitize_filename(filename));
        }
        
        result
    }
    
    fn is_problematic_unicode(&self, ch: char) -> bool {
        let code = ch as u32;
        
        // Check for various problematic Unicode ranges
        matches!(code,
            // Bidirectional override characters
            0x202A..=0x202E |
            // Zero-width characters
            0x200B..=0x200F |
            // High surrogates (should not appear alone in UTF-8)
            0xD800..=0xDBFF |
            // Low surrogates (should not appear alone in UTF-8)
            0xDC00..=0xDFFF |
            // Private use areas (may not display consistently)
            0xE000..=0xF8FF |
            // Specials block
            0xFFF0..=0xFFFF
        )
    }
    
    fn add_filename_warnings(&self, filename: &str, result: &mut FilenameValidationResult) {
        // Warn about leading periods (hidden files on Unix-like systems)
        if filename.starts_with('.') {
            result.add_warning("Filename starts with period (hidden file on Unix systems)".to_string());
        }
        
        // Warn about mixed case in extensions
        if let Some(extension_pos) = filename.rfind('.') {
            let extension = &filename[extension_pos + 1..];
            if extension.chars().any(|c| c.is_uppercase()) && extension.chars().any(|c| c.is_lowercase()) {
                result.add_warning("Extension has mixed case which may cause issues".to_string());
            }
        }
        
        // Warn about very long filenames (before the hard limit)
        if filename.len() > 200 {
            result.add_warning(format!(
                "Filename is {} characters long which may cause portability issues",
                filename.len()
            ));
        }
        
        // Warn about non-ASCII characters
        if !filename.is_ascii() {
            result.add_warning("Filename contains non-ASCII characters".to_string());
        }
        
        // Warn about multiple consecutive spaces
        if filename.contains("  ") {
            result.add_warning("Filename contains multiple consecutive spaces".to_string());
        }
        
        // Warn about very short extensions
        if let Some(extension_pos) = filename.rfind('.') {
            let extension = &filename[extension_pos + 1..];
            if extension.len() == 1 {
                result.add_warning("Single-character extension may not be recognized".to_string());
            }
        }
    }
    
    pub fn sanitize_filename(&self, filename: &str) -> String {
        if filename.is_empty() {
            return "unnamed_file".to_string();
        }
        
        let mut sanitized = String::new();
        let mut chars = filename.chars();
        
        // Skip leading spaces
        while let Some(ch) = chars.next() {
            if ch != ' ' {
                sanitized.push(ch);
                break;
            }
        }
        
        // Process remaining characters
        for ch in chars {
            if ch == '\0' {
                continue; // Skip null bytes
            } else if RESERVED_CHARS.contains(&ch) {
                sanitized.push('_'); // Replace reserved chars
            } else if (ch as u32) <= 31 {
                continue; // Skip control characters
            } else if self.is_problematic_unicode(ch) {
                sanitized.push('_'); // Replace problematic Unicode
            } else {
                sanitized.push(ch);
            }
        }
        
        // Remove trailing spaces and periods
        while let Some(last_char) = sanitized.chars().last() {
            if last_char == ' ' || last_char == '.' {
                sanitized.pop();
            } else {
                break;
            }
        }
        
        // If empty after sanitization, provide default
        if sanitized.is_empty() || sanitized.trim().is_empty() {
            sanitized = "sanitized_file".to_string();
        }
        
        // Handle reserved names
        let upper_sanitized = sanitized.to_uppercase();
        let base_name = upper_sanitized.split('.').next().unwrap_or("");
        
        if RESERVED_NAMES.contains(&base_name) {
            sanitized = format!("_{}", sanitized);
        }
        
        // Ensure reasonable length
        const MAX_SAFE_LENGTH: usize = 200;
        if sanitized.len() > MAX_SAFE_LENGTH {
            // Try to preserve extension
            if let Some(dot_pos) = sanitized.rfind('.') {
                let extension = &sanitized[dot_pos..];
                let name_part = &sanitized[..dot_pos];
                let truncated_name = &name_part[..MAX_SAFE_LENGTH.saturating_sub(extension.len())];
                sanitized = format!("{}{}", truncated_name, extension);
            } else {
                sanitized.truncate(MAX_SAFE_LENGTH);
            }
        }
        
        sanitized
    }
    
    pub fn validate_filename_quick(&self, filename: &str) -> bool {
        let result = self.validate_filename_detailed(filename);
        result.is_valid
    }
    
    pub fn get_filename_suggestions(&self, filename: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Always include sanitized version
        suggestions.push(self.sanitize_filename(filename));
        
        // Provide alternative suggestions
        if filename.contains(' ') {
            // Replace spaces with underscores
            let with_underscores = filename.replace(' ', "_");
            if self.validate_filename_quick(&with_underscores) {
                suggestions.push(with_underscores);
            }
            
            // Replace spaces with hyphens
            let with_hyphens = filename.replace(' ', "-");
            if self.validate_filename_quick(&with_hyphens) {
                suggestions.push(with_hyphens);
            }
        }
        
        // Remove duplicates
        suggestions.sort();
        suggestions.dedup();
        
        suggestions
    }
}
```

### 4. Add batch validation for paths
Add these methods for validating entire paths:
```rust
impl WindowsPathHandler {
    pub fn validate_path_components(&self, path: &Path) -> Vec<(String, FilenameValidationResult)> {
        let mut results = Vec::new();
        
        for component in path.components() {
            if let std::path::Component::Normal(os_str) = component {
                let filename = os_str.to_string_lossy();
                let validation = self.validate_filename_detailed(&filename);
                results.push((filename.to_string(), validation));
            }
        }
        
        results
    }
    
    pub fn is_path_safe_for_indexing(&self, path: &Path) -> bool {
        let component_results = self.validate_path_components(path);
        
        // Check if all components are valid
        component_results.iter().all(|(_, result)| result.is_valid)
    }
    
    pub fn sanitize_path(&self, path: &Path) -> Result<PathBuf> {
        let mut sanitized_components = Vec::new();
        
        // Handle the root/prefix part
        if let Some(prefix) = path.components().next() {
            match prefix {
                std::path::Component::Prefix(prefix_component) => {
                    sanitized_components.push(prefix_component.as_os_str().to_string_lossy().to_string());
                }
                std::path::Component::RootDir => {
                    sanitized_components.push("\\".to_string());
                }
                _ => {}
            }
        }
        
        // Sanitize each normal component
        for component in path.components().skip(1) {
            match component {
                std::path::Component::Normal(os_str) => {
                    let filename = os_str.to_string_lossy();
                    let sanitized = self.sanitize_filename(&filename);
                    sanitized_components.push(sanitized);
                }
                std::path::Component::RootDir => {
                    sanitized_components.push("\\".to_string());
                }
                std::path::Component::ParentDir => {
                    sanitized_components.push("..".to_string());
                }
                std::path::Component::CurDir => {
                    continue; // Skip current directory references
                }
                _ => {}
            }
        }
        
        // Reconstruct path
        let sanitized_path_str = sanitized_components.join("\\");
        Ok(PathBuf::from(sanitized_path_str))
    }
}
```

### 5. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_detailed_filename_validation() {
    let handler = WindowsPathHandler::new();
    
    // Valid filename
    let result = handler.validate_filename_detailed("valid_file.txt");
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    
    // Invalid filename with reserved character
    let result = handler.validate_filename_detailed("file<name>.txt");
    assert!(!result.is_valid);
    assert!(matches!(result.errors[0], FilenameValidationError::ContainsReservedChar { char: '<', position: 4 }));
    
    // Reserved name
    let result = handler.validate_filename_detailed("CON");
    assert!(!result.is_valid);
    assert!(matches!(result.errors[0], FilenameValidationError::ReservedName { .. }));
    
    // Ends with period
    let result = handler.validate_filename_detailed("file.");
    assert!(!result.is_valid);
    assert!(matches!(result.errors[0], FilenameValidationError::EndsWithSpaceOrPeriod { char: '.' }));
}

#[test]
fn test_filename_sanitization() {
    let handler = WindowsPathHandler::new();
    
    // Sanitize reserved characters
    assert_eq!(handler.sanitize_filename("file<name>.txt"), "file_name_.txt");
    
    // Sanitize reserved name
    assert_eq!(handler.sanitize_filename("CON"), "_CON");
    
    // Sanitize trailing period
    assert_eq!(handler.sanitize_filename("file."), "file");
    
    // Sanitize empty string
    assert_eq!(handler.sanitize_filename(""), "unnamed_file");
    
    // Sanitize control characters
    assert_eq!(handler.sanitize_filename("file\x01name.txt"), "filename.txt");
}

#[test]
fn test_filename_suggestions() {
    let handler = WindowsPathHandler::new();
    
    let suggestions = handler.get_filename_suggestions("my file name.txt");
    assert!(suggestions.len() > 1);
    assert!(suggestions.contains(&"my_file_name.txt".to_string()));
    assert!(suggestions.contains(&"my-file-name.txt".to_string()));
}

#[test]
fn test_path_component_validation() {
    let handler = WindowsPathHandler::new();
    
    let path = Path::new(r"C:\valid\path\file.txt");
    let results = handler.validate_path_components(&path);
    
    // Should have 3 components: valid, path, file.txt
    assert_eq!(results.len(), 3);
    
    for (_, result) in &results {
        assert!(result.is_valid);
    }
}

#[test]
fn test_path_safety_for_indexing() {
    let handler = WindowsPathHandler::new();
    
    // Safe path
    let safe_path = Path::new(r"C:\documents\report.pdf");
    assert!(handler.is_path_safe_for_indexing(safe_path));
    
    // Unsafe path with reserved character
    let unsafe_path = Path::new(r"C:\documents\report<copy>.pdf");
    assert!(!handler.is_path_safe_for_indexing(unsafe_path));
}

#[test]
fn test_path_sanitization() -> Result<()> {
    let handler = WindowsPathHandler::new();
    
    let problematic_path = Path::new(r"C:\my folder\file<name>.txt");
    let sanitized = handler.sanitize_path(problematic_path)?;
    
    let sanitized_str = sanitized.to_string_lossy();
    assert!(!sanitized_str.contains('<'));
    assert!(handler.is_path_safe_for_indexing(&sanitized));
    
    Ok(())
}

#[test]
fn test_unicode_filename_validation() {
    let handler = WindowsPathHandler::new();
    
    // Test with Unicode filename
    let result = handler.validate_filename_detailed("测试文件.txt");
    assert!(result.is_valid);
    assert!(result.has_warnings()); // Should warn about non-ASCII
    
    // Test with problematic Unicode
    let result = handler.validate_filename_detailed("file\u{202E}name.txt"); // Right-to-left override
    assert!(!result.is_valid || result.has_warnings());
}

#[test]
fn test_filename_length_validation() {
    let handler = WindowsPathHandler::new();
    
    // Very long filename
    let long_name = "x".repeat(300);
    let result = handler.validate_filename_detailed(&long_name);
    assert!(!result.is_valid);
    assert!(matches!(result.errors[0], FilenameValidationError::TooLong { .. }));
    
    // Sanitized version should be shorter
    let sanitized = handler.sanitize_filename(&long_name);
    assert!(sanitized.len() <= 200);
}

#[test]
fn test_control_character_handling() {
    let handler = WindowsPathHandler::new();
    
    // Filename with control character
    let filename_with_control = format!("file{}name.txt", '\x01');
    let result = handler.validate_filename_detailed(&filename_with_control);
    assert!(!result.is_valid);
    
    let sanitized = handler.sanitize_filename(&filename_with_control);
    assert_eq!(sanitized, "filename.txt");
}
```

## Success Criteria
- [ ] Detailed filename validation with comprehensive error reporting
- [ ] Filename sanitization produces valid Windows filenames
- [ ] Path component validation works for entire directory structures
- [ ] Unicode character handling includes problematic sequences
- [ ] Batch validation efficiently processes multiple filenames
- [ ] Integration ready for file indexing system
- [ ] All tests pass including edge cases
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Windows filename validation has many edge cases beyond reserved characters
- Unicode handling is critical for international filename support
- Sanitization should preserve filename meaning when possible
- Warning system helps users understand potential issues
- Path-level validation ensures entire directory structures are safe for indexing
- Performance considerations for batch validation of large directory trees