# Task 27: Implement Windows Extended Path Support

## Context
You are implementing Phase 4 of a vector indexing system. The Windows path handler has been created with basic normalization capabilities. Now you need to implement comprehensive extended path support to handle Windows long path scenarios that exceed the standard 260-character limit.

## Current State
- `src/windows.rs` exists with enhanced path normalization
- Path type detection is implemented
- Basic extended path conversion exists
- Need to add robust extended path handling for the indexing system

## Task Objective
Implement comprehensive Windows extended path support with automatic fallback mechanisms, long path validation, and integration with the file indexing system.

## Implementation Requirements

### 1. Add extended path configuration
Add this configuration struct to `src/windows.rs`:
```rust
#[derive(Debug, Clone)]
pub struct ExtendedPathConfig {
    pub enable_extended_paths: bool,
    pub auto_convert_long_paths: bool,
    pub max_path_length: usize,
    pub fallback_on_error: bool,
}

impl Default for ExtendedPathConfig {
    fn default() -> Self {
        Self {
            enable_extended_paths: true,
            auto_convert_long_paths: true,
            max_path_length: MAX_WINDOWS_PATH_LENGTH,
            fallback_on_error: true,
        }
    }
}

impl WindowsPathHandler {
    pub fn with_config(config: ExtendedPathConfig) -> Self {
        WindowsPathHandler { config }
    }
}
```

### 2. Update WindowsPathHandler struct
Modify the struct to include configuration:
```rust
pub struct WindowsPathHandler {
    config: ExtendedPathConfig,
}

impl WindowsPathHandler {
    pub fn new() -> Self {
        Self {
            config: ExtendedPathConfig::default(),
        }
    }
}
```

### 3. Add extended path handling methods
Add these comprehensive extended path methods:
```rust
pub fn ensure_extended_path(&self, path: &Path) -> Result<PathBuf> {
    if !self.config.enable_extended_paths {
        return Ok(path.to_path_buf());
    }
    
    let path_str = path.to_string_lossy();
    let path_length = path_str.len();
    
    // Check if conversion is needed
    if path_length <= MAX_STANDARD_PATH_LENGTH && !self.config.auto_convert_long_paths {
        return Ok(path.to_path_buf());
    }
    
    // Already an extended path
    if path_str.starts_with(r"\\?\") {
        return Ok(path.to_path_buf());
    }
    
    // Convert to extended path
    self.convert_to_extended_path(path)
}

fn convert_to_extended_path(&self, path: &Path) -> Result<PathBuf> {
    let normalized = self.normalize_path(path)?;
    let path_type = Self::get_path_type(&normalized);
    
    match path_type {
        WindowsPathType::DriveAbsolute => {
            let path_str = normalized.to_string_lossy();
            Ok(PathBuf::from(format!(r"\\?\{}", path_str)))
        }
        WindowsPathType::UNC => {
            let path_str = normalized.to_string_lossy();
            let without_unc = path_str.trim_start_matches(r"\\");
            Ok(PathBuf::from(format!(r"\\?\UNC\{}", without_unc)))
        }
        WindowsPathType::Relative => {
            // Convert to absolute first
            let absolute = std::env::current_dir()?.join(&normalized);
            self.convert_to_extended_path(&absolute)
        }
        _ => {
            // Already extended or device path
            Ok(normalized)
        }
    }
}

pub fn try_extended_path_operation<F, R>(&self, path: &Path, operation: F) -> Result<R>
where
    F: Fn(&Path) -> Result<R>,
{
    // Try with original path first
    match operation(path) {
        Ok(result) => Ok(result),
        Err(original_error) => {
            if !self.config.fallback_on_error {
                return Err(original_error);
            }
            
            // Try with extended path
            match self.ensure_extended_path(path) {
                Ok(extended_path) => {
                    operation(&extended_path).map_err(|extended_error| {
                        anyhow::anyhow!(
                            "Both original and extended path operations failed. Original: {}, Extended: {}",
                            original_error,
                            extended_error
                        )
                    })
                }
                Err(conversion_error) => {
                    Err(anyhow::anyhow!(
                        "Path conversion failed: {}. Original error: {}",
                        conversion_error,
                        original_error
                    ))
                }
            }
        }
    }
}

pub fn is_extended_path_needed(&self, path: &Path) -> bool {
    if !self.config.enable_extended_paths {
        return false;
    }
    
    let path_str = path.to_string_lossy();
    
    // Check path length
    if path_str.len() > MAX_STANDARD_PATH_LENGTH {
        return true;
    }
    
    // Check component lengths (each component has its own limit)
    for component in path.components() {
        if let std::path::Component::Normal(os_str) = component {
            if os_str.to_string_lossy().len() > 255 {
                return true;
            }
        }
    }
    
    false
}

pub fn validate_extended_path(&self, path: &Path) -> Result<()> {
    let path_str = path.to_string_lossy();
    
    // Check overall length
    if path_str.len() > self.config.max_path_length {
        return Err(anyhow::anyhow!(
            "Path length {} exceeds maximum allowed length {}",
            path_str.len(),
            self.config.max_path_length
        ));
    }
    
    // Validate extended path format
    if path_str.starts_with(r"\\?\") {
        let without_prefix = &path_str[4..];
        
        if without_prefix.starts_with(r"UNC\") {
            // Extended UNC path
            let unc_part = &without_prefix[4..];
            if unc_part.is_empty() {
                return Err(anyhow::anyhow!("Invalid extended UNC path format"));
            }
            
            let parts: Vec<&str> = unc_part.split('\\').collect();
            if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
                return Err(anyhow::anyhow!("Invalid UNC server/share format"));
            }
        } else {
            // Extended local path
            if without_prefix.len() < 3 || !without_prefix.chars().nth(1).map_or(false, |c| c == ':') {
                return Err(anyhow::anyhow!("Invalid extended local path format"));
            }
        }
    }
    
    // Validate each component
    for component in path.components() {
        if let std::path::Component::Normal(os_str) = component {
            let filename = os_str.to_string_lossy();
            if !Self::is_valid_windows_filename(&filename) {
                return Err(anyhow::anyhow!("Invalid filename component: {}", filename));
            }
        }
    }
    
    Ok(())
}
```

### 4. Add file system operation wrappers
Add these wrapper methods for common file operations:
```rust
pub fn open_file_extended(&self, path: &Path) -> Result<std::fs::File> {
    self.try_extended_path_operation(path, |p| {
        std::fs::File::open(p).map_err(|e| anyhow::anyhow!("Failed to open file: {}", e))
    })
}

pub fn create_file_extended(&self, path: &Path) -> Result<std::fs::File> {
    self.try_extended_path_operation(path, |p| {
        std::fs::File::create(p).map_err(|e| anyhow::anyhow!("Failed to create file: {}", e))
    })
}

pub fn metadata_extended(&self, path: &Path) -> Result<std::fs::Metadata> {
    self.try_extended_path_operation(path, |p| {
        std::fs::metadata(p).map_err(|e| anyhow::anyhow!("Failed to get metadata: {}", e))
    })
}

pub fn read_dir_extended(&self, path: &Path) -> Result<std::fs::ReadDir> {
    self.try_extended_path_operation(path, |p| {
        std::fs::read_dir(p).map_err(|e| anyhow::anyhow!("Failed to read directory: {}", e))
    })
}

pub fn canonicalize_extended(&self, path: &Path) -> Result<PathBuf> {
    self.try_extended_path_operation(path, |p| {
        p.canonicalize().map_err(|e| anyhow::anyhow!("Failed to canonicalize path: {}", e))
    })
}
```

### 5. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_extended_path_config() {
    let config = ExtendedPathConfig {
        enable_extended_paths: false,
        ..Default::default()
    };
    let handler = WindowsPathHandler::with_config(config);
    
    let long_path = PathBuf::from("C:\\".to_string() + &"very_long_directory\\".repeat(50) + "file.txt");
    let result = handler.ensure_extended_path(&long_path).unwrap();
    
    // Should not convert when disabled
    assert_eq!(result, long_path);
}

#[test]
fn test_extended_path_conversion() -> Result<()> {
    let handler = WindowsPathHandler::new();
    
    // Test drive path conversion
    let drive_path = Path::new(r"C:\test\file.txt");
    let extended = handler.ensure_extended_path(drive_path)?;
    assert!(extended.to_string_lossy().starts_with(r"\\?\"));
    
    // Test UNC path conversion
    let unc_path = Path::new(r"\\server\share\file.txt");
    let extended_unc = handler.ensure_extended_path(unc_path)?;
    assert!(extended_unc.to_string_lossy().starts_with(r"\\?\UNC\"));
    
    Ok(())
}

#[test]
fn test_extended_path_needed_detection() {
    let handler = WindowsPathHandler::new();
    
    // Short path - not needed
    let short_path = Path::new(r"C:\short\path.txt");
    assert!(!handler.is_extended_path_needed(short_path));
    
    // Long path - needed
    let long_path = PathBuf::from("C:\\".to_string() + &"very_long_directory_name\\".repeat(20) + "file.txt");
    assert!(handler.is_extended_path_needed(&long_path));
}

#[test]
fn test_extended_path_validation() -> Result<()> {
    let handler = WindowsPathHandler::new();
    
    // Valid extended paths
    assert!(handler.validate_extended_path(Path::new(r"\\?\C:\test\file.txt")).is_ok());
    assert!(handler.validate_extended_path(Path::new(r"\\?\UNC\server\share\file.txt")).is_ok());
    
    // Invalid extended paths
    assert!(handler.validate_extended_path(Path::new(r"\\?\")).is_err());
    assert!(handler.validate_extended_path(Path::new(r"\\?\UNC\")).is_err());
    assert!(handler.validate_extended_path(Path::new(r"\\?\UNC\server")).is_err());
    
    Ok(())
}

#[test]
fn test_extended_path_operations() -> Result<()> {
    use tempfile::TempDir;
    
    let handler = WindowsPathHandler::new();
    let temp_dir = TempDir::new()?;
    
    // Create a test file
    let test_file = temp_dir.path().join("test.txt");
    std::fs::write(&test_file, "test content")?;
    
    // Test file operations with extended path handling
    let metadata = handler.metadata_extended(&test_file)?;
    assert!(metadata.is_file());
    
    // Test opening file
    let _file = handler.open_file_extended(&test_file)?;
    
    // Test reading directory
    let _dir_entries = handler.read_dir_extended(temp_dir.path())?;
    
    Ok(())
}

#[test]
fn test_fallback_mechanism() -> Result<()> {
    let handler = WindowsPathHandler::new();
    
    // Test with non-existent path (should try extended path fallback)
    let non_existent = Path::new(r"C:\non_existent_path_for_testing.txt");
    let result = handler.try_extended_path_operation(non_existent, |p| {
        std::fs::metadata(p).map_err(|e| anyhow::anyhow!("Metadata error: {}", e))
    });
    
    // Should fail but with extended path error info
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Both original and extended path operations failed"));
    
    Ok(())
}

#[test]
fn test_component_length_validation() {
    let handler = WindowsPathHandler::new();
    
    // Create path with very long filename
    let long_filename = "x".repeat(300);
    let path_with_long_component = PathBuf::from(format!(r"C:\test\{}.txt", long_filename));
    
    // Should detect need for extended path due to component length
    assert!(handler.is_extended_path_needed(&path_with_long_component));
}
```

## Success Criteria
- [ ] Extended path configuration system works correctly
- [ ] Automatic extended path conversion handles all path types
- [ ] Fallback mechanism tries extended paths when operations fail
- [ ] File system operation wrappers work with extended paths
- [ ] Path length and component validation is comprehensive
- [ ] All tests pass including edge cases
- [ ] No compilation errors or warnings
- [ ] Integration ready for indexing system

## Time Limit
10 minutes

## Notes
- Extended paths bypass the 260-character limit on Windows
- The `\\?\` prefix enables extended-length path support
- UNC paths require special handling with `\\?\UNC\` prefix
- Component lengths also have limits (255 characters per component)
- Fallback mechanism ensures compatibility with older systems
- Configuration allows disabling extended paths if needed