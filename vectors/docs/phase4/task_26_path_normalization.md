# Task 26: Implement Windows Path Normalization

## Context
You are implementing Phase 4 of a vector indexing system. The basic Windows path handler was created in the previous task. Now you need to enhance the path normalization functionality to handle complex Windows path scenarios.

## Current State
- `src/windows.rs` exists with basic `WindowsPathHandler` struct
- Basic path normalization is implemented
- Filename validation is working

## Task Objective
Enhance the path normalization functionality to handle Windows UNC paths, network paths, junction points, and other complex Windows path scenarios.

## Implementation Requirements

### 1. Add enhanced path normalization
Replace the existing `normalize_path()` method with this enhanced version:
```rust
pub fn normalize_path(path: &Path) -> Result<PathBuf> {
    // Handle different types of Windows paths
    let path_str = path.to_string_lossy();
    
    // Handle UNC paths (\\server\share)
    if path_str.starts_with(r"\\\\") && !path_str.starts_with(r"\\?\\UNC\\") {
        return Self::normalize_unc_path(path);
    }
    
    // Handle device paths (\\?\\ or \\.\\ )
    if path_str.starts_with(r"\\?\\") || path_str.starts_with(r"\\.\\")
    {
        return Self::normalize_device_path(path);
    }
    
    // Handle regular paths
    Self::normalize_regular_path(path)
}

fn normalize_regular_path(path: &Path) -> Result<PathBuf> {
    // Try canonicalization first
    match path.canonicalize() {
        Ok(canonical) => {
            #[cfg(windows)]
            {
                Self::handle_windows_extended_path(canonical)
            }
            #[cfg(not(windows))]
            {
                Ok(canonical)
            }
        }
        Err(_) => {
            // If canonicalization fails, clean manually
            let cleaned = Self::clean_path_components(path);
            
            // Try to make it absolute if it's relative
            if cleaned.is_relative() {
                match std::env::current_dir() {
                    Ok(current) => Ok(current.join(cleaned)),
                    Err(_) => Ok(cleaned),
                }
            } else {
                Ok(cleaned)
            }
        }
    }
}

fn normalize_unc_path(path: &Path) -> Result<PathBuf> {
    let path_str = path.to_string_lossy();
    
    // UNC paths: \\server\share\path
    if path_str.len() > 2 && path_str.starts_with(r"\\\\") {
        // Already a UNC path, just clean it
        Ok(Self::clean_path_components(path))
    } else {
        Err(anyhow::anyhow!("Invalid UNC path: {}", path_str))
    }
}

fn normalize_device_path(path: &Path) -> Result<PathBuf> {
    let path_str = path.to_string_lossy();
    
    if path_str.starts_with(r"\\?\\") {
        // Extended-length path
        let without_prefix = &path_str[4..];
        
        if without_prefix.starts_with(r"UNC\\") {
            // \\?\UNC\server\share -> \\server\share
            let unc_part = &without_prefix[4..];
            Ok(PathBuf::from(format!(r"\\\\", unc_part)))
        } else {
            // \\?\C:\path -> C:\path
            Ok(PathBuf::from(without_prefix))
        }
    } else if path_str.starts_with(r"\\.\\")
    {
        // Device path - preserve as-is
        Ok(path.to_path_buf())
    } else {
        Err(anyhow::anyhow!("Invalid device path: {}", path_str))
    }
}
```

### 2. Add path type detection
Add these helper methods to identify different path types:
```rust
pub fn get_path_type(path: &Path) -> WindowsPathType {
    let path_str = path.to_string_lossy();
    
    if path_str.starts_with(r"\\?\\") {
        if path_str[4..].starts_with(r"UNC\\") {
            WindowsPathType::ExtendedUNC
        } else {
            WindowsPathType::ExtendedLength
        }
    } else if path_str.starts_with(r"\\.\\")
    {
        WindowsPathType::Device
    } else if path_str.starts_with(r"\\\\") {
        WindowsPathType::UNC
    } else if path_str.len() >= 3 && path_str.chars().nth(1) == Some(':') {
        WindowsPathType::DriveAbsolute
    } else if path_str.starts_with('\\') || path_str.starts_with('/') {
        WindowsPathType::RootRelative
    } else {
        WindowsPathType::Relative
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WindowsPathType {
    Relative,           // path/to/file
    DriveAbsolute,      // C:\path\to\file
    RootRelative,       // \path\to\file
    UNC,               // \\server\share\path
    ExtendedLength,     // \\?\C:\very\long\path
    ExtendedUNC,       // \\?\UNC\server\share\path
    Device,            // \\.\device
}
```

### 3. Add path validation with type checking
Add this enhanced validation method:
```rust
pub fn validate_path_with_type(path: &Path) -> Result<WindowsPathType> {
    let path_type = Self::get_path_type(path);
    
    // Validate based on path type
    match path_type {
        WindowsPathType::UNC => {
            let path_str = path.to_string_lossy();
            let parts: Vec<&str> = path_str.trim_start_matches(r"\\\\").split('\\').collect();
            
            if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
                return Err(anyhow::anyhow!("Invalid UNC path format: {}", path_str));
            }
        }
        WindowsPathType::DriveAbsolute => {
            let path_str = path.to_string_lossy();
            if path_str.len() < 3 {
                return Err(anyhow::anyhow!("Invalid drive path format: {}", path_str));
            }
            
            let drive_char = path_str.chars().nth(0).unwrap();
            if !drive_char.is_ascii_alphabetic() {
                return Err(anyhow::anyhow!("Invalid drive letter: {}", drive_char));
            }
        }
        WindowsPathType::ExtendedLength | WindowsPathType::ExtendedUNC => {
            // Extended paths can be very long, so just check basic format
            if !Self::is_valid_path_length(path) {
                return Err(anyhow::anyhow!("Path exceeds maximum length"));
            }
        }
        _ => {
            // Basic validation for other types
            Self::validate_windows_path(path)?;
        }
    }
    
    Ok(path_type)
}
```

### 4. Add path conversion utilities
Add these utility methods:
```rust
pub fn to_extended_path(path: &Path) -> Result<PathBuf> {
    let path_type = Self::get_path_type(path);
    
    match path_type {
        WindowsPathType::DriveAbsolute => {
            let path_str = path.to_string_lossy();
            Ok(PathBuf::from(format!(r"\\?\\{}", path_str)))
        }
        WindowsPathType::UNC => {
            let path_str = path.to_string_lossy();
            let without_unc = path_str.trim_start_matches(r"\\\\");
            Ok(PathBuf::from(format!(r"\\?\\UNC\\{}", without_unc)))
        }
        WindowsPathType::ExtendedLength | WindowsPathType::ExtendedUNC => {
            // Already extended
            Ok(path.to_path_buf())
        }
        _ => {
            // Convert relative to absolute first
            let absolute = Self::normalize_path(path)?;
            Self::to_extended_path(&absolute)
        }
    }
}

pub fn from_extended_path(path: &Path) -> Result<PathBuf> {
    let path_str = path.to_string_lossy();
    
    if path_str.starts_with(r"\\?\\UNC\\") {
        // \\?\UNC\server\share -> \\server\share
        let unc_part = &path_str[8..];
        Ok(PathBuf::from(format!(r"\\\\", unc_part)))
    } else if path_str.starts_with(r"\\?\\") {
        // \\?\C:\path -> C:\path
        Ok(PathBuf::from(&path_str[4..]))
    } else {
        // Not an extended path
        Ok(path.to_path_buf())
    }
}
```

### 5. Add comprehensive path normalization tests
Add these tests to the test module:
```rust
#[test]
fn test_path_type_detection() {
    assert_eq!(WindowsPathHandler::get_path_type(Path::new("file.txt")), 
               WindowsPathType::Relative);
    assert_eq!(WindowsPathHandler::get_path_type(Path::new(r"C:\\file.txt")), 
               WindowsPathType::DriveAbsolute);
    assert_eq!(WindowsPathHandler::get_path_type(Path::new(r"\\\\server\\share")), 
               WindowsPathType::UNC);
    assert_eq!(WindowsPathHandler::get_path_type(Path::new(r"\\?\\C:\\file.txt")), 
               WindowsPathType::ExtendedLength);
    assert_eq!(WindowsPathHandler::get_path_type(Path::new(r"\\?\\UNC\\server\\share")), 
               WindowsPathType::ExtendedUNC);
    assert_eq!(WindowsPathHandler::get_path_type(Path::new(r"\\\\.\\device")), 
               WindowsPathType::Device);
}

#[test]
fn test_unc_path_normalization() -> Result<()> {
    let unc_path = Path::new(r"\\\\server\\share\\file.txt");
    let normalized = WindowsPathHandler::normalize_path(unc_path)?;
    
    // Should preserve UNC format
    let normalized_str = normalized.to_string_lossy();
    assert!(normalized_str.starts_with(r"\\\\"));
    
    Ok(())
}

#[test]
fn test_extended_path_conversion() -> Result<()> {
    // Test drive path to extended
    let drive_path = Path::new(r"C:\\test\\file.txt");
    let extended = WindowsPathHandler::to_extended_path(drive_path)?;
    assert!(extended.to_string_lossy().starts_with(r"\\?\\"));
    
    // Test back conversion
    let back = WindowsPathHandler::from_extended_path(&extended)?;
    assert_eq!(back, drive_path);
    
    Ok(())
}

#[test]
fn test_path_validation_with_type() -> Result<()> {
    // Valid paths
    assert!(WindowsPathHandler::validate_path_with_type(Path::new(r"C:\\test.txt")).is_ok());
    assert!(WindowsPathHandler::validate_path_with_type(Path::new(r"\\\\server\\share")).is_ok());
    
    // Invalid paths
    assert!(WindowsPathHandler::validate_path_with_type(Path::new(r"\\\\invalid")).is_err());
    assert!(WindowsPathHandler::validate_path_with_type(Path::new(r"1:\\invalid")).is_err());
    
    Ok(())
}

#[test]
fn test_device_path_handling() -> Result<()> {
    let device_path = Path::new(r"\\\\.\\device");
    let path_type = WindowsPathHandler::get_path_type(device_path);
    assert_eq!(path_type, WindowsPathType::Device);
    
    // Device paths should be preserved as-is
    let normalized = WindowsPathHandler::normalize_path(device_path)?;
    assert_eq!(normalized, device_path);
    
    Ok(())
}

#[test]
fn test_relative_path_handling() -> Result<()> {
    let relative_path = Path::new("test/file.txt");
    let normalized = WindowsPathHandler::normalize_path(relative_path)?;
    
    // Should be converted to absolute
    assert!(normalized.is_absolute());
    
    Ok(())
}
```

## Success Criteria
- [ ] Enhanced path normalization handles all Windows path types
- [ ] Path type detection works correctly for all formats
- [ ] UNC path normalization preserves correct format
- [ ] Extended path conversion works bidirectionally
- [ ] Device paths are handled appropriately
- [ ] Path validation catches type-specific errors
- [ ] All tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Windows supports multiple path formats with different rules
- Extended paths allow longer filenames and paths
- UNC paths provide network resource access
- Device paths access special system devices
- Proper normalization ensures consistent path handling across the system