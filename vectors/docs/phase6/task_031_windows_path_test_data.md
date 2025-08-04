# Task 031: Generate Windows Path Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-030. Windows path handling is critical since the system needs to work across platforms, and Windows paths have unique characteristics that can break naive implementations.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_windows_path_tests()` method that creates test files with Windows-specific path patterns, drive letters, UNC paths, reserved names, and path length limitations to ensure cross-platform compatibility.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files with Windows drive letters (C:, D:, etc.)
3. Include UNC paths (\\server\share\path)
4. Test Windows reserved names (CON, PRN, AUX, NUL, etc.)
5. Create paths with Windows path separators (\) vs Unix (/)
6. Test maximum path length scenarios (260 character limit)
7. Include files with Windows-specific characters and escaping

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_windows_path_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Windows drive letter paths
        let drive_paths_content = self.generate_drive_letter_paths()?;
        let mut drive_file = self.create_test_file("windows_drive_paths.rs", &drive_paths_content, TestFileType::WindowsPaths)?;
        drive_file.expected_matches = vec![
            "C:\\".to_string(),
            "D:\\Program Files".to_string(),
            "E:\\Users\\".to_string(),
            "F:\\temp\\file.txt".to_string(),
            "PathBuf::from(\"C:\\\\Windows\")".to_string(),
        ];
        files.push(drive_file);
        
        // UNC paths (Universal Naming Convention)
        let unc_paths_content = self.generate_unc_paths()?;
        let mut unc_file = self.create_test_file("windows_unc_paths.rs", &unc_paths_content, TestFileType::WindowsPaths)?;
        unc_file.expected_matches = vec![
            "\\\\server\\share".to_string(),
            "\\\\localhost\\C$".to_string(),
            "\\\\fileserver\\documents".to_string(),
            "\\\\?\\C:\\very\\long\\path".to_string(),
            "UNC_PATH_MARKER".to_string(),
        ];
        files.push(unc_file);
        
        // Windows reserved names
        let reserved_names_content = self.generate_reserved_names_content()?;
        let mut reserved_file = self.create_test_file("windows_reserved_names.txt", &reserved_names_content, TestFileType::WindowsPaths)?;
        reserved_file.expected_matches = vec![
            "RESERVED_CON".to_string(),
            "RESERVED_PRN".to_string(),
            "RESERVED_AUX".to_string(),
            "RESERVED_NUL".to_string(),
            "RESERVED_COM1".to_string(),
            "RESERVED_LPT1".to_string(),
        ];
        files.push(reserved_file);
        
        // Mixed path separators
        let mixed_separators_content = self.generate_mixed_separator_paths()?;
        let mut mixed_file = self.create_test_file("windows_mixed_separators.rs", &mixed_separators_content, TestFileType::WindowsPaths)?;
        mixed_file.expected_matches = vec![
            "C:\\Windows/System32".to_string(),
            "unix/style/path".to_string(),
            "windows\\style\\path".to_string(),
            "Path::new(\"mixed/path\\\\style\")".to_string(),
            "MIXED_SEPARATOR_TEST".to_string(),
        ];
        files.push(mixed_file);
        
        // Long path names (testing 260 character limit)
        let long_paths_content = self.generate_long_path_content()?;
        let mut long_paths_file = self.create_test_file("windows_long_paths.txt", &long_paths_content, TestFileType::WindowsPaths)?;
        long_paths_file.expected_matches = vec![
            "LONG_PATH_START".to_string(),
            "LONG_PATH_END".to_string(),
            "very_long_directory_name".to_string(),
            "\\\\?\\".to_string(), // Long path prefix
        ];
        files.push(long_paths_file);
        
        // Windows special characters in paths
        let special_chars_content = self.generate_windows_special_chars()?;
        let mut special_file = self.create_test_file("windows_special_chars.rs", &special_chars_content, TestFileType::WindowsPaths)?;
        special_file.expected_matches = vec![
            "file with spaces.txt".to_string(),
            "file[with]brackets.txt".to_string(),
            "file(with)parens.txt".to_string(),
            "file'with'quotes.txt".to_string(),
            "SPECIAL_CHAR_TEST".to_string(),
        ];
        files.push(special_file);
        
        // Windows environment variable paths
        let env_var_paths_content = self.generate_env_var_paths()?;
        let mut env_var_file = self.create_test_file("windows_env_var_paths.bat", &env_var_paths_content, TestFileType::WindowsPaths)?;
        env_var_file.expected_matches = vec![
            "%USERPROFILE%".to_string(),
            "%PROGRAMFILES%".to_string(),
            "%SYSTEMROOT%".to_string(),
            "%TEMP%".to_string(),
            "ENV_VAR_TEST".to_string(),
        ];
        files.push(env_var_file);
        
        Ok(files)
    }
    
    /// Generate content with various Windows drive letter patterns
    fn generate_drive_letter_paths(&self) -> Result<String> {
        Ok(r#"//! Windows drive letter path handling tests
//! This file contains various Windows-style paths for testing

use std::path::{Path, PathBuf};
use std::fs;

/// Test function with common Windows paths
pub fn test_windows_drive_paths() -> Result<(), std::io::Error> {
    // Common Windows system paths
    let system_paths = vec![
        r"C:\Windows\System32",
        r"C:\Program Files\Common Files",
        r"C:\Users\Public\Documents",
        r"D:\Program Files (x86)\Application Data",
        r"E:\Users\Administrator\Desktop",
        r"F:\temp\file.txt",
    ];
    
    for path_str in system_paths {
        let path = PathBuf::from(path_str);
        println!("Testing path: {}", path.display());
        
        // Test path components
        if let Some(parent) = path.parent() {
            println!("Parent: {}", parent.display());
        }
        
        if let Some(filename) = path.file_name() {
            println!("Filename: {:?}", filename);
        }
    }
    
    // Test drive letter extraction
    let drive_paths = [
        ("C:\\Windows", 'C'),
        ("D:\\Program Files", 'D'),
        ("E:\\Users\\", 'E'),
        ("F:\\temp\\file.txt", 'F'),
        ("G:\\very\\deep\\directory\\structure\\file.exe", 'G'),
    ];
    
    for (path_str, expected_drive) in drive_paths {
        let path = Path::new(path_str);
        if let Some(drive) = extract_drive_letter(path) {
            assert_eq!(drive, expected_drive);
            println!("Drive letter {} extracted from {}", drive, path_str);
        }
    }
    
    // Test PathBuf construction with Windows paths
    let constructed_paths = vec![
        PathBuf::from("C:\\Windows"),
        PathBuf::from("D:\\").join("Program Files").join("MyApp"),
        PathBuf::from("E:\\Users").join("username").join("Documents").join("file.txt"),
    ];
    
    for path in constructed_paths {
        println!("Constructed path: {}", path.display());
        assert!(path.is_absolute());
    }
    
    Ok(())
}

/// Extract drive letter from Windows path
fn extract_drive_letter(path: &Path) -> Option<char> {
    let path_str = path.to_string_lossy();
    if path_str.len() >= 2 && path_str.chars().nth(1) == Some(':') {
        path_str.chars().next()
    } else {
        None
    }
}

/// Test relative vs absolute Windows paths
pub fn test_windows_path_types() {
    let absolute_paths = [
        r"C:\absolute\path",
        r"D:\another\absolute\path",
        r"\\server\share\network\path",
    ];
    
    let relative_paths = [
        r"relative\path",
        r".\current\directory",
        r"..\parent\directory",
        r"..\..\grandparent\path",
    ];
    
    for abs_path in absolute_paths {
        let path = Path::new(abs_path);
        assert!(path.is_absolute(), "Path should be absolute: {}", abs_path);
    }
    
    for rel_path in relative_paths {
        let path = Path::new(rel_path);
        assert!(path.is_relative(), "Path should be relative: {}", rel_path);
    }
}

/// Windows path normalization tests
pub fn normalize_windows_paths() {
    let paths_to_normalize = [
        (r"C:\Windows\..\Program Files", r"C:\Program Files"),
        (r"D:\folder\.\subfolder", r"D:\folder\subfolder"),
        (r"E:\path\with\..\double\dots", r"E:\path\double\dots"),
        (r"F:\\\multiple\\\backslashes", r"F:\multiple\backslashes"),
    ];
    
    for (input, expected) in paths_to_normalize {
        let normalized = normalize_path(input);
        println!("Normalized {} -> {}", input, normalized);
        // Note: actual normalization would require more complex logic
    }
}

fn normalize_path(path: &str) -> String {
    // Simplified normalization for demonstration
    path.replace("\\\\", "\\")
        .replace("\\.\\", "\\")
        // More complex .. handling would be needed in real implementation
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_drive_letter_extraction() {
        assert_eq!(extract_drive_letter(Path::new("C:\\Windows")), Some('C'));
        assert_eq!(extract_drive_letter(Path::new("D:\\Program Files")), Some('D'));
        assert_eq!(extract_drive_letter(Path::new("relative\\path")), None);
    }
    
    #[test]
    fn test_path_types() {
        assert!(Path::new("C:\\Windows").is_absolute());
        assert!(Path::new("relative\\path").is_relative());
        assert!(Path::new("\\\\server\\share").is_absolute());
    }
}
"#.to_string())
    }
    
    /// Generate UNC path test content
    fn generate_unc_paths(&self) -> Result<String> {
        Ok(r#"//! UNC (Universal Naming Convention) path tests
//! Testing network paths and long path formats

use std::path::{Path, PathBuf};

/// UNC path handling demonstrations
pub fn test_unc_paths() -> Result<(), Box<dyn std::error::Error>> {
    // Standard UNC paths
    let unc_paths = vec![
        r"\\server\share",
        r"\\fileserver\documents\folder",
        r"\\localhost\C$\Windows",
        r"\\192.168.1.100\shared\files",
        r"\\domain-controller\sysvol\domain.com",
    ];
    
    for unc_path in unc_paths {
        println!("UNC_PATH_MARKER: Processing {}", unc_path);
        let path = Path::new(unc_path);
        
        // Test UNC path properties
        assert!(path.is_absolute());
        assert!(is_unc_path(path));
        
        // Extract server and share
        if let Some((server, share)) = extract_unc_components(path) {
            println!("Server: {}, Share: {}", server, share);
        }
    }
    
    // Long path format (\\?\)
    let long_format_paths = vec![
        r"\\?\C:\very\long\path\that\exceeds\normal\limits",
        r"\\?\UNC\server\share\path",
        r"\\?\Volume{12345678-1234-1234-1234-123456789012}\path",
    ];
    
    for long_path in long_format_paths {
        println!("Long format path: {}", long_path);
        let path = Path::new(long_path);
        assert!(is_long_path_format(path));
    }
    
    // Device paths
    let device_paths = vec![
        r"\\.\COM1",
        r"\\.\LPT1",
        r"\\.\PhysicalDrive0",
        r"\\.\HarddiskVolume1",
    ];
    
    for device_path in device_paths {
        println!("Device path: {}", device_path);
        let path = Path::new(device_path);
        assert!(is_device_path(path));
    }
    
    Ok(())
}

/// Check if path is UNC format
fn is_unc_path(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    path_str.starts_with("\\\\") && !path_str.starts_with("\\\\?") && !path_str.starts_with("\\\\.")
}

/// Check if path uses long path format
fn is_long_path_format(path: &Path) -> bool {
    path.to_string_lossy().starts_with("\\\\?")
}

/// Check if path is a device path
fn is_device_path(path: &Path) -> bool {
    path.to_string_lossy().starts_with("\\\\.")
}

/// Extract server and share from UNC path
fn extract_unc_components(path: &Path) -> Option<(String, String)> {
    let path_str = path.to_string_lossy();
    if !is_unc_path(path) {
        return None;
    }
    
    let without_prefix = &path_str[2..]; // Remove \\
    let parts: Vec<&str> = without_prefix.split('\\').collect();
    
    if parts.len() >= 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

/// UNC path construction utilities
pub fn construct_unc_paths() {
    let servers = ["fileserver", "backupserver", "sharepoint"];
    let shares = ["documents", "backup", "projects", "shared"];
    
    for server in servers {
        for share in shares {
            let unc_path = format!("\\\\{}\\{}", server, share);
            let path = PathBuf::from(&unc_path);
            
            println!("UNC_PATH_MARKER: Constructed {}", path.display());
            
            // Test joining additional path components
            let full_path = path.join("subfolder").join("file.txt");
            println!("Full UNC path: {}", full_path.display());
        }
    }
}

/// Test UNC path comparison and canonicalization
pub fn test_unc_path_operations() {
    let unc_variants = [
        r"\\server\share\path",
        r"\\SERVER\SHARE\PATH",
        r"\\server\share\path\",
        r"\\server\share\path\.",
    ];
    
    for variant in unc_variants {
        let path = Path::new(variant);
        println!("UNC variant: {}", path.display());
        
        // In a real implementation, you'd want to canonicalize these
        // to handle case differences and trailing separators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unc_detection() {
        assert!(is_unc_path(Path::new(r"\\server\share")));
        assert!(is_unc_path(Path::new(r"\\fileserver\documents\folder")));
        assert!(!is_unc_path(Path::new(r"C:\Windows")));
        assert!(!is_unc_path(Path::new(r"\\?\C:\path")));
        assert!(!is_unc_path(Path::new(r"\\.\COM1")));
    }
    
    #[test]
    fn test_unc_component_extraction() {
        let (server, share) = extract_unc_components(Path::new(r"\\fileserver\documents\subfolder")).unwrap();
        assert_eq!(server, "fileserver");
        assert_eq!(share, "documents");
    }
    
    #[test]
    fn test_long_path_detection() {
        assert!(is_long_path_format(Path::new(r"\\?\C:\very\long\path")));
        assert!(is_long_path_format(Path::new(r"\\?\UNC\server\share")));
        assert!(!is_long_path_format(Path::new(r"C:\normal\path")));
    }
    
    #[test]
    fn test_device_path_detection() {
        assert!(is_device_path(Path::new(r"\\.\COM1")));
        assert!(is_device_path(Path::new(r"\\.\PhysicalDrive0")));
        assert!(!is_device_path(Path::new(r"C:\Windows")));
    }
}
"#.to_string())
    }
    
    /// Generate Windows reserved names content
    fn generate_reserved_names_content(&self) -> Result<String> {
        Ok(r#"Windows Reserved Names Test File
==================================

This file contains references to Windows reserved device names that
should be handled specially by file systems and applications.

RESERVED_CON: The CON device (console)
- CON is reserved and cannot be used as a filename
- CON.txt is also reserved
- CON.log would be problematic

RESERVED_PRN: The PRN device (printer)
- PRN represents the default printer
- PRN.doc would cause issues
- Applications should avoid PRN as filename

RESERVED_AUX: The AUX device (auxiliary)
- AUX is an old auxiliary device name
- AUX.dat would be reserved
- Historical compatibility requirement

RESERVED_NUL: The NUL device (null)
- NUL represents the null device
- NUL.txt is reserved
- Similar to /dev/null on Unix systems

Serial Port Names:
RESERVED_COM1: COM1 (serial port 1)
RESERVED_COM2: COM2 (serial port 2)
RESERVED_COM3: COM3 (serial port 3)
COM4, COM5, COM6, COM7, COM8, COM9 are also reserved

Parallel Port Names:
RESERVED_LPT1: LPT1 (parallel port 1)
RESERVED_LPT2: LPT2 (parallel port 2)
RESERVED_LPT3: LPT3 (parallel port 3)
LPT4, LPT5, LPT6, LPT7, LPT8, LPT9 are also reserved

Additional reserved patterns:
- Names with trailing periods: "filename."
- Names with trailing spaces: "filename "
- Case variations: con, CON, Con, etc.

Test cases that should be handled:
1. Direct reserved names: CON, PRN, AUX, NUL
2. Reserved names with extensions: CON.txt, PRN.log
3. Reserved names in different cases: con, Con, CON
4. Serial ports: COM1-COM9
5. Parallel ports: LPT1-LPT9

File system behavior:
- Windows will treat these as device references
- Cannot create files with these exact names
- Case-insensitive matching applies
- Extensions don't change reserved status

Search patterns should account for:
- Reserved name detection in file paths
- Case-insensitive matching
- Extension handling with reserved names
- Device path references vs regular files

End of reserved names test content.
"#.to_string())
    }
    
    /// Generate mixed path separator test content
    fn generate_mixed_separator_paths(&self) -> Result<String> {
        Ok(r#"//! Mixed path separator handling tests
//! Testing both forward slashes and backslashes in paths

use std::path::{Path, PathBuf, MAIN_SEPARATOR};

/// Test mixed separator handling
pub fn test_mixed_separators() -> Result<(), std::io::Error> {
    // MIXED_SEPARATOR_TEST: Various separator combinations
    let mixed_paths = vec![
        r"C:\Windows/System32",           // Mixed in same path
        r"D:\Program Files/Common Files", // Windows drive with Unix separators
        r"E:/Users\Administrator",        // Unix start with Windows separators
        r"unix/style/path",               // Pure Unix style
        r"windows\style\path",            // Pure Windows style
        r"C:\path/with\mixed/separators", // Multiple mixing
    ];
    
    for path_str in mixed_paths {
        println!("Testing mixed separator path: {}", path_str);
        let path = Path::new(path_str);
        
        // Test path components with mixed separators
        let components: Vec<_> = path.components().collect();
        println!("Components: {:?}", components);
        
        // Test path reconstruction
        let reconstructed = reconstruct_path(&components);
        println!("Reconstructed: {}", reconstructed);
    }
    
    // Test Path::new() with different separator styles
    let separator_tests = [
        ("forward/slash/path", "/"),
        (r"back\slash\path", "\\"),
        (r"mixed/and\combined", "mixed"),
    ];
    
    for (path_str, separator_type) in separator_tests {
        let path = Path::new(path_str);
        println!("Path with {} separators: {}", separator_type, path.display());
        
        // Convert to native separators
        let native_path = path.to_string_lossy().replace('/', &MAIN_SEPARATOR.to_string());
        println!("Native separators: {}", native_path);
    }
    
    // Test PathBuf construction with mixed separators
    let mut path_buf = PathBuf::new();
    path_buf.push("C:");
    path_buf.push("Users");      // Will use native separator
    path_buf.push("Documents");
    
    println!("PathBuf with native separators: {}", path_buf.display());
    
    // Test joining paths with different separator styles
    let base_paths = ["C:\\Windows", "D:/Program Files", "unix/style"];
    let extensions = ["System32", "subfolder/nested", "file.txt"];
    
    for base in base_paths {
        for ext in extensions {
            let joined = Path::new(base).join(ext);
            println!("Joined {} + {} = {}", base, ext, joined.display());
        }
    }
    
    Ok(())
}

fn reconstruct_path(components: &[std::path::Component]) -> String {
    components.iter()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join(&MAIN_SEPARATOR.to_string())
}

/// Normalize path separators for cross-platform compatibility
pub fn normalize_separators(path: &str) -> String {
    if cfg!(windows) {
        path.replace('/', "\\")
    } else {
        path.replace('\\', "/")
    }
}

/// Test separator normalization
pub fn test_separator_normalization() {
    let test_paths = [
        "C:\\Windows/System32",
        "unix/path\\with\\mixed",
        "/absolute/unix/path",
        "C:\\absolute\\windows\\path",
        "relative/unix/path",
        "relative\\windows\\path",
    ];
    
    for path in test_paths {
        let normalized = normalize_separators(path);
        println!("Original: {} -> Normalized: {}", path, normalized);
        
        // Test that Path::new handles both formats
        let path_obj = Path::new(path);
        let path_normalized = Path::new(&normalized);
        
        println!("Path object display: {}", path_obj.display());
        println!("Normalized display: {}", path_normalized.display());
    }
}

/// Cross-platform path utilities
pub struct PathUtils;

impl PathUtils {
    /// Convert path to Unix-style separators
    pub fn to_unix_separators(path: &str) -> String {
        path.replace('\\', "/")
    }
    
    /// Convert path to Windows-style separators
    pub fn to_windows_separators(path: &str) -> String {
        path.replace('/', "\\")
    }
    
    /// Get the appropriate separator for current platform
    pub fn native_separator() -> char {
        MAIN_SEPARATOR
    }
    
    /// Check if path uses mixed separators
    pub fn has_mixed_separators(path: &str) -> bool {
        path.contains('/') && path.contains('\\')
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mixed_separator_detection() {
        assert!(PathUtils::has_mixed_separators("C:\\Windows/System32"));
        assert!(PathUtils::has_mixed_separators("unix/path\\windows"));
        assert!(!PathUtils::has_mixed_separators("C:\\Windows\\System32"));
        assert!(!PathUtils::has_mixed_separators("unix/path/only"));
    }
    
    #[test]
    fn test_separator_conversion() {
        assert_eq!(PathUtils::to_unix_separators("C:\\Windows\\System32"), "C:/Windows/System32");
        assert_eq!(PathUtils::to_windows_separators("unix/path/style"), "unix\\path\\style");
    }
    
    #[test]
    #[cfg(windows)]
    fn test_windows_native_separator() {
        assert_eq!(PathUtils::native_separator(), '\\');
    }
    
    #[test]
    #[cfg(unix)]
    fn test_unix_native_separator() {
        assert_eq!(PathUtils::native_separator(), '/');
    }
}
"#.to_string())
    }
    
    /// Generate long path test content
    fn generate_long_path_content(&self) -> Result<String> {
        Ok(r#"Windows Long Path Testing
========================

LONG_PATH_START: Testing Windows path length limitations

Windows traditionally has a 260-character path limit (MAX_PATH), but
newer versions support longer paths with specific prefixes.

Standard MAX_PATH limit: 260 characters including null terminator
This means effectively 259 usable characters for the path string.

Long path examples:

1. Standard long path that approaches the limit:
C:\very_long_directory_name_that_approaches_the_traditional_windows_path_limit_of_260_characters\and_this_continues_with_more_subdirectories\and_even_more_nested_folders\with_very_descriptive_names\file.txt

2. Path using \\?\ prefix to exceed normal limits:
\\?\C:\this_path_uses_the_long_path_prefix_which_allows_it_to_exceed_the_traditional_260_character_limit_significantly\very_long_directory_name_with_many_subdirectories\another_level_of_nesting\and_another_level\with_a_very_long_filename_that_would_normally_cause_problems.txt

3. UNC path with long path prefix:
\\?\UNC\server\share\extremely_long_network_path_that_exceeds_normal_limitations\with_many_nested_subdirectories\and_very_long_filenames\that_would_cause_issues_without_proper_handling\final_file.log

4. Testing path components individually:

Directory names that are very long:
- very_long_directory_name_that_tests_individual_component_limits_in_windows_file_systems
- another_extremely_long_directory_name_for_comprehensive_testing_of_path_handling
- final_very_long_directory_name_to_ensure_robust_path_processing

Filename testing:
- very_long_filename_that_tests_the_limits_of_what_windows_can_handle_without_issues.txt
- another_long_filename_for_comprehensive_testing_of_file_naming_conventions.log
- final_extremely_long_filename_to_test_edge_cases_in_path_processing.data

Path construction patterns:

Base path: C:\Program Files\Application Name With Spaces\
Extended: C:\Program Files\Application Name With Spaces\very_long_subdirectory_name\
Further:  C:\Program Files\Application Name With Spaces\very_long_subdirectory_name\nested_folder\
Final:    C:\Program Files\Application Name With Spaces\very_long_subdirectory_name\nested_folder\very_long_filename.extension

Testing scenarios:
1. Paths at exactly 259 characters
2. Paths at 260 characters (should fail without \\?\)
3. Paths over 260 characters with \\?\ prefix
4. Paths with maximum individual component lengths
5. Network paths with long components

Error conditions to test:
- Path too long without \\?\ prefix
- Individual component exceeds limits
- Total path length exceeds system maximum
- Invalid characters in long paths
- Long path creation vs. long path access

Common long path issues:
- File operations failing silently
- Truncated paths in APIs
- Backup software skipping files
- Archive extraction failures
- Network access problems

Best practices for long path handling:
1. Use \\?\ prefix for paths over 260 characters
2. Check path length before operations
3. Provide meaningful error messages
4. Test with realistic long path scenarios
5. Consider relative vs absolute path alternatives

LONG_PATH_END: End of long path testing content
"#.to_string())
    }
    
    /// Generate Windows special characters test content
    fn generate_windows_special_chars(&self) -> Result<String> {
        Ok(r#"//! Windows special character handling in paths
//! Testing various characters that have special meaning in Windows

use std::path::{Path, PathBuf};

/// Test Windows path special characters
pub fn test_windows_special_characters() -> Result<(), Box<dyn std::error::Error>> {
    // SPECIAL_CHAR_TEST: Files with various special characters
    
    // Spaces in filenames and paths
    let space_paths = vec![
        r"C:\Program Files\Application Name",
        r"C:\Users\User Name\My Documents",
        r"file with spaces.txt",
        r"directory with spaces\file.txt",
    ];
    
    for path_str in space_paths {
        let path = Path::new(path_str);
        println!("Space path: {}", path.display());
        test_path_validity(path);
    }
    
    // Brackets and parentheses
    let bracket_paths = vec![
        r"file[with]brackets.txt",
        r"file(with)parens.txt",
        r"file{with}braces.txt",
        r"C:\Program Files (x86)\Application",
        r"directory[1]\file.txt",
    ];
    
    for path_str in bracket_paths {
        let path = Path::new(path_str);
        println!("Bracket path: {}", path.display());
        test_path_validity(path);
    }
    
    // Quotes and apostrophes
    let quote_paths = vec![
        r#"file"with"quotes.txt"#,
        r"file'with'quotes.txt",
        r"file`with`backticks.txt",
    ];
    
    for path_str in quote_paths {
        let path = Path::new(path_str);
        println!("Quote path: {}", path.display());
        test_path_validity(path);
    }
    
    // Unicode characters in paths
    let unicode_paths = vec![
        "файл.txt",                    // Cyrillic
        "文件.txt",                    // Chinese
        "ファイル.txt",                // Japanese
        "tëst_fîlé.txt",              // Accented characters
        "emoji_file_🚀_test.txt",     // Emoji (may not be supported)
    ];
    
    for path_str in unicode_paths {
        let path = Path::new(path_str);
        println!("Unicode path: {}", path.display());
        test_path_validity(path);
    }
    
    // Special symbols that are usually allowed
    let symbol_paths = vec![
        "file@email.txt",
        "file#hash.txt",
        "file$dollar.txt",
        "file%percent.txt",
        "file&ampersand.txt",
        "file+plus.txt",
        "file=equals.txt",
        "file~tilde.txt",
        "file-dash.txt",
        "file_underscore.txt",
    ];
    
    for path_str in symbol_paths {
        let path = Path::new(path_str);
        println!("Symbol path: {}", path.display());
        test_path_validity(path);
    }
    
    Ok(())
}

/// Characters that are NOT allowed in Windows filenames
pub fn test_invalid_windows_characters() {
    let invalid_chars = vec![
        '<', '>', ':', '"', '|', '?', '*',
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x09', '\x0A', '\x0B', '\x0C', '\x0D', '\x0E', '\x0F',
        '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
        '\x18', '\x19', '\x1A', '\x1B', '\x1C', '\x1D', '\x1E', '\x1F',
    ];
    
    println!("SPECIAL_CHAR_TEST: Invalid Windows filename characters:");
    for ch in invalid_chars {
        if ch.is_control() {
            println!("Control character: U+{:04X}", ch as u32);
        } else {
            println!("Invalid character: '{}'", ch);
        }
    }
}

/// Test path validity helper
fn test_path_validity(path: &Path) {
    // Basic validity checks
    if path.to_string_lossy().is_empty() {
        println!("  Empty path");
        return;
    }
    
    // Check for invalid Windows characters
    let path_str = path.to_string_lossy();
    let invalid_chars = ['<', '>', ':', '"', '|', '?', '*'];
    
    for &invalid_char in &invalid_chars {
        if path_str.contains(invalid_char) {
            println!("  Contains invalid character: '{}'", invalid_char);
        }
    }
    
    // Check for reserved names
    if let Some(file_stem) = path.file_stem() {
        let stem_str = file_stem.to_string_lossy().to_uppercase();
        let reserved_names = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ];
        
        if reserved_names.contains(&stem_str.as_str()) {
            println!("  Uses reserved name: {}", stem_str);
        }
    }
    
    // Check path length
    if path_str.len() > 260 {
        println!("  Path exceeds 260 characters: {} chars", path_str.len());
    }
    
    println!("  Path appears valid for Windows");
}

/// Escape special characters for command line usage
pub fn escape_for_command_line(path: &str) -> String {
    if path.contains(' ') || path.contains('&') || path.contains('^') {
        format!("\"{}\"", path)
    } else {
        path.to_string()
    }
}

/// Test command line escaping
pub fn test_command_line_escaping() {
    let paths_to_escape = vec![
        r"C:\Program Files\Application",
        r"file with spaces.txt",
        r"path&with&ampersands",
        r"path^with^carets",
        r"normal_path.txt",
    ];
    
    println!("SPECIAL_CHAR_TEST: Command line escaping:");
    for path in paths_to_escape {
        let escaped = escape_for_command_line(path);
        println!("Original: {} -> Escaped: {}", path, escaped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_path_with_spaces() {
        let path = Path::new("file with spaces.txt");
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), "file with spaces.txt");
    }
    
    #[test]
    fn test_command_line_escaping() {
        assert_eq!(escape_for_command_line("normal.txt"), "normal.txt");
        assert_eq!(escape_for_command_line("file with spaces.txt"), "\"file with spaces.txt\"");
        assert_eq!(escape_for_command_line("path&with&special"), "\"path&with&special\"");
    }
    
    #[test]
    fn test_unicode_paths() {
        let unicode_path = Path::new("tëst_fîlé.txt");
        assert!(unicode_path.file_name().is_some());
    }
}
"#.to_string())
    }
    
    /// Generate Windows environment variable path content
    fn generate_env_var_paths(&self) -> Result<String> {
        Ok(r#"@echo off
REM Windows Environment Variable Path Tests
REM This batch file demonstrates various Windows environment variables in paths

echo ENV_VAR_TEST: Testing Windows environment variable paths

REM Common user-related environment variables
echo User Profile: %USERPROFILE%
echo User Documents: %USERPROFILE%\Documents
echo User Desktop: %USERPROFILE%\Desktop
echo User AppData: %APPDATA%
echo Local AppData: %LOCALAPPDATA%
echo User Temp: %TEMP%

REM System-wide paths
echo Windows Directory: %SYSTEMROOT%
echo Windows System32: %SYSTEMROOT%\System32
echo Program Files: %PROGRAMFILES%
echo Program Files x86: %PROGRAMFILES(X86)%
echo Common Files: %COMMONPROGRAMFILES%
echo Common Files x86: %COMMONPROGRAMFILES(X86)%

REM Development and tools
echo System Path: %PATH%
echo Python Path: %PYTHONPATH%
echo Java Home: %JAVA_HOME%
echo Node Path: %NODE_PATH%

REM Testing path expansion
set TEST_VAR=C:\TestDirectory
echo Custom variable: %TEST_VAR%
echo Nested path: %TEST_VAR%\SubDirectory\File.txt

REM Environment variable in different contexts
dir "%USERPROFILE%\Documents" /B
copy nul "%TEMP%\test_file.txt"
del "%TEMP%\test_file.txt"

REM Testing with quotes
echo "Quoted path: %PROGRAMFILES%\Common Files"
echo 'Single quoted: %SYSTEMROOT%\System32'

REM Long environment variable paths
echo Long path example: %USERPROFILE%\Documents\Projects\LongProjectName\SubDirectory\AnotherSubDirectory\VeryLongFileName.extension

REM Testing case sensitivity (Windows is case-insensitive for env vars)
echo Lower case: %userprofile%
echo Mixed case: %UserProfile%
echo Upper case: %USERPROFILE%

REM Network paths with environment variables
echo Network path: \\%COMPUTERNAME%\C$\Users\%USERNAME%

REM Registry-based environment variables
echo All Users Profile: %ALLUSERSPROFILE%
echo Public Profile: %PUBLIC%
echo Program Data: %PROGRAMDATA%

REM Hardware and system info
echo Computer Name: %COMPUTERNAME%
echo Username: %USERNAME%
echo User Domain: %USERDOMAIN%
echo Processor: %PROCESSOR_ARCHITECTURE%
echo OS: %OS%

REM Date and time variables
echo Date: %DATE%
echo Time: %TIME%

REM Command line and batch specific
echo Command Line: %CMDCMDLINE%
echo Batch File: %~f0
echo Batch Directory: %~dp0

REM Testing undefined variables
echo Undefined variable: %UNDEFINED_VAR%
echo With default: %UNDEFINED_VAR:~0,10%

REM Combining multiple environment variables
echo Combined: %SYSTEMROOT%\%USERNAME%\%COMPUTERNAME%

REM Special characters in environment variables
set SPECIAL_VAR=Path with spaces & special chars
echo Special variable: %SPECIAL_VAR%

REM Testing environment variable substitution in paths
for %%i in ("%USERPROFILE%\*.*") do echo File: %%i

echo ENV_VAR_TEST: End of environment variable testing
"#.to_string())
    }
}
```

## Success Criteria
- Method generates 7 Windows-specific path test files
- Files cover drive letters, UNC paths, reserved names, and mixed separators
- Long path scenarios (260+ character limit) are tested properly
- Windows special characters and escaping patterns are included
- Environment variable path patterns are demonstrated
- All files include appropriate expected_matches for search validation
- Cross-platform compatibility considerations are addressed

## Time Limit
10 minutes maximum