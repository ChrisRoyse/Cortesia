# Task 32: Implement Cross-Platform Testing

## Context
You are implementing Phase 4 of a vector indexing system. Windows-specific tests have been setup. Now you need to create comprehensive cross-platform testing that ensures the Windows optimizations work correctly on all platforms, handles platform differences gracefully, and maintains compatibility across operating systems.

## Current State
- `src/windows.rs` has comprehensive Windows-specific test suite
- Windows functionality is fully implemented with system detection
- Need cross-platform compatibility testing and platform abstraction
- Must ensure graceful degradation on non-Windows systems

## Task Objective
Implement comprehensive cross-platform testing with platform abstraction, feature detection, graceful degradation, and unified testing across Windows, macOS, and Linux systems.

## Implementation Requirements

### 1. Create cross-platform abstraction layer
Add this platform abstraction to `src/windows.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Platform {
    Windows,
    MacOS,
    Linux,
    Unix,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    pub platform: Platform,
    pub case_sensitive_filesystem: bool,
    pub unicode_normalization: UnicodeNormalizationForm,
    pub max_path_length: Option<usize>,
    pub max_filename_length: Option<usize>,
    pub reserved_characters: Vec<char>,
    pub path_separator: char,
    pub supports_long_paths: bool,
    pub supports_symlinks: bool,
    pub supports_hard_links: bool,
    pub filesystem_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnicodeNormalizationForm {
    NFC,    // Canonical Composition
    NFD,    // Canonical Decomposition  
    NFKC,   // Compatibility Composition
    NFKD,   // Compatibility Decomposition
    None,   // No normalization required
}

impl PlatformCapabilities {
    pub fn detect_current_platform() -> Self {
        let platform = Self::detect_platform();
        
        match platform {
            Platform::Windows => Self::windows_capabilities(),
            Platform::MacOS => Self::macos_capabilities(),
            Platform::Linux => Self::linux_capabilities(),
            Platform::Unix => Self::unix_capabilities(),
            Platform::Other(_) => Self::generic_capabilities(),
        }
    }
    
    fn detect_platform() -> Platform {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOS
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else if cfg!(unix) {
            Platform::Unix
        } else {
            Platform::Other(std::env::consts::OS.to_string())
        }
    }
    
    fn windows_capabilities() -> Self {
        Self {
            platform: Platform::Windows,
            case_sensitive_filesystem: false,
            unicode_normalization: UnicodeNormalizationForm::NFC,
            max_path_length: Some(32767), // With extended paths
            max_filename_length: Some(255),
            reserved_characters: vec!['<', '>', ':', '"', '|', '?', '*'],
            path_separator: '\\',
            supports_long_paths: true, // With proper configuration
            supports_symlinks: true,   // Windows 10+ with developer mode
            supports_hard_links: true,
            filesystem_type: "NTFS".to_string(),
        }
    }
    
    fn macos_capabilities() -> Self {
        Self {
            platform: Platform::MacOS,
            case_sensitive_filesystem: false, // Default, can be case-sensitive
            unicode_normalization: UnicodeNormalizationForm::NFD,
            max_path_length: Some(1024),
            max_filename_length: Some(255),
            reserved_characters: vec![':'], // Colon is reserved
            path_separator: '/',
            supports_long_paths: true,
            supports_symlinks: true,
            supports_hard_links: true,
            filesystem_type: "APFS".to_string(),
        }
    }
    
    fn linux_capabilities() -> Self {
        Self {
            platform: Platform::Linux,
            case_sensitive_filesystem: true,
            unicode_normalization: UnicodeNormalizationForm::NFC,
            max_path_length: Some(4096),
            max_filename_length: Some(255),
            reserved_characters: vec!['/'], // Only forward slash
            path_separator: '/',
            supports_long_paths: true,
            supports_symlinks: true,
            supports_hard_links: true,
            filesystem_type: "ext4".to_string(),
        }
    }
    
    fn unix_capabilities() -> Self {
        Self {
            platform: Platform::Unix,
            case_sensitive_filesystem: true,
            unicode_normalization: UnicodeNormalizationForm::NFC,
            max_path_length: Some(1024),
            max_filename_length: Some(255),
            reserved_characters: vec!['/'],
            path_separator: '/',
            supports_long_paths: false,
            supports_symlinks: true,
            supports_hard_links: true,
            filesystem_type: "Unknown".to_string(),
        }
    }
    
    fn generic_capabilities() -> Self {
        Self {
            platform: Platform::Other("Unknown".to_string()),
            case_sensitive_filesystem: true, // Conservative assumption
            unicode_normalization: UnicodeNormalizationForm::NFC,
            max_path_length: Some(1024),
            max_filename_length: Some(255),
            reserved_characters: vec!['/', '\\'], // Both separators
            path_separator: std::path::MAIN_SEPARATOR,
            supports_long_paths: false,
            supports_symlinks: false, // Conservative
            supports_hard_links: false, // Conservative
            filesystem_type: "Unknown".to_string(),
        }
    }
    
    pub fn is_filename_valid_for_platform(&self, filename: &str) -> bool {
        // Check for reserved characters
        for ch in filename.chars() {
            if self.reserved_characters.contains(&ch) {
                return false;
            }
        }
        
        // Check length limits
        if let Some(max_len) = self.max_filename_length {
            if filename.len() > max_len {
                return false;
            }
        }
        
        // Platform-specific checks
        match self.platform {
            Platform::Windows => {
                // Use existing Windows validation
                WindowsPathHandler::new().validate_filename_quick(filename)
            }
            Platform::MacOS => {
                // macOS specific checks
                !filename.contains(':') && !filename.starts_with('.')
            }
            Platform::Linux | Platform::Unix => {
                // Unix-like systems
                !filename.contains('/') && filename != "." && filename != ".."
            }
            _ => {
                // Generic validation
                !filename.is_empty() && !filename.contains('\0')
            }
        }
    }
    
    pub fn normalize_path_for_platform(&self, path: &str) -> String {
        let mut normalized = path.to_string();
        
        // Normalize path separators
        if self.platform == Platform::Windows {
            normalized = normalized.replace('/', "\\");
        } else {
            normalized = normalized.replace('\\', "/");
        }
        
        // Apply Unicode normalization
        match self.unicode_normalization {
            UnicodeNormalizationForm::NFC => normalized.nfc().collect(),
            UnicodeNormalizationForm::NFD => normalized.nfd().collect(),
            UnicodeNormalizationForm::NFKC => normalized.nfkc().collect(),
            UnicodeNormalizationForm::NFKD => normalized.nfkd().collect(),
            UnicodeNormalizationForm::None => normalized,
        }
    }
}
```

### 2. Add cross-platform path handler
Add this unified path handler:
```rust
pub struct CrossPlatformPathHandler {
    windows_handler: WindowsPathHandler,
    capabilities: PlatformCapabilities,
}

impl CrossPlatformPathHandler {
    pub fn new() -> Self {
        Self {
            windows_handler: WindowsPathHandler::new(),
            capabilities: PlatformCapabilities::detect_current_platform(),
        }
    }
    
    pub fn with_capabilities(capabilities: PlatformCapabilities) -> Self {
        Self {
            windows_handler: WindowsPathHandler::new(),
            capabilities,
        }
    }
    
    pub fn validate_path_cross_platform(&self, path: &Path) -> CrossPlatformValidationResult {
        let mut result = CrossPlatformValidationResult::new();
        let path_str = path.to_string_lossy();
        
        // Always run Windows validation for comprehensive checking
        if self.capabilities.platform == Platform::Windows {
            let windows_result = self.windows_handler.validate_windows_path(path);
            result.windows_compatible = windows_result.is_ok();
            if let Err(e) = windows_result {
                result.issues.push(format!("Windows: {}", e));
            }
        } else {
            // Test Windows compatibility even on other platforms
            let windows_compatible = self.test_windows_compatibility(&path_str);
            result.windows_compatible = windows_compatible.is_ok();
            if let Err(e) = windows_compatible {
                result.issues.push(format!("Windows compatibility: {}", e));
            }
        }
        
        // Test current platform compatibility
        result.current_platform_compatible = self.test_current_platform_compatibility(&path_str);
        
        // Test other platforms
        result.macos_compatible = self.test_macos_compatibility(&path_str);
        result.linux_compatible = self.test_linux_compatibility(&path_str);
        
        // Overall compatibility
        result.cross_platform_safe = result.windows_compatible 
            && result.macos_compatible 
            && result.linux_compatible;
        
        // Generate suggestions if not compatible
        if !result.cross_platform_safe {
            result.suggested_alternatives = self.generate_cross_platform_alternatives(&path_str);
        }
        
        result
    }
    
    fn test_windows_compatibility(&self, path: &str) -> Result<()> {
        // Test against Windows rules even on other platforms
        let windows_caps = PlatformCapabilities::windows_capabilities();
        
        for component in Path::new(path).components() {
            if let std::path::Component::Normal(os_str) = component {
                let filename = os_str.to_string_lossy();
                
                // Check reserved characters
                for ch in filename.chars() {
                    if windows_caps.reserved_characters.contains(&ch) {
                        return Err(anyhow::anyhow!(
                            "Contains Windows reserved character: '{}'", ch
                        ));
                    }
                }
                
                // Check reserved names (simplified check)
                let upper_filename = filename.to_uppercase();
                let base_name = upper_filename.split('.').next().unwrap_or("");
                if RESERVED_NAMES.contains(&base_name) {
                    return Err(anyhow::anyhow!(
                        "Contains Windows reserved name: '{}'", base_name
                    ));
                }
                
                // Check filename length
                if filename.len() > 255 {
                    return Err(anyhow::anyhow!(
                        "Filename too long for Windows: {} characters", filename.len()
                    ));
                }
            }
        }
        
        // Check total path length
        if path.len() > 260 {
            return Err(anyhow::anyhow!(
                "Path too long for Windows (without extended paths): {} characters", path.len()
            ));
        }
        
        Ok(())
    }
    
    fn test_current_platform_compatibility(&self, path: &str) -> bool {
        for component in Path::new(path).components() {
            if let std::path::Component::Normal(os_str) = component {
                let filename = os_str.to_string_lossy();
                if !self.capabilities.is_filename_valid_for_platform(&filename) {
                    return false;
                }
            }
        }
        true
    }
    
    fn test_macos_compatibility(&self, path: &str) -> bool {
        for component in Path::new(path).components() {
            if let std::path::Component::Normal(os_str) = component {
                let filename = os_str.to_string_lossy();
                
                // macOS reserves colon and has issues with certain names
                if filename.contains(':') {
                    return false;
                }
                
                // Check for problematic names on macOS
                if filename.starts_with('.') && filename.len() == 1 {
                    return false; // Just "." is problematic
                }
                
                // Length check
                if filename.len() > 255 {
                    return false;
                }
            }
        }
        
        // Path length check
        path.len() <= 1024
    }
    
    fn test_linux_compatibility(&self, path: &str) -> bool {
        for component in Path::new(path).components() {
            if let std::path::Component::Normal(os_str) = component {
                let filename = os_str.to_string_lossy();
                
                // Linux only reserves forward slash and null
                if filename.contains('/') || filename.contains('\0') {
                    return false;
                }
                
                // Special directory names
                if filename == "." || filename == ".." {
                    return false;
                }
                
                // Length check
                if filename.len() > 255 {
                    return false;
                }
            }
        }
        
        // Path length check
        path.len() <= 4096
    }
    
    fn generate_cross_platform_alternatives(&self, path: &str) -> Vec<String> {
        let mut alternatives = Vec::new();
        
        // Convert to safe ASCII
        let ascii_safe = self.to_cross_platform_safe(path);
        alternatives.push(ascii_safe);
        
        // Replace problematic characters
        let mut char_replaced = path.to_string();
        for &ch in &['<', '>', ':', '"', '|', '?', '*', '\\', '/'] {
            char_replaced = char_replaced.replace(ch, "_");
        }
        alternatives.push(char_replaced);
        
        // Truncate if too long
        if path.len() > 200 {
            let truncated = format!("{}...", &path[..200]);
            alternatives.push(truncated);
        }
        
        // Remove duplicates
        alternatives.sort();
        alternatives.dedup();
        
        alternatives
    }
    
    pub fn to_cross_platform_safe(&self, path: &str) -> String {
        let mut safe_path = String::new();
        
        for ch in path.chars() {
            match ch {
                // Replace all potentially problematic characters
                '<' | '>' | ':' | '"' | '|' | '?' | '*' => safe_path.push('_'),
                '\\' | '/' => safe_path.push('-'), // Use dash for separators
                '\0' => continue, // Skip null characters
                c if c as u32 <= 31 => continue, // Skip control characters
                c => safe_path.push(c),
            }
        }
        
        // Ensure reasonable length
        if safe_path.len() > 200 {
            safe_path.truncate(200);
        }
        
        // Remove trailing dots and spaces
        while safe_path.ends_with('.') || safe_path.ends_with(' ') {
            safe_path.pop();
        }
        
        if safe_path.is_empty() {
            safe_path = "safe_filename".to_string();
        }
        
        safe_path
    }
    
    pub fn get_platform_capabilities(&self) -> &PlatformCapabilities {
        &self.capabilities
    }
    
    pub fn normalize_for_current_platform(&self, path: &str) -> String {
        self.capabilities.normalize_path_for_platform(path)
    }
}

#[derive(Debug, Clone)]
pub struct CrossPlatformValidationResult {
    pub cross_platform_safe: bool,
    pub windows_compatible: bool,
    pub macos_compatible: bool,
    pub linux_compatible: bool,
    pub current_platform_compatible: bool,
    pub issues: Vec<String>,
    pub suggested_alternatives: Vec<String>,
}

impl CrossPlatformValidationResult {
    pub fn new() -> Self {
        Self {
            cross_platform_safe: true,
            windows_compatible: true,
            macos_compatible: true,
            linux_compatible: true,
            current_platform_compatible: true,
            issues: Vec::new(),
            suggested_alternatives: Vec::new(),
        }
    }
    
    pub fn get_compatible_platforms(&self) -> Vec<Platform> {
        let mut platforms = Vec::new();
        
        if self.windows_compatible {
            platforms.push(Platform::Windows);
        }
        if self.macos_compatible {
            platforms.push(Platform::MacOS);
        }
        if self.linux_compatible {
            platforms.push(Platform::Linux);
        }
        
        platforms
    }
}
```

### 3. Add comprehensive cross-platform tests
Add these cross-platform test modules:
```rust
#[cfg(test)]
mod cross_platform_tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_platform_detection() {
        let capabilities = PlatformCapabilities::detect_current_platform();
        
        // Should detect a known platform
        assert_ne!(capabilities.platform, Platform::Other("Unknown".to_string()));
        
        println!("Detected platform: {:?}", capabilities.platform);
        println!("Capabilities: {:#?}", capabilities);
        
        // Validate platform-specific capabilities
        match capabilities.platform {
            Platform::Windows => {
                assert_eq!(capabilities.path_separator, '\\');
                assert!(!capabilities.case_sensitive_filesystem);
                assert_eq!(capabilities.unicode_normalization, UnicodeNormalizationForm::NFC);
            }
            Platform::MacOS => {
                assert_eq!(capabilities.path_separator, '/');
                assert_eq!(capabilities.unicode_normalization, UnicodeNormalizationForm::NFD);
            }
            Platform::Linux => {
                assert_eq!(capabilities.path_separator, '/');
                assert!(capabilities.case_sensitive_filesystem);
            }
            _ => {}
        }
    }
    
    #[test]
    fn test_cross_platform_validation() {
        let handler = CrossPlatformPathHandler::new();
        
        let test_paths = vec![
            ("safe_file.txt", true),
            ("file with spaces.txt", true),
            ("file<with>reserved.txt", false),
            ("file:with:colons.txt", false),
            ("CON.txt", false),
            ("very_long_filename_".to_string() + &"x".repeat(300) + ".txt", false),
            ("测试文件.txt", true), // Unicode should be safe
        ];
        
        for (path_str, should_be_safe) in test_paths {
            let path = Path::new(&path_str);
            let result = handler.validate_path_cross_platform(&path);
            
            assert_eq!(result.cross_platform_safe, should_be_safe,
                      "Cross-platform safety mismatch for '{}'. Issues: {:?}", 
                      path_str, result.issues);
            
            if !result.cross_platform_safe {
                assert!(!result.suggested_alternatives.is_empty(),
                       "Should provide alternatives for unsafe path: {}", path_str);
            }
        }
    }
    
    #[test]
    fn test_platform_specific_validation() {
        let handler = CrossPlatformPathHandler::new();
        
        // Test Windows-specific reserved names
        let windows_path = Path::new("CON.txt");
        let result = handler.validate_path_cross_platform(&windows_path);
        
        assert!(!result.windows_compatible);
        assert!(result.issues.iter().any(|issue| issue.contains("reserved")));
        
        // Test characters problematic on different platforms
        let colon_path = Path::new("file:name.txt");
        let result = handler.validate_path_cross_platform(&colon_path);
        
        // Should fail on Windows and macOS but might be OK on Linux
        assert!(!result.windows_compatible);
        // Note: macOS compatibility depends on filesystem
    }
    
    #[test]
    fn test_path_normalization() {
        let handler = CrossPlatformPathHandler::new();
        
        let test_cases = vec![
            ("path/with/forward/slashes", "Windows should convert to backslashes"),
            ("path\\with\\back\\slashes", "Unix should convert to forward slashes"),
            ("mixed/path\\separators", "Should normalize separators"),
        ];
        
        for (input, description) in test_cases {
            let normalized = handler.normalize_for_current_platform(input);
            println!("{}: '{}' -> '{}'", description, input, normalized);
            
            // Verify normalization matches platform expectations
            let expected_separator = handler.get_platform_capabilities().path_separator;
            if normalized.contains('/') || normalized.contains('\\') {
                assert!(normalized.chars().any(|c| c == expected_separator),
                       "Normalized path should use platform separator: {}", expected_separator);
            }
        }
    }
    
    #[test]
    fn test_cross_platform_safe_conversion() {
        let handler = CrossPlatformPathHandler::new();
        
        let problematic_paths = vec![
            "file<name>.txt",
            "file:name.txt", 
            "file|name.txt",
            "file?name.txt",
            "file*name.txt",
            "file\"name\".txt",
            "CON.txt",
            "file\x00name.txt", // Null character
            "file\x01name.txt", // Control character
        ];
        
        for problematic_path in problematic_paths {
            let safe_path = handler.to_cross_platform_safe(problematic_path);
            
            // Verify the safe path passes validation on all platforms
            let safe_validation = handler.validate_path_cross_platform(Path::new(&safe_path));
            assert!(safe_validation.cross_platform_safe,
                   "Safe conversion failed for '{}' -> '{}'. Issues: {:?}",
                   problematic_path, safe_path, safe_validation.issues);
        }
    }
    
    #[test]
    fn test_unicode_across_platforms() {
        let handler = CrossPlatformPathHandler::new();
        
        let unicode_paths = vec![
            "简单文件.txt",      // Chinese
            "файл.txt",         // Russian
            "ファイル.txt",      // Japanese
            "파일.txt",          // Korean
            "αρχείο.txt",       // Greek
            "café.txt",         // Accented characters
            "naïve.txt",        // Diacritics
        ];
        
        for unicode_path in unicode_paths {
            let result = handler.validate_path_cross_platform(Path::new(unicode_path));
            
            println!("Unicode path '{}' compatibility:", unicode_path);
            println!("  Windows: {}", result.windows_compatible);
            println!("  macOS: {}", result.macos_compatible);
            println!("  Linux: {}", result.linux_compatible);
            
            // Most Unicode should be compatible across modern platforms
            // but might have normalization differences
            if !result.cross_platform_safe {
                println!("  Issues: {:?}", result.issues);
                println!("  Alternatives: {:?}", result.suggested_alternatives);
            }
        }
    }
    
    #[test]
    fn test_length_limits_across_platforms() {
        let handler = CrossPlatformPathHandler::new();
        
        // Test various length scenarios
        let length_tests = vec![
            ("short.txt", true),
            ("x".repeat(100) + ".txt", true),
            ("x".repeat(255) + ".txt", false), // Most platforms limit filename to 255
            ("x".repeat(300) + ".txt", false),
        ];
        
        for (filename, should_be_safe) in length_tests {
            let result = handler.validate_path_cross_platform(Path::new(&filename));
            
            println!("Length test '{}' chars: cross-platform safe = {}",
                    filename.len(), result.cross_platform_safe);
            
            if filename.len() > 255 {
                // Very long filenames should generally fail
                assert!(!result.cross_platform_safe,
                       "Very long filename should not be cross-platform safe");
            }
        }
    }
    
    #[test]
    fn test_case_sensitivity_handling() {
        let handler = CrossPlatformPathHandler::new();
        
        let case_variants = vec![
            ("File.txt", "file.txt"),
            ("FILE.TXT", "file.txt"), 
            ("MyDocument.PDF", "mydocument.pdf"),
        ];
        
        for (original, lowercase) in case_variants {
            let original_result = handler.validate_path_cross_platform(Path::new(original));
            let lowercase_result = handler.validate_path_cross_platform(Path::new(lowercase));
            
            // Both should be valid, but behavior might differ on case-sensitive systems
            assert!(original_result.cross_platform_safe);
            assert!(lowercase_result.cross_platform_safe);
            
            println!("Case sensitivity test: '{}' and '{}' both valid", original, lowercase);
        }
    }
    
    #[test]
    fn test_special_filenames() {
        let handler = CrossPlatformPathHandler::new();
        
        let special_files = vec![
            (".hidden", "Hidden file (Unix convention)"),
            ("file.", "Ends with period (Windows issue)"),
            ("file ", "Ends with space (Windows issue)"),
            ("..parent", "Starts with double dots"),
            ("normal_file.txt", "Normal file"),
        ];
        
        for (filename, description) in special_files {
            let result = handler.validate_path_cross_platform(Path::new(filename));
            
            println!("{}: '{}' - cross-platform safe: {}", 
                    description, filename, result.cross_platform_safe);
            
            if !result.cross_platform_safe {
                println!("  Issues: {:?}", result.issues);
                println!("  Suggested alternatives: {:?}", result.suggested_alternatives);
            }
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_real_file_operations_cross_platform() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = CrossPlatformPathHandler::new();
        
        // Test creating files with cross-platform safe names
        let safe_files = vec![
            "document.pdf",
            "image.jpg",
            "data.json",
            "config.xml",
            "readme.md",
        ];
        
        for filename in safe_files {
            let file_path = temp_dir.path().join(filename);
            
            // Verify the path is cross-platform safe
            let validation = handler.validate_path_cross_platform(&file_path);
            assert!(validation.cross_platform_safe,
                   "File '{}' should be cross-platform safe", filename);
            
            // Create the file
            std::fs::write(&file_path, "test content")?;
            assert!(file_path.exists());
            
            // Test file operations
            let metadata = std::fs::metadata(&file_path)?;
            assert!(metadata.is_file());
        }
        
        Ok(())
    }
    
    #[test]  
    fn test_directory_structure_compatibility() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = CrossPlatformPathHandler::new();
        
        // Create a directory structure that should work on all platforms
        let structure = vec![
            "documents",
            "documents/reports",
            "documents/reports/2023",
            "documents/images",
            "documents/data",
        ];
        
        for dir_path in structure {
            let full_path = temp_dir.path().join(dir_path);
            
            // Validate cross-platform compatibility
            let validation = handler.validate_path_cross_platform(&full_path);
            assert!(validation.cross_platform_safe,
                   "Directory '{}' should be cross-platform safe", dir_path);
            
            // Create the directory
            std::fs::create_dir_all(&full_path)?;
            assert!(full_path.exists());
            assert!(full_path.is_dir());
        }
        
        Ok(())
    }
    
    #[test]
    fn test_indexing_system_integration() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = CrossPlatformPathHandler::new();
        
        // Create a mix of files that an indexing system might encounter
        let mixed_files = vec![
            ("document.pdf", true),
            ("image.jpg", true),
            ("data.json", true),
            ("script.sh", true),
            ("config.xml", true),
        ];
        
        let mut indexable_count = 0;
        
        for (filename, should_be_indexable) in mixed_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, "content")?;
            
            // Test cross-platform validation
            let validation = handler.validate_path_cross_platform(&file_path);
            
            if validation.cross_platform_safe {
                indexable_count += 1;
                
                // Additional indexing-related tests
                let path_str = file_path.to_string_lossy();
                let normalized = handler.normalize_for_current_platform(&path_str);
                
                println!("Indexable file: '{}' -> '{}'", filename, normalized);
            } else {
                println!("Non-indexable file: '{}', issues: {:?}", filename, validation.issues);
            }
            
            assert_eq!(validation.cross_platform_safe, should_be_indexable,
                      "Indexability mismatch for '{}'", filename);
        }
        
        // All test files should be indexable
        assert_eq!(indexable_count, 5);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Comprehensive platform detection and capability assessment
- [ ] Cross-platform path validation with platform-specific rules
- [ ] Graceful degradation of Windows features on other platforms
- [ ] Unicode normalization handling across different platforms
- [ ] Path length and character restrictions for all major platforms
- [ ] Case sensitivity handling for different filesystems
- [ ] Cross-platform safe filename generation
- [ ] Integration testing with real file operations
- [ ] Performance validation across platforms
- [ ] All tests pass on Windows, macOS, and Linux
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Different platforms have different Unicode normalization preferences (NFC vs NFD)
- Case sensitivity varies by platform and filesystem configuration
- Path length limits differ significantly between platforms
- Reserved characters and names are platform-specific
- Some features (like extended paths) are Windows-only
- Symlinks and hard links have different support levels
- File creation tests validate actual platform behavior
- Cross-platform compatibility is essential for distributed indexing systems