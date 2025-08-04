# Task 31: Setup Windows-Specific Test Suite

## Context
You are implementing Phase 4 of a vector indexing system. Unicode path support has been implemented. Now you need to create a comprehensive Windows-specific test suite that validates all Windows optimizations, handles platform-specific testing, and ensures robust validation across different Windows versions and configurations.

## Current State
- `src/windows.rs` has Unicode path support with comprehensive validation
- Reserved names checking and filename validation are implemented
- Extended path support is working
- Need comprehensive test suite for Windows-specific functionality

## Task Objective
Create a complete Windows-specific test suite with platform detection, version-specific tests, comprehensive edge case coverage, and integration with the existing test framework.

## Implementation Requirements

### 1. Create Windows test configuration
Add this test configuration system to `src/windows.rs`:
```rust
#[cfg(test)]
mod windows_tests {
    use super::*;
    use std::env;
    use std::process::Command;
    
    #[derive(Debug, Clone)]
    pub struct WindowsTestConfig {
        pub windows_version: WindowsVersion,
        pub filesystem_type: FileSystemType,
        pub long_path_support: bool,
        pub unicode_support_level: UnicodeSupport,
        pub admin_privileges: bool,
        pub test_data_path: PathBuf,
    }
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum WindowsVersion {
        Windows7,
        Windows8,
        Windows81,
        Windows10,
        Windows11,
        WindowsServer2008,
        WindowsServer2012,
        WindowsServer2016,
        WindowsServer2019,
        WindowsServer2022,
        Unknown,
    }
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum FileSystemType {
        NTFS,
        FAT32,
        ExFAT,
        ReFS,
        Unknown,
    }
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum UnicodeSupport {
        Full,      // Full Unicode support
        Limited,   // Basic Unicode support
        ASCII,     // ASCII only
    }
    
    impl WindowsTestConfig {
        pub fn detect_current_system() -> Self {
            Self {
                windows_version: Self::detect_windows_version(),
                filesystem_type: Self::detect_filesystem_type(),
                long_path_support: Self::detect_long_path_support(),
                unicode_support_level: Self::detect_unicode_support(),
                admin_privileges: Self::has_admin_privileges(),
                test_data_path: Self::setup_test_data_directory(),
            }
        }
        
        fn detect_windows_version() -> WindowsVersion {
            #[cfg(windows)]
            {
                use std::ffi::OsString;
                use std::os::windows::ffi::OsStringExt;
                
                // Try to get version from registry or system calls
                // This is a simplified detection - real implementation would use Windows APIs
                if let Ok(output) = Command::new("cmd")
                    .args(&["/c", "ver"])
                    .output()
                {
                    let version_str = String::from_utf8_lossy(&output.stdout);
                    
                    if version_str.contains("10.0.22") {
                        return WindowsVersion::Windows11;
                    } else if version_str.contains("10.0") {
                        return WindowsVersion::Windows10;
                    } else if version_str.contains("6.3") {
                        return WindowsVersion::Windows81;
                    } else if version_str.contains("6.2") {
                        return WindowsVersion::Windows8;
                    } else if version_str.contains("6.1") {
                        return WindowsVersion::Windows7;
                    }
                }
            }
            
            WindowsVersion::Unknown
        }
        
        fn detect_filesystem_type() -> FileSystemType {
            #[cfg(windows)]
            {
                // Try to detect filesystem type for the current drive
                if let Ok(output) = Command::new("fsutil")
                    .args(&["fsinfo", "volumeinfo", "C:"])
                    .output()
                {
                    let info = String::from_utf8_lossy(&output.stdout);
                    
                    if info.contains("NTFS") {
                        return FileSystemType::NTFS;
                    } else if info.contains("FAT32") {
                        return FileSystemType::FAT32;
                    } else if info.contains("exFAT") {
                        return FileSystemType::ExFAT;
                    } else if info.contains("ReFS") {
                        return FileSystemType::ReFS;
                    }
                }
            }
            
            FileSystemType::Unknown
        }
        
        fn detect_long_path_support() -> bool {
            #[cfg(windows)]
            {
                // Check if long path support is enabled in Windows
                // This would typically check registry or try creating a long path
                if let Ok(output) = Command::new("reg")
                    .args(&["query", "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem", "/v", "LongPathsEnabled"])
                    .output()
                {
                    let reg_output = String::from_utf8_lossy(&output.stdout);
                    return reg_output.contains("0x1");
                }
            }
            
            false
        }
        
        fn detect_unicode_support() -> UnicodeSupport {
            // Test Unicode support by trying to create files with Unicode names
            UnicodeSupport::Full // Assume full support on modern Windows
        }
        
        fn has_admin_privileges() -> bool {
            #[cfg(windows)]
            {
                // Try to write to a system directory to test admin privileges
                if let Ok(output) = Command::new("net")
                    .args(&["user"])
                    .output()
                {
                    return output.status.success();
                }
            }
            
            false
        }
        
        fn setup_test_data_directory() -> PathBuf {
            let temp_dir = env::temp_dir();
            let test_dir = temp_dir.join("windows_indexer_tests");
            
            if !test_dir.exists() {
                std::fs::create_dir_all(&test_dir).unwrap_or_default();
            }
            
            test_dir
        }
        
        pub fn supports_feature(&self, feature: WindowsFeature) -> bool {
            match feature {
                WindowsFeature::ExtendedPaths => {
                    matches!(self.windows_version, 
                        WindowsVersion::Windows10 | 
                        WindowsVersion::Windows11 |
                        WindowsVersion::WindowsServer2016 |
                        WindowsVersion::WindowsServer2019 |
                        WindowsVersion::WindowsServer2022
                    ) && self.long_path_support
                }
                WindowsFeature::UnicodeFilenames => {
                    self.unicode_support_level != UnicodeSupport::ASCII
                }
                WindowsFeature::NTFSFeatures => {
                    self.filesystem_type == FileSystemType::NTFS
                }
                WindowsFeature::AdminRequiredFeatures => {
                    self.admin_privileges
                }
            }
        }
    }
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum WindowsFeature {
        ExtendedPaths,
        UnicodeFilenames,
        NTFSFeatures,
        AdminRequiredFeatures,
    }
    
    // Test fixture setup
    pub struct WindowsTestFixture {
        pub config: WindowsTestConfig,
        pub temp_dir: tempfile::TempDir,
        pub handler: WindowsPathHandler,
        pub test_files: Vec<PathBuf>,
    }
    
    impl WindowsTestFixture {
        pub fn new() -> Result<Self> {
            let config = WindowsTestConfig::detect_current_system();
            let temp_dir = tempfile::TempDir::new()?;
            let handler = WindowsPathHandler::new();
            
            Ok(Self {
                config,
                temp_dir,
                handler,
                test_files: Vec::new(),
            })
        }
        
        pub fn create_test_file(&mut self, name: &str, content: &str) -> Result<PathBuf> {
            let file_path = self.temp_dir.path().join(name);
            std::fs::write(&file_path, content)?;
            self.test_files.push(file_path.clone());
            Ok(file_path)
        }
        
        pub fn create_test_directory(&mut self, name: &str) -> Result<PathBuf> {
            let dir_path = self.temp_dir.path().join(name);
            std::fs::create_dir_all(&dir_path)?;
            Ok(dir_path)
        }
        
        pub fn create_long_path_file(&mut self) -> Result<PathBuf> {
            if !self.config.supports_feature(WindowsFeature::ExtendedPaths) {
                return Err(anyhow::anyhow!("Extended paths not supported on this system"));
            }
            
            // Create a very long path
            let long_component = "very_long_directory_name_".repeat(20);
            let mut current_path = self.temp_dir.path().to_path_buf();
            
            // Build up a long path
            for i in 0..10 {
                current_path = current_path.join(format!("{}{}", long_component, i));
                std::fs::create_dir_all(&current_path)?;
            }
            
            let long_file = current_path.join("long_path_test_file.txt");
            std::fs::write(&long_file, "Long path test content")?;
            self.test_files.push(long_file.clone());
            
            Ok(long_file)
        }
        
        pub fn create_unicode_test_files(&mut self) -> Result<Vec<PathBuf>> {
            if !self.config.supports_feature(WindowsFeature::UnicodeFilenames) {
                return Err(anyhow::anyhow!("Unicode filenames not supported"));
            }
            
            let unicode_names = vec![
                "测试文件.txt",           // Chinese
                "тестовый файл.txt",      // Russian  
                "ملف الاختبار.txt",       // Arabic
                "טעסט טעקע.txt",          // Hebrew
                "परीक्षण फ़ाइल.txt",        // Hindi
                "テストファイル.txt",        // Japanese
                "테스트파일.txt",           // Korean
                "αρχείο δοκιμής.txt",     // Greek
                "café_résumé.txt",        // French accents
                "naïve_coöperation.txt",  // Diacritics
            ];
            
            let mut created_files = Vec::new();
            
            for name in unicode_names {
                match self.create_test_file(name, "Unicode test content") {
                    Ok(path) => created_files.push(path),
                    Err(e) => println!("Warning: Could not create Unicode file '{}': {}", name, e),
                }
            }
            
            Ok(created_files)
        }
        
        pub fn create_reserved_name_files(&mut self) -> Result<Vec<(PathBuf, bool)>> {
            let reserved_names = vec![
                ("CON.txt", true),
                ("PRN.log", true),
                ("COM1.dat", true),
                ("LPT1.txt", true),
                ("NUL.tmp", true),
                ("desktop.ini", true),
                ("thumbs.db", true),
                ("normal_file.txt", false),
            ];
            
            let mut results = Vec::new();
            
            for (name, should_be_reserved) in reserved_names {
                // Note: On Windows, we might not be able to actually create reserved name files
                // So we'll test the detection logic instead
                let test_path = self.temp_dir.path().join(name);
                
                if !should_be_reserved {
                    // Create normal files
                    match std::fs::write(&test_path, "test content") {
                        Ok(_) => {
                            self.test_files.push(test_path.clone());
                            results.push((test_path, should_be_reserved));
                        }
                        Err(e) => println!("Warning: Could not create file '{}': {}", name, e),
                    }
                } else {
                    // Just add to results for testing detection logic
                    results.push((test_path, should_be_reserved));
                }
            }
            
            Ok(results)
        }
        
        pub fn cleanup(&mut self) {
            self.test_files.clear();
            // temp_dir will be automatically cleaned up when dropped
        }
    }
}
```

### 2. Add comprehensive Windows-specific tests
Add these test functions:
```rust
#[cfg(test)]
mod windows_integration_tests {
    use super::windows_tests::*;
    use super::*;
    
    #[test]
    fn test_windows_system_detection() {
        let config = WindowsTestConfig::detect_current_system();
        
        // Should detect some version (even if unknown)
        assert_ne!(config.windows_version, WindowsVersion::Unknown);
        
        // Should detect filesystem type
        println!("Detected filesystem: {:?}", config.filesystem_type);
        
        // Print configuration for debugging
        println!("Windows Test Configuration: {:#?}", config);
    }
    
    #[test]
    fn test_path_handler_with_system_config() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        
        // Test basic functionality
        let test_file = fixture.create_test_file("test.txt", "content")?;
        assert!(test_file.exists());
        
        // Test path validation with actual file
        let validation = fixture.handler.validate_windows_path(&test_file);
        assert!(validation.is_ok());
        
        Ok(())
    }
    
    #[test]
    #[cfg(windows)]
    fn test_extended_path_support() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        
        if !fixture.config.supports_feature(WindowsFeature::ExtendedPaths) {
            println!("Skipping extended path test - not supported on this system");
            return Ok(());
        }
        
        // Try to create and work with long paths
        match fixture.create_long_path_file() {
            Ok(long_path) => {
                assert!(long_path.to_string_lossy().len() > 260);
                
                // Test our handler with the long path
                let extended = fixture.handler.ensure_extended_path(&long_path)?;
                assert!(extended.to_string_lossy().starts_with(r"\\?\"));
                
                // Test file operations with extended path
                let metadata = fixture.handler.metadata_extended(&long_path)?;
                assert!(metadata.is_file());
            }
            Err(e) => {
                println!("Could not create long path file: {}", e);
                // This might fail on systems without long path support enabled
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_unicode_filename_support() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        
        if !fixture.config.supports_feature(WindowsFeature::UnicodeFilenames) {
            println!("Skipping Unicode test - not supported on this system");
            return Ok(());
        }
        
        let unicode_files = fixture.create_unicode_test_files()?;
        
        for file_path in unicode_files {
            // Test Unicode validation
            let filename = file_path.file_name().unwrap().to_string_lossy();
            let validation = fixture.handler.validate_unicode_path(&filename);
            
            if !validation.is_valid {
                println!("Unicode validation failed for '{}': {:?}", filename, validation.errors);
            }
            
            // Test file operations
            if file_path.exists() {
                let metadata = fixture.handler.metadata_extended(&file_path);
                assert!(metadata.is_ok());
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_reserved_names_detection() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        let reserved_files = fixture.create_reserved_name_files()?;
        
        for (file_path, should_be_reserved) in reserved_files {
            let filename = file_path.file_name().unwrap().to_string_lossy();
            let check_result = fixture.handler.check_reserved_name(&filename);
            
            if should_be_reserved {
                assert!(check_result.is_reserved, 
                       "Expected '{}' to be detected as reserved", filename);
            } else {
                assert!(!check_result.is_reserved, 
                       "Expected '{}' to not be detected as reserved", filename);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_filesystem_specific_features() -> Result<()> {
        let fixture = WindowsTestFixture::new()?;
        
        match fixture.config.filesystem_type {
            FileSystemType::NTFS => {
                // Test NTFS-specific features
                println!("Testing NTFS-specific features");
                
                // NTFS supports long filenames, Unicode, etc.
                assert!(fixture.config.supports_feature(WindowsFeature::UnicodeFilenames));
            }
            FileSystemType::FAT32 => {
                // Test FAT32 limitations
                println!("Testing FAT32 limitations");
                
                // FAT32 has filename length limitations
                let long_filename = "x".repeat(300);
                let validation = fixture.handler.validate_filename_detailed(&long_filename);
                assert!(!validation.is_valid);
            }
            _ => {
                println!("Unknown or unsupported filesystem type: {:?}", fixture.config.filesystem_type);
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_cross_platform_compatibility() -> Result<()> {
        let fixture = WindowsTestFixture::new()?;
        
        let test_paths = vec![
            "normal_file.txt",
            "file with spaces.txt",
            "file-with-hyphens.txt",
            "file_with_underscores.txt",
            "UPPERCASE.TXT",
            "lowercase.txt",
            "Mixed.Case.txt",
            "file.multiple.extensions.txt",
        ];
        
        for path_str in test_paths {
            let is_safe = fixture.handler.is_cross_platform_safe(path_str);
            println!("Path '{}' is cross-platform safe: {}", path_str, is_safe);
            
            // All these should be cross-platform safe
            assert!(is_safe, "Path '{}' should be cross-platform safe", path_str);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_performance_with_large_directory() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        
        // Create a directory with many files
        let test_dir = fixture.create_test_directory("performance_test")?;
        
        let start_time = std::time::Instant::now();
        
        // Create 100 test files
        for i in 0..100 {
            let filename = format!("test_file_{:03}.txt", i);
            let file_path = test_dir.join(&filename);
            std::fs::write(&file_path, format!("Content for file {}", i))?;
        }
        
        let creation_time = start_time.elapsed();
        println!("Created 100 files in {:?}", creation_time);
        
        // Test batch validation
        let validation_start = std::time::Instant::now();
        let results = fixture.handler.check_directory_for_reserved_names(&test_dir)?;
        let validation_time = validation_start.elapsed();
        
        println!("Validated directory in {:?}", validation_time);
        println!("Found {} reserved names", results.len());
        
        // Should be fast even with many files
        assert!(validation_time.as_millis() < 1000, "Validation took too long: {:?}", validation_time);
        
        Ok(())
    }
    
    #[test]
    fn test_error_handling_and_recovery() -> Result<()> {
        let fixture = WindowsTestFixture::new()?;
        
        // Test handling of non-existent paths
        let non_existent = PathBuf::from("C:\\this\\path\\does\\not\\exist\\file.txt");
        let result = fixture.handler.metadata_extended(&non_existent);
        assert!(result.is_err());
        
        // Test handling of invalid Unicode
        let invalid_unicode = std::str::from_utf8(&[0xFF, 0xFE]).unwrap_or("invalid");
        let validation = fixture.handler.validate_unicode_path(invalid_unicode);
        // Should handle gracefully without panicking
        
        // Test handling of very long paths without extended path support
        let very_long_path = PathBuf::from("C:\\".to_string() + &"x".repeat(1000));
        let validation = fixture.handler.validate_windows_path(&very_long_path);
        // Should detect the length issue
        
        Ok(())
    }
    
    #[test]
    fn test_integration_with_indexing_system() -> Result<()> {
        let mut fixture = WindowsTestFixture::new()?;
        
        // Create a mix of files that would be encountered in real indexing
        let mixed_files = vec![
            ("document.pdf", "Safe indexable file"),
            ("IMG_001.jpg", "Image file"),
            ("data.json", "JSON data"),
            ("config.xml", "XML configuration"),
            ("script.ps1", "PowerShell script"),
            ("readme.md", "Markdown readme"),
        ];
        
        let mut indexable_files = Vec::new();
        
        for (filename, content) in mixed_files {
            let file_path = fixture.create_test_file(filename, content)?;
            
            // Test if file is safe for indexing
            if fixture.handler.is_path_safe_for_indexing(&file_path) {
                indexable_files.push(file_path);
            }
        }
        
        // All these files should be safe for indexing
        assert_eq!(indexable_files.len(), 6);
        
        // Test batch processing
        let stats = fixture.handler.get_reserved_name_statistics(fixture.temp_dir.path())?;
        assert_eq!(stats.total_reserved, 0); // No reserved names in our test files
        
        Ok(())
    }
}
```

### 3. Add benchmark tests
Add these performance tests:
```rust
#[cfg(test)]
mod windows_benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    #[ignore] // Run with --ignored flag
    fn benchmark_filename_validation() {
        let handler = WindowsPathHandler::new();
        let test_filenames = vec![
            "normal_file.txt",
            "file with spaces.txt",
            "very_long_filename_".to_string() + &"x".repeat(200) + ".txt",
            "测试文件.txt",
            "CON.txt",
            "file<invalid>.txt",
        ];
        
        let iterations = 10000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            for filename in &test_filenames {
                let _ = handler.validate_filename_detailed(filename);
            }
        }
        
        let duration = start.elapsed();
        let per_validation = duration / (iterations * test_filenames.len() as u32);
        
        println!("Filename validation benchmark:");
        println!("Total time: {:?}", duration);
        println!("Per validation: {:?}", per_validation);
        println!("Validations per second: {}", 1_000_000_000 / per_validation.as_nanos());
        
        // Should be very fast
        assert!(per_validation.as_micros() < 100);
    }
    
    #[test]
    #[ignore]
    fn benchmark_unicode_validation() {
        let handler = WindowsPathHandler::new();
        let unicode_paths = vec![
            "C:\\Documents\\测试文件.txt",
            "C:\\Документы\\тестовый файл.txt",
            "C:\\المستندات\\ملف الاختبار.txt",
            "C:\\ドキュメント\\テストファイル.txt",
        ];
        
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            for path in &unicode_paths {
                let _ = handler.validate_unicode_path(path);
            }
        }
        
        let duration = start.elapsed();
        let per_validation = duration / (iterations * unicode_paths.len() as u32);
        
        println!("Unicode validation benchmark:");
        println!("Total time: {:?}", duration);
        println!("Per validation: {:?}", per_validation);
        
        // Unicode validation is more complex but should still be reasonable
        assert!(per_validation.as_micros() < 1000);
    }
}
```

### 4. Add test utilities and helpers
Add these helper functions:
```rust
#[cfg(test)]
mod test_utils {
    use super::*;
    
    pub fn skip_if_not_windows() {
        if !cfg!(windows) {
            panic!("This test can only run on Windows");
        }
    }
    
    pub fn skip_if_no_admin() -> Result<()> {
        let config = windows_tests::WindowsTestConfig::detect_current_system();
        if !config.admin_privileges {
            return Err(anyhow::anyhow!("Test requires administrator privileges"));
        }
        Ok(())
    }
    
    pub fn skip_if_no_extended_paths() -> Result<()> {
        let config = windows_tests::WindowsTestConfig::detect_current_system();
        if !config.supports_feature(windows_tests::WindowsFeature::ExtendedPaths) {
            return Err(anyhow::anyhow!("Test requires extended path support"));
        }
        Ok(())
    }
    
    pub fn create_test_files_with_patterns(
        base_dir: &Path,
        patterns: &[&str]
    ) -> Result<Vec<PathBuf>> {
        let mut created_files = Vec::new();
        
        for pattern in patterns {
            let file_path = base_dir.join(pattern);
            
            // Create parent directories if needed
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            
            std::fs::write(&file_path, "test content")?;
            created_files.push(file_path);
        }
        
        Ok(created_files)
    }
    
    pub fn assert_path_characteristics(
        handler: &WindowsPathHandler,
        path: &Path,
        expected_safe: bool,
        expected_reserved: bool,
        expected_unicode_valid: bool,
    ) {
        let path_str = path.to_string_lossy();
        
        // Test safety for indexing
        let is_safe = handler.is_path_safe_for_indexing(path);
        assert_eq!(is_safe, expected_safe, 
                  "Path safety mismatch for '{}'", path_str);
        
        // Test reserved name detection
        if let Some(filename) = path.file_name() {
            let filename_str = filename.to_string_lossy();
            let reserved = handler.check_reserved_name(&filename_str);
            assert_eq!(reserved.is_reserved, expected_reserved,
                      "Reserved name detection mismatch for '{}'", filename_str);
        }
        
        // Test Unicode validation
        let unicode_validation = handler.validate_unicode_path(&path_str);
        assert_eq!(unicode_validation.is_valid, expected_unicode_valid,
                  "Unicode validation mismatch for '{}'", path_str);
    }
}
```

## Success Criteria
- [ ] Comprehensive Windows system detection and configuration
- [ ] Platform-specific test fixtures and utilities
- [ ] Extended path testing with system capability detection
- [ ] Unicode filename testing across different character sets
- [ ] Reserved name testing with actual file creation attempts
- [ ] Filesystem-specific feature testing
- [ ] Performance benchmarks for validation operations
- [ ] Error handling and recovery testing
- [ ] Integration testing with indexing system workflow
- [ ] All tests pass on supported Windows versions
- [ ] Tests properly skip unsupported features
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Tests should detect system capabilities and skip unsupported features
- Different Windows versions have different Unicode and long path support
- Admin privileges may be required for some advanced testing
- Filesystem type affects available features (NTFS vs FAT32 vs ExFAT)
- Real file creation tests validate actual system behavior
- Performance tests ensure scalability for large directories
- Cross-platform compatibility tests ensure portability
- Integration tests validate the complete indexing workflow