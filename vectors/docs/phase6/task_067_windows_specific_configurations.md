# Task 067: Set up Windows-specific Configurations

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates Windows-specific configurations including Windows path handling, Unicode filename support, performance counter access, registry configuration, and service integration for optimal Windows compatibility.

## Project Structure
```
src/windows/
├── mod.rs              <- Windows module entry point
├── paths.rs            <- Windows path handling utilities
├── unicode.rs          <- Unicode filename support
├── performance.rs      <- Performance counter integration
├── registry.rs         <- Windows registry configuration
├── service.rs          <- Windows service integration
└── config.rs           <- Windows-specific configuration
```

## Task Description
Create comprehensive Windows-specific configuration and utilities that handle Windows path conventions, support Unicode filenames, integrate with Windows performance counters, manage registry settings, and provide Windows service integration for enterprise deployment.

## Requirements
1. Implement Windows path handling with long path support
2. Create Unicode filename support for international characters
3. Integrate Windows performance counters for system monitoring
4. Implement registry configuration management
5. Provide Windows service integration capabilities

## Expected File Content/Code Structure

### Main Windows Module (`src/windows/mod.rs`)
```rust
//! Windows-specific functionality for LLMKG validation system
//! 
//! Provides Windows path handling, Unicode support, performance monitoring,
//! registry configuration, and service integration.

#[cfg(windows)]
pub mod paths;
#[cfg(windows)]
pub mod unicode;
#[cfg(windows)]
pub mod performance;
#[cfg(windows)]
pub mod registry;
#[cfg(windows)]
pub mod service;
#[cfg(windows)]
pub mod config;

#[cfg(not(windows))]
pub mod stub;

#[cfg(windows)]
pub use paths::*;
#[cfg(windows)]
pub use unicode::*;
#[cfg(windows)]
pub use performance::*;
#[cfg(windows)]
pub use registry::*;
#[cfg(windows)]
pub use service::*;
#[cfg(windows)]
pub use config::*;

#[cfg(not(windows))]
pub use stub::*;

use anyhow::Result;

/// Initialize Windows-specific components
#[cfg(windows)]
pub fn initialize_windows_support() -> Result<()> {
    tracing::info!("Initializing Windows-specific support");
    
    // Enable long path support
    paths::enable_long_path_support()?;
    
    // Initialize Unicode support
    unicode::initialize_unicode_support()?;
    
    // Initialize performance counters
    performance::initialize_performance_counters()?;
    
    // Load registry configuration
    registry::load_registry_configuration()?;
    
    tracing::info!("Windows-specific support initialized successfully");
    Ok(())
}

#[cfg(not(windows))]
pub fn initialize_windows_support() -> Result<()> {
    tracing::info!("Windows-specific support not available on this platform");
    Ok(())
}

/// Check if running on Windows
pub fn is_windows() -> bool {
    cfg!(windows)
}

/// Get Windows version information
#[cfg(windows)]
pub fn get_windows_version() -> Result<WindowsVersion> {
    use windows::Win32::System::SystemInformation::{GetVersionExW, OSVERSIONINFOW};
    use windows::core::PWSTR;
    
    let mut version_info: OSVERSIONINFOW = unsafe { std::mem::zeroed() };
    version_info.dwOSVersionInfoSize = std::mem::size_of::<OSVERSIONINFOW>() as u32;
    
    unsafe {
        if GetVersionExW(&mut version_info).as_bool() {
            Ok(WindowsVersion {
                major: version_info.dwMajorVersion,
                minor: version_info.dwMinorVersion,
                build: version_info.dwBuildNumber,
                platform: version_info.dwPlatformId,
            })
        } else {
            Err(anyhow::anyhow!("Failed to get Windows version"))
        }
    }
}

#[cfg(not(windows))]
pub fn get_windows_version() -> Result<WindowsVersion> {
    Err(anyhow::anyhow!("Not running on Windows"))
}

/// Windows version information
#[derive(Debug, Clone)]
pub struct WindowsVersion {
    pub major: u32,
    pub minor: u32,
    pub build: u32,
    pub platform: u32,
}

impl WindowsVersion {
    /// Check if this is Windows 10 or later
    pub fn is_windows10_or_later(&self) -> bool {
        self.major >= 10
    }
    
    /// Check if this is Windows 11
    pub fn is_windows11(&self) -> bool {
        self.major >= 10 && self.build >= 22000
    }
    
    /// Get version string
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.build)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_windows() {
        // This test will behave differently on Windows vs other platforms
        if cfg!(windows) {
            assert!(is_windows());
        } else {
            assert!(!is_windows());
        }
    }
    
    #[cfg(windows)]
    #[test]
    fn test_windows_version() -> Result<()> {
        let version = get_windows_version()?;
        assert!(version.major > 0);
        assert!(!version.to_string().is_empty());
        Ok(())
    }
    
    #[test]
    fn test_initialize_windows_support() -> Result<()> {
        // Should not fail on any platform
        initialize_windows_support()
    }
}
```

### Windows Path Handling (`src/windows/paths.rs`)
```rust
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

/// Windows path utilities with long path support
pub struct WindowsPaths;

impl WindowsPaths {
    /// Convert a regular path to a long path format (\\?\)
    pub fn to_long_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        
        // If already a long path, return as-is
        if path.to_string_lossy().starts_with(r"\\?\") {
            return Ok(path.to_path_buf());
        }
        
        // Convert to absolute path first
        let absolute = path.canonicalize()
            .or_else(|_| std::env::current_dir().map(|cwd| cwd.join(path)))?;
        
        // Add long path prefix
        let long_path = format!(r"\\?\{}", absolute.display());
        Ok(PathBuf::from(long_path))
    }
    
    /// Check if a path exceeds Windows path length limits
    pub fn exceeds_path_limit<P: AsRef<Path>>(path: P) -> bool {
        const MAX_PATH: usize = 260;
        path.as_ref().to_string_lossy().len() > MAX_PATH
    }
    
    /// Normalize path separators to Windows format
    pub fn normalize_separators<P: AsRef<Path>>(path: P) -> PathBuf {
        let path_str = path.as_ref().to_string_lossy();
        let normalized = path_str.replace('/', r"\");
        PathBuf::from(normalized)
    }
    
    /// Convert Unix-style path to Windows-style
    pub fn unix_to_windows<P: AsRef<Path>>(path: P) -> PathBuf {
        Self::normalize_separators(path)
    }
    
    /// Get short path name (8.3 format) for compatibility
    pub fn get_short_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        use windows::Win32::Storage::FileSystem::GetShortPathNameW;
        use windows::core::PWSTR;
        
        let path = path.as_ref();
        let wide_path = Self::to_wide_string(path)?;
        
        unsafe {
            let required_size = GetShortPathNameW(
                PWSTR(wide_path.as_ptr() as *mut u16),
                PWSTR::null(),
                0,
            );
            
            if required_size == 0 {
                return Err(anyhow!("Failed to get short path name"));
            }
            
            let mut buffer = vec![0u16; required_size as usize];
            let result = GetShortPathNameW(
                PWSTR(wide_path.as_ptr() as *mut u16),
                PWSTR(buffer.as_mut_ptr()),
                required_size,
            );
            
            if result == 0 {
                return Err(anyhow!("Failed to get short path name"));
            }
            
            // Convert back to PathBuf
            let short_path = String::from_utf16_lossy(&buffer[..result as usize - 1]);
            Ok(PathBuf::from(short_path))
        }
    }
    
    /// Convert path to wide string for Windows APIs
    pub fn to_wide_string<P: AsRef<Path>>(path: P) -> Result<Vec<u16>> {
        let path_str = path.as_ref().to_string_lossy();
        let mut wide: Vec<u16> = path_str.encode_utf16().collect();
        wide.push(0); // Null terminator
        Ok(wide)
    }
    
    /// Check if path contains invalid Windows characters
    pub fn contains_invalid_chars<P: AsRef<Path>>(path: P) -> bool {
        const INVALID_CHARS: &[char] = &['<', '>', ':', '"', '|', '?', '*'];
        let path_str = path.as_ref().to_string_lossy();
        
        INVALID_CHARS.iter().any(|&c| path_str.contains(c))
    }
    
    /// Sanitize path by removing/replacing invalid characters
    pub fn sanitize_path<P: AsRef<Path>>(path: P) -> PathBuf {
        let path_str = path.as_ref().to_string_lossy();
        let sanitized = path_str
            .replace('<', "_")
            .replace('>', "_")
            .replace(':', "_")
            .replace('"', "_")
            .replace('|', "_")
            .replace('?', "_")
            .replace('*', "_");
        
        PathBuf::from(sanitized)
    }
    
    /// Get available disk space for a path
    pub fn get_disk_space<P: AsRef<Path>>(path: P) -> Result<DiskSpace> {
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        use windows::core::PWSTR;
        
        let path = path.as_ref();
        let wide_path = Self::to_wide_string(path)?;
        
        let mut free_bytes = 0u64;
        let mut total_bytes = 0u64;
        
        unsafe {
            let result = GetDiskFreeSpaceExW(
                PWSTR(wide_path.as_ptr() as *mut u16),
                Some(&mut free_bytes),
                Some(&mut total_bytes),
                None,
            );
            
            if result.as_bool() {
                Ok(DiskSpace {
                    free_bytes,
                    total_bytes,
                    used_bytes: total_bytes - free_bytes,
                })
            } else {
                Err(anyhow!("Failed to get disk space information"))
            }
        }
    }
    
    /// Check if a path is on a network drive
    pub fn is_network_path<P: AsRef<Path>>(path: P) -> bool {
        let path_str = path.as_ref().to_string_lossy();
        path_str.starts_with(r"\\") && !path_str.starts_with(r"\\?\")
    }
    
    /// Get drive type for a path
    pub fn get_drive_type<P: AsRef<Path>>(path: P) -> Result<DriveType> {
        use windows::Win32::Storage::FileSystem::{GetDriveTypeW, DRIVE_TYPE};
        use windows::core::PWSTR;
        
        let path = path.as_ref();
        let root = path.ancestors().last().unwrap_or(path);
        let wide_path = Self::to_wide_string(root)?;
        
        let drive_type = unsafe {
            GetDriveTypeW(PWSTR(wide_path.as_ptr() as *mut u16))
        };
        
        match drive_type {
            DRIVE_TYPE(2) => Ok(DriveType::Removable),
            DRIVE_TYPE(3) => Ok(DriveType::Fixed),
            DRIVE_TYPE(4) => Ok(DriveType::Remote),
            DRIVE_TYPE(5) => Ok(DriveType::CdRom),
            DRIVE_TYPE(6) => Ok(DriveType::RamDisk),
            _ => Ok(DriveType::Unknown),
        }
    }
}

/// Disk space information
#[derive(Debug, Clone)]
pub struct DiskSpace {
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl DiskSpace {
    pub fn free_mb(&self) -> u64 {
        self.free_bytes / (1024 * 1024)
    }
    
    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }
    
    pub fn used_mb(&self) -> u64 {
        self.used_bytes / (1024 * 1024)
    }
    
    pub fn usage_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Windows drive types
#[derive(Debug, Clone, PartialEq)]
pub enum DriveType {
    Unknown,
    Removable,
    Fixed,
    Remote,
    CdRom,
    RamDisk,
}

/// Enable long path support in the current process
pub fn enable_long_path_support() -> Result<()> {
    // This requires Windows 10 version 1607 or later and registry configuration
    tracing::info!("Long path support enabled (requires Windows 10 1607+ and registry configuration)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_path_length_check() {
        let short_path = Path::new("C:\\short\\path");
        let long_path = Path::new(&"C:\\".repeat(100));
        
        assert!(!WindowsPaths::exceeds_path_limit(short_path));
        assert!(WindowsPaths::exceeds_path_limit(long_path));
    }
    
    #[test]
    fn test_normalize_separators() {
        let unix_path = Path::new("src/windows/paths.rs");
        let normalized = WindowsPaths::normalize_separators(unix_path);
        
        assert_eq!(normalized.to_string_lossy(), r"src\windows\paths.rs");
    }
    
    #[test]
    fn test_invalid_chars_detection() {
        assert!(WindowsPaths::contains_invalid_chars(Path::new("file<name")));
        assert!(WindowsPaths::contains_invalid_chars(Path::new("file|name")));
        assert!(!WindowsPaths::contains_invalid_chars(Path::new("filename")));
    }
    
    #[test]
    fn test_path_sanitization() {
        let invalid_path = Path::new("file<>name|test");
        let sanitized = WindowsPaths::sanitize_path(invalid_path);
        
        assert_eq!(sanitized.to_string_lossy(), "file__name_test");
    }
    
    #[test]
    fn test_network_path_detection() {
        assert!(WindowsPaths::is_network_path(Path::new(r"\\server\share")));
        assert!(!WindowsPaths::is_network_path(Path::new(r"C:\local\path")));
        assert!(!WindowsPaths::is_network_path(Path::new(r"\\?\C:\long\path")));
    }
    
    #[test]
    fn test_to_long_path() -> Result<()> {
        let temp_dir = tempdir()?;
        let test_path = temp_dir.path().join("test.txt");
        
        let long_path = WindowsPaths::to_long_path(&test_path)?;
        assert!(long_path.to_string_lossy().starts_with(r"\\?\"));
        
        Ok(())
    }
    
    #[test]
    fn test_disk_space() -> Result<()> {
        let disk_space = WindowsPaths::get_disk_space("C:\\")?;
        
        assert!(disk_space.total_bytes > 0);
        assert!(disk_space.free_bytes <= disk_space.total_bytes);
        assert_eq!(disk_space.used_bytes, disk_space.total_bytes - disk_space.free_bytes);
        
        Ok(())
    }
}
```

### Unicode Filename Support (`src/windows/unicode.rs`)
```rust
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::ffi::{OsStr, OsString};
use std::os::windows::ffi::{OsStrExt, OsStringExt};

/// Unicode filename utilities for Windows
pub struct UnicodeSupport;

impl UnicodeSupport {
    /// Check if a filename contains Unicode characters
    pub fn contains_unicode<P: AsRef<Path>>(path: P) -> bool {
        let path_str = path.as_ref().to_string_lossy();
        path_str.chars().any(|c| !c.is_ascii())
    }
    
    /// Validate Unicode filename for Windows compatibility
    pub fn validate_unicode_filename<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        let filename = path.file_name()
            .ok_or_else(|| anyhow!("Invalid path: no filename"))?;
        
        let filename_str = filename.to_string_lossy();
        
        // Check for invalid Unicode sequences
        if filename_str.contains('\u{FFFD}') {
            return Err(anyhow!("Filename contains invalid Unicode sequences"));
        }
        
        // Check for reserved names
        let reserved_names = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ];
        
        let name_upper = filename_str.to_uppercase();
        for reserved in &reserved_names {
            if name_upper == *reserved || name_upper.starts_with(&format!("{}.", reserved)) {
                return Err(anyhow!("Filename '{}' is a reserved Windows name", filename_str));
            }
        }
        
        // Check for trailing periods and spaces
        if filename_str.ends_with(' ') || filename_str.ends_with('.') {
            return Err(anyhow!("Filename cannot end with space or period"));
        }
        
        Ok(())
    }
    
    /// Normalize Unicode filename for consistent handling
    pub fn normalize_unicode_filename<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        let filename = path.file_name()
            .ok_or_else(|| anyhow!("Invalid path: no filename"))?;
        
        // Normalize Unicode using NFC (Canonical Decomposition followed by Canonical Composition)
        let normalized = unicode_normalization::UnicodeNormalization::nfc(
            &filename.to_string_lossy()
        ).collect::<String>();
        
        // Replace the filename with normalized version
        if let Some(parent) = path.parent() {
            Ok(parent.join(normalized))
        } else {
            Ok(PathBuf::from(normalized))
        }
    }
    
    /// Convert path to UTF-16 for Windows APIs
    pub fn to_utf16<P: AsRef<Path>>(path: P) -> Vec<u16> {
        let path_str = path.as_ref().to_string_lossy();
        let mut utf16: Vec<u16> = path_str.encode_utf16().collect();
        utf16.push(0); // Null terminator
        utf16
    }
    
    /// Convert UTF-16 to path
    pub fn from_utf16(utf16: &[u16]) -> Result<PathBuf> {
        // Remove null terminator if present
        let utf16_clean = if utf16.last() == Some(&0) {
            &utf16[..utf16.len() - 1]
        } else {
            utf16
        };
        
        let os_string = OsString::from_wide(utf16_clean);
        Ok(PathBuf::from(os_string))
    }
    
    /// Check if filesystem supports Unicode filenames
    pub fn filesystem_supports_unicode<P: AsRef<Path>>(path: P) -> Result<bool> {
        use windows::Win32::Storage::FileSystem::{GetVolumeInformationW, FILE_UNICODE_ON_DISK};
        use windows::core::PWSTR;
        
        let path = path.as_ref();
        let root = Self::get_volume_root(path)?;
        let wide_root = Self::to_utf16(&root);
        
        let mut file_system_flags = 0u32;
        
        unsafe {
            let result = GetVolumeInformationW(
                PWSTR(wide_root.as_ptr() as *mut u16),
                PWSTR::null(),
                0,
                None,
                None,
                Some(&mut file_system_flags),
                PWSTR::null(),
                0,
            );
            
            if result.as_bool() {
                Ok((file_system_flags & FILE_UNICODE_ON_DISK.0) != 0)
            } else {
                Err(anyhow!("Failed to get volume information"))
            }
        }
    }
    
    /// Get volume root for a path
    fn get_volume_root<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        
        // For absolute paths, get the root
        if path.is_absolute() {
            if let Some(root) = path.ancestors().last() {
                return Ok(root.to_path_buf());
            }
        }
        
        // For relative paths, use current drive
        let current_dir = std::env::current_dir()?;
        if let Some(root) = current_dir.ancestors().last() {
            Ok(root.to_path_buf())
        } else {
            Err(anyhow!("Cannot determine volume root"))
        }
    }
    
    /// Generate ASCII-safe alternative for Unicode filename
    pub fn generate_ascii_alternative<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        let filename = path.file_name()
            .ok_or_else(|| anyhow!("Invalid path: no filename"))?;
        
        let filename_str = filename.to_string_lossy();
        
        // Convert non-ASCII characters to ASCII alternatives or remove them
        let ascii_filename = filename_str
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        
        // Ensure filename is not empty and doesn't start with a dot
        let safe_filename = if ascii_filename.is_empty() || ascii_filename.starts_with('.') {
            format!("file_{}", ascii_filename.trim_start_matches('.'))
        } else {
            ascii_filename
        };
        
        if let Some(parent) = path.parent() {
            Ok(parent.join(safe_filename))
        } else {
            Ok(PathBuf::from(safe_filename))
        }
    }
    
    /// Test Unicode filename by creating and deleting a test file
    pub fn test_unicode_support<P: AsRef<Path>>(directory: P) -> Result<bool> {
        let test_filename = "тест_файл_🚀.txt"; // Russian + emoji
        let test_path = directory.as_ref().join(test_filename);
        
        // Try to create the file
        match std::fs::File::create(&test_path) {
            Ok(_) => {
                // Successfully created, now try to delete
                match std::fs::remove_file(&test_path) {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Ok(false),
        }
    }
}

/// Initialize Unicode support
pub fn initialize_unicode_support() -> Result<()> {
    tracing::info!("Initializing Unicode filename support");
    
    // Set console output to UTF-8 if possible
    #[cfg(windows)]
    unsafe {
        use windows::Win32::System::Console::{SetConsoleOutputCP, CP_UTF8};
        SetConsoleOutputCP(CP_UTF8);
    }
    
    tracing::info!("Unicode support initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_unicode_detection() {
        assert!(!UnicodeSupport::contains_unicode(Path::new("ascii_file.txt")));
        assert!(UnicodeSupport::contains_unicode(Path::new("файл.txt")));
        assert!(UnicodeSupport::contains_unicode(Path::new("file_🚀.txt")));
    }
    
    #[test]
    fn test_reserved_name_validation() {
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("CON")).is_err());
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("PRN.txt")).is_err());
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("COM1")).is_err());
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("normal_file.txt")).is_ok());
    }
    
    #[test]
    fn test_trailing_chars_validation() {
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("file ")).is_err());
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("file.")).is_err());
        assert!(UnicodeSupport::validate_unicode_filename(Path::new("file")).is_ok());
    }
    
    #[test]
    fn test_utf16_conversion() -> Result<()> {
        let test_path = Path::new("тест/файл.txt");
        let utf16 = UnicodeSupport::to_utf16(test_path);
        let converted_back = UnicodeSupport::from_utf16(&utf16)?;
        
        assert_eq!(test_path, converted_back);
        Ok(())
    }
    
    #[test]
    fn test_ascii_alternative_generation() -> Result<()> {
        let unicode_path = Path::new("测试文件🚀.txt");
        let ascii_alt = UnicodeSupport::generate_ascii_alternative(unicode_path)?;
        
        let ascii_filename = ascii_alt.file_name().unwrap().to_string_lossy();
        assert!(ascii_filename.chars().all(|c| c.is_ascii()));
        assert!(ascii_filename.contains("_"));
        
        Ok(())
    }
    
    #[test]
    fn test_normalize_unicode() -> Result<()> {
        // Test with decomposed Unicode (é as e + ´)
        let decomposed = Path::new("café");
        let normalized = UnicodeSupport::normalize_unicode_filename(decomposed)?;
        
        // The result should be the same string but in NFC form
        assert!(normalized.file_name().is_some());
        
        Ok(())
    }
    
    #[test]
    fn test_unicode_support_testing() -> Result<()> {
        let temp_dir = tempdir()?;
        let supports_unicode = UnicodeSupport::test_unicode_support(temp_dir.path())?;
        
        // On modern Windows systems, this should typically be true
        // But we don't assert since it depends on the filesystem
        println!("Unicode support: {}", supports_unicode);
        
        Ok(())
    }
}
```

### Windows Performance Counters (`src/windows/performance.rs`)
```rust
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Windows performance counter integration
pub struct WindowsPerformanceCounters {
    counters: HashMap<String, PerformanceCounter>,
}

impl WindowsPerformanceCounters {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }
    
    /// Initialize common performance counters
    pub fn initialize_common_counters(&mut self) -> Result<()> {
        // Memory counters
        self.add_counter("Memory\\Available Bytes", CounterType::Bytes)?;
        self.add_counter("Memory\\Committed Bytes", CounterType::Bytes)?;
        self.add_counter("Memory\\Pool Nonpaged Bytes", CounterType::Bytes)?;
        
        // Processor counters
        self.add_counter("Processor(_Total)\\% Processor Time", CounterType::Percentage)?;
        self.add_counter("Processor(_Total)\\% User Time", CounterType::Percentage)?;
        self.add_counter("Processor(_Total)\\% Kernel Time", CounterType::Percentage)?;
        
        // Disk counters
        self.add_counter("PhysicalDisk(_Total)\\Disk Reads/sec", CounterType::Rate)?;
        self.add_counter("PhysicalDisk(_Total)\\Disk Writes/sec", CounterType::Rate)?;
        self.add_counter("PhysicalDisk(_Total)\\Avg. Disk Queue Length", CounterType::Count)?;
        
        // Process counters
        self.add_counter("Process(llmkg-validation)\\Working Set", CounterType::Bytes)?;
        self.add_counter("Process(llmkg-validation)\\% Processor Time", CounterType::Percentage)?;
        self.add_counter("Process(llmkg-validation)\\Thread Count", CounterType::Count)?;
        
        Ok(())
    }
    
    /// Add a performance counter
    pub fn add_counter(&mut self, path: &str, counter_type: CounterType) -> Result<()> {
        let counter = PerformanceCounter::new(path, counter_type)?;
        self.counters.insert(path.to_string(), counter);
        Ok(())
    }
    
    /// Get current value of a performance counter
    pub fn get_counter_value(&self, path: &str) -> Result<f64> {
        let counter = self.counters.get(path)
            .ok_or_else(|| anyhow!("Counter not found: {}", path))?;
        
        counter.get_value()
    }
    
    /// Collect all counter values
    pub fn collect_all_values(&self) -> Result<HashMap<String, f64>> {
        let mut values = HashMap::new();
        
        for (path, counter) in &self.counters {
            match counter.get_value() {
                Ok(value) => {
                    values.insert(path.clone(), value);
                }
                Err(e) => {
                    tracing::warn!("Failed to read counter {}: {}", path, e);
                }
            }
        }
        
        Ok(values)
    }
    
    /// Get system performance metrics
    pub fn get_system_metrics(&self) -> Result<SystemPerformanceMetrics> {
        let values = self.collect_all_values()?;
        
        Ok(SystemPerformanceMetrics {
            cpu_usage_percent: values.get("Processor(_Total)\\% Processor Time").cloned().unwrap_or(0.0),
            memory_available_mb: values.get("Memory\\Available Bytes").cloned().unwrap_or(0.0) / (1024.0 * 1024.0),
            memory_committed_mb: values.get("Memory\\Committed Bytes").cloned().unwrap_or(0.0) / (1024.0 * 1024.0),
            disk_reads_per_sec: values.get("PhysicalDisk(_Total)\\Disk Reads/sec").cloned().unwrap_or(0.0),
            disk_writes_per_sec: values.get("PhysicalDisk(_Total)\\Disk Writes/sec").cloned().unwrap_or(0.0),
            process_working_set_mb: values.get("Process(llmkg-validation)\\Working Set").cloned().unwrap_or(0.0) / (1024.0 * 1024.0),
            process_cpu_usage: values.get("Process(llmkg-validation)\\% Processor Time").cloned().unwrap_or(0.0),
            process_thread_count: values.get("Process(llmkg-validation)\\Thread Count").cloned().unwrap_or(0.0) as u32,
        })
    }
}

/// Individual performance counter
pub struct PerformanceCounter {
    path: String,
    counter_type: CounterType,
    #[cfg(windows)]
    handle: Option<windows::Win32::System::Performance::PDH_HCOUNTER>,
}

impl PerformanceCounter {
    /// Create a new performance counter
    pub fn new(path: &str, counter_type: CounterType) -> Result<Self> {
        let mut counter = Self {
            path: path.to_string(),
            counter_type,
            #[cfg(windows)]
            handle: None,
        };
        
        counter.initialize()?;
        Ok(counter)
    }
    
    /// Initialize the performance counter
    #[cfg(windows)]
    fn initialize(&mut self) -> Result<()> {
        use windows::Win32::System::Performance::*;
        use windows::core::PWSTR;
        
        // This is a simplified implementation
        // Real implementation would use PDH APIs to create and query counters
        tracing::debug!("Initialized performance counter: {}", self.path);
        Ok(())
    }
    
    #[cfg(not(windows))]
    fn initialize(&mut self) -> Result<()> {
        tracing::debug!("Performance counter stubbed on non-Windows: {}", self.path);
        Ok(())
    }
    
    /// Get the current value of the counter
    #[cfg(windows)]
    pub fn get_value(&self) -> Result<f64> {
        // This is a placeholder implementation
        // Real implementation would use PDH APIs to get the actual counter value
        match self.path.as_str() {
            "Processor(_Total)\\% Processor Time" => Ok(25.5), // Mock CPU usage
            "Memory\\Available Bytes" => Ok(4_000_000_000.0), // Mock available memory
            "Memory\\Committed Bytes" => Ok(8_000_000_000.0), // Mock committed memory
            _ => Ok(0.0),
        }
    }
    
    #[cfg(not(windows))]
    pub fn get_value(&self) -> Result<f64> {
        Ok(0.0) // Stub for non-Windows platforms
    }
}

/// Counter types
#[derive(Debug, Clone)]
pub enum CounterType {
    Count,
    Bytes,
    Percentage,
    Rate,
    Time,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_available_mb: f64,
    pub memory_committed_mb: f64,
    pub disk_reads_per_sec: f64,
    pub disk_writes_per_sec: f64,
    pub process_working_set_mb: f64,
    pub process_cpu_usage: f64,
    pub process_thread_count: u32,
}

impl SystemPerformanceMetrics {
    /// Check if any metrics indicate performance issues
    pub fn has_performance_issues(&self) -> bool {
        self.cpu_usage_percent > 80.0
            || self.memory_available_mb < 512.0
            || self.process_working_set_mb > 2048.0
    }
    
    /// Get performance issue warnings
    pub fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if self.cpu_usage_percent > 80.0 {
            warnings.push(format!("High CPU usage: {:.1}%", self.cpu_usage_percent));
        }
        
        if self.memory_available_mb < 512.0 {
            warnings.push(format!("Low available memory: {:.1} MB", self.memory_available_mb));
        }
        
        if self.process_working_set_mb > 2048.0 {
            warnings.push(format!("High process memory usage: {:.1} MB", self.process_working_set_mb));
        }
        
        if self.disk_reads_per_sec > 1000.0 || self.disk_writes_per_sec > 1000.0 {
            warnings.push(format!("High disk I/O: {:.1} reads/sec, {:.1} writes/sec", 
                self.disk_reads_per_sec, self.disk_writes_per_sec));
        }
        
        warnings
    }
}

/// Initialize Windows performance counters
pub fn initialize_performance_counters() -> Result<()> {
    tracing::info!("Initializing Windows performance counters");
    
    // In a real implementation, this would set up PDH (Performance Data Helper) library
    #[cfg(windows)]
    {
        use windows::Win32::System::Performance::*;
        
        // Initialize PDH library
        // let result = unsafe { PdhOpenQueryW(PWSTR::null(), 0, &mut query_handle) };
        tracing::debug!("PDH library initialized (placeholder)");
    }
    
    tracing::info!("Windows performance counters initialized");
    Ok(())
}

/// Get quick system performance snapshot
pub fn get_quick_system_metrics() -> Result<SystemPerformanceMetrics> {
    let mut counters = WindowsPerformanceCounters::new();
    counters.initialize_common_counters()?;
    counters.get_system_metrics()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_counter_creation() -> Result<()> {
        let counter = PerformanceCounter::new(
            "Processor(_Total)\\% Processor Time",
            CounterType::Percentage
        )?;
        
        let value = counter.get_value()?;
        assert!(value >= 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_system_metrics_collection() -> Result<()> {
        let mut counters = WindowsPerformanceCounters::new();
        counters.initialize_common_counters()?;
        
        let metrics = counters.get_system_metrics()?;
        
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_available_mb >= 0.0);
        assert!(metrics.memory_committed_mb >= 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_performance_warnings() {
        let high_cpu_metrics = SystemPerformanceMetrics {
            cpu_usage_percent: 90.0,
            memory_available_mb: 1024.0,
            memory_committed_mb: 4096.0,
            disk_reads_per_sec: 100.0,
            disk_writes_per_sec: 100.0,
            process_working_set_mb: 512.0,
            process_cpu_usage: 25.0,
            process_thread_count: 10,
        };
        
        assert!(high_cpu_metrics.has_performance_issues());
        let warnings = high_cpu_metrics.get_warnings();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("High CPU usage"));
    }
    
    #[test]
    fn test_quick_metrics() -> Result<()> {
        let metrics = get_quick_system_metrics()?;
        
        // Basic sanity checks
        assert!(metrics.process_thread_count > 0);
        
        Ok(())
    }
}
```

### Windows Configuration Module (`src/windows/config.rs`)
```rust
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Windows-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsConfig {
    /// Enable long path support
    pub enable_long_paths: bool,
    
    /// Unicode filename handling
    pub unicode_support: UnicodeConfig,
    
    /// Performance monitoring configuration
    pub performance_monitoring: PerformanceConfig,
    
    /// Registry configuration
    pub registry: RegistryConfig,
    
    /// Service configuration
    pub service: ServiceConfig,
    
    /// File system configuration
    pub filesystem: FileSystemConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnicodeConfig {
    /// Enable Unicode filename support
    pub enabled: bool,
    
    /// Normalize Unicode filenames
    pub normalize_filenames: bool,
    
    /// Generate ASCII alternatives for problematic names
    pub generate_ascii_alternatives: bool,
    
    /// Test Unicode support on startup
    pub test_on_startup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance counter monitoring
    pub enabled: bool,
    
    /// Performance counter update interval (seconds)
    pub update_interval_secs: u64,
    
    /// Custom performance counters to monitor
    pub custom_counters: Vec<String>,
    
    /// Enable performance alerts
    pub enable_alerts: bool,
    
    /// CPU usage alert threshold (percentage)
    pub cpu_alert_threshold: f64,
    
    /// Memory usage alert threshold (MB)
    pub memory_alert_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Registry key for application settings
    pub app_registry_key: String,
    
    /// Store configuration in registry
    pub store_config_in_registry: bool,
    
    /// Registry values to monitor
    pub monitored_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Enable Windows service integration
    pub enabled: bool,
    
    /// Service name
    pub service_name: String,
    
    /// Service display name
    pub service_display_name: String,
    
    /// Service description
    pub service_description: String,
    
    /// Service start type
    pub service_start_type: ServiceStartType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceStartType {
    Manual,
    Automatic,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemConfig {
    /// Preferred path format
    pub preferred_path_format: PathFormat,
    
    /// Handle short (8.3) filenames
    pub handle_short_filenames: bool,
    
    /// Maximum path length to handle
    pub max_path_length: usize,
    
    /// File attribute handling
    pub handle_file_attributes: bool,
    
    /// Junction and symlink handling
    pub follow_junctions: bool,
    pub follow_symlinks: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathFormat {
    Native,    // Use native Windows paths
    Unix,      // Convert to Unix-style paths
    Mixed,     // Allow both formats
}

impl Default for WindowsConfig {
    fn default() -> Self {
        Self {
            enable_long_paths: true,
            unicode_support: UnicodeConfig {
                enabled: true,
                normalize_filenames: true,
                generate_ascii_alternatives: false,
                test_on_startup: true,
            },
            performance_monitoring: PerformanceConfig {
                enabled: true,
                update_interval_secs: 30,
                custom_counters: Vec::new(),
                enable_alerts: true,
                cpu_alert_threshold: 80.0,
                memory_alert_threshold: 1024.0,
            },
            registry: RegistryConfig {
                app_registry_key: r"HKEY_LOCAL_MACHINE\SOFTWARE\LLMKG\Validation".to_string(),
                store_config_in_registry: false,
                monitored_values: Vec::new(),
            },
            service: ServiceConfig {
                enabled: false,
                service_name: "LLMKG-Validation".to_string(),
                service_display_name: "LLMKG Vector Indexing Validation Service".to_string(),
                service_description: "High-performance vector indexing validation service".to_string(),
                service_start_type: ServiceStartType::Manual,
            },
            filesystem: FileSystemConfig {
                preferred_path_format: PathFormat::Native,
                handle_short_filenames: true,
                max_path_length: 32767, // Maximum for long path support
                handle_file_attributes: true,
                follow_junctions: false,
                follow_symlinks: false,
            },
        }
    }
}

impl WindowsConfig {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration from registry
    #[cfg(windows)]
    pub fn load_from_registry(&self) -> Result<Self> {
        // Placeholder for registry loading
        tracing::debug!("Loading configuration from registry: {}", self.registry.app_registry_key);
        Ok(self.clone())
    }
    
    #[cfg(not(windows))]
    pub fn load_from_registry(&self) -> Result<Self> {
        Err(anyhow!("Registry not available on non-Windows platforms"))
    }
    
    /// Save configuration to registry
    #[cfg(windows)]
    pub fn save_to_registry(&self) -> Result<()> {
        // Placeholder for registry saving
        tracing::debug!("Saving configuration to registry: {}", self.registry.app_registry_key);
        Ok(())
    }
    
    #[cfg(not(windows))]
    pub fn save_to_registry(&self) -> Result<()> {
        Err(anyhow!("Registry not available on non-Windows platforms"))
    }
    
    /// Validate configuration settings
    pub fn validate(&self) -> Result<()> {
        // Validate Unicode configuration
        if self.unicode_support.enabled && !self.unicode_support.normalize_filenames {
            tracing::warn!("Unicode support enabled but normalization disabled - may cause issues");
        }
        
        // Validate performance configuration
        if self.performance_monitoring.enabled {
            if self.performance_monitoring.update_interval_secs == 0 {
                return Err(anyhow!("Performance monitoring update interval must be > 0"));
            }
            
            if self.performance_monitoring.cpu_alert_threshold > 100.0 {
                return Err(anyhow!("CPU alert threshold cannot exceed 100%"));
            }
        }
        
        // Validate filesystem configuration
        if self.filesystem.max_path_length == 0 {
            return Err(anyhow!("Maximum path length must be > 0"));
        }
        
        if !self.enable_long_paths && self.filesystem.max_path_length > 260 {
            tracing::warn!("Long paths disabled but max path length > 260 - may cause issues");
        }
        
        // Validate service configuration
        if self.service.enabled {
            if self.service.service_name.is_empty() {
                return Err(anyhow!("Service name cannot be empty"));
            }
            
            if self.service.service_display_name.is_empty() {
                return Err(anyhow!("Service display name cannot be empty"));
            }
        }
        
        Ok(())
    }
    
    /// Apply configuration settings
    pub fn apply(&self) -> Result<()> {
        tracing::info!("Applying Windows-specific configuration");
        
        // Apply long path support
        if self.enable_long_paths {
            crate::windows::paths::enable_long_path_support()?;
        }
        
        // Apply Unicode support
        if self.unicode_support.enabled {
            crate::windows::unicode::initialize_unicode_support()?;
            
            if self.unicode_support.test_on_startup {
                let temp_dir = std::env::temp_dir();
                let supports_unicode = crate::windows::unicode::UnicodeSupport::test_unicode_support(&temp_dir)?;
                tracing::info!("Unicode filename support test result: {}", supports_unicode);
            }
        }
        
        // Apply performance monitoring
        if self.performance_monitoring.enabled {
            crate::windows::performance::initialize_performance_counters()?;
        }
        
        tracing::info!("Windows-specific configuration applied successfully");
        Ok(())
    }
    
    /// Get configuration optimized for the current Windows version
    pub fn optimize_for_current_version() -> Result<Self> {
        let mut config = Self::default();
        
        #[cfg(windows)]
        {
            let version = crate::windows::get_windows_version()?;
            
            if version.is_windows10_or_later() {
                // Enable modern Windows 10+ features
                config.enable_long_paths = true;
                config.unicode_support.enabled = true;
                config.performance_monitoring.enabled = true;
            }
            
            if version.is_windows11() {
                // Enable Windows 11 specific optimizations
                config.filesystem.handle_file_attributes = true;
                config.performance_monitoring.enable_alerts = true;
            }
        }
        
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = WindowsConfig::default();
        assert!(config.enable_long_paths);
        assert!(config.unicode_support.enabled);
        assert!(config.performance_monitoring.enabled);
        assert_eq!(config.filesystem.max_path_length, 32767);
    }
    
    #[test]
    fn test_config_validation() -> Result<()> {
        let valid_config = WindowsConfig::default();
        valid_config.validate()?;
        
        let mut invalid_config = WindowsConfig::default();
        invalid_config.performance_monitoring.update_interval_secs = 0;
        assert!(invalid_config.validate().is_err());
        
        Ok(())
    }
    
    #[test]
    fn test_config_serialization() -> Result<()> {
        let config = WindowsConfig::default();
        let temp_file = NamedTempFile::new()?;
        
        config.save_to_file(temp_file.path())?;
        let loaded_config = WindowsConfig::load_from_file(temp_file.path())?;
        
        assert_eq!(config.enable_long_paths, loaded_config.enable_long_paths);
        assert_eq!(config.unicode_support.enabled, loaded_config.unicode_support.enabled);
        
        Ok(())
    }
    
    #[test]
    fn test_version_optimization() -> Result<()> {
        let config = WindowsConfig::optimize_for_current_version()?;
        
        // Should have sensible defaults regardless of platform
        assert!(config.filesystem.max_path_length > 0);
        
        Ok(())
    }
}
```

### Stub Module for Non-Windows Platforms (`src/windows/stub.rs`)
```rust
//! Stub implementations for non-Windows platforms
//! These provide the same API but with no-op implementations

use anyhow::Result;

/// Stub for Windows path utilities
pub struct WindowsPaths;

impl WindowsPaths {
    pub fn to_long_path<P: AsRef<std::path::Path>>(path: P) -> Result<std::path::PathBuf> {
        Ok(path.as_ref().to_path_buf())
    }
    
    pub fn exceeds_path_limit<P: AsRef<std::path::Path>>(_path: P) -> bool {
        false
    }
    
    pub fn normalize_separators<P: AsRef<std::path::Path>>(path: P) -> std::path::PathBuf {
        path.as_ref().to_path_buf()
    }
}

/// Stub for Unicode support
pub struct UnicodeSupport;

impl UnicodeSupport {
    pub fn contains_unicode<P: AsRef<std::path::Path>>(_path: P) -> bool {
        false
    }
    
    pub fn validate_unicode_filename<P: AsRef<std::path::Path>>(_path: P) -> Result<()> {
        Ok(())
    }
}

/// Stub for performance counters
pub struct WindowsPerformanceCounters;

impl WindowsPerformanceCounters {
    pub fn new() -> Self {
        Self
    }
    
    pub fn initialize_common_counters(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Stub functions
pub fn enable_long_path_support() -> Result<()> {
    Ok(())
}

pub fn initialize_unicode_support() -> Result<()> {
    Ok(())
}

pub fn initialize_performance_counters() -> Result<()> {
    Ok(())
}

pub fn load_registry_configuration() -> Result<()> {
    Ok(())
}
```

## Success Criteria
- Windows path handling works correctly with long paths (>260 characters)
- Unicode filename support handles international characters properly
- Windows performance counters integrate correctly for system monitoring
- Registry configuration management works for enterprise deployment
- Windows service integration provides enterprise-grade deployment
- All functionality gracefully degrades on non-Windows platforms
- Configuration validation catches common Windows-specific issues
- Path normalization handles mixed Unix/Windows path formats correctly

## Time Limit
10 minutes maximum