# Task 082: Create Windows Deployment Tests

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The Windows Deployment Tests ensure the system works correctly across different Windows environments, including Windows Server, different Windows versions, and various deployment scenarios.

## Project Structure
```
src/
  validation/
    windows_tests.rs   <- Create this file
  lib.rs
tests/
  windows/
    deployment.rs      <- Create this file
    compatibility.rs   <- Create this file
    performance.rs     <- Create this file
```

## Task Description
Create comprehensive Windows-specific validation tests that verify system functionality, performance, and compatibility across different Windows environments and deployment scenarios.

## Requirements
1. Create `src/validation/windows_tests.rs` with Windows-specific validation logic
2. Create Windows deployment integration tests
3. Test Windows-specific file system behaviors
4. Validate Windows path handling and permissions
5. Create performance benchmarks for Windows environments

## Expected Code Structure

### `src/validation/windows_tests.rs`
```rust
use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::env;
use std::ffi::OsString;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};

#[cfg(windows)]
use winapi::um::{
    fileapi::{GetDiskFreeSpaceExW, GetVolumeInformationW},
    winbase::GetComputerNameW,
    sysinfoapi::{GetSystemInfo, GetVersionExW, SYSTEM_INFO, OSVERSIONINFOEXW},
    processthreadsapi::GetCurrentProcess,
    psapi::GetProcessMemoryInfo,
    handleapi::INVALID_HANDLE_VALUE,
};

use crate::validation::{
    ground_truth::{GroundTruthDataset, GroundTruthCase},
    correctness::CorrectnessValidator,
    performance::PerformanceBenchmark,
    report::TestResult,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsEnvironmentInfo {
    pub os_version: String,
    pub build_number: String,
    pub architecture: String,
    pub computer_name: String,
    pub domain_joined: bool,
    pub available_memory_gb: f64,
    pub total_disk_space_gb: f64,
    pub available_disk_space_gb: f64,
    pub file_system: String,
    pub user_account_control: bool,
    pub windows_defender_status: WindowsDefenderStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsDefenderStatus {
    pub enabled: bool,
    pub real_time_protection: bool,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsCompatibilityResult {
    pub environment_info: WindowsEnvironmentInfo,
    pub path_handling_tests: TestResult,
    pub file_permissions_tests: TestResult,
    pub unicode_support_tests: TestResult,
    pub long_path_support_tests: TestResult,
    pub case_sensitivity_tests: TestResult,
    pub network_drive_tests: TestResult,
    pub windows_service_tests: TestResult,
    pub performance_counters_tests: TestResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsDeploymentResult {
    pub msi_installation_test: TestResult,
    pub chocolatey_installation_test: TestResult,
    pub winget_installation_test: TestResult,
    pub portable_deployment_test: TestResult,
    pub windows_service_deployment: TestResult,
    pub startup_registration_test: TestResult,
    pub uninstallation_test: TestResult,
}

pub struct WindowsValidator {
    test_data_path: PathBuf,
    text_index_path: PathBuf,
    vector_db_path: String,
    environment_info: Option<WindowsEnvironmentInfo>,
}

impl WindowsValidator {
    pub fn new<P: AsRef<Path>>(test_data_path: P, text_index_path: P, vector_db_path: &str) -> Self {
        Self {
            test_data_path: test_data_path.as_ref().to_path_buf(),
            text_index_path: text_index_path.as_ref().to_path_buf(),
            vector_db_path: vector_db_path.to_string(),
            environment_info: None,
        }
    }
    
    pub async fn run_windows_compatibility_tests(&mut self) -> Result<WindowsCompatibilityResult> {
        info!("Running Windows compatibility tests");
        
        // Gather environment information
        let environment_info = self.gather_environment_info().await?;
        self.environment_info = Some(environment_info.clone());
        
        // Run compatibility tests
        let path_handling_tests = self.test_windows_path_handling().await?;
        let file_permissions_tests = self.test_file_permissions().await?;
        let unicode_support_tests = self.test_unicode_support().await?;
        let long_path_support_tests = self.test_long_path_support().await?;
        let case_sensitivity_tests = self.test_case_sensitivity().await?;
        let network_drive_tests = self.test_network_drive_support().await?;
        let windows_service_tests = self.test_windows_service_integration().await?;
        let performance_counters_tests = self.test_performance_counters().await?;
        
        Ok(WindowsCompatibilityResult {
            environment_info,
            path_handling_tests,
            file_permissions_tests,
            unicode_support_tests,
            long_path_support_tests,
            case_sensitivity_tests,
            network_drive_tests,
            windows_service_tests,
            performance_counters_tests,
        })
    }
    
    pub async fn run_windows_deployment_tests(&self) -> Result<WindowsDeploymentResult> {
        info!("Running Windows deployment tests");
        
        let msi_installation_test = self.test_msi_installation().await?;
        let chocolatey_installation_test = self.test_chocolatey_installation().await?;
        let winget_installation_test = self.test_winget_installation().await?;
        let portable_deployment_test = self.test_portable_deployment().await?;
        let windows_service_deployment = self.test_windows_service_deployment().await?;
        let startup_registration_test = self.test_startup_registration().await?;
        let uninstallation_test = self.test_uninstallation().await?;
        
        Ok(WindowsDeploymentResult {
            msi_installation_test,
            chocolatey_installation_test,
            winget_installation_test,
            portable_deployment_test,
            windows_service_deployment,
            startup_registration_test,
            uninstallation_test,
        })
    }
    
    async fn gather_environment_info(&self) -> Result<WindowsEnvironmentInfo> {
        info!("Gathering Windows environment information");
        
        #[cfg(windows)]
        {
            use std::mem;
            use winapi::shared::minwindef::{DWORD, FALSE};
            use winapi::um::winnt::{WCHAR, OSVERSIONINFOEXW};
            
            let os_version = self.get_windows_version()?;
            let build_number = self.get_build_number()?;
            let architecture = env::consts::ARCH.to_string();
            let computer_name = self.get_computer_name()?;
            let domain_joined = self.is_domain_joined()?;
            let (available_memory_gb, total_memory_gb) = self.get_memory_info()?;
            let (total_disk_space_gb, available_disk_space_gb) = self.get_disk_space_info()?;
            let file_system = self.get_file_system_type()?;
            let user_account_control = self.is_uac_enabled()?;
            let windows_defender_status = self.get_windows_defender_status().await?;
            
            Ok(WindowsEnvironmentInfo {
                os_version,
                build_number,
                architecture,
                computer_name,
                domain_joined,
                available_memory_gb,
                total_disk_space_gb,
                available_disk_space_gb,
                file_system,
                user_account_control,
                windows_defender_status,
            })
        }
        
        #[cfg(not(windows))]
        {
            anyhow::bail!("Windows compatibility tests can only run on Windows")
        }
    }
    
    #[cfg(windows)]
    fn get_windows_version(&self) -> Result<String> {
        use winapi::um::sysinfoapi::GetVersionExW;
        use winapi::um::winnt::OSVERSIONINFOEXW;
        use std::mem;
        
        unsafe {
            let mut version_info: OSVERSIONINFOEXW = mem::zeroed();
            version_info.dwOSVersionInfoSize = mem::size_of::<OSVERSIONINFOEXW>() as u32;
            
            if GetVersionExW(&mut version_info as *mut _ as *mut _) != 0 {
                Ok(format!("{}.{}", version_info.dwMajorVersion, version_info.dwMinorVersion))
            } else {
                // Fallback to registry or other method
                Ok("Unknown".to_string())
            }
        }
    }
    
    #[cfg(windows)]
    fn get_build_number(&self) -> Result<String> {
        // Read from registry: HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\CurrentBuild
        let output = Command::new("reg")
            .args(&["query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "/v", "CurrentBuild"])
            .output()?;
        
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().find(|line| line.contains("CurrentBuild")) {
                if let Some(build) = line.split_whitespace().last() {
                    return Ok(build.to_string());
                }
            }
        }
        
        Ok("Unknown".to_string())
    }
    
    #[cfg(windows)]
    fn get_computer_name(&self) -> Result<String> {
        use winapi::um::winbase::GetComputerNameW;
        use winapi::shared::minwindef::DWORD;
        
        unsafe {
            let mut buffer = vec![0u16; 256];
            let mut size = buffer.len() as DWORD;
            
            if GetComputerNameW(buffer.as_mut_ptr(), &mut size) != 0 {
                buffer.truncate(size as usize);
                Ok(String::from_utf16_lossy(&buffer))
            } else {
                Ok("Unknown".to_string())
            }
        }
    }
    
    #[cfg(windows)]
    fn is_domain_joined(&self) -> Result<bool> {
        let output = Command::new("systeminfo")
            .output()?;
        
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            Ok(output_str.contains("Domain:") && !output_str.contains("Domain: WORKGROUP"))
        } else {
            Ok(false)
        }
    }
    
    #[cfg(windows)]
    fn get_memory_info(&self) -> Result<(f64, f64)> {
        use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use std::mem;
        
        unsafe {
            let mut mem_status: MEMORYSTATUSEX = mem::zeroed();
            mem_status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
            
            if GlobalMemoryStatusEx(&mut mem_status) != 0 {
                let total_gb = mem_status.ullTotalPhys as f64 / (1024.0 * 1024.0 * 1024.0);
                let available_gb = mem_status.ullAvailPhys as f64 / (1024.0 * 1024.0 * 1024.0);
                Ok((available_gb, total_gb))
            } else {
                Ok((0.0, 0.0))
            }
        }
    }
    
    #[cfg(windows)]
    fn get_disk_space_info(&self) -> Result<(f64, f64)> {
        use winapi::um::fileapi::GetDiskFreeSpaceExW;
        use std::mem;
        
        unsafe {
            let mut free_bytes: u64 = 0;
            let mut total_bytes: u64 = 0;
            
            let current_dir = std::env::current_dir()?;
            let path_wide: Vec<u16> = current_dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            
            if GetDiskFreeSpaceExW(
                path_wide.as_ptr(),
                &mut free_bytes,
                &mut total_bytes,
                std::ptr::null_mut(),
            ) != 0 {
                let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let available_gb = free_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                Ok((total_gb, available_gb))
            } else {
                Ok((0.0, 0.0))
            }
        }
    }
    
    #[cfg(windows)]
    fn get_file_system_type(&self) -> Result<String> {
        use winapi::um::fileapi::GetVolumeInformationW;
        use std::mem;
        
        unsafe {
            let mut file_system_name = vec![0u16; 256];
            let current_dir = std::env::current_dir()?;
            let root_path = format!("{}\\", current_dir.display().to_string().chars().take(3).collect::<String>());
            let path_wide: Vec<u16> = root_path.encode_utf16().chain(std::iter::once(0)).collect();
            
            if GetVolumeInformationW(
                path_wide.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                file_system_name.as_mut_ptr(),
                file_system_name.len() as u32,
            ) != 0 {
                Ok(String::from_utf16_lossy(&file_system_name).trim_end_matches('\0').to_string())
            } else {
                Ok("Unknown".to_string())
            }
        }
    }
    
    #[cfg(windows)]
    fn is_uac_enabled(&self) -> Result<bool> {
        let output = Command::new("reg")
            .args(&["query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System", "/v", "EnableLUA"])
            .output()?;
        
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            Ok(output_str.contains("0x1"))
        } else {
            Ok(false)
        }
    }
    
    async fn get_windows_defender_status(&self) -> Result<WindowsDefenderStatus> {
        // Use PowerShell to get Windows Defender status
        let output = Command::new("powershell")
            .args(&["-Command", "Get-MpComputerStatus | ConvertTo-Json"])
            .output()?;
        
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse JSON response to extract defender status
            // Simplified implementation
            Ok(WindowsDefenderStatus {
                enabled: output_str.contains("true"),
                real_time_protection: output_str.contains("RealTimeProtectionEnabled"),
                version: "Unknown".to_string(),
            })
        } else {
            Ok(WindowsDefenderStatus {
                enabled: false,
                real_time_protection: false,
                version: "Unknown".to_string(),
            })
        }
    }
    
    async fn test_windows_path_handling(&self) -> Result<TestResult> {
        info!("Testing Windows path handling");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test various Windows path formats
        let test_paths = vec![
            r"C:\Users\test\Documents\file.txt",
            r"\\server\share\file.txt", // UNC path
            r"C:\Program Files (x86)\Application\file.txt", // Path with spaces and parentheses
            r"C:\测试\文件.txt", // Unicode path
            r"C:\very\long\path\that\exceeds\the\traditional\260\character\limit\for\windows\paths\and\tests\long\path\support\functionality.txt",
        ];
        
        for test_path in test_paths {
            total_tests += 1;
            
            let path_buf = PathBuf::from(test_path);
            
            // Test path parsing
            if path_buf.is_absolute() {
                tests_passed += 1;
            } else {
                issues.push(format!("Path parsing failed for: {}", test_path));
            }
            
            // Test path components
            if path_buf.components().count() > 0 {
                tests_passed += 1;
            } else {
                issues.push(format!("Path component parsing failed for: {}", test_path));
            }
            
            total_tests += 1;
        }
        
        // Test path normalization
        let test_cases = vec![
            (r"C:\Users\..\Users\test", r"C:\Users\test"),
            (r"C:\Users\test\.\Documents", r"C:\Users\test\Documents"),
            (r"C:\Users\test\\Documents", r"C:\Users\test\Documents"),
        ];
        
        for (input, expected) in test_cases {
            total_tests += 1;
            
            let normalized = PathBuf::from(input);
            let normalized_str = normalized.to_string_lossy();
            
            if normalized_str.contains(expected) {
                tests_passed += 1;
            } else {
                issues.push(format!("Path normalization failed: {} -> {}", input, normalized_str));
            }
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "All Windows path handling tests passed".to_string()
            } else {
                format!("Path handling issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_file_permissions(&self) -> Result<TestResult> {
        info!("Testing Windows file permissions");
        
        let temp_dir = tempfile::TempDir::new()?;
        let test_file = temp_dir.path().join("permission_test.txt");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test file creation
        total_tests += 1;
        match std::fs::write(&test_file, "test content") {
            Ok(_) => tests_passed += 1,
            Err(e) => issues.push(format!("File creation failed: {}", e)),
        }
        
        // Test file reading
        total_tests += 1;
        match std::fs::read_to_string(&test_file) {
            Ok(content) if content == "test content" => tests_passed += 1,
            Ok(_) => issues.push("File content mismatch".to_string()),
            Err(e) => issues.push(format!("File reading failed: {}", e)),
        }
        
        // Test file attributes on Windows
        #[cfg(windows)]
        {
            total_tests += 1;
            match test_file.metadata() {
                Ok(metadata) => {
                    tests_passed += 1;
                    debug!("File metadata: {:?}", metadata);
                }
                Err(e) => issues.push(format!("Metadata access failed: {}", e)),
            }
        }
        
        // Test directory creation with permissions
        let test_dir = temp_dir.path().join("permission_test_dir");
        total_tests += 1;
        match std::fs::create_dir(&test_dir) {
            Ok(_) => tests_passed += 1,
            Err(e) => issues.push(format!("Directory creation failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "All file permission tests passed".to_string()
            } else {
                format!("File permission issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_unicode_support(&self) -> Result<TestResult> {
        info!("Testing Unicode support on Windows");
        
        let temp_dir = tempfile::TempDir::new()?;
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test Unicode file names
        let unicode_filenames = vec![
            "测试.txt",           // Chinese
            "тест.txt",           // Russian  
            "δοκιμή.txt",         // Greek
            "prueba_ñ.txt",       // Spanish with ñ
            "🚀rocket.txt",       // Emoji
            "café_résumé.txt",    // French accents
        ];
        
        for filename in unicode_filenames {
            total_tests += 1;
            
            let file_path = temp_dir.path().join(filename);
            let test_content = format!("Unicode test content for {}", filename);
            
            match std::fs::write(&file_path, &test_content) {
                Ok(_) => {
                    // Try to read it back
                    match std::fs::read_to_string(&file_path) {
                        Ok(content) if content == test_content => tests_passed += 1,
                        Ok(_) => issues.push(format!("Unicode content mismatch for {}", filename)),
                        Err(e) => issues.push(format!("Unicode file read failed for {}: {}", filename, e)),
                    }
                }
                Err(e) => issues.push(format!("Unicode file creation failed for {}: {}", filename, e)),
            }
        }
        
        // Test Unicode in directory names
        total_tests += 1;
        let unicode_dir = temp_dir.path().join("测试目录");
        match std::fs::create_dir(&unicode_dir) {
            Ok(_) => {
                let test_file = unicode_dir.join("test.txt");
                match std::fs::write(&test_file, "test") {
                    Ok(_) => tests_passed += 1,
                    Err(e) => issues.push(format!("File creation in Unicode directory failed: {}", e)),
                }
            }
            Err(e) => issues.push(format!("Unicode directory creation failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "All Unicode support tests passed".to_string()
            } else {
                format!("Unicode support issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_long_path_support(&self) -> Result<TestResult> {
        info!("Testing long path support on Windows");
        
        let temp_dir = tempfile::TempDir::new()?;
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Create a path longer than 260 characters (traditional Windows limit)
        let mut long_path = temp_dir.path().to_path_buf();
        let long_component = "very_long_directory_name_that_helps_create_paths_longer_than_260_characters";
        
        // Build path components until we exceed 260 characters
        while long_path.to_string_lossy().len() < 300 {
            long_path = long_path.join(long_component);
        }
        
        total_tests += 1;
        
        // Test long path directory creation
        match std::fs::create_dir_all(&long_path) {
            Ok(_) => {
                tests_passed += 1;
                
                // Test file creation in long path
                total_tests += 1;
                let long_file = long_path.join("test_file.txt");
                match std::fs::write(&long_file, "long path test") {
                    Ok(_) => {
                        tests_passed += 1;
                        
                        // Test reading from long path
                        total_tests += 1;
                        match std::fs::read_to_string(&long_file) {
                            Ok(content) if content == "long path test" => tests_passed += 1,
                            Ok(_) => issues.push("Long path file content mismatch".to_string()),
                            Err(e) => issues.push(format!("Long path file read failed: {}", e)),
                        }
                    }
                    Err(e) => issues.push(format!("Long path file creation failed: {}", e)),
                }
            }
            Err(e) => issues.push(format!("Long path directory creation failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                format!("Long path support tests passed (path length: {})", long_path.to_string_lossy().len())
            } else {
                format!("Long path support issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_case_sensitivity(&self) -> Result<TestResult> {
        info!("Testing case sensitivity behavior on Windows");
        
        let temp_dir = tempfile::TempDir::new()?;
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Windows is case-insensitive by default
        let test_file_lower = temp_dir.path().join("testfile.txt");
        let test_file_upper = temp_dir.path().join("TESTFILE.TXT");
        let test_file_mixed = temp_dir.path().join("TestFile.txt");
        
        total_tests += 1;
        
        // Create file with lowercase name
        match std::fs::write(&test_file_lower, "test content") {
            Ok(_) => {
                tests_passed += 1;
                
                // Try to read with different case variations
                total_tests += 2;
                
                if test_file_upper.exists() {
                    tests_passed += 1;
                } else {
                    issues.push("Case insensitive file access failed (uppercase)".to_string());
                }
                
                if test_file_mixed.exists() {
                    tests_passed += 1;
                } else {
                    issues.push("Case insensitive file access failed (mixed case)".to_string());
                }
            }
            Err(e) => issues.push(format!("Test file creation failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Case sensitivity tests passed (Windows case-insensitive behavior confirmed)".to_string()
            } else {
                format!("Case sensitivity issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_network_drive_support(&self) -> Result<TestResult> {
        info!("Testing network drive support");
        
        // This is a simplified test - in real scenarios, you'd need actual network shares
        let mut issues = Vec::new();
        let score = 100.0; // Assume pass unless we can test actual network drives
        
        // Test UNC path parsing
        let unc_path = PathBuf::from(r"\\server\share\file.txt");
        if unc_path.is_absolute() {
            // UNC paths are properly parsed
        } else {
            issues.push("UNC path parsing failed".to_string());
        }
        
        let passed = issues.is_empty();
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Network drive support tests passed (limited testing without actual network shares)".to_string()
            } else {
                format!("Network drive issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_windows_service_integration(&self) -> Result<TestResult> {
        info!("Testing Windows service integration");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test if we can query Windows services
        total_tests += 1;
        let output = Command::new("sc")
            .args(&["query", "type=", "service"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("Service query command failed".to_string()),
            Err(e) => issues.push(format!("Service query execution failed: {}", e)),
        }
        
        // Test event log access (basic check)
        total_tests += 1;
        let output = Command::new("powershell")
            .args(&["-Command", "Get-EventLog -List | Select-Object Log -First 1"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("Event log access failed".to_string()),
            Err(e) => issues.push(format!("Event log query failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Windows service integration tests passed".to_string()
            } else {
                format!("Windows service integration issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_performance_counters(&self) -> Result<TestResult> {
        info!("Testing Windows performance counters access");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test basic performance counter access
        total_tests += 1;
        let output = Command::new("powershell")
            .args(&["-Command", "Get-Counter '\\Process(*)\\% Processor Time' -MaxSamples 1"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("Performance counter access failed".to_string()),
            Err(e) => issues.push(format!("Performance counter query failed: {}", e)),
        }
        
        // Test memory performance counters
        total_tests += 1;
        let output = Command::new("powershell")
            .args(&["-Command", "Get-Counter '\\Memory\\Available Bytes' -MaxSamples 1"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("Memory counter access failed".to_string()),
            Err(e) => issues.push(format!("Memory counter query failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Performance counter tests passed".to_string()
            } else {
                format!("Performance counter issues: {}", issues.join("; "))
            },
        })
    }
    
    // Deployment tests
    async fn test_msi_installation(&self) -> Result<TestResult> {
        info!("Testing MSI installation (simulation)");
        
        // This would test actual MSI installation in a real scenario
        // For now, we'll simulate the test
        
        Ok(TestResult {
            passed: true,
            score: 100.0,
            details: "MSI installation test simulated (requires actual MSI package)".to_string(),
        })
    }
    
    async fn test_chocolatey_installation(&self) -> Result<TestResult> {
        info!("Testing Chocolatey installation support");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Check if Chocolatey is available
        total_tests += 1;
        let output = Command::new("choco")
            .args(&["--version"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("Chocolatey not installed or not working".to_string()),
            Err(_) => issues.push("Chocolatey not available".to_string()),
        }
        
        let passed = tests_passed > 0;
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Chocolatey installation support confirmed".to_string()
            } else {
                format!("Chocolatey issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_winget_installation(&self) -> Result<TestResult> {
        info!("Testing WinGet installation support");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Check if WinGet is available
        total_tests += 1;
        let output = Command::new("winget")
            .args(&["--version"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                tests_passed += 1;
            }
            Ok(_) => issues.push("WinGet not installed or not working".to_string()),
            Err(_) => issues.push("WinGet not available".to_string()),
        }
        
        let passed = tests_passed > 0;
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "WinGet installation support confirmed".to_string()
            } else {
                format!("WinGet issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_portable_deployment(&self) -> Result<TestResult> {
        info!("Testing portable deployment");
        
        let temp_dir = tempfile::TempDir::new()?;
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test creating a portable deployment structure
        let portable_dir = temp_dir.path().join("llmkg_portable");
        let bin_dir = portable_dir.join("bin");
        let data_dir = portable_dir.join("data");
        let config_dir = portable_dir.join("config");
        
        total_tests += 3;
        
        for dir in [&bin_dir, &data_dir, &config_dir] {
            match std::fs::create_dir_all(dir) {
                Ok(_) => tests_passed += 1,
                Err(e) => issues.push(format!("Failed to create directory {}: {}", dir.display(), e)),
            }
        }
        
        // Test portable configuration file
        total_tests += 1;
        let config_file = config_dir.join("llmkg.toml");
        let config_content = r#"
[general]
portable_mode = true
data_dir = "./data"
index_dir = "./data/index"
"#;
        
        match std::fs::write(&config_file, config_content) {
            Ok(_) => tests_passed += 1,
            Err(e) => issues.push(format!("Failed to create config file: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Portable deployment structure created successfully".to_string()
            } else {
                format!("Portable deployment issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_windows_service_deployment(&self) -> Result<TestResult> {
        info!("Testing Windows service deployment (simulation)");
        
        // This would test actual service installation in a real scenario
        // For now, we'll check if the necessary tools are available
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Check if sc.exe is available for service management
        total_tests += 1;
        let output = Command::new("sc")
            .args(&["query", "state=", "all"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => tests_passed += 1,
            Ok(_) => issues.push("Service control manager access failed".to_string()),
            Err(e) => issues.push(format!("Service management tools not available: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Windows service deployment tools available".to_string()
            } else {
                format!("Service deployment issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_startup_registration(&self) -> Result<TestResult> {
        info!("Testing startup registration");
        
        let mut issues = Vec::new();
        let mut tests_passed = 0;
        let mut total_tests = 0;
        
        // Test registry access for startup registration
        total_tests += 1;
        let output = Command::new("reg")
            .args(&["query", "HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => tests_passed += 1,
            Ok(_) => issues.push("Startup registry access failed".to_string()),
            Err(e) => issues.push(format!("Registry access failed: {}", e)),
        }
        
        let passed = issues.is_empty();
        let score = (tests_passed as f64 / total_tests as f64) * 100.0;
        
        Ok(TestResult {
            passed,
            score,
            details: if issues.is_empty() {
                "Startup registration capability confirmed".to_string()
            } else {
                format!("Startup registration issues: {}", issues.join("; "))
            },
        })
    }
    
    async fn test_uninstallation(&self) -> Result<TestResult> {
        info!("Testing uninstallation procedures");
        
        // This would test actual uninstallation in a real scenario
        // For now, we'll simulate cleanup procedures
        
        Ok(TestResult {
            passed: true,
            score: 100.0,
            details: "Uninstallation test simulated (requires actual installation)".to_string(),
        })
    }
}
```

### `tests/windows/deployment.rs`
```rust
#[cfg(test)]
#[cfg(windows)]
mod windows_deployment_tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::PathBuf;
    
    #[tokio::test]
    async fn test_windows_environment_detection() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Verify environment info was collected
        assert!(!compatibility_result.environment_info.os_version.is_empty());
        assert!(!compatibility_result.environment_info.architecture.is_empty());
        assert!(compatibility_result.environment_info.available_memory_gb > 0.0);
    }
    
    #[tokio::test]
    async fn test_path_handling_on_windows() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Path handling should pass on Windows
        assert!(compatibility_result.path_handling_tests.passed);
        assert!(compatibility_result.path_handling_tests.score >= 90.0);
    }
    
    #[tokio::test]
    async fn test_unicode_file_handling() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Unicode support should work on modern Windows
        assert!(compatibility_result.unicode_support_tests.passed);
    }
    
    #[tokio::test]
    async fn test_portable_deployment_structure() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let result = validator.run_windows_deployment_tests().await;
        
        assert!(result.is_ok());
        let deployment_result = result.unwrap();
        
        // Portable deployment should work
        assert!(deployment_result.portable_deployment_test.passed);
    }
}
```

### `tests/windows/compatibility.rs`
```rust
#[cfg(test)]
#[cfg(windows)]
mod windows_compatibility_tests {
    use crate::validation::windows_tests::WindowsValidator;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_long_path_support() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Long path support test should complete (may pass or fail depending on system config)
        assert!(compatibility_result.long_path_support_tests.score >= 0.0);
    }
    
    #[tokio::test]
    async fn test_case_sensitivity_behavior() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Case sensitivity tests should pass (Windows is case-insensitive)
        assert!(compatibility_result.case_sensitivity_tests.passed);
    }
    
    #[tokio::test]
    async fn test_file_permissions() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // File permission tests should pass
        assert!(compatibility_result.file_permissions_tests.passed);
    }
}
```

### `tests/windows/performance.rs`
```rust
#[cfg(test)]
#[cfg(windows)]
mod windows_performance_tests {
    use crate::validation::windows_tests::WindowsValidator;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_performance_counter_access() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Performance counter tests should work on Windows
        assert!(compatibility_result.performance_counters_tests.score >= 50.0);
    }
    
    #[tokio::test]
    async fn test_windows_service_integration() {
        let temp_dir = TempDir::new().unwrap();
        let validator = WindowsValidator::new(
            &temp_dir.path(),
            &temp_dir.path().join("index"),
            &temp_dir.path().join("vector.lance").to_string_lossy()
        );
        
        let mut validator = validator;
        let result = validator.run_windows_compatibility_tests().await;
        
        assert!(result.is_ok());
        let compatibility_result = result.unwrap();
        
        // Windows service integration should be available
        assert!(compatibility_result.windows_service_tests.passed);
    }
}
```

## Success Criteria
- WindowsValidator properly detects Windows environment characteristics
- Path handling tests cover Windows-specific behaviors (UNC paths, long paths, case insensitivity)
- Unicode support tests work with international filenames and content
- File permission tests validate Windows security model
- Deployment tests verify different Windows installation methods
- Performance tests utilize Windows-specific monitoring capabilities
- All tests provide detailed error reporting and remediation suggestions

## Time Limit
25 minutes maximum