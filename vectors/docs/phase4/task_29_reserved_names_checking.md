# Task 29: Implement Windows Reserved Names Checking

## Context
You are implementing Phase 4 of a vector indexing system. Comprehensive filename validation has been implemented. Now you need to create a robust reserved names checking system that handles all Windows reserved names, their variations, and provides intelligent handling for the indexing system.

## Current State
- `src/windows.rs` has comprehensive filename validation
- Basic reserved names checking exists in the validation system
- Extended path support and sanitization are working
- Need enhanced reserved names handling with context awareness

## Task Objective
Implement a comprehensive Windows reserved names checking system with detailed categorization, context-aware validation, and intelligent workarounds for the file indexing system.

## Implementation Requirements

### 1. Add comprehensive reserved names database
Add this enhanced reserved names system to `src/windows.rs`:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ReservedNameCategory {
    Device,          // CON, PRN, AUX, NUL
    SerialPort,      // COM1-COM9
    ParallelPort,    // LPT1-LPT9
    Historical,      // CLOCK$, CONFIG$, etc.
    WindowsSpecial,  // desktop.ini, thumbs.db, etc.
}

#[derive(Debug, Clone)]
pub struct ReservedNameInfo {
    pub name: &'static str,
    pub category: ReservedNameCategory,
    pub description: &'static str,
    pub case_sensitive: bool,
    pub extension_matters: bool,
}

impl WindowsPathHandler {
    fn get_reserved_names_database() -> &'static [ReservedNameInfo] {
        &[
            // Device names
            ReservedNameInfo {
                name: "CON",
                category: ReservedNameCategory::Device,
                description: "Console input/output device",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "PRN",
                category: ReservedNameCategory::Device,
                description: "Default printer device",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "AUX",
                category: ReservedNameCategory::Device,
                description: "Auxiliary device",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "NUL",
                category: ReservedNameCategory::Device,
                description: "Null device",
                case_sensitive: false,
                extension_matters: false,
            },
            
            // Serial ports
            ReservedNameInfo {
                name: "COM1",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 1",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM2",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 2",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM3",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 3",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM4",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 4",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM5",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 5",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM6",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 6",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM7",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 7",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM8",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 8",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "COM9",
                category: ReservedNameCategory::SerialPort,
                description: "Serial port 9",
                case_sensitive: false,
                extension_matters: false,
            },
            
            // Parallel ports
            ReservedNameInfo {
                name: "LPT1",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 1",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT2",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 2",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT3",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 3",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT4",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 4",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT5",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 5",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT6",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 6",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT7",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 7",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT8",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 8",
                case_sensitive: false,
                extension_matters: false,
            },
            ReservedNameInfo {
                name: "LPT9",
                category: ReservedNameCategory::ParallelPort,
                description: "Parallel port 9",
                case_sensitive: false,
                extension_matters: false,
            },
            
            // Historical/legacy names
            ReservedNameInfo {
                name: "CLOCK$",
                category: ReservedNameCategory::Historical,
                description: "System clock device (legacy)",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "CONFIG$",
                category: ReservedNameCategory::Historical,
                description: "System configuration (legacy)",
                case_sensitive: false,
                extension_matters: true,
            },
        ]
    }
    
    fn get_windows_special_files() -> &'static [ReservedNameInfo] {
        &[
            ReservedNameInfo {
                name: "desktop.ini",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Desktop configuration file",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "thumbs.db",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Thumbnail cache database",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "ehthumbs.db",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Enhanced thumbnail cache",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "ehthumbs_vista.db",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Vista thumbnail cache",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "folder.htt",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Folder customization template",
                case_sensitive: false,
                extension_matters: true,
            },
            ReservedNameInfo {
                name: "autorun.inf",
                category: ReservedNameCategory::WindowsSpecial,
                description: "Autorun configuration file",
                case_sensitive: false,
                extension_matters: true,
            },
        ]
    }
}
```

### 2. Add comprehensive reserved name checking
Add these enhanced checking methods:
```rust
#[derive(Debug, Clone)]
pub struct ReservedNameCheckResult {
    pub is_reserved: bool,
    pub matched_info: Option<ReservedNameInfo>,
    pub suggested_alternatives: Vec<String>,
    pub severity: ReservedNameSeverity,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReservedNameSeverity {
    Critical,   // Will cause system errors
    High,       // May cause application issues
    Medium,     // May cause confusion or conflicts
    Low,        // Minor compatibility concerns
}

impl WindowsPathHandler {
    pub fn check_reserved_name(&self, filename: &str) -> ReservedNameCheckResult {
        // Check against main reserved names database
        if let Some(info) = self.find_reserved_name(filename, Self::get_reserved_names_database()) {
            return ReservedNameCheckResult {
                is_reserved: true,
                matched_info: Some(info.clone()),
                suggested_alternatives: self.generate_alternatives_for_reserved(filename, &info),
                severity: self.get_severity_for_category(&info.category),
                reason: format!("Matches reserved {} name: {}", 
                    self.category_description(&info.category), info.description),
            };
        }
        
        // Check against Windows special files
        if let Some(info) = self.find_reserved_name(filename, Self::get_windows_special_files()) {
            return ReservedNameCheckResult {
                is_reserved: true,
                matched_info: Some(info.clone()),
                suggested_alternatives: self.generate_alternatives_for_reserved(filename, &info),
                severity: ReservedNameSeverity::Medium,
                reason: format!("Matches Windows special file: {}", info.description),
            };
        }
        
        // Check for potential variants
        if let Some(variant_result) = self.check_reserved_name_variants(filename) {
            return variant_result;
        }
        
        ReservedNameCheckResult {
            is_reserved: false,
            matched_info: None,
            suggested_alternatives: Vec::new(),
            severity: ReservedNameSeverity::Low,
            reason: "Not a reserved name".to_string(),
        }
    }
    
    fn find_reserved_name(&self, filename: &str, database: &[ReservedNameInfo]) -> Option<&ReservedNameInfo> {
        for info in database {
            if self.matches_reserved_name(filename, info) {
                return Some(info);
            }
        }
        None
    }
    
    fn matches_reserved_name(&self, filename: &str, info: &ReservedNameInfo) -> bool {
        let filename_to_check = if info.case_sensitive {
            filename.to_string()
        } else {
            filename.to_uppercase()
        };
        
        let reserved_name = if info.case_sensitive {
            info.name.to_string()
        } else {
            info.name.to_uppercase()
        };
        
        if info.extension_matters {
            // Exact match required
            filename_to_check == reserved_name
        } else {
            // Match base name (before any extension)
            let base_name = filename_to_check.split('.').next().unwrap_or(&filename_to_check);
            base_name == reserved_name
        }
    }
    
    fn check_reserved_name_variants(&self, filename: &str) -> Option<ReservedNameCheckResult> {
        let upper_filename = filename.to_uppercase();
        
        // Check for numbered variants beyond the standard range
        if let Some(captures) = regex::Regex::new(r"^(COM|LPT)(\d+)").unwrap().captures(&upper_filename) {
            let device_type = &captures[1];
            let number: u32 = captures[2].parse().unwrap_or(0);
            
            if number > 9 {
                return Some(ReservedNameCheckResult {
                    is_reserved: true,
                    matched_info: None,
                    suggested_alternatives: vec![
                        format!("{}_{}", device_type.to_lowercase(), number),
                        format!("{}_port_{}", device_type.to_lowercase(), number),
                    ],
                    severity: ReservedNameSeverity::Medium,
                    reason: format!("Extended {} port number {}", device_type, number),
                });
            }
        }
        
        // Check for variations with spaces or special characters
        let cleaned = upper_filename.replace(&[' ', '-', '_'][..], "");
        for info in Self::get_reserved_names_database() {
            if cleaned == info.name && !info.extension_matters {
                return Some(ReservedNameCheckResult {
                    is_reserved: true,
                    matched_info: Some(info.clone()),
                    suggested_alternatives: self.generate_alternatives_for_reserved(filename, info),
                    severity: ReservedNameSeverity::High,
                    reason: format!("Variant of reserved name: {}", info.name),
                });
            }
        }
        
        None
    }
    
    fn generate_alternatives_for_reserved(&self, filename: &str, info: &ReservedNameInfo) -> Vec<String> {
        let mut alternatives = Vec::new();
        
        // Get base name and extension
        let (base, extension) = if let Some(dot_pos) = filename.rfind('.') {
            (&filename[..dot_pos], Some(&filename[dot_pos..]))
        } else {
            (filename, None)
        };
        
        // Generate alternatives based on category
        match info.category {
            ReservedNameCategory::Device => {
                alternatives.push(format!("{}_device", base.to_lowercase()));
                alternatives.push(format!("my_{}", base.to_lowercase()));
                alternatives.push(format!("{}_file", base.to_lowercase()));
            }
            ReservedNameCategory::SerialPort | ReservedNameCategory::ParallelPort => {
                alternatives.push(format!("{}_port", base.to_lowercase()));
                alternatives.push(format!("{}_{}", base.to_lowercase(), "interface"));
                alternatives.push(format!("virtual_{}", base.to_lowercase()));
            }
            ReservedNameCategory::Historical => {
                alternatives.push(format!("old_{}", base.to_lowercase()));
                alternatives.push(format!("{}_legacy", base.to_lowercase()));
            }
            ReservedNameCategory::WindowsSpecial => {
                alternatives.push(format!("custom_{}", base));
                alternatives.push(format!("user_{}", base));
                alternatives.push(format!("my_{}", base));
            }
        }
        
        // Add extension back if present
        if let Some(ext) = extension {
            alternatives = alternatives.into_iter()
                .map(|alt| format!("{}{}", alt, ext))
                .collect();
        }
        
        // Add generic alternatives
        alternatives.push(format!("_{}", filename));
        alternatives.push(format!("{}_", filename));
        
        // Remove duplicates and sort
        alternatives.sort();
        alternatives.dedup();
        
        alternatives
    }
    
    fn get_severity_for_category(&self, category: &ReservedNameCategory) -> ReservedNameSeverity {
        match category {
            ReservedNameCategory::Device => ReservedNameSeverity::Critical,
            ReservedNameCategory::SerialPort | ReservedNameCategory::ParallelPort => ReservedNameSeverity::High,
            ReservedNameCategory::Historical => ReservedNameSeverity::Medium,
            ReservedNameCategory::WindowsSpecial => ReservedNameSeverity::Medium,
        }
    }
    
    fn category_description(&self, category: &ReservedNameCategory) -> &'static str {
        match category {
            ReservedNameCategory::Device => "device",
            ReservedNameCategory::SerialPort => "serial port",
            ReservedNameCategory::ParallelPort => "parallel port",
            ReservedNameCategory::Historical => "historical/legacy",
            ReservedNameCategory::WindowsSpecial => "Windows special file",
        }
    }
    
    pub fn get_all_reserved_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        
        // Add all reserved names
        for info in Self::get_reserved_names_database() {
            names.push(info.name.to_string());
        }
        
        // Add Windows special files
        for info in Self::get_windows_special_files() {
            names.push(info.name.to_string());
        }
        
        names.sort();
        names
    }
    
    pub fn is_indexing_safe_name(&self, filename: &str) -> bool {
        let result = self.check_reserved_name(filename);
        
        // Only allow low severity or non-reserved names for indexing
        !result.is_reserved || result.severity == ReservedNameSeverity::Low
    }
    
    pub fn get_safe_indexing_alternative(&self, filename: &str) -> String {
        let result = self.check_reserved_name(filename);
        
        if result.is_reserved && result.severity != ReservedNameSeverity::Low {
            // Return the first suggested alternative or a safe default
            result.suggested_alternatives
                .into_iter()
                .next()
                .unwrap_or_else(|| format!("safe_{}", filename))
        } else {
            filename.to_string()
        }
    }
}
```

### 3. Add regex dependency
Add to the top of `src/windows.rs`:
```rust
// Note: You'll need to add regex to Cargo.toml:
// [dependencies]
// regex = "1.0"
```

### 4. Add bulk checking for directories
Add these methods for batch processing:
```rust
impl WindowsPathHandler {
    pub fn check_directory_for_reserved_names(&self, dir_path: &Path) -> Result<Vec<(PathBuf, ReservedNameCheckResult)>> {
        let mut results = Vec::new();
        
        if !dir_path.is_dir() {
            return Err(anyhow::anyhow!("Path is not a directory: {}", dir_path.display()));
        }
        
        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                let check_result = self.check_reserved_name(&filename_str);
                
                if check_result.is_reserved {
                    results.push((path, check_result));
                }
            }
        }
        
        Ok(results)
    }
    
    pub fn check_directory_recursively(&self, dir_path: &Path) -> Result<Vec<(PathBuf, ReservedNameCheckResult)>> {
        let mut all_results = Vec::new();
        
        fn visit_dir(
            handler: &WindowsPathHandler,
            dir: &Path,
            results: &mut Vec<(PathBuf, ReservedNameCheckResult)>
        ) -> Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if let Some(filename) = path.file_name() {
                    let filename_str = filename.to_string_lossy();
                    let check_result = handler.check_reserved_name(&filename_str);
                    
                    if check_result.is_reserved {
                        results.push((path.clone(), check_result));
                    }
                }
                
                if path.is_dir() {
                    visit_dir(handler, &path, results)?;
                }
            }
            Ok(())
        }
        
        visit_dir(self, dir_path, &mut all_results)?;
        Ok(all_results)
    }
    
    pub fn get_reserved_name_statistics(&self, dir_path: &Path) -> Result<ReservedNameStatistics> {
        let results = self.check_directory_recursively(dir_path)?;
        
        let mut stats = ReservedNameStatistics::new();
        
        for (_, result) in results {
            stats.total_reserved += 1;
            
            match result.severity {
                ReservedNameSeverity::Critical => stats.critical_count += 1,
                ReservedNameSeverity::High => stats.high_count += 1,
                ReservedNameSeverity::Medium => stats.medium_count += 1,
                ReservedNameSeverity::Low => stats.low_count += 1,
            }
            
            if let Some(info) = result.matched_info {
                *stats.category_counts.entry(info.category).or_insert(0) += 1;
            }
        }
        
        Ok(stats)
    }
}

#[derive(Debug, Clone)]
pub struct ReservedNameStatistics {
    pub total_reserved: usize,
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub category_counts: std::collections::HashMap<ReservedNameCategory, usize>,
}

impl ReservedNameStatistics {
    pub fn new() -> Self {
        Self {
            total_reserved: 0,
            critical_count: 0,
            high_count: 0,
            medium_count: 0,
            low_count: 0,
            category_counts: std::collections::HashMap::new(),
        }
    }
    
    pub fn has_critical_issues(&self) -> bool {
        self.critical_count > 0
    }
    
    pub fn needs_attention(&self) -> bool {
        self.critical_count > 0 || self.high_count > 0
    }
}
```

### 5. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_reserved_name_detection() {
    let handler = WindowsPathHandler::new();
    
    // Test device names
    let result = handler.check_reserved_name("CON");
    assert!(result.is_reserved);
    assert_eq!(result.severity, ReservedNameSeverity::Critical);
    
    let result = handler.check_reserved_name("con.txt");
    assert!(result.is_reserved);
    
    // Test serial ports
    let result = handler.check_reserved_name("COM1");
    assert!(result.is_reserved);
    assert_eq!(result.severity, ReservedNameSeverity::High);
    
    // Test case insensitivity
    let result = handler.check_reserved_name("prn");
    assert!(result.is_reserved);
    
    // Test valid names
    let result = handler.check_reserved_name("document.txt");
    assert!(!result.is_reserved);
}

#[test]
fn test_reserved_name_alternatives() {
    let handler = WindowsPathHandler::new();
    
    let result = handler.check_reserved_name("CON.txt");
    assert!(result.is_reserved);
    assert!(!result.suggested_alternatives.is_empty());
    assert!(result.suggested_alternatives.iter().any(|alt| alt.contains("con_device")));
}

#[test]
fn test_windows_special_files() {
    let handler = WindowsPathHandler::new();
    
    let result = handler.check_reserved_name("desktop.ini");
    assert!(result.is_reserved);
    assert_eq!(result.severity, ReservedNameSeverity::Medium);
    
    let result = handler.check_reserved_name("thumbs.db");
    assert!(result.is_reserved);
}

#[test]
fn test_reserved_name_variants() {
    let handler = WindowsPathHandler::new();
    
    // Test extended port numbers
    let result = handler.check_reserved_name("COM15");
    assert!(result.is_reserved);
    assert_eq!(result.severity, ReservedNameSeverity::Medium);
    
    // Test variations with separators
    let result = handler.check_reserved_name("C O N");
    assert!(result.is_reserved);
    assert_eq!(result.severity, ReservedNameSeverity::High);
}

#[test]
fn test_indexing_safety() {
    let handler = WindowsPathHandler::new();
    
    // Critical names should not be safe for indexing
    assert!(!handler.is_indexing_safe_name("CON"));
    assert!(!handler.is_indexing_safe_name("PRN.txt"));
    
    // Regular files should be safe
    assert!(handler.is_indexing_safe_name("document.pdf"));
    assert!(handler.is_indexing_safe_name("report.docx"));
}

#[test]
fn test_safe_alternatives() {
    let handler = WindowsPathHandler::new();
    
    let safe_name = handler.get_safe_indexing_alternative("CON.txt");
    assert_ne!(safe_name, "CON.txt");
    assert!(!handler.check_reserved_name(&safe_name).is_reserved);
}

#[test]
fn test_directory_checking() -> Result<()> {
    use tempfile::TempDir;
    
    let handler = WindowsPathHandler::new();
    let temp_dir = TempDir::new()?;
    
    // Create some test files
    std::fs::write(temp_dir.path().join("normal.txt"), "content")?;
    std::fs::write(temp_dir.path().join("CON.txt"), "content")?;
    std::fs::write(temp_dir.path().join("desktop.ini"), "content")?;
    
    let results = handler.check_directory_for_reserved_names(temp_dir.path())?;
    
    // Should find the reserved names
    assert_eq!(results.len(), 2); // CON.txt and desktop.ini
    
    Ok(())
}

#[test]
fn test_reserved_name_statistics() -> Result<()> {
    use tempfile::TempDir;
    
    let handler = WindowsPathHandler::new();
    let temp_dir = TempDir::new()?;
    
    // Create test files with different severity levels
    std::fs::write(temp_dir.path().join("CON.txt"), "content")?; // Critical
    std::fs::write(temp_dir.path().join("COM1.dat"), "content")?; // High
    std::fs::write(temp_dir.path().join("desktop.ini"), "content")?; // Medium
    
    let stats = handler.get_reserved_name_statistics(temp_dir.path())?;
    
    assert_eq!(stats.total_reserved, 3);
    assert_eq!(stats.critical_count, 1);
    assert_eq!(stats.high_count, 1);
    assert_eq!(stats.medium_count, 1);
    assert!(stats.has_critical_issues());
    assert!(stats.needs_attention());
    
    Ok(())
}

#[test]
fn test_all_reserved_names_list() {
    let handler = WindowsPathHandler::new();
    let all_names = handler.get_all_reserved_names();
    
    // Should include standard reserved names
    assert!(all_names.contains(&"CON".to_string()));
    assert!(all_names.contains(&"COM1".to_string()));
    assert!(all_names.contains(&"LPT1".to_string()));
    assert!(all_names.contains(&"desktop.ini".to_string()));
    
    // Should be sorted
    let mut sorted_names = all_names.clone();
    sorted_names.sort();
    assert_eq!(all_names, sorted_names);
}
```

## Success Criteria
- [ ] Comprehensive reserved names database with categorization
- [ ] Context-aware checking with severity levels
- [ ] Intelligent alternative name generation
- [ ] Batch processing for directory checking
- [ ] Statistical analysis of reserved name usage
- [ ] Integration with indexing safety checks
- [ ] All tests pass including edge cases
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Windows reserved names are case-insensitive and apply regardless of extension
- Some names like CLOCK$ and CONFIG$ are historical but may still cause issues
- Windows special files like desktop.ini have different handling requirements
- Extended port numbers (COM10+, LPT10+) are not reserved by Windows but may cause confusion
- Severity levels help prioritize which names need immediate attention during indexing
- Statistics help identify patterns in problematic filenames across large directory structures