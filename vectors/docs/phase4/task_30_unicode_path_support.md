# Task 30: Implement Unicode Path Support

## Context
You are implementing Phase 4 of a vector indexing system. Reserved names checking has been implemented. Now you need to create comprehensive Unicode path support to handle international filenames, special Unicode characters, and ensure proper encoding handling across the indexing system.

## Current State
- `src/windows.rs` has reserved names checking with severity levels
- Comprehensive filename validation exists
- Extended path support is working
- Need robust Unicode handling for international filename support

## Task Objective
Implement comprehensive Unicode path support with proper encoding validation, normalization handling, bidirectional text support, and integration with the file indexing system for international files.

## Implementation Requirements

### 1. Add Unicode validation and normalization
Add these Unicode handling structures to `src/windows.rs`:
```rust
use unicode_normalization::{UnicodeNormalization, is_nfc};
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, PartialEq)]
pub enum UnicodePathError {
    InvalidUtf8 { position: usize, bytes: Vec<u8> },
    InvalidSurrogate { position: usize, code_point: u32 },
    BidirectionalOverride { position: usize, character: char },
    InvalidNormalization { form: String },
    ZeroWidthCharacter { position: usize, character: char },
    PrivateUseArea { position: usize, character: char },
    UnassignedCodePoint { position: usize, code_point: u32 },
    ExcessiveLength { visual_width: usize, max_width: usize },
}

#[derive(Debug, Clone)]
pub struct UnicodePathValidation {
    pub is_valid: bool,
    pub errors: Vec<UnicodePathError>,
    pub warnings: Vec<String>,
    pub normalized_path: Option<String>,
    pub visual_width: usize,
    pub contains_rtl: bool,
    pub contains_combining: bool,
}

impl UnicodePathValidation {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            normalized_path: None,
            visual_width: 0,
            contains_rtl: false,
            contains_combining: false,
        }
    }
    
    pub fn add_error(&mut self, error: UnicodePathError) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

impl std::fmt::Display for UnicodePathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnicodePathError::InvalidUtf8 { position, bytes } => {
                write!(f, "Invalid UTF-8 sequence at position {}: {:?}", position, bytes)
            }
            UnicodePathError::InvalidSurrogate { position, code_point } => {
                write!(f, "Invalid surrogate code point U+{:04X} at position {}", code_point, position)
            }
            UnicodePathError::BidirectionalOverride { position, character } => {
                write!(f, "Bidirectional text override character '{}' at position {}", character, position)
            }
            UnicodePathError::InvalidNormalization { form } => {
                write!(f, "Path is not in {} normalization form", form)
            }
            UnicodePathError::ZeroWidthCharacter { position, character } => {
                write!(f, "Zero-width character U+{:04X} at position {}", *character as u32, position)
            }
            UnicodePathError::PrivateUseArea { position, character } => {
                write!(f, "Private use area character U+{:04X} at position {}", *character as u32, position)
            }
            UnicodePathError::UnassignedCodePoint { position, code_point } => {
                write!(f, "Unassigned code point U+{:04X} at position {}", code_point, position)
            }
            UnicodePathError::ExcessiveLength { visual_width, max_width } => {
                write!(f, "Visual width {} exceeds maximum {}", visual_width, max_width)
            }
        }
    }
}
```

### 2. Add comprehensive Unicode validation methods
Add these validation methods to `WindowsPathHandler`:
```rust
impl WindowsPathHandler {
    pub fn validate_unicode_path(&self, path: &str) -> UnicodePathValidation {
        let mut validation = UnicodePathValidation::new();
        
        // Check for valid UTF-8
        if !path.is_ascii() {
            self.validate_utf8_encoding(path, &mut validation);
        }
        
        // Calculate visual width
        let visual_width = path.width();
        validation.visual_width = visual_width;
        
        // Check visual width limits
        const MAX_VISUAL_WIDTH: usize = 500;
        if visual_width > MAX_VISUAL_WIDTH {
            validation.add_error(UnicodePathError::ExcessiveLength {
                visual_width,
                max_width: MAX_VISUAL_WIDTH,
            });
        }
        
        // Validate each character
        for (pos, ch) in path.char_indices() {
            self.validate_unicode_character(ch, pos, &mut validation);
        }
        
        // Check normalization
        self.check_unicode_normalization(path, &mut validation);
        
        // Check for bidirectional text issues
        self.check_bidirectional_text(path, &mut validation);
        
        // Generate normalized version if needed
        if !validation.is_valid || !is_nfc(path) {
            validation.normalized_path = Some(self.normalize_unicode_path(path));
        }
        
        validation
    }
    
    fn validate_utf8_encoding(&self, path: &str, validation: &mut UnicodePathValidation) {
        let bytes = path.as_bytes();
        let mut pos = 0;
        
        while pos < bytes.len() {
            match std::str::from_utf8(&bytes[pos..]) {
                Ok(_) => break,
                Err(e) => {
                    let error_pos = pos + e.valid_up_to();
                    let invalid_bytes = if let Some(len) = e.error_len() {
                        bytes[error_pos..error_pos + len].to_vec()
                    } else {
                        bytes[error_pos..].to_vec()
                    };
                    
                    validation.add_error(UnicodePathError::InvalidUtf8 {
                        position: error_pos,
                        bytes: invalid_bytes,
                    });
                    
                    pos = error_pos + e.error_len().unwrap_or(1);
                }
            }
        }
    }
    
    fn validate_unicode_character(&self, ch: char, pos: usize, validation: &mut UnicodePathValidation) {
        let code_point = ch as u32;
        
        // Check for invalid surrogates
        if (0xD800..=0xDFFF).contains(&code_point) {
            validation.add_error(UnicodePathError::InvalidSurrogate {
                position: pos,
                code_point,
            });
            return;
        }
        
        // Check for bidirectional override characters
        if self.is_bidirectional_override(ch) {
            validation.add_error(UnicodePathError::BidirectionalOverride {
                position: pos,
                character: ch,
            });
        }
        
        // Check for zero-width characters
        if self.is_zero_width_character(ch) {
            validation.add_error(UnicodePathError::ZeroWidthCharacter {
                position: pos,
                character: ch,
            });
        }
        
        // Check for private use area
        if self.is_private_use_area(ch) {
            validation.add_warning(format!(
                "Private use area character U+{:04X} at position {} may not display consistently",
                code_point, pos
            ));
        }
        
        // Check for right-to-left characters
        if self.is_rtl_character(ch) {
            validation.contains_rtl = true;
            validation.add_warning(format!(
                "Right-to-left character '{}' at position {} may cause display issues",
                ch, pos
            ));
        }
        
        // Check for combining characters
        if self.is_combining_character(ch) {
            validation.contains_combining = true;
        }
        
        // Check for unassigned code points (basic check)
        if self.is_likely_unassigned(ch) {
            validation.add_warning(format!(
                "Code point U+{:04X} at position {} may be unassigned",
                code_point, pos
            ));
        }
    }
    
    fn check_unicode_normalization(&self, path: &str, validation: &mut UnicodePathValidation) {
        if !is_nfc(path) {
            validation.add_warning("Path is not in NFC normalization form".to_string());
        }
        
        // Check if different normalizations would produce different results
        let nfc = path.nfc().collect::<String>();
        let nfd = path.nfd().collect::<String>();
        let nfkc = path.nfkc().collect::<String>();
        let nfkd = path.nfkd().collect::<String>();
        
        if nfc != nfd {
            validation.add_warning("Path has different NFC and NFD forms".to_string());
        }
        
        if nfc != nfkc {
            validation.add_warning("Path has compatibility characters that normalize differently".to_string());
        }
        
        if path != nfc {
            validation.normalized_path = Some(nfc);
        }
    }
    
    fn check_bidirectional_text(&self, path: &str, validation: &mut UnicodePathValidation) {
        let mut has_ltr = false;
        let mut has_rtl = false;
        
        for ch in path.chars() {
            if self.is_ltr_character(ch) {
                has_ltr = true;
            } else if self.is_rtl_character(ch) {
                has_rtl = true;
            }
        }
        
        if has_ltr && has_rtl {
            validation.add_warning("Path contains mixed left-to-right and right-to-left text".to_string());
        }
    }
    
    fn normalize_unicode_path(&self, path: &str) -> String {
        // Normalize to NFC form
        let mut normalized = path.nfc().collect::<String>();
        
        // Remove bidirectional override characters
        normalized = normalized.chars()
            .filter(|&ch| !self.is_bidirectional_override(ch))
            .collect();
        
        // Remove zero-width characters
        normalized = normalized.chars()
            .filter(|&ch| !self.is_zero_width_character(ch))
            .collect();
        
        // Replace private use area characters
        normalized = normalized.chars()
            .map(|ch| if self.is_private_use_area(ch) { '_' } else { ch })
            .collect();
        
        normalized
    }
    
    // Character classification methods
    fn is_bidirectional_override(&self, ch: char) -> bool {
        matches!(ch as u32,
            0x202A | // LEFT-TO-RIGHT EMBEDDING
            0x202B | // RIGHT-TO-LEFT EMBEDDING
            0x202C | // POP DIRECTIONAL FORMATTING
            0x202D | // LEFT-TO-RIGHT OVERRIDE
            0x202E | // RIGHT-TO-LEFT OVERRIDE
            0x2066 | // LEFT-TO-RIGHT ISOLATE
            0x2067 | // RIGHT-TO-LEFT ISOLATE
            0x2068 | // FIRST STRONG ISOLATE
            0x2069   // POP DIRECTIONAL ISOLATE
        )
    }
    
    fn is_zero_width_character(&self, ch: char) -> bool {
        matches!(ch as u32,
            0x200B | // ZERO WIDTH SPACE
            0x200C | // ZERO WIDTH NON-JOINER
            0x200D | // ZERO WIDTH JOINER
            0x200E | // LEFT-TO-RIGHT MARK
            0x200F | // RIGHT-TO-LEFT MARK
            0xFEFF   // ZERO WIDTH NO-BREAK SPACE (BOM)
        )
    }
    
    fn is_private_use_area(&self, ch: char) -> bool {
        let code_point = ch as u32;
        matches!(code_point,
            0xE000..=0xF8FF |   // Private Use Area
            0xF0000..=0xFFFFD | // Supplementary Private Use Area-A
            0x100000..=0x10FFFD // Supplementary Private Use Area-B
        )
    }
    
    fn is_rtl_character(&self, ch: char) -> bool {
        // Basic RTL detection - Hebrew and Arabic ranges
        let code_point = ch as u32;
        matches!(code_point,
            0x0590..=0x05FF | // Hebrew
            0x0600..=0x06FF | // Arabic
            0x0700..=0x074F | // Syriac
            0x0750..=0x077F | // Arabic Supplement
            0x0780..=0x07BF | // Thaana
            0x08A0..=0x08FF | // Arabic Extended-A
            0xFB1D..=0xFB4F | // Hebrew Presentation Forms
            0xFB50..=0xFDFF | // Arabic Presentation Forms-A
            0xFE70..=0xFEFF   // Arabic Presentation Forms-B
        )
    }
    
    fn is_ltr_character(&self, ch: char) -> bool {
        // Basic LTR detection - Latin, Cyrillic, etc.
        let code_point = ch as u32;
        matches!(code_point,
            0x0041..=0x005A | // ASCII uppercase
            0x0061..=0x007A | // ASCII lowercase
            0x00C0..=0x024F | // Latin Extended
            0x0400..=0x04FF | // Cyrillic
            0x0500..=0x052F   // Cyrillic Supplement
        )
    }
    
    fn is_combining_character(&self, ch: char) -> bool {
        // Basic combining character detection
        let code_point = ch as u32;
        matches!(code_point,
            0x0300..=0x036F | // Combining Diacritical Marks
            0x1AB0..=0x1AFF | // Combining Diacritical Marks Extended
            0x1DC0..=0x1DFF | // Combining Diacritical Marks Supplement
            0x20D0..=0x20FF | // Combining Diacritical Marks for Symbols
            0xFE20..=0xFE2F   // Combining Half Marks
        )
    }
    
    fn is_likely_unassigned(&self, ch: char) -> bool {
        let code_point = ch as u32;
        // Check some ranges that are likely to contain unassigned code points
        // This is a simplified check - full implementation would need Unicode database
        matches!(code_point,
            0xFDD0..=0xFDEF | // Non-characters
            0xFFFE | 0xFFFF | // Non-characters
            0x1FFFE | 0x1FFFF | // Non-characters
            0x2FFFE | 0x2FFFF | // Non-characters
            0x3FFFE | 0x3FFFF | // Non-characters
            0x4FFFE | 0x4FFFF | // Non-characters
            0x5FFFE | 0x5FFFF | // Non-characters
            0x6FFFE | 0x6FFFF | // Non-characters
            0x7FFFE | 0x7FFFF | // Non-characters
            0x8FFFE | 0x8FFFF | // Non-characters
            0x9FFFE | 0x9FFFF | // Non-characters
            0xAFFFE | 0xAFFFF | // Non-characters
            0xBFFFE | 0xBFFFF | // Non-characters
            0xCFFFE | 0xCFFFF | // Non-characters
            0xDFFFE | 0xDFFFF | // Non-characters
            0xEFFFE | 0xEFFFF | // Non-characters
            0xFFFFE | 0xFFFFF | // Non-characters
            0x10FFFE | 0x10FFFF // Non-characters
        )
    }
}
```

### 3. Add path conversion and compatibility methods
Add these methods for handling different encodings and platforms:
```rust
impl WindowsPathHandler {
    pub fn convert_to_safe_ascii(&self, path: &str) -> String {
        let mut result = String::new();
        
        for ch in path.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ' ') {
                result.push(ch);
            } else if ch == '\\' || ch == '/' {
                result.push('\\'); // Normalize to Windows separator
            } else {
                // Convert non-ASCII to ASCII-safe representation
                match ch {
                    'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => result.push('a'),
                    'Á' | 'À' | 'Â' | 'Ä' | 'Ã' | 'Å' => result.push('A'),
                    'é' | 'è' | 'ê' | 'ë' => result.push('e'),
                    'É' | 'È' | 'Ê' | 'Ë' => result.push('E'),
                    'í' | 'ì' | 'î' | 'ï' => result.push('i'),
                    'Í' | 'Ì' | 'Î' | 'Ï' => result.push('I'),
                    'ó' | 'ò' | 'ô' | 'ö' | 'õ' => result.push('o'),
                    'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' => result.push('O'),
                    'ú' | 'ù' | 'û' | 'ü' => result.push('u'),
                    'Ú' | 'Ù' | 'Û' | 'Ü' => result.push('U'),
                    'ñ' => result.push('n'),
                    'Ñ' => result.push('N'),
                    'ç' => result.push('c'),
                    'Ç' => result.push('C'),
                    _ => {
                        // Use Unicode code point as fallback
                        result.push_str(&format!("U{:04X}", ch as u32));
                    }
                }
            }
        }
        
        result
    }
    
    pub fn detect_encoding_issues(&self, path: &Path) -> Vec<String> {
        let mut issues = Vec::new();
        
        let path_str = path.to_string_lossy();
        
        // Check if lossy conversion occurred
        if path_str.contains('\u{FFFD}') {
            issues.push("Path contains invalid UTF-8 sequences (shown as �)".to_string());
        }
        
        // Check for mixed separators
        if path_str.contains('/') && path_str.contains('\\') {
            issues.push("Path contains mixed forward and backward slashes".to_string());
        }
        
        // Check for potential encoding artifacts
        if path_str.contains("Ã") {
            issues.push("Path may contain UTF-8 encoding artifacts".to_string());
        }
        
        // Check for byte order marks
        if path_str.starts_with('\u{FEFF}') {
            issues.push("Path starts with byte order mark (BOM)".to_string());
        }
        
        issues
    }
    
    pub fn is_cross_platform_safe(&self, path: &str) -> bool {
        // Check if path would be safe on multiple platforms
        for ch in path.chars() {
            // Check for characters that are problematic on various platforms
            if matches!(ch,
                '<' | '>' | ':' | '"' | '|' | '?' | '*' | // Windows reserved
                '\0' | // Null character (all platforms)
                '\x01'..='\x1F' // Control characters
            ) {
                return false;
            }
        }
        
        // Check for names that are reserved on any platform
        for component in std::path::Path::new(path).components() {
            if let std::path::Component::Normal(os_str) = component {
                let component_str = os_str.to_string_lossy();
                
                // Check Windows reserved names
                if self.check_reserved_name(&component_str).is_reserved {
                    return false;
                }
                
                // Check for leading dots (hidden on Unix)
                if component_str.starts_with('.') && component_str.len() > 1 {
                    // This is acceptable but worth noting
                }
            }
        }
        
        true
    }
    
    pub fn get_unicode_statistics(&self, path: &str) -> UnicodeStatistics {
        let mut stats = UnicodeStatistics::new();
        
        stats.total_chars = path.chars().count();
        stats.total_bytes = path.len();
        stats.visual_width = path.width();
        
        for ch in path.chars() {
            if ch.is_ascii() {
                stats.ascii_chars += 1;
            } else {
                stats.non_ascii_chars += 1;
            }
            
            if self.is_combining_character(ch) {
                stats.combining_chars += 1;
            }
            
            if self.is_rtl_character(ch) {
                stats.rtl_chars += 1;
            }
            
            if self.is_private_use_area(ch) {
                stats.private_use_chars += 1;
            }
            
            if self.is_zero_width_character(ch) {
                stats.zero_width_chars += 1;
            }
        }
        
        stats.is_normalized = is_nfc(path);
        stats.encoding_efficiency = if stats.total_chars > 0 {
            stats.total_bytes as f64 / stats.total_chars as f64
        } else {
            0.0
        };
        
        stats
    }
}

#[derive(Debug, Clone)]
pub struct UnicodeStatistics {
    pub total_chars: usize,
    pub total_bytes: usize,
    pub visual_width: usize,
    pub ascii_chars: usize,
    pub non_ascii_chars: usize,
    pub combining_chars: usize,
    pub rtl_chars: usize,
    pub private_use_chars: usize,
    pub zero_width_chars: usize,
    pub is_normalized: bool,
    pub encoding_efficiency: f64,
}

impl UnicodeStatistics {
    pub fn new() -> Self {
        Self {
            total_chars: 0,
            total_bytes: 0,
            visual_width: 0,
            ascii_chars: 0,
            non_ascii_chars: 0,
            combining_chars: 0,
            rtl_chars: 0,
            private_use_chars: 0,
            zero_width_chars: 0,
            is_normalized: true,
            encoding_efficiency: 0.0,
        }
    }
    
    pub fn ascii_percentage(&self) -> f64 {
        if self.total_chars > 0 {
            (self.ascii_chars as f64 / self.total_chars as f64) * 100.0
        } else {
            0.0
        }
    }
    
    pub fn has_complex_text(&self) -> bool {
        self.combining_chars > 0 || self.rtl_chars > 0 || self.zero_width_chars > 0
    }
}
```

### 4. Add dependencies note
Add this note at the top of the file:
```rust
// Note: Add these dependencies to Cargo.toml:
// [dependencies]
// unicode-normalization = "0.1"
// unicode-width = "0.1"
```

### 5. Add comprehensive tests
Add these tests to the test module:
```rust
#[test]
fn test_basic_unicode_validation() {
    let handler = WindowsPathHandler::new();
    
    // ASCII path should be valid
    let validation = handler.validate_unicode_path("C:\\Documents\\file.txt");
    assert!(validation.is_valid);
    assert_eq!(validation.errors.len(), 0);
    
    // Unicode path should be valid
    let validation = handler.validate_unicode_path("C:\\Documents\\测试文件.txt");
    assert!(validation.is_valid);
    assert!(validation.visual_width > 0);
}

#[test]
fn test_bidirectional_text_detection() {
    let handler = WindowsPathHandler::new();
    
    // Path with RTL override
    let path_with_rtl_override = format!("C:\\test{}file.txt", '\u{202E}');
    let validation = handler.validate_unicode_path(&path_with_rtl_override);
    assert!(!validation.is_valid);
    assert!(validation.errors.iter().any(|e| matches!(e, UnicodePathError::BidirectionalOverride { .. })));
}

#[test]
fn test_zero_width_character_detection() {
    let handler = WindowsPathHandler::new();
    
    // Path with zero-width space
    let path_with_zwsp = format!("C:\\test{}file.txt", '\u{200B}');
    let validation = handler.validate_unicode_path(&path_with_zwsp);
    assert!(!validation.is_valid);
    assert!(validation.errors.iter().any(|e| matches!(e, UnicodePathError::ZeroWidthCharacter { .. })));
}

#[test]
fn test_unicode_normalization() {
    let handler = WindowsPathHandler::new();
    
    // Create unnormalized path (combining characters)
    let unnormalized = "C:\\café.txt"; // This might not be NFC
    let validation = handler.validate_unicode_path(unnormalized);
    
    // Even if valid, might have normalization info
    if let Some(normalized) = &validation.normalized_path {
        assert_ne!(normalized, unnormalized);
    }
}

#[test]
fn test_private_use_area_detection() {
    let handler = WindowsPathHandler::new();
    
    // Path with private use character
    let path_with_pua = format!("C:\\test{}file.txt", '\u{E000}');
    let validation = handler.validate_unicode_path(&path_with_pua);
    
    // Should be valid but have warnings
    assert!(validation.is_valid);
    assert!(!validation.warnings.is_empty());
}

#[test]
fn test_ascii_conversion() {
    let handler = WindowsPathHandler::new();
    
    let unicode_path = "C:\\Documentación\\café.txt";
    let ascii_safe = handler.convert_to_safe_ascii(unicode_path);
    
    assert!(!ascii_safe.contains('ó'));
    assert!(!ascii_safe.contains('é'));
    assert!(ascii_safe.is_ascii());
}

#[test]
fn test_encoding_issue_detection() {
    let handler = WindowsPathHandler::new();
    
    // Test with replacement character
    let path_with_replacement = std::path::Path::new("C:\\test\u{FFFD}file.txt");
    let issues = handler.detect_encoding_issues(path_with_replacement);
    
    assert!(!issues.is_empty());
    assert!(issues.iter().any(|issue| issue.contains("invalid UTF-8")));
}

#[test]
fn test_cross_platform_safety() {
    let handler = WindowsPathHandler::new();
    
    // Safe path
    assert!(handler.is_cross_platform_safe("C:\\Documents\\file.txt"));
    
    // Unsafe path with reserved character
    assert!(!handler.is_cross_platform_safety("C:\\Documents\\file<copy>.txt"));
    
    // Path with reserved name
    assert!(!handler.is_cross_platform_safe("C:\\Documents\\CON.txt"));
}

#[test]
fn test_unicode_statistics() {
    let handler = WindowsPathHandler::new();
    
    let mixed_path = "C:\\Documents\\测试file.txt";
    let stats = handler.get_unicode_statistics(mixed_path);
    
    assert!(stats.total_chars > 0);
    assert!(stats.ascii_chars > 0);
    assert!(stats.non_ascii_chars > 0);
    assert!(stats.ascii_percentage() < 100.0);
    assert!(stats.encoding_efficiency > 1.0); // Unicode chars take more bytes
}

#[test]
fn test_rtl_text_handling() {
    let handler = WindowsPathHandler::new();
    
    // Path with Hebrew text
    let hebrew_path = "C:\\Documents\\שלום.txt";
    let validation = handler.validate_unicode_path(hebrew_path);
    
    assert!(validation.contains_rtl);
    assert!(!validation.warnings.is_empty());
}

#[test]
fn test_combining_characters() {
    let handler = WindowsPathHandler::new();
    
    // Path with combining diacritical marks
    let path_with_combining = "C:\\Documents\\e\u{0301}.txt"; // e with acute accent
    let validation = handler.validate_unicode_path(path_with_combining);
    
    assert!(validation.contains_combining);
    assert!(validation.visual_width < validation.normalized_path.as_ref().unwrap_or(&path_with_combining.to_string()).len());
}

#[test]
fn test_visual_width_calculation() {
    let handler = WindowsPathHandler::new();
    
    // Test with wide characters (CJK)
    let cjk_path = "C:\\Documents\\中文文件.txt";
    let validation = handler.validate_unicode_path(cjk_path);
    
    // Visual width should be larger than character count for CJK
    assert!(validation.visual_width > cjk_path.chars().count());
}

#[test]
fn test_excessive_length_detection() {
    let handler = WindowsPathHandler::new();
    
    // Create very long path with wide characters
    let long_path = format!("C:\\{}.txt", "中文".repeat(300));
    let validation = handler.validate_unicode_path(&long_path);
    
    assert!(!validation.is_valid);
    assert!(validation.errors.iter().any(|e| matches!(e, UnicodePathError::ExcessiveLength { .. })));
}
```

## Success Criteria
- [ ] Comprehensive Unicode character validation and classification
- [ ] Proper normalization form detection and conversion
- [ ] Bidirectional text and complex script handling
- [ ] Cross-platform compatibility checking
- [ ] Visual width calculation for display purposes
- [ ] Encoding issue detection and reporting
- [ ] ASCII conversion for compatibility
- [ ] Statistical analysis of Unicode usage
- [ ] All tests pass including international character sets
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Unicode normalization is critical for consistent filename handling
- Bidirectional text can cause security issues (homograph attacks)
- Zero-width characters can hide malicious content
- Private use area characters may not display consistently
- Visual width differs from character count for CJK and other scripts
- Cross-platform compatibility requires careful character validation
- Different operating systems handle Unicode normalization differently
- File systems may have varying levels of Unicode support