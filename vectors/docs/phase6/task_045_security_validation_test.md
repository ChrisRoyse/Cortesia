# Task 045: Security Validation Test

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates security validation tests that validates the system is protected against injection attacks, DoS attempts, and malicious input.

## Project Structure
tests/
  security_validation_test.rs  <- Create this file
src/
  validation/
  lib.rs
Cargo.toml

## Task Description
Create comprehensive security validation tests that verify the system's resilience against various attack vectors including injection attempts, denial of service, and malicious input patterns.

## Requirements
1. Create comprehensive integration test
2. Test protection against injection attacks
3. Validate DoS protection mechanisms
4. Handle malicious input gracefully
5. Ensure 10-minute completion scope

## Expected Code Structure
```rust
use anyhow::Result;
use tempfile::TempDir;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

#[tokio::test]
async fn test_sql_injection_protection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test various SQL injection patterns
    let sql_injection_attempts = vec![
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM sensitive_data --",
        "'; DELETE FROM * WHERE 1=1; --",
        "' OR 1=1 LIMIT 1 --",
        "admin'--",
        "admin'/*",
        "' OR 'x'='x",
        "') OR ('1'='1",
        "' AND (SELECT COUNT(*) FROM users) > 0 --",
        "1' OR '1'='1')) /*",
        "' WAITFOR DELAY '00:00:05' --",
    ];
    
    for injection_attempt in sql_injection_attempts {
        let result = security_validator.search(injection_attempt).await;
        
        match result {
            Ok(search_results) => {
                // Should return safe, sanitized results - not execute injection
                assert!(search_results.len() < 1000, "Suspicious large result set from injection attempt");
                println!("SQL injection attempt handled safely: {}", injection_attempt);
            },
            Err(e) => {
                // Errors are acceptable for malicious input
                println!("SQL injection attempt rejected: {} -> {}", injection_attempt, e);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_xss_injection_protection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test various XSS injection patterns
    let xss_injection_attempts = vec![
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src=javascript:alert('XSS')></iframe>",
        "<body onload=alert('XSS')>",
        "<input onfocus=alert('XSS') autofocus>",
        "<select onfocus=alert('XSS') autofocus>",
        "<textarea onfocus=alert('XSS') autofocus>",
        "<details open ontoggle=alert('XSS')>",
        "<marquee onstart=alert('XSS')>",
        "\"'><script>alert('XSS')</script>",
    ];
    
    for xss_attempt in xss_injection_attempts {
        let result = security_validator.search(xss_attempt).await;
        
        match result {
            Ok(search_results) => {
                // Verify results don't contain executable code
                for result in &search_results {
                    assert!(!result.content.contains("<script"), "Unescaped script tag in results");
                    assert!(!result.content.contains("javascript:"), "Unescaped javascript protocol");
                    assert!(!result.content.contains("onerror="), "Unescaped event handler");
                }
                println!("XSS attempt handled safely: {}", xss_attempt);
            },
            Err(e) => {
                println!("XSS attempt rejected: {} -> {}", xss_attempt, e);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_path_traversal_protection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test various path traversal patterns
    let path_traversal_attempts = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..%252f..%252f..%252fetc%252fpasswd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
        "\\..\\..\\..\\etc\\passwd",
        "....\\\\....\\\\....\\\\etc\\\\passwd",
        "file:///etc/passwd",
        "file://c:/windows/system32/config/sam",
        "..../..../..../etc/passwd",
    ];
    
    for traversal_attempt in path_traversal_attempts {
        let result = security_validator.search_file_path(traversal_attempt).await;
        
        match result {
            Ok(files) => {
                // Should not access files outside allowed directories
                for file_path in &files {
                    assert!(!file_path.contains("/etc/"), "Path traversal to /etc/ detected");
                    assert!(!file_path.contains("\\windows\\"), "Path traversal to Windows directory detected");
                    assert!(!file_path.contains("system32"), "Path traversal to system32 detected");
                }
                println!("Path traversal attempt contained: {}", traversal_attempt);
            },
            Err(e) => {
                println!("Path traversal attempt blocked: {} -> {}", traversal_attempt, e);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dos_protection_query_length() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test extremely long queries
    let long_query_sizes = vec![10_000, 100_000, 1_000_000, 10_000_000];
    
    for size in long_query_sizes {
        let long_query = "a".repeat(size);
        let start_time = Instant::now();
        
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            security_validator.search(&long_query)
        ).await;
        
        let elapsed = start_time.elapsed();
        
        match result {
            Ok(Ok(_)) => {
                // Should complete quickly even for long queries
                assert!(elapsed < Duration::from_secs(5), 
                       "Long query ({} chars) took too much time: {:?}", size, elapsed);
                println!("Long query ({} chars) handled in {:?}", size, elapsed);
            },
            Ok(Err(e)) => {
                // Rejecting long queries is acceptable
                println!("Long query ({} chars) rejected: {}", size, e);
            },
            Err(_) => {
                // Timeout protection worked
                println!("Long query ({} chars) timed out (protection activated)", size);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dos_protection_complex_regex() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test catastrophic backtracking regex patterns
    let malicious_regex_patterns = vec![
        "(a+)+b",
        "(a*)*b",
        "(a|a)*b",
        "(a|b)*ababababababababababababac",
        "^(a+)+$",
        "([a-zA-Z]+)*$",
        "(x+x+)+y",
        "(a*)*$",
        "^(([a-z])+.)+[A-Z]([a-z])+$",
        "^((a|b)*)*$",
    ];
    
    let test_strings = vec![
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ababababababababababababab",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaax",
    ];
    
    for pattern in malicious_regex_patterns {
        for test_string in &test_strings {
            let regex_query = format!("regex:/{}/", pattern);
            let start_time = Instant::now();
            
            let result = tokio::time::timeout(
                Duration::from_secs(5),
                security_validator.search(&regex_query)
            ).await;
            
            let elapsed = start_time.elapsed();
            
            match result {
                Ok(Ok(_)) => {
                    assert!(elapsed < Duration::from_secs(2), 
                           "Regex DoS pattern completed but took too long: {:?}", elapsed);
                    println!("Regex pattern {} handled safely in {:?}", pattern, elapsed);
                },
                Ok(Err(e)) => {
                    println!("Malicious regex pattern {} rejected: {}", pattern, e);
                },
                Err(_) => {
                    println!("Regex pattern {} timed out (DoS protection activated)", pattern);
                }
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_dos_protection_rate_limiting() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test rate limiting by sending many requests rapidly
    let rapid_requests = 1000;
    let success_counter = Arc::new(AtomicU64::new(0));
    let rate_limited_counter = Arc::new(AtomicU64::new(0));
    let error_counter = Arc::new(AtomicU64::new(0));
    
    let start_time = Instant::now();
    let mut handles = Vec::new();
    
    for i in 0..rapid_requests {
        let security_validator_clone = security_validator.clone();
        let success_counter_clone = success_counter.clone();
        let rate_limited_counter_clone = rate_limited_counter.clone();
        let error_counter_clone = error_counter.clone();
        
        let handle = tokio::spawn(async move {
            let query = format!("rate_limit_test_{}", i);
            
            match security_validator_clone.search(&query).await {
                Ok(_) => success_counter_clone.fetch_add(1, Ordering::SeqCst),
                Err(e) => {
                    let error_msg = e.to_string().to_lowercase();
                    if error_msg.contains("rate limit") || error_msg.contains("too many requests") {
                        rate_limited_counter_clone.fetch_add(1, Ordering::SeqCst)
                    } else {
                        error_counter_clone.fetch_add(1, Ordering::SeqCst)
                    }
                }
            };
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await?;
    }
    
    let total_time = start_time.elapsed();
    let successful_requests = success_counter.load(Ordering::SeqCst);
    let rate_limited_requests = rate_limited_counter.load(Ordering::SeqCst);
    let other_errors = error_counter.load(Ordering::SeqCst);
    
    let total_processed = successful_requests + rate_limited_requests + other_errors;
    let actual_qps = successful_requests as f64 / total_time.as_secs_f64();
    
    println!("Rate limiting test: {}/{} successful, {} rate-limited, {} other errors, {:.1} QPS", 
             successful_requests, total_processed, rate_limited_requests, other_errors, actual_qps);
    
    // Should either limit rate or handle all requests efficiently
    if rate_limited_requests > 0 {
        println!("Rate limiting activated successfully");
    } else {
        assert!(actual_qps >= 100.0, "System should handle rapid requests efficiently: {:.1} QPS", actual_qps);
    }
    
    assert!(other_errors < rapid_requests / 10, "Too many unexpected errors: {}", other_errors);
    
    Ok(())
}

#[tokio::test]
async fn test_malicious_unicode_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test various malicious Unicode patterns
    let malicious_unicode_patterns = vec![
        "\u{202E}override\u{202C}", // Right-to-Left Override
        "\u{2066}test\u{2069}",      // Isolate characters
        "\u{200D}\u{200C}",          // Zero-width joiners
        "\u{FEFF}",                  // Byte Order Mark
        "\u{0000}",                  // Null character
        "\u{FFFF}",                  // Non-character
        "\u{FFF0}\u{FFF1}\u{FFF2}",  // More non-characters
        "test\u{0008}\u{007F}",      // Control characters
        "\u{1F4A9}".repeat(1000),    // Emoji bomb
        "\u{0041}\u{0300}".repeat(100), // Combining character attack
    ];
    
    for malicious_pattern in malicious_unicode_patterns {
        let start_time = Instant::now();
        
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            security_validator.search(&malicious_pattern)
        ).await;
        
        let elapsed = start_time.elapsed();
        
        match result {
            Ok(Ok(results)) => {
                // Should handle Unicode safely
                assert!(elapsed < Duration::from_secs(2), 
                       "Unicode pattern took too long: {:?}", elapsed);
                assert!(results.len() < 10000, "Suspicious large result set from Unicode pattern");
                println!("Malicious Unicode pattern handled safely in {:?}", elapsed);
            },
            Ok(Err(e)) => {
                println!("Malicious Unicode pattern rejected: {}", e);
            },
            Err(_) => {
                println!("Unicode pattern timed out (protection activated)");
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_input_validation_boundaries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test various input boundary conditions
    let boundary_test_cases = vec![
        ("", "empty input"),
        (" ".repeat(10000).as_str(), "whitespace bomb"),
        ("\n".repeat(1000).as_str(), "newline bomb"),
        ("\t".repeat(1000).as_str(), "tab bomb"),
        ("A".repeat(1000000).as_str(), "mega string"),
        ("🚀".repeat(10000).as_str(), "emoji flood"),
        ("\\".repeat(1000).as_str(), "escape flood"),
        ("\"".repeat(1000).as_str(), "quote flood"),
        ("'".repeat(1000).as_str(), "apostrophe flood"),
        (";".repeat(1000).as_str(), "semicolon flood"),
    ];
    
    for (test_input, description) in boundary_test_cases {
        let start_time = Instant::now();
        
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            security_validator.search(test_input)
        ).await;
        
        let elapsed = start_time.elapsed();
        
        match result {
            Ok(Ok(_)) => {
                assert!(elapsed < Duration::from_secs(5), 
                       "Boundary test '{}' took too long: {:?}", description, elapsed);
                println!("Boundary test '{}' handled in {:?}", description, elapsed);
            },
            Ok(Err(e)) => {
                println!("Boundary test '{}' rejected: {}", description, e);
            },
            Err(_) => {
                println!("Boundary test '{}' timed out (protection activated)", description);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_file_system_security() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let security_validator = SecurityValidator::new(temp_dir.path()).await?;
    
    // Test that system doesn't access unauthorized files
    let forbidden_paths = vec![
        "/etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\System32\\config\\SAM",
        "C:\\Windows\\System32\\config\\SYSTEM",
        "/proc/version",
        "/proc/cpuinfo",
        "/proc/meminfo",
        "../../../etc/passwd",
        "..\\..\\..\\Windows\\System32",
    ];
    
    for forbidden_path in forbidden_paths {
        // Try to search for content that might reveal system files
        let result = security_validator.search(&format!("path:{}", forbidden_path)).await;
        
        match result {
            Ok(results) => {
                // Should not return system file contents
                for result in &results {
                    assert!(!result.content.contains("root:x:"), "Detected /etc/passwd content");
                    assert!(!result.content.contains("Administrator"), "Detected Windows admin content");
                    assert!(!result.content.contains("Linux version"), "Detected system version info");
                }
                println!("Forbidden path search contained: {}", forbidden_path);
            },
            Err(e) => {
                println!("Forbidden path search blocked: {} -> {}", forbidden_path, e);
            }
        }
    }
    
    Ok(())
}
```

## Success Criteria
- All SQL injection attempts are safely handled or rejected
- XSS injection attempts don't produce executable content in results
- Path traversal attempts are blocked from accessing system files
- DoS protection activates for oversized queries (timeout < 10s)
- Complex regex patterns don't cause catastrophic backtracking
- Rate limiting protects against rapid request floods
- Malicious Unicode patterns are handled safely (< 2s processing)
- Input boundary conditions don't crash the system
- File system access is restricted to authorized directories only
- No sensitive system information leaked in search results

## Time Limit
10 minutes maximum