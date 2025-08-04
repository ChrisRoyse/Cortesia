# Task 013: Advanced Content Validation with Pattern Matching

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 005-012, specifically extending the CorrectnessValidator with sophisticated content validation capabilities. The ContentValidator provides fuzzy matching, regex patterns, context-aware validation, and advanced pattern matching with escape handling.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `ContentValidator` that provides robust content validation with support for fuzzy matching, regex patterns, context-aware validation (distinguishing code from comments), and advanced pattern matching with configurable thresholds.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement `ContentValidator` with fuzzy matching capabilities
3. Add regex pattern support for must_contain/must_not_contain rules
4. Include context-aware validation for different content types
5. Add partial match scoring with configurable thresholds
6. Support advanced pattern matching with escape handling
7. Provide detailed validation reporting with match locations

## Expected Code Structure to Add
```rust
use regex::Regex;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Code,
    Comment,
    Documentation,
    PlainText,
    Json,
    Xml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    Fuzzy { threshold: f64 },
    Regex { pattern: String },
    Contains,
    StartsWith,
    EndsWith,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub pattern: String,
    pub match_type: MatchType,
    pub must_contain: bool, // true for must_contain, false for must_not_contain
    pub content_type: Option<ContentType>,
    pub case_sensitive: bool,
    pub weight: f64, // Weight for scoring
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub rule_index: usize,
    pub matched: bool,
    pub score: f64,
    pub locations: Vec<MatchLocation>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchLocation {
    pub start: usize,
    pub end: usize,
    pub line_number: usize,
    pub column: usize,
    pub context: String, // Surrounding context
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentValidationResult {
    pub overall_score: f64,
    pub passed: bool,
    pub rule_results: Vec<MatchResult>,
    pub content_analysis: ContentAnalysis,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAnalysis {
    pub detected_type: ContentType,
    pub confidence: f64,
    pub line_count: usize,
    pub character_count: usize,
    pub code_blocks: Vec<(usize, usize)>, // (start_line, end_line) pairs
    pub comment_blocks: Vec<(usize, usize)>,
}

pub struct ContentValidator {
    rules: Vec<ValidationRule>,
    min_score_threshold: f64,
    fuzzy_match_cache: HashMap<String, f64>,
}

impl ContentValidator {
    pub fn new(min_score_threshold: f64) -> Self {
        Self {
            rules: Vec::new(),
            min_score_threshold,
            fuzzy_match_cache: HashMap::new(),
        }
    }
    
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }
    
    pub fn add_must_contain(&mut self, pattern: String, match_type: MatchType, weight: f64) {
        self.rules.push(ValidationRule {
            pattern,
            match_type,
            must_contain: true,
            content_type: None,
            case_sensitive: true,
            weight,
        });
    }
    
    pub fn add_must_not_contain(&mut self, pattern: String, match_type: MatchType, weight: f64) {
        self.rules.push(ValidationRule {
            pattern,
            match_type,
            must_contain: false,
            content_type: None,
            case_sensitive: true,
            weight,
        });
    }
    
    pub fn validate_content(&mut self, content: &str) -> ContentValidationResult {
        let content_analysis = self.analyze_content(content);
        let mut rule_results = Vec::new();
        let mut total_weighted_score = 0.0;
        let mut total_weight = 0.0;
        let mut recommendations = Vec::new();
        
        for (index, rule) in self.rules.iter().enumerate() {
            let match_result = self.apply_rule(content, rule, index, &content_analysis);
            
            // Calculate weighted contribution
            total_weighted_score += match_result.score * rule.weight;
            total_weight += rule.weight;
            
            // Generate recommendations based on results
            if !match_result.matched && rule.must_contain {
                recommendations.push(format!(
                    "Missing required pattern '{}' (Match type: {:?})", 
                    rule.pattern, rule.match_type
                ));
            } else if match_result.matched && !rule.must_contain {
                recommendations.push(format!(
                    "Found forbidden pattern '{}' at {} locations", 
                    rule.pattern, match_result.locations.len()
                ));
            }
            
            rule_results.push(match_result);
        }
        
        let overall_score = if total_weight > 0.0 {
            total_weighted_score / total_weight
        } else {
            1.0 // No rules means perfect score
        };
        
        let passed = overall_score >= self.min_score_threshold;
        
        if !passed {
            recommendations.push(format!(
                "Overall score {:.3} below threshold {:.3}", 
                overall_score, self.min_score_threshold
            ));
        }
        
        ContentValidationResult {
            overall_score,
            passed,
            rule_results,
            content_analysis,
            recommendations,
        }
    }
    
    fn analyze_content(&self, content: &str) -> ContentAnalysis {
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();
        let character_count = content.len();
        
        // Detect content type based on patterns
        let (detected_type, confidence) = self.detect_content_type(content);
        
        // Find code and comment blocks
        let mut code_blocks = Vec::new();
        let mut comment_blocks = Vec::new();
        
        let mut in_code_block = false;
        let mut in_comment_block = false;
        let mut current_block_start = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect code blocks (```lang or ``` patterns)
            if trimmed.starts_with("```") {
                if in_code_block {
                    code_blocks.push((current_block_start, i));
                    in_code_block = false;
                } else {
                    current_block_start = i;
                    in_code_block = true;
                }
            }
            
            // Detect comment blocks (/* */ or // patterns)
            if trimmed.starts_with("/*") && !in_comment_block {
                current_block_start = i;
                in_comment_block = true;
            } else if trimmed.ends_with("*/") && in_comment_block {
                comment_blocks.push((current_block_start, i));
                in_comment_block = false;
            } else if trimmed.starts_with("//") && !in_comment_block {
                comment_blocks.push((i, i));
            }
        }
        
        // Close any open blocks
        if in_code_block {
            code_blocks.push((current_block_start, line_count - 1));
        }
        if in_comment_block {
            comment_blocks.push((current_block_start, line_count - 1));
        }
        
        ContentAnalysis {
            detected_type,
            confidence,
            line_count,
            character_count,
            code_blocks,
            comment_blocks,
        }
    }
    
    fn detect_content_type(&self, content: &str) -> (ContentType, f64) {
        let mut scores = HashMap::new();
        
        // Check for code patterns
        let code_patterns = [
            r"\bfn\s+\w+\s*\(", // Rust functions
            r"\bclass\s+\w+", // Class definitions
            r"\bimport\s+", // Import statements
            r"\{[\s\S]*\}", // Braces
            r";$", // Semicolons at end of line
        ];
        
        let mut code_score = 0.0;
        for pattern in &code_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                code_score += regex.find_iter(content).count() as f64;
            }
        }
        
        // Check for documentation patterns
        let doc_patterns = [
            r"^#\s+", // Markdown headers
            r"\*\*.*\*\*", // Bold text
            r"\[.*\]\(.*\)", // Links
            r"```", // Code blocks
        ];
        
        let mut doc_score = 0.0;
        for pattern in &doc_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                doc_score += regex.find_iter(content).count() as f64;
            }
        }
        
        // Check for JSON
        if content.trim_start().starts_with('{') && content.trim_end().ends_with('}') {
            if serde_json::from_str::<serde_json::Value>(content).is_ok() {
                return (ContentType::Json, 0.95);
            }
        }
        
        // Check for XML
        if content.trim_start().starts_with('<') && content.contains("</") {
            return (ContentType::Xml, 0.9);
        }
        
        scores.insert(ContentType::Code, code_score);
        scores.insert(ContentType::Documentation, doc_score);
        scores.insert(ContentType::PlainText, 1.0); // Default fallback
        
        let max_entry = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        
        let confidence = (*max_entry.1 / (code_score + doc_score + 1.0)).min(1.0);
        (max_entry.0.clone(), confidence)
    }
    
    fn apply_rule(&mut self, content: &str, rule: &ValidationRule, index: usize, 
                  analysis: &ContentAnalysis) -> MatchResult {
        // Filter content by type if specified
        let target_content = if let Some(content_type) = &rule.content_type {
            self.extract_content_by_type(content, content_type, analysis)
        } else {
            content.to_string()
        };
        
        let search_content = if rule.case_sensitive {
            target_content
        } else {
            target_content.to_lowercase()
        };
        
        let search_pattern = if rule.case_sensitive {
            rule.pattern.clone()
        } else {
            rule.pattern.to_lowercase()
        };
        
        match &rule.match_type {
            MatchType::Exact => self.exact_match(&search_content, &search_pattern, rule, index),
            MatchType::Fuzzy { threshold } => {
                self.fuzzy_match(&search_content, &search_pattern, *threshold, rule, index)
            },
            MatchType::Regex { pattern } => {
                self.regex_match(&search_content, pattern, rule, index)
            },
            MatchType::Contains => self.contains_match(&search_content, &search_pattern, rule, index),
            MatchType::StartsWith => self.starts_with_match(&search_content, &search_pattern, rule, index),
            MatchType::EndsWith => self.ends_with_match(&search_content, &search_pattern, rule, index),
        }
    }
    
    fn extract_content_by_type(&self, content: &str, content_type: &ContentType, 
                              analysis: &ContentAnalysis) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut result = String::new();
        
        match content_type {
            ContentType::Code => {
                for (start, end) in &analysis.code_blocks {
                    for i in *start..=(*end).min(lines.len() - 1) {
                        result.push_str(lines[i]);
                        result.push('\n');
                    }
                }
            },
            ContentType::Comment => {
                for (start, end) in &analysis.comment_blocks {
                    for i in *start..=(*end).min(lines.len() - 1) {
                        result.push_str(lines[i]);
                        result.push('\n');
                    }
                }
            },
            _ => return content.to_string(),
        }
        
        if result.is_empty() {
            content.to_string()
        } else {
            result
        }
    }
    
    fn exact_match(&self, content: &str, pattern: &str, rule: &ValidationRule, 
                   index: usize) -> MatchResult {
        let matched = content == pattern;
        let score = if matched == rule.must_contain { 1.0 } else { 0.0 };
        
        let locations = if matched {
            vec![MatchLocation {
                start: 0,
                end: content.len(),
                line_number: 1,
                column: 1,
                context: self.get_context(content, 0, content.len()),
            }]
        } else {
            Vec::new()
        };
        
        MatchResult {
            rule_index: index,
            matched,
            score,
            locations,
            error_message: if matched != rule.must_contain {
                Some(format!("Exact match {} for pattern '{}'", 
                           if matched { "found" } else { "not found" }, pattern))
            } else {
                None
            },
        }
    }
    
    fn fuzzy_match(&mut self, content: &str, pattern: &str, threshold: f64, 
                   rule: &ValidationRule, index: usize) -> MatchResult {
        let cache_key = format!("{}:{}", content, pattern);
        let similarity = if let Some(&cached_score) = self.fuzzy_match_cache.get(&cache_key) {
            cached_score
        } else {
            let score = self.calculate_levenshtein_similarity(content, pattern);
            self.fuzzy_match_cache.insert(cache_key, score);
            score
        };
        
        let matched = similarity >= threshold;
        let score = if matched == rule.must_contain { similarity } else { 1.0 - similarity };
        
        MatchResult {
            rule_index: index,
            matched,
            score,
            locations: if matched {
                vec![MatchLocation {
                    start: 0,
                    end: content.len(),
                    line_number: 1,
                    column: 1,
                    context: self.get_context(content, 0, content.len()),
                }]
            } else {
                Vec::new()
            },
            error_message: if matched != rule.must_contain {
                Some(format!("Fuzzy match similarity {:.3} {} threshold {:.3}", 
                           similarity, if matched { "above" } else { "below" }, threshold))
            } else {
                None
            },
        }
    }
    
    fn regex_match(&self, content: &str, pattern: &str, rule: &ValidationRule, 
                   index: usize) -> MatchResult {
        match Regex::new(pattern) {
            Ok(regex) => {
                let matches: Vec<_> = regex.find_iter(content).collect();
                let matched = !matches.is_empty();
                let score = if matched == rule.must_contain { 1.0 } else { 0.0 };
                
                let locations = matches.iter().map(|m| {
                    let (line_num, col) = self.get_line_column(content, m.start());
                    MatchLocation {
                        start: m.start(),
                        end: m.end(),
                        line_number: line_num,
                        column: col,
                        context: self.get_context(content, m.start(), m.end()),
                    }
                }).collect();
                
                MatchResult {
                    rule_index: index,
                    matched,
                    score,
                    locations,
                    error_message: if matched != rule.must_contain {
                        Some(format!("Regex pattern '{}' {} {} matches", 
                                   pattern, if matched { "found" } else { "found no" }, matches.len()))
                    } else {
                        None
                    },
                }
            },
            Err(e) => MatchResult {
                rule_index: index,
                matched: false,
                score: 0.0,
                locations: Vec::new(),
                error_message: Some(format!("Invalid regex pattern '{}': {}", pattern, e)),
            }
        }
    }
    
    fn contains_match(&self, content: &str, pattern: &str, rule: &ValidationRule, 
                     index: usize) -> MatchResult {
        let mut locations = Vec::new();
        let mut start = 0;
        
        while let Some(pos) = content[start..].find(pattern) {
            let absolute_pos = start + pos;
            let (line_num, col) = self.get_line_column(content, absolute_pos);
            locations.push(MatchLocation {
                start: absolute_pos,
                end: absolute_pos + pattern.len(),
                line_number: line_num,
                column: col,
                context: self.get_context(content, absolute_pos, absolute_pos + pattern.len()),
            });
            start = absolute_pos + 1;
        }
        
        let matched = !locations.is_empty();
        let score = if matched == rule.must_contain { 1.0 } else { 0.0 };
        
        MatchResult {
            rule_index: index,
            matched,
            score,
            locations,
            error_message: if matched != rule.must_contain {
                Some(format!("Pattern '{}' {} {} times", 
                           pattern, if matched { "found" } else { "not found" }, locations.len()))
            } else {
                None
            },
        }
    }
    
    fn starts_with_match(&self, content: &str, pattern: &str, rule: &ValidationRule, 
                        index: usize) -> MatchResult {
        let matched = content.starts_with(pattern);
        let score = if matched == rule.must_contain { 1.0 } else { 0.0 };
        
        MatchResult {
            rule_index: index,
            matched,
            score,
            locations: if matched {
                vec![MatchLocation {
                    start: 0,
                    end: pattern.len(),
                    line_number: 1,
                    column: 1,
                    context: self.get_context(content, 0, pattern.len()),
                }]
            } else {
                Vec::new()
            },
            error_message: if matched != rule.must_contain {
                Some(format!("Content {} start with '{}'", 
                           if matched { "does" } else { "does not" }, pattern))
            } else {
                None
            },
        }
    }
    
    fn ends_with_match(&self, content: &str, pattern: &str, rule: &ValidationRule, 
                      index: usize) -> MatchResult {
        let matched = content.ends_with(pattern);
        let score = if matched == rule.must_contain { 1.0 } else { 0.0 };
        
        let start_pos = if matched { content.len() - pattern.len() } else { 0 };
        
        MatchResult {
            rule_index: index,
            matched,
            score,
            locations: if matched {
                let (line_num, col) = self.get_line_column(content, start_pos);
                vec![MatchLocation {
                    start: start_pos,
                    end: content.len(),
                    line_number: line_num,
                    column: col,
                    context: self.get_context(content, start_pos, content.len()),
                }]
            } else {
                Vec::new()
            },
            error_message: if matched != rule.must_contain {
                Some(format!("Content {} end with '{}'", 
                           if matched { "does" } else { "does not" }, pattern))
            } else {
                None
            },
        }
    }
    
    fn calculate_levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 && len2 == 0 {
            return 1.0;
        }
        
        let max_len = len1.max(len2);
        if max_len == 0 {
            return 1.0;
        }
        
        let distance = self.levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }
    
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }
        
        matrix[len1][len2]
    }
    
    fn get_line_column(&self, content: &str, pos: usize) -> (usize, usize) {
        let prefix = &content[..pos];
        let line_num = prefix.matches('\n').count() + 1;
        let last_newline = prefix.rfind('\n').unwrap_or(0);
        let column = pos - last_newline;
        (line_num, column)
    }
    
    fn get_context(&self, content: &str, start: usize, end: usize) -> String {
        let context_size = 20;
        let content_start = start.saturating_sub(context_size);
        let content_end = (end + context_size).min(content.len());
        
        let mut context = String::new();
        if content_start > 0 {
            context.push_str("...");
        }
        context.push_str(&content[content_start..content_end]);
        if content_end < content.len() {
            context.push_str("...");
        }
        
        context
    }
}
```

## Success Criteria
- ContentValidator struct compiles without errors
- All match types (exact, fuzzy, regex, contains, starts_with, ends_with) work correctly
- Content type detection accurately identifies code, comments, documentation
- Fuzzy matching with configurable thresholds performs correctly
- Regex pattern matching handles complex patterns and edge cases
- Context-aware validation filters content by type appropriately
- Match location tracking provides accurate line/column information
- Performance is acceptable with caching for expensive operations

## Time Limit
10 minutes maximum