use crate::core::knowledge_types::{QuestionIntent, QuestionType, AnswerType, TimeRange};
use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    // Question word patterns
    static ref QUESTION_PATTERNS: Vec<(Regex, QuestionType)> = vec![
        (Regex::new(r"(?i)^what\s+").unwrap(), QuestionType::What),
        (Regex::new(r"(?i)^who\s+").unwrap(), QuestionType::Who),
        (Regex::new(r"(?i)^when\s+").unwrap(), QuestionType::When),
        (Regex::new(r"(?i)^where\s+").unwrap(), QuestionType::Where),
        (Regex::new(r"(?i)^why\s+").unwrap(), QuestionType::Why),
        (Regex::new(r"(?i)^how\s+").unwrap(), QuestionType::How),
        (Regex::new(r"(?i)^which\s+").unwrap(), QuestionType::Which),
        (Regex::new(r"(?i)^is\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^are\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^was\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^were\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^did\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^does\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^do\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^can\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^could\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^will\s+").unwrap(), QuestionType::Is),
        (Regex::new(r"(?i)^would\s+").unwrap(), QuestionType::Is),
    ];

    // Time-related patterns
    static ref TIME_INDICATORS: Regex = Regex::new(
        r"(?i)(in\s+\d{4}|during\s+\w+|before\s+\w+|after\s+\w+|between\s+\d{4}\s+and\s+\d{4}|from\s+\d{4}\s+to\s+\d{4})"
    ).unwrap();
}

pub struct QuestionParser;

impl QuestionParser {
    pub fn parse(question: &str) -> QuestionIntent {
        let question_type = Self::identify_question_type(question);
        let entities = Self::extract_entities(question);
        let expected_answer_type = Self::determine_answer_type(&question_type, question);
        let temporal_context = Self::extract_temporal_context(question);

        QuestionIntent {
            question_type,
            entities,
            expected_answer_type,
            temporal_context,
        }
    }

    fn identify_question_type(question: &str) -> QuestionType {
        for (pattern, qtype) in QUESTION_PATTERNS.iter() {
            if pattern.is_match(question) {
                return qtype.clone();
            }
        }
        
        // Default to What if no pattern matches
        QuestionType::What
    }

    fn extract_entities(question: &str) -> Vec<String> {
        let mut entities = Vec::new();
        
        // Remove the question word(s) to avoid false positives
        let clean_question = Self::remove_question_words(question);
        
        // Extract capitalized words (likely entities)
        let words: Vec<&str> = clean_question.split_whitespace().collect();
        let mut i = 0;
        
        while i < words.len() {
            let word = words[i];
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            
            if clean_word.is_empty() {
                i += 1;
                continue;
            }
            
            // Check if this word starts with uppercase (potential entity)
            if !clean_word.chars().next().unwrap().is_uppercase() {
                i += 1;
                continue;
            }
            
            // Skip if it's a stop word on its own
            if is_stop_word(clean_word) {
                i += 1;
                continue;
            }
            
            // Start of a potential multi-word entity
            let mut entity_words = vec![clean_word];
            let mut j = i + 1;
            
            // Continue collecting words while they're capitalized or connectors
            while j < words.len() {
                let next_word = words[j];
                let clean_next = next_word.trim_matches(|c: char| !c.is_alphanumeric());
                
                if clean_next.is_empty() {
                    j += 1;
                    continue;
                }
                
                // Special handling for "and" - it usually separates entities, not part of one
                if clean_next.to_lowercase() == "and" {
                    break;
                }
                
                // Check if it's a connector word (like "of", "the")
                if is_connector(clean_next) && j + 1 < words.len() {
                    // Look ahead to see if there's another capitalized word after the connector
                    let look_ahead = j + 1;
                    if look_ahead < words.len() {
                        let next_next = words[look_ahead].trim_matches(|c: char| !c.is_alphanumeric());
                        if !next_next.is_empty() && next_next.chars().next().unwrap().is_uppercase() {
                            entity_words.push(clean_next);
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }
                
                // Check if it's another capitalized word
                if clean_next.chars().next().unwrap().is_uppercase() {
                    entity_words.push(clean_next);
                    j += 1;
                } else {
                    break;
                }
            }
            
            let entity_name = entity_words.join(" ");
            if !entity_name.is_empty() && !is_stop_word(&entity_name) {
                entities.push(entity_name);
            }
            
            i = j;
        }
        
        // Extract quoted phrases
        let quote_pattern = Regex::new(r#"["']([^"']+)["']"#).unwrap();
        for cap in quote_pattern.captures_iter(question) {
            if let Some(match_) = cap.get(1) {
                entities.push(match_.as_str().to_string());
            }
        }
        
        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        entities.retain(|e| seen.insert(e.clone()));
        
        entities
    }

    fn determine_answer_type(question_type: &QuestionType, question: &str) -> AnswerType {
        let question_lower = question.to_lowercase();
        
        match question_type {
            QuestionType::What => {
                if question_lower.contains("time") || question_lower.contains("date") {
                    AnswerType::Time
                } else if question_lower.contains("number") || question_lower.contains("many") || question_lower.contains("much") {
                    AnswerType::Number
                } else if question_lower.contains("list") || question_lower.contains("examples") {
                    AnswerType::List
                } else {
                    AnswerType::Fact
                }
            }
            QuestionType::Who => AnswerType::Entity,
            QuestionType::When => AnswerType::Time,
            QuestionType::Where => AnswerType::Location,
            QuestionType::Why => AnswerType::Text,
            QuestionType::How => {
                if question_lower.contains("many") || question_lower.contains("much") {
                    AnswerType::Number
                } else {
                    AnswerType::Text
                }
            }
            QuestionType::Which => {
                if question_lower.contains("one") {
                    AnswerType::Entity
                } else {
                    AnswerType::List
                }
            }
            QuestionType::Is => AnswerType::Boolean,
        }
    }

    fn extract_temporal_context(question: &str) -> Option<TimeRange> {
        if let Some(captures) = TIME_INDICATORS.captures(question) {
            if let Some(match_) = captures.get(0) {
                let time_str = match_.as_str();
                
                // Parse different time patterns
                if time_str.contains("between") && time_str.contains("and") {
                    let parts: Vec<&str> = time_str.split(|c: char| !c.is_numeric()).filter(|s| !s.is_empty()).collect();
                    if parts.len() >= 2 {
                        return Some(TimeRange {
                            start: Some(parts[0].to_string()),
                            end: Some(parts[1].to_string()),
                        });
                    }
                } else if time_str.contains("from") && time_str.contains("to") {
                    let parts: Vec<&str> = time_str.split(|c: char| !c.is_numeric()).filter(|s| !s.is_empty()).collect();
                    if parts.len() >= 2 {
                        return Some(TimeRange {
                            start: Some(parts[0].to_string()),
                            end: Some(parts[1].to_string()),
                        });
                    }
                } else if let Some(year) = extract_year(time_str) {
                    return Some(TimeRange {
                        start: Some(year.clone()),
                        end: Some(year),
                    });
                }
            }
        }
        
        None
    }
    
    fn remove_question_words(question: &str) -> String {
        // Remove common question words from the beginning
        let question_word_pattern = Regex::new(
            r"(?i)^(what|who|when|where|why|how|which|is|are|was|were|did|does|do|can|could|will|would|has|have|had)\s+"
        ).unwrap();
        
        question_word_pattern.replace(question, "").to_string()
    }
}

fn is_connector(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "of" | "the" | "de" | "del" | "la" | "le" | "von" | "van" | "der"
    )
}

fn is_stop_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" |
        "of" | "with" | "by" | "from" | "as" | "is" | "was" | "are" | "were"
    )
}

fn extract_year(text: &str) -> Option<String> {
    let year_pattern = Regex::new(r"\b(19\d{2}|20\d{2})\b").unwrap();
    if let Some(captures) = year_pattern.captures(text) {
        if let Some(match_) = captures.get(0) {
            return Some(match_.as_str().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_what_question() {
        let intent = QuestionParser::parse("What did Einstein discover?");
        assert_eq!(intent.question_type, QuestionType::What);
        assert!(intent.entities.contains(&"Einstein".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Fact);
    }

    #[test]
    fn test_parse_who_question() {
        let intent = QuestionParser::parse("Who invented the telephone?");
        assert_eq!(intent.question_type, QuestionType::Who);
        assert_eq!(intent.expected_answer_type, AnswerType::Entity);
    }

    #[test]
    fn test_parse_when_question() {
        let intent = QuestionParser::parse("When was the Theory of Relativity published?");
        assert_eq!(intent.question_type, QuestionType::When);
        assert!(intent.entities.contains(&"Theory of Relativity".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Time);
    }

    #[test]
    fn test_parse_temporal_context() {
        let intent = QuestionParser::parse("What happened between 1900 and 1920?");
        assert!(intent.temporal_context.is_some());
        let range = intent.temporal_context.unwrap();
        assert_eq!(range.start, Some("1900".to_string()));
        assert_eq!(range.end, Some("1920".to_string()));
    }

    #[test]
    fn test_parse_is_question() {
        let intent = QuestionParser::parse("Is Einstein a physicist?");
        assert_eq!(intent.question_type, QuestionType::Is);
        assert!(intent.entities.contains(&"Einstein".to_string()));
        assert_eq!(intent.expected_answer_type, AnswerType::Boolean);
    }

    #[test]
    fn test_extract_multi_word_entities() {
        let intent = QuestionParser::parse("What is the Theory of Relativity?");
        assert!(intent.entities.contains(&"Theory of Relativity".to_string()));
    }
}