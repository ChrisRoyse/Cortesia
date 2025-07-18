use crate::core::triple::Triple;
use crate::error::Result;
use std::collections::HashMap;

/// Neural salience model for importance filtering
pub struct NeuralSalienceModel {
    importance_scorer: ImportanceScorer,
    content_filter: ContentFilter,
    threshold: f32,
}

impl NeuralSalienceModel {
    pub fn new(threshold: f32) -> Self {
        Self {
            importance_scorer: ImportanceScorer::new(),
            content_filter: ContentFilter::new(),
            threshold,
        }
    }

    pub async fn calculate_salience(&self, text: &str) -> Result<f32> {
        // Multi-factor salience calculation
        let importance_score = self.importance_scorer.score_text(text).await?;
        let content_score = self.content_filter.score_content(text).await?;
        
        // Weighted combination
        let salience = (importance_score * 0.7) + (content_score * 0.3);
        
        Ok(salience.max(0.0).min(1.0))
    }

    pub async fn should_store(&self, text: &str) -> Result<bool> {
        let salience = self.calculate_salience(text).await?;
        Ok(salience > self.threshold)
    }

    pub async fn score_triple(&self, triple: &Triple) -> Result<f32> {
        let natural_language = triple.to_natural_language();
        let text_salience = self.calculate_salience(&natural_language).await?;
        
        // Bonus for structured triples
        let structure_bonus = self.calculate_structure_bonus(triple);
        
        Ok((text_salience + structure_bonus).min(1.0))
    }

    fn calculate_structure_bonus(&self, triple: &Triple) -> f32 {
        let mut bonus = 0.0;
        
        // Bonus for proper nouns (likely entities)
        if triple.subject.chars().next().unwrap_or('a').is_uppercase() {
            bonus += 0.1;
        }
        if triple.object.chars().next().unwrap_or('a').is_uppercase() {
            bonus += 0.1;
        }
        
        // Bonus for important predicates
        match triple.predicate.as_str() {
            "invented" | "discovered" | "created" | "developed" => bonus += 0.2,
            "is" | "was" | "are" | "were" => bonus += 0.05,
            "born_in" | "died_in" | "works_at" => bonus += 0.1,
            _ => {}
        }
        
        // Penalty for very short or very long components
        if triple.subject.len() < 3 || triple.object.len() < 3 {
            bonus -= 0.1;
        }
        if triple.subject.len() > 50 || triple.object.len() > 50 {
            bonus -= 0.1;
        }
        
        bonus
    }
}

/// Importance scorer using multiple heuristics
pub struct ImportanceScorer {
    keyword_weights: HashMap<String, f32>,
    entity_patterns: Vec<regex::Regex>,
    concept_patterns: Vec<regex::Regex>,
}

impl ImportanceScorer {
    pub fn new() -> Self {
        let mut keyword_weights = HashMap::new();
        
        // High importance keywords
        keyword_weights.insert("invented".to_string(), 0.9);
        keyword_weights.insert("discovered".to_string(), 0.9);
        keyword_weights.insert("theory".to_string(), 0.8);
        keyword_weights.insert("principle".to_string(), 0.8);
        keyword_weights.insert("breakthrough".to_string(), 0.8);
        keyword_weights.insert("revolutionary".to_string(), 0.7);
        keyword_weights.insert("significant".to_string(), 0.6);
        keyword_weights.insert("important".to_string(), 0.6);
        keyword_weights.insert("major".to_string(), 0.5);
        keyword_weights.insert("notable".to_string(), 0.5);
        
        // Medium importance keywords
        keyword_weights.insert("created".to_string(), 0.6);
        keyword_weights.insert("developed".to_string(), 0.6);
        keyword_weights.insert("established".to_string(), 0.5);
        keyword_weights.insert("founded".to_string(), 0.5);
        keyword_weights.insert("published".to_string(), 0.4);
        keyword_weights.insert("wrote".to_string(), 0.4);
        
        // Low importance keywords
        keyword_weights.insert("mentioned".to_string(), 0.2);
        keyword_weights.insert("said".to_string(), 0.2);
        keyword_weights.insert("stated".to_string(), 0.2);
        keyword_weights.insert("noted".to_string(), 0.2);
        
        let entity_patterns = vec![
            regex::Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(), // Person names
            regex::Regex::new(r"\b[A-Z][a-z]+ (University|Institute|Foundation)\b").unwrap(), // Institutions
            regex::Regex::new(r"\b[A-Z][a-z]+ (Prize|Award|Medal)\b").unwrap(), // Awards
        ];
        
        let concept_patterns = vec![
            regex::Regex::new(r"\b[A-Z][a-z]+ (Theory|Principle|Law|Equation)\b").unwrap(), // Scientific concepts
            regex::Regex::new(r"\b[A-Z][a-z]+ (Algorithm|Method|Technique)\b").unwrap(), // Technical concepts
            regex::Regex::new(r"\b[A-Z][a-z]+ (Movement|Revolution|Era)\b").unwrap(), // Historical concepts
        ];
        
        Self {
            keyword_weights,
            entity_patterns,
            concept_patterns,
        }
    }

    pub async fn score_text(&self, text: &str) -> Result<f32> {
        let mut score = 0.0;
        let text_lower = text.to_lowercase();
        
        // Keyword-based scoring
        for (keyword, weight) in &self.keyword_weights {
            if text_lower.contains(keyword) {
                score += weight;
            }
        }
        
        // Entity-based scoring
        let entity_count = self.entity_patterns.iter()
            .map(|pattern| pattern.find_iter(text).count())
            .sum::<usize>();
        score += (entity_count as f32) * 0.1;
        
        // Concept-based scoring
        let concept_count = self.concept_patterns.iter()
            .map(|pattern| pattern.find_iter(text).count())
            .sum::<usize>();
        score += (concept_count as f32) * 0.15;
        
        // Length-based scoring (moderate length preferred)
        let length_score = self.calculate_length_score(text.len());
        score += length_score;
        
        // Complexity scoring (more complex sentences are more important)
        let complexity_score = self.calculate_complexity_score(text);
        score += complexity_score;
        
        Ok(score.min(1.0))
    }

    fn calculate_length_score(&self, length: usize) -> f32 {
        match length {
            0..=20 => 0.0,        // Too short
            21..=50 => 0.1,       // Short but okay
            51..=150 => 0.2,      // Good length
            151..=300 => 0.3,     // Optimal length
            301..=500 => 0.2,     // Long but acceptable
            501..=1000 => 0.1,    // Very long
            _ => 0.0,             // Too long
        }
    }

    fn calculate_complexity_score(&self, text: &str) -> f32 {
        let mut score = 0.0;
        
        // Count sentences
        let sentence_count = text.split('.').filter(|s| !s.trim().is_empty()).count();
        if sentence_count > 1 {
            score += 0.1;
        }
        
        // Count commas (indication of complex structure)
        let comma_count = text.chars().filter(|&c| c == ',').count();
        score += (comma_count as f32) * 0.02;
        
        // Count parentheses (additional information)
        let paren_count = text.chars().filter(|&c| c == '(' || c == ')').count();
        score += (paren_count as f32) * 0.01;
        
        // Count numbers and dates
        let number_count = text.split_whitespace()
            .filter(|word| word.chars().any(|c| c.is_numeric()))
            .count();
        score += (number_count as f32) * 0.05;
        
        score.min(0.3) // Cap complexity score
    }
}

/// Content filter to identify trivial or low-value content
pub struct ContentFilter {
    trivial_patterns: Vec<regex::Regex>,
    spam_patterns: Vec<regex::Regex>,
    quality_indicators: Vec<regex::Regex>,
}

impl ContentFilter {
    pub fn new() -> Self {
        let trivial_patterns = vec![
            regex::Regex::new(r"(?i)the sky is blue").unwrap(),
            regex::Regex::new(r"(?i)water is wet").unwrap(),
            regex::Regex::new(r"(?i)fire is hot").unwrap(),
            regex::Regex::new(r"(?i)grass is green").unwrap(),
            regex::Regex::new(r"(?i)snow is white").unwrap(),
            regex::Regex::new(r"(?i)ice is cold").unwrap(),
        ];
        
        let spam_patterns = vec![
            regex::Regex::new(r"(?i)click here").unwrap(),
            regex::Regex::new(r"(?i)buy now").unwrap(),
            regex::Regex::new(r"(?i)limited time").unwrap(),
            regex::Regex::new(r"(?i)free trial").unwrap(),
            regex::Regex::new(r"(?i)subscribe").unwrap(),
            regex::Regex::new(r"(?i)download now").unwrap(),
        ];
        
        let quality_indicators = vec![
            regex::Regex::new(r"(?i)according to").unwrap(),
            regex::Regex::new(r"(?i)research shows").unwrap(),
            regex::Regex::new(r"(?i)studies indicate").unwrap(),
            regex::Regex::new(r"(?i)evidence suggests").unwrap(),
            regex::Regex::new(r"(?i)peer.reviewed").unwrap(),
            regex::Regex::new(r"(?i)published in").unwrap(),
        ];
        
        Self {
            trivial_patterns,
            spam_patterns,
            quality_indicators,
        }
    }

    pub async fn score_content(&self, text: &str) -> Result<f32> {
        let mut score = 0.5f32; // Base score
        
        // Penalty for trivial content
        for pattern in &self.trivial_patterns {
            if pattern.is_match(text) {
                score -= 0.8;
            }
        }
        
        // Penalty for spam content
        for pattern in &self.spam_patterns {
            if pattern.is_match(text) {
                score -= 0.6;
            }
        }
        
        // Bonus for quality indicators
        for pattern in &self.quality_indicators {
            if pattern.is_match(text) {
                score += 0.3;
            }
        }
        
        // Bonus for proper citations/references
        if text.contains("(") && text.contains(")") {
            score += 0.1;
        }
        
        // Bonus for factual statements
        if self.contains_factual_structure(text) {
            score += 0.2;
        }
        
        Ok(score.max(0.0).min(1.0))
    }

    fn contains_factual_structure(&self, text: &str) -> bool {
        // Check for patterns that indicate factual information
        let factual_patterns = [
            r"\b\d{4}\b",                    // Years
            r"\b\d+(\.\d+)?\s*(percent|%)\b", // Percentages
            r"\b\d+(\.\d+)?\s*(million|billion|thousand)\b", // Large numbers
            r"\b(approximately|about|around)\s+\d+\b", // Approximate numbers
            r"\b(born|died|founded|established|created)\s+in\s+\d{4}\b", // Historical dates
        ];
        
        for pattern_str in factual_patterns {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                if pattern.is_match(text) {
                    return true;
                }
            }
        }
        
        false
    }
}

/// Importance filter that uses neural salience models
pub struct ImportanceFilter {
    salience_model: NeuralSalienceModel,
    threshold: f32,
    adaptive_threshold: bool,
}

impl ImportanceFilter {
    pub fn new(threshold: f32) -> Self {
        Self {
            salience_model: NeuralSalienceModel::new(threshold),
            threshold,
            adaptive_threshold: false,
        }
    }

    pub fn with_adaptive_threshold(mut self, adaptive: bool) -> Self {
        self.adaptive_threshold = adaptive;
        self
    }

    pub async fn filter_text(&self, text: &str) -> Result<FilterResult> {
        let salience_score = self.salience_model.calculate_salience(text).await?;
        let effective_threshold = if self.adaptive_threshold {
            self.calculate_adaptive_threshold(text).await?
        } else {
            self.threshold
        };
        
        let should_keep = salience_score > effective_threshold;
        
        Ok(FilterResult {
            original_text: text.to_string(),
            salience_score,
            threshold_used: effective_threshold,
            should_keep,
            filter_reasons: self.get_filter_reasons(text, salience_score, effective_threshold),
        })
    }

    pub async fn filter_triples(&self, triples: Vec<Triple>) -> Result<Vec<Triple>> {
        let mut filtered = Vec::new();
        
        for triple in triples {
            let score = self.salience_model.score_triple(&triple).await?;
            if score > self.threshold {
                filtered.push(triple);
            }
        }
        
        Ok(filtered)
    }

    pub async fn batch_filter(&self, texts: Vec<String>) -> Result<Vec<FilterResult>> {
        let mut results = Vec::new();
        
        for text in texts {
            let result = self.filter_text(&text).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    async fn calculate_adaptive_threshold(&self, text: &str) -> Result<f32> {
        // Adjust threshold based on text characteristics
        let base_threshold = self.threshold;
        let mut adjustment = 0.0;
        
        // Lower threshold for longer texts (more context)
        if text.len() > 200 {
            adjustment -= 0.1;
        }
        
        // Higher threshold for very short texts
        if text.len() < 50 {
            adjustment += 0.2;
        }
        
        // Lower threshold for texts with proper nouns
        let proper_noun_count = text.split_whitespace()
            .filter(|word| word.chars().next().unwrap_or('a').is_uppercase())
            .count();
        if proper_noun_count > 2 {
            adjustment -= 0.1;
        }
        
        Ok((base_threshold + adjustment).max(0.1).min(0.9))
    }

    fn get_filter_reasons(&self, text: &str, score: f32, threshold: f32) -> Vec<String> {
        let mut reasons = Vec::new();
        
        if score <= threshold {
            reasons.push(format!("Salience score {:.2} below threshold {:.2}", score, threshold));
        }
        
        if text.len() < 20 {
            reasons.push("Text too short".to_string());
        }
        
        if text.to_lowercase().contains("the sky is blue") {
            reasons.push("Contains trivial statement".to_string());
        }
        
        if text.chars().filter(|c| c.is_uppercase()).count() == 0 {
            reasons.push("No proper nouns detected".to_string());
        }
        
        reasons
    }
}

/// Result of importance filtering
#[derive(Debug, Clone)]
pub struct FilterResult {
    pub original_text: String,
    pub salience_score: f32,
    pub threshold_used: f32,
    pub should_keep: bool,
    pub filter_reasons: Vec<String>,
}

impl FilterResult {
    pub fn is_filtered(&self) -> bool {
        !self.should_keep
    }
    
    pub fn get_quality_assessment(&self) -> &str {
        match self.salience_score {
            s if s >= 0.8 => "High quality",
            s if s >= 0.6 => "Good quality",
            s if s >= 0.4 => "Moderate quality",
            s if s >= 0.2 => "Low quality",
            _ => "Very low quality",
        }
    }
}
