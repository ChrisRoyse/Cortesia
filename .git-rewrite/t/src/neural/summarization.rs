use crate::core::triple::KnowledgeNode;
use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Neural summarization for knowledge chunks
pub struct NeuralSummarizer {
    model: Arc<dyn SummarizationModel>,
    cache: Arc<RwLock<HashMap<String, CachedSummary>>>,
    max_input_length: usize,
    target_summary_length: usize,
}

impl NeuralSummarizer {
    pub fn new(model: Arc<dyn SummarizationModel>) -> Self {
        Self {
            model,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_input_length: 1024, // Maximum tokens to process
            target_summary_length: 128, // Target summary length
        }
    }

    pub async fn summarize_chunk(&self, chunk: &str) -> Result<String> {
        // Check cache first
        let cache_key = self.generate_cache_key(chunk);
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                if !cached.is_expired() {
                    return Ok(cached.summary.clone());
                }
            }
        }

        // Preprocess the chunk
        let processed_chunk = self.preprocess_chunk(chunk)?;
        
        // Generate summary
        let summary = self.model.summarize(&processed_chunk, self.target_summary_length).await?;
        
        // Post-process summary
        let final_summary = self.postprocess_summary(&summary)?;
        
        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, CachedSummary::new(final_summary.clone()));
        }
        
        Ok(final_summary)
    }

    pub async fn summarize_knowledge_node(&self, node: &KnowledgeNode) -> Result<String> {
        match &node.node_type {
            crate::core::triple::NodeType::Chunk => {
                // Extract text from chunk node
                let text = self.extract_text_from_node(node)?;
                self.summarize_chunk(&text).await
            }
            crate::core::triple::NodeType::Triple => {
                // For triple nodes, create a summary from the triples
                let triples = node.get_triples();
                let combined_text = triples.iter()
                    .map(|t| t.to_natural_language())
                    .collect::<Vec<_>>()
                    .join(". ");
                self.summarize_chunk(&combined_text).await
            }
            _ => {
                Ok("Summary not available for this node type".to_string())
            }
        }
    }

    pub async fn batch_summarize(&self, chunks: Vec<String>) -> Result<Vec<String>> {
        let mut summaries = Vec::new();
        
        for chunk in chunks {
            let summary = self.summarize_chunk(&chunk).await?;
            summaries.push(summary);
        }
        
        Ok(summaries)
    }

    pub async fn adaptive_summarize(&self, text: &str, target_reduction: f32) -> Result<String> {
        if target_reduction <= 0.0 || target_reduction >= 1.0 {
            return Err(GraphError::InvalidInput("Target reduction must be between 0 and 1".to_string()));
        }
        
        let target_length = (text.len() as f32 * (1.0 - target_reduction)) as usize;
        let token_target = target_length / 4; // Rough tokens estimate
        
        self.model.summarize(text, token_target).await
    }

    fn preprocess_chunk(&self, chunk: &str) -> Result<String> {
        // Remove excessive whitespace
        let cleaned = chunk.trim().to_string();
        
        // Truncate if too long
        if cleaned.len() > self.max_input_length * 4 { // Rough character to token conversion
            Ok(cleaned.chars().take(self.max_input_length * 4).collect())
        } else {
            Ok(cleaned)
        }
    }

    fn postprocess_summary(&self, summary: &str) -> Result<String> {
        // Ensure summary ends with proper punctuation
        let mut result = summary.trim().to_string();
        
        if !result.ends_with('.') && !result.ends_with('!') && !result.ends_with('?') {
            result.push('.');
        }
        
        // Ensure first letter is capitalized
        if let Some(first_char) = result.chars().next() {
            if first_char.is_lowercase() {
                result = first_char.to_uppercase().to_string() + &result[1..];
            }
        }
        
        Ok(result)
    }

    fn extract_text_from_node(&self, node: &KnowledgeNode) -> Result<String> {
        // Extract text content based on the node type
        match &node.content {
            crate::core::triple::NodeContent::Triple(triple) => {
                Ok(format!("{} {} {}", triple.subject, triple.predicate, triple.object))
            }
            crate::core::triple::NodeContent::Chunk { text, .. } => {
                Ok(text.clone())
            }
            crate::core::triple::NodeContent::Entity { name, .. } => {
                Ok(name.clone())
            }
            crate::core::triple::NodeContent::Relationship { predicate, .. } => {
                Ok(predicate.clone())
            }
        }
    }

    fn generate_cache_key(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    pub async fn get_cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        CacheStats {
            size: cache.len(),
            hit_count: cache.values().map(|c| c.access_count).sum(),
            total_requests: cache.values().map(|c| c.access_count + 1).sum(),
        }
    }
}

/// Trait for summarization models
#[async_trait::async_trait]
pub trait SummarizationModel: Send + Sync {
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String>;
    fn get_model_name(&self) -> &str;
    fn get_supported_languages(&self) -> Vec<&str>;
}

/// T5-based summarization model placeholder
/// 
/// NOTE: This currently uses extractive summarization instead of T5.
/// Real T5 requires ML framework integration (ONNX, PyTorch, etc.)
pub struct T5SummarizationModel {
    model_name: String,
    temperature: f32,
}

impl T5SummarizationModel {
    pub fn new() -> Self {
        Self {
            model_name: "t5-small".to_string(),
            temperature: 0.7,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

#[async_trait::async_trait]
impl SummarizationModel for T5SummarizationModel {
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String> {
        // T5 model integration requires:
        // 1. ONNX or PyTorch bindings for Rust
        // 2. Pre-trained T5 model weights
        // 3. Tokenizer implementation
        // 4. GPU support for inference
        // 
        // Using simple extractive summarization as fallback
        self.extractive_summarize(text, max_tokens)
    }

    fn get_model_name(&self) -> &str {
        &self.model_name
    }

    fn get_supported_languages(&self) -> Vec<&str> {
        vec!["en"] // English only for now
    }
}

impl T5SummarizationModel {
    fn extractive_summarize(&self, text: &str, max_tokens: usize) -> Result<String> {
        // Simple extractive summarization
        let sentences = self.split_into_sentences(text);
        let sentence_scores = self.score_sentences(&sentences);
        
        // Select top sentences
        let mut selected_sentences = Vec::new();
        let mut total_length = 0;
        
        for (sentence, score) in sentence_scores {
            if total_length + sentence.len() <= max_tokens * 4 && score > 0.3 {
                total_length += sentence.len();
                selected_sentences.push(sentence);
            }
        }
        
        if selected_sentences.is_empty() {
            // Fallback to first sentence if no good sentences found
            selected_sentences.push(sentences.get(0).cloned().unwrap_or_else(|| text.to_string()));
        }
        
        Ok(selected_sentences.join(" "))
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn score_sentences(&self, sentences: &[String]) -> Vec<(String, f32)> {
        let mut scored = Vec::new();
        
        for sentence in sentences {
            let score = self.calculate_sentence_score(sentence);
            scored.push((sentence.clone(), score));
        }
        
        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        scored
    }

    fn calculate_sentence_score(&self, sentence: &str) -> f32 {
        let mut score = 0.0;
        
        // Length score (prefer medium-length sentences)
        let length = sentence.len();
        if length > 20 && length < 200 {
            score += 0.3;
        }
        
        // Keyword score (prefer sentences with important keywords)
        let important_keywords = ["is", "was", "invented", "created", "discovered", "theory", "principle"];
        for keyword in important_keywords {
            if sentence.to_lowercase().contains(keyword) {
                score += 0.1;
            }
        }
        
        // Proper noun score (prefer sentences with proper nouns)
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let proper_noun_count = words.iter()
            .filter(|word| word.chars().next().unwrap_or('a').is_uppercase())
            .count();
        
        score += (proper_noun_count as f32) * 0.1;
        
        // Avoid very short sentences
        if sentence.len() < 10 {
            score -= 0.5;
        }
        
        score.max(0.0).min(1.0)
    }
}

/// BART-based summarization model placeholder
/// 
/// NOTE: This currently uses extractive summarization instead of BART.
/// Real BART requires ML framework integration (ONNX, PyTorch, etc.)
pub struct BARTSummarizationModel {
    model_name: String,
    beam_size: usize,
}

impl BARTSummarizationModel {
    pub fn new() -> Self {
        Self {
            model_name: "facebook/bart-large-cnn".to_string(),
            beam_size: 4,
        }
    }
}

#[async_trait::async_trait]
impl SummarizationModel for BARTSummarizationModel {
    async fn summarize(&self, text: &str, max_tokens: usize) -> Result<String> {
        // In a real implementation, this would use the BART model
        // For now, use a different extractive approach
        self.frequency_based_summarize(text, max_tokens)
    }

    fn get_model_name(&self) -> &str {
        &self.model_name
    }

    fn get_supported_languages(&self) -> Vec<&str> {
        vec!["en"]
    }
}

impl BARTSummarizationModel {
    fn frequency_based_summarize(&self, text: &str, max_tokens: usize) -> Result<String> {
        // Frequency-based extractive summarization
        let words = self.tokenize(text);
        let word_freq = self.calculate_word_frequency(&words);
        let sentences = self.split_into_sentences(text);
        
        // Score sentences based on word frequency
        let mut sentence_scores = Vec::new();
        for sentence in sentences {
            let score = self.score_sentence_by_frequency(&sentence, &word_freq);
            sentence_scores.push((sentence, score));
        }
        
        // Sort by score and select top sentences
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut summary = String::new();
        let mut total_length = 0;
        
        for (sentence, _) in sentence_scores {
            if total_length + sentence.len() <= max_tokens * 4 {
                if !summary.is_empty() {
                    summary.push(' ');
                }
                summary.push_str(&sentence);
                total_length += sentence.len();
            }
        }
        
        if summary.is_empty() {
            summary = text.chars().take(max_tokens * 4).collect();
        }
        
        Ok(summary)
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn calculate_word_frequency(&self, words: &[String]) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        
        // Skip common stop words
        let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"];
        
        for word in words {
            if !stop_words.contains(&word.as_str()) && word.len() > 2 {
                *freq.entry(word.clone()).or_insert(0) += 1;
            }
        }
        
        freq
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10)
            .collect()
    }

    fn score_sentence_by_frequency(&self, sentence: &str, word_freq: &HashMap<String, usize>) -> f32 {
        let sentence_words = self.tokenize(sentence);
        let mut score = 0.0;
        
        for word in sentence_words {
            if let Some(&freq) = word_freq.get(&word) {
                score += freq as f32;
            }
        }
        
        // Normalize by sentence length
        if sentence.len() > 0 {
            score / sentence.len() as f32
        } else {
            0.0
        }
    }
}

/// Cached summary with expiration
#[derive(Debug, Clone)]
struct CachedSummary {
    summary: String,
    created_at: std::time::Instant,
    access_count: usize,
    ttl_seconds: u64,
}

impl CachedSummary {
    fn new(summary: String) -> Self {
        Self {
            summary,
            created_at: std::time::Instant::now(),
            access_count: 0,
            ttl_seconds: 3600, // 1 hour
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed().as_secs() > self.ttl_seconds
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub hit_count: usize,
    pub total_requests: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hit_count as f32 / self.total_requests as f32
        }
    }
}

/// Salience-based filtering for knowledge chunks
pub struct SalienceFilter {
    model: Arc<dyn SalienceModel>,
    threshold: f32,
}

impl SalienceFilter {
    pub fn new(model: Arc<dyn SalienceModel>, threshold: f32) -> Self {
        Self { model, threshold }
    }

    pub async fn filter_chunk(&self, chunk: &str) -> Result<bool> {
        let salience_score = self.model.calculate_salience(chunk).await?;
        Ok(salience_score > self.threshold)
    }

    pub async fn score_chunk(&self, chunk: &str) -> Result<f32> {
        self.model.calculate_salience(chunk).await
    }

    pub async fn filter_chunks(&self, chunks: Vec<String>) -> Result<Vec<String>> {
        let mut filtered = Vec::new();
        
        for chunk in chunks {
            if self.filter_chunk(&chunk).await? {
                filtered.push(chunk);
            }
        }
        
        Ok(filtered)
    }
}

/// Trait for salience models
#[async_trait::async_trait]
pub trait SalienceModel: Send + Sync {
    async fn calculate_salience(&self, text: &str) -> Result<f32>;
}

/// Simple keyword-based salience model
pub struct KeywordSalienceModel {
    important_keywords: Vec<String>,
    weights: HashMap<String, f32>,
}

impl KeywordSalienceModel {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("invented".to_string(), 0.9);
        weights.insert("discovered".to_string(), 0.9);
        weights.insert("theory".to_string(), 0.8);
        weights.insert("principle".to_string(), 0.8);
        weights.insert("created".to_string(), 0.7);
        weights.insert("developed".to_string(), 0.7);
        weights.insert("important".to_string(), 0.6);
        weights.insert("significant".to_string(), 0.6);
        
        Self {
            important_keywords: weights.keys().cloned().collect(),
            weights,
        }
    }
}

#[async_trait::async_trait]
impl SalienceModel for KeywordSalienceModel {
    async fn calculate_salience(&self, text: &str) -> Result<f32> {
        let text_lower = text.to_lowercase();
        let mut score = 0.0;
        
        // Base score from text length (longer texts slightly more salient)
        score += (text.len() as f32 / 1000.0).min(0.2);
        
        // Keyword-based scoring
        for (keyword, weight) in &self.weights {
            if text_lower.contains(keyword) {
                score += weight;
            }
        }
        
        // Proper noun bonus
        let proper_noun_count = text.split_whitespace()
            .filter(|word| word.chars().next().unwrap_or('a').is_uppercase())
            .count();
        score += (proper_noun_count as f32) * 0.05;
        
        // Trivial statement penalty
        if text_lower.contains("the sky is blue") || 
           text_lower.contains("water is wet") ||
           text_lower.contains("fire is hot") {
            score -= 0.8;
        }
        
        Ok(score.max(0.0).min(1.0))
    }
}
