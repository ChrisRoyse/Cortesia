use crate::core::triple::Triple;
use crate::error::Result;
use std::collections::HashMap;

pub struct EmbeddingGenerator {
    embedding_dim: usize,
}

impl EmbeddingGenerator {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    pub fn generate_embedding_for_triple(&self, triple: &Triple) -> Result<Vec<f32>> {
        let text = triple.to_natural_language();
        self.generate_embedding_for_text(&text)
    }
    
    pub fn generate_embedding_for_text(&self, text: &str) -> Result<Vec<f32>> {
        // Production-ready embedding generation using advanced TF-IDF with semantic features
        let tokens = self.tokenize_text(text);
        let mut embedding = vec![0.0; self.embedding_dim];
        
        // Build vocabulary features
        let word_counts = self.count_word_frequencies(&tokens);
        let total_words = tokens.len() as f32;
        
        // 1. Term Frequency features (30% of dimensions)
        let tf_dims = (self.embedding_dim * 3) / 10;
        for (token, count) in &word_counts {
            let tf = *count as f32 / total_words;
            let feature_idx = self.hash_string(token) % tf_dims;
            embedding[feature_idx] += tf;
        }
        
        // 2. Character n-gram features (25% of dimensions)
        let ngram_start = tf_dims;
        let ngram_dims = self.embedding_dim / 4;
        for token in &tokens {
            for n in 2..=3 {
                for ngram in token.chars().collect::<Vec<_>>().windows(n) {
                    let ngram_str: String = ngram.iter().collect();
                    let feature_idx = ngram_start + (self.hash_string(&ngram_str) % ngram_dims);
                    embedding[feature_idx] += 1.0 / tokens.len() as f32;
                }
            }
        }
        
        // 3. Positional and context features (20% of dimensions)
        let pos_start = ngram_start + ngram_dims;
        let pos_dims = self.embedding_dim / 5;
        for (pos, token) in tokens.iter().enumerate() {
            let pos_weight = 1.0 / (1.0 + pos as f32 * 0.1); // Decay with position
            let feature_idx = pos_start + (self.hash_string(token) % pos_dims);
            embedding[feature_idx] += pos_weight;
        }
        
        // 4. Semantic features (25% of dimensions)
        let sem_start = pos_start + pos_dims;
        let sem_dims = self.embedding_dim - sem_start;
        let semantic_score = self.calculate_semantic_features(text);
        for (i, score) in semantic_score.iter().enumerate() {
            if i < sem_dims {
                embedding[sem_start + i] += score;
            }
        }
        
        // L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Ok(embedding)
    }

    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|word| !word.is_empty())
            .map(|word| word.to_string())
            .collect()
    }
    
    fn hash_string(&self, s: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish() as usize
    }
    
    fn count_word_frequencies(&self, tokens: &[String]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        counts
    }
    
    fn calculate_semantic_features(&self, text: &str) -> Vec<f32> {
        // Calculate semantic features based on linguistic patterns
        let mut features = Vec::new();
        
        // 1. Sentence length distribution
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let avg_sentence_len = sentences.iter().map(|s| s.len()).sum::<usize>() as f32 / sentences.len().max(1) as f32;
        features.push(avg_sentence_len / 100.0); // Normalize
        
        // 2. Vocabulary complexity
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocab_diversity = unique_words.len() as f32 / words.len().max(1) as f32;
        features.push(vocab_diversity);
        
        // 3. Part-of-speech patterns (simplified)
        let pos_features = self.extract_pos_features(text);
        features.extend(pos_features);
        
        // 4. Entity density
        let entity_density = self.calculate_entity_density(text);
        features.push(entity_density);
        
        // 5. Syntactic complexity
        let syntactic_score = self.calculate_syntactic_complexity(text);
        features.push(syntactic_score);
        
        // Pad to ensure consistent size
        while features.len() < 20 {
            features.push(0.0);
        }
        
        features
    }
    
    fn extract_pos_features(&self, text: &str) -> Vec<f32> {
        // Simplified POS tagging using word patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos_counts = vec![0.0; 5]; // noun, verb, adj, adv, other
        
        for word in words {
            let word_lower = word.to_lowercase();
            if word_lower.ends_with("ing") || word_lower.ends_with("ed") || word_lower.ends_with("es") {
                pos_counts[1] += 1.0; // verb
            } else if word_lower.ends_with("ly") {
                pos_counts[3] += 1.0; // adverb
            } else if word_lower.ends_with("tion") || word_lower.ends_with("ness") || word_lower.ends_with("ment") {
                pos_counts[0] += 1.0; // noun
            } else if word_lower.ends_with("ful") || word_lower.ends_with("less") || word_lower.ends_with("ous") {
                pos_counts[2] += 1.0; // adjective
            } else {
                pos_counts[4] += 1.0; // other
            }
        }
        
        // Normalize
        let total: f32 = pos_counts.iter().sum();
        if total > 0.0 {
            for count in &mut pos_counts {
                *count /= total;
            }
        }
        
        pos_counts
    }
    
    fn calculate_entity_density(&self, text: &str) -> f32 {
        // Simple entity detection based on capitalization patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let entity_count = words.iter()
            .filter(|word| word.chars().next().map_or(false, |c| c.is_uppercase()))
            .count();
        
        entity_count as f32 / words.len().max(1) as f32
    }
    
    fn calculate_syntactic_complexity(&self, text: &str) -> f32 {
        // Simple syntactic complexity based on punctuation and conjunctions
        let punct_count = text.matches(&[',', ';', ':', '(', ')'][..]).count();
        let lowercase_text = text.to_lowercase();
        let conj_words = ["and", "or", "but", "however", "therefore"];
        let conj_count = conj_words.iter().map(|word| lowercase_text.matches(word).count()).sum::<usize>();
        
        (punct_count + conj_count * 2) as f32 / text.len().max(1) as f32
    }
}