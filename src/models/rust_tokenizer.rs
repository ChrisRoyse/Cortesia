//! Native Rust tokenizer implementation

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Tokenizer for text processing
#[derive(Debug, Clone)]
pub struct RustTokenizer {
    vocab: HashMap<String, usize>,
    inv_vocab: HashMap<usize, String>,
    special_tokens: HashMap<String, usize>,
    max_length: usize,
    lowercase: bool,
}

/// Token information
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    pub input_ids: Vec<usize>,
    pub attention_mask: Vec<f32>,
    pub token_type_ids: Vec<usize>,
    pub offsets: Vec<(usize, usize)>,
    pub tokens: Vec<String>,
}

impl RustTokenizer {
    /// Create a new tokenizer with basic vocabulary
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut inv_vocab = HashMap::new();
        
        // Special tokens
        let special_tokens = vec![
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4),
        ];
        
        let mut idx = 0;
        for (token, id) in special_tokens.iter() {
            vocab.insert(token.to_string(), *id);
            inv_vocab.insert(*id, token.to_string());
            idx = *id + 1;
        }
        
        // Basic English vocabulary (simplified for demo)
        let common_words = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "is", "was", "are", "been", "has", "had", "were", "said", "did", "got",
            "albert", "einstein", "marie", "curie", "physics", "chemistry", "science", "scientist",
            "nobel", "prize", "theory", "relativity", "radioactivity", "poland", "france", "germany",
            "university", "professor", "research", "discovery", "element", "radium", "polonium",
        ];
        
        for word in common_words {
            vocab.insert(word.to_string(), idx);
            inv_vocab.insert(idx, word.to_string());
            idx += 1;
        }
        
        // Add subword tokens for better coverage
        for i in 0..26 {
            let letter = ((b'a' + i) as char).to_string();
            vocab.insert(format!("##{}", letter), idx);
            inv_vocab.insert(idx, format!("##{}", letter));
            idx += 1;
        }
        
        let mut special_tokens_map = HashMap::new();
        for (token, id) in special_tokens {
            special_tokens_map.insert(token.to_string(), id);
        }
        
        Self {
            vocab,
            inv_vocab,
            special_tokens: special_tokens_map,
            max_length: 512,
            lowercase: true,
        }
    }
    
    /// Load vocabulary from file
    pub fn from_file(vocab_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab_json = fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&vocab_json)?;
        
        let mut inv_vocab = HashMap::new();
        for (token, id) in &vocab {
            inv_vocab.insert(*id, token.clone());
        }
        
        let special_tokens = vec![
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4),
        ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        
        Ok(Self {
            vocab,
            inv_vocab,
            special_tokens,
            max_length: 512,
            lowercase: true,
        })
    }
    
    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let processed = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        
        // Simple whitespace tokenization with punctuation handling
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        
        for ch in processed.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
            } else if ch.is_alphanumeric() {
                current_word.push(ch);
            } else {
                // Handle punctuation
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(ch.to_string());
            }
        }
        
        if !current_word.is_empty() {
            tokens.push(current_word);
        }
        
        tokens
    }
    
    /// Convert tokens to IDs
    pub fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<usize> {
        tokens.iter()
            .map(|token| {
                self.vocab.get(token)
                    .or_else(|| self.special_tokens.get(token))
                    .copied()
                    .unwrap_or(self.special_tokens["[UNK]"])
            })
            .collect()
    }
    
    /// Encode text with special tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TokenizedInput {
        let tokens = self.tokenize(text);
        let mut token_strs = Vec::new();
        let mut offsets = Vec::new();
        
        if add_special_tokens {
            token_strs.push("[CLS]".to_string());
            offsets.push((0, 0));
        }
        
        // Calculate offsets
        let mut current_pos = 0;
        for token in &tokens {
            let start = text.find(token.as_str()).unwrap_or(current_pos);
            let end = start + token.len();
            token_strs.push(token.clone());
            offsets.push((start, end));
            current_pos = end;
        }
        
        if add_special_tokens {
            token_strs.push("[SEP]".to_string());
            offsets.push((text.len(), text.len()));
        }
        
        // Truncate if needed
        if token_strs.len() > self.max_length {
            token_strs.truncate(self.max_length - 1);
            offsets.truncate(self.max_length - 1);
            if add_special_tokens {
                token_strs.push("[SEP]".to_string());
                offsets.push((text.len(), text.len()));
            }
        }
        
        let input_ids = self.convert_tokens_to_ids(&token_strs);
        let attention_mask = vec![1.0; input_ids.len()];
        let token_type_ids = vec![0; input_ids.len()];
        
        TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            offsets,
            tokens: token_strs,
        }
    }
    
    /// Encode multiple texts with padding
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<TokenizedInput> {
        let mut batch = Vec::new();
        let mut max_len = 0;
        
        // First pass: encode all texts
        for text in texts {
            let encoded = self.encode(text, add_special_tokens);
            max_len = max_len.max(encoded.input_ids.len());
            batch.push(encoded);
        }
        
        // Second pass: pad to max length
        for encoded in &mut batch {
            let pad_len = max_len - encoded.input_ids.len();
            if pad_len > 0 {
                encoded.input_ids.extend(vec![self.special_tokens["[PAD]"]; pad_len]);
                encoded.attention_mask.extend(vec![0.0; pad_len]);
                encoded.token_type_ids.extend(vec![0; pad_len]);
                encoded.tokens.extend(vec!["[PAD]".to_string(); pad_len]);
                let last_offset = encoded.offsets.last().copied().unwrap_or((0, 0));
                encoded.offsets.extend(vec![last_offset; pad_len]);
            }
        }
        
        batch
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String {
        let tokens: Vec<String> = token_ids.iter()
            .filter_map(|&id| {
                let token = self.inv_vocab.get(&id)?;
                if skip_special_tokens && self.special_tokens.contains_key(token) {
                    None
                } else {
                    Some(token.clone())
                }
            })
            .collect();
        
        tokens.join(" ")
    }
}

/// Vocabulary configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct VocabConfig {
    pub vocab: HashMap<String, usize>,
    pub special_tokens: HashMap<String, usize>,
    pub max_length: usize,
    pub lowercase: bool,
}

impl RustTokenizer {
    /// Save vocabulary to file
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let config = VocabConfig {
            vocab: self.vocab.clone(),
            special_tokens: self.special_tokens.clone(),
            max_length: self.max_length,
            lowercase: self.lowercase,
        };
        
        let json = serde_json::to_string_pretty(&config)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    /// Load from config
    pub fn from_config(config: VocabConfig) -> Self {
        let mut inv_vocab = HashMap::new();
        for (token, id) in &config.vocab {
            inv_vocab.insert(*id, token.clone());
        }
        
        Self {
            vocab: config.vocab,
            inv_vocab,
            special_tokens: config.special_tokens,
            max_length: config.max_length,
            lowercase: config.lowercase,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenization() {
        let tokenizer = RustTokenizer::new();
        let tokens = tokenizer.tokenize("Albert Einstein was a physicist.");
        assert_eq!(tokens, vec!["albert", "einstein", "was", "a", "physicist", "."]);
    }
    
    #[test]
    fn test_encoding() {
        let tokenizer = RustTokenizer::new();
        let encoded = tokenizer.encode("Marie Curie", true);
        
        assert_eq!(encoded.tokens[0], "[CLS]");
        assert_eq!(encoded.tokens[encoded.tokens.len() - 1], "[SEP]");
        assert_eq!(encoded.input_ids.len(), encoded.attention_mask.len());
    }
    
    #[test]
    fn test_batch_encoding() {
        let tokenizer = RustTokenizer::new();
        let texts = vec!["Hello", "Hello world", "Hello world!"];
        let batch = tokenizer.encode_batch(&texts, true);
        
        // All should have same length after padding
        let len = batch[0].input_ids.len();
        assert!(batch.iter().all(|enc| enc.input_ids.len() == len));
    }
    
    #[test]
    fn test_decode() {
        let tokenizer = RustTokenizer::new();
        let encoded = tokenizer.encode("Hello world", false);
        let decoded = tokenizer.decode(&encoded.input_ids, false);
        assert!(decoded.contains("hello") || decoded.contains("world"));
    }
}