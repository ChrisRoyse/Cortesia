//! Model ID Translation Layer
//! 
//! Translates between model registry IDs and AI backend model IDs.
//! This bridges the gap between the abstract model registry and concrete AI implementations.

use std::collections::HashMap;

/// Get the model ID mapping
fn get_model_id_mapping() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    
    // Map SmolLM models to available BERT variants based on size
    // Low complexity (135M) -> Smallest available model
    map.insert("smollm2_135m", "sentence-transformers/all-MiniLM-L6-v2");
    
    // Medium complexity (360M) -> Medium available model  
    map.insert("smollm2_360m", "bert-base-uncased");
    
    // High complexity -> BERT large NER model
    // Note: 1.7B model removed as it's too large
    map.insert("smollm2_high", "dbmdz/bert-large-cased-finetuned-conll03-english");
    
    // Direct mappings for MiniLM variants
    map.insert("minilm_l6_v2", "sentence-transformers/all-MiniLM-L6-v2");
    map.insert("minilm_l12_v2", "sentence-transformers/all-MiniLM-L6-v2"); // L12 not available, use L6
    
    // Additional mappings for common model aliases
    map.insert("embedding_model", "sentence-transformers/all-MiniLM-L6-v2");
    map.insert("language_model", "bert-base-uncased");
    map.insert("ner_model", "dbmdz/bert-large-cased-finetuned-conll03-english");
    
    map
}

/// Translates a model registry ID to an AI backend model ID
pub fn translate_model_id(registry_id: &str) -> Option<&'static str> {
    let mapping = get_model_id_mapping();
    mapping.get(registry_id).copied()
}

/// Gets the backend model ID, falling back to the original if no translation exists
pub fn get_backend_model_id(registry_id: &str) -> &str {
    translate_model_id(registry_id).unwrap_or(registry_id)
}

/// Checks if a model ID needs translation
pub fn needs_translation(model_id: &str) -> bool {
    let mapping = get_model_id_mapping();
    mapping.contains_key(model_id)
}

/// Gets all available registry model IDs
pub fn get_registry_model_ids() -> Vec<&'static str> {
    let mapping = get_model_id_mapping();
    mapping.keys().copied().collect()
}

/// Gets the complexity level for a model based on parameter count
pub fn get_model_complexity(registry_id: &str) -> Option<crate::enhanced_knowledge_storage::types::ComplexityLevel> {
    use crate::enhanced_knowledge_storage::types::ComplexityLevel;
    
    match registry_id {
        "smollm2_135m" | "minilm_l6_v2" | "minilm_l12_v2" => Some(ComplexityLevel::Low),
        "smollm2_360m" | "language_model" => Some(ComplexityLevel::Medium),
        "smollm2_high" | "ner_model" => Some(ComplexityLevel::High),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_translation() {
        // Test SmolLM translations
        assert_eq!(
            translate_model_id("smollm2_135m"),
            Some("sentence-transformers/all-MiniLM-L6-v2")
        );
        assert_eq!(
            translate_model_id("smollm2_360m"),
            Some("bert-base-uncased")
        );
        assert_eq!(
            translate_model_id("smollm2_high"),
            Some("dbmdz/bert-large-cased-finetuned-conll03-english")
        );
        
        // Test MiniLM translations
        assert_eq!(
            translate_model_id("minilm_l6_v2"),
            Some("sentence-transformers/all-MiniLM-L6-v2")
        );
        
        // Test unknown model returns None
        assert_eq!(translate_model_id("unknown_model"), None);
    }
    
    #[test]
    fn test_get_backend_model_id() {
        // Known models get translated
        assert_eq!(
            get_backend_model_id("smollm2_360m"),
            "bert-base-uncased"
        );
        
        // Unknown models return as-is
        assert_eq!(
            get_backend_model_id("bert-base-uncased"),
            "bert-base-uncased"
        );
    }
    
    #[test]
    fn test_needs_translation() {
        assert!(needs_translation("smollm2_135m"));
        assert!(needs_translation("minilm_l6_v2"));
        assert!(!needs_translation("bert-base-uncased"));
        assert!(!needs_translation("unknown_model"));
    }
    
    #[test]
    fn test_model_complexity() {
        use crate::enhanced_knowledge_storage::types::ComplexityLevel;
        
        assert_eq!(
            get_model_complexity("smollm2_135m"),
            Some(ComplexityLevel::Low)
        );
        assert_eq!(
            get_model_complexity("smollm2_360m"),
            Some(ComplexityLevel::Medium)
        );
        assert_eq!(
            get_model_complexity("unknown_model"),
            None
        );
    }
}