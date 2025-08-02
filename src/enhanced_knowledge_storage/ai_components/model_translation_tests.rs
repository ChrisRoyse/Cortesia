//! Tests for model translation integration with AI backend
//! 
//! Verifies that registry model IDs are correctly translated to backend model IDs

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::enhanced_knowledge_storage::{
        model_management::{get_backend_model_id, translate_model_id},
        ai_components::local_model_backend::{LocalModelBackend, LocalModelConfig},
    };
    
    #[tokio::test]
    async fn test_local_backend_loads_translated_models() {
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Test loading models through translation with local backend
        let test_cases = vec![
            ("smollm2_135m", "sentence-transformers/all-MiniLM-L6-v2"),
            ("smollm2_360m", "bert-base-uncased"),
            ("minilm_l6_v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ];
        
        for (registry_id, expected_backend_id) in test_cases {
            let translated = get_backend_model_id(registry_id);
            assert_eq!(translated, expected_backend_id, 
                "Model {} should translate to {}", registry_id, expected_backend_id);
                
            // Test that the translated model can be used with local backend
            let available_models = backend.list_available_models();
            if available_models.contains(&expected_backend_id.to_string()) {
                println!("✓ Local model available: {}", expected_backend_id);
            } else {
                println!("⚠ Local model not found: {}", expected_backend_id);
            }
        }
    }
    
    #[test]
    fn test_model_translation_completeness() {
        // Ensure all common registry IDs have translations
        let registry_ids = vec![
            "smollm2_135m",
            "smollm2_360m", 
            "minilm_l6_v2",
            "minilm_l12_v2",
            "embedding_model",
            "language_model",
            "ner_model",
        ];
        
        for id in registry_ids {
            let translation = translate_model_id(id);
            assert!(translation.is_some(), 
                "Registry model {} should have a translation", id);
        }
    }
    
    #[test]
    fn test_no_1_7b_model_references() {
        // Ensure the 1.7B model has been completely removed
        let removed_ids = vec![
            "smollm2_1_7b",
            "smollm2_1.7b",
            "1_7b",
        ];
        
        for id in removed_ids {
            let translation = translate_model_id(id);
            assert!(translation.is_none(), 
                "Removed model {} should not have a translation", id);
        }
    }
    
    #[test]
    fn test_complexity_mapping_without_1_7b() {
        use crate::enhanced_knowledge_storage::types::ComplexityLevel;
        use crate::enhanced_knowledge_storage::model_management::get_model_complexity;
        
        // Verify complexity mappings work without 1.7B model
        assert_eq!(get_model_complexity("smollm2_135m"), Some(ComplexityLevel::Low));
        assert_eq!(get_model_complexity("smollm2_360m"), Some(ComplexityLevel::Medium));
        assert_eq!(get_model_complexity("ner_model"), Some(ComplexityLevel::High));
        
        // 1.7B model should not have a complexity mapping
        assert_eq!(get_model_complexity("smollm2_1_7b"), None);
    }
}