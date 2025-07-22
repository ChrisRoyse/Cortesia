// Simple test to verify extraction module compiles
#[cfg(test)]
mod test {
    use llmkg::extraction::{AdvancedEntityExtractor, EntityExtractor};

    #[tokio::test]
    async fn test_extraction_module_exists() {
        let extractor = AdvancedEntityExtractor::new();
        let text = "John Smith works at Google.";
        
        // Test that we can call the public API
        let result = extractor.extract_entities(text).await;
        assert!(result.is_ok());
        
        let entities = result.unwrap();
        // Basic check - should extract something from this text
        assert!(!entities.is_empty());
    }
}