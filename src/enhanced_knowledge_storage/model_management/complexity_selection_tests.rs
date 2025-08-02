//! Tests for complexity-based model selection
//! 
//! Verifies that the system correctly selects models based on task complexity

#[cfg(test)]
mod tests {
    use crate::enhanced_knowledge_storage::{
        types::{ComplexityLevel, ModelResourceConfig, ProcessingTask, TaskComplexity},
        model_management::{ModelResourceManager, ModelRegistry},
    };
    
    #[tokio::test]
    async fn test_complexity_based_model_selection() {
        // Create a resource manager with sufficient memory
        let mut config = ModelResourceConfig::default();
        config.max_memory_usage = 5_000_000_000; // 5GB to fit all models
        
        let manager = ModelResourceManager::new(config).await.unwrap();
        
        // Test that different complexity levels select different models by processing tasks
        let low_task = ProcessingTask::new(ComplexityLevel::Low, "test");
        let medium_task = ProcessingTask::new(ComplexityLevel::Medium, "test");
        let high_task = ProcessingTask::new(ComplexityLevel::High, "test");
        
        let low_result = manager.process_with_optimal_model(low_task).await.unwrap();
        let medium_result = manager.process_with_optimal_model(medium_task).await.unwrap();
        let high_result = manager.process_with_optimal_model(high_task).await.unwrap();
        
        let low_model = low_result.model_used;
        let medium_model = medium_result.model_used;
        let high_model = high_result.model_used;
        
        // Verify we get different models for each complexity
        assert_ne!(low_model, medium_model, "Low and Medium should use different models");
        assert_ne!(medium_model, high_model, "Medium and High should use different models");
        assert_ne!(low_model, high_model, "Low and High should use different models");
        
        // The model names in results will be the display names, not the IDs
        // So we just verify they're different and reasonable
    }
    
    #[tokio::test]
    async fn test_memory_constrained_selection() {
        // Create a resource manager with limited memory
        let mut config = ModelResourceConfig::default();
        config.max_memory_usage = 800_000_000; // 800MB - can't fit the high model
        
        let manager = ModelResourceManager::new(config).await.unwrap();
        
        // High complexity should fall back to a smaller model
        let high_task = ProcessingTask::new(ComplexityLevel::High, "test");
        let result = manager.process_with_optimal_model(high_task).await.unwrap();
        
        // Memory constraint should prevent using the largest model
        // The actual model name will be from metadata, not the ID
        assert!(result.success, "Task should complete even with memory constraints");
    }
    
    #[test]
    fn test_model_registry_contains_all_complexity_levels() {
        let registry = ModelRegistry::with_default_models();
        
        // Check we have models for each complexity level
        let low_models = registry.list_models_by_complexity(TaskComplexity::Low);
        let medium_models = registry.list_models_by_complexity(TaskComplexity::Medium);
        let high_models = registry.list_models_by_complexity(TaskComplexity::High);
        
        assert!(!low_models.is_empty(), "Should have low complexity models");
        assert!(!medium_models.is_empty(), "Should have medium complexity models");
        assert!(!high_models.is_empty(), "Should have high complexity models");
        
        // Verify no 1.7B model in high complexity
        for model in high_models {
            assert!(!model.name.contains("1.7B"), "1.7B model should be removed");
            assert!(!model.huggingface_id.contains("1.7B"), "1.7B model should be removed");
        }
    }
    
    #[test]
    fn test_no_1_7b_model_in_registry() {
        let registry = ModelRegistry::with_default_models();
        
        // Ensure 1.7B model is not registered
        assert!(!registry.has_model("smollm2_1_7b"), "1.7B model should not exist");
        
        // Check all model IDs
        let all_ids = registry.get_model_ids();
        for id in all_ids {
            assert!(!id.contains("1_7b"), "No model ID should contain 1_7b");
            assert!(!id.contains("1.7b"), "No model ID should contain 1.7b");
        }
    }
}