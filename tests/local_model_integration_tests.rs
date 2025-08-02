//! Integration tests for local model functionality
//! 
//! Tests the complete pipeline with locally cached models

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use llmkg::enhanced_knowledge_storage::{
        model_management::{ModelResourceManager, ModelResourceConfig},
        knowledge_processing::{IntelligentKnowledgeProcessor, KnowledgeProcessingConfig},
        hierarchical_storage::{HierarchicalStorageEngine, HierarchicalStorageConfig},
        retrieval_system::{RetrievalEngine, RetrievalConfig, RetrievalQuery},
        types::{ComplexityLevel, ProcessingTask},
        ai_components::{
            hybrid_model_backend::{HybridModelBackend, HybridModelConfig},
            local_model_backend::{LocalModelBackend, LocalModelConfig},
        },
    };
    
    /// Check if local models are available
    fn models_available() -> bool {
        PathBuf::from("model_weights/.models_ready").exists()
    }
    
    #[tokio::test]
    async fn test_local_model_backend_initialization() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available. Run scripts/setup_models.bat first");
            return;
        }
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config);
        
        assert!(backend.is_ok(), "Local backend should initialize successfully");
        
        let backend = backend.unwrap();
        let models = backend.list_available_models();
        
        assert!(!models.is_empty(), "Should have at least one model available");
        println!("Available local models: {:?}", models);
    }
    
    #[tokio::test]
    async fn test_hybrid_backend_prefers_local() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        let config = HybridModelConfig {
            prefer_local: true,
            enable_download: false, // Disable downloads to force local
            ..Default::default()
        };
        
        let backend = HybridModelBackend::new(config).await;
        assert!(backend.is_ok(), "Hybrid backend should initialize");
        
        let backend = backend.unwrap();
        
        // Try to load a model
        let result = backend.load_model("smollm2_135m").await;
        assert!(result.is_ok(), "Should load model from local backend");
        
        let handle = result.unwrap();
        assert_eq!(handle.backend_type, llmkg::enhanced_knowledge_storage::model_management::BackendType::Local);
    }
    
    #[tokio::test]
    async fn test_local_model_embeddings() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Test embedding generation
        let text = "This is a test sentence for embeddings.";
        let embeddings = backend.generate_embeddings("bert-base-uncased", text).await;
        
        assert!(embeddings.is_ok(), "Should generate embeddings");
        let embeddings = embeddings.unwrap();
        assert!(!embeddings.is_empty(), "Embeddings should not be empty");
        assert!(embeddings.len() > 100, "Embeddings should have reasonable dimensionality");
    }
    
    #[tokio::test]
    async fn test_full_pipeline_with_local_models() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        // Configure to use local models
        let mut resource_config = ModelResourceConfig::default();
        resource_config.max_memory_usage = 2_000_000_000; // 2GB
        
        // Set up hybrid backend in resource manager
        std::env::set_var("LLMKG_USE_LOCAL_MODELS", "true");
        std::env::set_var("LLMKG_MODEL_WEIGHTS_DIR", "model_weights");
        
        let manager = ModelResourceManager::new(resource_config).await;
        assert!(manager.is_ok(), "Resource manager should initialize with local models");
        
        let manager = manager.unwrap();
        
        // Test processing with different complexity levels
        let tasks = vec![
            ProcessingTask::new(ComplexityLevel::Low, "Simple test"),
            ProcessingTask::new(ComplexityLevel::Medium, "Medium complexity test"),
            ProcessingTask::new(ComplexityLevel::High, "High complexity test with more content"),
        ];
        
        for task in tasks {
            let result = manager.process_with_optimal_model(task).await;
            assert!(result.is_ok(), "Should process task with local model");
            
            let result = result.unwrap();
            assert!(result.success);
            assert!(!result.output.is_empty());
            println!("Processed with model: {}", result.model_used);
        }
    }
    
    #[tokio::test]
    async fn test_knowledge_processing_with_local_models() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        let resource_config = ModelResourceConfig::default();
        let manager = std::sync::Arc::new(
            ModelResourceManager::new(resource_config).await.unwrap()
        );
        
        let processor = IntelligentKnowledgeProcessor::new(
            manager.clone(),
            KnowledgeProcessingConfig::default(),
        );
        
        let document = "Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in Germany in 1879.";
        
        let result = processor.process_knowledge(document, "Einstein Biography").await;
        assert!(result.is_ok(), "Should process document with local models");
        
        let result = result.unwrap();
        assert!(!result.chunks.is_empty(), "Should create chunks");
        assert!(!result.global_entities.is_empty(), "Should extract entities");
        
        println!("Extracted {} entities", result.global_entities.len());
        println!("Created {} chunks", result.chunks.len());
    }
    
    #[tokio::test]
    async fn test_memory_management_with_local_models() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Load multiple models
        let models = vec![
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
        ];
        
        for model_id in &models {
            let result = backend.load_model(model_id).await;
            assert!(result.is_ok(), "Should load model: {}", model_id);
        }
        
        // Check memory usage
        let memory_usage = backend.get_memory_usage().await;
        assert!(!memory_usage.is_empty(), "Should track memory usage");
        
        for (model_id, usage) in memory_usage {
            println!("Model {} uses {} MB", model_id, usage / 1_000_000);
            assert!(usage > 0, "Memory usage should be positive");
        }
        
        // Clear cache
        backend.clear_cache().await;
        let memory_usage_after = backend.get_memory_usage().await;
        assert!(memory_usage_after.is_empty(), "Memory should be cleared");
    }
}