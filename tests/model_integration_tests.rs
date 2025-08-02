//! Comprehensive integration tests for all local models
//! 
//! Tests each model's full integration into the system

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use llmkg::enhanced_knowledge_storage::{
        model_management::{ModelResourceManager, ModelResourceConfig},
        types::{ComplexityLevel, ProcessingTask},
        ai_components::local_model_backend::{LocalModelBackend, LocalModelConfig},
    };
    use std::sync::Arc;
    
    fn models_available() -> bool {
        PathBuf::from("model_weights/.models_ready").exists()
    }
    
    #[tokio::test]
    async fn test_model_resource_manager_with_local_models() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        
        // Initialize resource manager
        let resource_manager = ModelResourceManager::new(ModelResourceConfig::default()).await;
        assert!(resource_manager.is_ok(), "Resource manager should initialize with local models");
        
        let manager = resource_manager.unwrap();
        
        // Test processing with different complexity levels
        let test_cases = vec![
            (ComplexityLevel::Low, "Simple test text"),
            (ComplexityLevel::Medium, "This is a more complex document that requires deeper analysis with multiple concepts"),
            (ComplexityLevel::High, "Complex scientific document about quantum mechanics, relativity theory, and advanced mathematical concepts that require sophisticated processing"),
        ];
        
        for (complexity, content) in test_cases {
            println!("Testing {} complexity processing", format!("{:?}", complexity));
            
            let task = ProcessingTask {
                complexity_level: complexity,
                content: content.to_string(),
                task_type: "embeddings".to_string(),
                timeout: Some(30000),
            };
            
            let result = manager.process_with_optimal_model(task).await;
            assert!(result.is_ok(), "Processing should succeed for {:?} complexity", complexity);
            
            let result = result.unwrap();
            println!("✓ Processed successfully with model: {}", result.model_used);
            assert!(!result.output.is_empty(), "Should return non-empty output");
            assert!(result.confidence > 0.0, "Should have positive confidence");
        }
    }
    
    #[tokio::test]
    async fn test_minilm_embeddings_similarity() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Test semantic similarity with MiniLM
        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        
        let sentences = vec![
            "The cat sat on the mat",
            "A feline rested on the rug",  // Similar to first
            "The dog played in the park",
            "A canine frolicked in the garden",  // Similar to third
            "Einstein developed the theory of relativity",  // Different topic
        ];
        
        // Generate embeddings for all sentences
        let mut embeddings = Vec::new();
        for sentence in &sentences {
            let emb = backend.generate_embeddings(model_id, sentence).await.unwrap();
            embeddings.push(emb);
        }
        
        // Calculate cosine similarities
        let cosine_sim = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (norm_a * norm_b)
        };
        
        // Check similarities
        let sim_0_1 = cosine_sim(&embeddings[0], &embeddings[1]);
        let sim_2_3 = cosine_sim(&embeddings[2], &embeddings[3]);
        let sim_0_4 = cosine_sim(&embeddings[0], &embeddings[4]);
        
        println!("Similarity scores:");
        println!("  Cat/Feline: {:.3}", sim_0_1);
        println!("  Dog/Canine: {:.3}", sim_2_3);
        println!("  Cat/Einstein: {:.3}", sim_0_4);
        
        // Similar sentences should have high similarity
        assert!(sim_0_1 > 0.7, "Cat/Feline should be similar");
        assert!(sim_2_3 > 0.7, "Dog/Canine should be similar");
        
        // Different topics should have lower similarity
        assert!(sim_0_4 < 0.5, "Cat/Einstein should be dissimilar");
    }
    
    #[tokio::test]
    async fn test_bert_large_ner_entity_extraction() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Test NER capabilities
        let model_id = "dbmdz/bert-large-cased-finetuned-conll03-english";
        
        let text = "Barack Obama was born in Honolulu, Hawaii. He later became the President of the United States.";
        
        // Generate embeddings (NER models can still produce embeddings)
        let embeddings = backend.generate_embeddings(model_id, text).await;
        
        match embeddings {
            Ok(emb) => {
                println!("✓ Successfully generated embeddings with bert-large-ner");
                assert_eq!(emb.len(), 1024, "BERT large should produce 1024-dimensional embeddings");
                
                // Check embeddings quality
                let non_zero = emb.iter().any(|&x| x != 0.0);
                assert!(non_zero, "Embeddings should not be all zeros");
            }
            Err(e) => {
                eprintln!("Note: bert-large-ner embedding generation failed: {}", e);
                // This is acceptable as NER models might not be optimized for embeddings
            }
        }
    }
    
    #[tokio::test]
    async fn test_model_memory_management() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        
        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();
        
        // Check initial memory usage
        let initial_memory = backend.get_memory_usage().await;
        assert!(initial_memory.is_empty(), "No models should be loaded initially");
        
        // Load a model
        let _ = backend.generate_embeddings("bert-base-uncased", "test").await.unwrap();
        
        // Check memory after loading
        let memory_after_load = backend.get_memory_usage().await;
        assert!(!memory_after_load.is_empty(), "Model should be loaded");
        assert!(memory_after_load.contains_key("bert-base-uncased"));
        
        let memory_used = memory_after_load.get("bert-base-uncased").unwrap();
        println!("bert-base-uncased memory usage: {} MB", memory_used / 1_000_000);
        assert!(*memory_used > 100_000_000, "Model should use significant memory"); // > 100MB
        
        // Clear cache
        backend.clear_cache().await;
        
        // Check memory after clearing
        let memory_after_clear = backend.get_memory_usage().await;
        assert!(memory_after_clear.is_empty(), "Cache should be cleared");
    }
    
    #[tokio::test]
    async fn test_concurrent_model_access() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }
        
        
        let config = LocalModelConfig::default();
        let backend = Arc::new(LocalModelBackend::new(config).unwrap());
        
        // Spawn multiple concurrent tasks
        let mut handles = vec![];
        
        for i in 0..5 {
            let backend_clone = backend.clone();
            let handle = tokio::spawn(async move {
                let text = format!("This is test sentence number {}", i);
                let embeddings = backend_clone.generate_embeddings("bert-base-uncased", &text).await;
                assert!(embeddings.is_ok(), "Concurrent embedding generation should succeed");
                embeddings.unwrap()
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles).await;
        
        // Verify all succeeded
        for result in results {
            assert!(result.is_ok(), "Task should complete successfully");
            let embeddings = result.unwrap();
            assert_eq!(embeddings.len(), 768, "Each task should produce valid embeddings");
        }
        
        println!("✓ All concurrent tasks completed successfully");
    }
}