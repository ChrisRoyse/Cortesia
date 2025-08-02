//! Simple test to verify all local models work

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use llmkg::enhanced_knowledge_storage::ai_components::local_model_backend::{
        LocalModelBackend, LocalModelConfig
    };

    fn models_available() -> bool {
        PathBuf::from("model_weights/.models_ready").exists()
    }

    #[tokio::test]
    async fn test_all_local_models_working() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available. Run python scripts/convert_to_candle.py first");
            return;
        }


        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();

        let models = backend.list_available_models();
        println!("Available models: {:?}", models);

        assert!(!models.is_empty(), "Should have at least one model available");

        // Test each model
        let test_text = "This is a test sentence for model validation.";
        
        for model_id in &models {
            println!("\n=== Testing model: {} ===", model_id);
            
            match backend.generate_embeddings(model_id, test_text).await {
                Ok(embeddings) => {
                    println!("✓ Model {} works! Generated {} dimensional embeddings", 
                            model_id, embeddings.len());
                    
                    // Basic validation
                    assert!(!embeddings.is_empty(), "Embeddings should not be empty");
                    assert!(embeddings.len() >= 384, "Should have reasonable dimensionality");
                    
                    // Check if embeddings contain actual values
                    let non_zero_count = embeddings.iter().filter(|&&x| x != 0.0).count();
                    assert!(non_zero_count > embeddings.len() / 2, 
                           "Most embeddings should be non-zero");
                    
                    // Check reasonable value range  
                    let max_abs = embeddings.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
                    assert!(max_abs < 100.0, "Embedding values should be in reasonable range");
                    
                    println!("  - Dimensions: {}", embeddings.len());
                    println!("  - Non-zero values: {}/{}", non_zero_count, embeddings.len());
                    println!("  - Max absolute value: {:.3}", max_abs);
                }
                Err(e) => {
                    eprintln!("✗ Model {} failed: {}", model_id, e);
                    panic!("Model {} should work but failed with: {}", model_id, e);
                }
            }
        }

        println!("\n✅ All {} models are working correctly!", models.len());
    }

    #[tokio::test]
    async fn test_model_consistency() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }


        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();

        let test_text = "Consistency test sentence";
        
        // Test that the same model produces the same embeddings
        if let Some(model_id) = backend.list_available_models().first() {
            println!("Testing consistency with model: {}", model_id);
            
            let embeddings1 = backend.generate_embeddings(model_id, test_text).await.unwrap();
            let embeddings2 = backend.generate_embeddings(model_id, test_text).await.unwrap();
            
            assert_eq!(embeddings1.len(), embeddings2.len(), 
                      "Embeddings should have same length");
            
            // Check if embeddings are identical or very close
            let max_diff = embeddings1.iter().zip(embeddings2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |acc, diff| acc.max(diff));
            
            println!("Maximum difference between runs: {:.6}", max_diff);
            assert!(max_diff < 0.001, "Embeddings should be consistent across runs");
            
            println!("✓ Model {} produces consistent embeddings", model_id);
        }
    }

    #[tokio::test]
    async fn test_memory_usage() {
        if !models_available() {
            eprintln!("Skipping test: Local models not available");
            return;
        }


        let config = LocalModelConfig::default();
        let backend = LocalModelBackend::new(config).unwrap();

        // Check memory before loading any models
        let initial_memory = backend.get_memory_usage().await;
        assert!(initial_memory.is_empty(), "No models should be loaded initially");

        // Load a model by generating embeddings
        if let Some(model_id) = backend.list_available_models().first() {
            println!("Testing memory usage with model: {}", model_id);
            
            let _ = backend.generate_embeddings(model_id, "test").await.unwrap();
            
            let memory_after = backend.get_memory_usage().await;
            assert!(!memory_after.is_empty(), "Model should be loaded");
            assert!(memory_after.contains_key(model_id), "Specific model should be tracked");
            
            let memory_used = memory_after[model_id];
            println!("Memory used by {}: {} bytes ({:.1} MB)", 
                    model_id, memory_used, memory_used as f64 / 1_000_000.0);
            
            assert!(memory_used > 50_000_000, "Model should use reasonable amount of memory (>50MB)");
            assert!(memory_used < 2_000_000_000, "Model shouldn't use excessive memory (<2GB)");
            
            // Test cache clearing
            backend.clear_cache().await;
            let memory_after_clear = backend.get_memory_usage().await;
            assert!(memory_after_clear.is_empty(), "Memory should be cleared");
            
            println!("✓ Memory management working correctly");
        }
    }
}