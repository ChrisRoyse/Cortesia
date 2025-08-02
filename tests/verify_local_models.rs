//! Standalone test to verify local models work correctly

use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Verifying Local Model Setup ===\n");
    
    // Check model files exist
    let model_weights_dir = PathBuf::from("model_weights");
    if !model_weights_dir.exists() {
        eprintln!("ERROR: model_weights directory not found!");
        return Err("Model weights directory missing".into());
    }
    
    let models = vec![
        ("bert-base-uncased", vec!["model.safetensors", "config.json", "vocab.txt"]),
        ("minilm-l6-v2", vec!["model.safetensors", "config.json", "vocab.txt"]),
    ];
    
    let mut all_good = true;
    for (model_name, required_files) in &models {
        let model_dir = model_weights_dir.join(model_name);
        println!("Checking {model_name}:");
        
        if !model_dir.exists() {
            eprintln!("  ✗ Directory not found");
            all_good = false;
            continue;
        }
        
        for file_name in required_files {
            let file_path = model_dir.join(file_name);
            if file_path.exists() {
                let size_mb = file_path.metadata()?.len() as f64 / (1024.0 * 1024.0);
                println!("  ✓ {file_name} ({size_mb:.1} MB)");
            } else {
                eprintln!("  ✗ {file_name} missing");
                all_good = false;
            }
        }
        println!();
    }
    
    if all_good {
        println!("✅ All models are downloaded and ready!");
        
        // Test loading with Candle
        println!("\nTesting Candle model loading...");
        
        // Import the local model backend
        use llmkg::enhanced_knowledge_storage::ai_components::local_model_backend::{
            LocalModelBackend, LocalModelConfig
        };
        
        let config = LocalModelConfig::default();
        match LocalModelBackend::new(config) {
            Ok(backend) => {
                println!("✓ Local model backend initialized successfully");
                
                let available = backend.list_available_models();
                println!("\nAvailable models:");
                for model in &available {
                    println!("  - {}", model);
                }
                
                // Try loading a model
                println!("\nAttempting to load bert-base-uncased...");
                match backend.load_model("bert-base-uncased").await {
                    Ok(handle) => {
                        println!("✓ Model loaded successfully!");
                        println!("  ID: {}", handle.id);
                        println!("  Backend: {:?}", handle.backend_type);
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to load model: {}", e);
                    }
                }
                
                // Try generating embeddings
                println!("\nTesting embedding generation...");
                let test_text = "This is a test sentence.";
                match backend.generate_embeddings("bert-base-uncased", test_text).await {
                    Ok(embeddings) => {
                        println!("✓ Embeddings generated successfully!");
                        println!("  Dimensions: {}", embeddings.len());
                        println!("  First few values: {:?}", &embeddings[..5.min(embeddings.len())]);
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to generate embeddings: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("✗ Failed to initialize local model backend: {}", e);
            }
        }
    } else {
        eprintln!("\n❌ Some models are missing!");
        eprintln!("Please run: python scripts/convert_to_candle.py");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_files_exist() {
        let model_weights_dir = PathBuf::from("model_weights");
        assert!(model_weights_dir.exists(), "model_weights directory should exist");
        
        let bert_model = model_weights_dir.join("bert-base-uncased/model.safetensors");
        let minilm_model = model_weights_dir.join("minilm-l6-v2/model.safetensors");
        
        assert!(bert_model.exists(), "BERT model should exist");
        assert!(minilm_model.exists(), "MiniLM model should exist");
    }
}