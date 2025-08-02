//! Simple test to verify Candle model loading works
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Candle model loading...\n");

    // Initialize local model backend directly

    // Import the necessary types
    use llmkg::enhanced_knowledge_storage::{
        ai_components::local_model_backend::{LocalModelBackend, LocalModelConfig},
    };

    // Create local model backend
    let config = LocalModelConfig {
        model_weights_dir: PathBuf::from("model_weights"),
        device: candle_core::Device::Cpu,
        max_sequence_length: 512,
        use_cache: true,
    };

    println!("Initializing local model backend...");
    let backend = match LocalModelBackend::new(config) {
        Ok(backend) => backend,
        Err(e) => {
            println!("✗ Failed to initialize local model backend: {}", e);
            return Ok(());
        }
    };

    println!("Attempting to generate embeddings...");
    let text = "Hello, this is a test of the local model backend.";
    
    match backend.generate_embeddings("bert-base-uncased", text).await {
        Ok(embeddings) => {
            println!("✓ Successfully generated embeddings!");
            println!("  Embedding dimensions: {}", embeddings.len());
            println!("  First few values: {:?}", &embeddings[..5.min(embeddings.len())]);
        }
        Err(e) => {
            println!("✗ Failed to generate embeddings: {}", e);
        }
    }

    println!("\nTest complete!");
    Ok(())
}