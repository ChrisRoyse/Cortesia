//! Simple test to verify local models are working
#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    
    #[test]
    fn test_models_downloaded() {
        let model_weights_dir = PathBuf::from("model_weights");
        assert!(model_weights_dir.exists(), "model_weights directory should exist");
        
        let bert_base = model_weights_dir.join("bert-base-uncased/model.safetensors");
        let minilm = model_weights_dir.join("minilm-l6-v2/model.safetensors");
        
        assert!(bert_base.exists(), "BERT base model should be downloaded");
        assert!(minilm.exists(), "MiniLM model should be downloaded");
        
        println!("✓ Models downloaded successfully");
    }
}