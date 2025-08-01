#!/usr/bin/env python3
"""
Convert PyTorch models to Candle-compatible format
"""

import os
import sys
import json
import torch
import safetensors.torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Model configurations
MODELS = {
    "bert-base-uncased": {
        "repo_id": "bert-base-uncased",
        "model_type": "bert",
    },
    "minilm-l6-v2": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2", 
        "model_type": "bert",
    },
    "bert-large-ner": {
        "repo_id": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "model_type": "bert",
    }
}

def download_and_convert_model(model_name, model_config):
    """Download model and convert to Candle format"""
    print(f"\nProcessing {model_name}...")
    
    output_dir = Path(f"model_weights/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and tokenizer
        print(f"  Loading model from HuggingFace...")
        model = AutoModel.from_pretrained(model_config["repo_id"])
        tokenizer = AutoTokenizer.from_pretrained(model_config["repo_id"])
        
        # Save config
        config = model.config.to_dict()
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  [OK] Saved config.json")
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_dir))
        print(f"  [OK] Saved tokenizer files")
        
        # Get model state dict
        state_dict = model.state_dict()
        
        # Convert to Candle-compatible format
        candle_weights = {}
        for key, tensor in state_dict.items():
            # Candle expects different key names
            candle_key = key.replace(".", "_")
            candle_weights[candle_key] = tensor
        
        # Save as safetensors (preferred by Candle)
        safetensors_path = output_dir / "model.safetensors"
        safetensors.torch.save_file(candle_weights, str(safetensors_path))
        print(f"  [OK] Saved model.safetensors")
        
        # Also save as PyTorch format for compatibility
        torch_path = output_dir / "pytorch_model.bin"
        torch.save(state_dict, str(torch_path))
        print(f"  [OK] Saved pytorch_model.bin")
        
        # Create metadata for Candle
        candle_metadata = {
            "model_type": model_config["model_type"],
            "architecture": config.get("architectures", ["unknown"])[0],
            "hidden_size": config.get("hidden_size", 768),
            "num_layers": config.get("num_hidden_layers", 12),
            "num_heads": config.get("num_attention_heads", 12),
            "vocab_size": config.get("vocab_size", 30522),
            "max_position_embeddings": config.get("max_position_embeddings", 512),
        }
        
        with open(output_dir / "candle_metadata.json", "w") as f:
            json.dump(candle_metadata, f, indent=2)
        print(f"  [OK] Created candle_metadata.json")
        
        # Create weight mapping for Candle
        weight_map = {}
        for key in candle_weights.keys():
            weight_map[key] = {
                "shape": list(candle_weights[key].shape),
                "dtype": str(candle_weights[key].dtype),
            }
        
        with open(output_dir / "weight_map.json", "w") as f:
            json.dump(weight_map, f, indent=2)
        print(f"  [OK] Created weight_map.json")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to process {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_candle_loader_code():
    """Create Rust code for loading models in Candle"""
    loader_code = '''//! Candle Model Loader
//! Auto-generated code for loading converted models

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, Module};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use std::collections::HashMap;

pub struct CandleModelLoader {
    device: Device,
}

impl CandleModelLoader {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
    
    pub fn load_bert_model(&self, model_path: &Path) -> candle_core::Result<BertModel> {
        // Load config
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        
        // Load weights
        let weights_path = model_path.join("model.safetensors");
        let weights = candle_core::safetensors::load(&weights_path, &self.device)?;
        
        // Create VarBuilder
        let vb = VarBuilder::from_tensors(weights, DType::F32, &self.device);
        
        // Load model
        BertModel::load(vb, &config)
    }
}
'''
    
    with open("src/enhanced_knowledge_storage/ai_components/candle_loader.rs", "w") as f:
        f.write(loader_code)
    print("\nCreated candle_loader.rs")

def main():
    """Main conversion function"""
    print("Model Converter for Candle")
    print("=" * 50)
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    successful = []
    failed = []
    
    for model_name, model_config in MODELS.items():
        if download_and_convert_model(model_name, model_config):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Create Candle loader code
    create_candle_loader_code()
    
    # Create marker file
    if successful:
        with open("model_weights/.models_ready", "w") as f:
            f.write(f"Models converted: {', '.join(successful)}\n")
    
    print("\n" + "=" * 50)
    print(f"Converted {len(successful)} models successfully:")
    for model in successful:
        print(f"  [OK] {model}")
    
    if failed:
        print(f"\nFailed to convert {len(failed)} models:")
        for model in failed:
            print(f"  [FAIL] {model}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())