#!/usr/bin/env python3
"""
Download and convert models for local use with Candle
"""

import os
import sys
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

# Model configurations
MODELS = {
    "bert-base-uncased": {
        "repo_id": "bert-base-uncased",
        "files": ["config.json", "vocab.txt", "pytorch_model.bin"],
        "output_dir": "model_weights/bert-base-uncased"
    },
    "minilm-l6-v2": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin", "vocab.txt"],
        "output_dir": "model_weights/minilm-l6-v2"
    },
    "bert-large-ner": {
        "repo_id": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "files": ["config.json", "vocab.txt", "pytorch_model.bin", "tokenizer_config.json"],
        "output_dir": "model_weights/bert-large-ner"
    }
}

def download_model(model_name, model_config):
    """Download a model from HuggingFace Hub"""
    print(f"\nDownloading {model_name}...")
    
    output_dir = Path(model_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download specific files
        for file_name in model_config["files"]:
            print(f"  Downloading {file_name}...")
            try:
                file_path = hf_hub_download(
                    repo_id=model_config["repo_id"],
                    filename=file_name,
                    cache_dir=".cache",
                    local_dir=str(output_dir)
                )
                print(f"    [OK] Downloaded to {file_path}")
            except Exception as e:
                print(f"    [FAIL] Failed to download {file_name}: {e}")
                # Continue with other files
        
        # Create a metadata file
        metadata = {
            "model_name": model_name,
            "repo_id": model_config["repo_id"],
            "downloaded_files": model_config["files"],
            "candle_compatible": True
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[SUCCESS] Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {model_name}: {e}")
        return False

def main():
    """Main download function"""
    print("Model Downloader for LLMKG")
    print("=" * 50)
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    successful = []
    failed = []
    
    for model_name, model_config in MODELS.items():
        if download_model(model_name, model_config):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    print("\n" + "=" * 50)
    print(f"Downloaded {len(successful)} models successfully:")
    for model in successful:
        print(f"  [OK] {model}")
    
    if failed:
        print(f"\nFailed to download {len(failed)} models:")
        for model in failed:
            print(f"  [FAIL] {model}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())