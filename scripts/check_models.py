#!/usr/bin/env python3
"""Check if models are downloaded and converted"""
import os
from pathlib import Path

def check_models():
    root = Path(__file__).parent.parent
    model_weights = root / "model_weights"
    
    models = {
        "bert-base-uncased": ["model.safetensors", "config.json", "vocab.txt"],
        "minilm-l6-v2": ["model.safetensors", "config.json", "vocab.txt"],
        "bert-large-ner": ["model.safetensors", "config.json", "vocab.txt"]
    }
    
    print("Checking local models...")
    print("=" * 50)
    
    all_good = True
    for model_name, required_files in models.items():
        model_dir = model_weights / model_name
        print(f"\n{model_name}:")
        
        if not model_dir.exists():
            print(f"  [FAIL] Directory not found: {model_dir}")
            all_good = False
            continue
            
        for file_name in required_files:
            file_path = model_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  [OK] {file_name} ({size_mb:.1f} MB)")
            else:
                print(f"  [MISSING] {file_name}")
                if file_name == "model.safetensors":
                    # Check for pytorch_model.bin as alternative
                    alt_path = model_dir / "pytorch_model.bin"
                    if alt_path.exists():
                        size_mb = alt_path.stat().st_size / (1024 * 1024)
                        print(f"  [OK] pytorch_model.bin ({size_mb:.1f} MB) - needs conversion")
                    else:
                        all_good = False
                else:
                    all_good = False
    
    print("\n" + "=" * 50)
    if all_good or (model_weights / ".models_ready").exists():
        print("[OK] Models are ready for local use!")
        print("\nTo use local models, set environment variable:")
        print("  export LLMKG_USE_LOCAL_MODELS=true")
        print("  # or on Windows:")
        print("  set LLMKG_USE_LOCAL_MODELS=true")
    else:
        print("[FAIL] Some models are missing or incomplete")
        print("Run: python scripts/convert_to_candle.py")
    
    return all_good

if __name__ == "__main__":
    check_models()