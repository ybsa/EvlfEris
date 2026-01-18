import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    # Use absolute path for clarity
    local_dir = Path("c:/Users/wind xebec/EvlfEris/models/base")
    
    print(f"🚀 Starting download of {model_id}...")
    print(f"📂 Target directory: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model"] # Only download relevant files
        )
        print("✅ Download completed successfully!")
        
        # Verify critical files
        expected_files = ["model.safetensors", "config.json", "tokenizer.json"]
        missing = []
        for f in expected_files:
            if not (local_dir / f).exists():
                # Check for sharded weights if main safetensors missing
                if f == "model.safetensors" and list(local_dir.glob("model-*.safetensors")):
                    continue
                missing.append(f)
        
        if missing:
            print(f"⚠️ Warning: Some expected files are missing: {missing}")
        else:
            print("✨ All critical files verified.")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_model()
