# 🔬 Evlf Eris - Setup & Usage Guide

## 📋 Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~20GB disk space

## 🚀 Quick Start

### 1. Download Base Model

First, download the Llama-3.2-3B-Instruct model:

```powershell
# Install huggingface-cli if needed
pip install huggingface-hub

# Login to HuggingFace (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Download model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir models/base
```

### 2. Install Dependencies

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 3. Run Model Surgery

**Step 1: Analyze the model**
```powershell
python surgery/analyze.py --model models/base
```
This creates `analysis_results.yaml` with pruning recommendations.

**Step 2: Prune the model**
```powershell
python surgery/prune.py --model models/base --output models/pruned --validate
```
This reduces model size by ~35% while maintaining quality.

**Step 3: Extract steering vectors**
```powershell
python surgery/steer.py --model models/pruned --extract-vectors
```
This creates personality vectors for romantic, caring, playful traits.

**Step 4: Embed personality facts (optional)**
```powershell
python surgery/edit.py --model models/pruned --embed --verify
```
This embeds facts like "I am your girlfriend" directly into weights.

### 4. Chat with Eris

```powershell
# Interactive chat
python inference/chat.py

# Or use the launcher
.\run_eris.ps1

# Single prompt
python inference/chat.py --prompt "Hello, how are you?"
```

## 📊 Expected Results

After surgery:
- **Model size**: ~4GB (down from ~6.4GB)
- **Speed**: ~1.3x faster inference
- **Quality**: Comparable to original
- **Personality**: Embedded in weights (no external DB)

## 🛠️ Advanced Usage

### Custom Pruning

```powershell
# More aggressive pruning
python surgery/prune.py --target-reduction 0.5

# Less aggressive
python surgery/prune.py --target-reduction 0.2
```

### Test Steering

```powershell
python surgery/steer.py --model models/pruned --test
```

### Modify Personality

Edit `configs/personality.yaml`:
```yaml
traits:
  romantic: 1.0  # Increase/decrease trait strength
  playful: 0.5
steering_strength: 1.2  # Overall intensity
```

## 🔧 Troubleshooting

**Out of VRAM?**
- Use 4-bit quantization: Edit scripts to use `load_in_4bit=True`
- Or use CPU: `device_map="cpu"` (slower)

**Model not loading?**
- Check you have the base model in `models/base/`
- Verify HuggingFace login: `huggingface-cli whoami`

**Steering not working?**
- Make sure you ran `steer.py --extract-vectors` first
- Check `steering_vectors.pkl` exists

## 📁 File Structure

```
EvlfEris/
├── models/
│   ├── base/              ← Download model here
│   └── pruned/            ← Pruned model saved here
├── surgery/
│   ├── analyze.py         ← Run first
│   ├── prune.py           ← Run second
│   ├── steer.py           ← Run third
│   └── edit.py            ← Optional fourth
├── inference/
│   └── chat.py            ← Chat interface
└── configs/
    └── personality.yaml   ← Edit personality here
```

## 🆚 Comparison with Evlf

| Feature | Evlf (RAG) | Eris (Surgery) |
|---------|------------|----------------|
| Memory | External DB | Embedded |
| Dependencies | ChromaDB | None |
| Size | 6.4GB + DB | ~4GB |
| Speed | Slower | Faster |
| Setup | Complex | Simple |
| Portability | Low | High |

## 💡 Tips

1. **Always analyze first** - Run `analyze.py` before pruning for best results
2. **Start conservative** - Begin with 30-35% pruning, increase if quality is good
3. **Test steering** - Verify personality vectors work before chatting
4. **Backup models** - Keep a copy of base model before editing

## 📚 Next Steps

- Experiment with different pruning levels
- Create custom steering vectors
- Try different personality configurations
- Compare with original Evlf

---

**Sister Project:** [Evlf](../Evlf) - Original RAG-based version
