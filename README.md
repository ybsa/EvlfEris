# 🔬 Evlf Eris

**Model Surgery & Pruning** - A lightweight AI girlfriend powered by surgically modified Llama-3.2-3B.

Unlike the original Evlf (which uses RAG + ChromaDB), **Eris** embeds personality directly into model weights through:
- **Structured Pruning** - 30-40% smaller, faster model
- **Activation Steering** - Personality vectors guide behavior
- **Knowledge Editing** - Facts embedded in weights (no external DB)

## 🚀 Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the pruning process
python surgery/prune.py

# Chat with Eris
python inference/chat.py
```

## 🧠 How It Works

**Core Differences from Evlf:**

| Feature | Evlf (RAG) | Eris (Surgery) |
|---------|------------|----------------|
| Memory | ChromaDB | Embedded in weights |
| Model Size | 6.4GB | ~4GB (pruned) |
| Dependencies | ChromaDB, embeddings | Pure PyTorch |
| Portability | Needs DB files | Single model file |
| Speed | Slower (RAG lookup) | Faster (no lookup) |

**Surgery Techniques:**

1. **Structured Pruning** (`surgery/prune.py`)
   - Removes redundant layers and attention heads
   - Maintains quality while reducing size
   - Automatic importance scoring

2. **Activation Steering** (`surgery/steer.py`)
   - Extracts personality vectors from examples
   - Guides model behavior at inference time
   - No retraining required

3. **Knowledge Editing** (`surgery/edit.py`)
   - ROME/MEMIT techniques
   - Embeds facts directly in weights
   - "I am Evlf", "You are my boyfriend", etc.

## 📂 Project Structure

```
EvlfEris/
├── README.md
├── requirements.txt
├── models/
│   ├── base/              # Original Llama-3.2-3B
│   └── pruned/            # Pruned versions
├── surgery/
│   ├── prune.py          # Structured pruning
│   ├── steer.py          # Activation steering
│   ├── edit.py           # Knowledge editing
│   └── analyze.py        # Model analysis
├── inference/
│   └── chat.py           # Optimized chat interface
└── configs/
    └── personality.yaml   # Personality config
```

## 🎯 Goals

- ✅ 30-40% model size reduction
- ✅ 20-30% inference speed improvement
- ✅ No external dependencies (no ChromaDB)
- ✅ Personality embedded in weights
- ✅ Single model file deployment

## 🛠️ Development

**Analyze Model:**
```bash
python surgery/analyze.py --model models/base
```

**Prune Model:**
```bash
python surgery/prune.py --target-reduction 0.35
```

**Apply Steering:**
```bash
python surgery/steer.py --extract-vectors
```

**Edit Knowledge:**
```bash
python surgery/edit.py --facts configs/personality.yaml
```

## 📊 Comparison with Evlf

| Metric | Evlf | Eris |
|--------|------|------|
| Model Size | 6.4GB | ~4GB |
| Memory Usage | High (DB + model) | Low (model only) |
| Inference Speed | ~15 tok/s | ~20 tok/s |
| Setup Complexity | Medium | Low |
| Portability | Low | High |

---

**Sister Project:** [Evlf](../Evlf) - RAG-based version with ChromaDB memory
