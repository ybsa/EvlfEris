# Evlf Eris (Next-Gen AI Agent)

**An ultra-efficient, local-first AI agent built on the 2026 "Next-Gen" stack (JEPA, mHC, MTP, TTT) optimized for consumer hardware (4GB VRAM).**

## 🚀 The Stack
- **Base Model**: Llama-3.2 (3B)
- **Training**: JEPA (Joint-Embedding Predictive Architecture) - Learning latent concepts instead of next-token prediction.
- **Architecture**:
    - **mHC (Manifold-Constrained)**: Stable local training.
    - **MTP (Multi-Token Prediction)**: Faster generation.
    - **TTT (Test-Time Training)**: Real-time learning and memory.
- **Optimization**: Structured Pruning (20% reduction) + 4-bit Quantization (bitsandbytes).

## 🗺️ Roadmap
1. **Preparation**: Clean the model (Pruning + Unlearning).
2. **Surgery**: Inject mHC, MTP, and TTT components.
3. **Training**: Teach it about Air Quality & AI Engineering using JEPA.
4. **Deployment**: Real-time TTT inference loop with Flutter UI.

## 🛠️ Setup
```bash
pip install -r requirements.txt
python download_model.py
```

## ⚠️ Hardware Constraints
Strict 4GB VRAM limit. All engineered components utilize LISA (Layer-wise Importance Sampling for Adam) and targeted quantization to fit.
