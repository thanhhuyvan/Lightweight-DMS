# Member Onboarding Guide: Hybrid Evolution

Welcome to the **Lightweight-DMS** R&D team. We are currently implementing a **Hybrid Multimodal Framework**.

## 👥 The Squad Structure
Work is divided into 3 specialized squads:

1.  **Data Squad (Module 1):**
    *   Focus: Image processing, Padding, solvePnP, and Data Loading.
    *   Stack: OpenCV, MediaPipe, Pandas.
2.  **Model Squad (Module 2):**
    *   Focus: CNN architecture, FiLM modulation layers, and GRU logic.
    *   Stack: PyTorch, Torchvision.
3.  **F1 Squad (Module 3):**
    *   Focus: Loss functions (Triplet Loss), Hyperparameter tuning, and Validation.
    *   Stack: Scikit-learn, PyTorch.

## 📐 Technical Mandates
*   **Isotropic Padding:** Never resize an image before making it square.
*   **Residual Fallback:** Your Deep Learning code must output a $\Delta S$ (delta score) to adjust the baseline, not replace it.
*   **No Leakage:** Check your `GroupKFold` logic twice. Participants must stay separated.

## 🚀 Getting Started
1.  Read [METHODOLOGY.md](./METHODOLOGY.md) for the math behind the pipeline.
2.  Check [KANBAN.md](./KANBAN.md) for your assigned task.
3.  Run `python src/train_final.py` to see the current $0.5422$ baseline.
