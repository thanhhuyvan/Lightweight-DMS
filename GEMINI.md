# Gemini CLI Mandates (Hybrid Strategy)

This file contains foundational instructions for the AI agent. These take precedence over general defaults.

## 🛡️ Architectural Mandates
1.  **Hybrid Logic:** Prioritize solutions that combine geometric data (EAR/Pose) with image patches.
2.  **Safety Net:** Ensure all classification scripts support the **Residual Fallback** formula: $S_{final} = S_{base} + \Delta S$.
3.  **Min-Max Standard:** Always apply per-participant scaling for ocular features.
4.  **Isotropic Rule:** Image patches MUST be padded to a 1:1 ratio before resizing.

## 📁 Workspace Standards
*   **Configs:** Reference `src.core_config` for all paths.
*   **Reports:** Save all diagnostic plots in `report/diagnostics/` and final charts in `report/final/`.
*   **Models:** Save all `.joblib` or `.pth` files in `models/`.

## 📊 Current Knowledge Base
*   **Baseline F1:** $0.5422$ (Validated on 10,944 windows, 10s duration).
*   **Current SOTA Goal:** F1 > $0.80$ using FiLM + GRU.
