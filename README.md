# Lightweight-DMS: Hybrid Driver Monitoring System

A high-performance, multi-stage hybrid monitoring pipeline designed for real-time drowsiness detection on edge devices.

## 🚀 Project Vision
To bridge the gap between deterministic geometric math and stochastic deep learning, ensuring robust detection in real-world driving environments.

## 🏗️ Architecture: Robust Hybrid Framework
The system leverages a **Residual Fallback** strategy to ensure industrial-grade reliability:
1.  **Branch A (Geometry):** MediaPipe + 3D solvePnP + Asymmetric EAR (Stable Backbone).
2.  **Branch B (Appearance):** Landmark-guided Isotropic Patches + MobileNetV3 (Visual Detail).
3.  **Fusion:** Feature-level Linear Modulation (FiLM) + Gated GRU.
4.  **Safety Net:** Residual adjustment $\Delta S$ preserves a guaranteed **Baseline F1-Score of 0.5422**.

## 📊 Current Status
- **Baseline (Geometry Only):** F1 = $0.5422$ (XGBoost, GroupKFold validated).
- **In Development:** Hybrid CNN-GRU integration with FiLM modulation.

## 🛠️ Tech Stack
- **Vision:** MediaPipe, OpenCV, solvePnP.
- **Learning:** XGBoost (Baseline), PyTorch (Hybrid).
- **Optimization:** Savitzky-Golay filtering, Isotropic Padding, Min-Max Scaling.

## 📖 Documentation
- [METHODOLOGY.md](./METHODOLOGY.md) - Full technical specification.
- [KANBAN.md](./KANBAN.md) - Real-time task tracking.
- [MEMBER_GUIDE.md](./MEMBER_GUIDE.md) - Onboarding and squad structure.
