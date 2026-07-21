"""
export_onnx.py — Export PyTorch FiLM+GRU Model to ONNX
------------------------------------------------------
Converts the trained Stage E FiLM+GRU model checkpoint to ONNX format.
This allows high-performance, lightweight deployment using ONNX Runtime
without requiring the heavy PyTorch package in the production environment.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Include src directory in python path to resolve imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import model definition from predict.py (or we can define it here to be self-contained)
from docker.predict import FiLMGRUModel

def export_to_onnx(model_path: str, output_path: str):
    print(f"Loading PyTorch checkpoint: {model_path}")
    device = torch.device("cpu")
    
    # Initialize model and load state dict
    model = FiLMGRUModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    # Define dummy inputs matching the shape expected by forward()
    # forward(self, patches, valid_mask, geo, confidence=None)
    batch_size = 1
    seq_len = 40
    num_channels = 3
    patch_size = 24
    geo_dim = 11
    
    dummy_patches = torch.randn(batch_size, seq_len, num_channels, patch_size, patch_size)
    dummy_vmask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    dummy_geo = torch.randn(batch_size, geo_dim)
    dummy_conf = torch.ones(batch_size, seq_len, dtype=torch.float32)
    
    print("Exporting model to ONNX...")
    
    # Export with static shapes (batch_size=1, seq_len=40) for maximum compatibility.
    torch.onnx.export(
        model,
        (dummy_patches, dummy_vmask, dummy_geo, dummy_conf),
        output_path,
        export_params=True,
        opset_version=14,  # Opset 14 is widely supported and handles nn.GRU well
        do_constant_folding=True,
        input_names=["patches", "valid_mask", "geo", "confidence"],
        output_names=["logits"]
    )
    print(f"ONNX model successfully saved to: {output_path}")

if __name__ == "__main__":
    # Get absolute project root (E:\Buồn_ngủ) relative to the script location
    project_root = Path(__file__).resolve().parents[1]
    
    model_checkpoint = str(project_root / "models" / "models" / "film_gru_fold3.pth")
    onnx_output = str(project_root / "models" / "models" / "film_gru_fold3.onnx")
    
    # Ensure models/models directory exists
    Path(onnx_output).parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(model_checkpoint, onnx_output)
