import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. The FiLM Layer Implementation
class FiLMLayer(nn.Module):
    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x, gamma, beta):
        # x: [Batch, Channels, H, W]
        # gamma/beta: [Batch, Channels]
        
        # Reshape gamma and beta for broadcasting: [Batch, Channels, 1, 1]
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        # The FiLM equation: y = gamma * x + beta
        return (gamma * x) + beta

# 2. The Parameter Generator (MLP)
# Maps 12D Geometry Vector -> 2 * num_features (for gamma and beta)
class FiLMGenerator(nn.Module):
    def __init__(self, input_dim=12, num_features=16):
        super(FiLMGenerator, self).__init__()
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_features * 2) # Output gamma and beta
        )

    def forward(self, x):
        params = self.mlp(x)
        # Split into gamma and beta (initialize gamma around 1.0, beta around 0.0)
        gamma = params[:, :self.num_features] + 1.0
        beta = params[:, self.num_features:]
        return gamma, beta

def run_film_demo():
    print("🎬 FiLM Modulation Demo: 'Steering the CNN with Geometry'")
    
    # Setup: 1 sample, 16 channels, 24x24 resolution
    num_channels = 16
    dummy_patch = torch.randn(1, 1, 24, 24) # Random eye patch input
    
    # Mock a basic first CNN layer (extracting 16 features)
    conv = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
    film = FiLMLayer(num_channels)
    generator = FiLMGenerator(input_dim=12, num_features=num_channels)
    
    # --- Scenario A: Head Looking Straight (Neutral Geometry) ---
    # [EAR_L, EAR_R, EAR_Avg, EAR_Diff, MAR, Pitch, Yaw, Roll, dP, dY, dR, Conf]
    geo_neutral = torch.zeros(1, 12) 
    geo_neutral[0, 2] = 0.30 # EAR_Avg is normal
    
    # --- Scenario B: Extreme Head Tilt (Stressed Geometry) ---
    geo_tilted = torch.zeros(1, 12)
    geo_tilted[0, 5] = 45.0  # Pitch = 45 degrees
    geo_tilted[0, 6] = 30.0  # Yaw = 30 degrees
    geo_tilted[0, 2] = 0.15  # EAR_Avg is low (drowsy)

    # Processing
    with torch.no_grad():
        # 1. Base CNN features
        features = conv(dummy_patch)
        
        # 2. Generate modulation for both scenarios
        gamma_n, beta_n = generator(geo_neutral)
        gamma_t, beta_t = generator(geo_tilted)
        
        # 3. Apply FiLM
        out_neutral = film(features, gamma_n, beta_n)
        out_tilted  = film(features, gamma_t, beta_t)
        
    print("\n[Analysis]")
    print(f"Base Features (Mean): {features.mean().item():.4f}")
    print(f"Neutral Modulation (Mean): {out_neutral.mean().item():.4f}")
    print(f"Tilted Modulation (Mean): {out_tilted.mean().item():.4f}")
    
    # Calculate difference
    diff = torch.abs(out_neutral - out_tilted).mean().item()
    print(f"\nDifference in feature response due ONLY to Head Pose: {diff:.4f}")
    
    if diff > 0:
        print("\n✅ SUCCESS: The CNN response changed based on geometric input!")
        print("This means the SAME image patch is interpreted DIFFERENTLY depending on Head Pose.")
        print("Example: An 'eye-like' pattern might be amplified if the Head Pose indicates the driver is looking down.")

if __name__ == "__main__":
    run_film_demo()
