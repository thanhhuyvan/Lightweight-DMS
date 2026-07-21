import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Applies an affine transformation to feature maps: y = gamma * x + beta.
    """
    def __init__(self, num_features):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x, gamma, beta):
        # x: [Batch, Channels, H, W]
        # gamma/beta: [Batch, Channels]
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return (gamma * x) + beta

class FiLMGenerator(nn.Module):
    """
    MLP that generates FiLM parameters (gamma, beta) from a geometry vector.
    """
    def __init__(self, input_dim=12, num_features=16):
        super(FiLMGenerator, self).__init__()
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_features * 2)
        )

    def forward(self, x):
        params = self.mlp(x)
        # Initialize gamma around 1.0, beta around 0.0
        gamma = params[:, :self.num_features] + 1.0
        beta = params[:, self.num_features:]
        return gamma, beta

class HybridBackbone(nn.Module):
    """
    MobileNetV3-Small modified with FiLM layers for geometry-steered vision.
    Input: [Batch, 1, 24, 24] grayscale patch.
    """
    def __init__(self, geo_dim=12, width_mult=0.5):
        super(HybridBackbone, self).__init__()
        
        # 1. Load MobileNetV3 Small (Lightweight)
        # We don't use pretrained weights because our input is 1-channel grayscale 24x24
        base_model = mobilenet_v3_small(weights=None)
        
        # 2. Modify first layer for 1-channel grayscale
        self.first_conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.first_bn = nn.BatchNorm2d(16)
        self.first_act = nn.Hardswish()
        
        # 3. FiLM Modulation for early features
        self.film1 = FiLMLayer(16)
        self.gen1 = FiLMGenerator(input_dim=geo_dim, num_features=16)
        
        # 4. Extract blocks from MobileNet
        # For 24x24 input, the spatial dimension reduces quickly.
        # We use the first few blocks.
        self.blocks = base_model.features[1:4] # Take up to index 3
        
        # 5. Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 24 # Feature dim after index 3 in MobileNetV3-Small (approx)
        
    def forward(self, patches, geo_vector):
        # patches: [Batch, 1, 24, 24]
        # geo_vector: [Batch, 12]
        
        x = self.first_conv(patches)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        # Apply FiLM at the earliest stage
        gamma, beta = self.gen1(geo_vector)
        x = self.film1(x, gamma, beta)
        
        # Pass through remaining blocks
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        return x

class HybridNet(nn.Module):
    """
    The Final Multimodal Model: Fuses 3 Patch Backbones + GRU for temporal modeling.
    """
    def __init__(self, geo_dim=12, hidden_dim=64):
        super(HybridNet, self).__init__()
        
        # Three separate heads or shared weights? 
        # Shared weights for L/R eyes, separate for Mouth is usually more efficient.
        self.eye_backbone = HybridBackbone(geo_dim=geo_dim)
        self.mouth_backbone = HybridBackbone(geo_dim=geo_dim)
        
        # Fusion Layer: [EyeL_Features + EyeR_Features + Mouth_Features + Geometry]
        # Dim: 24 + 24 + 24 + 12 = 84
        self.fusion_dim = 24 * 3 + geo_dim
        
        # GRU for temporal persistence
        self.gru = nn.GRU(input_size=self.fusion_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3) # 0: Alert, 1: Low, 2: Drowsy
        )
        
    def forward(self, l_eye, r_eye, mouth, geo_seq):
        # l_eye, r_eye, mouth: [Batch, Seq, 1, 24, 24]
        # geo_seq: [Batch, Seq, 12]
        
        batch_size, seq_len = l_eye.shape[:2]
        
        # Process each frame in the sequence
        # (Alternatively, reshape to [Batch*Seq, ...] then reshape back)
        l_feat = self.eye_backbone(l_eye.view(-1, 1, 24, 24), geo_seq.view(-1, 12))
        r_feat = self.eye_backbone(r_eye.view(-1, 1, 24, 24), geo_seq.view(-1, 12))
        m_feat = self.mouth_backbone(mouth.view(-1, 1, 24, 24), geo_seq.view(-1, 12))
        
        # Concatenate features
        # [Batch*Seq, FusionDim]
        fused = torch.cat([l_feat, r_feat, m_feat, geo_seq.view(-1, 12)], dim=1)
        
        # Reshape for GRU: [Batch, Seq, FusionDim]
        fused = fused.view(batch_size, seq_len, -1)
        
        # GRU
        gru_out, _ = self.gru(fused)
        
        # Take the last state for classification
        last_out = gru_out[:, -1, :]
        
        logits = self.classifier(last_out)
        return logits

if __name__ == "__main__":
    # Test Architecture
    model = HybridNet()
    print(f"HybridNet initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Mock Data: Batch=2, Seq=5 frames, 1x24x24 patches, 12D Geo
    l = torch.randn(2, 5, 1, 24, 24)
    r = torch.randn(2, 5, 1, 24, 24)
    m = torch.randn(2, 5, 1, 24, 24)
    g = torch.randn(2, 5, 12)
    
    out = model(l, r, m, g)
    print(f"Output shape: {out.shape}") # Should be [2, 3]
