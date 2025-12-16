import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetClassifier(nn.Module):
    
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        
        self.backbone.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":

    print("=" * 70)
    print("Model Test (cnn_model.py)")
    print("=" * 70)
    
    model = EfficientNetClassifier(num_classes=16, dropout_rate=0.3)
    model.eval()
    
    dummy_input = torch.randn(4, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample (first 5 values): {output[0][:5]}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("Model is ready for training!")
    print("=" * 70)