#Sydney & Matthew W.
# Group: Model Architects
# Branch: resnet-core

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def initialize_resnet50(num_classes=3):
    """
    Sets up ResNet-50 for 3-class Pediatric Pneumonia classification.
    Ref: He et al. (2016)
    """
    print("Loading Pre-trained ResNet-50...")
    
    # 1. Load weights from ImageNet training
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # 2. FREEZE the backbone 
    # This prevents the model from 'forgetting' basic shapes/edges
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. ATTACH THE NEW HEAD
    # We replace the final Fully Connected (fc) layer
    # 
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3), # Prevents overfitting
        nn.Linear(512, num_classes) # Outputs: Normal, Bacteria, Viral
    )
    
    print("Model ready for 3-class classification.")
    return model

# To check if it works:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_resnet50(num_classes=3).to(device)
    print(f"Model is running on: {device}")


