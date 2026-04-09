import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    """
    Creates the 'Bridge' between raw images and the ResNet-50 model.
    Handles resizing, normalization, and augmentation.
    """
    
    # 1. Define Transformations (Hard Requirements for ResNet-50)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ])
    }

    # 2. Create Datasets (Resilient Version)
    image_datasets = {}
    for x in ['train', 'val', 'test']:
        path = os.path.join(data_dir, x)
        if os.path.exists(path):
            try:
                dataset = datasets.ImageFolder(path, data_transforms[x])
                if len(dataset) > 0:
                    # Fix label order — ImageFolder defaults to alphabetical (BACTERIAL=0, NORMAL=1, VIRAL=2)
                    # but spec requires NORMAL=0, BACTERIAL=1, VIRAL=2
                    dataset.class_to_idx = {'NORMAL': 0, 'BACTERIAL': 1, 'VIRAL': 2}
                    dataset.targets = [dataset.class_to_idx[dataset.classes[t]] for t in dataset.targets]
                    image_datasets[x] = dataset
                else:
                    print(f"Warning: Folder '{x}' is empty. Skipping.")
            except Exception as e:
                print(f"Warning: Could not load '{x}' folder. Error: {e}")

    # 3. Create DataLoaders only for valid datasets
    loaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0)
        for x in image_datasets.keys()
    }
    
    classes = image_datasets['train'].classes if 'train' in image_datasets else []
    
    return loaders, classes


def calculate_weighted_loss(data_dir, device):
    """
    Calculates Class Weights to handle the 'Class Imbalance' problem.
    Formula: Weight = total / (num_classes * samples_in_class)
    """
    categories = ['NORMAL', 'BACTERIAL', 'VIRAL']
    counts = []
    
    for cat in categories:
        path = os.path.join(data_dir, 'train', cat)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            counts.append(max(count, 1))
        else:
            counts.append(1)
    
    total_samples = sum(counts)
    num_classes = len(categories)
    
    weights = [total_samples / (num_classes * c) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    
    return nn.CrossEntropyLoss(weight=weights_tensor)


# --- Integration Point ---
if __name__ == "__main__":
    base_data_path = "data/chest_xray/chest_xray"
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(base_data_path):
        dataloaders, class_names = get_data_loaders(base_data_path)
        
        if dataloaders:
            print(f"\nSUCCESS: DataLoaders ready for: {list(dataloaders.keys())}")
            if class_names:
                print(f"Classes identified: {class_names}")
            
            criterion = calculate_weighted_loss(base_data_path, current_device)
            print(f"Weighted Loss Function initialized on {current_device}")

            # Sanity check — verify batch shape and label mapping
            if 'train' in dataloaders:
                imgs, labels = next(iter(dataloaders['train']))
                print(f"Batch shape: {imgs.shape}")        # expect [32, 3, 224, 224]
                print(f"Unique labels: {labels.unique()}") # expect tensor([0, 1, 2])
        else:
            print("Error: No valid images found in any folder.")
    else:
        print(f"Error: Data directory not found at {base_data_path}")