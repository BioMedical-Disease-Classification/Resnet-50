import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL
model.load_state_dict(torch.load("resnet50.pth", map_location=device))
model.to(device)
model.eval()

# DATA (REAL TEST DATA)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("data/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(val_dataset.class_to_idx)  # MUST be correct mapping

# GET PREDICTIONS
def get_predictions(model, val_loader, device):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

y_true, y_pred = get_predictions(model, val_loader, device)

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
print(cm)


# PLOT
class_names = ["NORMAL", "BACTERIAL", "VIRAL"]

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
