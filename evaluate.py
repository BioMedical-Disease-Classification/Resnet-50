import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def get_predictions(model, val_loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # convert softmax output → class index
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


y_true, y_pred = get_predictions(model, val_loader, device)

cm = confusion_matrix(y_true, y_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

class_names = ["NORMAL", "BACTERIAL", "VIRAL"]

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#normalize
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#plot
sns.heatmap(cm_normalized, annot=True, fmt='.2f',
            xticklabels=class_names,
            yticklabels=class_names)

#print acuracy
accuracy = (cm.diagonal().sum()) / cm.sum()
print("Accuracy:", accuracy)