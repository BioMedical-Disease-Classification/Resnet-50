 Data Engineering Specifications
 Assigned to: Ibrahime & Amrith1. 
 Objective: To build the pipeline that feeds images into the ResNet-50 "Brain" created by Sydney & Matthew. You must ensure the data is normalized and balanced before it hits the model.
 Please implement the following in a new file:
Resize all images to 224x224.Apply ImageNet Normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
 Augmentation (Training only): Include random horizontal flips and rotations . Weighted Loss Calculation: * Calculate the frequency of the 3 classes (Normal, Bacteria, Virus).
 Provide a WeightedCrossEntropyLoss function to the Architects to handle the class imbalance.
 
 Integration Point
 Your train_loader and val_loader must be compatible with the train_model function in resnet_core.py.

** A general guidance:**
Your pipeline is the "bridge" between raw images and the model. 
To prevent crashes, you must follow these hard requirements:

**Dimensionality**: Every image must be exactly 224 x 224 pixels.
**Normalization**: You must use the ImageNet Mean and Standard Deviation. If you use random values, the pre-trained weights in the backbone will be useless.
mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

Handling the "Class Imbalance": Our dataset has significantly more Pneumonia cases than Normal cases. If you don't handle this, the model will learn to just "guess" Pneumonia every time to get a high accuracy score.
Solution: You must calculate Class Weights. 
**Formula**: Weight = total weight/ (number of classes * samples in class)
You will pass these weights into the nn.CrossEntropyLoss(weight=your_weights) function.

**Data Augmentation**: Since pediatric patients move during X-rays, we need to simulate that. However, do not over-distort the images.
These types of augmentations would be good, but you dont need to do all of them: Random Horizontal Flips, small Rotations +/- 10º, and slight Brightness adjustments. 

**Please do not do these adjustment**s: Unsafe: Vertical flips (hearts aren't upside down) or heavy cropping (we need to see the whole lung).
Categorization Logic Ensure your DataLoader maps the folders to these exact integers: 0: NORMAL; 1: BACTERIAL; 2: VIRAL. **If these are swapped, the confusion matrix will not be true.**
