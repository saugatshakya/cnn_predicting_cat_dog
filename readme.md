# Cat vs Dog Classification with Custom CNN

This repository contains a PyTorch-based Convolutional Neural Network (CNN) for classifying images of **cats and dogs**. The notebook demonstrates data loading, preprocessing, training, evaluation, error analysis, and inference on unseen images.

---

## 📂 Dataset

The notebook expects the dataset to be organized as follows:

```

data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
│   ├── cat/
│   └── dog/

```

- **Training images:** `data/train`
- **Testing images:** `data/test`

Images should be labeled by their folder names (`cat` or `dog`).

---

## ⚙️ Dependencies

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- PIL
- requests

Install dependencies with:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow requests
```

---

## 📝 Notebook Overview

1. **Data Loading & Augmentation**

   - Used `ImageFolder` to load images.
   - Applied optional **online augmentation**: `RandomRotation` and `RandomHorizontalFlip`.

2. **CNN Architecture**

   - Two convolutional layers followed by max pooling.
   - Three fully connected layers with ReLU activations.
   - Final layer outputs 2 classes: `cat` and `dog`.

3. **Training & Evaluation**

   - Loss: `CrossEntropyLoss`
   - Optimizer: `Adam`
   - Trained for 3 epochs with limited batch examples for demonstration.
   - Accuracy and loss are tracked per epoch.

4. **Metrics & Error Analysis**

   - Calculated **Accuracy**, **Precision**, **Recall**, and **F1-score**.
   - Plotted misclassified images to understand common errors.
   - Observed challenges with low-light, cropped, or ambiguous images.

5. **Inference on Unseen Images**

   - Tested the model on 4 internet images (2 cats, 2 dogs).
   - All predictions were correct, showing reasonable generalization.

---

## 📈 Results

| Setting                             | Test Accuracy |
| ----------------------------------- | ------------- |
| With augmentation (rotation + flip) | 75.3%         |
| Without augmentation                | 76.4%         |

> Observation: For this dataset, simple augmentation slightly reduced test accuracy, likely due to dataset consistency. Augmentation generally improves robustness on more varied datasets.

---

## 🖼 Misclassified Examples

The notebook visualizes misclassified images along with their true and predicted labels, highlighting areas where the model struggles (e.g., occluded or ambiguous images).

---

## 🔮 Inference

Example usage to predict a new image:

```python
from PIL import Image
import torch
from torchvision import transforms

# Load trained model
model = ConvolutionalNetwork()
model.load_state_dict(torch.load("models/CustomImageCNNModel.pt", map_location="cpu"))
model.eval()

# Preprocess image
img = Image.open("path_to_image.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {['cat', 'dog'][pred_class]}")
```

---

## 💡 Key Takeaways

- The CNN can distinguish cats and dogs with reasonable accuracy.
- Online data augmentation increases robustness but may slightly reduce accuracy on clean, consistent datasets.
- Misclassified images reveal limitations: ambiguous textures, partial occlusions, or low-quality inputs.
- For improved performance, consider **pretrained models** like ResNet or VGG, and additional augmentation strategies.

---

## 📁 File Structure

```
.
├── data/                     # Training & testing images
├── models/                   # Saved CNN model weights
├── Cat_vs_Dog_CNN.ipynb      # Main notebook
└── README.md
```

---

## ⚡ Future Improvements

- Use **transfer learning** with pretrained CNNs (ResNet, VGG) for better accuracy.
- Add **more diverse augmentations** (color jitter, random crops).
- Train on a **larger dataset** for better generalization.
- Implement **real-time inference** using webcam input.

---
