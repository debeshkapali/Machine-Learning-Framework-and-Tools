# Pneumonia Detection Using CNN

This project involves the implementation of a Convolutional Neural Network (CNN) model to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. The objective is to demonstrate a basic deep learning pipeline from preprocessing to evaluation, while also reflecting on performance metrics and potential improvements.

---

## 🔍 Problem Statement

Pneumonia is a serious respiratory infection that can be diagnosed through chest X-ray imaging. However, manual diagnosis is time-consuming and requires medical expertise. This project aims to automate the classification of chest X-rays into **Pneumonia** or **Normal** classes using a simple CNN model.

---

## 🎯 Objective

- Develop a CNN-based binary image classifier for chest X-ray images.
- Evaluate model performance using validation accuracy, confusion matrix, and qualitative observations.
- Identify shortcomings in the current model and propose directions for future improvement.

---

## 📂 Dataset Description

The dataset is organized into three subsets:
- **Training Set:** Used for model training.
- **Validation Set:** Used during training to tune hyperparameters.
- **Test Set:** Reserved for final evaluation (not used in this iteration).

Each subset contains two folders:
- `NORMAL/`: Chest X-rays of healthy individuals.
- `PNEUMONIA/`: Chest X-rays of pneumonia cases.

There are **8 images per class in the validation set**, totaling 16 images. The dataset is balanced in this set.

> 🔗 **Download Dataset:**  
> (https://drive.google.com/drive/folders/1zP3kek2mvDL2Emckrk0OJooje6t_V_9h?usp=drive_link)

---

## 🧠 Model Architecture

The model follows a basic CNN structure with three convolutional layers, max pooling, and two fully connected layers. Below is the model summary:

| Layer (type)                  | Output Shape         | Param #   |
|------------------------------|----------------------|-----------|
| Conv2D (32 filters, 3x3)      | (148, 148, 32)        | 896       |
| MaxPooling2D (2x2)            | (74, 74, 32)          | 0         |
| Conv2D (64 filters, 3x3)      | (72, 72, 64)          | 18,496    |
| MaxPooling2D (2x2)            | (36, 36, 64)          | 0         |
| Conv2D (128 filters, 3x3)     | (34, 34, 128)         | 73,856    |
| MaxPooling2D (2x2)            | (17, 17, 128)         | 0         |
| Flatten                       | (36992,)              | 0         |
| Dense (128 units)             | (128,)                | 4,735,104 |
| Dropout (rate=0.5)            | (128,)                | 0         |
| Dense (1 unit, sigmoid)       | (1,)                  | 129       |

> 🔢 **Total Trainable Parameters:** 4.83M

---

## 📈 Results and Observations

### ✅ Validation Metrics:
- **Final Validation Accuracy:** 81.25%
- **Final Validation Loss:** 0.5425

### 📊 Confusion Matrix:

Despite achieving a decent validation accuracy, the confusion matrix revealed **complete misclassification of pneumonia images** as normal:

- **True Positive (Normal):** 8/8 correct
- **False Negative (Pneumonia → Predicted Normal):** 8/8 wrong

This suggests the model has become biased toward the **Normal** class during validation, possibly due to:

- Small validation size (only 8 images per class).
- Overfitting during training.
- Insufficient model complexity or generalization.

---

## 🚧 Shortcomings

- **No test set evaluation:** The final model was not tested on an unseen test set.
- **Misclassification bias:** The model failed to generalize for the pneumonia class in validation.
- **Small validation size:** Reduces statistical confidence in validation metrics.

---

## 🔮 Conclusion & Future Recommendations

This project successfully implements a CNN for binary classification of chest X-rays with reasonable training accuracy. However, significant room for improvement exists:

### 🔧 Future Improvements:
- Use **data augmentation** to increase variation in training samples.
- Implement **class weighting** or **focal loss** to address misclassification bias.
- Train for more epochs with **early stopping** to prevent overfitting.
- Apply **transfer learning** using pretrained networks like ResNet or VGG.
- Evaluate performance on a **larger, separate test set** for real-world reliability.
