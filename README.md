## 🧠 Problem Statement
Pneumonia is a serious lung infection that requires timely diagnosis. Chest X-rays are a primary tool for detecting pneumonia, but manual diagnosis can be time-consuming and subject to human error. The goal is to automate this process using a deep learning model.

## 🎯 Objective
To build and evaluate a binary classification model that can distinguish between normal and pneumonia chest X-rays using a CNN-based deep learning approach.

## 📂 Dataset Description
The dataset consists of chest X-ray images divided into two categories: Normal and Pneumonia.

Each class in the validation set had 8 images, resulting in a total of 16 validation samples.

Images were preprocessed using rescaling and resizing to 150x150 pixels.

Data was split into training, validation, and test sets using ImageDataGenerator.

## 🏗️ Model Architecture
Layer (type)	Output Shape	Param #
Conv2D (32 filters)	(148, 148, 32)	896
MaxPooling2D	(74, 74, 32)	0
Conv2D (64 filters)	(72, 72, 64)	18,496
MaxPooling2D	(36, 36, 64)	0
Conv2D (128 filters)	(34, 34, 128)	73,856
MaxPooling2D	(17, 17, 128)	0
Flatten	(36992)	0
Dense (128 units)	(128)	4,735,104
Dropout (0.5 rate)	(128)	0
Dense (1 unit, sigmoid)	(1)	129
Total Parameters		4,828,481

## 📊 Results
Training Accuracy: 94.60%

Validation Accuracy: 81.25%

Validation Loss: 0.5425

Model was trained for 5 epochs. Performance was evaluated on the validation set of 16 balanced images.

## 📉 Confusion Matrix Insight
Predicted: Normal	Predicted: Pneumonia
Actual: Normal	8	0
Actual: Pneumonia	8	0

Although the model achieved a high validation accuracy of 81.25%, the confusion matrix shows a severe class imbalance in predictions. All pneumonia cases were misclassified as normal, while all normal cases were correctly identified.

## ⚠️ Limitations
Small validation set (only 16 images).

Validation results are misleading due to class-specific misclassification.

Model lacks generalization on unseen pneumonia samples.

No evaluation was conducted on the test set.

## 🔍 Future Recommendations
Evaluate on a larger and more diverse validation/test set.

Implement data augmentation to improve generalization.

Incorporate class weighting or focal loss to handle class imbalance.

Use transfer learning with pre-trained CNNs like ResNet, EfficientNet, etc.

🧪 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/pneumonia-cnn.git
