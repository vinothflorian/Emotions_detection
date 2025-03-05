# Emotion Classification Model

## Overview
This project is a **Deep Learning-based Emotion Classification Model** implemented using **PyTorch**. The model is trained on an image dataset containing different emotions and can classify images into one of seven emotion categories.

## Dataset
The dataset is organized into folders for each emotion and is loaded using `torchvision.datasets.ImageFolder`. The dataset is preprocessed and transformed before being passed to the model.

- Training Dataset Path: `D:\PYTHON\VSCODE-PY\emotions\Emotions_1\train`

## Dependencies
Make sure you have the following dependencies installed:

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn
```

## Model Architecture
The `EmotionClassifier` model consists of:
- **5 Convolutional Layers**
- **MaxPooling Layers** after every convolutional layer
- **Dropout Layer** to prevent overfitting
- **Fully Connected Layers** with ReLU activation 
- **Final Output Layer** with 7 classes (softmax activation is handled by CrossEntropyLoss)

## Training Process
The model is trained using:
- **Loss Function:** `CrossEntropyLoss`
- **Optimizer:** `Adam` with `lr=0.0001`
- **Batch Size:** `32`
- **Epochs:** `20`

Each epoch prints:
- **Loss per epoch**
- **Accuracy per epoch**

After training, evaluation metrics such as **Accuracy, Precision, Recall, and F1-Score** are calculated.

## Training the Model
Run the following Python script to train the model:

```python
python train.py
```

After training, the model is saved as:

```plaintext
D:\PYTHON\VSCODE-PY\emotions\emotion.pth
```

## Evaluation Metrics
The following metrics are computed after training:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Saving the Model
The trained model is saved as:
```python
torch.save(model.state_dict(), r"D:\PYTHON\VSCODE-PY\emotions\emotion.pth")
```

## Prediction for New Datapoints

The saved model is loaded in Emot_predict.py and used to do the prediction for new datapoints


