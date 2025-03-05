from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import streamlit as st

tab1, tab2 = st.tabs(["Home Page","Metrics"])

def metric(truevalue, predictedvalue):
    print(f"Accuracy : {accuracy_score(truevalue, predictedvalue)}")
    print(f"Precision : {precision_score(truevalue, predictedvalue, average= 'macro')}")
    print(f"Recall : {recall_score(truevalue, predictedvalue, average= 'macro')}")
    print(f"f1_Score: {f1_score(truevalue, predictedvalue, average= 'macro' )}")

transform  = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor()
])


test_dataset = datasets.ImageFolder(root=r"D:\PYTHON\VSCODE-PY\emotions\Emotions_1\test",transform= transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*1*1,256)
        self.fc2 = nn.Linear(256,7)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = torch.relu(self.pool(X))
        X = self.conv3(X)
        X = torch.relu(self.pool(X))
        X = self.conv4(X)
        X = torch.relu(self.pool(X))
        X = self.conv5(X)
        X = torch.relu(self.pool(X))
        X = torch.flatten(X,1)
        X = self.fc1(X)
        X= self.dropout(X)
        X = self.fc2(X)
        return X


model = EmotionClassifier()
model.load_state_dict(torch.load(r"D:\PYTHON\VSCODE-PY\emotions\emotion.pth"))

model.eval()  # Set the model to evaluation mode

test_true = []
test_pred = []

with torch.no_grad():
  for images,labels in test_loader:
    outputs = model(images)
    _,predictions=torch.max(outputs,1)
    test_true.extend(labels)
    test_pred.extend(predictions)

with tab1:

    st.title("Emotion Detection using Convolutional neural network")

    uploaded_file = st.file_uploader("Upload an Gray Scale image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB") 
        st.image(image, caption="Uploaded Image", use_column_width=True)
        inp_img = transform(image).unsqueeze(0)
    #new_img = r"C:\Users\vinot\Downloads\images.jpg"
    #image = Image.open(new_img).convert("RGB")



    if st.button("Prediction"):
        with torch.no_grad():
            output = model(inp_img)
        _,predictions=torch.max(output,1)

        class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        prd_lbl = class_labels[predictions.item()]
        out = f"The predicted Label is {prd_lbl}"
        #st.markdown(f"<h3 style='color: red;'><b>{out}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: red;'><b>The Predicted Label: {str(prd_lbl)}</b></h3>", unsafe_allow_html=True)

with tab2:

    accuracy = accuracy_score(test_true, test_pred)
    Precision = precision_score(test_true, test_pred, average= 'macro')
    Recall =    recall_score(test_true, test_pred, average= 'macro')
    f1_Score = f1_score(test_true, test_pred, average= 'macro' )


    st.title("Metrics - Test Dataset")
    st.markdown(f"<h3 style='color: blue;'><b>Accuracy  : {accuracy}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: blue;'><b>Precision  : {Precision}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: blue;'><b>Recall  : {Recall}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: blue;'><b>f1_score  : {f1_Score}</b></h3>", unsafe_allow_html=True)
