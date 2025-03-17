# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![Screenshot 2025-03-17 110723](https://github.com/user-attachments/assets/9ec0d7bf-98a6-4f8e-bd3c-3879b99a7590)

## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Model Design:
Input Layer: Number of neurons = features.
Hidden Layers: 2 layers with ReLU activation.
Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

## STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

## STEP 5:
Evaluation: Assess using accuracy, confusion matrix, precision, and recall.

## STEP 6:
Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).

## PROGRAM

### Name: KRISHNA KUMAR R
### Register Number:212223230107

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

```
```
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```
def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

## Dataset Information

![Screenshot 2025-03-17 111333](https://github.com/user-attachments/assets/7e3f36b7-a52b-41a1-90a1-48d8c662e577)

## OUTPUT



### Confusion Matrix

![Screenshot 2025-03-17 111417](https://github.com/user-attachments/assets/142cdadc-f9a4-47c8-86c0-4ec92253df06)

### Classification Report

![Screenshot 2025-03-17 111512](https://github.com/user-attachments/assets/a6d48b4d-ec71-4812-8331-6f7d25c054a0)

### New Sample Data Prediction

![Screenshot 2025-03-17 111541](https://github.com/user-attachments/assets/3970412f-8290-4107-bd6d-ce6e6111786a)

## RESULT
Thus the neural network classification model for the given dataset has been executed successfully.
