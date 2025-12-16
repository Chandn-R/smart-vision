import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import random

# --- CONFIGURATION ---
DATA_PATH = os.path.expanduser("../data/processed/lstm_data")
MODEL_SAVE_PATH = "../models/lstm_action_recognition.pth"
EPOCHS = 60           
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
MAX_NORMAL_SAMPLES = 7000 

# --- 1. DEFINE DATASET ---
class ActionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. DEFINE LSTM MODEL (FIXED) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        # FIX: Save these as class attributes so 'forward' can use them
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. TRAINING FUNCTION ---
def train_model():
    print(" Loading Data & Balancing Classes...")
    
    features = []
    labels = []
    
    class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    class_map = {name: i for i, name in enumerate(class_names)}
    
    print(f"   Classes Map: {class_map}")

    for class_name in class_names:
        class_dir = os.path.join(DATA_PATH, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        if class_name == "normal" and len(files) > MAX_NORMAL_SAMPLES:
            print(f"     Capping '{class_name}': {len(files)} -> {MAX_NORMAL_SAMPLES} samples")
            files = random.sample(files, MAX_NORMAL_SAMPLES)
        else:
            print(f"   ✅ Loading '{class_name}': {len(files)} samples")

        for file in files:
            data = np.load(os.path.join(class_dir, file))
            features.append(data)
            labels.append(class_map[class_name])

    X = np.array(features)
    y = np.array(labels)

    print("\n  Calculating Class Weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"   Weights: {class_weights}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(ActionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ActionDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Device Selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps') # Use Mac GPU
        
    print(f" Training on {device}...")

    input_size = X.shape[2]
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_map': class_map,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS
            }, MODEL_SAVE_PATH)

    print(f"\n Final Best Accuracy: {best_acc:.2f}%")
    print(f" Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()