import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import copy

# --- CONFIGURATION ---
DATA_PATH = os.path.expanduser("../data/processed/new_L_lstm_data")
MODEL_SAVE_PATH = "../models/lstm_action_recognition_pro.pth" # New filename
EPOCHS = 80              # More epochs for the scheduler to work
BATCH_SIZE = 64          # Bigger batch for stable gradients
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128        # Doubled brain size
NUM_LAYERS = 2
DROPOUT_RATE = 0.4       # Higher dropout to prevent overfitting

# TARGET SAMPLES per class (We force all classes to match this count)
TARGET_SAMPLES = 6000 

# --- 1. DATASET WITH NOISE AUGMENTATION ---
class ActionDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        # On-the-fly Data Augmentation (Only for Training)
        if self.augment:
            # Add random jitter (simulate shaky camera/sensor noise)
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
        return x, y

# --- 2. BIDIRECTIONAL LSTM MODEL ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # bidirectional=True doubles the output features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=DROPOUT_RATE, bidirectional=True)
        
        # Input to FC is hidden_size * 2 because of bidirectionality
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Initialize hidden states (num_layers * 2 for bidirectional)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Concatenate the final forward and backward states
        # out shape: [batch, seq, hidden*2]
        # We take the last time step
        out = self.fc(out[:, -1, :]) 
        return out

# --- 3. TRAINING FUNCTION ---
def train_model():
    print(" Starting Professional Training Pipeline...")
    
    features = []
    labels = []
    
    class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"   Classes Map: {class_map}")

    # --- ADVANCED BALANCING STRATEGY ---
    for class_name in class_names:
        class_dir = os.path.join(DATA_PATH, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        print(f"   Processing '{class_name}': Found {len(files)} files.")
        
        selected_files = []
        if len(files) >= TARGET_SAMPLES:
            selected_files = random.sample(files, TARGET_SAMPLES)
        else:
            # Oversampling (Duplication)
            repeats = TARGET_SAMPLES // len(files)
            remainder = TARGET_SAMPLES % len(files)
            selected_files = files * repeats + files[:remainder]
            print(f"      ↳ Oversampled: Copied data {repeats}x times to match size.")

        # Load Data
        for file in selected_files:
            data = np.load(os.path.join(class_dir, file))
            features.append(data)
            labels.append(class_map[class_name])

    X = np.array(features)
    y = np.array(labels)
    
    print(f"   Final Dataset Size: {len(X)} samples (Perfectly Balanced)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Enable augmentation ONLY for training data
    train_dataset = ActionDataset(X_train, y_train, augment=True)
    test_dataset = ActionDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')

    # Model Init
    input_size = X.shape[2]
    model = BidirectionalLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 
    
    # FIX: Removed 'verbose=True' to fix TypeError
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0

    print(f" Training on {device} with Bidirectional LSTM...")

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

        # Evaluate
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
        avg_loss = running_loss/len(train_loader)
        
        # Step the scheduler
        scheduler.step(acc)
        
        # We manually print the LR here, so we don't need verbose=True in the scheduler
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_map': class_map,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'bidirectional': True 
            }, MODEL_SAVE_PATH)

    print(f"\n Best Accuracy: {best_acc:.2f}%")
    print(f" Saved to {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    train_model()