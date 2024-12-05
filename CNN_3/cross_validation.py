# cross_validation.py
from sklearn.model_selection import KFold
import torch
from eca_cnn import ECACNN
from utils import load_data
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def cross_validate(data_dir, n_splits=5, epochs=50, batch_size=32, learning_rate=0.0005):
    kf = KFold(n_splits=n_splits, shuffle=True)
    dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{n_splits}")
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ECACNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        
        # Evaluar en el conjunto de validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    data_dir = "annotatted/SAMPLES/dataset_4_12/data"
    cross_validate(data_dir)