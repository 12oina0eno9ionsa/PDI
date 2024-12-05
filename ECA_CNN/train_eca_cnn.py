# train_eca_cnn.py
import torch
import torch.optim as optim
import torch.nn as nn
from eca_cnn import ECACNN
from utils import load_data
from torchvision import transforms
import numpy as np
import os

def compute_class_weights(data_dir):
    class_counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in ['high', 'low', 'mid']]
    total = sum(class_counts)
    weights = torch.FloatTensor([total/(3*count) for count in class_counts])
    return weights

def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.0005):
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    model = ECACNN().to(device)
    
    class_weights = compute_class_weights(data_dir)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = load_data(data_dir, batch_size, transform=transform)
    
    best_loss = float('inf')
    patience = 5  # Early stopping más estricto
    patience_counter = 0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        print(f"Epoch {epoch+1}")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
            torch.save(best_model, "eca_cnn_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    torch.save(best_model, "eca_cnn_model_v2.pth")
    print("Finished Training")

if __name__ == "__main__":
    data_dir = "annotatted/SAMPLES/dataset_4_12/data"
    train_model(data_dir)