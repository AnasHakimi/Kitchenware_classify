import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from model import KitchenwareCNN

def train_model():
    # Configuration
    data_dir = 'c:/Users/anash/Documents/MATLAB/Dataset'  # Corrected path
    batch_size = 20  # Matches MATLAB randperm(400,20) visualization but standard batch size
    learning_rate = 0.01
    num_epochs = 30
    
    # Data Transforms
    # MATLAB: imageInputLayer([100 100 3])
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    
    # Load Dataset
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_dir}")
        return

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    
    # Split Data
    # MATLAB: splitEachLabel(imds,numTrainFiles,'randomize'); numTrainFiles = 75
    # We need to manually split to ensure 75 per class if we want exact match, 
    # but random_split is easier and standard in PyTorch.
    # Total images approx 400? Let's check length.
    total_len = len(full_dataset)
    train_len = int(0.75 * total_len) # Approx split
    val_len = total_len - train_len
    
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KitchenwareCNN(num_classes=len(class_names)).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
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
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {100 * val_correct / val_total:.2f}%")
        
    # Save Model
    torch.save(model.state_dict(), 'kitchenware_model.pth')
    print("Model saved to kitchenware_model.pth")
    
    # Save class names for inference
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

if __name__ == '__main__':
    train_model()
