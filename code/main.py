import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MLPNet(nn.Module):
    def __init__(self, input_size=32*32*3, num_classes=10):
        super(MLPNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)
        
        self.fc4 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class CNNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10Classifier:
    def __init__(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.mlp_model = None
        self.cnn_model = None
        self.mlp_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.cnn_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def load_and_preprocess_data(self, batch_size=128):
        """Load and preprocess CIFAR-10 data"""
        print("Loading CIFAR-10 dataset...")
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform_test)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        val_dataset.dataset.transform = transform_test
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
        
    def build_models(self):
        print("\nBuilding models...")
        self.mlp_model = MLPNet().to(self.device)
        print(f"MLP Parameters: {sum(p.numel() for p in self.mlp_model.parameters()):,}")
        self.cnn_model = CNNNet().to(self.device)
        print(f"CNN Parameters: {sum(p.numel() for p in self.cnn_model.parameters()):,}")
        
    def train_model(self, model, model_name, epochs=20, lr=0.001):
        """Train a model and return training history"""
        print(f"\nTraining {model_name} model...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in self.val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            train_loss = train_loss / len(self.train_loader)
            val_loss = val_loss / len(self.val_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
                
            scheduler.step()
        
        return history
    
    def train_models(self, epochs=20):
        """Train both models"""
        # Train MLP
        self.mlp_history = self.train_model(self.mlp_model, "MLP", epochs)
        
        # Train CNN
        self.cnn_history = self.train_model(self.cnn_model, "CNN", epochs)
        
    def evaluate_model(self, model, model_name):
        """Evaluate model on test set"""
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        test_loss = test_loss / len(self.test_loader)
        
        print(f"\n{model_name} Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        return test_acc, all_predictions, all_targets
    
    def plot_learning_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MLP Learning Curves
        axes[0, 0].plot(self.mlp_history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.mlp_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('MLP - Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.mlp_history['train_acc'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.mlp_history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('MLP - Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].plot(self.cnn_history['train_loss'], label='Training Loss', color='blue')
        axes[1, 0].plot(self.cnn_history['val_loss'], label='Validation Loss', color='red')
        axes[1, 0].set_title('CNN - Training and Validation Loss')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.cnn_history['train_acc'], label='Training Accuracy', color='blue')
        axes[1, 1].plot(self.cnn_history['val_acc'], label='Validation Accuracy', color='red')
        axes[1, 1].set_title('CNN - Training and Validation Accuracy')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrices(self, mlp_pred, mlp_targets, cnn_pred, cnn_targets):
        """Plot confusion matrices for both models"""
        mlp_cm = confusion_matrix(mlp_targets, mlp_pred)
        cnn_cm = confusion_matrix(cnn_targets, cnn_pred)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title('MLP - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1])
        axes[1].set_title('CNN - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*50)
        
        print("\nMLP Classification Report:")
        print(classification_report(mlp_targets, mlp_pred, target_names=self.class_names))
        
        print("\nCNN Classification Report:")
        print(classification_report(cnn_targets, cnn_pred, target_names=self.class_names))
        
    def display_sample_predictions(self, num_samples=8):
        """Display sample predictions from both models"""
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        indices = torch.randperm(len(images))[:num_samples]
        sample_images = images[indices].to(self.device)
        sample_labels = labels[indices]
        self.mlp_model.eval()
        self.cnn_model.eval()
        
        with torch.no_grad():
            mlp_outputs = self.mlp_model(sample_images)
            cnn_outputs = self.cnn_model(sample_images)
            
            _, mlp_pred = mlp_outputs.max(1)
            _, cnn_pred = cnn_outputs.max(1)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            img = sample_images[i].cpu() * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0)
            
            true_label = sample_labels[i].item()
            mlp_pred_label = mlp_pred[i].item()
            cnn_pred_label = cnn_pred[i].item()
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'True: {self.class_names[true_label]}')
            axes[0, i].axis('off')
            
            # Predictions
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'MLP: {self.class_names[mlp_pred_label]}\nCNN: {self.class_names[cnn_pred_label]}',
                               fontsize=10)
            axes[1, i].axis('off')
        
        plt.suptitle('Sample Predictions Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
def main():
    classifier = CIFAR10Classifier()
    classifier.load_and_preprocess_data(batch_size=128)
    classifier.build_models()
    print("\nStarting training process...")
    classifier.train_models(epochs=15)
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    mlp_acc, mlp_pred, mlp_targets = classifier.evaluate_model(classifier.mlp_model, "MLP")
    cnn_acc, cnn_pred, cnn_targets = classifier.evaluate_model(classifier.cnn_model, "CNN")
    classifier.plot_learning_curves()
    classifier.plot_confusion_matrices(mlp_pred, mlp_targets, cnn_pred, cnn_targets)
    classifier.display_sample_predictions()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()