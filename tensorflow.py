import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CIFAR10Classifier:
    def __init__(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.mlp_model = None
        self.cnn_model = None
        self.mlp_history = None
        self.cnn_history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess CIFAR-10 data"""
        print("Loading CIFAR-10 dataset...")
        (x_train_full, y_train_full), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        
        # Split training data into train and validation sets
        val_size = 5000
        self.x_train = x_train_full[val_size:]
        self.y_train = y_train_full[val_size:]
        self.x_val = x_train_full[:val_size]
        self.y_val = y_train_full[:val_size]
        
        # Normalize pixel values to [0, 1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_val = self.x_val.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_val = keras.utils.to_categorical(self.y_val, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)
        
        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        print(f"Test set: {self.x_test.shape}")
        
    def build_mlp_model(self):
        """Build Multi-Layer Perceptron model with 3 layers"""
        print("\nBuilding MLP model...")
        self.mlp_model = keras.Sequential([
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(512, activation='relu', name='hidden_layer_1'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', name='hidden_layer_2'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name='hidden_layer_3'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax', name='output_layer')
        ])
        
        self.mlp_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("MLP Model Summary:")
        self.mlp_model.summary()
        
    def build_cnn_model(self):
        """Build Convolutional Neural Network with 3 convolution layers"""
        print("\nBuilding CNN model...")
        self.cnn_model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv_layer_1'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer_2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', name='conv_layer_3'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        self.cnn_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("CNN Model Summary:")
        self.cnn_model.summary()
        
    def train_mlp(self, epochs=20, batch_size=128):
        """Train the MLP model"""
        print("\nTraining MLP model...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        self.mlp_history = self.mlp_model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
    def train_cnn(self, epochs=20, batch_size=128):
        """Train the CNN model"""
        print("\nTraining CNN model...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        self.cnn_history = self.cnn_model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
    def plot_learning_curves(self):
        """Plot learning curves for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MLP Learning Curves
        axes[0, 0].plot(self.mlp_history.history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.mlp_history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('MLP - Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.mlp_history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.mlp_history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('MLP - Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # CNN Learning Curves
        axes[1, 0].plot(self.cnn_history.history['loss'], label='Training Loss', color='blue')
        axes[1, 0].plot(self.cnn_history.history['val_loss'], label='Validation Loss', color='red')
        axes[1, 0].set_title('CNN - Training and Validation Loss')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.cnn_history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[1, 1].plot(self.cnn_history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1, 1].set_title('CNN - Training and Validation Accuracy')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_models(self):
        """Evaluate both models on test data"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # MLP Evaluation
        mlp_test_loss, mlp_test_acc = self.mlp_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nMLP Test Results:")
        print(f"Test Loss: {mlp_test_loss:.4f}")
        print(f"Test Accuracy: {mlp_test_acc:.4f}")
        
        # CNN Evaluation
        cnn_test_loss, cnn_test_acc = self.cnn_model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nCNN Test Results:")
        print(f"Test Loss: {cnn_test_loss:.4f}")
        print(f"Test Accuracy: {cnn_test_acc:.4f}")
        
        return mlp_test_acc, cnn_test_acc
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for both models"""
        # Get predictions
        mlp_pred = self.mlp_model.predict(self.x_test, verbose=0)
        cnn_pred = self.cnn_model.predict(self.x_test, verbose=0)
        
        # Convert to class labels
        y_true = np.argmax(self.y_test, axis=1)
        mlp_pred_classes = np.argmax(mlp_pred, axis=1)
        cnn_pred_classes = np.argmax(cnn_pred, axis=1)
        
        # Create confusion matrices
        mlp_cm = confusion_matrix(y_true, mlp_pred_classes)
        cnn_cm = confusion_matrix(y_true, cnn_pred_classes)
        
        # Plot confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # MLP Confusion Matrix
        sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0])
        axes[0].set_title('MLP - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # CNN Confusion Matrix
        sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1])
        axes[1].set_title('CNN - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification reports
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*50)
        
        print("\nMLP Classification Report:")
        print(classification_report(y_true, mlp_pred_classes, target_names=self.class_names))
        
        print("\nCNN Classification Report:")
        print(classification_report(y_true, cnn_pred_classes, target_names=self.class_names))
        
    def display_sample_predictions(self, num_samples=8):
        """Display sample predictions from both models"""
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        # Get predictions
        mlp_pred = self.mlp_model.predict(self.x_test[indices], verbose=0)
        cnn_pred = self.cnn_model.predict(self.x_test[indices], verbose=0)
        
        # Plot samples
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        for i in range(num_samples):
            idx = indices[i]
            true_label = np.argmax(self.y_test[idx])
            mlp_pred_label = np.argmax(mlp_pred[i])
            cnn_pred_label = np.argmax(cnn_pred[i])
            
            # Original image
            axes[0, i].imshow(self.x_test[idx])
            axes[0, i].set_title(f'True: {self.class_names[true_label]}')
            axes[0, i].axis('off')
            
            # Predictions
            axes[1, i].imshow(self.x_test[idx])
            mlp_color = 'green' if mlp_pred_label == true_label else 'red'
            cnn_color = 'green' if cnn_pred_label == true_label else 'red'
            axes[1, i].set_title(f'MLP: {self.class_names[mlp_pred_label]}\nCNN: {self.class_names[cnn_pred_label]}',
                               color='black', fontsize=10)
            axes[1, i].axis('off')
        
        plt.suptitle('Sample Predictions Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def compare_and_discuss_results(self, mlp_acc, cnn_acc):
        """Compare and discuss the results of both models"""
        print("\n" + "="*60)
        print("COMPARISON AND DISCUSSION OF RESULTS")
        print("="*60)
        
        print(f"\nPerformance Comparison:")
        print(f"MLP Test Accuracy: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")
        print(f"CNN Test Accuracy: {cnn_acc:.4f} ({cnn_acc*100:.2f}%)")
        print(f"Improvement: {((cnn_acc - mlp_acc) * 100):.2f} percentage points")
        
        print(f"\nKey Observations:")
        print(f"1. **Architecture Advantage**: CNN shows {'superior' if cnn_acc > mlp_acc else 'similar'} performance")
        print(f"   - CNNs preserve spatial relationships in images")
        print(f"   - MLPs flatten images, losing spatial structure")
        
        print(f"\n2. **Feature Learning**:")
        print(f"   - CNNs learn hierarchical features (edges → shapes → objects)")
        print(f"   - MLPs learn global patterns but miss local spatial features")
        
        print(f"\n3. **Parameter Efficiency**:")
        mlp_params = self.mlp_model.count_params()
        cnn_params = self.cnn_model.count_params()
        print(f"   - MLP parameters: {mlp_params:,}")
        print(f"   - CNN parameters: {cnn_params:,}")
        print(f"   - CNN is {'more' if cnn_params < mlp_params else 'less'} parameter efficient")
        
        print(f"\n4. **Training Characteristics**:")
        print(f"   - Check learning curves for overfitting patterns")
        print(f"   - CNN typically shows better generalization")
        print(f"   - Batch normalization and dropout help regularization")
        
        print(f"\n5. **Practical Implications**:")
        print(f"   - For image tasks, CNNs are generally preferred")
        print(f"   - MLPs can still be useful for feature-engineered data")
        print(f"   - Trade-off between complexity and performance")

def main():
    """Main execution function"""
    # Initialize classifier
    classifier = CIFAR10Classifier()
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Build models
    classifier.build_mlp_model()
    classifier.build_cnn_model()
    
    # Train models
    print("\nStarting training process...")
    classifier.train_mlp(epochs=15, batch_size=128)
    classifier.train_cnn(epochs=15, batch_size=128)
    
    # Plot learning curves
    classifier.plot_learning_curves()
    
    # Evaluate models
    mlp_acc, cnn_acc = classifier.evaluate_models()
    
    # Plot confusion matrices
    classifier.plot_confusion_matrices()
    
    # Display sample predictions
    classifier.display_sample_predictions()
    
    # Compare and discuss results
    classifier.compare_and_discuss_results(mlp_acc, cnn_acc)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()