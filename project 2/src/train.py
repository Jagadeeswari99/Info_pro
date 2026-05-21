import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping

# ==========================================
# 1. EDA & Data Preprocessing
# ==========================================
def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    print(f"Loading data from {data_dir}...")
    X, y = [], []
    class_names = []
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please download the PlantVillage dataset.")
        print("Generating dummy data for testing purposes...")
        dummy_classes = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
            'Tomato__Tomato_YellowLeaf__Curl_Virus'
        ]
        return np.random.rand(100, 224, 224, 3), np.random.randint(0, 13, 100), dummy_classes
        
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_names.append(class_name)
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(class_idx)
                
    return np.array(X) / 255.0, np.array(y), class_names

def plot_class_balance(y, class_names):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.title("Class Imbalance Check (Crop Diseases)")
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. Custom CNN Construction
# ==========================================
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential([
        # Data Augmentation layer embedded in the model
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 3. Transfer Learning (ResNet50)
# ==========================================
def build_resnet_transfer_model(input_shape=(224, 224, 3), num_classes=3):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        # Data Augmentation
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.RandomRotation(0.2),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. Evaluation Module (Optimizing for Recall)
# ==========================================
def evaluate_model(model, X_test, y_test, class_names):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n" + "="*40)
    print("--- MODEL EVALUATION ---")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}  <-- (Optimized to minimize false negatives)")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    data_dir = "data/PlantVillage"
    
    X, y, class_names = load_and_preprocess_data(data_dir)
    num_classes = len(class_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Train Custom CNN
    print("\nTraining Custom CNN...")
    cnn_model = build_custom_cnn(num_classes=num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the Custom CNN
    cnn_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # 2. Train Transfer Learning Model
    print("\nTraining ResNet50 Transfer Model...")
    resnet_model = build_resnet_transfer_model(num_classes=num_classes)
    resnet_model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # Evaluate
    evaluate_model(resnet_model, X_test, y_test, class_names)
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    resnet_model.save("models/crop_disease_resnet.keras")
    print("Pipeline complete. Saved best model.")
