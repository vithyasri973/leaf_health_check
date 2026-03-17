# Model Training Guide

## Complete ML Pipeline Documentation

---

## 1. Dataset Preparation

### 1.1 Download PlantVillage Dataset

**Source:** https://www.kaggle.com/datasets/emmarex/plantvillage-dataset

**Dataset Structure:**
- **Total Images:** 54,306
- **Size:** ~2.6 GB
- **Image Format:** RGB JPG, 256×256 pixels
- **Split:** 80/20 (train/test recommended)

**Disease Distribution:**
```
Tomato Diseases (11 classes):
- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Septoria_leaf_spot
- Tomato___Spider_mites_Two_spotted_spider_mite
- Tomato___Target_Spot
- Tomato___Tomato_Yellow_Leaf_Curl_Virus
- Tomato___Tomato_mosaic_virus
- Tomato___healthy
- Tomato___Powdery_mildew

Similar distributions for: Potato, Apple, Corn, Grape, etc.
```

### 1.2 Organize Dataset

```bash
# Create directory structure
mkdir -p data/train data/val data/test

# Extract downloaded dataset
unzip plantvillage-dataset.zip

# Organize by disease
python scripts/organize_dataset.py
```

**Organization Script:**
```python
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

data_root = Path('plantvillage')
output_root = Path('data')

# Create directories
(output_root / 'train').mkdir(exist_ok=True)
(output_root / 'val').mkdir(exist_ok=True)
(output_root / 'test').mkdir(exist_ok=True)

# Organize images
for disease_dir in data_root.iterdir():
    if disease_dir.is_dir():
        images = list(disease_dir.glob('*.jpg'))
        
        # Split: 60% train, 20% val, 20% test
        train, test = train_test_split(images, test_size=0.2)
        train, val = train_test_split(train, test_size=0.25)
        
        # Copy files
        for img_set, dest in zip([train, val, test], ['train', 'val', 'test']):
            dest_dir = output_root / dest / disease_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in img_set:
                shutil.copy(img, dest_dir / img.name)
```

---

## 2. Data Loading & Preprocessing

### 2.1 Create Data Loaders

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Data directories
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Image specifications
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 8  # disease classes

# Create data generators for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Validation/Test generators (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create flows from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Print class mapping
print("Class Mapping:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"{class_idx}: {class_name}")
```

### 2.2 Data Augmentation Strategy

**Why Augmentation?**
- Increases effective dataset size
- Improves model generalization
- Simulates real-world variations

**Augmentation Techniques Applied:**
```
1. Rotation (±20°)
   - Simulates different leaf angles
   - Common in field captures

2. Width/Height Shift (±20%)
   - Simulates different framing
   - Shifts focus area

3. Shear (0.2)
   - Creates perspective shifts
   - Simulates tilted camera

4. Zoom (±20%)
   - Simulates distance variations
   - Creates crop effects

5. Horizontal Flip
   - Doubles leaf orientation variety
   - Improves symmetry handling

6. Vertical Flip
   - Simulates inverted perspective
   - Realistic in some scenarios

7. Brightness (0.8-1.2)
   - Simulates lighting variations
   - Morning/afternoon light

8. Fill Mode (nearest)
   - Fills empty pixels from neighbors
   - Natural appearance
```

---

## 3. Model Building

### 3.1 Import Required Libraries

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
```

### 3.2 Build Disease Detection Model

```python
def build_disease_model():
    """Build EfficientNetB0-based disease classification model."""
    
    # Load pretrained base model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model weights (transfer learning)
    base_model.trainable = False
    
    # Build full model
    model = models.Sequential([
        # Base model
        base_model,
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers with regularization
        layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Create model
disease_model = build_disease_model()
disease_model.summary()
```

### 3.3 Compile Model

```python
# Compile with optimizer and loss
disease_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC()
    ]
)
```

---

## 4. Training

### 4.1 Setup Callbacks

```python
# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Save best model
checkpoint = ModelCheckpoint(
    'model/leaf_disease_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# TensorBoard logging
tensorboard = TensorBoard(
    log_dir='logs',
    histogram_freq=1
)

callbacks_list = [early_stop, reduce_lr, checkpoint, tensorboard]
```

### 4.2 Train Model

```python
# Train
history = disease_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks_list,
    verbose=1
)

print("Training completed!")
```

### 4.3 Training Configuration

```
Epochs: 50 (will stop early if no improvement)
Batch Size: 32
Learning Rate: 0.001 (Adam optimizer)
Validation Frequency: Every epoch
Early Stopping Patience: 10 epochs
LR Reduction Factor: 0.5
LR Reduction Patience: 5 epochs
Minimum LR: 1e-7
```

---

## 5. Model Evaluation

### 5.1 Evaluate on Test Set

```python
# Evaluate
test_loss, test_acc, test_precision, test_recall = disease_model.evaluate(
    test_generator,
    verbose=1
)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

### 5.2 Detailed Metrics

```python
# Get predictions
y_pred = disease_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### 5.3 Plot Training History

```python
# Plot accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```

---

## 6. Model Fine-tuning (Optional)

### 6.1 Unfreeze Base Model Layers

```python
# Unfreeze last N layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
disease_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train again with limited epochs
history_finetune = disease_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
```

---

## 7. Save & Export Model

### 7.1 Save Trained Model

```python
# Save in HDF5 format (Keras native)
disease_model.save('model/leaf_disease_model.h5')

# Save in SavedModel format (TensorFlow native)
disease_model.save('model/leaf_disease_model_tf')

print("Model saved successfully!")
```

### 7.2 Load Model

```python
# Load from HDF5
loaded_model = keras.models.load_model('model/leaf_disease_model.h5')

# Or load from SavedModel
loaded_model = keras.models.load_model('model/leaf_disease_model_tf')

# Test loading
predictions = loaded_model.predict(test_image)
```

### 7.3 Convert to ONNX (Optional)

```python
import tf2onnx
import onnx

# Convert to ONNX format for broader compatibility
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = 'model/leaf_disease_model.onnx'

model_proto, _ = tf2onnx.convert.from_keras(disease_model, input_signature=spec, output_path=output_path)
```

---

## 8. Expected Performance

### Baseline Results (EfficientNetB0)

```
Disease Detection Model:
├── Training Accuracy: 94-97%
├── Validation Accuracy: 92-95%
├── Test Accuracy: 91-94%
├── Precision (Macro): 90-93%
├── Recall (Macro): 89-92%
└── F1-Score: 90-92%

Plant Species Model:
├── Training Accuracy: 98-99%
├── Validation Accuracy: 97-98%
├── Test Accuracy: 97-98%
├── Precision (Macro): 97-98%
├── Recall (Macro): 97-98%
└── F1-Score: 97-98%

Training Time:
├── Per Epoch: 2-5 minutes (32 batch, GPU)
├── Total (50 epochs): 1.5-4 hours
├── With Fine-tuning: +30-60 minutes

Inference Performance:
├── Single Image: 300-500ms (CPU)
├── Single Image: 100-200ms (GPU)
└── Batch (32): 1-2 seconds (GPU)
```

---

## 9. Advanced Techniques

### 9.1 Class Weighting

```python
# Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weight_dict = {
    i: class_weights[i]
    for i in range(NUM_CLASSES)
}

# Train with class weights
history = disease_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)
```

### 9.2 Knowledge Distillation

```python
# Train larger teacher model
teacher_model = build_large_model()
teacher_model.fit(train_generator, ...)

# Use teacher to train smaller student model
# Student learns from both labels and teacher predictions
```

### 9.3 Ensemble Methods

```python
# Train multiple models with different initializations
ensemble_models = []
for i in range(5):
    model = build_disease_model()
    model.fit(train_generator, ...)
    ensemble_models.append(model)

# Ensemble prediction
def ensemble_predict(image):
    predictions = [model.predict(image) for model in ensemble_models]
    return np.mean(predictions, axis=0)
```

---

## 10. Troubleshooting

### Issue: Overfitting

**Symptoms:**
- Training accuracy: 98%
- Validation accuracy: 70%

**Solutions:**
```python
# 1. Increase dropout
layers.Dropout(0.6)  # Was 0.5

# 2. Increase L2 regularization
kernel_regularizer=keras.regularizers.l2(0.005)

# 3. More data augmentation
rotation_range=30  # Was 20

# 4. Reduce model capacity
Dense(256)  # Was 512
```

### Issue: Underfitting

**Symptoms:**
- Training accuracy: 75%
- Validation accuracy: 73%

**Solutions:**
```python
# 1. Increase model complexity
Dense(1024)  # Was 512

# 2. Reduce dropout/regularization
layers.Dropout(0.3)  # Was 0.5

# 3. Train longer
epochs=100  # Was 50

# 4. Reduce early stopping patience
patience=20  # Was 10
```

### Issue: Out of Memory

**Solutions:**
```python
# 1. Reduce batch size
BATCH_SIZE = 16  # Was 32

# 2. Use smaller model
base_model = EfficientNetB0(...)  # vs ResNet50

# 3. Use mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

---

## 11. Complete Training Script

```python
"""
Complete training pipeline for Leaf Health Check models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 8

# Data directories
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    keras.callbacks.ModelCheckpoint('model/leaf_disease_model_best.h5', monitor='val_accuracy', save_best_only=True)
]

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save
model.save('model/leaf_disease_model.h5')
print("✓ Training complete! Model saved to model/leaf_disease_model.h5")
```

---

**Version:** 1.0  
**Last Updated:** January 2024
