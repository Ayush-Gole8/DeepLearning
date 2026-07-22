import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
import kagglehub

# ---------------------------------------------------------
# 1. DIRECTORY SETUP FOR EXPERIMENT TRACKING
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

for d in [OUTPUT_DIR, LOGS_DIR, MODELS_DIR, METRICS_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ---------------------------------------------------------
# 2. DATASET LOADING (FULL DATASET)
# ---------------------------------------------------------
print("Downloading and extracting dataset...")
path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
data_dir = os.path.join(path, "standardized_256")

BATCH_SIZE = 32
IMG_SIZE = (256, 256)

print("Loading 70% Training Data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.30,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading 30% Validation/Test Data...")
val_temp_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.30,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_dataset.class_names

# Split the 30% in half to get 15% Validation and 15% Test
val_batches = tf.data.experimental.cardinality(val_temp_dataset)
test_dataset = val_temp_dataset.take(val_batches // 2)
val_dataset = val_temp_dataset.skip(val_batches // 2)

print("\n--- Full Dataset Loaded ---")
print(f"Training batches: {len(train_dataset)}")
print(f"Validation batches: {len(val_dataset)}")
print(f"Testing batches: {len(test_dataset)}")

# Optimize for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------
# 3. ARCHITECTURE DEFINITION (MobileNetV2)
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = tf.keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc', multi_label=True),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn')
]

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)
model.summary()

# Save model configuration metadata
config_dict = {
    "model": "MobileNetV2",
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "classes": class_names
}
with open(os.path.join(METRICS_DIR, f"model_config_{timestamp}.json"), "w") as f:
    json.dump(config_dict, f, indent=4)

# ---------------------------------------------------------
# 4. CALLBACKS & TRAINING PIPELINE
# ---------------------------------------------------------
# Save the best model automatically
checkpoint_path = os.path.join(MODELS_DIR, "best_model.keras")
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Record loss and metrics to a CSV file
csv_logger = CSVLogger(os.path.join(LOGS_DIR, f"training_log_{timestamp}.csv"))

# TensorBoard logs for visualization
tensorboard_callback = TensorBoard(log_dir=os.path.join(LOGS_DIR, "fit_" + timestamp), histogram_freq=1)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

EPOCHS = 10 

print("\nStarting exhaustive training pipeline...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping, model_checkpoint, csv_logger, tensorboard_callback]
)

# ---------------------------------------------------------
# 5. TESTING & METRIC RECORDING
# ---------------------------------------------------------
print("\n--- Final Model Evaluation on Test Set ---")
test_results = model.evaluate(test_dataset, verbose=1)

eval_metrics_dict = {}
for metric_name, value in zip(model.metrics_names, test_results):
    print(f"{metric_name.upper():<15}: {value:.4f}")
    eval_metrics_dict[metric_name] = float(value)

# Save evaluation results to JSON
with open(os.path.join(METRICS_DIR, f"test_metrics_{timestamp}.json"), "w") as f:
    json.dump(eval_metrics_dict, f, indent=4)

# Save the final model (in case you want the last epoch regardless of checkpoint)
model.save(os.path.join(MODELS_DIR, f"final_model_{timestamp}.keras"))

# ---------------------------------------------------------
# 6. INFERENCE & VISUALIZATION COMPARISONS
# ---------------------------------------------------------
category_mapping = {
    'biological': 'Biodegradable',
    'paper': 'Recyclable',
    'cardboard': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'plastic': 'Recyclable',
    'clothes': 'Recyclable',
    'shoes': 'Recyclable',
    'trash': 'Non-recyclable',
    'battery': 'Non-recyclable (Hazardous)'
}

def test_single_image(img_array):
    img_batch = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_batch, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100
    municipal_category = category_mapping.get(predicted_class_name.lower(), "Unknown")
    return predicted_class_name, municipal_category, confidence

# Grab one batch and test an image to save a visual record
for images, labels in test_dataset.take(1):
    IMAGE_INDEX = 0
    sample_image = images[IMAGE_INDEX]
    true_index = np.argmax(labels[IMAGE_INDEX])
    true_class = class_names[true_index]
    
    predicted, mapped_bin, conf = test_single_image(sample_image)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image.numpy().astype("uint8"))
    plt.axis("off")
    
    color = "green" if true_class == predicted else "red"
    plt.title(f"True Class: {true_class}\n"
              f"Predicted: {predicted} ({conf:.2f}%)\n"
              f"Municipal Bin: {mapped_bin}",
              color=color, fontsize=14, pad=15)
    
    # Save the plot instead of just showing it
    plot_path = os.path.join(IMAGES_DIR, f"inference_sample_{timestamp}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nSaved inference visualization to: {plot_path}")
    break 

print("\nPipeline execution complete. All artifacts saved successfully.")