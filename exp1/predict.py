import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# Path to your saved model (auto-saved by your training script)
MODEL_PATH = os.path.join("outputs", "models", "best_model.keras")

# The exact class names from the dataset (alphabetical order)
CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass', 
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Mapping to municipal waste bins
CATEGORY_MAPPING = {
    'biological': 'Biodegradable (Green Bin)',
    'paper': 'Recyclable (Blue Bin)',
    'cardboard': 'Recyclable (Blue Bin)',
    'glass': 'Recyclable (Blue Bin)',
    'metal': 'Recyclable (Blue Bin)',
    'plastic': 'Recyclable (Blue Bin)',
    'clothes': 'Recyclable (Donation/Blue Bin)',
    'shoes': 'Recyclable (Donation/Blue Bin)',
    'trash': 'Non-recyclable (Black Bin)',
    'battery': 'Hazardous Waste (E-Waste Facility)'
}

# ---------------------------------------------------------
# 2. LOAD MODEL
# ---------------------------------------------------------
print("Loading model. This might take a few seconds...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run the training script first so 'best_model.keras' exists.")
    exit()

# ---------------------------------------------------------
# 3. PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_garbage_type(img_path):
    try:
        # Load and resize the image to match the training input (256x256)
        img = image.load_img(img_path, target_size=(256, 256))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Add a batch dimension (models expect a batch, so 1 image becomes shape (1, 256, 256, 3))
        img_batch = tf.expand_dims(img_array, 0)
        
        # Make the prediction
        predictions = model.predict(img_batch, verbose=0)
        
        # Get the index of the highest probability
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions[0]) * 100
        
        # Map to the correct municipal bin
        municipal_category = CATEGORY_MAPPING.get(predicted_class_name, "Unknown")
        
        # Print Results
        print("-" * 40)
        print(f"Prediction: {predicted_class_name.capitalize()}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Action: Put in {municipal_category}")
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"Error: Could not find an image at '{img_path}'. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# ---------------------------------------------------------
# 4. RUN INFERENCE
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        # Prompt the user for an image path
        user_input = input("\nEnter the file path of the image to test (or type 'quit' to exit):\n> ")
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            print("Exiting...")
            break
            
        # Clean up the path (removes accidental quotes if you drag-and-drop the file into the terminal)
        clean_path = user_input.strip('"\'') 
        
        predict_garbage_type(clean_path)