import sys
import numpy as np
import cv2
from keras.models import load_model

def predict_disease(image_path, model_path="models/crop_disease_resnet.keras"):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained and saved the model first.")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Ensure these match the sorted alphabetical order of the training data directories
    class_names = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
        'Tomato__Tomato_YellowLeaf__Curl_Virus'
    ] 
    
    predicted_disease = class_names[class_idx]
    
    print("\n" + "="*40)
    print("🌾 CROP DISEASE PREDICTION RESULT")
    print("="*40)
    print(f"Image Analysed: {image_path}")
    print(f"Predicted Disease: {predicted_disease}")
    print(f"Confidence Score: {confidence:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
    else:
        predict_disease(sys.argv[1])
