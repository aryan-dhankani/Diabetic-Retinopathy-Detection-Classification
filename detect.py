import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# Loading the trained model
model_path = "diabetic_retinopathy_resnet50_model.h5"  
model = load_model(model_path)

def preprocess_image(img_path):
    # Open the image and resize it
    img = Image.open(img_path)
    img = img.resize((32, 32))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Expand dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_diabetic_retinopathy(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Predict using the model
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    
    # Map the predicted class to its label (you can adjust this based on your dataset labels)
    class_labels = ["No_DR", "Moderate", "Mild", "Proliferate_DR", "Severe"]
    predicted_label = class_labels[predicted_class]
    
    return predicted_label