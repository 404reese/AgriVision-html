import streamlit as st
import numpy as np
import cv2  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Load the model
model = load_model('plant_disease_prediction_model.h5')

    
class_indices = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

treatment_solutions = {
    'Apple___Apple_scab': "Remove affected leaves and apply fungicides.",
    'Apple___Black_rot': "Prune affected areas and apply appropriate fungicides.",
    'Apple___Cedar_apple_rust': "Use resistant varieties and remove cedar trees nearby.",
    'Blueberry___healthy': "No action required; continue regular care.",
    'Cherry_(including_sour)___Powdery_mildew': "Use fungicides and ensure proper air circulation.",
    'Cherry_(including_sour)___healthy': "No action required; continue regular care.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops and use resistant varieties.",
    'Corn_(maize)___Common_rust_': "Apply fungicides and remove crop residues.",
    'Corn_(maize)___Northern_Leaf_Blight': "Practice crop rotation and apply fungicides.",
    'Corn_(maize)___healthy': "No action required; continue regular care.",
    'Grape___Black_rot': "Remove infected leaves and apply sulfur or other fungicides.",
    'Grape___Esca_(Black_Measles)': "Prune affected vines and improve air circulation.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides and remove affected leaves.",
    'Grape___healthy': "No action required; continue regular care.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove affected trees and use resistant varieties.",
    'Peach___Bacterial_spot': "Remove infected parts and apply bactericides.",
    'Peach___healthy': "No action required; continue regular care.",
    'Pepper,_bell___Bacterial_spot': "Remove infected plants and use resistant varieties.",
    'Pepper,_bell___healthy': "No action required; continue regular care.",
    'Potato___Early_blight': "Rotate crops and apply fungicides.",
    'Potato___Late_blight': "Use resistant varieties and apply appropriate fungicides.",
    'Potato___healthy': "No action required; continue regular care.",
    'Raspberry___healthy': "No action required; continue regular care.",
    'Soybean___healthy': "No action required; continue regular care.",
    'Squash___Powdery_mildew': "Use fungicides and improve air circulation.",
    'Strawberry___Leaf_scorch': "Water plants properly and remove affected leaves.",
    'Strawberry___healthy': "No action required; continue regular care.",
    'Tomato___Bacterial_spot': "Remove infected plants and use resistant varieties.",
    'Tomato___Early_blight': "Use crop rotation and apply fungicides.",
    'Tomato___Late_blight': "Use resistant varieties and fungicides.",
    'Tomato___Leaf_Mold': "Ensure proper ventilation and use fungicides.",
    'Tomato___Septoria_leaf_spot': "Apply fungicides and remove affected leaves.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides and promote beneficial insects.",
    'Tomato___Target_Spot': "Use fungicides and remove infected leaves.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants; no cure available.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants; no cure available.",
    'Tomato___healthy': "No action required; continue regular care."
}


def predict_image_class(model, image, class_indices):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    predicted_class = np.argmax(preds, axis=1)[0]
    return class_indices[predicted_class]

# Create the Streamlit app
st.title("Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode the image.")
    else:
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            predicted_class_name = predict_image_class(model, image, class_indices)
            st.success(f"Predicted Class: {predicted_class_name}")

            # Get the probability of the predicted class
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  
            img = img.astype("float32") / 255.0
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img)
            predicted_class = np.argmax(preds, axis=1)[0]
            predicted_class_prob = preds[0][predicted_class]

            # Display the certainty
            st.info(f"Certainty: {predicted_class_prob:.2f}%")

            if predicted_class_name in treatment_solutions:
                solution = treatment_solutions[predicted_class_name]
                st.info(f"Suggested Solution: {solution}")
            else:
                st.warning("No solution available for this disease.")
