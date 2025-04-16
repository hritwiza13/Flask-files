#load model
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
model = load_model("color_recommender_cnn.h5")

def predict_color(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Get the dominant color from the model
    prediction = model.predict(img)
    color_index = np.argmax(prediction)
    
    # Map the prediction to color names
    color_names = ["red", "green", "blue", "black", "white", "pink", "grey", "beige", "purple", "brown", "silver"]
    predicted_color = color_names[color_index]
    
    # Get RGB value for the predicted color
    rgb = color_name_to_rgb(predicted_color)
    
    return predicted_color, rgb

# Try to find an image file in the current directory
image_extensions = ['.jpg', '.jpeg', '.png']
image_path = None

for ext in image_extensions:
    for file in os.listdir('.'):
        if file.lower().endswith(ext):
            image_path = file
            break
    if image_path:
        break

if not image_path:
    print("No image file found in the current directory. Please add an image file with extension .jpg, .jpeg, or .png")
    exit(1)

try:
    predicted_color, rgb = predict_color(image_path)
    print(f"Detected Color: {predicted_color} (RGB: {rgb})")

    recommended = recommend_colors_from_cnn(predicted_color)
    print("Recommended Colors (RGB):")
    for col in recommended:
        print(col)

    import matplotlib.pyplot as plt

    def show_colors(base_color, recommended_colors):
        all_colors = [base_color] + recommended_colors
        labels = ["Detected"] + ["Suggested"] * len(recommended_colors)

        fig, ax = plt.subplots(1, len(all_colors), figsize=(10, 2))
        for i, col in enumerate(all_colors):
            ax[i].imshow([[col]])
            ax[i].set_title(labels[i])
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()

    show_colors(rgb, recommended)
except Exception as e:
    print(f"Error: {str(e)}")
    exit(1)

#dataset
outfit_dataset = [
    ["white", "blue", "grey"],
    ["black", "red", "white"],
    ["pink", "white", "beige"],
    ["green", "beige", "white"],
    ["brown", "beige", "black"],
    ["blue", "white", "black"],
    ["grey", "white", "black"],
    ["red", "white", "grey"],
    ["pink", "black", "white"],
    ["blue", "grey", "white"],
    ["purple", "black", "silver"],
    ["beige", "white", "brown"],
    ["red", "beige", "black"],
    ["white", "brown", "pink"],
    ["purple", "white", "grey"],
    ["grey", "black", "red"],
    ["blue", "beige", "white"],
    ["green", "white", "brown"],
    ["black", "grey", "white"],
    ["pink", "purple", "white"],
    ["white", "blue", "black"],
    ["brown", "red", "white"],
    ["grey", "blue", "beige"],
    ["red", "grey", "black"],
    ["blue", "white", "silver"]
]

#recommender
from collections import defaultdict, Counter

def color_name_to_rgb(name):
    mapping = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "pink": (255, 105, 180),
        "grey": (128, 128, 128),
        "beige": (245, 245, 220),
        "purple": (128, 0, 128),
        "brown": (139, 69, 19),
        "silver": (192, 192, 192)
    }
    return mapping.get(name.lower(), (128, 128, 128))

def build_color_relations(dataset):
    relation_map = defaultdict(Counter)
    for outfit in dataset:
        for c1 in outfit:
            for c2 in outfit:
                if c1 != c2:
                    relation_map[c1][c2] += 1
    return relation_map

def get_style2vec_recommendations(predicted_color, relation_map, top_k=3):
    color = predicted_color.lower()
    if color not in relation_map:
        return ["white", "black", "grey"]
    return [c for c, _ in relation_map[color].most_common(top_k)]

def recommend_colors_from_cnn(predicted_color):
    relation_map = build_color_relations(outfit_dataset)
    recommended_names = get_style2vec_recommendations(predicted_color, relation_map)
    return [color_name_to_rgb(c) for c in recommended_names]