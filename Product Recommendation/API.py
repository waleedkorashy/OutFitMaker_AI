import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import tensorflow

app = Flask(__name__)

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load pre-computed feature vectors and filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


@app.route('/recommend', methods=['POST'])
def recommend():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract features from the uploaded image
    query_features = extract_features(file_path, model)

    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([query_features])

    # Prepare recommended image paths
    recommended_images = [filenames[idx] for idx in indices[0]]

    return jsonify({'recommended_images': recommended_images})


# Load the model from the file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    input_data = tuple(data['input_data'])
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    predicted_class = prediction[0]
    classes=['XXS','S','M','L','XL','XXL','XXXL']
    class_name=classes[predicted_class]
    print(predicted_class, class_name)

    # Return the prediction
    return jsonify({'predicted_class': int(predicted_class),
                    "class_name": class_name,
                    })


if __name__ == '__main__':

    app.run(debug=True)
