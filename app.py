import PySimpleGUI as sg
from ultralytics import YOLO
import cv2
#from postprocessing import *
import h5py
import joblib
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import time
from PIL import Image
import numpy as np
from rembg import remove

# Define class names
class_names = ['Part A', 'Part B', 'Part C', 'Part D']

# Load data from HDF5 file
hdf5_file = 'Total_Image.h5'

with h5py.File(hdf5_file, 'r') as h5f:
    images = np.array(h5f['images'])
    labels = np.array(h5f['labels'])

# Normalize pixel values to be between 0 and 1
images = images.astype('float32') / 255.0

# Flatten the images
num_samples, height, width, channels = images.shape
images_flat = images.reshape(num_samples, height * width * channels)

# Encode labels to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
label_encoder.fit(class_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels_encoded, test_size=0.2, random_state=42)

# Apply PCA for dimensionality reduction (using the same PCA object as in training)
pca = PCA(n_components=0.95)  # keep 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Load the trained SVM model from the HDF5 file
hdf5_file_svm = 'SVM.h5'
with h5py.File(hdf5_file_svm, 'r') as h5f:
    model_byte_stream = io.BytesIO(h5f['svm_model'][()])
best_svm = joblib.load(model_byte_stream)

# Load the CNN model
cnn_model = load_model('cnn_model.h5')

# Function to preprocess and predict image using SVM
def preprocess_and_predict_svm(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((150, 150))  # Resize image
    img = img.convert('RGBA')  # Convert to RGBA for background removal  
    # Remove background
    img_no_bg = remove(img)
    # Convert image to array and normalize
    img_array = np.array(img_no_bg).astype(np.float32) / 255.0
    img_flat = img_array.flatten().reshape(1, -1) 
    # Apply PCA transformation and make prediction
    img_pca = pca.transform(img_flat)
    prediction = best_svm.predict(img_pca)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label, img_array
# Function to predict using CNN
def preprocess_and_predict_cnn(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((150, 150))  # Resize image
    img = img.convert('RGBA')  # Convert to RGBA for background removal  
    # Remove background
    img_no_bg = remove(img)
    # Convert image to array and normalize
    img_array = np.array(img_no_bg).astype(np.float32) / 255.0
    img_flat = img_array.flatten().reshape(1, -1) 
    # Apply PCA transformation and make prediction
    img_pca = pca.transform(img_flat)
    prediction = best_svm.predict(img_pca)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label, img_array

# Function to predict using YOLOv8
def predict_yolov8(image_path):
    
    # Load YOLO model
    model = YOLO("best.pt")
    results = model(image_path)
    part_a_count = part_b_count = part_c_count = part_d_count = 0
    img=None
    if not results:
        pass
    else:
    # Process results
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                class_id = int(box.cls)  # Assuming class ID is stored in 'class' attribute
                prob = box.conf
                # Check if confidence is above a threshold
                if prob > 0.80:
                    if class_id == 0:  # Assuming Part_A has class ID 0 (modify class IDs as needed)
                        part_a_count += 1
                    elif class_id == 1:  # Assuming Part_B has class ID 1
                        part_b_count += 1
                    elif class_id == 2:
                        part_c_count += 1
                    elif class_id == 3:
                        part_d_count += 1
            img=result.plot()  # display to screen

    total_count = part_a_count + part_b_count + part_c_count + part_d_count
    part_counts = {
        "Part A": part_a_count,
        "Part B": part_b_count,
        "Part C": part_c_count,
        "Part D": part_d_count,
        "Total": total_count
    }
    return part_counts, img

# Create Layout of the GUI
sg.theme('LightBlue2')  
# Create Layout of the GUI
layout = [
    [sg.Text('GUI Object Detection Vanguard', font=('Helvetica', 16), justification='center')],
    [sg.Frame(layout=[
        [sg.Text('Choose Model:', size=(15, 1)), sg.Combo(['SVM Model', 'CNN Model', 'YOLOv8 Model'], key='model_name', default_value='SVM Model')],
        [sg.Text('Upload Image:', size=(15, 1)), sg.InputText(key='image_path', enable_events=True), sg.FileBrowse()]
    ], title='Input Section', relief=sg.RELIEF_SUNKEN)],
    [sg.Button('Run', size=(10, 1)), sg.Button('Close', size=(10, 1))],
    [sg.Frame(layout=[
        [sg.Text('Result:', size=(50, 1), font=('Helvetica', 12), justification='center')],
        [sg.Text('', size=(50, 5), key='out', font=('Helvetica', 10))]
    ], title='Output Section', relief=sg.RELIEF_SUNKEN)],
    [sg.Frame(layout=[
        [sg.Image(key='image_display', size=(400, 400))]
    ], title='Image Display', relief=sg.RELIEF_SUNKEN)]
]

# Create the Window
window = sg.Window('VanguardGUI', layout)

# Function to resize and convert numpy image to bytes for PySimpleGUI
def resize_and_convert_to_bytes(img_array, size=(400, 400)):
    img = Image.fromarray((img_array * 255).astype(np.uint8)).resize(size)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()

def resize_image(img, size=(400, 400), convert_to_bytes=False):
    img_resized = cv2.resize(img, size)
    if convert_to_bytes:
        return cv2.imencode('.png', img_resized)[1].tobytes()
    else:
        return img_resized
        
# Event Loop to process "events"
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Close':
        break
    elif event == 'Run':
        model_choice = values['model_name']
        image_path = values['image_path']
        output_file = os.path.join('static', f'output_{time.time()}.png')
   
        if model_choice == 'SVM Model':
            predicted_label, img_array = preprocess_and_predict_svm(image_path)
            result_text = f'Predicted Label: {predicted_label}'
            window['out'].update(result_text)
            window['image_display'].update(data=resize_and_convert_to_bytes(img_array))

        elif model_choice == 'CNN Model':
            predicted_label, img_array = preprocess_and_predict_cnn(image_path)
            result_text = f'Predicted Label: {predicted_label}'
            window['out'].update(result_text)
            window['image_display'].update(data=resize_and_convert_to_bytes(img_array))

        elif model_choice == 'YOLOv8 Model':
            part_counts, img = predict_yolov8(image_path)
            result_text = (
                f"Part A Count: {part_counts['Part A']}\n"
                f"Part B Count: {part_counts['Part B']}\n"
                f"Part C Count: {part_counts['Part C']}\n"
                f"Part D Count: {part_counts['Part D']}\n"
                f"Total part found in this image: {part_counts['Total']}"
            )
            window['out'].update(result_text)
            if img is not None:
                img_resized = resize_image(img, size=(400, 400))
                img_bytes = cv2.imencode('.png', img_resized)[1].tobytes()
                window['image_display'].update(data=img_bytes)
# Close window
window.close()
