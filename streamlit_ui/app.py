import streamlit as st
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants

# st.set_page_config(layout="wide")
tf.keras.backend.clear_session()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

MODEL_PATH = './exported_model/saved_model/'
demo_image = './images/apple_48.jpg'
saved_model = tf.saved_model.load(
                str(MODEL_PATH), 
                )
model = saved_model.signatures['serving_default']
CLASSES = [
    "No Object Found",
    "Apple",
    "Banana",
    "Orange",
]

COLORS = np.random.uniform(0, 255, size=(4, 3))
LOADED_IMAGE = None 

def pre_process_image(img):
    print("i'm here")
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

@st.cache(allow_output_mutation=True)
def show_boxes(img, detections, confidence_threshold):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    original_h, original_w, _ = img.shape
    for i in range(detections['detection_boxes'].shape[0]):
        if detections['detection_scores'][i] > confidence_threshold:
            box = detections['detection_boxes'][i] * np.array([original_h,
                                                                original_w,
                                                                original_h,
                                                                original_w])
            (endX, startX, endY, startY) = box.astype("int")
            text_to_show = f"{CLASSES[detections['detection_classes'][i]]}: {int(detections['detection_scores'][i] * 100)}%"
            cv2.rectangle(
                img,
                (startX, endX), 
                (startY, endY), 
                COLORS[detections['detection_classes'][i]],
                2
            )
            
            cv2.putText(
                img, 
                text_to_show, 
                (startX, endX), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                COLORS[2], 
                2
            )
    return img

if __name__ == '__main__':
    st.title("Object detection with MobileNet SSD")

    img_file_buffer = st.file_uploader(
        "Upload an image", 
        type=["png", "jpg", "jpeg"]
        )

    if img_file_buffer is not None:
        original_image = np.array(Image.open(img_file_buffer))
    else:
        original_image = np.array(Image.open(demo_image))
        LOADED_IMAGE = original_image
        detections = pre_process_image(original_image)

    if not (LOADED_IMAGE == original_image).all():
        LOADED_IMAGE = original_image
        detections = pre_process_image(original_image)
    
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, .50, 0.05
    )

    image = show_boxes(original_image.copy(), detections, confidence_threshold)

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(LOADED_IMAGE, caption=f"Original image", use_column_width='auto')
    
    with col2:
        st.image(image, caption=f"Processed image", use_column_width='auto')