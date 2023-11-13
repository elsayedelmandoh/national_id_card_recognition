import streamlit as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image, gray, thresholded


def visualize_preprocess_image(image, gray, thresholded):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Grayscale Image")
    axes[1].axis('off')

    axes[2].imshow(thresholded, cmap='gray')
    axes[2].set_title("Thresholded Image")
    axes[2].axis('off')
    
    st.pyplot(fig)

def extract_from_whole_image(image):
    image, gray, thresholded = preprocess_image(image)
    text = pytesseract.image_to_string(thresholded)
    return text, image, gray, thresholded

def extract_custom_roi(image, x, y, width, height):
    roi = image[y:y+height, x:x+width]
    image, gray, thresholded = preprocess_image(roi)
    text = pytesseract.image_to_string(thresholded)
    return text, image, gray, thresholded

def draw_roi(image, x, y, width, height):
    roi_image = image.copy()
    cv2.rectangle(roi_image, (x, y), (x+width, y+height), (0, 255, 0), 2)
    return roi_image

def main():
    st.title("Tesseract OCR with Streamlit")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])        
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),  cv2.IMREAD_COLOR)
        st.write("Uploaded image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
        extract_option = st.radio("Choose Extraction Option:", ("Extract All Text", "Extract from Custom Region"))

        # Customize region selection
        if extract_option == "Extract from Custom Region":
            num_regions = st.number_input("Number of Regions to Extract", min_value=1, value=1, step=1)
            extracted_data = {}
            
            for i in range(num_regions):
                # get axis rectangle
                x_roi = st.slider(f"X Coordinate of ROI {i+1}", 0, image.shape[1], 0)
                y_roi = st.slider(f"Y Coordinate of ROI {i+1}", 0, image.shape[0], 0)
                width_roi = st.slider(f"Width of ROI {i+1}", 0, image.shape[1], image.shape[1])
                height_roi = st.slider(f"Height of ROI {i+1}", 0, image.shape[0], image.shape[0])
                roi_name = st.text_input(f"Name of ROI {i+1} Column", f"ROI_{i+1}")
                
                # draw rectangel 
                roi_image = draw_roi(image, x_roi, y_roi, width_roi, height_roi)
                st.image(roi_image, caption=f"Region of Interest {i+1}", use_column_width=True)
                
                if st.button(f"Extract Text from ROI {i+1}"):
                    extracted_text_roi, image, gray, thresholded = extract_custom_roi(image, x_roi, y_roi, width_roi, height_roi)
                    visualize_preprocess_image(image, gray, thresholded)
                    st.subheader(f"Extracted Text from ROI {i+1}:")
                    st.write(extracted_text_roi)
                    extracted_data[roi_name] = [extracted_text_roi]

            data = pd.DataFrame(extracted_data)
            st.dataframe(data)
            st.write(data)

        
        # Whole image
        if extract_option == "Extract All Text":
            if st.button("Extract Text from Whole Image"):
                extracted_text_image, image, gray, thresholded = extract_from_whole_image(image)
                visualize_preprocess_image(image, gray, thresholded)
                st.subheader("Extracted Text from the Whole Image:")
                st.write(extracted_text_image)
                
                data = pd.DataFrame({
                    "Text from Whole Image": [extracted_text_image]
                })
                st.dataframe(data)
            
if __name__ == "__main__":
    main()