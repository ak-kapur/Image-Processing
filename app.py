import streamlit as st
import cv2
import numpy as np
from PIL import Image


st.title("🖼️ Image Filter App with Advanced Transformations")
st.write("Upload an image and apply different filters like blur, edge detection, negative, and log transformation!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


filters = [
    "Original",
    "Grayscale",
    "Gaussian Blur",
    "Canny Edge Detection",
    "Median Blur",
    "Sharpen",
    "Negative",
    "Logarithmic Transformation"
]

selected_filter = st.selectbox("Choose a filter to apply:", filters)


def apply_filter(image, filter_name):
    if filter_name == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_name == "Gaussian Blur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_name == "Canny Edge Detection":
        return cv2.Canny(image, 100, 200)
    elif filter_name == "Median Blur":
        return cv2.medianBlur(image, 5)
    elif filter_name == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_name == "Negative":
        return cv2.bitwise_not(image)
    elif filter_name == "Logarithmic Transformation":
        # Convert to float and apply log transform
        image_float = image.astype(np.float32)
        image_log = np.log1p(image_float)  # log(1 + I)
        image_log = cv2.normalize(image_log, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(image_log)
    else:
        return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    processed_image = apply_filter(image_np, selected_filter)

    st.subheader("Filtered Image:")
    st.image(
        processed_image,
        channels="GRAY" if selected_filter in ["Grayscale", "Canny Edge Detection"] else "RGB",
        use_column_width=True
    )
    st.markdown("Developed by ARYAMAN KAPUR 229310218 ")
