import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

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
enable_crop = st.checkbox("Crop Image Before Applying Filter")

# ----------------------------
# Filter function
# ----------------------------
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
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_name == "Negative":
        return cv2.bitwise_not(image)
    elif filter_name == "Logarithmic Transformation":
        image_float = image.astype(np.float32)
        image_log = np.log1p(image_float)
        image_log = cv2.normalize(image_log, None, 0, 255, cv2.NORM_MINMAX)
        return image_log.astype(np.uint8)
    else:
        return image

# ----------------------------
# Convert for Download
# ----------------------------
def convert_to_bytes_for_download(image_array):
    if len(image_array.shape) == 2:
        image_pil = Image.fromarray(image_array.astype(np.uint8), mode="L")
    else:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb.astype(np.uint8))

    buf = BytesIO()
    image_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ----------------------------
# Main logic
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Fix: convert RGB to BGR

    # Remove alpha channel if present
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    h, w = image_np.shape[:2]

    # Optional cropping
    if enable_crop:
        st.subheader("Crop Settings")
        x1 = st.slider("Start X", 0, w - 1, 0)
        y1 = st.slider("Start Y", 0, h - 1, 0)
        x2 = st.slider("End X", x1 + 1, w, w)
        y2 = st.slider("End Y", y1 + 1, h, h)
        image_np = image_np[y1:y2, x1:x2]

    # Apply the selected filter
    processed_image = apply_filter(image_np, selected_filter)

    # Show the image
    st.subheader("Filtered Image:")
    if len(processed_image.shape) == 2:
        st.image(processed_image, channels="GRAY")
    else:
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")

    # Enable download
    image_bytes = convert_to_bytes_for_download(processed_image)
    st.download_button(label="📥 Download Processed Image",
                       data=image_bytes,
                       file_name="processed_image.png",
                       mime="image/png")

    st.markdown("Developed by **ARYAMAN KAPUR (229310218)** 🎓")
