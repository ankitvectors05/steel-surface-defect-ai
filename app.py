import streamlit as st
import numpy as np
import cv2
from PIL import Image

from inference import inspect_image


st.set_page_config(
    page_title="Steel Defect Inspection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Steel Surface Defect Inspection")

st.write(
    "YOLO-based segmentation system for detecting and analyzing "
    "surface defects in steel images."
)

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Input Image")

    st.image(
        image,
        width="stretch"
    )

    if st.button("Inspect Steel Surface"):

        with st.spinner("Analyzing steel surface..."):

            result = inspect_image(image_array)

        # -----------------------------
        # Inspection metrics
        # -----------------------------

        st.subheader("Inspection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Defects Detected",
            result["defect_count"]
        )

        col2.metric(
            "Defect Area",
            f"{result['defect_area_percent']:.2f}%"
        )

        col3.metric(
            "Image Size",
            f"{image_array.shape[1]} × {image_array.shape[0]}"
        )

        # -----------------------------
        # Defect information
        # -----------------------------

        st.subheader("Detected Defects")

        if result["predictions"]:

            for defect in result["predictions"]:

                st.write(
                    f"**{defect['class']}**  |  "
                    f"Confidence: "
                    f"{defect['confidence']:.2f}"
                )

        else:

            st.success("No defect detected.")

        # -----------------------------
        # Create overlay
        # -----------------------------

        mask = result["mask"]

        overlay = image_array.copy()

        # Create highlighted defect region
        highlight = np.zeros_like(overlay)
        highlight[:, :, 0] = 255

        overlay[mask] = (
            0.55 * overlay[mask] +
            0.45 * highlight[mask]
        ).astype(np.uint8)

        # -----------------------------
        # Visualization
        # -----------------------------

        st.subheader("Defect Localization")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image_array,
                caption="Original Steel Surface",
                width="stretch"
            )

        with col2:

            st.image(
                overlay,
                caption="Detected Defect Overlay",
                width="stretch"
            )

        # -----------------------------
        # Binary mask
        # -----------------------------

        st.subheader("Segmentation Mask")

        binary_mask = (
            mask.astype(np.uint8) * 255
        )

        st.image(
            binary_mask,
            caption="Predicted Defect Segmentation",
            width="stretch"
        )