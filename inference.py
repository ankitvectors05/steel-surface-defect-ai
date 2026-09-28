from ultralytics import YOLO
import cv2
import numpy as np


MODEL_PATH = "models/best.pt"

CLASS_NAMES = {
    0: "Defect Class 1",
    1: "Defect Class 2",
    2: "Defect Class 3",
    3: "Defect Class 4"
}


model = YOLO(MODEL_PATH)


def inspect_image(image):

    h, w = image.shape[:2]

    results = model.predict(
        source=image,
        conf=0.30,
        imgsz=640,
        retina_masks=True,
        verbose=False
    )

    result = results[0]

    predictions = []

    combined_mask = np.zeros(
        (h, w),
        dtype=bool
    )

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            binary_mask = mask > 0.5

            combined_mask |= binary_mask

            class_id = int(
                result.boxes.cls[i].item()
            )

            confidence = float(
                result.boxes.conf[i].item()
            )

            predictions.append({
                "class": CLASS_NAMES[class_id],
                "confidence": confidence
            })

    defect_pixels = int(combined_mask.sum())
    total_pixels = h * w

    defect_area_percent = (
        defect_pixels / total_pixels
    ) * 100

    return {
        "predictions": predictions,
        "defect_count": len(predictions),
        "defect_area_percent": defect_area_percent,
        "mask": combined_mask
    }