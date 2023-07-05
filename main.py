import cv2
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


# Step 1: Initialize model with the best available weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(
    weights=weights,
    score_thresh=0.8,
    detection_per_img=10,
    topk_candidates=3,
)
model = model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Initialize variables.
cap = cv2.VideoCapture(0)  # Capture from camera.
cap.set(3, 244)
cap.set(4, 244)
cap.set(cv2.CAP_PROP_FPS, 15)
winname = "Annotated"  # Window title.

# Exception definition.
BackendError = type("BackendError", (Exception,), {})


labels = []
boxes = []


def IsWindowVisible(winname):
    try:
        ret = cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE)
        if ret == -1:
            raise BackendError(
                "Use Qt as backend to check whether window is visible or not."
            )

        return bool(ret)

    except cv2.error:
        return False


def detect(orig_image):
    global labels
    global boxes
    rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image.transpose((2, 0, 1))
    rgb_image = rgb_image / 255.0
    rgb_image = torch.FloatTensor(rgb_image)

    batch = [preprocess(rgb_image)]

    with torch.no_grad():
        prediction = model(batch)[0]

    score_threshold = 0.5
    labels = [
        weights.meta["categories"][class_index] + f": {score_int:.2f}"
        for class_index, score_int in zip(prediction["labels"], prediction["scores"])
        if score_int > score_threshold
    ]
    boxes = prediction["boxes"][prediction["scores"] > score_threshold]


# i = 0

while True:
    # i += 1
    # Capture image by opencv.
    ret, orig_image = cap.read()
    if orig_image is None:
        continue

    # if i % 5 == 0:
    detect(orig_image)

    # Draw result.
    for box, label in zip(boxes, labels):
        cv2.rectangle(
            orig_image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 255, 0),
            4,
        )

        cv2.putText(
            orig_image,
            label,
            (int(box[0]) + 20, int(box[1]) + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

    # Display video.
    cv2.imshow(winname, orig_image)

    # Press the "q" key to finish.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Exit the program if there is no specified window.
    if not IsWindowVisible(winname):
        break

cap.release()
cv2.destroyAllWindows()
