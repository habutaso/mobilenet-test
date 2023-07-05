import cv2
import numpy as np
import time
import torch
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    SSD300_VGG16_Weights,
    ssd300_vgg16,
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

WINDOW_NAME = "Object detection"

# 1モデルの試行回数
TRIALS = 50


def display_video(image):
    cv2.imshow(WINDOW_NAME, image)


def detect(model, orig_image, preprocess):
    rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image.transpose((2, 0, 1))
    rgb_image = rgb_image / 255.0
    rgb_image = torch.FloatTensor(rgb_image)

    batch = [preprocess(rgb_image)]

    with torch.no_grad():
        prediction = model(batch)[0]

    return prediction


def measure_object_detection_time(cap, model, weights):
    model = model_function(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    detection_times = []

    for i in range(TRIALS):
        _, orig_image = cap.read()
        start_time = time.perf_counter()
        detect(model, orig_image, preprocess)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        detection_times.append(elapsed_time)
        print(f"{i}回目: {elapsed_time}")

    return detection_times


def write_csv(model_times):
    label = [" "]
    label += [f"{i + 1}回目" for i in range(TRIALS)]
    label.append("平均")

    model_result_format = [[key, *value, sum(value) / TRIALS] for key, value in model_times.items()]
    model_result_format.insert(0, label)
    print(model_result_format)
    transformed = list(zip(*model_result_format))

    with open("./result.csv", "w") as f:
        print(transformed)
        obj = [",".join(map(str, x)) + "\n" for x in transformed]
        f.writelines(obj)


# item = {"key": [i for i in range(TRIALS)], "key2": [i**2 for i in range(TRIALS)]}

# write_csv(item)


if __name__ == "__main__":
    model_functions = [
        fcos_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn_v2,
        fasterrcnn_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
        retinanet_resnet50_fpn,
        ssd300_vgg16,
        ssdlite320_mobilenet_v3_large,
    ]
    model_weights = [
        FCOS_ResNet50_FPN_Weights.DEFAULT,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        RetinaNet_ResNet50_FPN_Weights.DEFAULT,
        SSD300_VGG16_Weights.DEFAULT,
        SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    ]

    cap = cv2.VideoCapture(0)
    cap.set(3, 244)
    cap.set(4, 244)
    cap.set(cv2.CAP_PROP_FPS, 15)

    model_times = {}

    for model_function, weights in zip(model_functions, model_weights):
        detection_times = measure_object_detection_time(cap, model_function, weights)
        model_times[model_function.__name__] = detection_times

    write_csv(model_times)
