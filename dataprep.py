import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torch
import json

label_path = '/home/yec23006/projects/research/merfish/Result/pretrained/cellpose_mask.npy'
original_image_path = '/home/yec23006/projects/research/merfish/testimg/seg1_ori.png'

# Load files
label_image = np.load(label_path)
original_image = cv2.imread(original_image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Get unique object IDs (excluding background, typically 0)
unique_ids = np.unique(label_image)
unique_ids = unique_ids[unique_ids != 0]  # Exclude background

# Extract bounding boxes for each unique object
def get_bounding_boxes(label_image, unique_ids):
    bounding_boxes = []
    for obj_id in unique_ids:
        mask = (label_image == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes

bounding_boxes = get_bounding_boxes(label_image, unique_ids)

# Visualize bounding boxes on the original image
def visualize_bounding_boxes(image, boxes):
    vis_image = image.copy()
    for (x_min, y_min, x_max, y_max) in boxes:
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.imwrite('/home/yec23006/projects/research/merfish/FasterRCNN/Result/bbox.png', vis_image)


visualize_bounding_boxes(original_image, bounding_boxes)

# Prepare the data for Faster R-CNN
def prepare_data_for_faster_rcnn(image, bounding_boxes):
    image_tensor = F.to_tensor(image)  # Convert image to tensor
    boxes_tensor = torch.tensor(bounding_boxes, dtype=torch.float32)  # Bounding boxes
    labels_tensor = torch.ones((len(bounding_boxes),), dtype=torch.int64)  # Labels (all ones for a single class)

    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor
    }
    return image_tensor, target

image_tensor, target = prepare_data_for_faster_rcnn(original_image, bounding_boxes)

# Save target data to JSON
def save_target_as_json(target, output_path):
    target_serializable = {
        "boxes": target["boxes"].tolist(),
        "labels": target["labels"].tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(target_serializable, f, indent=4)
    print(f"Target data saved to {output_path}")

output_json_path = '/home/yec23006/projects/research/merfish/FasterRCNN/Result/target_data_frcnn.json'
save_target_as_json(target, output_json_path)
