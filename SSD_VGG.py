import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.models.detection as detection
import torch.optim as optim
from itertools import islice
import wandb
import argparse
import datetime

class PatchingCellDataset(Dataset):
    def __init__(self, root_dir, resize=False, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []
        self.resize = resize
        self.img_size = 300
        self.transform = transform

        # Traverse dataset directories
        for subdir in ["10xGenomics_DAPI", "DAPI", "ssDNA"]:
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
            for folder in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, folder, f"{folder}-img.tif")
                mask_path = os.path.join(subdir_path, folder, f"{folder}-mask.tif")
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load grayscale image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=-1)  # Convert to RGB (H, W, 3)

        # Convert binary mask to instance segmentation mask
        num_labels, instance_mask = cv2.connectedComponents(mask)

        # Generate bounding boxes for each instance
        bboxes = []
        for label in range(1, num_labels):  # Exclude background (label 0)
            ys, xs = np.where(instance_mask == label)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

            # Filter out invalid boxes (zero width or height)
            if x_max - x_min > 0 and y_max - y_min > 0:
                bboxes.append([x_min, y_min, x_max, y_max])

        bboxes = np.array(bboxes, dtype=np.float32)

        if self.resize : 
            # Resize image and adjust bounding boxes
            orig_h, orig_w = image.shape[:2]
            image = cv2.resize(image, (self.img_size, self.img_size))

            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] / orig_w) * self.img_size
                bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] / orig_h) * self.img_size

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)  # (N, 4)
        labels_tensor = torch.ones((bboxes_tensor.shape[0],), dtype=torch.int64)  # Class 1 for cells

        # Ensure there is at least one valid box, otherwise return a dummy box
        if len(bboxes_tensor) == 0:
            bboxes_tensor = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels_tensor = torch.tensor([0], dtype=torch.int64)  # Assign background class

        target = {
            "boxes": bboxes_tensor,
            "labels": labels_tensor
        }

        return image_tensor, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    
    # Filter out empty targets
    filtered_images, filtered_targets = [], []
    for img, tgt in zip(images, targets):
        if tgt["boxes"].numel() > 0:
            filtered_images.append(img)
            filtered_targets.append(tgt)
    
    if len(filtered_images) == 0:
        return None  # Handle this in training loop

    return filtered_images, filtered_targets

def predict_and_visualize_old(model, dataset, idx=0):
    model.eval()
    image = dataset[0]
    target = dataset[1]['boxes']

    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image)

    # Extract predicted bounding boxes and scores
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # Apply Non-Maximum Suppression (NMS)
    keep = torch.ops.torchvision.nms(torch.tensor(pred_boxes), torch.tensor(pred_scores), iou_threshold=0.3)
    pred_boxes = pred_boxes[keep.numpy()]

    # Target visualization
    image_t = image.cpu().numpy().copy()
    target_boxes = target.cpu().numpy()
    for bbox_t in target_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox_t)
        cv2.rectangle(image_t, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Prediction visualization
    image_p = image.cpu().numpy().copy()
    for bbox_p in pred_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox_p)
        cv2.rectangle(image_p, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return image_t, image_p

def predict_and_visualize(model, dataset, idx=0):
    model.eval()
    
    image, target = dataset[idx]  # ✅ Correctly extract image and target
    target_boxes = target["boxes"]
    
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image)

    # Extract predicted bounding boxes and scores
    pred_boxes = predictions[0]["boxes"].cpu().numpy()
    pred_scores = predictions[0]["scores"].cpu().numpy()

    # Apply Non-Maximum Suppression (NMS)
    keep = torch.ops.torchvision.nms(
        torch.tensor(pred_boxes), torch.tensor(pred_scores), iou_threshold=0.3
    )
    pred_boxes = pred_boxes[keep.numpy()]

    # Convert image to NumPy
    image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert (3, H, W) -> (H, W, 3)
    image_np = (image_np * 255).astype(np.uint8)  # Normalize if needed

    # Visualization for target (Ground Truth)
    target_img = image_np.copy()
    for bbox_t in target_boxes.cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, bbox_t)
        cv2.rectangle(target_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for GT

    # Visualization for prediction
    pred_img = image_np.copy()
    for bbox_p in pred_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox_p)
        cv2.rectangle(pred_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for Pred

    return target_img, pred_img


if __name__=="__main__" :
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    args = parser.parse_args()
    learning_rate = args.lr
    num_epochs = args.epochs
    batch_size = args.batch

    # WandB initialization
    wandb.init(project="Merfish",
               config={"learning_rate" : learning_rate,
                       "num_epochs": learning_rate,
                       "batch_size" : batch_size,
                       "architecture" : "SSDwVGG",
                       "data" : "CellBinDB"})
    wandb.run.name = args.run_name

    # Load train data
    dataset = PatchingCellDataset(root_dir="/data/CellBinDB")
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained SSD model
    model = detection.ssd300_vgg16(pretrained=True)
    model.num_classes = 2  # Background + Cells

    # Modify classification head to match our dataset (1 object class + background)
    # in_features = model.head.classification_head.num_classes
    model.head.classification_head.num_classes = 2  # 1 for cells, 1 for background

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    # Training loop

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        best_loss = 400 # init loss
        target_images, pred_images, input_images = [], [], []
        
        for images, targets in train_loader:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # visualization
        # for i in range(10):
        #     target_img, pred_img = predict_and_visualize(model, dataset[i])
        #     target_img_reshape = target_img.squeeze(0).transpose(1, 2, 0)
        #     pred_img_reshape = pred_img.squeeze(0).transpose(1, 2, 0)
        #     target_images.append(wandb.Image(target_img_reshape))
        #     pred_images.append(wandb.Image(pred_img_reshape))
        for i in range(10):
            target_img, pred_img = predict_and_visualize(model, dataset, i)

            target_images.append(wandb.Image(target_img, caption=f"Target {i}"))
            pred_images.append(wandb.Image(pred_img, caption=f"Prediction {i}"))

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch" : epoch + 1, "loss" : epoch_loss, "Target" : target_images, "Prediction" : pred_images})

        if epoch_loss < best_loss :
            best_loss = epoch_loss
            torch.save(model.state_dict(), "/home/yec23006/projects/research/merfish/FasterRCNN/Result/ssd_vgg.pth")
