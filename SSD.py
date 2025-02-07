import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import wandb
import argparse
import datetime
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

def draw_bboxes(image, bboxes, color):
    """
    Draw bounding boxes on an image.
    :param image: numpy array (H, W, 3)
    :param bboxes: list of [x_min, y_min, x_max, y_max]
    :param color: tuple (B, G, R)
    :param label: string
    :return: image with bboxes drawn
    """
    image = image.copy()
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to RGB
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        overlapped_image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return overlapped_image

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length bounding boxes.
    """
    images, instance_masks, bboxes = zip(*batch)  # Unpack batch
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    instance_masks = torch.stack(instance_masks, dim=0)  # Stack instance masks
    return images, instance_masks, list(bboxes)  # Keep bounding boxes as a list

# Dataset class
class PatchingCellDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []

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

        # Normalize images
        image = image.astype(np.float32) / 255.0

        # Convert binary mask to instance segmentation mask
        num_labels, instance_mask = cv2.connectedComponents(mask)

        # Generate bounding boxes for each instance
        bboxes = []
        for label in range(1, num_labels):  # Exclude background (label 0)
            ys, xs = np.where(instance_mask == label)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
            bboxes.append([x_min, y_min, x_max, y_max])

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        instance_mask_tensor = torch.tensor(instance_mask, dtype=torch.int64)  # (H, W)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)  # (N, 4)

        return image_tensor, instance_mask_tensor, bboxes_tensor


# SSD model
class SSD(nn.Module):
    def __init__(self, num_classes=2):  # 1 class + background
        super(SSD, self).__init__()
        
        # Load MobileNetV2 as feature extractor
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = mobilenet.features

        # Define SSD-specific layers (detection heads)
        self.loc_head = nn.Conv2d(1280, 4 * 6, kernel_size=3, padding=1)  # Bounding box predictions
        self.cls_head = nn.Conv2d(1280, num_classes * 6, kernel_size=3, padding=1)  # Class scores
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features
        locs = self.loc_head(x)  # Predict bounding boxes
        classes = self.cls_head(x)  # Predict class probabilities

        # Reshape for SSD format
        batch_size = x.shape[0]
        locs = locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        classes = classes.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # 2 classes (cell, background)

        return locs, classes

def bbox_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between two sets of bounding boxes.

    :param box1: (N, 4) tensor, predicted bounding boxes [x_min, y_min, x_max, y_max]
    :param box2: (N, 4) tensor, ground truth bounding boxes [x_min, y_min, x_max, y_max]
    :return: IoU tensor (N,)
    """
    # Intersection area
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union area
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area + eps  # Avoid division by zero

    return inter_area / union_area  # IoU

def giou_loss(pred_locs, true_locs):
    """
    Compute Generalized IoU (GIoU) Loss.

    :param pred_locs: (N, 4) Predicted bounding boxes
    :param true_locs: (N, 4) Ground truth bounding boxes
    :return: GIoU loss scalar
    """
    iou = bbox_iou(pred_locs, true_locs)

    # Smallest enclosing box
    enc_x1 = torch.min(pred_locs[:, 0], true_locs[:, 0])
    enc_y1 = torch.min(pred_locs[:, 1], true_locs[:, 1])
    enc_x2 = torch.max(pred_locs[:, 2], true_locs[:, 2])
    enc_y2 = torch.max(pred_locs[:, 3], true_locs[:, 3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)  # Area of enclosing box
    giou = iou - (enc_area - (iou * (true_locs[:, 2] - true_locs[:, 0]) * (true_locs[:, 3] - true_locs[:, 1]))) / (enc_area + 1e-7)

    return 1 - giou.mean()  # Minimize GIoU loss



# Loss function
def ssd_loss(pred_locs, pred_classes, true_locs, true_labels):
    """
    SSD Loss function: GIoU + Cross-Entropy Loss

    :param pred_locs: (N, 4) Predicted bounding boxes
    :param pred_classes: (N, num_classes) Predicted class scores
    :param true_locs: (N, 4) Ground truth bounding boxes
    :param true_labels: (N,) Ground truth labels
    :param alpha: weight for GIoU loss
    :return: total loss
    """
    giou = giou_loss(pred_locs, true_locs)
    class_loss = F.cross_entropy(pred_classes.view(-1, 2), true_labels.view(-1), reduction="sum")

    return 0.5 * giou + 0.5 * class_loss 

# Training function
def train_ssd(model, train_loader, num_epochs=200, learning_rate=0.0001, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        target_images, pred_images, input_images = [], [], []

        for images, _, bboxes_list in train_loader:
            images = images.repeat(1, 3, 1, 1).to(device)  # Convert grayscale to 3 channels

            optimizer.zero_grad()
            pred_locs, pred_classes = model(images)

            # Compute loss
            batch_loss = 0
            for i, bboxes in enumerate(bboxes_list):
                bboxes = bboxes.to(device)  # Send each to GPU
                true_labels = torch.ones(bboxes.shape[0], dtype=torch.long).to(device)  # One label per bbox

                # Select the corresponding predictions for this image
                pred_locs_img = pred_locs[i, :bboxes.shape[0], :]  # Matching number of bboxes
                pred_classes_img = pred_classes[i, :bboxes.shape[0], :]

                # Calculate loss per image
                loss = ssd_loss(pred_locs_img, pred_classes_img, bboxes, true_labels)
                batch_loss += loss

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

            # Convert tensor images to numpy
            images_np = images.cpu().numpy().transpose(0, 2, 3, 1) * 255  # Convert to (H, W, 3)
            images_np = images_np.astype(np.uint8)

            # Convert bboxes to numpy for visualization
            bboxes_np_list = [b.cpu().numpy() for b in bboxes_list]

            # Process and store up to 20 target/prediction images
            for i in range(min(len(images_np), 20)):  
                # Get corresponding bboxes (ground truth & predictions)
                gt_bboxes = bboxes_np_list[i]  # Ground truth bounding boxes
                pred_bboxes = pred_locs[i].cpu().detach().numpy() # [: gt_bboxes.shape[0]]  # Match the number of GT boxes

                # Draw bboxes
                target_img = draw_bboxes(images_np[i], gt_bboxes, color=(0, 255, 0))  # Green = Ground Truth
                pred_img = draw_bboxes(images_np[i], pred_bboxes, color=(255, 0, 0))  # Red = Predictions

                # Log images to WandB
                target_images.append(wandb.Image(target_img))
                pred_images.append(wandb.Image(pred_img))
                input_images.append(wandb.Image(images_np[i]))


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": total_loss, "Target" : target_images[0:20], "Prediction" : pred_images[0:20], "Image": input_images[0:20]})


# Main execution
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/merfish/FasterRCNN/Result')
    parser.add_argument("--model", type=str, default="SSD")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data", type=str, default="CellBinDB")
    parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    args = parser.parse_args()
    learning_rate = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save2path = args.save2

    # WandB initialization
    wandb.init(project="Merfish",
               config={"learning_rate": learning_rate,
                       "num_epochs": num_epochs,
                       "batch_size" : batch_size,
                       "architecture": args.model,
                       "data": args.data})
    wandb.run.name = args.run_name

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset & dataloader
    root_dir = "/data/CellBinDB"
    dataset = PatchingCellDataset(root_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, collate_fn=custom_collate_fn)
    # Model initialization with multi-GPU support
    model = SSD(num_classes=2).to(device)
    # print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Multi-GPU parallel training

    # Start training
    train_ssd(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)