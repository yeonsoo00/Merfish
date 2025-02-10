import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models.detection as detection
import torch.optim as optim
from sklearn.model_selection import train_test_split
from itertools import islice
import wandb
from torchvision.ops import box_iou
import argparse
import datetime
"""
TODO 
 - Add measurements : iou,  cell count number..

"""
class PatchingCellDataset(Dataset):
    def __init__(self, root_dir, resize=False, transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []
        self.resize = resize
        self.img_size = 256
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


def predict_and_visualize(model, image, target, iou_thres=0.3):
    target_boxes = target["boxes"]
    
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image)

    # Extract predicted bounding boxes and scores
    pred_boxes = predictions[0]["boxes"].cpu().numpy()
    pred_scores = predictions[0]["scores"].cpu().numpy()

    # Apply Non-Maximum Suppression (NMS)
    keep = torch.ops.torchvision.nms(
        torch.tensor(pred_boxes), torch.tensor(pred_scores), iou_threshold=iou_thres
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

def calculate_iou(gt_boxes, pred_boxes):
    """Computes IoU between ground truth boxes and predicted boxes."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0  # No matching boxes
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    return iou_matrix.diag().mean().item()

if __name__=="__main__" :
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument('--iou_thres', type=float, default=0.3)
    parser.add_argument('--trained', type=bool, default=True)

    args = parser.parse_args()
    learning_rate = args.lr
    num_epochs = args.epochs
    batch_size = args.batch
    iou_thres = args.iou_thres
    trained = args.trained

    # WandB initialization
    wandb.init(project="Merfish",
               config={"learning_rate" : learning_rate,
                       "num_epochs": num_epochs,
                       "iou_thres" : iou_thres,
                       "batch_size" : batch_size,
                       "architecture" : "SSDwVGG",
                       "data" : "CellBinDB"})
    wandb.run.name = args.run_name

    # Load and split data
    dataset = PatchingCellDataset(root_dir="/data/CellBinDB")
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained SSD model
    model = detection.ssd300_vgg16(pretrained=trained)
    model.num_classes = 2  # Background + Cells

    # Modify classification head to match our dataset (1 object class + background)
    # in_features = model.head.classification_head.num_classes
    model.head.classification_head.num_classes = 2  

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    # Training loop
    num_images_to_log = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        best_loss = 400  # Init loss

        # Training Loop
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        

        # Evaluation Loop
        with torch.no_grad():
            target_images, pred_images = [], []
            total_iou = 0
            total_objects = 0
            image_count = 0
            total_batches = 0

            for batch_idx, (images, targets) in enumerate(test_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                model.eval()
                predictions = model(images, targets)
                
                for i in range(len(images)):
                    pred_boxes = predictions[i]["boxes"].cpu()
                    pred_scores = predictions[i]["scores"].cpu()
                    gt_boxes = targets[i]["boxes"].cpu()

                    if len(pred_boxes) > 0:
                        keep = torch.ops.torchvision.nms(pred_boxes, pred_scores, iou_threshold=iou_thres)
                        pred_boxes = pred_boxes[keep]
                        pred_scores = pred_scores[keep]

                    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                        iou = calculate_iou(gt_boxes, pred_boxes)
                        total_iou += iou
                        total_objects += len(gt_boxes)

                    if image_count < num_images_to_log:
                        target_img, pred_img = predict_and_visualize(model, images[i], targets[i], iou_thres)
                        target_images.append(wandb.Image(target_img, caption=f"Ground Truth {image_count}"))
                        pred_images.append(wandb.Image(pred_img, caption=f"Prediction {image_count}"))
                        image_count += 1

                total_batches += 1

                if image_count >= num_images_to_log:
                    break
        avg_iou = total_iou / total_objects if total_objects > 0 else 0

        wandb.log({"Epoch" : epoch+1, "Train Loss" : epoch_loss/len(train_loader), "Avg IoU" : avg_iou, "GT" : target_images, "Prediction" : pred_images})
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {avg_iou:.4f}")


        # Save best model
        # if (test_loss/len(test_loader)) < best_loss:
        #     best_loss = test_loss/len(test_loader)
        #     torch.save(model.state_dict(), "/home/yec23006/projects/research/merfish/FasterRCNN/Result/ssd_vgg.pth")
