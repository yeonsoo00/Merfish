import torch
import cv2
import numpy as np
import os
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.anchor_utils import AnchorGenerator

def load_model(model_path):
    """ Load trained SSD model """
    model = ssd300_vgg16(pretrained=False)
    model.num_classes = 2  # Background + Cells
    model.head.classification_head.num_classes = 2  # Adjust classifier

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

class CellDetectionDataset(Dataset):
    def __init__(self, image_path, patch_size=256, overlap=32):
        """ Dataset class to generate patches from a large image """
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Apply stronger noise reduction using Non-Local Means Denoising
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)

        # Normalize image to range [0,1]
        self.image = self.image.astype(np.float32) / 255.0
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches, self.positions = self._extract_patches()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard normalization
        ])

    def _extract_patches(self):
        """ Extract overlapping patches from the large image """
        h, w, _ = self.image.shape
        patches, positions = [], []
        stride = self.patch_size - self.overlap
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch = self.image[y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)
                positions.append((x, y))
        
        return patches, positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        patch = (patch * 255).astype(np.uint8)  # Convert back to uint8
        return self.transform(patch), self.positions[idx]

def infer_patches(model, dataset, device):
    """ Run inference on each patch """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    
    with torch.no_grad():
        for img, (x, y) in loader:
            img = img.to(device)
            output = model(img)[0]
            results.append((output, x.item(), y.item()))
    
    return results

def stitch_detections(results, img_size, patch_size=256, overlap=32, iou_thres=0.2):
    """ Stitch detections back to the original image space with NMS """
    full_detections = []
    pred_boxes, pred_scores = [], []
    
    for output, x_offset, y_offset in results:
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5 : # Filter out low confidence predictions
                x1, y1, x2, y2 = box
                full_detections.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, score, label])
                pred_boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset])
                pred_scores.append(score)
    
    # Apply Non-Maximum Suppression (NMS)
    # Convert to PyTorch tensors with the same dtype and device
    pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32, device=device)
    pred_scores = torch.tensor(pred_scores, dtype=torch.float32, device=device)

    # Apply Non-Maximum Suppression (NMS)
    keep = torch.ops.torchvision.nms(pred_boxes, pred_scores, iou_threshold=iou_thres)

    keep = keep.cpu().numpy().astype(int)  # Ensure integer indices

    # Convert full_detections to a NumPy array before indexing
    full_detections = np.array(full_detections)[keep]

    return full_detections

def save_detections_json(detections, output_path):
    """ Save detections to a JSON file """
    with open(output_path, 'w') as f:
        json.dump(detections.tolist(), f)

def visualize_detections(image_path, detections, output_path):
    """ Visualize detections and save as PNG """
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2, score, label = map(int, det)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)


if __name__=="__main__":

    model_path = '/home/yec23006/projects/research/merfish/FasterRCNN/Result/ssd_vgg_aug.pth'
    path2root = '/data/Maye'
    dir2data = ['09122024_ID23009MEDWB', '09142024_ID23009MEDWB']
    subdir = 'cellpose_tiffs/cellpose_tiffs/'
    filelist = ['cellpose_section1.tif', 'cellpose_section2.tif', 'cellpose_section3.tif']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path).to(device)

    for directory in dir2data:
        os.mkdir(os.path.join('/home/yec23006/projects/research/merfish/FasterRCNN/Result/', directory+'_sddvggaug'))
        path2save = os.path.join('/home/yec23006/projects/research/merfish/FasterRCNN/Result/', directory+'_sddvggaug')

        for file in filelist:
            path2data = os.path.join(path2root, directory, subdir, file)
            dataset = CellDetectionDataset(path2data)
            results = infer_patches(model, dataset, device)
            detections = stitch_detections(results, dataset.image.shape[:2])

            output_json = os.path.join(path2save, file+'_bboxes.json')
            output_png = os.path.join(path2save, file+'_detections.png')
            save_detections_json(detections, output_json)
            visualize_detections(path2data, detections, output_png)