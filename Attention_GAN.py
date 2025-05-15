import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import jaccard_score, f1_score
import argparse
from AttentionUnetModel import *

class PatchingCellDataset(Dataset):
    def __init__(self, root_dir, add_gaussian_noise=False, add_sp_noise=False, gaussian_std=0.05, sp_prob=0.01):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []

        self.add_gaussian_noise = add_gaussian_noise
        self.add_sp_noise = add_sp_noise
        self.gaussian_std = gaussian_std
        self.sp_prob = sp_prob

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
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Apply optional noise
        if self.add_gaussian_noise:
            image = self.apply_gaussian_noise(image, std=self.gaussian_std)
        if self.add_sp_noise:
            image = self.apply_salt_pepper_noise(image, prob=self.sp_prob)

        image_tensor = torch.tensor(image).unsqueeze(0)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return image_tensor, mask_tensor

    def apply_gaussian_noise(self, img, std=0.05):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy_img = np.clip(img + noise, 0.0, 1.0)
        return noisy_img

    def apply_salt_pepper_noise(self, img, prob=0.01):
        noisy = img.copy()
        salt_pepper = np.random.rand(*img.shape)
        noisy[salt_pepper < prob / 2] = 0.0    # pepper
        noisy[salt_pepper > 1 - prob / 2] = 1.0  # salt
        return noisy


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),  # Patch-based output
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def compute_gradient_penalty(D, real_samples, fake_samples, device="cuda"):
    """Calculates the gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Custom Loss Function (BCE + Dice Loss)
def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    l1 = nn.L1Loss()(pred, target)  # L1 Regularization
    return 2*bce + dice + 0.1 * l1

def edge_loss(pred, target):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_target_x = F.conv2d(target, sobel_x, padding=1)
    grad_target_y = F.conv2d(target, sobel_y, padding=1)

    grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
    grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)

    return F.l1_loss(grad_pred, grad_target)


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.05):
        super(CombinedLoss, self).__init__()
        # self.bcew = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7], dtype=torch.float32).to(device))
        self.register_buffer("pos_weight", torch.tensor([0.7], dtype=torch.float32))
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        bcew_loss = self.bcew(pred, target)
        dice_loss_value = dice_loss(pred, target)
        edge_loss_value = edge_loss(pred, target)
        return (2*bcew_loss + dice_loss_value + 0.2 * edge_loss_value) / 3.2
    
    def forward(self, pred, target):
        bcew = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(pred.device))
        bcew_loss = bcew(pred, target)
        dice_loss_value = dice_loss(pred, target)
        edge_loss_value = edge_loss(pred, target)
        return (2 * bcew_loss + dice_loss_value + 0.2 * edge_loss_value) / 3.2
        
def postprocess_mask(mask):

    mask_np = (mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    return torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(mask.device) / 255.0


# Train the GAN with Patch Discriminator and Dice Loss
def train_gan(generator, discriminator, dataloader, num_epochs, lr, device):
    
    generator.to(device)
    discriminator.to(device)

    criterion = CombinedLoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=10*lr)
    
    
    for epoch in range(num_epochs):
        generated_image = []
        target_image = []

        for real_images, real_masks in dataloader:
            real_images, real_masks = real_images.to(device), real_masks.to(device)

            discriminator.zero_grad()
            real_inputs = torch.cat((real_images, real_masks), dim=1) # Input of the Discriminator
            real_outputs = discriminator(real_inputs) # Probability map (Confidence score)
            d_loss_real = criterion(real_outputs, torch.ones_like(real_outputs, device=real_outputs.device))

            fake_masks = generator(real_images)
            fake_masks = torch.cat([postprocess_mask(m.unsqueeze(0)) for m in fake_masks], dim=0)
            fake_inputs = torch.cat((real_images, fake_masks), dim=1)  
            fake_outputs = discriminator(fake_inputs.detach())
            d_loss_fake = criterion(fake_outputs, torch.zeros_like(fake_outputs, device=real_outputs.device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            generated_image += [wandb.Image(im) for im in fake_masks] # wandb
            target_image += [wandb.Image(im) for im in real_masks] # wandb

            generator.zero_grad()
            fake_outputs = discriminator(fake_inputs)
            g_loss_gan = criterion(fake_outputs, torch.ones_like(fake_outputs, device=fake_outputs.device))
            g_loss_dice = dice_loss(fake_masks, real_masks)  # Ensuring segmentation quality
            g_loss = g_loss_gan + 50 * g_loss_dice  # Combined loss
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        wandb.log({"Epoch" : epoch, "D loss" : d_loss, "G loss" : g_loss, "Generated Mask" : generated_image[0:5], "True Mask" : target_image[0:5]})
    print("Training completed.")



if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/merfish/Result/GAN')
    parser.add_argument("--model", type=str, default="AttentionUnet++GAN")
    parser.add_argument("--batch_size", type=int, default="64")
    parser.add_argument("--data", type=str, default="CellBinDBwNoise")
    parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    args = parser.parse_args()
    learning_rate = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save2path = args.save2

    # Wantdb initialization
    wandb.init(project = "Merfish",
            config = {
                "learning_rate" : learning_rate,
                "num_epochs" : num_epochs,
                "architecture" : args.model,
                "data" : args.data,
                "batch_size" : batch_size
            })
    wandb.run.name = args.run_name

    root_dir = "/data/CellBinDB"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPU(s)")

    dataset = PatchingCellDataset(
    root_dir=root_dir,
    add_gaussian_noise=True,
    add_sp_noise=True,
    gaussian_std=0.01,
    sp_prob=0.01
)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # generator = Generator()
    # discriminator = Discriminator()
    generator = nn.DataParallel(Generator()).to(device)
    discriminator = nn.DataParallel(Discriminator()).to(device)


    train_gan(generator, discriminator, dataloader, num_epochs=num_epochs, lr=learning_rate, device=device)

    torch.save(generator.state_dict(), os.path.join(save2path, "attentionGan_cell_detection_w_noise.pth"))
    print("Generator model saved.")
    wandb.finish()