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
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Normalize images and convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image_tensor, mask_tensor

# Define the Generator with Transposed Convolutions
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.model = nn.DataParallel(nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling instead of ConvTranspose2d
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Regular Conv2D
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),  # Another upsampling
#             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Regular Conv2D
#             nn.Sigmoid()
#         ))
    
#     def forward(self, x):
#         return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Ensure input matches expected channels
            nn.ReLU(inplace=True)
        )

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample 16x16 → 32x32
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample 32x32 → 64x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample 64x64 → 128x128
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample 128x128 → 256x256
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Ensure final mask matches input size
            nn.Sigmoid()
        )


        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Bottleneck with extra layers
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Decoder (Upsampling & Adjusting Channels)
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),  # Reduce from 256 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            self.upsample,  # First Upsample (128 → 64)

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # Ensure 64 → 64 consistency
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            self.final_layer  # Final Sigmoid Activation
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.DataParallel(nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),  # Patch-based output
            nn.Sigmoid()
        ))
    
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


class CombinedLoss(nn.Module):
    def __init__(self, lambda_gan=1, lambda_dice=50, lambda_l1=10):
        """
        Combined Loss for GAN-based Segmentation:
        - lambda_gan: Weight for GAN loss (adversarial loss)
        - lambda_dice: Weight for Dice loss (segmentation accuracy)
        - lambda_l1: Weight for L1 loss (optional for smoothness)
        """
        super(CombinedLoss, self).__init__()
        self.lambda_gan = lambda_gan
        self.lambda_dice = lambda_dice
        self.lambda_l1 = lambda_l1
        
        self.bce = nn.BCEWithLogitsLoss()  # Adversarial loss for D
        self.l1_loss = nn.L1Loss()  # Smoothness loss

    def dice_loss(self, preds, targets, smooth=1):
        """
        Dice Loss for segmentation accuracy.
        preds: Generated mask (from G, should be [0,1])
        targets: Ground truth mask (should be [0,1])
        """
        # preds = torch.sigmoid(preds)  # Ensure outputs are in [0,1]
        intersection = (preds * targets).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()  # Dice loss should be minimized

    def forward(self, fake_outputs, fake_masks=None, real_masks=None, real_or_fake="fake"):
        """
        Computes the combined loss for Generator or Discriminator.
        - fake_outputs: Output from discriminator
        - fake_masks: Output from generator (predicted mask)
        - real_masks: Ground truth mask
        - real_or_fake: "real" for D-loss, "fake" for G-loss
        """
        # Adversarial Loss (GAN Loss)
        if real_or_fake == "real":
            adversarial_loss = self.bce(fake_outputs, torch.ones_like(fake_outputs))
        else:
            adversarial_loss = self.bce(fake_outputs, torch.zeros_like(fake_outputs))

        # Compute Dice Loss ONLY if we are training the generator
        dice_loss = 0
        l1_loss = 0
        if fake_masks is not None and real_masks is not None:
            dice_loss = self.dice_loss(fake_masks, real_masks)
            l1_loss = self.l1_loss(fake_masks, real_masks)

        # Return loss based on mode (D or G)
        if real_or_fake == "fake":
            return self.lambda_gan * adversarial_loss + self.lambda_dice * dice_loss + self.lambda_l1 * l1_loss
        else:
            return adversarial_loss  # Only adversarial loss for Discriminator


# Train the GAN with Patch Discriminator and Dice Loss
def train_gan(generator, discriminator, dataloader, num_epochs, lr, device):
    generator.to(device)
    discriminator.to(device)

    criterion = CombinedLoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=10*lr)
    
    
    criterion = CombinedLoss(lambda_gan=1, lambda_dice=100, lambda_l1=10)  # Adjust weights

    for epoch in range(num_epochs):
        generated_image = []
        target_image = []
        for real_images, real_masks in dataloader:
            real_images, real_masks = real_images.to(device), real_masks.to(device)

            ### Train Discriminator ###
            discriminator.zero_grad()
            real_inputs = torch.cat((real_images, real_masks), dim=1)
            real_outputs = discriminator(real_inputs)
            d_loss_real = criterion(real_outputs, None, None, real_or_fake="real")  # Only GAN loss
            
            fake_masks = generator(real_images)
            fake_inputs = torch.cat((real_images, fake_masks), dim=1)
            fake_outputs = discriminator(fake_inputs.detach())  
            d_loss_fake = criterion(fake_outputs, None, None, real_or_fake="fake")  # Only GAN loss

            generated_image += [wandb.Image(im) for im in fake_masks] # wandb
            target_image += [wandb.Image(im) for im in real_masks] # wandb
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            ### Train Generator ###
            generator.zero_grad()
            fake_outputs = discriminator(fake_inputs)
            g_loss = criterion(fake_outputs, fake_masks, real_masks, real_or_fake="fake")  # Full loss
            
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        wandb.log({"Epoch" : epoch, "D loss" : d_loss, "G loss" : g_loss, "Generated Mask" : generated_image[0:20], "True Mask" : target_image[0:20]})
    print("Training completed.")



if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/merfish/Result/GAN')
    parser.add_argument("--model", type=str, default="BottleneckGAN")
    parser.add_argument("--batch_size", type=int, default="32")
    parser.add_argument("--data", type=str, default="CellBinDB")
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

    dataset = PatchingCellDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()

    train_gan(generator, discriminator, dataloader, num_epochs=num_epochs, lr=learning_rate, device=device)

    torch.save(generator.state_dict(), os.path.join(save2path, "gan_cell_detection.pth"))
    print("Generator model saved.")
    wandb.finish()