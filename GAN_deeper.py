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

class ResidualBlock(nn.Module):
    """ Residual block to help preserve features and stabilize training """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Adding multiple residual blocks in the bottleneck
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # Output Layer
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)  # Adding deep feature refinement
        x = self.decoder(x)
        return x




# ## Unet generator
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.encoder = nn.DataParallel(nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         ))
#         self.decoder = nn.DataParallel(nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         ))
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# Define the Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.DataParallel(nn.Sequential(
#             nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),  # Patch-based output
#             nn.Sigmoid()
#         ))
    
#     def forward(self, x):
#         return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Spectral Normalization helps stabilize training
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.DataParallel(nn.Sequential(
#             nn.utils.spectral_norm(nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.utils.spectral_norm(nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)),
#             nn.Sigmoid()
#         ))
    
#     def forward(self, x):
#         return self.model(x)

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


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.05):
        super(CombinedLoss, self).__init__()
        # self.bce = nn.BCELoss()
        self.bcew = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7], dtype=torch.float32).to(device))
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        # bce_loss = self.bce(pred, target)
        bcew_loss = self.bcew(pred, target)
        dice_loss_value = dice_loss(pred, target)
        l1_loss_value = self.l1(pred, target)
        return bcew_loss + dice_loss_value + self.l1_weight * l1_loss_value
        # return dice_loss_value + self.l1_weight * l1_loss_value

def wasserstein_loss(real_preds, fake_preds):
    """WGAN loss: real should be high (+1), fake should be low (-1)"""
    return -torch.mean(real_preds) + torch.mean(fake_preds)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--save2', type=str, default='/home/yec23006/projects/research/merfish/Result/GAN')
    parser.add_argument("--model", type=str, default="ResidualGAN")
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

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = CombinedLoss()

    beta1, beta2 = 0.5, 0.9
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    critic_steps = 5
    lambda_gp = 10
    lambda_pixel = 100

    for epoch in range(num_epochs):
        generated_image = []
        real_input_image = []
        target_image = []

        for i, (real_images, target_images) in enumerate(dataloader):  # Assuming `dataloader` is defined
            real_images = real_images.to(device)
            target_images = target_images.to(device)
            batch_size = real_images.size(0)

            ### Train Discriminator ###
            optimizer_D.zero_grad()

            # Generate fake images
            noise = torch.randn(batch_size, 1, real_images.shape[2], real_images.shape[3]).to(device)
            fake_images = generator(noise)

            # Discriminator outputs
            real_preds = discriminator(real_images)
            fake_preds = discriminator(fake_images.detach())  # Detach to prevent generator update

            # Compute gradient penalty
            gp = compute_gradient_penalty(discriminator, real_images, fake_images, device)

            d_loss_real = criterion(real_preds, torch.ones_like(real_preds, device=device))
            d_loss_fake = criterion(fake_preds, torch.ones_like(fake_preds, device=device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()


            if i % critic_steps == 0:
                optimizer_G.zero_grad()

                # Generate fake images again
                fake_images = generator(noise)
                fake_preds = discriminator(fake_images)

                # pixel_loss = torch.nn.functional.l1_loss(fake_images, target_images)

                # g_loss = -torch.mean(fake_preds) + lambda_pixel * pixel_loss
                g_loss = criterion(target_images, fake_images)

                # Backprop and update Generator
                g_loss.backward()
                optimizer_G.step()

                generated_image+=[wandb.Image(im) for im in fake_images]
                real_input_image+=[wandb.Image(im) for im in real_images]
                target_image+=[wandb.Image(im) for im in target_images]

        # Logging
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        wandb.log({"G Loss" : g_loss.item(), "D Loss" : d_loss.item(), "True Mask" : target_image[0:20], "Generated Mask" : generated_image[0:20], "Input Image" : real_input_image})
    

    torch.save(generator.state_dict(), os.path.join(save2path, "r_generator_cell_detection.pth"))
    print("Generator model saved.")
    wandb.finish()
