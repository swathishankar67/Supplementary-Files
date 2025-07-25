Steps for Code Implementation:

Step 1: Install Dependencies
pip install torch torchvision torchaudio transformers matplotlib numpy scikit-learn

Step 2: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import SwinTransformer
import matplotlib.pyplot as plt
import numpy as np

Step 3: Data Preprocessing
Step 3.1: Load MRI Data
class BrainTumorDataset(Dataset):
    def __init__ (self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])  # load MRI image
        mask = np.load(self.mask_paths[idx])    # load corresponding mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
Step 3.2: Image Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_dataset=BrainTumorDataset(image_paths='path_to_images',      
    mask_paths='path_to_masks', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


Step 4: Define U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Decoder
        self.dec1 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec3 = self.upconv_block(128, 64)
        self.dec4 = self.upconv_block(64, out_channels)
        self.pool = nn.MaxPool2d(2)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        # Decoder
        dec1 = self.dec1(enc4)
        dec2 = self.dec2(dec1 + enc3)
        dec3 = self.dec3(dec2 + enc2)
        dec4 = self.dec4(dec3 + enc1)
        return dec4


Step: 5 Define Class Conditional GAN (cGAN)
Step 5.1: Generator (U-Net)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super (Generator, self).__init__()
        self.unet = UNet(in_channels, out_channels)
    def forward (self, x):
        return self.unet(x)
Step 5.2: Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):  # Input channels: 1 (image) + 1 (condition)
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 16 * 16, 1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        return torch.sigmoid(self.fc(x))


Step: 6 Integrating Swin Transformer
class SwinTransformerExtractor(nn.Module):
    def __init__(self):
        super (SwinTransformerExtractor, self).__init__()
        self.swin_transformer=SwinTransformer.from_pretrained('swin-base-patch4- 
        window7-224')
    def forward (self, x):
        return self.swin_transformer(x)


Step: 7Training the Model
def train(generator, discriminator, dataloader, num_epochs=50):
    criterion_gan = nn.BCELoss()  # Binary Cross Entropy Loss for GAN
    criterion_l1 = nn.L1Loss()  # L1 loss for pixel-wise reconstruction
    
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (images, masks) in enumerate(dataloader):
            # Create labels for real and fake images
            real_labels = torch.ones(images.size(0), 1)
            fake_labels = torch.zeros(images.size(0), 1)

            # Train the discriminator
            optimizer_d.zero_grad()
            real_images = torch.cat([images, masks], dim=1)
            outputs = discriminator(real_images)
            d_loss_real = criterion_gan(outputs, real_labels)

            fake_images = generator(images)
            fake_images_with_condition = torch.cat([images, fake_images], dim=1)
            outputs_fake = discriminator(fake_images_with_condition)
            d_loss_fake = criterion_gan(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images_with_condition)
            g_loss_gan = criterion_gan(outputs, real_labels)

            g_loss_l1 = criterion_l1(fake_images, masks)
            g_loss = g_loss_gan + 100 * g_loss_l1
            g_loss.backward()
            optimizer_g.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss:  
         {g_loss.item()}")


Step: 8 Model Evaluation
def evaluate(generator, dataloader):
    generator.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            output = generator(images)
            # Calculate Dice Similarity Coefficient, Accuracy, etc.
            dice_score = calculate_dice(output, masks)
            print(f"Dice Score: {dice_score}")


Step: 9 Visualization
def visualize_output(generator, dataloader):
    generator.eval()
    images, masks = next(iter(dataloader))
    output = generator(images)
    # Visualize
    plt.subplot(1, 3, 1)
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(masks[0].squeeze(), cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 3)
    plt.imshow(output[0].squeeze(), cmap='gray')
    plt.title('Predicted Output')
    plt.show()

