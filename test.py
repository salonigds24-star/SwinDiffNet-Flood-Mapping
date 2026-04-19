#Load cdm model
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("/content/drive/MyDrive/checkpoints/cdm_final.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("CDM model loaded")

import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def mse(pred, gt):
    return F.mse_loss(pred, gt).item()

def psnr(pred, gt):
    m = F.mse_loss(pred, gt).item()
    if m < 1e-10:
        return 100.0
    return 10 * np.log10((2.0 ** 2) / m)

def ssim_score(pred, gt):
    pred = pred[0,0].cpu().numpy()
    gt = gt[0,0].cpu().numpy()
    return ssim(pred, gt, data_range=2.0)

mse_list, psnr_list, ssim_list = [], [], []
count = 0

# The test_loader should already be initialized from cell PMGoMGij8HiL
# No need to re-initialize test_dataset and test_loader here.

with torch.no_grad():
    for sar, ndwi, gt in test_loader:
        count += 1

        sar = sar.to(device)
        ndwi = ndwi.to(device)

        t = torch.tensor([20], device=device)
        noise = torch.randn_like(ndwi)

        x_t = forward_diffusion(ndwi, t, noise)
        gen_ndwi = simple_reverse(model, x_t, sar, t)

        mse_list.append(mse(gen_ndwi, ndwi))
        psnr_list.append(psnr(gen_ndwi, ndwi))
        ssim_list.append(ssim_score(gen_ndwi, ndwi))

print("\n===== TEST SET RESULTS  === DRIVE_MyDrive")
print(f"Samples evaluated : {count}")
print(f"MSE  : {np.mean(mse_list):.6f}")
print(f"PSNR : {np.mean(psnr_list):.2f} dB")
print(f"SSIM : {np.mean(ssim_list):.4f}")


#Ndwi Gen for test set

import os
import numpy as np
import torch
from tqdm import tqdm

# --- User-defined paths ---
INPUT_SAR_DIR = "/content/drive/MyDrive/Assam_TEST_nan-free/vh"  # <--- SET YOUR INPUT SAR DIRECTORY HERE
OUTPUT_NDWI_DIR = "/content/drive/MyDrive/Assam_test_gen_ndwi_npy" # <--- SET YOUR OUTPUT NDWI DIRECTORY HERE

# Ensure output directory exists
os.makedirs(OUTPUT_NDWI_DIR, exist_ok=True)

# Ensure cdm_model and device are loaded from previous steps
# (Assuming `cdm_model` and `device` are defined in the current Colab session)

# Re-initialize and load cdm_model within this cell to prevent NameError
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assuming ConditionalUNet class is defined in an earlier cell and available in scope
# If not, you might need to copy the class definition here or ensure that cell is run.
class ConditionalUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Conv2d(2, 64, 3, padding=1)
        self.enc2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = torch.nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, sar):
        x = torch.cat([x, sar], dim=1)
        x = torch.nn.functional.relu(self.enc1(x))
        x = torch.nn.functional.relu(self.enc2(x))
        x = torch.nn.functional.relu(self.dec1(x))
        return self.dec2(x)

cdm_model = ConditionalUNet().to(device)
checkpoint = torch.load("/content/drive/MyDrive/checkpoints/cdm_final.pth", map_location=device)
cdm_model.load_state_dict(checkpoint["model_state_dict"])
cdm_model.eval() # Set to evaluation mode

# Assuming simple_reverse function is defined in an earlier cell and available in scope
# If not, you might need to copy the function definition here or ensure that cell is run.
# T and alpha_bar should also be defined and available. (From cell gGNOmp7dBcAz)
T = 200
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alpha_bar = torch.cumprod(alphas, dim=0)

@torch.no_grad()
def simple_reverse(model, x_t, sar, t):
    eps = model(x_t, sar)
    x0 = (x_t - torch.sqrt(1 - alpha_bar[t]) * eps) / torch.sqrt(alpha_bar[t])
    return x0

print(f"Generating NDWI for SAR files in: {INPUT_SAR_DIR}")
print(f"Saving generated NDWI to: {OUTPUT_NDWI_DIR}")

sar_filenames = sorted([f for f in os.listdir(INPUT_SAR_DIR) if f.endswith('.npy')])

if not sar_filenames:
    print(f"No .npy files found in {INPUT_SAR_DIR}. Please check the path and file extensions.")
else:
    print(f"Found {len(sar_filenames)} SAR files.")

    # Set model to evaluation mode
    cdm_model.eval()

    with torch.no_grad():
        for fname in tqdm(sar_filenames, desc="Generating NDWI"):
            sar_path = os.path.join(INPUT_SAR_DIR, fname)
            sar = np.load(sar_path)

            # Ensure SAR has correct shape [1, 1, H, W] for model input
            if sar.ndim == 2:
                sar_tensor = torch.tensor(sar, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            elif sar.ndim == 3:
                sar_tensor = torch.tensor(sar, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                print(f"Skipping {fname}: Unsupported SAR dimension {sar.ndim}")
                continue

            # Normalize SAR for CDM input (0-1 then -1 to 1) as done in in3gSLUMBWtg
            # Also handle potential NaNs that might still be present in raw .npy files if not cleaned
            sar_tensor_cleaned = torch.nan_to_num(sar_tensor)
            sar_tensor_norm = (sar_tensor_cleaned - sar_tensor_cleaned.min()) / (sar_tensor_cleaned.max() - sar_tensor_cleaned.min() + 1e-8)
            sar_tensor_norm = sar_tensor_norm * 2 - 1

            # Generate NDWI using the Conditional Diffusion Model
            t_gen = torch.tensor([20], device=device) # Use a fixed time step for generation
            noise = torch.randn_like(sar_tensor_norm) # Noise should be same shape as normalized input
            generated_ndwi = simple_reverse(cdm_model, noise, sar_tensor_norm, t_gen)

            # Convert generated NDWI back to numpy and save
            output_ndwi_np = generated_ndwi[0, 0].cpu().numpy()
            output_path = os.path.join(OUTPUT_NDWI_DIR, fname) # Keep original filename
            np.save(output_path, output_ndwi_np)

    print("\n✅ NDWI generation for all files complete!")

swin-unet 
# ===============================
# INSTALL (run once in Colab)
# ===============================
!pip install timm einops

# ===============================
# IMPORTS
# ===============================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import rasterio # Added import for rasterio
import re # Import regular expression module for robust filename parsing

# Helper function to read .tif files
def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        # Handle potential NaN values, e.g., convert to 0 or a sensible value
        data = np.nan_to_num(data, nan=0.0) # Replace NaN with 0
        return data


# ===============================
# DATASET (50x50 → 64x64)
# ===============================
class FloodDataset(Dataset):
    def __init__(self, sar_dir, ndwi_dir, gen_ndwi_dir, mask_dir, dataset_type='train'):
        self.sar_dir = sar_dir
        self.ndwi_dir = ndwi_dir
        self.gen_ndwi_dir = gen_ndwi_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type

        # Filter for .tif files, as processed data is in .tif format
        self.files = sorted([f for f in os.listdir(sar_dir) if f.endswith(".tif")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sar_fname = self.files[idx] # e.g., sample_X_hflip.tif or S1_VH_patch_10.tif

        # Determine naming conventions based on dataset_type
        if self.dataset_type == 'train':
            ndwi_fname = sar_fname
            gt_fname = sar_fname
            gen_ndwi_fname = sar_fname.replace(".tif", "_gen_ndwi.tif")
        elif self.dataset_type == 'test':
            # Extract patch number from SAR filename (e.g., '10' from 'S1_VH_patch_10.tif')
            match = re.search(r'_patch_(\d+)\.tif$', sar_fname)
            if not match:
                raise ValueError(f"Could not parse patch number from filename: {sar_fname}")
            patch_number = match.group(1)

            ndwi_fname = f"NDWI_patch_{patch_number}.tif"
            gt_fname = f"GT_patch_{patch_number}.tif"
            gen_ndwi_fname = sar_fname.replace(".tif", "_test_gen_ndwi.tif")
        else:
            raise ValueError("Invalid dataset_type. Must be 'train' or 'test'.")

        sar = read_tif(os.path.join(self.sar_dir, sar_fname))
        ndwi = read_tif(os.path.join(self.ndwi_dir, ndwi_fname))
        gen_ndwi = read_tif(os.path.join(self.gen_ndwi_dir, gen_ndwi_fname))
        mask = read_tif(os.path.join(self.mask_dir, gt_fname))

        sar = sar.squeeze()
        ndwi = ndwi.squeeze()
        gen_ndwi = gen_ndwi.squeeze()
        mask = mask.squeeze()

        # Normalize (only for non-mask data, mask should be 0/1)
        sar = (sar - sar.mean()) / (sar.std() + 1e-6)
        # NDWI and Gen_NDWI are already normalized by previous steps to [-1, 1] or [0, 1]
        # If they are [0,1], clip to [-1,1] before stack if model expects this range.
        # Let's assume they are already normalized to [-1,1] or compatible for model.

        x = np.stack([sar, ndwi, gen_ndwi], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 🔥 Resize 50x50 → 224x224 for Swin-UNet encoder input
        x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=(224,224), mode="nearest").squeeze(0) # Mask resize with nearest

        return x, y

# ===============================
# SWIN-UNET MODEL
# ===============================
class SwinUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            in_chans=in_channels,
            features_only=True
        )

        enc_channels = self.encoder.feature_info.channels()  # [96,192,384,768]

        # Custom up_block that performs ConvTranspose2d, concatenates, and then applies Conv2d block
        class CustomUpBlock(nn.Module):
            def __init__(self, in_c, skip_c, out_c):
                super().__init__()
                self.upsample = nn.ConvTranspose2d(in_c, skip_c, kernel_size=2, stride=2)
                self.conv_block = nn.Sequential(
                    nn.Conv2d(skip_c + skip_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x, skip_feature):
                x = self.upsample(x)
                # Ensure size match for concatenation (can happen if encoder output is not perfectly 2x)
                if x.shape[-2:] != skip_feature.shape[-2:]:
                    x = F.interpolate(x, size=skip_feature.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_feature], dim=1)
                return self.conv_block(x)

        # Decoder blocks
        self.up4 = CustomUpBlock(enc_channels[3], enc_channels[2], enc_channels[2]) # From e4 to e3 resolution
        self.up3 = CustomUpBlock(enc_channels[2], enc_channels[1], enc_channels[1]) # From d4 to e2 resolution
        self.up2 = CustomUpBlock(enc_channels[1], enc_channels[0], enc_channels[0]) # From d3 to e1 resolution

        # Final upsampling block to original (224x224) input resolution. No skip from encoder here.
        # This block will just upsample and do convolutions without concatenation.
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[0], 64, kernel_size=2, stride=2), # From 56x56 to 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), # From 112x112 to 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        original_input_size = x.shape[-2:] # Store original 50x50 size

        # Encoder
        # Input x is already 224x224 from dataset
        feats = self.encoder(x)
        # Ensure features are in NCHW format by permuting from NHWC if necessary
        e1, e2, e3, e4 = [f.permute(0, 3, 1, 2) if f.ndim == 4 and f.shape[-1] == c else f
                          for f, c in zip(feats, self.encoder.feature_info.channels())]

        # Decoder path
        d4 = self.up4(e4, e3) # (B, 384, 14, 14)
        d3 = self.up3(d4, e2) # (B, 192, 28, 28)
        d2 = self.up2(d3, e1) # (B, 96, 56, 56)

        # Final upsampling to 224x224
        final_features = self.final_upsample(d2) # (B, 64, 224, 224)

        output = torch.sigmoid(self.out(final_features)) # (B, 1, 224, 224)

        # Resize output back to original 50x50 size
        return F.interpolate(output, size=original_input_size, mode='bilinear', align_corners=False)

# ===============================
# LOSS (BCE + DICE)
# ===============================
class DiceLoss(nn.Module):
    def forward(self, p, t):
        smooth = 1e-6
        p = p.view(-1)
        t = t.view(-1)
        intersection = (p * t).sum()
        return 1 - (2*(intersection)+smooth)/(p.sum()+t.sum()+smooth)

# Define a combined loss function - FIX for TypeError
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

criterion = CombinedLoss()

# ===============================
# TRAINING
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"



train_dataset = FloodDataset(
    sar_dir="/content/drive/MyDrive/Assam_nan-free/vh",
    ndwi_dir="/content/drive/MyDrive/Assam_nan-free/ndwi",
    gen_ndwi_dir="/content/drive/MyDrive/Assam_gen_ndwi_tif",
    mask_dir="/content/drive/MyDrive/Assam_nan-free/gt"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = SwinUNet(in_channels=3).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 40

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "swin_unet_flood_50x50.pth")
print("✅ Swin-UNet training complete & model saved")

#evaluation on train and testing set

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    cohen_kappa_score
)
import os
import rasterio
import re # Import regular expression module
import torch.nn.functional as F # Imported for F.interpolate

# Helper function to read .tif or .npy files
def read_image(path):
    if path.lower().endswith(".tif"):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            return data
    elif path.lower().endswith(".npy"):
        data = np.load(path).astype(np.float32)
        # Ensure it's 2D if loaded as 3D (e.g., [1, H, W])
        if data.ndim == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        data = np.nan_to_num(data, nan=0.0)
        return data
    else:
        raise ValueError(f"Unsupported file format: {path}")

# Local re-definition of FloodDataset for evaluation
class FloodDataset(Dataset):
    def __init__(self, sar_dir, ndwi_dir, gen_ndwi_dir, mask_dir, dataset_type='train'):
        self.sar_dir = sar_dir
        self.ndwi_dir = ndwi_dir
        self.gen_ndwi_dir = gen_ndwi_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type

        # Determine expected file extension based on sar_dir content
        sample_files = [f for f in os.listdir(sar_dir) if f.lower().startswith(('s1_vh_patch_', 'sample_'))]
        if not sample_files:
            raise FileNotFoundError(f"No SAR files found in {sar_dir} starting with 'S1_VH_patch_' or 'sample_'.")

        first_file_ext = os.path.splitext(sample_files[0])[1].lower()
        if first_file_ext == '.tif':
            self.file_ext = ".tif"
            self.files = sorted([f for f in os.listdir(sar_dir) if f.lower().endswith(".tif")])
        elif first_file_ext == '.npy':
            self.file_ext = ".npy"
            self.files = sorted([f for f in os.listdir(sar_dir) if f.lower().endswith(".npy")])
        else:
            raise ValueError(f"Unsupported file extension found in {sar_dir}: {first_file_ext}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sar_fname = self.files[idx]

        if self.dataset_type == 'train':
            ndwi_fname = sar_fname
            gt_fname = sar_fname
            gen_ndwi_fname = sar_fname.replace(".tif", "_gen_ndwi.tif")
        elif self.dataset_type == 'test':
            # Extract patch number from SAR filename (e.g., '10' from 'S1_VH_patch_10.npy')
            match = re.search(r'_patch_(\d+)' + re.escape(self.file_ext) + '$', sar_fname, re.IGNORECASE)
            if not match:
                raise ValueError(f"Could not parse patch number from filename: {sar_fname}")
            patch_number = match.group(1)

            ndwi_fname = f"NDWI_patch_{patch_number}{self.file_ext}"
            gt_fname = f"GT_patch_{patch_number}{self.file_ext}"
            # Generated NDWI files for the test set always end in _gen_ndwi.npy as per cell 4HV9XJ35jgbf
            gen_ndwi_fname = sar_fname.replace(self.file_ext, "_gen_ndwi.npy")
        else:
            raise ValueError("Invalid dataset_type. Must be 'train' or 'test'.")

        sar = read_image(os.path.join(self.sar_dir, sar_fname))
        ndwi = read_image(os.path.join(self.ndwi_dir, ndwi_fname))
        gen_ndwi = read_image(os.path.join(self.gen_ndwi_dir, gen_ndwi_fname))
        mask = read_image(os.path.join(self.mask_dir, gt_fname))

        sar = sar.squeeze()
        ndwi = ndwi.squeeze()
        gen_ndwi = gen_ndwi.squeeze()
        mask = mask.squeeze()

        # Normalize SAR as done in SwinUNet training (wDEOQCr0PKJ7)
        sar = (sar - sar.mean()) / (sar.std() + 1e-6)

        x = np.stack([sar, ndwi, gen_ndwi], axis=0)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Resize 50x50 → 224x224 for Swin-UNet encoder input (as done in wDEOQCr0PKJ7)
        x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=(224,224), mode="nearest").squeeze(0) # Mask resize with nearest

        return x, y


# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(model, dataloader, device, threshold=0.45):
    model.eval()
    criterion = nn.BCELoss()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()

            preds_bin = (preds > threshold).float()

            all_preds.append(preds_bin.cpu().numpy().ravel())
            all_labels.append(y.cpu().numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Balanced Accuracy": balanced_accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1-score": f1_score(all_labels, all_preds, zero_division=0),
        "IoU": jaccard_score(all_labels, all_preds, zero_division=0),
        "Cohen Kappa": cohen_kappa_score(all_labels, all_preds),
        "Loss": total_loss / len(dataloader)
    }

    return metrics

# ===============================
# DEVICE
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD TRAIN + TEST DATA
# ===============================
train_dataset = FloodDataset(
    sar_dir="/content/drive/MyDrive/Assam_nan-free/vh",
    ndwi_dir="/content/drive/MyDrive/Assam_nan-free/ndwi",
    gen_ndwi_dir="/content/drive/MyDrive/Assam_gen_ndwi_tif",
    mask_dir="/content/drive/MyDrive/Assam_nan-free/gt",
    dataset_type='train'
)

test_dataset = FloodDataset(
    sar_dir="/content/drive/MyDrive/Assam_TEST_nan-free/vh",
    ndwi_dir="/content/drive/MyDrive/Assam_TEST_nan-free/ndwi",
    gen_ndwi_dir="/content/drive/MyDrive/Assam_test_gen_ndwi_npy", # Corrected path
    mask_dir="/content/drive/MyDrive/Assam_TEST_nan-free/gt",
    dataset_type='test'
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = SwinUNet(in_channels=3).to(device)
model.load_state_dict(torch.load("swin_unet_flood_50x50.pth", map_location=device))

# ===============================
# EVALUATE TRAIN SET
# ===============================
train_metrics = evaluate_model(model, train_loader, device)

print("\n📊 TRAIN SET METRICS")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

# ===============================
# EVALUATE TEST SET
# ===============================
test_metrics = evaluate_model(model, test_loader, device)

print("\n📊 TEST SET METRICS")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
