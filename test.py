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
INPUT_SAR_DIR = ""  # <--- SET YOUR INPUT SAR DIRECTORY HERE
OUTPUT_NDWI_DIR = "" # <--- SET YOUR OUTPUT NDWI DIRECTORY HERE

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


#swin-unet for test evaluation

# ==========================================
# INSTALL (run once if needed)
# ==========================================
!pip install timm einops

# ==========================================
# IMPORTS
# ==========================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# SAFE DATASET (INDEX MATCHING)
# ==========================================
class FloodDataset(Dataset):
    def __init__(self, sar_dir, ndwi_dir, gen_ndwi_dir, mask_dir):

        self.sar_dir = sar_dir
        self.ndwi_dir = ndwi_dir
        self.gen_dir = gen_ndwi_dir
        self.mask_dir = mask_dir

        self.sar_files = sorted([f for f in os.listdir(sar_dir) if f.endswith(".npy")])
        self.ndwi_files = sorted([f for f in os.listdir(ndwi_dir) if f.endswith(".npy")])
        self.gen_files = sorted([f for f in os.listdir(gen_ndwi_dir) if f.endswith(".npy")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

        assert len(self.sar_files) == len(self.ndwi_files) == len(self.gen_files) == len(self.mask_files), \
            "Mismatch in number of files!"

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):

        sar = np.load(os.path.join(self.sar_dir, self.sar_files[idx])).astype(np.float32)
        ndwi = np.load(os.path.join(self.ndwi_dir, self.ndwi_files[idx])).astype(np.float32)
        gen_ndwi = np.load(os.path.join(self.gen_dir, self.gen_files[idx])).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx])).astype(np.float32)

        sar = sar.squeeze()
        ndwi = ndwi.squeeze()
        gen_ndwi = gen_ndwi.squeeze()
        mask = mask.squeeze()

        # Normalize SAR
        sar_std = sar.std()
        sar = (sar - sar.mean()) / (sar_std + 1e-6) if sar_std > 1e-6 else np.zeros_like(sar)

        x = np.stack([sar, ndwi, gen_ndwi], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # Resize to 224 (since model trained with 224 input)
        x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)

        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return x, y


# ==========================================
# SWIN-UNET MODEL
# ==========================================
class SwinUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            in_chans=in_channels,
            features_only=True
        )

        enc_channels = self.encoder.feature_info.channels()

        class UpBlock(nn.Module):
            def __init__(self, in_c, skip_c, out_c):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_c, skip_c, 2, stride=2)
                self.conv = nn.Sequential(
                    nn.Conv2d(skip_c + skip_c, out_c, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x, skip):
                x = self.up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                return self.conv(x)

        self.up4 = UpBlock(enc_channels[3], enc_channels[2], enc_channels[2])
        self.up3 = UpBlock(enc_channels[2], enc_channels[1], enc_channels[1])
        self.up2 = UpBlock(enc_channels[1], enc_channels[0], enc_channels[0])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[0], 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        e1, e2, e3, e4 = [f.permute(0,3,1,2) for f in feats]

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)

        out = torch.sigmoid(self.final(d2))
        return F.interpolate(out, size=(50,50), mode="bilinear", align_corners=False)


# ==========================================
# LOAD MODEL
# ==========================================
model = SwinUNet().to(DEVICE)
model.load_state_dict(torch.load("swin_unet_56.pth", map_location=DEVICE))
model.eval()


# ==========================================
# TEST DATA PATHS
# ==========================================
test_dataset = FloodDataset(
    sar_dir="",
    ndwi_dir="",
    gen_ndwi_dir="",
    mask_dir=""
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# ==========================================
# EVALUATION
# ==========================================
TP = FP = TN = FN = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        preds = model(x)
        preds = (preds > 0.5).float()

        TP += ((preds == 1) & (y == 1)).sum().item()
        TN += ((preds == 0) & (y == 0)).sum().item()
        FP += ((preds == 1) & (y == 0)).sum().item()
        FN += ((preds == 0) & (y == 1)).sum().item()

total = TP + TN + FP + FN

accuracy = (TP + TN) / (total + 1e-6)
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)
iou = TP / (TP + FP + FN + 1e-6)
dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

po = (TP + TN) / (total + 1e-6)
pe = (((TP + FP)*(TP + FN)) + ((FN + TN)*(FP + TN))) / ((total**2) + 1e-6)
kappa = (po - pe) / (1 - pe + 1e-6)

print("\n===== TEST RESULTS =====")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"IoU           : {iou:.4f}")
print(f"Dice          : {dice:.4f}")
print(f"Cohen Kappa   : {kappa:.4f}")
