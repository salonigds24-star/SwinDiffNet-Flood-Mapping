import os
import cv2
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch


# Main folder path
MAIN_DIR = "/content/drive/MyDrive/Punjab_Patches_2025"   # change this

# All subfolders
SUBFOLDERS = ["ndwi", "vh", "gt"]

for folder in SUBFOLDERS:
    folder_path = os.path.join(MAIN_DIR, folder)
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tif")])

    print("====================================")
    print(f"Folder: {folder}")
    print(f"Total patches: {len(files)}")
    print("====================================")

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)

        with rasterio.open(file_path) as src:
            data = src.read()

            print(f"Patch {i+1}: {file}")
            print(f"  ➤ Bands      : {data.shape[0]}")
            print(f"  ➤ Size       : {data.shape[1]} × {data.shape[2]}")
            print(f"  ➤ Data type  : {data.dtype}")
            print(f"  ➤ Min value  : {np.nanmin(data)}")
            print(f"  ➤ Max value  : {np.nanmax(data)}")
            print(f"  ➤ NaN count  : {np.isnan(data).sum()}")
            print(f"  ➤ NoData     : {src.nodata}")
            print("-" * 40)

    print("\n")

#data Processing

import os
import cv2
import rasterio
import numpy as np
from glob import glob

# ======================================================
# 🔒 SAFETY FLAG → prevents duplicates
# ======================================================
SAFE_FLAG = "/content/drive/MyDrive/sen1flood11_PREPROCESSING_DONE72.flag"

if os.path.exists(SAFE_FLAG):
    print("✔ Preprocessing already done before — skipping.")
    SKIP_PREPROCESS = True
else:
    print("⚠ Running preprocessing for the first time...")
    SKIP_PREPROCESS = False


# ======================================================
# PATHS
# ======================================================
RAW_DIR = ""
OUT_DIR = ""

folders = ["vh", "ndwi", "gt"]

# ======================================================
# Add a check for Google Drive mount
# ======================================================
if not os.path.exists("/content/drive/MyDrive"):
    print("🚨 Error: Google Drive is not mounted or accessible at /content/drive/MyDrive.")
    print("Please mount Google Drive first (e.g., using 'from google.colab import drive; drive.mount(\"/content/drive\")').")
    raise SystemExit("Google Drive not mounted.")

# Create output folders (safe) - These are created initially, but we'll add a more explicit one in the loop
for f in folders:
    folder_path = os.path.join(OUT_DIR, f)
    os.makedirs(folder_path, exist_ok=True)
    # Optional: Verify creation, though os.makedirs usually raises error if it truly fails (e.g., permissions)
    if not os.path.exists(folder_path):
        # This case is less likely if /content/drive/MyDrive exists, but good for debugging deeper issues.
        raise IOError(f"Failed to verify directory creation: {folder_path}. Check permissions or disk space.")
    else:
        print(f"Ensured directory exists: {folder_path}")


# ======================================================
# FUNCTION TO READ TIF
# ======================================================
def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


# ======================================================
# NORMALIZATION FUNCTIONS
# ======================================================
def normalize_0_1(img):
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn == 0:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def normalize_gt(img):
    if img.max() > 1:
        img = img / img.max()
    return (img > 0.5).astype(np.uint8)


# ======================================================
# PROCESSING LOOP (Runs ONLY once)
# ======================================================
if not SKIP_PREPROCESS:

    for folder in folders:
        print(f"\n🔄 Processing folder: {folder}")

        raw_files = sorted(glob(f"{RAW_DIR}/{folder}/*.tif"))

        for file in raw_files:

            img = read_tif(file)

            # Resize to 50-50
            img_resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_NEAREST)

            # Choose normalization
            if folder in ["vh", "ndwi"]:
                img_norm = normalize_0_1(img_resized)
            else:
                img_norm = normalize_gt(img_resized)

            # Output path
            out_path = os.path.join(OUT_DIR, folder, os.path.basename(file))

            # Ensure the directory exists explicitly right before writing to avoid FUSE issues
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save as float32 tif
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=img_norm.shape[0],
                width=img_norm.shape[1],
                count=1,
                dtype="float32",
            ) as dst:
                dst.write(img_norm.astype(np.float32), 1)

        print(f"✅ Done: {folder}")

    # Save the flag so it never runs again
    with open(SAFE_FLAG, "w") as f:
        f.write("done")

    print("\n🎉 Preprocessing completed ONCE and saved safely!")

else:
    print("⏩ Skipping preprocessing — using already processed data.")

# Comparision Raw vs Processed

# Paths
RAW_DIR = ""
PROC_DIR = ""

folders = ["vh", "ndwi", "gt"]

def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def visualize_comparison(folder, num_samples=2):

    raw_files = sorted(glob(f"{RAW_DIR}/{folder}/*.tif"))
    proc_files = sorted(glob(f"{PROC_DIR}/{folder}/*.tif"))

    # Select first 2 samples (or modify)
    sample_indices = [18, 65]

    print(f"\n📌 Folder: {folder.upper()}")

    for idx in sample_indices:

        raw = read_tif(raw_files[idx])
        proc = read_tif(proc_files[idx])

        plt.figure(figsize=(10,4))

        # Original
        plt.subplot(1,2,1)
        plt.imshow(raw, cmap="gray")
        plt.title(f"{folder.upper()} Original — {os.path.basename(raw_files[idx])}")
        plt.axis("off")

        # Preprocessed
        plt.subplot(1,2,2)
        plt.imshow(proc, cmap="gray", vmin=0, vmax=1)
        plt.title(f"{folder.upper()} Preprocessed — {os.path.basename(proc_files[idx])}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# ===============================
# RUN FOR EACH FOLDER
# ===============================
for folder in folders:
    visualize_comparison(folder, num_samples=2)



#Data Split-train, test, validation

SOURCE_FOR_SPLIT_DIR = "" # <<< FIXED THIS

OUT_DIR = "" # Output of the split operation

folders = ["vh", "ndwi", "gt"]

# ==========================================================
#  SAFETY FLAG → prevents duplication
# ==========================================================
SPLIT_FLAG = "/content/drive/MyDrive/SPLIT_PROCESSED_50_DONE72.flag" # Updated flag to reflect processed data split

if os.path.exists(SPLIT_FLAG):
    print("✔ Split already exists — skipping copying.")
    SKIP_SPLIT = True
else:
    print("⚠ Running split for the first time...")
    SKIP_SPLIT = False

# ==========================================================
# CREATE SPLIT FOLDERS SAFELY
# ==========================================================
for split in ["train", "val", "test"]:
    for f in folders:
        # Also clean existing split folders to ensure fresh copy of processed data
        path_to_clean = f"{OUT_DIR}/{split}/{f}"
        if os.path.exists(path_to_clean):
            shutil.rmtree(path_to_clean)
        os.makedirs(path_to_clean, exist_ok=True)

# ==========================================================
# IF FIRST TIME → PERFORM SPLIT
# ==========================================================
if not SKIP_SPLIT:

    # sorted lists (paired by index) from the *processed* data
    vh_files = sorted(glob(f"{SOURCE_FOR_SPLIT_DIR}/vh/*.tif"))
    ndwi_files = sorted(glob(f"{SOURCE_FOR_SPLIT_DIR}/ndwi/*.tif"))
    gt_files = sorted(glob(f"{SOURCE_FOR_SPLIT_DIR}/gt/*.tif"))

    # size check
    assert len(vh_files) == len(ndwi_files) == len(gt_files), "Folder sizes mismatch!"

    n = len(vh_files)

    # random shuffle
    indices = np.arange(n)
    np.random.shuffle(indices)

    # ratios
    train_ratio = 0.70
    val_ratio = 0.15

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # ==========================================================
    # SAFE COPY — COPIES ALL FILES, CLEANING DONE UPSTREAM
    # ==========================================================
    def safe_copy(src, dst_folder):
        dst_path = os.path.join(dst_folder, os.path.basename(src))
        shutil.copy(src, dst_folder)

    def copy_by_index(indices, split):
        for idx in indices:
            # We are copying processed files with their original names into the split folders
            safe_copy(vh_files[idx],   f"{OUT_DIR}/{split}/vh/")
            safe_copy(ndwi_files[idx], f"{OUT_DIR}/{split}/ndwi/")
            safe_copy(gt_files[idx],   f"{OUT_DIR}/{split}/gt/")

    # perform copy
    copy_by_index(train_idx, "train")
    copy_by_index(val_idx, "val")
    copy_by_index(test_idx, "test")

    # create safety flag
    with open(SPLIT_FLAG, "w") as f:
        f.write("done")

    # print summary
    print("✔ Split completed successfully!")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

else:
    print("⏩ Using previously created split (no duplicates added).")


#Data Augmentatiom

TRAIN_DIR = ""
OUT_DIR = ""

folders = ["vh", "ndwi", "gt"]

# ========================================================
# 🔒 SAFETY FLAG — Prevents duplicate augmentation
# ========================================================
AUG_FLAG = f"{OUT_DIR}/AUGMENT_DONE19.flag"

if os.path.exists(AUG_FLAG):
    print("✔ Augmentation already done — skipping.")
    SKIP_AUG = True
else:
    print("⚠ Running AUGMENTATION for the first time...")
    SKIP_AUG = False

# create output folders
for f in folders:
    os.makedirs(f"{OUT_DIR}/{f}", exist_ok=True)


def read_img(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def safe_save(path, img, profile):
    """Save image only if it does not already exist."""
    if not os.path.exists(path):
        profile.update(dtype="float32")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(img.astype(np.float32), 1)


if not SKIP_AUG:

    # aligned file lists
    vh_files = sorted(glob(f"{TRAIN_DIR}/vh/*.tif"))
    ndwi_files = sorted(glob(f"{TRAIN_DIR}/ndwi/*.tif"))
    gt_files = sorted(glob(f"{TRAIN_DIR}/gt/*.tif"))

    print(f"Total training samples to augment: {len(vh_files)}")

    for i in range(len(vh_files)):

        # read aligned VH, NDWI, GT
        vh, vh_prof = read_img(vh_files[i])
        ndwi, ndwi_prof = read_img(ndwi_files[i])
        gt, gt_prof = read_img(gt_files[i])

        base = f"sample_{i}"

        # ======================================================
        # 1. Horizontal Flip
        # ======================================================
        safe_save(f"{OUT_DIR}/vh/{base}_hflip.tif",  np.fliplr(vh),  vh_prof)
        safe_save(f"{OUT_DIR}/ndwi/{base}_hflip.tif", np.fliplr(ndwi), ndwi_prof)
        safe_save(f"{OUT_DIR}/gt/{base}_hflip.tif",   np.fliplr(gt), gt_prof)

        # ======================================================
        # 2. Vertical Flip
        # ======================================================
        safe_save(f"{OUT_DIR}/vh/{base}_vflip.tif",  np.flipud(vh), vh_prof)
        safe_save(f"{OUT_DIR}/ndwi/{base}_vflip.tif", np.flipud(ndwi), ndwi_prof)
        safe_save(f"{OUT_DIR}/gt/{base}_vflip.tif",   np.flipud(gt), gt_prof)

        # ======================================================
        # 3. Rotations (90,180,270)
        # ======================================================
        for k in [1, 2, 3]:
            safe_save(f"{OUT_DIR}/vh/{base}_rot{k}.tif",  np.rot90(vh, k), vh_prof)
            safe_save(f"{OUT_DIR}/ndwi/{base}_rot{k}.tif", np.rot90(ndwi, k), ndwi_prof)
            safe_save(f"{OUT_DIR}/gt/{base}_rot{k}.tif",   np.rot90(gt, k), gt_prof)

    # Create safety flag
    with open(AUG_FLAG, "w") as f:
        f.write("done")

    print("✔ Augmentation completed (NO NOISE).")

else:
    print("⏩ Using existing augmented dataset (no duplicates added).")

#Nan-free Patches only

in_sar  = ""
in_ndwi = ""
in_gt   = ""

out_sar  = ""
out_ndwi = ""
out_gt   = ""

os.makedirs(out_sar, exist_ok=True)
os.makedirs(out_ndwi, exist_ok=True)
os.makedirs(out_gt, exist_ok=True)

sar_files  = sorted(os.listdir(in_sar))
ndwi_files = sorted(os.listdir(in_ndwi))
gt_files   = sorted(os.listdir(in_gt))

kept = 0
removed = 0
skipped_existing = 0

for s, n, g in tqdm(zip(sar_files, ndwi_files, gt_files), total=len(sar_files)):

    out_s_path = os.path.join(out_sar, s)
    out_n_path = os.path.join(out_ndwi, n)
    out_g_path = os.path.join(out_gt, g)

    if os.path.exists(out_s_path) and os.path.exists(out_n_path) and os.path.exists(out_g_path):
        skipped_existing += 1
        continue

    with rasterio.open(os.path.join(in_sar, s)) as src:
        sar = src.read(1)

    with rasterio.open(os.path.join(in_ndwi, n)) as src:
        ndwi = src.read(1)

    with rasterio.open(os.path.join(in_gt, g)) as src:
        gt = src.read(1)

    if np.isnan(sar).any() or np.isnan(ndwi).any() or np.isnan(gt).any():
        removed += 1
        continue

    shutil.copy(os.path.join(in_sar, s),  out_s_path)
    shutil.copy(os.path.join(in_ndwi, n), out_n_path)
    shutil.copy(os.path.join(in_gt, g),   out_g_path)

    kept += 1

print("\n✅ Done (Safe Re-run Enabled)")
print(f"New patches copied    : {kept}")
print(f"Skipped (already done): {skipped_existing}")
print(f"Removed (NaNs)        : {removed}")

#Condition diffusion model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class PatchDataset(Dataset):
    def __init__(self, sar, ndwi, gt, patch_size=64):
        self.sar = sar.float()
        self.ndwi = ndwi.float()
        self.gt = gt.float() # Store ground truth
        self.ps = patch_size

    def __len__(self):
        return len(self.sar)

    def __getitem__(self, idx):
        sar = self.sar[idx]
        ndwi = self.ndwi[idx]
        gt = self.gt[idx] # Get ground truth

        _, H, W = sar.shape
        top = torch.randint(0, H - self.ps + 1, (1,)).item()
        left = torch.randint(0, W - self.ps + 1, (1,)).item()

        return (
            sar[:, top:top+self.ps, left:left+self.ps],
            ndwi[:, top:top+self.ps, left:left+self.ps],
            gt[:, top:top+self.ps, left:left+self.ps] # Return ground truth
        )

import os
import rasterio
import torch
import numpy as np
from glob import glob
import cv2 # Import OpenCV for resizing

# Define the root directory for the augmented data
TRAIN_DATA_ROOT = ""

# Define target patch size
TARGET_SIZE = (50, 50)

def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

# List files for sar (vh), ndwi, and gt, looking for .tif files
sar_files = sorted(glob(os.path.join(TRAIN_DATA_ROOT, "vh", "*.tif")))
ndwi_files = sorted(glob(os.path.join(TRAIN_DATA_ROOT, "ndwi", "*.tif")))
gt_files = sorted(glob(os.path.join(TRAIN_DATA_ROOT, "gt", "*.tif")))

# Load images into lists
loaded_sar_images = []
loaded_ndwi_images = []
loaded_gt_images = [] # New list for ground truth

# Add an assertion for file consistency
assert len(sar_files) == len(ndwi_files) == len(gt_files), "File count mismatch in augmented data folders!"

# Check if any files were found before proceeding
if not sar_files:
    raise ValueError(f"No .tif files found in {TRAIN_DATA_ROOT}/vh. Please check the path and file extensions.")

for s_file, n_file, g_file in zip(sar_files, ndwi_files, gt_files): # Zip all three file lists
    # Read .tif files using rasterio
    sar_img = read_tif(s_file)
    sar_img = np.nan_to_num(sar_img) # Convert NaNs to 0
    # Resize SAR image
    sar_img_resized = cv2.resize(sar_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    loaded_sar_images.append(sar_img_resized)

    ndwi_img = read_tif(n_file)
    ndwi_img = np.nan_to_num(ndwi_img) # Convert NaNs to 0
    # Resize NDWI image
    ndwi_img_resized = cv2.resize(ndwi_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    loaded_ndwi_images.append(ndwi_img_resized)

    gt_img = read_tif(g_file)
    gt_img = np.nan_to_num(gt_img) # Convert NaNs to 0
    # Resize GT image
    gt_img_resized = cv2.resize(gt_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    loaded_gt_images.append(gt_img_resized)

# Convert lists of numpy arrays to a single numpy array, then to a PyTorch tensor
# Add a channel dimension for consistency (assuming single band images)
sar_images = torch.from_numpy(np.array(loaded_sar_images)).unsqueeze(1).float()
ndwi_images = torch.from_numpy(np.array(loaded_ndwi_images)).unsqueeze(1).float()
gt_images = torch.from_numpy(np.array(loaded_gt_images)).unsqueeze(1).float() # Convert GT to tensor

# Perform normalization as originally intended for sar and ndwi
sar_images = (sar_images - sar_images.min()) / (sar_images.max() - sar_images.min() + 1e-8) # Add small epsilon to avoid division by zero
sar_images = sar_images * 2 - 1

ndwi_images = (ndwi_images - ndwi_images.min()) / (ndwi_images.max() - ndwi_images.min() + 1e-8) # Add small epsilon
ndwi_images = ndwi_images * 2 - 1

# For ground truth, ensure it's binary (0 or 1) if not already, and keep it in [0,1] range
# Assuming flood_mask values are already 0 or 1, or can be thresholded to binary.
# If gt_images are already normalized to [0,1], no further normalization needed for training BCEWithLogitsLoss.
# For safety, let's ensure it's binary if it's not already.
gt_images = (gt_images > 0.5).float()

dataset = PatchDataset(sar_images, ndwi_images, gt_images, patch_size=50)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 200   # IMPORTANT (not 1000)

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alpha_bar = torch.cumprod(alphas, dim=0)


class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(2, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)

        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, sar):
        x = torch.cat([x, sar], dim=1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        return self.dec2(x)


def forward_diffusion(x0, t, noise):
    sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1,1,1,1)
    sqrt_om = torch.sqrt(1 - alpha_bar[t]).view(-1,1,1,1)
    return sqrt_ab * x0 + sqrt_om * noise

@torch.no_grad()
def simple_reverse(model, x_t, sar, t):
    eps = model(x_t, sar)
    x0 = (x_t - torch.sqrt(1 - alpha_bar[t]) * eps) / torch.sqrt(alpha_bar[t])
    return x0

sar_vis, ndwi_vis, _ = next(iter(train_loader)) # Unpack all three, but only use sar_vis and ndwi_vis
sar_vis = sar_vis[:1].to(device)
ndwi_vis = ndwi_vis[:1].to(device)

model = ConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 150

for epoch in range(epochs):
    model.train()

    for sar, ndwi, _ in train_loader: # Unpack all three, but only use sar and ndwi
        sar = sar.to(device)
        ndwi = ndwi.to(device)

        t = torch.randint(0, T, (ndwi.size(0),), device=device)
        noise = torch.randn_like(ndwi)

        x_t = forward_diffusion(ndwi, t, noise)
        noise_pred = model(x_t, sar)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    # 🔍 VISUALIZE SAME PATCH EVERY 20 EPOCHS (INSIDE LOOP)
    if (epoch + 1) % 20 == 0:
        model.eval()

        t_vis = torch.tensor([20], device=device)
        noise = torch.randn_like(ndwi_vis)
        x_t = forward_diffusion(ndwi_vis, t_vis, noise)

        recon = simple_reverse(model, x_t, sar_vis, t_vis)

        plt.figure(figsize=(9,3))

        plt.subplot(1,3,1)
        plt.imshow(ndwi_vis[0,0].cpu(), cmap="gray")
        plt.title(f"GT NDWI (Epoch {epoch+1})")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(x_t[0,0].cpu(), cmap="gray")
        plt.title("Noisy NDWI (t=20)")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(recon[0,0].cpu(), cmap="gray")
        plt.title("Reconstructed NDWI")
        plt.axis("off")

        plt.show()

import os

# Create the directory if it does not exist
os.makedirs("/content/drive/MyDrive/checkpoints", exist_ok=True)

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
    "/content/drive/MyDrive/checkpoints/cdm_final.pth" # Save to Google Drive
)

print("Final model saved to Google Drive")


#Train set Evaluation On CDM model

import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def compute_mse(pred, gt):
    return F.mse_loss(pred, gt).item()

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt).item()
    return 10 * np.log10((2.0 ** 2) / mse)

def compute_ssim(pred, gt):
    pred = pred[0,0].cpu().numpy()
    gt = gt[0,0].cpu().numpy()
    return ssim(pred, gt, data_range=2.0)

model.eval()

mse_list = []
psnr_list = []
ssim_list = []

with torch.no_grad():
    for sar, ndwi, gt in train_loader: # Unpack all three values
        sar = sar.to(device)
        ndwi = ndwi.to(device)

        # fixed diffusion step for evaluation
        t = torch.tensor([20], device=device)
        noise = torch.randn_like(ndwi)

        x_t = forward_diffusion(ndwi, t, noise)
        gen = simple_reverse(model, x_t, sar, t)

        mse_list.append(compute_mse(gen, ndwi))
        psnr_list.append(compute_psnr(gen, ndwi))
        ssim_list.append(compute_ssim(gen, ndwi))

print("===== TRAINING SET METRICS ====")
print(f"MSE  : {np.mean(mse_list):.6f}")
print(f"PSNR : {np.mean(psnr_list):.2f} dB")
print(f"SSIM : {np.mean(ssim_list):.4f}")

#NDWI Generation for train set
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


#swin-Unet for flood segmentation 

# ==========================================
# INSTALL (Run once)
# ==========================================
!pip install timm einops

# ==========================================
# IMPORTS
# ==========================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# ==========================================
# CONFIG
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 40

# ==========================================
# DATASET (50 → 56 PAD ONLY)
# ==========================================
class FloodDataset(Dataset):
    def __init__(self, sar_dir, ndwi_dir, gen_ndwi_dir, mask_dir):
        self.sar_dir = sar_dir
        self.ndwi_dir = ndwi_dir
        self.gen_ndwi_dir = gen_ndwi_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(sar_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # All files (SAR, NDWI, Generated NDWI, GT) for a given index share the same base filename.
        # This was established by the previous augmentation and NPY conversion steps.
        fname = self.files[idx]

        sar = np.load(os.path.join(self.sar_dir, fname)).astype(np.float32)
        ndwi = np.load(os.path.join(self.ndwi_dir, fname)).astype(np.float32)
        gen_ndwi = np.load(os.path.join(self.gen_ndwi_dir, fname)).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, fname)).astype(np.float32)

        sar = sar.squeeze()
        ndwi = ndwi.squeeze()
        gen_ndwi = gen_ndwi.squeeze()
        mask = mask.squeeze()

        # Normalize SAR
        sar_std = sar.std()
        sar = (sar - sar.mean()) / (sar_std + 1e-6) if sar_std > 1e-6 else np.zeros_like(sar)

        x = np.stack([sar, ndwi, gen_ndwi], axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # •̣̣ RESIZE x to 224x224 for Swin-UNet encoder input
        x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
        # Keep y (mask) at 50x50 to match model output

        return x, y

# ==========================================
# SWIN-UNET (56 INPUT)
# ==========================================
class SwinUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
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

        # Convert NHWC → NCHW
        e1, e2, e3, e4 = [f.permute(0,3,1,2) for f in feats]

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)

        out = torch.sigmoid(self.final(d2))

        # •̣̣ Crop 56 → 50 (This cropping is correct as Swin-UNet operates on 224x224 internal feature maps)
        # However, the output needs to be 50x50 to match the target `y` which is now 50x50.
        # The `F.interpolate` from the other version of SwinUNet was better for aligning output to original size.
        # Let's revert the output handling to interpolate back to 50x50 from current `final` output which is 224x224.
        return F.interpolate(out, size=(50, 50), mode="bilinear", align_corners=False)

# ==========================================
# LOSS
# ==========================================
class DiceLoss(nn.Module):
    def forward(self, p, t):
        smooth = 1e-6
        p = p.view(-1)
        t = t.view(-1)
        intersection = (p * t).sum()
        return 1 - (2*intersection + smooth) / (p.sum() + t.sum() + smooth)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, p, t):
        return 0.5*self.bce(p,t) + 0.5*self.dice(p,t)

criterion = CombinedLoss()

# ==========================================
# DATA
# ==========================================
train_dataset = FloodDataset(
    sar_dir="",
    ndwi_dir="",
    gen_ndwi_dir="",
    mask_dir=""
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ==========================================
# TRAIN
# ==========================================
model = SwinUNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "swin_unet_56.pth")
print("✅ Training Complete (56x56 version)")

# ===============================
# IMPORTS
# ===============================
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



# ==========================================
# LOAD MODEL
# ==========================================
model = SwinUNet().to(DEVICE)
model.load_state_dict(torch.load("swin_unet_56.pth", map_location=DEVICE))
model.eval()


# ==========================================
# Train DATA PATHS
# ==========================================
train_dataset = FloodDataset(
    sar_dir="",
    ndwi_dir="",
    gen_ndwi_dir="",
    mask_dir=""
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)


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

print("\n===== Train RESULTS =====")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"IoU           : {iou:.4f}")
print(f"Dice          : {dice:.4f}")
print(f"Cohen Kappa   : {kappa:.4f}")
