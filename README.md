This repository contains the official implementation of our research work on flood mapping using SAR and optical data fusion.Flood mapping plays a critical role in disaster response and environmental
management. However, commonly used satellite data sources present inherent
limitations. Optical imagery enables the use of water indices such as the Nor-
malized Difference Water Index (NDWI) for identifying flooded regions, but its
availability is often hindered during flood events due to cloud cover. In contrast,
Synthetic Aperture Radar (SAR) provides all-weather, day-and-night observa-
tions, although its backscatter characteristics are noisy and less intuitive for
water delineation. To address these challenges, we propose SwinDiffNet, a con-
ditional diffusion-based framework that reconstructs NDWI-like representations
from SAR imagery, enabling the transformation of SAR data into a more inter-
pretable representation for water detection. The generated NDWI maps are
evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity
Index Measure (SSIM), achieving 30.39 dB and 0.7941, respectively, indicating
strong reconstruction quality. For downstream flood segmentation, SAR, origi-
nal NDWI, and reconstructed NDWI are jointly utilized as multi-modal inputs
within a Swin-UNet architecture. The results demonstrate that incorporating the
1
diffusion-reconstructed NDWI improves segmentation performance by approxi-
mately 83% compared to using SAR data alone. Overall, the proposed framework
provides a robust and reliable solution for flood mapping in scenarios where opti-
cal imagery is unavailable, enhancing flood extent estimation under challenging
environmental conditions.
