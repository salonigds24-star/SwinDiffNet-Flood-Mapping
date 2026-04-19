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

1. Data Generation

A multi-source dataset was constructed using real flood events from five flood-prone states of India: Punjab, Assam, Uttar Pradesh, Bihar, and West Bengal.

Satellite data was acquired from Google Earth Engine:
Sentinel-1 (SAR) imagery
Sentinel-2 (Optical) imagery
From Sentinel-2 data, the Normalized Difference Water Index (NDWI) was computed to enhance water body detection.
For each state, 100 image patches of size 50×50 pixels were extracted, resulting in a diverse dataset capturing different flood scenarios.
Ground Truth (GT) masks were generated using:
Thresholding on SAR (VH polarization)
NDWI-based water detection
Cross-verification with multiple external sources (e.g., satellite imagery platforms and flood reports) to ensure accuracy

This process ensures a reliable and representative dataset for flood segmentation tasks


2. Proposed Approach

To address the issue of missing or unreliable NDWI information due to cloud cover, a Conditional Diffusion Model (CDM) is employed.

The CDM is used to generate NDWI representations from SAR data, enabling reconstruction of optical information even under cloudy conditions.
The final input for segmentation is formed by concatenating:
Real SAR data
Generated NDWI (from CDM)
Available real NDWI
This fused multi-modal representation is then passed to a Swin-UNet architecture, which leverages transformer-based encoding for accurate flood segmentation.

This approach enhances model robustness by effectively combining real and generated data, improving performance in challenging environmental conditions.

3.Results
The performance of the proposed SwinDiffNet framework is first evaluated on the
curated multi-state dataset. To assess the effectiveness of the Conditional Diffu-
sion Model (CDM) for SAR-to-optical translation, the recovered NDWI images are
compared with existing approaches using Peak Signal-to-Noise Ratio (PSNR) and
Structural Similarity Index Measure (SSIM).The proposed SwinDiffNet achieves a PSNR of 30.39 dB and SSIM of
0.7941 on the combined dataset, significantly outperforming the competing methods
and demonstrating its ability to reconstruct structurally consistent NDWI represen-
tations from SAR imagery. The recovered multimodal features are then used for
downstream flood segmentation using the Swin-UNet architecture.By incorporating
SAR, real NDWI, and diffusion-recovered NDWI inputs, the proposed SwinDiffNet
framework achieves an overall flood segmentation accuracy of approximately 83% on
the combined dataset, indicating that diffusion-enhanced feature generation improves
segmentation reliability and flood boundary detection.

provide a detailed comparison of segmentation performance across mul-
tiple regions, namely Punjab, Bihar, West Bengal, Assam, and Uttar Pradesh. The
proposed SwinDiffNet framework is evaluated against baseline models including 1D-
CNN, 2D-CNN, Conditional-GAN, and Swin-UNet. Across all regions, SwinDiffNet
demonstrates consistently strong performance, achieving accuracy values of 0.8481
(Punjab), 0.8499 (Bihar), 0.8593 (West Bengal), 0.8843 (Assam), and 0.8199 (Uttar
Pradesh). The corresponding improvements observed in F1-score and Intersection over
Union (IoU) indicate that incorporating diffusion-based NDWI features contributes to
more reliable flood boundary detection under varying geographical conditions.

The proposed approach across diverse environments within the Sen1Floods11
dataset. Although the dataset includes regions with varying landscape charac-
teristics and flood dynamics, the model consistently achieves an accuracy close to
0.80 for Bolivia, Ghana, India, and Nigeria. This consistency suggests that the frame-
work is relatively robust to spatial domain variations. Among the evaluated regions,
Nigeria records the highest Recall value of 0.8900, indicating stronger capability in
identifying flooded areas. Ghana shows the highest Cohen’s Kappa score of 0.5661,
reflecting better agreement beyond random classification. The IoU values, ranging from
0.7200 to 0.7399, also support the reliability of the segmentation results produced by
the proposed model. The limited inter-country performance variation highlights the
model’s ability to learn domain-invariant flood representations, thereby improving its
applicability for large-scale and real-world flood monitoring scenarios.
