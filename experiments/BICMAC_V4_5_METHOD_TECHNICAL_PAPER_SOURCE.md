# Method: PET-Aware Multimodal Pseudo-CT Generation

This document describes the final V4.5 method used for three-dimensional pseudo-CT synthesis in the BIC-MAC challenge. It is intentionally restricted to the method: problem formulation, preprocessing, network architecture, patch construction, PET-aware objectives, optimization, and inference. Experimental results, leaderboard analysis, operational procedures, and submission instructions are documented elsewhere.

## 1. Problem Formulation

The objective is to estimate a whole-body CT volume in Hounsfield units (HU) from non-attenuation-corrected PET (NAC-PET), Dixon MRI, and a two-dimensional topogram. The predicted CT is subsequently converted to a 511-keV attenuation map and used by the official PET reconstruction pipeline. We therefore formulate pseudo-CT synthesis as attenuation estimation with downstream PET-response constraints, rather than as pure image-domain HU regression.

For a subject, the inference-time inputs are:

1. a three-dimensional NAC-PET volume;
2. a two-dimensional topogram aligned with the physical x-z plane;
3. three-dimensional Dixon MRI in-phase and out-of-phase volumes;
4. absolute spatial coordinate channels derived from the original subject grid.

The seven model channels are

```text
0  NAC-PET
1  z-coordinate
2  topogram
3  y-coordinate
4  Dixon MRI in-phase
5  Dixon MRI out-of-phase
6  |MRI in-phase - MRI out-of-phase|
```

The network predicts a single-channel pseudo-CT volume

$$
\hat{C}=f_{\theta}(P_{\mathrm{NAC}},T,M_{\mathrm{in}},M_{\mathrm{out}},c_y,c_z),
$$

where $\hat{C}$ is expressed in HU. The saved NIfTI prediction has the same shape, affine, orientation, qform, and sform as the NAC-PET reference. CT labels, segmentation masks, and reconstructed PET labels are used only to define training objectives and are never required at inference.

## 2. Input Preprocessing

### 2.1 Spatial convention

NIfTI volumes are loaded in spatial order $(x,y,z)$. After adding batch and channel dimensions, tensors are represented as $[B,C,D,H,W]$, where the implementation maps $D/H/W$ to NIfTI $x/y/z$, respectively. Anatomical interpretation is always derived from the NIfTI affine rather than from array indices alone.

All modalities belonging to a subject are transformed with the same spatial crop. The topogram is represented on the corresponding x-z plane and is lifted along y only inside the topogram fusion pathway.

### 2.2 Subject-level intensity normalization

Normalization statistics are computed over each complete subject before patch extraction. This avoids patch-dependent intensity scales and does not use CT labels.

For modality $m$, valid voxels are defined as positive voxels for NAC-PET and nonzero voxels for MRI and topogram. Let $q_{0.5}^{(m)}$ and $q_{99.5}^{(m)}$ denote the 0.5th and 99.5th percentiles over valid voxels. The clipped signal is

$$
\bar{x}^{(m)}=\operatorname{clip}\left(x^{(m)},q_{0.5}^{(m)},q_{99.5}^{(m)}\right).
$$

NAC-PET is compressed with a logarithmic transform,

$$
T_{\mathrm{NAC}}(x)=\log(1+\max(x,0)),
$$

whereas MRI and topogram use the identity transform. Subject-level mean and standard deviation are then computed over valid voxels:

$$
\mu_s^{(m)}=\operatorname{mean}\left(T_m(\bar{x}^{(m)})\right),
\qquad
\sigma_s^{(m)}=\operatorname{std}\left(T_m(\bar{x}^{(m)})\right).
$$

Every training crop and inference window is normalized using the same subject-level statistics:

$$
x_{\mathrm{norm}}^{(m)}=
\frac{T_m(\bar{x}^{(m)})-\mu_s^{(m)}}{\sigma_s^{(m)}+\epsilon}.
$$

The logarithmic transform reduces the strongly right-skewed NAC-PET dynamic range. MRI intensities are scanner-dependent arbitrary signals and are therefore percentile-clipped and standardized without a count-like logarithmic transform.

The Dixon absolute-difference channel is constructed before channel-wise normalization:

$$
M_{\mathrm{diff}}=|M_{\mathrm{in}}-M_{\mathrm{out}}|.
$$

This channel exposes complementary fat-water contrast while retaining the original in-phase and out-of-phase measurements.

### 2.3 Absolute coordinate channels

For an axis of original size $N$, the coordinate at index $i$ is

$$
c(i)=2\frac{i}{N-1}-1.
$$

Coordinates are computed from the crop origin and the full subject shape. Consequently, the same physical-relative position receives the same coordinate during random patch training and sliding-window inference. The z-coordinate provides superior-inferior context to the geometry branch. The y-coordinate provides anterior-posterior context when a two-dimensional topogram feature is lifted into a three-dimensional representation.

## 3. Network Architecture

### 3.1 Overview

The model is a geometry-anchored multimodal Residual 3D U-Net with 20.58 million trainable parameters. NAC-PET defines the main three-dimensional geometry pathway. Topogram and MRI features are injected through gated residual adapters, allowing the network to suppress a modality when it is locally unreliable. A compact whole-body encoder supplies global context to coarse local features through feature-wise linear modulation (FiLM).

```text
NAC-PET + z-coordinate -> 3D geometry encoder -------------------------+
                              ^                                        |
topogram -> 2D encoder -> lifted gated residual fusion                 |
Dixon MRI -> lightweight 3D encoder -> gated residual fusion          |
whole-body [NAC-PET, topogram, z] -> compact encoder -> FiLM           |
                                                                       v
             fused multiscale skips -> residual 3D decoder -> CT in HU
```

### 3.2 Geometry encoder and decoder

The geometry encoder receives normalized NAC-PET and the z-coordinate channel. Each residual block contains two $3\times3\times3$ convolutions, instance normalization, and leaky-ReLU activation. For an input $x$,

$$
R(x)=\operatorname{IN}\left(operatorname{Conv}_{3^3}\left(
\operatorname{LReLU}\left(operatorname{IN}\left(operatorname{Conv}_{3^3}(x)\right)\right)
\right)\right),
$$

$$
y=\operatorname{LReLU}(R(x)+S(x)),
$$

where $S$ is the identity mapping or a $1\times1\times1$ projection when the number of channels changes. Downsampling uses max pooling. The decoder uses transposed convolutions, concatenates the corresponding fused encoder skip, and applies another residual block.

For a local patch of $160\times160\times192$ voxels and base width 24, the geometry hierarchy is:

| Scale | Geometry tensor |
|---|---|
| s1 | $[B,24,160,160,192]$ |
| s2 | $[B,48,80,80,96]$ |
| s3 | $[B,96,40,40,48]$ |
| s4 | $[B,192,20,20,24]$ |
| bottleneck | $[B,384,10,10,12]$ |

A dropout probability of 0.2 is applied at the bottleneck. The final decoder layer produces one HU channel.

### 3.3 Two-dimensional topogram pathway

The topogram is a scout X-ray in the physical x-z plane. It contains global body silhouette and projection information but does not explicitly resolve depth along y. A dedicated two-dimensional encoder extracts features with widths $24,48,96,192,384$:

```text
s1 [B, 24,160,192]
s2 [B, 48, 80, 96]
s3 [B, 96, 40, 48]
s4 [B,192, 20, 24]
b  [B,384, 10, 12]
```

At each scale, the two-dimensional feature $T$ is lifted along y and projected to the geometry width. At s1-s3, topogram information is added through a conservative gated residual adapter:

$$
g=\sigma\left(f_g([G,T,|G-T|])\right),
$$

$$
F=G+g\odot Z(T),
$$

where $G$ is the geometry feature, $Z$ is a zero-initialized $1\times1\times1$ residual projection, and the gate bias is initialized to $-2$. This initialization preserves the NAC-PET geometry pathway at the beginning of training while permitting the topogram contribution to grow when useful.

At s4 and the bottleneck, the lifted feature is additionally conditioned on anterior-posterior position. A Fourier encoding with three frequency bands is projected to an eight-channel embedding $PE(y)$. The fusion is

$$
C=[G,T,|G-T|,PE(y)],
$$

$$
H=f_h(C),\qquad R=f_R(H),\qquad g=\sigma(f_g(H)),
$$

$$
F=G+g\odot R.
$$

The residual output is Kaiming-initialized and multiplied by 0.01. The final gate weights are initialized to zero and its bias to $-1$, giving an initial gate value of $\sigma(-1)\approx0.269$. The resulting correction is small but nonzero, so gradients reach the residual pathway from the first update.

### 3.4 Lightweight MRI pathway

The MRI pathway receives Dixon in-phase, out-of-phase, and absolute-difference volumes. It uses a separate three-dimensional encoder with base width 12 and channel widths $12,24,48,96,192$. At every encoder scale and at the bottleneck, MRI features are fused into the geometry stream by

$$
g_m=\sigma\left(f_m([G,M,|G-M|])\right),
$$

$$
F=G+g_m\odot Z_m(M),
$$

where $M$ is the MRI feature and $Z_m$ is a zero-initialized residual projection. The gate bias is initialized to $-2$.

The separate encoder prevents MRI from directly redefining the geometry anchor. This is important because Dixon MRI contributes soft-tissue and fat-water information but can be locally misregistered due to differences in scanner geometry, acquisition time, and patient pose. The residual gate can attenuate MRI corrections when local correspondence is unreliable.

### 3.5 Whole-body context conditioning

A local patch cannot observe the complete body extent. The method therefore processes a compact whole-body representation containing downsampled NAC-PET, topogram, and z-coordinate at $96\times96\times128$ voxels. A lightweight encoder with base width 8 produces:

```text
[  8,96,96,128]
[ 16,48,48, 64]
[ 32,24,24, 32]
[ 64,12,12, 16]
[128, 6, 6,  8]
```

After global average pooling, the context embedding predicts FiLM parameters for local s4 and bottleneck features:

$$
F'=F\odot(1+0.1\tanh\gamma)+0.1\beta.
$$

The FiLM linear layers are initialized to zero, so the initial network is close to the unconditioned local backbone. The whole-body overview is computed once and reused for all patches from the same subject. Activation checkpointing is used in the global encoder to control memory. An auxiliary global CT head is present in the implementation but is not supervised and does not contribute to the V4.5 objective.

## 4. Patch Construction and Full-FOV Composition

### 4.1 Spatially stratified patch sampling

Each subject record generates eight patches of $160\times160\times192$ voxels. Candidate centers are drawn from the official prediction mask and jittered independently by at most one quarter of the ROI size along each axis. A candidate is accepted when at least 35% of its voxels belong to the prediction mask. Up to 16 attempts are made; if none meets the threshold, the candidate with the largest mask fraction is retained.

Sampling is stratified over the body extent along the original NIfTI z-axis:

| Region | Relative body extent | Sampling rule |
|---|---:|---:|
| lower body | 0.00-0.22 | weight 0.15 |
| abdomen and pelvis | 0.22-0.55 | weight 0.30 |
| thorax | 0.55-0.78 | weight 0.30 |
| head and neck | 0.78-1.00 | exactly 2 of 8 patches |

The remaining six patches are sampled from the first three regions. Region boundaries are defined from the actual prediction-mask extent rather than from padded array dimensions.

### 4.2 Compact full-FOV error composition

The PET-response objectives require long-range attenuation information. For each subject record, the eight local attenuation-error patches are inserted into a compact $96\times96\times128$ whole-body grid using their absolute crop origins and the original subject shape. Overlapping contributions are averaged. Uncovered voxels are assigned zero attenuation error, equivalent to assuming $\hat\mu=\mu$ outside the sampled support for that optimization step.

This operation retains differentiability from the compact full-FOV response back to every covered local prediction. It also separates local high-resolution CT estimation from the lower-resolution global ray computation.

## 5. PET-Aware Training Objectives

### 5.1 HU-to-511-keV attenuation conversion

Let $h$ denote CT intensity in HU and $u=h+1000$. We use the official bilinear conversion to the 511-keV linear attenuation coefficient in $\mathrm{cm}^{-1}$:

$$
\mu(h)=
\begin{cases}
9.6\times10^{-5}u, & u<1047,\\
5.10\times10^{-5}u+4.71\times10^{-2}, & u\ge1047.
\end{cases}
$$

Negative attenuation values are clamped to zero. Predicted CT values are clipped to $[-1000,2000]$ HU before conversion for numerical stability; the ground-truth CT follows the official conversion without prediction clipping.

### 5.2 Raw attenuation-map loss

The raw attenuation objective directly matches the official CT quantity within the official CT evaluation mask $M_{CT}$:

$$
L_{\mu,\mathrm{raw}}=
\frac{\sum_v M_{CT,v}|\hat\mu_v-\mu_v|}
{\sum_v M_{CT,v}+\epsilon}.
$$

The mask excludes the evaluator-defined region around the superior liver boundary to reduce sensitivity to respiratory misalignment.

### 5.3 Effective smoothed attenuation loss

During official reconstruction, voxels outside the prediction mask are replaced with reference face-and-bed CT. Training reproduces this composition:

$$
\hat\mu_{\mathrm{eff}}=M_{\mathrm{pred}}\hat\mu+
(1-M_{\mathrm{pred}})\mu_{\mathrm{GT}}.
$$

Both effective prediction and target are smoothed with a 4-mm full-width-at-half-maximum Gaussian. For voxel spacing $\Delta_d$ along axis $d$,

$$
\sigma_{d,\mathrm{voxel}}=\frac{4/2.355}{\Delta_d}.
$$

The dimensionless smoothed attenuation loss is

$$
L_{\mu,\mathrm{smooth}}=
\frac{1}{\mu_{\mathrm{water}}}
\frac{\sum_v M_v
|G(\hat\mu_{\mathrm{eff}})_v-G(\mu_{\mathrm{GT}})_v|}
{\sum_v M_v+\epsilon},
$$

where $M=M_{\mathrm{pred}}\cap M_{\mathrm{body}}$ and $\mu_{\mathrm{water}}=0.096\,\mathrm{cm}^{-1}$.

### 5.4 Differentiable attenuation-response surrogate

PET attenuation correction depends on line integrals of $\mu$. For a line $l$ at transaxial angle $\theta$, the attenuation-integral error is

$$
\Delta\tau_{\theta}(l)=\int_l(\hat\mu-\mu)\,ds.
$$

Because the attenuation correction factor is $\mathrm{ACF}=\exp(\tau)$, its relative error is

$$
r_{\theta}(l)=
\frac{\widehat{\mathrm{ACF}}}{\mathrm{ACF}}-1
=\exp(\Delta\tau_{\theta}(l))-1.
$$

The implementation rotates the compact full-FOV attenuation-error volume in the transaxial plane over 24 angles from $0^\circ$ to $172.5^\circ$ in $7.5^\circ$ increments. It sums along each ray using physical spacing in centimeters, clamps $\Delta\tau$ to $[-0.5,0.5]$, applies `expm1`, backprojects the ray response, and averages by projection coverage. This produces a voxel response field $R$ that approximates the relative PET error induced by attenuation mismatch.

The response module is an attenuation-response surrogate, not a differentiable OSEM reconstruction. It does not model scatter, randoms, detector normalization, sensitivity updates, Poisson statistics, iterative image updates, or the final reconstruction filter.

### 5.5 Controlled activity-aware ray weighting

For PET-labelled subjects, the mean ground-truth PET activity on each supported ray is normalized by the mean activity over supported rays. Ray weights are

$$
w_l=\operatorname{clip}\left((1-\alpha)+\alpha a_l^{\mathrm{norm}},0.8,1.2\right),
\qquad \alpha=0.15.
$$

The activity tensor is detached from the computation graph. The bounded range prevents isolated hotspots from dominating the objective while giving modestly greater importance to attenuation errors on rays carrying more true PET activity. For subjects without reconstructed PET labels, $\alpha=0$ and ray weighting is uniform.

### 5.6 Whole-body PET-response loss

Let $A_v$ denote evaluator-scaled ground-truth PET activity and $M_{WB}$ the exact whole-body PET evaluation mask. Under a first-order response model,

$$
\hat P_v\approx P_v(1+R_v),
\qquad
|\hat P_v-P_v|\approx P_v|R_v|.
$$

The whole-body surrogate is

$$
L_{WB}=
\frac{\sum_v M_{WB,v}A_v|R_v|}
{\sum_v M_{WB,v}+\epsilon}.
$$

For PET-labelled subjects, $A$ and $M_{WB}$ follow the official evaluator scaling and mask definitions. For subjects without PET labels, detached NAC-PET activity and CT-space mask fallbacks maintain a conservative attenuation-response signal.

### 5.7 Signed eight-region organ-bias loss

The official organ metric averages the absolute signed mean bias over eight equally weighted regions: brain, liver, spleen, heart, pancreas, muscle, adipose tissue, and extremities. To preserve cancellation within a region before taking the absolute value, the surrogate computes

$$
b_k=
\frac{\sum_v M_{k,v}A_vR_v}
{\sum_v M_{k,v}A_v+\epsilon},
$$

$$
L_{\text{organ}}=
\frac{1}{8}\sum_{k=1}^{8}|b_k|.
$$

For PET-labelled subjects, the exact evaluator region masks are resampled by affine to a grid at twice the compact resolution and then average-pooled to $96\times96\times128$. This procedure retains fractional occupancy near small-organ boundaries, particularly for the pancreas and spleen.

### 5.8 Brain soft-outlier loss

The official brain score counts voxels whose relative PET error falls below 5%, 10%, and 15%. We approximate the complementary outlier indicator with a normalized soft threshold. Let $e_v=|R_v|$, threshold $t$, and temperature $T=0.01$:

$$
q_t(e)=\operatorname{clip}\left(
\frac{\sigma((e-t)/T)-\sigma(-t/T)}
{1-\sigma(-t/T)+\epsilon},0,1\right).
$$

The brain loss is

$$
L_{\mathrm{brain}}=
\frac{1}{3}\sum_{t\in\{0.05,0.10,0.15\}}
\frac{\sum_v M_{\mathrm{brain},v}^{\mathrm{valid}}q_t(e_v)}
{\sum_v M_{\mathrm{brain},v}^{\mathrm{valid}}+\epsilon}.
$$

This loss matches the threshold structure of the evaluator while remaining differentiable. It still inherits the approximation error of the attenuation-response field $R$.

### 5.9 Loss-scale normalization and total objective

The raw objectives have substantially different numerical scales. During the first 100 optimization steps, a reference value is estimated for each active component:

$$
\operatorname{ref}_j=
\frac{1}{100}\sum_{n=1}^{100}L_j^{(n)}.
$$

Each component is normalized before weighting:

$$
\tilde L_j=\frac{L_j}{\operatorname{ref}_j+\epsilon}.
$$

The complete objective is

$$
L_{\mathrm{total}}=
\lambda_{\mu,r}\tilde L_{\mu,\mathrm{raw}}
+\lambda_{\mu,s}\tilde L_{\mu,\mathrm{smooth}}
+\lambda_{WB}\tilde L_{WB}
+\lambda_{\mathrm{organ}}\tilde L_{\mathrm{organ}}
+\lambda_{\mathrm{brain}}\tilde L_{\mathrm{brain}}.
$$

The implemented schedule is:

| Epochs | $\lambda_{\mu,r}$ | $\lambda_{\mu,s}$ | $\lambda_{WB}$ | $\lambda_{\mathrm{organ}}$ | $\lambda_{\mathrm{brain}}$ |
|---|---:|---:|---:|---:|---:|
| 1-30 | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| 31-90 | 1.00 | 0.50 | 0.05 | 0.05 | 0.02 |
| 91-117 | 1.00 | 0.50 | 0.15 | 0.15 | 0.05 |
| 118-150 | 1.00 | 0.50 | 0.20 | 0.20 | 0.015 |

The first stage establishes attenuation-map accuracy before PET-response terms are activated. Later stages increase whole-body and regional PET-response supervision. The brain term is reduced in the final stage to preserve an already stable brain response while placing greater emphasis on whole-body SUV and organ bias.

Generic HU L1, gradient, orthogonal projection, generic organ, and auxiliary global-head losses have zero weight in V4.5 and do not contribute gradients.

## 6. PET-Label Sampling

All 75 training subjects contribute to CT attenuation learning. Exact reconstructed PET activity and evaluator masks are available for eight subjects. Each PET-labelled subject is represented by one additional training record per epoch:

```text
75 unique subject records + 8 PET-labelled replicas = 83 records per epoch
```

This gives 16 PET-supervised appearances per epoch, or 19.28% of all records, while retaining 80.72% non-replicated records. The replication increases exposure to exact PET supervision without excluding subjects that only provide CT labels. Each record still generates independently sampled spatial patches, so the repeated subject does not deterministically reproduce the same crop set.

## 7. Optimization

The network is optimized with the following settings:

| Parameter | Value |
|---|---:|
| optimizer | AdamW |
| initial learning rate | $1\times10^{-4}$ |
| weight decay | $1\times10^{-5}$ |
| scheduler | cosine annealing over 150 epochs |
| batch size | 1 subject record |
| patches per record | 8 |
| local patch size | $160\times160\times192$ |
| mixed precision | enabled |
| initial AMP scale | 4096 |
| gradient-norm clipping | 12 |
| random seed | 42 |

The input channels and network topology remain fixed throughout training; only loss weights change between phases. Whole-body context is reused across the eight patches of a subject record. The implementation supports checkpoint resumption with model, optimizer, scheduler, scaler, and loss-reference states.

## 8. Sliding-Window Inference

Inference uses a $160\times160\times192$ sliding window with 50% overlap and `sw_batch_size=1`. Patch predictions are accumulated on CPU to satisfy the deployment memory budget. The inference context matches the training patch size, preventing a train-inference receptive-field mismatch.

The final HU volume is written as `ct.nii.gz`. The output preserves the NAC-PET NIfTI spatial metadata exactly. No training-only masks, CT labels, organ segmentations, or PET labels are loaded during inference.

## 9. Method References

1. Carney JP, Townsend DW, Rappoport V, Bendriem B. Method for transforming CT images for attenuation correction in PET/CT imaging. *Medical Physics*. 2006;33(4):976-983. https://doi.org/10.1118/1.2174132
2. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://arxiv.org/abs/1505.04597
3. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
4. Perez E, Strub F, de Vries H, Dumoulin V, Courville A. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018. https://ojs.aaai.org/index.php/AAAI/article/view/11671
5. Liu R et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution. NeurIPS 2018. https://proceedings.neurips.cc/paper/2018/hash/60106888f8977b71e1f15db7bc9a88d1-Abstract.html
6. Zhang L, Rao A, Agrawala M. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html
7. Shi L et al. A Novel Loss Function Incorporating Imaging Acquisition Physics for PET Attenuation Map Generation using Deep Learning. MICCAI 2019. https://arxiv.org/abs/1909.01394
8. Torrado-Carvajal A et al. Dixon-VIBE Deep Learning (DIVIDE) Pseudo-CT Synthesis for Pelvis PET/MR Attenuation Correction. https://pmc.ncbi.nlm.nih.gov/articles/PMC6910626/
9. Dong X et al. Synthetic CT generation from non-attenuation-corrected PET images for whole-body PET imaging. https://pmc.ncbi.nlm.nih.gov/articles/PMC7759014/
10. BIC-MAC Challenge official evaluator. https://github.com/bic-mac-challenge/challenge-codebase/tree/main/src/evaluation
