# 3D CT foundation models — preprocessing & loading reference

Collected from hands-on integration of ten 3D chest-CT foundation models
(SPECTRE, COLIPRI, CT-SSG, CT-CLIP, CT-FM, Merlin, Pillar-0, Curia-2, VoxelFM,
FlexiCT) for embedding extraction + kNN / zero-shot / linear-probe evaluation. Dataset: RadChestCT (2284 valid samples,
`final_spacing` from `CT_Scan_Metadata_Complete_35747.csv`, NPZ axis order
verified as (I, P, L) in LPS physical space).

Primary use: **embedding extraction** for downstream classification / kNN /
retrieval. Most of the wiring also works unchanged for **report generation**
as long as you only need a volume-level feature vector (or per-scale tokens)
as the visual input to an LLM/decoder.

---

## General lessons

### 1. Never reinvent per-model preprocessing

Every author publishes a canonical inference pipeline (`scripts/data_inference_nii.py`,
`transforms.py`, `configs/ct_chest.yaml`, or equivalent). **Mirror those
parameters exactly**; don't paste "generic CT preprocessing." Mismatches
(e.g. wrong HU clamp, wrong spacing, wrong orientation, 0-1 vs -1-+1 scale)
show up as silent performance drops, not crashes.

Each model's preprocessing should be encapsulated in its own extractor class,
with a MONAI `Compose` (or the author's torchio pipeline) built from the
published parameters.

### 2. RadChest NPZ axis convention (I, P, L)

RadChest stores each volume as numpy `(Z, Y, X)` where:

- Axis 0 (Z): first voxel = superior end, last = inferior → positive direction **I**
- Axis 1 (Y): first voxel = anterior, last = posterior → positive direction **P**
- Axis 2 (X): first voxel = right, last = left → positive direction **L**

This was **verified visually** by dumping mid-plane slices (axial / coronal /
sagittal) and checking anatomy orientation. Do the same verification on any
new dataset — don't assume DICOM-standard conventions hold after a resampling
step.

Per-sample **isotropic voxel spacing** (column `final_spacing`) is required
for any model that does physical-spacing resampling (all of them except
SPECTRE-style grid models).

### 3. MONAI `MetaTensor` + `space` attribute gotcha

`MetaTensor(vol, affine=diag([s,s,s,1]))` with **no explicit space** defaults
to **RAS** world coords. If your data is actually LPS, MONAI will interpret
your positive affine as RAS → subsequent `Orientationd(axcodes="SLP")` will
reorient "relative to RAS" → produce the wrong anatomical axes.

Fix: set space explicitly.

```python
from monai.utils.enums import SpaceKeys
from monai.data.meta_tensor import MetaTensor

affine = _radchest_lps_affine_zyx(spacing)     # see helper below
mt = MetaTensor(vol, affine=affine, meta={"space": SpaceKeys.LPS})
```

The RadChest LPS affine for a volume stored as `(Z, Y, X)` with axis
directions (I, P, L):

```python
def _radchest_lps_affine_zyx(spacing: float) -> torch.Tensor:
    s = float(spacing)
    return torch.tensor(
        [[0,  0, s, 0],   # world X (L) from voxel 2
         [0,  s, 0, 0],   # world Y (P) from voxel 1
         [-s, 0, 0, 0],   # world Z (S) from -voxel 0 (axis 0 points I)
         [0,  0, 0, 1]],
        dtype=torch.float64,
    )
```

### 4. Torchio vs MONAI conventions

- **Torchio**: `tio.ScalarImage(tensor, affine)` wants the tensor in
  `(C, X, Y, Z)` order — permute `(C, Z, Y, X) → (C, X, Y, Z)` first. Its
  world convention is LPS.
- **MONAI**: transforms are affine-aware; keep the tensor in its native axis
  order and let the affine describe it. `Spacingd` / `Orientationd` use the
  affine, not the axis layout.

### 5. Dataset preprocessing lives in DataLoader workers

The heavy lift (NPZ inflate, resample, crop, window, normalize) goes in a
worker-side `preprocess` callable. The main thread only moves tensors to GPU
and runs the forward. **Never put `ToDeviced` inside a multi-worker
DataLoader** — each worker would need its own CUDA context.

### 6. TumorImagingBench ≠ whole-CT classification

[AIM-Harvard/TumorImagingBench](https://github.com/AIM-Harvard/TumorImagingBench)
is an excellent reference for each model's **orientation / spacing / HU /
normalization**, but its MONAI pipeline uses `SeedBasedPatchCropd` (a
48³–60³ patch around a lesion seed). For whole-volume classification / kNN /
report generation, replace that with each model's **native full-volume
input shape** (see per-model table below).

### 7. Most models have one of three embedding interfaces

- `encode_image(batch, pool=True, project=True, normalize=True)` → CLIP-style
  (COLIPRI, Pillar-0's `visual(...)`).
- `model.visual(...)` + manual pooling → same idea, slightly different
  plumbing.
- Forward hook on the classifier head's `nn.Linear` → supervised classifiers
  where the author didn't expose a pre-classifier feature API (CT-SSG).
- `return_encoded_tokens=True` + `adaptive_avg_pool3d` → token-based CLIP
  (CT-CLIP).

If there's no documented embedding API, **skip `model.forward` and call the
vision branch directly**, or hook the classifier input.

### 8. Some models require state_dict key surgery

- **Pillar-0**: safetensors keys have a leading `"model."` prefix that the
  in-memory `CustomTextCLIP` doesn't use — strip it on load. Also,
  `AutoModel.from_pretrained` fails on Pillar-0 with a meta-tensor `.item()`
  error; use `AutoModel.from_config` + manual `load_file` + `load_state_dict`
  on CPU.
- **CT-SSG**: state_dict loads cleanly, but `args.device` is a required
  attribute the README example doesn't set. Also move
  `model.blocks.edges.edges_{index,weight}` tensors to device manually —
  they're not in state_dict.
- **CT-CLIP**: filter state_dict to only `visual_transformer.*` keys, strip
  the prefix, load non-strict.

---

## Per-model preprocessing contracts

All tensors are single-channel 3D CT unless a "channels" column says
otherwise. Axis-code notation follows nibabel: `"SLP"` means voxel axis 0 goes
Superior, axis 1 goes Left, axis 2 goes Posterior (positive directions).

| model | install | weights | orient | spacing (mm) | input shape | HU clip → scale | channels | embed dim | per-sample code |
|---|---|---|---|---|---|---|---|---:|---|
| **SPECTRE** (Claessens et al. 2024) | `pip install spectre-fm` | via `spectre.MODEL_CONFIGS["spectre-large-pretrained"]` | RAS | (0.75, 0.75, 1.5) | 384×384×256 (grid-tiled from 128×128×64 patches) | [-1000, 1000] → [0, 1] | 1 | 1080 | `wrapper.infer_from_volume(vol)[:,0,:]` (CLS token) |
| **COLIPRI** (Wald et al. 2026, Microsoft) | `pip install colipri` | `get_model(pretrained=True, image_only=True)` | SAR | 2.0 iso | 192×192×192 | [-1000, 1000] → [-1, 1] (uses torchio `RescaleIntensity`) | 1 | 768 | `model.encode_image(batch, pool=True, project=True, normalize=True)` |
| **CT-SSG** (Di Piazza et al. 2026) | clone `github.com/theodpzz/ct-ssg` | HF `theodpzz/ct-ssg` → `model_state_dict.pt` | SLP | (1.5, 0.75, 0.75) | 240×480×480 | [-1000, 200] → [0, 1] − 0.449 (ImageNet mean) | 1 | 512 (z̄) | `z = model.blocks(model.stem(batch))` — skip classifier head |
| **CT-CLIP** (Hamamci et al. 2024) | `pip install transformer-maskgit` | HF dataset `ibrahimhamamci/CT-RATE` → `models/CT-CLIP-Related/CT-CLIP_v2.pt` (strip `visual_transformer.` prefix) | SLP | (1.5, 0.75, 0.75) | 240×480×480, pad with -1 | [-1000, 1000] → [-1, 1] | 1 | 512 | `adaptive_avg_pool3d(model(batch, return_encoded_tokens=True).permute(0,4,1,2,3), 1).flatten(1)` |
| **CT-FM** (Pai et al. 2025, IBM) | `pip install fmcib monai` | HF `surajpaib/CT-FM-SegResNet` → `pretrained_segresnet.torch` (strip `encoder.` prefix) | SPL | (3.0, 1.0, 1.0) | whole volume, then **sliding window 24×128×128 patches, mean-pool** | [-1024, 2048] → [0, 1] | 1 | 512 | `SegResEncoder(...)` with `adaptive_avg_pool3d(x[-1], 1)` head; `SlidingWindowSplitter(24,128,128)` → batch forward → mean across patches |
| **Merlin** (Blankemeier, Kumar et al. 2026, Stanford) | `pip install merlin` | via `Merlin(ImageEmbedding=True)` auto-download | RAS | (1.5, 1.5, 3.0) | 224×224×160 | [-1000, 1000] → [0, 1] | 1 | 2048 (ResNet152-3D-ish) | `outputs = model(batch); outputs[0]` |
| **Pillar-0** (Agrawal et al. 2025, YalaLab) | clone `github.com/YalaLab/rave` + `pip install -e third_party/rave` | HF `YalaLab/Pillar0-ChestCT` → `model.safetensors` (strip `model.` prefix) | LPS | (1.25, 1.25, 1.25) iso | 256×256×256 | none at worker stage (raw HU passed through) | **11 (multi-windowing)** via `rve.apply_windowing(vol, 'all', 'CT')` | 1152 | `model.model.visual({"chest_ct": batch}, {"anatomy": ["chest_ct"]})` |
| **Curia-2 B** (Saporta et al. 2026, Raidium) | `transformers` (no clone; *not* gated) | HF `raidium/curia-2` → `AutoModel.from_pretrained` (stock `Dinov2Model`) | PLS, then **per-axial-slice** | none (no mm resample; 512 px in-plane resize) | **2D slices** 512×512 (last axis = axial), mean-pool over slices | per-slice: ÷32767 → bicubic-resize → clamp_min(-1000); then **one per-volume z-score** | 1 | 768 (CLS) | slice-loop → `model(pixel_values=slices).pooler_output` (layernormed CLS) → `mean(0)` |
| **FlexiCT-3D / -VLM** (Li et al. 2026, GT/Emory) | clone `github.com/ricklisz/FlexiCT` → `third_party/FlexiCT` (added to sys.path; deps in its `requirements.txt`, our env already satisfies them) | HF `ricklisz123/FlexiCT` or Google-Drive (readme) → `ct_3d_vlm.pth` (VLM) / `ct_3d_teacher.pth` (3D SSL); env `FLEXICT_VLM_CHECKPOINT` / `FLEXICT_3D_CHECKPOINT` | **LPS** (FlexiCT canonical; = its `Orientationd(axcodes="LPS")`) | (2.0, 2.0, 2.0) iso | 160×160×160 (pad-with-min + center-crop; patch 8) | [-1000, 1000] clip → **whole-volume z-score** | 1 | **1728** kNN (concat CLS + mean patch) / **1024** ZS (VLM projection) | kNN: `model.model.vision_model(x, is_training=True)` → `cat([cls, patch.mean(1)])`; ZS img: `model.encode_image(x)`; ZS text: `model.encode_text(prompts)` (Qwen3) |
| **VoxelFM** (Maguado et al. 2026) | clone `github.com/rmaguado/VoxelFM` → `third_party/VoxelFM` (added to sys.path; deps already in env. Patch `dinov2/inference/distributed.py` with `from __future__ import annotations` — its `Queue[...]` annotation breaks the package import on Py≤3.11) | HF `rmaguado/VoxelFM` (Apache-2.0, **not** gated) → `vitb_3d/checkpoints/99999.pth` + `vitb_3d/config.yaml`; `build_model` takes `state["teacher"]`, strips `module.`/`backbone.`, drops `dino_head`/`ibot_head` | **axial-first** (through-plane axis → dim0; no L/R/A/P canonicalization — mirrors author loaders) | **adaptive isotropic** = max(min(native), 0.75) | variable (RoPE-3D, no fixed grid); patch 14; `resize_max_patches=25000` token budget | [-1000, 1900] clip → **global dataset z-score** (`config.datasets[0].norm.mean/std`) | 1 | **864** (mean-pooled patch tokens) | `crop_volume(k=21)` → `resize_isotropic` → `resize_max_patches` → `patch_crop` → `(x-fmean)/fstd` → `model(x)["x_norm_patchtokens"][0].mean(0)` |

**Notes per model:**

- **SPECTRE**: orientation-/spacing-agnostic by training; we verified the
  LPS-fix doesn't change macro PR-AUC. Grid-size ablation (3×3×4 → 4×4×4)
  gives a tiny lift that's not clearly outside the CI.
- **COLIPRI**: input must be a `torchio.ScalarImage` with correct affine
  (not a bare tensor). Its processor owns the whole pipeline; don't do
  intensity normalization upstream.
- **CT-SSG**: z̄ = 512-d mean-pooled post-ChebConv feature. Paper uses it
  for t-SNE and as the representation (Appendix D). Skip the classifier.
  Set `args.device = device` at construction, then move
  `model.blocks.edges.edges_{index,weight}` to device manually.
- **CT-CLIP**: `model(batch, return_encoded_tokens=True)` returns
  `[B, t, h, w, dim]` (no channel dim in the expected position). Permute
  to `[B, dim, t, h, w]` then `adaptive_avg_pool3d(..., 1).flatten(1)`.
- **CT-FM**: sliding-window required. Output shape is `[n_patches, 512]`;
  mean-pool over patches. The patch-level features **also** make sense as
  an embedding if your downstream task is patch-local (e.g. tumor tasks).
- **Merlin**: loads its own weights via `Merlin(ImageEmbedding=True)`. No
  manual checkpoint path. Embedding is the first element of the tuple
  returned by `model(image)`.
- **FlexiCT**: one ViT-Base backbone (patch 8, dim 864) shared by the SSL
  teacher (`Flexi_CT_3D`, ct_3d_teacher.pth) and the report-aligned VLM
  (`Flexi_CT_VLM`, ct_3d_vlm.pth). The **VLM serves both tracks** — its vision
  encoder gives the 1728-d concat for kNN and the 1024-d projection for ZS — so
  one extraction per dataset covers kNN + ZS. The 3D-SSL teacher is kept only
  for an SSL-vs-VLM kNN sanity check (`flexict3d_*` configs). Preprocessing
  mirrors the author's CT-RATE eval (`downstream/vlm/{dataset,ct_rate_eval}.py`):
  2 mm iso, clip[-1000,1000], **whole-volume** z-score, pad-with-min +
  center-crop to 160³. **Orientation = LPS**, author-sourced: FlexiCT's
  retrieval pipeline uses `Orientationd(axcodes="LPS")` and its C4KC/demo paths
  use sitk `DICOMOrient("LPS")`; the classification eval (`CT3D_CLIP`) skips
  reorientation only because CT-RATE is already LPS. We apply the same MONAI
  transform on our verified per-dataset LPS affines, so it's correct by
  construction (a one-time slice dump is still cheap insurance). ZS prompts: the author
  uses bare `"{cls} ."` / `"No {cls}."`, not our generic caption template.
  **Leakage**: CT-RATE + NLST are in FlexiCT's pretraining → CT-RATE eval is
  in-distribution; RadChest is likely the clean out-of-distribution test.
- **Pillar-0**: **most complex**. Needs `rve.apply_windowing(vol, 'all', 'CT')`
  to produce the 11-channel input (HU windows: lung, mediastinum, abdomen,
  liver, bone, brain, subdural, stroke, temporal_bone, soft_tissue, minmax).
  `rve` is a pip package (`rad-vision-engine==1.0.0`) available from
  `github.com/YalaLab/rave`. Model loading must use `AutoModel.from_config`
  + manual `load_state_dict` because `from_pretrained` hits a meta-tensor
  `.item()` error in Pillar-0's custom modeling code.
- **Curia-2**: the **only slice-based model** in the lineup, and the only one
  that natively targets **MRI** as well as CT — MRI is its headline
  experiment. The released checkpoint is a stock HF `Dinov2Model`, so it loads
  directly with `AutoModel.from_pretrained("raidium/curia-2")` (public, no
  gating/token). No text tower → kNN / linear-probe only (peer = CT-FM).
  Preprocessing replicates `curia_image_processor.py` (`CuriaImageProcessor`)
  op-for-op: per axial slice `to(int16) → convert_image_dtype(float32)`
  (= ÷32767) `→ F.interpolate(512, bicubic, align_corners=False, antialias)
  → clamp_min(-1000)`, then a **single per-volume z-score** over the whole
  stacked slice tensor (`_zscore_per_image` reduces over the full 3D stack —
  it is *not* per-slice). The ÷32767 cancels under the z-score and the clamp
  is inert on the scaled values; both are kept for bit-faithfulness (cos≈1.0
  vs the cleaned-up equivalent). The 3D rule is the authors' own: "for
  slice-based models, features are averaged on the volume." Orient to **PLS**
  so the last axis is axial and each slice is indexed `[P, L]` ("PL for axial"
  in the processor docstring); the model gives the 768-d layernormed CLS
  (`pooler_output`) per slice, mean-pooled over slices. Slower than the 3D
  models (N_slices forwards/volume) — slices are batched (`slice_batch_size`).
- **VoxelFM**: DINOv2-style 3D ViT (DINO CLS + iBOT masked-patch
  self-distillation), no text tower → kNN / linear-probe only (peer = CT-FM,
  Curia-2). **The only extractor that bypasses MONAI**: its canonical
  inference (`dinov2/inference/distributed.py::worker`) uses the author's own
  torch helpers and an *adaptive* per-volume isotropic resample
  (`ts = max(min(native_spacing), 0.75)` mm), which a fixed `Spacingd` cannot
  reproduce — so `embed()` calls `crop_volume`/`resize_isotropic`/
  `resize_max_patches`/`patch_crop` directly. Worker is IO-only; orientation +
  spacing are recovered in `embed()` from the `meta` (`spacing` scalar for
  RadChest, already [I,P,L] = axial-first; LPS `affine` for CT-RATE →
  per-axis spacing from column norms, through-plane axis (max |world-Z|) moved
  to dim0, mirroring the NIfTI loader's `x y z -> z y x` reversal). HU clip is
  **[-1000, 1900]** (the paper range + the NIfTI/MHD loader; note the upstream
  *DICOM* loader inconsistently clips to [-1000, 1000]). Normalization is a
  **global dataset z-score** read from the shipped `config.yaml`
  (`datasets[0].norm.{mean,std}`) — *not* per-volume like Curia-2/FlexiCT.
  Embedding = **mean-pooled patch tokens** (`x_norm_patchtokens`, 864-d), which
  is the author-canonical kNN feature (their CT-RATE head-to-head uses
  `select_feature="patch", pooling="avg"`, the `output/dino/patch` runs);
  `feature="cls"` selects the CLS token instead. `embed_dim`/`patch_size`/
  `fmean`/`fstd` are read from the config, never hardcoded. Authors run fp32
  (`use_amp=false` default). VoxelFM also evaluates with **kNN** itself
  (`NearestNeighbors(k=5, metric="cosine")`) — the same protocol as this repo.

---

## Metric + evaluation guidance

Not model-specific, but worth preserving:

1. **PR-AUC, not ROC-AUC** for imbalanced multilabel. Always print class
   prevalence next to PR-AUC (random baseline = prevalence, not 0.5).
2. **Pool all out-of-fold predictions** into one `[N, C]` matrix; compute
   metrics once on the pool. Don't average per-fold point estimates.
3. **Bootstrap over samples** (n_boot=1000) for 95% CIs — captures
   test-set-size uncertainty, the thing you actually care about.
4. **Paired bootstrap** for model comparisons: same resample indices for
   both models on the same test set → tighter CI of the *difference*.
   Non-overlapping marginal CIs → `p < 0.05`; paired CI excluding zero →
   `p < 0.05` (usually much stronger power).

