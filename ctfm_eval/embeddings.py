from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectre import MODEL_CONFIGS, SpectreImageFeatureExtractor
from spectre_wrapper import SpectrePatchConfig, SpectreVolumeWrapper


# ----- Shared helpers -------------------------------------------------------

def _torchio_preprocess_worker(transform):
    """Dict->dict worker callable for extractors whose authors use torchio
    (e.g. COLIPRI). Wraps 'image' as a tio.ScalarImage, runs the supplied
    torchio Compose, writes the preprocessed tensor back.

    Two input shapes are supported:
      - RadChest: image is plain [1, Z, Y, X] HU tensor. Sample has
        `spacing` scalar. Permute to (1, X, Y, Z) and use isotropic-diag
        LPS affine.
      - CT-RATE: image is plain [1, X, Y, Z] HU tensor (from
        `_LoadCTRateNii`) and `affine` is a precomputed 4×4 LPS matrix
        in nibabel (X, Y, Z) voxel order. No permute, use the supplied
        affine directly.

    The CT-RATE branch is only entered when `sample["affine"]` is present;
    RadChest batches do not carry that key, so the RadChest code path is
    bit-identical to the pre-CT-RATE implementation (verified by
    `scripts/test_radchest_identity.py`).
    """

    def _fn(sample: dict) -> dict:
        import torchio as tio
        img = sample["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if img.ndim != 4 or img.shape[0] != 1:
            raise ValueError(f"expected [1, *, *, *], got {tuple(img.shape)}")
        aff = sample.get("affine")
        if aff is not None:
            # CT-RATE path: tensor is [1, X, Y, Z], affine is LPS (X,Y,Z).
            vol = img.contiguous().float()
            affine = np.asarray(aff, dtype=np.float32)
        else:
            # RadChest path (unchanged): tensor is [1, Z, Y, X]; permute to
            # torchio's (1, X, Y, Z) and build an isotropic-diag LPS affine.
            sp = float(sample.get("spacing", 1.0))
            vol = img.permute(0, 3, 2, 1).contiguous().float()
            affine = np.diag([sp, sp, sp, 1.0]).astype(np.float32)
        tio_img = tio.ScalarImage(tensor=vol, affine=affine)
        tio_img = transform(tio_img)
        sample["image"] = tio_img.data.contiguous()      # [1, X', Y', Z']
        return sample
    return _fn


def _radchest_lps_affine_zyx(spacing: float) -> torch.Tensor:
    """4x4 affine that correctly describes a RadChest-stored [Z, Y, X] volume
    in LPS world coords. Verified visually on a sample (exp/ctssg_orient_check):
    NPZ axis 0 = I, axis 1 = P, axis 2 = L. In LPS (+X=L, +Y=P, +Z=S):

        world[X,Y,Z] = A @ [v0, v1, v2, 1]
        A = [[ 0,  0, +s, 0],   # world X (L)  from voxel 2
             [ 0, +s,  0, 0],   # world Y (P)  from voxel 1
             [-s,  0,  0, 0],   # world Z (S)  from -voxel 0 (axis 0 goes I)
             [ 0,  0,  0, 1]]
    """
    s = float(spacing)
    return torch.tensor(
        [[0, 0, s, 0], [0, s, 0, 0], [-s, 0, 0, 0], [0, 0, 0, 1]],
        dtype=torch.float64,
    )


def _monai_preprocess_worker(transform):
    """Dict->dict worker callable for RadChest + any MONAI pipeline.

    Wraps the raw [1, Z, Y, X] volume as a MetaTensor with a correct LPS
    affine (matching RadChest's stored (I, P, L) axis directions) and declares
    space=LPS so MONAI's Orientationd/Spacingd reorient from the true source
    orientation rather than default RAS. Runs the provided transform and
    returns the preprocessed tensor."""

    def _fn(sample: dict) -> dict:
        from monai.data.meta_tensor import MetaTensor
        from monai.utils.enums import SpaceKeys

        img = sample["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if img.ndim != 4 or img.shape[0] != 1:
            raise ValueError(f"expected [1, Z, Y, X], got {tuple(img.shape)}")
        sp = float(sample.get("spacing", 1.0))
        mt = MetaTensor(
            img.float(),
            affine=_radchest_lps_affine_zyx(sp),
            meta={"space": SpaceKeys.LPS},
        )
        out = transform({"image": mt})
        sample["image"] = out["image"].as_tensor().contiguous()
        return sample
    return _fn


# ----- Main-loop metadata unwrapper ----------------------------------------

_RESERVED_BATCH_KEYS = {"image", "label", "id"}


def _sample_meta(batch: dict) -> dict:
    """Unpack a batch_size=1 DataLoader batch into scalar meta for extractors."""
    out: dict = {}
    for k, v in batch.items():
        if k in _RESERVED_BATCH_KEYS:
            continue
        if isinstance(v, list) and v:
            out[k] = v[0]
        elif isinstance(v, torch.Tensor):
            out[k] = v[0].item() if v.numel() == 1 else v[0]
        else:
            out[k] = v
    return out


# ----- Embedding cache ------------------------------------------------------

@dataclass(slots=True)
class EmbeddingBatch:
    """Cached embeddings + labels + ids + provenance.

    `groups` (optional) carries a per-sample group key (e.g. patient_id for a
    cohort with multiple scans per patient) so downstream evaluation can do
    GroupKFold without re-reading the source manifest. None for caches that
    don't need grouping (RadChest / CT-RATE).
    """
    embeddings: torch.Tensor       # [N, D]
    labels: torch.Tensor           # [N, C]
    ids: list[str]                 # length N
    label_columns: list[str]       # length C
    model_name: str
    dataset_name: str
    groups: list[str] | None = None  # length N when set

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embeddings": self.embeddings,
            "labels": self.labels,
            "ids": self.ids,
            "label_columns": self.label_columns,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "groups": self.groups,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingBatch":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            embeddings=d["embeddings"],
            labels=d["labels"],
            ids=d["ids"],
            label_columns=d["label_columns"],
            model_name=d["model_name"],
            dataset_name=d["dataset_name"],
            groups=d.get("groups"),
        )

    def subset_labels(self, keep: list[str]) -> "EmbeddingBatch":
        """Return a new batch with labels/label_columns restricted to `keep`
        (in the given order). Raises if any requested column is missing."""
        idx_map = {c: i for i, c in enumerate(self.label_columns)}
        missing = [c for c in keep if c not in idx_map]
        if missing:
            raise KeyError(
                f"label columns missing from cache {self.model_name!r}: {missing}"
            )
        idx = torch.tensor([idx_map[c] for c in keep], dtype=torch.long)
        return EmbeddingBatch(
            embeddings=self.embeddings,
            labels=self.labels.index_select(1, idx),
            ids=self.ids,
            label_columns=list(keep),
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            groups=self.groups,
        )


# ----- Extractor protocol ---------------------------------------------------

class EmbeddingExtractor(Protocol):
    """Extracts a fixed-size embedding from a preprocessed volume tensor.

    Split across threads:
      - `worker_preprocess` (optional): dict->dict callable that runs in a
        DataLoader worker. Takes raw loaded volumes to model-ready tensors.
      - `embed`: runs in the main thread on GPU; only the forward pass.
    """

    name: str
    embedding_dim: int
    worker_preprocess: Callable[[dict], dict] | None

    def to(self, device: torch.device) -> "EmbeddingExtractor": ...
    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor: ...


# ----- SPECTRE --------------------------------------------------------------

class SpectreExtractor:
    """SPECTRE feature extractor. Worker-side preprocessing mirrors the
    authors' canonical SSL recipe (MONAI: intensity [-1000,1000]->[0,1],
    orient RAS, resample to 0.75x0.75x1.5 mm, crop/pad to 384x384x256). Main
    thread only does the model forward, taking the CLS token of the feature
    combiner (1080-d for spectre-large)."""

    def __init__(
        self,
        model_config: str = "spectre-large-pretrained",
        patch_size: tuple[int, int, int] = (128, 128, 64),
        grid_size: tuple[int, int, int] = (3, 3, 4),
        target_spacing: tuple[float, float, float] = (0.75, 0.75, 1.5),
        target_shape: tuple[int, int, int] = (384, 384, 256),
        use_amp: bool = True,
        cls_plus_mean: bool = False,
    ) -> None:
        if model_config not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown SPECTRE config {model_config!r}. Available: {list(MODEL_CONFIGS)}"
            )
        self.name = f"spectre:{model_config}"
        cfg = MODEL_CONFIGS[model_config]
        model = SpectreImageFeatureExtractor.from_config(cfg).eval()
        self.wrapper = SpectreVolumeWrapper(
            model=model,
            config=SpectrePatchConfig(patch_size=patch_size, grid_size=grid_size),
        )
        self.use_amp = use_amp
        self._cls_plus_mean = bool(cls_plus_mean)
        self._device = torch.device("cpu")

        from monai.transforms import (
            Compose,
            CropForegroundd,  # noqa: F401  (not used here; left for future clarity)
            Orientationd,
            ResizeWithPadOrCropd,
            ScaleIntensityRanged,
            Spacingd,
        )
        self._transform = Compose([
            ScaleIntensityRanged(
                keys="image", a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True,
            ),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=target_spacing, mode="bilinear"),
            ResizeWithPadOrCropd(keys="image", spatial_size=target_shape, value=0.0),
        ])
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "SpectreExtractor":
        self._device = device
        self.wrapper.model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        base = getattr(self.wrapper.model, "embed_dim", -1)
        return 2 * base if self._cls_plus_mean and base > 0 else base

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        feats = self.wrapper.infer_from_volume(
            volume, device=self._device, use_amp=self.use_amp,
        )
        if self._cls_plus_mean:
            # SigLIP-head convention: concat(CLS, mean(patches)).
            out = torch.cat([feats[:, 0, :], feats[:, 1:, :].mean(dim=1)], dim=-1)
            return out.squeeze(0).detach().cpu().float()
        return feats[:, 0, :].squeeze(0).detach().cpu().float()


# ----- COLIPRI --------------------------------------------------------------

class ColipriExtractor:
    """Microsoft COLIPRI-CRM (3D chest-CT CLIP). Image branch only.

    Uses the *authentic* shipped torchio Compose (ToOrientation SAR -> Resample
    2mm -> Clamp [-1000,1000] -> RescaleIntensity [-1,1] -> CropOrPad 192^3).
    Preprocessing runs inside DataLoader workers; main thread does just the
    model forward.
    """

    def __init__(
        self,
        pool: bool = True,
        project: bool = True,
        normalize: bool = True,
        input_size: int | None = None,       # None -> COLIPRI default (192)
        spacing_mm: float | None = None,     # None -> COLIPRI default (2.0)
    ) -> None:
        from colipri import get_model, get_processor

        proc_overrides: dict = {}
        if input_size is not None:
            proc_overrides["input_size"] = input_size
        if spacing_mm is not None:
            proc_overrides["spacing"] = spacing_mm

        self.name = "colipri-crm"
        self._model = get_model(pretrained=True, image_only=True).eval()
        self._processor = get_processor(image_only=True, **proc_overrides)
        self._device = torch.device("cpu")
        self._pool = pool
        self._project = project
        self._normalize = normalize

        # Authoritative worker preprocess: the full shipped COLIPRI Compose.
        self.worker_preprocess = _torchio_preprocess_worker(self._processor._image_transform)

    def to(self, device: torch.device) -> "ColipriExtractor":
        self._device = device
        self._model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return 768

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        # volume: [1, D1, D2, D3] already preprocessed to 192^3 (float32).
        # COLIPRI's encode_image expects [B, 1, D1, D2, D3].
        batch = volume.unsqueeze(0).to(self._device, non_blocking=True)
        emb = self._model.encode_image(
            batch, pool=self._pool, project=self._project, normalize=self._normalize,
        )
        return emb.squeeze(0).detach().cpu().float()


# ----- CT-SSG ---------------------------------------------------------------

def _ctssg_compose(
    input_size: int,
    slice_count: int,
    target_spacing: tuple[float, float, float],
    hu_range: tuple[float, float],
    imagenet_mean: float,
):
    """MONAI Compose mirroring the CT-SSG paper's CT-scan processing exactly."""
    from monai.transforms import (
        Compose,
        Lambdad,
        Orientationd,
        ResizeWithPadOrCropd,
        ScaleIntensityRanged,
        Spacingd,
    )
    return Compose([
        Orientationd(keys="image", axcodes="SLP"),
        Spacingd(keys="image", pixdim=target_spacing, mode="trilinear"),
        ScaleIntensityRanged(
            keys="image",
            a_min=hu_range[0], a_max=hu_range[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Lambdad(keys="image", func=lambda x: x - imagenet_mean),
        ResizeWithPadOrCropd(
            keys="image",
            spatial_size=(slice_count, input_size, input_size),
            value=-imagenet_mean,                          # pad with [0,1]->0 = "air"
        ),
    ])


class CTSSGExtractor:
    """CT-SSG (Di Piazza et al. 2026) — supervised 3D CT graph model.

    We extract z̄, the 512-d mean-pooled post-ChebConv feature from
    OperatorBlocks — it's the volume-level representation the paper uses for
    t-SNE and for comparison against other foundation models at dim=512 in
    Table 14. Classifier head is ignored.

    Weights come from HF 'theodpzz/ct-ssg' (downloaded at first construction).
    Repo source (cloned into third_party/ct-ssg) provides the model class.
    """

    def __init__(
        self,
        repo_path: str | Path = "third_party/ct-ssg",
        input_size: int = 480,
        slice_count: int = 240,
        target_spacing: tuple[float, float, float] = (1.5, 0.75, 0.75),
        hu_range: tuple[float, float] = (-1000.0, 200.0),
        imagenet_mean: float = 0.449,
        n_outputs: int = 18,
        embed_dim: int = 512,
        hidden_size: tuple[int, ...] = (512,),
        window_size: tuple[int, ...] = (16,),
        nb_triplets: int = 80,
        K: int = 3,
        dropout: float = 0.2,
        spacing_z: float = 1.5,
        weights_revision: str | None = "b3b841803bbd21e428710839758e31ab56039361",
    ) -> None:
        import sys as _sys
        from argparse import Namespace
        from huggingface_hub import snapshot_download

        repo_path = Path(repo_path)
        if not repo_path.is_absolute():
            # Resolve against repo root (two levels up from this file).
            repo_path = (Path(__file__).resolve().parents[1] / repo_path).resolve()
        if not (repo_path / "src" / "model.py").exists():
            raise FileNotFoundError(
                f"CT-SSG repo not found at {repo_path}. Clone from "
                f"https://github.com/theodpzz/ct-ssg"
            )
        if str(repo_path) not in _sys.path:
            _sys.path.insert(0, str(repo_path))

        from src.model import Model  # type: ignore[import-not-found]

        args = Namespace(
            n_outputs=n_outputs, embed_dim=embed_dim,
            depth=len(hidden_size), hidden_size=list(hidden_size),
            window_size=list(window_size), nb_triplets=nb_triplets,
            K=K, path_resnet=None, dropout=dropout, bias=True,
            spacing_z=spacing_z, device="cpu",
        )
        model = Model(args).eval()

        # Pin the weights revision: the HF repo's HEAD moved past the snapshot
        # that ships `model_state_dict.pt` (the new tip caches only README), so
        # the default refs/main resolution now 404s the weights. `b3b841…` is
        # the revision the cached RadChest/CT-RATE CT-SSG embeddings were built
        # from — pinning it keeps the weights present AND bit-identical.
        weights_dir = Path(snapshot_download("theodpzz/ct-ssg", revision=weights_revision))
        ckpt_path = weights_dir / "model_state_dict.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"CT-SSG state_dict mismatch: missing={missing} unexpected={unexpected}"
            )

        self.name = "ctssg"
        self._model = model
        self._device = torch.device("cpu")
        self._input_size = input_size
        self._slice_count = slice_count

        self._transform = _ctssg_compose(
            input_size=input_size,
            slice_count=slice_count,
            target_spacing=target_spacing,
            hu_range=hu_range,
            imagenet_mean=imagenet_mean,
        )
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "CTSSGExtractor":
        self._device = device
        self._model.to(device)
        # Edges module stores edge_index / edge_weight tensors that aren't in
        # state_dict and default to CPU; move them with the model.
        edges = self._model.blocks.edges
        edges.edges_index  = [t.to(device) for t in edges.edges_index]
        edges.edges_weight = [t.to(device) for t in edges.edges_weight]
        # OperatorBlocks stores a `device` attribute (read in forward_operator).
        self._model.blocks.device = device
        return self

    @property
    def embedding_dim(self) -> int:
        return 512

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        # volume: [1, 240, 480, 480], float32, values in [-0.449, 0.551].
        if tuple(volume.shape) != (1, self._slice_count, self._input_size, self._input_size):
            raise ValueError(
                f"CT-SSG expects [1, {self._slice_count}, {self._input_size}, "
                f"{self._input_size}]; got {tuple(volume.shape)}"
            )
        batch = volume.unsqueeze(0).to(self._device, non_blocking=True)  # [1, 1, 240, 480, 480]
        x = self._model.stem(batch)                                      # [1, 80, 512]
        z_bar = self._model.blocks(x)                                    # [1, 512]
        return z_bar.squeeze(0).detach().cpu().float()


# ----- CT-CLIP --------------------------------------------------------------

def _ctclip_compose(
    input_shape: tuple[int, int, int] = (240, 480, 480),   # (S, L, P)
    target_spacing: tuple[float, float, float] = (1.5, 0.75, 0.75),
    hu_range: tuple[float, float] = (-1000.0, 1000.0),
    pad_value: float = -1.0,
):
    """MONAI pipeline matching CT-CLIP's official inference (data_inference_nii.py).

    Their recipe: clip HU to [-1000, 1000] then divide by 1000 → [-1, 1], resample
    to (1.5, 0.75, 0.75) mm with D/H/W order, center-crop/pad to (240, 480, 480)
    (which in the author's axis convention corresponds to SLP voxel order), pad
    with -1. We keep that identical here but use MONAI's affine-aware transforms
    so we can guarantee source orientation.
    """
    from monai.transforms import (
        Compose,
        Lambdad,
        Orientationd,
        ResizeWithPadOrCropd,
        ScaleIntensityRanged,
        Spacingd,
    )
    return Compose([
        Orientationd(keys="image", axcodes="SLP"),
        Spacingd(keys="image", pixdim=target_spacing, mode="trilinear"),
        ScaleIntensityRanged(
            keys="image",
            a_min=hu_range[0], a_max=hu_range[1],
            b_min=-1.0, b_max=1.0, clip=True,
        ),
        ResizeWithPadOrCropd(
            keys="image", spatial_size=input_shape, value=pad_value,
        ),
    ])


class CTCLIPExtractor:
    """CT-CLIP (Hamamci et al. 2024) — 3D CLIP on CT-RATE. Image branch only.

    CTViT visual tower produces encoded tokens; we follow the TumorImagingBench
    reduction (adaptive_avg_pool3d over spatial tokens, flatten) to get a single
    volume-level feature — same reduction the CTViT's own inference code uses
    when `return_encoded_tokens=True`.

    Weights: HF dataset `ibrahimhamamci/CT-RATE`, file
    `models/CT-CLIP-Related/CT-CLIP_v2.pt`. We strip the `visual_transformer.`
    prefix to load only the visual tower.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (240, 480, 480),
        target_spacing: tuple[float, float, float] = (1.5, 0.75, 0.75),
        hu_range: tuple[float, float] = (-1000.0, 1000.0),
        image_size: int = 480,
        patch_size: int = 20,
        temporal_patch_size: int = 10,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        dim: int = 512,
        dim_head: int = 32,
        heads: int = 8,
        codebook_size: int = 8192,
        project: bool = False,
    ) -> None:
        from transformer_maskgit import CTViT
        from huggingface_hub import hf_hub_download

        model = CTViT(
            dim=dim, codebook_size=codebook_size,
            image_size=image_size, patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_depth=spatial_depth, temporal_depth=temporal_depth,
            dim_head=dim_head, heads=heads,
        ).eval()

        weights_path = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            repo_type="dataset",
            filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        vit_state = {
            k.replace("visual_transformer.", ""): v
            for k, v in ckpt.items()
            if k.startswith("visual_transformer.")
        }
        missing, unexpected = model.load_state_dict(vit_state, strict=False)
        if unexpected:
            raise RuntimeError(f"CT-CLIP unexpected keys: {unexpected[:5]}...")

        self.name = "ct-clip"
        self._model = model
        self._device = torch.device("cpu")
        self._input_shape = input_shape
        self._project = bool(project)

        if self._project:
            w = ckpt.get("to_visual_latent.weight")
            b = ckpt.get("to_visual_latent.bias")
            if w is None:
                raise RuntimeError("to_visual_latent missing from CT-CLIP_v2.pt")
            proj = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
            with torch.no_grad():
                proj.weight.copy_(w)
                if b is not None:
                    proj.bias.copy_(b)
            self._to_visual_latent = proj.eval()
            self.name = "ctclip-zs"

        self._transform = _ctclip_compose(
            input_shape=input_shape,
            target_spacing=target_spacing,
            hu_range=hu_range,
        )
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "CTCLIPExtractor":
        self._device = device
        self._model.to(device)
        if self._project:
            self._to_visual_latent.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return 512            # CTViT 'dim' parameter

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        if tuple(volume.shape) != (1,) + self._input_shape:
            raise ValueError(
                f"CT-CLIP expects [1, {self._input_shape[0]}, "
                f"{self._input_shape[1]}, {self._input_shape[2]}]; got {tuple(volume.shape)}"
            )
        batch = volume.unsqueeze(0).to(self._device, non_blocking=True)   # [1,1,240,480,480]
        tokens = self._model(batch, return_encoded_tokens=True)           # [1, t, h, w, dim]
        if self._project:
            # CT-CLIP-native encode: mean over temporal, flatten h*w*dim, Linear → 512.
            enc = tokens.mean(dim=1).flatten(start_dim=1)                 # [1, 294912]
            emb = self._to_visual_latent(enc)
        else:
            tokens = tokens.permute(0, 4, 1, 2, 3)                        # [1, dim, t, h, w]
            emb = torch.nn.functional.adaptive_avg_pool3d(tokens, 1).flatten(start_dim=1)
        return emb.squeeze(0).detach().cpu().float()


# ----- Merlin ---------------------------------------------------------------

def _merlin_compose(
    input_shape: tuple[int, int, int] = (224, 224, 160),   # (R, A, S)
    target_spacing: tuple[float, float, float] = (1.5, 1.5, 3.0),
    hu_range: tuple[float, float] = (-1000.0, 1000.0),
):
    """Mirror Merlin's official `merlin.data.monai_transforms.ImageTransforms`."""
    from monai.transforms import (
        Compose,
        CenterSpatialCropd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
    )
    return Compose([
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys="image",
            a_min=hu_range[0], a_max=hu_range[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        SpatialPadd(keys="image", spatial_size=input_shape),
        CenterSpatialCropd(keys="image", roi_size=input_shape),
    ])


class MerlinExtractor:
    """Merlin (Blankemeier, Kumar et al., Nature 2026).

    Image-only branch: `Merlin(ImageEmbedding=True)` returns an image embedding
    via `outputs[0]` on forward. Weights download happens at construction via
    the package.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 160),
        target_spacing: tuple[float, float, float] = (1.5, 1.5, 3.0),
        hu_range: tuple[float, float] = (-1000.0, 1000.0),
    ) -> None:
        from merlin import Merlin

        self.name = "merlin"
        self._model = Merlin(ImageEmbedding=True).eval()
        self._device = torch.device("cpu")
        self._input_shape = input_shape

        self._transform = _merlin_compose(
            input_shape=input_shape,
            target_spacing=target_spacing,
            hu_range=hu_range,
        )
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "MerlinExtractor":
        self._device = device
        self._model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return -1             # inferred from forward; ResNet152-3D embedding

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        if tuple(volume.shape) != (1,) + self._input_shape:
            raise ValueError(
                f"Merlin expects [1, {self._input_shape[0]}, "
                f"{self._input_shape[1]}, {self._input_shape[2]}]; got {tuple(volume.shape)}"
            )
        batch = volume.unsqueeze(0).to(self._device, non_blocking=True)   # [1,1,R,A,S]
        outputs = self._model(batch)
        emb = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return emb.squeeze(0).detach().cpu().float().flatten()


# ----- CT-FM ----------------------------------------------------------------

def _ctfm_compose(
    target_spacing: tuple[float, float, float] = (3.0, 1.0, 1.0),
    hu_range: tuple[float, float] = (-1024.0, 2048.0),
):
    """Mirror CT-FM's official `scripts/feature_extractor.py` transforms."""
    from monai.transforms import (
        Compose,
        CropForegroundd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )
    return Compose([
        Orientationd(keys="image", axcodes="SPL"),
        Spacingd(keys="image", pixdim=target_spacing, mode="bilinear"),
        CropForegroundd(keys="image", source_key="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=hu_range[0], a_max=hu_range[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ])


class CTFMExtractor:
    """CT-FM (Pai et al. 2025) — SSL SegResEncoder on 148k CTs.

    The official feature extractor (`CT-FM/scripts/feature_extractor.py`) runs
    a sliding window over the whole volume after MONAI-based preprocessing,
    producing one patch-level feature per window. We aggregate to a single
    volume-level vector by mean-pooling across patches.

    Weights: HF `surajpaib/CT-FM-SegResNet`, file `pretrained_segresnet.torch`.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (24, 128, 128),
        overlap: float = 0.0,
        batch_size: int = 16,
        target_spacing: tuple[float, float, float] = (3.0, 1.0, 1.0),
        hu_range: tuple[float, float] = (-1024.0, 2048.0),
    ) -> None:
        from huggingface_hub import hf_hub_download
        from monai.networks.nets.segresnet_ds import SegResEncoder

        model = SegResEncoder(
            blocks_down=(1, 2, 2, 4, 4),
            head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(
                x[-1], 1
            ).flatten(start_dim=1),
        ).eval()

        weights_path = hf_hub_download(
            repo_id="surajpaib/CT-FM-SegResNet",
            filename="pretrained_segresnet.torch",
        )
        weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        weights = {k.replace("encoder.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=False)

        self.name = "ct-fm"
        self._model = model
        self._device = torch.device("cpu")
        self._patch_size = patch_size
        self._overlap = overlap
        self._batch_size = batch_size

        self._transform = _ctfm_compose(target_spacing=target_spacing, hu_range=hu_range)
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "CTFMExtractor":
        self._device = device
        self._model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return -1             # inferred from forward; 512 for default SegResEncoder

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        import monai

        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"CT-FM expects [1, S, P, L]; got {tuple(volume.shape)}")
        vol = volume.unsqueeze(0).to(self._device, non_blocking=True)       # [1, 1, S, P, L]

        # Sliding-window split -> iterator of (patch, coords). We only need the patches.
        splitter = monai.inferers.SlidingWindowSplitter(self._patch_size, self._overlap)

        feats: list[torch.Tensor] = []
        patch_batch: list[torch.Tensor] = []
        for patch, _ in splitter(vol):
            patch_batch.append(patch.squeeze(0))                            # [1, pD, pH, pW]
            if len(patch_batch) >= self._batch_size:
                feats.append(self._run(patch_batch))
                patch_batch.clear()
        if patch_batch:
            feats.append(self._run(patch_batch))

        all_feats = torch.cat(feats, dim=0)                                 # [n_patches, D]
        return all_feats.mean(dim=0).detach().cpu().float()

    def _run(self, patch_batch: list[torch.Tensor]) -> torch.Tensor:
        batch = torch.stack(patch_batch, dim=0).to(self._device, non_blocking=True)
        return self._model(batch).detach()


# ----- Pillar-0 -------------------------------------------------------------

def _pillar0_compose(
    target_spacing: tuple[float, float, float] = (1.25, 1.25, 1.25),
    input_shape: tuple[int, int, int] = (256, 256, 256),    # (S, L, P)
):
    """CT-chest preprocessing matching RAVE's `configs/ct_chest.yaml`.

    Steps (authors' pipeline in `vision_engine/processing/ct_processor.py`):
      1. Resample to (1.25, 1.25, 1.25) mm isotropic (SITK linear).
      2. Slice selection: middle 256 slices if Z > 256, else keep.
      3. Center crop/pad H,W to 256 with pad_value = volume.min().

    Downstream, pillar-finetune's CSVDataset symmetrically pads/crops the Z
    dim to `num_images=256`. We fold that into the same crop/pad step here.
    We also orient to 'LPS' explicitly (the DICOM default RAVE preserves).
    """
    from monai.transforms import (
        Compose,
        Lambdad,
        Orientationd,
        ResizeWithPadOrCropd,
        Spacingd,
    )
    return Compose([
        Orientationd(keys="image", axcodes="LPS"),
        Spacingd(keys="image", pixdim=target_spacing, mode="trilinear"),
        # ResizeWithPadOrCrop pads with a constant; use image min (~-1000 HU for air).
        Lambdad(keys="image", func=lambda x: x),        # no-op, just a landing zone
        ResizeWithPadOrCropd(
            keys="image", spatial_size=input_shape, value=-1000.0,
        ),
    ])


class Pillar0Extractor:
    """Pillar-0 ChestCT (Agrawal et al. 2025, YalaLab).

    Multi-windowing CT CLIP. Input contract (from RAVE `ct_chest.yaml` +
    HF config.json):
      - Shape (256, 256, 256) at (1.25, 1.25, 1.25) mm isotropic, orientation
        LPS.
      - 11 CT windows stacked as channels: lung, mediastinum, abdomen,
        liver, bone, brain, subdural, stroke, temporal_bone, soft_tissue,
        minmax (via `rve.apply_windowing(vol, 'all', 'CT')`).

    We run the visual tower (`model.model.visual`) which does multi-scale
    Atlas pooling internally and returns a 1152-d pooled volume feature.

    Weights: HF `YalaLab/Pillar0-ChestCT` (`model.safetensors`, `model.` prefix
    stripped on load because the model's own state_dict uses `visual.`/`text.`
    keys directly).
    """

    def __init__(
        self,
        target_spacing: tuple[float, float, float] = (1.25, 1.25, 1.25),
        input_shape: tuple[int, int, int] = (256, 256, 256),
        modality: str = "chest_ct",
        hf_repo: str = "YalaLab/Pillar0-ChestCT",
    ) -> None:
        import os
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        # Pillar-0's custom modeling code calls `.item()` on torch tensors
        # during init. The transformers meta-device init path breaks this,
        # so we force CPU during `from_config` and load weights manually.
        prev_default_device = torch.get_default_device() if hasattr(torch, "get_default_device") else None
        torch.set_default_device("cpu")
        try:
            cfg = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
            model = AutoModel.from_config(cfg, trust_remote_code=True)
        finally:
            if prev_default_device is not None:
                torch.set_default_device(prev_default_device)

        weights_path = hf_hub_download(hf_repo, "model.safetensors")
        sd = load_file(weights_path, device="cpu")
        # Safetensors keys have an extra leading "model." that the in-memory
        # CustomTextCLIP doesn't use.
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Pillar-0 state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}\n"
                f"first missing: {missing[:3]}\nfirst unexpected: {unexpected[:3]}"
            )
        model.eval()

        self.name = "pillar0-chestct"
        self._model = model
        self._visual = model.model.visual
        self._modality = modality
        self._device = torch.device("cpu")
        self._input_shape = input_shape

        self._transform = _pillar0_compose(target_spacing=target_spacing, input_shape=input_shape)
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "Pillar0Extractor":
        self._device = device
        self._model.to(device)
        self._visual = self._model.model.visual   # re-bind (in case .to() reassigned)
        return self

    @property
    def embedding_dim(self) -> int:
        return 1152

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        import rve                                                      # type: ignore[import-not-found]

        # volume: [1, 256, 256, 256] in raw HU (float).
        if tuple(volume.shape) != (1,) + self._input_shape:
            raise ValueError(
                f"Pillar-0 expects [1, {self._input_shape[0]}, "
                f"{self._input_shape[1]}, {self._input_shape[2]}]; got {tuple(volume.shape)}"
            )
        vol = volume.squeeze(0).cpu()                                    # [D, H, W] HU

        # 11 CT windows as channels (authors' canonical 'all' setting).
        multi = rve.apply_windowing(vol, "all", "CT")                    # [11, D, H, W]
        if not torch.is_tensor(multi):
            multi = torch.as_tensor(multi)
        batch = multi.unsqueeze(0).to(self._device, non_blocking=True)   # [1, 11, D, H, W]

        out = self._visual({self._modality: batch.float()}, {"anatomy": [self._modality]})
        # visual() returns a pooled volume-level feature [B, 1152].
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(0).detach().cpu().float()


# ----- Curia-2 --------------------------------------------------------------

def _curia2_orient_compose(axcodes: str = "PLS"):
    """Orientation-only MONAI compose for Curia-2.

    Curia-2 is a 2D-slice DINOv2 model, so the only *volume-level* step is
    putting axial slices on the last axis in the author-prescribed in-plane
    order. `curia_image_processor.py` states the input must be in "PL for
    axial" — i.e. each 2D axial slice indexed [P, L]. We orient the volume to
    (P, L, S); `embed()` then iterates the last (S = axial / through-plane)
    axis, so each slice is [P, L]. No spacing resample and no HU windowing
    happen here — the per-slice resize / clamp / z-score run in `embed()`,
    mirroring `CuriaImageProcessor` exactly.

    This compose plugs into the same two-path plumbing as the other MONAI
    extractors: `worker_preprocess` for RadChest (runs in DataLoader workers)
    and `gpu_compose` for CT-RATE (runs on-device in the main loop).
    """
    from monai.transforms import Compose, Orientationd
    return Compose([Orientationd(keys="image", axcodes=axcodes)])


class Curia2Extractor:
    """Curia-2 B (Raidium, HF `raidium/curia-2`) — DINOv2 SSL vision FM for
    CT **and** MRI. No text tower → kNN / linear-probe track only (peer =
    CT-FM). The first *slice-based* extractor in the repo.

    The released checkpoint is a stock HF `Dinov2Model` (hidden_size=768,
    image_size=512, patch_size=16, num_channels=1, apply_layernorm=true,
    swiglu ffn, layerscale). We load it directly with `AutoModel` (no custom
    modeling code; only the *image processor* carries remote code, which we
    replicate instead of importing).

    Per the authors' 3D rule ("for slice-based models, features are averaged
    on the volume for all 3D tasks"), `embed()`:
      1. receives the volume already oriented to (P, L, S);
      2. takes every axial slice along the last axis;
      3. preprocesses each slice op-for-op like `CuriaImageProcessor`
         (curia_image_processor.py, documented numpy path):
           ``to(int16) -> convert_image_dtype(float32)  [= ÷32767]
             -> F.interpolate(512, bicubic, align_corners=False, antialias=True)
             -> clamp_min(-1000)  [clip_below_air]``
         then **one per-volume z-score** over the whole stacked slice tensor
         (eps=1e-6; center-only if std<eps) — note `_zscore_per_image` in the
         processor reduces over the *entire* 3D stack, i.e. a single global
         mean/std, not per-slice;
      4. runs the encoder (slices batched for GPU throughput), takes the
         768-d layernormed CLS (`pooler_output`) per slice, and **mean-pools
         over all slices** → one [768] volume embedding.

    The ÷32767 from `convert_image_dtype` cancels under the z-score and the
    clamp is inert on the scaled values, but both are kept so the pipeline is
    bit-faithful to the author's processor rather than a "cleaned-up"
    equivalent (verified cos≈1.0 either way).
    """

    def __init__(
        self,
        hf_repo: str = "raidium/curia-2",
        axcodes: str = "PLS",
        crop_size: int = 512,
        clip_below_air: bool = True,
        eps: float = 1e-6,
        slice_batch_size: int = 96,
        use_amp: bool = True,
    ) -> None:
        from transformers import AutoModel

        self.name = "curia2"
        self._model = AutoModel.from_pretrained(hf_repo).eval()
        self._device = torch.device("cpu")
        self._crop = int(crop_size)
        self._clip = bool(clip_below_air)
        self._eps = float(eps)
        self._bs = int(slice_batch_size)
        self._use_amp = bool(use_amp)

        self._transform = _curia2_orient_compose(axcodes=axcodes)
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "Curia2Extractor":
        self._device = device
        self._model.to(device)
        if device.type == "cuda":
            # Slice forwards are a fixed 512×512 single-channel shape, so let
            # cuDNN autotune the patch-embed conv once and reuse the plan.
            torch.backends.cudnn.benchmark = True
        return self

    @property
    def embedding_dim(self) -> int:
        return 768

    def _preprocess_slices(self, volume: torch.Tensor) -> torch.Tensor:
        """[1, P, L, S] HU volume → [S, 1, crop, crop], faithful to
        `CuriaImageProcessor.__call__`'s 3D branch (vectorised across slices;
        F.interpolate is per-sample independent, so batching the resize is
        bit-identical to the author's per-slice loop)."""
        from torchvision.transforms.functional import convert_image_dtype

        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"Curia-2 expects [1, P, L, S]; got {tuple(volume.shape)}")
        vol = volume.to(self._device, non_blocking=True)[0]      # [P, L, S] HU
        slices = vol.permute(2, 0, 1).to(torch.int16)            # [S, P, L]  (_to_tensor numpy path)
        slices = convert_image_dtype(slices, torch.float32)      # ÷32767
        slices = slices.unsqueeze(1)                             # [S, 1, P, L]
        slices = torch.nn.functional.interpolate(
            slices, size=(self._crop, self._crop),
            mode="bicubic", align_corners=False, antialias=True,
        )                                                        # [S, 1, crop, crop]
        if self._clip:
            torch.clamp_min(slices, -1000.0, out=slices)
        # Single per-volume z-score (CuriaImageProcessor._zscore_per_image over
        # the whole stacked volume — one mean/std, not per-slice).
        mean = float(slices.mean())
        std = float(slices.std())
        if std < self._eps:
            slices = slices - mean
        else:
            slices = (slices - mean) / std
        return slices

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        slices = self._preprocess_slices(volume)                 # [S, 1, crop, crop]
        # bf16 autocast on CUDA: ~2× tensor-core throughput + half the
        # activation memory for the DINOv2-B forwards (the dominant cost of
        # this slice-loop model). Preprocessing stays fp32 (z-score precision).
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_amp and self._device.type == "cuda"
            else contextlib.nullcontext()
        )
        feats: list[torch.Tensor] = []
        with amp:
            for i in range(0, slices.shape[0], self._bs):
                out = self._model(pixel_values=slices[i:i + self._bs])
                feats.append(out.pooler_output.float())          # [b, 768] layernormed CLS
        emb = torch.cat(feats, dim=0).mean(dim=0)                # [768]
        return emb.detach().cpu().float()


def _flexict_compose(
    axcodes: str = "LPS",
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    clip: tuple[float, float] = (-1000.0, 1000.0),
):
    """Volume MONAI compose for FlexiCT, mirroring the author's CT-RATE eval
    (`downstream/vlm/dataset.py::CT3D_CLIP` + `downstream/vlm/ct_rate_eval.py`):

      reorient -> 2.0 mm isotropic resample -> clip [-1000, 1000] HU ->
      whole-volume z-score.

    The released loader uses `spacing=(2.0,2.0,2.0)`, `torch.clamp(-1000,1000)`,
    then a whole-volume z-score (`zero_to_one`, misnamed), and finally
    pad-with-min + center-crop to 160^3. The 160^3 pad/crop runs in `embed()`
    (it needs the per-volume min as pad value, which a static Compose can't
    express). 160 is divisible by the patch size 8.

    AXCODES = "LPS" — grounded in FlexiCT's own code: the retrieval pipeline
    (`downstream/retrieval/ours_radio_retrieval.py`) uses
    `monai.transforms.Orientationd(axcodes="LPS")` (default `orientation="LPS"`)
    and the C4KC/demo paths use sitk `DICOMOrient(..., "LPS")`. The
    classification/ZS eval (`CT3D_CLIP`) skips reorientation only because
    CT-RATE's native order is already LPS. We canonicalise all three datasets to
    LPS with the same MONAI transform the author uses, on our per-dataset LPS
    affines (already verified: RadChest (I,P,L) via `_radchest_lps_affine_zyx`;
    CT-RATE carries an LPS affine + space=LPS). A one-time mid-plane slice dump
    is still cheap insurance, but "LPS" is author-sourced, not a guess.
    """
    from monai.transforms import (
        Compose,
        NormalizeIntensityd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    return Compose([
        Orientationd(keys="image", axcodes=axcodes),
        Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
        # b_min/b_max == a_min/a_max + clip=True -> clamp to [-1000,1000], no rescale.
        ScaleIntensityRanged(
            keys="image", a_min=clip[0], a_max=clip[1],
            b_min=clip[0], b_max=clip[1], clip=True,
        ),
        NormalizeIntensityd(keys="image"),          # whole-volume z-score (mean/std)
    ])


class FlexiCTExtractor:
    """FlexiCT (Li et al. 2026, arXiv 2605.21906; Georgia Tech / Emory) —
    agglomerative DINO/iBOT CT foundation model. Native 3D ViT-Base backbone
    (patch 8, embed dim 864).

    Two checkpoints are wired (download from HF `ricklisz123/FlexiCT` or the
    Google-Drive links in the repo readme; pass the path or set
    FLEXICT_VLM_CHECKPOINT / FLEXICT_3D_CHECKPOINT):

      - variant="vlm" (default): Stage-3 vision-language model
        (`Flexi_CT_VLM`, ct_3d_vlm.pth). Its image encoder is the SAME backbone
        as FlexiCT-3D but report-aligned, and serves BOTH tracks:
          * project=False -> 1728-d concat([CLS], mean patch tokens)   (kNN)
          * project=True  -> 1024-d VLM projection, L2-normalized       (ZS image)
        The Qwen3-Embedding text tower (for ZS prompts) lives in
        `ctfm_eval.zero_shot.FlexiCTScorer`, not here.
      - variant="3d": Stage-2 pure-SSL teacher (`Flexi_CT_3D`, ct_3d_teacher.pth).
        1728-d concat, kNN only (no text tower). For the SSL-vs-VLM kNN sanity
        comparison; name="flexict3d".

    Preprocessing (`_flexict_compose` + the 160^3 pad/crop here) mirrors the
    author's CT-RATE eval exactly: orient -> 2 mm iso -> clip[-1000,1000] ->
    whole-volume z-score -> pad-with-min + center-crop to 160^3. The kNN concat
    is returned RAW (kNN cosine normalises); the ZS projection is already
    L2-normalized by `encode_image`.

    `third_party/FlexiCT` must be importable (added to sys.path in __init__).
    """

    def __init__(
        self,
        variant: str = "vlm",
        project: bool = False,
        checkpoint_path: str | None = None,
        axcodes: str = "LPS",
        spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
        roi: int = 160,
        use_amp: bool = True,
    ) -> None:
        import sys

        root = str(Path(__file__).resolve().parent.parent / "third_party" / "FlexiCT")
        if root not in sys.path:
            sys.path.insert(0, root)

        if variant not in ("vlm", "3d"):
            raise ValueError(f"FlexiCT variant must be 'vlm' or '3d', got {variant!r}")
        if project and variant != "vlm":
            raise ValueError("project=True (1024-d ZS) requires variant='vlm'")

        self._variant = variant
        self._project = bool(project)
        self._roi = int(roi)
        self._use_amp = bool(use_amp)
        self._device = torch.device("cpu")
        self.name = "flexict" if variant == "vlm" else "flexict3d"

        if variant == "vlm":
            from flexi_ct import Flexi_CT_VLM
            self._model = Flexi_CT_VLM(checkpoint_path=checkpoint_path, device="cpu").eval()
        else:
            from flexi_ct import Flexi_CT_3D
            self._model = Flexi_CT_3D(checkpoint_path=checkpoint_path, device="cpu").eval()

        self._transform = _flexict_compose(axcodes=axcodes, spacing=spacing)
        self.worker_preprocess = _monai_preprocess_worker(self._transform)
        self.gpu_compose = self._transform

    def to(self, device: torch.device) -> "FlexiCTExtractor":
        self._device = device
        self._model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return 1024 if self._project else 1728

    def _pad_crop(self, x: torch.Tensor) -> torch.Tensor:
        """[1, D, H, W] -> [1, roi, roi, roi]: center-pad with the volume min
        (matching `CT3D_CLIP.pad_min`), then center-crop. roi=160 is /8."""
        import torch.nn.functional as F

        t = self._roi
        d, h, w = x.shape[-3:]
        pd, ph, pw = max(0, t - d), max(0, t - h), max(0, t - w)
        if pd or ph or pw:
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2)
            x = F.pad(x, pad, mode="constant", value=float(x.amin()))
        d, h, w = x.shape[-3:]
        sd, sh, sw = (d - t) // 2, (h - t) // 2, (w - t) // 2
        return x[..., sd:sd + t, sh:sh + t, sw:sw + t]

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"FlexiCT expects [1, D, H, W]; got {tuple(volume.shape)}")
        vol = self._pad_crop(volume.to(self._device, non_blocking=True))
        x = vol.unsqueeze(0).float()                                   # [1, 1, 160, 160, 160]
        with torch.autocast("cuda", enabled=self._use_amp and self._device.type == "cuda"):
            if self._variant == "vlm":
                if self._project:
                    emb = self._model.encode_image(x)                 # [1, 1024] L2-norm
                else:
                    feats = self._model.model.vision_model(x, is_training=True)
                    emb = torch.cat(
                        [feats["x_norm_clstoken"], feats["x_norm_patchtokens"].mean(dim=1)],
                        dim=-1,
                    )                                                 # [1, 1728]
            else:
                feats = self._model(x)                                # {cls_token, patch_tokens}
                emb = torch.cat(
                    [feats["cls_token"], feats["patch_tokens"].mean(dim=1)], dim=-1,
                )                                                     # [1, 1728]
        return emb[0].detach().cpu().float()


class VoxelFMExtractor:
    """VoxelFM (Maguado et al., arXiv 2604.04133; `rmaguado/VoxelFM`,
    Apache-2.0) — a 3D ViT CT foundation model trained with DINOv2-style
    self-distillation (DINO CLS + iBOT masked-patch), no language tower →
    kNN / linear-probe track only (peer = CT-FM, Curia-2).

    Unlike every other extractor in this repo, VoxelFM's canonical inference
    pipeline is **not** MONAI-based and its isotropic resample is *adaptive
    per volume* (target spacing = max(min(native_spacing), 0.75) mm), so a
    fixed `Spacingd` cannot reproduce it. We therefore run the author's own
    `dinov2.inference` helpers inside `embed()`, mirroring the canonical eval
    `worker` (third_party/VoxelFM/dinov2/inference/distributed.py):

        load (HU clamp) -> crop_volume(k=21) -> resize_isotropic(min=0.75)
        -> resize_max_patches(max_patches) -> patch_crop(patch_size)
        -> (x - fmean)/fstd -> 3D-ViT forward

    The embedding is the author-canonical kNN feature: **mean-pooled patch
    tokens** (`x_norm_patchtokens`, 864-d for vitb_3d). This is what their
    own CT-RATE kNN head-to-head reports (evaluation/CT_RATE: `select_feature
    ="patch", pooling="avg"`, the `output/dino/patch` runs). `feature="cls"`
    selects the 864-d CLS token instead. `fmean`/`fstd`/`patch_size`/
    `embed_dim` are read from the shipped `config.yaml` (global dataset
    z-score), never hardcoded.

    Worker stays IO-only: the dataset hands `embed()` a raw HU volume plus
    `spacing` (RadChest, isotropic scalar — stored [I,P,L] = already
    axial-first) or an LPS `affine` (CT-RATE, [X,Y,Z] order — we derive
    per-axis spacing and reorder the through-plane axis to dim0, mirroring
    the VoxelFM NIfTI loader's axis reversal). No `worker_preprocess` /
    `gpu_compose` is set, so the main loop passes the raw tensor straight
    through with `meta` carrying spacing/affine.
    """

    def __init__(
        self,
        hf_repo: str = "rmaguado/VoxelFM",
        subfolder: str = "vitb_3d",
        checkpoint_file: str = "vitb_3d/checkpoints/99999.pth",
        config_file: str = "vitb_3d/config.yaml",
        repo_path: str | Path = "third_party/VoxelFM",
        feature: str = "patch",
        max_patches: int = 25000,
        min_spacing: float = 0.75,
        crop_background: bool = True,
        crop_kernel: int = 21,
        hu_range: tuple[float, float] = (-1000.0, 1900.0),
        use_amp: bool = False,
    ) -> None:
        import sys as _sys

        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        if feature not in ("patch", "cls"):
            raise ValueError(f"feature must be 'patch' or 'cls'; got {feature!r}")

        # Vendored upstream is not pip-packaged; put it on sys.path so its
        # `dinov2.inference` (build_model + processing helpers) imports.
        repo = Path(repo_path)
        if not repo.is_absolute():
            repo = (Path(__file__).resolve().parents[1] / repo).resolve()
        if not (repo / "dinov2" / "inference" / "processing.py").exists():
            raise FileNotFoundError(
                f"VoxelFM repo not found at {repo}. Clone from "
                "https://github.com/rmaguado/VoxelFM into third_party/VoxelFM."
            )
        if str(repo) not in _sys.path:
            _sys.path.insert(0, str(repo))

        from dinov2.inference import (  # type: ignore
            build_model,
            crop_volume,
            patch_crop,
            resize_isotropic,
            resize_max_patches,
        )

        ckpt = hf_hub_download(hf_repo, checkpoint_file)
        cfg_path = hf_hub_download(hf_repo, config_file)
        config = OmegaConf.load(cfg_path)

        self.name = "voxelfm"
        self._feature = feature
        self._patch_size = int(config.student.patch_size)
        self._embed_dim = int(config.student.embed_dim)
        self._fmean = float(config.datasets[0].norm.mean)
        self._fstd = float(config.datasets[0].norm.std)
        self._max_patches = int(max_patches)
        self._min_spacing = float(min_spacing)
        self._crop_bg = bool(crop_background)
        self._crop_k = int(crop_kernel)
        self._hu = (float(hu_range[0]), float(hu_range[1]))
        self._use_amp = bool(use_amp)

        # Bind the author's helpers (imported lazily above).
        self._crop_volume = crop_volume
        self._resize_isotropic = resize_isotropic
        self._resize_max_patches = resize_max_patches
        self._patch_crop = patch_crop

        self._device = torch.device("cpu")
        self._config = config
        self._build_model = build_model
        self._ckpt = ckpt
        self._model = build_model(ckpt, config, device=self._device)

        # Worker is IO-only; all preprocessing happens in embed() with the
        # native spacing/affine, so no worker_preprocess / gpu_compose.
        self.worker_preprocess = None

    def to(self, device: torch.device) -> "VoxelFMExtractor":
        self._device = device
        self._model.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def _to_dhw_spacing(
        self, volume: torch.Tensor, meta: dict | None
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        """Return an axial-first [D, H, W] HU volume + per-axis spacing (mm),
        mirroring what VoxelFM's loaders feed `resize_isotropic`."""
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"VoxelFM expects [1, *, *, *]; got {tuple(volume.shape)}")
        meta = meta or {}
        affine = meta.get("affine")
        if affine is None:
            # RadChest: [1, Z=I, Y=P, X=L]; dim0 = I = axial. Isotropic scalar.
            sp = float(meta.get("spacing", 1.0))
            return volume[0], (sp, sp, sp)
        # CT-RATE: [1, X, Y, Z] with an LPS affine. Per-axis spacing is
        # the column norm of the 3×3; the axial (S-aligned) voxel axis is the
        # one whose world-Z row component dominates. Move it to dim0 and
        # reverse the in-plane pair (mirrors load_nifti's "x y z -> z y x").
        aff = torch.as_tensor(affine, dtype=torch.float32)
        R = aff[:3, :3]
        spacing = torch.linalg.norm(R, dim=0)                      # (sX, sY, sZ)
        axial = int(torch.argmax(torch.abs(R[2])).item())
        rest = [k for k in (2, 1, 0) if k != axial]
        perm = [axial] + rest
        vol = volume[0].permute(*perm).contiguous()
        sp = tuple(float(spacing[k]) for k in perm)
        return vol, sp

    @torch.inference_mode()
    def embed(self, volume: torch.Tensor, meta: dict | None = None) -> torch.Tensor:
        vol, spacing = self._to_dhw_spacing(volume, meta)
        dev = self._device
        vol = vol.to(dev, non_blocking=True).float().clamp_(self._hu[0], self._hu[1])
        if self._crop_bg:
            vol = self._crop_volume(vol, dev, self._crop_k)
        vol = self._resize_isotropic(vol, spacing, dev, min_spacing=self._min_spacing)
        if self._max_patches:
            vol = self._resize_max_patches(vol, self._patch_size, self._max_patches, dev)
        vol = self._patch_crop(vol, self._patch_size)
        x = ((vol - self._fmean) / self._fstd).unsqueeze(0).to(device=dev, dtype=torch.float)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_amp and dev.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp:
            feats = self._model(x)                                 # dict of [1, *, embed_dim]
        if self._feature == "patch":
            emb = feats["x_norm_patchtokens"][0].float().mean(dim=0)   # [embed_dim]
        else:
            emb = feats["x_norm_clstoken"][0].float()                 # [embed_dim]
        return emb.detach().cpu().float()


def extract_embeddings(
    extractor: EmbeddingExtractor,
    dataset,
    dataset_name: str,
    device: torch.device,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool | None = None,
) -> EmbeddingBatch:
    """Iterate dataset with a DataLoader (batch=1) and stack embeddings.

    Worker pipeline (CPU, parallel):
        NPZ inflate -> channel-first -> `extractor.worker_preprocess` (if any)
        -> tensor cast.
    Main loop (GPU):
        `extractor.embed(image, meta)`.
    """
    extractor.to(device)
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Optional main-thread GPU preprocessing. Activated when the batch carries
    # an `affine` field (CT-RATE path). RadChest does NOT include `affine`, so
    # its existing path is untouched and embeddings remain bit-identical.
    gpu_compose = getattr(extractor, "gpu_compose", None)
    from monai.data.meta_tensor import MetaTensor

    embs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[str] = []
    groups: list[str] = []
    for batch in tqdm(loader, desc=f"extract[{extractor.name}]", total=len(loader)):
        image = batch["image"][0]                   # [C, ...]  (CPU; extractor handles .to(device))
        meta = _sample_meta(batch)
        if gpu_compose is not None and batch.get("affine") is not None:
            affine = batch["affine"][0]
            # Only declare LPS to MONAI when the dataset asks for it. CT-RATE
            # has historically been extracted without this metadata (MONAI
            # then defaults to RAS interpretation of the affine), and changing
            # it would break bit-identity of the cached CT-RATE caches. A
            # dataset may instead emit `_space="LPS"` to opt in to correct
            # interpretation when its scans carry heterogeneous `axcodes`.
            mt_kwargs: dict = {}
            space_field = batch.get("_space")
            if space_field is not None:
                space = space_field[0] if isinstance(space_field, list) else space_field
                if space:
                    from monai.utils.enums import SpaceKeys
                    mt_kwargs["meta"] = {"space": SpaceKeys(space)}
            image_gpu = MetaTensor(
                image.float().to(device),
                affine=affine.float().to(device),
                **mt_kwargs,
            )
            out = gpu_compose({"image": image_gpu})
            image = out["image"].as_tensor() if hasattr(out["image"], "as_tensor") else out["image"]
        emb = extractor.embed(image, meta)          # [D_emb]
        embs.append(emb)
        labels.append(batch["label"][0].cpu().float())
        ids.append(batch["id"][0])
        if "patient_id" in batch:
            groups.append(batch["patient_id"][0])

    return EmbeddingBatch(
        embeddings=torch.stack(embs, dim=0),
        labels=torch.stack(labels, dim=0),
        ids=ids,
        label_columns=list(dataset.label_columns),
        model_name=extractor.name,
        dataset_name=dataset_name,
        groups=groups if groups else None,
    )
