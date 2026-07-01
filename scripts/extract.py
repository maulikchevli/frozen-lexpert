"""Extract embeddings for a (model, dataset) pair and cache them to disk.

The extractor owns its own preprocessing pipeline. The dataset receives the
extractor's `worker_preprocess` callable so heavy per-sample preprocessing
runs inside DataLoader workers. The main thread only moves tensors to GPU
and runs the model forward.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ctfm_eval.datasets import CTRateDataset, RadChestCTDataset
from ctfm_eval.embeddings import (
    CTCLIPExtractor,
    CTFMExtractor,
    CTSSGExtractor,
    ColipriExtractor,
    Curia2Extractor,
    FlexiCTExtractor,
    MerlinExtractor,
    Pillar0Extractor,
    SpectreExtractor,
    VoxelFMExtractor,
    extract_embeddings,
)


def _build_extractor(cfg: DictConfig):
    name = cfg.model.name
    if name == "spectre":
        m = cfg.model
        return SpectreExtractor(
            model_config=m.model_config,
            patch_size=tuple(m.patch_size),
            grid_size=tuple(m.grid_size),
            target_spacing=tuple(m.get("target_spacing", (0.75, 0.75, 1.5))),
            target_shape=tuple(m.get("target_shape", (384, 384, 256))),
            use_amp=bool(m.get("use_amp", True)),
            cls_plus_mean=bool(m.get("cls_plus_mean", False)),
        )
    if name == "colipri":
        m = cfg.model
        return ColipriExtractor(
            pool=bool(m.get("pool", True)),
            project=bool(m.get("project", True)),
            normalize=bool(m.get("normalize", True)),
            input_size=m.get("input_size"),
            spacing_mm=m.get("spacing_mm"),
        )
    if name == "ctssg":
        m = cfg.model
        return CTSSGExtractor(
            repo_path=str(m.get("repo_path", "third_party/ct-ssg")),
            input_size=int(m.get("input_size", 480)),
            slice_count=int(m.get("slice_count", 240)),
            target_spacing=tuple(m.get("target_spacing", (1.5, 0.75, 0.75))),
            hu_range=tuple(m.get("hu_range", (-1000.0, 200.0))),
            imagenet_mean=float(m.get("imagenet_mean", 0.449)),
        )
    if name == "ctclip":
        m = cfg.model
        return CTCLIPExtractor(
            input_shape=tuple(m.get("input_shape", (240, 480, 480))),
            target_spacing=tuple(m.get("target_spacing", (1.5, 0.75, 0.75))),
            hu_range=tuple(m.get("hu_range", (-1000.0, 1000.0))),
            project=bool(m.get("project", False)),
        )
    if name == "merlin":
        m = cfg.model
        return MerlinExtractor(
            input_shape=tuple(m.get("input_shape", (224, 224, 160))),
            target_spacing=tuple(m.get("target_spacing", (1.5, 1.5, 3.0))),
            hu_range=tuple(m.get("hu_range", (-1000.0, 1000.0))),
        )
    if name == "ctfm":
        m = cfg.model
        return CTFMExtractor(
            patch_size=tuple(m.get("patch_size", (24, 128, 128))),
            overlap=float(m.get("overlap", 0.0)),
            batch_size=int(m.get("batch_size", 16)),
            target_spacing=tuple(m.get("target_spacing", (3.0, 1.0, 1.0))),
            hu_range=tuple(m.get("hu_range", (-1024.0, 2048.0))),
        )
    if name == "pillar0":
        m = cfg.model
        return Pillar0Extractor(
            target_spacing=tuple(m.get("target_spacing", (1.25, 1.25, 1.25))),
            input_shape=tuple(m.get("input_shape", (256, 256, 256))),
            modality=str(m.get("modality", "chest_ct")),
            hf_repo=str(m.get("hf_repo", "YalaLab/Pillar0-ChestCT")),
        )
    if name == "curia2":
        m = cfg.model
        return Curia2Extractor(
            hf_repo=str(m.get("hf_repo", "raidium/curia-2")),
            axcodes=str(m.get("axcodes", "PLS")),
            crop_size=int(m.get("crop_size", 512)),
            clip_below_air=bool(m.get("clip_below_air", True)),
            eps=float(m.get("eps", 1e-6)),
            slice_batch_size=int(m.get("slice_batch_size", 96)),
            use_amp=bool(m.get("use_amp", True)),
        )
    if name == "voxelfm":
        m = cfg.model
        return VoxelFMExtractor(
            hf_repo=str(m.get("hf_repo", "rmaguado/VoxelFM")),
            subfolder=str(m.get("subfolder", "vitb_3d")),
            checkpoint_file=str(m.get("checkpoint_file", "vitb_3d/checkpoints/99999.pth")),
            config_file=str(m.get("config_file", "vitb_3d/config.yaml")),
            repo_path=str(m.get("repo_path", "third_party/VoxelFM")),
            feature=str(m.get("feature", "patch")),
            max_patches=int(m.get("max_patches", 25000)),
            min_spacing=float(m.get("min_spacing", 0.75)),
            crop_background=bool(m.get("crop_background", True)),
            crop_kernel=int(m.get("crop_kernel", 21)),
            hu_range=tuple(m.get("hu_range", (-1000.0, 1900.0))),
            use_amp=bool(m.get("use_amp", False)),
        )
    if name == "flexict":
        m = cfg.model
        return FlexiCTExtractor(
            variant=str(m.get("variant", "vlm")),
            project=bool(m.get("project", False)),
            checkpoint_path=m.get("checkpoint_path"),
            axcodes=str(m.get("axcodes", "LPS")),
            spacing=tuple(m.get("spacing", (2.0, 2.0, 2.0))),
            roi=int(m.get("roi", 160)),
            use_amp=bool(m.get("use_amp", True)),
        )
    raise ValueError(f"unknown model {name!r}")


def _build_dataset(cfg: DictConfig, extractor):
    """Build the dataset and decide where the heavy preprocessing lives.

    For CT-RATE we default to GPU-side preprocessing (via
    `extractor.gpu_compose`) for MONAI-based extractors, which dominates
    the Spacingd cost. For extractors without `gpu_compose` (COLIPRI,
    which uses torchio), we keep the per-extractor worker pipeline
    running in DataLoader workers so its torchio transforms still apply.
    """
    name = cfg.dataset.name
    if name == "radchestct":
        d = cfg.dataset
        return RadChestCTDataset(
            json_path=d.json_path,
            npz_key=d.npz_key,
            id_key=d.id_key,
            max_samples=int(d.get("max_samples", 0)),
            metadata_csv=d.get("metadata_csv"),
            spacing_column=str(d.get("spacing_column", "final_spacing")),
            worker_preprocess=extractor.worker_preprocess,
        )
    if name == "ctrate":
        d = cfg.dataset
        gpu_compose = getattr(extractor, "gpu_compose", None)
        # If the extractor exposes a GPU-ready compose we skip the worker
        # preprocess entirely and let the main loop run it on-device.
        # Otherwise (torchio path) keep the worker pipeline.
        worker_pp = None if gpu_compose is not None else extractor.worker_preprocess
        return CTRateDataset(
            labels_csv=d.labels_csv,
            data_dir=d.data_dir,
            metadata_csv=d.get("metadata_csv"),
            reports_csv=d.get("reports_csv"),
            max_samples=int(d.get("max_samples", 0)),
            worker_preprocess=worker_pp,
        )
    raise ValueError(f"unknown dataset {name!r}")


def _device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs/extract", config_name=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    extractor = _build_extractor(cfg)
    dataset = _build_dataset(cfg, extractor)
    device = _device(cfg.get("device"))
    print(f"device={device}  n_samples={len(dataset)}  labels={len(dataset.label_columns)}")

    loader_cfg = cfg.get("loader", {})
    batch = extract_embeddings(
        extractor=extractor,
        dataset=dataset,
        dataset_name=cfg.dataset.name,
        device=device,
        num_workers=int(loader_cfg.get("num_workers", 8)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", 4)),
        pin_memory=loader_cfg.get("pin_memory"),
    )
    out = Path(cfg.cache_path)
    batch.save(out)
    print(f"saved {batch.embeddings.shape} -> {out}")


if __name__ == "__main__":
    main()
