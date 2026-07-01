from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from monai.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Transform


@dataclass(slots=True)
class Sample:
    sample_id: str
    image: torch.Tensor
    label: torch.Tensor


class EmbeddingDataset(Protocol):
    """A dataset that yields samples ready for embedding extraction.

    Each item is a dict with keys: 'id' (str), 'image' (Tensor), 'label' (Tensor
    [num_labels]), and optionally 'spacing' (float, mm isotropic). The 'image'
    tensor's shape and preprocessing state depend on whatever `worker_preprocess`
    callable is attached (extractor-owned).
    """

    label_columns: list[str]

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...


_ZIP_MAGIC = b"PK\x03\x04"


def _is_zip_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == _ZIP_MAGIC
    except OSError:
        return False


def _load_spacing_csv(
    csv_path: str | Path, id_column: str, spacing_column: str,
) -> dict[str, float]:
    """Map NoteAcc_DEID -> isotropic spacing (mm) from RadChest's metadata CSV."""
    import csv
    out: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if id_column not in reader.fieldnames or spacing_column not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: need columns {id_column!r} and {spacing_column!r}; "
                f"have {reader.fieldnames}"
            )
        for row in reader:
            sid = row[id_column]
            try:
                out[sid] = float(row[spacing_column])
            except (TypeError, ValueError):
                continue
    return out


class _LoadNpzVolume(Transform):
    def __init__(self, image_key: str, npz_key: str) -> None:
        self.image_key = image_key
        self.npz_key = npz_key

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        path = out[self.image_key]
        try:
            npz = np.load(path, allow_pickle=False)
        except ValueError:
            # Some RadChest .npz files carry object/pickle metadata alongside
            # the array. The CT itself is plain numeric.
            npz = np.load(path, allow_pickle=True)
        try:
            if self.npz_key not in npz:
                raise KeyError(f"npz key '{self.npz_key}' missing in {path}")
            out[self.image_key] = np.asarray(npz[self.npz_key], dtype=np.float32)
        finally:
            npz.close()
        return out


class _CallablePreprocessd(Transform):
    """Wrap a dict->dict preprocessing callable as a MONAI Transform."""

    def __init__(self, fn: Callable[[dict], dict]) -> None:
        self._fn = fn

    def __call__(self, data: dict) -> dict:
        return self._fn(dict(data))


def _build_radchest_transforms(
    npz_key: str,
    worker_preprocess: Callable[[dict], dict] | None,
) -> Compose:
    """Minimal CPU-side pipeline. Preprocessing that the extractor actually
    wants (orientation / spacing / HU normalisation) is provided by the extractor
    as `worker_preprocess` and is run inside DataLoader workers."""
    tfs: list = [
        _LoadNpzVolume(image_key="image", npz_key=npz_key),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
    ]
    if worker_preprocess is not None:
        tfs.append(_CallablePreprocessd(worker_preprocess))
    tfs.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False))
    return Compose(tfs)


def _ct_rate_nii_path(data_dir: Path, vol_name: str) -> Path:
    """CT-RATE layout: <data_dir>/<patient>/<scan>/<vol_name>.nii.gz.

    VolumeName encodes the hierarchy: `valid_5_b_2.nii.gz` →
      patient = "valid_5", scan = "valid_5_b", volume = valid_5_b_2.nii.gz
    """
    stem = vol_name.replace(".nii.gz", "")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected CT-RATE VolumeName {vol_name!r}")
    patient = "_".join(parts[:2])          # valid_5
    scan = "_".join(parts[:3])             # valid_5_b
    return data_dir / patient / scan / vol_name


def _ct_rate_load_metadata(csv_path: Path) -> dict[str, dict]:
    """Map VolumeName → dict(slope, intercept, xy_spacing, z_spacing).

    CT-RATE NIfTI headers carry NaN slope/intercept; the real values ship in
    CT-CHAT's metadata CSV.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        xy = row["XYSpacing"]
        if isinstance(xy, str):
            xy = float(xy.strip("[]").split(",")[0])
        try:
            out[row["VolumeName"]] = {
                "slope": float(row["RescaleSlope"]),
                "intercept": float(row["RescaleIntercept"]),
                "xy_spacing": float(xy),
                "z_spacing": float(row["ZSpacing"]),
            }
        except (TypeError, ValueError):
            continue
    return out


class _LoadCTRateNii(Transform):
    """Worker-side NIfTI loader for CT-RATE.

    - Loads .nii.gz via nibabel (MONAI's image_only wrapper),
    - applies CT-RATE's DICOM `slope*raw + intercept` with HU-guard (max < -500
      after rescale ⇒ raw is already HU),
    - converts nibabel's RAS affine to LPS voxel-order-aware form and stores
      it as a plain tensor under `affine` so default_collate + the main loop
      can re-wrap the volume as a GPU MetaTensor.

    The volume is returned as a plain float32 tensor in nibabel-native
    (X, Y, Z) order — the LPS affine describes this exactly, so MONAI's
    Orientationd in the extractor's gpu_compose will reorient correctly.
    """

    def __init__(self, image_key: str) -> None:
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        import nibabel as nib
        out = dict(data)
        path = out[self.image_key]
        img = nib.load(str(path))
        raw = np.asarray(img.get_fdata(), dtype=np.float32)     # (X, Y, Z)
        slope = float(out.get("_slope", 1.0))
        intercept = float(out.get("_intercept", 0.0))
        # CT-RATE volumes ship in two intensity states:
        #   (a) ALREADY HU — the rescale was baked in during DICOM->NIfTI
        #       conversion, so the air floor already sits at ~-1000/-1024 and
        #       bone reaches ~+3000. Here the metadata RescaleIntercept must
        #       NOT be re-applied.
        #   (b) raw stored values — air sits near 0 and slope*raw+intercept is
        #       required to reach HU.
        # The previous guard tested the POST-rescale stats and missed case (a):
        # for an already-HU volume, raw-1024 gives max ~+2047 and mean ~-1700,
        # neither of which tripped the old `< -500 / < -2000` thresholds, so it
        # silently double-shifted every volume down by 1024 HU (air -> ~-2048,
        # median -> ~-1880). That wrecked zero-shot on CT-RATE (intensity
        # windowing on garbage HU) while leaving kNN intact (the ~constant
        # shift preserves relative neighbour structure). Detect already-HU on
        # the RAW values: a calibrated CT has its air floor clearly negative.
        if float(raw.min()) <= -800.0:
            hu = raw                            # already HU; do not re-apply intercept
        else:
            hu = slope * raw + intercept        # raw stored values -> HU

        # nibabel stores (X, Y, Z) with RAS-oriented affine.
        # Convert to LPS so downstream MONAI (space default = LPS for us) is
        # consistent with the RadChest path.
        aff = np.asarray(img.affine, dtype=np.float32)
        aff_lps = aff.copy()
        aff_lps[0, :] *= -1.0     # R → L
        aff_lps[1, :] *= -1.0     # A → P

        out[self.image_key] = hu                  # [X, Y, Z]; EnsureChannelFirstd adds C
        out["affine"] = aff_lps                   # plain np/torch-castable 4×4
        return out


def _build_ctrate_transforms(
    worker_preprocess: Callable[[dict], dict] | None = None,
) -> Compose:
    """CT-RATE transform chain.

    Always runs (in workers): `LoadCTRateNii` (nibabel read + HU calibrate +
    LPS affine), `EnsureChannelFirstd` → [1, X, Y, Z].

    If `worker_preprocess` is provided (torchio extractors such as COLIPRI),
    run it in-worker — the preprocessed plain tensor then arrives in the main
    loop and the `affine` field is dropped so the extractor's `gpu_compose`
    (if any) is not re-applied.

    Otherwise (MONAI extractors), leave the image unprocessed — the main
    extraction loop wraps the volume + affine as a GPU MetaTensor and runs
    `extractor.gpu_compose` on-device.
    """
    tfs: list = [
        _LoadCTRateNii(image_key="image"),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
    ]
    if worker_preprocess is not None:
        # Hand off (image+affine) to the extractor's worker callable, then
        # drop `affine` — the main loop's guard (`batch.get("affine")`) will
        # therefore NOT re-run a gpu_compose for this sample.
        def _drop_affine(s: dict) -> dict:
            s = dict(s)
            s.pop("affine", None)
            return s
        tfs.append(_CallablePreprocessd(worker_preprocess))
        tfs.append(_CallablePreprocessd(_drop_affine))
    tfs.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False))
    return Compose(tfs)


class CTRateDataset(Dataset):
    """CT-RATE loaded from the shipped labels CSV + metadata CSV.

    NIfTI volumes live under `<data_dir>/<patient>/<scan>/<vol>.nii.gz`.
    Labels come from the pathology-prediction CSV (18 binary classes).

    Each sample dict has: `id` (VolumeName), `image` (path, resolved to tensor
    in the transform), `label` [C] float, `spacing` (anisotropic xy mm — used
    only as a scalar default for extractors that still expect one), plus
    `_slope`, `_intercept`, `z_spacing` for the NIfTI loader, and `affine` is
    populated by the loader transform.
    """

    def __init__(
        self,
        labels_csv: str | Path,
        data_dir: str | Path,
        metadata_csv: str | Path | None = None,
        reports_csv: str | Path | None = None,
        label_columns: list[str] | None = None,
        max_samples: int = 0,
        worker_preprocess: Callable[[dict], dict] | None = None,
    ) -> None:
        import pandas as pd
        labels_df = pd.read_csv(labels_csv)
        if "VolumeName" not in labels_df.columns:
            raise ValueError(f"{labels_csv}: missing VolumeName column")

        if label_columns is None:
            label_columns = [c for c in labels_df.columns if c != "VolumeName"]
        self.label_columns = list(label_columns)

        meta_map = _ct_rate_load_metadata(Path(metadata_csv)) if metadata_csv else {}

        reports_map: dict[str, dict] = {}
        if reports_csv is not None:
            rdf = pd.read_csv(reports_csv)
            for _, row in rdf.iterrows():
                reports_map[row["VolumeName"]] = {
                    "findings": str(row.get("Findings_EN", "")),
                    "impressions": str(row.get("Impressions_EN", "")),
                }

        data_dir = Path(data_dir)
        items: list[dict] = []
        n_missing = n_no_meta = 0
        for _, row in labels_df.iterrows():
            vol = row["VolumeName"]
            path = _ct_rate_nii_path(data_dir, vol)
            if not path.exists():
                n_missing += 1
                continue
            meta = meta_map.get(vol)
            if meta is None:
                n_no_meta += 1
                continue
            label = np.asarray(
                [float(row[c]) for c in self.label_columns], dtype=np.float32,
            )
            sample: dict = {
                "id": vol,
                "image": str(path),
                "label": label,
                "spacing": float(meta["xy_spacing"]),   # scalar fallback
                "_slope": float(meta["slope"]),
                "_intercept": float(meta["intercept"]),
                "z_spacing": float(meta["z_spacing"]),
            }
            if reports_map and vol in reports_map:
                sample["findings"] = reports_map[vol]["findings"]
                sample["impressions"] = reports_map[vol]["impressions"]
            items.append(sample)
            if max_samples and len(items) >= max_samples:
                break

        if n_missing or n_no_meta:
            print(f"[CTRateDataset] skipped: missing_nii={n_missing} no_metadata={n_no_meta}")
        if not items:
            raise RuntimeError("No valid CT-RATE samples after filtering")

        transform = _build_ctrate_transforms(worker_preprocess=worker_preprocess)
        super().__init__(data=items, transform=transform)


class RadChestCTDataset(Dataset):
    """RadChestCT loaded from a MONAI-style JSON list.

    Each JSON entry is {NoteAcc_DEID, image, <label_name>: bool, ...}.
    Labels are inferred as the boolean keys in the first record.

    The dataset carries per-sample voxel spacing (from RadChest's metadata CSV)
    under the 'spacing' key so extractor-side preprocessing can build correct
    affines for orientation / physical-spacing resample.
    """

    def __init__(
        self,
        json_path: str | Path,
        npz_key: str = "ct",
        id_key: str = "NoteAcc_DEID",
        label_columns: list[str] | None = None,
        max_samples: int = 0,
        metadata_csv: str | Path | None = None,
        spacing_column: str = "final_spacing",
        worker_preprocess: Callable[[dict], dict] | None = None,
    ) -> None:
        path = Path(json_path)
        with path.open() as f:
            records: list[dict] = json.load(f)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected non-empty JSON list at {path}")

        if label_columns is None:
            label_columns = [k for k, v in records[0].items() if isinstance(v, bool)]
        if not label_columns:
            raise ValueError("No boolean label columns found in JSON")
        self.label_columns = list(label_columns)

        # Per-sample isotropic voxel spacing. Required when the extractor's
        # worker_preprocess expects 'spacing' (SPECTRE, COLIPRI both do).
        spacing_by_id: dict[str, float] = {}
        if metadata_csv is not None:
            spacing_by_id = _load_spacing_csv(metadata_csv, id_key, spacing_column)

        items: list[dict] = []
        n_missing = n_broken = 0
        for rec in records:
            img = rec.get("image")
            if not img or not Path(img).exists():
                n_missing += 1
                continue
            # Reject files that aren't valid .npz (zip magic = PK\x03\x04).
            # Some RadChest .npz files are 92-byte HTML error pages from
            # failed Zenodo downloads.
            if not _is_zip_file(img):
                n_broken += 1
                continue
            label = np.asarray(
                [float(rec.get(c, False)) for c in self.label_columns], dtype=np.float32
            )
            sample: dict = {
                "id": str(rec[id_key]),
                "image": img,
                "label": label,
            }
            if spacing_by_id:
                sp = spacing_by_id.get(str(rec[id_key]))
                if sp is None:
                    # A known sample with no spacing is almost certainly a data
                    # bug; skip rather than silently fabricating a bogus affine.
                    n_broken += 1
                    continue
                sample["spacing"] = float(sp)
            items.append(sample)
            if max_samples and len(items) >= max_samples:
                break
        if n_missing or n_broken:
            print(f"[RadChestCTDataset] skipped: missing={n_missing} broken_npz={n_broken}")
        if not items:
            raise RuntimeError("No valid samples after filtering")

        transform = _build_radchest_transforms(
            npz_key=npz_key,
            worker_preprocess=worker_preprocess,
        )
        super().__init__(data=items, transform=transform)
