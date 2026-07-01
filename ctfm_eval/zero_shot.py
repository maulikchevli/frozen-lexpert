"""Zero-shot classification on cached VL-model image embeddings.

Each scorer implements `encode_text(prompts) -> [P, D_shared]` and
`project_image(cached) -> [N, D_shared]`. Scoring is cosine-similarity of
the projected image against the positive prompt minus cosine against the
negative prompt, squashed with sigmoid for a probability-like score.

Currently cached image embeddings live in assorted projection states (see
`docs/zero_shot_resume.md`). Each scorer projects at scoring time rather
than re-extracting, so cached `.pt` files stay reusable.
"""
from __future__ import annotations

from typing import Protocol

import torch


# ----- Prompts --------------------------------------------------------------

POSITIVE_TEMPLATE = "A chest CT scan showing {finding}."
NEGATIVE_TEMPLATE = "A chest CT scan showing no {finding}."

# Report-style templates — match typical radiology-report phrasing for models
# (SPECTRE in particular) whose text heads were trained on actual reports
# rather than generic "A chest CT showing X" captions.
#
# Design: keep the wording close to what appears in CT-RATE / RadChest train
# reports — short, declarative, with "Findings:"/"Impression:" cues.
REPORT_POS_TEMPLATE = "Findings: {finding} is present."
REPORT_NEG_TEMPLATE = "Findings: No {finding}."

# A pool of report-style paraphrases for ensembling. The first pair is the
# single-prompt default (see REPORT_POS_TEMPLATE above). All pairs are
# structurally matched so (pos, neg) describe the same finding.
REPORT_TEMPLATE_ENSEMBLE: list[tuple[str, str]] = [
    ("Findings: {finding} is present.",               "Findings: No {finding}."),
    ("There is {finding}.",                           "There is no {finding}."),
    ("The scan demonstrates {finding}.",              "The scan demonstrates no {finding}."),
    ("Evidence of {finding} is seen.",                "No evidence of {finding} is seen."),
    ("Impression: {finding}.",                        "Impression: No {finding}."),
    ("{finding} is noted on this chest CT.",          "No {finding} is noted on this chest CT."),
    ("{finding} is visible.",                         "{finding} is not visible."),
    ("The findings include {finding}.",               "The findings do not include {finding}."),
]

# Natural-language phrase for each RadChest label column. Missing entries
# fall back to snake-case -> space. Covers the 65 diseases+findings used for
# zero-shot; devices + surgical-history labels aren't zero-shot-evaluated,
# so their phrasings aren't included here.
RADCHEST_FINDINGS: dict[str, str] = {
    # --- original 20 (preserved wording) ---
    "coronary_artery_disease": "coronary artery disease",
    "cancer": "cancer",
    "atelectasis": "atelectasis",
    "atherosclerosis": "atherosclerosis",
    "emphysema": "emphysema",
    "pleural_effusion": "pleural effusion",
    "interstitial_lung_disease": "interstitial lung disease",
    "bronchiectasis": "bronchiectasis",
    "pericardial_effusion": "pericardial effusion",
    "mass": "a lung mass",
    "cardiomegaly": "cardiomegaly",
    "nodulegr1cm": "a pulmonary nodule greater than 1 cm",
    "aspiration": "aspiration",
    "pneumonia": "pneumonia",
    "pulmonary_edema": "pulmonary edema",
    "pneumothorax": "pneumothorax",
    "aneurysm": "an aortic aneurysm",
    "heart_failure": "heart failure",
    "tuberculosis": "tuberculosis",
    "hemothorax": "hemothorax",
    # --- expanded diseases (category = diseases) ---
    "infection": "an infection",
    "arthritis": "arthritis",
    "scarring": "scarring",
    "lymphadenopathy": "lymphadenopathy",
    "fibrosis": "fibrosis",
    "hernia": "a hernia",
    "inflammation": "inflammation",
    "pneumonitis": "pneumonitis",
    "bronchiolitis": "bronchiolitis",
    "bronchitis": "bronchitis",
    # --- expanded findings (category = findings) ---
    "nodule": "a pulmonary nodule",
    "calcification": "calcification",
    "opacity": "a pulmonary opacity",
    "groundglass": "a ground-glass opacity",
    "lesion": "a lesion",
    "scattered_nod": "scattered pulmonary nodules",
    "scattered_calc": "scattered calcifications",
    "bandlike_or_linear": "a bandlike or linear opacity",
    "soft_tissue": "a soft tissue abnormality",
    "cyst": "a cyst",
    "airspace_disease": "airspace disease",
    "consolidation": "consolidation",
    "reticulation": "a reticular pattern",
    "density": "an abnormal density",
    "bronchial_wall_thickening": "bronchial wall thickening",
    "granuloma": "a granuloma",
    "pleural_thickening": "pleural thickening",
    "septal_thickening": "septal thickening",
    "fracture": "a fracture",
    "deformity": "a deformity",
    "dilation_or_ectasia": "dilation or ectasia",
    "mucous_plugging": "mucous plugging",
    "cavitation": "a cavitary lesion",
    "debris": "debris",
    "air_trapping": "air trapping",
    "pericardial_thickening": "pericardial thickening",
    "infiltrate": "a pulmonary infiltrate",
    "honeycombing": "honeycombing",
    "tree_in_bud": "a tree-in-bud pattern",
    "plaque": "a pleural plaque",
    "secretion": "airway secretions",
    "lucency": "an abnormal lucency",
    "distention": "distention",
    "bronchiolectasis": "bronchiolectasis",
    "congestion": "vascular congestion",
    # --- location-split polyorgan labels (diseases) ---
    "cancer_lung": "lung cancer",
    "cancer_mediastinal": "mediastinal cancer",
    "cancer_extrathoracic": "extrathoracic cancer",
    "infection_lung": "a pulmonary infection",
    "infection_extrapulmonary": "an extrapulmonary infection",
    "inflammation_lung": "pulmonary inflammation",
    "mass_lung": "a lung mass",
    "mass_mediastinal": "a mediastinal mass",
    "mass_extrathoracic": "an extrathoracic mass",
    "scarring_lung": "pulmonary scarring",
    "fibrosis_lung": "pulmonary fibrosis",
    "lymphadenopathy_mediastinal": "mediastinal lymphadenopathy",
    "lymphadenopathy_axillary": "axillary lymphadenopathy",
    # --- location-split polyorgan labels (findings) ---
    "nodule_lung": "a pulmonary nodule",
    "lesion_lung": "a pulmonary lesion",
    "lesion_extrapulmonary": "an extrapulmonary lesion",
    "calcification_cardiac": "cardiac calcification",
    "calcification_vascular": "vascular calcification",
    "calcification_other": "calcification elsewhere",
}


def _finding_text(lbl: str) -> str:
    """Label → human-language finding phrase, with CT-RATE / RadChest lookup
    and snake-case fallback. Used by both single-template and ensemble prompt
    builders."""
    if lbl in CTRATE_FINDINGS:
        return CTRATE_FINDINGS[lbl]
    return RADCHEST_FINDINGS.get(lbl, lbl.replace("_", " ").lower())


def build_prompts(labels: list[str], style: str = "generic") -> tuple[list[str], list[str]]:
    """Return `(positives, negatives)` parallel to `labels`.

    style:
      - "generic" (default): "A chest CT scan showing {finding}." / "A chest
        CT scan showing no {finding}." — unchanged from the original spec.
      - "report": radiology-report phrasing. Intended for SPECTRE whose
        SigLIP text head was trained on actual reports — generic prompts
        leave ~20 PR-AUC points on the table there.
    """
    pos: list[str] = []
    neg: list[str] = []
    if style == "generic":
        pos_t, neg_t = POSITIVE_TEMPLATE, NEGATIVE_TEMPLATE
    elif style == "report":
        pos_t, neg_t = REPORT_POS_TEMPLATE, REPORT_NEG_TEMPLATE
    else:
        raise ValueError(f"unknown style: {style!r}")
    for lbl in labels:
        finding = _finding_text(lbl)
        pos.append(pos_t.format(finding=finding))
        neg.append(neg_t.format(finding=finding))
    return pos, neg


def build_prompts_ensemble(
    labels: list[str], templates: list[tuple[str, str]] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return `(positives, negatives)` where each element is a LIST of
    paraphrased prompts for that label — one per template in `templates`.

    The caller averages embeddings across the paraphrases to get a single
    per-label text embedding (template-ensemble zero-shot).
    """
    tpls = templates or REPORT_TEMPLATE_ENSEMBLE
    pos: list[list[str]] = []
    neg: list[list[str]] = []
    for lbl in labels:
        finding = _finding_text(lbl)
        pos.append([p.format(finding=finding) for p, _ in tpls])
        neg.append([n.format(finding=finding) for _, n in tpls])
    return pos, neg


# CT-RATE ships 18 binary pathology labels. Each maps to a natural-language
# finding phrase for zero-shot scoring.
CTRATE_FINDINGS: dict[str, str] = {
    "Medical material":                   "medical material",
    "Arterial wall calcification":        "arterial wall calcification",
    "Cardiomegaly":                       "cardiomegaly",
    "Pericardial effusion":               "pericardial effusion",
    "Coronary artery wall calcification": "coronary artery wall calcification",
    "Hiatal hernia":                      "a hiatal hernia",
    "Lymphadenopathy":                    "lymphadenopathy",
    "Emphysema":                          "emphysema",
    "Atelectasis":                        "atelectasis",
    "Lung nodule":                        "a lung nodule",
    "Lung opacity":                       "a lung opacity",
    "Pulmonary fibrotic sequela":         "pulmonary fibrotic sequela",
    "Pleural effusion":                   "pleural effusion",
    "Mosaic attenuation pattern":         "a mosaic attenuation pattern",
    "Peribronchial thickening":           "peribronchial thickening",
    "Consolidation":                      "consolidation",
    "Bronchiectasis":                     "bronchiectasis",
    "Interlobular septal thickening":     "interlobular septal thickening",
}


class ZeroShotScorer(Protocol):
    name: str
    shared_dim: int

    def to(self, device: torch.device) -> "ZeroShotScorer": ...
    def encode_text(self, prompts: list[str]) -> torch.Tensor: ...
    def project_image(self, cached: torch.Tensor) -> torch.Tensor: ...


@torch.inference_mode()
def score_against_prompts(
    image_feats: torch.Tensor,
    pos_prompts: list[str],
    neg_prompts: list[str],
    scorer: ZeroShotScorer,
    device: torch.device,
) -> torch.Tensor:
    """Returns `[N, C]` probability-like scores (sigmoid of pos-neg cos margin).

    Both image and text embeddings are L2-normalized by the scorer, so the
    dot-product below is cosine similarity. We use `sigmoid(cos_pos - cos_neg)`
    rather than a softmax over the pos/neg pair so each class gets an
    independent score (appropriate for multi-label).
    """
    scorer.to(device)
    img = scorer.project_image(image_feats.to(device))       # [N, D]
    pos = scorer.encode_text(pos_prompts)                    # [C, D]
    neg = scorer.encode_text(neg_prompts)                    # [C, D]
    cos_pos = img @ pos.T                                    # [N, C]
    cos_neg = img @ neg.T                                    # [N, C]
    return torch.sigmoid(cos_pos - cos_neg).float().cpu()


def score_against_prompts_ensemble(
    image_feats: torch.Tensor,
    pos_prompt_sets: list[list[str]],   # [C][K] — K paraphrases per class
    neg_prompt_sets: list[list[str]],
    scorer: "ZeroShotScorer",
    device: torch.device,
) -> torch.Tensor:
    """Ensemble variant of `score_against_prompts`.

    Encodes every paraphrase, averages the K text embeddings per class
    into one prototype, then scores as usual. Averaging happens in the
    already-L2-normalised embedding space (scorer.encode_text applies
    normalization) then re-normalizes the mean so cosine math stays valid.
    """
    scorer.to(device)
    img = scorer.project_image(image_feats.to(device))            # [N, D]
    C = len(pos_prompt_sets)
    assert len(neg_prompt_sets) == C

    def _avg(sets: list[list[str]]) -> torch.Tensor:
        # Flatten, encode once, then mean-pool per class, then re-normalize.
        flat = [p for row in sets for p in row]
        per_class = [len(row) for row in sets]
        enc = scorer.encode_text(flat)                             # [sum_k, D]
        out = torch.empty(C, enc.shape[1], device=enc.device, dtype=enc.dtype)
        offset = 0
        for c, k in enumerate(per_class):
            out[c] = enc[offset : offset + k].mean(dim=0)
            offset += k
        return torch.nn.functional.normalize(out, dim=-1)

    pos = _avg(pos_prompt_sets)                                    # [C, D]
    neg = _avg(neg_prompt_sets)                                    # [C, D]
    cos_pos = img @ pos.T
    cos_neg = img @ neg.T
    return torch.sigmoid(cos_pos - cos_neg).float().cpu()


# ----- COLIPRI --------------------------------------------------------------

class ColipriScorer:
    """COLIPRI cached features are already `encode_image(project=True,
    normalize=True)` so `project_image` is just a safety L2-norm."""

    name = "colipri-crm"
    shared_dim = 768

    def __init__(self) -> None:
        from colipri import get_model, get_processor
        self._model = get_model(pretrained=True, image_only=False).eval()
        self._proc = get_processor(image_only=False)
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "ColipriScorer":
        self._device = device
        self._model.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        token_ids, attn = self._proc.process_text(prompts)
        token_ids = token_ids.to(self._device)
        attn = attn.to(self._device)
        return self._model.encode_text(
            token_ids, attn, pool=True, normalize=True,
        )

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(cached.to(self._device), dim=-1)


# ----- CT-CLIP --------------------------------------------------------------

class CTClipScorer:
    """CT-CLIP zero-shot.

    Expects cached image features to be the POST-projection 512-d feature
    (from `CTCLIPExtractor(project=True)` i.e. mean-over-time + flatten-spatial
    + `to_visual_latent`). This path loads only the BiomedVLP-CXR-BERT text
    tower + `to_text_latent` (768 -> 512) to score against cached images.

    Note: the default `ctclip_radchest.pt` cache is AVG-POOLED pre-projection
    (different composition, incompatible with `to_visual_latent`'s weights).
    Use `ctclip-zs_radchest.pt` for zero-shot.
    """

    name = "ctclip"
    shared_dim = 512

    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, BertModel

        text_encoder = BertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True,
        ).eval()

        weights_path = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            repo_type="dataset",
            filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Load text_transformer weights into the BertModel we just built.
        txt_state = {
            k[len("text_transformer."):]: v
            for k, v in ckpt.items() if k.startswith("text_transformer.")
        }
        text_encoder.load_state_dict(txt_state, strict=False)

        # Standalone to_text_latent Linear (768 -> 512).
        w = ckpt["to_text_latent.weight"]
        b = ckpt.get("to_text_latent.bias")
        to_text_latent = torch.nn.Linear(
            w.shape[1], w.shape[0], bias=b is not None,
        )
        with torch.no_grad():
            to_text_latent.weight.copy_(w)
            if b is not None:
                to_text_latent.bias.copy_(b)

        self._text_encoder = text_encoder.eval()
        self._to_text_latent = to_text_latent.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True,
        )
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "CTClipScorer":
        self._device = device
        self._text_encoder.to(device)
        self._to_text_latent.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        t = self._tokenizer(
            prompts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=512,
        ).to(self._device)
        enc = self._text_encoder(
            t.input_ids, attention_mask=t.attention_mask,
        )[0][:, 0, :]                                                        # CLS
        return torch.nn.functional.normalize(self._to_text_latent(enc), dim=-1)

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        if cached.shape[-1] != 512:
            raise ValueError(
                f"CTClipScorer expects 512-d post-projection features; got {cached.shape[-1]}. "
                f"Re-extract with ctclip_zs_radchest.yaml (project=True)."
            )
        return torch.nn.functional.normalize(cached.to(self._device), dim=-1)


# ----- Merlin ---------------------------------------------------------------

class MerlinScorer:
    """Merlin: cached features are 2048-d ResNet152-3D pre-projection. Default
    Merlin checkpoint ships with both `encode_image.i3_resnet.contrastive_head`
    (Conv3d 2048->512) and a full TextEncoder (Clinical-Longformer + Linear
    768->512). Building with `ImageEmbedding=False` materialises both."""

    name = "merlin"
    shared_dim = 512

    def __init__(self) -> None:
        from merlin import Merlin
        self._model = Merlin(ImageEmbedding=False).eval()
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "MerlinScorer":
        self._device = device
        self._model.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        emb = self._model.model.encode_text(prompts)                          # [P, 512]
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        head = self._model.model.encode_image.i3_resnet.contrastive_head      # Conv3d(2048, 512, 1)
        x = cached.to(self._device)[..., None, None, None]                    # [N, 2048, 1, 1, 1]
        proj = head(x).flatten(start_dim=1)                                   # [N, 512]
        return torch.nn.functional.normalize(proj, dim=-1)


# ----- Pillar-0 -------------------------------------------------------------

class Pillar0Scorer:
    """Pillar-0: cached features are already the shared 1152-d space (the
    visual-side atlas output). Text side is Qwen3-Embedding-8B mean-pooled +
    L2-normalized, then passed through the shipped `model.model.text` MLP
    (4096 -> 2624 -> 1152)."""

    name = "pillar0"
    shared_dim = 1152

    def __init__(self, hf_repo: str = "YalaLab/Pillar0-ChestCT") -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        # Pillar-0's custom modeling code hits meta-tensor `.item()` during
        # `from_pretrained` init — same workaround as Pillar0Extractor.
        prev_default = torch.get_default_device() if hasattr(torch, "get_default_device") else None
        torch.set_default_device("cpu")
        try:
            cfg = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
            clip = AutoModel.from_config(cfg, trust_remote_code=True)
        finally:
            if prev_default is not None:
                torch.set_default_device(prev_default)
        weights_path = hf_hub_download(hf_repo, "model.safetensors")
        sd = load_file(weights_path, device="cpu")
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        clip.load_state_dict(sd, strict=False)
        clip.eval()
        self._clip = clip

        self._qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
        self._qenc = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-8B", torch_dtype=torch.bfloat16,
        ).eval()
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "Pillar0Scorer":
        self._device = device
        self._clip.to(device)
        self._qenc.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        inp = self._qtok(
            prompts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self._device)
        h = self._qenc(**inp).last_hidden_state                           # [P, T, 4096]
        mask = inp.attention_mask.unsqueeze(-1).float().expand(h.shape)
        pooled = (h.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean-pool
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)        # Qwen convention
        proj = self._clip.model.text(pooled)                              # [P, 1152]
        return torch.nn.functional.normalize(proj, dim=-1)

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(cached.to(self._device), dim=-1)


# ----- SPECTRE --------------------------------------------------------------

class SpectreScorer:
    """SPECTRE SigLIP zero-shot path.

    Cached image features MUST be 2160-d = `concat(CLS, mean(patch_tokens))`
    from the feature combiner (see `_build_extractor` + `SpectreExtractor(...,
    cls_plus_mean=True)`). Standard 1080-d CLS-only caches won't work.

    Text tower: Qwen3-Embedding-0.6B with LoRA adapters (`r=16, alpha=64`)
    applied to q/k/v/o projections + last-token-pool + 1024->512 SigLIP
    projection head. All three extra weights files live on `cclaess/SPECTRE`.
    """

    name = "spectre"
    shared_dim = 512

    def __init__(self) -> None:
        from huggingface_hub import hf_hub_download
        from spectre.ssl.heads.siglip_head import SigLIPProjectionHead
        from spectre.utils.lora import add_lora_adapters
        from transformers import AutoModel, AutoTokenizer

        qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        qenc = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", torch_dtype=torch.float32,
        ).eval()
        add_lora_adapters(
            qenc, r=16, lora_alpha=64, lora_dropout=0.05,
            target_keywords=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        lora_sd = torch.load(
            hf_hub_download("cclaess/SPECTRE", "spectre_qwen3_embedding_0.6B_lora.pt"),
            map_location="cpu", weights_only=False,
        )
        qenc.load_state_dict(lora_sd, strict=False)

        img_proj = SigLIPProjectionHead(input_dim=2160, output_dim=512, layer_norm=False).eval()
        txt_proj = SigLIPProjectionHead(input_dim=1024, output_dim=512, layer_norm=False).eval()
        img_proj.load_state_dict(torch.load(
            hf_hub_download("cclaess/SPECTRE", "SigLIP_projection_head_image.pt"),
            map_location="cpu", weights_only=False,
        ))
        txt_proj.load_state_dict(torch.load(
            hf_hub_download("cclaess/SPECTRE", "SigLIP_projection_head_text.pt"),
            map_location="cpu", weights_only=False,
        ))

        self._qtok = qtok
        self._qenc = qenc
        self._img_proj = img_proj
        self._txt_proj = txt_proj
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "SpectreScorer":
        self._device = device
        self._qenc.to(device)
        self._img_proj.to(device)
        self._txt_proj.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        from spectre.utils import last_token_pool

        inp = self._qtok(
            prompts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self._device)
        h = self._qenc(
            input_ids=inp.input_ids, attention_mask=inp.attention_mask,
        ).last_hidden_state
        pooled = last_token_pool(h, inp.attention_mask)                   # [P, 1024]
        return torch.nn.functional.normalize(self._txt_proj(pooled), dim=-1)

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        if cached.shape[-1] != 2160:
            raise ValueError(
                f"SpectreScorer expects 2160-d features (CLS ++ mean-patches); "
                f"got {cached.shape[-1]}. Re-extract with cls_plus_mean=True."
            )
        return torch.nn.functional.normalize(
            self._img_proj(cached.to(self._device)), dim=-1,
        )


# ----- FlexiCT --------------------------------------------------------------

class FlexiCTScorer:
    """FlexiCT-3D-VLM zero-shot (Li et al. 2026).

    Expects cached image features = 1024-d VLM projection from
    `FlexiCTExtractor(variant='vlm', project=True)` (the `flexict-zs_*` cache).
    Loads `Flexi_CT_VLM` (Qwen3-Embedding text tower) and scores prompts against
    the cached images, matching `downstream/vlm/ct_rate_eval.py`.

    Checkpoint comes from FLEXICT_VLM_CHECKPOINT / FLEXICT_CHECKPOINT (or pass
    `checkpoint_path`). `third_party/FlexiCT` is added to sys.path.

    NOTE: the author's canonical CT-RATE prompts are bare ``"{cls} ."`` /
    ``"No {cls}."`` (lowercased) — not our default
    ``"A chest CT scan showing {finding}."``. To reproduce their numbers, drive
    the prompt text from the zero_shot config accordingly.
    """

    name = "flexict"
    shared_dim = 1024

    def __init__(self, checkpoint_path: str | None = None) -> None:
        import sys
        from pathlib import Path

        root = str(Path(__file__).resolve().parent.parent / "third_party" / "FlexiCT")
        if root not in sys.path:
            sys.path.insert(0, root)
        from flexi_ct import Flexi_CT_VLM

        self._model = Flexi_CT_VLM(checkpoint_path=checkpoint_path, device="cpu").eval()
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "FlexiCTScorer":
        self._device = device
        self._model.to(device)
        return self

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        # The Qwen text tower runs in bf16; cast to float32 so the cosine
        # matmul matches the float32 cached image features (every other scorer
        # returns float32). Fixes `img @ pos.T` dtype mismatch in
        # score_against_prompts.
        return self._model.encode_text(prompts).float().to(self._device)  # [C, 1024] L2-norm

    @torch.inference_mode()
    def project_image(self, cached: torch.Tensor) -> torch.Tensor:
        if cached.shape[-1] != 1024:
            raise ValueError(
                f"FlexiCTScorer expects 1024-d VLM-projected features; got {cached.shape[-1]}. "
                f"Re-extract with flexict_zs_*.yaml (variant=vlm, project=true)."
            )
        return torch.nn.functional.normalize(cached.to(self._device), dim=-1)


# ----- Factory --------------------------------------------------------------

def build_scorer(name: str) -> ZeroShotScorer:
    if name == "colipri-crm":
        return ColipriScorer()
    if name == "ctclip":
        return CTClipScorer()
    if name == "merlin":
        return MerlinScorer()
    if name == "pillar0":
        return Pillar0Scorer()
    if name == "spectre":
        return SpectreScorer()
    if name == "flexict":
        return FlexiCTScorer()
    raise ValueError(f"unknown zero-shot scorer {name!r}")
