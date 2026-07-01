"""Concept-feature table: maps every label to an organ system + finding type.

Maps every label (RadChestCT 92 / CT-RATE 18) to:
  - organ_system : lung | airways | pleura | cardiac | mediastinum_vascular |
                   bones_chestwall | chestwall_extrathoracic | devices_surgical
  - finding_type : texture_diffuse | focal_object | morphometric_fluid |
                   calcification | device_foreign | skeletal
                   (imaging-appearance / structural grouping, the axis the
                   finding-type analysis uses; labels that fit no class are left
                   empty / NaN and excluded from the finding-type grouping)

Approach: transparent keyword rules on the finding token, with explicit
overrides for ambiguous labels. Emit ${CTFM_RESULTS}/paper/concept_features.csv.
THIS MAPPING IS CLINICALLY LOAD-BEARING — review before any numbers depend on it.

Reads the per-class prevalence CSVs produced by
`python -m scripts.evaluate --config-name compare_models[_ctrate]`.
A ready-made public concept_features.csv is also shipped under results/paper/;
this script regenerates it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(os.environ.get("CTFM_RESULTS", "results"))
OUT = HERE / "paper" / "concept_features.csv"

COHORT_CSV = {
    "RadChestCT": "knn_eval/compare_models/per_class_pr_auc.csv",
    "CT-RATE":    "knn_eval/compare_models_ctrate/per_class_pr_auc.csv",
}

# ---- explicit organ for RadChest / CT-RATE ------------------------------
ORGAN_OVERRIDE = {
    # RadChest
    "coronary_artery_disease": "cardiac", "atherosclerosis": "mediastinum_vascular",
    "atelectasis": "lung", "arthritis": "bones_chestwall", "emphysema": "lung",
    "pleural_effusion": "pleura", "interstitial_lung_disease": "lung",
    "pericardial_effusion": "cardiac", "bronchiectasis": "airways",
    "hernia": "mediastinum_vascular", "cardiomegaly": "cardiac",
    "aspiration": "lung", "pneumonia": "lung", "pulmonary_edema": "lung",
    "pneumothorax": "pleura", "pneumonitis": "lung", "aneurysm": "mediastinum_vascular",
    "bronchiolitis": "airways", "bronchitis": "airways", "heart_failure": "cardiac",
    "tuberculosis": "lung", "hemothorax": "pleura", "nodulegr1cm": "lung",
    "cancer_lung": "lung", "cancer_mediastinal": "mediastinum_vascular",
    "cancer_extrathoracic": "chestwall_extrathoracic", "infection_lung": "lung",
    "infection_extrapulmonary": "chestwall_extrathoracic", "inflammation_lung": "lung",
    "mass_lung": "lung", "mass_mediastinal": "mediastinum_vascular",
    "mass_extrathoracic": "chestwall_extrathoracic", "scarring_lung": "lung",
    "fibrosis_lung": "lung", "lymphadenopathy_mediastinal": "mediastinum_vascular",
    "lymphadenopathy_axillary": "chestwall_extrathoracic", "opacity": "lung",
    "groundglass": "lung", "scattered_nod": "lung", "scattered_calc": "lung",
    "bandlike_or_linear": "lung", "soft_tissue": "chestwall_extrathoracic",
    "cyst": "lung", "airspace_disease": "lung", "consolidation": "lung",
    "reticulation": "lung", "density": "lung", "bronchial_wall_thickening": "airways",
    "granuloma": "lung", "pleural_thickening": "pleura", "septal_thickening": "lung",
    "fracture": "bones_chestwall", "deformity": "bones_chestwall",
    "dilation_or_ectasia": "mediastinum_vascular", "mucous_plugging": "airways",
    "cavitation": "lung", "debris": "lung", "air_trapping": "lung",
    "pericardial_thickening": "cardiac", "infiltrate": "lung", "honeycombing": "lung",
    "tree_in_bud": "airways", "plaque": "mediastinum_vascular", "secretion": "airways",
    "lucency": "lung", "distention": "mediastinum_vascular", "bronchiolectasis": "airways",
    "congestion": "lung", "nodule_lung": "lung", "lesion_lung": "lung",
    "lesion_extrapulmonary": "chestwall_extrathoracic", "calcification_cardiac": "cardiac",
    "calcification_vascular": "mediastinum_vascular", "calcification_other": "chestwall_extrathoracic",
    "catheter_or_port": "devices_surgical", "clip": "devices_surgical",
    "pacemaker_or_defib": "devices_surgical", "stent": "devices_surgical",
    "staple": "devices_surgical", "chest_tube": "devices_surgical", "suture": "devices_surgical",
    "gi_tube": "devices_surgical", "breast_implant": "devices_surgical",
    "tracheal_tube": "devices_surgical", "hardware": "devices_surgical",
    "postsurgical": "devices_surgical", "lung_resection": "devices_surgical",
    "sternotomy": "devices_surgical", "transplant": "devices_surgical", "cabg": "devices_surgical",
    "heart_valve_replacement": "devices_surgical", "breast_surgery": "devices_surgical",
    # CT-RATE
    "Medical material": "devices_surgical", "Arterial wall calcification": "mediastinum_vascular",
    "Cardiomegaly": "cardiac", "Pericardial effusion": "cardiac",
    "Coronary artery wall calcification": "cardiac", "Hiatal hernia": "mediastinum_vascular",
    "Lymphadenopathy": "mediastinum_vascular", "Emphysema": "lung", "Atelectasis": "lung",
    "Lung nodule": "lung", "Lung opacity": "lung", "Pulmonary fibrotic sequela": "lung",
    "Pleural effusion": "pleura", "Mosaic attenuation pattern": "lung",
    "Peribronchial thickening": "airways", "Consolidation": "lung", "Bronchiectasis": "airways",
    "Interlobular septal thickening": "lung",
}


# ---- finding_type keyword rules (applied to lowercased finding token) ----
# order matters: device/calcification before texture/focal.
def finding_type(label: str, organ: str) -> str | None:
    s = label.lower()
    if organ == "devices_surgical":
        return "device_foreign"
    # guard: 'distention'/'distension' contain the substring 'stent' but are
    # morphometric, not devices.
    if "distent" in s or "distens" in s:
        return "morphometric_fluid"
    if any(w in s for w in ["stent", "cabg", "sternotomy", "pacemaker", "defib",
                            "_icd", "catheter", "_port", "implant", "suture", "staple",
                            "chest_tube", "ecmo", "gi_tube",
                            "valve_replacement", "tracheal_tube", "hardware",
                            "medical material", "clip"]):
        return "device_foreign"
    if any(w in s for w in ["calcification", "calc", "atheroscler", "plaque",
                            "coronary_artery_disease", "arterial wall"]):
        return "calcification"
    if any(w in s for w in ["effusion", "hemothorax", "empyema", "pneumothorax",
                            "pneumomediastinum", "edema", "cardiomegaly", "hernia",
                            "dilat", "ectasia", "aneurysm", "distention",
                            "congestion", "heart_failure"]):
        return "morphometric_fluid"
    if any(w in s for w in ["nodule", "scattered_nod", "mass", "lesion", "cancer",
                            "granuloma", "cyst", "bulla", "cavitation", "lymphadenopathy"]):
        return "focal_object"
    if any(w in s for w in ["emphysema", "fibro", "interstitial", "reticulation",
                            "honeycomb", "mosaic", "septal", "ground_glass", "groundglass",
                            "crazy_paving", "consolidation", "opacity", "infiltrate",
                            "airspace", "atelectasis", "aspiration", "pneumonia", "pneumonitis",
                            "bronchiolitis", "tree_in_bud", "air_trapping", "scarring",
                            "scar", "bandlike", "density", "lucency", "debris", "secretion",
                            "tuberculosis", "infection", "inflammation", "bronchitis",
                            "peribronchial", "bronchial_wall", "mucous", "bronchiectasis",
                            "bronchiolectasis", "thickening", "pleural_thickening"]):
        return "texture_diffuse"
    if any(w in s for w in ["fracture", "arthritis", "osteopenia", "deformity",
                            "post_resection_scar"]):
        return "skeletal"
    # Uncategorised: morphologically ill-defined labels (e.g. soft_tissue) that
    # fit no class. Returned as None -> empty in the CSV -> NaN on read ->
    # dropped by the downstream dropna(subset=["finding_type"]); i.e. excluded
    # from the finding-type grouping while still present as a label elsewhere.
    return None


def organ_for(cohort, label):
    return ORGAN_OVERRIDE.get(label, "other")


def main():
    rows = []
    for cohort, rel in COHORT_CSV.items():
        df = pd.read_csv(HERE / rel)
        for _, r in df.iterrows():
            lab = r["label"]
            organ = organ_for(cohort, lab)
            ftype = finding_type(lab, organ)
            rows.append({
                "cohort": cohort, "label": lab,
                "finding_clean": lab,
                "organ_system": organ, "finding_type": ftype,
                "prevalence": round(float(r["prevalence"]), 4),
            })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} labels)")
    print("\norgan_system distribution:")
    print(out.organ_system.value_counts())
    print("\nfinding_type distribution:")
    print(out.finding_type.value_counts())
    unc = out[out.finding_type.isna()]
    print(f"\nuncategorised (excluded from finding-type grouping): {len(unc)}")
    for _, r in unc.iterrows():
        print(f"  {r.cohort:12s} {r.label}")


if __name__ == "__main__":
    main()
