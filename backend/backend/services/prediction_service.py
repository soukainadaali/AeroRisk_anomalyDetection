"""
services/prediction_service.py — Service de Prédiction ML (v3)
===============================================================
Charge les artefacts produits par les notebooks et reproduit leur chaîne
d'inférence à l'identique :

  JSON brut (schéma ntsb_clean_final.csv)
    → _payload_to_dataframe()       # dict → DataFrame 1 ligne
    → apply_feature_engineering()   # NB03 §2-5
    → preprocessing_pipeline        # NB03 ColumnTransformer  (103 → 1518)
    → variance_threshold            # NB03 VarianceThreshold  (1518 → 515)
    → SplitConformalClassifier      # NB05 MAPIE 1.x (LightGBM + LAC)
        .predict_set()  → (classe majoritaire, masque des classes plausibles)

Artefacts (notebooks/) :
  outputs/preprocessing_pipeline.pkl
  outputs/variance_threshold.pkl
  outputs/feature_names.csv
  uncertainty_outputs/mapie_classifier_lac.pkl  (fallback : baseline_models/best_model.pkl)
"""

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import NOTEBOOKS_DIR
from utils.feature_engineering import apply_feature_engineering, RAW_INPUT_COLS

logger = logging.getLogger(__name__)

# ── Mapping classes ───────────────────────────────────────────────────────────
# Ordre entier identique à injury_order de NB03 : {NONE:0, MINR:1, SERS:2, FATL:3}
RISK_LABELS: dict[int, str] = {
    0: "NONE",
    1: "MINR",
    2: "SERS",
    3: "FATL",
}


class PredictionService:
    """
    Service : charge le preprocessing, le VarianceThreshold et le modèle
    et expose predict_accident() pour les routes Flask.
    """

    def __init__(self) -> None:
        self._preprocessor = self._load("outputs/preprocessing_pipeline.pkl")
        self._vt = self._load("outputs/variance_threshold.pkl")
        self._feature_names = self._load_feature_names()

        mapie_path = NOTEBOOKS_DIR / "uncertainty_outputs" / "mapie_classifier_lac.pkl"
        if mapie_path.exists():
            self._model = self._load("uncertainty_outputs/mapie_classifier_lac.pkl")
        else:
            logger.warning("MAPIE introuvable — fallback sur le modèle de base (sans intervalle).")
            self._model = self._load("baseline_models/best_model.pkl")

        self._is_mapie = hasattr(self._model, "predict_set")
        # SplitConformalClassifier stocke alpha = 1 - confidence_level
        alphas = getattr(self._model, "_alphas", None)
        self._confidence_level = (1.0 - float(alphas[0])) if self._is_mapie and alphas else None

    # ── Chargement des artefacts ──────────────────────────────────────────────

    @staticmethod
    def _load(relative_path: str) -> Any:
        path = NOTEBOOKS_DIR / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact introuvable : '{path}'. "
                "Assurez-vous que les notebooks ont été exécutés."
            )
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Échec chargement '{path}' : {exc}") from exc
        logger.info("Artefact chargé : %s (%s)", path, type(obj).__name__)
        return obj

    @staticmethod
    def _load_feature_names() -> list[str]:
        """Noms des 515 colonnes conservées après VarianceThreshold."""
        path = NOTEBOOKS_DIR / "outputs" / "feature_names.csv"
        try:
            return pd.read_csv(path).iloc[:, 0].tolist()
        except Exception as exc:
            raise RuntimeError(f"Échec chargement feature_names.csv : {exc}") from exc

    # ── Conversion payload → DataFrame brut ──────────────────────────────────

    @staticmethod
    def _payload_to_dataframe(payload: dict) -> pd.DataFrame:
        """
        Convertit le JSON brut frontend en DataFrame 1 ligne.
        Seules les clés présentes dans RAW_INPUT_COLS sont conservées.
        """
        row: dict[str, Any] = {col: np.nan for col in RAW_INPUT_COLS}
        unknown_keys = set(payload.keys()) - set(RAW_INPUT_COLS)
        row.update({k: v for k, v in payload.items() if k in row and v is not None})

        if unknown_keys:
            logger.debug("Clés payload ignorées (hors contrat) : %s", unknown_keys)

        return pd.DataFrame([row], columns=RAW_INPUT_COLS)

    # ── Extraction de l'intervalle MAPIE ─────────────────────────────────────

    @staticmethod
    def _extract_mapie_interval(pred_set_row: np.ndarray) -> list[str]:
        """Masque booléen (n_classes,) → labels plausibles, ex: ["MINR", "SERS"]."""
        plausible = [
            RISK_LABELS[i]
            for i, included in enumerate(pred_set_row)
            if bool(included) and i in RISK_LABELS
        ]
        if not plausible:
            logger.warning("Prediction set MAPIE vide — intervalle retourné vide.")
        return plausible

    # ── Transformation complète ───────────────────────────────────────────────

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Lignes au format NB01 → matrice de 515 features attendue par le modèle."""
        X_fe = apply_feature_engineering(raw_df)
        X_processed = self._preprocessor.transform(X_fe)
        X_vt = self._vt.transform(X_processed)
        return pd.DataFrame(X_vt, columns=self._feature_names)

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def predict_accident(self, payload: dict) -> dict:
        """
        Flux complet : JSON brut → prédiction structurée.

        Returns:
            {
              "prediction":         str   — classe majoritaire
              "confidence_level":   float — 1 - alpha MAPIE (0.90)
              "uncertainty_set":    list  — classes plausibles MAPIE
              "raw_prediction_idx": int   — indice brut (debug)
            }

        Raises:
            ValueError  : Payload vide ou non-conforme.
            RuntimeError: Erreur interne du pipeline.
        """
        if not payload or not isinstance(payload, dict):
            raise ValueError("Le payload doit être un dictionnaire non vide.")

        raw_df = self._payload_to_dataframe(payload)

        try:
            X = self.transform(raw_df)
        except Exception as exc:
            raise RuntimeError(f"Échec de la préparation des features : {exc}") from exc

        try:
            if self._is_mapie:
                y_pred, y_pred_set = self._model.predict_set(X)
                plausible = self._extract_mapie_interval(y_pred_set[0, :, 0])
            else:
                y_pred = self._model.predict(X)
                plausible = None
        except Exception as exc:
            raise RuntimeError(f"Erreur modèle.predict() : {exc}") from exc

        raw_idx = int(y_pred[0])
        majority_class = RISK_LABELS.get(raw_idx, f"CLASS_{raw_idx}")

        result = {
            "prediction":         majority_class,
            "confidence_level":   self._confidence_level,
            "uncertainty_set":    plausible if plausible is not None else [majority_class],
            "raw_prediction_idx": raw_idx,
        }

        logger.info("Prédiction : %s | Intervalle MAPIE : %s", majority_class, result["uncertainty_set"])
        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
try:
    prediction_service = PredictionService()
except (FileNotFoundError, RuntimeError) as e:
    logger.error("IMPOSSIBLE DE CHARGER LE PIPELINE : %s", e)
    prediction_service = None  # type: ignore[assignment]
