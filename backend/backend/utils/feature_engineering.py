"""
utils/feature_engineering.py — Feature Engineering Pipeline
==========================================================
Reproduit fidèlement les transformations de notebooks/03_feature_engineering.ipynb.

Entrée : une ligne au format de data/processed/ntsb_clean_final.csv (sortie NB01,
         unités déjà converties en SI : wx_temp_c, vis_km, sky_ceil_ht_m, ...).
Sortie : DataFrame avec FEATURE_COLS, prêt pour preprocessing_pipeline.pkl.

Flux :
  1. Normalisation des types (JSON → dtypes vus à l'entraînement)
  2. Corrections d'anomalies (afm_hrs==0, gust sans wind)          — NB03 §2
  3. Groupement des constructeurs (top 25 + OTHER)                 — NB03 §3
  4. Features dérivées (météo, maintenance, cycliques, âge, log)   — NB03 §4-5
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Groupes de colonnes (identiques à NB03, cellule 22) ─────────────────────────
BINARY_YN_COLS: List[str] = [
    'homebuilt', 'second_pilot', 'afm_hrs_since',
    'elt_install', 'elt_oper',
    'crew_tox_perf', 'latlong_acq', 'ev_nr_apt_loc',
    'wind_dir_ind', 'wind_vel_ind', 'gust_ind', 'pilot_flying',
]

NUMERIC_BINARY_COLS: List[str] = [
    'noaa_fog', 'noaa_rain', 'noaa_snow', 'noaa_thunder',
    'adverse_weather', 'poor_visibility', 'high_wind',
    'extreme_temp', 'fog_risk', 'is_imc',
    'is_missing_rwy_len', 'is_missing_rwy_width', 'is_missing_pax_seats',
    'is_missing_crew_age', 'is_missing_fuel_on_board', 'is_missing_afm_hrs',
    'is_missing_afm_hrs_last_insp', 'is_missing_acft_year', 'is_missing_wind_vel_kts',
]

OHE_COLS: List[str] = [
    'acft_category', 'ev_season', 'light_cond',
    'sky_cond_nonceil', 'sky_cond_ceil', 'wx_src_iic',
    'type_last_insp', 'crew_category', 'med_certf',
    'med_crtf_vldty', 'fixed_retractable', 'acft_make_grouped', 'seat_occ_pic',
]

OHE_HIGH_COLS: List[str] = [
    'ev_state', 'far_part', 'type_fly', 'flt_plan_filed',
    'flight_plan_activated', 'pc_profession', 'infl_rest_inst',
    'dprt_pt_same_ev', 'elt_type', 'available_restraint',
    'restraint_used', 'med_crtf_limit',
]

ORDINAL_COLS: List[str] = ['crew_sex']

NUMERIC_COLS: List[str] = [
    'noaa_temp_c', 'noaa_temp_max_c', 'noaa_temp_min_c',
    'noaa_wind_knots', 'noaa_maxwind_knots', 'noaa_gust_knots',
    'noaa_slp_hpa', 'noaa_visib_km', 'noaa_prcp_mm', 'noaa_dist_km',
    'gust_excess', 'wx_temp_c', 'wx_dew_pt_c',
    'wind_dir_deg', 'wind_vel_kts', 'gust_kts', 'vis_km',
    'sky_ceil_ht_m', 'sky_nonceil_ht_m', 'altimeter_hpa', 'td_spread',
    'apt_elev_m', 'apt_dist_km', 'wx_obs_elev_m', 'wx_obs_dist_km',
    'rwy_width_m', 'rwy_len_m_log', 'num_eng', 'fc_seats', 'pax_seats',
    'total_seats', 'afm_hrs_log', 'afm_hrs_last_insp',
    'cert_max_gr_wt_kg_log', 'fuel_on_board_l_log',
    'acft_age', 'maintenance_ratio', 'crew_age',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'ev_year', 'wx_obs_time',
]

# ── Colonnes finales attendues par le pipeline ML (ordre NB03) ──────────────────
FEATURE_COLS: List[str] = (
    BINARY_YN_COLS + NUMERIC_BINARY_COLS + OHE_COLS
    + OHE_HIGH_COLS + ORDINAL_COLS + NUMERIC_COLS
)

# Features calculées ici (absentes du CSV propre)
DERIVED_COLS: List[str] = [
    'adverse_weather', 'poor_visibility', 'high_wind', 'extreme_temp',
    'fog_risk', 'is_imc', 'td_spread', 'maintenance_ratio',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'acft_age', 'gust_excess', 'acft_make_grouped',
    'afm_hrs_log', 'cert_max_gr_wt_kg_log', 'fuel_on_board_l_log', 'rwy_len_m_log',
]

# Colonnes sources utilisées pour dériver les features, puis supprimées
SOURCE_COLS: List[str] = [
    'ev_time', 'ev_dow', 'ev_month', 'wx_cond_basic', 'acft_make', 'acft_year',
    'afm_hrs', 'cert_max_gr_wt_kg', 'fuel_on_board_l', 'rwy_len_m',
]

# ── Colonnes brutes attendues du frontend (schéma de ntsb_clean_final.csv) ──────
RAW_INPUT_COLS: List[str] = (
    [c for c in FEATURE_COLS if c not in DERIVED_COLS] + SOURCE_COLS
)

# Colonnes catégorielles (str dans le CSV d'entraînement)
CATEGORICAL_COLS: set[str] = (
    set(BINARY_YN_COLS + OHE_COLS + OHE_HIGH_COLS + ORDINAL_COLS
        + ['ev_dow', 'wx_cond_basic', 'acft_make'])
    - {'pilot_flying', 'acft_make_grouped'}
)
# Colonnes booléennes (bool dans le CSV, castées en int par NB03)
BOOL_COLS: set[str] = {'pilot_flying', 'noaa_fog', 'noaa_rain', 'noaa_snow', 'noaa_thunder'}

TOP_MAKES: List[str] = [
    'CESSNA', 'PIPER', 'BEECH', 'ROBINSON', 'BOEING', 'BELL',
    'CIRRUS', 'AIR TRACTOR', 'MOONEY', 'AIRBUS', 'SCHWEIZER',
    'BELLANCA', 'DE HAVILLAND', 'AERONCA', 'MAULE', 'DIAMOND',
    'HUGHES', 'BOMBARDIER', 'VANS', 'CHAMPION', 'LUSCOMBE',
    'EMBRAER', 'STINSON', 'EUROCOPTER', 'NORTH AMERICAN',
]

MAKE_ALIASES: dict[str, str] = {
    'ROBINSON HELICOPTER COMPANY': 'ROBINSON', 'ROBINSON HELICOPTER': 'ROBINSON',
    'CIRRUS DESIGN CORP': 'CIRRUS',            'CIRRUS DESIGN': 'CIRRUS',
    'DIAMOND AIRCRAFT IND INC': 'DIAMOND',     'DIAMOND AIRCRAFT': 'DIAMOND',
    'AIR TRACTOR INC': 'AIR TRACTOR',          'BOMBARDIER INC': 'BOMBARDIER',
    'DEHAVILLAND': 'DE HAVILLAND',             'DE HAVILLAND CANADA': 'DE HAVILLAND',
}

# Flags is_missing_* (NB01) → colonne SI correspondante, pour les payloads sans flag
MISSING_FLAG_SOURCES: dict[str, str] = {
    'is_missing_rwy_len': 'rwy_len_m',
    'is_missing_rwy_width': 'rwy_width_m',
    'is_missing_pax_seats': 'pax_seats',
    'is_missing_crew_age': 'crew_age',
    'is_missing_fuel_on_board': 'fuel_on_board_l',
    'is_missing_afm_hrs': 'afm_hrs',
    'is_missing_afm_hrs_last_insp': 'afm_hrs_last_insp',
    'is_missing_acft_year': 'acft_year',
    'is_missing_wind_vel_kts': 'wind_vel_kts',
}


# ── Normalisation des types ─────────────────────────────────────────────────────

def _to_category(value):
    """JSON → str comme lu par pandas dans le CSV (ex: 91 → '091' pour far_part)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(int(value)) if float(value).is_integer() else str(value)
    value = str(value)
    return value if value.strip() != '' else np.nan


def _to_bool_int(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('true', 'y', 'yes', '1'):
            return 1
        if value in ('false', 'n', 'no', '0'):
            return 0
        return np.nan
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return int(bool(value))


def normalize_types(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les dtypes d'un payload JSON sur ceux du CSV d'entraînement."""
    columns = {}
    for col in raw_df.columns:
        if col in CATEGORICAL_COLS:
            columns[col] = raw_df[col].map(_to_category).astype(object)
        elif col in BOOL_COLS:
            columns[col] = raw_df[col].map(_to_bool_int).astype(float)
        else:
            columns[col] = pd.to_numeric(raw_df[col], errors='coerce')
    df = pd.DataFrame(columns, index=raw_df.index)
    if 'far_part' in df.columns:
        # far_part est codé sur 3 chiffres ('091') — rétablit le zéro perdu en JSON
        df['far_part'] = df['far_part'].map(
            lambda v: v.zfill(3) if isinstance(v, str) and v.isdigit() else v
        )
    return df


# ── Feature engineering (NB03) ──────────────────────────────────────────────────

def apply_feature_engineering(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le feature engineering de NB03 sur des lignes au format NB01.

    Args:
        raw_df: DataFrame avec (un sous-ensemble de) RAW_INPUT_COLS

    Returns:
        DataFrame avec FEATURE_COLS prêt pour le preprocessing pipeline
    """
    df = normalize_types(raw_df.reindex(columns=RAW_INPUT_COLS))

    # Flags NB01 absents du payload → recalculés depuis la valeur source
    for flag, source in MISSING_FLAG_SOURCES.items():
        df[flag] = df[flag].fillna(df[source].isna().astype(int))

    # ── §2 Corrections d'anomalies ───────────────────────────────────────────
    mask_zero_hrs = (df['afm_hrs'] == 0) & (~df['is_missing_afm_hrs'].astype(bool))
    df.loc[mask_zero_hrs, 'afm_hrs'] = np.nan

    mask_gust_no_wind = (df['noaa_gust_knots'].notna()) & (df['noaa_wind_knots'] == 0)
    df.loc[mask_gust_no_wind, 'noaa_wind_knots'] = df.loc[mask_gust_no_wind, 'noaa_gust_knots']

    # ── §3 Constructeurs ─────────────────────────────────────────────────────
    df['acft_make'] = df['acft_make'].str.upper().str.strip().replace(MAKE_ALIASES)
    df['acft_make_grouped'] = df['acft_make'].apply(lambda x: x if x in TOP_MAKES else 'OTHER')

    # ── §4.1 Flags météo ─────────────────────────────────────────────────────
    # NaN.astype(bool) vaut True en pandas : NB03 travaillait sur des bool sans NaN
    noaa_flags = df[['noaa_fog', 'noaa_rain', 'noaa_snow', 'noaa_thunder']].fillna(0).astype(bool)
    df['adverse_weather'] = noaa_flags.any(axis=1).astype(int)
    df['poor_visibility'] = (df['vis_km'] < 5).astype(int)
    df['high_wind']       = (df['noaa_wind_knots'] > 25).astype(int)
    df['extreme_temp']    = ((df['noaa_temp_c'] < -20) | (df['noaa_temp_c'] > 35)).astype(int)
    df['td_spread']       = df['wx_temp_c'] - df['wx_dew_pt_c']
    df['fog_risk']        = (df['td_spread'] < 3).astype(int)
    df['is_imc']          = (df['wx_cond_basic'] == 'IMC').astype(int)

    # ── §4.2 Maintenance ratio ───────────────────────────────────────────────
    eps = 1e-3
    df['maintenance_ratio'] = (
        (df['afm_hrs'] - df['afm_hrs_last_insp']).clip(lower=0) / (df['afm_hrs'] + eps)
    ).clip(0, 1)

    # ── §4.3 Features temporelles cycliques ──────────────────────────────────
    ev_hour = (df['ev_time'] // 100).clip(0, 23)
    df['hour_sin'] = np.sin(2 * np.pi * ev_hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ev_hour / 24)

    dow_map = {'Mo': 0, 'Tu': 1, 'We': 2, 'Th': 3, 'Fr': 4, 'Sa': 5, 'Su': 6}
    ev_dow_num = df['ev_dow'].map(dow_map)
    df['dow_sin'] = np.sin(2 * np.pi * ev_dow_num / 7)
    df['dow_cos'] = np.cos(2 * np.pi * ev_dow_num / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['ev_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['ev_month'] / 12)

    # ── §4.4-4.5 Âge appareil, excès de rafale ───────────────────────────────
    df['acft_age'] = (df['ev_year'] - df['acft_year']).clip(lower=0, upper=80)
    df['gust_excess'] = (df['noaa_gust_knots'] - df['noaa_wind_knots']).clip(lower=0).fillna(0)

    # ── §5 Log transforms ────────────────────────────────────────────────────
    for col in ['afm_hrs', 'cert_max_gr_wt_kg', 'fuel_on_board_l', 'rwy_len_m']:
        df[f'{col}_log'] = np.log1p(df[col])

    return df[FEATURE_COLS].copy()
