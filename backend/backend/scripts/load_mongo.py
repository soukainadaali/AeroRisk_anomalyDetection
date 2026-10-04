"""
scripts/load_mongo.py — Chargement de la base MongoDB
=====================================================
Charge data/processed/ntsb_clean_final.csv dans la collection `accidents`
et ajoute à chaque document le champ `split` (train / val / test), recalculé
avec exactement le même découpage que notebooks/03_feature_engineering.ipynb
(stratifié, 60/20/20, random_state=42). Le mongo_service exclut ensuite
systématiquement split == "test".

Usage (depuis backend/backend) :
    python scripts/load_mongo.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CLEAN_DATA_PATH  # noqa: E402

SEED = 42
INJURY_ORDER = {"NONE": 0, "MINR": 1, "SERS": 2, "FATL": 3}


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Reproduit les deux train_test_split de NB03 (cellule 26)."""
    y = df["ev_highest_injury"].map(INJURY_ORDER)
    idx_temp, idx_test, y_temp, _ = train_test_split(
        df.index, y, test_size=0.20, random_state=SEED, stratify=y)
    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp)

    split = pd.Series("train", index=df.index)
    split[idx_val] = "val"
    split[idx_test] = "test"
    return split


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB_NAME", "aviation_risk")

    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    df["split"] = assign_splits(df)
    print(f"{len(df):,} lignes — {df['split'].value_counts().to_dict()}")

    # to_json → types Python natifs, NaN → null
    records = json.loads(df.to_json(orient="records"))

    collection = MongoClient(uri)[db_name]["accidents"]
    collection.drop()
    collection.insert_many(records)
    collection.create_index([("split", ASCENDING)])
    collection.create_index([("ev_year", ASCENDING), ("ev_month", ASCENDING)])
    print(f"{collection.count_documents({}):,} documents insérés dans {db_name}.accidents")


if __name__ == "__main__":
    main()
