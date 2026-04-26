import os
import pandas as pd


def save(df: pd.DataFrame, name: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def load(name: str, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{name}.parquet")
    return pd.read_parquet(path)


def exists(name: str, data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, f"{name}.parquet"))
