import pandas as pd
from typing import Optional


DEFAULT_LLM_SAMPLE_RUNS = 1
DEFAULT_LLM_MAX_ROWS = 999999
DEFAULT_LLM_MAX_ROWS_PER_REGIME = 999999
DEFAULT_LLM_SAMPLE_SEED = 42


def choose_llm_sample_size(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_LLM_MAX_ROWS,
) -> int:
    if df is None or df.shape[0] == 0:
        return 0
    return min(len(df), max_rows)


def sample_llm_rows(
    df: pd.DataFrame,
    sample_size: int,
    seed: int = DEFAULT_LLM_SAMPLE_SEED,
) -> pd.DataFrame:
    """Sample rows directly from the dataframe provided for this run.

    Callers should pass the original full datapair here for repeated LLM
    experiments, not a display/preview subset.
    """
    if df is None or df.shape[0] == 0:
        return df
    if sample_size <= 0 or len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def get_llm_sampling_plan(
    df: pd.DataFrame,
    data_format: str,
) -> dict:
    if data_format == "segmented_regimes":
        return {
            "n_samples": DEFAULT_LLM_SAMPLE_RUNS,
            "mode": "repeated_random_rows_per_regime_from_full_datapair",
            "max_rows_per_regime": DEFAULT_LLM_MAX_ROWS_PER_REGIME,
            "seed": DEFAULT_LLM_SAMPLE_SEED,
        }

    return {
        "n_samples": DEFAULT_LLM_SAMPLE_RUNS,
        "mode": "repeated_random_rows_from_full_datapair",
        "max_rows": DEFAULT_LLM_MAX_ROWS,
        "seed": DEFAULT_LLM_SAMPLE_SEED,
    }
